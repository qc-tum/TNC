use log::debug;
use mpi::collective::SystemOperation;
use mpi::topology::{Color, Process, SimpleCommunicator};
use mpi::traits::{BufferMut, Communicator, Destination, Equivalence, Root, Source};
use mpi::Rank;
use mpi_ext::RootExtension;

use super::mpi_types::RankTensorMapping;
use super::serialization::{deserialize_tensor, serialize_tensor};
use crate::mpi::mpi_types::MessageBinaryBlob;
use crate::mpi::serialization::{deserialize, serialize};
use crate::tensornetwork::contraction::contract_tensor_network;
use crate::tensornetwork::tensor::Tensor;
use crate::tensornetwork::tensordata::TensorData;
use crate::types::{ContractionIndex, SlicingPlan, SlicingTask};

/// Broadcasts a vector of `data` from `root` to all processes in `world`. For the
/// receivers, `data` can just be an empty vector.
fn broadcast_vec<T>(data: &mut Vec<T>, root: &Process)
where
    T: Clone + Default,
    Vec<T>: BufferMut,
{
    // Broadcast length
    let mut len = if root.is_self() { data.len() } else { 0 };
    root.broadcast_into(&mut len);

    // Broadcast data
    if !root.is_self() {
        data.resize(len, Default::default());
    }
    root.broadcast_into(data);
}

/// Extracts the communication path from the total contraction path.
#[must_use]
pub fn extract_communication_path(path: &[ContractionIndex]) -> Vec<ContractionIndex> {
    path.iter()
        .filter(|a| matches!(a, ContractionIndex::Pair(_, _)))
        .cloned()
        .collect()
}

/// Broadcast a contraction index `path` from `root` to all processes in `world`. For
/// the receivers, `path` can just be an empty slice.
pub fn broadcast_path(path: &mut Vec<ContractionIndex>, root: &Process) {
    // Serialize path
    let mut data = if root.is_self() {
        serialize(&path)
    } else {
        Default::default()
    };

    // Broadcast data
    broadcast_vec(&mut data, root);

    // Deserialize path
    if !root.is_self() {
        *path = deserialize(&data);
    }

    debug!(path:serde; "Received broadcasted path");
}

/// Broadcast a value by serializing it and sending it as byte array.
fn broadcast_serializing<T>(data: T, root: &Process) -> T
where
    T: serde::Serialize + serde::de::DeserializeOwned + Clone,
{
    let mut raw_value = if root.is_self() {
        serialize(&data)
    } else {
        Default::default()
    };

    broadcast_vec(&mut raw_value, root);

    if root.is_self() {
        data
    } else {
        deserialize(&raw_value)
    }
}

/// Sends the `tensor` to `receiver` via MPI.
fn send_tensor(tensor: &Tensor, receiver: Rank, world: &SimpleCommunicator) {
    let data = serialize_tensor(tensor);
    world.process_at_rank(receiver).send(&data);
}

/// Receives a tensor from `sender` via MPI.
fn receive_tensor(sender: Rank, world: &SimpleCommunicator) -> Tensor {
    // Receive the buffer
    let (data, _status) = world
        .process_at_rank(sender)
        .receive_vec::<MessageBinaryBlob>();

    if !data.is_empty() {
        deserialize_tensor(&data)
    } else {
        Tensor::default()
    }
}

/// Determines the tensor mapping and slice groups for the given contraction `path`.
/// Also returns the number of used ranks.
fn get_tensor_mapping_and_slice_groups(
    r_tn: &Tensor,
    path: &[ContractionIndex],
    size: Rank,
) -> (RankTensorMapping, Vec<i32>, Rank) {
    let mut used_ranks = 0;
    let mut tensor_mapping = RankTensorMapping::with_capacity(size as usize);
    let mut slice_groups = vec![-1; size as usize];
    let mut used_groups = 0;

    let Some(last) = path.last() else {
        // Empty path
        return (tensor_mapping, slice_groups, used_ranks);
    };
    let &ContractionIndex::Pair(final_tensor, _) = last else {
        panic!("Last part of path should be a pair")
    };

    // Reserve rank 0 for the final tensor
    used_ranks = 1;
    for pair in path {
        if let ContractionIndex::Path(i, slicing, _) = pair {
            if *i == final_tensor {
                // Assign the final tensor to rank 0
                tensor_mapping.insert(0, *i);
            } else {
                // Assign the next available rank to tensor `i`
                tensor_mapping.insert(used_ranks, *i);
                used_ranks += 1;
            }

            if let Some(slicing) = slicing {
                // Determine how many ranks are needed for doing all slices in parallel
                let local_tensor = r_tn.tensor(*i);
                let additional_needed_ranks: i32 =
                    (slicing.task_count(local_tensor) - 1).try_into().unwrap();

                // Assign each of the needed ranks the same color
                slice_groups[tensor_mapping.rank(*i) as usize] = used_groups;
                for rank in used_ranks..used_ranks + additional_needed_ranks {
                    slice_groups[rank as usize] = used_groups;
                }
                used_groups += 1;
                used_ranks += additional_needed_ranks;
            }
        }
    }
    assert!(
        used_ranks <= size,
        "Not enough MPI ranks available, got {size} but need {used_ranks}!"
    );
    (tensor_mapping, slice_groups, used_ranks)
}

/// Scatters `slice` on the rank corresponding to the `root` process to all processes
/// in the communicator of `root`, such that rank `i` gets `slice[i]`. Processes
/// other than the root can pass an empty slice.
fn scatter_slice<T>(slice: &[T], root: &Process) -> T
where
    T: Default + Equivalence,
{
    let mut recv_buffer = T::default();
    if root.is_self() {
        root.scatter_into_root(slice, &mut recv_buffer);
    } else {
        root.scatter_into(&mut recv_buffer);
    }
    recv_buffer
}

/// Information needed for communication during contraction of the tensor network.
pub struct Communication {
    /// A mapping between MPI ranks and their owned composite tensors. In slice
    /// groups, only the slice root rank is assigned the tensor.
    tensor_mapping: RankTensorMapping,

    /// The communicator for the slice group. `None` if the rank is not part of a
    /// slice group.
    slice_comm: Option<SimpleCommunicator>,
}

/// Distributes the partitioned tensor network to the various processes via MPI.
pub fn scatter_tensor_network(
    r_tn: &Tensor,
    path: &[ContractionIndex],
    rank: Rank,
    size: Rank,
    world: &SimpleCommunicator,
) -> (
    Tensor,
    Vec<ContractionIndex>,
    Option<SlicingTask>,
    Communication,
) {
    debug!(rank, size; "Scattering tensor network");
    let root = world.process_at_rank(0);

    // Get information about used ranks
    let (tensor_mapping, slicing_grouping, _) = if rank == 0 {
        get_tensor_mapping_and_slice_groups(r_tn, path, size)
    } else {
        Default::default()
    };

    // Tell the ranks the tensor they are responsible for (if any)
    let tensor_mapping = broadcast_serializing(tensor_mapping, &root);
    let is_tensor_owner = tensor_mapping.tensor(rank).is_some();

    // Send the slicing information
    let slice_group = scatter_slice(&slicing_grouping, &root);

    // Create communicators for the slice groups
    let color = if slice_group >= 0 {
        Color::with_value(slice_group)
    } else {
        Color::undefined()
    };
    let slice_comm = world.split_by_color(color);
    debug!(tensor_mapping:serde, is_tensor_owner, slice_group; "Scattered organizational data");

    // Send the local paths
    let mut local_path_raw = if rank == 0 {
        debug!("Sending local paths");
        let mut local_path = None;
        for contraction_path in path {
            if let ContractionIndex::Path(i, slicing, local) = contraction_path {
                let payload = (slicing, local);
                let target_rank = tensor_mapping.rank(*i);
                if target_rank == 0 {
                    // This is the path for the root, no need to send it
                    local_path = Some(payload);
                    continue;
                }

                world
                    .process_at_rank(target_rank)
                    .send(&serialize(&payload));
            }
        }
        serialize(&local_path.unwrap())
    } else if is_tensor_owner {
        debug!("Receiving local path");
        let (raw_path, _status) = world.process_at_rank(0).receive_vec::<u8>();
        raw_path
    } else {
        Default::default()
    };

    // Broadcast the path in the slicing group
    if let Some(slice_comm) = &slice_comm {
        debug!("Broadcasting local paths to slice group");
        broadcast_vec(&mut local_path_raw, &slice_comm.process_at_rank(0));
    }

    // Deserialize the path
    let (slicing, local_path): (Option<SlicingPlan>, Vec<ContractionIndex>) =
        if !local_path_raw.is_empty() {
            deserialize(&local_path_raw)
        } else {
            Default::default()
        };
    debug!(slicing:serde; "Received local path");

    // Send the tensors
    let mut local_tn_raw = if rank == 0 {
        debug!("Sending tensors");
        let mut local_tn = None;
        for &(target_rank, tensor_index) in &tensor_mapping {
            let tensor = r_tn.tensor(tensor_index);
            if target_rank == 0 {
                // This is the tensor for the root, no need to send it
                local_tn = Some(tensor);
                continue;
            }

            send_tensor(tensor, target_rank, world);
        }
        serialize_tensor(local_tn.unwrap())
    } else if is_tensor_owner {
        debug!("Receiving tensor");
        let (raw_tensor, _status) = world.process_at_rank(0).receive_vec::<MessageBinaryBlob>();
        raw_tensor
    } else {
        Default::default()
    };

    // Broadcast the tensor in the slicing group
    if let Some(slice_comm) = &slice_comm {
        debug!("Broadcasting tensors to slice group");
        broadcast_vec(&mut local_tn_raw, &slice_comm.process_at_rank(0));
    }

    // Deserialize the tensor
    let local_tn = if !local_tn_raw.is_empty() {
        deserialize_tensor(&local_tn_raw)
    } else {
        Default::default()
    };
    debug!("Received tensor");

    // Get the slicing task (based on the rank in the slicing group), if any
    let slicing_task = if let Some(slice_comm) = &slice_comm {
        let slicing_group_rank = slice_comm.rank();
        let task = slicing
            .unwrap()
            .get_task(&local_tn, slicing_group_rank as usize);
        debug!(task:serde; "Got slicing task");
        Some(task)
    } else {
        assert!(slicing.is_none());
        Default::default()
    };

    debug!("Scattered tensor network");

    // Return the local tensor, path and communication information
    (
        local_tn,
        local_path,
        slicing_task,
        Communication {
            tensor_mapping,
            slice_comm,
        },
    )
}

/// Uses the `path` as a communication blueprint to iteratively send tensors and contract them in a fan-in.
/// Assumes that `path` is a valid contraction path.
pub fn intermediate_reduce_tensor_network(
    local_tn: &mut Tensor,
    path: &[ContractionIndex],
    rank: Rank,
    world: &SimpleCommunicator,
    communication: &Communication,
) {
    debug!(rank, path:serde; "Reducing tensor network (intermediate)");
    assert!(local_tn.is_leaf());

    // Reduce the slice groups
    if let Some(slice_comm) = &communication.slice_comm {
        // Get raw data access (data might be transposed, but it's the same permutation on every rank,
        // so summing is valid)
        let tensor_data = std::mem::take(&mut local_tn.tensordata);
        let mut data_tensor = tensor_data.into_data();
        let raw_data_view = data_tensor.raw_data_mut();
        debug!(slice_rank=slice_comm.rank(), group_size=slice_comm.size(); "Reducing slice group");

        // Directly reduce into the root data tensor
        // TODO: this only supports tensors up to ~275 GB. If we want to go bigger, we need to do multiple broadcasts.
        let slice_root = slice_comm.process_at_rank(0);
        let op = SystemOperation::sum();
        if slice_root.is_self() {
            slice_root.reduce_into_root_inplace(raw_data_view, op);
        } else {
            slice_root.reduce_into(raw_data_view, op);
        }

        // Put the data back into the tensor
        local_tn.set_tensor_data(TensorData::Matrix(data_tensor));
        debug!("Reduced slice group");
    }

    let mut final_rank = 0;
    for pair in path {
        match pair {
            ContractionIndex::Pair(x, y) => {
                let receiver = communication.tensor_mapping.rank(*x);
                let sender = communication.tensor_mapping.rank(*y);
                final_rank = receiver;
                if receiver == rank {
                    // Receive tensor
                    debug!(sender; "Start receiving tensor");
                    let received_tensor = receive_tensor(sender, world);
                    debug!(sender; "Finish receiving tensor");

                    // Add local tensor and received tensor into a new tensor network
                    let tensor_network =
                        Tensor::new_composite(vec![std::mem::take(local_tn), received_tensor]);

                    // Contract tensors
                    let result =
                        contract_tensor_network(tensor_network, &[ContractionIndex::Pair(0, 1)]);
                    *local_tn = result;
                }
                if sender == rank {
                    debug!(receiver; "Start sending tensor");
                    send_tensor(local_tn, receiver, world);
                    debug!(receiver; "Finish sending tensor");
                }
            }
            ContractionIndex::Path(..) => panic!("Requires pair"),
        }
    }

    // Only runs if the final contracted process is not process 0
    if final_rank != 0 {
        debug!(rank, final_rank; "Final rank is not 0");
        if rank == 0 {
            debug!(sender = final_rank; "Receiving final tensor");
            let received_tensor = receive_tensor(final_rank, world);
            *local_tn = received_tensor;
        }
        if rank == final_rank {
            debug!(receiver = 0; "Sending final tensor");
            send_tensor(local_tn, 0, world);
        }
    }
    debug!("Reduced tensor network");
}

#[cfg(test)]
mod tests {
    use mpi::traits::Communicator;
    use mpi_test::mpi_test;
    use rustc_hash::FxHashMap;

    use super::*;
    use crate::path;

    #[mpi_test(2)]
    fn test_broadcast_contraction_path() {
        let universe = mpi::initialize().unwrap();
        let world = universe.world();
        let rank = world.rank();
        let root_process = world.process_at_rank(0);
        let max = usize::MAX;

        let ref_contraction_indices = path![
            (0, 4),
            (1, 5),
            (2, 16),
            (7, max),
            (max, 5),
            (64, 2),
            (4, 55),
            (81, 21),
            (2, 72),
            (23, 3),
            (40, 5),
            (2, 26)
        ]
        .to_vec();

        let mut contraction_indices = if rank == 0 {
            ref_contraction_indices.clone()
        } else {
            Default::default()
        };
        broadcast_path(&mut contraction_indices, &root_process);

        assert_eq!(contraction_indices, ref_contraction_indices);
    }

    #[test]
    fn test_serialize_empty_vec_is_nonempty_holds() {
        // The code relies on the fact that an empty Vec is serialized to a non-empty
        // sequence of bytes, in order to discriminate between a provided empty Vec
        // and a Default::default(). This test checks that we can rely on this.
        let empty = Vec::<ContractionIndex>::new();
        let serialized = serialize(&empty);

        assert!(!serialized.is_empty());
    }

    #[test]
    fn test_serialize_tuple_of_references() {
        // Check that a tuple of references can be serialized and deserialized
        // as expected.
        let slicing_plan_ref = Some(SlicingPlan {
            slices: vec![1, 2, 3],
        });
        let local_path_ref = vec![
            ContractionIndex::Pair(1, 2),
            ContractionIndex::Pair(2, 3),
            ContractionIndex::Pair(3, 4),
        ];

        let data = (&slicing_plan_ref, &local_path_ref);

        let serialized = serialize(&data);

        let (slicing_plan, local_path): (Option<SlicingPlan>, Vec<ContractionIndex>) =
            deserialize(&serialized);

        assert_eq!(slicing_plan, slicing_plan_ref);
        assert_eq!(local_path, local_path_ref);
    }

    #[mpi_test(3)]
    fn test_scatter_slice() {
        let universe = mpi::initialize().unwrap();
        let world = universe.world();
        let rank = world.rank();
        let root_process = world.process_at_rank(0);

        let slice = if rank == 0 {
            vec![0, 1, 2]
        } else {
            Default::default()
        };

        let received_number = scatter_slice(&slice, &root_process);

        assert_eq!(received_number, rank);
    }

    fn setup_tn() -> Tensor {
        let bond_dims = FxHashMap::from_iter([
            (0, 2),
            (1, 3),
            (2, 2),
            (3, 6),
            (4, 3),
            (5, 4),
            (6, 2),
            (7, 5),
            (8, 2),
            (9, 2),
        ]);
        let t1 = Tensor::new_from_map(vec![0, 1, 2], &bond_dims);
        let t2 = Tensor::new_from_map(vec![2, 3], &bond_dims);
        let t3 = Tensor::new_from_map(vec![1, 3, 4], &bond_dims);
        let tn1 = Tensor::new_composite(vec![t1, t2, t3]);
        let t4 = Tensor::new_from_map(vec![5, 6], &bond_dims);
        let t5 = Tensor::new_from_map(vec![4, 6, 7], &bond_dims);
        let t6 = Tensor::new_from_map(vec![7], &bond_dims);
        let tn2 = Tensor::new_composite(vec![t4, t5, t6]);
        let t7 = Tensor::new_from_map(vec![8, 9], &bond_dims);
        let t8 = Tensor::new_from_map(vec![0, 9], &bond_dims);
        let tn3 = Tensor::new_composite(vec![t7, t8]);

        Tensor::new_composite(vec![tn1, tn2, tn3])
    }

    #[test]
    fn test_tensor_mapping_no_slicing() {
        let tn = setup_tn();

        let path = path![
            (0, [(0, 2), (0, 1)]),
            (2, [(0, 1)]),
            (1, [(0, 1), (0, 1)]),
            (0, 2),
            (0, 1)
        ];

        let (tensor_mapping, slice_groups, used_ranks) =
            get_tensor_mapping_and_slice_groups(&tn, path, 4);

        assert_eq!(used_ranks, 3);
        assert_eq!(tensor_mapping.rank(0), 0);
        assert_eq!(tensor_mapping.rank(1), 2);
        assert_eq!(tensor_mapping.rank(2), 1);
        assert_eq!(tensor_mapping.tensor(0), Some(0));
        assert_eq!(tensor_mapping.tensor(1), Some(2));
        assert_eq!(tensor_mapping.tensor(2), Some(1));
        assert_eq!(tensor_mapping.tensor(3), None);
        assert_eq!(slice_groups, [-1; 4]);
    }

    #[test]
    fn test_tensor_mapping_one_slicing_group() {
        let tn = setup_tn();

        let path = path![
            (0, [(0, 2), (0, 1)]),
            (2, [(0, 1)]),
            (1, [6], [(0, 1), (0, 1)]),
            (1, 0),
            (2, 1)
        ];
        // final tensor is 2 -> should end up on rank 0

        let (tensor_mapping, slice_groups, used_ranks) =
            get_tensor_mapping_and_slice_groups(&tn, path, 5);

        assert_eq!(used_ranks, 4);
        assert_eq!(slice_groups, [-1, -1, 0, 0, -1]);
        assert_eq!(tensor_mapping.rank(0), 1);
        assert_eq!(tensor_mapping.rank(1), 2);
        assert_eq!(tensor_mapping.rank(2), 0);
        assert_eq!(tensor_mapping.tensor(0), Some(2));
        assert_eq!(tensor_mapping.tensor(1), Some(0));
        assert_eq!(tensor_mapping.tensor(2), Some(1));
        assert_eq!(tensor_mapping.tensor(3), None);
        assert_eq!(tensor_mapping.tensor(4), None);
    }

    #[test]
    fn test_tensor_mapping_two_slicing_groups() {
        let tn = setup_tn();

        let path = path![
            (0, [1], [(0, 2), (0, 1)]),
            (2, [(0, 1)]),
            (1, [6], [(0, 1), (0, 1)]),
            (1, 0),
            (0, 2)
        ];

        let (tensor_mapping, slice_groups, used_ranks) =
            get_tensor_mapping_and_slice_groups(&tn, path, 6);

        assert_eq!(used_ranks, 6);
        assert_eq!(slice_groups, [0, 0, 0, -1, 1, 1]);
        assert_eq!(tensor_mapping.rank(0), 0);
        assert_eq!(tensor_mapping.rank(1), 4);
        assert_eq!(tensor_mapping.rank(2), 3);
        assert_eq!(tensor_mapping.tensor(0), Some(0));
        assert_eq!(tensor_mapping.tensor(1), None);
        assert_eq!(tensor_mapping.tensor(2), None);
        assert_eq!(tensor_mapping.tensor(3), Some(2));
        assert_eq!(tensor_mapping.tensor(4), Some(1));
        assert_eq!(tensor_mapping.tensor(5), None);
    }
}
