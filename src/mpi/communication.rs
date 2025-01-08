use itertools::Itertools;
use log::{debug, warn};
use mpi::topology::{Color, Process, SimpleCommunicator};
use mpi::traits::{BufferMut, Communicator, Destination, Equivalence, Group, Root, Source};
use mpi::Rank;
use rustc_hash::{FxHashMap, FxHashSet};

use super::mpi_types::{BondDim, OptionalTensorIndex};
use super::serialization::{deserialize_tensor, serialize_tensor};
use crate::mpi::mpi_types::MessageBinaryBlob;
use crate::mpi::serialization::{deserialize, serialize};
use crate::tensornetwork::contraction::contract_tensor_network;
use crate::tensornetwork::tensor::Tensor;
use crate::types::{ContractionIndex, EdgeIndex, TensorIndex};

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

/// Sends the `tensor` to `receiver` via MPI.
fn send_tensor(tensor: &Tensor, receiver: Rank, world: &SimpleCommunicator) {
    let data = serialize_tensor(tensor);
    world.process_at_rank(receiver).send(&data);
}

/// Sends a MPI message to `receiver` that signals that no tensors are being sent,
/// i.e., that `receiver` doesn't contribute in the contraction.
fn send_no_tensor(receiver: Rank, world: &SimpleCommunicator) {
    let empty = Vec::<MessageBinaryBlob>::new();
    world.process_at_rank(receiver).send(&empty);
}

/// Receives a leaf tensor from `sender` via MPI.
fn receive_tensor(
    sender: Rank,
    world: &SimpleCommunicator,
    bond_dims: Option<&FxHashMap<EdgeIndex, u64>>,
) -> Tensor {
    // Receive the buffer
    let (data, _status) = world
        .process_at_rank(sender)
        .receive_vec::<MessageBinaryBlob>();

    if !data.is_empty() {
        deserialize_tensor(&data, bond_dims)
    } else {
        Tensor::default()
    }
}

/// Returns the ranks that don't have any local contractions.
fn get_idle_ranks(path: &[ContractionIndex], size: Rank) -> FxHashSet<Rank> {
    let mut idle_ranks = (0..size).collect::<FxHashSet<_>>();
    for pair in path {
        if let ContractionIndex::Path(i, _, _) = pair {
            idle_ranks.remove(&(*i as Rank));
        }
    }
    idle_ranks
}

/// A bidirectional mapping between MPI ranks and composite tensors.
#[derive(Debug, Clone, Default)]
struct RankTensorMapping {
    rank_to_tensor: Vec<OptionalTensorIndex>,
    tensor_to_rank: FxHashMap<TensorIndex, Rank>,
}

impl RankTensorMapping {
    fn new(ranks: Rank) -> Self {
        Self {
            rank_to_tensor: vec![OptionalTensorIndex::default(); ranks as usize],
            tensor_to_rank: Default::default(),
        }
    }

    fn add(&mut self, rank: Rank, tensor: TensorIndex) {
        self.rank_to_tensor[rank as usize] = OptionalTensorIndex::new(tensor);
        self.tensor_to_rank.insert(tensor, rank);
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
    let mut tensor_mapping = RankTensorMapping::new(size);
    let mut slice_groups = vec![-1; size as usize];
    let mut used_groups = 0;

    for pair in path {
        if let ContractionIndex::Path(i, slicing, _) = pair {
            // Assign the next available rank to tensor `i`
            tensor_mapping.add(used_ranks, *i);

            if let Some(slicing) = slicing {
                // Determine how many ranks are needed for doing all slices in parallel
                let local_tensor = r_tn.tensor(*i);
                let needed_ranks: i32 = slicing.size(local_tensor).try_into().unwrap();

                // Assign each of the needed ranks the same color
                for rank in used_ranks..used_ranks + needed_ranks {
                    slice_groups[rank as usize] = used_groups;
                }
                used_groups += 1;
                used_ranks += needed_ranks;
            } else {
                used_ranks += 1;
            }
        }
    }
    assert!(
        used_ranks < size,
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
    /// The tensor that the rank is responsible for. In a slice group, only the group
    /// root has this set.
    assigned_tensor: Option<TensorIndex>,

    /// The communicator for the slice group. `None` if the rank is not part of a
    /// slice group.
    slice_comm: Option<SimpleCommunicator>,
}

pub fn scatter_tensor_network2(
    r_tn: &Tensor,
    path: &[ContractionIndex],
    rank: Rank,
    size: Rank,
    world: &SimpleCommunicator,
) -> (Tensor, Vec<ContractionIndex>, Communication) {
    let root = world.process_at_rank(0);

    // Get information about used ranks
    let (tensor_mapping, slicing_grouping, _) = if rank == 0 {
        get_tensor_mapping_and_slice_groups(r_tn, path, size)
    } else {
        Default::default()
    };

    // Tell the ranks the tensor they are responsible for (if any)
    let assigned_tensor = scatter_slice(&tensor_mapping.rank_to_tensor, &root);
    let assigned_tensor: Option<TensorIndex> = assigned_tensor.into();

    // Send the slicing information
    let slice_group = scatter_slice(&slicing_grouping, &root);

    // Create communicators for the slice groups
    let color = if slice_group >= 0 {
        Color::with_value(slice_group)
    } else {
        Color::undefined()
    };
    let slice_comm = world.split_by_color(color);

    // Send the bond dimensions
    let mut bond_vec = if rank == 0 {
        r_tn.bond_dims()
            .iter()
            .map(|(&bond_id, &bond_size)| BondDim { bond_id, bond_size })
            .collect_vec()
    } else {
        Default::default()
    };
    broadcast_vec(&mut bond_vec, &root);
    let bond_dims: FxHashMap<usize, u64> =
        bond_vec.iter().map(|e| (e.bond_id, e.bond_size)).collect();

    // Send the local paths
    let mut local_path_raw = if rank == 0 {
        let mut local_path = None;
        for contraction_path in path {
            if let ContractionIndex::Path(i, _, local) = contraction_path {
                // TODO: send slicing information as well
                let target_rank = tensor_mapping.tensor_to_rank[i];
                if target_rank == 0 {
                    // This is the path for the root, no need to send it
                    local_path = Some(local.clone());
                    continue;
                }

                world.process_at_rank(target_rank).send(&serialize(&local));
            }
        }
        serialize(&local_path.unwrap())
    } else if assigned_tensor.is_some() {
        let (raw_path, _status) = world.process_at_rank(0).receive_vec::<u8>();
        raw_path
    } else {
        Default::default()
    };

    // Broadcast the path in the slicing group
    if let Some(slice_comm) = &slice_comm {
        broadcast_vec(&mut local_path_raw, &slice_comm.process_at_rank(0));
    }

    // Deserialize the path
    let local_path = if !local_path_raw.is_empty() {
        deserialize(&local_path_raw)
    } else {
        Default::default()
    };

    // Send the tensors
    let mut local_tn_raw = if rank == 0 {
        let mut local_tn = None;
        for (tensor_index, target_rank) in tensor_mapping.tensor_to_rank {
            let tensor = r_tn.tensor(tensor_index);
            if target_rank == 0 {
                // This is the tensor for the root, no need to send it
                local_tn = Some(tensor.clone());
                continue;
            }

            send_tensor(tensor, target_rank, world);
        }
        serialize_tensor(&local_tn.unwrap())
    } else if assigned_tensor.is_some() {
        let (raw_tensor, _status) = world.process_at_rank(0).receive_vec::<MessageBinaryBlob>();
        raw_tensor
    } else {
        Default::default()
    };

    // Broadcast the tensor in the slicing group
    if let Some(slice_comm) = &slice_comm {
        broadcast_vec(&mut local_tn_raw, &slice_comm.process_at_rank(0));
    }

    // Deserialize the tensor
    let local_tn = if !local_tn_raw.is_empty() {
        deserialize_tensor(&local_tn_raw, Some(&bond_dims))
    } else {
        Default::default()
    };

    // Return the local tensor, path and communication information
    (
        local_tn,
        local_path,
        Communication {
            assigned_tensor,
            slice_comm,
        },
    )
}

/// Distributes the partitioned tensor network to the various processes via MPI.
pub fn scatter_tensor_network(
    r_tn: &Tensor,
    path: &[ContractionIndex],
    rank: Rank,
    size: Rank,
    world: &SimpleCommunicator,
) -> (Tensor, Vec<ContractionIndex>) {
    debug!(rank, size, path:serde; "Scattering tensor network");
    let root_process = world.process_at_rank(0);

    // Distribute bond_dims
    debug!(bond_dims:serde = *r_tn.bond_dims(); "Distributing bond dimensions");
    let mut bond_vec = if rank == 0 {
        r_tn.bond_dims()
            .iter()
            .map(|(&bond_id, &bond_size)| BondDim { bond_id, bond_size })
            .collect_vec()
    } else {
        Vec::new()
    };
    broadcast_vec(&mut bond_vec, &root_process);
    let bond_dims = bond_vec.iter().map(|e| (e.bond_id, e.bond_size)).collect();

    // Distribute tensors
    debug!("Distributing paths and tensors");
    let (local_tn, local_path) = if rank == 0 {
        // Get the idle ranks
        let idle_ranks = get_idle_ranks(path, size);
        if !idle_ranks.is_empty() {
            warn!(idle_ranks:serde; "There are idle ranks");
        }

        // Send the local paths to the other processes
        let mut local_path = Vec::new();
        for contraction_path in path {
            if let ContractionIndex::Path(i, _, local) = contraction_path {
                if *i == 0 {
                    // This is the path for the root process, no need to send it
                    local_path = local.clone();
                } else {
                    debug!(receiver = i, local:serde = local; "Sending local path");
                    world.process_at_rank(*i as Rank).send(&serialize(&local));
                }
            }
        }
        // Send empty paths to non-participating ranks
        for i in &idle_ranks {
            debug!(receiver = i; "Sending empty local path");
            world
                .process_at_rank(*i)
                .send(&serialize(&Vec::<ContractionIndex>::new()));
        }
        debug!("Sent all local paths");

        // Send the tensors to the other processes
        let local_tn = r_tn.tensor(0).clone();
        for (i, tensor) in r_tn.tensors().iter().enumerate().skip(1) {
            debug!(receiver = i; "Sending tensor");
            send_tensor(tensor, i as Rank, world);
        }
        // Send zero tensors to non-participating ranks
        let used_ranks = r_tn.tensors().len() as Rank;
        for i in used_ranks..size {
            debug!(receiver = i; "Sending no-tensor flag to non-participating rank");
            send_no_tensor(i, world);
        }
        debug!("Sent all tensors");
        (local_tn, local_path)
    } else {
        // Receive local path
        debug!(sender = 0; "Receiving local path");
        let (raw_path, _status) = world.process_at_rank(0).receive_vec::<u8>();
        let local_path = deserialize(&raw_path);
        debug!(local_path:serde; "Received local path");

        // Receive tensors
        debug!(sender = 0; "Receiving tensor");
        let local_tn = receive_tensor(0, world, Some(&bond_dims));
        debug!("Received tensor");
        (local_tn, local_path)
    };
    debug!("Scattered tensor network");
    (local_tn, local_path)
}

/// Uses the `path` as a communication blueprint to iteratively send tensors and contract them in a fan-in.
/// Assumes that `path` is a valid contraction path.
pub fn intermediate_reduce_tensor_network(
    local_tn: &mut Tensor,
    path: &[ContractionIndex],
    rank: Rank,
    world: &SimpleCommunicator,
) {
    debug!(rank, path:serde; "Reducing tensor network (intermediate)");
    let mut final_rank = 0;
    for pair in path {
        match pair {
            ContractionIndex::Pair(x, y) => {
                let receiver = *x as Rank;
                let sender = *y as Rank;
                final_rank = receiver;
                if receiver == rank {
                    // Insert received tensor into local tensor
                    debug!(sender; "Start receiving tensor");
                    let received_tensor = receive_tensor(sender, world, None);
                    debug!(sender; "Finish receiving tensor");
                    local_tn.push_tensor(received_tensor, None);

                    // Contract tensors
                    contract_tensor_network(local_tn, &[ContractionIndex::Pair(0, 1)]);
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
            let received_tensor = receive_tensor(final_rank, world, None);
            *local_tn = received_tensor;
        }
        if rank == final_rank {
            debug!(receiver = 0; "Sending final tensor");
            send_tensor(local_tn, 0, world);
        }
    }
    debug!("Reduced tensor network");
}

/// Sends all tensors to the root process before contracting all tensors.
pub fn naive_reduce_tensor_network(
    local_tn: &mut Tensor,
    path: &[ContractionIndex],
    rank: Rank,
    size: Rank,
    world: &SimpleCommunicator,
) {
    debug!(rank, size, path:serde; "Reducing tensor network (naive)");
    if rank == 0 {
        for i in 1..size {
            // Add received tensor to final tensor network
            debug!(sender = i; "Receiving tensor");
            let received_tensor = receive_tensor(i, world, None);
            local_tn.push_tensor(received_tensor, None);
        }
    } else {
        debug!(receiver = 0; "Sending tensor");
        send_tensor(local_tn, 0, world);
    }

    if rank == 0 {
        // Contract the final tensor network
        contract_tensor_network(local_tn, &path[(size as usize)..path.len()]);
    }
    debug!("Reduced tensor network");
}

#[cfg(test)]
mod tests {
    use mpi::traits::Communicator;
    use mpi_test::mpi_test;

    use super::*;
    use crate::{path, types::Slicing};

    #[test]
    fn test_idle_ranks() {
        let path = path![(2, [(0, 1)]), (0, 2), (1, 3), (0, 1)];
        let idle_ranks = get_idle_ranks(path, 6);
        assert_eq!(idle_ranks, [0, 1, 3, 4, 5].into_iter().collect());
    }

    #[mpi_test(8)]
    fn test_my_scatter() {
        let universe = mpi::initialize().unwrap();
        let world = universe.world();
        let rank = world.rank();
        let size = world.size();
        let (tn, path) = if rank == 0 {
            let bond_dims = FxHashMap::from_iter([(1, 2), (2, 2), (3, 2), (4, 2), (5, 2), (6, 2)]);
            let mut ta = Tensor::default();
            let t2 = Tensor::new(vec![1, 2, 3]);
            let t3 = Tensor::new(vec![2, 3, 4]);
            let t4 = Tensor::new(vec![4, 5]);
            ta.push_tensors(vec![t2, t3, t4], Some(&bond_dims), None);
            let mut tb = Tensor::default();
            let t5 = Tensor::new(vec![5, 6]);
            let t6 = Tensor::new(vec![6]);
            tb.push_tensors(vec![t5, t6], Some(&bond_dims), None);
            let mut tc = Tensor::default();
            tc.push_tensors(vec![ta, tb], Some(&bond_dims), None);

            let path = vec![
                ContractionIndex::Path(
                    0,
                    Some(Slicing { slices: vec![2] }),
                    vec![ContractionIndex::Pair(0, 1), ContractionIndex::Pair(0, 2)],
                ),
                ContractionIndex::Path(1, None, vec![ContractionIndex::Pair(0, 1)]),
                ContractionIndex::Pair(0, 1),
            ];

            (tc, path)
        } else {
            Default::default()
        };

        scatter_tensor_network2(&tn, &path, rank, size, &world);
    }

    #[mpi_test(2)]
    fn test_sendrecv_contraction_index() {
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
}
