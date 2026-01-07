use log::debug;
use mpi::topology::{Process, SimpleCommunicator};
use mpi::traits::{BufferMut, Communicator, Destination, Root, Source};
use mpi::Rank;

use crate::contractionpath::ContractionIndex;
use crate::mpi::mpi_types::{MessageBinaryBlob, RankTensorMapping};
use crate::mpi::serialization::{deserialize, deserialize_tensor, serialize, serialize_tensor};
use crate::tensornetwork::contraction::contract_tensor_network;
use crate::tensornetwork::tensor::Tensor;

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
pub fn broadcast_serializing<T>(data: T, root: &Process) -> T
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

    deserialize_tensor(&data)
}

/// Determines the tensor mapping for the given contraction `path`.
/// Also returns the number of used ranks.
fn get_tensor_mapping(path: &[ContractionIndex], size: Rank) -> RankTensorMapping {
    let mut tensor_mapping = RankTensorMapping::with_capacity(size as usize);

    let Some(last) = path.last() else {
        // Empty path
        return tensor_mapping;
    };
    let &ContractionIndex::Pair(final_tensor, _) = last else {
        panic!("Last part of path should be a pair")
    };

    // Reserve rank 0 for the final tensor
    let mut used_ranks = 1;
    for pair in path {
        if let ContractionIndex::Path(i, _) = pair {
            if *i == final_tensor {
                // Assign the final tensor to rank 0
                tensor_mapping.insert(0, *i);
            } else {
                // Assign the next available rank to tensor `i`
                tensor_mapping.insert(used_ranks, *i);
                used_ranks += 1;
            }
        }
    }
    assert!(
        used_ranks <= size,
        "Not enough MPI ranks available, got {size} but need {used_ranks}!"
    );
    tensor_mapping
}

/// Information needed for communication during contraction of the tensor network.
pub struct Communication {
    /// A mapping between MPI ranks and their owned composite tensors. In slice
    /// groups, only the slice root rank is assigned the tensor.
    tensor_mapping: RankTensorMapping,
}

/// Distributes the partitioned tensor network to the various processes via MPI.
pub fn scatter_tensor_network(
    r_tn: &Tensor,
    path: &[ContractionIndex],
    rank: Rank,
    size: Rank,
    world: &SimpleCommunicator,
) -> (Tensor, Vec<ContractionIndex>, Communication) {
    debug!(rank, size; "Scattering tensor network");
    let root = world.process_at_rank(0);

    // Get information about used ranks
    let tensor_mapping = if rank == 0 {
        get_tensor_mapping(path, size)
    } else {
        Default::default()
    };

    // Tell the ranks the tensor they are responsible for (if any)
    let tensor_mapping = broadcast_serializing(tensor_mapping, &root);
    let is_tensor_owner = tensor_mapping.tensor(rank).is_some();
    debug!(tensor_mapping:serde, is_tensor_owner; "Scattered organizational data");

    // Send the local paths
    let local_path = if rank == 0 {
        debug!("Sending local paths");
        let mut local_path = None;
        for contraction_path in path {
            if let ContractionIndex::Path(i, local) = contraction_path {
                let target_rank = tensor_mapping.rank(*i);
                if target_rank == 0 {
                    // This is the path for the root, no need to send it
                    local_path = Some(local.clone());
                    continue;
                }

                world.process_at_rank(target_rank).send(&serialize(&local));
            }
        }
        local_path.unwrap()
    } else if is_tensor_owner {
        debug!("Receiving local path");
        let (raw_path, _status) = world.process_at_rank(0).receive_vec::<u8>();
        deserialize(&raw_path)
    } else {
        Default::default()
    };

    // Send the tensors
    let local_tn = if rank == 0 {
        debug!("Sending tensors");
        let mut local_tn = None;
        for &(target_rank, tensor_index) in &tensor_mapping {
            let tensor = r_tn.tensor(tensor_index);
            if target_rank == 0 {
                // This is the tensor for the root, no need to send it
                local_tn = Some(tensor.clone());
                continue;
            }

            send_tensor(tensor, target_rank, world);
        }
        local_tn.unwrap()
    } else if is_tensor_owner {
        debug!("Receiving tensor");
        receive_tensor(0, world)
    } else {
        Default::default()
    };
    debug!("Scattered tensor network");

    // Return the local tensor, path and communication information
    (local_tn, local_path, Communication { tensor_mapping })
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
    use super::*;

    use std::sync::Mutex;

    use mpi::traits::Communicator;
    use mpi_test::mpi_test;

    use crate::path;

    static MPI_SERIAL_TEST_LOCK: Mutex<()> = Mutex::new(());

    #[mpi_test(2)]
    fn test_broadcast_contraction_path() {
        let _lock = MPI_SERIAL_TEST_LOCK.lock().unwrap();
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
    fn test_tensor_mapping() {
        let path = path![
            (0, [(0, 2), (0, 1)]),
            (2, [(0, 1)]),
            (1, [(0, 1), (0, 1)]),
            (0, 2),
            (0, 1)
        ];

        let tensor_mapping = get_tensor_mapping(path, 4);

        assert_eq!(tensor_mapping.len(), 3);
        assert_eq!(tensor_mapping.rank(0), 0);
        assert_eq!(tensor_mapping.rank(1), 2);
        assert_eq!(tensor_mapping.rank(2), 1);
        assert_eq!(tensor_mapping.tensor(0), Some(0));
        assert_eq!(tensor_mapping.tensor(1), Some(2));
        assert_eq!(tensor_mapping.tensor(2), Some(1));
        assert_eq!(tensor_mapping.tensor(3), None);
    }
}
