use std::iter::zip;

use mpi::topology::{Process, SimpleCommunicator};
use mpi::traits::{BufferMut, Communicator, Destination, Root, Source};
use mpi::Rank;

use super::mpi_types::BondDim;
use crate::tensornetwork::contraction::contract_tensor_network;
use crate::tensornetwork::tensor::Tensor;
use crate::tensornetwork::tensordata::TensorData;
use crate::types::{ContractionIndex, EdgeIndex};

/// Serializes data to a byte array.
fn serialize<S>(data: &S) -> Vec<u8>
where
    S: serde::Serialize,
{
    bincode::serialize(data).unwrap()
}

/// Deserializes data from a byte array.
fn deserialize<D>(data: &[u8]) -> D
where
    D: serde::de::DeserializeOwned,
{
    bincode::deserialize(data).unwrap()
}

pub enum CommunicationScheme {
    Greedy,
    Bipartition,
}

/// Broadcasts a vector of `data` from `root` to all processes in `world`. For the
/// receivers, `data` can just be an empty vector.
fn broadcast_vec<T>(data: &mut Vec<T>, root: &Process, world: &SimpleCommunicator)
where
    T: Clone + Default,
    Vec<T>: BufferMut,
{
    // Broadcast length
    let mut len = if world.rank() == root.rank() {
        data.len()
    } else {
        0
    };
    root.broadcast_into(&mut len);

    // Broadcast data
    if world.rank() != root.rank() {
        data.resize(len, Default::default());
    }
    root.broadcast_into(data);
}

/// Broadcast a contraction index `path` from `root` to all processes in `world`. For
/// the receivers, `path` can just be an empty slice.
#[must_use]
pub fn broadcast_path(
    path: &[ContractionIndex],
    root: &Process,
    world: &SimpleCommunicator,
) -> Vec<ContractionIndex> {
    // Serialize path
    let mut data = if world.rank() == root.rank() {
        serialize(&path)
    } else {
        Default::default()
    };

    // Broadcast data
    broadcast_vec(&mut data, root, world);

    // Deserialize path
    if world.rank() == root.rank() {
        path.to_vec()
    } else {
        deserialize(&data)
    }
}

/// Sends the leaf tensor `tensor` to `receiver` via MPI.
fn send_leaf_tensor(tensor: &Tensor, receiver: Rank, world: &SimpleCommunicator) {
    assert!(tensor.is_leaf());

    // Send legs
    let legs = tensor.legs();
    world.process_at_rank(receiver).send(legs);

    // Send data
    let tensor_data = tensor.tensor_data();
    world
        .process_at_rank(receiver)
        .send(&serialize(&*tensor_data));
}

/// Receives a leaf tensor from `sender` via MPI.
fn receive_leaf_tensor(sender: Rank, world: &SimpleCommunicator) -> Tensor {
    // Receive legs
    let (legs, _status) = world.process_at_rank(sender).receive_vec::<EdgeIndex>();

    // Receive data
    let (raw_data, _status) = world.process_at_rank(sender).receive_vec::<u8>();
    let tensor_data: TensorData = deserialize(&raw_data);

    // Create tensor
    let mut new_tensor = Tensor::new(legs);
    new_tensor.set_tensor_data(tensor_data);
    new_tensor
}

/// Partitions input Tensor `r_tn` into `size` partitions using `KaHyPar` and distributes the partitions to the various processes via `rank` via MPI.
pub fn scatter_tensor_network(
    r_tn: &Tensor,
    path: &[ContractionIndex],
    rank: Rank,
    size: Rank,
    world: &SimpleCommunicator,
) -> (Tensor, Vec<ContractionIndex>) {
    let root_process = world.process_at_rank(0);

    // Distribute bond_dims
    let mut bond_vec = if rank == 0 {
        r_tn.bond_dims()
            .iter()
            .map(|(&bond_id, &bond_size)| BondDim { bond_id, bond_size })
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    broadcast_vec(&mut bond_vec, &root_process, world);
    let bond_dims = bond_vec.iter().map(|e| (e.bond_id, e.bond_size)).collect();

    // Distribute tensors
    let (local_tn, local_path) = if rank == 0 {
        let local_path = path[0].clone().get_data();
        let local_tn = r_tn.tensor(0).clone();

        // Send the local paths to the other processes
        for (i, contraction_path) in zip(1..size, path[1..size as usize].iter()) {
            match contraction_path {
                ContractionIndex::Path(_, local) => {
                    world.process_at_rank(i).send(&serialize(&local));
                }
                ContractionIndex::Pair(_, _) => panic!("Requires path"),
            }
        }

        // Send the tensors to the other processes
        for (i, tensor) in zip(1..size, r_tn.tensors()[1..size as usize].iter()) {
            let num_tensors = tensor.tensors().len();
            world.process_at_rank(i).send(&num_tensors);
            for inner_tensor in tensor.tensors() {
                send_leaf_tensor(inner_tensor, i, world);
            }
        }
        (local_tn, local_path)
    } else {
        // Receive local path
        let (raw_path, _status) = world.any_process().receive_vec::<u8>();
        let local_path = deserialize(&raw_path);

        // Receive tensors
        let mut local_tn = Tensor::default();
        let (num_tensors, _status) = world.any_process().receive::<usize>();
        for _ in 0..num_tensors {
            let new_tensor = receive_leaf_tensor(0, world);
            local_tn.push_tensor(new_tensor, Some(&bond_dims));
        }
        (local_tn, local_path)
    };
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
    let mut final_rank = 0;
    for pair in path {
        match pair {
            ContractionIndex::Pair(x, y) => {
                let receiver: Rank = (*x).try_into().unwrap();
                let sender: Rank = (*y).try_into().unwrap();
                final_rank = receiver;
                if receiver == rank {
                    // Insert received tensor into local tensor
                    let received_tensor = receive_leaf_tensor(sender, world);
                    local_tn.push_tensor(received_tensor, None);

                    // Contract tensors
                    contract_tensor_network(local_tn, &[ContractionIndex::Pair(0, 1)]);
                }
                if sender == rank {
                    send_leaf_tensor(local_tn, receiver, world);
                }
            }
            ContractionIndex::Path(..) => panic!("Requires pair"),
        }
    }

    // Only runs if the final contracted process is not process 0
    if final_rank != 0 {
        if rank == 0 {
            let received_tensor = receive_leaf_tensor(final_rank, world);
            *local_tn = received_tensor;
        }
        if rank == final_rank {
            send_leaf_tensor(local_tn, 0, world);
        }
    }
}

/// Sends all tensors to the root process before contracting all tensors.
pub fn naive_reduce_tensor_network(
    local_tn: &mut Tensor,
    path: &[ContractionIndex],
    rank: Rank,
    size: Rank,
    world: &SimpleCommunicator,
) {
    if rank == 0 {
        for i in 1..size {
            // Add received tensor to final tensor network
            let received_tensor = receive_leaf_tensor(i, world);
            local_tn.push_tensor(received_tensor, None);
        }
    } else {
        send_leaf_tensor(local_tn, 0, world);
    }

    if rank == 0 {
        // Contract the final tensor network
        contract_tensor_network(local_tn, &path[(size as usize)..path.len()]);
    }
}

#[cfg(test)]
mod tests {
    use mpi::traits::Communicator;
    use mpi_test::mpi_test;

    use super::*;
    use crate::path;

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

        let contraction_indices = if rank == 0 {
            let contraction_indices = &ref_contraction_indices;
            broadcast_path(contraction_indices, &root_process, &world)
        } else {
            broadcast_path(&[], &root_process, &world)
        };

        assert_eq!(contraction_indices, ref_contraction_indices);
    }
}
