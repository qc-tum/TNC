use std::sync::{Arc, RwLock};

use log::{debug, warn};
use mpi::topology::{Process, SimpleCommunicator};
use mpi::traits::{BufferMut, Communicator, Destination, Root, Source};
use mpi::Rank;
use rustc_hash::FxHashSet;

use super::mpi_types::BondDim;
use crate::mpi::mpi_types::MessageBinaryBlob;
use crate::tensornetwork::contraction::contract_tensor_network;
use crate::tensornetwork::tensor::Tensor;
use crate::tensornetwork::tensordata::TensorData;
use crate::types::{ContractionIndex, EdgeIndex};

/// Serializes data to a byte array.
fn serialize<S>(value: &S) -> Vec<u8>
where
    S: serde::Serialize,
{
    bincode::serialize(value).unwrap()
}

/// Serializes data into a writer.
fn serialize_into<W, S>(writer: W, value: &S)
where
    W: std::io::Write,
    S: serde::Serialize,
{
    bincode::serialize_into(writer, value).unwrap();
}

/// Returns the serialized size of the data (i.e., the number of bytes).
fn serialized_size<S>(value: &S) -> u64
where
    S: serde::Serialize,
{
    bincode::serialized_size(value).unwrap()
}

/// Deserializes data from a byte array.
fn deserialize<D>(data: &[u8]) -> D
where
    D: serde::de::DeserializeOwned,
{
    bincode::deserialize(data).unwrap()
}

/// Deserializes data from a reader.
fn deserialize_from<R, D>(reader: R) -> D
where
    R: std::io::Read,
    D: serde::de::DeserializeOwned,
{
    bincode::deserialize_from(reader).unwrap()
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
pub fn broadcast_path(
    path: &mut Vec<ContractionIndex>,
    root: &Process,
    world: &SimpleCommunicator,
) {
    debug!(root=root.rank(), rank=world.rank(), path:serde; "Broadcasting path");

    // Serialize path
    let mut data = if world.rank() == root.rank() {
        serialize(&path)
    } else {
        Default::default()
    };

    // Broadcast data
    broadcast_vec(&mut data, root, world);

    // Deserialize path
    if world.rank() != root.rank() {
        *path = deserialize(&data);
    }

    debug!(path:serde; "Received broadcasted path");
}

/// Sends the leaf tensor `tensor` to `receiver` via MPI.
fn send_leaf_tensor(tensor: &Tensor, receiver: Rank, world: &SimpleCommunicator) {
    assert!(tensor.is_leaf());

    // MPI uses an i32 to store the number of elements in a buffer. This means, we
    // can send at most `i32::MAX * sizeof(datatype)`. Hence, if we only use byte
    // arrays, we can send at most `i32::MAX` bytes (~2GB) which is not enough.
    // Instead, we interpret the byte arrays as arrays of a artifical, larger data
    // type, which allows us to send more bytes.

    let legs = tensor.legs();
    let tensor_data = tensor.tensor_data();

    // Get the total serialized size
    let size = serialized_size(&legs) + serialized_size(&*tensor_data);
    let size: usize = size.try_into().unwrap();

    // Allocate a buffer of blobs
    let element_size = std::mem::size_of::<MessageBinaryBlob>();
    let elements = size.div_ceil(element_size);
    let mut buffer = Vec::<MessageBinaryBlob>::with_capacity(elements);

    // Get a bytes view of the buffer
    let mut write_view = unsafe {
        std::slice::from_raw_parts_mut(
            buffer.as_mut_ptr() as *mut u8,
            buffer.capacity() * element_size,
        )
    };

    // Serialize legs and data into the buffer
    serialize_into(&mut write_view, legs);
    serialize_into(&mut write_view, &*tensor_data);
    unsafe { buffer.set_len(buffer.capacity()) };

    // Send the buffer
    world.process_at_rank(receiver).send(&buffer);
}

/// Receives a leaf tensor from `sender` via MPI.
fn receive_leaf_tensor(sender: Rank, world: &SimpleCommunicator) -> Tensor {
    // Receive the buffer
    let (buffer, _status) = world
        .process_at_rank(sender)
        .receive_vec::<MessageBinaryBlob>();

    // Get a bytes view of the buffer
    let element_size = std::mem::size_of::<MessageBinaryBlob>();
    let mut read_buffer = unsafe {
        std::slice::from_raw_parts(buffer.as_ptr() as *const u8, buffer.len() * element_size)
    };

    // Deserialize legs and data
    let legs: Vec<EdgeIndex> = deserialize_from(&mut read_buffer);
    let tensor_data: TensorData = deserialize_from(&mut read_buffer);

    // Create tensor
    let mut new_tensor = Tensor::new(legs);
    new_tensor.set_tensor_data(tensor_data);
    new_tensor
}

/// Returns the ranks that don't have any local contractions.
fn get_idle_ranks(path: &[ContractionIndex], size: Rank) -> FxHashSet<Rank> {
    let mut idle_ranks = (0..size).collect::<FxHashSet<_>>();
    for pair in path {
        if let ContractionIndex::Path(i, _) = pair {
            idle_ranks.remove(&(*i as Rank));
        }
    }
    idle_ranks
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
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    broadcast_vec(&mut bond_vec, &root_process, world);
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
            if let ContractionIndex::Path(i, local) = contraction_path {
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
            // Defining the tensor count as `1 + number of subtensors` discriminates
            // between no tensor (0), leaf tensor (1) and composite tensor (2..).
            let num_subtensors = tensor.tensors().len();
            let tensor_count = num_subtensors + 1;
            debug!(receiver = i, tensor_count; "Sending tensor count");
            world.process_at_rank(i as Rank).send(&tensor_count);
            debug!(receiver = i; "Sending tensor(s)");
            if tensor.is_leaf() {
                debug!(receiver = i, tensor:?; "Sending leaf tensor");
                send_leaf_tensor(tensor, i as Rank, world);
            } else {
                debug!(receiver = i, tensor:?; "Sending composite tensor");
                for inner_tensor in tensor.tensors() {
                    send_leaf_tensor(inner_tensor, i as Rank, world);
                }
            }
        }
        // Send zero tensors to non-participating ranks
        let used_ranks = r_tn.tensors().len() as Rank;
        for i in used_ranks..size {
            debug!(receiver = i; "Sending zero tensor count");
            world.process_at_rank(i).send(&0usize);
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
        debug!(sender = 0; "Receiving tensor count");
        let (tensor_count, _status) = world.process_at_rank(0).receive::<usize>();
        let mut local_tn = Tensor::default();
        debug!(sender = 0, tensor_count; "Receiving tensors");
        match tensor_count {
            // No tensor
            0 => (),
            // Leaf tensor
            1 => {
                let new_tensor = receive_leaf_tensor(0, world);
                local_tn = new_tensor;
                local_tn.bond_dims = Arc::new(RwLock::new(bond_dims));
            }
            // Composite tensor
            _ => {
                for _ in 0..tensor_count - 1 {
                    let new_tensor = receive_leaf_tensor(0, world);
                    local_tn.push_tensor(new_tensor, Some(&bond_dims));
                }
                // TODO: can use `push_tensors` to push all  at once?
            }
        }
        debug!("Received all tensors");
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
                    let received_tensor = receive_leaf_tensor(sender, world);
                    debug!(sender; "Finish receiving tensor");
                    local_tn.push_tensor(received_tensor, None);

                    // Contract tensors
                    contract_tensor_network(local_tn, &[ContractionIndex::Pair(0, 1)]);
                }
                if sender == rank {
                    debug!(receiver; "Start sending tensor");
                    send_leaf_tensor(local_tn, receiver, world);
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
            let received_tensor = receive_leaf_tensor(final_rank, world);
            *local_tn = received_tensor;
        }
        if rank == final_rank {
            debug!(receiver = 0; "Sending final tensor");
            send_leaf_tensor(local_tn, 0, world);
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
            let received_tensor = receive_leaf_tensor(i, world);
            local_tn.push_tensor(received_tensor, None);
        }
    } else {
        debug!(receiver = 0; "Sending tensor");
        send_leaf_tensor(local_tn, 0, world);
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
    use crate::path;

    #[test]
    fn test_idle_ranks() {
        let path = vec![
            ContractionIndex::Path(2, vec![ContractionIndex::Pair(0, 1)]),
            ContractionIndex::Pair(0, 2),
            ContractionIndex::Pair(1, 3),
            ContractionIndex::Pair(0, 1),
        ];
        let idle_ranks = get_idle_ranks(&path, 6);
        assert_eq!(idle_ranks, [0, 1, 3, 4, 5].into_iter().collect());
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
        broadcast_path(&mut contraction_indices, &root_process, &world);

        assert_eq!(contraction_indices, ref_contraction_indices);
    }
}
