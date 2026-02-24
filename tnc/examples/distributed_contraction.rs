use mpi::{topology::SimpleCommunicator, traits::Communicator};
use rand::{rngs::StdRng, SeedableRng};
use tnc::{
    builders::sycamore_circuit::sycamore_circuit,
    contractionpath::paths::{
        cotengrust::{Cotengrust, OptMethod},
        FindPath,
    },
    mpi::communication::{
        broadcast_path, intermediate_reduce_tensor_network, scatter_tensor_network,
    },
    tensornetwork::{
        contraction::contract_tensor_network,
        partitioning::{
            find_partitioning, partition_config::PartitioningStrategy, partition_tensor_network,
        },
        tensor::Tensor,
    },
};

fn main() {
    // Read from file
    let qubits = 10;
    let mut rng = StdRng::seed_from_u64(0);
    let circuit = sycamore_circuit(qubits, 10, &mut rng);
    let (tensor, perm) = circuit.into_amplitude_network(&"0".repeat(qubits));

    // The result will be a scalar, so no permutation is required
    assert!(perm.is_identity());

    // Set up MPI
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank();

    // Perform the contraction
    let result = distributed_contraction(tensor, &world);

    // Print the result
    if rank == 0 {
        println!("{result:?}");
    }
}

fn distributed_contraction(tensor: Tensor, world: &SimpleCommunicator) -> Tensor {
    let rank = world.rank();
    let size = world.size();
    let root = world.process_at_rank(0);

    let (partitioned_tn, path) = if rank == 0 {
        // Find a partitioning for the tensor network
        let partitioning = find_partitioning(&tensor, size, PartitioningStrategy::MinCut, true);
        let partitioned_tn = partition_tensor_network(tensor, &partitioning);

        // Find a contraction path for the individual partitions and the final fan-in
        let mut opt = Cotengrust::new(&partitioned_tn, OptMethod::Greedy);
        opt.find_path();
        let path = opt.get_best_replace_path();

        (partitioned_tn, path)
    } else {
        Default::default()
    };

    // Distribute the part of the path that describes the final fan-in between ranks
    let mut communication_path = if rank == 0 {
        path.toplevel.clone()
    } else {
        Default::default()
    };
    broadcast_path(&mut communication_path, &root);

    // Distribute the partitions to the ranks
    let (mut local_tn, local_path, comm) =
        scatter_tensor_network(&partitioned_tn, &path, rank, size, world);

    // Contract the partitions on each rank
    local_tn = contract_tensor_network(local_tn, &local_path);

    // Perform the final fan-in, sending tensors between ranks and contracting them
    // until there is only the final tensor left, which will end up on rank 0.
    intermediate_reduce_tensor_network(&mut local_tn, &communication_path, rank, world, &comm);

    local_tn
}
