use rand::{rngs::StdRng, SeedableRng};
use tensorcontraction::{
    contractionpath::{
        contraction_tree::{
            balancing::CommunicationScheme,
            repartitioning::{balance_partitions_genetic, calculate_fitness},
        },
        paths::{greedy::Greedy, CostType, OptimizePath},
    },
    networks::{connectivity::ConnectivityLayout, sycamore::random_circuit},
    tensornetwork::{
        contraction::contract_tensor_network,
        partitioning::{
            find_partitioning, partition_config::PartitioningStrategy, partition_tensor_network,
        },
    },
};

fn main() {
    let mut rng = StdRng::seed_from_u64(42);

    let num_qubits = 40;
    let circuit_depth = 20;
    let single_qubit_probability = 0.4;
    let two_qubit_probability = 0.4;
    let connectivity = ConnectivityLayout::Osprey;
    let num_partitions = 4;
    let communication_scheme = CommunicationScheme::Greedy;

    // Generate the tensor network
    let tensor = random_circuit(
        num_qubits,
        circuit_depth,
        single_qubit_probability,
        two_qubit_probability,
        &mut rng,
        connectivity,
    );

    // Find an initial partitioning with KaHyPar
    let partitioning =
        find_partitioning(&tensor, num_partitions, PartitioningStrategy::MinCut, true);

    // Calculate the initial fitness
    let initial_fitness = calculate_fitness(&tensor, &partitioning, communication_scheme);
    println!("Initial fitness: {initial_fitness:?}");

    // Try to find a better partitioning with a genetic algorithm
    let (partitioning, final_fitness) = balance_partitions_genetic(
        &tensor,
        num_partitions as usize,
        &partitioning,
        communication_scheme,
    );
    println!("Final fitness: {final_fitness:?}");

    // Partition the tensor network with the found partitioning and contract
    let mut partitioned_tn = partition_tensor_network(&tensor, &partitioning);

    let mut greedy = Greedy::new(&partitioned_tn, CostType::Flops);
    greedy.optimize_path();
    let path = greedy.get_best_replace_path();

    contract_tensor_network(&mut partitioned_tn, &path);
    println!("{:?}", partitioned_tn.tensor_data());
}
