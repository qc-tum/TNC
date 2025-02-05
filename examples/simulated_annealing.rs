use rand::{rngs::StdRng, Rng, SeedableRng};
use tensorcontraction::{
    contractionpath::contraction_tree::{
        balancing::CommunicationScheme,
        repartitioning::{
            compute_solution,
            simulated_annealing::{
                balance_partitions, LeafPartitioningModel, NaivePartitioningModel,
            },
        },
    },
    networks::{connectivity::ConnectivityLayout, random_circuit::random_circuit},
    tensornetwork::{
        contraction::contract_tensor_network,
        partitioning::{find_partitioning, partition_config::PartitioningStrategy},
        tensor::Tensor,
    },
};

fn compute_partitioning_cost<R>(
    tensor: &Tensor,
    partitioning: &[usize],
    communication_scheme: CommunicationScheme,
    rng: Option<&mut R>,
) -> f64
where
    R: ?Sized + Rng,
{
    compute_solution(tensor, partitioning, communication_scheme, rng).2
}

fn main() {
    let mut rng = StdRng::seed_from_u64(42);

    let num_qubits = 40;
    let circuit_depth = 20;
    let single_qubit_probability = 0.4;
    let two_qubit_probability = 0.4;
    let connectivity = ConnectivityLayout::Osprey;
    let num_partitions = 4;
    let communication_scheme = CommunicationScheme::WeightedBranchBound;

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
    let initial_partitioning =
        find_partitioning(&tensor, num_partitions, PartitioningStrategy::MinCut, true);

    // Calculate the initial score
    let initial_score = compute_partitioning_cost(
        &tensor,
        &initial_partitioning,
        communication_scheme,
        Some(&mut rng),
    );
    println!("Initial score: {initial_score:?}");

    // Try to find a better partitioning with undirected simulated annealing
    let (_, final_score) = balance_partitions::<_, NaivePartitioningModel>(
        &tensor,
        num_partitions as usize,
        initial_partitioning.clone(),
        communication_scheme,
        &mut rng,
        None,
    );
    println!("Normal final score: {final_score:?}");

    // Try to find a better partitioning with directed simulated annealing
    let mut intermediate_tensors = vec![Tensor::default(); num_partitions as usize];
    for (index, partition) in initial_partitioning.iter().enumerate() {
        intermediate_tensors[*partition] ^= tensor.tensor(index);
    }
    let (partitioning, final_score) = balance_partitions::<_, LeafPartitioningModel>(
        &tensor,
        num_partitions as usize,
        (initial_partitioning, intermediate_tensors),
        communication_scheme,
        &mut rng,
        None,
    );
    println!("Directed final score: {final_score:?}");

    // Partition the tensor network with the found partitioning and contract
    let (tensor, path, _) =
        compute_solution::<StdRng>(&tensor, &partitioning.0, communication_scheme, None);

    let tensor = contract_tensor_network(tensor, &path);
    println!("{:?}", tensor.tensor_data());
}
