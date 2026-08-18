use std::time::Duration;

use ndarray::{linalg::kron, Array2};
use num_complex::Complex64;
use rand::{rngs::StdRng, SeedableRng};
use tnc::{
    builders::sycamore_circuit::sycamore_circuit,
    contractionpath::{
        communication_schemes::CommunicationScheme,
        contraction_cost::{communication_path_cost, contract_path_cost},
        paths::{
            cotengrust::{Cotengrust, OptMethod},
            ContractionPathResult, Pathfinder,
        },
        repartitioning::simulated_annealing::{self, IntermediatePartitioningModel},
    },
    tensornetwork::{
        partitioning::{find_partitioning, partition_tensor_network, PartitioningStrategy},
        tensor::CompositeTensor,
        tensordata::TensorData,
    },
};

fn compute_cost(tensor: &CompositeTensor, partitioning: &[usize]) -> f64 {
    // Partition the tensor
    let partitioned_tn = partition_tensor_network(tensor.clone(), partitioning);

    // Find a contraction path
    let mut opt = Cotengrust::new(OptMethod::Greedy);
    let result = opt.find_path(&partitioned_tn);
    let path = result.replace_path();

    // Get the serial contraction cost of each partition
    let cost_per_partition = partitioned_tn
        .tensors()
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let (ops, _mem) =
                contract_path_cost(t.as_composite().unwrap().tensors(), &path.nested[&i], true);
            ops
        })
        .collect::<Vec<_>>();

    // Get the shape of the partition tensors (i.e., the tensors resulting from the contraction of the respective partition)
    let partition_tensors = partitioned_tn
        .tensors()
        .iter()
        .map(|t| t.as_composite().unwrap().external_tensor())
        .collect::<Vec<_>>();

    // Get the contraction cost of contracting the final partition tensors in parallel
    let (final_cost, _) = communication_path_cost(
        &partition_tensors,
        &path.toplevel,
        true,
        true,
        Some(&cost_per_partition),
    );
    final_cost
}

fn rebalance(
    tensor: &CompositeTensor,
    initial_partitioning: &[usize],
    rng: &mut StdRng,
    communication_scheme: CommunicationScheme,
) -> Vec<usize> {
    let model = IntermediatePartitioningModel {
        tensor,
        communication_scheme,
        memory_limit: None,
    };
    let initial_solution = model.compute_initial_solution(initial_partitioning, None);

    // Run the rebalancing
    let (solution, _) = simulated_annealing::balance_partitions(
        model,
        initial_solution,
        rng,
        Duration::from_secs(20),
    );

    // Return the new partitioning
    let (partitioning, ..) = solution;
    partitioning
}

fn z_chain(qubits: usize) -> TensorData {
    let mut matrix = Array2::eye(1);
    let z = Array2::from_shape_vec(
        (2, 2),
        vec![
            Complex64::ONE,
            Complex64::ZERO,
            Complex64::ZERO,
            -Complex64::ONE,
        ],
    )
    .unwrap();
    for _ in 0..qubits {
        matrix = kron(&matrix, &z);
    }
    TensorData::Matrix(matrix.into_dyn())
}

fn main() {
    // Create a small Sycamore circuit
    let qubits = 10;
    let mut rng = StdRng::seed_from_u64(0);
    let circuit = sycamore_circuit(qubits, 10, &mut rng);

    // Create a tensor network that computes the expectation value
    let observable = z_chain(qubits);
    let indices = circuit.qubits().collect::<Vec<_>>();
    let tensor = circuit.into_expectation_value_network(observable, &indices);

    // Find a partitioning into 8 partitions
    let naive_partitioning = find_partitioning(&tensor, 8, PartitioningStrategy::MinCut, true);

    // Compute cost
    let naive_cost = compute_cost(&tensor, &naive_partitioning);
    println!("Cost with naive partitioning     : {naive_cost:>10}");

    // Run the simulated annealing rebalancing to find better partitions
    let better_partitioning = rebalance(
        &tensor,
        &naive_partitioning,
        &mut rng,
        CommunicationScheme::RandomGreedy,
    );

    // Compute cost
    let better_cost = compute_cost(&tensor, &better_partitioning);
    println!("Cost with rebalanced partitioning: {better_cost:>10}");
}
