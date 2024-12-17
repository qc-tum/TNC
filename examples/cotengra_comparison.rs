use std::fs::File;
use std::io::{BufWriter, Write};

use ordered_float::NotNan;
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use tensorcontraction::contractionpath::contraction_cost::{
    compute_memory_requirements, contract_size_tensors_exact,
};
use tensorcontraction::contractionpath::contraction_tree::balancing::CommunicationScheme;
use tensorcontraction::contractionpath::contraction_tree::repartitioning::compute_solution;
use tensorcontraction::contractionpath::contraction_tree::repartitioning::genetic::{self};
use tensorcontraction::contractionpath::contraction_tree::repartitioning::simulated_annealing::{
    self, LeafPartitioningModel, NaivePartitioningModel,
};
use tensorcontraction::networks::connectivity::ConnectivityLayout;
use tensorcontraction::networks::random_circuit::random_circuit;

use tensorcontraction::tensornetwork::partitioning::find_partitioning;
use tensorcontraction::tensornetwork::partitioning::partition_config::PartitioningStrategy;
use tensorcontraction::tensornetwork::tensor::Tensor;

#[derive(Serialize, Deserialize, Debug)]
struct TensorResult {
    seed: u64,
    num_qubits: usize,
    circuit_depth: usize,
    method: String,
    flops: f64,
    mem: f64,
    flops_ratio: f64,
    mem_ratio: f64,
}

fn main() {
    let mut results = Vec::new();
    for num_qubits in (10..=20).step_by(10) {
        for circuit_depth in (10..=40).step_by(10) {
            println!("Circuit: {num_qubits}, {circuit_depth}");
            for i in 15..=20 {
                println!("Iteration {i}");
                let mut rng = StdRng::seed_from_u64(i);

                // let num_qubits = 40;
                // let circuit_depth = 40;
                let single_qubit_probability = 0.4;
                let two_qubit_probability = 0.4;
                let connectivity = ConnectivityLayout::Osprey;
                let num_partitions = 4;
                let communication_scheme = CommunicationScheme::WeightedBranchBound;

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
                let (partitioned_tensor, contraction_path, original_flops) =
                    compute_solution(&tensor, &initial_partitioning, communication_scheme);
                let original_memory = compute_memory_requirements(
                    partitioned_tensor.tensors(),
                    &contraction_path,
                    contract_size_tensors_exact,
                ) * 16.0;

                results.push(TensorResult {
                    seed: i,
                    num_qubits,
                    circuit_depth,
                    method: "Generic".to_string(),
                    flops: original_flops,
                    mem: original_memory,
                    flops_ratio: 1f64,
                    mem_ratio: 1f64,
                });
                println!("Original: {original_flops} / {original_memory}");
                let mut intermediate_tensors =
                    vec![Tensor::new(Vec::new()); num_partitions as usize];
                for (index, partition) in initial_partitioning.iter().enumerate() {
                    intermediate_tensors[*partition] ^= tensor.tensor(index);
                }
                // Try to find a better partitioning with a simulated annealing algorithm
                let ((partitioning, _), _): ((Vec<usize>, Vec<Tensor>), NotNan<f64>) =
                    simulated_annealing::balance_partitions::<_, LeafPartitioningModel>(
                        &tensor,
                        num_partitions as usize,
                        (initial_partitioning.clone(), intermediate_tensors),
                        communication_scheme,
                        &mut rng,
                        None,
                    );

                let (partitioned_tensor, contraction_path, flops) =
                    compute_solution(&tensor, &partitioning, communication_scheme);
                let memory = compute_memory_requirements(
                    partitioned_tensor.tensors(),
                    &contraction_path,
                    contract_size_tensors_exact,
                ) * 16.0;

                results.push(TensorResult {
                    seed: i,
                    num_qubits,
                    circuit_depth,
                    method: "SAD".to_string(),
                    flops,
                    mem: memory,
                    flops_ratio: flops / original_flops,
                    mem_ratio: memory / original_memory,
                });
                println!(
                    "SAD: {} / {}",
                    flops / original_flops,
                    memory / original_memory
                );
                let (partitioning, _): (Vec<usize>, NotNan<f64>) =
                    simulated_annealing::balance_partitions::<_, NaivePartitioningModel>(
                        &tensor,
                        num_partitions as usize,
                        initial_partitioning.clone(),
                        communication_scheme,
                        &mut rng,
                        None,
                    );

                let (partitioned_tensor, contraction_path, flops) =
                    compute_solution(&tensor, &partitioning, communication_scheme);
                let memory = compute_memory_requirements(
                    partitioned_tensor.tensors(),
                    &contraction_path,
                    contract_size_tensors_exact,
                ) * 16.0;

                results.push(TensorResult {
                    seed: i,
                    num_qubits,
                    circuit_depth,
                    method: "SA".to_string(),
                    flops,
                    mem: memory,
                    flops_ratio: flops / original_flops,
                    mem_ratio: memory / original_memory,
                });
                println!(
                    "SA: {} / {}",
                    flops / original_flops,
                    memory / original_memory
                );
                let (partitioning, _) = genetic::balance_partitions(
                    &tensor,
                    num_partitions as usize,
                    &initial_partitioning,
                    communication_scheme,
                    None,
                );

                let (partitioned_tensor, contraction_path, flops) =
                    compute_solution(&tensor, &partitioning, communication_scheme);
                let memory = compute_memory_requirements(
                    partitioned_tensor.tensors(),
                    &contraction_path,
                    contract_size_tensors_exact,
                ) * 16.0;

                results.push(TensorResult {
                    seed: i,
                    num_qubits,
                    circuit_depth,
                    method: "GA".to_string(),
                    flops,
                    mem: memory,
                    flops_ratio: flops / original_flops,
                    mem_ratio: memory / original_memory,
                });

                println!(
                    "GA: {} / {}",
                    flops / original_flops,
                    memory / original_memory
                );
            }
        }
    }
    // println!("results: {:?}", results);
    let file = File::create("results.json").unwrap();
    let mut writer = BufWriter::new(file);
    serde_json::to_writer(&mut writer, &results).unwrap();
    writer.flush().unwrap();
}
