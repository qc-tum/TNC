use std::fs::{self};
use std::io::{BufWriter, Write};
use std::panic;

use ordered_float::NotNan;
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use tensorcontraction::contractionpath::contraction_cost::{
    compute_memory_requirements, contract_size_tensors_exact,
};
use tensorcontraction::contractionpath::contraction_tree::balancing::CommunicationScheme;
use tensorcontraction::contractionpath::contraction_tree::repartitioning::compute_solution;
use tensorcontraction::contractionpath::contraction_tree::repartitioning::simulated_annealing::{
    self, IntermediatePartitioningModel, LeafPartitioningModel, NaivePartitioningModel,
};
use tensorcontraction::contractionpath::paths::greedy::Greedy;
use tensorcontraction::contractionpath::paths::{CostType, OptimizePath};
use tensorcontraction::networks::connectivity::ConnectivityLayout;
use tensorcontraction::networks::random_circuit::random_circuit;

use tensorcontraction::tensornetwork::partitioning::find_partitioning;
use tensorcontraction::tensornetwork::partitioning::partition_config::PartitioningStrategy;
use tensorcontraction::tensornetwork::tensor::Tensor;
use tensorcontraction::types::ContractionIndex;

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

    // let file = File::create(format!("results.json")).unwrap();
    let file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("results.json")
        .unwrap();
    let mut writer = BufWriter::new(file);
    for num_qubits in (10..=20).step_by(10) {
        for circuit_depth in (20..=40).step_by(10) {
            println!("Circuit: {num_qubits}, {circuit_depth}");
            for i in 15..=20 {
                println!("Iteration {i}");
                let mut local_results = Vec::new();

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
                    &mut StdRng::seed_from_u64(i),
                    connectivity,
                );

                // Find an initial partitioning with KaHyPar
                let (
                    initial_partitioning,
                    initial_partitioned_tensor,
                    initial_contraction_path,
                    original_flops,
                ) = match panic::catch_unwind(|| {
                    initial_problem(
                        &tensor,
                        num_partitions,
                        communication_scheme,
                        &mut StdRng::seed_from_u64(i),
                    )
                }) {
                    Ok((
                        initial_partitioning,
                        initial_partitioned_tensor,
                        initial_contraction_path,
                        original_flops,
                    )) => (
                        initial_partitioning,
                        initial_partitioned_tensor,
                        initial_contraction_path,
                        original_flops,
                    ),
                    Err(_) => {
                        local_results.push(TensorResult {
                            seed: i,
                            num_qubits,
                            circuit_depth,
                            method: "Generic".to_string(),
                            flops: -1f64,
                            mem: -1f64,
                            flops_ratio: 1f64,
                            mem_ratio: 1f64,
                        });
                        continue;
                    }
                };
                let original_memory = compute_memory_requirements(
                    initial_partitioned_tensor.tensors(),
                    &initial_contraction_path,
                    contract_size_tensors_exact,
                );

                local_results.push(TensorResult {
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

                let mut intermediate_tensors = vec![Tensor::default(); num_partitions as usize];
                for (index, partition) in initial_partitioning.iter().enumerate() {
                    intermediate_tensors[*partition] ^= tensor.tensor(index);
                }
                // Try to find a better partitioning with a simulated annealing algorithm
                let (flops, memory, flops_ratio, mem_ratio) = match panic::catch_unwind(|| {
                    sad_run(
                        &tensor,
                        num_partitions,
                        &initial_partitioning,
                        &intermediate_tensors,
                        communication_scheme,
                        &mut StdRng::seed_from_u64(i),
                    )
                }) {
                    Ok((flops, memory)) => (
                        flops,
                        memory,
                        flops / original_flops,
                        memory / original_memory,
                    ),
                    Err(_) => (-1f64, -1f64, -1f64, -1f64),
                };

                results.push(TensorResult {
                    seed: i,
                    num_qubits,
                    circuit_depth,
                    method: "SAD".to_string(),
                    flops,
                    mem: memory,
                    flops_ratio,
                    mem_ratio,
                });
                println!(
                    "SAD: {} / {}",
                    flops / original_flops,
                    memory / original_memory
                );

                let mut opt = Greedy::new(&initial_partitioned_tensor, CostType::Flops);
                opt.optimize_path();
                let mut initial_contractions = Vec::new();
                for contraction_path in initial_contraction_path {
                    if let ContractionIndex::Path(_, path) = contraction_path {
                        initial_contractions.push(path);
                    }
                }

                let (flops, memory, flops_ratio, mem_ratio) = match panic::catch_unwind(|| {
                    iad_run(
                        &tensor,
                        num_partitions,
                        &initial_partitioning,
                        &intermediate_tensors,
                        initial_contractions,
                        communication_scheme,
                        &mut StdRng::seed_from_u64(i),
                    )
                }) {
                    Ok((flops, memory)) => (
                        flops,
                        memory,
                        flops / original_flops,
                        memory / original_memory,
                    ),
                    Err(_) => (-1f64, -1f64, -1f64, -1f64),
                };

                local_results.push(TensorResult {
                    seed: i,
                    num_qubits,
                    circuit_depth,
                    method: "IAD".to_string(),
                    flops,
                    mem: memory,
                    flops_ratio,
                    mem_ratio,
                });
                println!(
                    "IAD: {} / {}",
                    flops / original_flops,
                    memory / original_memory
                );

                let (flops, memory, flops_ratio, mem_ratio) = match panic::catch_unwind(|| {
                    sa_run(
                        &tensor,
                        num_partitions,
                        &initial_partitioning,
                        communication_scheme,
                        &mut StdRng::seed_from_u64(i),
                    )
                }) {
                    Ok((flops, memory)) => (
                        flops,
                        memory,
                        flops / original_flops,
                        memory / original_memory,
                    ),
                    Err(_) => (-1f64, -1f64, -1f64, -1f64),
                };

                local_results.push(TensorResult {
                    seed: i,
                    num_qubits,
                    circuit_depth,
                    method: "SA".to_string(),
                    flops,
                    mem: memory,
                    flops_ratio,
                    mem_ratio,
                });
                println!(
                    "SA: {} / {}",
                    flops / original_flops,
                    memory / original_memory
                );
                // let (flops, memory, flops_ratio, mem_ratio) = match panic::catch_unwind(|| {
                //     ga_run(
                //         tensor,
                //         num_partitions,
                //         initial_partitioning,
                //         communication_scheme,
                //     )
                // }) {
                //     Ok((flops, memory)) => (
                //         flops,
                //         memory,
                //         flops / original_flops,
                //         memory / original_memory,
                //     ),
                //     Err(_) => (-1f64, -1f64, -1f64, -1f64),
                // };

                // local_results.push(TensorResult {
                //     seed: i,
                //     num_qubits,
                //     circuit_depth,
                //     method: "GA".to_string(),
                //     flops,
                //     mem: memory,
                //     flops_ratio,
                //     mem_ratio,
                // });

                // println!(
                //     "GA: {} / {}",
                //     flops / original_flops,
                //     memory / original_memory
                // );

                serde_json::to_writer(&mut writer, &local_results).unwrap();
                writer.flush().unwrap();
                results.append(&mut local_results);
            }
        }
    }
}

// fn ga_run(
//     tensor: Tensor,
//     num_partitions: i32,
//     initial_partitioning: Vec<usize>,
//     communication_scheme: CommunicationScheme,
// ) -> (f64, f64) {
//     let (partitioning, _) = genetic::balance_partitions(
//         &tensor,
//         num_partitions as usize,
//         &initial_partitioning,
//         communication_scheme,
//         None,
//     );

//     let (partitioned_tensor, contraction_path, flops) =
//         compute_solution::<StdRng>(&tensor, &partitioning, communication_scheme, None);
//     let memory = compute_memory_requirements(
//         partitioned_tensor.tensors(),
//         &contraction_path,
//         contract_size_tensors_exact,
//     );
//     (flops, memory)
// }

fn sa_run(
    tensor: &Tensor,
    num_partitions: i32,
    initial_partitioning: &[usize],
    communication_scheme: CommunicationScheme,
    rng: &mut StdRng,
) -> (f64, f64) {
    let (partitioning, _): (Vec<usize>, NotNan<f64>) =
        simulated_annealing::balance_partitions::<_, NaivePartitioningModel>(
            tensor,
            num_partitions as usize,
            initial_partitioning.to_vec(),
            communication_scheme,
            rng,
            None,
        );

    let (partitioned_tensor, contraction_path, flops) =
        compute_solution(tensor, &partitioning, communication_scheme, Some(rng));
    let memory = compute_memory_requirements(
        partitioned_tensor.tensors(),
        &contraction_path,
        contract_size_tensors_exact,
    );
    (flops, memory)
}

fn iad_run(
    tensor: &Tensor,
    num_partitions: i32,
    initial_partitioning: &[usize],
    intermediate_tensors: &[Tensor],
    initial_contractions: Vec<Vec<ContractionIndex>>,
    communication_scheme: CommunicationScheme,
    rng: &mut StdRng,
) -> (f64, f64) {
    // Try to find a better partitioning with a simulated annealing algorithm
    let (solution, _) = simulated_annealing::balance_partitions::<_, IntermediatePartitioningModel>(
        tensor,
        num_partitions as usize,
        (
            initial_partitioning.to_vec(),
            intermediate_tensors.to_vec(),
            initial_contractions,
        ),
        communication_scheme,
        rng,
        None,
    );
    let (partitioning, ..) = solution;

    let (partitioned_tensor, contraction_path, flops) =
        compute_solution(tensor, &partitioning, communication_scheme, Some(rng));
    let memory = compute_memory_requirements(
        partitioned_tensor.tensors(),
        &contraction_path,
        contract_size_tensors_exact,
    );
    (flops, memory)
}

fn sad_run(
    tensor: &Tensor,
    num_partitions: i32,
    initial_partitioning: &[usize],
    intermediate_tensors: &[Tensor],
    communication_scheme: CommunicationScheme,
    rng: &mut StdRng,
) -> (f64, f64) {
    let (solution, _) = simulated_annealing::balance_partitions::<_, LeafPartitioningModel>(
        tensor,
        num_partitions as usize,
        (initial_partitioning.to_vec(), intermediate_tensors.to_vec()),
        communication_scheme,
        rng,
        None,
    );
    let (partitioning, ..) = solution;

    let (partitioned_tensor, contraction_path, flops) =
        compute_solution(tensor, &partitioning, communication_scheme, Some(rng));
    let memory = compute_memory_requirements(
        partitioned_tensor.tensors(),
        &contraction_path,
        contract_size_tensors_exact,
    );
    (flops, memory)
}

fn initial_problem(
    tensor: &Tensor,
    num_partitions: i32,
    communication_scheme: CommunicationScheme,
    rng: &mut StdRng,
) -> (Vec<usize>, Tensor, Vec<ContractionIndex>, f64) {
    let initial_partitioning =
        find_partitioning(tensor, num_partitions, PartitioningStrategy::MinCut, true);
    let (initial_partitioned_tensor, initial_contraction_path, original_flops) = compute_solution(
        tensor,
        &initial_partitioning,
        communication_scheme,
        Some(rng),
    );
    (
        initial_partitioning,
        initial_partitioned_tensor,
        initial_contraction_path,
        original_flops,
    )
}
