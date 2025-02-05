use std::{fs, panic};

use clap::Parser;
use itertools::Itertools;
use log::info;
use rand::distributions::Standard;
use rand::rngs::StdRng;
use rand::{thread_rng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use tensorcontraction::contractionpath::contraction_cost::{
    compute_memory_requirements, contract_size_tensors_exact,
};
use tensorcontraction::contractionpath::contraction_tree::balancing::CommunicationScheme;

use tensorcontraction::contractionpath::contraction_tree::repartitioning::compute_solution;
use tensorcontraction::networks::connectivity::ConnectivityLayout;
use tensorcontraction::networks::random_circuit::random_circuit;
use tensorcontraction::tensornetwork::partitioning::find_partitioning;
use tensorcontraction::tensornetwork::partitioning::partition_config::PartitioningStrategy;
use tensorcontraction::tensornetwork::tensor::Tensor;
use tensorcontraction::types::ContractionIndex;

#[derive(Parser)]
struct Cli {
    single_qubit_probability: f64,
    two_qubit_probability: f64,
    out_file: String,
}

#[derive(Serialize, Deserialize, Debug)]
struct TensorResult {
    seed: u64,
    num_qubits: usize,
    circuit_depth: usize,
    partitions: i32,
    method: String,
    flops: f64,
    mem: f64,
    flops_ratio: f64,
    mem_ratio: f64,
}

impl TensorResult {
    fn new_invalid(
        seed: u64,
        num_qubits: usize,
        circuit_depth: usize,
        partitions: i32,
        method: impl Into<String>,
    ) -> Self {
        TensorResult {
            seed,
            num_qubits,
            circuit_depth,
            partitions,
            method: method.into(),
            flops: -1.0,
            mem: -1.0,
            flops_ratio: -1.0,
            mem_ratio: -1.0,
        }
    }
}

fn main() {
    let args = Cli::parse();
    let out_file = args.out_file;
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(out_file)
        .unwrap();

    let single_qubit_probability = args.single_qubit_probability;
    let two_qubit_probability = args.two_qubit_probability;
    // let observable_probability = 1.0;
    let connectivity = ConnectivityLayout::Sycamore;

    let qubit_range = (50..60).step_by(10).collect_vec();
    let circuit_depth_range = (20..40).step_by(10).collect_vec();
    let partition_range = (7..8).map(|p| 2i32.pow(p)).collect_vec();
    let rng = thread_rng();
    let seed_range = rng.sample_iter(Standard).take(10).collect_vec();

    let mut write = |result: TensorResult| {
        serde_json::to_writer(&mut file, &[result]).unwrap();
    };

    for &num_qubits in &qubit_range {
        println!("qubits: {num_qubits}");
        for &circuit_depth in &circuit_depth_range {
            println!("circuit_depth: {:?}", circuit_depth);
            for &seed in &seed_range {
                println!("seed: {:?}", seed);
                for &num_partitions in &partition_range {
                    println!("partitions: {:?}", num_partitions);
                    info!(seed, num_qubits, circuit_depth, single_qubit_probability, two_qubit_probability, connectivity:?; "Configuration set");
                    let tensor = random_circuit(
                        num_qubits,
                        circuit_depth,
                        single_qubit_probability,
                        two_qubit_probability,
                        // observable_probability,
                        &mut StdRng::seed_from_u64(seed),
                        connectivity,
                    );

                    let (_, initial_partitioned_tensor, initial_contraction_path, greedy_flops) =
                        match panic::catch_unwind(|| {
                            initial_problem(
                                &tensor,
                                num_partitions,
                                CommunicationScheme::Greedy,
                                &mut StdRng::seed_from_u64(seed),
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
                                write(TensorResult::new_invalid(
                                    seed,
                                    num_qubits,
                                    circuit_depth,
                                    num_partitions,
                                    CommunicationScheme::Greedy.to_string(),
                                ));
                                continue;
                            }
                        };
                    let greedy_memory = compute_memory_requirements(
                        initial_partitioned_tensor.tensors(),
                        &initial_contraction_path,
                        contract_size_tensors_exact,
                    );
                    write(TensorResult {
                        seed,
                        num_qubits,
                        circuit_depth,
                        partitions: num_partitions,
                        method: CommunicationScheme::Greedy.to_string(),
                        flops: greedy_flops,
                        mem: greedy_memory,
                        flops_ratio: 1.0,
                        mem_ratio: 1.0,
                    });

                    for communication_scheme in [
                        CommunicationScheme::Bipartition,
                        // CommunicationScheme::WeightedBranchBound,
                        // CommunicationScheme::BranchBound,
                        CommunicationScheme::RandomGreedy,
                        CommunicationScheme::RandomGreedyLatency,
                        CommunicationScheme::BipartitionSweep,
                    ] {
                        info!(seed, num_qubits, circuit_depth, single_qubit_probability, two_qubit_probability, connectivity:?; "Configuration set");

                        let (
                            _,
                            initial_partitioned_tensor,
                            initial_contraction_path,
                            strategy_flops,
                        ) = match panic::catch_unwind(|| {
                            initial_problem(
                                &tensor,
                                num_partitions,
                                communication_scheme,
                                &mut StdRng::seed_from_u64(seed),
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
                                write(TensorResult::new_invalid(
                                    seed,
                                    num_qubits,
                                    circuit_depth,
                                    num_partitions,
                                    communication_scheme.to_string(),
                                ));
                                continue;
                            }
                        };
                        let strategy_memory = compute_memory_requirements(
                            initial_partitioned_tensor.tensors(),
                            &initial_contraction_path,
                            contract_size_tensors_exact,
                        );
                        write(TensorResult {
                            seed,
                            num_qubits,
                            circuit_depth,
                            partitions: num_partitions,
                            method: communication_scheme.to_string(),
                            flops: strategy_flops,
                            mem: strategy_memory,
                            flops_ratio: strategy_flops / greedy_flops,
                            mem_ratio: strategy_memory / greedy_memory,
                        });
                    }
                }
            }
        }
    }
}

fn initial_problem<R>(
    tensor: &Tensor,
    num_partitions: i32,
    communication_scheme: CommunicationScheme,
    rng: &mut R,
) -> (Vec<usize>, Tensor, Vec<ContractionIndex>, f64)
where
    R: ?Sized + Rng,
{
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
