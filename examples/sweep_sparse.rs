use std::{fs, panic};

use clap::Parser;
use itertools::Itertools;
use log::info;
use ordered_float::NotNan;
use rand::distributions::Standard;
use rand::rngs::StdRng;
use rand::{thread_rng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use tensorcontraction::contractionpath::contraction_cost::{
    compute_memory_requirements, contract_size_tensors_exact,
};
use tensorcontraction::contractionpath::contraction_tree::balancing::{
    balance_partitions_iter, BalanceSettings, BalancingScheme, CommunicationScheme,
};
use tensorcontraction::contractionpath::contraction_tree::repartitioning::simulated_annealing::{
    IntermediatePartitioningModel, LeafPartitioningModel, NaivePartitioningModel,
};
use tensorcontraction::contractionpath::contraction_tree::repartitioning::{
    compute_solution, simulated_annealing,
};
use tensorcontraction::contractionpath::contraction_tree::ContractionTree;
use tensorcontraction::contractionpath::paths::tree_reconfiguration::TreeReconfigure;
use tensorcontraction::contractionpath::paths::{greedy::Greedy, CostType, OptimizePath};
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

    let qubit_range = (5..15).step_by(5).collect_vec();
    let circuit_depth_range = (5..15).step_by(5).collect_vec();
    let partition_range = (1..4).map(|p| 2i32.pow(p)).collect_vec();
    let rng = thread_rng();
    let seed_range = rng.sample_iter(Standard).take(10).collect_vec();
    let communication_scheme = CommunicationScheme::RandomGreedyLatency;

    let mut write = |result: TensorResult| {
        serde_json::to_writer(&mut file, &[result]).unwrap();
    };

    for num_qubits in qubit_range {
        println!("qubits: {num_qubits}");
        for &circuit_depth in &circuit_depth_range {
            println!("circuit_depth: {:?}", circuit_depth);
            for &seed in &seed_range {
                for &num_partitions in &partition_range {
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
                                "Generic".to_string(),
                            ));
                            continue;
                        }
                    };
                    let original_memory = compute_memory_requirements(
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
                        flops: original_flops,
                        mem: original_memory,
                        flops_ratio: 1f64,
                        mem_ratio: 1f64,
                    });

                    let mut intermediate_tensors = vec![Tensor::default(); num_partitions as usize];
                    for (index, partition) in initial_partitioning.iter().enumerate() {
                        intermediate_tensors[*partition] ^= tensor.tensor(index);
                    }

                    let (flops, memory, flops_ratio, mem_ratio) = match panic::catch_unwind(|| {
                        greedy_balancing_run(
                            &initial_partitioned_tensor,
                            &initial_contraction_path,
                            40,
                            communication_scheme,
                            BalancingScheme::AlternatingIntermediateTensors { height_limit: 8 },
                            &mut StdRng::seed_from_u64(seed),
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

                    write(TensorResult {
                        seed,
                        num_qubits,
                        circuit_depth,
                        partitions: num_partitions,
                        method: "GreedyBalance".to_string(),
                        flops,
                        mem: memory,
                        flops_ratio,
                        mem_ratio,
                    });

                    let (flops, memory, flops_ratio, mem_ratio, cotengra_partitions) =
                        match panic::catch_unwind(|| {
                            cotengra_run(
                                &tensor,
                                num_partitions,
                                communication_scheme,
                                &mut StdRng::seed_from_u64(seed),
                            )
                        }) {
                            Ok((flops, memory, cotengra_partitions)) => (
                                flops,
                                memory,
                                flops / original_flops,
                                memory / original_memory,
                                cotengra_partitions,
                            ),
                            Err(_) => (-1f64, -1f64, -1f64, -1f64, 0usize),
                        };

                    write(TensorResult {
                        seed,
                        num_qubits,
                        circuit_depth,
                        partitions: cotengra_partitions as i32,
                        method: "cotengra".to_string(),
                        flops,
                        mem: memory,
                        flops_ratio,
                        mem_ratio,
                    });

                    // Try to find a better partitioning with a simulated annealing algorithm
                    let (flops, memory, flops_ratio, mem_ratio) = match panic::catch_unwind(|| {
                        sad_run(
                            &tensor,
                            num_partitions,
                            &initial_partitioning,
                            &intermediate_tensors,
                            communication_scheme,
                            &mut StdRng::seed_from_u64(seed),
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

                    write(TensorResult {
                        seed,
                        num_qubits,
                        circuit_depth,
                        partitions: num_partitions,
                        method: "SAD".to_string(),
                        flops,
                        mem: memory,
                        flops_ratio,
                        mem_ratio,
                    });

                    let mut opt = Greedy::new(&initial_partitioned_tensor, CostType::Flops);
                    opt.optimize_path();
                    let mut initial_contractions = Vec::new();
                    for contraction_path in initial_contraction_path {
                        if let ContractionIndex::Path(_, _, path) = contraction_path {
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
                            &mut StdRng::seed_from_u64(seed),
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

                    write(TensorResult {
                        seed,
                        num_qubits,
                        circuit_depth,
                        partitions: num_partitions,
                        method: "IAD".to_string(),
                        flops,
                        mem: memory,
                        flops_ratio,
                        mem_ratio,
                    });

                    let (flops, memory, flops_ratio, mem_ratio) = match panic::catch_unwind(|| {
                        sa_run(
                            &tensor,
                            num_partitions,
                            &initial_partitioning,
                            communication_scheme,
                            &mut StdRng::seed_from_u64(seed),
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

                    write(TensorResult {
                        seed,
                        num_qubits,
                        circuit_depth,
                        partitions: num_partitions,
                        method: "SA".to_string(),
                        flops,
                        mem: memory,
                        flops_ratio,
                        mem_ratio,
                    });
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

fn objective_function(a: &Tensor, b: &Tensor) -> f64 {
    a.size() + b.size() - (a ^ b).size()
}

fn greedy_balancing_run(
    tensor: &Tensor,
    initial_contractions: &[ContractionIndex],
    iterations: usize,
    communication_scheme: CommunicationScheme,
    balancing_scheme: BalancingScheme,
    rng: &mut StdRng,
) -> (f64, f64)
where
{
    let balance_settings = BalanceSettings::new(
        1,
        iterations,
        objective_function,
        communication_scheme,
        balancing_scheme,
        None,
    );
    let (best_iteration, partitioned_tensor, contraction_path, max_costs) =
        balance_partitions_iter(tensor, initial_contractions, balance_settings, None, rng);
    let flops = max_costs[best_iteration];

    let memory = compute_memory_requirements(
        partitioned_tensor.tensors(),
        &contraction_path,
        contract_size_tensors_exact,
    );
    (flops, memory)
}

fn cotengra_run<R>(
    tensor: &Tensor,
    num_bipartitions: i32,
    communication_scheme: CommunicationScheme,
    rng: &mut R,
) -> (f64, f64, usize)
where
    R: ?Sized + Rng,
{
    let mut tree = TreeReconfigure::new(tensor, 4, CostType::Flops);
    tree.optimize_path();
    let best_path = tree.get_best_replace_path();

    let contraction_tree = ContractionTree::from_contraction_path(tensor, &best_path);

    let tree_root = contraction_tree.root_id().unwrap();
    let mut leaves = vec![];
    let mut traversal = vec![tree_root];
    let num_partitions = 1 << num_bipartitions;
    while (leaves.len() + traversal.len()) < num_partitions && !traversal.is_empty() {
        let node_id = traversal.pop().unwrap();
        let node = contraction_tree.node(node_id);
        if node.is_leaf() {
            leaves.push(node_id);
        } else {
            traversal.push(node.left_child_id().unwrap());
            traversal.push(node.right_child_id().unwrap());
        }
    }
    traversal.append(&mut leaves);
    let num_partitions = traversal.len();
    let mut partitioning = vec![0; tensor.tensors().len()];
    for (i, partition_root) in traversal.iter().enumerate() {
        let leaf_tensors = contraction_tree.leaf_ids(*partition_root);
        for leaf in leaf_tensors {
            partitioning[leaf] = i;
        }
    }

    let (partitioned_tensor, contraction_path, flops) =
        compute_solution(tensor, &partitioning, communication_scheme, Some(rng));
    let memory = compute_memory_requirements(
        partitioned_tensor.tensors(),
        &contraction_path,
        contract_size_tensors_exact,
    );
    (flops, memory, num_partitions)
}
