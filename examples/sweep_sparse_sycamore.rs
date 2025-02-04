use std::{fs, panic};

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
    compute_solution, genetic, simulated_annealing,
};
use tensorcontraction::contractionpath::contraction_tree::ContractionTree;
use tensorcontraction::contractionpath::paths::tree_reconfiguration::TreeReconfigure;
use tensorcontraction::contractionpath::paths::{greedy::Greedy, CostType, OptimizePath};
use tensorcontraction::networks::connectivity::ConnectivityLayout;
use tensorcontraction::networks::random_circuit::random_circuit_with_observable;
use tensorcontraction::tensornetwork::partitioning::find_partitioning;
use tensorcontraction::tensornetwork::partitioning::partition_config::PartitioningStrategy;
use tensorcontraction::tensornetwork::tensor::Tensor;
use tensorcontraction::types::ContractionIndex;

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
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("sweep_sparse_sycamore.json")
        .unwrap();

    let single_qubit_probability = 0.4;
    let two_qubit_probability = 0.4;
    let observable_probability = 1.0;
    let connectivity = ConnectivityLayout::Sycamore;

    let qubit_range = (10..50).step_by(5).collect_vec();
    let circuit_depth_range = (10..50).step_by(5).collect_vec();
    let partition_range = (2..5).map(|p| 2i32.pow(p)).collect_vec();
    let rng = thread_rng();
    let communication_scheme = CommunicationScheme::WeightedBranchBound;
    let seed_range = rng.sample_iter(Standard).take(10).collect_vec();

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
                    let tensor = random_circuit_with_observable(
                        num_qubits,
                        circuit_depth,
                        single_qubit_probability,
                        two_qubit_probability,
                        observable_probability,
                        &mut StdRng::seed_from_u64(seed),
                        connectivity,
                    );

                    let (
                        initial_partitioning,
                        initial_partitioned_tensor,
                        initial_contraction_path,
                        original_flops,
                    ) = match panic::catch_unwind(|| {
                        initial_problem(&tensor, num_partitions, communication_scheme)
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
                                "Generic",
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
                        method: "Generic".to_string(),
                        flops: original_flops,
                        mem: original_memory,
                        flops_ratio: 1f64,
                        mem_ratio: 1f64,
                    });

                    let (flops, memory, flops_ratio, mem_ratio) = match panic::catch_unwind(|| {
                        greedy_balancing_run(
                            &initial_partitioned_tensor,
                            &initial_contraction_path,
                            40,
                            communication_scheme,
                            BalancingScheme::AlternatingIntermediateTensors { height_limit: 8 },
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
                            initial_contractions.clone(),
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
                        cotengra_run(&tensor, num_partitions, communication_scheme)
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
                        method: "cotengra".to_string(),
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

                    let (flops, memory, flops_ratio, mem_ratio) = match panic::catch_unwind(|| {
                        ga_run(
                            tensor,
                            num_partitions,
                            initial_partitioning,
                            communication_scheme,
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
                        method: "GA".to_string(),
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

fn initial_problem(
    tensor: &Tensor,
    num_partitions: i32,
    communication_scheme: CommunicationScheme,
) -> (Vec<usize>, Tensor, Vec<ContractionIndex>, f64) {
    let initial_partitioning =
        find_partitioning(tensor, num_partitions, PartitioningStrategy::MinCut, true);
    let (initial_partitioned_tensor, initial_contraction_path, original_flops) =
        compute_solution(tensor, &initial_partitioning, communication_scheme);
    (
        initial_partitioning,
        initial_partitioned_tensor,
        initial_contraction_path,
        original_flops,
    )
}

fn ga_run(
    tensor: Tensor,
    num_partitions: i32,
    initial_partitioning: Vec<usize>,
    communication_scheme: CommunicationScheme,
) -> (f64, f64) {
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
    );
    (flops, memory)
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
        compute_solution(tensor, &partitioning, communication_scheme);
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
        compute_solution(tensor, &partitioning, communication_scheme);
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
        compute_solution(tensor, &partitioning, communication_scheme);
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
) -> (f64, f64) {
    let balance_settings = BalanceSettings::new(
        1,
        iterations,
        objective_function,
        communication_scheme,
        balancing_scheme,
        None,
    );
    let (best_iteration, partitioned_tensor, contraction_path, max_costs) =
        balance_partitions_iter(tensor, initial_contractions, balance_settings, None);
    let flops = max_costs[best_iteration];

    let memory = compute_memory_requirements(
        partitioned_tensor.tensors(),
        &contraction_path,
        contract_size_tensors_exact,
    );
    (flops, memory)
}

fn cotengra_run(
    tensor: &Tensor,
    num_bipartitions: i32,
    communication_scheme: CommunicationScheme,
) -> (f64, f64) {
    let mut tree = TreeReconfigure::new(tensor, 8, CostType::Flops);
    tree.optimize_path();
    let best_path = tree.get_best_path();
    let contraction_tree = ContractionTree::from_contraction_path(tensor, best_path);
    let tree_root = contraction_tree.root_id().unwrap();
    let mut partitions = vec![tree_root];
    for _ in 0..num_bipartitions {
        let mut temp = Vec::new();
        for partition in partitions {
            let left_child = contraction_tree.node(partition).left_child_id();
            let right_child = contraction_tree.node(partition).right_child_id();
            temp.push(left_child.unwrap());
            temp.push(right_child.unwrap());
        }
        partitions = temp;
    }
    let mut partitioning = vec![0; tensor.tensors().len()];
    for (i, partition_root) in partitions.iter().enumerate() {
        let leaf_tensors = contraction_tree.leaf_ids(*partition_root);
        for leaf in leaf_tensors {
            partitioning[leaf] = i;
        }
    }

    let (partitioned_tensor, contraction_path, flops) =
        compute_solution(tensor, &partitioning, communication_scheme);
    let memory = compute_memory_requirements(
        partitioned_tensor.tensors(),
        &contraction_path,
        contract_size_tensors_exact,
    );
    (flops, memory)
}
