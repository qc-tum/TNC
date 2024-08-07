use flexi_logger::{json_format, Duplicate, FileSpec, Logger};
use log::{info, LevelFilter};
use mpi::Rank;
use rand::rngs::StdRng;
use rand::SeedableRng;

use tensorcontraction::contractionpath::contraction_cost::contract_cost_tensors;
use tensorcontraction::contractionpath::contraction_tree::{
    balance_partitions_iter, BalanceSettings,
};
use tensorcontraction::contractionpath::paths::greedy::Greedy;
use tensorcontraction::contractionpath::paths::{CostType, OptimizePath};
use tensorcontraction::mpi::communication::CommunicationScheme;
use tensorcontraction::networks::connectivity::ConnectivityLayout;
use tensorcontraction::networks::sycamore::random_circuit;
use tensorcontraction::tensornetwork::partitioning::{find_partitioning, partition_tensor_network};
use tensorcontraction::tensornetwork::tensor::Tensor;

/// Sets up logging for rank `rank`. Each rank logs to a separate file and to stdout.
fn setup_logging_mpi(rank: Rank) {
    let _logger = Logger::with(LevelFilter::Debug)
        .format(json_format)
        .log_to_file(
            FileSpec::default()
                .discriminant(format!("rank{}", rank))
                .suppress_timestamp()
                .suffix("log.json"),
        )
        .duplicate_to_stdout(Duplicate::Info)
        .start()
        .unwrap();
}

fn greedy_cost_fn(t1: &Tensor, t2: &Tensor) -> f64 {
    t1.size() as f64 + t2.size() as f64 - (t1 ^ t2).size() as f64
}

fn main() {
    let mut rng: StdRng = StdRng::seed_from_u64(27);

    let num_qubits = 20;
    let circuit_depth = 30;
    let single_qubit_probability = 0.4;
    let two_qubit_probability = 0.4;
    let connectivity = ConnectivityLayout::Osprey;
    let num_partitions = 4;

    setup_logging_mpi(0);

    let tensor = random_circuit(
        num_qubits,
        circuit_depth,
        single_qubit_probability,
        two_qubit_probability,
        &mut rng,
        connectivity,
    );

    // println!("Tensor: {:?}", tensor.total_num_tensors());
    // Find vec of partitions
    let partitioning = find_partitioning(
        &tensor,
        num_partitions,
        String::from("tests/cut_kKaHyPar_sea20.ini"),
        true,
    );

    let partitioned_tn = partition_tensor_network(&tensor, &partitioning);
    let mut opt = Greedy::new(&partitioned_tn, CostType::Flops);
    opt.optimize_path();
    let path = opt.get_best_replace_path();

    let rebalance_depth = 1;
    let mut best_cost = f64::MAX;
    let mut best_iteration = 0;
    let mut best_method = String::new();
    for communication_scheme in [
        CommunicationScheme::Greedy,
        CommunicationScheme::Bipartition,
        CommunicationScheme::WeightedBranchBound,
    ] {
        let (num, mut _new_tn, _contraction_path, costs) = balance_partitions_iter(
            &partitioned_tn,
            &path,
            BalanceSettings {
                random_balance: false,
                rebalance_depth,
                iterations: 120,
                output_file: format!("output/{:?}_trial", communication_scheme),
                dendogram_cost_function: contract_cost_tensors,
                greedy_cost_function: greedy_cost_fn,
                communication_scheme,
            },
        );
        info!(
            "Best iteration for {:?} is {} at {}",
            communication_scheme, num, costs[num]
        );
        if costs[num] < best_cost {
            best_cost = costs[num];
            best_method = format!("{:?}", communication_scheme);
            best_iteration = num;
        }
    }
    info!(
        "Best scheme: {:?} with {} at {}",
        best_method, best_cost, best_iteration
    );

    // contract_tensor_network(&mut new_tn, &contraction_path);
}
