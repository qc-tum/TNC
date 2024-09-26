use flexi_logger::{json_format, Duplicate, FileSpec, Logger};
use log::{info, LevelFilter};
use mpi::Rank;
use rand::rngs::StdRng;
use rand::SeedableRng;

use tensorcontraction::contractionpath::contraction_cost::contract_cost_tensors;
use tensorcontraction::contractionpath::contraction_tree::balancing::balancing_schemes::BalancingScheme;
use tensorcontraction::contractionpath::contraction_tree::balancing::communication_schemes::CommunicationScheme;
use tensorcontraction::contractionpath::contraction_tree::balancing::{
    balance_partitions_iter, BalanceSettings,
};
use tensorcontraction::contractionpath::contraction_tree::export::DendogramSettings;
use tensorcontraction::contractionpath::paths::greedy::Greedy;
use tensorcontraction::contractionpath::paths::{CostType, OptimizePath};
use tensorcontraction::networks::connectivity::ConnectivityLayout;
use tensorcontraction::networks::sycamore::random_circuit;
use tensorcontraction::tensornetwork::partitioning::partition_config::PartitioningStrategy;
use tensorcontraction::tensornetwork::partitioning::{find_partitioning, partition_tensor_network};
use tensorcontraction::tensornetwork::tensor::Tensor;

/// Sets up logging for rank `rank`. Each rank logs to a separate file and to stdout.
fn setup_logging_mpi(rank: Rank) {
    let _logger = Logger::with(LevelFilter::Debug)
        .format(json_format)
        .log_to_file(
            FileSpec::default()
                .discriminant(format!("rank{rank}"))
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

    // Find vec of partitions
    let partitioning =
        find_partitioning(&tensor, num_partitions, PartitioningStrategy::MinCut, true);

    let partitioned_tn = partition_tensor_network(&tensor, &partitioning);
    let mut opt = Greedy::new(&partitioned_tn, CostType::Flops);
    opt.optimize_path();
    let path = opt.get_best_replace_path();

    #[derive(Debug, Clone, Copy)]
    struct Candidate {
        pub cost: f64,
        _iteration: usize,
        _method: CommunicationScheme,
    }

    let rebalance_depth = 1;
    let mut best = None;

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
                greedy_cost_function: greedy_cost_fn,
                communication_scheme,
                balancing_scheme: BalancingScheme::BestWorst,
            },
            &Some(DendogramSettings {
                output_file: format!("output/{communication_scheme:?}_trial"),
                cost_function: contract_cost_tensors,
            }),
        );
        let candidate = Candidate {
            cost: costs[num],
            _iteration: num,
            _method: communication_scheme,
        };
        info!("Candidate: {candidate:?}");

        if best.map_or(true, |best: Candidate| best.cost > candidate.cost) {
            best = Some(candidate);
        }
    }
    info!("Best scheme: {best:?}");

    // contract_tensor_network(&mut new_tn, &contraction_path);
}
