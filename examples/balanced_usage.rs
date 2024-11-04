use flexi_logger::writers::FileLogWriter;
use flexi_logger::{json_format, Duplicate, FileSpec, Logger, LoggerHandle};
use itertools::iproduct;
use log::{debug, info, LevelFilter};
use mpi::topology::{Process, SimpleCommunicator};
use mpi::traits::Communicator;
use mpi::Rank;
use rand::rngs::StdRng;
use rand::SeedableRng;
use tensorcontraction::contractionpath::contraction_cost::contract_cost_tensors;
use tensorcontraction::contractionpath::contraction_tree::balancing::balancing_schemes::BalancingScheme;
use tensorcontraction::contractionpath::contraction_tree::balancing::communication_schemes::CommunicationScheme;
use tensorcontraction::contractionpath::contraction_tree::balancing::{
    balance_partitions_iter, BalanceSettings,
};
use tensorcontraction::contractionpath::contraction_tree::export::{
    to_dendogram_format, to_pdf, DendogramSettings,
};
use tensorcontraction::contractionpath::contraction_tree::ContractionTree;
use tensorcontraction::contractionpath::paths::{greedy::Greedy, CostType, OptimizePath};
use tensorcontraction::mpi::communication::{
    broadcast_path, extract_communication_path, intermediate_reduce_tensor_network,
    scatter_tensor_network,
};
use tensorcontraction::networks::connectivity::ConnectivityLayout;
use tensorcontraction::networks::sycamore::random_circuit;
use tensorcontraction::tensornetwork::contraction::contract_tensor_network;
use tensorcontraction::tensornetwork::partitioning::partition_config::PartitioningStrategy;
use tensorcontraction::tensornetwork::partitioning::{find_partitioning, partition_tensor_network};
use tensorcontraction::tensornetwork::tensor::Tensor;
use tensorcontraction::types::ContractionIndex;

static LOGGING_FOLDER: &str = "logs/run";
/// Sets up logging for rank `rank`. Each rank logs to a separate file and to stdout.
fn setup_logging_mpi(rank: Rank) -> LoggerHandle {
    Logger::with(LevelFilter::Debug)
        .format(json_format)
        .log_to_file(
            FileSpec::default()
                .discriminant(format!("rank{rank}"))
                .suppress_timestamp()
                .suffix("log.json")
                .directory(LOGGING_FOLDER),
        )
        .duplicate_to_stdout(Duplicate::Info)
        .start()
        .unwrap()
}

fn objective_function(t1: &Tensor, t2: &Tensor) -> f64 {
    t1.size() as f64 + t2.size() as f64 - (t1 ^ t2).size() as f64
}

// Run with at least 2 processes
fn main() {
    let mut rng = StdRng::seed_from_u64(23);
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let size = world.size();
    let rank = world.rank();
    let root = world.process_at_rank(0);
    let logger = setup_logging_mpi(rank);
    info!(rank, size; "Logging setup");

    let seed = 23;
    let qubits = 15;
    let depth = 40;
    let single_qubit_probability = 0.4;
    let two_qubit_probability = 0.4;
    let connectivity = ConnectivityLayout::Osprey;

    let unopt_partitioned_tn = if rank == 0 {
        info!(seed, qubits, depth, single_qubit_probability, two_qubit_probability, connectivity:?; "Configuration set");
        let r_tn = random_circuit(
            qubits,
            depth,
            single_qubit_probability,
            two_qubit_probability,
            &mut rng,
            connectivity,
        );
        let partitioning = find_partitioning(&r_tn, size, PartitioningStrategy::MinCut, true);
        debug!(tn_size = partitioning.len(); "TN size");
        debug!(partitioning:serde; "Partitioning created");
        partition_tensor_network(&r_tn, &partitioning)
    } else {
        Default::default()
    };

    let unopt_path = if rank == 0 {
        let mut opt = Greedy::new(&unopt_partitioned_tn, CostType::Flops);
        opt.optimize_path();
        let unopt_path = opt.get_best_replace_path();
        debug!(unopt_path:serde; "Found contraction path");

        let contraction_tree =
            ContractionTree::from_contraction_path(&unopt_partitioned_tn, &unopt_path);
        let dendogram_entries = to_dendogram_format(
            &contraction_tree,
            &unopt_partitioned_tn,
            contract_cost_tensors,
        );
        to_pdf(
            &format!("{LOGGING_FOLDER}/path_Unoptimized"),
            &dendogram_entries,
            None,
        );
        unopt_path
    } else {
        Default::default()
    };

    // Experimental setup here.
    let balancing_schemes = [
        // BalancingScheme::BestWorst,
        // BalancingScheme::Tensor,
        BalancingScheme::Tensors,
    ];
    let communication_schemes = [
        CommunicationScheme::Greedy,
        // CommunicationScheme::Bipartition,
    ];
    let iterations = 30;
    let rebalance_depth = 1;

    for (balancing_scheme, communication_scheme) in
        iproduct!(balancing_schemes, communication_schemes)
    {
        let name = format!("{balancing_scheme:?}_{communication_scheme:?}");

        let (partitioned_tn, path) = if rank == 0 {
            info!("Running: {name}");

            let (num, partitioned_tn, path, _costs) = balance_partitions_iter(
                &unopt_partitioned_tn,
                &unopt_path,
                BalanceSettings {
                    random_balance: None,
                    rebalance_depth,
                    iterations,
                    objective_function,
                    communication_scheme,
                    balancing_scheme,
                },
                Some(&DendogramSettings {
                    output_file: format!("output/{name}_trial"),
                    objective_function: contract_cost_tensors,
                }),
            );
            info!(num; "Found best balancing iteration");
            let contraction_tree = ContractionTree::from_contraction_path(&partitioned_tn, &path);
            let dendogram_entries =
                to_dendogram_format(&contraction_tree, &partitioned_tn, contract_cost_tensors);
            to_pdf(
                &format!("{LOGGING_FOLDER}/path_{name}"),
                &dendogram_entries,
                None,
            );
            (partitioned_tn, path)
        } else {
            Default::default()
        };

        bench_run(&logger, &name, partitioned_tn, &path, &world, root);
    }

    let name = "Unoptimized";

    bench_run(
        &logger,
        name,
        unopt_partitioned_tn,
        &unopt_path,
        &world,
        root,
    );
}

fn bench_run(
    logger: &LoggerHandle,
    name: &str,
    mut partitioned_tn: Tensor,
    path: &[ContractionIndex],
    world: &SimpleCommunicator,
    root: Process,
) {
    let rank = world.rank();
    let size = world.size();
    logger.flush();
    logger
        .reset_flw(
            &FileLogWriter::builder(
                FileSpec::default()
                    .discriminant(format!("rank{rank}"))
                    .suppress_timestamp()
                    .suffix(format!("{name}.log.json"))
                    .directory(LOGGING_FOLDER),
            )
            .format(json_format),
        )
        .unwrap();

    // Distribute tensor network and contract
    let local_tn = if size > 1 {
        let (mut local_tn, local_path) =
            scatter_tensor_network(&partitioned_tn, path, rank, size, world);
        contract_tensor_network(&mut local_tn, &local_path);

        let mut communication_path = if rank == 0 {
            extract_communication_path(path)
        } else {
            Default::default()
        };
        broadcast_path(&mut communication_path, &root, world);

        intermediate_reduce_tensor_network(&mut local_tn, &communication_path, rank, world);
        local_tn
    } else {
        contract_tensor_network(&mut partitioned_tn, path);
        partitioned_tn
    };

    std::hint::black_box(local_tn);
}
