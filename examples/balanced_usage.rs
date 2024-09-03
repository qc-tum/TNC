use flexi_logger::writers::FileLogWriter;
use flexi_logger::{json_format, Duplicate, FileSpec, Logger, LoggerHandle};
use log::{debug, info, LevelFilter};
use mpi::traits::Communicator;
use mpi::Rank;
use rand::rngs::StdRng;
use rand::SeedableRng;
use tensorcontraction::contractionpath::contraction_cost::contract_cost_tensors;
use tensorcontraction::contractionpath::contraction_tree::export::{to_dendogram_format, to_pdf};
use tensorcontraction::contractionpath::contraction_tree::{
    balance_partitions_iter, BalanceSettings, ContractionTree, DendogramSettings,
};
use tensorcontraction::contractionpath::paths::{greedy::Greedy, CostType, OptimizePath};
use tensorcontraction::mpi::communication::{
    broadcast_path, intermediate_reduce_tensor_network, scatter_tensor_network, CommunicationScheme,
};
use tensorcontraction::networks::connectivity::ConnectivityLayout;
use tensorcontraction::networks::sycamore::random_circuit;
use tensorcontraction::tensornetwork::contraction::contract_tensor_network;
use tensorcontraction::tensornetwork::partitioning::partition_config::PartitioningStrategy;
use tensorcontraction::tensornetwork::partitioning::{find_partitioning, partition_tensor_network};
use tensorcontraction::tensornetwork::tensor::Tensor;

/// Sets up logging for rank `rank`. Each rank logs to a separate file and to stdout.
fn setup_logging_mpi(rank: Rank) -> LoggerHandle {
    Logger::with(LevelFilter::Debug)
        .format(json_format)
        .log_to_file(
            FileSpec::default()
                .discriminant(format!("rank{rank}"))
                .suppress_timestamp()
                .suffix("log.json")
                .directory("logs/run2"),
        )
        .duplicate_to_stdout(Duplicate::Info)
        .start()
        .unwrap()
}

fn greedy_cost_fn(t1: &Tensor, t2: &Tensor) -> f64 {
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
    let qubits = 30;
    let depth = 40;
    let single_qubit_probability = 0.4;
    let two_qubit_probability = 0.4;
    let connectivity = ConnectivityLayout::Osprey;
    info!(seed, qubits, depth, single_qubit_probability, two_qubit_probability, connectivity:?; "Configuration set");

    // Setup tensor network
    let (mut partitioned_tn, path, mut unopt_partitioned_tn, unopt_path) = if rank == 0 {
        let r_tn = random_circuit(
            qubits,
            depth,
            single_qubit_probability,
            two_qubit_probability,
            &mut rng,
            connectivity,
        );

        let unopt_partitioned_tn = if size > 1 {
            let partitioning = find_partitioning(&r_tn, size, PartitioningStrategy::MinCut, true);
            debug!(partitioning:serde; "Partitioning created");
            partition_tensor_network(&r_tn, &partitioning)
        } else {
            r_tn
        };
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
        to_pdf("unoptimized_path", &dendogram_entries);

        let rebalance_depth = 1;
        let communication_scheme = CommunicationScheme::Greedy;
        let (num, partitioned_tn, path, _costs) = balance_partitions_iter(
            &unopt_partitioned_tn,
            &unopt_path,
            BalanceSettings {
                random_balance: false,
                rebalance_depth,
                iterations: 40,
                greedy_cost_function: greedy_cost_fn,
                communication_scheme,
            },
            Some(DendogramSettings {
                output_file: format!("output/{communication_scheme:?}_trial"),
                cost_function: contract_cost_tensors,
            }),
        );
        info!(num; "Found best balancing iteration");

        let contraction_tree = ContractionTree::from_contraction_path(&partitioned_tn, &path);
        let dendogram_entries =
            to_dendogram_format(&contraction_tree, &partitioned_tn, contract_cost_tensors);
        to_pdf("optimized_path", &dendogram_entries);

        (partitioned_tn, path, unopt_partitioned_tn, unopt_path)
    } else {
        Default::default()
    };
    logger.flush();
    logger
        .reset_flw(
            &FileLogWriter::builder(
                FileSpec::default()
                    .discriminant(format!("rank{}", rank))
                    .suppress_timestamp()
                    .suffix("opt.log.json"),
            )
            .format(json_format),
        )
        .unwrap();

    // Distribute tensor network and contract
    let local_tn = if size > 1 {
        let (mut local_tn, local_path) =
            scatter_tensor_network(&partitioned_tn, &path, rank, size, &world);
        contract_tensor_network(&mut local_tn, &local_path);

        let path = if rank == 0 {
            broadcast_path(&path[(size as usize)..path.len()], &root, &world)
        } else {
            broadcast_path(&[], &root, &world)
        };
        intermediate_reduce_tensor_network(&mut local_tn, &path, rank, &world);
        local_tn
    } else {
        contract_tensor_network(&mut partitioned_tn, &path);
        partitioned_tn
    };

    // Print the result
    if rank == 0 {
        info!("{local_tn:?}");
    }

    logger
        .reset_flw(
            &FileLogWriter::builder(
                FileSpec::default()
                    .discriminant(format!("rank{}", rank))
                    .suppress_timestamp()
                    .suffix("unopt.log.json"),
            )
            .format(json_format)
            .append(),
        )
        .unwrap();

    // Distribute tensor network and contract
    let local_tn = if size > 1 {
        let (mut local_tn, local_path) =
            scatter_tensor_network(&unopt_partitioned_tn, &unopt_path, rank, size, &world);
        contract_tensor_network(&mut local_tn, &local_path);

        let path = if rank == 0 {
            broadcast_path(&path[(size as usize)..path.len()], &root, &world)
        } else {
            broadcast_path(&[], &root, &world)
        };
        intermediate_reduce_tensor_network(&mut local_tn, &path, rank, &world);
        local_tn
    } else {
        contract_tensor_network(&mut unopt_partitioned_tn, &unopt_path);
        unopt_partitioned_tn
    };

    // Print the result
    if rank == 0 {
        info!("{local_tn:?}");
    }
}
