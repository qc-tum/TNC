use flexi_logger::{json_format, Duplicate, FileSpec, Logger};
use log::{debug, info, LevelFilter};
use mpi::traits::Communicator;
use mpi::Rank;
use rand::rngs::StdRng;
use rand::SeedableRng;
use tnc::builders::connectivity::ConnectivityLayout;
use tnc::builders::random_circuit::random_circuit;
use tnc::contractionpath::contraction_cost::contract_cost_tensors;
use tnc::contractionpath::contraction_tree::export::{to_dendogram_format, to_pdf};
use tnc::contractionpath::contraction_tree::ContractionTree;
use tnc::contractionpath::paths::cotengrust::{Cotengrust, OptMethod};
use tnc::contractionpath::paths::OptimizePath;
use tnc::mpi::communication::{
    broadcast_path, extract_communication_path, intermediate_reduce_tensor_network,
    scatter_tensor_network,
};
use tnc::tensornetwork::contraction::contract_tensor_network;
use tnc::tensornetwork::partitioning::partition_config::PartitioningStrategy;
use tnc::tensornetwork::partitioning::{find_partitioning, partition_tensor_network};

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

// Run with at least 2 processes
fn main() {
    let mut rng = StdRng::seed_from_u64(23);
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let size = world.size();
    let rank = world.rank();
    let root = world.process_at_rank(0);
    setup_logging_mpi(rank);
    info!(rank, size; "Logging setup");

    let seed = 23;
    let qubits = 5;
    let depth = 30;
    let single_qubit_probability = 0.4;
    let two_qubit_probability = 0.4;
    let connectivity = ConnectivityLayout::Osprey;
    info!(seed, qubits, depth, single_qubit_probability, two_qubit_probability, connectivity:?; "Configuration set");

    // Setup tensor network
    let (partitioned_tn, path) = if rank == 0 {
        let r_tn = random_circuit(
            qubits,
            depth,
            single_qubit_probability,
            two_qubit_probability,
            &mut rng,
            connectivity,
        );

        let partitioned_tn = if size > 1 {
            let partitioning = find_partitioning(&r_tn, size, PartitioningStrategy::MinCut, true);
            debug!(partitioning:serde; "Partitioning created");
            partition_tensor_network(r_tn, &partitioning)
        } else {
            r_tn
        };

        let mut opt = Cotengrust::new(&partitioned_tn, OptMethod::Greedy);
        opt.optimize_path();
        let path = opt.get_best_replace_path();
        debug!(path:serde; "Found contraction path");

        let contraction_tree = ContractionTree::from_contraction_path(&partitioned_tn, &path);
        let dendogram_entries =
            to_dendogram_format(&contraction_tree, &partitioned_tn, contract_cost_tensors);
        to_pdf("from_path", &dendogram_entries, None);
        (partitioned_tn, path)
    } else {
        Default::default()
    };

    // Distribute tensor network and contract
    let local_tn = if size > 1 {
        let (mut local_tn, local_path, comm) =
            scatter_tensor_network(&partitioned_tn, &path, rank, size, &world);
        local_tn = contract_tensor_network(local_tn, &local_path);

        let mut communication_path = if rank == 0 {
            extract_communication_path(&path)
        } else {
            Default::default()
        };
        broadcast_path(&mut communication_path, &root);

        intermediate_reduce_tensor_network(&mut local_tn, &communication_path, rank, &world, &comm);
        local_tn
    } else {
        contract_tensor_network(partitioned_tn, &path)
    };

    // Print the result
    if rank == 0 {
        info!("{local_tn:?}");
    }
}
