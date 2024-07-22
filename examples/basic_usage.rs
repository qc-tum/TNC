use flexi_logger::{opt_format, Duplicate, FileSpec, Logger};
use log::{debug, info, LevelFilter};
use mpi::traits::Communicator;
use mpi::Rank;
use rand::rngs::StdRng;
use rand::SeedableRng;
use tensorcontraction::contractionpath::paths::{greedy::Greedy, CostType, OptimizePath};
use tensorcontraction::mpi::communication::{
    broadcast_path, intermediate_reduce_tensor_network, scatter_tensor_network,
};
use tensorcontraction::networks::connectivity::ConnectivityLayout;
use tensorcontraction::networks::sycamore::random_circuit;
use tensorcontraction::tensornetwork::contraction::contract_tensor_network;
use tensorcontraction::tensornetwork::partitioning::{find_partitioning, partition_tensor_network};

/// Sets up logging for rank `rank`. Each rank logs to a separate file and to stdout.
fn setup_logging_mpi(rank: Rank) {
    let _logger = Logger::with(LevelFilter::Debug)
        .format(opt_format)
        .log_to_file(
            FileSpec::default()
                .discriminant(format!("rank{}", rank))
                .suppress_timestamp(),
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
    info!("Running basic_usage");
    info!("This is rank {rank} of {size}");

    let seed = 23;
    let qubits = 20;
    let depth = 10;
    let single_qubit_probability = 0.4;
    let two_qubit_probability = 0.4;
    let connectivity = ConnectivityLayout::Osprey;
    info!("Configuration: seed={seed}, qubits={qubits}, depth={depth}, single_qubit_probability={single_qubit_probability}, two_qubit_probability={two_qubit_probability}, connectivity={connectivity:?}");

    // Setup tensor network
    let (mut partitioned_tn, path) = if rank == 0 {
        let r_tn = random_circuit(
            qubits,
            depth,
            single_qubit_probability,
            two_qubit_probability,
            &mut rng,
            connectivity,
        );
        debug!("Tensors: {}", r_tn.total_num_tensors());

        let partitioned_tn = if size > 1 {
            let partitioning = find_partitioning(
                &r_tn,
                size,
                String::from("tests/km1_kKaHyPar_sea20.ini"),
                true,
            );
            debug!("Partitioning: {partitioning:?}");
            partition_tensor_network(&r_tn, &partitioning)
        } else {
            r_tn
        };
        let mut opt = Greedy::new(&partitioned_tn, CostType::Flops);

        opt.optimize_path();
        let path = opt.get_best_replace_path();
        debug!("Contraction path: {path:?}");
        (partitioned_tn, path)
    } else {
        Default::default()
    };

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
}
