use mpi::traits::{Communicator, CommunicatorCollectives};
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

// Run with at least 2 processes
fn main() {
    let mut rng = StdRng::seed_from_u64(23);
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let size = world.size();
    let rank = world.rank();
    let root = world.process_at_rank(0);

    if rank == 0 {
        println!("Running with {size} processes");
    }

    let k = 20;

    // Setup tensor network
    // TODO: Do we need to know communication beforehand?
    let (mut partitioned_tn, path) = if rank == 0 {
        let r_tn = random_circuit(k, 10, 0.4, 0.4, &mut rng, ConnectivityLayout::Osprey);
        let partitioned_tn = if size > 1 {
            let partitioning = find_partitioning(
                &r_tn,
                size,
                String::from("tests/km1_kKaHyPar_sea20.ini"),
                true,
            );
            partition_tensor_network(&r_tn, &partitioning)
        } else {
            r_tn
        };
        let mut opt = Greedy::new(&partitioned_tn, CostType::Flops);

        opt.optimize_path();
        let path = opt.get_best_replace_path();
        (partitioned_tn, path)
    } else {
        Default::default()
    };
    world.barrier();

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
        world.barrier();
        intermediate_reduce_tensor_network(&mut local_tn, &path, rank, size, &world);
        local_tn
    } else {
        contract_tensor_network(&mut partitioned_tn, &path);
        partitioned_tn
    };

    // Print the result
    if rank == 0 {
        println!("{local_tn:?}");
    }
}
