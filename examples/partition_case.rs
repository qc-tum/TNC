use rand::rngs::StdRng;
use rand::SeedableRng;
use tensorcontraction::contractionpath::contraction_cost::contract_cost_tensors;
use tensorcontraction::contractionpath::contraction_tree::balance_partitions_iter;
use tensorcontraction::contractionpath::paths::{greedy::Greedy, CostType, OptimizePath};
use tensorcontraction::mpi::communication::{
    broadcast_path, intermediate_reduce_tensor_network, scatter_tensor_network, CommunicationScheme,
};
use tensorcontraction::networks::connectivity::ConnectivityLayout;
use tensorcontraction::networks::sycamore::random_circuit;
use tensorcontraction::tensornetwork::contraction::contract_tensor_network;
use tensorcontraction::tensornetwork::partitioning::{find_partitioning, partition_tensor_network};

use mpi::traits::{Communicator, CommunicatorCollectives};
use tensorcontraction::tensornetwork::tensor::Tensor;

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

    if rank == 0 {
        println!("Running with {size} processes");
    }

    let qubits = 5;
    let rounds = 10;

    // Setup tensor network
    // TODO: Do we need to know communication beforehand?
    let (mut partitioned_tn, path) = if rank == 0 {
        let r_tn = random_circuit(
            qubits,
            rounds,
            0.4,
            0.4,
            &mut rng,
            ConnectivityLayout::Osprey,
        );
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

        let rebalance_depth = 1;
        let (_cost, _partitioned_tn, _path, _costs) = balance_partitions_iter(
            &partitioned_tn,
            &path,
            false,
            rebalance_depth,
            10,
            String::from("output/rebalance_trial"),
            contract_cost_tensors,
            greedy_cost_fn,
            CommunicationScheme::Greedy,
        );

        (partitioned_tn, path)
    } else {
        Default::default()
    };
    // println!("partitioned_tn: {:?}", partitioned_tn);
    // println!("path: {:?}", path);
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
        println!("{local_tn:?}");
    }
}
