extern crate tensorcontraction;

use rand::rngs::StdRng;
use rand::SeedableRng;
use tensorcontraction::circuits::sycamore::sycamore_circuit;
use tensorcontraction::contractionpath::paths::{CostType, Greedy, OptimizePath};
use tensorcontraction::mpi::scatter::{gather_tensor_network, scatter_tensor_network};
use tensorcontraction::tensornetwork::contraction::contract_tensor_network;
use tensorcontraction::tensornetwork::partitioning::{find_partitioning, partition_tensor_network};

use mpi::traits::*;
use tensorcontraction::tensornetwork::tensor::Tensor;

// Run with at least 2 processes
fn main() {
    let mut rng = StdRng::seed_from_u64(23);

    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let size = world.size();
    let rank = world.rank();

    // let status;
    let mut partitioned_tn = Tensor::default();
    let mut path = Vec::new();
    if rank == 0 {
        let k = 5;
        let mut r_tn = sycamore_circuit(k, 5, None, None, &mut rng);
        let partitioning =
            find_partitioning(&mut r_tn, size, std::string::String::from("tests/km1"));
        partitioned_tn = partition_tensor_network(&r_tn, &partitioning);
        let mut opt = Greedy::new(&partitioned_tn, CostType::Flops);

        opt.optimize_path();
        path = opt.get_best_replace_path();
    }
    world.barrier();
    let (mut local_tn, local_path) =
        scatter_tensor_network(partitioned_tn, &path, rank, size, &world);
    contract_tensor_network(&mut local_tn, &local_path);

    gather_tensor_network(local_tn, &path, rank, size, &world);
}
