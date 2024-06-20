extern crate tensorcontraction;

use rand::rngs::StdRng;
use rand::SeedableRng;

use tensorcontraction::contractionpath::contraction_cost::contract_cost_tensors;
use tensorcontraction::contractionpath::contraction_tree::balance_partitions_iter;
use tensorcontraction::contractionpath::paths::greedy::Greedy;
use tensorcontraction::contractionpath::paths::{CostType, OptimizePath};
use tensorcontraction::networks::connectivity::ConnectivityLayout;
use tensorcontraction::networks::sycamore::random_circuit;
use tensorcontraction::tensornetwork::partitioning::{find_partitioning, partition_tensor_network};

use std::time::{Duration, Instant};

fn main() {
    let mut rng: StdRng = StdRng::seed_from_u64(27);

    let num_qubits = 20;
    let circuit_depth = 10;
    let single_qubit_probability = 0.4;
    let two_qubit_probability = 0.4;
    let connectivity = ConnectivityLayout::Osprey;
    let num_partitions = 4;

    let tensor = random_circuit(
        num_qubits,
        circuit_depth,
        single_qubit_probability,
        two_qubit_probability,
        &mut rng,
        connectivity,
    );
    // Find vec of partitions
    let partitioning = find_partitioning(
        &tensor,
        num_partitions,
        String::from("tests/km1_kKaHyPar_sea20.ini"),
        true,
    );

    let partitioned_tn = partition_tensor_network(&tensor, &partitioning);
    let mut opt = Greedy::new(&partitioned_tn, CostType::Flops);
    opt.optimize_path();
    let path = opt.get_best_replace_path();
    let now = Instant::now();
    let (cost, new_tree, contraction_path, costs) = balance_partitions_iter(
        &partitioned_tn,
        &path,
        false,
        1,
        120,
        String::from("output/rebalance_trial"),
        contract_cost_tensors,
    );
    println!("{:?}", now.duration_since(now));
}
