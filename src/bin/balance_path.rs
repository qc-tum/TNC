extern crate tensorcontraction;

use rand::rngs::StdRng;
use rand::SeedableRng;

use tensorcontraction::contractionpath::contraction_cost::contract_cost_tensors;
use tensorcontraction::contractionpath::contraction_tree::{
    balance_partitions_iter, repartition_tn,
};
use tensorcontraction::contractionpath::paths::greedy::Greedy;
use tensorcontraction::contractionpath::paths::{CostType, OptimizePath};
use tensorcontraction::networks::connectivity::ConnectivityLayout;
use tensorcontraction::networks::sycamore::random_circuit;
use tensorcontraction::tensornetwork::contraction::contract_tensor_network;
use tensorcontraction::tensornetwork::partitioning::{find_partitioning, partition_tensor_network};

use std::time::{Duration, Instant};

fn main() {
    let mut rng: StdRng = StdRng::seed_from_u64(27);

    let num_qubits = 15;
    let circuit_depth = 5;
    let single_qubit_probability = 0.4;
    let two_qubit_probability = 0.4;
    let connectivity = ConnectivityLayout::Osprey;
    let num_partitions = 3;

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
    let rebalance_depth = 1;
    let (cost, mut new_tn, contraction_path, costs) = balance_partitions_iter(
        &partitioned_tn,
        &path,
        false,
        rebalance_depth,
        200,
        String::from("output/rebalance_trial"),
        contract_cost_tensors,
    );
    // repartition_tn(&partitioned_tn, &new_tree, rebalance_depth);
    println!("partitioned_tn: {:?}", new_tn.num_tensors());
    contract_tensor_network(&mut new_tn, &contraction_path);
    println!("new_tn: {:?}", new_tn);
    println!("{:?}", now.duration_since(now));
}
