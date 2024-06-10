extern crate tensorcontraction;

use rand::rngs::StdRng;
use rand::SeedableRng;
use std::cmp::max;
use tensorcontraction::contractionpath::contraction_cost::{
    self, contract_cost_tensors, contract_path_cost, contract_size_tensors,
};
use tensorcontraction::contractionpath::contraction_tree::{
    balance_path, balance_path_iter, find_min_max_subtree, to_dendogram, to_png,
    tree_contraction_cost, ContractionTree,
};
use tensorcontraction::contractionpath::paths::{greedy::Greedy, CostType, OptimizePath};
use tensorcontraction::mpi::communication::{
    intermediate_reduce_tensor_network, scatter_tensor_network,
};
use tensorcontraction::networks::connectivity::{self, ConnectivityLayout};
use tensorcontraction::networks::sycamore::sycamore_circuit;
use tensorcontraction::path;
use tensorcontraction::tensornetwork::create_tensor_network;
use tensorcontraction::tensornetwork::partitioning::{find_partitioning, partition_tensor_network};
use tensorcontraction::tensornetwork::tensor::Tensor;
use tensorcontraction::types::ContractionIndex;

fn setup_complex() -> (Tensor, Vec<ContractionIndex>) {
    (
        create_tensor_network(
            vec![
                Tensor::new(vec![4, 3, 2]),
                Tensor::new(vec![0, 1, 3, 2]),
                Tensor::new(vec![4, 5, 6]),
                Tensor::new(vec![6, 8, 9]),
                Tensor::new(vec![10, 8, 9]),
                Tensor::new(vec![5, 1, 0]),
            ],
            &[
                (0, 4),
                (1, 3),
                (2, 2),
                (3, 3),
                (4, 2),
                (5, 3),
                (6, 4),
                (7, 1),
                (8, 2),
                (9, 3),
                (10, 5),
            ]
            .into(),
            None,
        ),
        path![(1, 5), (0, 1), (3, 4), (2, 3), (0, 2)].to_vec(),
    )
}

/// Returns Schroedinger contraction space complexity of fully contracting a nested [Tensor] object
///
/// # Arguments
///
/// * `inputs` - First tensor to determine contraction cost.
/// * `ssa_path`  - Contraction order as replacement path
/// * `bond_dims`- Dict of bond dimensions.
pub fn custom_contract_path_cost(
    inputs: &[Tensor],
    contract_path: &[ContractionIndex],
) -> (u64, u64) {
    let mut op_cost = 0;
    let mut mem_cost = 0;
    let mut inputs = inputs.to_vec();
    for index in contract_path {
        if let ContractionIndex::Pair(i, j) = *index {
            op_cost += contract_cost_tensors(&inputs[i], &inputs[j]);
            let k12 = &inputs[i] ^ &inputs[j];
            let new_mem_cost = contract_size_tensors(&inputs[i], &inputs[j]);
            mem_cost = max(mem_cost, new_mem_cost);
            inputs[i] = k12;
        }
    }

    (op_cost, mem_cost)
}
// Run with at least 2 processes
fn main() {
    let mut rng = StdRng::seed_from_u64(27);
    // let (tn, _) = setup_complex();
    let size = 6;
    let round = 5;
    let single_qubit_probability = 0.4;
    let two_qubit_probability = 0.4;
    let connectivity = ConnectivityLayout::Osprey;

    let tn = sycamore_circuit(
        size,
        round,
        single_qubit_probability,
        two_qubit_probability,
        &mut rng,
        connectivity,
    );

    println!("tn: {:?}", tn.tensors().len());
    let mut opt = Greedy::new(&tn, CostType::Flops);

    opt.optimize_path();
    let path = opt.get_best_replace_path();

    let contraction_tree = ContractionTree::from_contraction_path(&tn, &path);
    to_dendogram(
        &contraction_tree,
        &tn,
        contract_cost_tensors,
        String::from("output-test"),
    );

    let rebalance_depth = 1;
    let random_balance = false;
    let output_file = String::from("output/sycamore_experiment");
    let iterations = 120;

    let (best_iteration, best_path) = balance_path_iter(
        &tn,
        &path,
        random_balance,
        rebalance_depth,
        iterations,
        output_file,
        contract_cost_tensors,
    );

    println!("Best path:{:?}", best_iteration);
    println!("Best contraction: {:?}", best_path);
}
