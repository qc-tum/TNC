use rand::rngs::StdRng;
use rand::SeedableRng;

use tensorcontraction::contractionpath::contraction_cost::contract_cost_tensors;
use tensorcontraction::contractionpath::contraction_tree::{
    balance_partitions_iter, BalanceSettings,
};
use tensorcontraction::contractionpath::paths::greedy::Greedy;
use tensorcontraction::contractionpath::paths::{CostType, OptimizePath};
use tensorcontraction::mpi::communication::CommunicationScheme;
use tensorcontraction::networks::connectivity::ConnectivityLayout;
use tensorcontraction::networks::sycamore::random_circuit;
use tensorcontraction::tensornetwork::contraction::contract_tensor_network;
use tensorcontraction::tensornetwork::partitioning::partition_config::PartitioningStrategy;
use tensorcontraction::tensornetwork::partitioning::{find_partitioning, partition_tensor_network};
use tensorcontraction::tensornetwork::tensor::Tensor;

fn greedy_cost_fn(t1: &Tensor, t2: &Tensor) -> f64 {
    t1.size() as f64 + t2.size() as f64 - (t1 ^ t2).size() as f64
}

fn main() {
    let mut rng: StdRng = StdRng::seed_from_u64(27);

    let num_qubits = 20;
    let circuit_depth = 30;
    let single_qubit_probability = 0.4;
    let two_qubit_probability = 0.4;
    let connectivity = ConnectivityLayout::Osprey;
    let num_partitions = 8;

    let tensor = random_circuit(
        num_qubits,
        circuit_depth,
        single_qubit_probability,
        two_qubit_probability,
        &mut rng,
        connectivity,
    );

    // println!("Tensor: {:?}", tensor.total_num_tensors());
    // Find vec of partitions
    let partitioning =
        find_partitioning(&tensor, num_partitions, PartitioningStrategy::MinCut, true);

    let partitioned_tn = partition_tensor_network(&tensor, &partitioning);
    let mut opt = Greedy::new(&partitioned_tn, CostType::Flops);
    opt.optimize_path();
    let path = opt.get_best_replace_path();

    let rebalance_depth = 1;
    let (_num, mut new_tn, contraction_path, _costs) = balance_partitions_iter(
        &partitioned_tn,
        &path,
        BalanceSettings {
            random_balance: false,
            rebalance_depth,
            iterations: 30,
            output_file: String::from("output/rebalance_trial"),
            dendogram_cost_function: contract_cost_tensors,
            greedy_cost_function: greedy_cost_fn,
            communication_scheme: CommunicationScheme::Greedy,
        },
    );
    contract_tensor_network(&mut new_tn, &contraction_path);
}
