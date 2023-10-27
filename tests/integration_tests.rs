use rand::{rngs::StdRng, SeedableRng};
use tensorcontraction::{
    circuits::sycamore::sycamore_circuit,
    contractionpath::paths::{CostType, Greedy, OptimizePath},
    tensornetwork::{
        contraction::contract_tensor_network,
        partitioning::{find_partitioning, partition_tensor_network},
    },
};

#[test]
fn test_partitioned_contraction() {
    let mut rng = StdRng::seed_from_u64(52);
    let k = 5;

    let mut r_tn = sycamore_circuit(k, 15, None, None, &mut rng);
    let partitioning = find_partitioning(&mut r_tn, 5, std::string::String::from("tests/km1"));
    let partitioned_tn = partition_tensor_network(&r_tn, &partitioning);
    let mut opt = Greedy::new(&partitioned_tn, CostType::Flops);
    opt.optimize_path();
    let path = opt.get_best_replace_path();
    contract_tensor_network(&mut partitioned_tn.clone(), &path);
}
