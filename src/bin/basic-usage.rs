extern crate tensorcontraction;
use tensorcontraction::contractionpath::paths::{BranchBound, CostType, OptimizePath};
use tensorcontraction::random::tensorgeneration::{random_sparse_tensor, random_tensor_network};
// use tensorcontraction::tensornetwork::contraction::tn_contract;
use tensorcontraction::tensornetwork::TensorNetwork;

fn main() {
    let r_tn = random_tensor_network(4, 3);
    let mut d_tn = Vec::new();
    for r_t in r_tn.get_tensors() {
        d_tn.push(random_sparse_tensor(r_t, r_tn.get_bond_dims(), None));
    }
    let mut opt = BranchBound::new(&r_tn, None, 20, CostType::Flops);
    opt.optimize_path();
    let opt_path = opt.get_best_replace_path();

    // tn_contract(r_tn, d_tn, &opt_path);

    let mut rng = StdRng::seed_from_u64(52);

    let k = 5;
    let tn: TensorNetwork = sycamore_circuit(5, k, None, None, &mut rng);
    sycamore_contract(tn.clone())
    // mul_group.bench_function(BenchmarkId::from_parameter(k), |b| {
    //     b.iter(|| sycamore_contract(tn.clone()));
    // });
}
