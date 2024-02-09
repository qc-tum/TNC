extern crate tensorcontraction;
use rand::rngs::StdRng;
use rand::SeedableRng;
use tensorcontraction::{
    circuits::sycamore::{sycamore_circuit, sycamore_contract},
    contractionpath::paths::{BranchBound, CostType, OptimizePath},
    random::tensorgeneration::{random_sparse_tensor, random_tensor_network},
    tensornetwork::TensorNetwork,
};

fn main() {
    let r_tn = random_tensor_network(4, 3);
    let mut d_tn = Vec::new();
    for r_t in r_tn.get_tensors() {
        d_tn.push(random_sparse_tensor(r_t, r_tn.get_bond_dims(), None));
    }
    let mut opt = BranchBound::new(&r_tn, None, 20, CostType::Flops);
    opt.optimize_path();

    let mut rng = StdRng::seed_from_u64(52);

    let k = 5;
    let tn = sycamore_circuit(5, k, None, None, &mut rng);
    sycamore_contract(tn);
}
