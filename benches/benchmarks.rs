#![feature(test)]

use tensorcontraction::tensornetwork::contraction::tn_contract;
use tensorcontraction::{
    contractionpath::paths::{BranchBound, BranchBoundType, OptimizePath},
    random::tensorgeneration::{random_sparse_tensor, random_tensor_network},
    tensornetwork::{tensor::Tensor, TensorNetwork},
};

extern crate test;
use test::Bencher;

fn _setup() -> TensorNetwork {
    TensorNetwork::from_vector(
        vec![Tensor::new(vec![4, 3, 2]), Tensor::new(vec![0, 1, 3, 2])],
        vec![17, 18, 19, 12, 22],
    )
}

#[bench]
fn build_tensor(_b: &mut Bencher) {
    let n = 5;
    let cycles = 5;
    random_tensor_network(n, cycles);
}

#[bench]
fn contract_tensor_(_b: &mut Bencher) {
    let n = 5;
    let cycles = 5;
    let tn = random_tensor_network(n, cycles);
    let mut bb = BranchBound::new(tn.clone(), None, 50, BranchBoundType::Flops);
    bb.optimize_path(None);
    let mut d_tn = Vec::new();
    for r_t in tn.clone().get_tensors() {
        d_tn.push(random_sparse_tensor(
            r_t.clone(),
            &tn.get_bond_dims(),
            None,
        ));
    }
    tn_contract(tn, d_tn, &vec![(0, 1), (0, 1)]);
}
