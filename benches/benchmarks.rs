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
fn build_tensor(b: &mut Bencher) {
    let n = 5;
    let cycles = 5;
    b.iter(|| random_tensor_network(n, cycles));
}

// TODO: Implement benchmarking for contraction.