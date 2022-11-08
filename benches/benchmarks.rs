#![feature(test)]

use tensorcontraction::tensornetwork::{TensorNetwork, tensor::Tensor};

extern crate test;
use test::Bencher;

fn setup() -> TensorNetwork {
    TensorNetwork::new(
        vec![Tensor::new(vec![4, 3, 2]), Tensor::new(vec![0, 1, 3, 2])],
        vec![17, 18, 19, 12, 22],
    )
}

#[bench]
fn build_tensor(b: &mut Bencher) {
    b.iter(|| {
        setup();
    })
}
#[bench]
fn contract_tensor(b: &mut Bencher) {
    b.iter(|| {
        let mut t = setup();
        t.contraction(0, 1);
    })
}