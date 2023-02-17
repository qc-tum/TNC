#![feature(test)]
use rand::{rngs::StdRng, SeedableRng};
use tensorcontraction::{
    random::tensorgeneration::{random_tensor_network_with_rng},
    tensornetwork::{tensor::Tensor, TensorNetwork},
};

extern crate test;
use test::Bencher;

fn _setup() -> TensorNetwork {
    TensorNetwork::from_vector(
        vec![Tensor::new(vec![4, 3, 2]), Tensor::new(vec![0, 1, 3, 2])],
        vec![17, 18, 19, 12, 22],
        None
    )
}

#[bench]
fn build_tensor(b: &mut Bencher) {
    let n = 5;
    let cycles = 5;
    let mut rng: StdRng = SeedableRng::seed_from_u64(0);
    b.iter(|| random_tensor_network_with_rng(n, cycles, &mut rng));
}

// TODO: Implement benchmarking for contraction.