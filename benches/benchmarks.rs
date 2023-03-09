use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, SamplingMode};
use rand::Rng;
use rand::{rngs::StdRng, thread_rng, SeedableRng};
use std::time::Duration;
use tensorcontraction::{
    contractionpath::paths::{BranchBound, BranchBoundType, OptimizePath},
    random::tensorgeneration::{
        random_sparse_tensor_with_rng, random_sparse_tetra_tensor_with_rng,
        random_tensor_network_with_rng,
    },
    tensornetwork::{
        contraction::{tn_contract, tn_tetra_contract},
        tensor::Tensor,
        TensorNetwork,
    },
};

extern crate test;
use test::Bencher;

fn taco_contraction<R>(r_tn: TensorNetwork, opt_path: &Vec<(usize, usize)>, rng: &mut R)
where
    R: Rng + ?Sized,
{
    let d_tn = r_tn
        .get_tensors()
        .iter()
        .map(|tensor| {
            random_sparse_tensor_with_rng(tensor.clone(), r_tn.get_bond_dims(), None, rng)
        })
        .collect();
    tn_contract(r_tn, d_tn, opt_path);
}

fn tetra_contraction<R>(r_tn: TensorNetwork, opt_path: &Vec<(usize, usize)>, rng: &mut R)
where
    R: Rng + ?Sized,
{
    let d_tn = r_tn
        .get_tensors()
        .iter()
        .map(|tensor| {
            random_sparse_tetra_tensor_with_rng(tensor.clone(), r_tn.get_bond_dims(), None, rng)
        })
        .collect();
    tn_tetra_contract(r_tn, d_tn, opt_path);
}

fn sized_contraction(r_tn: TensorNetwork, d_tn: Vec<tetra::Tensor>) {
    tn_tetra_contract(r_tn, d_tn, &vec![(0, 1)]);
}

fn sized_transpose(mut d_tn: tetra::Tensor) {
    d_tn.transpose(&[1, 0, 3, 2]);
    d_tn.materialize_transpose();
}

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(52);
    let mut mul_group = c.benchmark_group("Multiplication");
    mul_group.measurement_time(Duration::from_secs(10));
    mul_group.sampling_mode(SamplingMode::Flat);

    for k in [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384] {
        let n = 64;
        let t1 = Tensor::new(vec![0, 1]);
        let t2 = Tensor::new(vec![2, 1, 3, 4]);
        let r_tn = TensorNetwork::from_vector(vec![t1, t2], vec![n, k, n, n, n], None);

        let d_tn: Vec<tetra::Tensor> = r_tn
            .get_tensors()
            .iter()
            .map(|tensor| {
                random_sparse_tetra_tensor_with_rng(
                    tensor.clone(),
                    r_tn.get_bond_dims(),
                    None,
                    &mut rng,
                )
            })
            .collect();
        mul_group.bench_function(BenchmarkId::from_parameter(k), |b| {
            b.iter(|| sized_contraction(r_tn.clone(), d_tn.clone()));
        });
    }
    mul_group.finish();
    // trans_group.finish();
    // let k = 10000;

    // let r_tn = TensorNetwork::from_vector(vec![t1, t2], vec![k, k, k], None);
    // let d_tn :Vec<tetra::Tensor> = r_tn
    //     .get_tensors()
    //     .iter()
    //     .map(|tensor| {
    //         random_sparse_tetra_tensor_with_rng(tensor.clone(), r_tn.get_bond_dims(), None, &mut rng)
    //     })
    //     .collect();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);

// TODO: Implement benchmarking for contraction.
