use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, SamplingMode};
use rand::{rngs::StdRng, SeedableRng};
use std::time::Duration;
use tensorcontraction::{
    random::tensorgeneration::create_filled_tensor_network,
    tensornetwork::{contraction::contract_tensor_network, tensor::Tensor},
};

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(52);
    let mut mul_group = c.benchmark_group("Multiplication");
    mul_group.measurement_time(Duration::from_secs(10));
    mul_group.sampling_mode(SamplingMode::Flat);

    for k in [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384] {
        let n = 64;
        let t1 = Tensor::new(vec![0, 1]);
        let t2 = Tensor::new(vec![2, 1, 3, 4]);
        let bond_dims = [(0, n), (1, k), (2, n), (3, n), (4, n)];
        let r_tn = create_filled_tensor_network(vec![t1, t2], &bond_dims.into(), None, &mut rng);

        mul_group.bench_function(BenchmarkId::from_parameter(k), |b| {
            b.iter(|| contract_tensor_network(&mut r_tn.clone(), &vec![(0, 1)]));
        });
    }
    mul_group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
