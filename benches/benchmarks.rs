use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, SamplingMode};
use rand::{rngs::StdRng, SeedableRng};
use std::time::Duration;
use tensorcontraction::circuits::sycamore::sycamore_circuit;
use tensorcontraction::contractionpath::paths::{CostType, Greedy, OptimizePath};
use tensorcontraction::tensornetwork::partitioning::{find_partitioning, partition_tensor_network};
use tensorcontraction::types::ContractionIndex;
use tensorcontraction::{
    path,
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
            b.iter(|| contract_tensor_network(&mut r_tn.clone(), &[path![0, 1]]));
        });
    }
    mul_group.finish();
}

pub fn partition_benchmark(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(52);
    let mut mul_group = c.benchmark_group("Partition");
    mul_group.measurement_time(Duration::from_secs(10));
    mul_group.sampling_mode(SamplingMode::Flat);

    for k in [15, 20] {
        let mut r_tn = sycamore_circuit(k, 5, None, None, &mut rng);
        let partitioning = find_partitioning(&mut r_tn, 5, std::string::String::from("tests/km1"));
        let partitioned_tn = partition_tensor_network(&r_tn, &partitioning);
        // let mut opt = BranchBound::new(&r_tn, None, 20, CostType::Flops);
        let mut opt = Greedy::new(&partitioned_tn, CostType::Flops);

        opt.optimize_path();
        let path = opt.get_best_replace_path();

        mul_group.bench_function(BenchmarkId::from_parameter(k), |b| {
            b.iter(|| {
                let mut local_r_tn = partitioned_tn.clone();
                contract_tensor_network(&mut local_r_tn, &path);
            });
        });
    }
    mul_group.finish();
}

criterion_group!(benches, partition_benchmark, criterion_benchmark,);
criterion_main!(benches);
