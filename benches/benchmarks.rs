use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, SamplingMode};
use rand::{rngs::StdRng, SeedableRng};
use std::time::Duration;
use tensorcontraction::{
    circuits::sycamore::{sycamore_circuit, sycamore_contract},
    contractionpath::paths::{CostType, Greedy, OptimizePath},
    random::tensorgeneration::create_filled_tensor_network,
    random::tensorgeneration::random_sparse_tensor_with_rng,
    tensornetwork::{contraction::contract_tensor_network, tensor::Tensor},
    tensornetwork::{
        contraction::{tn_contract, tn_contract_partition},
        partitioning::partition_tn,
        tensor::Tensor,
        TensorNetwork,
    },
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

pub fn partition_benchmark(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(52);
    let mut mul_group = c.benchmark_group("Partition");
    mul_group.measurement_time(Duration::from_secs(10));
    mul_group.sampling_mode(SamplingMode::Flat);

    for k in [5, 10, 15, 20] {
        let mut r_tn = sycamore_circuit(k, 10, None, None, &mut rng);

        let d_tn: Vec<tetra::Tensor> = r_tn
            .get_tensors()
            .iter()
            .map(|tensor| {
                random_sparse_tensor_with_rng(tensor, r_tn.get_bond_dims(), None, &mut rng)
            })
            .collect();

        let mut partitioning: Vec<i32> = vec![-1; r_tn.get_tensors().len() as usize];
        partition_tn(
            &mut partitioning,
            &mut r_tn,
            k as i32,
            std::string::String::from("test/km1"),
        );
        let mut opt = Greedy::new(&r_tn, CostType::Flops);
        let mut opt_paths = vec![];
        for i in 0..5 {
            opt.optimize_partitioned_path(i);
            opt_paths.push(opt.get_best_partition_replace_path(i as usize));
        }

        mul_group.bench_function(BenchmarkId::from_parameter(k), |b| {
            b.iter(|| {
                let mut local_r_tn = r_tn.clone();
                let mut local_d_tn = d_tn.clone();
                for i in 0..5 {
                    tn_contract_partition(
                        &mut local_r_tn,
                        &mut local_d_tn,
                        i as i32,
                        &opt_paths[i],
                    );
                }
            });
        });
    }
    mul_group.finish();
}

criterion_group!(benches, partition_benchmark, criterion_benchmark);
criterion_main!(benches);
