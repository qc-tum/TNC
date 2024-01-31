use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, SamplingMode};
use rand::{rngs::StdRng, SeedableRng};
use std::time::Duration;
use std::ffi::CString;
use tensorcontraction::circuits::sycamore::sycamore_circuit;
use tensorcontraction::contractionpath::paths::OptimizePath;
use tensorcontraction::contractionpath::paths::{greedy::Greedy, CostType};
use tensorcontraction::mpi::scatter::{
    broadcast_path, intermediate_gather_tensor_network, naive_gather_tensor_network,
    scatter_tensor_network,
};
// use tensorcontraction::tensornetwork::parallel_contraction::parallel_contract_tensor_network;
use mpi::traits::*;

use tensorcontraction::tensornetwork::partitioning::{find_partitioning, partition_tensor_network};
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
    let mut rng = StdRng::seed_from_u64(23);
    let mut part_group = c.benchmark_group("Partition");
    part_group.measurement_time(Duration::from_secs(10));
    part_group.sampling_mode(SamplingMode::Flat);

    for k in [10, 15, 20, 25] {
        let r_tn = sycamore_circuit(k, 5, None, None, &mut rng, "Osprey");
        let partitioning =
            find_partitioning(&r_tn, 5, CString::new("tests/km1").expect("CString::new failed"), true);
        let partitioned_tn = partition_tensor_network(&r_tn, &partitioning);
        // let mut opt = BranchBound::new(&r_tn, None, 20, CostType::Flops);
        let mut opt = Greedy::new(&partitioned_tn, CostType::Flops);

        opt.optimize_path();

        let path = opt.get_best_replace_path();
        part_group.bench_function(BenchmarkId::from_parameter(k), |b| {
            b.iter(|| {
                contract_tensor_network(&mut partitioned_tn.clone(), &path.clone());
            });
        });
    }
    part_group.finish();
}

// Run with 4 processes
// Current assumption is that we have the number of processes equal to intermediate tensors
pub fn parallel_naive_benchmark(c: &mut Criterion) {
    let mut rng: StdRng = StdRng::seed_from_u64(51);

    let mut par_part_group = c.benchmark_group("MPI Naive");
    par_part_group.measurement_time(Duration::from_secs(30));
    par_part_group.sampling_mode(SamplingMode::Flat);

    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let size = world.size();
    let rank = world.rank();

    // Do we need to know communication beforehand?
    for k in [25, 30] {
        let mut partitioned_tn = Tensor::default();
        let mut path = Vec::new();
        if rank == 0 {
            let r_tn = sycamore_circuit(k, 20, None, None, &mut rng, "Osprey");
            let partitioning =
                find_partitioning(&r_tn, size, CString::new("tests/km1").expect("CString::new failed"), true);
            partitioned_tn = partition_tensor_network(&r_tn, &partitioning);
            let mut opt = Greedy::new(&partitioned_tn, CostType::Flops);

            opt.optimize_path();
            path = opt.get_best_replace_path();
        }
        world.barrier();

        par_part_group.bench_function(BenchmarkId::from_parameter(k), |b| {
            b.iter(|| {
                let (mut local_tn, local_path) =
                    scatter_tensor_network(partitioned_tn.clone(), &path, rank, size, &world);
                contract_tensor_network(&mut local_tn, &local_path);

                naive_gather_tensor_network(local_tn.clone(), &path, rank, size, &world);
            });
        });
    }
    par_part_group.finish();
}

pub fn parallel_partition_benchmark(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(52);

    let mut par_part_group = c.benchmark_group("MPI Partition");
    par_part_group.measurement_time(Duration::from_secs(10));
    par_part_group.sampling_mode(SamplingMode::Flat);

    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let size = world.size();
    let rank = world.rank();

    // Do we need to know communication beforehand?
    for k in [30, 35] {
        let mut partitioned_tn = Tensor::default();
        let mut path = Vec::new();
        if rank == 0 {
            let r_tn = sycamore_circuit(k, 2, None, None, &mut rng, "Osprey");
            let partitioning =
                find_partitioning(&r_tn, size, CString::new("tests/km1").expect("CString::new failed"), true);
            partitioned_tn = partition_tensor_network(&r_tn, &partitioning);
            let mut opt = Greedy::new(&partitioned_tn, CostType::Flops);

            opt.optimize_path();
            path = opt.get_best_replace_path();
        }
        world.barrier();

        par_part_group.bench_function(BenchmarkId::from_parameter(k), |b| {
            b.iter(|| {
                let (mut local_tn, local_path) =
                    scatter_tensor_network(partitioned_tn.clone(), &path, rank, size, &world);
                contract_tensor_network(&mut local_tn, &local_path);

                let path = if rank == 0 {
                    broadcast_path(&path[(size as usize)..path.len()], &world)
                } else {
                    broadcast_path(&[], &world)
                };
                world.barrier();
                intermediate_gather_tensor_network(&mut local_tn, &path, rank, size, &world);
            });
        });
    }
    par_part_group.finish();
}

criterion_group!(
    benches,
    // partition_benchmark,
    // criterion_benchmark,
    // parallel_naive_benchmark,
    parallel_partition_benchmark
);
criterion_main!(benches);
