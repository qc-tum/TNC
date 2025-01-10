use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use mpi::environment::Universe;
use mpi::traits::{Communicator, CommunicatorCollectives};
use rand::{rngs::StdRng, SeedableRng};
use rustc_hash::FxHashMap;
use static_init::dynamic;
use tensorcontraction::contractionpath::paths::OptimizePath;
use tensorcontraction::contractionpath::paths::{greedy::Greedy, CostType};
use tensorcontraction::mpi::communication::{
    broadcast_path, extract_communication_path, intermediate_reduce_tensor_network,
    scatter_tensor_network,
};
use tensorcontraction::networks::connectivity::ConnectivityLayout;
use tensorcontraction::networks::sycamore::random_circuit;

use tensorcontraction::tensornetwork::partitioning::partition_config::PartitioningStrategy;
use tensorcontraction::tensornetwork::partitioning::{find_partitioning, partition_tensor_network};
use tensorcontraction::{
    path,
    random::tensorgeneration::create_filled_tensor_network,
    tensornetwork::{contraction::contract_tensor_network, tensor::Tensor},
};

/// The shared MPI universe.
#[dynamic(lazy, drop)]
static mut MPI_UNIVERSE: Universe = mpi::initialize().unwrap();

/// Benchmark for the contraction of two tensors.
pub fn multiplication_benchmark(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(52);
    let mut mul_group = c.benchmark_group("Multiplication");

    for k in [16, 32, 64] {
        let n = 64;
        let t1 = Tensor::new(vec![0, 1]);
        let t2 = Tensor::new(vec![2, 1, 3, 4]);
        let bond_dims = FxHashMap::from_iter([(0, n), (1, k), (2, n), (3, n), (4, n)]);
        let r_tn = create_filled_tensor_network(vec![t1, t2], &bond_dims, &mut rng);

        mul_group.bench_function(BenchmarkId::from_parameter(k), |b| {
            b.iter_batched_ref(
                || r_tn.clone(),
                |tn| contract_tensor_network(tn, black_box(&[path![0, 1]])),
                BatchSize::SmallInput,
            );
        });
    }
    mul_group.finish();
}

/// Benchmark for the contraction of a partitioned tensor network on a single node.
pub fn partitioned_contraction_benchmark(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(23);
    let mut part_group = c.benchmark_group("Partition");

    for k in [10, 15, 20, 25] {
        let r_tn = random_circuit(k, 5, 0.4, 0.4, &mut rng, ConnectivityLayout::Osprey);
        let partitioning =
            find_partitioning(&r_tn, 5, PartitioningStrategy::CommunityFinding, true);
        let partitioned_tn = partition_tensor_network(&r_tn, &partitioning);

        let mut opt = Greedy::new(&partitioned_tn, CostType::Flops);
        opt.optimize_path();
        let path = opt.get_best_replace_path();

        part_group.bench_function(BenchmarkId::from_parameter(k), |b| {
            b.iter_batched_ref(
                || partitioned_tn.clone(),
                |tn| contract_tensor_network(tn, &path),
                BatchSize::SmallInput,
            );
        });
    }
    part_group.finish();
}

/// Benchmark for the parallel contraction of a partitioned tensor network on
/// multiple nodes (using MPI). This benchmark uses
/// [`intermediate_reduce_tensor_network`].
pub fn parallel_partition_benchmark(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(52);

    let mut par_part_group = c.benchmark_group("MPI Partition");

    let universe = MPI_UNIVERSE.read();
    let world = universe.world();
    let size = world.size();
    let rank = world.rank();
    let root = world.process_at_rank(0);

    // TODO: Do we need to know communication beforehand?
    for k in [30, 35, 45] {
        let (partitioned_tn, path) = if rank == 0 {
            let r_tn = random_circuit(k, 20, 0.4, 0.4, &mut rng, ConnectivityLayout::Osprey);
            let partitioning =
                find_partitioning(&r_tn, size, PartitioningStrategy::CommunityFinding, true);
            let partitioned_tn = partition_tensor_network(&r_tn, &partitioning);

            let mut opt = Greedy::new(&partitioned_tn, CostType::Flops);
            opt.optimize_path();
            let path = opt.get_best_replace_path();
            (partitioned_tn, path)
        } else {
            Default::default()
        };
        world.barrier();

        par_part_group.bench_function(BenchmarkId::from_parameter(k), |b| {
            b.iter(|| {
                let (mut local_tn, local_path, slicing_task, comm) =
                    scatter_tensor_network(&partitioned_tn, &path, rank, size, &world);
                assert!(slicing_task.is_none());
                contract_tensor_network(&mut local_tn, &local_path);

                let mut communication_path = if rank == 0 {
                    extract_communication_path(&path)
                } else {
                    Default::default()
                };
                broadcast_path(&mut communication_path, &root);

                intermediate_reduce_tensor_network(
                    &mut local_tn,
                    &communication_path,
                    rank,
                    &world,
                    &comm,
                );
                local_tn
            });
        });
    }
    par_part_group.finish();
}

criterion_group!(
    benches,
    multiplication_benchmark,
    partitioned_contraction_benchmark,
    parallel_partition_benchmark
);
criterion_main!(benches);
