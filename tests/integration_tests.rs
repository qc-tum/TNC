use mpi::traits::Communicator;
use mpi_test::mpi_test;
use rand::{rngs::StdRng, SeedableRng};
use tensorcontraction::{
    contractionpath::paths::{
        cotengrust::{Cotengrust, OptMethod},
        OptimizePath,
    },
    mpi::communication::{
        broadcast_path, extract_communication_path, intermediate_reduce_tensor_network,
        scatter_tensor_network,
    },
    networks::{connectivity::ConnectivityLayout, random_circuit::random_circuit},
    tensornetwork::{
        contraction::contract_tensor_network,
        partitioning::{
            find_partitioning, partition_config::PartitioningStrategy, partition_tensor_network,
        },
    },
};

#[test]
fn test_partitioned_contraction_random() {
    let mut rng = StdRng::seed_from_u64(52);
    let k = 15;

    let r_tn = random_circuit(k, 10, 0.5, 0.5, &mut rng, ConnectivityLayout::Eagle);
    let ref_tn = r_tn.clone();
    let mut ref_opt = Cotengrust::new(&ref_tn, OptMethod::RandomGreedy(10));
    ref_opt.optimize_path();
    let ref_path = ref_opt.get_best_replace_path();
    let ref_result = contract_tensor_network(ref_tn, &ref_path);

    let partitioning = find_partitioning(&r_tn, 12, PartitioningStrategy::MinCut, true);
    let partitioned_tn = partition_tensor_network(r_tn, &partitioning);
    let mut opt = Cotengrust::new(&partitioned_tn, OptMethod::RandomGreedy(10));
    opt.optimize_path();
    let path = opt.get_best_replace_path();
    let result = contract_tensor_network(partitioned_tn, &path);
    assert!(&ref_result.approx_eq(&result, 1e-12));
}

#[test]
fn test_partitioned_contraction() {
    let mut rng = StdRng::seed_from_u64(52);
    let k = 15;

    let r_tn = random_circuit(k, 10, 0.5, 0.5, &mut rng, ConnectivityLayout::Osprey);
    let ref_tn = r_tn.clone();
    let mut ref_opt = Cotengrust::new(&ref_tn, OptMethod::Greedy);
    ref_opt.optimize_path();
    let ref_path = ref_opt.get_best_replace_path();
    let ref_result = contract_tensor_network(ref_tn, &ref_path);

    let partitioning = find_partitioning(&r_tn, 12, PartitioningStrategy::MinCut, true);
    let partitioned_tn = partition_tensor_network(r_tn, &partitioning);
    let mut opt = Cotengrust::new(&partitioned_tn, OptMethod::Greedy);
    opt.optimize_path();
    let path = opt.get_best_replace_path();
    let result = contract_tensor_network(partitioned_tn, &path);
    assert!(&ref_result.approx_eq(&result, 1e-12));
}

#[test]
fn test_partitioned_contraction_mixed() {
    let mut rng = StdRng::seed_from_u64(52);
    let k = 15;

    let r_tn = random_circuit(k, 10, 0.5, 0.5, &mut rng, ConnectivityLayout::Condor);
    let ref_tn = r_tn.clone();
    let mut ref_opt = Cotengrust::new(&ref_tn, OptMethod::Greedy);
    ref_opt.optimize_path();
    let ref_path = ref_opt.get_best_replace_path();
    let ref_result = contract_tensor_network(ref_tn, &ref_path);

    let partitioning = find_partitioning(&r_tn, 12, PartitioningStrategy::MinCut, true);
    let partitioned_tn = partition_tensor_network(r_tn, &partitioning);
    let mut opt = Cotengrust::new(&partitioned_tn, OptMethod::RandomGreedy(15));
    opt.optimize_path();
    let path = opt.get_best_replace_path();
    let result = contract_tensor_network(partitioned_tn, &path);
    assert!(&ref_result.approx_eq(&result, 1e-12));
}

#[mpi_test(4)]
fn test_partitioned_contraction_need_mpi() {
    let mut rng = StdRng::seed_from_u64(23);

    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let size = world.size();
    let rank = world.rank();
    let root = world.process_at_rank(0);

    let (ref_tn, partitioned_tn, path) = if rank == 0 {
        let k = 10;
        let r_tn = random_circuit(k, 10, 0.4, 0.4, &mut rng, ConnectivityLayout::Osprey);
        let ref_tn = r_tn.clone();
        let partitioning = find_partitioning(&r_tn, size, PartitioningStrategy::MinCut, true);
        let partitioned_tn = partition_tensor_network(r_tn, &partitioning);
        let mut opt = Cotengrust::new(&partitioned_tn, OptMethod::Greedy);
        opt.optimize_path();
        let path = opt.get_best_replace_path();
        (ref_tn, partitioned_tn, path)
    } else {
        Default::default()
    };

    let (mut local_tn, local_path, comm) =
        scatter_tensor_network(&partitioned_tn, &path, rank, size, &world);
    local_tn = contract_tensor_network(local_tn, &local_path);

    let mut communication_path = if rank == 0 {
        extract_communication_path(&path)
    } else {
        Default::default()
    };
    broadcast_path(&mut communication_path, &root);

    intermediate_reduce_tensor_network(&mut local_tn, &communication_path, rank, &world, &comm);

    if rank == 0 {
        let mut ref_opt = Cotengrust::new(&ref_tn, OptMethod::RandomGreedy(10));
        ref_opt.optimize_path();
        let ref_path = ref_opt.get_best_replace_path();

        let ref_tn = contract_tensor_network(ref_tn, &ref_path);

        assert!(local_tn.tensor_data().approx_eq(ref_tn.tensor_data(), 1e-8));
    }
}
