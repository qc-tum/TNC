use mpi::traits::{Communicator, CommunicatorCollectives};
use mpi_test::mpi_test;
use rand::{rngs::StdRng, SeedableRng};
use tensorcontraction::{
    contractionpath::{
        paths::{greedy::Greedy, CostType, OptimizePath},
        random_paths::RandomOptimizePath,
    },
    mpi::communication::{naive_reduce_tensor_network, scatter_tensor_network},
    networks::{connectivity::ConnectivityLayout, sycamore::random_circuit},
    tensornetwork::{
        contraction::contract_tensor_network,
        partitioning::{find_partitioning, partition_tensor_network},
    },
};

#[test]
fn test_partitioned_contraction_random() {
    let mut rng = StdRng::seed_from_u64(52);
    let k = 15;

    let r_tn = random_circuit(k, 10, 0.5, 0.5, &mut rng, ConnectivityLayout::Eagle);
    let mut ref_tn = r_tn.clone();
    let mut ref_opt = Greedy::new(&ref_tn, CostType::Flops);
    ref_opt.random_optimize_path(10, &mut StdRng::seed_from_u64(42));
    let ref_path = ref_opt.get_best_replace_path();
    contract_tensor_network(&mut ref_tn, &ref_path);

    let partitioning = find_partitioning(
        &r_tn,
        12,
        String::from("tests/km1_kKaHyPar_sea20.ini"),
        true,
    );
    let mut partitioned_tn = partition_tensor_network(&r_tn, &partitioning);
    let mut opt = Greedy::new(&partitioned_tn, CostType::Flops);
    opt.random_optimize_path(10, &mut StdRng::seed_from_u64(42));
    let path = opt.get_best_replace_path();
    contract_tensor_network(&mut partitioned_tn, &path);
    assert!(&ref_tn.approx_eq(&partitioned_tn, 1e-12));
}

#[test]
fn test_partitioned_contraction() {
    let mut rng = StdRng::seed_from_u64(52);
    let k = 15;

    let r_tn = random_circuit(k, 10, 0.5, 0.5, &mut rng, ConnectivityLayout::Osprey);
    let mut ref_tn = r_tn.clone();
    let mut ref_opt = Greedy::new(&ref_tn, CostType::Flops);
    ref_opt.optimize_path();
    let ref_path = ref_opt.get_best_replace_path();
    contract_tensor_network(&mut ref_tn, &ref_path);

    let partitioning = find_partitioning(
        &r_tn,
        12,
        String::from("tests/km1_kKaHyPar_sea20.ini"),
        true,
    );
    let mut partitioned_tn = partition_tensor_network(&r_tn, &partitioning);
    let mut opt = Greedy::new(&partitioned_tn, CostType::Flops);
    opt.optimize_path();
    let path = opt.get_best_replace_path();
    contract_tensor_network(&mut partitioned_tn, &path);
    assert!(&ref_tn.approx_eq(&partitioned_tn, 1e-12));
}

#[test]
fn test_partitioned_contraction_mixed() {
    let mut rng = StdRng::seed_from_u64(52);
    let k = 15;

    let r_tn = random_circuit(k, 10, 0.5, 0.5, &mut rng, ConnectivityLayout::Condor);
    let mut ref_tn = r_tn.clone();
    let mut ref_opt = Greedy::new(&ref_tn, CostType::Flops);
    ref_opt.optimize_path();
    let ref_path = ref_opt.get_best_replace_path();
    contract_tensor_network(&mut ref_tn, &ref_path);

    let partitioning = find_partitioning(
        &r_tn,
        12,
        String::from("tests/km1_kKaHyPar_sea20.ini"),
        true,
    );
    let mut partitioned_tn = partition_tensor_network(&r_tn, &partitioning);
    let mut opt = Greedy::new(&partitioned_tn, CostType::Flops);
    opt.random_optimize_path(15, &mut StdRng::seed_from_u64(42));
    let path = opt.get_best_replace_path();
    contract_tensor_network(&mut partitioned_tn, &path);
    assert!(&ref_tn.approx_eq(&partitioned_tn, 1e-12));
}

#[mpi_test(4)]
fn test_partitioned_contraction_need_mpi() {
    let mut rng = StdRng::seed_from_u64(23);

    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let size = world.size();
    let rank = world.rank();

    let (mut ref_tn, partitioned_tn, path) = if rank == 0 {
        let k = 10;
        let r_tn = random_circuit(k, 10, 0.4, 0.4, &mut rng, ConnectivityLayout::Osprey);
        let ref_tn = r_tn.clone();
        let partitioning = find_partitioning(
            &r_tn,
            size,
            String::from("tests/km1_kKaHyPar_sea20.ini"),
            true,
        );
        let partitioned_tn = partition_tensor_network(&r_tn, &partitioning);
        let mut opt = Greedy::new(&partitioned_tn, CostType::Flops);
        opt.optimize_path();
        let path = opt.get_best_replace_path();
        (ref_tn, partitioned_tn, path)
    } else {
        Default::default()
    };
    world.barrier();
    let (mut local_tn, local_path) =
        scatter_tensor_network(&partitioned_tn, &path, rank, size, &world);
    contract_tensor_network(&mut local_tn, &local_path);

    naive_reduce_tensor_network(&mut local_tn, &path, rank, size, &world);
    world.barrier();

    if rank == 0 {
        let mut ref_opt = Greedy::new(&ref_tn, CostType::Flops);

        ref_opt.random_optimize_path(10, &mut StdRng::seed_from_u64(42));
        let ref_path = ref_opt.get_best_replace_path();
        contract_tensor_network(&mut ref_tn, &ref_path);
        assert!(local_tn.approx_eq(&ref_tn, 1e-8));
    }
}
