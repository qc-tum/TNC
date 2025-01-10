use float_cmp::assert_approx_eq;
use mpi::traits::Communicator;
use mpi_test::mpi_test;
use num_complex::c64;
use rand::{rngs::StdRng, SeedableRng};
use rustc_hash::FxHashMap;
use tensorcontraction::{
    contractionpath::{
        paths::{greedy::Greedy, CostType, OptimizePath},
        random_paths::RandomOptimizePath,
    },
    mpi::communication::{
        broadcast_path, extract_communication_path, intermediate_reduce_tensor_network,
        scatter_tensor_network,
    },
    networks::{connectivity::ConnectivityLayout, sycamore::random_circuit},
    tensornetwork::{
        contraction::contract_tensor_network,
        create_tensor_network,
        partitioning::{
            find_partitioning, partition_config::PartitioningStrategy, partition_tensor_network,
        },
        tensor::Tensor,
        tensordata::TensorData,
    },
    types::{ContractionIndex, SlicingPlan},
};
use tetra::Layout;

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

    let partitioning = find_partitioning(&r_tn, 12, PartitioningStrategy::MinCut, true);
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

    let partitioning = find_partitioning(&r_tn, 12, PartitioningStrategy::MinCut, true);
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

    let partitioning = find_partitioning(&r_tn, 12, PartitioningStrategy::MinCut, true);
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
    let root = world.process_at_rank(0);

    let (mut ref_tn, partitioned_tn, path) = if rank == 0 {
        let k = 10;
        let r_tn = random_circuit(k, 10, 0.4, 0.4, &mut rng, ConnectivityLayout::Osprey);
        let ref_tn = r_tn.clone();
        let partitioning = find_partitioning(&r_tn, size, PartitioningStrategy::MinCut, true);
        let partitioned_tn = partition_tensor_network(&r_tn, &partitioning);
        let mut opt = Greedy::new(&partitioned_tn, CostType::Flops);
        opt.optimize_path();
        let path = opt.get_best_replace_path();
        (ref_tn, partitioned_tn, path)
    } else {
        Default::default()
    };

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

    intermediate_reduce_tensor_network(&mut local_tn, &communication_path, rank, &world, &comm);

    if rank == 0 {
        let mut ref_opt = Greedy::new(&ref_tn, CostType::Flops);
        ref_opt.random_optimize_path(10, &mut StdRng::seed_from_u64(42));
        let ref_path = ref_opt.get_best_replace_path();

        contract_tensor_network(&mut ref_tn, &ref_path);

        assert!(local_tn.tensor_data().approx_eq(ref_tn.tensor_data(), 1e-8));
    }
}

#[mpi_test(5)]
fn test_sliced_small_contraction_mpi() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let size = world.size();
    let rank = world.rank();
    let root = world.process_at_rank(0);

    // Create test scenario (tensor network and path)
    let (tensor, path) = if rank == 0 {
        let mut t0 = Tensor::new(vec![0, 1]);
        t0.set_tensor_data(TensorData::new_from_data(
            &[2, 4],
            (1..=8).map(|x| c64(x, 0)).collect(),
            Some(Layout::RowMajor),
        ));
        let mut t1 = Tensor::new(vec![1, 2]);
        t1.set_tensor_data(TensorData::new_from_data(
            &[4, 3],
            (1..=12).map(|x| c64(x, 0)).collect(),
            Some(Layout::RowMajor),
        ));
        let bond_dims = FxHashMap::from_iter([(0, 2), (1, 4), (2, 3)]);
        let tc = create_tensor_network(vec![t0, t1], &bond_dims);
        let mut t3 = Tensor::new(vec![0, 2]);
        t3.set_tensor_data(TensorData::new_from_data(
            &[2, 3],
            (1..=6).map(|x| c64(x, 0)).collect(),
            Some(Layout::RowMajor),
        ));
        let mut tensor = Tensor::default();
        tensor.push_tensors(vec![tc, t3], Some(&bond_dims));
        let path = vec![
            ContractionIndex::Path(
                0,
                Some(SlicingPlan { slices: vec![1] }),
                vec![ContractionIndex::Pair(0, 1)],
            ),
            ContractionIndex::Path(1, None, vec![]),
            ContractionIndex::Pair(0, 1),
        ];
        (tensor, path)
    } else {
        Default::default()
    };

    // Distribute tensor network
    let (mut local_tn, local_path, slicing_task, comm) =
        scatter_tensor_network(&tensor, &path, rank, size, &world);

    // Slice the local tensor network if necessary
    if let Some(slicing_task) = slicing_task {
        local_tn.apply_slicing(&slicing_task);
    }

    // Contract the local tensor network
    contract_tensor_network(&mut local_tn, &local_path);

    // Broadcast the communication path
    let mut communication_path = if rank == 0 {
        extract_communication_path(&path)
    } else {
        Default::default()
    };
    broadcast_path(&mut communication_path, &root);

    // Reduce the tensor network
    intermediate_reduce_tensor_network(&mut local_tn, &communication_path, rank, &world, &comm);

    if rank == 0 {
        let data = local_tn.tensor_data().clone().into_data();
        let data = data.elements();
        let [element] = &data[..] else {
            panic!("Expected one element")
        };
        assert_approx_eq!(f64, element.re, 3312.0);
        assert_approx_eq!(f64, element.im, 0.0);
    }
}
