use std::iter::zip;

use float_cmp::approx_eq;
use itertools::Itertools;
use mpi::traits::{Communicator, CommunicatorCollectives};
use rand::{rngs::StdRng, SeedableRng};
use tensorcontraction::{
    circuits::{connectivity::ConnectivityLayout, sycamore::sycamore_circuit},
    contractionpath::paths::{greedy::Greedy, CostType, OptimizePath},
    mpi::scatter::{naive_reduce_tensor_network, scatter_tensor_network},
    tensornetwork::{
        contraction::contract_tensor_network,
        partitioning::{find_partitioning, partition_tensor_network},
        tensor::Tensor,
        tensordata::TensorData,
    },
};

fn check_tensordata_eq(tensor1: &TensorData, other: &TensorData) -> bool {
    match (tensor1, other) {
        (TensorData::File(l0), TensorData::File(r0)) => l0 == r0,
        (TensorData::Gate((l0, angles_l)), TensorData::Gate((r0, angles_r))) => {
            if l0.to_lowercase() != r0.to_lowercase() {
                return false;
            }
            for (angle1, angle2) in zip(angles_l.iter(), angles_r.iter()) {
                if !approx_eq!(f64, *angle1, *angle2, ulps = 2) {
                    return false;
                }
            }
            true
        }
        (TensorData::Matrix(l0), TensorData::Matrix(r0)) => {
            let range = l0.shape().iter().map(|e| 0..*e).multi_cartesian_product();

            for index in range {
                if !approx_eq!(f64, l0.get(&index).im, r0.get(&index).im, epsilon = 1e-8) {
                    return false;
                }
                if !approx_eq!(f64, l0.get(&index).re, r0.get(&index).re, epsilon = 1e-8) {
                    return false;
                }
            }
            true
        }
        (TensorData::Uncontracted, TensorData::Uncontracted) => true,
        _ => false,
    }
}

#[test]
fn test_partitioned_contraction() {
    let mut rng = StdRng::seed_from_u64(52);
    let k = 10;

    let r_tn = sycamore_circuit(k, 10, 0.4, 0.4, &mut rng, ConnectivityLayout::Osprey);
    let mut ref_tn = r_tn.clone();
    let mut ref_opt = Greedy::new(&ref_tn, CostType::Flops);
    ref_opt.optimize_path();
    let ref_path = ref_opt.get_best_replace_path();
    contract_tensor_network(&mut ref_tn, &ref_path);

    let partitioning =
        find_partitioning(&r_tn, 3, String::from("tests/km1_kKaHyPar_sea20.ini"), true);
    let mut partitioned_tn = partition_tensor_network(&r_tn, &partitioning);
    let mut opt = Greedy::new(&partitioned_tn, CostType::Flops);
    opt.optimize_path();
    let path = opt.get_best_replace_path();
    contract_tensor_network(&mut partitioned_tn, &path);
    assert!(check_tensordata_eq(
        &ref_tn.get_tensor_data(),
        &partitioned_tn.get_tensor_data()
    ));
}

// Ignored as requires MPI to run
#[ignore]
#[test]
fn test_mpi_partitioned_contraction() {
    let mut rng = StdRng::seed_from_u64(23);

    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let size = world.size();
    let rank = world.rank();

    // let status;
    let mut partitioned_tn = Tensor::default();
    let mut path = Vec::new();
    let mut ref_tn = Tensor::default();
    if rank == 0 {
        let k = 5;
        let r_tn = sycamore_circuit(k, 10, 0.4, 0.4, &mut rng, ConnectivityLayout::Osprey);
        ref_tn = r_tn.clone();
        let partitioning = find_partitioning(&r_tn, size, String::from("tests/km1"), true);
        partitioned_tn = partition_tensor_network(&r_tn, &partitioning);
        let mut opt = Greedy::new(&partitioned_tn, CostType::Flops);

        opt.optimize_path();
        path = opt.get_best_replace_path();
    }
    world.barrier();
    let (mut local_tn, local_path) =
        scatter_tensor_network(partitioned_tn, &path, rank, size, &world);
    contract_tensor_network(&mut local_tn, &local_path);

    naive_reduce_tensor_network(&mut local_tn, &path, rank, size, &world);
    world.barrier();

    if rank == 0 {
        let mut ref_opt = Greedy::new(&ref_tn, CostType::Flops);

        ref_opt.optimize_path();
        let ref_path = ref_opt.get_best_replace_path();
        contract_tensor_network(&mut ref_tn, &ref_path);
        assert!(check_tensordata_eq(
            &local_tn.get_tensor_data(),
            &ref_tn.get_tensor_data()
        ));
    }
}
