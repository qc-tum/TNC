use std::f64::consts::FRAC_1_SQRT_2;

use approx::assert_abs_diff_eq;
use mpi::traits::Communicator;
use mpi_test::mpi_test;
use ndarray::array;
use num_complex::Complex64;
use rand::{rngs::StdRng, SeedableRng};
use tnc::{
    builders::{connectivity::ConnectivityLayout, random_circuit::random_circuit},
    contractionpath::paths::{
        cotengrust::{Cotengrust, OptMethod},
        ContractionPathResult, Pathfinder,
    },
    io::qasm::import_qasm,
    mpi::communication::{
        broadcast_path, intermediate_reduce_tensor_network, scatter_tensor_network,
    },
    tensornetwork::{
        contraction::contract_tensor_network,
        partitioning::{find_partitioning, partition_tensor_network, PartitioningStrategy},
    },
};

#[test]
fn test_partitioned_contraction_random() {
    let mut rng = StdRng::seed_from_u64(52);
    let k = 15;

    let r_tn = random_circuit(k, 10, 0.5, 0.5, &mut rng, ConnectivityLayout::Eagle);
    let ref_tn = r_tn.clone();
    let mut ref_opt = Cotengrust::new(OptMethod::RandomGreedy(10));
    let result = ref_opt.find_path(&ref_tn);
    let ref_path = result.replace_path();
    let ref_result = contract_tensor_network(ref_tn, &ref_path);

    let partitioning = find_partitioning(&r_tn, 12, PartitioningStrategy::MinCut, true);
    let partitioned_tn = partition_tensor_network(r_tn, &partitioning);
    let mut opt = Cotengrust::new(OptMethod::RandomGreedy(10));
    let result = opt.find_path(&partitioned_tn);
    let path = result.replace_path();
    let result = contract_tensor_network(partitioned_tn, &path);
    assert_abs_diff_eq!(&result, &ref_result);
}

#[test]
fn test_partitioned_contraction() {
    let mut rng = StdRng::seed_from_u64(52);
    let k = 15;

    let r_tn = random_circuit(k, 10, 0.5, 0.5, &mut rng, ConnectivityLayout::Osprey);
    let ref_tn = r_tn.clone();
    let mut ref_opt = Cotengrust::new(OptMethod::Greedy);
    let result = ref_opt.find_path(&ref_tn);
    let ref_path = result.replace_path();
    let ref_result = contract_tensor_network(ref_tn, &ref_path);

    let partitioning = find_partitioning(&r_tn, 12, PartitioningStrategy::MinCut, true);
    let partitioned_tn = partition_tensor_network(r_tn, &partitioning);
    let mut opt = Cotengrust::new(OptMethod::Greedy);
    let result = opt.find_path(&partitioned_tn);
    let path = result.replace_path();
    let result = contract_tensor_network(partitioned_tn, &path);
    assert_abs_diff_eq!(&result, &ref_result);
}

#[test]
fn test_partitioned_contraction_mixed() {
    let mut rng = StdRng::seed_from_u64(52);
    let k = 15;

    let r_tn = random_circuit(k, 10, 0.5, 0.5, &mut rng, ConnectivityLayout::Condor);
    let ref_tn = r_tn.clone();
    let mut ref_opt = Cotengrust::new(OptMethod::Greedy);
    let result = ref_opt.find_path(&ref_tn);
    let ref_path = result.replace_path();
    let ref_result = contract_tensor_network(ref_tn, &ref_path);

    let partitioning = find_partitioning(&r_tn, 12, PartitioningStrategy::MinCut, true);
    let partitioned_tn = partition_tensor_network(r_tn, &partitioning);
    let mut opt = Cotengrust::new(OptMethod::RandomGreedy(15));
    let result = opt.find_path(&partitioned_tn);
    let path = result.replace_path();
    let result = contract_tensor_network(partitioned_tn, &path);
    assert_abs_diff_eq!(&result, &ref_result);
}

#[mpi_test(2)]
fn test_broadcast_contraction_path() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank();
    let root_process = world.process_at_rank(0);
    let max = usize::MAX;

    let ref_contraction_indices = vec![
        (0, 4),
        (1, 5),
        (2, 16),
        (7, max),
        (max, 5),
        (64, 2),
        (4, 55),
        (81, 21),
        (2, 72),
        (23, 3),
        (40, 5),
        (2, 26),
    ];

    let mut contraction_indices = if rank == 0 {
        ref_contraction_indices.clone()
    } else {
        Default::default()
    };
    broadcast_path(&mut contraction_indices, &root_process);

    assert_eq!(contraction_indices, ref_contraction_indices);
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
        let mut opt = Cotengrust::new(OptMethod::Greedy);
        let result = opt.find_path(&partitioned_tn);
        let path = result.replace_path();
        (ref_tn, partitioned_tn, path)
    } else {
        Default::default()
    };

    let (mut local_tn, local_path, comm) =
        scatter_tensor_network(&partitioned_tn, &path, rank, size, &world);
    local_tn = contract_tensor_network(local_tn, &local_path);

    let mut communication_path = if rank == 0 {
        path.toplevel
    } else {
        Default::default()
    };
    broadcast_path(&mut communication_path, &root);

    intermediate_reduce_tensor_network(&mut local_tn, &communication_path, rank, &world, &comm);

    if rank == 0 {
        let mut ref_opt = Cotengrust::new(OptMethod::RandomGreedy(10));
        let result = ref_opt.find_path(&ref_tn);
        let ref_path = result.replace_path();

        let ref_tn = contract_tensor_network(ref_tn, &ref_path);

        assert_abs_diff_eq!(&local_tn.tensor_data(), &ref_tn.tensor_data());
    }
}

#[test]
fn dj_4qubits_statevector() {
    let code = "OPENQASM 2.0;
    include \"qelib1.inc\";
    qreg q[4];
    creg c[3];
    u2(0,0) q[0];
    u2(0,0) q[1];
    h q[2];
    u2(-pi,-pi) q[3];
    cx q[0],q[3];
    u2(-pi,-pi) q[0];
    cx q[1],q[3];
    u2(-pi,-pi) q[1];
    cx q[2],q[3];
    h q[2];";

    let circuit = import_qasm(code);
    let (tensor_network, permutator) = circuit.into_statevector_network();

    let mut opt = Cotengrust::new(OptMethod::Greedy);
    let result = opt.find_path(&tensor_network);
    let path = result.replace_path();

    let final_tensor = contract_tensor_network(tensor_network, &path);
    let statevector = permutator.apply(final_tensor);
    let data = statevector.into_tensor_data().into_data();

    // Result is 1/sqrt(2) * (|1110> - |1111>)
    let expected = array![
        Complex64::ZERO,
        Complex64::ZERO,
        Complex64::ZERO,
        Complex64::ZERO,
        Complex64::ZERO,
        Complex64::ZERO,
        Complex64::ZERO,
        Complex64::ZERO,
        Complex64::ZERO,
        Complex64::ZERO,
        Complex64::ZERO,
        Complex64::ZERO,
        Complex64::ZERO,
        Complex64::ZERO,
        Complex64::new(FRAC_1_SQRT_2, 0.0),
        Complex64::new(-FRAC_1_SQRT_2, 0.0),
    ];
    assert_abs_diff_eq!(data.flatten(), expected, epsilon = 1e-15);
}

#[test]
fn qft_2qubits_expectation() {
    let code = "OPENQASM 2.0;
    include \"qelib1.inc\";
    qreg q[2];
    creg meas[2];
    h q[1];
    cx q[1],q[0];
    h q[1];
    cp(pi/2) q[1],q[0];
    h q[0];
    swap q[0],q[1];";

    let circuit = import_qasm(code);
    let tensor_network = circuit.into_expectation_value_network();

    let mut opt = Cotengrust::new(OptMethod::RandomGreedy(3));
    let result = opt.find_path(&tensor_network);
    let path = result.replace_path();

    let final_tensor = contract_tensor_network(tensor_network, &path);
    let data = final_tensor.into_tensor_data().into_data();

    let expected = array![Complex64::new(0.5, 0.0)];
    assert_abs_diff_eq!(data.flatten(), expected, epsilon = 1e-15);
}
