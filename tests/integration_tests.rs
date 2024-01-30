use mpi::traits::{Communicator, CommunicatorCollectives};
use rand::{rngs::StdRng, SeedableRng};
use tensorcontraction::{
    circuits::sycamore::sycamore_circuit,
    contractionpath::paths::{greedy::Greedy, CostType, OptimizePath},
    mpi::scatter::{naive_gather_tensor_network, scatter_tensor_network},
    tensornetwork::{
        contraction::contract_tensor_network,
        partitioning::{find_partitioning, partition_tensor_network},
        tensor::Tensor,
    },
};

#[test]
fn test_partitioned_contraction() {
    let mut rng = StdRng::seed_from_u64(52);
    let k = 5;

    let r_tn = sycamore_circuit(k, 10, None, None, &mut rng);
    let mut ref_tn = r_tn.clone();
    let mut ref_opt = Greedy::new(&ref_tn, CostType::Flops);
    ref_opt.optimize_path();
    let ref_path = ref_opt.get_best_replace_path();
    contract_tensor_network(&mut ref_tn, &ref_path);

    let partitioning = find_partitioning(&r_tn, 5, std::string::String::from("tests/km1"), true);
    let mut partitioned_tn = partition_tensor_network(&r_tn, &partitioning);
    let mut opt = Greedy::new(&partitioned_tn, CostType::Flops);
    opt.optimize_path();
    let path = opt.get_best_replace_path();
    contract_tensor_network(&mut partitioned_tn, &path);

    assert_eq!(*ref_tn.get_tensor_data(), *partitioned_tn.get_tensor_data());
}

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
        let r_tn = sycamore_circuit(k, 10, None, None, &mut rng);
        ref_tn = r_tn.clone();
        let partitioning =
            find_partitioning(&r_tn, size, std::string::String::from("tests/km1"), true);
        partitioned_tn = partition_tensor_network(&r_tn, &partitioning);
        let mut opt = Greedy::new(&partitioned_tn, CostType::Flops);

        opt.optimize_path();
        path = opt.get_best_replace_path();
    }
    world.barrier();
    let (mut local_tn, local_path) =
        scatter_tensor_network(partitioned_tn, &path, rank, size, &world);
    contract_tensor_network(&mut local_tn, &local_path);

    let return_tn = naive_gather_tensor_network(local_tn, &path, rank, size, &world);
    world.barrier();

    if rank == 0 {
        let mut ref_opt = Greedy::new(&ref_tn, CostType::Flops);

        ref_opt.optimize_path();
        let ref_path = ref_opt.get_best_replace_path();
        contract_tensor_network(&mut ref_tn, &ref_path);
        assert_eq!(*return_tn.get_tensor_data(), *ref_tn.get_tensor_data());
    }
}
