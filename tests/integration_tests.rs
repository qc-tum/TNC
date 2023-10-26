use rand::{rngs::StdRng, SeedableRng};
use tensorcontraction::{
    contractionpath::paths::{CostType, Greedy, OptimizePath},
    random::tensorgeneration::create_filled_tensor_network,
    tensornetwork::{
        contraction::contract_tensor_network,
        partitioning::{find_partitioning, partition_tensor_network},
        tensor::Tensor,
    },
};

fn setup_complex() -> Tensor {
    let mut rng = StdRng::seed_from_u64(52);
    create_filled_tensor_network(
        vec![
            Tensor::new(vec![4, 3, 2]),
            Tensor::new(vec![0, 1, 3, 2]),
            Tensor::new(vec![4, 5, 6]),
            Tensor::new(vec![6, 8, 9]),
            Tensor::new(vec![10, 8, 9]),
            Tensor::new(vec![5, 1, 0]),
        ],
        &[
            (0, 27),
            (1, 18),
            (2, 12),
            (3, 15),
            (4, 5),
            (5, 3),
            (6, 18),
            (7, 22),
            (8, 45),
            (9, 65),
            (10, 5),
            (11, 17),
        ]
        .into(),
        None,
        &mut rng,
    )
}

#[test]
fn test_partitioned_tn_contraction() {
    let mut tn = setup_complex();
    let partitioning = find_partitioning(&mut tn, 3, std::string::String::from("tests/km1"));
    let mut partitioned_tn: Tensor = partition_tensor_network(&tn, partitioning.as_slice());
    // for tensor in partitioned_tn.get_tensors().iter() {
    //     for tn in tensor.get_tensors().iter() {
    //         println!("{:?}", tn.shape());
    //     }
    // }
    // let mut opt = Greedy::new(&partitioned_tn, CostType::Flops);
    // opt.optimize_path();
    // contract_tensor_network(&mut partitioned_tn);
}
