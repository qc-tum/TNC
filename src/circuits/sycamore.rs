use rand::distributions::Uniform;
use rand::{thread_rng, Rng};
use std::collections::HashMap;

use crate::contractionpath::paths::{CostType, Greedy, OptimizePath};
use crate::contractionpath::random_paths::RandomOptimizePath;
use crate::random::tensorgeneration::random_sparse_tensor;
use crate::tensornetwork::contraction::tn_contract;
use crate::tensornetwork::{tensor::Tensor, TensorNetwork};

const DEFAULT_TWO_QUBIT_PROBABILITY: f64 = 0.4;
const DEFAULT_SINGLE_QUBIT_PROBABILITY: f64 = 0.4;

pub fn sycamore_circuit<R>(
    size: usize,
    round: usize,
    single_qubit: Option<f64>,
    two_qubit: Option<f64>,
    rng: &mut R,
) -> TensorNetwork
where
    R: Rng + ?Sized,
{
    let sycamore_connect = vec![
        (0, 1),
        (0, 3),
        (1, 4),
        (2, 3),
        (3, 4),
        (4, 5),
        (2, 6),
        (3, 7),
        (4, 8),
        (5, 9),
        (6, 7),
        (7, 8),
        (8, 9),
        (9, 10),
        (6, 13),
        (7, 14),
        (8, 15),
        (9, 16),
        (10, 17),
        (11, 12),
        (12, 13),
        (13, 14),
        (14, 15),
        (15, 16),
        (16, 17),
        (17, 18),
        (11, 20),
        (12, 21),
        (13, 22),
        (14, 23),
        (15, 24),
        (16, 25),
        (17, 26),
        (18, 27),
        (19, 20),
        (20, 21),
        (21, 22),
        (22, 23),
        (23, 24),
        (24, 25),
        (25, 26),
        (26, 27),
        (19, 29),
        (20, 31),
        (21, 31),
        (22, 32),
        (23, 33),
        (24, 34),
        (25, 35),
        (26, 36),
        (28, 29),
        (29, 30),
        (30, 31),
        (31, 32),
        (32, 33),
        (33, 34),
        (34, 35),
        (35, 36),
        (29, 37),
        (30, 38),
        (31, 39),
        (32, 40),
        (33, 41),
        (34, 42),
        (35, 43),
        (37, 38),
        (38, 39),
        (39, 40),
        (40, 41),
        (41, 42),
        (42, 43),
        (38, 44),
        (39, 45),
        (40, 46),
        (41, 47),
        (42, 48),
        (44, 45),
        (45, 46),
        (46, 47),
        (47, 48),
        (45, 49),
        (46, 50),
        (47, 51),
        (49, 50),
        (50, 51),
        (50, 52),
    ];

    let mut open_edges = HashMap::new();

    // Initialize tensornetwork of size `usize`
    let mut sycamore_tn = TensorNetwork::empty_tensor_network();
    let mut sycamore_bonddims = HashMap::new();
    let mut tensors = Vec::with_capacity(size);
    for i in 0..size {
        tensors.push(Tensor::new(vec![i]));
        sycamore_bonddims.insert(i, 2);
        open_edges.insert(i, i);
    }

    // Filter connectivity map
    let filtered_connectivity = sycamore_connect
        .iter()
        .filter(|&&(u, v)| u < size && v < size)
        .collect::<Vec<_>>();

    let single_qubit_probability = if let Some(single_qubit_probability) = single_qubit {
        assert!(
            single_qubit_probability <= 1.0,
            "Probability should range [0.0, 1.0], values greater than one are not acceptable"
        );
        assert!(
            single_qubit_probability >= 0.0,
            "Probability should range [0.0, 1.0], values less than zero are not acceptable"
        );
        single_qubit_probability
    } else {
        DEFAULT_SINGLE_QUBIT_PROBABILITY
    };
    let two_qubit_probability = if let Some(two_qubit_probability) = two_qubit {
        assert!(
            two_qubit_probability <= 1.0,
            "Probability should range [0.0, 1.0], values greater than one are not acceptable"
        );
        assert!(
            two_qubit_probability >= 0.0,
            "Probability should range [0.0, 1.0], values less than zero are not acceptable"
        );
        two_qubit_probability
    } else {
        DEFAULT_TWO_QUBIT_PROBABILITY
    };
    let mut next_edge = size;
    let uniform_prob = Uniform::new(0.0, 1.0);
    sycamore_tn.push_tensors(tensors, Some(&sycamore_bonddims), None);
    for _ in 1..round {
        for i in 0..size {
            if rng.sample(uniform_prob) < single_qubit_probability {
                sycamore_bonddims.insert(next_edge, 2);
                sycamore_tn.push_tensor(
                    Tensor::new(vec![open_edges[&i], next_edge]),
                    Some(&sycamore_bonddims),
                    None,
                );
                open_edges.entry(i).insert_entry(next_edge);
                next_edge += 1;
            }
        }
        for (i, j) in filtered_connectivity.iter() {
            if rng.sample(uniform_prob) < two_qubit_probability {
                sycamore_bonddims.insert(next_edge, 2);
                sycamore_bonddims.insert(next_edge + 1, 2);
                sycamore_tn.push_tensor(
                    Tensor::new(vec![
                        open_edges[&i],
                        open_edges[&j],
                        next_edge,
                        next_edge + 1,
                    ]),
                    Some(&sycamore_bonddims),
                    None,
                );
                open_edges.entry(*i).insert_entry(next_edge);
                open_edges.entry(*j).insert_entry(next_edge + 1);
                next_edge += 2;
            }
        }
    }
    sycamore_tn
}

pub fn sycamore_contract(tn: TensorNetwork) -> TensorNetwork {
    let mut opt = Greedy::new(&tn, CostType::Flops);
    opt.random_optimize_path(32, &mut thread_rng());
    let contract_path = opt.get_best_replace_path();
    let mut d_tn = Vec::new();
    for t in tn.get_tensors() {
        d_tn.push(random_sparse_tensor(t, tn.get_bond_dims(), None));
    }
    let (tn, _dt) = tn_contract(tn, d_tn, &contract_path);
    tn
}
