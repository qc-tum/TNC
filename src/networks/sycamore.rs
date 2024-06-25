use super::connectivity::{Connectivity, ConnectivityLayout};
use rand::distributions::Uniform;
use rand::prelude::Distribution;
use rand::{thread_rng, Rng};
use std::collections::HashMap;

use crate::contractionpath::paths::OptimizePath;
use crate::contractionpath::paths::{greedy::Greedy, CostType};
use crate::contractionpath::random_paths::RandomOptimizePath;
use crate::random::tensorgeneration::random_sparse_tensor_data_with_rng;
use crate::tensornetwork::contraction::contract_tensor_network;
use crate::tensornetwork::tensor::Tensor;
use crate::tensornetwork::tensordata::TensorData;

macro_rules! fsim {
    ($a:expr, $b:expr) => {
        $crate::tensornetwork::tensordata::TensorData::Gate((String::from("FSIM"), vec![$a, $b]))
    };
}

pub fn sycamore_circuit<R>(
    size: usize,
    round: usize,
    single_qubit_probability: f64,
    two_qubit_probability: f64,
    rng: &mut R,
    connectivity: ConnectivityLayout,
) -> Tensor
where
    R: Rng + ?Sized,
{
    assert!(
        (0.0..=1.0).contains(&single_qubit_probability),
        "Probabilities should range from 0.0 to 1.0"
    );
    assert!(
        (0.0..=1.0).contains(&two_qubit_probability),
        "Probabilities should range from 0.0 to 1.0"
    );
    let single_qubit_gate = HashMap::from([
        (0, TensorData::Gate((String::from("SQRX"), Vec::new()))),
        (1, TensorData::Gate((String::from("SQRY"), Vec::new()))),
        (2, TensorData::Gate((String::from("SQRZ"), Vec::new()))),
    ]);

    let mut open_edges = HashMap::new();

    // Initialize tensornetwork of size `usize`
    let mut sycamore_tn = Tensor::default();
    let mut sycamore_bonddims = HashMap::new();

    let sycamore_connect = Connectivity::new(connectivity);
    // Filter connectivity map
    let filtered_connectivity = sycamore_connect
        .connectivity
        .iter()
        .filter(|&&(u, v)| u < size && v < size)
        .collect::<Vec<_>>();

    let mut next_edge = size;
    let uniform_prob = Uniform::new(0.0, 1.0);

    // set up initial state
    let mut initial_state = Vec::with_capacity(size);
    for i in 0..size {
        let mut new_state = Tensor::new(vec![i]);
        new_state.set_tensor_data(random_sparse_tensor_data_with_rng(&[2], None, rng));
        sycamore_bonddims.insert(i, 2);
        open_edges.insert(i, i);
        initial_state.push(new_state);
    }

    sycamore_tn.push_tensors(initial_state, Some(&sycamore_bonddims), None);

    let die = Uniform::from(0..3);
    let mut intermediate_gates = Vec::new();
    for _ in 1..round {
        for i in 0..size {
            // Placing of random single qubit gate
            if rng.sample(uniform_prob) < single_qubit_probability {
                sycamore_bonddims.insert(next_edge, 2);
                let mut new_tensor = Tensor::new(vec![open_edges[&i], next_edge]);
                new_tensor.set_tensor_data(single_qubit_gate[&die.sample(rng)].clone());
                intermediate_gates.push(new_tensor);
                open_edges.entry(i).insert_entry(next_edge);
                next_edge += 1;
            }
        }
        for (i, j) in &filtered_connectivity {
            // Placing of random two qubit gate
            if rng.sample(uniform_prob) < two_qubit_probability {
                sycamore_bonddims.insert(next_edge, 2);
                sycamore_bonddims.insert(next_edge + 1, 2);
                let mut new_tensor =
                    Tensor::new(vec![open_edges[i], open_edges[j], next_edge, next_edge + 1]);
                new_tensor.set_tensor_data(fsim!(0.3, 0.2));
                intermediate_gates.push(new_tensor);
                open_edges.entry(*i).insert_entry(next_edge);
                open_edges.entry(*j).insert_entry(next_edge + 1);
                next_edge += 2;
            }
        }
    }
    sycamore_tn.push_tensors(intermediate_gates, Some(&sycamore_bonddims), None);

    // set up final state
    let mut final_state = Vec::with_capacity(open_edges.len());
    for (_index, i) in open_edges {
        let mut new_state = Tensor::new(vec![i]);
        new_state.set_tensor_data(random_sparse_tensor_data_with_rng(&[2], None, rng));
        final_state.push(new_state);
    }
    sycamore_tn.push_tensors(final_state, Some(&sycamore_bonddims), None);

    sycamore_tn
}

pub fn sycamore_contract(mut tn: Tensor) {
    let mut opt = Greedy::new(&tn, CostType::Flops);
    opt.random_optimize_path(32, &mut thread_rng());
    let contract_path = opt.get_best_replace_path();
    contract_tensor_network(&mut tn, &contract_path);
}
