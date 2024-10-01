use super::connectivity::{Connectivity, ConnectivityLayout};
use rand::distributions::Uniform;
use rand::prelude::Distribution;
use rand::{thread_rng, Rng};
use rustc_hash::FxHashMap;

use crate::contractionpath::paths::OptimizePath;
use crate::contractionpath::paths::{greedy::Greedy, CostType};
use crate::contractionpath::random_paths::RandomOptimizePath;
use crate::random::tensorgeneration::random_sparse_tensor_data_with_rng;
use crate::tensornetwork::contraction::contract_tensor_network;
use crate::tensornetwork::tensor::Tensor;
use crate::tensornetwork::tensordata::TensorData;
use crate::utils::datastructures::UnionFind;

macro_rules! fsim {
    ($a:expr, $b:expr) => {
        $crate::tensornetwork::tensordata::TensorData::Gate((String::from("fsim"), vec![$a, $b]))
    };
}

pub fn random_connected_circuit<R>(
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
    const MAX_TRIES: usize = 100;
    for _ in 0..MAX_TRIES {
        let (is_connected, tn) = random_circuit(
            size,
            round,
            single_qubit_probability,
            two_qubit_probability,
            rng,
            connectivity,
        );
        if is_connected {
            return tn;
        }
    }
    panic!("Could not generate a connected circuit in {MAX_TRIES} tries");
}

pub fn random_circuit<R>(
    size: usize,
    round: usize,
    single_qubit_probability: f64,
    two_qubit_probability: f64,
    rng: &mut R,
    connectivity: ConnectivityLayout,
) -> (bool, Tensor)
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
    let single_qubit_gates = [
        TensorData::Gate((String::from("sx"), Vec::new())),
        TensorData::Gate((String::from("sy"), Vec::new())),
        TensorData::Gate((String::from("sz"), Vec::new())),
    ];

    let mut open_edges = FxHashMap::default();

    // Initialize tensornetwork of size `usize`
    let mut sycamore_tn = Tensor::default();
    let mut sycamore_bonddims = FxHashMap::default();

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
        new_state.set_tensor_data(random_sparse_tensor_data_with_rng(&[2], Some(1f32), rng));
        sycamore_bonddims.insert(i, 2);
        open_edges.insert(i, i);
        initial_state.push(new_state);
    }
    sycamore_tn.push_tensors(initial_state, Some(&sycamore_bonddims), None);

    // set up intermediate gates
    let mut connected = UnionFind::new(size);
    let die = Uniform::from(0..single_qubit_gates.len());
    let mut intermediate_gates = Vec::new();
    for _ in 1..round {
        for i in 0..size {
            // Placing of random single qubit gate
            if rng.sample(uniform_prob) < single_qubit_probability {
                sycamore_bonddims.insert(next_edge, 2);
                let mut new_tensor = Tensor::new(vec![open_edges[&i], next_edge]);
                new_tensor.set_tensor_data(single_qubit_gates[die.sample(rng)].clone());
                intermediate_gates.push(new_tensor);
                open_edges.insert(i, next_edge);
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
                open_edges.insert(*i, next_edge);
                open_edges.insert(*j, next_edge + 1);
                next_edge += 2;
                connected.union(*i, *j);
            }
        }
    }
    sycamore_tn.push_tensors(intermediate_gates, Some(&sycamore_bonddims), None);

    // set up final state
    let mut final_state = Vec::with_capacity(open_edges.len());
    for i in 0..size {
        let mut new_state = Tensor::new(vec![open_edges[&i]]);
        new_state.set_tensor_data(random_sparse_tensor_data_with_rng(&[2], Some(1f32), rng));
        final_state.push(new_state);
    }
    sycamore_tn.push_tensors(final_state, Some(&sycamore_bonddims), None);

    let is_connected = connected.count_sets() == 1;
    (is_connected, sycamore_tn)
}

pub fn sycamore_contract(mut tn: Tensor) {
    let mut opt = Greedy::new(&tn, CostType::Flops);
    opt.random_optimize_path(32, &mut thread_rng());
    let contract_path = opt.get_best_replace_path();
    contract_tensor_network(&mut tn, &contract_path);
}
