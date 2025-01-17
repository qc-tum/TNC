use super::connectivity::{Connectivity, ConnectivityLayout};
use itertools::Itertools;
use rand::distributions::Uniform;
use rand::prelude::Distribution;
use rand::Rng;
use rustc_hash::FxHashMap;

use crate::random::tensorgeneration::random_sparse_tensor_data_with_rng;
use crate::tensornetwork::tensor::Tensor;
use crate::tensornetwork::tensordata::TensorData;

macro_rules! fsim {
    ($a:expr, $b:expr, $c:expr) => {
        $crate::tensornetwork::tensordata::TensorData::Gate((
            String::from("fsim"),
            vec![$a, $b],
            $c,
        ))
    };
}

pub fn random_circuit<R>(
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
    let single_qubit_gates = [
        TensorData::Gate((String::from("sx"), Vec::new(), false)),
        TensorData::Gate((String::from("sy"), Vec::new(), false)),
        TensorData::Gate((String::from("sz"), Vec::new(), false)),
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
        .collect_vec();

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

    sycamore_tn.push_tensors(initial_state, Some(&sycamore_bonddims));

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
                new_tensor.set_tensor_data(fsim!(0.3, 0.2, false));
                intermediate_gates.push(new_tensor);
                open_edges.insert(*i, next_edge);
                open_edges.insert(*j, next_edge + 1);
                next_edge += 2;
            }
        }
    }
    sycamore_tn.push_tensors(intermediate_gates, Some(&sycamore_bonddims));

    // set up final state
    let mut final_state = Vec::with_capacity(open_edges.len());
    for i in 0..size {
        let mut new_state = Tensor::new(vec![open_edges[&i]]);
        new_state.set_tensor_data(random_sparse_tensor_data_with_rng(&[2], Some(1f32), rng));
        final_state.push(new_state);
    }
    sycamore_tn.push_tensors(final_state, Some(&sycamore_bonddims));

    sycamore_tn
}

pub fn random_circuit_with_observable<R>(
    size: usize,
    round: usize,
    single_qubit_probability: f64,
    two_qubit_probability: f64,
    observable_probability: f64,
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

    assert!(
        (0.0..=1.0).contains(&observable_probability),
        "Probabilities should range from 0.0 to 1.0"
    );

    let single_qubit_gates = [
        TensorData::Gate((String::from("sx"), Vec::new(), false)),
        TensorData::Gate((String::from("sy"), Vec::new(), false)),
        TensorData::Gate((String::from("sz"), Vec::new(), false)),
    ];

    let single_qubit_adjoint_gates = [
        TensorData::Gate((String::from("sx"), Vec::new(), true)),
        TensorData::Gate((String::from("sy"), Vec::new(), true)),
        TensorData::Gate((String::from("sz"), Vec::new(), true)),
    ];

    let observables = [
        TensorData::Gate((String::from("x"), Vec::new(), false)),
        TensorData::Gate((String::from("y"), Vec::new(), false)),
        TensorData::Gate((String::from("z"), Vec::new(), false)),
    ];

    let uniform_prob = Uniform::new(0.0, 1.0);

    // Initialize tensornetwork of size `usize`
    let mut random_tn = Tensor::default();
    let mut bond_dims = FxHashMap::default();

    let mut open_edges = FxHashMap::default();

    let observable_die = Uniform::from(0..observables.len());
    let mut next_edge = 0;

    for i in 0..size {
        // Placing of random observable
        if rng.sample(uniform_prob) < observable_probability {
            bond_dims.insert(next_edge, 2);
            bond_dims.insert(next_edge + 1, 2);
            open_edges.insert(i, (next_edge, next_edge + 1));
            next_edge += 2;

            let new_observable = observables[observable_die.sample(rng)].clone();

            let mut new_tensor = Tensor::new(vec![open_edges[&i].0, open_edges[&i].1]);

            new_tensor.set_tensor_data(new_observable);
            random_tn.push_tensor(new_tensor, Some(&bond_dims));
        } else {
            // set empty positions
            open_edges.insert(i, (0, 0));
        }
    }

    let single_qubit_gate_die = Uniform::from(0..single_qubit_gates.len());
    let connectivity_graph = Connectivity::new(connectivity);
    // Filter connectivity map
    let filtered_connectivity = connectivity_graph
        .connectivity
        .iter()
        .filter(|&&(u, v)| u < size && v < size)
        .collect_vec();

    // setup intermediate gates. only place gates if all legs are not -1 (since they will cancel out otherwise)
    let mut intermediate_gates = Vec::new();
    for _ in 1..round {
        // Placing of random two qubit gate if affects outcome of observable
        for (i, j) in &filtered_connectivity {
            // Placing of random two qubit gate
            if rng.sample(uniform_prob) < two_qubit_probability
                && (open_edges[i].0 != open_edges[i].1 || open_edges[j].0 != open_edges[j].1)
            {
                let i_indices = if open_edges[i].0 != open_edges[i].1 {
                    (open_edges[i].0, open_edges[i].1)
                } else {
                    bond_dims.insert(next_edge, 2);
                    next_edge += 1;
                    (next_edge - 1, next_edge - 1)
                };

                let j_indices = if open_edges[j].0 != open_edges[j].1 {
                    (open_edges[j].0, open_edges[j].1)
                } else {
                    bond_dims.insert(next_edge, 2);
                    next_edge += 1;
                    (next_edge - 1, next_edge - 1)
                };

                bond_dims.insert(next_edge, 2);
                bond_dims.insert(next_edge + 1, 2);

                let mut left_new_tensor =
                    Tensor::new(vec![next_edge, next_edge + 1, i_indices.0, j_indices.0]);
                left_new_tensor.set_tensor_data(fsim!(0.3, 0.2, false));
                intermediate_gates.push(left_new_tensor);

                bond_dims.insert(next_edge + 2, 2);
                bond_dims.insert(next_edge + 3, 2);
                let mut right_new_tensor =
                    Tensor::new(vec![i_indices.1, j_indices.1, next_edge + 2, next_edge + 3]);
                right_new_tensor.set_tensor_data(fsim!(0.3, 0.2, true));
                intermediate_gates.push(right_new_tensor);

                open_edges.insert(*i, (next_edge, next_edge + 2));
                open_edges.insert(*j, (next_edge + 1, next_edge + 3));

                next_edge += 4;
            }
        }

        for i in 0..size {
            // Placing of random single qubit gate if affects outcome of observable
            if rng.sample(uniform_prob) < single_qubit_probability
                && open_edges[&i].0 != open_edges[&i].1
            {
                let new_gate_index = single_qubit_gate_die.sample(rng);
                let left_new_gate = single_qubit_gates[new_gate_index].clone();
                bond_dims.insert(next_edge, 2);
                let mut left_new_tensor = Tensor::new(vec![next_edge, open_edges[&i].0]);
                left_new_tensor.set_tensor_data(left_new_gate);
                intermediate_gates.push(left_new_tensor);

                let right_new_gate = single_qubit_adjoint_gates[new_gate_index].clone();
                bond_dims.insert(next_edge + 1, 2);
                let mut right_new_tensor = Tensor::new(vec![open_edges[&i].1, next_edge + 1]);
                right_new_tensor.set_tensor_data(right_new_gate);
                intermediate_gates.push(right_new_tensor);

                open_edges.insert(i, (next_edge, next_edge + 1));
                next_edge += 2;
            }
        }
    }
    random_tn.push_tensors(intermediate_gates, Some(&bond_dims));

    // set up random initial state
    let mut initial_state = Vec::new();
    for i in 0..size {
        let (left_index, right_index) = open_edges[&i];
        if left_index != right_index {
            let random_sparse_tensor = random_sparse_tensor_data_with_rng(&[2], Some(1f32), rng);

            let mut left_new_state = Tensor::new(vec![left_index]);
            left_new_state.set_tensor_data(random_sparse_tensor.clone());
            initial_state.push(left_new_state);

            let mut right_new_state = Tensor::new(vec![right_index]);
            right_new_state.set_tensor_data(random_sparse_tensor);
            initial_state.push(right_new_state);
        }
    }

    random_tn.push_tensors(initial_state, Some(&bond_dims));

    random_tn
}
