use super::connectivity::{Connectivity, ConnectivityLayout};
use itertools::Itertools;
use rand::distributions::Bernoulli;
use rand::seq::SliceRandom;
use rand::Rng;
use rustc_hash::FxHashMap;

use crate::random::tensorgeneration::random_sparse_tensor_data_with_rng;
use crate::tensornetwork::tensor::Tensor;
use crate::tensornetwork::tensordata::TensorData;
use crate::utils::traits::WithCapacity;

macro_rules! fsim {
    ($a:expr, $b:expr, $c:expr) => {
        $crate::tensornetwork::tensordata::TensorData::Gate((
            String::from("fsim"),
            vec![$a, $b],
            $c,
        ))
    };
}

/// Creates a random circuit with `rounds` many rounds of single and two qubit gate
/// layers. Places the gates with the given probabilities and only on qubit pairs
/// specified by the `connectivity`.
pub fn random_circuit<R>(
    qubits: usize,
    rounds: usize,
    single_qubit_probability: f64,
    two_qubit_probability: f64,
    rng: &mut R,
    connectivity: ConnectivityLayout,
) -> Tensor
where
    R: Rng + ?Sized,
{
    let single_qubit_gates = [
        TensorData::Gate((String::from("sx"), Vec::new(), false)),
        TensorData::Gate((String::from("sy"), Vec::new(), false)),
        TensorData::Gate((String::from("sz"), Vec::new(), false)),
    ];

    let mut open_edges = FxHashMap::with_capacity(qubits);

    // Initialize tensornetwork of size `usize`
    let mut circuit_tn = Tensor::default();

    // Get connectivity for given size
    let connectivity_graph = Connectivity::new(connectivity);
    let filtered_connectivity = connectivity_graph
        .connectivity
        .iter()
        .filter(|&&(u, v)| u < qubits && v < qubits)
        .collect_vec();

    let mut next_edge = qubits;
    let single_qubit_die = Bernoulli::new(single_qubit_probability).unwrap();
    let two_qubit_die = Bernoulli::new(two_qubit_probability).unwrap();

    // set up initial state
    let mut initial_state = Vec::with_capacity(qubits);
    for i in 0..qubits {
        let mut new_state = Tensor::new_from_const(vec![i], 2);
        new_state.set_tensor_data(random_sparse_tensor_data_with_rng(&[2], Some(1f32), rng));
        open_edges.insert(i, i);
        initial_state.push(new_state);
    }
    circuit_tn.push_tensors(initial_state);

    let mut intermediate_gates = Vec::new();
    for _ in 1..rounds {
        for i in 0..qubits {
            // Placing of random single qubit gate
            if rng.sample(single_qubit_die) {
                let mut new_tensor = Tensor::new_from_const(vec![open_edges[&i], next_edge], 2);
                new_tensor.set_tensor_data(single_qubit_gates.choose(rng).unwrap().clone());
                intermediate_gates.push(new_tensor);
                open_edges.insert(i, next_edge);
                next_edge += 1;
            }
        }
        for (i, j) in &filtered_connectivity {
            // Placing of random two qubit gate
            if rng.sample(two_qubit_die) {
                let mut new_tensor = Tensor::new_from_const(
                    vec![open_edges[i], open_edges[j], next_edge, next_edge + 1],
                    2,
                );
                new_tensor.set_tensor_data(fsim!(0.3, 0.2, false));
                intermediate_gates.push(new_tensor);
                open_edges.insert(*i, next_edge);
                open_edges.insert(*j, next_edge + 1);
                next_edge += 2;
            }
        }
    }
    circuit_tn.push_tensors(intermediate_gates);

    // set up final state
    let mut final_state = Vec::with_capacity(qubits);
    for i in 0..qubits {
        let mut new_state = Tensor::new_from_const(vec![open_edges[&i]], 2);
        new_state.set_tensor_data(random_sparse_tensor_data_with_rng(&[2], Some(1f32), rng));
        final_state.push(new_state);
    }
    circuit_tn.push_tensors(final_state);

    circuit_tn
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
    let observable_locations = (0..size)
        .filter(|_| rng.gen_bool(observable_probability))
        .collect();

    random_circuit_with_set_observable(
        size,
        round,
        single_qubit_probability,
        two_qubit_probability,
        observable_locations,
        rng,
        connectivity,
    )
}

pub fn random_circuit_with_set_observable<R>(
    size: usize,
    round: usize,
    single_qubit_probability: f64,
    two_qubit_probability: f64,
    observable_location: Vec<usize>,
    rng: &mut R,
    connectivity: ConnectivityLayout,
) -> Tensor
where
    R: Rng + ?Sized,
{
    let single_qubit_gates = [
        (
            TensorData::Gate((String::from("sx"), Vec::new(), false)),
            TensorData::Gate((String::from("sx"), Vec::new(), true)),
        ),
        (
            TensorData::Gate((String::from("sy"), Vec::new(), false)),
            TensorData::Gate((String::from("sx"), Vec::new(), true)),
        ),
        (
            TensorData::Gate((String::from("sz"), Vec::new(), false)),
            TensorData::Gate((String::from("sx"), Vec::new(), true)),
        ),
    ];

    let observables = [
        TensorData::Gate((String::from("x"), Vec::new(), false)),
        TensorData::Gate((String::from("y"), Vec::new(), false)),
        TensorData::Gate((String::from("z"), Vec::new(), false)),
    ];

    let single_qubit_die = Bernoulli::new(single_qubit_probability).unwrap();
    let two_qubit_die = Bernoulli::new(two_qubit_probability).unwrap();

    // Initialize tensornetwork of size `usize`
    let mut random_tn = Tensor::default();

    let mut open_edges = FxHashMap::with_capacity(size);

    let mut next_edge = 0;

    let mut final_state = Vec::with_capacity(observable_location.len());
    for i in 0..size {
        // Placing of random observable
        if observable_location.contains(&i) {
            open_edges.insert(i, (next_edge, next_edge + 1));
            next_edge += 2;

            let new_observable = observables.choose(rng).unwrap().clone();
            let mut new_tensor =
                Tensor::new_from_const(vec![open_edges[&i].0, open_edges[&i].1], 2);
            new_tensor.set_tensor_data(new_observable);
            final_state.push(new_tensor);
        } else {
            // set empty positions
            open_edges.insert(i, (0, 0));
        }
    }
    random_tn.push_tensors(final_state);

    // Get connectivity for given size
    let connectivity_graph = Connectivity::new(connectivity);
    let filtered_connectivity = connectivity_graph
        .connectivity
        .iter()
        .filter(|&&(u, v)| u < size && v < size)
        .collect_vec();

    let mut intermediate_gates = Vec::new();
    for _ in 1..round {
        // Placing of random two qubit gate if affects outcome of observable
        for (i, j) in &filtered_connectivity {
            // Placing of random two qubit gate
            if rng.sample(two_qubit_die)
                && (open_edges[i].0 != open_edges[i].1 || open_edges[j].0 != open_edges[j].1)
            {
                let (left_i_index, right_i_index) = if open_edges[i].0 != open_edges[i].1 {
                    (open_edges[i].0, open_edges[i].1)
                } else {
                    next_edge += 1;
                    (next_edge - 1, next_edge - 1)
                };

                let (left_j_index, right_j_index) = if open_edges[j].0 != open_edges[j].1 {
                    (open_edges[j].0, open_edges[j].1)
                } else {
                    next_edge += 1;
                    (next_edge - 1, next_edge - 1)
                };

                let mut left_new_tensor = Tensor::new_from_const(
                    vec![next_edge, next_edge + 1, left_i_index, left_j_index],
                    2,
                );
                left_new_tensor.set_tensor_data(fsim!(0.3, 0.2, false));
                intermediate_gates.push(left_new_tensor);

                let mut right_new_tensor = Tensor::new_from_const(
                    vec![right_i_index, right_j_index, next_edge + 2, next_edge + 3],
                    2,
                );
                right_new_tensor.set_tensor_data(fsim!(0.3, 0.2, true));
                intermediate_gates.push(right_new_tensor);

                open_edges.insert(*i, (next_edge, next_edge + 2));
                open_edges.insert(*j, (next_edge + 1, next_edge + 3));

                next_edge += 4;
            }
        }

        for i in 0..size {
            // Placing of random single qubit gate if affects outcome of observable
            let (left_index, right_index) = open_edges[&i];
            if rng.sample(single_qubit_die) && left_index != right_index {
                let (left_new_gate, right_new_gate) =
                    single_qubit_gates.choose(rng).unwrap().clone();

                let mut left_new_tensor = Tensor::new_from_const(vec![next_edge, left_index], 2);
                left_new_tensor.set_tensor_data(left_new_gate);
                intermediate_gates.push(left_new_tensor);

                let mut right_new_tensor =
                    Tensor::new_from_const(vec![right_index, next_edge + 1], 2);
                right_new_tensor.set_tensor_data(right_new_gate);
                intermediate_gates.push(right_new_tensor);

                open_edges.insert(i, (next_edge, next_edge + 1));
                next_edge += 2;
            }
        }
    }
    random_tn.push_tensors(intermediate_gates);

    // set up random initial state
    let mut initial_state = Vec::with_capacity(size);
    for i in 0..size {
        let (left_index, right_index) = open_edges[&i];
        if left_index != right_index {
            let random_sparse_tensor = random_sparse_tensor_data_with_rng(&[2], Some(1f32), rng);

            let mut left_new_state = Tensor::new_from_const(vec![left_index], 2);
            left_new_state.set_tensor_data(random_sparse_tensor.clone());
            initial_state.push(left_new_state);

            let mut right_new_state = Tensor::new_from_const(vec![right_index], 2);
            right_new_state.set_tensor_data(random_sparse_tensor);
            initial_state.push(right_new_state);
        }
    }
    random_tn.push_tensors(initial_state);

    random_tn
}

#[cfg(test)]
mod tests {
    use std::iter::zip;

    use rand::thread_rng;

    use crate::{
        networks::{
            connectivity::ConnectivityLayout, random_circuit::random_circuit_with_set_observable,
        },
        tensornetwork::tensor::Tensor,
    };

    use super::random_circuit_with_observable;

    #[test]
    fn test_random_circuit_with_observable() {
        let size = 4;
        let rounds = 3;
        let single_qubit_probability = 1f64;
        let two_qubit_probability = 1f64;
        let observable_probability = 1f64;
        // results should be independent of rng used.
        let mut rng = thread_rng();
        let connectivity = ConnectivityLayout::Line(size);
        let circuit = random_circuit_with_observable(
            size,
            rounds,
            single_qubit_probability,
            two_qubit_probability,
            observable_probability,
            &mut rng,
            connectivity,
        );
        let ref_legs = [
            Tensor::new_from_const(vec![0, 1], 2),
            Tensor::new_from_const(vec![2, 3], 2),
            Tensor::new_from_const(vec![4, 5], 2),
            Tensor::new_from_const(vec![6, 7], 2),
            Tensor::new_from_const(vec![8, 9, 0, 2], 2),
            Tensor::new_from_const(vec![1, 3, 10, 11], 2),
            Tensor::new_from_const(vec![12, 13, 9, 4], 2),
            Tensor::new_from_const(vec![11, 5, 14, 15], 2),
            Tensor::new_from_const(vec![16, 17, 13, 6], 2),
            Tensor::new_from_const(vec![15, 7, 18, 19], 2),
            Tensor::new_from_const(vec![20, 8], 2),
            Tensor::new_from_const(vec![10, 21], 2),
            Tensor::new_from_const(vec![22, 12], 2),
            Tensor::new_from_const(vec![14, 23], 2),
            Tensor::new_from_const(vec![24, 16], 2),
            Tensor::new_from_const(vec![18, 25], 2),
            Tensor::new_from_const(vec![26, 17], 2),
            Tensor::new_from_const(vec![19, 27], 2),
            Tensor::new_from_const(vec![28, 29, 20, 22], 2),
            Tensor::new_from_const(vec![21, 23, 30, 31], 2),
            Tensor::new_from_const(vec![32, 33, 29, 24], 2),
            Tensor::new_from_const(vec![31, 25, 34, 35], 2),
            Tensor::new_from_const(vec![36, 37, 33, 26], 2),
            Tensor::new_from_const(vec![35, 27, 38, 39], 2),
            Tensor::new_from_const(vec![40, 28], 2),
            Tensor::new_from_const(vec![30, 41], 2),
            Tensor::new_from_const(vec![42, 32], 2),
            Tensor::new_from_const(vec![34, 43], 2),
            Tensor::new_from_const(vec![44, 36], 2),
            Tensor::new_from_const(vec![38, 45], 2),
            Tensor::new_from_const(vec![46, 37], 2),
            Tensor::new_from_const(vec![39, 47], 2),
            Tensor::new_from_const(vec![40], 2),
            Tensor::new_from_const(vec![41], 2),
            Tensor::new_from_const(vec![42], 2),
            Tensor::new_from_const(vec![43], 2),
            Tensor::new_from_const(vec![44], 2),
            Tensor::new_from_const(vec![45], 2),
            Tensor::new_from_const(vec![46], 2),
            Tensor::new_from_const(vec![47], 2),
        ];

        assert_eq!(circuit.tensors().len(), 40);
        for (tensor, ref_tensor) in zip(circuit.tensors(), ref_legs) {
            assert_eq!(tensor.legs(), ref_tensor.legs());
            assert_eq!(tensor.bond_dims(), ref_tensor.bond_dims());
        }
    }

    #[test]
    fn test_random_circuit_with_set_observable() {
        let size = 4;
        let rounds = 3;
        let single_qubit_probability = 1f64;
        let two_qubit_probability = 1f64;
        let observable_location = vec![2];
        // results should be independent of rng used.
        let mut rng = thread_rng();
        let connectivity = ConnectivityLayout::Line(size);
        let circuit = random_circuit_with_set_observable(
            size,
            rounds,
            single_qubit_probability,
            two_qubit_probability,
            observable_location,
            &mut rng,
            connectivity,
        );
        let ref_legs = [
            Tensor::new_from_const(vec![0, 1], 2),
            Tensor::new_from_const(vec![3, 4, 2, 0], 2),
            Tensor::new_from_const(vec![2, 1, 5, 6], 2),
            Tensor::new_from_const(vec![8, 9, 4, 7], 2),
            Tensor::new_from_const(vec![6, 7, 10, 11], 2),
            Tensor::new_from_const(vec![12, 3], 2),
            Tensor::new_from_const(vec![5, 13], 2),
            Tensor::new_from_const(vec![14, 8], 2),
            Tensor::new_from_const(vec![10, 15], 2),
            Tensor::new_from_const(vec![16, 9], 2),
            Tensor::new_from_const(vec![11, 17], 2),
            Tensor::new_from_const(vec![19, 20, 18, 12], 2),
            Tensor::new_from_const(vec![18, 13, 21, 22], 2),
            Tensor::new_from_const(vec![23, 24, 20, 14], 2),
            Tensor::new_from_const(vec![22, 15, 25, 26], 2),
            Tensor::new_from_const(vec![27, 28, 24, 16], 2),
            Tensor::new_from_const(vec![26, 17, 29, 30], 2),
            Tensor::new_from_const(vec![31, 19], 2),
            Tensor::new_from_const(vec![21, 32], 2),
            Tensor::new_from_const(vec![33, 23], 2),
            Tensor::new_from_const(vec![25, 34], 2),
            Tensor::new_from_const(vec![35, 27], 2),
            Tensor::new_from_const(vec![29, 36], 2),
            Tensor::new_from_const(vec![37, 28], 2),
            Tensor::new_from_const(vec![30, 38], 2),
            Tensor::new_from_const(vec![31], 2),
            Tensor::new_from_const(vec![32], 2),
            Tensor::new_from_const(vec![33], 2),
            Tensor::new_from_const(vec![34], 2),
            Tensor::new_from_const(vec![35], 2),
            Tensor::new_from_const(vec![36], 2),
            Tensor::new_from_const(vec![37], 2),
            Tensor::new_from_const(vec![38], 2),
        ];

        assert_eq!(circuit.tensors().len(), 33);
        for (tensor, ref_tensor) in zip(circuit.tensors(), ref_legs) {
            assert_eq!(tensor.legs(), ref_tensor.legs());
            assert_eq!(tensor.bond_dims(), ref_tensor.bond_dims());
        }
    }
}
