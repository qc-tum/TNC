use std::f64::consts::PI;

use rand::{seq::SliceRandom, Rng};
use rustc_hash::FxHashMap;

use crate::{
    networks::connectivity::{sycamore_a, sycamore_b, sycamore_c, sycamore_d},
    random::tensorgeneration::random_sparse_tensor_data_with_rng,
    tensornetwork::{tensor::Tensor, tensordata::TensorData},
};

macro_rules! fsim {
    ($a:expr, $b:expr, $c:expr) => {
        $crate::tensornetwork::tensordata::TensorData::Gate((
            String::from("fsim"),
            vec![$a, $b],
            $c,
        ))
    };
}

pub fn sycamore_circuit<R>(qubits: usize, depth: usize, rng: &mut R) -> Tensor
where
    R: ?Sized + Rng,
{
    let mut rounds = [
        sycamore_a, sycamore_b, sycamore_c, sycamore_d, sycamore_c, sycamore_d, sycamore_a,
        sycamore_b,
    ]
    .iter()
    .cycle();
    let single_qubit_gates = [
        TensorData::Gate((String::from("sx"), Vec::new(), false)),
        TensorData::Gate((String::from("sy"), Vec::new(), false)),
        TensorData::Gate((String::from("sz"), Vec::new(), false)),
    ];

    // Initialize tensornetwork of size `usize`
    let mut circuit_tn = Tensor::default();
    let mut next_edge = qubits;
    let mut open_edges = FxHashMap::default();

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
    for _ in 0..depth {
        for i in 0..qubits {
            // Placing of random single qubit gate
            let mut new_tensor = Tensor::new_from_const(vec![open_edges[&i], next_edge], 2);
            new_tensor.set_tensor_data(single_qubit_gates.choose(rng).unwrap().clone());
            intermediate_gates.push(new_tensor);
            open_edges.insert(i, next_edge);
            next_edge += 1;
        }

        let layer = rounds.next().unwrap()();
        for (i, j) in layer {
            if i > qubits || j > qubits {
                continue;
            }
            let i = i - 1;
            let j = j - 1;
            let mut new_tensor = Tensor::new_from_const(
                vec![open_edges[&i], open_edges[&j], next_edge, next_edge + 1],
                2,
            );
            new_tensor.set_tensor_data(fsim!(PI / 2., PI / 6., false));
            intermediate_gates.push(new_tensor);
            open_edges.insert(i, next_edge);
            open_edges.insert(j, next_edge + 1);
            next_edge += 2;
        }
    }

    for i in 0..qubits {
        // Placing of random single qubit gate
        let mut new_tensor = Tensor::new_from_const(vec![open_edges[&i], next_edge], 2);
        new_tensor.set_tensor_data(single_qubit_gates.choose(rng).unwrap().clone());
        intermediate_gates.push(new_tensor);
        open_edges.insert(i, next_edge);
        next_edge += 1;
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
