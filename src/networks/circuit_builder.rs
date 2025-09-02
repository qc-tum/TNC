use std::f64::consts::FRAC_1_SQRT_2;

use itertools::Itertools;
use num_complex::Complex64;
use rand::Rng;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::{
    random::tensorgeneration::random_sparse_tensor_data_with_rng,
    tensornetwork::{tensor::Tensor, tensordata::TensorData},
    types::EdgeIndex,
};

/// A quantum circuit builder that constructs a tensor network representing a quantum
/// circuit.
#[derive(Debug)]
pub struct Circuit {
    /// Maps qubit indices to the last open edge on that qubit.
    open_edges: FxHashMap<usize, EdgeIndex>,
    /// The next edge to be used.
    next_edge: usize,
    /// The tensor network representing the circuit.
    tensor_network: Tensor,
}

impl Circuit {
    /// The |0> state.
    fn ket0() -> TensorData {
        TensorData::new_from_data(&[2], vec![Complex64::ONE, Complex64::ZERO], None)
    }

    /// The |1> state.
    fn ket1() -> TensorData {
        TensorData::new_from_data(&[2], vec![Complex64::ZERO, Complex64::ONE], None)
    }

    /// The |+> state.
    fn ketplus() -> TensorData {
        TensorData::new_from_data(
            &[2],
            vec![
                Complex64::new(FRAC_1_SQRT_2, 0.0),
                Complex64::new(FRAC_1_SQRT_2, 0.0),
            ],
            None,
        )
    }

    /// The |-> state.
    fn ketminus() -> TensorData {
        TensorData::new_from_data(
            &[2],
            vec![
                Complex64::new(FRAC_1_SQRT_2, 0.0),
                Complex64::new(-FRAC_1_SQRT_2, 0.0),
            ],
            None,
        )
    }

    /// Initializes a circuit with qubits in the |0> state.
    pub fn initialize_ket0(qubits: usize) -> Self {
        Self::initialize(qubits, |_| Self::ket0())
    }

    /// Initializes a circuit given a bitstring containing `0`, `1`, `+`, or `-`.
    pub fn initialize_bitstring(bitstring: &str) -> Self {
        let qubits = bitstring.len();
        Self::initialize(qubits, |i| match bitstring.chars().nth(i).unwrap() {
            '0' => Self::ket0(),
            '1' => Self::ket1(),
            '+' => Self::ketplus(),
            '-' => Self::ketminus(),
            _ => panic!("Invalid bitstring"),
        })
    }

    /// Initializes a circuit with qubits in a random product state.
    pub fn initialize_random<R>(qubits: usize, rng: &mut R) -> Self
    where
        R: ?Sized + Rng,
    {
        Self::initialize(qubits, |_| {
            random_sparse_tensor_data_with_rng(&[2], Some(1.0), rng)
        })
    }

    /// Initializes a circuit with a function that returns a qubit state for each
    /// qubit.
    fn initialize<F>(qubits: usize, mut tensor_data_getter: F) -> Self
    where
        F: FnMut(usize) -> TensorData,
    {
        let open_edges = FxHashMap::from_iter((0..qubits).map(|i| (i, i)));
        let next_edge = qubits;

        let tensors = (0..qubits)
            .map(|i| {
                let mut tensor = Tensor::new_from_const(vec![i], 2);
                tensor.set_tensor_data(tensor_data_getter(i));
                tensor
            })
            .collect_vec();
        let tensor_network = Tensor::new_composite(tensors);

        Circuit {
            open_edges,
            next_edge,
            tensor_network,
        }
    }

    /// Appends a gate to the circuit on the specified qubits.
    pub fn append_gate(&mut self, gate: TensorData, indices: &[usize]) {
        assert_eq!(
            FxHashSet::from_iter(indices).len(),
            indices.len(),
            "Qubit indices must be unique"
        );

        let old_edges = indices.iter().map(|i| self.open_edges[i]).collect_vec();
        let mut new_edges = (self.next_edge..self.next_edge + indices.len()).collect_vec();
        self.next_edge += indices.len();

        // Update the open edges
        for (i, next_edge) in indices.iter().zip(&new_edges) {
            self.open_edges.insert(*i, *next_edge);
        }

        // Create the new tensor
        let mut edges = old_edges;
        edges.append(&mut new_edges);
        let mut new_tensor = Tensor::new_from_const(edges, 2);
        new_tensor.set_tensor_data(gate);

        // Push the new tensor to the tensor network
        self.tensor_network.push_tensor(new_tensor);
    }

    /// Applies a layer of <0| to the end and returns the final tensor network.
    pub fn finalize_ket0(self) -> Tensor {
        self.finalize(|_| Self::ket0())
    }

    /// Applies a layer of qubit states given by a bitstring containing `0`, `1`, `+`
    /// , or `-` and returns the final tensor network.
    pub fn finalize_bitstring(self, bitstring: &str) -> Tensor {
        self.finalize(|i| match bitstring.chars().nth(i).unwrap() {
            '0' => Self::ket0(),
            '1' => Self::ket1(),
            '+' => Self::ketplus(),
            '-' => Self::ketminus(),
            _ => panic!("Invalid bitstring"),
        })
    }

    /// Applies a random product state layer to the end and returns the final tensor network.
    pub fn finalize_random<R>(self, rng: &mut R) -> Tensor
    where
        R: ?Sized + Rng,
    {
        self.finalize(|_| random_sparse_tensor_data_with_rng(&[2], Some(1.0), rng))
    }

    /// Applies a function that returns a qubit state for each qubit to the end and
    /// returns the final tensor network.
    fn finalize<F>(mut self, mut tensor_data_getter: F) -> Tensor
    where
        F: FnMut(usize) -> TensorData,
    {
        let qubits = self.open_edges.len();
        let tensors = (0..qubits)
            .map(|i| {
                let mut tensor = Tensor::new_from_const(vec![self.open_edges[&i]], 2);
                tensor.set_tensor_data(tensor_data_getter(i));
                tensor
            })
            .collect_vec();
        self.tensor_network.push_tensors(tensors);
        self.tensor_network
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::f64::consts::FRAC_1_SQRT_2;

    use num_complex::Complex64;

    use crate::{
        contractionpath::paths::{
            cotengrust::{Cotengrust, OptMethod},
            OptimizePath,
        },
        tensornetwork::{
            contraction::contract_tensor_network, tensor::Tensor, tensordata::TensorData,
        },
    };

    #[test]
    fn hadmards_expectation() {
        let qubits = 5;
        let mut circuit = Circuit::initialize_ket0(qubits);
        for i in 0..qubits {
            circuit.append_gate(TensorData::Gate((String::from("h"), vec![], true)), &[i]);
        }
        let tensor_network = circuit.finalize_ket0();

        let mut opt = Cotengrust::new(&tensor_network, OptMethod::Greedy);
        opt.optimize_path();
        let path = opt.get_best_replace_path();

        let result = contract_tensor_network(tensor_network, &path);

        let mut tn_ref = Tensor::default();
        tn_ref.set_tensor_data(TensorData::new_from_data(
            &[],
            vec![Complex64::new(FRAC_1_SQRT_2.powi(qubits as i32), 0.0)],
            None,
        ));

        assert!(result.approx_eq(&tn_ref, 1e-8));
    }

    #[test]
    fn superposition_expectation() {
        let qubits = 3;
        let circuit = Circuit::initialize_bitstring("+++");
        let tensor_network = circuit.finalize_ket0();

        let mut opt = Cotengrust::new(&tensor_network, OptMethod::Greedy);
        opt.optimize_path();
        let path = opt.get_best_replace_path();

        let result = contract_tensor_network(tensor_network, &path);

        let mut tn_ref = Tensor::default();
        tn_ref.set_tensor_data(TensorData::new_from_data(
            &[],
            vec![Complex64::new(FRAC_1_SQRT_2.powi(qubits as i32), 0.0)],
            None,
        ));

        assert!(result.approx_eq(&tn_ref, 1e-8));
    }

    #[test]
    #[should_panic(expected = "Qubit indices must be unique")]
    fn duplicate_qubit_arg() {
        let mut circuit = Circuit::initialize_ket0(2);
        circuit.append_gate(
            TensorData::Gate((String::from("cx"), vec![], true)),
            &[1, 1],
        );
    }
}
