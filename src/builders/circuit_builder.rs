use std::marker::PhantomData;

use itertools::Itertools;
use num_complex::Complex64;

use crate::{
    tensornetwork::{tensor::Tensor, tensordata::TensorData},
    types::EdgeIndex,
};

/// A quantum register, i.e., an array of qubits. Similar to the Qiskit / QASM
/// idea, quantum registers group qubits (for instance, one qreg for ancillas), and
/// a circuit can act on multiple qregs.
#[derive(Debug)]
pub struct QuantumRegister<'a> {
    base: usize,
    size: usize,
    phantom: PhantomData<&'a Circuit>,
}

impl QuantumRegister<'_> {
    /// Creates a new quantum register without any associated circuit. This is mainly
    /// for testing.
    #[cfg(test)]
    pub(crate) fn new(size: usize) -> Self {
        QuantumRegister {
            base: 0,
            size,
            phantom: PhantomData,
        }
    }

    /// Returns the qubit at a given index.
    pub fn qubit(&self, index: usize) -> Qubit {
        assert!(index < self.size);
        Qubit {
            index: self.base + index,
            phantom: PhantomData,
        }
    }

    /// Returns an iterator over all qubits in this register.
    pub fn qubits(&self) -> impl Iterator<Item = Qubit> {
        (self.base..self.base + self.size).map(|i| Qubit {
            index: i,
            phantom: PhantomData,
        })
    }

    /// Returns the size of the register.
    #[inline]
    pub fn len(&self) -> usize {
        self.size
    }

    /// Returns whether this register is empty, i.e., doesn't contain any qubits.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }
}

/// A single qubit from a quantum register.
pub struct Qubit<'a> {
    index: usize,
    phantom: PhantomData<&'a Circuit>,
}

/// A quantum circuit builder that constructs a tensor network representing a quantum
/// circuit.
#[derive(Debug, Default)]
pub struct Circuit {
    /// The last open edges on each qubit.
    open_edges: Vec<EdgeIndex>,
    /// The next edge to be used.
    next_edge: usize,
    /// The tensors representing the circuit.
    tensors: Vec<Tensor>,
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

    /// Creates a new edge id.
    fn new_edge(&mut self) -> usize {
        let edge = self.next_edge;
        self.next_edge += 1;
        edge
    }

    /// Returns the total number of qubits allocated in this circuit.
    #[inline]
    pub fn num_qubits(&self) -> usize {
        self.open_edges.len()
    }

    /// Allocates a new quantum register. The qubits are initialized in the |0>
    /// state.
    pub fn allocate_register<'a>(&mut self, size: usize) -> QuantumRegister<'a> {
        let previous_qubits = self.num_qubits();

        self.open_edges.reserve(size);
        self.tensors.reserve(size);
        for _ in 0..size {
            let edge = self.new_edge();
            self.open_edges.push(edge);
            let mut ket0 = Tensor::new_from_const(vec![edge], 2);
            ket0.set_tensor_data(Self::ket0());
            self.tensors.push(ket0);
        }

        QuantumRegister {
            base: previous_qubits,
            size,
            phantom: PhantomData,
        }
    }

    /// Appends a gate to the circuit on the specified qubits.
    pub fn append_gate(&mut self, gate: TensorData, indices: &[Qubit]) {
        assert!(
            indices.iter().map(|q| q.index).all_unique(),
            "Qubit arguments must be unique"
        );

        // Get the old and new edges
        let old_edges = indices.iter().map(|q| self.open_edges[q.index]);
        let new_edges = (0..indices.len()).map(|e| e + self.next_edge);
        let edges = new_edges.chain(old_edges.rev()).collect_vec();
        self.next_edge += indices.len();

        // Update the open edges
        for (q, next_edge) in indices.iter().zip(&edges[..indices.len()]) {
            self.open_edges[q.index] = *next_edge;
        }

        // Create the new tensor
        let mut new_tensor = Tensor::new_from_const(edges, 2);
        new_tensor.set_tensor_data(gate);

        // Push the new tensor to the tensors
        self.tensors.push(new_tensor);
    }

    /// Converts the circuit to a tensor network that computes the amplitude for the
    /// given bitstring.
    pub fn into_amplitude_network(mut self, bitstring: &str) -> Tensor {
        assert_eq!(bitstring.len(), self.num_qubits());

        self.tensors.reserve(bitstring.len());
        for (c, e) in bitstring.chars().zip(self.open_edges) {
            let bra = match c {
                '0' => Self::ket0(),
                '1' => Self::ket1(),
                '*' => continue, // leave this edge open
                _ => panic!("Only 0, 1 and * are allowed in bitstring"),
            };
            let mut tensor = Tensor::new_from_const(vec![e], 2);
            tensor.set_tensor_data(bra);
            self.tensors.push(tensor);
        }
        Tensor::new_composite(self.tensors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::f64::consts::FRAC_1_SQRT_2;

    use float_cmp::assert_approx_eq;
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
        let mut circuit = Circuit::default();
        let qr = circuit.allocate_register(qubits);
        for q in qr.qubits() {
            circuit.append_gate(TensorData::Gate((String::from("h"), vec![], true)), &[q]);
        }
        let tensor_network = circuit.into_amplitude_network("00000");

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

        assert_approx_eq!(&Tensor, &result, &tn_ref);
    }

    #[test]
    #[should_panic(expected = "Qubit arguments must be unique")]
    fn duplicate_qubit_arg() {
        let mut circuit = Circuit::default();
        let qr = circuit.allocate_register(2);
        circuit.append_gate(
            TensorData::Gate((String::from("cx"), vec![], true)),
            &[qr.qubit(1), qr.qubit(1)],
        );
    }
}
