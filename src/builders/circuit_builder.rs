//! Building a tensor network from a quantum circuit.

use std::marker::PhantomData;

use itertools::Itertools;
use num_complex::Complex64;
use permutation::Permutation;

use crate::tensornetwork::{
    tensor::{EdgeIndex, Tensor},
    tensordata::TensorData,
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
    pub fn qubit(&self, index: usize) -> Qubit<'_> {
        assert!(index < self.size);
        Qubit {
            index: self.base + index,
            phantom: PhantomData,
        }
    }

    /// Returns an iterator over all qubits in this register.
    pub fn qubits(&self) -> impl Iterator<Item = Qubit<'_>> {
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

/// A struct holding a permutation to be applied to a tensor.
#[derive(Debug, Clone)]
pub struct Permutor {
    target_leg_order: Vec<EdgeIndex>,
}

impl Permutor {
    fn new(target_legs: Vec<EdgeIndex>) -> Self {
        Self {
            target_leg_order: target_legs,
        }
    }

    /// Permutates the tensor according to the stored permutation.
    pub fn apply(&self, tensor: Tensor) -> Tensor {
        assert!(tensor.is_leaf());
        if self.is_empty() {
            return tensor;
        }

        let Tensor {
            mut legs,
            mut bond_dims,
            tensordata,
            ..
        } = tensor;
        let mut data = tensordata.into_data();

        // Find the permutation
        let mut perm = Self::permutation_between(&legs, &self.target_leg_order);

        // Permute legs, shape and data
        perm.apply_slice_in_place(&mut legs);
        perm.apply_slice_in_place(&mut bond_dims);
        data.transpose(&perm);

        Tensor {
            tensors: vec![],
            legs,
            bond_dims,
            tensordata: TensorData::Matrix(data),
        }
    }

    /// Returns whether the permutor is empty, in which case it won't have any
    /// effect.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.target_leg_order.is_empty()
    }

    /// Computes the permutation that, when applied to `given`, returns `target`.
    /// Assumes that `given` and `target` are equal up to permutation.
    fn permutation_between(given: &[usize], target: &[usize]) -> Permutation {
        let given_to_sorted = permutation::sort_unstable(given);
        let target_to_sorted = permutation::sort_unstable(target);
        &target_to_sorted.inverse() * &given_to_sorted
    }
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

    /// The Z gate.
    fn z() -> TensorData {
        TensorData::Gate((String::from("z"), vec![], false))
    }

    /// Creates a new edge id.
    fn new_edge(&mut self) -> usize {
        let edge = self.next_edge;
        self.next_edge += 1;
        edge
    }

    /// Returns the total number of qubits allocated in this circuit.
    ///
    /// # Examples
    /// ```
    /// # use tnc::builders::circuit_builder::Circuit;
    /// let mut circuit = Circuit::default();
    /// let q1 = circuit.allocate_register(2);
    /// let q2 = circuit.allocate_register(3);
    /// assert_eq!(circuit.num_qubits(), 5);
    /// ```
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
        let edges = old_edges.chain(new_edges).collect_vec();
        self.next_edge += indices.len();

        // Update the open edges
        for (q, next_edge) in indices.iter().zip(&edges[indices.len()..]) {
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
    ///
    /// The bitstring can also contain wildcards `*`, in which case the tensor leg
    /// corresponding to this qubit is left open. For every wildcard, the output
    /// tensor will be doubled in size. In the extreme case where there's only
    /// wildcards, the full statevector will be computed.
    ///
    /// Since the final tensor can end up with arbitrary permutation, a [`Permutor`]
    /// is returned that can transpose the final tensor after contraction to the
    /// natural order, i.e., sorted by increasing qubit number. If the bitstring
    /// contains no wildcards, the final result is a scalar and the permutator can be
    /// ignored.
    pub fn into_amplitude_network(mut self, bitstring: &str) -> (Tensor, Permutor) {
        assert_eq!(bitstring.len(), self.num_qubits());

        // Apply the final bras
        self.tensors.reserve(bitstring.len());
        let mut final_legs = Vec::new();
        for (c, e) in bitstring.chars().zip(self.open_edges) {
            let bra = match c {
                '0' => Self::ket0(),
                '1' => Self::ket1(),
                '*' => {
                    final_legs.push(e);
                    continue; // leave this edge open
                }
                _ => panic!("Only 0, 1 and * are allowed in bitstring"),
            };
            let mut tensor = Tensor::new_from_const(vec![e], 2);
            tensor.set_tensor_data(bra);
            self.tensors.push(tensor);
        }

        // The contraction can re-order the legs depending on the contraction order,
        // so we need to return the order we want the legs to be in at the end, such
        // that the user can transpose the final tensor and has the expected order of
        // elements.
        let out = Tensor::new_composite(self.tensors);
        let permutor = Permutor::new(final_legs);
        (out, permutor)
    }

    /// Converts the circuit to a tensor network that computes the full statevector.
    ///
    /// Since the final tensor can end up with arbitrary permutation, a [`Permutor`]
    /// is returned that can transpose the final tensor after contraction to the
    /// natural order, i.e., sorted by increasing qubit number.
    #[inline]
    pub fn into_statevector_network(self) -> (Tensor, Permutor) {
        let qubits = self.num_qubits();
        self.into_amplitude_network(&"*".repeat(qubits))
    }

    /// Creates the adjoint tensor of a given tensor. This not only modifies the
    /// data, but also the order of legs and the bond_dims vec. The legs of the new
    /// tensor are offset by `leg_offset`.
    fn tensor_adjoint(tensor: &Tensor, leg_offset: usize) -> Tensor {
        // Transpose legs and shape of tensor
        let half = tensor.legs().len() / 2;
        let legs = tensor.legs()[half..]
            .iter()
            .chain(&tensor.legs()[..half])
            .map(|l| l + leg_offset)
            .collect();
        let bond_dims = tensor.bond_dims()[half..]
            .iter()
            .chain(&tensor.bond_dims()[..half])
            .copied()
            .collect();

        // Take actual adjoint of tensor data
        let data = tensor.tensor_data().clone();
        let data = data.adjoint();

        let mut adjoint = Tensor::new(legs, bond_dims);
        adjoint.set_tensor_data(data);
        adjoint
    }

    /// Converts the circuit to a tensor network that computes the expectation value
    /// with respect to standard observables (`Z`) on all qubits.
    ///
    /// The tensor network is roughly twice the size of the circuit, as it needs to
    /// compute the adjoint of the circuit as well.
    pub fn into_expectation_value_network(mut self) -> Tensor {
        let offset = self.next_edge;
        self.tensors.reserve(self.tensors.len() + self.num_qubits());

        // Add the mirrored tensor network
        let mut adjoint_tensors = Vec::with_capacity(self.tensors.len());
        for tensor in &self.tensors {
            let adjoint = Self::tensor_adjoint(tensor, offset);
            adjoint_tensors.push(adjoint);
        }
        self.tensors.append(&mut adjoint_tensors);

        // Add the layer of observables
        for e in self.open_edges {
            let mut t = Tensor::new_from_const(vec![e, e + offset], 2);
            t.set_tensor_data(Self::z());
            self.tensors.push(t);
        }

        Tensor::new_composite(self.tensors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::f64::consts::{FRAC_1_SQRT_2, FRAC_PI_3, FRAC_PI_4};

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

    fn test_permutation_between(given: &[usize], target: &[usize]) {
        let perm = Permutor::permutation_between(given, target);
        assert_eq!(perm.apply_slice(given), target);
    }

    #[test]
    fn permutation_between() {
        test_permutation_between(&[1, 2, 3, 4], &[1, 2, 3, 4]);
        test_permutation_between(&[1, 2, 3, 4], &[4, 3, 2, 1]);
        test_permutation_between(&[4, 3, 2, 1], &[1, 2, 3, 4]);
        test_permutation_between(&[4, 1, 3, 2], &[2, 4, 3, 1]);
        test_permutation_between(&[5, 1, 4, 3, 2, 6], &[1, 6, 3, 5, 2, 4]);
    }

    #[test]
    fn hadamards_amplitude() {
        let qubits = 5;
        let mut circuit = Circuit::default();
        let qr = circuit.allocate_register(qubits);
        for q in qr.qubits() {
            circuit.append_gate(TensorData::Gate((String::from("h"), vec![], false)), &[q]);
        }
        let (tensor_network, permutor) = circuit.into_amplitude_network("00000");
        assert!(permutor.is_empty());

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
    fn rx_expectation_value() {
        let qubits = 2;
        let mut circuit = Circuit::default();
        let qr = circuit.allocate_register(qubits);
        circuit.append_gate(
            TensorData::Gate((String::from("rx"), vec![FRAC_PI_4], false)),
            &[qr.qubit(0)],
        );
        circuit.append_gate(
            TensorData::Gate((String::from("rx"), vec![FRAC_PI_3], false)),
            &[qr.qubit(1)],
        );
        let tensor_network = circuit.into_expectation_value_network();

        let mut opt = Cotengrust::new(&tensor_network, OptMethod::Greedy);
        opt.optimize_path();
        let path = opt.get_best_replace_path();

        let result = contract_tensor_network(tensor_network, &path);

        let mut tn_ref = Tensor::default();
        tn_ref.set_tensor_data(TensorData::new_from_data(
            &[],
            vec![Complex64::new(FRAC_1_SQRT_2 * 0.5, 0.0)],
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
