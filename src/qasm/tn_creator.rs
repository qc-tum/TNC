use itertools::Itertools;
use num_complex::Complex64;
use rustc_hash::FxHashMap;

use crate::tensornetwork::{create_tensor_network, tensor::Tensor};

use super::ast::{Argument, Program, Statement};
use crate::tensornetwork::tensordata::TensorData;

type EdgeId = usize;

/// Struct to create a tensor network from an QASM2 AST.
#[derive(Debug, Default)]
pub struct TensorNetworkCreator {
    edge_counter: EdgeId,
}

impl TensorNetworkCreator {
    /// Gets a new edge id.
    fn new_edge(&mut self) -> EdgeId {
        let res = self.edge_counter;
        self.edge_counter += 1;
        res
    }

    /// Given the quantum arguments to a gate call, applies the broadcast rules and
    /// returns the list of quantum arguments for each single call.
    fn broadcast(
        qargs: &[Argument],
        register_sizes: &FxHashMap<String, u32>,
    ) -> Vec<Vec<Argument>> {
        // Get the size of all register arguments (i.e. those without qubit index specified)
        let sizes = qargs
            .iter()
            .filter(|arg| arg.1.is_none())
            .map(|arg| register_sizes.get(&arg.0).unwrap())
            .minmax();

        if sizes == itertools::MinMaxResult::NoElements {
            // Empty iterator
            // -> No registers, only single qubit args
            // -> No broadcasting needed
            vec![qargs.to_vec()]
        } else {
            let common_size = match sizes {
                itertools::MinMaxResult::OneElement(&x) => x,
                itertools::MinMaxResult::MinMax(&min, &max) => {
                    assert_eq!(
                        min, max,
                        "Broadcast of registers with different sizes is not possible"
                    );
                    min
                }
                itertools::MinMaxResult::NoElements => unreachable!(),
            };

            // All registers have the same size
            // -> They are zipped together
            let mut out = Vec::with_capacity(common_size as usize);
            for i in 0..common_size {
                let actual_qargs: Vec<Argument> = qargs
                    .iter()
                    .map(|arg| Argument(arg.0.clone(), arg.1.or(Some(i))))
                    .collect();
                out.push(actual_qargs);
            }
            out
        }
    }

    /// Creates a |0> state vector.
    fn ket0() -> tetra::Tensor {
        let mut out = tetra::Tensor::new(&[2]);
        out.set(&[0], Complex64::new(1.0, 0.0));
        out
    }

    /// Creates a tensor network from the AST. Assumes that all gate calls
    /// have been inlined and all expressions have been simplified to literals.
    pub fn create_tensornetwork(&mut self, program: &Program) -> Tensor {
        // Map qubits to the last open edge on the corresponding wire
        let mut wires = FxHashMap::default();
        let mut register_sizes = FxHashMap::default();
        let mut tensors = Vec::new();
        let ket0 = Self::ket0();

        for statement in &program.statements {
            match statement {
                Statement::Declaration {
                    is_quantum,
                    name,
                    count,
                } => {
                    if *is_quantum {
                        // New wires are initialized with |0>
                        // Thus, create new tensors with a single edge for each qubit
                        tensors.reserve(*count as usize);
                        for i in 0..*count {
                            let edge = self.new_edge();
                            let mut tensor = Tensor::new(vec![edge]);
                            tensor.set_tensor_data(TensorData::Matrix(ket0.clone()));
                            tensors.push(tensor);
                            wires.insert(Argument(name.clone(), Some(i)), edge);
                        }
                        register_sizes.insert(name.clone(), *count);
                    }
                }
                Statement::GateCall(call) => {
                    // Convert arg expressions to actual numbers
                    let args = call
                        .args
                        .iter()
                        .map(|arg| arg.try_into().unwrap())
                        .collect_vec();

                    for single_call in Self::broadcast(&call.qargs, &register_sizes) {
                        let mut open_edges = Vec::with_capacity(single_call.len());
                        let mut out_edges = Vec::with_capacity(2 * single_call.len());

                        // Get all input legs and create new output legs
                        for wire in &single_call {
                            let open_edge = wires.get_mut(wire).unwrap();
                            open_edges.push(*open_edge);
                            let out_edge = self.new_edge();
                            out_edges.push(out_edge);
                            *open_edge = out_edge;
                        }

                        // Create the tensor
                        open_edges.reverse();
                        out_edges.append(&mut open_edges);
                        let mut tensor = Tensor::new(out_edges);
                        tensor.set_tensor_data(TensorData::Gate((
                            call.name.to_ascii_lowercase(),
                            args.clone(),
                        )));
                        tensors.push(tensor);
                    }
                }
                _ => (),
            }
        }

        let bond_dims = (0..self.edge_counter)
            .map(|e| (e, 2u64))
            .collect::<FxHashMap<_, _>>();

        create_tensor_network(tensors, &bond_dims)
    }
}

#[cfg(test)]
mod tests {
    use rustc_hash::FxHashMap;

    use crate::qasm::ast::Argument;

    use super::TensorNetworkCreator;

    #[test]
    fn broadcasting_2qargs() {
        let mut register_sizes = FxHashMap::default();
        register_sizes.insert(String::from("a"), 3);
        register_sizes.insert(String::from("b"), 3);

        let a = Argument(String::from("a"), None);
        let a0 = Argument(String::from("a"), Some(0));
        let a1 = Argument(String::from("a"), Some(1));
        let a2 = Argument(String::from("a"), Some(2));
        let b = Argument(String::from("b"), None);
        let b0 = Argument(String::from("b"), Some(0));
        let b1 = Argument(String::from("b"), Some(1));
        let b2 = Argument(String::from("b"), Some(2));

        let no_broadcast_args = &[a2.clone(), b0.clone()];
        let a_broadcast_args = &[a.clone(), b0.clone()];
        let b_broadcast_args = &[a1.clone(), b.clone()];
        let both_broadcast_args = &[a, b];

        let no_broadcast_calls =
            TensorNetworkCreator::broadcast(no_broadcast_args, &register_sizes);
        assert_eq!(no_broadcast_calls.len(), 1);
        assert_eq!(no_broadcast_calls[0], no_broadcast_args);

        let a_broadcast_calls = TensorNetworkCreator::broadcast(a_broadcast_args, &register_sizes);
        assert_eq!(a_broadcast_calls.len(), 3);
        assert_eq!(a_broadcast_calls[0], vec![a0.clone(), b0.clone()]);
        assert_eq!(a_broadcast_calls[1], vec![a1.clone(), b0.clone()]);
        assert_eq!(a_broadcast_calls[2], vec![a2.clone(), b0.clone()]);

        let b_broadcast_calls = TensorNetworkCreator::broadcast(b_broadcast_args, &register_sizes);
        assert_eq!(b_broadcast_calls.len(), 3);
        assert_eq!(b_broadcast_calls[0], vec![a1.clone(), b0.clone()]);
        assert_eq!(b_broadcast_calls[1], vec![a1.clone(), b1.clone()]);
        assert_eq!(b_broadcast_calls[2], vec![a1.clone(), b2.clone()]);

        let both_broadcast_calls =
            TensorNetworkCreator::broadcast(both_broadcast_args, &register_sizes);
        assert_eq!(both_broadcast_calls.len(), 3);
        assert_eq!(both_broadcast_calls[0], vec![a0, b0]);
        assert_eq!(both_broadcast_calls[1], vec![a1, b1]);
        assert_eq!(both_broadcast_calls[2], vec![a2, b2]);
    }

    #[test]
    fn broadcasting_1qarg() {
        let mut register_sizes = FxHashMap::default();
        register_sizes.insert(String::from("a"), 2);

        let a = Argument(String::from("a"), None);
        let a0 = Argument(String::from("a"), Some(0));
        let a1 = Argument(String::from("a"), Some(1));

        let no_broadcast_args = &[a1.clone()];
        let broadcast_args = &[a];

        let no_broadcast_calls =
            TensorNetworkCreator::broadcast(no_broadcast_args, &register_sizes);
        assert_eq!(no_broadcast_calls.len(), 1);
        assert_eq!(no_broadcast_calls[0], no_broadcast_args);

        let broadcast_calls = TensorNetworkCreator::broadcast(broadcast_args, &register_sizes);
        assert_eq!(broadcast_calls.len(), 2);
        assert_eq!(broadcast_calls[0], vec![a0]);
        assert_eq!(broadcast_calls[1], vec![a1]);
    }
}
