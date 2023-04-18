use std::collections::HashMap;

use itertools::Itertools;

use crate::tensornetwork::{tensor::Tensor, TensorNetwork};

use super::ast::{Argument, Program, Statement};

type EdgeId = i32;

#[derive(Debug, Default)]
/// Struct to create a tensor network from an QASM2 AST.
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
    fn broadcast(qargs: &[Argument], register_sizes: &HashMap<String, u32>) -> Vec<Vec<Argument>> {
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
            let (min, max) = if let itertools::MinMaxResult::OneElement(&x) = sizes {
                (x, x)
            } else if let itertools::MinMaxResult::MinMax(&x, &y) = sizes {
                (x, y)
            } else {
                unreachable!()
            };

            if min != max {
                panic!("Broadcast of registers with different sizes is not possible");
            }

            // All registers have the same size
            // -> They are zipped together
            let mut out = Vec::with_capacity(max as usize);
            for i in 0..max {
                let actual_qargs: Vec<Argument> = qargs
                    .iter()
                    .map(|arg| Argument(arg.0.clone(), arg.1.or_else(|| Some(i))))
                    .collect();
                out.push(actual_qargs);
            }
            out
        }
    }

    /// Creates a tensor network from the AST. Assumes that all gate calls
    /// have been inlined and all expressions have been simplified to literals.
    pub fn create_tensornetwork(&mut self, program: &Program) -> TensorNetwork {
        // Map qubits to the last open edge on the corresponding wire
        let mut wires = HashMap::new();
        let mut register_sizes = HashMap::new();
        let mut tensors = Vec::new();

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
                            let tensor = Tensor::new(vec![edge]);
                            tensors.push(tensor);
                            wires.insert(Argument(name.clone(), Some(i)), edge);
                        }
                        register_sizes.insert(name.clone(), *count);
                    }
                }
                Statement::GateCall(call) => {
                    // TODO: broadcast calls, where e.g. a CX is applied to two registers
                    if call.name == "U" {
                        for single_call in Self::broadcast(&call.qargs, &register_sizes) {
                            let open_edge = wires.get_mut(&single_call[0]).unwrap();
                            let out_edge = self.new_edge();
                            let tensor = Tensor::new(vec![out_edge, *open_edge]);
                            tensors.push(tensor);
                            *open_edge = out_edge;
                        }
                    } else if call.name == "CX" {
                        for single_call in Self::broadcast(&call.qargs, &register_sizes) {
                            let [open_edge1, open_edge2] = wires
                                .get_many_mut([&single_call[0], &single_call[1]])
                                .unwrap();
                            let out_edge1 = self.new_edge();
                            let out_edge2 = self.new_edge();
                            let tensor =
                                Tensor::new(vec![out_edge1, out_edge2, *open_edge1, *open_edge2]);
                            tensors.push(tensor);
                            *open_edge1 = out_edge1;
                            *open_edge2 = out_edge2;
                        }
                    } else {
                        panic!("Non-builtin gate call encountered");
                    }
                }
                _ => (),
            }
        }

        let bond_dims = vec![2u64; self.edge_counter as usize];
        TensorNetwork::from_vector(tensors, bond_dims, None)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::qasm::ast::Argument;

    use super::TensorNetworkCreator;

    #[test]
    fn broadcasting_2qargs() {
        let mut register_sizes = HashMap::new();
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
        let both_broadcast_args = &[a.clone(), b.clone()];

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
        assert_eq!(both_broadcast_calls[0], vec![a0.clone(), b0.clone()]);
        assert_eq!(both_broadcast_calls[1], vec![a1.clone(), b1.clone()]);
        assert_eq!(both_broadcast_calls[2], vec![a2.clone(), b2.clone()]);
    }

    #[test]
    fn broadcasting_1qarg() {
        let mut register_sizes = HashMap::new();
        register_sizes.insert(String::from("a"), 2);

        let a = Argument(String::from("a"), None);
        let a0 = Argument(String::from("a"), Some(0));
        let a1 = Argument(String::from("a"), Some(1));

        let no_broadcast_args = &[a1.clone()];
        let broadcast_args = &[a.clone()];

        let no_broadcast_calls =
            TensorNetworkCreator::broadcast(no_broadcast_args, &register_sizes);
        assert_eq!(no_broadcast_calls.len(), 1);
        assert_eq!(no_broadcast_calls[0], no_broadcast_args);

        let broadcast_calls = TensorNetworkCreator::broadcast(broadcast_args, &register_sizes);
        assert_eq!(broadcast_calls.len(), 2);
        assert_eq!(broadcast_calls[0], vec![a0.clone()]);
        assert_eq!(broadcast_calls[1], vec![a1.clone()]);
    }
}
