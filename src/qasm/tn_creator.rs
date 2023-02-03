use std::collections::HashMap;

use crate::tensornetwork::{tensor::Tensor, TensorNetwork};

use super::ast::{Argument, Program, Statement};

type EdgeId = i32;

#[derive(Debug, Default)]
pub struct TensorNetworkCreator {
    edge_counter: EdgeId,
}

impl TensorNetworkCreator {
    fn new_edge(&mut self) -> EdgeId {
        let res = self.edge_counter;
        self.edge_counter += 1;
        res
    }

    /// Creates a tensor network from the AST. Assumes that all gate calls
    /// have been inlined and all expressions have been simplified to literals.
    pub fn create_tensornetwork(&mut self, program: &Program) -> TensorNetwork {
        // Map qubits to the last open edge on the corresponding wire
        let mut wires = HashMap::new();
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
                            tensors.push(Tensor::new(vec![edge]));
                            wires.insert(Argument(name.clone(), Some(i)), edge);
                        }
                    }
                }
                Statement::GateCall(call) => {
                    // TODO: broadcast calls, where e.g. a CX is applied to two registers
                    if call.name == "U" {
                        let open_edge = wires.get_mut(&call.qargs[0]).unwrap();
                        let out_edge = self.new_edge();
                        tensors.push(Tensor::new(vec![out_edge, *open_edge]));
                        *open_edge = out_edge;
                    } else if call.name == "CX" {
                        let [open_edge1, open_edge2] = wires
                            .get_many_mut([&call.qargs[0], &call.qargs[1]])
                            .unwrap();
                        let out_edge1 = self.new_edge();
                        let out_edge2 = self.new_edge();
                        tensors.push(Tensor::new(vec![
                            out_edge1,
                            out_edge2,
                            *open_edge1,
                            *open_edge2,
                        ]));
                        *open_edge1 = out_edge1;
                        *open_edge2 = out_edge2;
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
