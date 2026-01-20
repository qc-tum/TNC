use itertools::Itertools;
use rustc_hash::FxHashMap;

use crate::builders::circuit_builder::{Circuit, QuantumRegister};
use crate::qasm::ast::{Argument, Program, Statement};
use crate::tensornetwork::tensordata::TensorData;
use crate::utils::traits::HashMapInsertNew;

/// Struct to create a circuit from a QASM2 AST.
#[derive(Debug)]
pub struct CircuitCreator;

impl CircuitCreator {
    /// Given the quantum arguments to a gate call, applies the broadcast rules and
    /// returns the list of quantum arguments for each single call.
    fn broadcast(
        qargs: &[Argument],
        registers: &FxHashMap<String, QuantumRegister<'_>>,
    ) -> Vec<Vec<Argument>> {
        // Get the size of all register arguments (i.e. those without qubit index specified)
        let sizes = qargs
            .iter()
            .filter(|arg| arg.1.is_none())
            .map(|arg| registers[arg.0.as_str()].len())
            .minmax();

        if sizes == itertools::MinMaxResult::NoElements {
            // Empty iterator
            // -> No registers, only single qubit args
            // -> No broadcasting needed
            vec![qargs.to_vec()]
        } else {
            let common_size = match sizes {
                itertools::MinMaxResult::OneElement(x) => x,
                itertools::MinMaxResult::MinMax(min, max) => {
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
            let mut out = Vec::with_capacity(common_size);
            for i in 0..common_size {
                let i = i.try_into().unwrap();
                let actual_qargs = qargs
                    .iter()
                    .map(|arg| Argument(arg.0.clone(), arg.1.or(Some(i))))
                    .collect();
                out.push(actual_qargs);
            }
            out
        }
    }

    /// Creates a circuit from the AST. Assumes that all gate calls have been inlined
    /// and all expressions have been evaluated to literals.
    pub fn create_circuit(&self, program: &Program) -> Circuit {
        let mut circuit = Circuit::default();
        let mut registers = FxHashMap::default();

        for statement in &program.statements {
            match statement {
                Statement::Declaration {
                    is_quantum,
                    name,
                    count,
                } => {
                    if *is_quantum {
                        // Allocate a new register in |0> state
                        let register = circuit.allocate_register((*count).try_into().unwrap());
                        registers.insert_new(name.to_owned(), register);
                    }
                }
                Statement::GateCall(call) => {
                    // Convert arg expressions to actual numbers
                    let args = call
                        .args
                        .iter()
                        .map(|arg| arg.try_into().unwrap())
                        .collect_vec();

                    for single_call in Self::broadcast(&call.qargs, &registers) {
                        // Translate arguments to Qubit args for the circuit
                        let qargs = single_call
                            .into_iter()
                            .map(|arg| registers[&arg.0].qubit(arg.1.unwrap().try_into().unwrap()))
                            .collect_vec();

                        // Append the gate to the circuit
                        let data =
                            TensorData::Gate((call.name.to_ascii_lowercase(), args.clone(), false));
                        circuit.append_gate(data, &qargs);
                    }
                }
                _ => (),
            }
        }

        circuit
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::slice;

    use rustc_hash::FxHashMap;

    use crate::qasm::ast::Argument;

    #[test]
    fn broadcasting_2qargs() {
        let mut registers = FxHashMap::default();
        registers.insert(String::from("a"), QuantumRegister::new(3));
        registers.insert(String::from("b"), QuantumRegister::new(3));

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

        let no_broadcast_calls = CircuitCreator::broadcast(no_broadcast_args, &registers);
        assert_eq!(no_broadcast_calls.len(), 1);
        assert_eq!(no_broadcast_calls[0], no_broadcast_args);

        let a_broadcast_calls = CircuitCreator::broadcast(a_broadcast_args, &registers);
        assert_eq!(a_broadcast_calls.len(), 3);
        assert_eq!(a_broadcast_calls[0], vec![a0.clone(), b0.clone()]);
        assert_eq!(a_broadcast_calls[1], vec![a1.clone(), b0.clone()]);
        assert_eq!(a_broadcast_calls[2], vec![a2.clone(), b0.clone()]);

        let b_broadcast_calls = CircuitCreator::broadcast(b_broadcast_args, &registers);
        assert_eq!(b_broadcast_calls.len(), 3);
        assert_eq!(b_broadcast_calls[0], vec![a1.clone(), b0.clone()]);
        assert_eq!(b_broadcast_calls[1], vec![a1.clone(), b1.clone()]);
        assert_eq!(b_broadcast_calls[2], vec![a1.clone(), b2.clone()]);

        let both_broadcast_calls = CircuitCreator::broadcast(both_broadcast_args, &registers);
        assert_eq!(both_broadcast_calls.len(), 3);
        assert_eq!(both_broadcast_calls[0], vec![a0, b0]);
        assert_eq!(both_broadcast_calls[1], vec![a1, b1]);
        assert_eq!(both_broadcast_calls[2], vec![a2, b2]);
    }

    #[test]
    fn broadcasting_1qarg() {
        let mut registers = FxHashMap::default();
        registers.insert(String::from("a"), QuantumRegister::new(2));

        let a = Argument(String::from("a"), None);
        let a0 = Argument(String::from("a"), Some(0));
        let a1 = Argument(String::from("a"), Some(1));

        let no_broadcast_args = slice::from_ref(&a1);
        let broadcast_args = slice::from_ref(&a);

        let no_broadcast_calls = CircuitCreator::broadcast(no_broadcast_args, &registers);
        assert_eq!(no_broadcast_calls.len(), 1);
        assert_eq!(no_broadcast_calls[0], no_broadcast_args);

        let broadcast_calls = CircuitCreator::broadcast(broadcast_args, &registers);
        assert_eq!(broadcast_calls.len(), 2);
        assert_eq!(broadcast_calls[0], vec![a0]);
        assert_eq!(broadcast_calls[1], vec![a1]);
    }
}
