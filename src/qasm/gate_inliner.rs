use std::{collections::HashMap, iter::zip, mem::take};

use super::{
    ast::{Expr, GateCallData, GateDeclarationData, Program, Statement},
    utils::cast,
};

#[derive(Debug, Default)]
pub struct GateInliner {
    definitions: HashMap<String, GateDeclarationData>,
}

impl GateInliner {
    // Traverses an expression and replaces all variables with the expressions given by the context.
    fn replace_vars(expr: &mut Expr, context: &HashMap<&String, &Expr>) {
        match expr {
            Expr::Variable(x) => {
                *expr = (**context.get(x).expect("Unknown variable name in gate call")).clone()
            }
            Expr::Unary(_, inner) => GateInliner::replace_vars(inner, context),
            Expr::Binary(_, lhs, rhs) => {
                GateInliner::replace_vars(lhs, context);
                GateInliner::replace_vars(rhs, context)
            }
            Expr::Function(_, inner) => GateInliner::replace_vars(inner, context),
            _ => (),
        }
    }

    /// Returns a copy of the body of callee with the specific data filled in from the gate call.
    fn get_body(&self, call: &GateCallData, callee: &GateDeclarationData) -> Vec<Statement> {
        if let Some(body) = &callee.body {
            // Map the names in the declaration to the actual values passed in the call
            let name_to_expr: HashMap<_, _> = zip(&callee.params, &call.args).collect();
            let name_to_qreg: HashMap<_, _> = zip(&callee.qubits, &call.qargs).collect();

            let mut statements = Vec::with_capacity(body.len());
            for statement in body {
                let call = cast!(statement, Statement::GateCall);

                // Ensure that all calls of the gates to be inlined are already inlined
                assert!(call.name == "U" || call.name == "CX");

                // Replace any used variables by their actual values given by the caller
                let mut new_args = Vec::with_capacity(call.args.len());
                for arg in call.args.iter() {
                    let mut arg = arg.clone();
                    GateInliner::replace_vars(&mut arg, &name_to_expr);
                    new_args.push(arg);
                }

                // Replace any used qubit by the actual qreg reference
                let mut new_qargs = Vec::with_capacity(call.qargs.len());
                for qarg in call.qargs.iter() {
                    let global_name = *name_to_qreg
                        .get(&qarg.0)
                        .expect("Unknown variable name in gate call");
                    new_qargs.push(global_name.clone());
                }

                statements.push(Statement::GateCall(GateCallData {
                    name: call.name.clone(),
                    args: new_args,
                    qargs: new_qargs,
                }));
            }
            statements
        } else {
            panic!("Unable to inline opaque gate {}", callee.name);
        }
    }

    /// Returns the body of the called gate, with all parameters filled in.
    ///
    /// # Panics
    /// Panics if the called gate was not yet defined.
    fn get_inlined_body(&self, call: &GateCallData) -> Vec<Statement> {
        let callee = self.definitions.get(&call.name);
        if let Some(callee) = callee {
            self.get_body(call, callee)
        } else {
            panic!("Call to gate {} which was not defined yet", call.name);
        }
    }

    /// Takes a list of statements and inlines every gate call with the corresponding
    /// gate definition. Gate definitions are removed and added to the internal list
    /// of known gates.
    fn inline(&mut self, statements: &mut Vec<Statement>) {
        enum Change {
            Remove(usize),
            Replace(usize, Vec<Statement>),
        }

        let mut changes = Vec::new();
        for (i, statement) in statements.iter_mut().enumerate() {
            match statement {
                Statement::GateDeclaration(data) => {
                    // Take ownership of gate body from AST node
                    let mut data = take(data);

                    // Inline all gate calls inside the gate body
                    if let Some(body) = data.body.as_mut() {
                        self.inline(body);
                    }

                    // Store the gate and mark statement as to be removed
                    self.register_gate(data);
                    changes.push(Change::Remove(i));
                }
                Statement::GateCall(call) => {
                    if !call.is_builtin() {
                        // Get the inlined body and mark the call as to be replaced
                        let body = self.get_inlined_body(call);
                        changes.push(Change::Replace(i, body));
                    }
                }
                _ => (),
            }
        }

        // Since we are done iterating, apply all changes.
        // Changes are applied in reverse, as the indices would otherwise become wrong.
        for change in changes.into_iter().rev() {
            match change {
                Change::Remove(idx) => {
                    statements.remove(idx);
                }
                Change::Replace(idx, body) => {
                    statements.splice(idx..idx + 1, body);
                }
            };
        }
    }

    /// Registers the gate as known.
    fn register_gate(&mut self, gate: GateDeclarationData) {
        self.definitions.insert(gate.name.clone(), gate);
    }

    /// Inlines all gate calls in the program.
    pub fn inline_program(&mut self, program: &mut Program) {
        self.inline(&mut program.statements);
    }
}

#[cfg(test)]
mod tests {
    use crate::qasm::ast::{Argument, BinOp, Expr, Program, Statement};

    use super::GateInliner;

    #[test]
    fn inline() {
        // INPUT:
        // gate x (a, b) q {
        //   U(a + b, 0, b) q;
        // }
        // gate y (a) q1, q2 {
        //   x(a, 0) q2;
        //   U(1, 2, 3) q1;
        //   x(1, a) q1;
        // }
        // y(2) q[0], q[1];

        // RESULT:
        // U(2 + 0, 0, 0) q[1];
        // U(1, 2, 3) q[0];
        // U(1 + 2, 0, 2) q[0];

        let mut program = Program {
            statements: vec![
                Statement::gate_declaration(
                    "x",
                    vec![String::from("a"), String::from("b")],
                    vec![String::from("q")],
                    Some(vec![Statement::gate_call(
                        "U",
                        vec![
                            Expr::Binary(
                                BinOp::Add,
                                Box::new(Expr::Variable(String::from("a"))),
                                Box::new(Expr::Variable(String::from("b"))),
                            ),
                            Expr::Int(0),
                            Expr::Variable(String::from("b")),
                        ],
                        vec![Argument(String::from("q"), None)],
                    )]),
                ),
                Statement::gate_declaration(
                    "y",
                    vec![String::from("a")],
                    vec![String::from("q1"), String::from("q2")],
                    Some(vec![
                        Statement::gate_call(
                            "x",
                            vec![Expr::Variable(String::from("a")), Expr::Int(0)],
                            vec![Argument(String::from("q2"), None)],
                        ),
                        Statement::gate_call(
                            "U",
                            vec![Expr::Int(1), Expr::Int(2), Expr::Int(3)],
                            vec![Argument(String::from("q1"), None)],
                        ),
                        Statement::gate_call(
                            "x",
                            vec![Expr::Int(1), Expr::Variable(String::from("a"))],
                            vec![Argument(String::from("q1"), None)],
                        ),
                    ]),
                ),
                Statement::gate_call(
                    "y",
                    vec![Expr::Int(2)],
                    vec![
                        Argument(String::from("q"), Some(0)),
                        Argument(String::from("q"), Some(1)),
                    ],
                ),
            ],
        };

        let mut inliner = GateInliner::default();
        inliner.inline_program(&mut program);

        assert_eq!(
            program,
            Program {
                statements: vec![
                    Statement::gate_call(
                        "U",
                        vec![
                            Expr::Binary(
                                BinOp::Add,
                                Box::new(Expr::Int(2)),
                                Box::new(Expr::Int(0))
                            ),
                            Expr::Int(0),
                            Expr::Int(0)
                        ],
                        vec![Argument(String::from("q"), Some(1))]
                    ),
                    Statement::gate_call(
                        "U",
                        vec![Expr::Int(1), Expr::Int(2), Expr::Int(3)],
                        vec![Argument(String::from("q"), Some(0))]
                    ),
                    Statement::gate_call(
                        "U",
                        vec![
                            Expr::Binary(
                                BinOp::Add,
                                Box::new(Expr::Int(1)),
                                Box::new(Expr::Int(2))
                            ),
                            Expr::Int(0),
                            Expr::Int(2)
                        ],
                        vec![Argument(String::from("q"), Some(0))]
                    )
                ]
            }
        );
    }
}
