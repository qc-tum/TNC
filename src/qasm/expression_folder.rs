use super::{ast::Visitor, expression_simplification::simplify};

#[derive(Debug, Default)]
pub struct ExpressionFolder;

impl Visitor for ExpressionFolder {
    fn visit_expression(&mut self, expression: &mut super::ast::Expr) {
        simplify(expression);
    }
}

#[cfg(test)]
mod tests {
    use crate::qasm::ast::{BinOp, Expr, FuncType, Program, Statement, UnOp, Visitor};

    use super::ExpressionFolder;

    #[test]
    fn simplify_gatecall() {
        // Argument 1
        let x = Expr::Binary(
            BinOp::Mul,
            Box::new(Expr::Binary(
                BinOp::Add,
                Box::new(Expr::Int(2)),
                Box::new(Expr::Int(3)),
            )),
            Box::new(Expr::Float(4.0)),
        );

        // Argument 2
        let y = Expr::Binary(
            BinOp::Add,
            Box::new(Expr::Unary(UnOp::Neg, Box::new(Expr::Int(2)))),
            Box::new(Expr::Int(2)),
        );

        // Gate call with the two arguments
        let mut gc = Statement::gate_call(String::from("abc"), vec![x, y], Vec::new());

        let mut visitor = ExpressionFolder::default();
        visitor.visit_statement(&mut gc);

        // Check modified AST
        if let Statement::GateCall(data) = gc {
            assert_eq!(data.args.len(), 2);
            assert_eq!(data.args[0], Expr::Float((2 + 3) as f64 * 4.0));
            assert_eq!(data.args[1], Expr::Int(-2 + 2));
        } else {
            panic!("Expected a gate call");
        }
    }

    #[test]
    fn simplify_program() {
        let mut program = Program {
            statements: vec![
                Statement::gate_call(
                    String::from("a"),
                    vec![
                        Expr::Binary(
                            BinOp::Mul,
                            Box::new(Expr::Int(3)),
                            Box::new(Expr::Float(0.5)),
                        ),
                        Expr::Binary(
                            BinOp::Div,
                            Box::new(Expr::Float(3.6)),
                            Box::new(Expr::Float(-2.4)),
                        ),
                    ],
                    Vec::new(),
                ),
                Statement::gate_call(
                    String::from("b"),
                    vec![Expr::Function(FuncType::Sqrt, Box::new(Expr::Int(4)))],
                    Vec::new(),
                ),
            ],
        };

        let mut visitor = ExpressionFolder::default();
        visitor.visit_program(&mut program);

        assert_eq!(program.statements.len(), 2);

        // Check first call
        if let Statement::GateCall(data) = &program.statements[0] {
            assert_eq!(data.args.len(), 2);
            assert_eq!(data.args[0], Expr::Float(1.5));
            assert_eq!(data.args[1], Expr::Float(3.6 / -2.4));
        } else {
            panic!("Expected a gate call");
        }

        // Check second call
        if let Statement::GateCall(data) = &program.statements[1] {
            assert_eq!(data.args.len(), 1);
            assert_eq!(data.args[0], Expr::Float(4f64.sqrt()));
        } else {
            panic!("Expected a gate call");
        }
    }
}
