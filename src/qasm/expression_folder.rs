use super::{
    ast::{Statement, Visitor},
    expression_simplification::simplify,
};

struct ExpressionFolder;

impl Visitor for ExpressionFolder {
    fn visit_program(&mut self, program: &mut super::ast::Program) {
        for statement in program.statements.iter_mut() {
            self.visit_statement(statement);
        }
    }

    fn visit_expression(&mut self, expression: &mut super::ast::Expr) {
        simplify(expression);
    }

    fn visit_statement(&mut self, statement: &mut super::ast::Statement) {
        match statement {
            Statement::GateDeclaration {
                name: _,
                params: _,
                qubits: _,
                body,
            } => {
                if let Some(body) = body {
                    for statement in body.iter_mut() {
                        self.visit_statement(statement);
                    }
                }
            }
            Statement::GateCall { name, args, qargs } => {
                for expr in args.iter_mut() {
                    self.visit_expression(expr);
                }
            }
            _ => (),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::qasm::ast::{BinOp, Expr, Statement, UnOp, Visitor};

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
        let mut gc = Statement::GateCall {
            name: String::from("abc"),
            args: vec![x, y],
            qargs: Vec::new(),
        };

        let mut visitor = ExpressionFolder {};
        visitor.visit_statement(&mut gc);

        // Check modified AST
        if let Statement::GateCall { name, args, qargs } = gc {
            assert_eq!(args.len(), 2);
            assert_eq!(args[0], Expr::Float((2 + 3) as f64 * 4.0));
            assert_eq!(args[1], Expr::Int(-2 + 2));
        } else {
            panic!("Expected a gate call");
        }
    }
}
