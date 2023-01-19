use super::{
    ast::{BodyStatement, GCall, QOperation, Statement, Visitor},
    expression_simplification::simplify,
};

struct ExpressionFolder;

impl ExpressionFolder {
    fn simplify_gatecall(gcall: &mut Box<GCall>) {
        // GCall is the only place where expressions occur
        for expr in gcall.args.iter_mut() {
            simplify(expr);
        }
    }
}

impl Visitor for ExpressionFolder {
    fn visit_program(&mut self, program: &mut super::ast::Program){
        for statement in program.statements.iter_mut() {
            self.visit_statement(statement);
        }
    }

    fn visit_statement(&mut self, statement: &mut super::ast::Statement){
        match statement {
            Statement::GateDeclaration {
                name: _,
                params: _,
                qubits: _,
                body,
            } => {
                if let Some(body) = body {
                    for bstatement in body.iter_mut() {
                        self.visit_body_statement(bstatement);
                    }
                }
            }
            Statement::QuantumOperation(qop) => self.visit_qoperation(qop),
            _ => (),
        }
    }

    fn visit_body_statement(&mut self, statement: &mut super::ast::BodyStatement){
        match statement {
            BodyStatement::GateCall(gcall) => ExpressionFolder::simplify_gatecall(gcall),
            _ => (),
        }
    }

    fn visit_qoperation(&mut self, qoperation: &mut super::ast::QOperation){
        match qoperation {
            QOperation::GateCall(gcall) => ExpressionFolder::simplify_gatecall(gcall),
            _ => (),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::qasm::{
        ast::{BinOp, Expr, GCall, QOperation, UnOp, Visitor},
        utils::cast,
    };

    use super::ExpressionFolder;

    #[test]
    fn simplify_gatecall() {
        // Argument 1
        let x = Box::new(Expr::Binary(
            BinOp::Mul,
            Box::new(Expr::Binary(
                BinOp::Add,
                Box::new(Expr::Int(2)),
                Box::new(Expr::Int(3)),
            )),
            Box::new(Expr::Float(4.0)),
        ));

        // Argument 2
        let y = Box::new(Expr::Binary(
            BinOp::Add,
            Box::new(Expr::Unary(UnOp::Neg, Box::new(Expr::Int(2)))),
            Box::new(Expr::Int(2)),
        ));

        // Gate call with the two arguments
        let mut qop = QOperation::GateCall(Box::new(GCall {
            name: String::from("abc"),
            args: vec![x, y],
            qargs: Vec::new(),
        }));

        let mut visitor = ExpressionFolder {};
        visitor.visit_qoperation(&mut qop);

        // Check modified AST
        let gcall = cast!(qop, QOperation::GateCall);
        assert_eq!(gcall.args.len(), 2);
        assert_eq!(*gcall.args[0], Expr::Float((2 + 3) as f64 * 4.0));
        assert_eq!(*gcall.args[1], Expr::Int(-2 + 2));
    }
}
