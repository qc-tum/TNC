use antlr_rust::common_token_stream::CommonTokenStream;
use antlr_rust::tree::{ParseTree, ParseTreeVisitorCompat, Visitable};
use antlr_rust::InputStream;

use super::ast::{Argument, BinOp, Expr, FuncType, Program, Statement, UnOp};
use super::qasm2lexer::Qasm2Lexer;
use super::qasm2parser::{
    AdditiveExpressionContextAttrs, ArgumentContextAttrs, BitwiseXorExpressionContextAttrs,
    BodyStatementContextAttrs, DeclarationContextAttrs, DesignatorContextAttrs,
    ExplistContextAttrs, FunctionExpressionContextAttrs, GateCallContextAttrs,
    GateDeclarationContextAttrs, IdlistContextAttrs, MixedlistContextAttrs,
    MultiplicativeExpressionContextAttrs, ParenthesisExpressionContextAttrs, ProgramContextAttrs,
    Qasm2Parser, QuantumOperationContextAttrs, StatementContextAttrs, UnaryExpressionContextAttrs,
};
use super::qasm2parser::{LiteralExpressionContextAttrs, Qasm2ParserContextType};
use super::qasm2parservisitor::Qasm2ParserVisitorCompat;
use super::utils::cast;

#[derive(Debug)]
enum ReturnVal {
    None,
    Int(u32),
    Arg(Argument),
    ArgList(Vec<Argument>),
    IdentifierList(Vec<String>),
    Expression(Box<Expr>),
    ExpressionList(Vec<Expr>),
    Statement(Box<Statement>),
    Program(Box<Program>),
}

impl Default for ReturnVal {
    fn default() -> Self {
        Self::None
    }
}

#[derive(Debug, Default)]
struct MyVisitor {
    tmp: ReturnVal,
}

impl ParseTreeVisitorCompat<'_> for MyVisitor {
    type Node = Qasm2ParserContextType;
    type Return = ReturnVal;

    fn temp_result(&mut self) -> &mut Self::Return {
        &mut self.tmp
    }
}

impl Qasm2ParserVisitorCompat<'_> for MyVisitor {
    fn visit_program(&mut self, ctx: &super::qasm2parser::ProgramContext) -> Self::Return {
        let mut statements = Vec::new();
        while let Some(sctx) = ctx.statement(statements.len()) {
            let statement = self.visit(&*sctx);
            let statement = cast!(statement, ReturnVal::Statement);
            statements.push(*statement);
        }
        ReturnVal::Program(Box::new(Program { statements }))
    }

    fn visit_statement(&mut self, ctx: &super::qasm2parser::StatementContext) -> Self::Return {
        if let Some(qctx) = ctx.quantumOperation() {
            self.visit(&*qctx)
        } else if let Some(gctx) = ctx.declaration() {
            self.visit(&*gctx)
        } else if let Some(gctx) = ctx.gateDeclaration() {
            self.visit(&*gctx)
        } else if ctx.includeStatement().is_some() {
            panic!("Include statements should have been resolved already")
        } else if ctx.ifStatement().is_some() {
            panic!("If statements are not supported")
        } else if ctx.barrier().is_some() {
            panic!("Barrier statements are not supported")
        } else {
            panic!("Unhandled statement")
        }
    }

    fn visit_declaration(&mut self, ctx: &super::qasm2parser::DeclarationContext) -> Self::Return {
        let is_quantum = ctx.QREG().is_some();
        let name = ctx.Identifier().unwrap().get_text();
        let dctx = ctx
            .designator()
            .expect("Declaration must specify register size");
        let count = self.visit(&*dctx);
        let count = cast!(count, ReturnVal::Int);
        ReturnVal::Statement(Box::new(Statement::Declaration {
            is_quantum,
            name,
            count,
        }))
    }

    fn visit_gateDeclaration(
        &mut self,
        ctx: &super::qasm2parser::GateDeclarationContext,
    ) -> Self::Return {
        let is_opaque = ctx.OPAQUE().is_some();
        let name = ctx.Identifier().unwrap().get_text();

        if is_opaque {
            panic!("Opaque gates are not supported");
        }

        // Parse params. If not given, return empty vector
        let params = ctx.params.as_ref().map_or_else(Vec::new, |ctx| {
            cast!(self.visit(&**ctx), ReturnVal::IdentifierList)
        });

        // Parse qubit params (must be given)
        let qubits = self.visit(&**ctx.qubits.as_ref().unwrap());
        let qubits = cast!(qubits, ReturnVal::IdentifierList);

        let body = if is_opaque {
            None
        } else {
            let mut out = Vec::new();
            while let Some(bctx) = ctx.bodyStatement(out.len()) {
                let statement = self.visit(&*bctx);
                let statement = cast!(statement, ReturnVal::Statement);
                out.push(*statement);
            }
            Some(out)
        };

        ReturnVal::Statement(Box::new(Statement::GateDeclaration {
            name,
            params,
            qubits,
            body,
        }))
    }

    fn visit_quantumOperation(
        &mut self,
        ctx: &super::qasm2parser::QuantumOperationContext,
    ) -> Self::Return {
        if let Some(gctx) = ctx.gateCall() {
            self.visit(&*gctx)
        } else if ctx.MEASURE().is_some() || ctx.RESET().is_some() {
            panic!("Measure and Reset are not supported");
        } else {
            panic!("Unhandled quantum operation!");
        }
    }

    fn visit_bodyStatement(
        &mut self,
        ctx: &super::qasm2parser::BodyStatementContext,
    ) -> Self::Return {
        if let Some(gctx) = ctx.gateCall() {
            self.visit(&*gctx)
        } else {
            panic!("Barrier is not supported")
        }
    }

    fn visit_gateCall(&mut self, ctx: &super::qasm2parser::GateCallContext) -> Self::Return {
        let name = if let Some(id) = ctx.Identifier() {
            id.get_text()
        } else if ctx.U().is_some() {
            String::from("U")
        } else if ctx.CX().is_some() {
            String::from("CX")
        } else {
            panic!("Unhandled gate call variant")
        };

        // Parse parameters
        let args = ctx.explist().map_or_else(Vec::new, |ectx| {
            cast!(self.visit(&*ectx), ReturnVal::ExpressionList)
        });

        // Parse qubit params
        let qargs = self.visit(&*ctx.mixedlist().unwrap());
        let qargs = cast!(qargs, ReturnVal::ArgList);

        ReturnVal::Statement(Box::new(Statement::GateCall { name, args, qargs }))
    }

    fn visit_idlist(&mut self, ctx: &super::qasm2parser::IdlistContext) -> Self::Return {
        let mut identifiers = Vec::new();
        while let Some(ictx) = ctx.Identifier(identifiers.len()) {
            identifiers.push(ictx.get_text());
        }
        ReturnVal::IdentifierList(identifiers)
    }

    fn visit_mixedlist(&mut self, ctx: &super::qasm2parser::MixedlistContext) -> Self::Return {
        let mut arguments = Vec::new();
        while let Some(actx) = ctx.argument(arguments.len()) {
            let arg = self.visit(&*actx);
            let arg = cast!(arg, ReturnVal::Arg);
            arguments.push(arg);
        }
        ReturnVal::ArgList(arguments)
    }

    fn visit_argument(&mut self, ctx: &super::qasm2parser::ArgumentContext) -> Self::Return {
        let name = ctx.Identifier().unwrap().get_text();
        let count = ctx
            .designator()
            .map(|dctx| cast!(self.visit(&*dctx), ReturnVal::Int));
        ReturnVal::Arg(Argument(name, count))
    }

    fn visit_designator(&mut self, ctx: &super::qasm2parser::DesignatorContext) -> Self::Return {
        let number = ctx.Integer().unwrap().get_text();
        let number = number.parse().unwrap();
        ReturnVal::Int(number)
    }

    fn visit_explist(&mut self, ctx: &super::qasm2parser::ExplistContext) -> Self::Return {
        let mut expressions = Vec::new();
        while let Some(ectx) = ctx.exp(expressions.len()) {
            let expr = self.visit(&*ectx);
            let expr = cast!(expr, ReturnVal::Expression);
            expressions.push(*expr);
        }
        ReturnVal::ExpressionList(expressions)
    }

    fn visit_parenthesisExpression(
        &mut self,
        ctx: &super::qasm2parser::ParenthesisExpressionContext,
    ) -> Self::Return {
        self.visit(&*ctx.exp().unwrap())
    }

    fn visit_additiveExpression(
        &mut self,
        ctx: &super::qasm2parser::AdditiveExpressionContext,
    ) -> Self::Return {
        let lhs = self.visit(&**ctx.lhs.as_ref().unwrap());
        let lhs = cast!(lhs, ReturnVal::Expression);
        let rhs = self.visit(&**ctx.rhs.as_ref().unwrap());
        let rhs = cast!(rhs, ReturnVal::Expression);
        let op = if ctx.PLUS().is_some() {
            BinOp::Add
        } else if ctx.MINUS().is_some() {
            BinOp::Sub
        } else {
            panic!("Unhandled operator");
        };
        ReturnVal::Expression(Box::new(Expr::Binary(op, lhs, rhs)))
    }

    fn visit_multiplicativeExpression(
        &mut self,
        ctx: &super::qasm2parser::MultiplicativeExpressionContext,
    ) -> Self::Return {
        let lhs = self.visit(&**ctx.lhs.as_ref().unwrap());
        let lhs = cast!(lhs, ReturnVal::Expression);
        let rhs = self.visit(&**ctx.rhs.as_ref().unwrap());
        let rhs = cast!(rhs, ReturnVal::Expression);
        let op = if ctx.ASTERISK().is_some() {
            BinOp::Mul
        } else if ctx.SLASH().is_some() {
            BinOp::Div
        } else {
            panic!("Unhandled operator");
        };
        ReturnVal::Expression(Box::new(Expr::Binary(op, lhs, rhs)))
    }

    fn visit_bitwiseXorExpression(
        &mut self,
        ctx: &super::qasm2parser::BitwiseXorExpressionContext,
    ) -> Self::Return {
        let lhs = self.visit(&**ctx.lhs.as_ref().unwrap());
        let lhs = cast!(lhs, ReturnVal::Expression);
        let rhs = self.visit(&**ctx.rhs.as_ref().unwrap());
        let rhs = cast!(rhs, ReturnVal::Expression);
        let op = if ctx.CARET().is_some() {
            BinOp::BitXor
        } else {
            panic!("Unhandled operator");
        };
        ReturnVal::Expression(Box::new(Expr::Binary(op, lhs, rhs)))
    }

    fn visit_unaryExpression(
        &mut self,
        ctx: &super::qasm2parser::UnaryExpressionContext,
    ) -> Self::Return {
        let inner = self.visit(&*ctx.exp().unwrap());
        let inner = cast!(inner, ReturnVal::Expression);
        ReturnVal::Expression(Box::new(Expr::Unary(UnOp::Neg, inner)))
    }

    fn visit_functionExpression(
        &mut self,
        ctx: &super::qasm2parser::FunctionExpressionContext,
    ) -> Self::Return {
        let inner = self.visit(&*ctx.exp().unwrap());
        let inner = cast!(inner, ReturnVal::Expression);
        let kind = if ctx.SIN().is_some() {
            FuncType::Sin
        } else if ctx.COS().is_some() {
            FuncType::Cos
        } else if ctx.TAN().is_some() {
            FuncType::Tan
        } else if ctx.EXP().is_some() {
            FuncType::Exp
        } else if ctx.LN().is_some() {
            FuncType::Ln
        } else if ctx.SQRT().is_some() {
            FuncType::Sqrt
        } else {
            panic!("Unhandled function type");
        };
        ReturnVal::Expression(Box::new(Expr::Function(kind, inner)))
    }

    fn visit_literalExpression(
        &mut self,
        ctx: &super::qasm2parser::LiteralExpressionContext,
    ) -> Self::Return {
        let expr = if let Some(val) = ctx.Float() {
            Expr::Float(val.get_text().parse::<f64>().unwrap())
        } else if let Some(val) = ctx.Integer() {
            Expr::Int(val.get_text().parse::<i32>().unwrap())
        } else if ctx.PI().is_some() {
            Expr::Float(std::f64::consts::PI)
        } else if let Some(val) = ctx.Identifier() {
            Expr::Variable(val.get_text())
        } else {
            panic!("Unhandled literal");
        };
        ReturnVal::Expression(Box::new(expr))
    }
}

fn parse(code: &str) {
    let lexer = Qasm2Lexer::new(InputStream::new(code));
    let token_source = CommonTokenStream::new(lexer);
    let mut parser = Qasm2Parser::new(token_source);
    let parsed = parser.program().unwrap();

    let mut visitor = MyVisitor::default();
    parsed.accept(&mut visitor);
    println!("{visitor:#?}");
}

#[cfg(test)]
mod tests {
    use std::fs;

    use crate::qasm::include_resolver::expand_includes;

    use super::parse;

    #[test]
    fn it_works() {
        let mut code = fs::read_to_string("test.qasm").unwrap();
        expand_includes(&mut code);
        parse(&code);
    }
}
