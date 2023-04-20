#![allow(nonstandard_style)]
#![allow(clippy::all, clippy::restriction, clippy::pedantic, clippy::nursery)]
// Generated from Qasm2Parser.g4 by ANTLR 4.8
use antlr_rust::tree::{ParseTreeVisitor,ParseTreeVisitorCompat};
use super::qasm2parser::*;

/**
 * This interface defines a complete generic visitor for a parse tree produced
 * by {@link Qasm2Parser}.
 */
pub trait Qasm2ParserVisitor<'input>: ParseTreeVisitor<'input,Qasm2ParserContextType>{
	/**
	 * Visit a parse tree produced by {@link Qasm2Parser#program}.
	 * @param ctx the parse tree
	 */
	fn visit_program(&mut self, ctx: &ProgramContext<'input>) { self.visit_children(ctx) }

	/**
	 * Visit a parse tree produced by {@link Qasm2Parser#version}.
	 * @param ctx the parse tree
	 */
	fn visit_version(&mut self, ctx: &VersionContext<'input>) { self.visit_children(ctx) }

	/**
	 * Visit a parse tree produced by {@link Qasm2Parser#statement}.
	 * @param ctx the parse tree
	 */
	fn visit_statement(&mut self, ctx: &StatementContext<'input>) { self.visit_children(ctx) }

	/**
	 * Visit a parse tree produced by {@link Qasm2Parser#includeStatement}.
	 * @param ctx the parse tree
	 */
	fn visit_includeStatement(&mut self, ctx: &IncludeStatementContext<'input>) { self.visit_children(ctx) }

	/**
	 * Visit a parse tree produced by {@link Qasm2Parser#declaration}.
	 * @param ctx the parse tree
	 */
	fn visit_declaration(&mut self, ctx: &DeclarationContext<'input>) { self.visit_children(ctx) }

	/**
	 * Visit a parse tree produced by {@link Qasm2Parser#gateDeclaration}.
	 * @param ctx the parse tree
	 */
	fn visit_gateDeclaration(&mut self, ctx: &GateDeclarationContext<'input>) { self.visit_children(ctx) }

	/**
	 * Visit a parse tree produced by {@link Qasm2Parser#quantumOperation}.
	 * @param ctx the parse tree
	 */
	fn visit_quantumOperation(&mut self, ctx: &QuantumOperationContext<'input>) { self.visit_children(ctx) }

	/**
	 * Visit a parse tree produced by {@link Qasm2Parser#ifStatement}.
	 * @param ctx the parse tree
	 */
	fn visit_ifStatement(&mut self, ctx: &IfStatementContext<'input>) { self.visit_children(ctx) }

	/**
	 * Visit a parse tree produced by {@link Qasm2Parser#barrier}.
	 * @param ctx the parse tree
	 */
	fn visit_barrier(&mut self, ctx: &BarrierContext<'input>) { self.visit_children(ctx) }

	/**
	 * Visit a parse tree produced by {@link Qasm2Parser#bodyStatement}.
	 * @param ctx the parse tree
	 */
	fn visit_bodyStatement(&mut self, ctx: &BodyStatementContext<'input>) { self.visit_children(ctx) }

	/**
	 * Visit a parse tree produced by {@link Qasm2Parser#gateCall}.
	 * @param ctx the parse tree
	 */
	fn visit_gateCall(&mut self, ctx: &GateCallContext<'input>) { self.visit_children(ctx) }

	/**
	 * Visit a parse tree produced by {@link Qasm2Parser#idlist}.
	 * @param ctx the parse tree
	 */
	fn visit_idlist(&mut self, ctx: &IdlistContext<'input>) { self.visit_children(ctx) }

	/**
	 * Visit a parse tree produced by {@link Qasm2Parser#mixedlist}.
	 * @param ctx the parse tree
	 */
	fn visit_mixedlist(&mut self, ctx: &MixedlistContext<'input>) { self.visit_children(ctx) }

	/**
	 * Visit a parse tree produced by {@link Qasm2Parser#argument}.
	 * @param ctx the parse tree
	 */
	fn visit_argument(&mut self, ctx: &ArgumentContext<'input>) { self.visit_children(ctx) }

	/**
	 * Visit a parse tree produced by {@link Qasm2Parser#designator}.
	 * @param ctx the parse tree
	 */
	fn visit_designator(&mut self, ctx: &DesignatorContext<'input>) { self.visit_children(ctx) }

	/**
	 * Visit a parse tree produced by {@link Qasm2Parser#explist}.
	 * @param ctx the parse tree
	 */
	fn visit_explist(&mut self, ctx: &ExplistContext<'input>) { self.visit_children(ctx) }

	/**
	 * Visit a parse tree produced by the {@code bitwiseXorExpression}
	 * labeled alternative in {@link Qasm2Parser#exp}.
	 * @param ctx the parse tree
	 */
	fn visit_bitwiseXorExpression(&mut self, ctx: &BitwiseXorExpressionContext<'input>) { self.visit_children(ctx) }

	/**
	 * Visit a parse tree produced by the {@code additiveExpression}
	 * labeled alternative in {@link Qasm2Parser#exp}.
	 * @param ctx the parse tree
	 */
	fn visit_additiveExpression(&mut self, ctx: &AdditiveExpressionContext<'input>) { self.visit_children(ctx) }

	/**
	 * Visit a parse tree produced by the {@code parenthesisExpression}
	 * labeled alternative in {@link Qasm2Parser#exp}.
	 * @param ctx the parse tree
	 */
	fn visit_parenthesisExpression(&mut self, ctx: &ParenthesisExpressionContext<'input>) { self.visit_children(ctx) }

	/**
	 * Visit a parse tree produced by the {@code multiplicativeExpression}
	 * labeled alternative in {@link Qasm2Parser#exp}.
	 * @param ctx the parse tree
	 */
	fn visit_multiplicativeExpression(&mut self, ctx: &MultiplicativeExpressionContext<'input>) { self.visit_children(ctx) }

	/**
	 * Visit a parse tree produced by the {@code unaryExpression}
	 * labeled alternative in {@link Qasm2Parser#exp}.
	 * @param ctx the parse tree
	 */
	fn visit_unaryExpression(&mut self, ctx: &UnaryExpressionContext<'input>) { self.visit_children(ctx) }

	/**
	 * Visit a parse tree produced by the {@code literalExpression}
	 * labeled alternative in {@link Qasm2Parser#exp}.
	 * @param ctx the parse tree
	 */
	fn visit_literalExpression(&mut self, ctx: &LiteralExpressionContext<'input>) { self.visit_children(ctx) }

	/**
	 * Visit a parse tree produced by the {@code functionExpression}
	 * labeled alternative in {@link Qasm2Parser#exp}.
	 * @param ctx the parse tree
	 */
	fn visit_functionExpression(&mut self, ctx: &FunctionExpressionContext<'input>) { self.visit_children(ctx) }

}

pub trait Qasm2ParserVisitorCompat<'input>:ParseTreeVisitorCompat<'input, Node= Qasm2ParserContextType>{
	/**
	 * Visit a parse tree produced by {@link Qasm2Parser#program}.
	 * @param ctx the parse tree
	 */
		fn visit_program(&mut self, ctx: &ProgramContext<'input>) -> Self::Return {
			self.visit_children(ctx)
		}

	/**
	 * Visit a parse tree produced by {@link Qasm2Parser#version}.
	 * @param ctx the parse tree
	 */
		fn visit_version(&mut self, ctx: &VersionContext<'input>) -> Self::Return {
			self.visit_children(ctx)
		}

	/**
	 * Visit a parse tree produced by {@link Qasm2Parser#statement}.
	 * @param ctx the parse tree
	 */
		fn visit_statement(&mut self, ctx: &StatementContext<'input>) -> Self::Return {
			self.visit_children(ctx)
		}

	/**
	 * Visit a parse tree produced by {@link Qasm2Parser#includeStatement}.
	 * @param ctx the parse tree
	 */
		fn visit_includeStatement(&mut self, ctx: &IncludeStatementContext<'input>) -> Self::Return {
			self.visit_children(ctx)
		}

	/**
	 * Visit a parse tree produced by {@link Qasm2Parser#declaration}.
	 * @param ctx the parse tree
	 */
		fn visit_declaration(&mut self, ctx: &DeclarationContext<'input>) -> Self::Return {
			self.visit_children(ctx)
		}

	/**
	 * Visit a parse tree produced by {@link Qasm2Parser#gateDeclaration}.
	 * @param ctx the parse tree
	 */
		fn visit_gateDeclaration(&mut self, ctx: &GateDeclarationContext<'input>) -> Self::Return {
			self.visit_children(ctx)
		}

	/**
	 * Visit a parse tree produced by {@link Qasm2Parser#quantumOperation}.
	 * @param ctx the parse tree
	 */
		fn visit_quantumOperation(&mut self, ctx: &QuantumOperationContext<'input>) -> Self::Return {
			self.visit_children(ctx)
		}

	/**
	 * Visit a parse tree produced by {@link Qasm2Parser#ifStatement}.
	 * @param ctx the parse tree
	 */
		fn visit_ifStatement(&mut self, ctx: &IfStatementContext<'input>) -> Self::Return {
			self.visit_children(ctx)
		}

	/**
	 * Visit a parse tree produced by {@link Qasm2Parser#barrier}.
	 * @param ctx the parse tree
	 */
		fn visit_barrier(&mut self, ctx: &BarrierContext<'input>) -> Self::Return {
			self.visit_children(ctx)
		}

	/**
	 * Visit a parse tree produced by {@link Qasm2Parser#bodyStatement}.
	 * @param ctx the parse tree
	 */
		fn visit_bodyStatement(&mut self, ctx: &BodyStatementContext<'input>) -> Self::Return {
			self.visit_children(ctx)
		}

	/**
	 * Visit a parse tree produced by {@link Qasm2Parser#gateCall}.
	 * @param ctx the parse tree
	 */
		fn visit_gateCall(&mut self, ctx: &GateCallContext<'input>) -> Self::Return {
			self.visit_children(ctx)
		}

	/**
	 * Visit a parse tree produced by {@link Qasm2Parser#idlist}.
	 * @param ctx the parse tree
	 */
		fn visit_idlist(&mut self, ctx: &IdlistContext<'input>) -> Self::Return {
			self.visit_children(ctx)
		}

	/**
	 * Visit a parse tree produced by {@link Qasm2Parser#mixedlist}.
	 * @param ctx the parse tree
	 */
		fn visit_mixedlist(&mut self, ctx: &MixedlistContext<'input>) -> Self::Return {
			self.visit_children(ctx)
		}

	/**
	 * Visit a parse tree produced by {@link Qasm2Parser#argument}.
	 * @param ctx the parse tree
	 */
		fn visit_argument(&mut self, ctx: &ArgumentContext<'input>) -> Self::Return {
			self.visit_children(ctx)
		}

	/**
	 * Visit a parse tree produced by {@link Qasm2Parser#designator}.
	 * @param ctx the parse tree
	 */
		fn visit_designator(&mut self, ctx: &DesignatorContext<'input>) -> Self::Return {
			self.visit_children(ctx)
		}

	/**
	 * Visit a parse tree produced by {@link Qasm2Parser#explist}.
	 * @param ctx the parse tree
	 */
		fn visit_explist(&mut self, ctx: &ExplistContext<'input>) -> Self::Return {
			self.visit_children(ctx)
		}

	/**
	 * Visit a parse tree produced by the {@code bitwiseXorExpression}
	 * labeled alternative in {@link Qasm2Parser#exp}.
	 * @param ctx the parse tree
	 */
		fn visit_bitwiseXorExpression(&mut self, ctx: &BitwiseXorExpressionContext<'input>) -> Self::Return {
			self.visit_children(ctx)
		}

	/**
	 * Visit a parse tree produced by the {@code additiveExpression}
	 * labeled alternative in {@link Qasm2Parser#exp}.
	 * @param ctx the parse tree
	 */
		fn visit_additiveExpression(&mut self, ctx: &AdditiveExpressionContext<'input>) -> Self::Return {
			self.visit_children(ctx)
		}

	/**
	 * Visit a parse tree produced by the {@code parenthesisExpression}
	 * labeled alternative in {@link Qasm2Parser#exp}.
	 * @param ctx the parse tree
	 */
		fn visit_parenthesisExpression(&mut self, ctx: &ParenthesisExpressionContext<'input>) -> Self::Return {
			self.visit_children(ctx)
		}

	/**
	 * Visit a parse tree produced by the {@code multiplicativeExpression}
	 * labeled alternative in {@link Qasm2Parser#exp}.
	 * @param ctx the parse tree
	 */
		fn visit_multiplicativeExpression(&mut self, ctx: &MultiplicativeExpressionContext<'input>) -> Self::Return {
			self.visit_children(ctx)
		}

	/**
	 * Visit a parse tree produced by the {@code unaryExpression}
	 * labeled alternative in {@link Qasm2Parser#exp}.
	 * @param ctx the parse tree
	 */
		fn visit_unaryExpression(&mut self, ctx: &UnaryExpressionContext<'input>) -> Self::Return {
			self.visit_children(ctx)
		}

	/**
	 * Visit a parse tree produced by the {@code literalExpression}
	 * labeled alternative in {@link Qasm2Parser#exp}.
	 * @param ctx the parse tree
	 */
		fn visit_literalExpression(&mut self, ctx: &LiteralExpressionContext<'input>) -> Self::Return {
			self.visit_children(ctx)
		}

	/**
	 * Visit a parse tree produced by the {@code functionExpression}
	 * labeled alternative in {@link Qasm2Parser#exp}.
	 * @param ctx the parse tree
	 */
		fn visit_functionExpression(&mut self, ctx: &FunctionExpressionContext<'input>) -> Self::Return {
			self.visit_children(ctx)
		}

}

impl<'input,T> Qasm2ParserVisitor<'input> for T
where
	T: Qasm2ParserVisitorCompat<'input>
{
	fn visit_program(&mut self, ctx: &ProgramContext<'input>){
		let result = <Self as Qasm2ParserVisitorCompat>::visit_program(self, ctx);
        *<Self as ParseTreeVisitorCompat>::temp_result(self) = result;
	}

	fn visit_version(&mut self, ctx: &VersionContext<'input>){
		let result = <Self as Qasm2ParserVisitorCompat>::visit_version(self, ctx);
        *<Self as ParseTreeVisitorCompat>::temp_result(self) = result;
	}

	fn visit_statement(&mut self, ctx: &StatementContext<'input>){
		let result = <Self as Qasm2ParserVisitorCompat>::visit_statement(self, ctx);
        *<Self as ParseTreeVisitorCompat>::temp_result(self) = result;
	}

	fn visit_includeStatement(&mut self, ctx: &IncludeStatementContext<'input>){
		let result = <Self as Qasm2ParserVisitorCompat>::visit_includeStatement(self, ctx);
        *<Self as ParseTreeVisitorCompat>::temp_result(self) = result;
	}

	fn visit_declaration(&mut self, ctx: &DeclarationContext<'input>){
		let result = <Self as Qasm2ParserVisitorCompat>::visit_declaration(self, ctx);
        *<Self as ParseTreeVisitorCompat>::temp_result(self) = result;
	}

	fn visit_gateDeclaration(&mut self, ctx: &GateDeclarationContext<'input>){
		let result = <Self as Qasm2ParserVisitorCompat>::visit_gateDeclaration(self, ctx);
        *<Self as ParseTreeVisitorCompat>::temp_result(self) = result;
	}

	fn visit_quantumOperation(&mut self, ctx: &QuantumOperationContext<'input>){
		let result = <Self as Qasm2ParserVisitorCompat>::visit_quantumOperation(self, ctx);
        *<Self as ParseTreeVisitorCompat>::temp_result(self) = result;
	}

	fn visit_ifStatement(&mut self, ctx: &IfStatementContext<'input>){
		let result = <Self as Qasm2ParserVisitorCompat>::visit_ifStatement(self, ctx);
        *<Self as ParseTreeVisitorCompat>::temp_result(self) = result;
	}

	fn visit_barrier(&mut self, ctx: &BarrierContext<'input>){
		let result = <Self as Qasm2ParserVisitorCompat>::visit_barrier(self, ctx);
        *<Self as ParseTreeVisitorCompat>::temp_result(self) = result;
	}

	fn visit_bodyStatement(&mut self, ctx: &BodyStatementContext<'input>){
		let result = <Self as Qasm2ParserVisitorCompat>::visit_bodyStatement(self, ctx);
        *<Self as ParseTreeVisitorCompat>::temp_result(self) = result;
	}

	fn visit_gateCall(&mut self, ctx: &GateCallContext<'input>){
		let result = <Self as Qasm2ParserVisitorCompat>::visit_gateCall(self, ctx);
        *<Self as ParseTreeVisitorCompat>::temp_result(self) = result;
	}

	fn visit_idlist(&mut self, ctx: &IdlistContext<'input>){
		let result = <Self as Qasm2ParserVisitorCompat>::visit_idlist(self, ctx);
        *<Self as ParseTreeVisitorCompat>::temp_result(self) = result;
	}

	fn visit_mixedlist(&mut self, ctx: &MixedlistContext<'input>){
		let result = <Self as Qasm2ParserVisitorCompat>::visit_mixedlist(self, ctx);
        *<Self as ParseTreeVisitorCompat>::temp_result(self) = result;
	}

	fn visit_argument(&mut self, ctx: &ArgumentContext<'input>){
		let result = <Self as Qasm2ParserVisitorCompat>::visit_argument(self, ctx);
        *<Self as ParseTreeVisitorCompat>::temp_result(self) = result;
	}

	fn visit_designator(&mut self, ctx: &DesignatorContext<'input>){
		let result = <Self as Qasm2ParserVisitorCompat>::visit_designator(self, ctx);
        *<Self as ParseTreeVisitorCompat>::temp_result(self) = result;
	}

	fn visit_explist(&mut self, ctx: &ExplistContext<'input>){
		let result = <Self as Qasm2ParserVisitorCompat>::visit_explist(self, ctx);
        *<Self as ParseTreeVisitorCompat>::temp_result(self) = result;
	}

	fn visit_bitwiseXorExpression(&mut self, ctx: &BitwiseXorExpressionContext<'input>){
		let result = <Self as Qasm2ParserVisitorCompat>::visit_bitwiseXorExpression(self, ctx);
        *<Self as ParseTreeVisitorCompat>::temp_result(self) = result;
	}

	fn visit_additiveExpression(&mut self, ctx: &AdditiveExpressionContext<'input>){
		let result = <Self as Qasm2ParserVisitorCompat>::visit_additiveExpression(self, ctx);
        *<Self as ParseTreeVisitorCompat>::temp_result(self) = result;
	}

	fn visit_parenthesisExpression(&mut self, ctx: &ParenthesisExpressionContext<'input>){
		let result = <Self as Qasm2ParserVisitorCompat>::visit_parenthesisExpression(self, ctx);
        *<Self as ParseTreeVisitorCompat>::temp_result(self) = result;
	}

	fn visit_multiplicativeExpression(&mut self, ctx: &MultiplicativeExpressionContext<'input>){
		let result = <Self as Qasm2ParserVisitorCompat>::visit_multiplicativeExpression(self, ctx);
        *<Self as ParseTreeVisitorCompat>::temp_result(self) = result;
	}

	fn visit_unaryExpression(&mut self, ctx: &UnaryExpressionContext<'input>){
		let result = <Self as Qasm2ParserVisitorCompat>::visit_unaryExpression(self, ctx);
        *<Self as ParseTreeVisitorCompat>::temp_result(self) = result;
	}

	fn visit_literalExpression(&mut self, ctx: &LiteralExpressionContext<'input>){
		let result = <Self as Qasm2ParserVisitorCompat>::visit_literalExpression(self, ctx);
        *<Self as ParseTreeVisitorCompat>::temp_result(self) = result;
	}

	fn visit_functionExpression(&mut self, ctx: &FunctionExpressionContext<'input>){
		let result = <Self as Qasm2ParserVisitorCompat>::visit_functionExpression(self, ctx);
        *<Self as ParseTreeVisitorCompat>::temp_result(self) = result;
	}

}