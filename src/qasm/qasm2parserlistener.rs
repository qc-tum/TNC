#![allow(nonstandard_style)]
#![allow(clippy::all, clippy::restriction, clippy::pedantic, clippy::nursery)]
// Generated from Qasm2Parser.g4 by ANTLR 4.8
use super::qasm2parser::*;
use antlr_rust::tree::ParseTreeListener;

pub trait Qasm2ParserListener<'input>: ParseTreeListener<'input, Qasm2ParserContextType> {
    /**
     * Enter a parse tree produced by {@link Qasm2Parser#program}.
     * @param ctx the parse tree
     */
    fn enter_program(&mut self, _ctx: &ProgramContext<'input>) {}
    /**
     * Exit a parse tree produced by {@link Qasm2Parser#program}.
     * @param ctx the parse tree
     */
    fn exit_program(&mut self, _ctx: &ProgramContext<'input>) {}
    /**
     * Enter a parse tree produced by {@link Qasm2Parser#version}.
     * @param ctx the parse tree
     */
    fn enter_version(&mut self, _ctx: &VersionContext<'input>) {}
    /**
     * Exit a parse tree produced by {@link Qasm2Parser#version}.
     * @param ctx the parse tree
     */
    fn exit_version(&mut self, _ctx: &VersionContext<'input>) {}
    /**
     * Enter a parse tree produced by {@link Qasm2Parser#statement}.
     * @param ctx the parse tree
     */
    fn enter_statement(&mut self, _ctx: &StatementContext<'input>) {}
    /**
     * Exit a parse tree produced by {@link Qasm2Parser#statement}.
     * @param ctx the parse tree
     */
    fn exit_statement(&mut self, _ctx: &StatementContext<'input>) {}
    /**
     * Enter a parse tree produced by {@link Qasm2Parser#includeStatement}.
     * @param ctx the parse tree
     */
    fn enter_includeStatement(&mut self, _ctx: &IncludeStatementContext<'input>) {}
    /**
     * Exit a parse tree produced by {@link Qasm2Parser#includeStatement}.
     * @param ctx the parse tree
     */
    fn exit_includeStatement(&mut self, _ctx: &IncludeStatementContext<'input>) {}
    /**
     * Enter a parse tree produced by {@link Qasm2Parser#declaration}.
     * @param ctx the parse tree
     */
    fn enter_declaration(&mut self, _ctx: &DeclarationContext<'input>) {}
    /**
     * Exit a parse tree produced by {@link Qasm2Parser#declaration}.
     * @param ctx the parse tree
     */
    fn exit_declaration(&mut self, _ctx: &DeclarationContext<'input>) {}
    /**
     * Enter a parse tree produced by {@link Qasm2Parser#gateDeclaration}.
     * @param ctx the parse tree
     */
    fn enter_gateDeclaration(&mut self, _ctx: &GateDeclarationContext<'input>) {}
    /**
     * Exit a parse tree produced by {@link Qasm2Parser#gateDeclaration}.
     * @param ctx the parse tree
     */
    fn exit_gateDeclaration(&mut self, _ctx: &GateDeclarationContext<'input>) {}
    /**
     * Enter a parse tree produced by {@link Qasm2Parser#quantumOperation}.
     * @param ctx the parse tree
     */
    fn enter_quantumOperation(&mut self, _ctx: &QuantumOperationContext<'input>) {}
    /**
     * Exit a parse tree produced by {@link Qasm2Parser#quantumOperation}.
     * @param ctx the parse tree
     */
    fn exit_quantumOperation(&mut self, _ctx: &QuantumOperationContext<'input>) {}
    /**
     * Enter a parse tree produced by {@link Qasm2Parser#ifStatement}.
     * @param ctx the parse tree
     */
    fn enter_ifStatement(&mut self, _ctx: &IfStatementContext<'input>) {}
    /**
     * Exit a parse tree produced by {@link Qasm2Parser#ifStatement}.
     * @param ctx the parse tree
     */
    fn exit_ifStatement(&mut self, _ctx: &IfStatementContext<'input>) {}
    /**
     * Enter a parse tree produced by {@link Qasm2Parser#barrier}.
     * @param ctx the parse tree
     */
    fn enter_barrier(&mut self, _ctx: &BarrierContext<'input>) {}
    /**
     * Exit a parse tree produced by {@link Qasm2Parser#barrier}.
     * @param ctx the parse tree
     */
    fn exit_barrier(&mut self, _ctx: &BarrierContext<'input>) {}
    /**
     * Enter a parse tree produced by {@link Qasm2Parser#bodyStatement}.
     * @param ctx the parse tree
     */
    fn enter_bodyStatement(&mut self, _ctx: &BodyStatementContext<'input>) {}
    /**
     * Exit a parse tree produced by {@link Qasm2Parser#bodyStatement}.
     * @param ctx the parse tree
     */
    fn exit_bodyStatement(&mut self, _ctx: &BodyStatementContext<'input>) {}
    /**
     * Enter a parse tree produced by {@link Qasm2Parser#gateCall}.
     * @param ctx the parse tree
     */
    fn enter_gateCall(&mut self, _ctx: &GateCallContext<'input>) {}
    /**
     * Exit a parse tree produced by {@link Qasm2Parser#gateCall}.
     * @param ctx the parse tree
     */
    fn exit_gateCall(&mut self, _ctx: &GateCallContext<'input>) {}
    /**
     * Enter a parse tree produced by {@link Qasm2Parser#idlist}.
     * @param ctx the parse tree
     */
    fn enter_idlist(&mut self, _ctx: &IdlistContext<'input>) {}
    /**
     * Exit a parse tree produced by {@link Qasm2Parser#idlist}.
     * @param ctx the parse tree
     */
    fn exit_idlist(&mut self, _ctx: &IdlistContext<'input>) {}
    /**
     * Enter a parse tree produced by {@link Qasm2Parser#mixedlist}.
     * @param ctx the parse tree
     */
    fn enter_mixedlist(&mut self, _ctx: &MixedlistContext<'input>) {}
    /**
     * Exit a parse tree produced by {@link Qasm2Parser#mixedlist}.
     * @param ctx the parse tree
     */
    fn exit_mixedlist(&mut self, _ctx: &MixedlistContext<'input>) {}
    /**
     * Enter a parse tree produced by {@link Qasm2Parser#argument}.
     * @param ctx the parse tree
     */
    fn enter_argument(&mut self, _ctx: &ArgumentContext<'input>) {}
    /**
     * Exit a parse tree produced by {@link Qasm2Parser#argument}.
     * @param ctx the parse tree
     */
    fn exit_argument(&mut self, _ctx: &ArgumentContext<'input>) {}
    /**
     * Enter a parse tree produced by {@link Qasm2Parser#designator}.
     * @param ctx the parse tree
     */
    fn enter_designator(&mut self, _ctx: &DesignatorContext<'input>) {}
    /**
     * Exit a parse tree produced by {@link Qasm2Parser#designator}.
     * @param ctx the parse tree
     */
    fn exit_designator(&mut self, _ctx: &DesignatorContext<'input>) {}
    /**
     * Enter a parse tree produced by {@link Qasm2Parser#explist}.
     * @param ctx the parse tree
     */
    fn enter_explist(&mut self, _ctx: &ExplistContext<'input>) {}
    /**
     * Exit a parse tree produced by {@link Qasm2Parser#explist}.
     * @param ctx the parse tree
     */
    fn exit_explist(&mut self, _ctx: &ExplistContext<'input>) {}
    /**
     * Enter a parse tree produced by the {@code bitwiseXorExpression}
     * labeled alternative in {@link Qasm2Parser#exp}.
     * @param ctx the parse tree
     */
    fn enter_bitwiseXorExpression(&mut self, _ctx: &BitwiseXorExpressionContext<'input>) {}
    /**
     * Exit a parse tree produced by the {@code bitwiseXorExpression}
     * labeled alternative in {@link Qasm2Parser#exp}.
     * @param ctx the parse tree
     */
    fn exit_bitwiseXorExpression(&mut self, _ctx: &BitwiseXorExpressionContext<'input>) {}
    /**
     * Enter a parse tree produced by the {@code additiveExpression}
     * labeled alternative in {@link Qasm2Parser#exp}.
     * @param ctx the parse tree
     */
    fn enter_additiveExpression(&mut self, _ctx: &AdditiveExpressionContext<'input>) {}
    /**
     * Exit a parse tree produced by the {@code additiveExpression}
     * labeled alternative in {@link Qasm2Parser#exp}.
     * @param ctx the parse tree
     */
    fn exit_additiveExpression(&mut self, _ctx: &AdditiveExpressionContext<'input>) {}
    /**
     * Enter a parse tree produced by the {@code parenthesisExpression}
     * labeled alternative in {@link Qasm2Parser#exp}.
     * @param ctx the parse tree
     */
    fn enter_parenthesisExpression(&mut self, _ctx: &ParenthesisExpressionContext<'input>) {}
    /**
     * Exit a parse tree produced by the {@code parenthesisExpression}
     * labeled alternative in {@link Qasm2Parser#exp}.
     * @param ctx the parse tree
     */
    fn exit_parenthesisExpression(&mut self, _ctx: &ParenthesisExpressionContext<'input>) {}
    /**
     * Enter a parse tree produced by the {@code multiplicativeExpression}
     * labeled alternative in {@link Qasm2Parser#exp}.
     * @param ctx the parse tree
     */
    fn enter_multiplicativeExpression(&mut self, _ctx: &MultiplicativeExpressionContext<'input>) {}
    /**
     * Exit a parse tree produced by the {@code multiplicativeExpression}
     * labeled alternative in {@link Qasm2Parser#exp}.
     * @param ctx the parse tree
     */
    fn exit_multiplicativeExpression(&mut self, _ctx: &MultiplicativeExpressionContext<'input>) {}
    /**
     * Enter a parse tree produced by the {@code unaryExpression}
     * labeled alternative in {@link Qasm2Parser#exp}.
     * @param ctx the parse tree
     */
    fn enter_unaryExpression(&mut self, _ctx: &UnaryExpressionContext<'input>) {}
    /**
     * Exit a parse tree produced by the {@code unaryExpression}
     * labeled alternative in {@link Qasm2Parser#exp}.
     * @param ctx the parse tree
     */
    fn exit_unaryExpression(&mut self, _ctx: &UnaryExpressionContext<'input>) {}
    /**
     * Enter a parse tree produced by the {@code literalExpression}
     * labeled alternative in {@link Qasm2Parser#exp}.
     * @param ctx the parse tree
     */
    fn enter_literalExpression(&mut self, _ctx: &LiteralExpressionContext<'input>) {}
    /**
     * Exit a parse tree produced by the {@code literalExpression}
     * labeled alternative in {@link Qasm2Parser#exp}.
     * @param ctx the parse tree
     */
    fn exit_literalExpression(&mut self, _ctx: &LiteralExpressionContext<'input>) {}
    /**
     * Enter a parse tree produced by the {@code functionExpression}
     * labeled alternative in {@link Qasm2Parser#exp}.
     * @param ctx the parse tree
     */
    fn enter_functionExpression(&mut self, _ctx: &FunctionExpressionContext<'input>) {}
    /**
     * Exit a parse tree produced by the {@code functionExpression}
     * labeled alternative in {@link Qasm2Parser#exp}.
     * @param ctx the parse tree
     */
    fn exit_functionExpression(&mut self, _ctx: &FunctionExpressionContext<'input>) {}
}

antlr_rust::coerce_from! { 'input : Qasm2ParserListener<'input> }
