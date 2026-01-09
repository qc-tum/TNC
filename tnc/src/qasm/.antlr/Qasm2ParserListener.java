// Generated from /work_fast/ga87com/QuantumResearch/TensorStuff/GITHUB/tensornetworkcontractions/src/qasm/Qasm2Parser.g4 by ANTLR 4.13.1
import org.antlr.v4.runtime.tree.ParseTreeListener;

/**
 * This interface defines a complete listener for a parse tree produced by
 * {@link Qasm2Parser}.
 */
public interface Qasm2ParserListener extends ParseTreeListener {
	/**
	 * Enter a parse tree produced by {@link Qasm2Parser#program}.
	 * @param ctx the parse tree
	 */
	void enterProgram(Qasm2Parser.ProgramContext ctx);
	/**
	 * Exit a parse tree produced by {@link Qasm2Parser#program}.
	 * @param ctx the parse tree
	 */
	void exitProgram(Qasm2Parser.ProgramContext ctx);
	/**
	 * Enter a parse tree produced by {@link Qasm2Parser#version}.
	 * @param ctx the parse tree
	 */
	void enterVersion(Qasm2Parser.VersionContext ctx);
	/**
	 * Exit a parse tree produced by {@link Qasm2Parser#version}.
	 * @param ctx the parse tree
	 */
	void exitVersion(Qasm2Parser.VersionContext ctx);
	/**
	 * Enter a parse tree produced by {@link Qasm2Parser#statement}.
	 * @param ctx the parse tree
	 */
	void enterStatement(Qasm2Parser.StatementContext ctx);
	/**
	 * Exit a parse tree produced by {@link Qasm2Parser#statement}.
	 * @param ctx the parse tree
	 */
	void exitStatement(Qasm2Parser.StatementContext ctx);
	/**
	 * Enter a parse tree produced by {@link Qasm2Parser#includeStatement}.
	 * @param ctx the parse tree
	 */
	void enterIncludeStatement(Qasm2Parser.IncludeStatementContext ctx);
	/**
	 * Exit a parse tree produced by {@link Qasm2Parser#includeStatement}.
	 * @param ctx the parse tree
	 */
	void exitIncludeStatement(Qasm2Parser.IncludeStatementContext ctx);
	/**
	 * Enter a parse tree produced by {@link Qasm2Parser#declaration}.
	 * @param ctx the parse tree
	 */
	void enterDeclaration(Qasm2Parser.DeclarationContext ctx);
	/**
	 * Exit a parse tree produced by {@link Qasm2Parser#declaration}.
	 * @param ctx the parse tree
	 */
	void exitDeclaration(Qasm2Parser.DeclarationContext ctx);
	/**
	 * Enter a parse tree produced by {@link Qasm2Parser#gateDeclaration}.
	 * @param ctx the parse tree
	 */
	void enterGateDeclaration(Qasm2Parser.GateDeclarationContext ctx);
	/**
	 * Exit a parse tree produced by {@link Qasm2Parser#gateDeclaration}.
	 * @param ctx the parse tree
	 */
	void exitGateDeclaration(Qasm2Parser.GateDeclarationContext ctx);
	/**
	 * Enter a parse tree produced by {@link Qasm2Parser#quantumOperation}.
	 * @param ctx the parse tree
	 */
	void enterQuantumOperation(Qasm2Parser.QuantumOperationContext ctx);
	/**
	 * Exit a parse tree produced by {@link Qasm2Parser#quantumOperation}.
	 * @param ctx the parse tree
	 */
	void exitQuantumOperation(Qasm2Parser.QuantumOperationContext ctx);
	/**
	 * Enter a parse tree produced by {@link Qasm2Parser#ifStatement}.
	 * @param ctx the parse tree
	 */
	void enterIfStatement(Qasm2Parser.IfStatementContext ctx);
	/**
	 * Exit a parse tree produced by {@link Qasm2Parser#ifStatement}.
	 * @param ctx the parse tree
	 */
	void exitIfStatement(Qasm2Parser.IfStatementContext ctx);
	/**
	 * Enter a parse tree produced by {@link Qasm2Parser#barrier}.
	 * @param ctx the parse tree
	 */
	void enterBarrier(Qasm2Parser.BarrierContext ctx);
	/**
	 * Exit a parse tree produced by {@link Qasm2Parser#barrier}.
	 * @param ctx the parse tree
	 */
	void exitBarrier(Qasm2Parser.BarrierContext ctx);
	/**
	 * Enter a parse tree produced by {@link Qasm2Parser#bodyStatement}.
	 * @param ctx the parse tree
	 */
	void enterBodyStatement(Qasm2Parser.BodyStatementContext ctx);
	/**
	 * Exit a parse tree produced by {@link Qasm2Parser#bodyStatement}.
	 * @param ctx the parse tree
	 */
	void exitBodyStatement(Qasm2Parser.BodyStatementContext ctx);
	/**
	 * Enter a parse tree produced by {@link Qasm2Parser#gateCall}.
	 * @param ctx the parse tree
	 */
	void enterGateCall(Qasm2Parser.GateCallContext ctx);
	/**
	 * Exit a parse tree produced by {@link Qasm2Parser#gateCall}.
	 * @param ctx the parse tree
	 */
	void exitGateCall(Qasm2Parser.GateCallContext ctx);
	/**
	 * Enter a parse tree produced by {@link Qasm2Parser#idlist}.
	 * @param ctx the parse tree
	 */
	void enterIdlist(Qasm2Parser.IdlistContext ctx);
	/**
	 * Exit a parse tree produced by {@link Qasm2Parser#idlist}.
	 * @param ctx the parse tree
	 */
	void exitIdlist(Qasm2Parser.IdlistContext ctx);
	/**
	 * Enter a parse tree produced by {@link Qasm2Parser#mixedlist}.
	 * @param ctx the parse tree
	 */
	void enterMixedlist(Qasm2Parser.MixedlistContext ctx);
	/**
	 * Exit a parse tree produced by {@link Qasm2Parser#mixedlist}.
	 * @param ctx the parse tree
	 */
	void exitMixedlist(Qasm2Parser.MixedlistContext ctx);
	/**
	 * Enter a parse tree produced by {@link Qasm2Parser#argument}.
	 * @param ctx the parse tree
	 */
	void enterArgument(Qasm2Parser.ArgumentContext ctx);
	/**
	 * Exit a parse tree produced by {@link Qasm2Parser#argument}.
	 * @param ctx the parse tree
	 */
	void exitArgument(Qasm2Parser.ArgumentContext ctx);
	/**
	 * Enter a parse tree produced by {@link Qasm2Parser#designator}.
	 * @param ctx the parse tree
	 */
	void enterDesignator(Qasm2Parser.DesignatorContext ctx);
	/**
	 * Exit a parse tree produced by {@link Qasm2Parser#designator}.
	 * @param ctx the parse tree
	 */
	void exitDesignator(Qasm2Parser.DesignatorContext ctx);
	/**
	 * Enter a parse tree produced by {@link Qasm2Parser#explist}.
	 * @param ctx the parse tree
	 */
	void enterExplist(Qasm2Parser.ExplistContext ctx);
	/**
	 * Exit a parse tree produced by {@link Qasm2Parser#explist}.
	 * @param ctx the parse tree
	 */
	void exitExplist(Qasm2Parser.ExplistContext ctx);
	/**
	 * Enter a parse tree produced by the {@code bitwiseXorExpression}
	 * labeled alternative in {@link Qasm2Parser#exp}.
	 * @param ctx the parse tree
	 */
	void enterBitwiseXorExpression(Qasm2Parser.BitwiseXorExpressionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code bitwiseXorExpression}
	 * labeled alternative in {@link Qasm2Parser#exp}.
	 * @param ctx the parse tree
	 */
	void exitBitwiseXorExpression(Qasm2Parser.BitwiseXorExpressionContext ctx);
	/**
	 * Enter a parse tree produced by the {@code additiveExpression}
	 * labeled alternative in {@link Qasm2Parser#exp}.
	 * @param ctx the parse tree
	 */
	void enterAdditiveExpression(Qasm2Parser.AdditiveExpressionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code additiveExpression}
	 * labeled alternative in {@link Qasm2Parser#exp}.
	 * @param ctx the parse tree
	 */
	void exitAdditiveExpression(Qasm2Parser.AdditiveExpressionContext ctx);
	/**
	 * Enter a parse tree produced by the {@code parenthesisExpression}
	 * labeled alternative in {@link Qasm2Parser#exp}.
	 * @param ctx the parse tree
	 */
	void enterParenthesisExpression(Qasm2Parser.ParenthesisExpressionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code parenthesisExpression}
	 * labeled alternative in {@link Qasm2Parser#exp}.
	 * @param ctx the parse tree
	 */
	void exitParenthesisExpression(Qasm2Parser.ParenthesisExpressionContext ctx);
	/**
	 * Enter a parse tree produced by the {@code multiplicativeExpression}
	 * labeled alternative in {@link Qasm2Parser#exp}.
	 * @param ctx the parse tree
	 */
	void enterMultiplicativeExpression(Qasm2Parser.MultiplicativeExpressionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code multiplicativeExpression}
	 * labeled alternative in {@link Qasm2Parser#exp}.
	 * @param ctx the parse tree
	 */
	void exitMultiplicativeExpression(Qasm2Parser.MultiplicativeExpressionContext ctx);
	/**
	 * Enter a parse tree produced by the {@code unaryExpression}
	 * labeled alternative in {@link Qasm2Parser#exp}.
	 * @param ctx the parse tree
	 */
	void enterUnaryExpression(Qasm2Parser.UnaryExpressionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code unaryExpression}
	 * labeled alternative in {@link Qasm2Parser#exp}.
	 * @param ctx the parse tree
	 */
	void exitUnaryExpression(Qasm2Parser.UnaryExpressionContext ctx);
	/**
	 * Enter a parse tree produced by the {@code literalExpression}
	 * labeled alternative in {@link Qasm2Parser#exp}.
	 * @param ctx the parse tree
	 */
	void enterLiteralExpression(Qasm2Parser.LiteralExpressionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code literalExpression}
	 * labeled alternative in {@link Qasm2Parser#exp}.
	 * @param ctx the parse tree
	 */
	void exitLiteralExpression(Qasm2Parser.LiteralExpressionContext ctx);
	/**
	 * Enter a parse tree produced by the {@code functionExpression}
	 * labeled alternative in {@link Qasm2Parser#exp}.
	 * @param ctx the parse tree
	 */
	void enterFunctionExpression(Qasm2Parser.FunctionExpressionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code functionExpression}
	 * labeled alternative in {@link Qasm2Parser#exp}.
	 * @param ctx the parse tree
	 */
	void exitFunctionExpression(Qasm2Parser.FunctionExpressionContext ctx);
}