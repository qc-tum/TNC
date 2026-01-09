parser grammar Qasm2Parser;

options {
	tokenVocab = Qasm2Lexer;
}

program: version statement* EOF;
version: OPENQASM VersionSpecifier SEMICOLON;

statement:
	includeStatement
	| declaration
	| gateDeclaration
	| quantumOperation
	| ifStatement
	| barrier;

includeStatement: INCLUDE StringLiteral SEMICOLON;

declaration: (QREG | CREG) Identifier designator SEMICOLON;

gateDeclaration:
	GATE Identifier (LPAREN params = idlist? RPAREN)? qubits = idlist LBRACE body = bodyStatement*
		RBRACE
	| OPAQUE Identifier (LPAREN params = idlist? RPAREN)? qubits = idlist SEMICOLON;

quantumOperation:
	gateCall
	| MEASURE src = argument ARROW dest = argument SEMICOLON
	| RESET dest = argument SEMICOLON;

ifStatement:
	IF LPAREN Identifier EQUALS Integer RPAREN quantumOperation;

barrier: BARRIER mixedlist SEMICOLON;

bodyStatement: gateCall | BARRIER idlist SEMICOLON;

gateCall:
	(U | CX | Identifier) (LPAREN explist? RPAREN)? mixedlist SEMICOLON;

idlist: Identifier (COMMA Identifier)*;

mixedlist: argument (COMMA argument)*;

argument: Identifier designator?;

designator: LBRACKET Integer RBRACKET;

explist: exp (COMMA exp)*;

exp:
	LPAREN exp RPAREN												# parenthesisExpression
	| op = MINUS exp												# unaryExpression
	| lhs = exp op = (PLUS | MINUS) rhs = exp						# additiveExpression
	| lhs = exp op = (ASTERISK | SLASH) rhs = exp					# multiplicativeExpression
	| lhs = exp op = CARET rhs = exp								# bitwiseXorExpression
	| op = (SIN | COS | TAN | EXP | LN | SQRT) LPAREN exp RPAREN	# functionExpression
	| (Float | Integer | PI | Identifier)							# literalExpression;