lexer grammar Qasm2Lexer;

OPENQASM: 'OPENQASM' -> pushMode(VERSION_IDENTIFIER);

INCLUDE: 'include';
QREG: 'qreg';
CREG: 'creg';
GATE: 'gate';
OPAQUE: 'opaque';
RESET: 'reset';
MEASURE: 'measure';
BARRIER: 'barrier';
IF: 'if';
PI: 'pi';
U: 'U';
CX: 'CX';

LBRACKET: '[';
RBRACKET: ']';
LBRACE: '{';
RBRACE: '}';
LPAREN: '(';
RPAREN: ')';

SEMICOLON: ';';
COMMA: ',';
DOT: '.';
ARROW: '->';

EQUALS: '==';
PLUS: '+';
MINUS: '-';
ASTERISK: '*';
SLASH: '/';
CARET: '^';
SIN: 'sin';
COS: 'cos';
TAN: 'tan';
EXP: 'exp';
LN: 'ln';
SQRT: 'sqrt';

Integer: '0' | [1-9] [0-9]*;

fragment FloatLiteralExponent: [eE] (PLUS | MINUS)? Integer;
Float:
    Integer FloatLiteralExponent
    | DOT [0-9]+ FloatLiteralExponent?
    | Integer DOT [0-9]* FloatLiteralExponent?;

StringLiteral: '"' ~["\r\t\n]+? '"';

Whitespace: [ \t]+ -> channel(HIDDEN) ;
Newline: [\r\n]+ -> channel(HIDDEN) ;
LineComment : '//' ~[\r\n]* -> skip;
BlockComment : '/*' .*? '*/' -> skip;

Identifier: [a-z][A-Za-z0-9_]*;

mode VERSION_IDENTIFIER;
    VERSION_IDENTIFER_WHITESPACE: [ \t\r\n]+ -> channel(HIDDEN);
    VersionSpecifier: [0-9]+ ('.' [0-9]+)? -> popMode;
