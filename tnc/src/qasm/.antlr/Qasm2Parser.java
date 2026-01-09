// Generated from /work_fast/ga87com/QuantumResearch/TensorStuff/GITHUB/tensornetworkcontractions/src/qasm/Qasm2Parser.g4 by ANTLR 4.13.1
import org.antlr.v4.runtime.atn.*;
import org.antlr.v4.runtime.dfa.DFA;
import org.antlr.v4.runtime.*;
import org.antlr.v4.runtime.misc.*;
import org.antlr.v4.runtime.tree.*;
import java.util.List;
import java.util.Iterator;
import java.util.ArrayList;

@SuppressWarnings({"all", "warnings", "unchecked", "unused", "cast", "CheckReturnValue"})
public class Qasm2Parser extends Parser {
	static { RuntimeMetaData.checkVersion("4.13.1", RuntimeMetaData.VERSION); }

	protected static final DFA[] _decisionToDFA;
	protected static final PredictionContextCache _sharedContextCache =
		new PredictionContextCache();
	public static final int
		OPENQASM=1, INCLUDE=2, QREG=3, CREG=4, GATE=5, OPAQUE=6, RESET=7, MEASURE=8, 
		BARRIER=9, IF=10, PI=11, U=12, CX=13, LBRACKET=14, RBRACKET=15, LBRACE=16, 
		RBRACE=17, LPAREN=18, RPAREN=19, SEMICOLON=20, COMMA=21, DOT=22, ARROW=23, 
		EQUALS=24, PLUS=25, MINUS=26, ASTERISK=27, SLASH=28, CARET=29, SIN=30, 
		COS=31, TAN=32, EXP=33, LN=34, SQRT=35, Integer=36, Float=37, StringLiteral=38, 
		Whitespace=39, Newline=40, LineComment=41, BlockComment=42, Identifier=43, 
		VERSION_IDENTIFER_WHITESPACE=44, VersionSpecifier=45;
	public static final int
		RULE_program = 0, RULE_version = 1, RULE_statement = 2, RULE_includeStatement = 3, 
		RULE_declaration = 4, RULE_gateDeclaration = 5, RULE_quantumOperation = 6, 
		RULE_ifStatement = 7, RULE_barrier = 8, RULE_bodyStatement = 9, RULE_gateCall = 10, 
		RULE_idlist = 11, RULE_mixedlist = 12, RULE_argument = 13, RULE_designator = 14, 
		RULE_explist = 15, RULE_exp = 16;
	private static String[] makeRuleNames() {
		return new String[] {
			"program", "version", "statement", "includeStatement", "declaration", 
			"gateDeclaration", "quantumOperation", "ifStatement", "barrier", "bodyStatement", 
			"gateCall", "idlist", "mixedlist", "argument", "designator", "explist", 
			"exp"
		};
	}
	public static final String[] ruleNames = makeRuleNames();

	private static String[] makeLiteralNames() {
		return new String[] {
			null, "'OPENQASM'", "'include'", "'qreg'", "'creg'", "'gate'", "'opaque'", 
			"'reset'", "'measure'", "'barrier'", "'if'", "'pi'", "'U'", "'CX'", "'['", 
			"']'", "'{'", "'}'", "'('", "')'", "';'", "','", "'.'", "'->'", "'=='", 
			"'+'", "'-'", "'*'", "'/'", "'^'", "'sin'", "'cos'", "'tan'", "'exp'", 
			"'ln'", "'sqrt'"
		};
	}
	private static final String[] _LITERAL_NAMES = makeLiteralNames();
	private static String[] makeSymbolicNames() {
		return new String[] {
			null, "OPENQASM", "INCLUDE", "QREG", "CREG", "GATE", "OPAQUE", "RESET", 
			"MEASURE", "BARRIER", "IF", "PI", "U", "CX", "LBRACKET", "RBRACKET", 
			"LBRACE", "RBRACE", "LPAREN", "RPAREN", "SEMICOLON", "COMMA", "DOT", 
			"ARROW", "EQUALS", "PLUS", "MINUS", "ASTERISK", "SLASH", "CARET", "SIN", 
			"COS", "TAN", "EXP", "LN", "SQRT", "Integer", "Float", "StringLiteral", 
			"Whitespace", "Newline", "LineComment", "BlockComment", "Identifier", 
			"VERSION_IDENTIFER_WHITESPACE", "VersionSpecifier"
		};
	}
	private static final String[] _SYMBOLIC_NAMES = makeSymbolicNames();
	public static final Vocabulary VOCABULARY = new VocabularyImpl(_LITERAL_NAMES, _SYMBOLIC_NAMES);

	/**
	 * @deprecated Use {@link #VOCABULARY} instead.
	 */
	@Deprecated
	public static final String[] tokenNames;
	static {
		tokenNames = new String[_SYMBOLIC_NAMES.length];
		for (int i = 0; i < tokenNames.length; i++) {
			tokenNames[i] = VOCABULARY.getLiteralName(i);
			if (tokenNames[i] == null) {
				tokenNames[i] = VOCABULARY.getSymbolicName(i);
			}

			if (tokenNames[i] == null) {
				tokenNames[i] = "<INVALID>";
			}
		}
	}

	@Override
	@Deprecated
	public String[] getTokenNames() {
		return tokenNames;
	}

	@Override

	public Vocabulary getVocabulary() {
		return VOCABULARY;
	}

	@Override
	public String getGrammarFileName() { return "Qasm2Parser.g4"; }

	@Override
	public String[] getRuleNames() { return ruleNames; }

	@Override
	public String getSerializedATN() { return _serializedATN; }

	@Override
	public ATN getATN() { return _ATN; }

	public Qasm2Parser(TokenStream input) {
		super(input);
		_interp = new ParserATNSimulator(this,_ATN,_decisionToDFA,_sharedContextCache);
	}

	@SuppressWarnings("CheckReturnValue")
	public static class ProgramContext extends ParserRuleContext {
		public VersionContext version() {
			return getRuleContext(VersionContext.class,0);
		}
		public TerminalNode EOF() { return getToken(Qasm2Parser.EOF, 0); }
		public List<StatementContext> statement() {
			return getRuleContexts(StatementContext.class);
		}
		public StatementContext statement(int i) {
			return getRuleContext(StatementContext.class,i);
		}
		public ProgramContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_program; }
	}

	public final ProgramContext program() throws RecognitionException {
		ProgramContext _localctx = new ProgramContext(_ctx, getState());
		enterRule(_localctx, 0, RULE_program);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(34);
			version();
			setState(38);
			_errHandler.sync(this);
			_la = _input.LA(1);
			while ((((_la) & ~0x3f) == 0 && ((1L << _la) & 8796093036540L) != 0)) {
				{
				{
				setState(35);
				statement();
				}
				}
				setState(40);
				_errHandler.sync(this);
				_la = _input.LA(1);
			}
			setState(41);
			match(EOF);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class VersionContext extends ParserRuleContext {
		public TerminalNode OPENQASM() { return getToken(Qasm2Parser.OPENQASM, 0); }
		public TerminalNode VersionSpecifier() { return getToken(Qasm2Parser.VersionSpecifier, 0); }
		public TerminalNode SEMICOLON() { return getToken(Qasm2Parser.SEMICOLON, 0); }
		public VersionContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_version; }
	}

	public final VersionContext version() throws RecognitionException {
		VersionContext _localctx = new VersionContext(_ctx, getState());
		enterRule(_localctx, 2, RULE_version);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(43);
			match(OPENQASM);
			setState(44);
			match(VersionSpecifier);
			setState(45);
			match(SEMICOLON);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class StatementContext extends ParserRuleContext {
		public IncludeStatementContext includeStatement() {
			return getRuleContext(IncludeStatementContext.class,0);
		}
		public DeclarationContext declaration() {
			return getRuleContext(DeclarationContext.class,0);
		}
		public GateDeclarationContext gateDeclaration() {
			return getRuleContext(GateDeclarationContext.class,0);
		}
		public QuantumOperationContext quantumOperation() {
			return getRuleContext(QuantumOperationContext.class,0);
		}
		public IfStatementContext ifStatement() {
			return getRuleContext(IfStatementContext.class,0);
		}
		public BarrierContext barrier() {
			return getRuleContext(BarrierContext.class,0);
		}
		public StatementContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_statement; }
	}

	public final StatementContext statement() throws RecognitionException {
		StatementContext _localctx = new StatementContext(_ctx, getState());
		enterRule(_localctx, 4, RULE_statement);
		try {
			setState(53);
			_errHandler.sync(this);
			switch (_input.LA(1)) {
			case INCLUDE:
				enterOuterAlt(_localctx, 1);
				{
				setState(47);
				includeStatement();
				}
				break;
			case QREG:
			case CREG:
				enterOuterAlt(_localctx, 2);
				{
				setState(48);
				declaration();
				}
				break;
			case GATE:
			case OPAQUE:
				enterOuterAlt(_localctx, 3);
				{
				setState(49);
				gateDeclaration();
				}
				break;
			case RESET:
			case MEASURE:
			case U:
			case CX:
			case Identifier:
				enterOuterAlt(_localctx, 4);
				{
				setState(50);
				quantumOperation();
				}
				break;
			case IF:
				enterOuterAlt(_localctx, 5);
				{
				setState(51);
				ifStatement();
				}
				break;
			case BARRIER:
				enterOuterAlt(_localctx, 6);
				{
				setState(52);
				barrier();
				}
				break;
			default:
				throw new NoViableAltException(this);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class IncludeStatementContext extends ParserRuleContext {
		public TerminalNode INCLUDE() { return getToken(Qasm2Parser.INCLUDE, 0); }
		public TerminalNode StringLiteral() { return getToken(Qasm2Parser.StringLiteral, 0); }
		public TerminalNode SEMICOLON() { return getToken(Qasm2Parser.SEMICOLON, 0); }
		public IncludeStatementContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_includeStatement; }
	}

	public final IncludeStatementContext includeStatement() throws RecognitionException {
		IncludeStatementContext _localctx = new IncludeStatementContext(_ctx, getState());
		enterRule(_localctx, 6, RULE_includeStatement);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(55);
			match(INCLUDE);
			setState(56);
			match(StringLiteral);
			setState(57);
			match(SEMICOLON);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class DeclarationContext extends ParserRuleContext {
		public TerminalNode Identifier() { return getToken(Qasm2Parser.Identifier, 0); }
		public DesignatorContext designator() {
			return getRuleContext(DesignatorContext.class,0);
		}
		public TerminalNode SEMICOLON() { return getToken(Qasm2Parser.SEMICOLON, 0); }
		public TerminalNode QREG() { return getToken(Qasm2Parser.QREG, 0); }
		public TerminalNode CREG() { return getToken(Qasm2Parser.CREG, 0); }
		public DeclarationContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_declaration; }
	}

	public final DeclarationContext declaration() throws RecognitionException {
		DeclarationContext _localctx = new DeclarationContext(_ctx, getState());
		enterRule(_localctx, 8, RULE_declaration);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(59);
			_la = _input.LA(1);
			if ( !(_la==QREG || _la==CREG) ) {
			_errHandler.recoverInline(this);
			}
			else {
				if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
				_errHandler.reportMatch(this);
				consume();
			}
			setState(60);
			match(Identifier);
			setState(61);
			designator();
			setState(62);
			match(SEMICOLON);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class GateDeclarationContext extends ParserRuleContext {
		public IdlistContext params;
		public IdlistContext qubits;
		public BodyStatementContext body;
		public TerminalNode GATE() { return getToken(Qasm2Parser.GATE, 0); }
		public TerminalNode Identifier() { return getToken(Qasm2Parser.Identifier, 0); }
		public TerminalNode LBRACE() { return getToken(Qasm2Parser.LBRACE, 0); }
		public TerminalNode RBRACE() { return getToken(Qasm2Parser.RBRACE, 0); }
		public List<IdlistContext> idlist() {
			return getRuleContexts(IdlistContext.class);
		}
		public IdlistContext idlist(int i) {
			return getRuleContext(IdlistContext.class,i);
		}
		public TerminalNode LPAREN() { return getToken(Qasm2Parser.LPAREN, 0); }
		public TerminalNode RPAREN() { return getToken(Qasm2Parser.RPAREN, 0); }
		public List<BodyStatementContext> bodyStatement() {
			return getRuleContexts(BodyStatementContext.class);
		}
		public BodyStatementContext bodyStatement(int i) {
			return getRuleContext(BodyStatementContext.class,i);
		}
		public TerminalNode OPAQUE() { return getToken(Qasm2Parser.OPAQUE, 0); }
		public TerminalNode SEMICOLON() { return getToken(Qasm2Parser.SEMICOLON, 0); }
		public GateDeclarationContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_gateDeclaration; }
	}

	public final GateDeclarationContext gateDeclaration() throws RecognitionException {
		GateDeclarationContext _localctx = new GateDeclarationContext(_ctx, getState());
		enterRule(_localctx, 10, RULE_gateDeclaration);
		int _la;
		try {
			setState(95);
			_errHandler.sync(this);
			switch (_input.LA(1)) {
			case GATE:
				enterOuterAlt(_localctx, 1);
				{
				setState(64);
				match(GATE);
				setState(65);
				match(Identifier);
				setState(71);
				_errHandler.sync(this);
				_la = _input.LA(1);
				if (_la==LPAREN) {
					{
					setState(66);
					match(LPAREN);
					setState(68);
					_errHandler.sync(this);
					_la = _input.LA(1);
					if (_la==Identifier) {
						{
						setState(67);
						((GateDeclarationContext)_localctx).params = idlist();
						}
					}

					setState(70);
					match(RPAREN);
					}
				}

				setState(73);
				((GateDeclarationContext)_localctx).qubits = idlist();
				setState(74);
				match(LBRACE);
				setState(78);
				_errHandler.sync(this);
				_la = _input.LA(1);
				while ((((_la) & ~0x3f) == 0 && ((1L << _la) & 8796093035008L) != 0)) {
					{
					{
					setState(75);
					((GateDeclarationContext)_localctx).body = bodyStatement();
					}
					}
					setState(80);
					_errHandler.sync(this);
					_la = _input.LA(1);
				}
				setState(81);
				match(RBRACE);
				}
				break;
			case OPAQUE:
				enterOuterAlt(_localctx, 2);
				{
				setState(83);
				match(OPAQUE);
				setState(84);
				match(Identifier);
				setState(90);
				_errHandler.sync(this);
				_la = _input.LA(1);
				if (_la==LPAREN) {
					{
					setState(85);
					match(LPAREN);
					setState(87);
					_errHandler.sync(this);
					_la = _input.LA(1);
					if (_la==Identifier) {
						{
						setState(86);
						((GateDeclarationContext)_localctx).params = idlist();
						}
					}

					setState(89);
					match(RPAREN);
					}
				}

				setState(92);
				((GateDeclarationContext)_localctx).qubits = idlist();
				setState(93);
				match(SEMICOLON);
				}
				break;
			default:
				throw new NoViableAltException(this);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class QuantumOperationContext extends ParserRuleContext {
		public ArgumentContext src;
		public ArgumentContext dest;
		public GateCallContext gateCall() {
			return getRuleContext(GateCallContext.class,0);
		}
		public TerminalNode MEASURE() { return getToken(Qasm2Parser.MEASURE, 0); }
		public TerminalNode ARROW() { return getToken(Qasm2Parser.ARROW, 0); }
		public TerminalNode SEMICOLON() { return getToken(Qasm2Parser.SEMICOLON, 0); }
		public List<ArgumentContext> argument() {
			return getRuleContexts(ArgumentContext.class);
		}
		public ArgumentContext argument(int i) {
			return getRuleContext(ArgumentContext.class,i);
		}
		public TerminalNode RESET() { return getToken(Qasm2Parser.RESET, 0); }
		public QuantumOperationContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_quantumOperation; }
	}

	public final QuantumOperationContext quantumOperation() throws RecognitionException {
		QuantumOperationContext _localctx = new QuantumOperationContext(_ctx, getState());
		enterRule(_localctx, 12, RULE_quantumOperation);
		try {
			setState(108);
			_errHandler.sync(this);
			switch (_input.LA(1)) {
			case U:
			case CX:
			case Identifier:
				enterOuterAlt(_localctx, 1);
				{
				setState(97);
				gateCall();
				}
				break;
			case MEASURE:
				enterOuterAlt(_localctx, 2);
				{
				setState(98);
				match(MEASURE);
				setState(99);
				((QuantumOperationContext)_localctx).src = argument();
				setState(100);
				match(ARROW);
				setState(101);
				((QuantumOperationContext)_localctx).dest = argument();
				setState(102);
				match(SEMICOLON);
				}
				break;
			case RESET:
				enterOuterAlt(_localctx, 3);
				{
				setState(104);
				match(RESET);
				setState(105);
				((QuantumOperationContext)_localctx).dest = argument();
				setState(106);
				match(SEMICOLON);
				}
				break;
			default:
				throw new NoViableAltException(this);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class IfStatementContext extends ParserRuleContext {
		public TerminalNode IF() { return getToken(Qasm2Parser.IF, 0); }
		public TerminalNode LPAREN() { return getToken(Qasm2Parser.LPAREN, 0); }
		public TerminalNode Identifier() { return getToken(Qasm2Parser.Identifier, 0); }
		public TerminalNode EQUALS() { return getToken(Qasm2Parser.EQUALS, 0); }
		public TerminalNode Integer() { return getToken(Qasm2Parser.Integer, 0); }
		public TerminalNode RPAREN() { return getToken(Qasm2Parser.RPAREN, 0); }
		public QuantumOperationContext quantumOperation() {
			return getRuleContext(QuantumOperationContext.class,0);
		}
		public IfStatementContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_ifStatement; }
	}

	public final IfStatementContext ifStatement() throws RecognitionException {
		IfStatementContext _localctx = new IfStatementContext(_ctx, getState());
		enterRule(_localctx, 14, RULE_ifStatement);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(110);
			match(IF);
			setState(111);
			match(LPAREN);
			setState(112);
			match(Identifier);
			setState(113);
			match(EQUALS);
			setState(114);
			match(Integer);
			setState(115);
			match(RPAREN);
			setState(116);
			quantumOperation();
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class BarrierContext extends ParserRuleContext {
		public TerminalNode BARRIER() { return getToken(Qasm2Parser.BARRIER, 0); }
		public MixedlistContext mixedlist() {
			return getRuleContext(MixedlistContext.class,0);
		}
		public TerminalNode SEMICOLON() { return getToken(Qasm2Parser.SEMICOLON, 0); }
		public BarrierContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_barrier; }
	}

	public final BarrierContext barrier() throws RecognitionException {
		BarrierContext _localctx = new BarrierContext(_ctx, getState());
		enterRule(_localctx, 16, RULE_barrier);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(118);
			match(BARRIER);
			setState(119);
			mixedlist();
			setState(120);
			match(SEMICOLON);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class BodyStatementContext extends ParserRuleContext {
		public GateCallContext gateCall() {
			return getRuleContext(GateCallContext.class,0);
		}
		public TerminalNode BARRIER() { return getToken(Qasm2Parser.BARRIER, 0); }
		public IdlistContext idlist() {
			return getRuleContext(IdlistContext.class,0);
		}
		public TerminalNode SEMICOLON() { return getToken(Qasm2Parser.SEMICOLON, 0); }
		public BodyStatementContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_bodyStatement; }
	}

	public final BodyStatementContext bodyStatement() throws RecognitionException {
		BodyStatementContext _localctx = new BodyStatementContext(_ctx, getState());
		enterRule(_localctx, 18, RULE_bodyStatement);
		try {
			setState(127);
			_errHandler.sync(this);
			switch (_input.LA(1)) {
			case U:
			case CX:
			case Identifier:
				enterOuterAlt(_localctx, 1);
				{
				setState(122);
				gateCall();
				}
				break;
			case BARRIER:
				enterOuterAlt(_localctx, 2);
				{
				setState(123);
				match(BARRIER);
				setState(124);
				idlist();
				setState(125);
				match(SEMICOLON);
				}
				break;
			default:
				throw new NoViableAltException(this);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class GateCallContext extends ParserRuleContext {
		public MixedlistContext mixedlist() {
			return getRuleContext(MixedlistContext.class,0);
		}
		public TerminalNode SEMICOLON() { return getToken(Qasm2Parser.SEMICOLON, 0); }
		public TerminalNode U() { return getToken(Qasm2Parser.U, 0); }
		public TerminalNode CX() { return getToken(Qasm2Parser.CX, 0); }
		public TerminalNode Identifier() { return getToken(Qasm2Parser.Identifier, 0); }
		public TerminalNode LPAREN() { return getToken(Qasm2Parser.LPAREN, 0); }
		public TerminalNode RPAREN() { return getToken(Qasm2Parser.RPAREN, 0); }
		public ExplistContext explist() {
			return getRuleContext(ExplistContext.class,0);
		}
		public GateCallContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_gateCall; }
	}

	public final GateCallContext gateCall() throws RecognitionException {
		GateCallContext _localctx = new GateCallContext(_ctx, getState());
		enterRule(_localctx, 20, RULE_gateCall);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(129);
			_la = _input.LA(1);
			if ( !((((_la) & ~0x3f) == 0 && ((1L << _la) & 8796093034496L) != 0)) ) {
			_errHandler.recoverInline(this);
			}
			else {
				if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
				_errHandler.reportMatch(this);
				consume();
			}
			setState(135);
			_errHandler.sync(this);
			_la = _input.LA(1);
			if (_la==LPAREN) {
				{
				setState(130);
				match(LPAREN);
				setState(132);
				_errHandler.sync(this);
				_la = _input.LA(1);
				if ((((_la) & ~0x3f) == 0 && ((1L << _la) & 9069964560384L) != 0)) {
					{
					setState(131);
					explist();
					}
				}

				setState(134);
				match(RPAREN);
				}
			}

			setState(137);
			mixedlist();
			setState(138);
			match(SEMICOLON);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class IdlistContext extends ParserRuleContext {
		public List<TerminalNode> Identifier() { return getTokens(Qasm2Parser.Identifier); }
		public TerminalNode Identifier(int i) {
			return getToken(Qasm2Parser.Identifier, i);
		}
		public List<TerminalNode> COMMA() { return getTokens(Qasm2Parser.COMMA); }
		public TerminalNode COMMA(int i) {
			return getToken(Qasm2Parser.COMMA, i);
		}
		public IdlistContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_idlist; }
	}

	public final IdlistContext idlist() throws RecognitionException {
		IdlistContext _localctx = new IdlistContext(_ctx, getState());
		enterRule(_localctx, 22, RULE_idlist);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(140);
			match(Identifier);
			setState(145);
			_errHandler.sync(this);
			_la = _input.LA(1);
			while (_la==COMMA) {
				{
				{
				setState(141);
				match(COMMA);
				setState(142);
				match(Identifier);
				}
				}
				setState(147);
				_errHandler.sync(this);
				_la = _input.LA(1);
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class MixedlistContext extends ParserRuleContext {
		public List<ArgumentContext> argument() {
			return getRuleContexts(ArgumentContext.class);
		}
		public ArgumentContext argument(int i) {
			return getRuleContext(ArgumentContext.class,i);
		}
		public List<TerminalNode> COMMA() { return getTokens(Qasm2Parser.COMMA); }
		public TerminalNode COMMA(int i) {
			return getToken(Qasm2Parser.COMMA, i);
		}
		public MixedlistContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_mixedlist; }
	}

	public final MixedlistContext mixedlist() throws RecognitionException {
		MixedlistContext _localctx = new MixedlistContext(_ctx, getState());
		enterRule(_localctx, 24, RULE_mixedlist);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(148);
			argument();
			setState(153);
			_errHandler.sync(this);
			_la = _input.LA(1);
			while (_la==COMMA) {
				{
				{
				setState(149);
				match(COMMA);
				setState(150);
				argument();
				}
				}
				setState(155);
				_errHandler.sync(this);
				_la = _input.LA(1);
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class ArgumentContext extends ParserRuleContext {
		public TerminalNode Identifier() { return getToken(Qasm2Parser.Identifier, 0); }
		public DesignatorContext designator() {
			return getRuleContext(DesignatorContext.class,0);
		}
		public ArgumentContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_argument; }
	}

	public final ArgumentContext argument() throws RecognitionException {
		ArgumentContext _localctx = new ArgumentContext(_ctx, getState());
		enterRule(_localctx, 26, RULE_argument);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(156);
			match(Identifier);
			setState(158);
			_errHandler.sync(this);
			_la = _input.LA(1);
			if (_la==LBRACKET) {
				{
				setState(157);
				designator();
				}
			}

			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class DesignatorContext extends ParserRuleContext {
		public TerminalNode LBRACKET() { return getToken(Qasm2Parser.LBRACKET, 0); }
		public TerminalNode Integer() { return getToken(Qasm2Parser.Integer, 0); }
		public TerminalNode RBRACKET() { return getToken(Qasm2Parser.RBRACKET, 0); }
		public DesignatorContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_designator; }
	}

	public final DesignatorContext designator() throws RecognitionException {
		DesignatorContext _localctx = new DesignatorContext(_ctx, getState());
		enterRule(_localctx, 28, RULE_designator);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(160);
			match(LBRACKET);
			setState(161);
			match(Integer);
			setState(162);
			match(RBRACKET);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class ExplistContext extends ParserRuleContext {
		public List<ExpContext> exp() {
			return getRuleContexts(ExpContext.class);
		}
		public ExpContext exp(int i) {
			return getRuleContext(ExpContext.class,i);
		}
		public List<TerminalNode> COMMA() { return getTokens(Qasm2Parser.COMMA); }
		public TerminalNode COMMA(int i) {
			return getToken(Qasm2Parser.COMMA, i);
		}
		public ExplistContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_explist; }
	}

	public final ExplistContext explist() throws RecognitionException {
		ExplistContext _localctx = new ExplistContext(_ctx, getState());
		enterRule(_localctx, 30, RULE_explist);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(164);
			exp(0);
			setState(169);
			_errHandler.sync(this);
			_la = _input.LA(1);
			while (_la==COMMA) {
				{
				{
				setState(165);
				match(COMMA);
				setState(166);
				exp(0);
				}
				}
				setState(171);
				_errHandler.sync(this);
				_la = _input.LA(1);
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class ExpContext extends ParserRuleContext {
		public ExpContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_exp; }
	 
		public ExpContext() { }
		public void copyFrom(ExpContext ctx) {
			super.copyFrom(ctx);
		}
	}
	@SuppressWarnings("CheckReturnValue")
	public static class BitwiseXorExpressionContext extends ExpContext {
		public ExpContext lhs;
		public Token op;
		public ExpContext rhs;
		public List<ExpContext> exp() {
			return getRuleContexts(ExpContext.class);
		}
		public ExpContext exp(int i) {
			return getRuleContext(ExpContext.class,i);
		}
		public TerminalNode CARET() { return getToken(Qasm2Parser.CARET, 0); }
		public BitwiseXorExpressionContext(ExpContext ctx) { copyFrom(ctx); }
	}
	@SuppressWarnings("CheckReturnValue")
	public static class AdditiveExpressionContext extends ExpContext {
		public ExpContext lhs;
		public Token op;
		public ExpContext rhs;
		public List<ExpContext> exp() {
			return getRuleContexts(ExpContext.class);
		}
		public ExpContext exp(int i) {
			return getRuleContext(ExpContext.class,i);
		}
		public TerminalNode PLUS() { return getToken(Qasm2Parser.PLUS, 0); }
		public TerminalNode MINUS() { return getToken(Qasm2Parser.MINUS, 0); }
		public AdditiveExpressionContext(ExpContext ctx) { copyFrom(ctx); }
	}
	@SuppressWarnings("CheckReturnValue")
	public static class ParenthesisExpressionContext extends ExpContext {
		public TerminalNode LPAREN() { return getToken(Qasm2Parser.LPAREN, 0); }
		public ExpContext exp() {
			return getRuleContext(ExpContext.class,0);
		}
		public TerminalNode RPAREN() { return getToken(Qasm2Parser.RPAREN, 0); }
		public ParenthesisExpressionContext(ExpContext ctx) { copyFrom(ctx); }
	}
	@SuppressWarnings("CheckReturnValue")
	public static class MultiplicativeExpressionContext extends ExpContext {
		public ExpContext lhs;
		public Token op;
		public ExpContext rhs;
		public List<ExpContext> exp() {
			return getRuleContexts(ExpContext.class);
		}
		public ExpContext exp(int i) {
			return getRuleContext(ExpContext.class,i);
		}
		public TerminalNode ASTERISK() { return getToken(Qasm2Parser.ASTERISK, 0); }
		public TerminalNode SLASH() { return getToken(Qasm2Parser.SLASH, 0); }
		public MultiplicativeExpressionContext(ExpContext ctx) { copyFrom(ctx); }
	}
	@SuppressWarnings("CheckReturnValue")
	public static class UnaryExpressionContext extends ExpContext {
		public Token op;
		public ExpContext exp() {
			return getRuleContext(ExpContext.class,0);
		}
		public TerminalNode MINUS() { return getToken(Qasm2Parser.MINUS, 0); }
		public UnaryExpressionContext(ExpContext ctx) { copyFrom(ctx); }
	}
	@SuppressWarnings("CheckReturnValue")
	public static class LiteralExpressionContext extends ExpContext {
		public TerminalNode Float() { return getToken(Qasm2Parser.Float, 0); }
		public TerminalNode Integer() { return getToken(Qasm2Parser.Integer, 0); }
		public TerminalNode PI() { return getToken(Qasm2Parser.PI, 0); }
		public TerminalNode Identifier() { return getToken(Qasm2Parser.Identifier, 0); }
		public LiteralExpressionContext(ExpContext ctx) { copyFrom(ctx); }
	}
	@SuppressWarnings("CheckReturnValue")
	public static class FunctionExpressionContext extends ExpContext {
		public Token op;
		public TerminalNode LPAREN() { return getToken(Qasm2Parser.LPAREN, 0); }
		public ExpContext exp() {
			return getRuleContext(ExpContext.class,0);
		}
		public TerminalNode RPAREN() { return getToken(Qasm2Parser.RPAREN, 0); }
		public TerminalNode SIN() { return getToken(Qasm2Parser.SIN, 0); }
		public TerminalNode COS() { return getToken(Qasm2Parser.COS, 0); }
		public TerminalNode TAN() { return getToken(Qasm2Parser.TAN, 0); }
		public TerminalNode EXP() { return getToken(Qasm2Parser.EXP, 0); }
		public TerminalNode LN() { return getToken(Qasm2Parser.LN, 0); }
		public TerminalNode SQRT() { return getToken(Qasm2Parser.SQRT, 0); }
		public FunctionExpressionContext(ExpContext ctx) { copyFrom(ctx); }
	}

	public final ExpContext exp() throws RecognitionException {
		return exp(0);
	}

	private ExpContext exp(int _p) throws RecognitionException {
		ParserRuleContext _parentctx = _ctx;
		int _parentState = getState();
		ExpContext _localctx = new ExpContext(_ctx, _parentState);
		ExpContext _prevctx = _localctx;
		int _startState = 32;
		enterRecursionRule(_localctx, 32, RULE_exp, _p);
		int _la;
		try {
			int _alt;
			enterOuterAlt(_localctx, 1);
			{
			setState(185);
			_errHandler.sync(this);
			switch (_input.LA(1)) {
			case LPAREN:
				{
				_localctx = new ParenthesisExpressionContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;

				setState(173);
				match(LPAREN);
				setState(174);
				exp(0);
				setState(175);
				match(RPAREN);
				}
				break;
			case MINUS:
				{
				_localctx = new UnaryExpressionContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;
				setState(177);
				((UnaryExpressionContext)_localctx).op = match(MINUS);
				setState(178);
				exp(6);
				}
				break;
			case SIN:
			case COS:
			case TAN:
			case EXP:
			case LN:
			case SQRT:
				{
				_localctx = new FunctionExpressionContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;
				setState(179);
				((FunctionExpressionContext)_localctx).op = _input.LT(1);
				_la = _input.LA(1);
				if ( !((((_la) & ~0x3f) == 0 && ((1L << _la) & 67645734912L) != 0)) ) {
					((FunctionExpressionContext)_localctx).op = (Token)_errHandler.recoverInline(this);
				}
				else {
					if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
					_errHandler.reportMatch(this);
					consume();
				}
				setState(180);
				match(LPAREN);
				setState(181);
				exp(0);
				setState(182);
				match(RPAREN);
				}
				break;
			case PI:
			case Integer:
			case Float:
			case Identifier:
				{
				_localctx = new LiteralExpressionContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;
				setState(184);
				_la = _input.LA(1);
				if ( !((((_la) & ~0x3f) == 0 && ((1L << _la) & 9002251454464L) != 0)) ) {
				_errHandler.recoverInline(this);
				}
				else {
					if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
					_errHandler.reportMatch(this);
					consume();
				}
				}
				break;
			default:
				throw new NoViableAltException(this);
			}
			_ctx.stop = _input.LT(-1);
			setState(198);
			_errHandler.sync(this);
			_alt = getInterpreter().adaptivePredict(_input,18,_ctx);
			while ( _alt!=2 && _alt!=org.antlr.v4.runtime.atn.ATN.INVALID_ALT_NUMBER ) {
				if ( _alt==1 ) {
					if ( _parseListeners!=null ) triggerExitRuleEvent();
					_prevctx = _localctx;
					{
					setState(196);
					_errHandler.sync(this);
					switch ( getInterpreter().adaptivePredict(_input,17,_ctx) ) {
					case 1:
						{
						_localctx = new AdditiveExpressionContext(new ExpContext(_parentctx, _parentState));
						((AdditiveExpressionContext)_localctx).lhs = _prevctx;
						pushNewRecursionContext(_localctx, _startState, RULE_exp);
						setState(187);
						if (!(precpred(_ctx, 5))) throw new FailedPredicateException(this, "precpred(_ctx, 5)");
						setState(188);
						((AdditiveExpressionContext)_localctx).op = _input.LT(1);
						_la = _input.LA(1);
						if ( !(_la==PLUS || _la==MINUS) ) {
							((AdditiveExpressionContext)_localctx).op = (Token)_errHandler.recoverInline(this);
						}
						else {
							if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
							_errHandler.reportMatch(this);
							consume();
						}
						setState(189);
						((AdditiveExpressionContext)_localctx).rhs = exp(6);
						}
						break;
					case 2:
						{
						_localctx = new MultiplicativeExpressionContext(new ExpContext(_parentctx, _parentState));
						((MultiplicativeExpressionContext)_localctx).lhs = _prevctx;
						pushNewRecursionContext(_localctx, _startState, RULE_exp);
						setState(190);
						if (!(precpred(_ctx, 4))) throw new FailedPredicateException(this, "precpred(_ctx, 4)");
						setState(191);
						((MultiplicativeExpressionContext)_localctx).op = _input.LT(1);
						_la = _input.LA(1);
						if ( !(_la==ASTERISK || _la==SLASH) ) {
							((MultiplicativeExpressionContext)_localctx).op = (Token)_errHandler.recoverInline(this);
						}
						else {
							if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
							_errHandler.reportMatch(this);
							consume();
						}
						setState(192);
						((MultiplicativeExpressionContext)_localctx).rhs = exp(5);
						}
						break;
					case 3:
						{
						_localctx = new BitwiseXorExpressionContext(new ExpContext(_parentctx, _parentState));
						((BitwiseXorExpressionContext)_localctx).lhs = _prevctx;
						pushNewRecursionContext(_localctx, _startState, RULE_exp);
						setState(193);
						if (!(precpred(_ctx, 3))) throw new FailedPredicateException(this, "precpred(_ctx, 3)");
						setState(194);
						((BitwiseXorExpressionContext)_localctx).op = match(CARET);
						setState(195);
						((BitwiseXorExpressionContext)_localctx).rhs = exp(4);
						}
						break;
					}
					} 
				}
				setState(200);
				_errHandler.sync(this);
				_alt = getInterpreter().adaptivePredict(_input,18,_ctx);
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			unrollRecursionContexts(_parentctx);
		}
		return _localctx;
	}

	public boolean sempred(RuleContext _localctx, int ruleIndex, int predIndex) {
		switch (ruleIndex) {
		case 16:
			return exp_sempred((ExpContext)_localctx, predIndex);
		}
		return true;
	}
	private boolean exp_sempred(ExpContext _localctx, int predIndex) {
		switch (predIndex) {
		case 0:
			return precpred(_ctx, 5);
		case 1:
			return precpred(_ctx, 4);
		case 2:
			return precpred(_ctx, 3);
		}
		return true;
	}

	public static final String _serializedATN =
		"\u0004\u0001-\u00ca\u0002\u0000\u0007\u0000\u0002\u0001\u0007\u0001\u0002"+
		"\u0002\u0007\u0002\u0002\u0003\u0007\u0003\u0002\u0004\u0007\u0004\u0002"+
		"\u0005\u0007\u0005\u0002\u0006\u0007\u0006\u0002\u0007\u0007\u0007\u0002"+
		"\b\u0007\b\u0002\t\u0007\t\u0002\n\u0007\n\u0002\u000b\u0007\u000b\u0002"+
		"\f\u0007\f\u0002\r\u0007\r\u0002\u000e\u0007\u000e\u0002\u000f\u0007\u000f"+
		"\u0002\u0010\u0007\u0010\u0001\u0000\u0001\u0000\u0005\u0000%\b\u0000"+
		"\n\u0000\f\u0000(\t\u0000\u0001\u0000\u0001\u0000\u0001\u0001\u0001\u0001"+
		"\u0001\u0001\u0001\u0001\u0001\u0002\u0001\u0002\u0001\u0002\u0001\u0002"+
		"\u0001\u0002\u0001\u0002\u0003\u00026\b\u0002\u0001\u0003\u0001\u0003"+
		"\u0001\u0003\u0001\u0003\u0001\u0004\u0001\u0004\u0001\u0004\u0001\u0004"+
		"\u0001\u0004\u0001\u0005\u0001\u0005\u0001\u0005\u0001\u0005\u0003\u0005"+
		"E\b\u0005\u0001\u0005\u0003\u0005H\b\u0005\u0001\u0005\u0001\u0005\u0001"+
		"\u0005\u0005\u0005M\b\u0005\n\u0005\f\u0005P\t\u0005\u0001\u0005\u0001"+
		"\u0005\u0001\u0005\u0001\u0005\u0001\u0005\u0001\u0005\u0003\u0005X\b"+
		"\u0005\u0001\u0005\u0003\u0005[\b\u0005\u0001\u0005\u0001\u0005\u0001"+
		"\u0005\u0003\u0005`\b\u0005\u0001\u0006\u0001\u0006\u0001\u0006\u0001"+
		"\u0006\u0001\u0006\u0001\u0006\u0001\u0006\u0001\u0006\u0001\u0006\u0001"+
		"\u0006\u0001\u0006\u0003\u0006m\b\u0006\u0001\u0007\u0001\u0007\u0001"+
		"\u0007\u0001\u0007\u0001\u0007\u0001\u0007\u0001\u0007\u0001\u0007\u0001"+
		"\b\u0001\b\u0001\b\u0001\b\u0001\t\u0001\t\u0001\t\u0001\t\u0001\t\u0003"+
		"\t\u0080\b\t\u0001\n\u0001\n\u0001\n\u0003\n\u0085\b\n\u0001\n\u0003\n"+
		"\u0088\b\n\u0001\n\u0001\n\u0001\n\u0001\u000b\u0001\u000b\u0001\u000b"+
		"\u0005\u000b\u0090\b\u000b\n\u000b\f\u000b\u0093\t\u000b\u0001\f\u0001"+
		"\f\u0001\f\u0005\f\u0098\b\f\n\f\f\f\u009b\t\f\u0001\r\u0001\r\u0003\r"+
		"\u009f\b\r\u0001\u000e\u0001\u000e\u0001\u000e\u0001\u000e\u0001\u000f"+
		"\u0001\u000f\u0001\u000f\u0005\u000f\u00a8\b\u000f\n\u000f\f\u000f\u00ab"+
		"\t\u000f\u0001\u0010\u0001\u0010\u0001\u0010\u0001\u0010\u0001\u0010\u0001"+
		"\u0010\u0001\u0010\u0001\u0010\u0001\u0010\u0001\u0010\u0001\u0010\u0001"+
		"\u0010\u0001\u0010\u0003\u0010\u00ba\b\u0010\u0001\u0010\u0001\u0010\u0001"+
		"\u0010\u0001\u0010\u0001\u0010\u0001\u0010\u0001\u0010\u0001\u0010\u0001"+
		"\u0010\u0005\u0010\u00c5\b\u0010\n\u0010\f\u0010\u00c8\t\u0010\u0001\u0010"+
		"\u0000\u0001 \u0011\u0000\u0002\u0004\u0006\b\n\f\u000e\u0010\u0012\u0014"+
		"\u0016\u0018\u001a\u001c\u001e \u0000\u0006\u0001\u0000\u0003\u0004\u0002"+
		"\u0000\f\r++\u0001\u0000\u001e#\u0003\u0000\u000b\u000b$%++\u0001\u0000"+
		"\u0019\u001a\u0001\u0000\u001b\u001c\u00d3\u0000\"\u0001\u0000\u0000\u0000"+
		"\u0002+\u0001\u0000\u0000\u0000\u00045\u0001\u0000\u0000\u0000\u00067"+
		"\u0001\u0000\u0000\u0000\b;\u0001\u0000\u0000\u0000\n_\u0001\u0000\u0000"+
		"\u0000\fl\u0001\u0000\u0000\u0000\u000en\u0001\u0000\u0000\u0000\u0010"+
		"v\u0001\u0000\u0000\u0000\u0012\u007f\u0001\u0000\u0000\u0000\u0014\u0081"+
		"\u0001\u0000\u0000\u0000\u0016\u008c\u0001\u0000\u0000\u0000\u0018\u0094"+
		"\u0001\u0000\u0000\u0000\u001a\u009c\u0001\u0000\u0000\u0000\u001c\u00a0"+
		"\u0001\u0000\u0000\u0000\u001e\u00a4\u0001\u0000\u0000\u0000 \u00b9\u0001"+
		"\u0000\u0000\u0000\"&\u0003\u0002\u0001\u0000#%\u0003\u0004\u0002\u0000"+
		"$#\u0001\u0000\u0000\u0000%(\u0001\u0000\u0000\u0000&$\u0001\u0000\u0000"+
		"\u0000&\'\u0001\u0000\u0000\u0000\')\u0001\u0000\u0000\u0000(&\u0001\u0000"+
		"\u0000\u0000)*\u0005\u0000\u0000\u0001*\u0001\u0001\u0000\u0000\u0000"+
		"+,\u0005\u0001\u0000\u0000,-\u0005-\u0000\u0000-.\u0005\u0014\u0000\u0000"+
		".\u0003\u0001\u0000\u0000\u0000/6\u0003\u0006\u0003\u000006\u0003\b\u0004"+
		"\u000016\u0003\n\u0005\u000026\u0003\f\u0006\u000036\u0003\u000e\u0007"+
		"\u000046\u0003\u0010\b\u00005/\u0001\u0000\u0000\u000050\u0001\u0000\u0000"+
		"\u000051\u0001\u0000\u0000\u000052\u0001\u0000\u0000\u000053\u0001\u0000"+
		"\u0000\u000054\u0001\u0000\u0000\u00006\u0005\u0001\u0000\u0000\u0000"+
		"78\u0005\u0002\u0000\u000089\u0005&\u0000\u00009:\u0005\u0014\u0000\u0000"+
		":\u0007\u0001\u0000\u0000\u0000;<\u0007\u0000\u0000\u0000<=\u0005+\u0000"+
		"\u0000=>\u0003\u001c\u000e\u0000>?\u0005\u0014\u0000\u0000?\t\u0001\u0000"+
		"\u0000\u0000@A\u0005\u0005\u0000\u0000AG\u0005+\u0000\u0000BD\u0005\u0012"+
		"\u0000\u0000CE\u0003\u0016\u000b\u0000DC\u0001\u0000\u0000\u0000DE\u0001"+
		"\u0000\u0000\u0000EF\u0001\u0000\u0000\u0000FH\u0005\u0013\u0000\u0000"+
		"GB\u0001\u0000\u0000\u0000GH\u0001\u0000\u0000\u0000HI\u0001\u0000\u0000"+
		"\u0000IJ\u0003\u0016\u000b\u0000JN\u0005\u0010\u0000\u0000KM\u0003\u0012"+
		"\t\u0000LK\u0001\u0000\u0000\u0000MP\u0001\u0000\u0000\u0000NL\u0001\u0000"+
		"\u0000\u0000NO\u0001\u0000\u0000\u0000OQ\u0001\u0000\u0000\u0000PN\u0001"+
		"\u0000\u0000\u0000QR\u0005\u0011\u0000\u0000R`\u0001\u0000\u0000\u0000"+
		"ST\u0005\u0006\u0000\u0000TZ\u0005+\u0000\u0000UW\u0005\u0012\u0000\u0000"+
		"VX\u0003\u0016\u000b\u0000WV\u0001\u0000\u0000\u0000WX\u0001\u0000\u0000"+
		"\u0000XY\u0001\u0000\u0000\u0000Y[\u0005\u0013\u0000\u0000ZU\u0001\u0000"+
		"\u0000\u0000Z[\u0001\u0000\u0000\u0000[\\\u0001\u0000\u0000\u0000\\]\u0003"+
		"\u0016\u000b\u0000]^\u0005\u0014\u0000\u0000^`\u0001\u0000\u0000\u0000"+
		"_@\u0001\u0000\u0000\u0000_S\u0001\u0000\u0000\u0000`\u000b\u0001\u0000"+
		"\u0000\u0000am\u0003\u0014\n\u0000bc\u0005\b\u0000\u0000cd\u0003\u001a"+
		"\r\u0000de\u0005\u0017\u0000\u0000ef\u0003\u001a\r\u0000fg\u0005\u0014"+
		"\u0000\u0000gm\u0001\u0000\u0000\u0000hi\u0005\u0007\u0000\u0000ij\u0003"+
		"\u001a\r\u0000jk\u0005\u0014\u0000\u0000km\u0001\u0000\u0000\u0000la\u0001"+
		"\u0000\u0000\u0000lb\u0001\u0000\u0000\u0000lh\u0001\u0000\u0000\u0000"+
		"m\r\u0001\u0000\u0000\u0000no\u0005\n\u0000\u0000op\u0005\u0012\u0000"+
		"\u0000pq\u0005+\u0000\u0000qr\u0005\u0018\u0000\u0000rs\u0005$\u0000\u0000"+
		"st\u0005\u0013\u0000\u0000tu\u0003\f\u0006\u0000u\u000f\u0001\u0000\u0000"+
		"\u0000vw\u0005\t\u0000\u0000wx\u0003\u0018\f\u0000xy\u0005\u0014\u0000"+
		"\u0000y\u0011\u0001\u0000\u0000\u0000z\u0080\u0003\u0014\n\u0000{|\u0005"+
		"\t\u0000\u0000|}\u0003\u0016\u000b\u0000}~\u0005\u0014\u0000\u0000~\u0080"+
		"\u0001\u0000\u0000\u0000\u007fz\u0001\u0000\u0000\u0000\u007f{\u0001\u0000"+
		"\u0000\u0000\u0080\u0013\u0001\u0000\u0000\u0000\u0081\u0087\u0007\u0001"+
		"\u0000\u0000\u0082\u0084\u0005\u0012\u0000\u0000\u0083\u0085\u0003\u001e"+
		"\u000f\u0000\u0084\u0083\u0001\u0000\u0000\u0000\u0084\u0085\u0001\u0000"+
		"\u0000\u0000\u0085\u0086\u0001\u0000\u0000\u0000\u0086\u0088\u0005\u0013"+
		"\u0000\u0000\u0087\u0082\u0001\u0000\u0000\u0000\u0087\u0088\u0001\u0000"+
		"\u0000\u0000\u0088\u0089\u0001\u0000\u0000\u0000\u0089\u008a\u0003\u0018"+
		"\f\u0000\u008a\u008b\u0005\u0014\u0000\u0000\u008b\u0015\u0001\u0000\u0000"+
		"\u0000\u008c\u0091\u0005+\u0000\u0000\u008d\u008e\u0005\u0015\u0000\u0000"+
		"\u008e\u0090\u0005+\u0000\u0000\u008f\u008d\u0001\u0000\u0000\u0000\u0090"+
		"\u0093\u0001\u0000\u0000\u0000\u0091\u008f\u0001\u0000\u0000\u0000\u0091"+
		"\u0092\u0001\u0000\u0000\u0000\u0092\u0017\u0001\u0000\u0000\u0000\u0093"+
		"\u0091\u0001\u0000\u0000\u0000\u0094\u0099\u0003\u001a\r\u0000\u0095\u0096"+
		"\u0005\u0015\u0000\u0000\u0096\u0098\u0003\u001a\r\u0000\u0097\u0095\u0001"+
		"\u0000\u0000\u0000\u0098\u009b\u0001\u0000\u0000\u0000\u0099\u0097\u0001"+
		"\u0000\u0000\u0000\u0099\u009a\u0001\u0000\u0000\u0000\u009a\u0019\u0001"+
		"\u0000\u0000\u0000\u009b\u0099\u0001\u0000\u0000\u0000\u009c\u009e\u0005"+
		"+\u0000\u0000\u009d\u009f\u0003\u001c\u000e\u0000\u009e\u009d\u0001\u0000"+
		"\u0000\u0000\u009e\u009f\u0001\u0000\u0000\u0000\u009f\u001b\u0001\u0000"+
		"\u0000\u0000\u00a0\u00a1\u0005\u000e\u0000\u0000\u00a1\u00a2\u0005$\u0000"+
		"\u0000\u00a2\u00a3\u0005\u000f\u0000\u0000\u00a3\u001d\u0001\u0000\u0000"+
		"\u0000\u00a4\u00a9\u0003 \u0010\u0000\u00a5\u00a6\u0005\u0015\u0000\u0000"+
		"\u00a6\u00a8\u0003 \u0010\u0000\u00a7\u00a5\u0001\u0000\u0000\u0000\u00a8"+
		"\u00ab\u0001\u0000\u0000\u0000\u00a9\u00a7\u0001\u0000\u0000\u0000\u00a9"+
		"\u00aa\u0001\u0000\u0000\u0000\u00aa\u001f\u0001\u0000\u0000\u0000\u00ab"+
		"\u00a9\u0001\u0000\u0000\u0000\u00ac\u00ad\u0006\u0010\uffff\uffff\u0000"+
		"\u00ad\u00ae\u0005\u0012\u0000\u0000\u00ae\u00af\u0003 \u0010\u0000\u00af"+
		"\u00b0\u0005\u0013\u0000\u0000\u00b0\u00ba\u0001\u0000\u0000\u0000\u00b1"+
		"\u00b2\u0005\u001a\u0000\u0000\u00b2\u00ba\u0003 \u0010\u0006\u00b3\u00b4"+
		"\u0007\u0002\u0000\u0000\u00b4\u00b5\u0005\u0012\u0000\u0000\u00b5\u00b6"+
		"\u0003 \u0010\u0000\u00b6\u00b7\u0005\u0013\u0000\u0000\u00b7\u00ba\u0001"+
		"\u0000\u0000\u0000\u00b8\u00ba\u0007\u0003\u0000\u0000\u00b9\u00ac\u0001"+
		"\u0000\u0000\u0000\u00b9\u00b1\u0001\u0000\u0000\u0000\u00b9\u00b3\u0001"+
		"\u0000\u0000\u0000\u00b9\u00b8\u0001\u0000\u0000\u0000\u00ba\u00c6\u0001"+
		"\u0000\u0000\u0000\u00bb\u00bc\n\u0005\u0000\u0000\u00bc\u00bd\u0007\u0004"+
		"\u0000\u0000\u00bd\u00c5\u0003 \u0010\u0006\u00be\u00bf\n\u0004\u0000"+
		"\u0000\u00bf\u00c0\u0007\u0005\u0000\u0000\u00c0\u00c5\u0003 \u0010\u0005"+
		"\u00c1\u00c2\n\u0003\u0000\u0000\u00c2\u00c3\u0005\u001d\u0000\u0000\u00c3"+
		"\u00c5\u0003 \u0010\u0004\u00c4\u00bb\u0001\u0000\u0000\u0000\u00c4\u00be"+
		"\u0001\u0000\u0000\u0000\u00c4\u00c1\u0001\u0000\u0000\u0000\u00c5\u00c8"+
		"\u0001\u0000\u0000\u0000\u00c6\u00c4\u0001\u0000\u0000\u0000\u00c6\u00c7"+
		"\u0001\u0000\u0000\u0000\u00c7!\u0001\u0000\u0000\u0000\u00c8\u00c6\u0001"+
		"\u0000\u0000\u0000\u0013&5DGNWZ_l\u007f\u0084\u0087\u0091\u0099\u009e"+
		"\u00a9\u00b9\u00c4\u00c6";
	public static final ATN _ATN =
		new ATNDeserializer().deserialize(_serializedATN.toCharArray());
	static {
		_decisionToDFA = new DFA[_ATN.getNumberOfDecisions()];
		for (int i = 0; i < _ATN.getNumberOfDecisions(); i++) {
			_decisionToDFA[i] = new DFA(_ATN.getDecisionState(i), i);
		}
	}
}