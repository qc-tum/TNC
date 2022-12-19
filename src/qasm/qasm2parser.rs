// Generated from Qasm2Parser.g4 by ANTLR 4.8
#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
#![allow(nonstandard_style)]
#![allow(unused_imports)]
#![allow(unused_mut)]
#![allow(unused_braces)]
use antlr_rust::PredictionContextCache;
use antlr_rust::parser::{Parser, BaseParser, ParserRecog, ParserNodeType};
use antlr_rust::token_stream::TokenStream;
use antlr_rust::TokenSource;
use antlr_rust::parser_atn_simulator::ParserATNSimulator;
use antlr_rust::errors::*;
use antlr_rust::rule_context::{BaseRuleContext, CustomRuleContext, RuleContext};
use antlr_rust::recognizer::{Recognizer,Actions};
use antlr_rust::atn_deserializer::ATNDeserializer;
use antlr_rust::dfa::DFA;
use antlr_rust::atn::{ATN, INVALID_ALT};
use antlr_rust::error_strategy::{ErrorStrategy, DefaultErrorStrategy};
use antlr_rust::parser_rule_context::{BaseParserRuleContext, ParserRuleContext,cast,cast_mut};
use antlr_rust::tree::*;
use antlr_rust::token::{TOKEN_EOF,OwningToken,Token};
use antlr_rust::int_stream::EOF;
use antlr_rust::vocabulary::{Vocabulary,VocabularyImpl};
use antlr_rust::token_factory::{CommonTokenFactory,TokenFactory, TokenAware};
use super::qasm2parserlistener::*;
use super::qasm2parservisitor::*;

use antlr_rust::lazy_static;
use antlr_rust::{TidAble,TidExt};

use std::marker::PhantomData;
use std::sync::Arc;
use std::rc::Rc;
use std::convert::TryFrom;
use std::cell::RefCell;
use std::ops::{DerefMut, Deref};
use std::borrow::{Borrow,BorrowMut};
use std::any::{Any,TypeId};

		pub const OPENQASM:isize=1; 
		pub const INCLUDE:isize=2; 
		pub const QREG:isize=3; 
		pub const CREG:isize=4; 
		pub const GATE:isize=5; 
		pub const OPAQUE:isize=6; 
		pub const RESET:isize=7; 
		pub const MEASURE:isize=8; 
		pub const BARRIER:isize=9; 
		pub const IF:isize=10; 
		pub const PI:isize=11; 
		pub const U:isize=12; 
		pub const CX:isize=13; 
		pub const LBRACKET:isize=14; 
		pub const RBRACKET:isize=15; 
		pub const LBRACE:isize=16; 
		pub const RBRACE:isize=17; 
		pub const LPAREN:isize=18; 
		pub const RPAREN:isize=19; 
		pub const SEMICOLON:isize=20; 
		pub const COMMA:isize=21; 
		pub const DOT:isize=22; 
		pub const ARROW:isize=23; 
		pub const EQUALS:isize=24; 
		pub const PLUS:isize=25; 
		pub const MINUS:isize=26; 
		pub const ASTERISK:isize=27; 
		pub const SLASH:isize=28; 
		pub const CARET:isize=29; 
		pub const SIN:isize=30; 
		pub const COS:isize=31; 
		pub const TAN:isize=32; 
		pub const EXP:isize=33; 
		pub const LN:isize=34; 
		pub const SQRT:isize=35; 
		pub const Integer:isize=36; 
		pub const Float:isize=37; 
		pub const StringLiteral:isize=38; 
		pub const Whitespace:isize=39; 
		pub const Newline:isize=40; 
		pub const LineComment:isize=41; 
		pub const BlockComment:isize=42; 
		pub const Identifier:isize=43; 
		pub const VERSION_IDENTIFER_WHITESPACE:isize=44; 
		pub const VersionSpecifier:isize=45;
	pub const RULE_program:usize = 0; 
	pub const RULE_version:usize = 1; 
	pub const RULE_statement:usize = 2; 
	pub const RULE_includeStatement:usize = 3; 
	pub const RULE_declaration:usize = 4; 
	pub const RULE_gateDeclaration:usize = 5; 
	pub const RULE_quantumOperation:usize = 6; 
	pub const RULE_ifStatement:usize = 7; 
	pub const RULE_barrier:usize = 8; 
	pub const RULE_bodyStatement:usize = 9; 
	pub const RULE_gateCall:usize = 10; 
	pub const RULE_idlist:usize = 11; 
	pub const RULE_mixedlist:usize = 12; 
	pub const RULE_argument:usize = 13; 
	pub const RULE_designator:usize = 14; 
	pub const RULE_explist:usize = 15; 
	pub const RULE_exp:usize = 16;
	pub const ruleNames: [&'static str; 17] =  [
		"program", "version", "statement", "includeStatement", "declaration", 
		"gateDeclaration", "quantumOperation", "ifStatement", "barrier", "bodyStatement", 
		"gateCall", "idlist", "mixedlist", "argument", "designator", "explist", 
		"exp"
	];


	pub const _LITERAL_NAMES: [Option<&'static str>;36] = [
		None, Some("'OPENQASM'"), Some("'include'"), Some("'qreg'"), Some("'creg'"), 
		Some("'gate'"), Some("'opaque'"), Some("'reset'"), Some("'measure'"), 
		Some("'barrier'"), Some("'if'"), Some("'pi'"), Some("'U'"), Some("'CX'"), 
		Some("'['"), Some("']'"), Some("'{'"), Some("'}'"), Some("'('"), Some("')'"), 
		Some("';'"), Some("','"), Some("'.'"), Some("'->'"), Some("'=='"), Some("'+'"), 
		Some("'-'"), Some("'*'"), Some("'/'"), Some("'^'"), Some("'sin'"), Some("'cos'"), 
		Some("'tan'"), Some("'exp'"), Some("'ln'"), Some("'sqrt'")
	];
	pub const _SYMBOLIC_NAMES: [Option<&'static str>;46]  = [
		None, Some("OPENQASM"), Some("INCLUDE"), Some("QREG"), Some("CREG"), Some("GATE"), 
		Some("OPAQUE"), Some("RESET"), Some("MEASURE"), Some("BARRIER"), Some("IF"), 
		Some("PI"), Some("U"), Some("CX"), Some("LBRACKET"), Some("RBRACKET"), 
		Some("LBRACE"), Some("RBRACE"), Some("LPAREN"), Some("RPAREN"), Some("SEMICOLON"), 
		Some("COMMA"), Some("DOT"), Some("ARROW"), Some("EQUALS"), Some("PLUS"), 
		Some("MINUS"), Some("ASTERISK"), Some("SLASH"), Some("CARET"), Some("SIN"), 
		Some("COS"), Some("TAN"), Some("EXP"), Some("LN"), Some("SQRT"), Some("Integer"), 
		Some("Float"), Some("StringLiteral"), Some("Whitespace"), Some("Newline"), 
		Some("LineComment"), Some("BlockComment"), Some("Identifier"), Some("VERSION_IDENTIFER_WHITESPACE"), 
		Some("VersionSpecifier")
	];
	lazy_static!{
	    static ref _shared_context_cache: Arc<PredictionContextCache> = Arc::new(PredictionContextCache::new());
		static ref VOCABULARY: Box<dyn Vocabulary> = Box::new(VocabularyImpl::new(_LITERAL_NAMES.iter(), _SYMBOLIC_NAMES.iter(), None));
	}


type BaseParserType<'input, I> =
	BaseParser<'input,Qasm2ParserExt<'input>, I, Qasm2ParserContextType , dyn Qasm2ParserListener<'input> + 'input >;

type TokenType<'input> = <LocalTokenFactory<'input> as TokenFactory<'input>>::Tok;
pub type LocalTokenFactory<'input> = CommonTokenFactory;

pub type Qasm2ParserTreeWalker<'input,'a> =
	ParseTreeWalker<'input, 'a, Qasm2ParserContextType , dyn Qasm2ParserListener<'input> + 'a>;

/// Parser for Qasm2Parser grammar
pub struct Qasm2Parser<'input,I,H>
where
    I: TokenStream<'input, TF = LocalTokenFactory<'input> > + TidAble<'input>,
    H: ErrorStrategy<'input,BaseParserType<'input,I>>
{
	base:BaseParserType<'input,I>,
	interpreter:Arc<ParserATNSimulator>,
	_shared_context_cache: Box<PredictionContextCache>,
    pub err_handler: H,
}

impl<'input, I, H> Qasm2Parser<'input, I, H>
where
    I: TokenStream<'input, TF = LocalTokenFactory<'input> > + TidAble<'input>,
    H: ErrorStrategy<'input,BaseParserType<'input,I>>
{
	pub fn get_serialized_atn() -> &'static str { _serializedATN }

    pub fn set_error_strategy(&mut self, strategy: H) {
        self.err_handler = strategy
    }

    pub fn with_strategy(input: I, strategy: H) -> Self {
		antlr_rust::recognizer::check_version("0","3");
		let interpreter = Arc::new(ParserATNSimulator::new(
			_ATN.clone(),
			_decision_to_DFA.clone(),
			_shared_context_cache.clone(),
		));
		Self {
			base: BaseParser::new_base_parser(
				input,
				Arc::clone(&interpreter),
				Qasm2ParserExt{
					_pd: Default::default(),
				}
			),
			interpreter,
            _shared_context_cache: Box::new(PredictionContextCache::new()),
            err_handler: strategy,
        }
    }

}

type DynStrategy<'input,I> = Box<dyn ErrorStrategy<'input,BaseParserType<'input,I>> + 'input>;

impl<'input, I> Qasm2Parser<'input, I, DynStrategy<'input,I>>
where
    I: TokenStream<'input, TF = LocalTokenFactory<'input> > + TidAble<'input>,
{
    pub fn with_dyn_strategy(input: I) -> Self{
    	Self::with_strategy(input,Box::new(DefaultErrorStrategy::new()))
    }
}

impl<'input, I> Qasm2Parser<'input, I, DefaultErrorStrategy<'input,Qasm2ParserContextType>>
where
    I: TokenStream<'input, TF = LocalTokenFactory<'input> > + TidAble<'input>,
{
    pub fn new(input: I) -> Self{
    	Self::with_strategy(input,DefaultErrorStrategy::new())
    }
}

/// Trait for monomorphized trait object that corresponds to the nodes of parse tree generated for Qasm2Parser
pub trait Qasm2ParserContext<'input>:
	for<'x> Listenable<dyn Qasm2ParserListener<'input> + 'x > + 
	for<'x> Visitable<dyn Qasm2ParserVisitor<'input> + 'x > + 
	ParserRuleContext<'input, TF=LocalTokenFactory<'input>, Ctx=Qasm2ParserContextType>
{}

antlr_rust::coerce_from!{ 'input : Qasm2ParserContext<'input> }

impl<'input, 'x, T> VisitableDyn<T> for dyn Qasm2ParserContext<'input> + 'input
where
    T: Qasm2ParserVisitor<'input> + 'x,
{
    fn accept_dyn(&self, visitor: &mut T) {
        self.accept(visitor as &mut (dyn Qasm2ParserVisitor<'input> + 'x))
    }
}

impl<'input> Qasm2ParserContext<'input> for TerminalNode<'input,Qasm2ParserContextType> {}
impl<'input> Qasm2ParserContext<'input> for ErrorNode<'input,Qasm2ParserContextType> {}

antlr_rust::tid! { impl<'input> TidAble<'input> for dyn Qasm2ParserContext<'input> + 'input }

antlr_rust::tid! { impl<'input> TidAble<'input> for dyn Qasm2ParserListener<'input> + 'input }

pub struct Qasm2ParserContextType;
antlr_rust::tid!{Qasm2ParserContextType}

impl<'input> ParserNodeType<'input> for Qasm2ParserContextType{
	type TF = LocalTokenFactory<'input>;
	type Type = dyn Qasm2ParserContext<'input> + 'input;
}

impl<'input, I, H> Deref for Qasm2Parser<'input, I, H>
where
    I: TokenStream<'input, TF = LocalTokenFactory<'input> > + TidAble<'input>,
    H: ErrorStrategy<'input,BaseParserType<'input,I>>
{
    type Target = BaseParserType<'input,I>;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl<'input, I, H> DerefMut for Qasm2Parser<'input, I, H>
where
    I: TokenStream<'input, TF = LocalTokenFactory<'input> > + TidAble<'input>,
    H: ErrorStrategy<'input,BaseParserType<'input,I>>
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.base
    }
}

pub struct Qasm2ParserExt<'input>{
	_pd: PhantomData<&'input str>,
}

impl<'input> Qasm2ParserExt<'input>{
}
antlr_rust::tid! { Qasm2ParserExt<'a> }

impl<'input> TokenAware<'input> for Qasm2ParserExt<'input>{
	type TF = LocalTokenFactory<'input>;
}

impl<'input,I: TokenStream<'input, TF = LocalTokenFactory<'input> > + TidAble<'input>> ParserRecog<'input, BaseParserType<'input,I>> for Qasm2ParserExt<'input>{}

impl<'input,I: TokenStream<'input, TF = LocalTokenFactory<'input> > + TidAble<'input>> Actions<'input, BaseParserType<'input,I>> for Qasm2ParserExt<'input>{
	fn get_grammar_file_name(&self) -> & str{ "Qasm2Parser.g4"}

   	fn get_rule_names(&self) -> &[& str] {&ruleNames}

   	fn get_vocabulary(&self) -> &dyn Vocabulary { &**VOCABULARY }
	fn sempred(_localctx: Option<&(dyn Qasm2ParserContext<'input> + 'input)>, rule_index: isize, pred_index: isize,
			   recog:&mut BaseParserType<'input,I>
	)->bool{
		match rule_index {
					16 => Qasm2Parser::<'input,I,_>::exp_sempred(_localctx.and_then(|x|x.downcast_ref()), pred_index, recog),
			_ => true
		}
	}
}

impl<'input, I> Qasm2Parser<'input, I, DefaultErrorStrategy<'input,Qasm2ParserContextType>>
where
    I: TokenStream<'input, TF = LocalTokenFactory<'input> > + TidAble<'input>,
{
	fn exp_sempred(_localctx: Option<&ExpContext<'input>>, pred_index:isize,
						recog:&mut <Self as Deref>::Target
		) -> bool {
		match pred_index {
				0=>{
					recog.precpred(None, 5)
				}
				1=>{
					recog.precpred(None, 4)
				}
				2=>{
					recog.precpred(None, 3)
				}
			_ => true
		}
	}
}
//------------------- program ----------------
pub type ProgramContextAll<'input> = ProgramContext<'input>;


pub type ProgramContext<'input> = BaseParserRuleContext<'input,ProgramContextExt<'input>>;

#[derive(Clone)]
pub struct ProgramContextExt<'input>{
ph:PhantomData<&'input str>
}

impl<'input> Qasm2ParserContext<'input> for ProgramContext<'input>{}

impl<'input,'a> Listenable<dyn Qasm2ParserListener<'input> + 'a> for ProgramContext<'input>{
		fn enter(&self,listener: &mut (dyn Qasm2ParserListener<'input> + 'a)) {
			listener.enter_every_rule(self);
			listener.enter_program(self);
		}
		fn exit(&self,listener: &mut (dyn Qasm2ParserListener<'input> + 'a)) {
			listener.exit_program(self);
			listener.exit_every_rule(self);
		}
}

impl<'input,'a> Visitable<dyn Qasm2ParserVisitor<'input> + 'a> for ProgramContext<'input>{
	fn accept(&self,visitor: &mut (dyn Qasm2ParserVisitor<'input> + 'a)) {
		visitor.visit_program(self);
	}
}

impl<'input> CustomRuleContext<'input> for ProgramContextExt<'input>{
	type TF = LocalTokenFactory<'input>;
	type Ctx = Qasm2ParserContextType;
	fn get_rule_index(&self) -> usize { RULE_program }
	//fn type_rule_index() -> usize where Self: Sized { RULE_program }
}
antlr_rust::tid!{ProgramContextExt<'a>}

impl<'input> ProgramContextExt<'input>{
	fn new(parent: Option<Rc<dyn Qasm2ParserContext<'input> + 'input > >, invoking_state: isize) -> Rc<ProgramContextAll<'input>> {
		Rc::new(
			BaseParserRuleContext::new_parser_ctx(parent, invoking_state,ProgramContextExt{
				ph:PhantomData
			}),
		)
	}
}

pub trait ProgramContextAttrs<'input>: Qasm2ParserContext<'input> + BorrowMut<ProgramContextExt<'input>>{

fn version(&self) -> Option<Rc<VersionContextAll<'input>>> where Self:Sized{
	self.child_of_type(0)
}
/// Retrieves first TerminalNode corresponding to token EOF
/// Returns `None` if there is no child corresponding to token EOF
fn EOF(&self) -> Option<Rc<TerminalNode<'input,Qasm2ParserContextType>>> where Self:Sized{
	self.get_token(EOF, 0)
}
fn statement_all(&self) ->  Vec<Rc<StatementContextAll<'input>>> where Self:Sized{
	self.children_of_type()
}
fn statement(&self, i: usize) -> Option<Rc<StatementContextAll<'input>>> where Self:Sized{
	self.child_of_type(i)
}

}

impl<'input> ProgramContextAttrs<'input> for ProgramContext<'input>{}

impl<'input, I, H> Qasm2Parser<'input, I, H>
where
    I: TokenStream<'input, TF = LocalTokenFactory<'input> > + TidAble<'input>,
    H: ErrorStrategy<'input,BaseParserType<'input,I>>
{
	pub fn program(&mut self,)
	-> Result<Rc<ProgramContextAll<'input>>,ANTLRError> {
		let mut recog = self;
		let _parentctx = recog.ctx.take();
		let mut _localctx = ProgramContextExt::new(_parentctx.clone(), recog.base.get_state());
        recog.base.enter_rule(_localctx.clone(), 0, RULE_program);
        let mut _localctx: Rc<ProgramContextAll> = _localctx;
		let mut _la: isize = -1;
		let result: Result<(), ANTLRError> = (|| {

			//recog.base.enter_outer_alt(_localctx.clone(), 1);
			recog.base.enter_outer_alt(None, 1);
			{
			/*InvokeRule version*/
			recog.base.set_state(34);
			recog.version()?;

			recog.base.set_state(38);
			recog.err_handler.sync(&mut recog.base)?;
			_la = recog.base.input.la(1);
			while (((_la) & !0x3f) == 0 && ((1usize << _la) & ((1usize << INCLUDE) | (1usize << QREG) | (1usize << CREG) | (1usize << GATE) | (1usize << OPAQUE) | (1usize << RESET) | (1usize << MEASURE) | (1usize << BARRIER) | (1usize << IF) | (1usize << U) | (1usize << CX))) != 0) || _la==Identifier {
				{
				{
				/*InvokeRule statement*/
				recog.base.set_state(35);
				recog.statement()?;

				}
				}
				recog.base.set_state(40);
				recog.err_handler.sync(&mut recog.base)?;
				_la = recog.base.input.la(1);
			}
			recog.base.set_state(41);
			recog.base.match_token(EOF,&mut recog.err_handler)?;

			}
			Ok(())
		})();
		match result {
		Ok(_)=>{},
        Err(e @ ANTLRError::FallThrough(_)) => return Err(e),
		Err(ref re) => {
				//_localctx.exception = re;
				recog.err_handler.report_error(&mut recog.base, re);
				recog.err_handler.recover(&mut recog.base, re)?;
			}
		}
		recog.base.exit_rule();

		Ok(_localctx)
	}
}
//------------------- version ----------------
pub type VersionContextAll<'input> = VersionContext<'input>;


pub type VersionContext<'input> = BaseParserRuleContext<'input,VersionContextExt<'input>>;

#[derive(Clone)]
pub struct VersionContextExt<'input>{
ph:PhantomData<&'input str>
}

impl<'input> Qasm2ParserContext<'input> for VersionContext<'input>{}

impl<'input,'a> Listenable<dyn Qasm2ParserListener<'input> + 'a> for VersionContext<'input>{
		fn enter(&self,listener: &mut (dyn Qasm2ParserListener<'input> + 'a)) {
			listener.enter_every_rule(self);
			listener.enter_version(self);
		}
		fn exit(&self,listener: &mut (dyn Qasm2ParserListener<'input> + 'a)) {
			listener.exit_version(self);
			listener.exit_every_rule(self);
		}
}

impl<'input,'a> Visitable<dyn Qasm2ParserVisitor<'input> + 'a> for VersionContext<'input>{
	fn accept(&self,visitor: &mut (dyn Qasm2ParserVisitor<'input> + 'a)) {
		visitor.visit_version(self);
	}
}

impl<'input> CustomRuleContext<'input> for VersionContextExt<'input>{
	type TF = LocalTokenFactory<'input>;
	type Ctx = Qasm2ParserContextType;
	fn get_rule_index(&self) -> usize { RULE_version }
	//fn type_rule_index() -> usize where Self: Sized { RULE_version }
}
antlr_rust::tid!{VersionContextExt<'a>}

impl<'input> VersionContextExt<'input>{
	fn new(parent: Option<Rc<dyn Qasm2ParserContext<'input> + 'input > >, invoking_state: isize) -> Rc<VersionContextAll<'input>> {
		Rc::new(
			BaseParserRuleContext::new_parser_ctx(parent, invoking_state,VersionContextExt{
				ph:PhantomData
			}),
		)
	}
}

pub trait VersionContextAttrs<'input>: Qasm2ParserContext<'input> + BorrowMut<VersionContextExt<'input>>{

/// Retrieves first TerminalNode corresponding to token OPENQASM
/// Returns `None` if there is no child corresponding to token OPENQASM
fn OPENQASM(&self) -> Option<Rc<TerminalNode<'input,Qasm2ParserContextType>>> where Self:Sized{
	self.get_token(OPENQASM, 0)
}
/// Retrieves first TerminalNode corresponding to token VersionSpecifier
/// Returns `None` if there is no child corresponding to token VersionSpecifier
fn VersionSpecifier(&self) -> Option<Rc<TerminalNode<'input,Qasm2ParserContextType>>> where Self:Sized{
	self.get_token(VersionSpecifier, 0)
}
/// Retrieves first TerminalNode corresponding to token SEMICOLON
/// Returns `None` if there is no child corresponding to token SEMICOLON
fn SEMICOLON(&self) -> Option<Rc<TerminalNode<'input,Qasm2ParserContextType>>> where Self:Sized{
	self.get_token(SEMICOLON, 0)
}

}

impl<'input> VersionContextAttrs<'input> for VersionContext<'input>{}

impl<'input, I, H> Qasm2Parser<'input, I, H>
where
    I: TokenStream<'input, TF = LocalTokenFactory<'input> > + TidAble<'input>,
    H: ErrorStrategy<'input,BaseParserType<'input,I>>
{
	pub fn version(&mut self,)
	-> Result<Rc<VersionContextAll<'input>>,ANTLRError> {
		let mut recog = self;
		let _parentctx = recog.ctx.take();
		let mut _localctx = VersionContextExt::new(_parentctx.clone(), recog.base.get_state());
        recog.base.enter_rule(_localctx.clone(), 2, RULE_version);
        let mut _localctx: Rc<VersionContextAll> = _localctx;
		let result: Result<(), ANTLRError> = (|| {

			//recog.base.enter_outer_alt(_localctx.clone(), 1);
			recog.base.enter_outer_alt(None, 1);
			{
			recog.base.set_state(43);
			recog.base.match_token(OPENQASM,&mut recog.err_handler)?;

			recog.base.set_state(44);
			recog.base.match_token(VersionSpecifier,&mut recog.err_handler)?;

			recog.base.set_state(45);
			recog.base.match_token(SEMICOLON,&mut recog.err_handler)?;

			}
			Ok(())
		})();
		match result {
		Ok(_)=>{},
        Err(e @ ANTLRError::FallThrough(_)) => return Err(e),
		Err(ref re) => {
				//_localctx.exception = re;
				recog.err_handler.report_error(&mut recog.base, re);
				recog.err_handler.recover(&mut recog.base, re)?;
			}
		}
		recog.base.exit_rule();

		Ok(_localctx)
	}
}
//------------------- statement ----------------
pub type StatementContextAll<'input> = StatementContext<'input>;


pub type StatementContext<'input> = BaseParserRuleContext<'input,StatementContextExt<'input>>;

#[derive(Clone)]
pub struct StatementContextExt<'input>{
ph:PhantomData<&'input str>
}

impl<'input> Qasm2ParserContext<'input> for StatementContext<'input>{}

impl<'input,'a> Listenable<dyn Qasm2ParserListener<'input> + 'a> for StatementContext<'input>{
		fn enter(&self,listener: &mut (dyn Qasm2ParserListener<'input> + 'a)) {
			listener.enter_every_rule(self);
			listener.enter_statement(self);
		}
		fn exit(&self,listener: &mut (dyn Qasm2ParserListener<'input> + 'a)) {
			listener.exit_statement(self);
			listener.exit_every_rule(self);
		}
}

impl<'input,'a> Visitable<dyn Qasm2ParserVisitor<'input> + 'a> for StatementContext<'input>{
	fn accept(&self,visitor: &mut (dyn Qasm2ParserVisitor<'input> + 'a)) {
		visitor.visit_statement(self);
	}
}

impl<'input> CustomRuleContext<'input> for StatementContextExt<'input>{
	type TF = LocalTokenFactory<'input>;
	type Ctx = Qasm2ParserContextType;
	fn get_rule_index(&self) -> usize { RULE_statement }
	//fn type_rule_index() -> usize where Self: Sized { RULE_statement }
}
antlr_rust::tid!{StatementContextExt<'a>}

impl<'input> StatementContextExt<'input>{
	fn new(parent: Option<Rc<dyn Qasm2ParserContext<'input> + 'input > >, invoking_state: isize) -> Rc<StatementContextAll<'input>> {
		Rc::new(
			BaseParserRuleContext::new_parser_ctx(parent, invoking_state,StatementContextExt{
				ph:PhantomData
			}),
		)
	}
}

pub trait StatementContextAttrs<'input>: Qasm2ParserContext<'input> + BorrowMut<StatementContextExt<'input>>{

fn includeStatement(&self) -> Option<Rc<IncludeStatementContextAll<'input>>> where Self:Sized{
	self.child_of_type(0)
}
fn declaration(&self) -> Option<Rc<DeclarationContextAll<'input>>> where Self:Sized{
	self.child_of_type(0)
}
fn gateDeclaration(&self) -> Option<Rc<GateDeclarationContextAll<'input>>> where Self:Sized{
	self.child_of_type(0)
}
fn quantumOperation(&self) -> Option<Rc<QuantumOperationContextAll<'input>>> where Self:Sized{
	self.child_of_type(0)
}
fn ifStatement(&self) -> Option<Rc<IfStatementContextAll<'input>>> where Self:Sized{
	self.child_of_type(0)
}
fn barrier(&self) -> Option<Rc<BarrierContextAll<'input>>> where Self:Sized{
	self.child_of_type(0)
}

}

impl<'input> StatementContextAttrs<'input> for StatementContext<'input>{}

impl<'input, I, H> Qasm2Parser<'input, I, H>
where
    I: TokenStream<'input, TF = LocalTokenFactory<'input> > + TidAble<'input>,
    H: ErrorStrategy<'input,BaseParserType<'input,I>>
{
	pub fn statement(&mut self,)
	-> Result<Rc<StatementContextAll<'input>>,ANTLRError> {
		let mut recog = self;
		let _parentctx = recog.ctx.take();
		let mut _localctx = StatementContextExt::new(_parentctx.clone(), recog.base.get_state());
        recog.base.enter_rule(_localctx.clone(), 4, RULE_statement);
        let mut _localctx: Rc<StatementContextAll> = _localctx;
		let result: Result<(), ANTLRError> = (|| {

			recog.base.set_state(53);
			recog.err_handler.sync(&mut recog.base)?;
			match recog.base.input.la(1) {
			 INCLUDE 
				=> {
					//recog.base.enter_outer_alt(_localctx.clone(), 1);
					recog.base.enter_outer_alt(None, 1);
					{
					/*InvokeRule includeStatement*/
					recog.base.set_state(47);
					recog.includeStatement()?;

					}
				}

			 QREG | CREG 
				=> {
					//recog.base.enter_outer_alt(_localctx.clone(), 2);
					recog.base.enter_outer_alt(None, 2);
					{
					/*InvokeRule declaration*/
					recog.base.set_state(48);
					recog.declaration()?;

					}
				}

			 GATE | OPAQUE 
				=> {
					//recog.base.enter_outer_alt(_localctx.clone(), 3);
					recog.base.enter_outer_alt(None, 3);
					{
					/*InvokeRule gateDeclaration*/
					recog.base.set_state(49);
					recog.gateDeclaration()?;

					}
				}

			 RESET | MEASURE | U | CX | Identifier 
				=> {
					//recog.base.enter_outer_alt(_localctx.clone(), 4);
					recog.base.enter_outer_alt(None, 4);
					{
					/*InvokeRule quantumOperation*/
					recog.base.set_state(50);
					recog.quantumOperation()?;

					}
				}

			 IF 
				=> {
					//recog.base.enter_outer_alt(_localctx.clone(), 5);
					recog.base.enter_outer_alt(None, 5);
					{
					/*InvokeRule ifStatement*/
					recog.base.set_state(51);
					recog.ifStatement()?;

					}
				}

			 BARRIER 
				=> {
					//recog.base.enter_outer_alt(_localctx.clone(), 6);
					recog.base.enter_outer_alt(None, 6);
					{
					/*InvokeRule barrier*/
					recog.base.set_state(52);
					recog.barrier()?;

					}
				}

				_ => Err(ANTLRError::NoAltError(NoViableAltError::new(&mut recog.base)))?
			}
			Ok(())
		})();
		match result {
		Ok(_)=>{},
        Err(e @ ANTLRError::FallThrough(_)) => return Err(e),
		Err(ref re) => {
				//_localctx.exception = re;
				recog.err_handler.report_error(&mut recog.base, re);
				recog.err_handler.recover(&mut recog.base, re)?;
			}
		}
		recog.base.exit_rule();

		Ok(_localctx)
	}
}
//------------------- includeStatement ----------------
pub type IncludeStatementContextAll<'input> = IncludeStatementContext<'input>;


pub type IncludeStatementContext<'input> = BaseParserRuleContext<'input,IncludeStatementContextExt<'input>>;

#[derive(Clone)]
pub struct IncludeStatementContextExt<'input>{
ph:PhantomData<&'input str>
}

impl<'input> Qasm2ParserContext<'input> for IncludeStatementContext<'input>{}

impl<'input,'a> Listenable<dyn Qasm2ParserListener<'input> + 'a> for IncludeStatementContext<'input>{
		fn enter(&self,listener: &mut (dyn Qasm2ParserListener<'input> + 'a)) {
			listener.enter_every_rule(self);
			listener.enter_includeStatement(self);
		}
		fn exit(&self,listener: &mut (dyn Qasm2ParserListener<'input> + 'a)) {
			listener.exit_includeStatement(self);
			listener.exit_every_rule(self);
		}
}

impl<'input,'a> Visitable<dyn Qasm2ParserVisitor<'input> + 'a> for IncludeStatementContext<'input>{
	fn accept(&self,visitor: &mut (dyn Qasm2ParserVisitor<'input> + 'a)) {
		visitor.visit_includeStatement(self);
	}
}

impl<'input> CustomRuleContext<'input> for IncludeStatementContextExt<'input>{
	type TF = LocalTokenFactory<'input>;
	type Ctx = Qasm2ParserContextType;
	fn get_rule_index(&self) -> usize { RULE_includeStatement }
	//fn type_rule_index() -> usize where Self: Sized { RULE_includeStatement }
}
antlr_rust::tid!{IncludeStatementContextExt<'a>}

impl<'input> IncludeStatementContextExt<'input>{
	fn new(parent: Option<Rc<dyn Qasm2ParserContext<'input> + 'input > >, invoking_state: isize) -> Rc<IncludeStatementContextAll<'input>> {
		Rc::new(
			BaseParserRuleContext::new_parser_ctx(parent, invoking_state,IncludeStatementContextExt{
				ph:PhantomData
			}),
		)
	}
}

pub trait IncludeStatementContextAttrs<'input>: Qasm2ParserContext<'input> + BorrowMut<IncludeStatementContextExt<'input>>{

/// Retrieves first TerminalNode corresponding to token INCLUDE
/// Returns `None` if there is no child corresponding to token INCLUDE
fn INCLUDE(&self) -> Option<Rc<TerminalNode<'input,Qasm2ParserContextType>>> where Self:Sized{
	self.get_token(INCLUDE, 0)
}
/// Retrieves first TerminalNode corresponding to token StringLiteral
/// Returns `None` if there is no child corresponding to token StringLiteral
fn StringLiteral(&self) -> Option<Rc<TerminalNode<'input,Qasm2ParserContextType>>> where Self:Sized{
	self.get_token(StringLiteral, 0)
}
/// Retrieves first TerminalNode corresponding to token SEMICOLON
/// Returns `None` if there is no child corresponding to token SEMICOLON
fn SEMICOLON(&self) -> Option<Rc<TerminalNode<'input,Qasm2ParserContextType>>> where Self:Sized{
	self.get_token(SEMICOLON, 0)
}

}

impl<'input> IncludeStatementContextAttrs<'input> for IncludeStatementContext<'input>{}

impl<'input, I, H> Qasm2Parser<'input, I, H>
where
    I: TokenStream<'input, TF = LocalTokenFactory<'input> > + TidAble<'input>,
    H: ErrorStrategy<'input,BaseParserType<'input,I>>
{
	pub fn includeStatement(&mut self,)
	-> Result<Rc<IncludeStatementContextAll<'input>>,ANTLRError> {
		let mut recog = self;
		let _parentctx = recog.ctx.take();
		let mut _localctx = IncludeStatementContextExt::new(_parentctx.clone(), recog.base.get_state());
        recog.base.enter_rule(_localctx.clone(), 6, RULE_includeStatement);
        let mut _localctx: Rc<IncludeStatementContextAll> = _localctx;
		let result: Result<(), ANTLRError> = (|| {

			//recog.base.enter_outer_alt(_localctx.clone(), 1);
			recog.base.enter_outer_alt(None, 1);
			{
			recog.base.set_state(55);
			recog.base.match_token(INCLUDE,&mut recog.err_handler)?;

			recog.base.set_state(56);
			recog.base.match_token(StringLiteral,&mut recog.err_handler)?;

			recog.base.set_state(57);
			recog.base.match_token(SEMICOLON,&mut recog.err_handler)?;

			}
			Ok(())
		})();
		match result {
		Ok(_)=>{},
        Err(e @ ANTLRError::FallThrough(_)) => return Err(e),
		Err(ref re) => {
				//_localctx.exception = re;
				recog.err_handler.report_error(&mut recog.base, re);
				recog.err_handler.recover(&mut recog.base, re)?;
			}
		}
		recog.base.exit_rule();

		Ok(_localctx)
	}
}
//------------------- declaration ----------------
pub type DeclarationContextAll<'input> = DeclarationContext<'input>;


pub type DeclarationContext<'input> = BaseParserRuleContext<'input,DeclarationContextExt<'input>>;

#[derive(Clone)]
pub struct DeclarationContextExt<'input>{
ph:PhantomData<&'input str>
}

impl<'input> Qasm2ParserContext<'input> for DeclarationContext<'input>{}

impl<'input,'a> Listenable<dyn Qasm2ParserListener<'input> + 'a> for DeclarationContext<'input>{
		fn enter(&self,listener: &mut (dyn Qasm2ParserListener<'input> + 'a)) {
			listener.enter_every_rule(self);
			listener.enter_declaration(self);
		}
		fn exit(&self,listener: &mut (dyn Qasm2ParserListener<'input> + 'a)) {
			listener.exit_declaration(self);
			listener.exit_every_rule(self);
		}
}

impl<'input,'a> Visitable<dyn Qasm2ParserVisitor<'input> + 'a> for DeclarationContext<'input>{
	fn accept(&self,visitor: &mut (dyn Qasm2ParserVisitor<'input> + 'a)) {
		visitor.visit_declaration(self);
	}
}

impl<'input> CustomRuleContext<'input> for DeclarationContextExt<'input>{
	type TF = LocalTokenFactory<'input>;
	type Ctx = Qasm2ParserContextType;
	fn get_rule_index(&self) -> usize { RULE_declaration }
	//fn type_rule_index() -> usize where Self: Sized { RULE_declaration }
}
antlr_rust::tid!{DeclarationContextExt<'a>}

impl<'input> DeclarationContextExt<'input>{
	fn new(parent: Option<Rc<dyn Qasm2ParserContext<'input> + 'input > >, invoking_state: isize) -> Rc<DeclarationContextAll<'input>> {
		Rc::new(
			BaseParserRuleContext::new_parser_ctx(parent, invoking_state,DeclarationContextExt{
				ph:PhantomData
			}),
		)
	}
}

pub trait DeclarationContextAttrs<'input>: Qasm2ParserContext<'input> + BorrowMut<DeclarationContextExt<'input>>{

/// Retrieves first TerminalNode corresponding to token Identifier
/// Returns `None` if there is no child corresponding to token Identifier
fn Identifier(&self) -> Option<Rc<TerminalNode<'input,Qasm2ParserContextType>>> where Self:Sized{
	self.get_token(Identifier, 0)
}
fn designator(&self) -> Option<Rc<DesignatorContextAll<'input>>> where Self:Sized{
	self.child_of_type(0)
}
/// Retrieves first TerminalNode corresponding to token SEMICOLON
/// Returns `None` if there is no child corresponding to token SEMICOLON
fn SEMICOLON(&self) -> Option<Rc<TerminalNode<'input,Qasm2ParserContextType>>> where Self:Sized{
	self.get_token(SEMICOLON, 0)
}
/// Retrieves first TerminalNode corresponding to token QREG
/// Returns `None` if there is no child corresponding to token QREG
fn QREG(&self) -> Option<Rc<TerminalNode<'input,Qasm2ParserContextType>>> where Self:Sized{
	self.get_token(QREG, 0)
}
/// Retrieves first TerminalNode corresponding to token CREG
/// Returns `None` if there is no child corresponding to token CREG
fn CREG(&self) -> Option<Rc<TerminalNode<'input,Qasm2ParserContextType>>> where Self:Sized{
	self.get_token(CREG, 0)
}

}

impl<'input> DeclarationContextAttrs<'input> for DeclarationContext<'input>{}

impl<'input, I, H> Qasm2Parser<'input, I, H>
where
    I: TokenStream<'input, TF = LocalTokenFactory<'input> > + TidAble<'input>,
    H: ErrorStrategy<'input,BaseParserType<'input,I>>
{
	pub fn declaration(&mut self,)
	-> Result<Rc<DeclarationContextAll<'input>>,ANTLRError> {
		let mut recog = self;
		let _parentctx = recog.ctx.take();
		let mut _localctx = DeclarationContextExt::new(_parentctx.clone(), recog.base.get_state());
        recog.base.enter_rule(_localctx.clone(), 8, RULE_declaration);
        let mut _localctx: Rc<DeclarationContextAll> = _localctx;
		let mut _la: isize = -1;
		let result: Result<(), ANTLRError> = (|| {

			//recog.base.enter_outer_alt(_localctx.clone(), 1);
			recog.base.enter_outer_alt(None, 1);
			{
			recog.base.set_state(59);
			_la = recog.base.input.la(1);
			if { !(_la==QREG || _la==CREG) } {
				recog.err_handler.recover_inline(&mut recog.base)?;

			}
			else {
				if  recog.base.input.la(1)==TOKEN_EOF { recog.base.matched_eof = true };
				recog.err_handler.report_match(&mut recog.base);
				recog.base.consume(&mut recog.err_handler);
			}
			recog.base.set_state(60);
			recog.base.match_token(Identifier,&mut recog.err_handler)?;

			/*InvokeRule designator*/
			recog.base.set_state(61);
			recog.designator()?;

			recog.base.set_state(62);
			recog.base.match_token(SEMICOLON,&mut recog.err_handler)?;

			}
			Ok(())
		})();
		match result {
		Ok(_)=>{},
        Err(e @ ANTLRError::FallThrough(_)) => return Err(e),
		Err(ref re) => {
				//_localctx.exception = re;
				recog.err_handler.report_error(&mut recog.base, re);
				recog.err_handler.recover(&mut recog.base, re)?;
			}
		}
		recog.base.exit_rule();

		Ok(_localctx)
	}
}
//------------------- gateDeclaration ----------------
pub type GateDeclarationContextAll<'input> = GateDeclarationContext<'input>;


pub type GateDeclarationContext<'input> = BaseParserRuleContext<'input,GateDeclarationContextExt<'input>>;

#[derive(Clone)]
pub struct GateDeclarationContextExt<'input>{
	pub params: Option<Rc<IdlistContextAll<'input>>>,
	pub qubits: Option<Rc<IdlistContextAll<'input>>>,
	pub body: Option<Rc<BodyStatementContextAll<'input>>>,
ph:PhantomData<&'input str>
}

impl<'input> Qasm2ParserContext<'input> for GateDeclarationContext<'input>{}

impl<'input,'a> Listenable<dyn Qasm2ParserListener<'input> + 'a> for GateDeclarationContext<'input>{
		fn enter(&self,listener: &mut (dyn Qasm2ParserListener<'input> + 'a)) {
			listener.enter_every_rule(self);
			listener.enter_gateDeclaration(self);
		}
		fn exit(&self,listener: &mut (dyn Qasm2ParserListener<'input> + 'a)) {
			listener.exit_gateDeclaration(self);
			listener.exit_every_rule(self);
		}
}

impl<'input,'a> Visitable<dyn Qasm2ParserVisitor<'input> + 'a> for GateDeclarationContext<'input>{
	fn accept(&self,visitor: &mut (dyn Qasm2ParserVisitor<'input> + 'a)) {
		visitor.visit_gateDeclaration(self);
	}
}

impl<'input> CustomRuleContext<'input> for GateDeclarationContextExt<'input>{
	type TF = LocalTokenFactory<'input>;
	type Ctx = Qasm2ParserContextType;
	fn get_rule_index(&self) -> usize { RULE_gateDeclaration }
	//fn type_rule_index() -> usize where Self: Sized { RULE_gateDeclaration }
}
antlr_rust::tid!{GateDeclarationContextExt<'a>}

impl<'input> GateDeclarationContextExt<'input>{
	fn new(parent: Option<Rc<dyn Qasm2ParserContext<'input> + 'input > >, invoking_state: isize) -> Rc<GateDeclarationContextAll<'input>> {
		Rc::new(
			BaseParserRuleContext::new_parser_ctx(parent, invoking_state,GateDeclarationContextExt{
				params: None, qubits: None, body: None, 
				ph:PhantomData
			}),
		)
	}
}

pub trait GateDeclarationContextAttrs<'input>: Qasm2ParserContext<'input> + BorrowMut<GateDeclarationContextExt<'input>>{

/// Retrieves first TerminalNode corresponding to token GATE
/// Returns `None` if there is no child corresponding to token GATE
fn GATE(&self) -> Option<Rc<TerminalNode<'input,Qasm2ParserContextType>>> where Self:Sized{
	self.get_token(GATE, 0)
}
/// Retrieves first TerminalNode corresponding to token Identifier
/// Returns `None` if there is no child corresponding to token Identifier
fn Identifier(&self) -> Option<Rc<TerminalNode<'input,Qasm2ParserContextType>>> where Self:Sized{
	self.get_token(Identifier, 0)
}
/// Retrieves first TerminalNode corresponding to token LBRACE
/// Returns `None` if there is no child corresponding to token LBRACE
fn LBRACE(&self) -> Option<Rc<TerminalNode<'input,Qasm2ParserContextType>>> where Self:Sized{
	self.get_token(LBRACE, 0)
}
/// Retrieves first TerminalNode corresponding to token RBRACE
/// Returns `None` if there is no child corresponding to token RBRACE
fn RBRACE(&self) -> Option<Rc<TerminalNode<'input,Qasm2ParserContextType>>> where Self:Sized{
	self.get_token(RBRACE, 0)
}
fn idlist_all(&self) ->  Vec<Rc<IdlistContextAll<'input>>> where Self:Sized{
	self.children_of_type()
}
fn idlist(&self, i: usize) -> Option<Rc<IdlistContextAll<'input>>> where Self:Sized{
	self.child_of_type(i)
}
/// Retrieves first TerminalNode corresponding to token LPAREN
/// Returns `None` if there is no child corresponding to token LPAREN
fn LPAREN(&self) -> Option<Rc<TerminalNode<'input,Qasm2ParserContextType>>> where Self:Sized{
	self.get_token(LPAREN, 0)
}
/// Retrieves first TerminalNode corresponding to token RPAREN
/// Returns `None` if there is no child corresponding to token RPAREN
fn RPAREN(&self) -> Option<Rc<TerminalNode<'input,Qasm2ParserContextType>>> where Self:Sized{
	self.get_token(RPAREN, 0)
}
fn bodyStatement_all(&self) ->  Vec<Rc<BodyStatementContextAll<'input>>> where Self:Sized{
	self.children_of_type()
}
fn bodyStatement(&self, i: usize) -> Option<Rc<BodyStatementContextAll<'input>>> where Self:Sized{
	self.child_of_type(i)
}
/// Retrieves first TerminalNode corresponding to token OPAQUE
/// Returns `None` if there is no child corresponding to token OPAQUE
fn OPAQUE(&self) -> Option<Rc<TerminalNode<'input,Qasm2ParserContextType>>> where Self:Sized{
	self.get_token(OPAQUE, 0)
}
/// Retrieves first TerminalNode corresponding to token SEMICOLON
/// Returns `None` if there is no child corresponding to token SEMICOLON
fn SEMICOLON(&self) -> Option<Rc<TerminalNode<'input,Qasm2ParserContextType>>> where Self:Sized{
	self.get_token(SEMICOLON, 0)
}

}

impl<'input> GateDeclarationContextAttrs<'input> for GateDeclarationContext<'input>{}

impl<'input, I, H> Qasm2Parser<'input, I, H>
where
    I: TokenStream<'input, TF = LocalTokenFactory<'input> > + TidAble<'input>,
    H: ErrorStrategy<'input,BaseParserType<'input,I>>
{
	pub fn gateDeclaration(&mut self,)
	-> Result<Rc<GateDeclarationContextAll<'input>>,ANTLRError> {
		let mut recog = self;
		let _parentctx = recog.ctx.take();
		let mut _localctx = GateDeclarationContextExt::new(_parentctx.clone(), recog.base.get_state());
        recog.base.enter_rule(_localctx.clone(), 10, RULE_gateDeclaration);
        let mut _localctx: Rc<GateDeclarationContextAll> = _localctx;
		let mut _la: isize = -1;
		let result: Result<(), ANTLRError> = (|| {

			recog.base.set_state(95);
			recog.err_handler.sync(&mut recog.base)?;
			match recog.base.input.la(1) {
			 GATE 
				=> {
					//recog.base.enter_outer_alt(_localctx.clone(), 1);
					recog.base.enter_outer_alt(None, 1);
					{
					recog.base.set_state(64);
					recog.base.match_token(GATE,&mut recog.err_handler)?;

					recog.base.set_state(65);
					recog.base.match_token(Identifier,&mut recog.err_handler)?;

					recog.base.set_state(71);
					recog.err_handler.sync(&mut recog.base)?;
					_la = recog.base.input.la(1);
					if _la==LPAREN {
						{
						recog.base.set_state(66);
						recog.base.match_token(LPAREN,&mut recog.err_handler)?;

						recog.base.set_state(68);
						recog.err_handler.sync(&mut recog.base)?;
						_la = recog.base.input.la(1);
						if _la==Identifier {
							{
							/*InvokeRule idlist*/
							recog.base.set_state(67);
							let tmp = recog.idlist()?;
							 cast_mut::<_,GateDeclarationContext >(&mut _localctx).params = Some(tmp.clone());
							  

							}
						}

						recog.base.set_state(70);
						recog.base.match_token(RPAREN,&mut recog.err_handler)?;

						}
					}

					/*InvokeRule idlist*/
					recog.base.set_state(73);
					let tmp = recog.idlist()?;
					 cast_mut::<_,GateDeclarationContext >(&mut _localctx).qubits = Some(tmp.clone());
					  

					recog.base.set_state(74);
					recog.base.match_token(LBRACE,&mut recog.err_handler)?;

					recog.base.set_state(78);
					recog.err_handler.sync(&mut recog.base)?;
					_la = recog.base.input.la(1);
					while (((_la) & !0x3f) == 0 && ((1usize << _la) & ((1usize << BARRIER) | (1usize << U) | (1usize << CX))) != 0) || _la==Identifier {
						{
						{
						/*InvokeRule bodyStatement*/
						recog.base.set_state(75);
						let tmp = recog.bodyStatement()?;
						 cast_mut::<_,GateDeclarationContext >(&mut _localctx).body = Some(tmp.clone());
						  

						}
						}
						recog.base.set_state(80);
						recog.err_handler.sync(&mut recog.base)?;
						_la = recog.base.input.la(1);
					}
					recog.base.set_state(81);
					recog.base.match_token(RBRACE,&mut recog.err_handler)?;

					}
				}

			 OPAQUE 
				=> {
					//recog.base.enter_outer_alt(_localctx.clone(), 2);
					recog.base.enter_outer_alt(None, 2);
					{
					recog.base.set_state(83);
					recog.base.match_token(OPAQUE,&mut recog.err_handler)?;

					recog.base.set_state(84);
					recog.base.match_token(Identifier,&mut recog.err_handler)?;

					recog.base.set_state(90);
					recog.err_handler.sync(&mut recog.base)?;
					_la = recog.base.input.la(1);
					if _la==LPAREN {
						{
						recog.base.set_state(85);
						recog.base.match_token(LPAREN,&mut recog.err_handler)?;

						recog.base.set_state(87);
						recog.err_handler.sync(&mut recog.base)?;
						_la = recog.base.input.la(1);
						if _la==Identifier {
							{
							/*InvokeRule idlist*/
							recog.base.set_state(86);
							let tmp = recog.idlist()?;
							 cast_mut::<_,GateDeclarationContext >(&mut _localctx).params = Some(tmp.clone());
							  

							}
						}

						recog.base.set_state(89);
						recog.base.match_token(RPAREN,&mut recog.err_handler)?;

						}
					}

					/*InvokeRule idlist*/
					recog.base.set_state(92);
					let tmp = recog.idlist()?;
					 cast_mut::<_,GateDeclarationContext >(&mut _localctx).qubits = Some(tmp.clone());
					  

					recog.base.set_state(93);
					recog.base.match_token(SEMICOLON,&mut recog.err_handler)?;

					}
				}

				_ => Err(ANTLRError::NoAltError(NoViableAltError::new(&mut recog.base)))?
			}
			Ok(())
		})();
		match result {
		Ok(_)=>{},
        Err(e @ ANTLRError::FallThrough(_)) => return Err(e),
		Err(ref re) => {
				//_localctx.exception = re;
				recog.err_handler.report_error(&mut recog.base, re);
				recog.err_handler.recover(&mut recog.base, re)?;
			}
		}
		recog.base.exit_rule();

		Ok(_localctx)
	}
}
//------------------- quantumOperation ----------------
pub type QuantumOperationContextAll<'input> = QuantumOperationContext<'input>;


pub type QuantumOperationContext<'input> = BaseParserRuleContext<'input,QuantumOperationContextExt<'input>>;

#[derive(Clone)]
pub struct QuantumOperationContextExt<'input>{
	pub src: Option<Rc<ArgumentContextAll<'input>>>,
	pub dest: Option<Rc<ArgumentContextAll<'input>>>,
ph:PhantomData<&'input str>
}

impl<'input> Qasm2ParserContext<'input> for QuantumOperationContext<'input>{}

impl<'input,'a> Listenable<dyn Qasm2ParserListener<'input> + 'a> for QuantumOperationContext<'input>{
		fn enter(&self,listener: &mut (dyn Qasm2ParserListener<'input> + 'a)) {
			listener.enter_every_rule(self);
			listener.enter_quantumOperation(self);
		}
		fn exit(&self,listener: &mut (dyn Qasm2ParserListener<'input> + 'a)) {
			listener.exit_quantumOperation(self);
			listener.exit_every_rule(self);
		}
}

impl<'input,'a> Visitable<dyn Qasm2ParserVisitor<'input> + 'a> for QuantumOperationContext<'input>{
	fn accept(&self,visitor: &mut (dyn Qasm2ParserVisitor<'input> + 'a)) {
		visitor.visit_quantumOperation(self);
	}
}

impl<'input> CustomRuleContext<'input> for QuantumOperationContextExt<'input>{
	type TF = LocalTokenFactory<'input>;
	type Ctx = Qasm2ParserContextType;
	fn get_rule_index(&self) -> usize { RULE_quantumOperation }
	//fn type_rule_index() -> usize where Self: Sized { RULE_quantumOperation }
}
antlr_rust::tid!{QuantumOperationContextExt<'a>}

impl<'input> QuantumOperationContextExt<'input>{
	fn new(parent: Option<Rc<dyn Qasm2ParserContext<'input> + 'input > >, invoking_state: isize) -> Rc<QuantumOperationContextAll<'input>> {
		Rc::new(
			BaseParserRuleContext::new_parser_ctx(parent, invoking_state,QuantumOperationContextExt{
				src: None, dest: None, 
				ph:PhantomData
			}),
		)
	}
}

pub trait QuantumOperationContextAttrs<'input>: Qasm2ParserContext<'input> + BorrowMut<QuantumOperationContextExt<'input>>{

fn gateCall(&self) -> Option<Rc<GateCallContextAll<'input>>> where Self:Sized{
	self.child_of_type(0)
}
/// Retrieves first TerminalNode corresponding to token MEASURE
/// Returns `None` if there is no child corresponding to token MEASURE
fn MEASURE(&self) -> Option<Rc<TerminalNode<'input,Qasm2ParserContextType>>> where Self:Sized{
	self.get_token(MEASURE, 0)
}
/// Retrieves first TerminalNode corresponding to token ARROW
/// Returns `None` if there is no child corresponding to token ARROW
fn ARROW(&self) -> Option<Rc<TerminalNode<'input,Qasm2ParserContextType>>> where Self:Sized{
	self.get_token(ARROW, 0)
}
/// Retrieves first TerminalNode corresponding to token SEMICOLON
/// Returns `None` if there is no child corresponding to token SEMICOLON
fn SEMICOLON(&self) -> Option<Rc<TerminalNode<'input,Qasm2ParserContextType>>> where Self:Sized{
	self.get_token(SEMICOLON, 0)
}
fn argument_all(&self) ->  Vec<Rc<ArgumentContextAll<'input>>> where Self:Sized{
	self.children_of_type()
}
fn argument(&self, i: usize) -> Option<Rc<ArgumentContextAll<'input>>> where Self:Sized{
	self.child_of_type(i)
}
/// Retrieves first TerminalNode corresponding to token RESET
/// Returns `None` if there is no child corresponding to token RESET
fn RESET(&self) -> Option<Rc<TerminalNode<'input,Qasm2ParserContextType>>> where Self:Sized{
	self.get_token(RESET, 0)
}

}

impl<'input> QuantumOperationContextAttrs<'input> for QuantumOperationContext<'input>{}

impl<'input, I, H> Qasm2Parser<'input, I, H>
where
    I: TokenStream<'input, TF = LocalTokenFactory<'input> > + TidAble<'input>,
    H: ErrorStrategy<'input,BaseParserType<'input,I>>
{
	pub fn quantumOperation(&mut self,)
	-> Result<Rc<QuantumOperationContextAll<'input>>,ANTLRError> {
		let mut recog = self;
		let _parentctx = recog.ctx.take();
		let mut _localctx = QuantumOperationContextExt::new(_parentctx.clone(), recog.base.get_state());
        recog.base.enter_rule(_localctx.clone(), 12, RULE_quantumOperation);
        let mut _localctx: Rc<QuantumOperationContextAll> = _localctx;
		let result: Result<(), ANTLRError> = (|| {

			recog.base.set_state(108);
			recog.err_handler.sync(&mut recog.base)?;
			match recog.base.input.la(1) {
			 U | CX | Identifier 
				=> {
					//recog.base.enter_outer_alt(_localctx.clone(), 1);
					recog.base.enter_outer_alt(None, 1);
					{
					/*InvokeRule gateCall*/
					recog.base.set_state(97);
					recog.gateCall()?;

					}
				}

			 MEASURE 
				=> {
					//recog.base.enter_outer_alt(_localctx.clone(), 2);
					recog.base.enter_outer_alt(None, 2);
					{
					recog.base.set_state(98);
					recog.base.match_token(MEASURE,&mut recog.err_handler)?;

					/*InvokeRule argument*/
					recog.base.set_state(99);
					let tmp = recog.argument()?;
					 cast_mut::<_,QuantumOperationContext >(&mut _localctx).src = Some(tmp.clone());
					  

					recog.base.set_state(100);
					recog.base.match_token(ARROW,&mut recog.err_handler)?;

					/*InvokeRule argument*/
					recog.base.set_state(101);
					let tmp = recog.argument()?;
					 cast_mut::<_,QuantumOperationContext >(&mut _localctx).dest = Some(tmp.clone());
					  

					recog.base.set_state(102);
					recog.base.match_token(SEMICOLON,&mut recog.err_handler)?;

					}
				}

			 RESET 
				=> {
					//recog.base.enter_outer_alt(_localctx.clone(), 3);
					recog.base.enter_outer_alt(None, 3);
					{
					recog.base.set_state(104);
					recog.base.match_token(RESET,&mut recog.err_handler)?;

					/*InvokeRule argument*/
					recog.base.set_state(105);
					let tmp = recog.argument()?;
					 cast_mut::<_,QuantumOperationContext >(&mut _localctx).dest = Some(tmp.clone());
					  

					recog.base.set_state(106);
					recog.base.match_token(SEMICOLON,&mut recog.err_handler)?;

					}
				}

				_ => Err(ANTLRError::NoAltError(NoViableAltError::new(&mut recog.base)))?
			}
			Ok(())
		})();
		match result {
		Ok(_)=>{},
        Err(e @ ANTLRError::FallThrough(_)) => return Err(e),
		Err(ref re) => {
				//_localctx.exception = re;
				recog.err_handler.report_error(&mut recog.base, re);
				recog.err_handler.recover(&mut recog.base, re)?;
			}
		}
		recog.base.exit_rule();

		Ok(_localctx)
	}
}
//------------------- ifStatement ----------------
pub type IfStatementContextAll<'input> = IfStatementContext<'input>;


pub type IfStatementContext<'input> = BaseParserRuleContext<'input,IfStatementContextExt<'input>>;

#[derive(Clone)]
pub struct IfStatementContextExt<'input>{
ph:PhantomData<&'input str>
}

impl<'input> Qasm2ParserContext<'input> for IfStatementContext<'input>{}

impl<'input,'a> Listenable<dyn Qasm2ParserListener<'input> + 'a> for IfStatementContext<'input>{
		fn enter(&self,listener: &mut (dyn Qasm2ParserListener<'input> + 'a)) {
			listener.enter_every_rule(self);
			listener.enter_ifStatement(self);
		}
		fn exit(&self,listener: &mut (dyn Qasm2ParserListener<'input> + 'a)) {
			listener.exit_ifStatement(self);
			listener.exit_every_rule(self);
		}
}

impl<'input,'a> Visitable<dyn Qasm2ParserVisitor<'input> + 'a> for IfStatementContext<'input>{
	fn accept(&self,visitor: &mut (dyn Qasm2ParserVisitor<'input> + 'a)) {
		visitor.visit_ifStatement(self);
	}
}

impl<'input> CustomRuleContext<'input> for IfStatementContextExt<'input>{
	type TF = LocalTokenFactory<'input>;
	type Ctx = Qasm2ParserContextType;
	fn get_rule_index(&self) -> usize { RULE_ifStatement }
	//fn type_rule_index() -> usize where Self: Sized { RULE_ifStatement }
}
antlr_rust::tid!{IfStatementContextExt<'a>}

impl<'input> IfStatementContextExt<'input>{
	fn new(parent: Option<Rc<dyn Qasm2ParserContext<'input> + 'input > >, invoking_state: isize) -> Rc<IfStatementContextAll<'input>> {
		Rc::new(
			BaseParserRuleContext::new_parser_ctx(parent, invoking_state,IfStatementContextExt{
				ph:PhantomData
			}),
		)
	}
}

pub trait IfStatementContextAttrs<'input>: Qasm2ParserContext<'input> + BorrowMut<IfStatementContextExt<'input>>{

/// Retrieves first TerminalNode corresponding to token IF
/// Returns `None` if there is no child corresponding to token IF
fn IF(&self) -> Option<Rc<TerminalNode<'input,Qasm2ParserContextType>>> where Self:Sized{
	self.get_token(IF, 0)
}
/// Retrieves first TerminalNode corresponding to token LPAREN
/// Returns `None` if there is no child corresponding to token LPAREN
fn LPAREN(&self) -> Option<Rc<TerminalNode<'input,Qasm2ParserContextType>>> where Self:Sized{
	self.get_token(LPAREN, 0)
}
/// Retrieves first TerminalNode corresponding to token Identifier
/// Returns `None` if there is no child corresponding to token Identifier
fn Identifier(&self) -> Option<Rc<TerminalNode<'input,Qasm2ParserContextType>>> where Self:Sized{
	self.get_token(Identifier, 0)
}
/// Retrieves first TerminalNode corresponding to token EQUALS
/// Returns `None` if there is no child corresponding to token EQUALS
fn EQUALS(&self) -> Option<Rc<TerminalNode<'input,Qasm2ParserContextType>>> where Self:Sized{
	self.get_token(EQUALS, 0)
}
/// Retrieves first TerminalNode corresponding to token Integer
/// Returns `None` if there is no child corresponding to token Integer
fn Integer(&self) -> Option<Rc<TerminalNode<'input,Qasm2ParserContextType>>> where Self:Sized{
	self.get_token(Integer, 0)
}
/// Retrieves first TerminalNode corresponding to token RPAREN
/// Returns `None` if there is no child corresponding to token RPAREN
fn RPAREN(&self) -> Option<Rc<TerminalNode<'input,Qasm2ParserContextType>>> where Self:Sized{
	self.get_token(RPAREN, 0)
}
fn quantumOperation(&self) -> Option<Rc<QuantumOperationContextAll<'input>>> where Self:Sized{
	self.child_of_type(0)
}

}

impl<'input> IfStatementContextAttrs<'input> for IfStatementContext<'input>{}

impl<'input, I, H> Qasm2Parser<'input, I, H>
where
    I: TokenStream<'input, TF = LocalTokenFactory<'input> > + TidAble<'input>,
    H: ErrorStrategy<'input,BaseParserType<'input,I>>
{
	pub fn ifStatement(&mut self,)
	-> Result<Rc<IfStatementContextAll<'input>>,ANTLRError> {
		let mut recog = self;
		let _parentctx = recog.ctx.take();
		let mut _localctx = IfStatementContextExt::new(_parentctx.clone(), recog.base.get_state());
        recog.base.enter_rule(_localctx.clone(), 14, RULE_ifStatement);
        let mut _localctx: Rc<IfStatementContextAll> = _localctx;
		let result: Result<(), ANTLRError> = (|| {

			//recog.base.enter_outer_alt(_localctx.clone(), 1);
			recog.base.enter_outer_alt(None, 1);
			{
			recog.base.set_state(110);
			recog.base.match_token(IF,&mut recog.err_handler)?;

			recog.base.set_state(111);
			recog.base.match_token(LPAREN,&mut recog.err_handler)?;

			recog.base.set_state(112);
			recog.base.match_token(Identifier,&mut recog.err_handler)?;

			recog.base.set_state(113);
			recog.base.match_token(EQUALS,&mut recog.err_handler)?;

			recog.base.set_state(114);
			recog.base.match_token(Integer,&mut recog.err_handler)?;

			recog.base.set_state(115);
			recog.base.match_token(RPAREN,&mut recog.err_handler)?;

			/*InvokeRule quantumOperation*/
			recog.base.set_state(116);
			recog.quantumOperation()?;

			}
			Ok(())
		})();
		match result {
		Ok(_)=>{},
        Err(e @ ANTLRError::FallThrough(_)) => return Err(e),
		Err(ref re) => {
				//_localctx.exception = re;
				recog.err_handler.report_error(&mut recog.base, re);
				recog.err_handler.recover(&mut recog.base, re)?;
			}
		}
		recog.base.exit_rule();

		Ok(_localctx)
	}
}
//------------------- barrier ----------------
pub type BarrierContextAll<'input> = BarrierContext<'input>;


pub type BarrierContext<'input> = BaseParserRuleContext<'input,BarrierContextExt<'input>>;

#[derive(Clone)]
pub struct BarrierContextExt<'input>{
ph:PhantomData<&'input str>
}

impl<'input> Qasm2ParserContext<'input> for BarrierContext<'input>{}

impl<'input,'a> Listenable<dyn Qasm2ParserListener<'input> + 'a> for BarrierContext<'input>{
		fn enter(&self,listener: &mut (dyn Qasm2ParserListener<'input> + 'a)) {
			listener.enter_every_rule(self);
			listener.enter_barrier(self);
		}
		fn exit(&self,listener: &mut (dyn Qasm2ParserListener<'input> + 'a)) {
			listener.exit_barrier(self);
			listener.exit_every_rule(self);
		}
}

impl<'input,'a> Visitable<dyn Qasm2ParserVisitor<'input> + 'a> for BarrierContext<'input>{
	fn accept(&self,visitor: &mut (dyn Qasm2ParserVisitor<'input> + 'a)) {
		visitor.visit_barrier(self);
	}
}

impl<'input> CustomRuleContext<'input> for BarrierContextExt<'input>{
	type TF = LocalTokenFactory<'input>;
	type Ctx = Qasm2ParserContextType;
	fn get_rule_index(&self) -> usize { RULE_barrier }
	//fn type_rule_index() -> usize where Self: Sized { RULE_barrier }
}
antlr_rust::tid!{BarrierContextExt<'a>}

impl<'input> BarrierContextExt<'input>{
	fn new(parent: Option<Rc<dyn Qasm2ParserContext<'input> + 'input > >, invoking_state: isize) -> Rc<BarrierContextAll<'input>> {
		Rc::new(
			BaseParserRuleContext::new_parser_ctx(parent, invoking_state,BarrierContextExt{
				ph:PhantomData
			}),
		)
	}
}

pub trait BarrierContextAttrs<'input>: Qasm2ParserContext<'input> + BorrowMut<BarrierContextExt<'input>>{

/// Retrieves first TerminalNode corresponding to token BARRIER
/// Returns `None` if there is no child corresponding to token BARRIER
fn BARRIER(&self) -> Option<Rc<TerminalNode<'input,Qasm2ParserContextType>>> where Self:Sized{
	self.get_token(BARRIER, 0)
}
fn mixedlist(&self) -> Option<Rc<MixedlistContextAll<'input>>> where Self:Sized{
	self.child_of_type(0)
}
/// Retrieves first TerminalNode corresponding to token SEMICOLON
/// Returns `None` if there is no child corresponding to token SEMICOLON
fn SEMICOLON(&self) -> Option<Rc<TerminalNode<'input,Qasm2ParserContextType>>> where Self:Sized{
	self.get_token(SEMICOLON, 0)
}

}

impl<'input> BarrierContextAttrs<'input> for BarrierContext<'input>{}

impl<'input, I, H> Qasm2Parser<'input, I, H>
where
    I: TokenStream<'input, TF = LocalTokenFactory<'input> > + TidAble<'input>,
    H: ErrorStrategy<'input,BaseParserType<'input,I>>
{
	pub fn barrier(&mut self,)
	-> Result<Rc<BarrierContextAll<'input>>,ANTLRError> {
		let mut recog = self;
		let _parentctx = recog.ctx.take();
		let mut _localctx = BarrierContextExt::new(_parentctx.clone(), recog.base.get_state());
        recog.base.enter_rule(_localctx.clone(), 16, RULE_barrier);
        let mut _localctx: Rc<BarrierContextAll> = _localctx;
		let result: Result<(), ANTLRError> = (|| {

			//recog.base.enter_outer_alt(_localctx.clone(), 1);
			recog.base.enter_outer_alt(None, 1);
			{
			recog.base.set_state(118);
			recog.base.match_token(BARRIER,&mut recog.err_handler)?;

			/*InvokeRule mixedlist*/
			recog.base.set_state(119);
			recog.mixedlist()?;

			recog.base.set_state(120);
			recog.base.match_token(SEMICOLON,&mut recog.err_handler)?;

			}
			Ok(())
		})();
		match result {
		Ok(_)=>{},
        Err(e @ ANTLRError::FallThrough(_)) => return Err(e),
		Err(ref re) => {
				//_localctx.exception = re;
				recog.err_handler.report_error(&mut recog.base, re);
				recog.err_handler.recover(&mut recog.base, re)?;
			}
		}
		recog.base.exit_rule();

		Ok(_localctx)
	}
}
//------------------- bodyStatement ----------------
pub type BodyStatementContextAll<'input> = BodyStatementContext<'input>;


pub type BodyStatementContext<'input> = BaseParserRuleContext<'input,BodyStatementContextExt<'input>>;

#[derive(Clone)]
pub struct BodyStatementContextExt<'input>{
ph:PhantomData<&'input str>
}

impl<'input> Qasm2ParserContext<'input> for BodyStatementContext<'input>{}

impl<'input,'a> Listenable<dyn Qasm2ParserListener<'input> + 'a> for BodyStatementContext<'input>{
		fn enter(&self,listener: &mut (dyn Qasm2ParserListener<'input> + 'a)) {
			listener.enter_every_rule(self);
			listener.enter_bodyStatement(self);
		}
		fn exit(&self,listener: &mut (dyn Qasm2ParserListener<'input> + 'a)) {
			listener.exit_bodyStatement(self);
			listener.exit_every_rule(self);
		}
}

impl<'input,'a> Visitable<dyn Qasm2ParserVisitor<'input> + 'a> for BodyStatementContext<'input>{
	fn accept(&self,visitor: &mut (dyn Qasm2ParserVisitor<'input> + 'a)) {
		visitor.visit_bodyStatement(self);
	}
}

impl<'input> CustomRuleContext<'input> for BodyStatementContextExt<'input>{
	type TF = LocalTokenFactory<'input>;
	type Ctx = Qasm2ParserContextType;
	fn get_rule_index(&self) -> usize { RULE_bodyStatement }
	//fn type_rule_index() -> usize where Self: Sized { RULE_bodyStatement }
}
antlr_rust::tid!{BodyStatementContextExt<'a>}

impl<'input> BodyStatementContextExt<'input>{
	fn new(parent: Option<Rc<dyn Qasm2ParserContext<'input> + 'input > >, invoking_state: isize) -> Rc<BodyStatementContextAll<'input>> {
		Rc::new(
			BaseParserRuleContext::new_parser_ctx(parent, invoking_state,BodyStatementContextExt{
				ph:PhantomData
			}),
		)
	}
}

pub trait BodyStatementContextAttrs<'input>: Qasm2ParserContext<'input> + BorrowMut<BodyStatementContextExt<'input>>{

fn gateCall(&self) -> Option<Rc<GateCallContextAll<'input>>> where Self:Sized{
	self.child_of_type(0)
}
/// Retrieves first TerminalNode corresponding to token BARRIER
/// Returns `None` if there is no child corresponding to token BARRIER
fn BARRIER(&self) -> Option<Rc<TerminalNode<'input,Qasm2ParserContextType>>> where Self:Sized{
	self.get_token(BARRIER, 0)
}
fn idlist(&self) -> Option<Rc<IdlistContextAll<'input>>> where Self:Sized{
	self.child_of_type(0)
}
/// Retrieves first TerminalNode corresponding to token SEMICOLON
/// Returns `None` if there is no child corresponding to token SEMICOLON
fn SEMICOLON(&self) -> Option<Rc<TerminalNode<'input,Qasm2ParserContextType>>> where Self:Sized{
	self.get_token(SEMICOLON, 0)
}

}

impl<'input> BodyStatementContextAttrs<'input> for BodyStatementContext<'input>{}

impl<'input, I, H> Qasm2Parser<'input, I, H>
where
    I: TokenStream<'input, TF = LocalTokenFactory<'input> > + TidAble<'input>,
    H: ErrorStrategy<'input,BaseParserType<'input,I>>
{
	pub fn bodyStatement(&mut self,)
	-> Result<Rc<BodyStatementContextAll<'input>>,ANTLRError> {
		let mut recog = self;
		let _parentctx = recog.ctx.take();
		let mut _localctx = BodyStatementContextExt::new(_parentctx.clone(), recog.base.get_state());
        recog.base.enter_rule(_localctx.clone(), 18, RULE_bodyStatement);
        let mut _localctx: Rc<BodyStatementContextAll> = _localctx;
		let result: Result<(), ANTLRError> = (|| {

			recog.base.set_state(127);
			recog.err_handler.sync(&mut recog.base)?;
			match recog.base.input.la(1) {
			 U | CX | Identifier 
				=> {
					//recog.base.enter_outer_alt(_localctx.clone(), 1);
					recog.base.enter_outer_alt(None, 1);
					{
					/*InvokeRule gateCall*/
					recog.base.set_state(122);
					recog.gateCall()?;

					}
				}

			 BARRIER 
				=> {
					//recog.base.enter_outer_alt(_localctx.clone(), 2);
					recog.base.enter_outer_alt(None, 2);
					{
					recog.base.set_state(123);
					recog.base.match_token(BARRIER,&mut recog.err_handler)?;

					/*InvokeRule idlist*/
					recog.base.set_state(124);
					recog.idlist()?;

					recog.base.set_state(125);
					recog.base.match_token(SEMICOLON,&mut recog.err_handler)?;

					}
				}

				_ => Err(ANTLRError::NoAltError(NoViableAltError::new(&mut recog.base)))?
			}
			Ok(())
		})();
		match result {
		Ok(_)=>{},
        Err(e @ ANTLRError::FallThrough(_)) => return Err(e),
		Err(ref re) => {
				//_localctx.exception = re;
				recog.err_handler.report_error(&mut recog.base, re);
				recog.err_handler.recover(&mut recog.base, re)?;
			}
		}
		recog.base.exit_rule();

		Ok(_localctx)
	}
}
//------------------- gateCall ----------------
pub type GateCallContextAll<'input> = GateCallContext<'input>;


pub type GateCallContext<'input> = BaseParserRuleContext<'input,GateCallContextExt<'input>>;

#[derive(Clone)]
pub struct GateCallContextExt<'input>{
ph:PhantomData<&'input str>
}

impl<'input> Qasm2ParserContext<'input> for GateCallContext<'input>{}

impl<'input,'a> Listenable<dyn Qasm2ParserListener<'input> + 'a> for GateCallContext<'input>{
		fn enter(&self,listener: &mut (dyn Qasm2ParserListener<'input> + 'a)) {
			listener.enter_every_rule(self);
			listener.enter_gateCall(self);
		}
		fn exit(&self,listener: &mut (dyn Qasm2ParserListener<'input> + 'a)) {
			listener.exit_gateCall(self);
			listener.exit_every_rule(self);
		}
}

impl<'input,'a> Visitable<dyn Qasm2ParserVisitor<'input> + 'a> for GateCallContext<'input>{
	fn accept(&self,visitor: &mut (dyn Qasm2ParserVisitor<'input> + 'a)) {
		visitor.visit_gateCall(self);
	}
}

impl<'input> CustomRuleContext<'input> for GateCallContextExt<'input>{
	type TF = LocalTokenFactory<'input>;
	type Ctx = Qasm2ParserContextType;
	fn get_rule_index(&self) -> usize { RULE_gateCall }
	//fn type_rule_index() -> usize where Self: Sized { RULE_gateCall }
}
antlr_rust::tid!{GateCallContextExt<'a>}

impl<'input> GateCallContextExt<'input>{
	fn new(parent: Option<Rc<dyn Qasm2ParserContext<'input> + 'input > >, invoking_state: isize) -> Rc<GateCallContextAll<'input>> {
		Rc::new(
			BaseParserRuleContext::new_parser_ctx(parent, invoking_state,GateCallContextExt{
				ph:PhantomData
			}),
		)
	}
}

pub trait GateCallContextAttrs<'input>: Qasm2ParserContext<'input> + BorrowMut<GateCallContextExt<'input>>{

fn mixedlist(&self) -> Option<Rc<MixedlistContextAll<'input>>> where Self:Sized{
	self.child_of_type(0)
}
/// Retrieves first TerminalNode corresponding to token SEMICOLON
/// Returns `None` if there is no child corresponding to token SEMICOLON
fn SEMICOLON(&self) -> Option<Rc<TerminalNode<'input,Qasm2ParserContextType>>> where Self:Sized{
	self.get_token(SEMICOLON, 0)
}
/// Retrieves first TerminalNode corresponding to token U
/// Returns `None` if there is no child corresponding to token U
fn U(&self) -> Option<Rc<TerminalNode<'input,Qasm2ParserContextType>>> where Self:Sized{
	self.get_token(U, 0)
}
/// Retrieves first TerminalNode corresponding to token CX
/// Returns `None` if there is no child corresponding to token CX
fn CX(&self) -> Option<Rc<TerminalNode<'input,Qasm2ParserContextType>>> where Self:Sized{
	self.get_token(CX, 0)
}
/// Retrieves first TerminalNode corresponding to token Identifier
/// Returns `None` if there is no child corresponding to token Identifier
fn Identifier(&self) -> Option<Rc<TerminalNode<'input,Qasm2ParserContextType>>> where Self:Sized{
	self.get_token(Identifier, 0)
}
/// Retrieves first TerminalNode corresponding to token LPAREN
/// Returns `None` if there is no child corresponding to token LPAREN
fn LPAREN(&self) -> Option<Rc<TerminalNode<'input,Qasm2ParserContextType>>> where Self:Sized{
	self.get_token(LPAREN, 0)
}
/// Retrieves first TerminalNode corresponding to token RPAREN
/// Returns `None` if there is no child corresponding to token RPAREN
fn RPAREN(&self) -> Option<Rc<TerminalNode<'input,Qasm2ParserContextType>>> where Self:Sized{
	self.get_token(RPAREN, 0)
}
fn explist(&self) -> Option<Rc<ExplistContextAll<'input>>> where Self:Sized{
	self.child_of_type(0)
}

}

impl<'input> GateCallContextAttrs<'input> for GateCallContext<'input>{}

impl<'input, I, H> Qasm2Parser<'input, I, H>
where
    I: TokenStream<'input, TF = LocalTokenFactory<'input> > + TidAble<'input>,
    H: ErrorStrategy<'input,BaseParserType<'input,I>>
{
	pub fn gateCall(&mut self,)
	-> Result<Rc<GateCallContextAll<'input>>,ANTLRError> {
		let mut recog = self;
		let _parentctx = recog.ctx.take();
		let mut _localctx = GateCallContextExt::new(_parentctx.clone(), recog.base.get_state());
        recog.base.enter_rule(_localctx.clone(), 20, RULE_gateCall);
        let mut _localctx: Rc<GateCallContextAll> = _localctx;
		let mut _la: isize = -1;
		let result: Result<(), ANTLRError> = (|| {

			//recog.base.enter_outer_alt(_localctx.clone(), 1);
			recog.base.enter_outer_alt(None, 1);
			{
			recog.base.set_state(129);
			_la = recog.base.input.la(1);
			if { !(((((_la - 12)) & !0x3f) == 0 && ((1usize << (_la - 12)) & ((1usize << (U - 12)) | (1usize << (CX - 12)) | (1usize << (Identifier - 12)))) != 0)) } {
				recog.err_handler.recover_inline(&mut recog.base)?;

			}
			else {
				if  recog.base.input.la(1)==TOKEN_EOF { recog.base.matched_eof = true };
				recog.err_handler.report_match(&mut recog.base);
				recog.base.consume(&mut recog.err_handler);
			}
			recog.base.set_state(135);
			recog.err_handler.sync(&mut recog.base)?;
			_la = recog.base.input.la(1);
			if _la==LPAREN {
				{
				recog.base.set_state(130);
				recog.base.match_token(LPAREN,&mut recog.err_handler)?;

				recog.base.set_state(132);
				recog.err_handler.sync(&mut recog.base)?;
				_la = recog.base.input.la(1);
				if (((_la) & !0x3f) == 0 && ((1usize << _la) & ((1usize << PI) | (1usize << LPAREN) | (1usize << MINUS) | (1usize << SIN) | (1usize << COS))) != 0) || ((((_la - 32)) & !0x3f) == 0 && ((1usize << (_la - 32)) & ((1usize << (TAN - 32)) | (1usize << (EXP - 32)) | (1usize << (LN - 32)) | (1usize << (SQRT - 32)) | (1usize << (Integer - 32)) | (1usize << (Float - 32)) | (1usize << (Identifier - 32)))) != 0) {
					{
					/*InvokeRule explist*/
					recog.base.set_state(131);
					recog.explist()?;

					}
				}

				recog.base.set_state(134);
				recog.base.match_token(RPAREN,&mut recog.err_handler)?;

				}
			}

			/*InvokeRule mixedlist*/
			recog.base.set_state(137);
			recog.mixedlist()?;

			recog.base.set_state(138);
			recog.base.match_token(SEMICOLON,&mut recog.err_handler)?;

			}
			Ok(())
		})();
		match result {
		Ok(_)=>{},
        Err(e @ ANTLRError::FallThrough(_)) => return Err(e),
		Err(ref re) => {
				//_localctx.exception = re;
				recog.err_handler.report_error(&mut recog.base, re);
				recog.err_handler.recover(&mut recog.base, re)?;
			}
		}
		recog.base.exit_rule();

		Ok(_localctx)
	}
}
//------------------- idlist ----------------
pub type IdlistContextAll<'input> = IdlistContext<'input>;


pub type IdlistContext<'input> = BaseParserRuleContext<'input,IdlistContextExt<'input>>;

#[derive(Clone)]
pub struct IdlistContextExt<'input>{
ph:PhantomData<&'input str>
}

impl<'input> Qasm2ParserContext<'input> for IdlistContext<'input>{}

impl<'input,'a> Listenable<dyn Qasm2ParserListener<'input> + 'a> for IdlistContext<'input>{
		fn enter(&self,listener: &mut (dyn Qasm2ParserListener<'input> + 'a)) {
			listener.enter_every_rule(self);
			listener.enter_idlist(self);
		}
		fn exit(&self,listener: &mut (dyn Qasm2ParserListener<'input> + 'a)) {
			listener.exit_idlist(self);
			listener.exit_every_rule(self);
		}
}

impl<'input,'a> Visitable<dyn Qasm2ParserVisitor<'input> + 'a> for IdlistContext<'input>{
	fn accept(&self,visitor: &mut (dyn Qasm2ParserVisitor<'input> + 'a)) {
		visitor.visit_idlist(self);
	}
}

impl<'input> CustomRuleContext<'input> for IdlistContextExt<'input>{
	type TF = LocalTokenFactory<'input>;
	type Ctx = Qasm2ParserContextType;
	fn get_rule_index(&self) -> usize { RULE_idlist }
	//fn type_rule_index() -> usize where Self: Sized { RULE_idlist }
}
antlr_rust::tid!{IdlistContextExt<'a>}

impl<'input> IdlistContextExt<'input>{
	fn new(parent: Option<Rc<dyn Qasm2ParserContext<'input> + 'input > >, invoking_state: isize) -> Rc<IdlistContextAll<'input>> {
		Rc::new(
			BaseParserRuleContext::new_parser_ctx(parent, invoking_state,IdlistContextExt{
				ph:PhantomData
			}),
		)
	}
}

pub trait IdlistContextAttrs<'input>: Qasm2ParserContext<'input> + BorrowMut<IdlistContextExt<'input>>{

/// Retrieves all `TerminalNode`s corresponding to token Identifier in current rule
fn Identifier_all(&self) -> Vec<Rc<TerminalNode<'input,Qasm2ParserContextType>>>  where Self:Sized{
	self.children_of_type()
}
/// Retrieves 'i's TerminalNode corresponding to token Identifier, starting from 0.
/// Returns `None` if number of children corresponding to token Identifier is less or equal than `i`.
fn Identifier(&self, i: usize) -> Option<Rc<TerminalNode<'input,Qasm2ParserContextType>>> where Self:Sized{
	self.get_token(Identifier, i)
}
/// Retrieves all `TerminalNode`s corresponding to token COMMA in current rule
fn COMMA_all(&self) -> Vec<Rc<TerminalNode<'input,Qasm2ParserContextType>>>  where Self:Sized{
	self.children_of_type()
}
/// Retrieves 'i's TerminalNode corresponding to token COMMA, starting from 0.
/// Returns `None` if number of children corresponding to token COMMA is less or equal than `i`.
fn COMMA(&self, i: usize) -> Option<Rc<TerminalNode<'input,Qasm2ParserContextType>>> where Self:Sized{
	self.get_token(COMMA, i)
}

}

impl<'input> IdlistContextAttrs<'input> for IdlistContext<'input>{}

impl<'input, I, H> Qasm2Parser<'input, I, H>
where
    I: TokenStream<'input, TF = LocalTokenFactory<'input> > + TidAble<'input>,
    H: ErrorStrategy<'input,BaseParserType<'input,I>>
{
	pub fn idlist(&mut self,)
	-> Result<Rc<IdlistContextAll<'input>>,ANTLRError> {
		let mut recog = self;
		let _parentctx = recog.ctx.take();
		let mut _localctx = IdlistContextExt::new(_parentctx.clone(), recog.base.get_state());
        recog.base.enter_rule(_localctx.clone(), 22, RULE_idlist);
        let mut _localctx: Rc<IdlistContextAll> = _localctx;
		let mut _la: isize = -1;
		let result: Result<(), ANTLRError> = (|| {

			//recog.base.enter_outer_alt(_localctx.clone(), 1);
			recog.base.enter_outer_alt(None, 1);
			{
			recog.base.set_state(140);
			recog.base.match_token(Identifier,&mut recog.err_handler)?;

			recog.base.set_state(145);
			recog.err_handler.sync(&mut recog.base)?;
			_la = recog.base.input.la(1);
			while _la==COMMA {
				{
				{
				recog.base.set_state(141);
				recog.base.match_token(COMMA,&mut recog.err_handler)?;

				recog.base.set_state(142);
				recog.base.match_token(Identifier,&mut recog.err_handler)?;

				}
				}
				recog.base.set_state(147);
				recog.err_handler.sync(&mut recog.base)?;
				_la = recog.base.input.la(1);
			}
			}
			Ok(())
		})();
		match result {
		Ok(_)=>{},
        Err(e @ ANTLRError::FallThrough(_)) => return Err(e),
		Err(ref re) => {
				//_localctx.exception = re;
				recog.err_handler.report_error(&mut recog.base, re);
				recog.err_handler.recover(&mut recog.base, re)?;
			}
		}
		recog.base.exit_rule();

		Ok(_localctx)
	}
}
//------------------- mixedlist ----------------
pub type MixedlistContextAll<'input> = MixedlistContext<'input>;


pub type MixedlistContext<'input> = BaseParserRuleContext<'input,MixedlistContextExt<'input>>;

#[derive(Clone)]
pub struct MixedlistContextExt<'input>{
ph:PhantomData<&'input str>
}

impl<'input> Qasm2ParserContext<'input> for MixedlistContext<'input>{}

impl<'input,'a> Listenable<dyn Qasm2ParserListener<'input> + 'a> for MixedlistContext<'input>{
		fn enter(&self,listener: &mut (dyn Qasm2ParserListener<'input> + 'a)) {
			listener.enter_every_rule(self);
			listener.enter_mixedlist(self);
		}
		fn exit(&self,listener: &mut (dyn Qasm2ParserListener<'input> + 'a)) {
			listener.exit_mixedlist(self);
			listener.exit_every_rule(self);
		}
}

impl<'input,'a> Visitable<dyn Qasm2ParserVisitor<'input> + 'a> for MixedlistContext<'input>{
	fn accept(&self,visitor: &mut (dyn Qasm2ParserVisitor<'input> + 'a)) {
		visitor.visit_mixedlist(self);
	}
}

impl<'input> CustomRuleContext<'input> for MixedlistContextExt<'input>{
	type TF = LocalTokenFactory<'input>;
	type Ctx = Qasm2ParserContextType;
	fn get_rule_index(&self) -> usize { RULE_mixedlist }
	//fn type_rule_index() -> usize where Self: Sized { RULE_mixedlist }
}
antlr_rust::tid!{MixedlistContextExt<'a>}

impl<'input> MixedlistContextExt<'input>{
	fn new(parent: Option<Rc<dyn Qasm2ParserContext<'input> + 'input > >, invoking_state: isize) -> Rc<MixedlistContextAll<'input>> {
		Rc::new(
			BaseParserRuleContext::new_parser_ctx(parent, invoking_state,MixedlistContextExt{
				ph:PhantomData
			}),
		)
	}
}

pub trait MixedlistContextAttrs<'input>: Qasm2ParserContext<'input> + BorrowMut<MixedlistContextExt<'input>>{

fn argument_all(&self) ->  Vec<Rc<ArgumentContextAll<'input>>> where Self:Sized{
	self.children_of_type()
}
fn argument(&self, i: usize) -> Option<Rc<ArgumentContextAll<'input>>> where Self:Sized{
	self.child_of_type(i)
}
/// Retrieves all `TerminalNode`s corresponding to token COMMA in current rule
fn COMMA_all(&self) -> Vec<Rc<TerminalNode<'input,Qasm2ParserContextType>>>  where Self:Sized{
	self.children_of_type()
}
/// Retrieves 'i's TerminalNode corresponding to token COMMA, starting from 0.
/// Returns `None` if number of children corresponding to token COMMA is less or equal than `i`.
fn COMMA(&self, i: usize) -> Option<Rc<TerminalNode<'input,Qasm2ParserContextType>>> where Self:Sized{
	self.get_token(COMMA, i)
}

}

impl<'input> MixedlistContextAttrs<'input> for MixedlistContext<'input>{}

impl<'input, I, H> Qasm2Parser<'input, I, H>
where
    I: TokenStream<'input, TF = LocalTokenFactory<'input> > + TidAble<'input>,
    H: ErrorStrategy<'input,BaseParserType<'input,I>>
{
	pub fn mixedlist(&mut self,)
	-> Result<Rc<MixedlistContextAll<'input>>,ANTLRError> {
		let mut recog = self;
		let _parentctx = recog.ctx.take();
		let mut _localctx = MixedlistContextExt::new(_parentctx.clone(), recog.base.get_state());
        recog.base.enter_rule(_localctx.clone(), 24, RULE_mixedlist);
        let mut _localctx: Rc<MixedlistContextAll> = _localctx;
		let mut _la: isize = -1;
		let result: Result<(), ANTLRError> = (|| {

			//recog.base.enter_outer_alt(_localctx.clone(), 1);
			recog.base.enter_outer_alt(None, 1);
			{
			/*InvokeRule argument*/
			recog.base.set_state(148);
			recog.argument()?;

			recog.base.set_state(153);
			recog.err_handler.sync(&mut recog.base)?;
			_la = recog.base.input.la(1);
			while _la==COMMA {
				{
				{
				recog.base.set_state(149);
				recog.base.match_token(COMMA,&mut recog.err_handler)?;

				/*InvokeRule argument*/
				recog.base.set_state(150);
				recog.argument()?;

				}
				}
				recog.base.set_state(155);
				recog.err_handler.sync(&mut recog.base)?;
				_la = recog.base.input.la(1);
			}
			}
			Ok(())
		})();
		match result {
		Ok(_)=>{},
        Err(e @ ANTLRError::FallThrough(_)) => return Err(e),
		Err(ref re) => {
				//_localctx.exception = re;
				recog.err_handler.report_error(&mut recog.base, re);
				recog.err_handler.recover(&mut recog.base, re)?;
			}
		}
		recog.base.exit_rule();

		Ok(_localctx)
	}
}
//------------------- argument ----------------
pub type ArgumentContextAll<'input> = ArgumentContext<'input>;


pub type ArgumentContext<'input> = BaseParserRuleContext<'input,ArgumentContextExt<'input>>;

#[derive(Clone)]
pub struct ArgumentContextExt<'input>{
ph:PhantomData<&'input str>
}

impl<'input> Qasm2ParserContext<'input> for ArgumentContext<'input>{}

impl<'input,'a> Listenable<dyn Qasm2ParserListener<'input> + 'a> for ArgumentContext<'input>{
		fn enter(&self,listener: &mut (dyn Qasm2ParserListener<'input> + 'a)) {
			listener.enter_every_rule(self);
			listener.enter_argument(self);
		}
		fn exit(&self,listener: &mut (dyn Qasm2ParserListener<'input> + 'a)) {
			listener.exit_argument(self);
			listener.exit_every_rule(self);
		}
}

impl<'input,'a> Visitable<dyn Qasm2ParserVisitor<'input> + 'a> for ArgumentContext<'input>{
	fn accept(&self,visitor: &mut (dyn Qasm2ParserVisitor<'input> + 'a)) {
		visitor.visit_argument(self);
	}
}

impl<'input> CustomRuleContext<'input> for ArgumentContextExt<'input>{
	type TF = LocalTokenFactory<'input>;
	type Ctx = Qasm2ParserContextType;
	fn get_rule_index(&self) -> usize { RULE_argument }
	//fn type_rule_index() -> usize where Self: Sized { RULE_argument }
}
antlr_rust::tid!{ArgumentContextExt<'a>}

impl<'input> ArgumentContextExt<'input>{
	fn new(parent: Option<Rc<dyn Qasm2ParserContext<'input> + 'input > >, invoking_state: isize) -> Rc<ArgumentContextAll<'input>> {
		Rc::new(
			BaseParserRuleContext::new_parser_ctx(parent, invoking_state,ArgumentContextExt{
				ph:PhantomData
			}),
		)
	}
}

pub trait ArgumentContextAttrs<'input>: Qasm2ParserContext<'input> + BorrowMut<ArgumentContextExt<'input>>{

/// Retrieves first TerminalNode corresponding to token Identifier
/// Returns `None` if there is no child corresponding to token Identifier
fn Identifier(&self) -> Option<Rc<TerminalNode<'input,Qasm2ParserContextType>>> where Self:Sized{
	self.get_token(Identifier, 0)
}
fn designator(&self) -> Option<Rc<DesignatorContextAll<'input>>> where Self:Sized{
	self.child_of_type(0)
}

}

impl<'input> ArgumentContextAttrs<'input> for ArgumentContext<'input>{}

impl<'input, I, H> Qasm2Parser<'input, I, H>
where
    I: TokenStream<'input, TF = LocalTokenFactory<'input> > + TidAble<'input>,
    H: ErrorStrategy<'input,BaseParserType<'input,I>>
{
	pub fn argument(&mut self,)
	-> Result<Rc<ArgumentContextAll<'input>>,ANTLRError> {
		let mut recog = self;
		let _parentctx = recog.ctx.take();
		let mut _localctx = ArgumentContextExt::new(_parentctx.clone(), recog.base.get_state());
        recog.base.enter_rule(_localctx.clone(), 26, RULE_argument);
        let mut _localctx: Rc<ArgumentContextAll> = _localctx;
		let mut _la: isize = -1;
		let result: Result<(), ANTLRError> = (|| {

			//recog.base.enter_outer_alt(_localctx.clone(), 1);
			recog.base.enter_outer_alt(None, 1);
			{
			recog.base.set_state(156);
			recog.base.match_token(Identifier,&mut recog.err_handler)?;

			recog.base.set_state(158);
			recog.err_handler.sync(&mut recog.base)?;
			_la = recog.base.input.la(1);
			if _la==LBRACKET {
				{
				/*InvokeRule designator*/
				recog.base.set_state(157);
				recog.designator()?;

				}
			}

			}
			Ok(())
		})();
		match result {
		Ok(_)=>{},
        Err(e @ ANTLRError::FallThrough(_)) => return Err(e),
		Err(ref re) => {
				//_localctx.exception = re;
				recog.err_handler.report_error(&mut recog.base, re);
				recog.err_handler.recover(&mut recog.base, re)?;
			}
		}
		recog.base.exit_rule();

		Ok(_localctx)
	}
}
//------------------- designator ----------------
pub type DesignatorContextAll<'input> = DesignatorContext<'input>;


pub type DesignatorContext<'input> = BaseParserRuleContext<'input,DesignatorContextExt<'input>>;

#[derive(Clone)]
pub struct DesignatorContextExt<'input>{
ph:PhantomData<&'input str>
}

impl<'input> Qasm2ParserContext<'input> for DesignatorContext<'input>{}

impl<'input,'a> Listenable<dyn Qasm2ParserListener<'input> + 'a> for DesignatorContext<'input>{
		fn enter(&self,listener: &mut (dyn Qasm2ParserListener<'input> + 'a)) {
			listener.enter_every_rule(self);
			listener.enter_designator(self);
		}
		fn exit(&self,listener: &mut (dyn Qasm2ParserListener<'input> + 'a)) {
			listener.exit_designator(self);
			listener.exit_every_rule(self);
		}
}

impl<'input,'a> Visitable<dyn Qasm2ParserVisitor<'input> + 'a> for DesignatorContext<'input>{
	fn accept(&self,visitor: &mut (dyn Qasm2ParserVisitor<'input> + 'a)) {
		visitor.visit_designator(self);
	}
}

impl<'input> CustomRuleContext<'input> for DesignatorContextExt<'input>{
	type TF = LocalTokenFactory<'input>;
	type Ctx = Qasm2ParserContextType;
	fn get_rule_index(&self) -> usize { RULE_designator }
	//fn type_rule_index() -> usize where Self: Sized { RULE_designator }
}
antlr_rust::tid!{DesignatorContextExt<'a>}

impl<'input> DesignatorContextExt<'input>{
	fn new(parent: Option<Rc<dyn Qasm2ParserContext<'input> + 'input > >, invoking_state: isize) -> Rc<DesignatorContextAll<'input>> {
		Rc::new(
			BaseParserRuleContext::new_parser_ctx(parent, invoking_state,DesignatorContextExt{
				ph:PhantomData
			}),
		)
	}
}

pub trait DesignatorContextAttrs<'input>: Qasm2ParserContext<'input> + BorrowMut<DesignatorContextExt<'input>>{

/// Retrieves first TerminalNode corresponding to token LBRACKET
/// Returns `None` if there is no child corresponding to token LBRACKET
fn LBRACKET(&self) -> Option<Rc<TerminalNode<'input,Qasm2ParserContextType>>> where Self:Sized{
	self.get_token(LBRACKET, 0)
}
/// Retrieves first TerminalNode corresponding to token Integer
/// Returns `None` if there is no child corresponding to token Integer
fn Integer(&self) -> Option<Rc<TerminalNode<'input,Qasm2ParserContextType>>> where Self:Sized{
	self.get_token(Integer, 0)
}
/// Retrieves first TerminalNode corresponding to token RBRACKET
/// Returns `None` if there is no child corresponding to token RBRACKET
fn RBRACKET(&self) -> Option<Rc<TerminalNode<'input,Qasm2ParserContextType>>> where Self:Sized{
	self.get_token(RBRACKET, 0)
}

}

impl<'input> DesignatorContextAttrs<'input> for DesignatorContext<'input>{}

impl<'input, I, H> Qasm2Parser<'input, I, H>
where
    I: TokenStream<'input, TF = LocalTokenFactory<'input> > + TidAble<'input>,
    H: ErrorStrategy<'input,BaseParserType<'input,I>>
{
	pub fn designator(&mut self,)
	-> Result<Rc<DesignatorContextAll<'input>>,ANTLRError> {
		let mut recog = self;
		let _parentctx = recog.ctx.take();
		let mut _localctx = DesignatorContextExt::new(_parentctx.clone(), recog.base.get_state());
        recog.base.enter_rule(_localctx.clone(), 28, RULE_designator);
        let mut _localctx: Rc<DesignatorContextAll> = _localctx;
		let result: Result<(), ANTLRError> = (|| {

			//recog.base.enter_outer_alt(_localctx.clone(), 1);
			recog.base.enter_outer_alt(None, 1);
			{
			recog.base.set_state(160);
			recog.base.match_token(LBRACKET,&mut recog.err_handler)?;

			recog.base.set_state(161);
			recog.base.match_token(Integer,&mut recog.err_handler)?;

			recog.base.set_state(162);
			recog.base.match_token(RBRACKET,&mut recog.err_handler)?;

			}
			Ok(())
		})();
		match result {
		Ok(_)=>{},
        Err(e @ ANTLRError::FallThrough(_)) => return Err(e),
		Err(ref re) => {
				//_localctx.exception = re;
				recog.err_handler.report_error(&mut recog.base, re);
				recog.err_handler.recover(&mut recog.base, re)?;
			}
		}
		recog.base.exit_rule();

		Ok(_localctx)
	}
}
//------------------- explist ----------------
pub type ExplistContextAll<'input> = ExplistContext<'input>;


pub type ExplistContext<'input> = BaseParserRuleContext<'input,ExplistContextExt<'input>>;

#[derive(Clone)]
pub struct ExplistContextExt<'input>{
ph:PhantomData<&'input str>
}

impl<'input> Qasm2ParserContext<'input> for ExplistContext<'input>{}

impl<'input,'a> Listenable<dyn Qasm2ParserListener<'input> + 'a> for ExplistContext<'input>{
		fn enter(&self,listener: &mut (dyn Qasm2ParserListener<'input> + 'a)) {
			listener.enter_every_rule(self);
			listener.enter_explist(self);
		}
		fn exit(&self,listener: &mut (dyn Qasm2ParserListener<'input> + 'a)) {
			listener.exit_explist(self);
			listener.exit_every_rule(self);
		}
}

impl<'input,'a> Visitable<dyn Qasm2ParserVisitor<'input> + 'a> for ExplistContext<'input>{
	fn accept(&self,visitor: &mut (dyn Qasm2ParserVisitor<'input> + 'a)) {
		visitor.visit_explist(self);
	}
}

impl<'input> CustomRuleContext<'input> for ExplistContextExt<'input>{
	type TF = LocalTokenFactory<'input>;
	type Ctx = Qasm2ParserContextType;
	fn get_rule_index(&self) -> usize { RULE_explist }
	//fn type_rule_index() -> usize where Self: Sized { RULE_explist }
}
antlr_rust::tid!{ExplistContextExt<'a>}

impl<'input> ExplistContextExt<'input>{
	fn new(parent: Option<Rc<dyn Qasm2ParserContext<'input> + 'input > >, invoking_state: isize) -> Rc<ExplistContextAll<'input>> {
		Rc::new(
			BaseParserRuleContext::new_parser_ctx(parent, invoking_state,ExplistContextExt{
				ph:PhantomData
			}),
		)
	}
}

pub trait ExplistContextAttrs<'input>: Qasm2ParserContext<'input> + BorrowMut<ExplistContextExt<'input>>{

fn exp_all(&self) ->  Vec<Rc<ExpContextAll<'input>>> where Self:Sized{
	self.children_of_type()
}
fn exp(&self, i: usize) -> Option<Rc<ExpContextAll<'input>>> where Self:Sized{
	self.child_of_type(i)
}
/// Retrieves all `TerminalNode`s corresponding to token COMMA in current rule
fn COMMA_all(&self) -> Vec<Rc<TerminalNode<'input,Qasm2ParserContextType>>>  where Self:Sized{
	self.children_of_type()
}
/// Retrieves 'i's TerminalNode corresponding to token COMMA, starting from 0.
/// Returns `None` if number of children corresponding to token COMMA is less or equal than `i`.
fn COMMA(&self, i: usize) -> Option<Rc<TerminalNode<'input,Qasm2ParserContextType>>> where Self:Sized{
	self.get_token(COMMA, i)
}

}

impl<'input> ExplistContextAttrs<'input> for ExplistContext<'input>{}

impl<'input, I, H> Qasm2Parser<'input, I, H>
where
    I: TokenStream<'input, TF = LocalTokenFactory<'input> > + TidAble<'input>,
    H: ErrorStrategy<'input,BaseParserType<'input,I>>
{
	pub fn explist(&mut self,)
	-> Result<Rc<ExplistContextAll<'input>>,ANTLRError> {
		let mut recog = self;
		let _parentctx = recog.ctx.take();
		let mut _localctx = ExplistContextExt::new(_parentctx.clone(), recog.base.get_state());
        recog.base.enter_rule(_localctx.clone(), 30, RULE_explist);
        let mut _localctx: Rc<ExplistContextAll> = _localctx;
		let mut _la: isize = -1;
		let result: Result<(), ANTLRError> = (|| {

			//recog.base.enter_outer_alt(_localctx.clone(), 1);
			recog.base.enter_outer_alt(None, 1);
			{
			/*InvokeRule exp*/
			recog.base.set_state(164);
			recog.exp_rec(0)?;

			recog.base.set_state(169);
			recog.err_handler.sync(&mut recog.base)?;
			_la = recog.base.input.la(1);
			while _la==COMMA {
				{
				{
				recog.base.set_state(165);
				recog.base.match_token(COMMA,&mut recog.err_handler)?;

				/*InvokeRule exp*/
				recog.base.set_state(166);
				recog.exp_rec(0)?;

				}
				}
				recog.base.set_state(171);
				recog.err_handler.sync(&mut recog.base)?;
				_la = recog.base.input.la(1);
			}
			}
			Ok(())
		})();
		match result {
		Ok(_)=>{},
        Err(e @ ANTLRError::FallThrough(_)) => return Err(e),
		Err(ref re) => {
				//_localctx.exception = re;
				recog.err_handler.report_error(&mut recog.base, re);
				recog.err_handler.recover(&mut recog.base, re)?;
			}
		}
		recog.base.exit_rule();

		Ok(_localctx)
	}
}
//------------------- exp ----------------
#[derive(Debug)]
pub enum ExpContextAll<'input>{
	BitwiseXorExpressionContext(BitwiseXorExpressionContext<'input>),
	AdditiveExpressionContext(AdditiveExpressionContext<'input>),
	ParenthesisExpressionContext(ParenthesisExpressionContext<'input>),
	MultiplicativeExpressionContext(MultiplicativeExpressionContext<'input>),
	UnaryExpressionContext(UnaryExpressionContext<'input>),
	LiteralExpressionContext(LiteralExpressionContext<'input>),
	FunctionExpressionContext(FunctionExpressionContext<'input>),
Error(ExpContext<'input>)
}
antlr_rust::tid!{ExpContextAll<'a>}

impl<'input> antlr_rust::parser_rule_context::DerefSeal for ExpContextAll<'input>{}

impl<'input> Qasm2ParserContext<'input> for ExpContextAll<'input>{}

impl<'input> Deref for ExpContextAll<'input>{
	type Target = dyn ExpContextAttrs<'input> + 'input;
	fn deref(&self) -> &Self::Target{
		use ExpContextAll::*;
		match self{
			BitwiseXorExpressionContext(inner) => inner,
			AdditiveExpressionContext(inner) => inner,
			ParenthesisExpressionContext(inner) => inner,
			MultiplicativeExpressionContext(inner) => inner,
			UnaryExpressionContext(inner) => inner,
			LiteralExpressionContext(inner) => inner,
			FunctionExpressionContext(inner) => inner,
Error(inner) => inner
		}
	}
}
impl<'input,'a> Visitable<dyn Qasm2ParserVisitor<'input> + 'a> for ExpContextAll<'input>{
	fn accept(&self, visitor: &mut (dyn Qasm2ParserVisitor<'input> + 'a)) { self.deref().accept(visitor) }
}
impl<'input,'a> Listenable<dyn Qasm2ParserListener<'input> + 'a> for ExpContextAll<'input>{
    fn enter(&self, listener: &mut (dyn Qasm2ParserListener<'input> + 'a)) { self.deref().enter(listener) }
    fn exit(&self, listener: &mut (dyn Qasm2ParserListener<'input> + 'a)) { self.deref().exit(listener) }
}



pub type ExpContext<'input> = BaseParserRuleContext<'input,ExpContextExt<'input>>;

#[derive(Clone)]
pub struct ExpContextExt<'input>{
ph:PhantomData<&'input str>
}

impl<'input> Qasm2ParserContext<'input> for ExpContext<'input>{}

impl<'input,'a> Listenable<dyn Qasm2ParserListener<'input> + 'a> for ExpContext<'input>{
}

impl<'input,'a> Visitable<dyn Qasm2ParserVisitor<'input> + 'a> for ExpContext<'input>{
}

impl<'input> CustomRuleContext<'input> for ExpContextExt<'input>{
	type TF = LocalTokenFactory<'input>;
	type Ctx = Qasm2ParserContextType;
	fn get_rule_index(&self) -> usize { RULE_exp }
	//fn type_rule_index() -> usize where Self: Sized { RULE_exp }
}
antlr_rust::tid!{ExpContextExt<'a>}

impl<'input> ExpContextExt<'input>{
	fn new(parent: Option<Rc<dyn Qasm2ParserContext<'input> + 'input > >, invoking_state: isize) -> Rc<ExpContextAll<'input>> {
		Rc::new(
		ExpContextAll::Error(
			BaseParserRuleContext::new_parser_ctx(parent, invoking_state,ExpContextExt{
				ph:PhantomData
			}),
		)
		)
	}
}

pub trait ExpContextAttrs<'input>: Qasm2ParserContext<'input> + BorrowMut<ExpContextExt<'input>>{


}

impl<'input> ExpContextAttrs<'input> for ExpContext<'input>{}

pub type BitwiseXorExpressionContext<'input> = BaseParserRuleContext<'input,BitwiseXorExpressionContextExt<'input>>;

pub trait BitwiseXorExpressionContextAttrs<'input>: Qasm2ParserContext<'input>{
	fn exp_all(&self) ->  Vec<Rc<ExpContextAll<'input>>> where Self:Sized{
		self.children_of_type()
	}
	fn exp(&self, i: usize) -> Option<Rc<ExpContextAll<'input>>> where Self:Sized{
		self.child_of_type(i)
	}
	/// Retrieves first TerminalNode corresponding to token CARET
	/// Returns `None` if there is no child corresponding to token CARET
	fn CARET(&self) -> Option<Rc<TerminalNode<'input,Qasm2ParserContextType>>> where Self:Sized{
		self.get_token(CARET, 0)
	}
}

impl<'input> BitwiseXorExpressionContextAttrs<'input> for BitwiseXorExpressionContext<'input>{}

pub struct BitwiseXorExpressionContextExt<'input>{
	base:ExpContextExt<'input>,
	pub lhs: Option<Rc<ExpContextAll<'input>>>,
	pub op: Option<TokenType<'input>>,
	pub rhs: Option<Rc<ExpContextAll<'input>>>,
	ph:PhantomData<&'input str>
}

antlr_rust::tid!{BitwiseXorExpressionContextExt<'a>}

impl<'input> Qasm2ParserContext<'input> for BitwiseXorExpressionContext<'input>{}

impl<'input,'a> Listenable<dyn Qasm2ParserListener<'input> + 'a> for BitwiseXorExpressionContext<'input>{
	fn enter(&self,listener: &mut (dyn Qasm2ParserListener<'input> + 'a)) {
		listener.enter_every_rule(self);
		listener.enter_bitwiseXorExpression(self);
	}
	fn exit(&self,listener: &mut (dyn Qasm2ParserListener<'input> + 'a)) {
		listener.exit_bitwiseXorExpression(self);
		listener.exit_every_rule(self);
	}
}

impl<'input,'a> Visitable<dyn Qasm2ParserVisitor<'input> + 'a> for BitwiseXorExpressionContext<'input>{
	fn accept(&self,visitor: &mut (dyn Qasm2ParserVisitor<'input> + 'a)) {
		visitor.visit_bitwiseXorExpression(self);
	}
}

impl<'input> CustomRuleContext<'input> for BitwiseXorExpressionContextExt<'input>{
	type TF = LocalTokenFactory<'input>;
	type Ctx = Qasm2ParserContextType;
	fn get_rule_index(&self) -> usize { RULE_exp }
	//fn type_rule_index() -> usize where Self: Sized { RULE_exp }
}

impl<'input> Borrow<ExpContextExt<'input>> for BitwiseXorExpressionContext<'input>{
	fn borrow(&self) -> &ExpContextExt<'input> { &self.base }
}
impl<'input> BorrowMut<ExpContextExt<'input>> for BitwiseXorExpressionContext<'input>{
	fn borrow_mut(&mut self) -> &mut ExpContextExt<'input> { &mut self.base }
}

impl<'input> ExpContextAttrs<'input> for BitwiseXorExpressionContext<'input> {}

impl<'input> BitwiseXorExpressionContextExt<'input>{
	fn new(ctx: &dyn ExpContextAttrs<'input>) -> Rc<ExpContextAll<'input>>  {
		Rc::new(
			ExpContextAll::BitwiseXorExpressionContext(
				BaseParserRuleContext::copy_from(ctx,BitwiseXorExpressionContextExt{
					op:None, 
        			lhs:None, rhs:None, 
        			base: ctx.borrow().clone(),
        			ph:PhantomData
				})
			)
		)
	}
}

pub type AdditiveExpressionContext<'input> = BaseParserRuleContext<'input,AdditiveExpressionContextExt<'input>>;

pub trait AdditiveExpressionContextAttrs<'input>: Qasm2ParserContext<'input>{
	fn exp_all(&self) ->  Vec<Rc<ExpContextAll<'input>>> where Self:Sized{
		self.children_of_type()
	}
	fn exp(&self, i: usize) -> Option<Rc<ExpContextAll<'input>>> where Self:Sized{
		self.child_of_type(i)
	}
	/// Retrieves first TerminalNode corresponding to token PLUS
	/// Returns `None` if there is no child corresponding to token PLUS
	fn PLUS(&self) -> Option<Rc<TerminalNode<'input,Qasm2ParserContextType>>> where Self:Sized{
		self.get_token(PLUS, 0)
	}
	/// Retrieves first TerminalNode corresponding to token MINUS
	/// Returns `None` if there is no child corresponding to token MINUS
	fn MINUS(&self) -> Option<Rc<TerminalNode<'input,Qasm2ParserContextType>>> where Self:Sized{
		self.get_token(MINUS, 0)
	}
}

impl<'input> AdditiveExpressionContextAttrs<'input> for AdditiveExpressionContext<'input>{}

pub struct AdditiveExpressionContextExt<'input>{
	base:ExpContextExt<'input>,
	pub lhs: Option<Rc<ExpContextAll<'input>>>,
	pub op: Option<TokenType<'input>>,
	pub rhs: Option<Rc<ExpContextAll<'input>>>,
	ph:PhantomData<&'input str>
}

antlr_rust::tid!{AdditiveExpressionContextExt<'a>}

impl<'input> Qasm2ParserContext<'input> for AdditiveExpressionContext<'input>{}

impl<'input,'a> Listenable<dyn Qasm2ParserListener<'input> + 'a> for AdditiveExpressionContext<'input>{
	fn enter(&self,listener: &mut (dyn Qasm2ParserListener<'input> + 'a)) {
		listener.enter_every_rule(self);
		listener.enter_additiveExpression(self);
	}
	fn exit(&self,listener: &mut (dyn Qasm2ParserListener<'input> + 'a)) {
		listener.exit_additiveExpression(self);
		listener.exit_every_rule(self);
	}
}

impl<'input,'a> Visitable<dyn Qasm2ParserVisitor<'input> + 'a> for AdditiveExpressionContext<'input>{
	fn accept(&self,visitor: &mut (dyn Qasm2ParserVisitor<'input> + 'a)) {
		visitor.visit_additiveExpression(self);
	}
}

impl<'input> CustomRuleContext<'input> for AdditiveExpressionContextExt<'input>{
	type TF = LocalTokenFactory<'input>;
	type Ctx = Qasm2ParserContextType;
	fn get_rule_index(&self) -> usize { RULE_exp }
	//fn type_rule_index() -> usize where Self: Sized { RULE_exp }
}

impl<'input> Borrow<ExpContextExt<'input>> for AdditiveExpressionContext<'input>{
	fn borrow(&self) -> &ExpContextExt<'input> { &self.base }
}
impl<'input> BorrowMut<ExpContextExt<'input>> for AdditiveExpressionContext<'input>{
	fn borrow_mut(&mut self) -> &mut ExpContextExt<'input> { &mut self.base }
}

impl<'input> ExpContextAttrs<'input> for AdditiveExpressionContext<'input> {}

impl<'input> AdditiveExpressionContextExt<'input>{
	fn new(ctx: &dyn ExpContextAttrs<'input>) -> Rc<ExpContextAll<'input>>  {
		Rc::new(
			ExpContextAll::AdditiveExpressionContext(
				BaseParserRuleContext::copy_from(ctx,AdditiveExpressionContextExt{
					op:None, 
        			lhs:None, rhs:None, 
        			base: ctx.borrow().clone(),
        			ph:PhantomData
				})
			)
		)
	}
}

pub type ParenthesisExpressionContext<'input> = BaseParserRuleContext<'input,ParenthesisExpressionContextExt<'input>>;

pub trait ParenthesisExpressionContextAttrs<'input>: Qasm2ParserContext<'input>{
	/// Retrieves first TerminalNode corresponding to token LPAREN
	/// Returns `None` if there is no child corresponding to token LPAREN
	fn LPAREN(&self) -> Option<Rc<TerminalNode<'input,Qasm2ParserContextType>>> where Self:Sized{
		self.get_token(LPAREN, 0)
	}
	fn exp(&self) -> Option<Rc<ExpContextAll<'input>>> where Self:Sized{
		self.child_of_type(0)
	}
	/// Retrieves first TerminalNode corresponding to token RPAREN
	/// Returns `None` if there is no child corresponding to token RPAREN
	fn RPAREN(&self) -> Option<Rc<TerminalNode<'input,Qasm2ParserContextType>>> where Self:Sized{
		self.get_token(RPAREN, 0)
	}
}

impl<'input> ParenthesisExpressionContextAttrs<'input> for ParenthesisExpressionContext<'input>{}

pub struct ParenthesisExpressionContextExt<'input>{
	base:ExpContextExt<'input>,
	ph:PhantomData<&'input str>
}

antlr_rust::tid!{ParenthesisExpressionContextExt<'a>}

impl<'input> Qasm2ParserContext<'input> for ParenthesisExpressionContext<'input>{}

impl<'input,'a> Listenable<dyn Qasm2ParserListener<'input> + 'a> for ParenthesisExpressionContext<'input>{
	fn enter(&self,listener: &mut (dyn Qasm2ParserListener<'input> + 'a)) {
		listener.enter_every_rule(self);
		listener.enter_parenthesisExpression(self);
	}
	fn exit(&self,listener: &mut (dyn Qasm2ParserListener<'input> + 'a)) {
		listener.exit_parenthesisExpression(self);
		listener.exit_every_rule(self);
	}
}

impl<'input,'a> Visitable<dyn Qasm2ParserVisitor<'input> + 'a> for ParenthesisExpressionContext<'input>{
	fn accept(&self,visitor: &mut (dyn Qasm2ParserVisitor<'input> + 'a)) {
		visitor.visit_parenthesisExpression(self);
	}
}

impl<'input> CustomRuleContext<'input> for ParenthesisExpressionContextExt<'input>{
	type TF = LocalTokenFactory<'input>;
	type Ctx = Qasm2ParserContextType;
	fn get_rule_index(&self) -> usize { RULE_exp }
	//fn type_rule_index() -> usize where Self: Sized { RULE_exp }
}

impl<'input> Borrow<ExpContextExt<'input>> for ParenthesisExpressionContext<'input>{
	fn borrow(&self) -> &ExpContextExt<'input> { &self.base }
}
impl<'input> BorrowMut<ExpContextExt<'input>> for ParenthesisExpressionContext<'input>{
	fn borrow_mut(&mut self) -> &mut ExpContextExt<'input> { &mut self.base }
}

impl<'input> ExpContextAttrs<'input> for ParenthesisExpressionContext<'input> {}

impl<'input> ParenthesisExpressionContextExt<'input>{
	fn new(ctx: &dyn ExpContextAttrs<'input>) -> Rc<ExpContextAll<'input>>  {
		Rc::new(
			ExpContextAll::ParenthesisExpressionContext(
				BaseParserRuleContext::copy_from(ctx,ParenthesisExpressionContextExt{
        			base: ctx.borrow().clone(),
        			ph:PhantomData
				})
			)
		)
	}
}

pub type MultiplicativeExpressionContext<'input> = BaseParserRuleContext<'input,MultiplicativeExpressionContextExt<'input>>;

pub trait MultiplicativeExpressionContextAttrs<'input>: Qasm2ParserContext<'input>{
	fn exp_all(&self) ->  Vec<Rc<ExpContextAll<'input>>> where Self:Sized{
		self.children_of_type()
	}
	fn exp(&self, i: usize) -> Option<Rc<ExpContextAll<'input>>> where Self:Sized{
		self.child_of_type(i)
	}
	/// Retrieves first TerminalNode corresponding to token ASTERISK
	/// Returns `None` if there is no child corresponding to token ASTERISK
	fn ASTERISK(&self) -> Option<Rc<TerminalNode<'input,Qasm2ParserContextType>>> where Self:Sized{
		self.get_token(ASTERISK, 0)
	}
	/// Retrieves first TerminalNode corresponding to token SLASH
	/// Returns `None` if there is no child corresponding to token SLASH
	fn SLASH(&self) -> Option<Rc<TerminalNode<'input,Qasm2ParserContextType>>> where Self:Sized{
		self.get_token(SLASH, 0)
	}
}

impl<'input> MultiplicativeExpressionContextAttrs<'input> for MultiplicativeExpressionContext<'input>{}

pub struct MultiplicativeExpressionContextExt<'input>{
	base:ExpContextExt<'input>,
	pub lhs: Option<Rc<ExpContextAll<'input>>>,
	pub op: Option<TokenType<'input>>,
	pub rhs: Option<Rc<ExpContextAll<'input>>>,
	ph:PhantomData<&'input str>
}

antlr_rust::tid!{MultiplicativeExpressionContextExt<'a>}

impl<'input> Qasm2ParserContext<'input> for MultiplicativeExpressionContext<'input>{}

impl<'input,'a> Listenable<dyn Qasm2ParserListener<'input> + 'a> for MultiplicativeExpressionContext<'input>{
	fn enter(&self,listener: &mut (dyn Qasm2ParserListener<'input> + 'a)) {
		listener.enter_every_rule(self);
		listener.enter_multiplicativeExpression(self);
	}
	fn exit(&self,listener: &mut (dyn Qasm2ParserListener<'input> + 'a)) {
		listener.exit_multiplicativeExpression(self);
		listener.exit_every_rule(self);
	}
}

impl<'input,'a> Visitable<dyn Qasm2ParserVisitor<'input> + 'a> for MultiplicativeExpressionContext<'input>{
	fn accept(&self,visitor: &mut (dyn Qasm2ParserVisitor<'input> + 'a)) {
		visitor.visit_multiplicativeExpression(self);
	}
}

impl<'input> CustomRuleContext<'input> for MultiplicativeExpressionContextExt<'input>{
	type TF = LocalTokenFactory<'input>;
	type Ctx = Qasm2ParserContextType;
	fn get_rule_index(&self) -> usize { RULE_exp }
	//fn type_rule_index() -> usize where Self: Sized { RULE_exp }
}

impl<'input> Borrow<ExpContextExt<'input>> for MultiplicativeExpressionContext<'input>{
	fn borrow(&self) -> &ExpContextExt<'input> { &self.base }
}
impl<'input> BorrowMut<ExpContextExt<'input>> for MultiplicativeExpressionContext<'input>{
	fn borrow_mut(&mut self) -> &mut ExpContextExt<'input> { &mut self.base }
}

impl<'input> ExpContextAttrs<'input> for MultiplicativeExpressionContext<'input> {}

impl<'input> MultiplicativeExpressionContextExt<'input>{
	fn new(ctx: &dyn ExpContextAttrs<'input>) -> Rc<ExpContextAll<'input>>  {
		Rc::new(
			ExpContextAll::MultiplicativeExpressionContext(
				BaseParserRuleContext::copy_from(ctx,MultiplicativeExpressionContextExt{
					op:None, 
        			lhs:None, rhs:None, 
        			base: ctx.borrow().clone(),
        			ph:PhantomData
				})
			)
		)
	}
}

pub type UnaryExpressionContext<'input> = BaseParserRuleContext<'input,UnaryExpressionContextExt<'input>>;

pub trait UnaryExpressionContextAttrs<'input>: Qasm2ParserContext<'input>{
	fn exp(&self) -> Option<Rc<ExpContextAll<'input>>> where Self:Sized{
		self.child_of_type(0)
	}
	/// Retrieves first TerminalNode corresponding to token MINUS
	/// Returns `None` if there is no child corresponding to token MINUS
	fn MINUS(&self) -> Option<Rc<TerminalNode<'input,Qasm2ParserContextType>>> where Self:Sized{
		self.get_token(MINUS, 0)
	}
}

impl<'input> UnaryExpressionContextAttrs<'input> for UnaryExpressionContext<'input>{}

pub struct UnaryExpressionContextExt<'input>{
	base:ExpContextExt<'input>,
	pub op: Option<TokenType<'input>>,
	ph:PhantomData<&'input str>
}

antlr_rust::tid!{UnaryExpressionContextExt<'a>}

impl<'input> Qasm2ParserContext<'input> for UnaryExpressionContext<'input>{}

impl<'input,'a> Listenable<dyn Qasm2ParserListener<'input> + 'a> for UnaryExpressionContext<'input>{
	fn enter(&self,listener: &mut (dyn Qasm2ParserListener<'input> + 'a)) {
		listener.enter_every_rule(self);
		listener.enter_unaryExpression(self);
	}
	fn exit(&self,listener: &mut (dyn Qasm2ParserListener<'input> + 'a)) {
		listener.exit_unaryExpression(self);
		listener.exit_every_rule(self);
	}
}

impl<'input,'a> Visitable<dyn Qasm2ParserVisitor<'input> + 'a> for UnaryExpressionContext<'input>{
	fn accept(&self,visitor: &mut (dyn Qasm2ParserVisitor<'input> + 'a)) {
		visitor.visit_unaryExpression(self);
	}
}

impl<'input> CustomRuleContext<'input> for UnaryExpressionContextExt<'input>{
	type TF = LocalTokenFactory<'input>;
	type Ctx = Qasm2ParserContextType;
	fn get_rule_index(&self) -> usize { RULE_exp }
	//fn type_rule_index() -> usize where Self: Sized { RULE_exp }
}

impl<'input> Borrow<ExpContextExt<'input>> for UnaryExpressionContext<'input>{
	fn borrow(&self) -> &ExpContextExt<'input> { &self.base }
}
impl<'input> BorrowMut<ExpContextExt<'input>> for UnaryExpressionContext<'input>{
	fn borrow_mut(&mut self) -> &mut ExpContextExt<'input> { &mut self.base }
}

impl<'input> ExpContextAttrs<'input> for UnaryExpressionContext<'input> {}

impl<'input> UnaryExpressionContextExt<'input>{
	fn new(ctx: &dyn ExpContextAttrs<'input>) -> Rc<ExpContextAll<'input>>  {
		Rc::new(
			ExpContextAll::UnaryExpressionContext(
				BaseParserRuleContext::copy_from(ctx,UnaryExpressionContextExt{
					op:None, 
        			base: ctx.borrow().clone(),
        			ph:PhantomData
				})
			)
		)
	}
}

pub type LiteralExpressionContext<'input> = BaseParserRuleContext<'input,LiteralExpressionContextExt<'input>>;

pub trait LiteralExpressionContextAttrs<'input>: Qasm2ParserContext<'input>{
	/// Retrieves first TerminalNode corresponding to token Float
	/// Returns `None` if there is no child corresponding to token Float
	fn Float(&self) -> Option<Rc<TerminalNode<'input,Qasm2ParserContextType>>> where Self:Sized{
		self.get_token(Float, 0)
	}
	/// Retrieves first TerminalNode corresponding to token Integer
	/// Returns `None` if there is no child corresponding to token Integer
	fn Integer(&self) -> Option<Rc<TerminalNode<'input,Qasm2ParserContextType>>> where Self:Sized{
		self.get_token(Integer, 0)
	}
	/// Retrieves first TerminalNode corresponding to token PI
	/// Returns `None` if there is no child corresponding to token PI
	fn PI(&self) -> Option<Rc<TerminalNode<'input,Qasm2ParserContextType>>> where Self:Sized{
		self.get_token(PI, 0)
	}
	/// Retrieves first TerminalNode corresponding to token Identifier
	/// Returns `None` if there is no child corresponding to token Identifier
	fn Identifier(&self) -> Option<Rc<TerminalNode<'input,Qasm2ParserContextType>>> where Self:Sized{
		self.get_token(Identifier, 0)
	}
}

impl<'input> LiteralExpressionContextAttrs<'input> for LiteralExpressionContext<'input>{}

pub struct LiteralExpressionContextExt<'input>{
	base:ExpContextExt<'input>,
	ph:PhantomData<&'input str>
}

antlr_rust::tid!{LiteralExpressionContextExt<'a>}

impl<'input> Qasm2ParserContext<'input> for LiteralExpressionContext<'input>{}

impl<'input,'a> Listenable<dyn Qasm2ParserListener<'input> + 'a> for LiteralExpressionContext<'input>{
	fn enter(&self,listener: &mut (dyn Qasm2ParserListener<'input> + 'a)) {
		listener.enter_every_rule(self);
		listener.enter_literalExpression(self);
	}
	fn exit(&self,listener: &mut (dyn Qasm2ParserListener<'input> + 'a)) {
		listener.exit_literalExpression(self);
		listener.exit_every_rule(self);
	}
}

impl<'input,'a> Visitable<dyn Qasm2ParserVisitor<'input> + 'a> for LiteralExpressionContext<'input>{
	fn accept(&self,visitor: &mut (dyn Qasm2ParserVisitor<'input> + 'a)) {
		visitor.visit_literalExpression(self);
	}
}

impl<'input> CustomRuleContext<'input> for LiteralExpressionContextExt<'input>{
	type TF = LocalTokenFactory<'input>;
	type Ctx = Qasm2ParserContextType;
	fn get_rule_index(&self) -> usize { RULE_exp }
	//fn type_rule_index() -> usize where Self: Sized { RULE_exp }
}

impl<'input> Borrow<ExpContextExt<'input>> for LiteralExpressionContext<'input>{
	fn borrow(&self) -> &ExpContextExt<'input> { &self.base }
}
impl<'input> BorrowMut<ExpContextExt<'input>> for LiteralExpressionContext<'input>{
	fn borrow_mut(&mut self) -> &mut ExpContextExt<'input> { &mut self.base }
}

impl<'input> ExpContextAttrs<'input> for LiteralExpressionContext<'input> {}

impl<'input> LiteralExpressionContextExt<'input>{
	fn new(ctx: &dyn ExpContextAttrs<'input>) -> Rc<ExpContextAll<'input>>  {
		Rc::new(
			ExpContextAll::LiteralExpressionContext(
				BaseParserRuleContext::copy_from(ctx,LiteralExpressionContextExt{
        			base: ctx.borrow().clone(),
        			ph:PhantomData
				})
			)
		)
	}
}

pub type FunctionExpressionContext<'input> = BaseParserRuleContext<'input,FunctionExpressionContextExt<'input>>;

pub trait FunctionExpressionContextAttrs<'input>: Qasm2ParserContext<'input>{
	/// Retrieves first TerminalNode corresponding to token LPAREN
	/// Returns `None` if there is no child corresponding to token LPAREN
	fn LPAREN(&self) -> Option<Rc<TerminalNode<'input,Qasm2ParserContextType>>> where Self:Sized{
		self.get_token(LPAREN, 0)
	}
	fn exp(&self) -> Option<Rc<ExpContextAll<'input>>> where Self:Sized{
		self.child_of_type(0)
	}
	/// Retrieves first TerminalNode corresponding to token RPAREN
	/// Returns `None` if there is no child corresponding to token RPAREN
	fn RPAREN(&self) -> Option<Rc<TerminalNode<'input,Qasm2ParserContextType>>> where Self:Sized{
		self.get_token(RPAREN, 0)
	}
	/// Retrieves first TerminalNode corresponding to token SIN
	/// Returns `None` if there is no child corresponding to token SIN
	fn SIN(&self) -> Option<Rc<TerminalNode<'input,Qasm2ParserContextType>>> where Self:Sized{
		self.get_token(SIN, 0)
	}
	/// Retrieves first TerminalNode corresponding to token COS
	/// Returns `None` if there is no child corresponding to token COS
	fn COS(&self) -> Option<Rc<TerminalNode<'input,Qasm2ParserContextType>>> where Self:Sized{
		self.get_token(COS, 0)
	}
	/// Retrieves first TerminalNode corresponding to token TAN
	/// Returns `None` if there is no child corresponding to token TAN
	fn TAN(&self) -> Option<Rc<TerminalNode<'input,Qasm2ParserContextType>>> where Self:Sized{
		self.get_token(TAN, 0)
	}
	/// Retrieves first TerminalNode corresponding to token EXP
	/// Returns `None` if there is no child corresponding to token EXP
	fn EXP(&self) -> Option<Rc<TerminalNode<'input,Qasm2ParserContextType>>> where Self:Sized{
		self.get_token(EXP, 0)
	}
	/// Retrieves first TerminalNode corresponding to token LN
	/// Returns `None` if there is no child corresponding to token LN
	fn LN(&self) -> Option<Rc<TerminalNode<'input,Qasm2ParserContextType>>> where Self:Sized{
		self.get_token(LN, 0)
	}
	/// Retrieves first TerminalNode corresponding to token SQRT
	/// Returns `None` if there is no child corresponding to token SQRT
	fn SQRT(&self) -> Option<Rc<TerminalNode<'input,Qasm2ParserContextType>>> where Self:Sized{
		self.get_token(SQRT, 0)
	}
}

impl<'input> FunctionExpressionContextAttrs<'input> for FunctionExpressionContext<'input>{}

pub struct FunctionExpressionContextExt<'input>{
	base:ExpContextExt<'input>,
	pub op: Option<TokenType<'input>>,
	ph:PhantomData<&'input str>
}

antlr_rust::tid!{FunctionExpressionContextExt<'a>}

impl<'input> Qasm2ParserContext<'input> for FunctionExpressionContext<'input>{}

impl<'input,'a> Listenable<dyn Qasm2ParserListener<'input> + 'a> for FunctionExpressionContext<'input>{
	fn enter(&self,listener: &mut (dyn Qasm2ParserListener<'input> + 'a)) {
		listener.enter_every_rule(self);
		listener.enter_functionExpression(self);
	}
	fn exit(&self,listener: &mut (dyn Qasm2ParserListener<'input> + 'a)) {
		listener.exit_functionExpression(self);
		listener.exit_every_rule(self);
	}
}

impl<'input,'a> Visitable<dyn Qasm2ParserVisitor<'input> + 'a> for FunctionExpressionContext<'input>{
	fn accept(&self,visitor: &mut (dyn Qasm2ParserVisitor<'input> + 'a)) {
		visitor.visit_functionExpression(self);
	}
}

impl<'input> CustomRuleContext<'input> for FunctionExpressionContextExt<'input>{
	type TF = LocalTokenFactory<'input>;
	type Ctx = Qasm2ParserContextType;
	fn get_rule_index(&self) -> usize { RULE_exp }
	//fn type_rule_index() -> usize where Self: Sized { RULE_exp }
}

impl<'input> Borrow<ExpContextExt<'input>> for FunctionExpressionContext<'input>{
	fn borrow(&self) -> &ExpContextExt<'input> { &self.base }
}
impl<'input> BorrowMut<ExpContextExt<'input>> for FunctionExpressionContext<'input>{
	fn borrow_mut(&mut self) -> &mut ExpContextExt<'input> { &mut self.base }
}

impl<'input> ExpContextAttrs<'input> for FunctionExpressionContext<'input> {}

impl<'input> FunctionExpressionContextExt<'input>{
	fn new(ctx: &dyn ExpContextAttrs<'input>) -> Rc<ExpContextAll<'input>>  {
		Rc::new(
			ExpContextAll::FunctionExpressionContext(
				BaseParserRuleContext::copy_from(ctx,FunctionExpressionContextExt{
					op:None, 
        			base: ctx.borrow().clone(),
        			ph:PhantomData
				})
			)
		)
	}
}

impl<'input, I, H> Qasm2Parser<'input, I, H>
where
    I: TokenStream<'input, TF = LocalTokenFactory<'input> > + TidAble<'input>,
    H: ErrorStrategy<'input,BaseParserType<'input,I>>
{
	pub fn  exp(&mut self,)
	-> Result<Rc<ExpContextAll<'input>>,ANTLRError> {
		self.exp_rec(0)
	}

	fn exp_rec(&mut self, _p: isize)
	-> Result<Rc<ExpContextAll<'input>>,ANTLRError> {
		let recog = self;
		let _parentctx = recog.ctx.take();
		let _parentState = recog.base.get_state();
		let mut _localctx = ExpContextExt::new(_parentctx.clone(), recog.base.get_state());
		recog.base.enter_recursion_rule(_localctx.clone(), 32, RULE_exp, _p);
	    let mut _localctx: Rc<ExpContextAll> = _localctx;
        let mut _prevctx = _localctx.clone();
		let _startState = 32;
		let mut _la: isize = -1;
		let result: Result<(), ANTLRError> = (|| {
			let mut _alt: isize;
			//recog.base.enter_outer_alt(_localctx.clone(), 1);
			recog.base.enter_outer_alt(None, 1);
			{
			recog.base.set_state(185);
			recog.err_handler.sync(&mut recog.base)?;
			match recog.base.input.la(1) {
			 LPAREN 
				=> {
					{
					let mut tmp = ParenthesisExpressionContextExt::new(&**_localctx);
					recog.ctx = Some(tmp.clone());
					_localctx = tmp;
					_prevctx = _localctx.clone();


					recog.base.set_state(173);
					recog.base.match_token(LPAREN,&mut recog.err_handler)?;

					/*InvokeRule exp*/
					recog.base.set_state(174);
					recog.exp_rec(0)?;

					recog.base.set_state(175);
					recog.base.match_token(RPAREN,&mut recog.err_handler)?;

					}
				}

			 MINUS 
				=> {
					{
					let mut tmp = UnaryExpressionContextExt::new(&**_localctx);
					recog.ctx = Some(tmp.clone());
					_localctx = tmp;
					_prevctx = _localctx.clone();
					recog.base.set_state(177);
					let tmp = recog.base.match_token(MINUS,&mut recog.err_handler)?;
					if let ExpContextAll::UnaryExpressionContext(ctx) = cast_mut::<_,ExpContextAll >(&mut _localctx){
					ctx.op = Some(tmp.clone()); } else {unreachable!("cant cast");}  

					/*InvokeRule exp*/
					recog.base.set_state(178);
					recog.exp_rec(6)?;

					}
				}

			 SIN | COS | TAN | EXP | LN | SQRT 
				=> {
					{
					let mut tmp = FunctionExpressionContextExt::new(&**_localctx);
					recog.ctx = Some(tmp.clone());
					_localctx = tmp;
					_prevctx = _localctx.clone();
					recog.base.set_state(179);
					if let ExpContextAll::FunctionExpressionContext(ctx) = cast_mut::<_,ExpContextAll >(&mut _localctx){
					ctx.op = recog.base.input.lt(1).cloned(); } else {unreachable!("cant cast");} 
					_la = recog.base.input.la(1);
					if { !(((((_la - 30)) & !0x3f) == 0 && ((1usize << (_la - 30)) & ((1usize << (SIN - 30)) | (1usize << (COS - 30)) | (1usize << (TAN - 30)) | (1usize << (EXP - 30)) | (1usize << (LN - 30)) | (1usize << (SQRT - 30)))) != 0)) } {
						let tmp = recog.err_handler.recover_inline(&mut recog.base)?;
						if let ExpContextAll::FunctionExpressionContext(ctx) = cast_mut::<_,ExpContextAll >(&mut _localctx){
						ctx.op = Some(tmp.clone()); } else {unreachable!("cant cast");}  

					}
					else {
						if  recog.base.input.la(1)==TOKEN_EOF { recog.base.matched_eof = true };
						recog.err_handler.report_match(&mut recog.base);
						recog.base.consume(&mut recog.err_handler);
					}
					recog.base.set_state(180);
					recog.base.match_token(LPAREN,&mut recog.err_handler)?;

					/*InvokeRule exp*/
					recog.base.set_state(181);
					recog.exp_rec(0)?;

					recog.base.set_state(182);
					recog.base.match_token(RPAREN,&mut recog.err_handler)?;

					}
				}

			 PI | Integer | Float | Identifier 
				=> {
					{
					let mut tmp = LiteralExpressionContextExt::new(&**_localctx);
					recog.ctx = Some(tmp.clone());
					_localctx = tmp;
					_prevctx = _localctx.clone();
					recog.base.set_state(184);
					_la = recog.base.input.la(1);
					if { !(_la==PI || ((((_la - 36)) & !0x3f) == 0 && ((1usize << (_la - 36)) & ((1usize << (Integer - 36)) | (1usize << (Float - 36)) | (1usize << (Identifier - 36)))) != 0)) } {
						recog.err_handler.recover_inline(&mut recog.base)?;

					}
					else {
						if  recog.base.input.la(1)==TOKEN_EOF { recog.base.matched_eof = true };
						recog.err_handler.report_match(&mut recog.base);
						recog.base.consume(&mut recog.err_handler);
					}
					}
				}

				_ => Err(ANTLRError::NoAltError(NoViableAltError::new(&mut recog.base)))?
			}

			let tmp = recog.input.lt(-1).cloned();
			recog.ctx.as_ref().unwrap().set_stop(tmp);
			recog.base.set_state(198);
			recog.err_handler.sync(&mut recog.base)?;
			_alt = recog.interpreter.adaptive_predict(18,&mut recog.base)?;
			while { _alt!=2 && _alt!=INVALID_ALT } {
				if _alt==1 {
					recog.trigger_exit_rule_event();
					_prevctx = _localctx.clone();
					{
					recog.base.set_state(196);
					recog.err_handler.sync(&mut recog.base)?;
					match  recog.interpreter.adaptive_predict(17,&mut recog.base)? {
						1 =>{
							{
							/*recRuleLabeledAltStartAction*/
							let mut tmp = AdditiveExpressionContextExt::new(&**ExpContextExt::new(_parentctx.clone(), _parentState));
							if let ExpContextAll::AdditiveExpressionContext(ctx) = cast_mut::<_,ExpContextAll >(&mut tmp){
								ctx.lhs = Some(_prevctx.clone());
							} else {unreachable!("cant cast");}
							recog.push_new_recursion_context(tmp.clone(), _startState, RULE_exp);
							_localctx = tmp;
							recog.base.set_state(187);
							if !({recog.precpred(None, 5)}) {
								Err(FailedPredicateError::new(&mut recog.base, Some("recog.precpred(None, 5)".to_owned()), None))?;
							}
							recog.base.set_state(188);
							if let ExpContextAll::AdditiveExpressionContext(ctx) = cast_mut::<_,ExpContextAll >(&mut _localctx){
							ctx.op = recog.base.input.lt(1).cloned(); } else {unreachable!("cant cast");} 
							_la = recog.base.input.la(1);
							if { !(_la==PLUS || _la==MINUS) } {
								let tmp = recog.err_handler.recover_inline(&mut recog.base)?;
								if let ExpContextAll::AdditiveExpressionContext(ctx) = cast_mut::<_,ExpContextAll >(&mut _localctx){
								ctx.op = Some(tmp.clone()); } else {unreachable!("cant cast");}  

							}
							else {
								if  recog.base.input.la(1)==TOKEN_EOF { recog.base.matched_eof = true };
								recog.err_handler.report_match(&mut recog.base);
								recog.base.consume(&mut recog.err_handler);
							}
							/*InvokeRule exp*/
							recog.base.set_state(189);
							let tmp = recog.exp_rec(6)?;
							if let ExpContextAll::AdditiveExpressionContext(ctx) = cast_mut::<_,ExpContextAll >(&mut _localctx){
							ctx.rhs = Some(tmp.clone()); } else {unreachable!("cant cast");}  

							}
						}
					,
						2 =>{
							{
							/*recRuleLabeledAltStartAction*/
							let mut tmp = MultiplicativeExpressionContextExt::new(&**ExpContextExt::new(_parentctx.clone(), _parentState));
							if let ExpContextAll::MultiplicativeExpressionContext(ctx) = cast_mut::<_,ExpContextAll >(&mut tmp){
								ctx.lhs = Some(_prevctx.clone());
							} else {unreachable!("cant cast");}
							recog.push_new_recursion_context(tmp.clone(), _startState, RULE_exp);
							_localctx = tmp;
							recog.base.set_state(190);
							if !({recog.precpred(None, 4)}) {
								Err(FailedPredicateError::new(&mut recog.base, Some("recog.precpred(None, 4)".to_owned()), None))?;
							}
							recog.base.set_state(191);
							if let ExpContextAll::MultiplicativeExpressionContext(ctx) = cast_mut::<_,ExpContextAll >(&mut _localctx){
							ctx.op = recog.base.input.lt(1).cloned(); } else {unreachable!("cant cast");} 
							_la = recog.base.input.la(1);
							if { !(_la==ASTERISK || _la==SLASH) } {
								let tmp = recog.err_handler.recover_inline(&mut recog.base)?;
								if let ExpContextAll::MultiplicativeExpressionContext(ctx) = cast_mut::<_,ExpContextAll >(&mut _localctx){
								ctx.op = Some(tmp.clone()); } else {unreachable!("cant cast");}  

							}
							else {
								if  recog.base.input.la(1)==TOKEN_EOF { recog.base.matched_eof = true };
								recog.err_handler.report_match(&mut recog.base);
								recog.base.consume(&mut recog.err_handler);
							}
							/*InvokeRule exp*/
							recog.base.set_state(192);
							let tmp = recog.exp_rec(5)?;
							if let ExpContextAll::MultiplicativeExpressionContext(ctx) = cast_mut::<_,ExpContextAll >(&mut _localctx){
							ctx.rhs = Some(tmp.clone()); } else {unreachable!("cant cast");}  

							}
						}
					,
						3 =>{
							{
							/*recRuleLabeledAltStartAction*/
							let mut tmp = BitwiseXorExpressionContextExt::new(&**ExpContextExt::new(_parentctx.clone(), _parentState));
							if let ExpContextAll::BitwiseXorExpressionContext(ctx) = cast_mut::<_,ExpContextAll >(&mut tmp){
								ctx.lhs = Some(_prevctx.clone());
							} else {unreachable!("cant cast");}
							recog.push_new_recursion_context(tmp.clone(), _startState, RULE_exp);
							_localctx = tmp;
							recog.base.set_state(193);
							if !({recog.precpred(None, 3)}) {
								Err(FailedPredicateError::new(&mut recog.base, Some("recog.precpred(None, 3)".to_owned()), None))?;
							}
							recog.base.set_state(194);
							let tmp = recog.base.match_token(CARET,&mut recog.err_handler)?;
							if let ExpContextAll::BitwiseXorExpressionContext(ctx) = cast_mut::<_,ExpContextAll >(&mut _localctx){
							ctx.op = Some(tmp.clone()); } else {unreachable!("cant cast");}  

							/*InvokeRule exp*/
							recog.base.set_state(195);
							let tmp = recog.exp_rec(4)?;
							if let ExpContextAll::BitwiseXorExpressionContext(ctx) = cast_mut::<_,ExpContextAll >(&mut _localctx){
							ctx.rhs = Some(tmp.clone()); } else {unreachable!("cant cast");}  

							}
						}

						_ => {}
					}
					} 
				}
				recog.base.set_state(200);
				recog.err_handler.sync(&mut recog.base)?;
				_alt = recog.interpreter.adaptive_predict(18,&mut recog.base)?;
			}
			}
			Ok(())
		})();
		match result {
		Ok(_) => {},
        Err(e @ ANTLRError::FallThrough(_)) => return Err(e),
		Err(ref re)=>{
			//_localctx.exception = re;
			recog.err_handler.report_error(&mut recog.base, re);
	        recog.err_handler.recover(&mut recog.base, re)?;}
		}
		recog.base.unroll_recursion_context(_parentctx);

		Ok(_localctx)
	}
}

lazy_static! {
    static ref _ATN: Arc<ATN> =
        Arc::new(ATNDeserializer::new(None).deserialize(_serializedATN.chars()));
    static ref _decision_to_DFA: Arc<Vec<antlr_rust::RwLock<DFA>>> = {
        let mut dfa = Vec::new();
        let size = _ATN.decision_to_state.len();
        for i in 0..size {
            dfa.push(DFA::new(
                _ATN.clone(),
                _ATN.get_decision_state(i),
                i as isize,
            ).into())
        }
        Arc::new(dfa)
    };
}



const _serializedATN:&'static str =
	"\x03\u{608b}\u{a72a}\u{8133}\u{b9ed}\u{417c}\u{3be7}\u{7786}\u{5964}\x03\
	\x2f\u{cc}\x04\x02\x09\x02\x04\x03\x09\x03\x04\x04\x09\x04\x04\x05\x09\x05\
	\x04\x06\x09\x06\x04\x07\x09\x07\x04\x08\x09\x08\x04\x09\x09\x09\x04\x0a\
	\x09\x0a\x04\x0b\x09\x0b\x04\x0c\x09\x0c\x04\x0d\x09\x0d\x04\x0e\x09\x0e\
	\x04\x0f\x09\x0f\x04\x10\x09\x10\x04\x11\x09\x11\x04\x12\x09\x12\x03\x02\
	\x03\x02\x07\x02\x27\x0a\x02\x0c\x02\x0e\x02\x2a\x0b\x02\x03\x02\x03\x02\
	\x03\x03\x03\x03\x03\x03\x03\x03\x03\x04\x03\x04\x03\x04\x03\x04\x03\x04\
	\x03\x04\x05\x04\x38\x0a\x04\x03\x05\x03\x05\x03\x05\x03\x05\x03\x06\x03\
	\x06\x03\x06\x03\x06\x03\x06\x03\x07\x03\x07\x03\x07\x03\x07\x05\x07\x47\
	\x0a\x07\x03\x07\x05\x07\x4a\x0a\x07\x03\x07\x03\x07\x03\x07\x07\x07\x4f\
	\x0a\x07\x0c\x07\x0e\x07\x52\x0b\x07\x03\x07\x03\x07\x03\x07\x03\x07\x03\
	\x07\x03\x07\x05\x07\x5a\x0a\x07\x03\x07\x05\x07\x5d\x0a\x07\x03\x07\x03\
	\x07\x03\x07\x05\x07\x62\x0a\x07\x03\x08\x03\x08\x03\x08\x03\x08\x03\x08\
	\x03\x08\x03\x08\x03\x08\x03\x08\x03\x08\x03\x08\x05\x08\x6f\x0a\x08\x03\
	\x09\x03\x09\x03\x09\x03\x09\x03\x09\x03\x09\x03\x09\x03\x09\x03\x0a\x03\
	\x0a\x03\x0a\x03\x0a\x03\x0b\x03\x0b\x03\x0b\x03\x0b\x03\x0b\x05\x0b\u{82}\
	\x0a\x0b\x03\x0c\x03\x0c\x03\x0c\x05\x0c\u{87}\x0a\x0c\x03\x0c\x05\x0c\u{8a}\
	\x0a\x0c\x03\x0c\x03\x0c\x03\x0c\x03\x0d\x03\x0d\x03\x0d\x07\x0d\u{92}\x0a\
	\x0d\x0c\x0d\x0e\x0d\u{95}\x0b\x0d\x03\x0e\x03\x0e\x03\x0e\x07\x0e\u{9a}\
	\x0a\x0e\x0c\x0e\x0e\x0e\u{9d}\x0b\x0e\x03\x0f\x03\x0f\x05\x0f\u{a1}\x0a\
	\x0f\x03\x10\x03\x10\x03\x10\x03\x10\x03\x11\x03\x11\x03\x11\x07\x11\u{aa}\
	\x0a\x11\x0c\x11\x0e\x11\u{ad}\x0b\x11\x03\x12\x03\x12\x03\x12\x03\x12\x03\
	\x12\x03\x12\x03\x12\x03\x12\x03\x12\x03\x12\x03\x12\x03\x12\x03\x12\x05\
	\x12\u{bc}\x0a\x12\x03\x12\x03\x12\x03\x12\x03\x12\x03\x12\x03\x12\x03\x12\
	\x03\x12\x03\x12\x07\x12\u{c7}\x0a\x12\x0c\x12\x0e\x12\u{ca}\x0b\x12\x03\
	\x12\x02\x03\x22\x13\x02\x04\x06\x08\x0a\x0c\x0e\x10\x12\x14\x16\x18\x1a\
	\x1c\x1e\x20\x22\x02\x08\x03\x02\x05\x06\x04\x02\x0e\x0f\x2d\x2d\x03\x02\
	\x20\x25\x05\x02\x0d\x0d\x26\x27\x2d\x2d\x03\x02\x1b\x1c\x03\x02\x1d\x1e\
	\x02\u{d5}\x02\x24\x03\x02\x02\x02\x04\x2d\x03\x02\x02\x02\x06\x37\x03\x02\
	\x02\x02\x08\x39\x03\x02\x02\x02\x0a\x3d\x03\x02\x02\x02\x0c\x61\x03\x02\
	\x02\x02\x0e\x6e\x03\x02\x02\x02\x10\x70\x03\x02\x02\x02\x12\x78\x03\x02\
	\x02\x02\x14\u{81}\x03\x02\x02\x02\x16\u{83}\x03\x02\x02\x02\x18\u{8e}\x03\
	\x02\x02\x02\x1a\u{96}\x03\x02\x02\x02\x1c\u{9e}\x03\x02\x02\x02\x1e\u{a2}\
	\x03\x02\x02\x02\x20\u{a6}\x03\x02\x02\x02\x22\u{bb}\x03\x02\x02\x02\x24\
	\x28\x05\x04\x03\x02\x25\x27\x05\x06\x04\x02\x26\x25\x03\x02\x02\x02\x27\
	\x2a\x03\x02\x02\x02\x28\x26\x03\x02\x02\x02\x28\x29\x03\x02\x02\x02\x29\
	\x2b\x03\x02\x02\x02\x2a\x28\x03\x02\x02\x02\x2b\x2c\x07\x02\x02\x03\x2c\
	\x03\x03\x02\x02\x02\x2d\x2e\x07\x03\x02\x02\x2e\x2f\x07\x2f\x02\x02\x2f\
	\x30\x07\x16\x02\x02\x30\x05\x03\x02\x02\x02\x31\x38\x05\x08\x05\x02\x32\
	\x38\x05\x0a\x06\x02\x33\x38\x05\x0c\x07\x02\x34\x38\x05\x0e\x08\x02\x35\
	\x38\x05\x10\x09\x02\x36\x38\x05\x12\x0a\x02\x37\x31\x03\x02\x02\x02\x37\
	\x32\x03\x02\x02\x02\x37\x33\x03\x02\x02\x02\x37\x34\x03\x02\x02\x02\x37\
	\x35\x03\x02\x02\x02\x37\x36\x03\x02\x02\x02\x38\x07\x03\x02\x02\x02\x39\
	\x3a\x07\x04\x02\x02\x3a\x3b\x07\x28\x02\x02\x3b\x3c\x07\x16\x02\x02\x3c\
	\x09\x03\x02\x02\x02\x3d\x3e\x09\x02\x02\x02\x3e\x3f\x07\x2d\x02\x02\x3f\
	\x40\x05\x1e\x10\x02\x40\x41\x07\x16\x02\x02\x41\x0b\x03\x02\x02\x02\x42\
	\x43\x07\x07\x02\x02\x43\x49\x07\x2d\x02\x02\x44\x46\x07\x14\x02\x02\x45\
	\x47\x05\x18\x0d\x02\x46\x45\x03\x02\x02\x02\x46\x47\x03\x02\x02\x02\x47\
	\x48\x03\x02\x02\x02\x48\x4a\x07\x15\x02\x02\x49\x44\x03\x02\x02\x02\x49\
	\x4a\x03\x02\x02\x02\x4a\x4b\x03\x02\x02\x02\x4b\x4c\x05\x18\x0d\x02\x4c\
	\x50\x07\x12\x02\x02\x4d\x4f\x05\x14\x0b\x02\x4e\x4d\x03\x02\x02\x02\x4f\
	\x52\x03\x02\x02\x02\x50\x4e\x03\x02\x02\x02\x50\x51\x03\x02\x02\x02\x51\
	\x53\x03\x02\x02\x02\x52\x50\x03\x02\x02\x02\x53\x54\x07\x13\x02\x02\x54\
	\x62\x03\x02\x02\x02\x55\x56\x07\x08\x02\x02\x56\x5c\x07\x2d\x02\x02\x57\
	\x59\x07\x14\x02\x02\x58\x5a\x05\x18\x0d\x02\x59\x58\x03\x02\x02\x02\x59\
	\x5a\x03\x02\x02\x02\x5a\x5b\x03\x02\x02\x02\x5b\x5d\x07\x15\x02\x02\x5c\
	\x57\x03\x02\x02\x02\x5c\x5d\x03\x02\x02\x02\x5d\x5e\x03\x02\x02\x02\x5e\
	\x5f\x05\x18\x0d\x02\x5f\x60\x07\x16\x02\x02\x60\x62\x03\x02\x02\x02\x61\
	\x42\x03\x02\x02\x02\x61\x55\x03\x02\x02\x02\x62\x0d\x03\x02\x02\x02\x63\
	\x6f\x05\x16\x0c\x02\x64\x65\x07\x0a\x02\x02\x65\x66\x05\x1c\x0f\x02\x66\
	\x67\x07\x19\x02\x02\x67\x68\x05\x1c\x0f\x02\x68\x69\x07\x16\x02\x02\x69\
	\x6f\x03\x02\x02\x02\x6a\x6b\x07\x09\x02\x02\x6b\x6c\x05\x1c\x0f\x02\x6c\
	\x6d\x07\x16\x02\x02\x6d\x6f\x03\x02\x02\x02\x6e\x63\x03\x02\x02\x02\x6e\
	\x64\x03\x02\x02\x02\x6e\x6a\x03\x02\x02\x02\x6f\x0f\x03\x02\x02\x02\x70\
	\x71\x07\x0c\x02\x02\x71\x72\x07\x14\x02\x02\x72\x73\x07\x2d\x02\x02\x73\
	\x74\x07\x1a\x02\x02\x74\x75\x07\x26\x02\x02\x75\x76\x07\x15\x02\x02\x76\
	\x77\x05\x0e\x08\x02\x77\x11\x03\x02\x02\x02\x78\x79\x07\x0b\x02\x02\x79\
	\x7a\x05\x1a\x0e\x02\x7a\x7b\x07\x16\x02\x02\x7b\x13\x03\x02\x02\x02\x7c\
	\u{82}\x05\x16\x0c\x02\x7d\x7e\x07\x0b\x02\x02\x7e\x7f\x05\x18\x0d\x02\x7f\
	\u{80}\x07\x16\x02\x02\u{80}\u{82}\x03\x02\x02\x02\u{81}\x7c\x03\x02\x02\
	\x02\u{81}\x7d\x03\x02\x02\x02\u{82}\x15\x03\x02\x02\x02\u{83}\u{89}\x09\
	\x03\x02\x02\u{84}\u{86}\x07\x14\x02\x02\u{85}\u{87}\x05\x20\x11\x02\u{86}\
	\u{85}\x03\x02\x02\x02\u{86}\u{87}\x03\x02\x02\x02\u{87}\u{88}\x03\x02\x02\
	\x02\u{88}\u{8a}\x07\x15\x02\x02\u{89}\u{84}\x03\x02\x02\x02\u{89}\u{8a}\
	\x03\x02\x02\x02\u{8a}\u{8b}\x03\x02\x02\x02\u{8b}\u{8c}\x05\x1a\x0e\x02\
	\u{8c}\u{8d}\x07\x16\x02\x02\u{8d}\x17\x03\x02\x02\x02\u{8e}\u{93}\x07\x2d\
	\x02\x02\u{8f}\u{90}\x07\x17\x02\x02\u{90}\u{92}\x07\x2d\x02\x02\u{91}\u{8f}\
	\x03\x02\x02\x02\u{92}\u{95}\x03\x02\x02\x02\u{93}\u{91}\x03\x02\x02\x02\
	\u{93}\u{94}\x03\x02\x02\x02\u{94}\x19\x03\x02\x02\x02\u{95}\u{93}\x03\x02\
	\x02\x02\u{96}\u{9b}\x05\x1c\x0f\x02\u{97}\u{98}\x07\x17\x02\x02\u{98}\u{9a}\
	\x05\x1c\x0f\x02\u{99}\u{97}\x03\x02\x02\x02\u{9a}\u{9d}\x03\x02\x02\x02\
	\u{9b}\u{99}\x03\x02\x02\x02\u{9b}\u{9c}\x03\x02\x02\x02\u{9c}\x1b\x03\x02\
	\x02\x02\u{9d}\u{9b}\x03\x02\x02\x02\u{9e}\u{a0}\x07\x2d\x02\x02\u{9f}\u{a1}\
	\x05\x1e\x10\x02\u{a0}\u{9f}\x03\x02\x02\x02\u{a0}\u{a1}\x03\x02\x02\x02\
	\u{a1}\x1d\x03\x02\x02\x02\u{a2}\u{a3}\x07\x10\x02\x02\u{a3}\u{a4}\x07\x26\
	\x02\x02\u{a4}\u{a5}\x07\x11\x02\x02\u{a5}\x1f\x03\x02\x02\x02\u{a6}\u{ab}\
	\x05\x22\x12\x02\u{a7}\u{a8}\x07\x17\x02\x02\u{a8}\u{aa}\x05\x22\x12\x02\
	\u{a9}\u{a7}\x03\x02\x02\x02\u{aa}\u{ad}\x03\x02\x02\x02\u{ab}\u{a9}\x03\
	\x02\x02\x02\u{ab}\u{ac}\x03\x02\x02\x02\u{ac}\x21\x03\x02\x02\x02\u{ad}\
	\u{ab}\x03\x02\x02\x02\u{ae}\u{af}\x08\x12\x01\x02\u{af}\u{b0}\x07\x14\x02\
	\x02\u{b0}\u{b1}\x05\x22\x12\x02\u{b1}\u{b2}\x07\x15\x02\x02\u{b2}\u{bc}\
	\x03\x02\x02\x02\u{b3}\u{b4}\x07\x1c\x02\x02\u{b4}\u{bc}\x05\x22\x12\x08\
	\u{b5}\u{b6}\x09\x04\x02\x02\u{b6}\u{b7}\x07\x14\x02\x02\u{b7}\u{b8}\x05\
	\x22\x12\x02\u{b8}\u{b9}\x07\x15\x02\x02\u{b9}\u{bc}\x03\x02\x02\x02\u{ba}\
	\u{bc}\x09\x05\x02\x02\u{bb}\u{ae}\x03\x02\x02\x02\u{bb}\u{b3}\x03\x02\x02\
	\x02\u{bb}\u{b5}\x03\x02\x02\x02\u{bb}\u{ba}\x03\x02\x02\x02\u{bc}\u{c8}\
	\x03\x02\x02\x02\u{bd}\u{be}\x0c\x07\x02\x02\u{be}\u{bf}\x09\x06\x02\x02\
	\u{bf}\u{c7}\x05\x22\x12\x08\u{c0}\u{c1}\x0c\x06\x02\x02\u{c1}\u{c2}\x09\
	\x07\x02\x02\u{c2}\u{c7}\x05\x22\x12\x07\u{c3}\u{c4}\x0c\x05\x02\x02\u{c4}\
	\u{c5}\x07\x1f\x02\x02\u{c5}\u{c7}\x05\x22\x12\x06\u{c6}\u{bd}\x03\x02\x02\
	\x02\u{c6}\u{c0}\x03\x02\x02\x02\u{c6}\u{c3}\x03\x02\x02\x02\u{c7}\u{ca}\
	\x03\x02\x02\x02\u{c8}\u{c6}\x03\x02\x02\x02\u{c8}\u{c9}\x03\x02\x02\x02\
	\u{c9}\x23\x03\x02\x02\x02\u{ca}\u{c8}\x03\x02\x02\x02\x15\x28\x37\x46\x49\
	\x50\x59\x5c\x61\x6e\u{81}\u{86}\u{89}\u{93}\u{9b}\u{a0}\u{ab}\u{bb}\u{c6}\
	\u{c8}";

