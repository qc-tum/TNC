use std::collections::HashMap;
use std::rc::Rc;

use antlr_rust::common_token_stream::CommonTokenStream;
use antlr_rust::tree::{ParseTree, ParseTreeVisitorCompat, Visitable};
use antlr_rust::InputStream;

use crate::tensornetwork::tensor::Tensor;

use super::qasm2parser::{
    ArgumentContextAttrs, DeclarationContextAttrs, DesignatorContextAttrs, GateCallContextAttrs,
    MixedlistContextAttrs,
};
use super::{
    include_resolver::expand_includes,
    qasm2lexer::Qasm2Lexer,
    qasm2parser::{Qasm2Parser, Qasm2ParserContextType},
    qasm2parservisitor::Qasm2ParserVisitorCompat,
};

macro_rules! cast {
    ($target: expr, $pat: path) => {{
        if let $pat(a) = $target {
            a
        } else {
            panic!("Could not cast to {}", stringify!($pat));
        }
    }};
}

type EdgeId = i32;

#[derive(Debug, PartialEq, Eq, Hash)]
struct Qubit {
    qreg: Rc<String>,
    index: i32,
}

#[derive(Debug)]
enum ReturnVal {
    Void,
    Int(i32),
    Float(f64),
    Identifier(Rc<String>),
    QRef(Qubit),
    List(Vec<ReturnVal>),
}

impl Default for ReturnVal {
    fn default() -> Self {
        Self::Void
    }
}

#[derive(Debug, Default)]
struct CreatorVisitor {
    tmp: ReturnVal,
    edge_counter: EdgeId,
    tensors: Vec<Tensor>,

    /// Maps qubits to the last open edge on the corresponding wire
    wires: HashMap<Qubit, EdgeId>,
}

impl CreatorVisitor {
    pub fn new_edge(&mut self) -> EdgeId {
        let res = self.edge_counter;
        self.edge_counter += 1;
        res
    }
}

impl ParseTreeVisitorCompat<'_> for CreatorVisitor {
    type Node = Qasm2ParserContextType;
    type Return = ReturnVal;

    fn temp_result(&mut self) -> &mut Self::Return {
        &mut self.tmp
    }
}

impl Qasm2ParserVisitorCompat<'_> for CreatorVisitor {
    fn visit_declaration(&mut self, ctx: &super::qasm2parser::DeclarationContext) -> Self::Return {
        // Only consider qregs
        if ctx.QREG().is_some() {
            let name = ctx.Identifier().unwrap().get_text();
            let designator = ctx.designator().unwrap();
            let res = self.visit(&*designator);
            let count = cast!(res, ReturnVal::Int);

            // New wires are initialized with |0>
            // Thus, create new tensors with a single edge
            let name = Rc::new(name);
            self.tensors.reserve(count as usize);
            for i in 0..count {
                let edge = self.new_edge();
                self.tensors.push(Tensor::new(vec![edge]));
                self.wires.insert(
                    Qubit {
                        qreg: name.clone(),
                        index: i,
                    },
                    edge,
                );
            }
        }
        ReturnVal::Void
    }

    fn visit_designator(&mut self, ctx: &super::qasm2parser::DesignatorContext) -> Self::Return {
        let number = ctx.Integer().unwrap().get_text();
        let number = number.parse().unwrap();
        ReturnVal::Int(number)
    }

    fn visit_gateCall(&mut self, ctx: &super::qasm2parser::GateCallContext) -> Self::Return {
        // Parse quantum arguments
        let ml = ctx.mixedlist().unwrap();
        let res = self.visit(&*ml);
        let list = cast!(res, ReturnVal::List);
        let qargs: Vec<&Qubit> = list.iter().map(|ret| cast!(ret, ReturnVal::QRef)).collect();

        if ctx.U().is_some() {
            assert_eq!(qargs.len(), 1);
            
            let in_edge = self.wires[qargs[0]];
            let out_edge = self.new_edge();
            self.tensors.push(Tensor::new(vec![out_edge, in_edge]));
            *self.wires.get_mut(qargs[0]).unwrap() = out_edge;
        } else if ctx.CX().is_some() {
            assert_eq!(qargs.len(), 2);

            let in_edge1 = self.wires[qargs[0]];
            let in_edge2 = self.wires[qargs[1]];
            let out_edge1 = self.new_edge();
            let out_edge2 = self.new_edge();
            self.tensors.push(Tensor::new(vec![out_edge1, out_edge2, in_edge1, in_edge2]));
            *self.wires.get_mut(qargs[0]).unwrap() = out_edge1;
            *self.wires.get_mut(qargs[1]).unwrap() = out_edge2;
        } else {
            todo!("User-defined gates are not yet supported ({})", ctx.Identifier().unwrap().get_text());
        }
        
        
        ReturnVal::Void
    }

    fn visit_mixedlist(&mut self, ctx: &super::qasm2parser::MixedlistContext) -> Self::Return {
        let results = ctx
            .argument_all()
            .iter()
            .map(|arg_ctx| self.visit(&**arg_ctx))
            .collect();
        ReturnVal::List(results)
    }

    fn visit_argument(&mut self, ctx: &super::qasm2parser::ArgumentContext) -> Self::Return {
        let name = ctx.Identifier().unwrap().get_text();
        let name = Rc::new(name);
        if let Some(des) = ctx.designator() {
            let res = self.visit(&*des);
            let index = cast!(res, ReturnVal::Int);
            ReturnVal::QRef(Qubit { qreg: name, index })
        } else {
            ReturnVal::Identifier(name)
        }
    }
}

pub fn create_tensornetwork(code: &str) {
    let mut full_code = String::from(code);
    expand_includes(&mut full_code);

    let lexer = Qasm2Lexer::new(InputStream::new(full_code.as_str()));
    let token_source = CommonTokenStream::new(lexer);
    let mut parser = Qasm2Parser::new(token_source);
    let parsed = parser.program().unwrap();

    let mut visitor = CreatorVisitor::default();
    parsed.accept(&mut visitor);

    println!("{:#?}", visitor);
}

#[cfg(test)]
mod tests {

    use std::fs;

    use super::create_tensornetwork;

    #[test]
    fn it_works() {
        let code = fs::read_to_string("test.qasm").unwrap();
        create_tensornetwork(&code);
    }
}
