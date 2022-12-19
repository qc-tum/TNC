use antlr_rust::tree::{ParseTree, ParseTreeVisitorCompat};

use super::qasm2parser::VersionContext;
use super::{
    qasm2parser::{Qasm2ParserContextType, VersionContextAttrs},
    qasm2parservisitor::Qasm2ParserVisitorCompat,
};

pub struct MyVisitor {
    result: Option<f32>,
}

impl Default for MyVisitor {
    fn default() -> Self {
        Self { result: None }
    }
}

impl ParseTreeVisitorCompat<'_> for MyVisitor {
    type Node = Qasm2ParserContextType;
    type Return = Option<f32>;

    fn temp_result(&mut self) -> &mut Self::Return {
        &mut self.result
    }

    fn aggregate_results(&self, aggregate: Self::Return, next: Self::Return) -> Self::Return {
        aggregate.or(next)
    }
}

impl Qasm2ParserVisitorCompat<'_> for MyVisitor {
    fn visit_version(&mut self, ctx: &VersionContext) -> Self::Return {
        Some(ctx.VersionSpecifier()
            .unwrap()
            .get_text()
            .parse()
            .unwrap())
    }
}

#[cfg(test)]
mod tests {

    use std::fs;

    use antlr_rust::{common_token_stream::CommonTokenStream, InputStream};

    use antlr_rust::tree::Visitable;

    use crate::qasm::{qasm2lexer::Qasm2Lexer, qasm2parser::Qasm2Parser};

    use super::MyVisitor;

    #[test]
    fn it_works() {
        let input = fs::read_to_string("test.qasm").unwrap();
        let lexer = Qasm2Lexer::new(InputStream::new(input.as_str()));
        let token_source = CommonTokenStream::new(lexer);
        let mut parser = Qasm2Parser::new(token_source);
        let result = parser.program().unwrap();
        let mut visitor = MyVisitor::default();
        result.accept(&mut visitor);
        println!("Visitor result: {:?}", visitor.result);
    }
}
