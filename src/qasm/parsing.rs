use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;

use antlr_rust::common_token_stream::CommonTokenStream;
use antlr_rust::tree::{ParseTree, ParseTreeVisitorCompat};
use antlr_rust::InputStream;

use super::qasm2lexer::Qasm2Lexer;
use super::qasm2parser::Qasm2ParserContextType;
use super::qasm2parser::{IncludeStatementContextAttrs, Qasm2Parser};
use super::qasm2parservisitor::Qasm2ParserVisitorCompat;
use antlr_rust::tree::Visitable;

static QELIB: &'static str = include_str!("qelib1.inc");

/// Holds the position and content of an include statement in a string.
#[derive(Clone, Debug)]
pub struct IncludeInstruction {
    path: String,
    start: usize,
    end: usize,
}

/// Visitor collecting all include statements in a file.
pub struct IncludeVisitor {
    result: Vec<IncludeInstruction>,
}

impl Default for IncludeVisitor {
    fn default() -> Self {
        Self { result: Vec::new() }
    }
}

impl ParseTreeVisitorCompat<'_> for IncludeVisitor {
    type Node = Qasm2ParserContextType;
    type Return = Vec<IncludeInstruction>;

    fn temp_result(&mut self) -> &mut Self::Return {
        &mut self.result
    }

    fn aggregate_results(&self, aggregate: Self::Return, next: Self::Return) -> Self::Return {
        let mut res = aggregate.clone();
        res.extend_from_slice(&next);
        res
    }
}

impl Qasm2ParserVisitorCompat<'_> for IncludeVisitor {
    fn visit_includeStatement(
        &mut self,
        ctx: &super::qasm2parser::IncludeStatementContext,
    ) -> Self::Return {
        let start = ctx.INCLUDE().unwrap().symbol.start as usize;
        let end = ctx.SEMICOLON().unwrap().symbol.stop as usize;
        let include_path = ctx.StringLiteral().unwrap().get_text();
        let include_path = &include_path[1..include_path.len() - 1];
        vec![IncludeInstruction {
            path: String::from(include_path),
            start,
            end,
        }]
    }
}

/// Parses the QASM2 code and returns a list of all include instructions in the order
/// they were found.
fn parse_includes(code: &str) -> Vec<IncludeInstruction> {
    let lexer = Qasm2Lexer::new(InputStream::new(code));
    let token_source = CommonTokenStream::new(lexer);
    let mut parser = Qasm2Parser::new(token_source);
    let parsed = parser.program().unwrap();

    let mut visitor = IncludeVisitor::default();
    parsed.accept(&mut visitor);
    visitor.result
}

/// Parses the file for include statements, loads the corresponding files and
/// replaces the statements by the file contents. Includes of the standard library
/// "qelib1.inc" will directly be replaced by the corresponding code. This does only
/// resolve includes in the original file, not in the included files.
pub fn expand_includes(code: &mut String) {
    let mut already_included = HashSet::new();
    let includes = parse_includes(&code);

    for include in includes.iter().rev() {
        let is_stdlib = include.path == "qelib1.inc";

        let path = if is_stdlib {
            PathBuf::from(&include.path)
        } else {
            // Canonicalize paths to external files, such that relative and absolute
            // paths to the same file are not seen as different paths
            PathBuf::from(&include.path).canonicalize().unwrap()
        };

        // Check if the file was already included
        if already_included.contains(&path) {
            // We can safely ignore the include: the file was already included
            // earlier, hence all definitions are already available
            continue;
        }

        // Get code to include
        let file_content;
        let included_text = if is_stdlib {
            // Use builtin qelib code
            QELIB
        } else {
            // Load from external file
            file_content = fs::read_to_string(&path).unwrap();
            &file_content
        };

        // Include code in the root code
        code.replace_range(include.start..=include.end, included_text);
        already_included.insert(path);
    }
}

#[cfg(test)]
mod tests {

    use std::fs;

    use crate::qasm::parsing::expand_includes;

    #[test]
    fn it_works() {
        let mut code = fs::read_to_string("test.qasm").unwrap();
        expand_includes(&mut code);
        println!("{code}");
    }
}
