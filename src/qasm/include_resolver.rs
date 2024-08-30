use std::fs;
use std::ops::RangeInclusive;
use std::path::PathBuf;

use antlr_rust::common_token_stream::CommonTokenStream;
use antlr_rust::tree::{ParseTree, ParseTreeVisitorCompat, Visitable};
use antlr_rust::InputStream;
use rustc_hash::FxHashSet;

use super::qasm2lexer::Qasm2Lexer;
use super::qasm2parser::Qasm2ParserContextType;
use super::qasm2parser::{IncludeStatementContextAttrs, Qasm2Parser};
use super::qasm2parservisitor::Qasm2ParserVisitorCompat;

static QELIB: &str = include_str!("qelib1.inc");

/// Holds the position and content of an include statement in a string.
#[derive(Clone, Debug)]
struct IncludeInstruction {
    path: String,
    span: RangeInclusive<usize>,
}

/// Visitor collecting all include statements in a file.
#[derive(Debug, Default)]
struct IncludeVisitor {
    result: Vec<IncludeInstruction>,
}

impl ParseTreeVisitorCompat<'_> for IncludeVisitor {
    type Node = Qasm2ParserContextType;
    type Return = Vec<IncludeInstruction>;

    fn temp_result(&mut self) -> &mut Self::Return {
        &mut self.result
    }

    fn aggregate_results(&self, mut aggregate: Self::Return, next: Self::Return) -> Self::Return {
        aggregate.extend_from_slice(&next);
        aggregate
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
            span: start..=end,
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
    let mut already_included = FxHashSet::default();
    let includes = parse_includes(code);

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
        code.replace_range(include.span.clone(), included_text);
        already_included.insert(path);
    }
}
