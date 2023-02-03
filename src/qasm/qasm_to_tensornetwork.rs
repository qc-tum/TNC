use crate::tensornetwork::TensorNetwork;

use super::{
    ast::Visitor, expression_folder::ExpressionFolder, gate_inliner::GateInliner,
    include_resolver::expand_includes, parser::parse, tn_creator::TensorNetworkCreator,
};

/// Creates a tensor network from QASM2 code. All gates are inlined, such that only U
/// and CX gates remain. Since all qubits are initialized to zero, this method adds a
/// tensor for all initial states. The tensor network is not closed, i.e. for each
/// wire in the circuit there is an unbounded leg.
pub fn create_tensornetwork<S>(code: S) -> TensorNetwork
where
    S: Into<String>,
{
    // Expand all includes
    let mut full_code = code.into();
    expand_includes(&mut full_code);

    // Parse to AST
    let mut program = parse(&full_code);

    // Simplify expressions (not strictly needed)
    let mut expression_folder = ExpressionFolder::default();
    expression_folder.visit_program(&mut program);

    // Inline gate calls
    let mut inliner = GateInliner::default();
    inliner.inline_program(&mut program);

    // Simplify expressions after inline (needed)
    let mut expression_folder = ExpressionFolder::default();
    expression_folder.visit_program(&mut program);

    // Create the tensornetwork
    let mut tn_creator = TensorNetworkCreator::default();
    tn_creator.create_tensornetwork(&program)
}
