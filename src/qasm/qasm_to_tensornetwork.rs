use crate::tensornetwork::tensor::Tensor;

use super::{
    ast::Visitor, expression_folder::ExpressionFolder, gate_inliner::GateInliner,
    include_resolver::expand_includes, parser::parse, tn_creator::TensorNetworkCreator,
};

/// Creates a tensor network from QASM2 code. All gates are inlined, such that only U
/// and CX gates remain. Since all qubits are initialized to zero, this method adds a
/// tensor for all initial states. The tensor network is not closed, i.e. for each
/// wire in the circuit there is an unbounded leg.
pub fn create_tensornetwork<S>(code: S) -> Tensor
where
    S: Into<String>,
{
    // Expand all includes
    let mut full_code = code.into();
    expand_includes(&mut full_code);

    // Parse to AST
    let mut program = parse(&full_code);

    // Simplify expressions (not strictly needed)
    let mut expression_folder = ExpressionFolder;
    expression_folder.visit_program(&mut program);

    // Inline gate calls
    let mut inliner = GateInliner::default();
    inliner.inline_program(&mut program);

    // Simplify expressions after inline (needed)
    let mut expression_folder = ExpressionFolder;
    expression_folder.visit_program(&mut program);

    // Create the tensornetwork
    let mut tn_creator = TensorNetworkCreator::default();
    tn_creator.create_tensornetwork(&program)
}

#[cfg(test)]
mod tests {

    use float_cmp::assert_approx_eq;
    use itertools::Itertools;

    use crate::{
        tensornetwork::{
            contraction::{contract_tensor_network, TensorContraction},
            tensor::Tensor,
        },
        types::{ContractionIndex, Vertex},
    };

    use super::create_tensornetwork;

    /// Returns whether the edge connects the two tensors.
    fn edge_connects(edge_id: usize, t1_id: usize, t2_id: usize, tn: &Tensor) -> bool {
        let edge = tn.edges().get(&edge_id).unwrap();
        if let [Vertex::Closed(from), Vertex::Closed(to)] = edge[..] {
            t1_id == from && t2_id == to || t1_id == to && t2_id == from
        } else {
            false
        }
    }

    /// Returns whether the edge is an open edge of the tensor.
    fn is_open_edge_of(edge_id: usize, t1_id: usize, tn: &Tensor) -> bool {
        let edge = tn.edges().get(&edge_id).unwrap();
        if let [Vertex::Closed(from), Vertex::Open] = edge[..] {
            t1_id == from
        } else {
            false
        }
    }

    struct IdTensor<'a> {
        id: usize,
        tensor: &'a Tensor,
    }

    fn get_quantum_tensors(tn: &Tensor) -> (Vec<IdTensor>, Vec<IdTensor>, Vec<IdTensor>) {
        let mut kets = Vec::new();
        let mut single_qubit_gates = Vec::new();
        let mut two_qubit_gates = Vec::new();
        for (tid, tensor) in tn.tensors().iter().enumerate() {
            let id: usize = tid;
            let legs = tensor.legs().len();
            match legs {
                1 => kets.push(IdTensor { id, tensor }),
                2 => single_qubit_gates.push(IdTensor { id, tensor }),
                4 => two_qubit_gates.push(IdTensor { id, tensor }),
                _ => panic!("Tensor with unexpected leg count {legs} in quantum tensor network"),
            }
        }
        (kets, single_qubit_gates, two_qubit_gates)
    }

    #[test]
    fn bell_tensornetwork_construction() {
        let code = "OPENQASM 2.0;
        include \"qelib1.inc\";
        qreg q[2];
        h q[0];
        cx q[0], q[1];
        ";
        let tn = create_tensornetwork(code);

        let (kets, single_qubit_gates, two_qubit_gates) = get_quantum_tensors(&tn);
        let [k0, k1] = kets.as_slice() else { panic!() };
        let [h] = single_qubit_gates.as_slice() else {
            panic!()
        };
        let [cx] = two_qubit_gates.as_slice() else {
            panic!()
        };

        // Find out which tensor is the first/top qubit (the one connected to the H gate tensor)
        // and which is the second/bottom qubit
        let first_qubit_id = h.tensor.legs()[1];
        let (first_qubit, second_qubit) = if first_qubit_id == k0.id {
            (k0, k1)
        } else if first_qubit_id == k1.id {
            (k1, k0)
        } else {
            panic!("H gate tensor not connected to any ket tensor");
        };

        // Check edges
        let fq_to_h_id = first_qubit.tensor.legs()[0];
        assert_eq!(h.tensor.legs()[1], fq_to_h_id);
        assert!(edge_connects(fq_to_h_id, first_qubit_id, h.id, &tn));

        let h_to_cx_c_id = h.tensor.legs()[0];
        assert_eq!(cx.tensor.legs()[2], h_to_cx_c_id);
        assert!(edge_connects(h_to_cx_c_id, h.id, cx.id, &tn));

        let sq_to_cx_t_id = second_qubit.tensor.legs()[0];
        assert_eq!(cx.tensor.legs()[3], sq_to_cx_t_id);
        assert!(edge_connects(sq_to_cx_t_id, second_qubit.id, cx.id, &tn));

        let cx_c_to_open_id = cx.tensor.legs()[0];
        assert!(is_open_edge_of(cx_c_to_open_id, cx.id, &tn));

        let cx_t_to_open_id = cx.tensor.legs()[1];
        assert!(is_open_edge_of(cx_t_to_open_id, cx.id, &tn));
    }

    #[test]
    fn bell_contract() {
        let code = "OPENQASM 2.0;
        include \"qelib1.inc\";
        qreg q[2];
        h q[0];
        cx q[0], q[1];
        ";
        let mut tn = create_tensornetwork(code);
        let opt_path = (1..tn.tensors().len())
            .map(|tid| ContractionIndex::Pair(0, tid))
            .collect_vec();
        contract_tensor_network(&mut tn, &opt_path);

        let resulting_state = tn.get_data();
        assert_eq!(resulting_state.shape(), &[2, 2]);
        assert_approx_eq!(f64, resulting_state.get(&[0, 0]).re, 1.0 / 2.0f64.sqrt());
        assert_approx_eq!(f64, resulting_state.get(&[0, 0]).im, 0.0);
        assert_approx_eq!(f64, resulting_state.get(&[0, 1]).re, 0.0);
        assert_approx_eq!(f64, resulting_state.get(&[0, 1]).im, 0.0);
        assert_approx_eq!(f64, resulting_state.get(&[1, 0]).re, 0.0);
        assert_approx_eq!(f64, resulting_state.get(&[1, 0]).im, 0.0);
        assert_approx_eq!(f64, resulting_state.get(&[1, 1]).re, 1.0 / 2.0f64.sqrt());
        assert_approx_eq!(f64, resulting_state.get(&[1, 1]).im, 0.0);
    }
}
