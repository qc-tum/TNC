use crate::builders::circuit_builder::Circuit;
use crate::qasm::{
    ast::Visitor, circuit_creator::CircuitCreator, expression_folder::ExpressionFolder,
    gate_inliner::GateInliner, include_resolver::expand_includes, parser::parse,
};

/// Creates a [`Circuit`] from OpenQASM2 code.
///
/// All gates are inlined up to the known gates defined in [`crate::gates`]. All
/// qubits are initialized to zero. Note that not all QASM instructions are
/// supported, such as `measure` or `if`.
pub fn import_qasm<S>(code: S) -> Circuit
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

    // Create the circuit
    let mut circuit_creator = CircuitCreator;
    circuit_creator.create_circuit(&program)
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::f64::consts::FRAC_1_SQRT_2;

    use float_cmp::assert_approx_eq;
    use itertools::Itertools;
    use num_complex::{c64, Complex64};

    use crate::{
        builders::circuit_builder::Permutor,
        contractionpath::ContractionIndex,
        tensornetwork::{
            contraction::contract_tensor_network,
            tensor::{EdgeIndex, Tensor, TensorIndex},
            tensordata::TensorData,
        },
    };

    /// Returns whether the edge connects the two tensors.
    fn edge_connects(
        edge_id: EdgeIndex,
        t1_id: TensorIndex,
        t2_id: TensorIndex,
        tn: &Tensor,
    ) -> bool {
        let overlap = tn.tensor(t1_id) & tn.tensor(t2_id);
        overlap.legs().contains(&edge_id)
    }

    /// Returns whether the edge is an open edge of the tensor.
    fn is_open_edge_of(edge_id: EdgeIndex, t1_id: TensorIndex, tn: &Tensor) -> bool {
        // Check if the edge is a leg of the tensor
        if !tn.tensor(t1_id).legs().contains(&edge_id) {
            return false;
        }

        // Check if the edge is not connected to any other tensor
        for (tensor_id, tensor) in tn.tensors().iter().enumerate() {
            if tensor_id != t1_id && tensor.legs().contains(&edge_id) {
                return false;
            }
        }
        true
    }

    struct IdTensor<'a> {
        id: usize,
        tensor: &'a Tensor,
    }

    fn get_quantum_tensors(
        tn: &Tensor,
    ) -> (Vec<IdTensor<'_>>, Vec<IdTensor<'_>>, Vec<IdTensor<'_>>) {
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
        let circuit = import_qasm(code);
        let (tn, _) = circuit.into_statevector_network();

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
        let first_qubit_id = h.tensor.legs()[0];
        let (first_qubit, second_qubit) = if first_qubit_id == k0.id {
            (k0, k1)
        } else if first_qubit_id == k1.id {
            (k1, k0)
        } else {
            panic!("H gate tensor not connected to any ket tensor");
        };

        // Check edges
        let fq_to_h_id = first_qubit.tensor.legs()[0];
        assert_eq!(h.tensor.legs()[0], fq_to_h_id);
        assert!(edge_connects(fq_to_h_id, first_qubit_id, h.id, &tn));

        let sq_to_cx_t_id = second_qubit.tensor.legs()[0];
        assert_eq!(cx.tensor.legs()[1], sq_to_cx_t_id);
        assert!(edge_connects(sq_to_cx_t_id, second_qubit.id, cx.id, &tn));

        let h_to_cx_c_id = h.tensor.legs()[1];
        assert_eq!(cx.tensor.legs()[0], h_to_cx_c_id);
        assert!(edge_connects(h_to_cx_c_id, h.id, cx.id, &tn));

        let cx_c_to_open_id = cx.tensor.legs()[2];
        assert!(is_open_edge_of(cx_c_to_open_id, cx.id, &tn));

        let cx_t_to_open_id = cx.tensor.legs()[3];
        assert!(is_open_edge_of(cx_t_to_open_id, cx.id, &tn));
    }

    /// Contracts the tensor network with an arbitrary contraction order, then
    /// returns the correctly permuted tensor data.
    fn contract_tn(tn: Tensor, perm: Permutor) -> TensorData {
        let opt_path = (1..tn.tensors().len())
            .map(|tid| ContractionIndex::Pair(0, tid))
            .collect_vec();
        let tn = contract_tensor_network(tn, &opt_path);
        let mut tn = perm.apply(tn);
        std::mem::take(&mut tn.tensordata)
    }

    #[test]
    fn bell_contract() {
        let code = "OPENQASM 2.0;
        include \"qelib1.inc\";
        qreg q[2];
        h q[0];
        cx q[0], q[1];
        ";
        let circuit = import_qasm(code);
        let (tn, perm) = circuit.into_statevector_network();
        let resulting_state = contract_tn(tn, perm);

        let expected = TensorData::new_from_data(
            &[2, 2],
            vec![
                c64(FRAC_1_SQRT_2, 0.),
                c64(0, 0),
                c64(0, 0),
                c64(FRAC_1_SQRT_2, 0.),
            ],
            None,
        );
        assert_approx_eq!(&TensorData, &resulting_state, &expected);
    }

    #[test]
    fn custom_swap() {
        let code = "OPENQASM 2.0;
        include \"qelib1.inc\";
        qreg q[2];
        gate myswap a, b {
            cx a, b;
            cx b, a;
            cx a, b;
        }
        x q[0];
        myswap q[1], q[0];
        ";
        let circuit = import_qasm(code);
        let (tn, perm) = circuit.into_statevector_network();
        let resulting_state = contract_tn(tn, perm);

        let expected = TensorData::new_from_data(
            &[2, 2],
            vec![
                Complex64::ZERO,
                Complex64::ONE,
                Complex64::ZERO,
                Complex64::ZERO,
            ],
            None,
        );
        assert_approx_eq!(&TensorData, &resulting_state, &expected);
    }

    fn odd_test_circuit() -> Circuit {
        // Test with odd numbers to check the order of the statevector is correct
        let code = "OPENQASM 2.0;
        include \"qelib1.inc\";
        qreg q[3];
        rx(0.5) q[0];
        rx(0.2) q[1];
        rx(0.3) q[2];
        cx q[0], q[1];
        cx q[1], q[2];";
        import_qasm(code)
    }

    #[test]
    fn statevector_order() {
        let circuit = odd_test_circuit();
        let (tn, perm) = circuit.into_statevector_network();
        let resulting_state = contract_tn(tn, perm);

        let expected = TensorData::new_from_data(
            &[2, 2, 2],
            vec![
                Complex64::new(0.953246407214305, 0.0),
                Complex64::new(0.0, -0.14406910361762032),
                Complex64::new(-0.014455126269118733, 0.0),
                Complex64::new(0.0, -0.09564366568448116),
                Complex64::new(-0.024421837348497916, 0.0),
                Complex64::new(0.0, 0.0036909997130494475),
                Complex64::new(-0.03678688170631573, 0.0),
                Complex64::new(0.0, -0.24340376901515096),
            ],
            None,
        );
        assert_approx_eq!(&TensorData, &resulting_state, &expected);
    }

    #[test]
    fn statevector_order_two_fixed_qubits() {
        let circuit = odd_test_circuit();
        // 1*0 should get a vec with amplitudes |100> and |110>
        let (tn, perm) = circuit.into_amplitude_network("1*0");
        let resulting_state = contract_tn(tn, perm);

        let expected = TensorData::new_from_data(
            &[2],
            vec![
                Complex64::new(-0.024421837348497916, 0.0),
                Complex64::new(-0.03678688170631573, 0.0),
            ],
            None,
        );
        assert_approx_eq!(&TensorData, &resulting_state, &expected);
    }

    #[test]
    fn statevector_order_one_fixed_qubit() {
        let circuit = odd_test_circuit();
        // *1* should get a vec with amplitudes |010>, |011>, |110>, |111>
        let (tn, perm) = circuit.into_amplitude_network("*1*");
        let resulting_state = contract_tn(tn, perm);

        let expected = TensorData::new_from_data(
            &[2, 2],
            vec![
                Complex64::new(-0.014455126269118733, 0.0),
                Complex64::new(0.0, -0.09564366568448116),
                Complex64::new(-0.03678688170631573, 0.0),
                Complex64::new(0.0, -0.24340376901515096),
            ],
            None,
        );
        assert_approx_eq!(&TensorData, &resulting_state, &expected);
    }
}
