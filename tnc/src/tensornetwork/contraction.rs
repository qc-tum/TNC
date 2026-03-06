//! Functionality to contract tensor networks.
use itertools::Itertools;
use log::debug;
use tetra::contract;

use crate::{
    contractionpath::ContractionPath,
    tensornetwork::{
        tensor::{CompositeTensor, LeafTensor, Tensor},
        tensordata::TensorData,
    },
};

/// Fully contracts `tn` based on the given `contract_path` using ReplaceLeft format.
/// Returns the resulting tensor.
///
/// # Examples
/// ```
/// # use tnc::{
/// #   contractionpath::paths::{cotengrust::{Cotengrust, OptMethod}, FindPath},
/// #   builders::sycamore_circuit::sycamore_circuit,
/// #   tensornetwork::tensor::Tensor,
/// #   tensornetwork::contraction::contract_tensor_network,
/// # };
/// # use rand::rngs::StdRng;
/// # use rand::SeedableRng;
/// let mut r = StdRng::seed_from_u64(42);
/// let mut r_tn = sycamore_circuit(2, 1, &mut r).into_expectation_value_network();
/// let mut opt = Cotengrust::new(&r_tn, OptMethod::Greedy);
/// opt.find_path();
/// let opt_path = opt.get_best_replace_path();
/// let result = contract_tensor_network(r_tn, &opt_path);
/// ```
pub fn contract_tensor_network(tn: CompositeTensor, contract_path: &ContractionPath) -> LeafTensor {
    debug!(len = tn.len(); "Start contracting tensor network");

    // Wrap the tensors into options, so we can take out used tensors
    let mut tensors = tn.into_tensors().into_iter().map(Some).collect_vec();

    // Contract child composite tensors first
    for (index, inner_path) in &contract_path.nested {
        let tc = tensors[*index]
            .take()
            .and_then(Tensor::into_composite)
            .unwrap();
        let contracted = contract_tensor_network(tc, inner_path);
        tensors[*index] = Some(contracted.into());
    }

    // Contract all leaf tensors
    for (i, j) in &contract_path.toplevel {
        let ti = tensors[*i].take().and_then(Tensor::into_leaf).unwrap();
        let tj = tensors[*j].take().and_then(Tensor::into_leaf).unwrap();
        let contracted = contract_tensors(ti, tj);
        tensors[*i] = Some(contracted.into());
    }
    debug!("Completed tensor network contraction");

    // Remove all the None values and return the final tensor
    tensors
        .into_iter()
        .flatten()
        .exactly_one()
        .unwrap()
        .into_leaf()
        .unwrap()
}

fn contract_tensors(tensor_a: LeafTensor, tensor_b: LeafTensor) -> LeafTensor {
    let mut tensor_symmetric_difference = &tensor_a ^ &tensor_b;

    let (a_legs, _, a_data) = tensor_a.into_inner();
    let (b_legs, _, b_data) = tensor_b.into_inner();

    let result = contract(
        tensor_symmetric_difference.legs(),
        &a_legs,
        a_data.into_data(),
        &b_legs,
        b_data.into_data(),
    );

    tensor_symmetric_difference.set_tensor_data(TensorData::Matrix(result));
    tensor_symmetric_difference
}

#[cfg(test)]
mod tests {
    use super::*;

    use float_cmp::assert_approx_eq;
    use num_complex::Complex64;
    use rustc_hash::FxHashMap;
    use serde::Deserialize;

    use crate::{path, tensornetwork::tensordata::TensorData};

    #[derive(Debug, Deserialize)]
    struct TestTensor {
        legs: Vec<usize>,
        shape: Vec<u64>,
        data: Vec<Complex64>,
    }

    type TestData = FxHashMap<String, TestTensor>;

    static TEST_DATA: &str = include_str!("contraction_test_data.json");

    fn load_test_data() -> TestData {
        serde_json::from_str(TEST_DATA).unwrap()
    }

    fn pop_test_tensor(name: &str, data: &mut TestData) -> LeafTensor {
        let test_tensor = data.remove(name).unwrap();
        let mut tensor = LeafTensor::new(test_tensor.legs, test_tensor.shape);
        tensor.set_tensor_data(TensorData::new_from_data(
            &tensor.shape().unwrap(),
            test_tensor.data,
            None,
        ));
        tensor
    }

    #[test]
    fn test_tensor_contraction() {
        let mut data = load_test_data();
        // t1 is of shape [3, 2, 7]
        let t1 = pop_test_tensor("A", &mut data);
        // t2 is of shape [7, 8, 6]
        let t2 = pop_test_tensor("B", &mut data);
        // t3 is of shape [3, 5, 8]
        let t3 = pop_test_tensor("C", &mut data);
        // t12 is of shape [8, 6, 3, 2]
        let t12 = pop_test_tensor("AxB", &mut data);
        // t23 is of shape [3, 5, 7, 6]
        let t23 = pop_test_tensor("BxC", &mut data);

        let out = contract_tensors(t2.clone(), t1);
        assert_approx_eq!(&LeafTensor, &out, &t12, epsilon = 1e-14);

        let out = contract_tensors(t3, t2);
        assert_approx_eq!(&LeafTensor, &out, &t23, epsilon = 1e-14);
    }

    #[test]
    fn test_tn_contraction() {
        let mut data = load_test_data();
        // t1 is of shape [3, 2, 7]
        let t1 = pop_test_tensor("A", &mut data);
        // t2 is of shape [7, 8, 6]
        let t2 = pop_test_tensor("B", &mut data);
        // t3 is of shape [3, 5, 8]
        let t3 = pop_test_tensor("C", &mut data);
        // tout is of shape [5, 6, 2]
        let tout = pop_test_tensor("ABxC", &mut data);

        let tn = CompositeTensor::new(vec![t1, t2, t3]);
        let contract_path = path![(1, 0), (2, 1)];

        let result = contract_tensor_network(tn, &contract_path);
        assert_approx_eq!(&LeafTensor, &result, &tout, epsilon = 1e-14);
    }

    #[test]
    fn test_outer_product_contraction() {
        let bond_dims = FxHashMap::from_iter([(0, 3), (1, 2)]);
        let mut t1 = LeafTensor::new_from_map(vec![0], &bond_dims);
        let mut t2 = LeafTensor::new_from_map(vec![1], &bond_dims);
        t1.set_tensor_data(TensorData::new_from_data(
            &[3],
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 5.0),
                Complex64::new(3.0, -1.0),
            ],
            None,
        ));
        t2.set_tensor_data(TensorData::new_from_data(
            &[2],
            vec![Complex64::new(-4.0, 2.0), Complex64::new(0.0, -1.0)],
            None,
        ));
        let t3 = CompositeTensor::new(vec![t1, t2]);
        let contract_path = path![(0, 1)];

        let mut tn_ref = LeafTensor::new_from_map(vec![0, 1], &bond_dims);
        tn_ref.set_tensor_data(TensorData::new_from_data(
            &[3, 2],
            vec![
                Complex64::new(-4.0, 2.0),
                Complex64::new(0.0, -1.0),
                Complex64::new(-18.0, -16.0),
                Complex64::new(5.0, -2.0),
                Complex64::new(-10.0, 10.0),
                Complex64::new(-1.0, -3.0),
            ],
            None,
        ));

        let result = contract_tensor_network(t3, &contract_path);
        assert_approx_eq!(&LeafTensor, &result, &tn_ref);
    }
}
