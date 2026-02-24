//! Functionality to contract tensor networks.
use log::debug;
use tetra::contract;

use crate::{
    contractionpath::ContractionPath,
    tensornetwork::{tensor::Tensor, tensordata::TensorData},
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
pub fn contract_tensor_network(mut tn: Tensor, contract_path: &ContractionPath) -> Tensor {
    debug!(len = tn.tensors().len(); "Start contracting tensor network");

    // Contract child composite tensors first
    for (index, inner_path) in &contract_path.nested {
        let composite = std::mem::take(&mut tn.tensors[*index]);
        let contracted = contract_tensor_network(composite, inner_path);
        tn.tensors[*index] = contracted;
    }

    // Contract all leaf tensors
    for (i, j) in &contract_path.toplevel {
        debug!(i, j; "Contracting tensors");
        tn.contract_tensors(*i, *j);
        debug!(i, j; "Finished contracting tensors");
    }
    debug!("Completed tensor network contraction");

    tn.tensors
        .retain(|x| !matches!(x.tensor_data(), TensorData::Uncontracted) || x.is_composite());
    assert!(tn.tensors().len() <= 1, "Not fully contracted");
    tn.tensors.pop().unwrap_or(tn)
}

trait TensorContraction {
    /// Contracts two tensors.
    fn contract_tensors(&mut self, tensor_a_loc: usize, tensor_b_loc: usize);
}

impl TensorContraction for Tensor {
    fn contract_tensors(&mut self, tensor_a_loc: usize, tensor_b_loc: usize) {
        let tensor_a = std::mem::take(&mut self.tensors[tensor_a_loc]);
        let tensor_b = std::mem::take(&mut self.tensors[tensor_b_loc]);

        let mut tensor_symmetric_difference = &tensor_b ^ &tensor_a;

        let Tensor {
            legs: a_legs,
            tensordata: a_data,
            ..
        } = tensor_a;

        let Tensor {
            legs: b_legs,
            tensordata: b_data,
            ..
        } = tensor_b;

        let result = contract(
            &tensor_symmetric_difference.legs,
            &a_legs,
            a_data.into_data(),
            &b_legs,
            b_data.into_data(),
        );

        tensor_symmetric_difference.set_tensor_data(TensorData::Matrix(result));
        self.tensors[tensor_a_loc] = tensor_symmetric_difference;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use float_cmp::assert_approx_eq;
    use num_complex::Complex64;
    use rustc_hash::FxHashMap;
    use serde::Deserialize;

    use crate::{
        path,
        tensornetwork::{contraction::TensorContraction, tensor::Tensor, tensordata::TensorData},
    };

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

    #[test]
    fn test_tensor_contraction() {
        let mut data = load_test_data();
        let ta = data.remove("A").unwrap();
        let tb = data.remove("B").unwrap();
        let tc = data.remove("C").unwrap();
        let tab = data.remove("AxB").unwrap();
        let tbc = data.remove("BxC").unwrap();

        // t1 is of shape [3, 2, 7]
        let mut t1 = Tensor::new(ta.legs, ta.shape);
        // t2 is of shape [7, 8, 6]
        let mut t2 = Tensor::new(tb.legs, tb.shape);
        // t3 is of shape [3, 5, 8]
        let mut t3 = Tensor::new(tc.legs, tc.shape);
        // t12 is of shape [8, 6, 3, 2]
        let mut t12 = Tensor::new(tab.legs, tab.shape);
        // t23 is of shape [3, 5, 7, 6]
        let mut t23 = Tensor::new(tbc.legs, tbc.shape);

        t1.set_tensor_data(TensorData::new_from_data(
            &t1.shape().unwrap(),
            ta.data,
            None,
        ));

        t2.set_tensor_data(TensorData::new_from_data(
            &t2.shape().unwrap(),
            tb.data,
            None,
        ));
        t3.set_tensor_data(TensorData::new_from_data(
            &t3.shape().unwrap(),
            tc.data,
            None,
        ));

        t12.set_tensor_data(TensorData::new_from_data(
            &t12.shape().unwrap(),
            tab.data,
            None,
        ));

        t23.set_tensor_data(TensorData::new_from_data(
            &t23.shape().unwrap(),
            tbc.data,
            None,
        ));

        let mut tn_12 = Tensor::new_composite(vec![t1.clone(), t2.clone(), t3.clone()]);

        tn_12.contract_tensors(0, 1);
        assert_approx_eq!(&Tensor, tn_12.tensor(0), &t12, epsilon = 1e-14);

        let mut tn_23 = Tensor::new_composite(vec![t1, t2, t3]);

        tn_23.contract_tensors(1, 2);
        assert_approx_eq!(&Tensor, tn_23.tensor(1), &t23, epsilon = 1e-14);
    }

    #[test]
    fn test_tn_contraction() {
        let mut data = load_test_data();
        let ta = data.remove("A").unwrap();
        let tb = data.remove("B").unwrap();
        let tc = data.remove("C").unwrap();
        let tabc = data.remove("ABxC").unwrap();

        // t1 is of shape [3, 2, 7]
        let mut t1 = Tensor::new(ta.legs, ta.shape);
        // t2 is of shape [7, 8, 6]
        let mut t2 = Tensor::new(tb.legs, tb.shape);
        // t3 is of shape [3, 5, 8]
        let mut t3 = Tensor::new(tc.legs, tc.shape);
        // tout is of shape [5, 6, 2]
        let mut tout = Tensor::new(tabc.legs, tabc.shape);

        t1.set_tensor_data(TensorData::new_from_data(
            &t1.shape().unwrap(),
            ta.data,
            None,
        ));

        t2.set_tensor_data(TensorData::new_from_data(
            &t2.shape().unwrap(),
            tb.data,
            None,
        ));
        t3.set_tensor_data(TensorData::new_from_data(
            &t3.shape().unwrap(),
            tc.data,
            None,
        ));
        tout.set_tensor_data(TensorData::new_from_data(
            &tout.shape().unwrap(),
            tabc.data,
            None,
        ));

        let tn = Tensor::new_composite(vec![t1, t2, t3]);
        let contract_path = path![(0, 1), (0, 2)];

        let result = contract_tensor_network(tn, &contract_path);
        assert_approx_eq!(&Tensor, &result, &tout, epsilon = 1e-14);
    }

    #[test]
    fn test_outer_product_contraction() {
        let bond_dims = FxHashMap::from_iter([(0, 3), (1, 2)]);
        let mut t1 = Tensor::new_from_map(vec![0], &bond_dims);
        let mut t2 = Tensor::new_from_map(vec![1], &bond_dims);
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
        let t3 = Tensor::new_composite(vec![t1, t2]);
        let contract_path = path![(0, 1)];

        let mut tn_ref = Tensor::new_from_map(vec![1, 0], &bond_dims);
        tn_ref.set_tensor_data(TensorData::new_from_data(
            &[2, 3],
            vec![
                Complex64::new(-4.0, 2.0),
                Complex64::new(-18.0, -16.0),
                Complex64::new(-10.0, 10.0),
                Complex64::new(0.0, -1.0),
                Complex64::new(5.0, -2.0),
                Complex64::new(-1.0, -3.0),
            ],
            None,
        ));

        let result = contract_tensor_network(t3, &contract_path);
        assert_approx_eq!(&Tensor, &result, &tn_ref);
    }
}
