use crate::tensornetwork::tensor::Tensor;
use itertools::Itertools;
use num_complex::Complex64;
use std::collections::HashMap;
use tetra::Tensor as _TetraTensor;

pub fn from_array(t: &Tensor, bond_dims: &HashMap<i32, u64>, data: &[Complex64]) -> _TetraTensor {
    let tensor_dims: Vec<i32> = t.get_legs().iter().map(|e| bond_dims[e] as i32).collect();
    let mut tct = _TetraTensor::new(tensor_dims.as_slice());
    let t_ranges = tensor_dims
        .iter()
        .map(|e| (0..*e))
        .multi_cartesian_product();
    let mut data_iter = data.iter();
    for dim in t_ranges {
        tct.insert(&dim, *data_iter.next().unwrap());
    }
    tct
}
