use self::tensor::Tensor;

pub mod contraction;
pub mod partitioning;
pub mod tensor;
pub mod tensordata;

use std::collections::HashMap;

pub fn create_tensor_network(
    tensors: Vec<Tensor>,
    bond_dims: &HashMap<usize, u64>,
    external_legs: Option<&Vec<usize>>,
) -> Tensor {
    let mut tensor = Tensor::default();
    tensor.push_tensors(tensors, Some(bond_dims), external_legs);
    tensor
}
