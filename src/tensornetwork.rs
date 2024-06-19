use self::tensor::Tensor;

pub mod contraction;
pub mod partitioning;
pub mod tensor;
pub mod tensordata;

use std::collections::HashMap;

/// Creates a tensor network from a list of tensors and the bond dimensions.
#[must_use]
pub fn create_tensor_network(
    tensors: Vec<Tensor>,
    bond_dims: &HashMap<usize, u64>,
    external_legs: Option<&HashMap<usize, usize>>,
) -> Tensor {
    let mut tensor = Tensor::default();
    tensor.push_tensors(tensors, Some(bond_dims), external_legs);
    tensor
}
