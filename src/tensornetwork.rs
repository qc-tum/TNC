use rustc_hash::FxHashMap;

use self::tensor::Tensor;

pub mod contraction;
pub mod partitioning;
pub mod tensor;
pub mod tensordata;

/// Creates a tensor network from a list of tensors and the bond dimensions.
#[must_use]
pub fn create_tensor_network(tensors: Vec<Tensor>, bond_dims: &FxHashMap<usize, u64>) -> Tensor {
    let mut tensor = Tensor::default();
    tensor.push_tensors(tensors, Some(bond_dims));
    tensor
}
