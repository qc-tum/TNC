use crate::tensornetwork::tensor::Tensor;

pub mod contraction_cost;
pub mod paths;

// type alias to store contraction candidate when searching for optimal contraction path.
struct Candidate {
    flop_cost: u64,
    size_cost: u64,
    parent_ids: (usize, usize),
    child_id: usize,
    child_tensor: Tensor,
}
