use crate::tensornetwork::tensor::Tensor;

pub mod contraction_cost;
pub mod paths;

// type alias to store contraction candidate when searching for optimal contraction path.
type Candidate = (u64, u64, (usize, usize), usize, Tensor);
