use std::cmp::Ordering;

use crate::tensornetwork::tensor::Tensor;

// type alias to store contraction candidate when searching for optimal contraction path.
#[derive(Eq, PartialEq, Hash, Clone, Debug)]
pub(crate) struct Candidate {
    pub(crate) flop_cost: i64,
    pub(crate) size_cost: i64,
    pub(crate) parent_ids: (usize, usize),
    pub(crate) parent_tensors: Option<(Tensor, Tensor)>,
    pub(crate) child_id: usize,
    pub(crate) child_tensor: Option<Tensor>,
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .size_cost
            .cmp(&self.size_cost)
            .then_with(|| other.flop_cost.cmp(&self.flop_cost))
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
