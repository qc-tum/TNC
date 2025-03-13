use std::cmp::Ordering;

use float_cmp::approx_eq;

/// Struct to store contraction candidate information when searching for optimal contraction path.
#[derive(Clone, Debug)]
pub(crate) struct Candidate {
    pub flop_cost: f64,
    pub size_cost: f64,
    pub parent_ids: (usize, usize),
    pub child_id: usize,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        approx_eq!(f64, self.flop_cost, other.flop_cost)
            && approx_eq!(f64, self.size_cost, other.size_cost)
            && self.parent_ids == other.parent_ids
            && self.child_id == other.child_id
    }
}

impl Eq for Candidate {}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .size_cost
            .total_cmp(&self.size_cost)
            .then_with(|| self.flop_cost.total_cmp(&other.flop_cost))
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
