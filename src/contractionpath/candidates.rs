use std::cmp::Ordering;

/// Struct to store contraction candidate information when searching for optimal contraction path.
#[derive(Clone, Debug)]
pub(crate) struct Candidate {
    pub(crate) flop_cost: f64,
    pub(crate) size_cost: f64,
    pub(crate) parent_ids: (usize, usize),
    pub(crate) child_id: usize,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        (self.flop_cost - other.flop_cost).abs() < f64::EPSILON
            && (self.size_cost - other.size_cost).abs() < f64::EPSILON
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
