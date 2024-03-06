use std::cmp::Ordering;

/// Struct to store contraction candidate information when searching for optimal contraction path.
#[derive(Eq, PartialEq, Hash, Clone, Debug)]
pub(crate) struct Candidate {
    pub(crate) flop_cost: i64,
    pub(crate) size_cost: i64,
    pub(crate) parent_ids: (usize, usize),
    pub(crate) child_id: usize,
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .size_cost
            .cmp(&self.size_cost)
            .then_with(|| self.flop_cost.cmp(&other.flop_cost))
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
