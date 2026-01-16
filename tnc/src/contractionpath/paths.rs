//! Contraction path finders.

use crate::contractionpath::ContractionPath;

pub mod branchbound;
pub mod cotengrust;
pub mod hyperoptimization;
#[cfg(feature = "cotengra")]
pub mod tree_annealing;
#[cfg(feature = "cotengra")]
pub mod tree_reconfiguration;
#[cfg(feature = "cotengra")]
pub mod tree_tempering;
pub mod weighted_branchbound;

/// An optimizer for finding a contraction path.
pub trait FindPath {
    /// Finds a contraction path.
    fn find_path(&mut self);

    /// Returns the best found contraction path in SSA format.
    fn get_best_path(&self) -> &ContractionPath;

    /// Returns the best found contraction path in ReplaceLeft format.
    fn get_best_replace_path(&self) -> ContractionPath;

    /// Returns the total op count of the best path found.
    fn get_best_flops(&self) -> f64;

    /// Returns the max memory (in number of elements) of the best path found.
    fn get_best_size(&self) -> f64;
}

/// The cost metric to optimize for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CostType {
    /// Number of flops or operations.
    Flops,
    /// Size of the biggest contraction.
    Size,
}

pub(crate) fn validate_path(path: &ContractionPath) {
    let mut contracted = Vec::<usize>::new();
    for nested in path.nested.values() {
        validate_path(nested);
    }

    for (u, v) in &path.toplevel {
        assert!(
            !contracted.contains(u),
            "Contracting already contracted tensors: {u:?}, path: {path:?}"
        );
        contracted.push(*v);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::path;

    #[test]
    #[should_panic(
        expected = "Contracting already contracted tensors: 1, path: ContractionPath { nested: {}, toplevel: [(0, 1), (1, 2)] }"
    )]
    fn test_validate_paths() {
        let invalid_path = path![(0, 1), (1, 2)];
        validate_path(&invalid_path);
    }
}
