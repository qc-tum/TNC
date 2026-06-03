//! Contraction path finders.

use crate::{
    contractionpath::{ssa_replace_ordering, ContractionPath},
    tensornetwork::tensor::Tensor,
};

pub mod branchbound;
pub mod cotengrust;
#[cfg(feature = "cotengra")]
pub mod hyperoptimization;
#[cfg(feature = "cotengra")]
pub mod tree_annealing;
#[cfg(feature = "cotengra")]
pub mod tree_reconfiguration;
#[cfg(feature = "cotengra")]
pub mod tree_tempering;
pub mod weighted_branchbound;

/// An optimizer for finding a contraction path.
pub trait Pathfinder {
    type Result: ContractionPathResult;

    /// Finds a contraction path for the `tensor`.
    ///
    /// Uses `&mut self` to allow for internal state such as caching.
    fn find_path(&mut self, tensor: &Tensor) -> Self::Result;
}

/// The result of running a contraction [`Pathfinder`].
pub trait ContractionPathResult {
    /// Returns the best found contraction path in SSA format.
    fn ssa_path(&self) -> &ContractionPath;

    /// Returns the best found contraction path in ReplaceLeft format.
    fn replace_path(&self) -> ContractionPath;

    /// Returns the total op count of the best path found.
    fn flops(&self) -> f64;

    /// Returns the max memory (in number of elements) of the best path found.
    fn size(&self) -> f64;
}

/// Basic result data from running a contraction [`Pathfinder`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BasicContractionPathResult {
    /// The found path in SSA format.
    ssa_path: ContractionPath,
    /// The computational cost of the found path.
    flops: f64,
    /// The peak memory of the found path.
    size: f64,
}

impl ContractionPathResult for BasicContractionPathResult {
    #[inline]
    fn ssa_path(&self) -> &ContractionPath {
        &self.ssa_path
    }

    #[inline]
    fn replace_path(&self) -> ContractionPath {
        ssa_replace_ordering(&self.ssa_path)
    }

    #[inline]
    fn flops(&self) -> f64 {
        self.flops
    }

    #[inline]
    fn size(&self) -> f64 {
        self.size
    }
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
