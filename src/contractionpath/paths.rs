use rand::Rng;
use rustc_hash::FxHashMap;
use std::collections::BinaryHeap;

use crate::contractionpath::candidates::Candidate;
use crate::tensornetwork::tensor::Tensor;
use crate::types::ContractionIndex;
pub mod branchbound;
pub mod cotengrust;
pub mod greedy;
pub mod tree_annealing;
pub mod tree_reconfiguration;
pub mod tree_tempering;
pub mod weighted_branchbound;
//pub mod parallel_greedy;

/// An optimizer for finding a contraction path.
pub trait OptimizePath {
    /// Finds a contraction path.
    fn optimize_path(&mut self);

    /// Returns the best found contraction path in SSA format.
    fn get_best_path(&self) -> &Vec<ContractionIndex>;

    /// Returns the best found contraction path in ReplaceLeft format.
    fn get_best_replace_path(&self) -> Vec<ContractionIndex>;

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

pub(crate) fn validate_path(path: &[ContractionIndex]) {
    let mut contracted = Vec::<usize>::new();
    for index in path {
        match index {
            ContractionIndex::Pair(u, v) => {
                assert!(
                    !contracted.contains(u),
                    "Contracting already contracted tensors: {u:?}, path: {path:?}"
                );
                contracted.push(*v);
            }
            ContractionIndex::Path(_, path) => {
                validate_path(path);
            }
        }
    }
}

type CostFnType = dyn Fn(f64, f64, f64, &Tensor, &Tensor, &Tensor) -> f64;

/// Define a trait for functions that take an RNG as an argument.
pub(crate) trait RNGChooser {
    fn choose<R>(
        &self,
        queue: &mut BinaryHeap<Candidate>,
        remaining_tensors: &FxHashMap<u64, usize>,
        nbranch: usize,
        temperature: f64,
        rel_temperature: bool,
        rng: &mut R,
    ) -> Option<Candidate>
    where
        R: ?Sized + Rng;
}

#[cfg(test)]
mod tests {
    use crate::path;

    use super::validate_path;

    #[test]
    #[should_panic(
        expected = "Contracting already contracted tensors: 1, path: [Pair(0, 1), Pair(1, 2)]"
    )]
    fn test_validate_paths() {
        let invalid_path = path![(0, 1), (1, 2)];
        validate_path(invalid_path);
    }
}
