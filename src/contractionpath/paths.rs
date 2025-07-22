use crate::types::ContractionIndex;
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
