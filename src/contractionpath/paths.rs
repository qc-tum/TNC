use rand::Rng;
use std::collections::{BinaryHeap, HashMap};

use crate::contractionpath::candidates::Candidate;
use crate::tensornetwork::tensor::Tensor;
use crate::types::ContractionIndex;
use rand::Rng;
use std::collections::{BinaryHeap, HashMap};

pub mod branchbound;
pub mod greedy;
//pub mod parallel_greedy;

pub trait OptimizePath {
    fn optimize_path(&mut self);

    fn get_best_path(&self) -> &Vec<ContractionIndex>;
    fn get_best_replace_path(&self) -> Vec<ContractionIndex>;
    fn get_best_flops(&self) -> u64;
    fn get_best_size(&self) -> u64;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CostType {
    Flops,
    Size,
}

pub(crate) fn validate_path(path: &[ContractionIndex]) {
    let mut contracted = Vec::<usize>::new();
    for index in path {
        match index {
            ContractionIndex::Pair(u, v) => {
                if contracted.contains(u) {
                    panic!(
                        "Contracting already contracted tensors: {:?}, path: {:?}",
                        u, path,
                    )
                } else {
                    contracted.push(*v);
                }
            }
            ContractionIndex::Path(_, path) => {
                validate_path(path);
            }
        }
    }
}

type CostFnType = dyn Fn(i64, i64, i64, &Tensor, &Tensor, &Tensor) -> i64;

// Define a trait for functions that take an RNG as an argument
pub(crate) trait RNGChooser {
    fn choose<R>(
        &self,
        queue: &mut BinaryHeap<Candidate>,
        remaining_tensors: &HashMap<u64, usize>,
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
    #[should_panic]
    fn test_validate_paths() {
        let invalid_path = path![(0, 1), (1, 2)];
        validate_path(invalid_path);
    }
}
