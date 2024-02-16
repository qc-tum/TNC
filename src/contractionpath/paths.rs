use rand::rngs::StdRng;
use std::option::Option;

use crate::contractionpath::candidates::Candidate;
use crate::tensornetwork::tensor::Tensor;
use crate::types::ContractionIndex;

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

#[derive(Debug, Clone)]
pub enum CostType {
    Flops,
    Size,
}

pub(crate) fn validate_path(path: &Vec<ContractionIndex>) {
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
type ChoiceFnType = dyn for<'b, 'c> Fn(
    &'b mut std::collections::BinaryHeap<Candidate>,
    &'c std::collections::HashMap<u64, usize>,
    usize,
    f64,
    bool,
    &mut StdRng,
) -> Option<Candidate>;

#[cfg(test)]
mod tests {

    use crate::path;

    use super::validate_path;

    #[test]
    #[should_panic]
    fn test_validate_paths() {
        let invalid_path = path![(0, 1), (1, 2)];
        validate_path(&invalid_path);
    }
}
