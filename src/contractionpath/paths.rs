use rand::rngs::StdRng;
use std::collections::HashMap;
use std::option::Option;

use crate::contractionpath::candidates::Candidate;
use crate::tensornetwork::tensor::Tensor;
use crate::types::ContractionIndex;

pub mod branchbound;
pub mod greedy;
//pub mod parallel_greedy;

pub trait OptimizePath {
    fn optimize_path(&mut self);

    fn get_best_path(&self) -> &Vec<(usize, usize)>;
    fn get_best_replace_path(&self) -> Vec<(usize, usize)>;
    fn get_best_flops(&self) -> u64;
    fn get_best_size(&self) -> u64;
}

#[derive(Debug, Clone)]
pub enum CostType {
    Flops = 0,
    Size = 1,
}

pub(crate) fn validate_path(path: &Vec<ContractionIndex>) -> Result<bool, String> {
    let mut contracted = Vec::<usize>::new();
    for index in path {
        match index {
            ContractionIndex::Pair(u, v) => {
                if contracted.contains(u) {
                    panic!(
                        "Contracting already contracted tensors: {:?}, path: {:?}",
                        u, path
                    );
                } else {
                    contracted.push(*v);
                }
            }
            ContractionIndex::Path(_, path) => {
                let _ = validate_path(path);
            }
        }
    }
    Ok(true)
}

type CostFnType = dyn Fn(&HashMap<usize, u64>, i64, i64, i64, &Tensor, &Tensor, &Tensor) -> i64;
type ChoiceFnType = dyn for<'b, 'c> Fn(
    &'b mut std::collections::BinaryHeap<Candidate>,
    &'c std::collections::HashMap<u64, usize>,
    usize,
    f64,
    bool,
    &mut StdRng,
) -> Option<Candidate>;
