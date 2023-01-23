use itertools::Itertools;
use std::cmp::max;
use std::collections::HashMap;
// use std::iter::zip;

use crate::contractionpath::contraction_cost::{_contract_cost, _contract_size, size};
use crate::tensornetwork::tensor::Tensor;
use crate::tensornetwork::TensorNetwork;

pub trait OptimizePath {
    fn _optimize_path(&mut self, output: Option<Vec<u32>>);

    fn get_best_path(self) -> Vec<(usize, usize)>;
    fn get_best_replace_path(self) -> Vec<(usize, usize)>;
    fn get_best_flops(self) -> u64;
    fn get_best_size(self) -> u64;
}

pub enum BranchBoundType {
    Flops = 0,
    Size = 1,
}

/// A struct with an OptimizePath implementation that explores possible pair contractions in a depth-first manner.
pub struct BranchBound {
    tn: TensorNetwork,
    nbranch: Option<u32>,
    cutoff_flops_factor: u64,
    minimize: BranchBoundType,
    best_flops: u64,
    best_size: u64,
    best_path: Vec<(usize, usize)>,
    // best_path: Vec<(usize, usize)>,
    best_progress: HashMap<usize, u64>,
    result_cache: HashMap<Vec<usize>, usize>,
    flop_cache: HashMap<usize, u64>,
    size_cache: HashMap<usize, u64>,
    tensor_cache: HashMap<usize, Tensor>,
}

/// The contraction ordering labels [Tensor] objects from each possible contraction with a
/// unique identifier in ssa format. As only a subset of these [Tensor] objects are seen in
/// a contraction path, the tensors in the optimal path search are not sequential. This converts
/// the output to strictly obey an ssa format.
///
/// # Arguments
///
/// * `path` - Output path as Vec<(usize, usize)> after an [_optimize_path] call.
/// # Returns
///
/// Identical path using ssa format
fn ssa_ordering(path: &Vec<(usize, usize, usize)>, mut n: usize) -> Vec<(usize, usize)> {
    let mut ssa_path = Vec::with_capacity(path.len());
    let mut hs = HashMap::new();
    let path_len = n;
    for (u1, u2, u3) in path {
        let t1;
        let t2;
        if *u1 > path_len {
            t1 = hs[u1];
        } else {
            t1 = *u1;
        }
        if *u2 > path_len {
            t2 = hs[u2];
        } else {
            t2 = *u2;
        }
        hs.entry(*u3).or_insert(n);
        n += 1;
        ssa_path.push((t1, t2));
    }
    ssa_path
}

/// Accepts a contraction path that is in ssa format and returns a contraction path
/// assuming that all contracted tensors replace the left input tensor.
///
/// # Arguments
///
/// * `path` - Output path as Vec<(usize, usize)> that is in ssa format.
/// # Returns
///
/// Identical path that replaces the left input tensor upon contraction
fn ssa_replace_ordering(path: &Vec<(usize, usize)>, mut n: usize) -> Vec<(usize, usize)> {
    let mut ssa_path = Vec::with_capacity(path.len());
    let mut hs = HashMap::new();
    for i in 0..path.len() {
        let mut tup = path[i];
        if hs.contains_key(&tup.0) {
            tup.0 = hs[&tup.0];
        }
        if hs.contains_key(&tup.1) {
            tup.1 = hs[&tup.1];
        }
        hs.entry(n).or_insert(tup.0);
        ssa_path.push(tup);
        n += 1;
    }
    ssa_path
}

impl BranchBound {
    pub fn new(
        tn: TensorNetwork,
        nbranch: Option<u32>,
        cutoff_flops_factor: u64,
        minimize: BranchBoundType,
    ) -> Self {
        Self {
            tn,
            nbranch,
            cutoff_flops_factor,
            minimize,
            best_flops: u64::MAX,
            best_size: u64::MAX,
            best_path: Vec::new(),
            best_progress: HashMap::<usize, u64>::new(),
            result_cache: HashMap::<Vec<usize>, usize>::new(),
            flop_cache: HashMap::<usize, u64>::new(),
            size_cache: HashMap::<usize, u64>::new(),
            tensor_cache: HashMap::<usize, Tensor>::new(),
        }
    }

    /// Explores possible pair contractions in a depth-first
    /// recursive manner like the `optimal` approach, but with extra heuristic early pruning of branches
    /// as well sieving by `memory_limit` and the best path found so far. A rust implementation of
    /// the Python based `opt_einsum` implementation. Found at github.com/dgasmith/opt_einsum.
    fn _branch_iterate(
        &mut self,
        path: Vec<(usize, usize, usize)>,
        remaining: Vec<u32>,
        mut flops: u64,
        mut size: u64,
    ) {
        if remaining.len() == 1 {
            self.best_size = size;
            self.best_flops = flops;
            self.best_path = ssa_ordering(&path, self.tn.get_tensors().len());
        }

        let mut assess_candidate =
            |i: usize, j: usize| -> Option<(u64, u64, (usize, usize), usize, Tensor)> {
                let flops_12: u64;
                let size_12: u64;
                let k12: usize;
                let k12_tensor: Tensor;
                if self.result_cache.contains_key(&vec![i, j]) {
                    k12 = self.result_cache[&vec![i, j]];
                    flops_12 = self.flop_cache[&k12];
                    size_12 = self.size_cache[&k12];
                    k12_tensor = self.tensor_cache[&k12].clone();
                } else {
                    k12 = self.tensor_cache.len();
                    flops_12 = _contract_cost(
                        self.tensor_cache[&i].clone(),
                        self.tensor_cache[&j].clone(),
                        self.tn.get_bond_dims(),
                    );
                    (k12_tensor, size_12) = _contract_size(
                        self.tensor_cache[&i].clone(),
                        self.tensor_cache[&j].clone(),
                        self.tn.get_bond_dims(),
                    );
                    self.result_cache.entry(vec![i, j]).or_insert(k12);
                    self.flop_cache.entry(k12).or_insert(flops_12);
                    self.size_cache.entry(k12).or_insert(size_12);
                    // self.tn.push_tensor(k12_tensor.clone(), None);
                    self.tensor_cache.entry(k12).or_insert(k12_tensor.clone());
                }
                flops += flops_12 as u64;
                size = max(size, size_12 as u64);

                if flops > self.best_flops && size > self.best_size {
                    return None;
                }
                let best_flops: u64;
                if self.best_progress.contains_key(&remaining.len()) {
                    best_flops = self.best_progress[&remaining.len()];
                } else {
                    best_flops = flops;
                    self.best_progress.entry(remaining.len()).or_insert(flops);
                }

                if flops < best_flops as u64 {
                    self.best_progress
                        .entry(remaining.len())
                        .insert_entry(flops);
                } else if flops > self.cutoff_flops_factor * self.best_progress[&remaining.len()] {
                    return None;
                }

                return Some((flops, size, (i, j), k12, k12_tensor));
            };

        let mut candidates = Vec::new();
        for i in remaining.iter().combinations(2) {
            let candidate = assess_candidate(*i[0] as usize, *i[1] as usize);
            if !candidate.is_none() {
                candidates.push(candidate);
            }
        }

        let bi = 0;
        let mut new_remaining;
        let mut new_path: Vec<(usize, usize, usize)>;
        while self.nbranch.is_none() || bi < self.nbranch.unwrap() {
            if candidates.is_empty() {
                break;
            }
            let (new_flops, new_size, (i, j), k12, _k12_tensor) =
                candidates.pop().unwrap().unwrap();
            new_remaining = remaining.clone();
            new_remaining.retain(|e| *e != i as u32);
            new_remaining.retain(|e| *e != j as u32);
            new_remaining.insert(new_remaining.len(), k12 as u32);
            new_path = path.clone();
            new_path.push((i, j, k12));
            BranchBound::_branch_iterate(self, new_path, new_remaining, new_flops, new_size);
        }
    }
}


impl OptimizePath for BranchBound {
    fn _optimize_path(&mut self, _output: Option<Vec<u32>>) {
        let tensors = self.tn.get_tensors();
        if self.tn.is_empty() {
            return;
        }

        self.flop_cache.clear();
        self.size_cache.clear();

        // Get the initial space requirements for uncontracted tensors
        for index in 0usize..tensors.len() {
            self.size_cache
                .entry(index)
                .or_insert(size(&self.tn, index));
            self.tensor_cache
                .entry(index)
                .or_insert(tensors[index].clone());
        }

        let remaining: Vec<u32> = (0u32..self.tn.get_tensors().len() as u32).collect();
        BranchBound::_branch_iterate(self, vec![], remaining, 0, 0);
    }

    fn get_best_flops(self) -> u64 {
        self.best_flops
    }

    fn get_best_size(self) -> u64 {
        self.best_size
    }

    fn get_best_path(self) -> Vec<(usize, usize)> {
        self.best_path
    }

    fn get_best_replace_path(self) -> Vec<(usize, usize)> {
        ssa_replace_ordering(&self.best_path, self.tn.get_tensors().len())
    }
}

#[cfg(test)]
mod tests {
    // use rand::distributions::{Distribution, Uniform};
    // TODO: Use random tensors
    use super::ssa_ordering;
    use super::ssa_replace_ordering;
    use crate::contractionpath::paths::BranchBound;
    use crate::contractionpath::paths::BranchBoundType;
    use crate::contractionpath::paths::OptimizePath;
    use crate::tensornetwork::tensor::Tensor;
    use crate::tensornetwork::TensorNetwork;

    fn setup_simple() -> TensorNetwork {
        TensorNetwork::from_vector(
            vec![
                Tensor::new(vec![4, 3, 2]),
                Tensor::new(vec![0, 1, 3, 2]),
                Tensor::new(vec![4, 5, 6]),
            ],
            vec![27, 18, 12, 15, 5, 3, 18],
        )
    }

    fn setup_complex() -> TensorNetwork {
        TensorNetwork::from_vector(
            vec![
                Tensor::new(vec![4, 3, 2]),
                Tensor::new(vec![0, 1, 3, 2]),
                Tensor::new(vec![4, 5, 6]),
                Tensor::new(vec![6, 8, 9]),
                Tensor::new(vec![10, 8, 9]),
                Tensor::new(vec![5, 1, 0]),
            ],
            vec![27, 18, 12, 15, 5, 3, 18, 22, 45, 65, 5, 17],
        )
    }

    #[test]
    fn test_ssa_ordering() {
        let path = vec![
            (0, 3, 15),
            (1, 2, 44),
            (6, 4, 8),
            (5, 15, 22),
            (8, 44, 12),
            (12, 22, 99),
        ];
        let new_path = ssa_ordering(&path, 7);

        assert_eq!(
            new_path,
            vec![(0, 3), (1, 2), (6, 4), (5, 7), (9, 8), (11, 10)]
        )
    }

    #[test]
    fn test_ssa_replace_ordering() {
        let path = vec![(0, 3), (1, 2), (6, 4), (5, 7), (9, 8), (11, 10)];
        let new_path = ssa_replace_ordering(&path, 7);

        assert_eq!(
            new_path,
            vec![(0, 3), (1, 2), (6, 4), (5, 0), (6, 1), (6, 5)]
        )
    }

    #[test]
    fn test_contract_order_simple() {
        let tn = setup_simple();
        let mut opt = BranchBound::new(tn, None, 20, BranchBoundType::Flops);
        opt._optimize_path(None);

        assert_eq!(opt.best_flops, 568620);
        assert_eq!(opt.best_size, 90810);
        assert_eq!(opt.best_path, vec![(0, 1), (2, 3)]);
        assert_eq!(opt.get_best_replace_path(), vec![(0, 1), (2, 0)]);
    }

    #[test]
    fn test_contract_order_complex() {
        let tn = setup_complex();
        let mut opt = BranchBound::new(tn, None, 20, BranchBoundType::Flops);
        opt._optimize_path(None);
        
        assert_eq!(opt.best_flops, 5614200);
        assert_eq!(opt.best_size, 3963645);
        assert_eq!(opt.best_path, vec![(0, 1), (2, 5), (3, 4), (6, 7), (8, 9)]);
        assert_eq!(opt.get_best_replace_path(), vec![(0, 1), (2, 5), (3, 4), (0, 2), (3, 0)]);
    }
}
