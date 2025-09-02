use std::collections::BinaryHeap;

use itertools::Itertools;
use rustc_hash::FxHashMap;

use crate::{
    contractionpath::{
        candidates::Candidate,
        contraction_cost::{contract_cost_tensors, contract_size_tensors},
        paths::{CostType, OptimizePath},
        ssa_ordering, ssa_replace_ordering,
    },
    tensornetwork::tensor::Tensor,
    types::ContractionIndex,
    utils::traits::HashMapInsertNew,
};

/// A struct with an [`OptimizePath`] implementation that explores possible pair contractions in a depth-first manner.
pub struct BranchBound<'a> {
    tn: &'a Tensor,
    nbranch: Option<usize>,
    cutoff_flops_factor: f64,
    minimize: CostType,
    best_flops: f64,
    best_size: f64,
    best_path: Vec<ContractionIndex>,
    best_progress: FxHashMap<usize, f64>,
    result_cache: FxHashMap<(usize, usize), usize>,
    flop_cache: FxHashMap<usize, f64>,
    size_cache: FxHashMap<usize, f64>,
    tensor_cache: FxHashMap<usize, Tensor>,
}

impl<'a> BranchBound<'a> {
    pub fn new(
        tn: &'a Tensor,
        nbranch: Option<usize>,
        cutoff_flops_factor: f64,
        minimize: CostType,
    ) -> Self {
        Self {
            tn,
            nbranch,
            cutoff_flops_factor,
            minimize,
            best_flops: f64::INFINITY,
            best_size: f64::INFINITY,
            best_path: Vec::new(),
            best_progress: FxHashMap::default(),
            result_cache: FxHashMap::default(),
            flop_cache: FxHashMap::default(),
            size_cache: FxHashMap::default(),
            tensor_cache: FxHashMap::default(),
        }
    }

    fn assess_candidate(
        &mut self,
        mut i: usize,
        mut j: usize,
        flops: f64,
        size: f64,
        remaining_len: usize,
    ) -> Option<Candidate> {
        let flops_12: f64;
        let size_12: f64;
        let k12: usize;
        let k12_tensor: Tensor;
        let mut current_flops = flops;
        let mut current_size = size;
        // Ensure that larger tensor is always to the left.
        if self.tensor_cache[&j].size() > self.tensor_cache[&i].size() {
            (i, j) = (j, i);
        }

        if self.result_cache.contains_key(&(i, j)) {
            k12 = self.result_cache[&(i, j)];
            flops_12 = self.flop_cache[&k12];
            size_12 = self.size_cache[&k12];
        } else {
            k12 = self.tensor_cache.len();
            flops_12 = contract_cost_tensors(&self.tensor_cache[&i], &self.tensor_cache[&j]);
            size_12 = contract_size_tensors(&self.tensor_cache[&i], &self.tensor_cache[&j]);
            k12_tensor = &self.tensor_cache[&i] ^ &self.tensor_cache[&j];

            self.result_cache.entry((i, j)).or_insert_with(|| k12);
            self.flop_cache.entry(k12).or_insert_with(|| flops_12);
            self.size_cache.entry(k12).or_insert_with(|| size_12);
            self.tensor_cache.insert_new(k12, k12_tensor);
        }
        current_flops += flops_12;
        current_size = current_size.max(size_12);

        if current_flops > self.best_flops && current_size > self.best_size {
            return None;
        }
        let best_flops = *self
            .best_progress
            .entry(remaining_len)
            .or_insert(current_flops);

        if current_flops < best_flops {
            self.best_progress.insert(remaining_len, current_flops);
        } else if current_flops > self.cutoff_flops_factor * best_flops {
            return None;
        }

        Some(Candidate {
            flop_cost: current_flops,
            size_cost: current_size,
            parent_ids: (i, j),
            child_id: k12,
        })
    }

    /// Explores possible pair contractions in a depth-first
    /// recursive manner like the `optimal` approach, but with extra heuristic early pruning of branches
    /// as well sieving by `memory_limit` and the best path found so far. A rust implementation of
    /// the Python based `opt_einsum` implementation. Found at <https://github.com/dgasmith/opt_einsum>.
    fn branch_iterate(
        &mut self,
        path: &[(usize, usize, usize)],
        remaining: &[usize],
        flops: f64,
        size: f64,
    ) {
        if remaining.len() == 1 {
            match self.minimize {
                CostType::Flops => {
                    if self.best_flops > flops {
                        self.best_flops = flops;
                        self.best_size = size;
                        self.best_path = ssa_ordering(path, self.tn.tensors().len());
                    }
                }
                CostType::Size => {
                    if self.best_size > size {
                        self.best_flops = flops;
                        self.best_size = size;
                        self.best_path = ssa_ordering(path, self.tn.tensors().len());
                    }
                }
            }
            return;
        }

        let mut candidates = BinaryHeap::<Candidate>::new();
        for i in remaining.iter().copied().combinations(2) {
            let candidate = self.assess_candidate(i[0], i[1], flops, size, remaining.len());
            if let Some(new_candidate) = candidate {
                candidates.push(new_candidate);
            }
        }
        let mut new_remaining;
        let mut new_path: Vec<(usize, usize, usize)>;
        let mut bi = 0;
        while self.nbranch.is_none() || bi < self.nbranch.unwrap() {
            bi += 1;
            let Some(Candidate {
                flop_cost,
                size_cost,
                parent_ids,
                child_id,
            }) = candidates.pop()
            else {
                break;
            };
            new_remaining = remaining.to_vec();
            new_remaining.retain(|e| *e != parent_ids.0 && *e != parent_ids.1);
            new_remaining.push(child_id);
            new_path = path.to_vec();
            new_path.push((parent_ids.0, parent_ids.1, child_id));
            self.branch_iterate(&new_path, &new_remaining, flop_cost, size_cost);
        }
    }
}

impl OptimizePath for BranchBound<'_> {
    fn optimize_path(&mut self) {
        if self.tn.is_leaf() {
            return;
        }
        let tensors = self.tn.tensors().clone();
        self.flop_cache.clear();
        self.size_cache.clear();
        self.result_cache.clear();
        self.tensor_cache.clear();
        let mut sub_tensor_contraction = Vec::new();
        // Get the initial space requirements for uncontracted tensors
        for (index, mut tensor) in tensors.into_iter().enumerate() {
            // Check that tensor has sub-tensors and doesn't have external legs set
            if !tensor.tensors().is_empty() && tensor.legs().is_empty() {
                let mut bb = BranchBound::new(
                    &tensor,
                    self.nbranch,
                    self.cutoff_flops_factor,
                    self.minimize,
                );
                bb.optimize_path();
                sub_tensor_contraction
                    .push(ContractionIndex::Path(index, bb.get_best_path().clone()));
                tensor = tensor.external_tensor();
            }
            self.size_cache
                .entry(index)
                .or_insert_with(|| tensor.size());

            self.tensor_cache.insert_new(index, tensor);
        }
        let remaining = (0..self.tn.tensors().len()).collect_vec();
        self.branch_iterate(&[], &remaining, 0f64, 0f64);
        sub_tensor_contraction.extend_from_slice(&self.best_path);
        self.best_path = sub_tensor_contraction;
    }

    fn get_best_flops(&self) -> f64 {
        self.best_flops
    }

    fn get_best_size(&self) -> f64 {
        self.best_size
    }

    fn get_best_path(&self) -> &Vec<ContractionIndex> {
        &self.best_path
    }

    fn get_best_replace_path(&self) -> Vec<ContractionIndex> {
        ssa_replace_ordering(&self.best_path, self.tn.tensors().len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use rustc_hash::FxHashMap;

    use crate::contractionpath::paths::{CostType, OptimizePath};
    use crate::path;
    use crate::tensornetwork::tensor::Tensor;

    fn setup_simple() -> Tensor {
        let bond_dims =
            FxHashMap::from_iter([(0, 5), (1, 2), (2, 6), (3, 8), (4, 1), (5, 3), (6, 4)]);
        Tensor::new_composite(vec![
            Tensor::new_from_map(vec![4, 3, 2], &bond_dims),
            Tensor::new_from_map(vec![0, 1, 3, 2], &bond_dims),
            Tensor::new_from_map(vec![4, 5, 6], &bond_dims),
        ])
    }

    fn setup_complex() -> Tensor {
        let bond_dims = FxHashMap::from_iter([
            (0, 27),
            (1, 18),
            (2, 12),
            (3, 15),
            (4, 5),
            (5, 3),
            (6, 18),
            (7, 22),
            (8, 45),
            (9, 65),
            (10, 5),
            (11, 17),
        ]);
        Tensor::new_composite(vec![
            Tensor::new_from_map(vec![4, 3, 2], &bond_dims),
            Tensor::new_from_map(vec![0, 1, 3, 2], &bond_dims),
            Tensor::new_from_map(vec![4, 5, 6], &bond_dims),
            Tensor::new_from_map(vec![6, 8, 9], &bond_dims),
            Tensor::new_from_map(vec![10, 8, 9], &bond_dims),
            Tensor::new_from_map(vec![5, 1, 0], &bond_dims),
        ])
    }

    #[test]
    fn test_contract_order_simple() {
        let tn = setup_simple();
        let mut opt = BranchBound::new(&tn, None, 20., CostType::Flops);
        opt.optimize_path();

        assert_eq!(opt.best_flops, 4540.);
        assert_eq!(opt.best_size, 538.);
        assert_eq!(opt.get_best_path(), &path![(1, 0), (2, 3)]);
        assert_eq!(opt.get_best_replace_path(), path![(1, 0), (2, 1)]);
    }

    #[test]
    fn test_contract_order_complex() {
        let tn = setup_complex();
        let mut opt = BranchBound::new(&tn, None, 20., CostType::Flops);
        opt.optimize_path();

        assert_eq!(opt.best_flops, 2654474.);
        assert_eq!(opt.best_size, 89478.);
        assert_eq!(opt.best_path, path![(1, 5), (0, 6), (2, 7), (3, 8), (4, 9)]);
        assert_eq!(
            opt.get_best_replace_path(),
            path![(1, 5), (0, 1), (2, 0), (3, 2), (4, 3)]
        );
    }
}
