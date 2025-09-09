use std::collections::BinaryHeap;

use itertools::Itertools;
use rustc_hash::FxHashMap;

use crate::{
    contractionpath::{
        candidates::Candidate,
        contraction_cost::{contract_op_cost_tensors, contract_size_tensors},
        paths::{CostType, OptimizePath},
        ssa_ordering, ssa_replace_ordering,
    },
    tensornetwork::tensor::Tensor,
    types::ContractionIndex,
    utils::traits::HashMapInsertNew,
};

/// A struct with an [`OptimizePath`] implementation that explores possible pair contractions in a depth-first manner.
pub struct WeightedBranchBound<'a> {
    tn: &'a Tensor,
    nbranch: Option<usize>,
    cutoff_flops_factor: f64,
    minimize: CostType,
    best_flops: f64,
    best_size: f64,
    best_path: Vec<ContractionIndex>,
    best_progress: FxHashMap<usize, f64>,
    largest_latency: f64,
    result_cache: FxHashMap<(usize, usize), (usize, f64, f64)>,
    comm_cache: FxHashMap<usize, f64>,
    tensor_cache: FxHashMap<usize, Tensor>,
}

impl<'a> WeightedBranchBound<'a> {
    pub fn new(
        tn: &'a Tensor,
        nbranch: Option<usize>,
        cutoff_flops_factor: f64,
        latency_map: FxHashMap<usize, f64>,
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
            largest_latency: Default::default(),
            result_cache: FxHashMap::default(),
            comm_cache: latency_map,
            tensor_cache: FxHashMap::default(),
        }
    }

    fn assess_candidate(
        &mut self,
        mut i: usize,
        mut j: usize,
        size: f64,
        remaining_len: usize,
    ) -> Option<Candidate> {
        if self.tensor_cache[&j].size() > self.tensor_cache[&i].size() {
            (i, j) = (j, i);
        }

        let &mut (k12, flops_12, size_12) = self.result_cache.entry((i, j)).or_insert_with(|| {
            let k12 = self.tensor_cache.len();
            let flops_12 = contract_op_cost_tensors(&self.tensor_cache[&i], &self.tensor_cache[&j]);
            let size_12 = contract_size_tensors(&self.tensor_cache[&i], &self.tensor_cache[&j]);
            let k12_tensor = &self.tensor_cache[&i] ^ &self.tensor_cache[&j];
            self.tensor_cache.insert_new(k12, k12_tensor);
            (k12, flops_12, size_12)
        });

        let current_flops = if let Some(total_flops) = self.comm_cache.get(&k12) {
            *total_flops
        } else {
            let total_flops = flops_12 + self.comm_cache[&i].max(self.comm_cache[&j]);
            self.comm_cache.insert(k12, total_flops);
            total_flops
        };
        let current_size = size.max(size_12);

        if current_flops > self.best_flops && current_size > self.best_size {
            return None;
        }
        let best_flops = *self
            .best_progress
            .entry(remaining_len)
            .or_insert(current_flops);

        if current_flops < best_flops {
            self.best_progress.insert(remaining_len, current_flops);
        } else if current_flops > (self.cutoff_flops_factor * best_flops + self.largest_latency) {
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

        let mut candidates = BinaryHeap::with_capacity(remaining.len() * (remaining.len() - 1) / 2);
        for pair in remaining.iter().copied().combinations(2) {
            let candidate = self.assess_candidate(pair[0], pair[1], size, remaining.len());
            if let Some(new_candidate) = candidate {
                candidates.push(new_candidate);
            }
        }
        let mut candidates = candidates.into_sorted_vec();
        if let Some(limit) = self.nbranch {
            candidates.truncate(limit);
        }

        let mut new_path = Vec::with_capacity(path.len() + 1);
        new_path.extend_from_slice(path);

        for candidate in candidates.into_iter().rev() {
            let Candidate {
                flop_cost,
                size_cost,
                parent_ids,
                child_id,
            } = candidate;
            let mut new_remaining = remaining.to_vec();
            new_remaining.retain(|e| *e != parent_ids.0 && *e != parent_ids.1);
            new_remaining.push(child_id);
            new_path.push((parent_ids.0, parent_ids.1, child_id));
            self.branch_iterate(&new_path, &new_remaining, flop_cost, size_cost);
            new_path.pop();
        }
    }
}

impl OptimizePath for WeightedBranchBound<'_> {
    fn optimize_path(&mut self) {
        if self.tn.is_leaf() {
            return;
        }
        let tensors = self.tn.tensors().clone();
        self.result_cache.clear();
        self.tensor_cache.clear();
        self.largest_latency = *self
            .comm_cache
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).expect("Tried to compare NaN"))
            .unwrap()
            .1;
        let mut sub_tensor_contraction = Vec::new();
        // Get the initial space requirements for uncontracted tensors
        for (index, mut tensor) in tensors.into_iter().enumerate() {
            // Check that tensor has sub-tensors and doesn't have external legs set
            if tensor.is_composite() && tensor.legs().is_empty() {
                let mut bb = WeightedBranchBound::new(
                    &tensor,
                    self.nbranch,
                    self.cutoff_flops_factor,
                    self.comm_cache.clone(),
                    self.minimize,
                );
                bb.optimize_path();
                sub_tensor_contraction
                    .push(ContractionIndex::Path(index, bb.get_best_path().clone()));
                tensor = tensor.external_tensor();
            }
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

    use crate::contractionpath::paths::CostType;
    use crate::contractionpath::paths::OptimizePath;
    use crate::path;
    use crate::tensornetwork::tensor::Tensor;

    fn setup_simple() -> (Tensor, FxHashMap<usize, f64>) {
        let bond_dims =
            FxHashMap::from_iter([(0, 5), (1, 2), (2, 6), (3, 8), (4, 1), (5, 3), (6, 4)]);
        (
            Tensor::new_composite(vec![
                Tensor::new_from_map(vec![4, 3, 2], &bond_dims),
                Tensor::new_from_map(vec![0, 1, 3, 2], &bond_dims),
                Tensor::new_from_map(vec![4, 5, 6], &bond_dims),
            ]),
            FxHashMap::from_iter([(0, 20.), (1, 40.), (2, 85.)]),
        )
    }

    fn setup_complex() -> (Tensor, FxHashMap<usize, f64>) {
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
        (
            Tensor::new_composite(vec![
                Tensor::new_from_map(vec![4, 3, 2], &bond_dims),
                Tensor::new_from_map(vec![0, 1, 3, 2], &bond_dims),
                Tensor::new_from_map(vec![4, 5, 6], &bond_dims),
                Tensor::new_from_map(vec![6, 8, 9], &bond_dims),
                Tensor::new_from_map(vec![10, 8, 9], &bond_dims),
                Tensor::new_from_map(vec![5, 1, 0], &bond_dims),
            ]),
            FxHashMap::from_iter([(0, 120.), (1, 0.), (2, 15.), (3, 15.), (4, 85.), (5, 15.)]),
        )
    }

    #[test]
    fn test_contract_order_simple() {
        let (tn, latency_costs) = setup_simple();
        let mut opt = WeightedBranchBound::new(&tn, None, 20., latency_costs, CostType::Flops);
        opt.optimize_path();

        assert_eq!(opt.best_flops, 640.);
        assert_eq!(opt.best_size, 538.);
        assert_eq!(opt.get_best_path(), &path![(1, 0), (2, 3)]);
        assert_eq!(opt.get_best_replace_path(), path![(1, 0), (2, 1)]);
    }

    #[test]
    fn test_contract_order_complex() {
        let (tn, latency_costs) = setup_complex();
        let mut opt = WeightedBranchBound::new(&tn, None, 20., latency_costs, CostType::Flops);
        opt.optimize_path();

        assert_eq!(opt.best_flops, 265230.);
        assert_eq!(opt.best_size, 89478.);
        assert_eq!(opt.best_path, path![(3, 4), (2, 6), (1, 5), (0, 8), (7, 9)]);
        assert_eq!(
            opt.get_best_replace_path(),
            path![(3, 4), (2, 3), (1, 5), (0, 1), (2, 0)]
        );
    }
}
