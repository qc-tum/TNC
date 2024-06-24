use std::{
    cmp::max,
    collections::{BinaryHeap, HashMap},
};

use itertools::Itertools;

use crate::{
    contractionpath::{
        candidates::Candidate,
        contraction_cost::{contract_cost_tensors, contract_size_tensors},
        ssa_ordering, ssa_replace_ordering,
    },
    tensornetwork::tensor::Tensor,
    types::ContractionIndex,
};

use super::{CostType, OptimizePath};

/// A struct with an [`OptimizePath`] implementation that explores possible pair contractions in a depth-first manner.
pub struct BranchBound<'a> {
    tn: &'a Tensor,
    nbranch: Option<u32>,
    cutoff_flops_factor: u128,
    minimize: CostType,
    best_flops: u128,
    best_size: u128,
    best_path: Vec<ContractionIndex>,
    best_progress: HashMap<usize, u128>,
    result_cache: HashMap<(usize, usize), usize>,
    flop_cache: HashMap<usize, u128>,
    size_cache: HashMap<usize, u128>,
    tensor_cache: HashMap<usize, Tensor>,
}

impl<'a> BranchBound<'a> {
    pub fn new(
        tn: &'a Tensor,
        nbranch: Option<u32>,
        cutoff_flops_factor: u128,
        minimize: CostType,
    ) -> Self {
        Self {
            tn,
            nbranch,
            cutoff_flops_factor,
            minimize,
            best_flops: u128::MAX,
            best_size: u128::MAX,
            best_path: Vec::new(),
            best_progress: HashMap::new(),
            result_cache: HashMap::new(),
            flop_cache: HashMap::new(),
            size_cache: HashMap::new(),
            tensor_cache: HashMap::new(),
        }
    }

    fn assess_candidate(
        &mut self,
        i: usize,
        j: usize,
        flops: u128,
        size: u128,
        remaining: &[u32],
    ) -> Option<Candidate> {
        let flops_12: u128;
        let size_12: u128;
        let k12: usize;
        let k12_tensor: Tensor;
        let mut current_flops = flops;
        let mut current_size = size;
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
            self.tensor_cache
                .entry(k12)
                .or_insert_with(|| k12_tensor.clone());
        }
        current_flops += flops_12;
        current_size = max(current_size, size_12);

        if current_flops > self.best_flops && current_size > self.best_size {
            return None;
        }
        let best_flops = *self
            .best_progress
            .entry(remaining.len())
            .or_insert(current_flops);

        if current_flops < best_flops {
            self.best_progress.insert(remaining.len(), current_flops);
        } else if current_flops > self.cutoff_flops_factor * best_flops {
            return None;
        }

        Some(Candidate {
            flop_cost: current_flops as i64,
            size_cost: current_size as i64,
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
        path: Vec<(usize, usize, usize)>,
        remaining: Vec<u32>,
        flops: u128,
        size: u128,
    ) {
        if remaining.len() == 1 {
            match self.minimize {
                CostType::Flops => {
                    if self.best_flops > flops {
                        self.best_flops = flops;
                        self.best_size = size;
                        self.best_path = ssa_ordering(&path, self.tn.tensors().len());
                    }
                }
                CostType::Size => {
                    if self.best_size > size {
                        self.best_flops = flops;
                        self.best_size = size;
                        self.best_path = ssa_ordering(&path, self.tn.tensors().len());
                    }
                }
            }
            return;
        }

        let mut candidates = BinaryHeap::<Candidate>::new();
        for i in remaining.iter().copied().combinations(2) {
            let candidate =
                self.assess_candidate(i[0] as usize, i[1] as usize, flops, size, &remaining);
            if let Some(new_candidate) = candidate {
                candidates.push(new_candidate);
            }
        }
        let bi = 0;
        let mut new_remaining;
        let mut new_path: Vec<(usize, usize, usize)>;
        while self.nbranch.is_none() || bi < self.nbranch.unwrap() {
            let Some(Candidate {
                flop_cost,
                size_cost,
                parent_ids,
                child_id,
            }) = candidates.pop()
            else {
                break;
            };
            new_remaining = remaining.clone();
            new_remaining.retain(|e| *e != parent_ids.0 as u32 && *e != parent_ids.1 as u32);
            new_remaining.push(child_id as u32);
            new_path = path.clone();
            new_path.push((parent_ids.0, parent_ids.1, child_id));
            BranchBound::branch_iterate(
                self,
                new_path,
                new_remaining,
                flop_cost as u128,
                size_cost as u128,
            );
        }
    }
}

impl<'a> OptimizePath for BranchBound<'a> {
    fn optimize_path(&mut self) {
        if self.tn.is_single_tensor() {
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
                tensor.set_legs(tensor.external_edges());
            }
            self.size_cache
                .entry(index)
                .or_insert_with(|| tensor.shape().iter().product::<u128>());

            self.tensor_cache.entry(index).or_insert_with(|| tensor);
        }
        let remaining = (0u32..self.tn.tensors().len() as u32).collect();
        BranchBound::branch_iterate(self, vec![], remaining, 0, 0);
        sub_tensor_contraction.extend_from_slice(&self.best_path);
        self.best_path = sub_tensor_contraction;
    }

    fn get_best_flops(&self) -> u128 {
        self.best_flops
    }

    fn get_best_size(&self) -> u128 {
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
    use crate::contractionpath::paths::branchbound::BranchBound;
    use crate::contractionpath::paths::CostType;
    use crate::contractionpath::paths::OptimizePath;
    use crate::path;
    use crate::tensornetwork::create_tensor_network;
    use crate::tensornetwork::tensor::Tensor;

    fn setup_simple() -> Tensor {
        create_tensor_network(
            vec![
                Tensor::new(vec![4, 3, 2]),
                Tensor::new(vec![0, 1, 3, 2]),
                Tensor::new(vec![4, 5, 6]),
            ],
            &[(0, 5), (1, 2), (2, 6), (3, 8), (4, 1), (5, 3), (6, 4)].into(),
            None,
        )
    }

    fn setup_complex() -> Tensor {
        create_tensor_network(
            vec![
                Tensor::new(vec![4, 3, 2]),
                Tensor::new(vec![0, 1, 3, 2]),
                Tensor::new(vec![4, 5, 6]),
                Tensor::new(vec![6, 8, 9]),
                Tensor::new(vec![10, 8, 9]),
                Tensor::new(vec![5, 1, 0]),
            ],
            &[
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
            ]
            .into(),
            None,
        )
    }

    #[test]
    fn test_contract_order_simple() {
        let tn = setup_simple();
        let mut opt = BranchBound::new(&tn, None, 20, CostType::Flops);
        opt.optimize_path();

        assert_eq!(opt.best_flops, 3694);
        assert_eq!(opt.best_size, 538);
        assert_eq!(opt.get_best_path(), &path![(0, 1), (2, 3)]);
        assert_eq!(opt.get_best_replace_path(), path![(0, 1), (0, 2)]);
    }

    #[test]
    fn test_contract_order_complex() {
        let tn = setup_complex();
        let mut opt = BranchBound::new(&tn, None, 20, CostType::Flops);
        opt.optimize_path();

        assert_eq!(opt.best_flops, 2003348);
        assert_eq!(opt.best_size, 89478);
        assert_eq!(opt.best_path, path![(1, 5), (0, 6), (2, 7), (3, 8), (4, 9)]);
        assert_eq!(
            opt.get_best_replace_path(),
            path![(1, 5), (0, 1), (0, 2), (0, 3), (0, 4)]
        );
    }
}
