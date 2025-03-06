use itertools::Itertools;
use rand::{distributions::WeightedIndex, prelude::*};
use rustc_hash::FxHashMap;
use std::collections::BinaryHeap;

use crate::{tensornetwork::tensor::Tensor, types::ContractionIndex};

use super::{
    candidates::Candidate,
    contraction_cost::contract_path_cost,
    paths::{CostType, OptimizePath, RNGChooser},
    ssa_replace_ordering,
};
use crate::contractionpath::paths::greedy::Greedy;

pub trait RandomOptimizePath {
    fn random_optimize_path<R>(&mut self, trials: usize, rng: &mut R)
    where
        R: ?Sized + Rng;
}

struct ThermalChooser;

impl RNGChooser for ThermalChooser {
    fn choose<R>(
        &self,
        queue: &mut BinaryHeap<Candidate>,
        remaining_tensors: &FxHashMap<u64, usize>,
        nbranch: usize,
        mut temperature: f64,
        rel_temperature: bool,
        rng: &mut R,
    ) -> Option<Candidate>
    where
        R: ?Sized + Rng,
    {
        let mut choices = Vec::new();
        while !queue.is_empty() && choices.len() <= nbranch {
            let candidate = queue.pop();
            if let Some(Candidate {
                flop_cost,
                size_cost,
                parent_ids: (k1, k2),
                child_id,
            }) = candidate
            {
                if !remaining_tensors.values().any(|&x| x == k1)
                    && !remaining_tensors.values().any(|&x| x == k2)
                {
                    continue;
                }
                choices.push(Candidate {
                    flop_cost,
                    size_cost,
                    parent_ids: (k1, k2),
                    child_id,
                });
            }
        }
        let n = choices.len();
        if n == 0 {
            return None;
        }
        if n == 1 {
            return choices.pop();
        }

        let costs = choices.iter().map(|e| e.size_cost).collect_vec();
        let min_cost = costs[0];

        // adjust by the overall scale to account for fluctuating absolute costs
        if rel_temperature {
            temperature *= min_cost.abs().max(1f64);
        }

        // compute relative probability for each potential contraction
        let mut weights = Vec::new();
        if temperature == 0.0 {
            weights = vec![0.0; costs.len()];
            weights[0] = 1.0;
        } else {
            for cost in costs {
                weights.push((-(cost - min_cost) / temperature).exp());
            }
        }
        let dist = WeightedIndex::new(&weights).unwrap();
        let chosen = dist.sample(rng);
        let candidate = choices.get(chosen);
        for (index, other) in choices.iter().enumerate() {
            if index != chosen {
                queue.push(other.clone());
            }
        }
        candidate.cloned()
    }
}

impl RandomOptimizePath for Greedy<'_> {
    fn random_optimize_path<R>(&mut self, trials: usize, rng: &mut R)
    where
        R: ?Sized + Rng,
    {
        let mut inputs: Vec<Tensor> = self.tn.tensors().clone();
        for (index, input_tensor) in inputs.iter_mut().enumerate() {
            if input_tensor.is_composite() {
                let mut best_path = vec![];
                let mut best_cost = f64::INFINITY;
                let mut best_size = f64::INFINITY;
                let external_tensor = input_tensor.external_tensor();
                for _ in 0..trials {
                    let ssa_path = self.ssa_greedy_optimize(
                        input_tensor.tensors(),
                        &external_tensor,
                        &ThermalChooser,
                        Box::new(&Greedy::cost_memory_removed),
                        rng,
                    );
                    let (cost, size) = contract_path_cost(
                        input_tensor.tensors(),
                        &ssa_replace_ordering(&ssa_path, input_tensor.tensors().len()),
                        false,
                    );
                    match self.minimize {
                        CostType::Size => {
                            if size < best_size {
                                best_size = size;
                                best_path = ssa_path;
                            }
                        }
                        CostType::Flops => {
                            if cost < best_cost {
                                best_cost = cost;
                                best_path = ssa_path;
                            }
                        }
                    }
                }
                if !best_path.is_empty() {
                    let best_path = ssa_replace_ordering(&best_path, input_tensor.tensors().len());
                    self.best_path
                        .push(ContractionIndex::Path(index, None, best_path));
                }
                *input_tensor = external_tensor;
            }
        }
        // Vector of output leg ids
        let output_dims = self.tn.external_tensor();
        // Dictionary that maps leg id to bond dimension
        let mut best_path = vec![];
        let mut best_cost = f64::INFINITY;
        let mut best_size = f64::INFINITY;
        for _ in 0..trials {
            let ssa_path = self.ssa_greedy_optimize(
                &inputs,
                &output_dims,
                &ThermalChooser,
                Box::new(&Greedy::cost_memory_removed),
                rng,
            );
            let (cost, size) = contract_path_cost(
                &inputs,
                &ssa_replace_ordering(&ssa_path, inputs.len()),
                false,
            );

            match self.minimize {
                CostType::Size => {
                    if size < best_size {
                        best_size = size;
                        best_path = ssa_path;
                    }
                }
                CostType::Flops => {
                    if cost < best_cost {
                        best_cost = cost;
                        best_path = ssa_path;
                    }
                }
            }
        }
        self.best_path.append(&mut best_path);
        let (op_cost, mem_cost) =
            contract_path_cost(self.tn.tensors(), &self.get_best_replace_path(), false);
        self.best_size = mem_cost;
        self.best_flops = op_cost;
    }
}

#[cfg(test)]
mod tests {
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use rustc_hash::FxHashMap;

    use crate::contractionpath::paths::CostType;
    use crate::contractionpath::paths::OptimizePath;
    use crate::contractionpath::random_paths::Greedy;
    use crate::contractionpath::random_paths::RandomOptimizePath;
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
    fn test_contract_order_greedy_simple() {
        let tn = setup_simple();
        let mut opt = Greedy::new(&tn, CostType::Flops);
        opt.random_optimize_path(120, &mut StdRng::seed_from_u64(42));
        assert_eq!(opt.best_flops, 4540.);
        assert_eq!(opt.best_size, 538.);
        assert_eq!(opt.best_path, path![(1, 0), (2, 3)]);
        assert_eq!(opt.get_best_replace_path(), path![(1, 0), (2, 1)]);
    }

    #[test]
    fn test_contract_order_greedy_complex() {
        let tn = setup_complex();
        let mut opt = Greedy::new(&tn, CostType::Flops);
        opt.random_optimize_path(120, &mut StdRng::seed_from_u64(42));

        assert_eq!(opt.best_flops, 4228664.);
        assert_eq!(opt.best_size, 89478.);
        assert_eq!(opt.best_path, path![(1, 5), (3, 4), (0, 6), (2, 8), (7, 9)]);
        assert_eq!(
            opt.get_best_replace_path(),
            path![(1, 5), (3, 4), (0, 1), (2, 0), (3, 2)]
        );
    }
}
