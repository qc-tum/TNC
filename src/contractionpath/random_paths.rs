use rand::{distributions::WeightedIndex, prelude::*};
use std::{
    cmp::max,
    collections::{BinaryHeap, HashMap},
};

use crate::{tensornetwork::tensor::Tensor, types::calculate_hash};

use super::{
    candidates::Candidate, contraction_cost::contract_path_cost, paths::RNGChooser,
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
        remaining_tensors: &HashMap<u64, usize>,
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
                parent_ids,
                parent_tensors: Some((k1, k2)),
                child_id,
                child_tensor,
            }) = candidate
            {
                let k1_hash = calculate_hash(&k1);
                let k2_hash = calculate_hash(&k2);
                if !remaining_tensors.contains_key(&k1_hash)
                    || !remaining_tensors.contains_key(&k2_hash)
                {
                    continue;
                }
                choices.push(Candidate {
                    flop_cost,
                    size_cost,
                    parent_ids,
                    parent_tensors: Some((k1, k2)),
                    child_id,
                    child_tensor,
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

        let costs = choices.iter().map(|e| e.size_cost).collect::<Vec<i64>>();
        let cmin = costs[0];

        // adjust by the overall scale to account for fluctuating absolute costs
        if rel_temperature {
            temperature *= max(1, cmin.abs()) as f64;
        }

        // compute relative probability for each potential contraction
        let mut weights = Vec::new();
        if temperature == 0.0 {
            weights = vec![0.0; costs.len()];
            weights[0] = 1.0;
        } else {
            for c in costs {
                weights.push((-(c - cmin) as f64 / temperature).exp());
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

impl<'a> RandomOptimizePath for Greedy<'a> {
    fn random_optimize_path<R>(&mut self, trials: usize, rng: &mut R)
    where
        R: ?Sized + Rng,
    {
        let inputs = self.tn.get_tensors();

        let output_dims = Tensor::new(self.tn.get_external_edges());

        // Dictionary that maps leg id to bond dimension
        for _ in 0..trials {
            let ssa_path = self.ssa_greedy_optimize(
                inputs,
                &output_dims,
                ThermalChooser,
                Box::new(&Greedy::_cost_memory_removed),
                rng,
            );
            let (cost, size) =
                contract_path_cost(inputs, &ssa_replace_ordering(&ssa_path, inputs.len()));

            if cost < self.best_flops {
                self.best_flops = cost;
                self.best_size = size;
                self.best_path = ssa_path;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    use crate::contractionpath::paths::CostType;
    // use rand::distributions::{Distribution, Uniform};
    // TODO: Use random tensors
    use crate::contractionpath::paths::OptimizePath;
    use crate::contractionpath::random_paths::Greedy;
    use crate::contractionpath::random_paths::RandomOptimizePath;
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
    #[ignore]
    fn test_contract_order_greedy_simple() {
        let tn = setup_simple();
        let mut opt = Greedy::new(&tn, CostType::Flops);
        opt.random_optimize_path(120, &mut StdRng::seed_from_u64(42));
        assert_eq!(opt.best_flops, 600);
        assert_eq!(opt.best_size, 538);
        assert_eq!(opt.best_path, path![(0, 1), (2, 3)]);
        assert_eq!(opt.get_best_replace_path(), path![(0, 1), (2, 0)]);
    }

    #[test]
    #[ignore]
    fn test_contract_order_greedy_complex() {
        let tn = setup_complex();
        let mut opt = Greedy::new(&tn, CostType::Flops);
        opt.random_optimize_path(120, &mut StdRng::seed_from_u64(42));

        assert_eq!(opt.best_flops, 528750);
        assert_eq!(opt.best_size, 89478);
        assert_eq!(opt.best_path, path![(1, 5), (3, 4), (0, 6), (2, 8), (7, 9)]);
        assert_eq!(
            opt.get_best_replace_path(),
            path![(1, 5), (3, 4), (0, 1), (2, 0), (3, 2)]
        );
    }
}
