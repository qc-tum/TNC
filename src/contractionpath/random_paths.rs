use rand::{distributions::WeightedIndex, prelude::*};
use std::{
    cmp::max,
    collections::{BinaryHeap, HashMap},
};

use crate::tensornetwork::tensor::Tensor;

use super::{
    candidates::Candidate, contraction_cost::_contract_path_cost, paths::Greedy,
    ssa_replace_ordering,
};

pub trait RandomOptimizePath {
    fn random_optimize_path<R>(&mut self, trials: usize, rng: &mut R)
    where
        R: ?Sized + Rng;
}

// __all__ = ["RandomGreedy", "random_greedy", "random_greedy_128"]
impl<'a> Greedy<'a> {
    pub(crate) fn _thermal_chooser<R: Rng + ?Sized>(
        queue: &mut BinaryHeap<Candidate>,
        remaining_tensors: &HashMap<Tensor, usize>,
        nbranch: usize,
        mut temperature: f64,
        rel_temperature: bool,
        mut rng: &mut R,
    ) -> Option<Candidate> {
        let mut n = 0;
        let mut choices = Vec::new();
        while !queue.is_empty() && n <= nbranch {
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
                if !remaining_tensors.contains_key(&k1) || !remaining_tensors.contains_key(&k2) {
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
                n += 1;
            }
        }

        if n == 0 {
            return None;
        }
        if n == 1 {
            return Some(choices[0].clone());
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
                weights.push((-(-c - cmin) as f64).exp());
            }
        }
        let dist = WeightedIndex::new(&weights).unwrap();
        let chosen = dist.sample(&mut rng);
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
        let inputs: Vec<Tensor> = self.tn.get_tensors().clone();

        let output_dims = Tensor::new(self.tn.get_ext_edges().clone());

        // Dictionary that maps leg id to bond dimension
        let bond_dims = self.tn.get_bond_dims();
        for _ in 0..trials {
            let ssa_path = self._ssa_greedy_optimize(
                &inputs,
                &output_dims,
                bond_dims,
                Box::new(&Greedy::_thermal_chooser),
                Box::new(&Greedy::_cost_memory_removed),
            );
            let (cost, size) = _contract_path_cost(
                &inputs,
                &ssa_replace_ordering(&ssa_path, inputs.len()),
                bond_dims,
            );

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
    use rand::thread_rng;

    use crate::contractionpath::paths::CostType;
    // use rand::distributions::{Distribution, Uniform};
    // TODO: Use random tensors
    use crate::contractionpath::paths::Greedy;
    use crate::contractionpath::paths::OptimizePath;
    use crate::contractionpath::random_paths::RandomOptimizePath;
    use crate::tensornetwork::tensor::Tensor;
    use crate::tensornetwork::TensorNetwork;

    fn setup_simple() -> TensorNetwork {
        TensorNetwork::from_vector(
            vec![
                Tensor::new(vec![4, 3, 2]),
                Tensor::new(vec![0, 1, 3, 2]),
                Tensor::new(vec![4, 5, 6]),
            ],
            vec![5, 2, 6, 8, 1, 3, 4],
            None,
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
            None,
        )
    }

    #[test]
    fn test_contract_order_greedy_simple() {
        let tn = setup_simple();
        let mut opt = Greedy::new(&tn, CostType::Flops);
        opt.random_optimize_path(32, &mut thread_rng());
        assert_eq!(opt.best_flops, 600);
        assert_eq!(opt.best_size, 538);
        assert_eq!(opt.best_path, vec![(0, 1), (2, 3)]);
        assert_eq!(opt.get_best_replace_path(), vec![(0, 1), (2, 0)]);
    }
    #[test]
    fn test_contract_order_greedy_complex() {
        let tn = setup_complex();
        let mut opt = Greedy::new(&tn, CostType::Flops);
        opt.random_optimize_path(32, &mut thread_rng());

        assert_eq!(opt.best_flops, 528750);
        assert_eq!(opt.best_size, 89478);
        assert_eq!(opt.best_path, vec![(1, 5), (3, 4), (0, 6), (2, 8), (7, 9)]);
        assert_eq!(
            opt.get_best_replace_path(),
            vec![(1, 5), (3, 4), (0, 1), (2, 0), (3, 2)]
        );
    }
}
