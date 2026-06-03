use itertools::Itertools;
use rustc_hash::FxHashMap;
use rustengra::{cotengra_check, cotengra_sa_tree};

use crate::{
    contractionpath::{
        contraction_cost::contract_path_cost,
        paths::{BasicContractionPathResult, CostType, Pathfinder},
        ssa_replace_ordering, ContractionPath,
    },
    tensornetwork::tensor::Tensor,
};

/// Creates an interface to `rustengra` an interface to access `Cotengra` methods in
/// Rust. Specifically exposes `simulated_anneal_tree` method.
pub struct TreeAnnealing<'a> {
    tensor: &'a Tensor,
    temperature_steps: Option<usize>,
    numiter: Option<usize>,
    seed: Option<u64>,
}

impl<'a> TreeAnnealing<'a> {
    pub fn new(
        tensor: &'a Tensor,
        seed: Option<u64>,
        minimize: CostType,
        temperature_steps: Option<usize>,
        numiter: Option<usize>,
    ) -> Self {
        cotengra_check().expect("Needs python and cotengra installed");
        assert_eq!(
            minimize,
            CostType::Flops,
            "Currently, only Flops is supported"
        );
        Self {
            tensor,
            temperature_steps,
            numiter,
            seed,
        }
    }
}

impl Pathfinder for TreeAnnealing<'_> {
    type Result = BasicContractionPathResult;

    fn find_path(&mut self) -> BasicContractionPathResult {
        // Map tensors to legs
        let inputs = self
            .tensor
            .tensors()
            .iter()
            .map(|tensor| tensor.legs().clone())
            .collect_vec();
        let outputs = self.tensor.external_tensor();
        let size_dict = self.tensor.tensors().iter().map(Tensor::edges).fold(
            FxHashMap::default(),
            |mut acc, edges| {
                acc.extend(edges);
                acc
            },
        );

        let best_path = cotengra_sa_tree(
            &inputs,
            outputs.legs(),
            self.temperature_steps,
            self.numiter,
            &size_dict,
            self.seed,
        )
        .unwrap();

        let best_path = ContractionPath::simple(best_path);
        let replace_path = ssa_replace_ordering(&best_path);

        let (op_cost, mem_cost) = contract_path_cost(self.tensor.tensors(), &replace_path, true);

        BasicContractionPathResult {
            ssa_path: best_path,
            flops: op_cost,
            size: mem_cost,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use rustc_hash::FxHashMap;

    use crate::{
        contractionpath::paths::{CostType, Pathfinder},
        path,
        tensornetwork::tensor::Tensor,
    };

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
    fn test_anneal_tree_contract_order_simple() {
        let tn = setup_simple();
        let mut opt = TreeAnnealing::new(&tn, Some(8), CostType::Flops, Some(100), Some(50));
        let result = opt.find_path();

        assert_eq!(
            result,
            BasicContractionPathResult {
                ssa_path: path![(0, 1), (2, 3)],
                flops: 600.,
                size: 538.
            }
        );
    }

    #[test]
    fn test_anneal_tree_contract_order_complex() {
        let tn = setup_complex();
        let mut opt = TreeAnnealing::new(&tn, Some(8), CostType::Flops, Some(100), Some(50));
        let result = opt.find_path();

        assert_eq!(
            result,
            BasicContractionPathResult {
                ssa_path: path![(1, 5), (0, 6), (2, 7), (3, 8), (4, 9)],
                flops: 332685.,
                size: 89478.
            }
        );
    }
}
