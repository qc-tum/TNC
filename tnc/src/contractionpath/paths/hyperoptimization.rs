use itertools::Itertools;
use rustc_hash::FxHashMap;
use rustengra::hyper::cotengra_hyperoptimizer;

use crate::{
    contractionpath::{
        contraction_cost::contract_path_cost,
        paths::{BasicContractionPathResult, ContractionPathResult, CostType, Pathfinder},
        ssa_replace_ordering, ContractionPath,
    },
    tensornetwork::tensor::Tensor,
};

pub use rustengra::hyper::HyperOptions;

/// Creates an interface to access `Cotengra` methods in Rust. Specifically exposes
/// `search` method of `HyperOptimizer`.
pub struct Hyperoptimizer<'a> {
    tensor: &'a Tensor,
    hyper_options: HyperOptions,
}

impl<'a> Hyperoptimizer<'a> {
    pub fn new(tensor: &'a Tensor, minimize: CostType, hyper_options: HyperOptions) -> Self {
        assert_eq!(
            minimize,
            CostType::Flops,
            "Currently, only Flops is supported"
        );
        Self {
            tensor,
            hyper_options,
        }
    }
}

impl Pathfinder for Hyperoptimizer<'_> {
    type Result = BasicContractionPathResult;

    fn find_path(&mut self) -> BasicContractionPathResult {
        // Handle nested tensors first
        let mut nested_paths = FxHashMap::default();
        let inputs = self
            .tensor
            .tensors()
            .iter()
            .enumerate()
            .map(|(index, tensor)| {
                if tensor.is_composite() {
                    let mut hp =
                        Hyperoptimizer::new(tensor, CostType::Flops, self.hyper_options.clone());
                    let result = hp.find_path();
                    nested_paths.insert(index, result.ssa_path().clone());
                    tensor.external_tensor().legs
                } else {
                    tensor.legs.clone()
                }
            })
            .collect_vec();

        let outputs = self.tensor.external_tensor();
        let size_dict = self.tensor.tensors().iter().map(Tensor::edges).fold(
            FxHashMap::default(),
            |mut acc, edges| {
                acc.extend(edges);
                acc
            },
        );

        let ssa_path = cotengra_hyperoptimizer(
            &inputs,
            outputs.legs(),
            &size_dict,
            "kahypar",
            &self.hyper_options,
        )
        .unwrap();

        let best_path = ContractionPath {
            nested: nested_paths,
            toplevel: ssa_path,
        };
        let replace_path = ssa_replace_ordering(&best_path);

        let (op_cost, mem_cost) = contract_path_cost(self.tensor.tensors(), &&replace_path, true);

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
    fn test_hyper_tree_contract_order_simple() {
        let tn = setup_simple();
        let mut opt = Hyperoptimizer::new(
            &tn,
            CostType::Flops,
            HyperOptions::new()
                .with_max_repeats(16)
                .with_parallel(false),
        );
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
    fn test_hyper_tree_contract_order_complex() {
        let tn = setup_complex();
        let mut opt = Hyperoptimizer::new(
            &tn,
            CostType::Flops,
            HyperOptions::new()
                .with_max_repeats(32)
                .with_parallel(false),
        );
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
