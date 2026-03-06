use rustc_hash::FxHashMap;
use rustengra::hyper::cotengra_hyperoptimizer;

use crate::{
    contractionpath::{
        contraction_cost::contract_path_cost,
        paths::{BasicContractionPathResult, ContractionPathResult, CostType, Pathfinder},
        ssa_replace_ordering, ContractionPath,
    },
    tensornetwork::tensor::{CompositeTensor, LeafTensor, TensorType},
};

pub use rustengra::hyper::HyperOptions;

/// Creates an interface to access `Cotengra` methods in Rust. Specifically exposes
/// `search` method of `HyperOptimizer`.
pub struct Hyperoptimizer {
    hyper_options: HyperOptions,
}

impl Hyperoptimizer {
    #[inline]
    pub fn new(minimize: CostType, hyper_options: HyperOptions) -> Self {
        assert_eq!(
            minimize,
            CostType::Flops,
            "Currently, only Flops is supported"
        );
        Self { hyper_options }
    }
}

impl Pathfinder for Hyperoptimizer {
    type Result = BasicContractionPathResult;

    fn find_path(&mut self, tensor: &CompositeTensor) -> BasicContractionPathResult {
        // Handle nested tensors first
        let mut nested_paths = FxHashMap::default();
        let mut inputs = Vec::with_capacity(tensor.len());
        let mut output = LeafTensor::default();
        let mut size_dict = FxHashMap::default();

        for (index, tensor) in tensor.tensors().iter().enumerate() {
            match tensor.kind() {
                TensorType::Composite => {
                    // Find a path for the nested tensors
                    let composite = tensor.as_composite().unwrap();
                    let result = self.find_path(composite);
                    nested_paths.insert(index, result.ssa_path().clone());

                    // Get the outer tensor after contraction
                    let leaf = composite.external_tensor();
                    size_dict.extend(leaf.edges());
                    output ^= &leaf;
                    inputs.push(leaf.into_legs());
                }
                TensorType::Leaf => {
                    let leaf = tensor.as_leaf().unwrap();
                    size_dict.extend(leaf.edges());
                    output ^= leaf;
                    inputs.push(leaf.legs().clone());
                }
            }
        }

        let (ssa_path, _) = cotengra_hyperoptimizer(
            &inputs,
            output.legs(),
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

        let (op_cost, mem_cost) = contract_path_cost(tensor.tensors(), &replace_path, true);

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
    };

    fn setup_simple() -> CompositeTensor {
        let bond_dims =
            FxHashMap::from_iter([(0, 5), (1, 2), (2, 6), (3, 8), (4, 1), (5, 3), (6, 4)]);
        CompositeTensor::new(vec![
            LeafTensor::new_from_map(vec![4, 3, 2], &bond_dims),
            LeafTensor::new_from_map(vec![0, 1, 3, 2], &bond_dims),
            LeafTensor::new_from_map(vec![4, 5, 6], &bond_dims),
        ])
    }

    fn setup_complex() -> CompositeTensor {
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
        CompositeTensor::new(vec![
            LeafTensor::new_from_map(vec![4, 3, 2], &bond_dims),
            LeafTensor::new_from_map(vec![0, 1, 3, 2], &bond_dims),
            LeafTensor::new_from_map(vec![4, 5, 6], &bond_dims),
            LeafTensor::new_from_map(vec![6, 8, 9], &bond_dims),
            LeafTensor::new_from_map(vec![10, 8, 9], &bond_dims),
            LeafTensor::new_from_map(vec![5, 1, 0], &bond_dims),
        ])
    }

    #[test]
    fn test_hyper_tree_contract_order_simple() {
        let tn = setup_simple();
        let mut opt = Hyperoptimizer::new(
            CostType::Flops,
            HyperOptions::new()
                .with_max_repeats(16)
                .with_parallel(false),
        );
        let result = opt.find_path(&tn);

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
            CostType::Flops,
            HyperOptions::new()
                .with_max_repeats(32)
                .with_parallel(false),
        );
        let result = opt.find_path(&tn);

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
