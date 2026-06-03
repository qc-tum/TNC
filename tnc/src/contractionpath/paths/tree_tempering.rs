use itertools::Itertools;
use rustc_hash::FxHashMap;
use rustengra::{cotengra_check, cotengra_tree_tempering};

use crate::{
    contractionpath::{
        contraction_cost::contract_path_cost,
        paths::{BasicContractionPathResult, CostType, Pathfinder},
        ssa_replace_ordering, ContractionPath,
    },
    tensornetwork::tensor::Tensor,
};

/// Creates an interface to `rustengra` an interface to access `Cotengra` methods in
/// Rust. Specifically exposes `parallel_temper_tree` method.
pub struct TreeTempering {
    numiter: Option<usize>,
    seed: Option<u64>,
}

impl TreeTempering {
    pub fn new(seed: Option<u64>, minimize: CostType, numiter: Option<usize>) -> Self {
        cotengra_check().expect("Needs python and cotengra installed");
        assert_eq!(
            minimize,
            CostType::Flops,
            "Currently, only Flops is supported"
        );
        Self { numiter, seed }
    }
}

impl Pathfinder for TreeTempering {
    type Result = BasicContractionPathResult;

    fn find_path(&mut self, tensor: &Tensor) -> BasicContractionPathResult {
        // Map tensors to legs
        let inputs = tensor
            .tensors()
            .iter()
            .map(|tensor| tensor.legs().clone())
            .collect_vec();
        let outputs = tensor.external_tensor();
        let size_dict = tensor.tensors().iter().map(Tensor::edges).fold(
            FxHashMap::default(),
            |mut acc, edges| {
                acc.extend(edges);
                acc
            },
        );

        let best_path =
            cotengra_tree_tempering(&inputs, outputs.legs(), self.numiter, &size_dict, self.seed)
                .unwrap();

        let best_path = ContractionPath::simple(best_path);
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
    #[ignore = "flaky test due to a bug in cotengra"]
    fn test_temper_tree_contract_order_simple() {
        let tn = setup_simple();
        let mut opt = TreeTempering::new(Some(8), CostType::Flops, Some(100));
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
    #[ignore = "flaky test due to a bug in cotengra"]
    fn test_temper_tree_contract_order_complex() {
        let tn = setup_complex();
        let mut opt = TreeTempering::new(Some(8), CostType::Flops, Some(100));
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
