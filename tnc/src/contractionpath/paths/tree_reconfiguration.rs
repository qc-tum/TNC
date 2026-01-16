use itertools::Itertools;
use rustc_hash::FxHashMap;
use rustengra::{
    cotengra_check, cotengra_optimized_greedy, replace_to_ssa_path, tensor_legs_to_digit,
};

use crate::{
    contractionpath::{
        contraction_cost::contract_path_cost,
        paths::{CostType, FindPath},
        ssa_replace_ordering, ContractionPath,
    },
    tensornetwork::tensor::Tensor,
};

/// Creates an interface to `rustengra` an interface to access `Cotengra` methods in
/// Rust. Specifically exposes `subtree_reconfigure` method.
pub struct TreeReconfigure<'a> {
    tensor: &'a Tensor,
    subtree_size: usize,
    best_flops: f64,
    best_size: f64,
    best_path: ContractionPath,
}

impl<'a> TreeReconfigure<'a> {
    /// Creates a new [`TreeReconfigure`] instance. `subtree_size` is the
    /// size of subtrees that is considered (increases the optimization cost
    /// exponentially!).
    pub fn new(tensor: &'a Tensor, subtree_size: usize, minimize: CostType) -> Self {
        assert!(cotengra_check().is_ok());
        assert_eq!(
            minimize,
            CostType::Flops,
            "Currently, only Flops is supported"
        );
        Self {
            tensor,
            subtree_size,
            best_flops: f64::INFINITY,
            best_size: f64::INFINITY,
            best_path: ContractionPath::default(),
        }
    }
}

impl FindPath for TreeReconfigure<'_> {
    fn find_path(&mut self) {
        // Map tensors to legs
        let inputs = self
            .tensor
            .tensors()
            .iter()
            .map(|tensor| tensor.legs().clone())
            .collect_vec();
        let outputs = self.tensor.external_tensor();
        let size_dict = self.tensor.tensors().iter().map(|t| t.edges()).fold(
            FxHashMap::default(),
            |mut acc, edges| {
                acc.extend(edges);
                acc
            },
        );

        let (inputs, outputs, size_dict) =
            tensor_legs_to_digit(&inputs, outputs.legs(), &size_dict);

        let replace_path =
            cotengra_optimized_greedy(&inputs, outputs, size_dict, self.subtree_size).unwrap();

        let best_path = replace_to_ssa_path(replace_path, self.tensor.tensors().len());

        self.best_path = ContractionPath::simple(best_path);

        let (op_cost, mem_cost) =
            contract_path_cost(self.tensor.tensors(), &self.get_best_replace_path(), true);

        self.best_flops = op_cost;
        self.best_size = mem_cost;
    }

    fn get_best_flops(&self) -> f64 {
        self.best_flops
    }

    fn get_best_size(&self) -> f64 {
        self.best_size
    }

    fn get_best_path(&self) -> &ContractionPath {
        &self.best_path
    }

    fn get_best_replace_path(&self) -> ContractionPath {
        ssa_replace_ordering(&self.best_path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use rustc_hash::FxHashMap;

    use crate::{
        contractionpath::paths::{CostType, FindPath},
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
    fn test_tree_contract_order_simple() {
        let tn = setup_simple();
        let mut opt = TreeReconfigure::new(&tn, 8, CostType::Flops);
        opt.find_path();

        assert_eq!(opt.best_flops, 600.);
        assert_eq!(opt.best_size, 538.);
        assert_eq!(opt.get_best_path(), &path![(0, 1), (2, 3)]);
        assert_eq!(opt.get_best_replace_path(), path![(0, 1), (2, 0)]);
    }

    #[test]
    fn test_tree_contract_order_complex() {
        let tn = setup_complex();
        let mut opt = TreeReconfigure::new(&tn, 8, CostType::Flops);
        opt.find_path();

        assert_eq!(opt.best_flops, 332685.);
        assert_eq!(opt.best_size, 89478.);
        assert_eq!(opt.best_path, path![(1, 5), (0, 6), (2, 7), (3, 8), (4, 9)]);
        assert_eq!(
            opt.get_best_replace_path(),
            path![(1, 5), (0, 1), (2, 0), (3, 2), (4, 3)]
        );
    }
}
