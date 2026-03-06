use rustc_hash::FxHashMap;
use rustengra::hyper::cotengra_hyperoptimizer;

use crate::{
    contractionpath::{
        contraction_cost::contract_path_cost,
        paths::{CostType, FindPath},
        ssa_replace_ordering, ContractionPath,
    },
    tensornetwork::tensor::{CompositeTensor, LeafTensor, TensorType},
};

pub use rustengra::hyper::HyperOptions;

/// Creates an interface to access `Cotengra` methods in Rust. Specifically exposes
/// `search` method of `HyperOptimizer`.
pub struct Hyperoptimizer<'a> {
    tensor: &'a CompositeTensor,
    hyper_options: HyperOptions,
    best_flops: f64,
    best_size: f64,
    best_path: ContractionPath,
}

impl<'a> Hyperoptimizer<'a> {
    pub fn new(
        tensor: &'a CompositeTensor,
        minimize: CostType,
        hyper_options: HyperOptions,
    ) -> Self {
        assert_eq!(
            minimize,
            CostType::Flops,
            "Currently, only Flops is supported"
        );
        Self {
            tensor,
            hyper_options,
            best_flops: f64::INFINITY,
            best_size: f64::INFINITY,
            best_path: ContractionPath::default(),
        }
    }
}

impl FindPath for Hyperoptimizer<'_> {
    fn find_path(&mut self) {
        // Handle nested tensors first
        let mut nested_paths = FxHashMap::default();
        let mut inputs = Vec::with_capacity(self.tensor.len());
        let mut output = LeafTensor::default();
        let mut size_dict = FxHashMap::default();

        for (index, tensor) in self.tensor.tensors().iter().enumerate() {
            match tensor.kind() {
                TensorType::Composite => {
                    // Find a path for the nested tensors
                    let composite = tensor.as_composite().unwrap();
                    let mut hp =
                        Hyperoptimizer::new(composite, CostType::Flops, self.hyper_options.clone());
                    hp.find_path();
                    nested_paths.insert(index, hp.get_best_path().clone());

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

        let ssa_path = cotengra_hyperoptimizer(
            &inputs,
            output.legs(),
            &size_dict,
            "kahypar",
            &self.hyper_options,
        )
        .unwrap();

        self.best_path = ContractionPath {
            nested: nested_paths,
            toplevel: ssa_path,
        };

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
            &tn,
            CostType::Flops,
            HyperOptions::new()
                .with_max_repeats(16)
                .with_parallel(false),
        );
        opt.find_path();

        assert_eq!(opt.best_flops, 600.);
        assert_eq!(opt.best_size, 538.);
        assert_eq!(opt.get_best_path(), &path![(0, 1), (2, 3)]);
        assert_eq!(opt.get_best_replace_path(), path![(0, 1), (2, 0)]);
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
