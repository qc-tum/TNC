use itertools::Itertools;
use rustc_hash::FxHashMap;
use rustengra::{
    cotengra_check, create_and_optimize_tree, replace_to_ssa_path, tensor_legs_to_digit,
};

use crate::{
    contractionpath::{contraction_cost::contract_path_cost, ssa_replace_ordering},
    tensornetwork::tensor::Tensor,
    types::ContractionIndex,
};

use super::{CostType, OptimizePath};

/// Creates an interface to `rustengra` an interface to access `Cotengra` methods in rust.
/// Specifically exposes `subtree_reconfigure` method.
pub struct TreeReconfigure<'a> {
    tensor: &'a Tensor,
    minimize: CostType,
    subtree_size: usize,
    best_flops: f64,
    best_size: f64,
    best_path: Vec<ContractionIndex>,
    best_progress: FxHashMap<usize, f64>,
}

impl<'a> TreeReconfigure<'a> {
    /// Creates a new [`TreeReconfigure`] instance. The `initial_path` is an initial
    /// contraction path in SSA format that is to be optimized. `subtree_size` is the
    /// size of subtrees that is considered (increases the optimization cost
    /// exponentially!).
    pub fn new(
        tensor: &'a Tensor,
        initial_path: Vec<ContractionIndex>,
        subtree_size: usize,
        minimize: CostType,
    ) -> Self {
        assert!(cotengra_check().is_ok());
        assert!(
            initial_path
                .iter()
                .all(|index| matches!(index, ContractionIndex::Pair(..))),
            "This method does not support nested Paths"
        );
        Self {
            tensor,
            minimize,
            subtree_size,
            best_flops: f64::INFINITY,
            best_size: f64::INFINITY,
            best_path: initial_path,
            best_progress: FxHashMap::default(),
        }
    }
}

impl OptimizePath for TreeReconfigure<'_> {
    fn optimize_path(&mut self) {
        // Map tensors to legs
        let inputs = self
            .tensor
            .tensors()
            .iter()
            .map(|tensor| tensor.legs().clone())
            .collect_vec();
        let outputs = self.tensor.external_edges();
        let size_dict = self.tensor.bond_dims();

        let (inputs, outputs, size_dict) = tensor_legs_to_digit(&inputs, &outputs, &size_dict);

        // Map ContractIndex to (i, j) tuples
        let best_path = self
            .best_path
            .iter()
            .map(|a| {
                if let ContractionIndex::Pair(i, j) = a {
                    (*i, *j)
                } else {
                    panic!("This method does not support nested Paths")
                }
            })
            .collect_vec();

        let is_ssa = true;
        let replace_path = create_and_optimize_tree(
            &inputs,
            outputs,
            size_dict,
            best_path,
            self.subtree_size,
            is_ssa,
        )
        .unwrap();

        let best_path = replace_to_ssa_path(replace_path, self.tensor.tensors().len());

        self.best_path = best_path
            .iter()
            .map(|(i, j)| ContractionIndex::Pair(*i, *j))
            .collect_vec();

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

    fn get_best_path(&self) -> &Vec<ContractionIndex> {
        &self.best_path
    }

    fn get_best_replace_path(&self) -> Vec<ContractionIndex> {
        ssa_replace_ordering(&self.best_path, self.tensor.tensors().len())
    }
}

#[cfg(test)]
mod tests {
    use rand::{rngs::StdRng, SeedableRng};
    use rustc_hash::FxHashMap;

    use crate::{
        contractionpath::paths::{
            greedy::Greedy, tree_reconfiguration::TreeReconfigure, CostType, OptimizePath,
        },
        networks::{connectivity::ConnectivityLayout, random_circuit::random_circuit},
        path,
        tensornetwork::{create_tensor_network, tensor::Tensor},
    };

    fn setup_simple() -> Tensor {
        create_tensor_network(
            vec![
                Tensor::new(vec![4, 3, 2]),
                Tensor::new(vec![0, 1, 3, 2]),
                Tensor::new(vec![4, 5, 6]),
            ],
            &FxHashMap::from_iter([(0, 5), (1, 2), (2, 6), (3, 8), (4, 1), (5, 3), (6, 4)]),
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
            &FxHashMap::from_iter([
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
            ]),
        )
    }

    fn setup_large() -> Tensor {
        let mut rng = StdRng::seed_from_u64(23);
        let qubits = 15;
        let depth = 40;
        let single_qubit_probability = 0.4;
        let two_qubit_probability = 0.4;
        let connectivity = ConnectivityLayout::Osprey;
        random_circuit(
            qubits,
            depth,
            single_qubit_probability,
            two_qubit_probability,
            &mut rng,
            connectivity,
        )
    }

    #[test]
    #[ignore]
    fn test_tree_contract_order_simple() {
        let tn = setup_simple();
        let mut greedy_opt = Greedy::new(&tn, CostType::Flops);
        greedy_opt.optimize_path();
        let mut opt = TreeReconfigure::new(
            &tn,
            greedy_opt.get_best_path().to_owned(),
            8,
            CostType::Flops,
        );
        opt.optimize_path();

        assert_eq!(opt.best_flops, 600.);
        assert_eq!(opt.best_size, 538.);
        assert_eq!(opt.get_best_path(), &path![(0, 1), (2, 3)]);
        assert_eq!(opt.get_best_replace_path(), path![(0, 1), (2, 0)]);
    }

    #[test]
    #[ignore]
    fn test_tree_contract_order_complex() {
        let tn = setup_complex();
        let mut greedy_opt = Greedy::new(&tn, CostType::Flops);
        greedy_opt.optimize_path();
        let mut opt = TreeReconfigure::new(
            &tn,
            greedy_opt.get_best_path().to_owned(),
            8,
            CostType::Flops,
        );
        opt.optimize_path();

        assert_eq!(opt.best_flops, 332685.);
        assert_eq!(opt.best_size, 89478.);
        assert_eq!(opt.best_path, path![(1, 5), (0, 6), (2, 7), (3, 8), (4, 9)]);
        assert_eq!(
            opt.get_best_replace_path(),
            path![(1, 5), (0, 1), (2, 0), (3, 2), (4, 3)]
        );
    }

    #[test]
    #[ignore]
    fn test_tree_large() {
        let tn = setup_large();
        let mut greedy_opt = Greedy::new(&tn, CostType::Flops);
        greedy_opt.optimize_path();
        let mut opt = TreeReconfigure::new(
            &tn,
            greedy_opt.get_best_path().to_owned(),
            8,
            CostType::Flops,
        );
        opt.optimize_path();
    }
}
