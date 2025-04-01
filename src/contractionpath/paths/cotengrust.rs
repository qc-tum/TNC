use std::iter::zip;

use crate::{
    contractionpath::{contraction_cost::contract_path_cost, ssa_replace_ordering},
    tensornetwork::tensor::Tensor,
    types::ContractionIndex,
};

use super::OptimizePath;
use cotengrust::{optimize_greedy_rust, optimize_optimal_rust, optimize_random_greedy_rust};
use itertools::Itertools;
use rustc_hash::FxHashMap;

/// The optimization method to use.
#[derive(Debug, Clone)]
pub enum OptMethod {
    /// Searches for the optimal path, quite slow.
    Optimal,
    /// Uses a greedy algorithm to find a path.
    Greedy,
    /// Tries multiple greedy paths and selects the best one.
    RandomGreedy(usize),
}

/// A contraction path finder using the `cotengrust` library.
#[derive(Debug, Clone)]
pub struct Cotengrust<'a> {
    tensor: &'a Tensor,
    best_path: Vec<ContractionIndex>,
    best_flops: f64,
    best_size: f64,
    opt_method: OptMethod,
}

impl<'a> Cotengrust<'a> {
    /// Creates a new Cotengrust optimizer using the specified optimization method.
    pub fn new(tensor: &'a Tensor, opt_method: OptMethod) -> Self {
        Self {
            tensor,
            opt_method,
            best_path: Vec::new(),
            best_flops: f64::INFINITY,
            best_size: f64::INFINITY,
        }
    }

    fn optimize_single(&self, inputs: &[Tensor], output: &Tensor) -> Vec<ContractionIndex> {
        // Convert the inputs to the cotengra format
        let (inputs, output, size_dict) = tensor_legs_to_digit(inputs, output);

        // Find the contraction path
        let path = match &self.opt_method {
            OptMethod::Greedy => {
                optimize_greedy_rust(inputs, output, size_dict, None, None, Some(42), false, true)
            }
            &OptMethod::RandomGreedy(ntrials) => {
                optimize_random_greedy_rust(
                    inputs,
                    output,
                    size_dict,
                    ntrials,
                    None,
                    None,
                    Some(42),
                    false,
                    true,
                )
                .0
            }
            OptMethod::Optimal => {
                optimize_optimal_rust(inputs, output, size_dict, None, None, None, false, true)
            }
        };

        // Convert the path back to our format
        path.into_iter()
            .map(|pair| {
                let [a, b] = pair[..] else {
                    panic!("Expected two indices in contraction path pair")
                };
                ContractionIndex::Pair(a as _, b as _)
            })
            .collect_vec()
    }
}

/// Converts tensor leg inputs to chars. Creates new inputs, outputs and size_dict that can be fed to Cotengra.
fn tensor_legs_to_digit(
    inputs: &[Tensor],
    output: &Tensor,
) -> (Vec<Vec<char>>, Vec<char>, FxHashMap<char, f32>) {
    fn leg_to_char(leg: usize) -> char {
        char::from_u32(leg.try_into().unwrap()).unwrap()
    }
    let mut new_inputs = vec![Vec::new(); inputs.len()];
    let new_output = output.legs().iter().copied().map(leg_to_char).collect();
    let mut new_size_dict = FxHashMap::default();

    for (tensor, labels) in zip(inputs, new_inputs.iter_mut()) {
        labels.reserve_exact(tensor.legs().len());
        for (leg, dim) in tensor.edges() {
            let character = leg_to_char(*leg);
            labels.push(character);
            new_size_dict.insert(character, *dim as f32);
        }
    }
    (new_inputs, new_output, new_size_dict)
}

impl OptimizePath for Cotengrust<'_> {
    fn optimize_path(&mut self) {
        // Handle nested tensors first
        let mut inputs = self.tensor.tensors().clone();
        for (index, input_tensor) in inputs.iter_mut().enumerate() {
            if input_tensor.is_composite() {
                let external_tensor = input_tensor.external_tensor();
                let path = self.optimize_single(input_tensor.tensors(), &external_tensor);
                self.best_path.push(ContractionIndex::Path(index, path));
                *input_tensor = external_tensor;
            }
        }

        // Now handle the outer tensor
        let external_tensor = self.tensor.external_tensor();
        let mut outer_path = self.optimize_single(&inputs, &external_tensor);
        self.best_path.append(&mut outer_path);

        // Compute the cost
        let (op_cost, mem_cost) =
            contract_path_cost(self.tensor.tensors(), &self.get_best_replace_path(), true);
        self.best_size = mem_cost;
        self.best_flops = op_cost;
    }

    fn get_best_path(&self) -> &Vec<ContractionIndex> {
        &self.best_path
    }

    fn get_best_replace_path(&self) -> Vec<ContractionIndex> {
        ssa_replace_ordering(&self.best_path, self.tensor.tensors().len())
    }

    fn get_best_flops(&self) -> f64 {
        self.best_flops
    }

    fn get_best_size(&self) -> f64 {
        self.best_size
    }
}

#[cfg(test)]
mod tests {
    use crate::path;

    use super::*;

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

    fn setup_simple_inner_product() -> Tensor {
        let bond_dims =
            FxHashMap::from_iter([(0, 5), (1, 2), (2, 6), (3, 8), (4, 1), (5, 3), (6, 4)]);
        Tensor::new_composite(vec![
            Tensor::new_from_map(vec![4, 3, 2], &bond_dims),
            Tensor::new_from_map(vec![4, 3, 2], &bond_dims),
            Tensor::new_from_map(vec![0, 1, 5], &bond_dims),
            Tensor::new_from_map(vec![1, 6], &bond_dims),
        ])
    }

    fn setup_simple_outer_product() -> Tensor {
        let bond_dims = FxHashMap::from_iter([(0, 3), (1, 2), (2, 2)]);
        Tensor::new_composite(vec![
            Tensor::new_from_map(vec![0], &bond_dims),
            Tensor::new_from_map(vec![1], &bond_dims),
            Tensor::new_from_map(vec![2], &bond_dims),
        ])
    }

    fn setup_complex_outer_product() -> Tensor {
        let bond_dims = FxHashMap::from_iter([(0, 5), (1, 4)]);
        Tensor::new_composite(vec![
            Tensor::new_from_map(vec![0], &bond_dims),
            Tensor::new_from_map(vec![0], &bond_dims),
            Tensor::new_from_map(vec![1], &bond_dims),
            Tensor::new_from_map(vec![1], &bond_dims),
        ])
    }

    #[test]
    fn test_contract_order_greedy_simple() {
        let tn = setup_simple();
        let mut opt = Cotengrust::new(&tn, OptMethod::Greedy);
        opt.optimize_path();

        assert_eq!(opt.get_best_flops(), 600.);
        assert_eq!(opt.get_best_size(), 538.);
        assert_eq!(opt.get_best_path(), path![(0, 1), (3, 2)]);
        assert_eq!(opt.get_best_replace_path(), path![(0, 1), (0, 2)]);
    }

    #[test]
    fn test_contract_order_greedy_simple_inner() {
        let tn = setup_simple_inner_product();
        let mut opt = Cotengrust::new(&tn, OptMethod::Greedy);
        opt.optimize_path();

        assert_eq!(opt.get_best_flops(), 228.);
        assert_eq!(opt.get_best_size(), 121.);
        assert_eq!(opt.get_best_path(), path![(0, 1), (2, 3), (4, 5)]);
        assert_eq!(opt.get_best_replace_path(), path![(0, 1), (2, 3), (0, 2)]);
    }

    #[test]
    fn test_contract_order_greedy_simple_outer() {
        let tn = setup_simple_outer_product();
        let mut opt = Cotengrust::new(&tn, OptMethod::Greedy);
        opt.optimize_path();

        assert_eq!(opt.get_best_flops(), 16.);
        assert_eq!(opt.get_best_size(), 19.);
        assert_eq!(opt.get_best_path(), path![(2, 1), (0, 3)]);
        assert_eq!(opt.get_best_replace_path(), path![(2, 1), (0, 2)]);
    }

    #[test]
    fn test_contract_order_greedy_complex_outer() {
        let tn = setup_complex_outer_product();
        let mut opt = Cotengrust::new(&tn, OptMethod::Greedy);
        opt.optimize_path();

        assert_eq!(opt.get_best_flops(), 10.);
        assert_eq!(opt.get_best_size(), 11.);
        assert_eq!(opt.get_best_path(), path![(0, 1), (2, 3), (5, 4)]);
        assert_eq!(opt.get_best_replace_path(), path![(0, 1), (2, 3), (2, 0)]);
    }

    #[test]
    fn test_contract_order_greedy_complex() {
        let tn = setup_complex();
        let mut opt = Cotengrust::new(&tn, OptMethod::Greedy);
        opt.optimize_path();

        assert_eq!(opt.get_best_flops(), 529815.);
        assert_eq!(opt.get_best_size(), 89478.);
        assert_eq!(
            opt.get_best_path(),
            path![(1, 5), (3, 4), (6, 0), (7, 2), (9, 8)]
        );
        assert_eq!(
            opt.get_best_replace_path(),
            path![(1, 5), (3, 4), (1, 0), (3, 2), (3, 1)]
        );
    }
}
