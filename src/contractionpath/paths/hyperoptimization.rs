use std::{
    iter::zip,
    process::{Command, Stdio},
    time::Duration,
};

use itertools::Itertools;
use rustc_hash::FxHashMap;
use serde::Serialize;
use serde_pickle::{DeOptions, SerOptions};

use crate::{
    contractionpath::{
        contraction_cost::contract_path_cost,
        paths::{CostType, OptimizePath},
        ssa_replace_ordering,
    },
    tensornetwork::tensor::Tensor,
    types::ContractionIndex,
};

/// Creates an interface to access `Cotengra` methods in Rust. Specifically exposes
/// `search` method of `HyperOptimizer`.
pub struct Hyperoptimizer<'a> {
    tensor: &'a Tensor,
    hyper_options: HyperOptions,
    best_flops: f64,
    best_size: f64,
    best_path: Vec<ContractionIndex>,
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
            best_flops: f64::INFINITY,
            best_size: f64::INFINITY,
            best_path: vec![],
        }
    }
}

/// The keyword options for the cotengra Hyperoptimizer. Unassigned options will not
/// be passed to the function and hence the Python default values will be used.
/// Please see the cotengra documentation for details on the parameters.
#[derive(Serialize, Default)]
pub struct HyperOptions {
    max_time: Option<u64>,
    max_repeats: Option<usize>,
}

impl HyperOptions {
    /// Creates the default HyperOptimizer options.
    pub fn new() -> Self {
        HyperOptions::default()
    }

    /// Sets the `max_time` argument for the HyperOptimizer.
    pub fn with_max_time(mut self, time: &Duration) -> Self {
        self.max_time = Some(time.as_secs());
        self
    }

    /// Sets the `max_repeats` argument for the HyperOptimizer.
    pub fn with_max_repeats(mut self, repeats: usize) -> Self {
        self.max_repeats = Some(repeats);
        self
    }
}

/// Runs the Hyperoptimizer of cotengra on the given inputs. Additional inputs to the
/// Hyperoptimizer can be passed with the [`HyperOptions`] struct.
///
/// # Python Dependency
/// Python 3 must be installed with `cotengra` and `kahypar` packages installed.
/// Can also work with virtual environments if the binary is run from a terminal with
/// actived virtual environment.
fn python_hyperoptimizer(
    inputs: &[Vec<char>],
    outputs: &[char],
    size_dict: &FxHashMap<char, f32>,
    hyper_options: &HyperOptions,
) -> Vec<(usize, usize)> {
    // Python code to be executed (WARNING: command line length limits might silently
    // truncate the code! These are usually around >100,000 characters. Make sure the
    // code is not too long.)
    const PYTHON_CODE: &str = include_str!("hyperoptimization.py");

    // Spawn python process
    let mut child = Command::new("python3")
        .arg("-c")
        .arg(PYTHON_CODE)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();

    // Send serialized data
    serde_pickle::to_writer(
        &mut stdin,
        &(inputs, outputs, size_dict, hyper_options),
        SerOptions::default(),
    )
    .unwrap();

    // Wait for completion
    let out = child.wait_with_output().unwrap();

    // Deserialize SSA path
    serde_pickle::from_slice(&out.stdout, DeOptions::default()).unwrap()
}

/// Converts tensor leg inputs to chars. Creates new inputs, outputs and size_dict that can be fed to Cotengra.
fn tensor_legs_to_chars(
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

impl OptimizePath for Hyperoptimizer<'_> {
    fn optimize_path(&mut self) {
        let (inputs, outputs, size_dict) =
            tensor_legs_to_chars(self.tensor.tensors(), &self.tensor.external_tensor());

        let ssa_path = python_hyperoptimizer(&inputs, &outputs, &size_dict, &self.hyper_options);

        self.best_path = ssa_path
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
    use super::*;

    use std::time::Duration;

    use rustc_hash::FxHashMap;

    use crate::{
        contractionpath::paths::{CostType, OptimizePath},
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
            HyperOptions::new().with_max_time(&Duration::from_secs(25)),
        );
        opt.optimize_path();

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
            HyperOptions::new().with_max_time(&Duration::from_secs(45)),
        );
        opt.optimize_path();

        assert_eq!(opt.best_flops, 529815.);
        assert_eq!(opt.best_size, 89478.);
        assert_eq!(opt.best_path, path![(1, 5), (0, 6), (3, 4), (2, 8), (7, 9)]);
        assert_eq!(
            opt.get_best_replace_path(),
            path![(1, 5), (0, 1), (3, 4), (2, 3), (0, 2)]
        );
    }
}
