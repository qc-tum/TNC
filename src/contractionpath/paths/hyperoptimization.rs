use std::process::{Command, Stdio};

use itertools::Itertools;
use rustc_hash::FxHashMap;
use rustengra::tensor_legs_to_digit;
use serde::Serialize;
use serde_pickle::{DeOptions, SerOptions};

use crate::{
    contractionpath::{
        contraction_cost::contract_path_cost,
        contraction_tree::repartitioning::simulated_annealing::TerminationCondition,
        ssa_replace_ordering,
    },
    tensornetwork::tensor::Tensor,
    types::ContractionIndex,
};

use super::{CostType, OptimizePath};

/// Creates an interface to `rustengra` an interface to access `Cotengra` methods in rust.
/// Specifically exposes `subtree_reconfigure` method.
pub struct Hyperoptimizer<'a> {
    tensor: &'a Tensor,
    termination_condition: TerminationCondition,
    best_flops: f64,
    best_size: f64,
    best_path: Vec<ContractionIndex>,
}

impl<'a> Hyperoptimizer<'a> {
    /// Creates a new [`TreeReconfigure`] instance. The `initial_path` is an initial
    /// contraction path in SSA format that is to be optimized. `subtree_size` is the
    /// size of subtrees that is considered (increases the optimization cost
    /// exponentially!).
    pub fn new(
        tensor: &'a Tensor,
        minimize: CostType,
        termination_condition: TerminationCondition,
    ) -> Self {
        assert_eq!(
            minimize,
            CostType::Flops,
            "Currently, only Flops is supported"
        );
        Self {
            tensor,
            termination_condition,
            best_flops: f64::INFINITY,
            best_size: f64::INFINITY,
            best_path: vec![],
        }
    }
}

/// The keyword options for the cotengra Hyperoptimizer. Unassigned options will not
/// be passed to the function and hence the Python default values will be used.
#[derive(Serialize, Default)]
struct HyperOptions {
    max_time: Option<u64>,
    max_repeats: Option<usize>,
}

/// Runs the Hyperoptimizer of cotengra on the given inputs. An optional time limit
/// can be given with `max_time`. Returns an SSA contraction path.
///
/// # Python Dependency
/// Python 3 must be installed with `cotengra` and `kahypar` packages installed.
/// Can also work with virtual environments if the binary is run from a terminal with
/// actived virtual environment.
fn python_hyperoptimizer(
    inputs: &[Vec<String>],
    outputs: &[String],
    size_dict: &FxHashMap<String, u64>,
    termination_condition: &TerminationCondition,
) -> Vec<(usize, usize)> {
    let mut options = HyperOptions::default();
    match termination_condition {
        TerminationCondition::Iterations { n_iter, .. } => {
            options.max_repeats = Some(*n_iter);
        }
        TerminationCondition::Time { max_time } => {
            options.max_time = Some(max_time.as_secs());
        }
    }

    // Spawn python process
    let mut child = Command::new("python3")
        .arg("hyperoptimization.py")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();

    // Send serialized data
    serde_pickle::to_writer(
        &mut stdin,
        &(inputs, outputs, size_dict, &options),
        SerOptions::default(),
    )
    .unwrap();

    // Wait for completion
    let out = child.wait_with_output().unwrap();

    // Get output
    let ssa_path = serde_pickle::from_slice(&out.stdout, DeOptions::default()).unwrap();
    ssa_path
}

impl OptimizePath for Hyperoptimizer<'_> {
    fn optimize_path(&mut self) {
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

        let ssa_path =
            python_hyperoptimizer(&inputs, &outputs, &size_dict, &self.termination_condition);

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

    use num_complex::Complex64;
    use rand::{rngs::StdRng, SeedableRng};
    use rustc_hash::FxHashMap;

    use crate::{
        contractionpath::{
            contraction_tree::repartitioning::simulated_annealing::TerminationCondition,
            paths::{CostType, OptimizePath},
        },
        networks::{connectivity::ConnectivityLayout, random_circuit::random_circuit},
        path,
        qasm::qasm_to_tensornetwork::create_tensornetwork,
        tensornetwork::{tensor::Tensor, tensordata::TensorData},
    };

    /// Reads a circuit from the given qasm file.
    fn read_circuit(source: &str) -> Tensor {
        let (mut tensor, open_legs) = create_tensornetwork(source);

        // Add bras to each open leg
        for leg in open_legs {
            let mut bra = Tensor::new_from_const(vec![leg], 2);
            bra.set_tensor_data(TensorData::new_from_data(
                &[2],
                vec![Complex64::ONE, Complex64::ONE],
                None,
            ));
            tensor.push_tensor(bra);
        }

        // last_values.replace((file.into(), tensor.clone()));
        tensor
    }

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

    fn setup_circuit() -> Tensor {
        let bond_dims = FxHashMap::from_iter([
            (0, 2),
            (1, 2),
            (2, 2),
            (3, 2),
            (4, 2),
            (5, 2),
            (6, 2),
            (7, 2),
            (8, 2),
            (9, 2),
            (10, 2),
            (11, 2),
            (12, 2),
            (13, 2),
            (14, 2),
            (15, 2),
            (16, 2),
            (17, 2),
            (18, 2),
            (19, 2),
        ]);

        Tensor::new_composite(vec![
            Tensor::new_from_map(vec![0], &bond_dims),
            Tensor::new_from_map(vec![1], &bond_dims),
            Tensor::new_from_map(vec![2], &bond_dims),
            Tensor::new_from_map(vec![3], &bond_dims),
            // Tensor::new_from_map(vec![4], &bond_dims),
            // Tensor::new_from_map(vec![5], &bond_dims),
            // Tensor::new_from_map(vec![6], &bond_dims),
            // Tensor::new_from_map(vec![7], &bond_dims),
            // Tensor::new_from_map(vec![8], &bond_dims),
            // Tensor::new_from_map(vec![9], &bond_dims),
            Tensor::new_from_map(vec![10, 0], &bond_dims),
            Tensor::new_from_map(vec![11, 1], &bond_dims),
            Tensor::new_from_map(vec![12, 2], &bond_dims),
            Tensor::new_from_map(vec![13, 3], &bond_dims),
            // Tensor::new_from_map(vec![14, 4], &bond_dims),
            // Tensor::new_from_map(vec![15, 5], &bond_dims),
            // Tensor::new_from_map(vec![16, 6], &bond_dims),
            // Tensor::new_from_map(vec![17, 7], &bond_dims),
            // Tensor::new_from_map(vec![18, 8], &bond_dims),
            // Tensor::new_from_map(vec![19, 9], &bond_dims),
            Tensor::new_from_map(vec![11], &bond_dims),
            // Tensor::new_from_map(vec![18], &bond_dims),
            // Tensor::new_from_map(vec![14], &bond_dims),
            // Tensor::new_from_map(vec![10], &bond_dims),
            // Tensor::new_from_map(vec![17], &bond_dims),
            Tensor::new_from_map(vec![13], &bond_dims),
            // Tensor::new_from_map(vec![16], &bond_dims),
            Tensor::new_from_map(vec![12], &bond_dims),
            // Tensor::new_from_map(vec![15], &bond_dims),
        ])
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
    fn test_hyper_tree_contract_order_simple() {
        let tn = setup_simple();
        let mut opt = Hyperoptimizer::new(
            &tn,
            CostType::Flops,
            TerminationCondition::Time {
                max_time: Duration::from_secs(25),
            },
        );
        opt.optimize_path();

        assert_eq!(opt.best_flops, 600.);
        assert_eq!(opt.best_size, 538.);
        assert_eq!(opt.get_best_path(), &path![(0, 1), (2, 3)]);
        assert_eq!(opt.get_best_replace_path(), path![(0, 1), (2, 0)]);
    }

    #[test]
    #[ignore = "HyperOptimizer is not deterministic"]
    fn test_hyper_tree_contract_order_complex() {
        let tn = setup_complex();
        let mut opt = Hyperoptimizer::new(
            &tn,
            CostType::Flops,
            TerminationCondition::Time {
                max_time: Duration::from_secs(45),
            },
        );
        opt.optimize_path();

        assert_eq!(opt.best_flops, 529815.);
        assert_eq!(opt.best_size, 89478.);
        assert_eq!(opt.best_path, path![(1, 5), (0, 6), (2, 7), (3, 4), (8, 9)]);
        assert_eq!(
            opt.get_best_replace_path(),
            path![(1, 5), (0, 1), (2, 0), (3, 4), (2, 3)]
        );
    }

    #[test]
    #[ignore = "HyperOptimizer is not deterministic"]
    fn test_hyper_tree_custom_circuit() {
        let inputs = [
            vec![String::from("0")],
            vec![String::from("1")],
            vec![String::from("2")],
            vec![String::from("3")],
            vec![String::from("4")],
            vec![String::from("5")],
            vec![String::from("6")],
            vec![String::from("7")],
            vec![String::from("8")],
            vec![String::from("9")],
            vec![String::from("10"), String::from("0")],
            vec![String::from("11"), String::from("1")],
            vec![String::from("12"), String::from("2")],
            vec![String::from("13"), String::from("3")],
            vec![String::from("14"), String::from("4")],
            vec![String::from("15"), String::from("5")],
            vec![String::from("16"), String::from("6")],
            vec![String::from("17"), String::from("7")],
            vec![String::from("18"), String::from("8")],
            vec![String::from("19"), String::from("9")],
            vec![String::from("11")],
            vec![String::from("18")],
            vec![String::from("14")],
            vec![String::from("10")],
            vec![String::from("17")],
            vec![String::from("13")],
            vec![String::from("16")],
            vec![String::from("12")],
            vec![String::from("19")],
            vec![String::from("15")],
        ];
        let outputs = vec![];

        let size_dict = FxHashMap::from_iter([
            (String::from("0"), 2),
            (String::from("1"), 2),
            (String::from("2"), 2),
            (String::from("3"), 2),
            (String::from("4"), 2),
            (String::from("5"), 2),
            (String::from("6"), 2),
            (String::from("7"), 2),
            (String::from("8"), 2),
            (String::from("9"), 2),
            (String::from("10"), 2),
            (String::from("11"), 2),
            (String::from("12"), 2),
            (String::from("13"), 2),
            (String::from("14"), 2),
            (String::from("15"), 2),
            (String::from("16"), 2),
            (String::from("17"), 2),
            (String::from("18"), 2),
            (String::from("19"), 2),
        ]);
        let ssa_path = python_hyperoptimizer(
            &inputs,
            &outputs,
            &size_dict,
            &TerminationCondition::Iterations {
                n_iter: 10,
                patience: 0,
            },
        );

        assert_eq!(
            ssa_path,
            vec![
                (0, 10),
                (23, 0),
                (2, 12),
                (27, 2),
                (23, 27),
                (8, 18),
                (21, 8),
                (23, 21),
                (3, 13),
                (25, 3),
                (6, 16),
                (26, 6),
                (25, 26),
                (23, 25),
                (1, 11),
                (20, 1),
                (4, 14),
                (22, 4),
                (20, 22),
                (9, 19),
                (28, 9),
                (20, 28),
                (5, 15),
                (29, 5),
                (7, 17),
                (24, 7),
                (29, 24),
                (20, 29),
                (23, 20)
            ]
        )
    }

    #[test]
    #[ignore = "HyperOptimizer is not deterministic"]
    fn test_hyper_tree_contract_order_circuit() {
        let circuit = r###"OPENQASM 2.0;
include "qelib1.inc";
qreg eval[9];
qreg q[1];
creg meas[10];
u2(0,-pi) eval[0];
u2(0,-pi) eval[1];
u2(0,-pi) eval[2];
u2(0,-pi) eval[3];
u2(0,-pi) eval[4];
u2(0,-pi) eval[5];
u2(0,-pi) eval[6];
u2(0,-pi) eval[7];
u2(0,-pi) eval[8];
u(237.38757580841272,0,0) q[0];"###;
        let tn = read_circuit(circuit);
        let mut opt = Hyperoptimizer::new(
            &tn,
            CostType::Flops,
            TerminationCondition::Time {
                max_time: Duration::from_secs(45),
            },
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
