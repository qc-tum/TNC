use std::iter::zip;

use itertools::Itertools;
use ordered_float::NotNan;
use rand::{rngs::StdRng, Rng, SeedableRng};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use rustc_hash::FxHashSet;

use crate::{
    contractionpath::{
        contraction_cost::{compute_memory_requirements, contract_size_tensors_exact},
        contraction_tree::{
            balancing::communication_schemes::CommunicationScheme, repartitioning::compute_solution,
        },
        paths::{
            cotengrust::{Cotengrust, OptMethod},
            OptimizePath,
        },
    },
    tensornetwork::tensor::Tensor,
    types::ContractionIndex,
};

type ScoreType = NotNan<f64>;
type EvalScoreType = (NotNan<f64>, NotNan<f64>);

/// OptModel is a trait that defines requirements to be used with optimization algorithm
pub trait OptModel<'a>: Sync + Send {
    /// Type of the Solution
    type SolutionType: Clone + Sync + Send;

    /// Generate a new trial solution from current solution
    fn generate_trial_solution<R: Rng>(
        &self,
        current_solution: Self::SolutionType,
        rng: &mut R,
    ) -> Self::SolutionType;

    /// Evaluate the score of the solution
    fn evaluate<R: Rng>(&self, solution: &Self::SolutionType, rng: &mut R) -> EvalScoreType;
}

/// Optimizer that implements the simulated annealing algorithm
#[derive(Clone, Copy)]
pub struct SimulatedAnnealingOptimizer {
    /// Number of candidate solutions to generate and evaluate in each iteration.
    n_trials: usize,
    /// Number of iterations without improvement after which the algorithm should
    /// restart from the best solution found so far.
    restart_iter: usize,
    w: f64,
}

/// Termination condition for the [`SimulatedAnnealingOptimizer`].
#[derive(Debug, Clone)]
pub enum TerminationCondition {
    Iterations {
        /// Number of iterations.
        n_iter: usize,
        /// Number of iterations without improvement after which the algorithm should
        /// terminate.
        patience: usize,
    },
    Time {
        /// Maximum time allowed for optimization.
        max_time: std::time::Duration,
    },
}

impl<'a> SimulatedAnnealingOptimizer {
    /// Start optimization with given temperature range
    ///
    /// - `model` : the model to optimize
    /// - `initial_solution` : the initial solution to start optimization.
    /// - `n_iter`: maximum iterations
    #[allow(clippy::too_many_arguments)]
    fn optimize_with_temperature<M, R>(
        &self,
        model: &M,
        initial_solution: M::SolutionType,
        termination: &TerminationCondition,
        rng: &mut R,
    ) -> (M::SolutionType, ScoreType)
    where
        M: OptModel<'a>,
        R: Rng,
    {
        let mut current_score = model.evaluate(&initial_solution, rng).0;
        let mut current_solution = initial_solution;
        let mut best_solution = current_solution.clone();
        let mut best_score = current_score;
        let mut last_improvement = 0;

        let mut rngs = (0..self.n_trials)
            .map(|_| StdRng::seed_from_u64(rng.gen()))
            .collect_vec();
        let iterations = match termination {
            TerminationCondition::Iterations { n_iter, .. } => *n_iter,
            TerminationCondition::Time { .. } => usize::MAX,
        };
        let start_time = std::time::Instant::now();
        for _ in 0..iterations {
            // Generate and evaluate candidate solutions to find the minimum objective
            let (_, trial_solution, (trial_score, _)) = rngs
                .par_iter_mut()
                .enumerate()
                .map(|(index, rng)| {
                    let trial = model.generate_trial_solution(current_solution.clone(), rng);
                    let score = model.evaluate(&trial, rng);
                    (index, trial, score)
                })
                .min_by_key(|(index, _, score)| (*score, *index))
                .unwrap();

            let diff = (trial_score - current_score) / current_score;
            let acceptance_probability = (-self.w * diff.into_inner()).exp();
            let random_value = rng.gen();

            // Accept this solution with the given acceptance probability
            if acceptance_probability >= random_value {
                current_solution = trial_solution;
                current_score = trial_score;
            }

            // Update the best solution if the current solution is better
            if current_score < best_score {
                best_solution = current_solution.clone();
                best_score = current_score;
                last_improvement = 0;
            }

            last_improvement += 1;

            // Check if we should terminate
            match termination {
                TerminationCondition::Iterations { patience, .. } => {
                    if last_improvement >= *patience {
                        break;
                    }
                }
                TerminationCondition::Time { max_time } => {
                    if start_time.elapsed() >= *max_time {
                        break;
                    }
                }
            }

            // Check if we should restart from the best solution
            if last_improvement == self.restart_iter {
                current_solution = best_solution.clone();
                current_score = best_score;
            }
        }

        (best_solution, best_score)
    }
}

/// A simulated annealing model that moves a random tensor between random partitions.
pub struct NaivePartitioningModel<'a> {
    pub tensor: &'a Tensor,
    pub num_partitions: usize,
    pub communication_scheme: CommunicationScheme,
    pub memory_limit: Option<f64>,
}

impl<'a> OptModel<'a> for NaivePartitioningModel<'a> {
    type SolutionType = Vec<usize>;

    fn generate_trial_solution<R: Rng>(
        &self,
        mut current_solution: Self::SolutionType,
        rng: &mut R,
    ) -> Self::SolutionType {
        let tensor_index = rng.gen_range(0..current_solution.len());
        let current_partition = current_solution[tensor_index];
        let new_partition = loop {
            let b = rng.gen_range(0..self.num_partitions);
            if b != current_partition {
                break b;
            }
        };
        current_solution[tensor_index] = new_partition;
        current_solution
    }

    fn evaluate<R: Rng>(&self, partitioning: &Self::SolutionType, rng: &mut R) -> EvalScoreType {
        // Construct the tensor network and contraction path from the partitioning
        let (partitioned_tn, path, parallel_cost, sum_cost) = compute_solution(
            self.tensor,
            partitioning,
            self.communication_scheme,
            Some(rng),
        );

        // Compute memory usage
        let mem = compute_memory_requirements(
            partitioned_tn.tensors(),
            &path,
            contract_size_tensors_exact,
        );

        // If the memory limit is exceeded, return infinity
        if self.memory_limit.is_some_and(|limit| mem > limit) {
            unsafe {
                (
                    NotNan::new_unchecked(f64::INFINITY),
                    NotNan::new_unchecked(f64::INFINITY),
                )
            }
        } else {
            (
                NotNan::new(parallel_cost).unwrap(),
                NotNan::new(sum_cost).unwrap(),
            )
        }
    }
}

/// A simulated annealing model that moves a random tensor to the partition that
/// maximizes memory reduction.
pub struct LeafPartitioningModel<'a> {
    pub tensor: &'a Tensor,
    pub communication_scheme: CommunicationScheme,
    pub memory_limit: Option<f64>,
}

impl<'a> OptModel<'a> for LeafPartitioningModel<'a> {
    type SolutionType = (Vec<usize>, Vec<Tensor>);

    fn generate_trial_solution<R: Rng>(
        &self,
        current_solution: Self::SolutionType,
        rng: &mut R,
    ) -> Self::SolutionType {
        let (mut partitioning, mut partition_tensors) = current_solution;
        let tensor_index = rng.gen_range(0..partitioning.len());
        let shifted_tensor = self.tensor.tensor(tensor_index);
        let source_partition = partitioning[tensor_index];

        let (new_partition, _) = partition_tensors
            .iter()
            .enumerate()
            .filter_map(|(i, partition_tensor)| {
                if i != source_partition {
                    Some((
                        i,
                        (shifted_tensor ^ partition_tensor).size() - partition_tensor.size(),
                    ))
                } else {
                    // Don't consider old partition as move target (would be a NOOP)
                    None
                }
            })
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap();

        partitioning[tensor_index] = new_partition;
        partition_tensors[source_partition] ^= shifted_tensor;
        partition_tensors[new_partition] ^= shifted_tensor;
        (partitioning, partition_tensors)
    }

    fn evaluate<R: Rng>(&self, partitioning: &Self::SolutionType, rng: &mut R) -> EvalScoreType {
        // Construct the tensor network and contraction path from the partitioning
        let (partitioned_tn, path, parallel_cost, sum_cost) = compute_solution(
            self.tensor,
            &partitioning.0,
            self.communication_scheme,
            Some(rng),
        );

        // Compute memory usage
        let mem = compute_memory_requirements(
            partitioned_tn.tensors(),
            &path,
            contract_size_tensors_exact,
        );

        // If the memory limit is exceeded, return infinity
        if self.memory_limit.is_some_and(|limit| mem > limit) {
            unsafe {
                (
                    NotNan::new_unchecked(f64::INFINITY),
                    NotNan::new_unchecked(f64::INFINITY),
                )
            }
        } else {
            (
                NotNan::new(parallel_cost).unwrap(),
                NotNan::new(sum_cost).unwrap(),
            )
        }
    }
}

/// A simulated annealing model that moves a random intermediate tensor, i.e., a
/// random number of tensors from one partition to the partition that maximizes
/// memory reduction.
pub struct IntermediatePartitioningModel<'a> {
    pub tensor: &'a Tensor,
    pub communication_scheme: CommunicationScheme,
    pub memory_limit: Option<f64>,
}

impl<'a> OptModel<'a> for IntermediatePartitioningModel<'a> {
    type SolutionType = (Vec<usize>, Vec<Tensor>, Vec<Vec<ContractionIndex>>);

    fn generate_trial_solution<R: Rng>(
        &self,
        current_solution: Self::SolutionType,
        rng: &mut R,
    ) -> Self::SolutionType {
        let (mut partitioning, mut partition_tensors, mut contraction_paths) = current_solution;

        // Select source partition (with more than one tensor)
        let viable_partitions = contraction_paths
            .iter()
            .enumerate()
            .filter_map(|(contraction_id, contraction)| {
                if contraction.len() >= 3 {
                    Some(contraction_id)
                } else {
                    None
                }
            })
            .collect_vec();

        if viable_partitions.is_empty() {
            // No viable partitions, return the current solution
            return (partitioning, partition_tensors, contraction_paths);
        }
        let trial = rng.gen_range(0..viable_partitions.len());
        let source_partition = viable_partitions[trial];

        // Select random tensor contraction in source partition
        let pair_index = rng.gen_range(0..contraction_paths[source_partition].len() - 1);
        let ContractionIndex::Pair(i, j) = contraction_paths[source_partition][pair_index] else {
            panic!("Partitioned contractions should not contain Path elements")
        };
        let mut tensor_leaves = FxHashSet::from_iter([i, j]);

        // Gather all tensors that contribute to the selected contraction
        for contraction in contraction_paths[source_partition]
            .iter()
            .take(pair_index)
            .rev()
        {
            let ContractionIndex::Pair(i, j) = contraction else {
                panic!("Expected pair")
            };
            if tensor_leaves.contains(i) {
                tensor_leaves.insert(*j);
            }
        }

        let mut shifted_tensor = Tensor::default();
        let mut shifted_indices = Vec::with_capacity(tensor_leaves.len());
        for (partition_tensor_index, (i, _partition)) in partitioning
            .iter()
            .enumerate()
            .filter(|(_, partition)| *partition == &source_partition)
            .enumerate()
        {
            if tensor_leaves.contains(&partition_tensor_index) {
                shifted_tensor ^= self.tensor.tensor(i);
                shifted_indices.push(i);
            }
        }

        // Find best target partition
        // Cost function is actually quite important!!
        let (target_partition, _) = partition_tensors
            .iter()
            .enumerate()
            .filter_map(|(i, partition_tensor)| {
                if i != source_partition {
                    Some((
                        i,
                        (&shifted_tensor ^ partition_tensor).size() - partition_tensor.size(),
                    ))
                } else {
                    // Don't consider old partition as move target (would be a NOOP)
                    None
                }
            })
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap();

        // Change partition
        for index in shifted_indices {
            partitioning[index] = target_partition;
        }

        // Recompute the tensors for both partitions
        partition_tensors[source_partition] ^= &shifted_tensor;
        partition_tensors[target_partition] ^= &shifted_tensor;

        // Recompute the contraction path for both partitions
        let mut from_tensor = Tensor::default();
        let mut to_tensor = Tensor::default();
        for (partition_index, tensor) in zip(&partitioning, self.tensor.tensors()) {
            if *partition_index == source_partition {
                from_tensor.push_tensor(tensor.clone());
            } else if *partition_index == target_partition {
                to_tensor.push_tensor(tensor.clone());
            }
        }

        let mut from_opt = Cotengrust::new(&from_tensor, OptMethod::Greedy);
        from_opt.optimize_path();
        let from_path = from_opt.get_best_replace_path();
        contraction_paths[source_partition] = from_path;

        let mut to_opt = Cotengrust::new(&to_tensor, OptMethod::Greedy);
        to_opt.optimize_path();
        let to_path = to_opt.get_best_replace_path();
        contraction_paths[target_partition] = to_path;

        (partitioning, partition_tensors, contraction_paths)
    }

    fn evaluate<R: Rng>(&self, partitioning: &Self::SolutionType, rng: &mut R) -> EvalScoreType {
        // Construct the tensor network and contraction path from the partitioning
        let (partitioned_tn, path, parallel_cost, sum_cost) = compute_solution(
            self.tensor,
            &partitioning.0,
            self.communication_scheme,
            Some(rng),
        );

        // Compute memory usage
        let mem = compute_memory_requirements(
            partitioned_tn.tensors(),
            &path,
            contract_size_tensors_exact,
        );

        // If the memory limit is exceeded, return infinity
        if self.memory_limit.is_some_and(|limit| mem > limit) {
            unsafe {
                (
                    NotNan::new_unchecked(f64::INFINITY),
                    NotNan::new_unchecked(f64::INFINITY),
                )
            }
        } else {
            (
                NotNan::new(parallel_cost).unwrap(),
                NotNan::new(sum_cost).unwrap(),
            )
        }
    }
}

/// Runs simulated annealing to find a better partitioning.
pub fn balance_partitions<'a, R, M>(
    model: M,
    initial_solution: M::SolutionType,
    rng: &mut R,
    termination_condition: &TerminationCondition,
) -> (M::SolutionType, ScoreType)
where
    R: Rng,
    M: OptModel<'a>,
{
    let optimizer = SimulatedAnnealingOptimizer {
        n_trials: 48,
        restart_iter: 100,
        w: 1.0,
    };
    optimizer.optimize_with_temperature::<M, _>(
        &model,
        initial_solution,
        termination_condition,
        rng,
    )
}
