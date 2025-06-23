use std::{
    iter::zip,
    time::{Duration, Instant},
};

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

/// Number of threads to use for processing candidate solutions in parallel. This is
/// a constant (and not hardware-aware) for reproducibility.
const PROCESSING_THREADS: usize = 48;

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
    fn evaluate<R: Rng>(&self, solution: &Self::SolutionType, rng: &mut R) -> ScoreType;
}

/// Optimizer that implements the simulated annealing algorithm
#[derive(Clone, Copy)]
pub struct SimulatedAnnealingOptimizer {
    /// Number of candidate solutions to generate and evaluate in each iteration.
    n_trials: usize,
    /// Total duration to take for the optimization
    max_time: Duration,
    /// Number of steps to take in each temperature iteration.
    n_steps: usize,
    /// Number of iterations without improvement after which the algorithm should
    /// restart from the best solution found so far.
    restart_iter: usize,
    /// The initial temperature to start the annealing process with.
    initial_temperature: f64,
    /// The final temperature to reach at the end of the annealing process.
    final_temperature: f64,
}

/// Given the `current` log temperature, the final `stop` log temperature and the
/// number of iterations that are still to be performed, returns the next log
/// temperature.
///
/// This method allows for dynamically adapting to changing number of remaining
/// iterations during the optimization.
fn next_temperature(current: f64, stop: f64, iters: u64) -> f64 {
    if iters == 0 {
        stop
    } else {
        current * (stop / current).powf(1.0 / iters as f64)
    }
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
        rng: &mut R,
    ) -> (M::SolutionType, ScoreType)
    where
        M: OptModel<'a>,
        R: Rng,
    {
        let mut current_score = model.evaluate(&initial_solution, rng);
        let mut current_solution = initial_solution;
        let mut best_solution = current_solution.clone();
        let mut best_score = current_score;
        let mut last_improvement = 0;
        let steps_per_thread = self.n_steps.div_ceil(self.n_trials);
        let end_time = Instant::now() + self.max_time;

        let mut temperature = self.initial_temperature;
        let mut rngs = (0..self.n_trials)
            .map(|_| StdRng::seed_from_u64(rng.gen()))
            .collect_vec();
        loop {
            // Time the iteration
            let iteration_start = Instant::now();

            // Generate and evaluate candidate solutions to find the minimum objective
            let (_, trial_solution, trial_score) = rngs
                .par_iter_mut()
                .enumerate()
                .map(|(index, rng)| {
                    let mut trial_score = current_score;
                    let mut trial_solution = current_solution.clone();
                    for _ in 0..steps_per_thread {
                        let solution = model.generate_trial_solution(trial_solution.clone(), rng);
                        let score = model.evaluate(&solution, rng);

                        let diff = (score / trial_score).log2();
                        let acceptance_probability = (-diff / temperature).exp();
                        let random_value = rng.gen();

                        if acceptance_probability >= random_value {
                            trial_solution = solution;
                            trial_score = score;
                        }
                    }
                    (index, trial_solution, trial_score)
                })
                .min_by_key(|(index, _, score)| (*score, *index))
                .unwrap();

            current_score = trial_score;
            current_solution = trial_solution;

            // Update the best solution if the current solution is better
            if current_score < best_score {
                best_solution = current_solution.clone();
                best_score = current_score;
                last_improvement = 0;
            }

            last_improvement += 1;

            // Check if we should restart from the best solution
            if last_improvement == self.restart_iter {
                current_solution = best_solution.clone();
                current_score = best_score;
            }

            // Estimate the number of remaining iterations and adapt the temperature
            let now = Instant::now();
            if now > end_time {
                // We've reached the time limit
                break;
            }
            let remaining_time = end_time - now;
            let iteration_time = iteration_start.elapsed();
            let remaining_iterations =
                remaining_time.as_millis_f64() / iteration_time.as_millis_f64();
            let remaining_iterations = remaining_iterations.ceil() as u64;
            temperature =
                next_temperature(temperature, self.final_temperature, remaining_iterations);
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

    fn evaluate<R: Rng>(&self, solution: &Self::SolutionType, rng: &mut R) -> ScoreType {
        // Construct the tensor network and contraction path from the partitioning
        let (partitioned_tn, path, parallel_cost, _) =
            compute_solution(self.tensor, solution, self.communication_scheme, Some(rng));

        // Compute memory usage
        let mem = compute_memory_requirements(
            partitioned_tn.tensors(),
            &path,
            contract_size_tensors_exact,
        );

        // If the memory limit is exceeded, return infinity
        if self.memory_limit.is_some_and(|limit| mem > limit) {
            unsafe { NotNan::new_unchecked(f64::INFINITY) }
        } else {
            NotNan::new(parallel_cost).unwrap()
        }
    }
}

/// A simulated annealing model that moves a random subtree between random partitions.
pub struct NaiveIntermediatePartitioningModel<'a> {
    pub tensor: &'a Tensor,
    pub num_partitions: usize,
    pub communication_scheme: CommunicationScheme,
    pub memory_limit: Option<f64>,
}

impl<'a> OptModel<'a> for NaiveIntermediatePartitioningModel<'a> {
    type SolutionType = (Vec<usize>, Vec<Vec<ContractionIndex>>);

    fn generate_trial_solution<R: Rng>(
        &self,
        current_solution: Self::SolutionType,
        rng: &mut R,
    ) -> Self::SolutionType {
        let (mut partitioning, mut contraction_paths) = current_solution;

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
            return (partitioning, contraction_paths);
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

        let mut shifted_indices = Vec::with_capacity(tensor_leaves.len());
        for (partition_tensor_index, (i, _partition)) in partitioning
            .iter()
            .enumerate()
            .filter(|(_, partition)| *partition == &source_partition)
            .enumerate()
        {
            if tensor_leaves.contains(&partition_tensor_index) {
                shifted_indices.push(i);
            }
        }

        // Select random target partition
        let target_partition = loop {
            let b = rng.gen_range(0..self.num_partitions);
            if b != source_partition {
                break b;
            }
        };

        // Change partition
        for index in shifted_indices {
            partitioning[index] = target_partition;
        }

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

        (partitioning, contraction_paths)
    }

    fn evaluate<R: Rng>(&self, solution: &Self::SolutionType, rng: &mut R) -> ScoreType {
        // Construct the tensor network and contraction path from the partitioning
        let (partitioned_tn, path, parallel_cost, _) = compute_solution(
            self.tensor,
            &solution.0,
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
            unsafe { NotNan::new_unchecked(f64::INFINITY) }
        } else {
            NotNan::new(parallel_cost).unwrap()
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

    fn evaluate<R: Rng>(&self, solution: &Self::SolutionType, rng: &mut R) -> ScoreType {
        // Construct the tensor network and contraction path from the partitioning
        let (partitioned_tn, path, parallel_cost, _) = compute_solution(
            self.tensor,
            &solution.0,
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
            unsafe { NotNan::new_unchecked(f64::INFINITY) }
        } else {
            NotNan::new(parallel_cost).unwrap()
        }
    }
}

/// A simulated annealing model that moves a random subtree to the partition that
/// maximizes memory reduction.
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

    fn evaluate<R: Rng>(&self, solution: &Self::SolutionType, rng: &mut R) -> ScoreType {
        // Construct the tensor network and contraction path from the partitioning
        let (partitioned_tn, path, parallel_cost, _) = compute_solution(
            self.tensor,
            &solution.0,
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
            unsafe { NotNan::new_unchecked(f64::INFINITY) }
        } else {
            NotNan::new(parallel_cost).unwrap()
        }
    }
}

/// Runs simulated annealing to find a better partitioning.
pub fn balance_partitions<'a, R, M>(
    model: M,
    initial_solution: M::SolutionType,
    rng: &mut R,
    max_time: Duration,
) -> (M::SolutionType, ScoreType)
where
    R: Rng,
    M: OptModel<'a>,
{
    let optimizer = SimulatedAnnealingOptimizer {
        n_trials: PROCESSING_THREADS,
        max_time,
        n_steps: PROCESSING_THREADS * 10,
        restart_iter: 50,
        initial_temperature: 2.0,
        final_temperature: 0.05,
    };
    optimizer.optimize_with_temperature::<M, _>(&model, initial_solution, rng)
}

#[cfg(test)]
mod tests {
    use super::*;
    use float_cmp::assert_approx_eq;
    use itertools::Itertools;

    #[test]
    fn test_next_temperature_without_adapting() {
        let start = 2.0f64;
        let stop = 0.05f64;
        let iters = 10;

        // Compute all values at once
        let log_start = start.log2();
        let log_stop = stop.log2();
        let step = (log_stop - log_start) / (iters - 1) as f64;
        let expected_logvals = (0..iters)
            .map(|i| 2.0f64.powf(log_start + i as f64 * step))
            .collect_vec();

        // Compute them step-wise using next_temperature
        let mut temperature = start;
        let mut logvals = Vec::new();
        for remaining in (0..iters).rev() {
            logvals.push(temperature);
            temperature = next_temperature(temperature, stop, remaining);
        }

        // Compare
        assert_eq!(logvals.len(), expected_logvals.len());
        for (actual, expected) in std::iter::zip(logvals, expected_logvals) {
            assert_approx_eq!(f64, actual, expected);
        }
    }
}
