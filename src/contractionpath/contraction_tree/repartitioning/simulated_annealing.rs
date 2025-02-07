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
            balancing::communication_schemes::CommunicationScheme,
            repartitioning::{compute_partitioning_cost, compute_solution},
        },
        paths::{greedy::Greedy, CostType, OptimizePath},
    },
    tensornetwork::tensor::Tensor,
    types::ContractionIndex,
};

type ScoreType = NotNan<f64>;

/// OptModel is a trait that defines requirements to be used with optimization algorithm
pub trait OptModel<'a>: Sync + Send {
    /// Type of the Solution
    type SolutionType: Clone + Sync + Send;

    fn new(
        tensor: &'a Tensor,
        num_partitions: usize,
        communication_scheme: CommunicationScheme,
        memory_limit: Option<f64>,
    ) -> Self
    where
        Self: Sized;

    /// Generate a new trial solution from current solution
    fn generate_trial_solution<R: Rng + Sized>(
        &self,
        current_solution: Self::SolutionType,
        rng: &mut R,
    ) -> Self::SolutionType;

    /// Evaluate the score of the solution
    fn evaluate(&self, solution: &Self::SolutionType) -> ScoreType;
}

/// Optimizer that implements the simulated annealing algorithm
#[derive(Clone, Copy)]
pub struct SimulatedAnnealingOptimizer {
    n_trials: usize,
    patience: usize,
    restart_iter: usize,
    w: f64,
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
        n_iter: usize,
        rng: &mut R,
    ) -> (M::SolutionType, ScoreType)
    where
        M: OptModel<'a>,
        R: Rng + Sized,
    {
        let mut current_score = model.evaluate(&initial_solution);
        let mut current_solution = initial_solution;
        let mut best_solution = current_solution.clone();
        let mut best_score = current_score;
        let mut last_improvement = 0;

        let mut rngs = (0..self.n_trials)
            .map(|_| StdRng::seed_from_u64(rng.gen()))
            .collect_vec();
        for _ in 0..n_iter {
            // Generate and evaluate candidate solutions to find the minimum objective
            let (_, trial_solution, trial_score) = rngs
                .par_iter_mut()
                .enumerate()
                .map(|(index, rng)| {
                    let trial = model.generate_trial_solution(current_solution.clone(), rng);
                    let score = model.evaluate(&trial);
                    (index, trial, score)
                })
                .min_by_key(|(index, _, score)| (*score, *index))
                .unwrap();

            let diff = (trial_score - current_score) / current_score;
            let acceptance_probability = (-self.w * diff.into_inner()).exp();
            let random_value = rng.gen();

            if acceptance_probability >= random_value {
                current_solution = trial_solution;
                current_score = trial_score;
            }

            if current_score < best_score {
                best_solution = current_solution.clone();
                best_score = current_score;
                last_improvement = 0;
            }

            last_improvement += 1;

            if last_improvement == self.restart_iter {
                current_solution = best_solution.clone();
                current_score = best_score;
            }

            if last_improvement == self.patience {
                break;
            }
        }

        (best_solution, best_score)
    }
}

/// A simulated annealing model that moves a random tensor between random partitions.
pub struct NaivePartitioningModel<'a> {
    tensor: &'a Tensor,
    num_partitions: usize,
    communication_scheme: CommunicationScheme,
    memory_limit: Option<f64>,
}

impl<'a> OptModel<'a> for NaivePartitioningModel<'a> {
    type SolutionType = Vec<usize>;

    fn generate_trial_solution<R: Rng + Sized>(
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

    fn evaluate(&self, partitioning: &Self::SolutionType) -> ScoreType {
        // Construct the tensor network and contraction path from the partitioning
        let (partitioned_tn, path, cost) =
            compute_solution(self.tensor, partitioning, self.communication_scheme);

        // Compute memory usage
        let mem = compute_memory_requirements(
            partitioned_tn.tensors(),
            &path,
            contract_size_tensors_exact,
        );

        // If the memory limit is exceeded, return infinity
        let score = if self.memory_limit.is_some_and(|limit| mem > limit) {
            f64::INFINITY
        } else {
            cost
        };
        NotNan::new(score).unwrap()
    }

    fn new(
        tensor: &'a Tensor,
        num_partitions: usize,
        communication_scheme: CommunicationScheme,
        memory_limit: Option<f64>,
    ) -> Self
    where
        Self: Sized,
    {
        Self {
            tensor,
            num_partitions,
            communication_scheme,
            memory_limit,
        }
    }
}

/// A simulated annealing model that moves a random tensor to the partition that
/// maximizes memory reduction.
pub struct LeafPartitioningModel<'a> {
    tensor: &'a Tensor,
    communication_scheme: CommunicationScheme,
    memory_limit: Option<f64>,
}

impl<'a> OptModel<'a> for LeafPartitioningModel<'a> {
    type SolutionType = (Vec<usize>, Vec<Tensor>);

    fn generate_trial_solution<R: Rng + Sized>(
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

    fn evaluate(&self, partitioning: &Self::SolutionType) -> ScoreType {
        // Construct the tensor network and contraction path from the partitioning
        let (partitioned_tn, path, cost) =
            compute_solution(self.tensor, &partitioning.0, self.communication_scheme);

        // Compute memory usage
        let mem = compute_memory_requirements(
            partitioned_tn.tensors(),
            &path,
            contract_size_tensors_exact,
        );

        // If the memory limit is exceeded, return infinity
        let score = if self
            .memory_limit
            .map(|limit| mem > limit)
            .unwrap_or_default()
        {
            f64::INFINITY
        } else {
            cost
        };
        NotNan::new(score).unwrap()
    }

    fn new(
        tensor: &'a Tensor,
        _num_partitions: usize,
        communication_scheme: CommunicationScheme,
        memory_limit: Option<f64>,
    ) -> Self
    where
        Self: Sized,
    {
        Self {
            tensor,
            communication_scheme,
            memory_limit,
        }
    }
}

/// A simulated annealing model that moves a random intermediate tensor, i.e., a
/// random number of tensors from one partition to the partition that maximizes
/// memory reduction.
pub struct IntermediatePartitioningModel<'a> {
    tensor: &'a Tensor,
    num_partitions: usize,
    communication_scheme: CommunicationScheme,
    memory_limit: Option<f64>,
}

impl<'a> OptModel<'a> for IntermediatePartitioningModel<'a> {
    type SolutionType = (Vec<usize>, Vec<Tensor>, Vec<Vec<ContractionIndex>>);

    fn generate_trial_solution<R: Rng + Sized>(
        &self,
        current_solution: Self::SolutionType,
        rng: &mut R,
    ) -> Self::SolutionType {
        let (mut partitioning, mut partition_tensors, mut contraction_paths) = current_solution;

        // Select source partition (with more than one tensor)
        let source_partition = loop {
            let trial_partition = rng.gen_range(0..self.num_partitions);
            if contraction_paths[trial_partition].len() > 3 {
                break trial_partition;
            }
        };

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

        let mut from_opt = Greedy::new(&from_tensor, CostType::Flops);
        from_opt.optimize_path();
        let from_path = from_opt.get_best_replace_path();
        contraction_paths[source_partition] = from_path;

        let mut to_opt = Greedy::new(&to_tensor, CostType::Flops);
        to_opt.optimize_path();
        let to_path = to_opt.get_best_replace_path();
        contraction_paths[target_partition] = to_path;

        (partitioning, partition_tensors, contraction_paths)
    }

    fn evaluate(&self, partitioning: &Self::SolutionType) -> ScoreType {
        // Construct the tensor network and contraction path from the partitioning
        let (partitioned_tn, path, cost) =
            compute_solution(self.tensor, &partitioning.0, self.communication_scheme);

        // Compute memory usage
        let mem = compute_memory_requirements(
            partitioned_tn.tensors(),
            &path,
            contract_size_tensors_exact,
        );

        // If the memory limit is exceeded, return infinity
        let score = if self
            .memory_limit
            .map(|limit| mem > limit)
            .unwrap_or_default()
        {
            f64::INFINITY
        } else {
            cost
        };
        NotNan::new(score).unwrap()
    }

    fn new(
        tensor: &'a Tensor,
        num_partitions: usize,
        communication_scheme: CommunicationScheme,
        memory_limit: Option<f64>,
    ) -> Self
    where
        Self: Sized,
    {
        Self {
            tensor,
            num_partitions,
            communication_scheme,
            memory_limit,
        }
    }
}

/// Runs simulated annealing to find a better partitioning.
pub fn balance_partitions<'a, R, M>(
    tensor_network: &'a Tensor,
    num_partitions: usize,
    initial_solution: M::SolutionType,
    communication_scheme: CommunicationScheme,
    rng: &mut R,
    memory_limit: Option<f64>,
) -> (M::SolutionType, ScoreType)
where
    R: Rng + Sized,
    M: OptModel<'a>,
{
    let model = M::new(
        tensor_network,
        num_partitions,
        communication_scheme,
        memory_limit,
    );

    let optimizer = SimulatedAnnealingOptimizer {
        patience: 300,
        n_trials: 48,
        restart_iter: 100,
        w: 1.0,
    };
    optimizer.optimize_with_temperature::<M, _>(&model, initial_solution, 1000, rng)
}

/// Computes the score of a partitioning.
pub fn calculate_score(
    tensor: &Tensor,
    partitioning: &[usize],
    communication_scheme: CommunicationScheme,
) -> f64 {
    compute_partitioning_cost(tensor, partitioning, communication_scheme)
}
