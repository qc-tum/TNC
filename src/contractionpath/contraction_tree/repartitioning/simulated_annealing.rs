use std::iter::zip;

use itertools::Itertools;
use ordered_float::NotNan;
use rand::Rng;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

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
        Self: std::marker::Sized;

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

        for _ in 0..n_iter {
            // Generate candidates sequentially (TODO: how to parallelize the RNG?)
            let candidates = (0..self.n_trials)
                .map(|_| model.generate_trial_solution(current_solution.clone(), rng))
                .collect_vec();

            // Evaluate candidates in parallel
            let (trial_solution, trial_score) = candidates
                .into_par_iter()
                .map(|candidate| {
                    let score = model.evaluate(&candidate);
                    (candidate, score)
                })
                .min_by_key(|(_, score)| *score)
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
        ) * 16.0;

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
        Self: std::marker::Sized,
    {
        Self {
            tensor,
            num_partitions,
            communication_scheme,
            memory_limit,
        }
    }
}

pub struct LeafPartitioningModel<'a> {
    tensor: &'a Tensor,
    num_partitions: usize,
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

        let (new_partition, _) = partition_tensors
            .iter()
            .enumerate()
            .map(|(i, partition_tensor)| {
                (
                    i,
                    (shifted_tensor ^ partition_tensor).size() - partition_tensor.size(),
                )
            })
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap();
        let old_partition = partitioning[tensor_index];
        partitioning[tensor_index] = new_partition;
        partition_tensors[old_partition] ^= shifted_tensor;
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
        ) * 16.0;

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
        Self: std::marker::Sized,
    {
        Self {
            tensor,
            num_partitions,
            communication_scheme,
            memory_limit,
        }
    }
}

pub struct IntermediatePartitioningModel<'a> {
    tensor: &'a Tensor,
    num_partitions: usize,
    communication_scheme: CommunicationScheme,
    memory_limit: Option<f64>,
}

impl<'a> IntermediatePartitioningModel<'a> {}

impl<'a> OptModel<'a> for IntermediatePartitioningModel<'a> {
    type SolutionType = (Vec<usize>, Vec<Tensor>, Vec<Vec<ContractionIndex>>);

    fn generate_trial_solution<R: Rng + Sized>(
        &self,
        current_solution: Self::SolutionType,
        rng: &mut R,
    ) -> Self::SolutionType {
        let (mut partitioning, mut partition_tensors, mut partition_contractions) =
            current_solution;
        let partition_index = loop{
            let trial_partition = rng.gen_range(0..self.num_partitions);
            if partition_contractions[trial_partition].len() > 3{
                break trial_partition;
            }
        };

        let tensor_index = rng.gen_range(0..partition_contractions[partition_index].len() - 1);
        let mut tensor_leaves = if let ContractionIndex::Pair(i, j) =
            partition_contractions[partition_index][tensor_index]
        {
            vec![i, j]
        } else {
            panic!("Partitioned contractions should not contain Path elements")
        };

        for contraction in partition_contractions[partition_index]
            .iter()
            .take(tensor_index)
            .rev()
        {
            if let ContractionIndex::Pair(i, j) = contraction {
                if tensor_leaves.contains(i) {
                    tensor_leaves.push(*j);
                }
            }
        }

        let mut shifted_tensor = Tensor::new(Vec::new());
        let mut shifted_indices = Vec::new();
        for (partition_tensor_index, (i, _partition)) in partitioning
            .iter()
            .enumerate()
            .filter(|(_, partition)| *partition == &partition_index)
            .enumerate()
        {
            if tensor_leaves.contains(&partition_tensor_index) {
                shifted_tensor ^= self.tensor.tensor(i);
                shifted_indices.push(i);
            }
        }

        // Cost function is actually quite important!!
        let (new_partition, _) = partition_tensors
            .iter()
            .enumerate()
            .map(|(i, partition_tensor)| {
                (
                    i,
                    (&shifted_tensor ^ partition_tensor).size_hint() - partition_tensor.size_hint(),
                )
            })
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap();
        let old_partition = partitioning[tensor_index];
        for index in shifted_indices {
            partitioning[index] = new_partition;
        }

        let mut from_tensor = Tensor::new(Vec::new());
        let mut to_tensor = Tensor::new(Vec::new());

        for (partition_index, tensor) in zip(&partitioning, self.tensor.tensors()) {
            if *partition_index == old_partition {
                from_tensor.push_tensor(tensor.clone(), Some(&tensor.bond_dims()));
            }
            if *partition_index == new_partition {
                to_tensor.push_tensor(tensor.clone(), Some(&tensor.bond_dims()));
            }
        }

        // Redo greedy here!
        let mut from_opt = Greedy::new(&from_tensor, CostType::Flops);
        from_opt.optimize_path();
        let from_path = from_opt.get_best_replace_path();
        partition_contractions[old_partition] = from_path;

        let mut to_opt = Greedy::new(&to_tensor, CostType::Flops);
        to_opt.optimize_path();
        let to_path = from_opt.get_best_replace_path();
        partition_contractions[new_partition] = to_path;

        partition_tensors[old_partition] ^= &shifted_tensor;
        partition_tensors[new_partition] ^= &shifted_tensor;

        (partitioning, partition_tensors, partition_contractions)
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
        ) * 16.0;

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
        Self: std::marker::Sized,
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
        patience: 1000,
        n_trials: 50,
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
