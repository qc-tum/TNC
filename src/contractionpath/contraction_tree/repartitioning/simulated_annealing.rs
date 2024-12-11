use std::sync::Arc;

use itertools::Itertools;
use ordered_float::NotNan;
use rand::Rng;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::{
    contractionpath::{
        contraction_cost::contract_path_cost,
        contraction_tree::{
            balancing::communication_schemes::CommunicationScheme,
            repartitioning::{compute_partitioning_cost, compute_solution},
        },
    },
    tensornetwork::tensor::Tensor,
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
        initial_partitioning: Self::SolutionType,
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

impl SimulatedAnnealingOptimizer {
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

pub struct PartitioningModel<'a> {
    tensor: &'a Tensor,
    num_partitions: usize,
    communication_scheme: CommunicationScheme,
    memory_limit: Option<f64>,
}

impl<'a> OptModel<'a> for PartitioningModel<'a> {
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
        _partitioning_scheme: Self::SolutionType,
    ) -> Self
    where
        Self: std::marker::Sized,
    {
        Self {
            tensor,
            num_partitions,
            communication_scheme,
        }
    }
}

pub struct DirectedPartitioningModel<'a> {
    tensor: &'a Tensor,
    num_partitions: usize,
    communication_scheme: CommunicationScheme,
    partition_tensors: Vec<Tensor>,
}

impl<'a> DirectedPartitioningModel<'a> {
    pub fn new(
        tensor: &'a Tensor,
        num_partitions: usize,
        initial_partitioning: Vec<usize>,
        communication_scheme: CommunicationScheme,
    ) -> Self {
        let mut partition_tensors = [Tensor::new_with_bonddims(
            Vec::new(),
            Arc::clone(&tensor.bond_dims),
        )];
        for (tensor_index, partition_index) in initial_partitioning.iter().enumerate() {
            partition_tensors[*partition_index] ^= tensor.tensors()[tensor_index].clone();
        }
        Self {
            tensor,
            num_partitions,
            communication_scheme,
            partition_tensors: partition_tensors.to_vec(),
        }
    }
}

impl<'a> OptModel<'a> for DirectedPartitioningModel<'a> {
    type SolutionType = Vec<usize>;

    fn generate_trial_solution<R: Rng + Sized>(
        &self,
        mut current_solution: Self::SolutionType,
        rng: &mut R,
    ) -> Self::SolutionType {
        let tensor_index = rng.gen_range(0..current_solution.len());
        // let current_partition = current_solution[tensor_index];
        let random_tensor = self.tensor.tensor(tensor_index);
        let (new_partition, _) = self
            .partition_tensors
            .iter()
            .enumerate()
            .map(|(i, tensor)| {
                (
                    i,
                    (random_tensor ^ tensor).size() as i64
                        - random_tensor.size() as i64
                        - tensor.size() as i64,
                )
            })
            .min_by(|a, b| a.1.cmp(&b.1))
            .unwrap();

        current_solution[tensor_index] = new_partition;
        current_solution
    }

    fn evaluate(&self, partitioning: &Self::SolutionType) -> ScoreType {
        let cost = compute_partitioning_cost(self.tensor, partitioning, self.communication_scheme);
        NotNan::new(cost).unwrap()
    }

    fn new(
        tensor: &'a Tensor,
        num_partitions: usize,
        communication_scheme: CommunicationScheme,
        initial_partitioning: Self::SolutionType,
    ) -> Self
    where
        Self: std::marker::Sized,
    {
        let mut partition_tensors =
            vec![
                Tensor::new_with_bonddims(Vec::new(), Arc::clone(&tensor.bond_dims));
                num_partitions
            ];
        for (i, partition_num) in initial_partitioning.iter().enumerate() {
            partition_tensors[*partition_num] ^= tensor.tensor(i).clone();
        }
        Self {
            tensor,
            num_partitions,
            communication_scheme,
            partition_tensors,
        }
    }
}

/// Runs simulated annealing to find a better partitioning.
pub fn balance_partitions<'a, R, M>(
    tensor_network: &'a Tensor,
    num_partitions: usize,
    initial_partitioning: M::SolutionType,
    communication_scheme: CommunicationScheme,
    rng: &mut R,
    memory_limit: Option<f64>,
) -> (Vec<usize>, ScoreType)
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
    optimizer.optimize_with_temperature(&model, initial_partitioning, 1000, rng)
}

/// Computes the score of a partitioning.
pub fn calculate_score(
    tensor: &Tensor,
    partitioning: &[usize],
    communication_scheme: CommunicationScheme,
) -> f64 {
    compute_partitioning_cost(tensor, partitioning, communication_scheme)
}
