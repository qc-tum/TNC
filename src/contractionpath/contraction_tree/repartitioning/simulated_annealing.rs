//! Adapted from <https://github.com/lucidfrontier45/localsearch>

use itertools::Itertools;
use ordered_float::NotNan;
use rand::Rng;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::{
    contractionpath::contraction_tree::{
        balancing::communication_schemes::CommunicationScheme,
        repartitioning::compute_partitioning_cost,
    },
    tensornetwork::tensor::Tensor,
};

type ScoreType = NotNan<f64>;

/// OptModel is a trait that defines requirements to be used with optimization algorithm
pub trait OptModel: Sync + Send {
    /// Type of the Solution
    type SolutionType: Clone + Sync + Send;

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
    patience: usize,
    n_trials: usize,
    max_temperature: f64,
    min_temperature: f64,
}

impl SimulatedAnnealingOptimizer {
    /// Constructor of SimulatedAnnealingOptimizer
    ///
    /// - `patience` : the optimizer will give up
    ///   if there is no improvement of the score after this number of iterations
    /// - `n_trials` : number of trial solutions to generate and evaluate at each iteration
    /// - `max_temperature` : maximum temperature
    /// - `min_temperature` : minimum temperature
    pub fn new(
        patience: usize,
        n_trials: usize,
        max_temperature: f64,
        min_temperature: f64,
    ) -> Self {
        Self {
            patience,
            n_trials,
            max_temperature,
            min_temperature,
        }
    }
}

impl SimulatedAnnealingOptimizer {
    #[allow(clippy::too_many_arguments)]
    /// Start optimization with given temperature range
    ///
    /// - `model` : the model to optimize
    /// - `initial_solution` : the initial solution to start optimization.
    /// - `n_iter`: maximum iterations
    /// - `max_temperature` : maximum temperature
    /// - `min_temperature` : minimum temperature
    fn optimize_with_temperature<M, R>(
        &self,
        model: &M,
        initial_solution: M::SolutionType,
        n_iter: usize,
        rng: &mut R,
    ) -> (M::SolutionType, ScoreType)
    where
        M: OptModel,
        R: Rng + Sized,
    {
        let mut current_score = model.evaluate(&initial_solution);
        let mut current_solution = initial_solution;
        let mut best_solution = current_solution.clone();
        let mut best_score = current_score;
        let mut temperature = self.max_temperature;
        let t_factor = (self.min_temperature / self.max_temperature).ln();
        let mut last_improvement = 0;

        for it in 0..n_iter {
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

            let ds = trial_score - current_score;
            let p = (-ds / temperature).exp();
            let r = rng.gen();

            if p > r {
                current_solution = trial_solution;
                current_score = trial_score;
            }

            if current_score < best_score {
                best_solution = current_solution.clone();
                best_score = current_score;
                last_improvement = 0;
            }

            temperature = self.max_temperature * (t_factor * (it as f64 / n_iter as f64)).exp();

            last_improvement += 1;
            if last_improvement == self.patience {
                break;
            }
        }

        (best_solution, best_score)
    }
}

struct PartitioningModel<'a> {
    tensor: &'a Tensor,
    num_partitions: usize,
    communication_scheme: CommunicationScheme,
}

impl OptModel for PartitioningModel<'_> {
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
        let cost = compute_partitioning_cost(self.tensor, partitioning, self.communication_scheme);
        NotNan::new(cost).unwrap()
    }
}

/// Runs simulated annealing to find a better partitioning.
pub fn balance_partitions<R>(
    tensor: &Tensor,
    num_partitions: usize,
    initial_partitioning: Vec<usize>,
    communication_scheme: CommunicationScheme,
    rng: &mut R,
) -> (Vec<usize>, ScoreType)
where
    R: Rng + Sized,
{
    let model = PartitioningModel {
        tensor,
        num_partitions,
        communication_scheme,
    };

    let optimizer = SimulatedAnnealingOptimizer::new(1000, 50, 1.0, 0.1);
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
