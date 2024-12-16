use genetic_algorithm::{
    crossover::CrossoverUniform,
    fitness::{Fitness, FitnessChromosome, FitnessOrdering, FitnessValue},
    genotype::{Genotype, RangeGenotype},
    mutate::MutateSingleGene,
    select::SelectTournament,
    strategy::{evolve::Evolve, prelude::EvolveReporterDuration, Strategy},
};
use ordered_float::NotNan;

use crate::{
    contractionpath::{
        contraction_cost::{compute_memory_requirements, contract_size_tensors_exact},
        contraction_tree::{
            balancing::communication_schemes::CommunicationScheme, repartitioning::compute_solution,
        },
    },
    tensornetwork::tensor::Tensor,
};

#[derive(Clone, Debug)]
struct PartitioningFitness<'a> {
    tensor: &'a Tensor,
    communication_scheme: CommunicationScheme,
    memory_limit: Option<f64>,
}

impl PartitioningFitness<'_> {
    fn calculate_fitness(&self, partitioning: &[usize]) -> NotNan<f64> {
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
}

impl Fitness for PartitioningFitness<'_> {
    type Genotype = RangeGenotype<usize>;

    fn calculate_for_chromosome(
        &mut self,
        chromosome: &FitnessChromosome<Self>,
        _genotype: &Self::Genotype,
    ) -> Option<FitnessValue> {
        Some(self.calculate_fitness(&chromosome.genes))
    }
}

/// Balances partitions using a genetic algorithm. Finds the partitioning that reduces
/// the total contraction cost.
pub fn balance_partitions(
    tensor: &Tensor,
    num_partitions: usize,
    initial_partitioning: &[usize],
    communication_scheme: CommunicationScheme,
    memory_limit: Option<f64>,
) -> (Vec<usize>, f64) {
    // Chromosomes: Possible partitions, e.g. [0, 1, 0, 2, 2, 1, 0, 0, 1, 1]
    // Genes: tensor (in vector)
    // Alleles: partition id

    let num_tensors = initial_partitioning.len();

    let genotype = RangeGenotype::builder()
        .with_genes_size(num_tensors)
        .with_allele_range(0..=num_partitions - 1)
        .with_seed_genes_list(vec![initial_partitioning.to_vec()])
        .build()
        .unwrap();

    let fitness = PartitioningFitness {
        tensor,
        communication_scheme,
        memory_limit,
    };

    let evolve = Evolve::builder()
        .with_genotype(genotype)
        .with_target_population_size(100)
        .with_max_stale_generations(100)
        .with_fitness(fitness)
        .with_fitness_ordering(FitnessOrdering::Minimize)
        .with_mutate(MutateSingleGene::new(0.2))
        .with_crossover(CrossoverUniform::new())
        .with_select(SelectTournament::new(4, 0.9))
        .with_reporter(EvolveReporterDuration::new())
        .with_par_fitness(true)
        .with_rng_seed_from_u64(0)
        .call()
        .unwrap();

    evolve
        .best_genes_and_fitness_score()
        .map(|(partitioning, score)| (partitioning, score.into_inner()))
        .unwrap()
}

/// Calculates the fitness of a partitioning. The fitness is the total contraction
/// cost (max parallel contraction cost + communication cost).
pub fn calculate_fitness(
    tensor: &Tensor,
    partitioning: &[usize],
    communication_scheme: CommunicationScheme,
) -> f64 {
    let fitness = PartitioningFitness {
        tensor,
        communication_scheme,
        memory_limit: None,
    };

    fitness.calculate_fitness(partitioning).into_inner()
}
