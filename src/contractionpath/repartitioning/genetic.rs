use genetic_algorithm::{
    crossover::CrossoverUniform,
    fitness::{Fitness, FitnessChromosome, FitnessOrdering, FitnessValue},
    genotype::{Genotype, RangeGenotype},
    mutate::MutateSingleGene,
    select::SelectTournament,
    strategy::{evolve::Evolve, Strategy},
};
use ordered_float::NotNan;
use rand::rngs::StdRng;

use crate::{
    contractionpath::{
        communication_schemes::CommunicationScheme,
        contraction_cost::{compute_memory_requirements, contract_size_tensors_exact},
        repartitioning::compute_solution,
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
        let (partitioned_tn, path, cost, _) =
            compute_solution::<StdRng>(self.tensor, partitioning, self.communication_scheme, None);

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
        .with_crossover(CrossoverUniform::new(1.0, 1.0))
        .with_select(SelectTournament::new(1.0, 0.02, 4))
        // .with_reporter(EvolveReporterDuration::new())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_partitioning() {
        let t1 = Tensor::new_from_const(vec![0, 1], 2);
        let t2 = Tensor::new_from_const(vec![2, 3], 2);
        let t3 = Tensor::new_from_const(vec![0, 1, 4], 2);
        let t4 = Tensor::new_from_const(vec![2, 3, 4], 2);
        let tn = Tensor::new_composite(vec![t1, t2, t3, t4]);
        let initial_partitioning = vec![0, 0, 1, 1];

        let (partitioning, _) = balance_partitions(
            &tn,
            2,
            &initial_partitioning,
            CommunicationScheme::RandomGreedy,
            None,
        );
        // Normalize for comparability
        let ref_partitioning = if partitioning[0] == 0 {
            [0, 1, 0, 1]
        } else {
            [1, 0, 1, 0]
        };
        assert_eq!(partitioning, ref_partitioning);
    }
}
