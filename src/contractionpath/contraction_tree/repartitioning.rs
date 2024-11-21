use std::sync::Arc;

use cast::isize;
use genetic_algorithm::{
    crossover::CrossoverUniform,
    fitness::{Fitness, FitnessChromosome, FitnessOrdering, FitnessValue},
    genotype::{Genotype, RangeGenotype},
    mutate::MutateSingleGene,
    select::SelectTournament,
    strategy::{evolve::Evolve, prelude::EvolveReporterDuration, Strategy},
};
use itertools::Itertools;
use rustc_hash::FxHashMap;

use crate::{
    contractionpath::{
        contraction_cost::contract_path_cost,
        paths::{greedy::Greedy, CostType, OptimizePath},
    },
    tensornetwork::{partitioning::partition_tensor_network, tensor::Tensor},
    types::ContractionIndex,
};

use self::communication_schemes::CommunicationScheme;

use super::balancing::communication_schemes;

#[derive(Clone, Debug)]
struct PartitioningFitness<'a> {
    tensor: &'a Tensor,
    communication_scheme: CommunicationScheme,
}

impl PartitioningFitness<'_> {
    fn calculate_fitness(&self, partitioning: &[usize]) -> isize {
        // Partition the tensor network with the proposed solution
        let partitioned_tn = partition_tensor_network(self.tensor, partitioning);

        // Find contraction path
        let mut greedy = Greedy::new(&partitioned_tn, CostType::Flops);
        greedy.optimize_path();
        let path = greedy.get_best_replace_path();

        // Find communication path separately
        let children_tensors = partitioned_tn
            .tensors()
            .iter()
            .map(|t| Tensor::new_with_bonddims(t.external_edges(), Arc::clone(&t.bond_dims)))
            .collect_vec();
        let bond_dims = partitioned_tn.bond_dims();
        let mut latency_map = FxHashMap::default();
        for p in &path {
            if let ContractionIndex::Path(i, local) = p {
                let (local_cost, _) =
                    contract_path_cost(partitioned_tn.tensor(*i).tensors(), local, true);
                latency_map.insert(*i, local_cost);
            }
        }
        let (communication_cost, _) = match self.communication_scheme {
            CommunicationScheme::Greedy => {
                communication_schemes::greedy(&children_tensors, &bond_dims, &latency_map)
            }
            CommunicationScheme::Bipartition => {
                communication_schemes::bipartition(&children_tensors, &bond_dims, &latency_map)
            }
            CommunicationScheme::WeightedBranchBound => {
                communication_schemes::weighted_branchbound(
                    &children_tensors,
                    &bond_dims,
                    &latency_map,
                )
            }
        };

        isize(communication_cost).unwrap()
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
pub fn balance_partitions_genetic(
    tensor: &Tensor,
    num_partitions: usize,
    initial_partitioning: &[usize],
    communication_scheme: CommunicationScheme,
) -> (Vec<usize>, isize) {
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

    evolve.best_genes_and_fitness_score().unwrap()
}

/// Calculates the fitness of a partitioning. The fitness is the total contraction
/// cost (max parallel contraction cost + communication cost).
pub fn calculate_fitness(
    tensor: &Tensor,
    partitioning: &[usize],
    communication_scheme: CommunicationScheme,
) -> isize {
    let fitness = PartitioningFitness {
        tensor,
        communication_scheme,
    };

    fitness.calculate_fitness(partitioning)
}
