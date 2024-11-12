use cast::isize;
use genetic_algorithm::{
    crossover::CrossoverUniform,
    fitness::{Fitness, FitnessChromosome, FitnessOrdering, FitnessValue},
    genotype::{Genotype, RangeGenotype},
    mutate::MutateSingleGene,
    select::SelectTournament,
    strategy::{evolve::Evolve, prelude::EvolveReporterDuration, Strategy},
};

use crate::{
    contractionpath::{
        contraction_cost::contract_op_cost_tensors,
        contraction_tree::ContractionTree,
        paths::{greedy::Greedy, CostType, OptimizePath},
    },
    tensornetwork::{partitioning::partition_tensor_network, tensor::Tensor},
};

fn combine_tensor_index(left_tensor_index: &[usize], right_tensor_index: &[usize]) -> Vec<usize> {
    match (left_tensor_index, right_tensor_index) {
        ([p1, a], [p2, _b]) if p1 == p2 => vec![*p1, *a], // Contraction within a partition
        ([p1, _a], [_p2, _b]) => vec![*p1],               // Contraction of two partition roots
        ([p1, _a], [_b]) => vec![*p1], // Contraction of a partition root with a fan-in tensor
        ([a], [_p2, _b]) => vec![*a],  // Contraction of a fan-in tensor with a partition root
        ([a], [_b]) => vec![*a],       // Contraction of two fan-in tensors
        _ => panic!("Invalid tensor index combination"),
    }
}

fn contraction_cost(contraction_tree: &ContractionTree, tensor_network: &Tensor) -> f64 {
    let root_id = contraction_tree.root_id().unwrap();
    let (cost, _, _) = contraction_cost_recursive(contraction_tree, root_id, tensor_network);
    cost
}

fn contraction_cost_recursive(
    contraction_tree: &ContractionTree,
    node_id: usize,
    tensor_network: &Tensor,
) -> (f64, Tensor, Vec<usize>) {
    let left_child_id = contraction_tree.node(node_id).left_child_id();
    let right_child_id = contraction_tree.node(node_id).right_child_id();
    if let (Some(left_child_id), Some(right_child_id)) = (left_child_id, right_child_id) {
        let (left_op_cost, t1, left_ancestor) =
            contraction_cost_recursive(contraction_tree, left_child_id, tensor_network);
        let (right_op_cost, t2, right_ancestor) =
            contraction_cost_recursive(contraction_tree, right_child_id, tensor_network);
        let current_tensor = &t1 ^ &t2;
        let contraction_cost = contract_op_cost_tensors(&t1, &t2);
        let tensor_index = combine_tensor_index(&left_ancestor, &right_ancestor);
        let is_local_contraction = tensor_index.len() == 2;

        if is_local_contraction {
            (
                left_op_cost + right_op_cost + contraction_cost,
                current_tensor,
                tensor_index,
            )
        } else {
            (
                left_op_cost.max(right_op_cost) + contraction_cost,
                current_tensor,
                tensor_index,
            )
        }
    } else {
        let tensor_id = contraction_tree
            .node(node_id)
            .tensor_index()
            .clone()
            .unwrap();
        let tensor = tensor_network.nested_tensor(&tensor_id).clone();
        (0.0, tensor, tensor_id)
    }
}

#[derive(Clone, Debug)]
struct PartitioningFitness<'a> {
    tensor: &'a Tensor,
}

impl PartitioningFitness<'_> {
    fn calculate_fitness(&self, partitioning: &[usize]) -> isize {
        // Partition the tensor network with the proposed solution
        let partitioned_tn = partition_tensor_network(self.tensor, partitioning);

        // Find contraction path
        let mut greedy = Greedy::new(&partitioned_tn, CostType::Flops);
        greedy.optimize_path();
        let path = greedy.get_best_replace_path();

        // Calculate the parallel contraction cost
        let tree = ContractionTree::from_contraction_path(&partitioned_tn, &path);
        let total_cost = contraction_cost(&tree, &partitioned_tn);
        isize(total_cost).unwrap()
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
/// the total contraction cost (max parallel contraction cost + communication cost).
pub fn balance_partitions_genetic(
    tensor: &Tensor,
    num_partitions: usize,
    initial_partitioning: &[usize],
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

    let fitness = PartitioningFitness { tensor };

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
pub fn calculate_fitness(tensor: &Tensor, partitioning: &[usize]) -> isize {
    let fitness = PartitioningFitness { tensor };

    fitness.calculate_fitness(partitioning)
}
