use crate::{
    contractionpath::{
        contraction_cost::contract_path_cost,
        paths::{greedy::Greedy, CostType, OptimizePath},
    },
    tensornetwork::{partitioning::partition_tensor_network, tensor::Tensor},
    types::ContractionIndex,
};
use itertools::Itertools;
use rustc_hash::FxHashMap;
use std::sync::Arc;

use super::balancing::{communication_schemes, CommunicationScheme};

pub mod genetic;
pub mod simulated_annealing;

/// Computes the total cost of contraction the `tensor` when partitioning it using
/// the `partitioning` list and the `communication_scheme` for finding the
/// communication path.
fn compute_partitioning_cost(
    tensor: &Tensor,
    partitioning: &[usize],
    communication_scheme: CommunicationScheme,
) -> f64 {
    // Partition the tensor network with the proposed solution
    let partitioned_tn = partition_tensor_network(tensor, partitioning);

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
    let (communication_cost, _) = match communication_scheme {
        CommunicationScheme::Greedy => {
            communication_schemes::greedy(&children_tensors, &bond_dims, &latency_map)
        }
        CommunicationScheme::Bipartition => {
            communication_schemes::bipartition(&children_tensors, &bond_dims, &latency_map)
        }
        CommunicationScheme::WeightedBranchBound => {
            communication_schemes::weighted_branchbound(&children_tensors, &bond_dims, &latency_map)
        }
    };

    communication_cost
}
