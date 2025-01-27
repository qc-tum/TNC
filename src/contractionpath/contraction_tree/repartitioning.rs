use crate::{
    contractionpath::{
        contraction_cost::{communication_path_cost, contract_path_cost},
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

/// Given a `tensor` and a `partitioning` for it, this constructs the partitioned
/// tensor and finds a contraction path for it.
pub fn compute_solution(
    tensor: &Tensor,
    partitioning: &[usize],
    communication_scheme: CommunicationScheme,
) -> (Tensor, Vec<ContractionIndex>, f64) {
    // Partition the tensor network with the proposed solution
    let partitioned_tn = partition_tensor_network(tensor.clone(), partitioning);

    // Find contraction path
    let mut greedy = Greedy::new(&partitioned_tn, CostType::Flops);
    greedy.optimize_path();
    let path = greedy.get_best_replace_path();

    // Store the local paths (and costs)
    let mut latency_map =
        FxHashMap::from_iter((0..partitioned_tn.tensors().len()).map(|i| (i, 0.0)));
    let mut final_path = Vec::with_capacity(tensor.tensors().len());
    for p in path {
        if let ContractionIndex::Path(i, slicing, local_path) = p {
            let (local_cost, _) =
                contract_path_cost(partitioned_tn.tensor(i).tensors(), &local_path, true);
            latency_map.insert(i, local_cost);
            final_path.push(ContractionIndex::Path(i, slicing, local_path));
        }
    }

    // Find communication path separately
    let children_tensors = partitioned_tn
        .tensors()
        .iter()
        .map(|t| Tensor::new_with_bonddims(t.external_edges(), Arc::clone(&t.bond_dims)))
        .collect_vec();
    let mut communication_path = {
        let bond_dims = partitioned_tn.bond_dims();
        match communication_scheme {
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
        }
    };
    let tensor_costs = (0..children_tensors.len())
        .map(|i| latency_map[&i])
        .collect_vec();
    let (communication_cost, _) = communication_path_cost(
        &children_tensors,
        &communication_path,
        true,
        Some(&tensor_costs),
    );

    // Add the communication path to the local paths
    final_path.append(&mut communication_path);

    (partitioned_tn, final_path, communication_cost)
}

/// Computes the total cost of contraction the `tensor` when partitioning it using
/// the `partitioning` list and the `communication_scheme` for finding the
/// communication path.
#[inline]
fn compute_partitioning_cost(
    tensor: &Tensor,
    partitioning: &[usize],
    communication_scheme: CommunicationScheme,
) -> f64 {
    compute_solution(tensor, partitioning, communication_scheme).2
}
