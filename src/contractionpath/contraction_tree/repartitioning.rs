use crate::{
    contractionpath::{
        contraction_cost::{communication_path_cost, contract_path_cost},
        paths::{greedy::Greedy, CostType, OptimizePath},
    },
    tensornetwork::{partitioning::partition_tensor_network, tensor::Tensor},
    types::ContractionIndex,
};
use itertools::Itertools;
use rand::{thread_rng, Rng};
use rustc_hash::FxHashMap;

use super::balancing::{communication_schemes, CommunicationScheme};

pub mod genetic;
pub mod simulated_annealing;

/// Given a `tensor` and a `partitioning` for it, this constructs the partitioned
/// tensor and finds a contraction path for it.
pub fn compute_solution<R>(
    tensor: &Tensor,
    partitioning: &[usize],
    communication_scheme: CommunicationScheme,
    rng: Option<&mut R>,
) -> (Tensor, Vec<ContractionIndex>, f64)
where
    R: ?Sized + Rng,
{
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
        if let ContractionIndex::Path(i, local_path) = p {
            let (local_cost, _) =
                contract_path_cost(partitioned_tn.tensor(i).tensors(), &local_path, true);
            latency_map.insert(i, local_cost);
            final_path.push(ContractionIndex::Path(i, local_path));
        }
    }

    // Find communication path separately
    let children_tensors = partitioned_tn
        .tensors()
        .iter()
        .map(Tensor::external_tensor)
        .collect_vec();
    let mut communication_path = {
        match communication_scheme {
            CommunicationScheme::Greedy => {
                communication_schemes::greedy(&children_tensors, &latency_map)
            }
            CommunicationScheme::RandomGreedy => {
                if let Some(rng) = rng {
                    communication_schemes::random_greedy(&children_tensors, rng)
                } else {
                    communication_schemes::random_greedy(&children_tensors, &mut thread_rng())
                }
            }
            CommunicationScheme::RandomGreedyLatency => {
                if let Some(rng) = rng {
                    communication_schemes::random_greedy_latency(
                        &children_tensors,
                        &latency_map,
                        rng,
                    )
                } else {
                    communication_schemes::random_greedy_latency(
                        &children_tensors,
                        &latency_map,
                        &mut thread_rng(),
                    )
                }
            }
            CommunicationScheme::Bipartition => {
                communication_schemes::bipartition(&children_tensors, &latency_map)
            }
            CommunicationScheme::BipartitionSweep => {
                if let Some(rng) = rng {
                    communication_schemes::bipartition_sweep(&children_tensors, &latency_map, rng)
                } else {
                    communication_schemes::bipartition_sweep(
                        &children_tensors,
                        &latency_map,
                        &mut thread_rng(),
                    )
                }
            }

            CommunicationScheme::WeightedBranchBound => {
                communication_schemes::weighted_branchbound(&children_tensors, &latency_map)
            }
            CommunicationScheme::BranchBound => {
                communication_schemes::branchbound(&children_tensors)
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
