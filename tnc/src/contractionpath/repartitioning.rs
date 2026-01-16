//! Methods for improving the partitioning of a tensor network to improve time-to-solution.

use itertools::Itertools;
use rand::Rng;
use rustc_hash::FxHashMap;

use crate::{
    contractionpath::{
        communication_schemes::CommunicationScheme,
        contraction_cost::{communication_path_op_costs, contract_path_cost},
        paths::{
            cotengrust::{Cotengrust, OptMethod},
            FindPath,
        },
        ContractionPath,
    },
    tensornetwork::{partitioning::partition_tensor_network, tensor::Tensor},
};

pub mod genetic;
pub mod simulated_annealing;

/// Given a `tensor` and a `partitioning` for it, this constructs the partitioned
/// tensor and finds a contraction path for it.
pub fn compute_solution<R>(
    tensor: &Tensor,
    partitioning: &[usize],
    communication_scheme: CommunicationScheme,
    rng: Option<&mut R>,
) -> (Tensor, ContractionPath, f64, f64)
where
    R: ?Sized + Rng,
{
    // Partition the tensor network with the proposed solution
    let partitioned_tn = partition_tensor_network(tensor.clone(), partitioning);

    // Find contraction path
    let mut greedy = Cotengrust::new(&partitioned_tn, OptMethod::Greedy);
    greedy.find_path();
    let path = greedy.get_best_replace_path();

    // Store the local paths (and costs)
    let mut latency_map =
        FxHashMap::from_iter((0..partitioned_tn.tensors().len()).map(|i| (i, 0.0)));
    for (i, local_path) in &path.nested {
        let (local_cost, _) =
            contract_path_cost(partitioned_tn.tensor(*i).tensors(), local_path, true);
        latency_map.insert(*i, local_cost);
    }

    // Find communication path separately
    let children_tensors = partitioned_tn
        .tensors()
        .iter()
        .map(Tensor::external_tensor)
        .collect_vec();
    let communication_path =
        communication_scheme.communication_path(&children_tensors, &latency_map, rng);
    let tensor_costs = (0..children_tensors.len())
        .map(|i| latency_map[&i])
        .collect_vec();
    let ((parallel_cost, sum_cost), _) = communication_path_op_costs(
        &children_tensors,
        &communication_path,
        true,
        Some(&tensor_costs),
    );

    // Add the communication path to the local paths
    let final_path = ContractionPath {
        nested: path.nested,
        toplevel: communication_path,
    };

    (partitioned_tn, final_path, parallel_cost, sum_cost)
}
