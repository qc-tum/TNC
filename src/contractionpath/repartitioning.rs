use crate::{
    contractionpath::{
        communication_schemes::CommunicationScheme,
        contraction_cost::{communication_path_op_costs, contract_path_cost},
        paths::{
            cotengrust::{Cotengrust, OptMethod},
            OptimizePath,
        },
    },
    tensornetwork::{partitioning::partition_tensor_network, tensor::Tensor},
    types::ContractionIndex,
};
use itertools::Itertools;
use rand::Rng;
use rustc_hash::FxHashMap;

pub mod genetic;
pub mod simulated_annealing;

/// Given a `tensor` and a `partitioning` for it, this constructs the partitioned
/// tensor and finds a contraction path for it.
pub fn compute_solution<R>(
    tensor: &Tensor,
    partitioning: &[usize],
    communication_scheme: CommunicationScheme,
    rng: Option<&mut R>,
) -> (Tensor, Vec<ContractionIndex>, f64, f64)
where
    R: ?Sized + Rng,
{
    // Partition the tensor network with the proposed solution
    let partitioned_tn = partition_tensor_network(tensor.clone(), partitioning);

    // Find contraction path
    let mut greedy = Cotengrust::new(&partitioned_tn, OptMethod::Greedy);
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
    let mut communication_path =
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
    final_path.append(&mut communication_path);

    (partitioned_tn, final_path, parallel_cost, sum_cost)
}
