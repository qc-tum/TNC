use rustc_hash::FxHashMap;

use crate::contractionpath::contraction_cost::contract_cost_tensors;
use crate::pair;
use crate::tensornetwork::partitioning::partition_config::PartitioningStrategy;

use crate::tensornetwork::partitioning::communication_partitioning;

use std::cmp::minmax;

use crate::types::ContractionIndex;

use crate::tensornetwork::tensor::Tensor;

#[derive(Debug, Copy, Clone)]
pub enum CommunicationScheme {
    /// Uses Greedy scheme to find contraction path for communication
    Greedy,
    /// Uses repeated bipartitioning to identify communication path
    Bipartition,
}

/// Uses recursive bipartitioning to identify a communication scheme for final tensors
/// Returns root id of subtree, parallel contraction cost as f64, resultant tensor and prior contraction sequence
pub(crate) fn tensor_bipartition_recursive(
    children_tensor: &[(usize, Tensor)],
    bond_dims: &FxHashMap<usize, u64>,
) -> (usize, f64, Tensor, Vec<ContractionIndex>) {
    let k = 2;
    let min = true;

    if children_tensor.len() == 1 {
        return (
            children_tensor[0].0,
            0.0,
            children_tensor[0].1.clone(),
            Vec::new(),
        );
    }
    if children_tensor.len() == 2 {
        let t1 = children_tensor[0].0;
        let t2 = children_tensor[1].0;
        let [t1, t2] = minmax(t1, t2);
        let tensor = &children_tensor[0].1 ^ &children_tensor[1].1;
        let contraction = &children_tensor[0].1 | &children_tensor[1].1;
        return (t1, contraction.size() as f64, tensor, vec![pair!(t1, t2)]);
    }

    let partitioning = communication_partitioning(
        children_tensor,
        bond_dims,
        k,
        PartitioningStrategy::MinCut,
        min,
    );

    let mut partition_iter = partitioning.iter();
    let (children_1, children_2): (Vec<_>, Vec<_>) = children_tensor
        .iter()
        .cloned()
        .partition(|_| partition_iter.next() == Some(&0));

    let (id_1, cost_1, t1, mut contraction_1) =
        tensor_bipartition_recursive(&children_1, bond_dims);

    let (id_2, cost_2, t2, mut contraction_2) =
        tensor_bipartition_recursive(&children_2, bond_dims);

    let cost = cost_1.max(cost_2) + contract_cost_tensors(&t1, &t2);
    let tensor = &t1 ^ &t2;

    contraction_1.append(&mut contraction_2);
    let [id_1, id_2] = minmax(id_1, id_2);
    contraction_1.push(pair!(id_1, id_2));
    (id_1, cost, tensor, contraction_1)
}

/// Repeatedly bipartitions tensor network to obtain communication scheme
/// Assumes that all tensors contracted do so in parallel
pub(crate) fn tensor_bipartition(
    children_tensor: &[(usize, Tensor)],
    bond_dims: &FxHashMap<usize, u64>,
) -> (f64, Vec<ContractionIndex>) {
    let (_, contraction_cost, _, contraction_path) =
        tensor_bipartition_recursive(children_tensor, bond_dims);
    (contraction_cost, contraction_path)
}
