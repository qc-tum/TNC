use itertools::Itertools;
use rustc_hash::FxHashMap;

use crate::contractionpath::contraction_cost::contract_cost_tensors;
use crate::contractionpath::contraction_tree::utils::parallel_tree_contraction_cost;
use crate::contractionpath::contraction_tree::ContractionTree;
use crate::contractionpath::paths::greedy::Greedy;
use crate::contractionpath::paths::weighted_branchbound::WeightedBranchBound;
use crate::contractionpath::paths::{CostType, OptimizePath};
use crate::pair;
use crate::tensornetwork::partitioning::communication_partitioning;
use crate::tensornetwork::partitioning::partition_config::PartitioningStrategy;

use std::sync::RwLockReadGuard;

use crate::types::ContractionIndex;

use crate::tensornetwork::tensor::Tensor;

#[derive(Debug, Copy, Clone)]
pub enum CommunicationScheme {
    /// Uses Greedy scheme to find contraction path for communication
    Greedy,
    /// Uses repeated bipartitioning to identify communication path
    Bipartition,
    /// Uses a filtered search that considered time to intermediate tensor
    WeightedBranchBound,
}

pub(super) fn greedy_communication_scheme(
    children_tensors: &[Tensor],
    bond_dims: &RwLockReadGuard<FxHashMap<usize, u64>>,
) -> (f64, Vec<ContractionIndex>) {
    let mut communication_tensors = Tensor::default();
    communication_tensors.push_tensors(children_tensors.to_vec(), Some(bond_dims), None);

    let mut opt = Greedy::new(&communication_tensors, CostType::Flops);
    opt.optimize_path();
    let final_contraction = opt.get_best_replace_path();
    let contraction_tree =
        ContractionTree::from_contraction_path(&communication_tensors, &final_contraction);
    // let (final_op_cost, _) = contract_path_cost(&children_tensors, &final_contraction);
    let (final_op_cost, _, _) = parallel_tree_contraction_cost(
        &contraction_tree,
        contraction_tree.root_id().unwrap(),
        &communication_tensors,
    );
    (final_op_cost, final_contraction)
}

pub(super) fn bipartition_communication_scheme(
    children_tensors: &[Tensor],
    bond_dims: &RwLockReadGuard<FxHashMap<usize, u64>>,
) -> (f64, Vec<ContractionIndex>) {
    let children_tensors = children_tensors.iter().cloned().enumerate().collect_vec();
    let (final_op_cost, final_contraction) = tensor_bipartition(&children_tensors, bond_dims);

    (final_op_cost, final_contraction)
}

pub(super) fn weighted_branchbound_communication_scheme(
    children_tensors: &[Tensor],
    bond_dims: &RwLockReadGuard<FxHashMap<usize, u64>>,
    latency_map: FxHashMap<usize, f64>,
) -> (f64, Vec<ContractionIndex>) {
    let mut communication_tensors = Tensor::default();
    communication_tensors.push_tensors(children_tensors.to_vec(), Some(bond_dims), None);

    let mut opt = WeightedBranchBound::new(
        &communication_tensors,
        None,
        20.0,
        latency_map,
        CostType::Flops,
    );
    opt.optimize_path();
    let final_contraction = opt.get_best_replace_path();
    let contraction_tree =
        ContractionTree::from_contraction_path(&communication_tensors, &final_contraction);
    // let (final_op_cost, _) = contract_path_cost(&children_tensors, &final_contraction);
    let (final_op_cost, _, _) = parallel_tree_contraction_cost(
        &contraction_tree,
        contraction_tree.root_id().unwrap(),
        &communication_tensors,
    );
    (final_op_cost, final_contraction)
}

/// Uses recursive bipartitioning to identify a communication scheme for final tensors
/// Returns root id of subtree, parallel contraction cost as f64, resultant tensor and prior contraction sequence
pub fn tensor_bipartition_recursive(
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
        // Always ensure that the larger tensor size is on the left.
        let (t1, t2) = if children_tensor[1].1.size() > children_tensor[0].1.size() {
            (children_tensor[1].0, children_tensor[0].0)
        } else {
            (children_tensor[0].0, children_tensor[1].0)
        };
        let tensor = &children_tensor[0].1 ^ &children_tensor[1].1;
        let contraction_cost = contract_cost_tensors(&children_tensor[0].1, &children_tensor[1].1);
        return (t1, contraction_cost, tensor, vec![pair!(t1, t2)]);
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
    let (id_1, id_2) = if t2.size() > t1.size() {
        (id_2, id_1)
    } else {
        (id_1, id_2)
    };

    contraction_1.push(pair!(id_1, id_2));
    (id_1, cost, tensor, contraction_1)
}

/// Repeatedly bipartitions tensor network to obtain communication scheme
/// Assumes that all tensors contracted do so in parallel
pub fn tensor_bipartition(
    children_tensor: &[(usize, Tensor)],
    bond_dims: &FxHashMap<usize, u64>,
) -> (f64, Vec<ContractionIndex>) {
    let (_, contraction_cost, _, contraction_path) =
        tensor_bipartition_recursive(children_tensor, bond_dims);
    (contraction_cost, contraction_path)
}
