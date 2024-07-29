use std::collections::HashMap;

use itertools::Itertools;

use crate::{
    contractionpath::{
        contraction_cost::{contract_cost_tensors, contract_path_cost, contract_size_tensors},
        paths::{greedy::Greedy, CostType, OptimizePath},
    },
    pair,
    tensornetwork::{create_tensor_network, tensor::Tensor},
    types::ContractionIndex,
};

use super::ContractionTree;

/// Returns contraction cost of subtree in `contraction_tree`.
///
/// # Arguments
/// * `contraction_tree` - [`ContractionTree`] object
/// * `node_id` - root of subtree to examine
/// * `tn` - [`Tensor`] object containing bond dimension and leaf node information
///
/// # Returns
/// Total op cost and maximum memory required of fully contracting subtree rooted at `node_id`
pub fn tree_contraction_cost(
    contraction_tree: &ContractionTree,
    node_id: usize,
    tn: &Tensor,
) -> (f64, f64) {
    let (local_tensors, local_contraction_path) = subtensor_network(contraction_tree, node_id, tn);

    contract_path_cost(&local_tensors, &local_contraction_path)
}

/// Returns contraction cost of subtree in `contraction_tree` if all subtrees can be contracted in parallel..
///
/// # Arguments
/// * `contraction_tree` - [`ContractionTree`] object
/// * `node_id` - root of subtree to examine
/// * `tn` - [`Tensor`] object containing bond dimension and leaf node information
///
/// # Returns
/// Total op cost and maximum memory required of fully contracting subtree rooted at `node_id` in parallel
pub fn parallel_tree_contraction_cost(
    contraction_tree: &ContractionTree,
    node_id: usize,
    tn: &Tensor,
) -> (f64, f64, Tensor) {
    let left_child_id = contraction_tree.node(node_id).left_child_id();
    let right_child_id = contraction_tree.node(node_id).right_child_id();
    if left_child_id.is_some() && right_child_id.is_some() {
        let (left_op_cost, left_mem_cost, t1) =
            parallel_tree_contraction_cost(contraction_tree, left_child_id.unwrap(), tn);
        let (right_op_cost, right_mem_cost, t2) =
            parallel_tree_contraction_cost(contraction_tree, right_child_id.unwrap(), tn);
        let current_tensor = &t1 ^ &t2;
        let contraction_cost = contract_cost_tensors(&t1, &t2);
        let current_mem_cost = contract_size_tensors(&t1, &t2);

        (
            left_op_cost.max(right_op_cost) + contraction_cost,
            current_mem_cost.max(left_mem_cost.max(right_mem_cost)),
            current_tensor,
        )
    } else {
        let tensor_id = contraction_tree.node(node_id).tensor_index.clone().unwrap();
        let tensor = tn.nested_tensor(&tensor_id).clone();
        (0.0, tensor.size() as f64, tensor)
    }
}

/// Identifies the contraction path designated by subtree rooted at `node_id` in TN
pub(super) fn subtensor_network(
    contraction_tree: &ContractionTree,
    node_id: usize,
    tn: &Tensor,
) -> (Vec<Tensor>, Vec<ContractionIndex>) {
    let leaf_ids = contraction_tree.leaf_ids(node_id);
    let local_tensors = leaf_ids
        .iter()
        .map(|&id| tn.nested_tensor(contraction_tree.node(id).tensor_index.as_ref().unwrap()))
        .cloned()
        .collect_vec();
    let local_mapping = leaf_ids
        .iter()
        .enumerate()
        .map(|(local_idx, leaf_id)| (leaf_id, local_idx))
        .collect::<HashMap<_, _>>();

    let contraction_path = contraction_tree.to_flat_contraction_path(node_id, true);

    let local_contraction_path = contraction_path
        .into_iter()
        .map(|e| {
            if let ContractionIndex::Pair(a, b) = e {
                ContractionIndex::Pair(local_mapping[&a], local_mapping[&b])
            } else {
                panic!("No recursive path from flat contraction path!");
            }
        })
        .collect_vec();

    (local_tensors, local_contraction_path)
}

/// Generates a local contraction path for a subtree in a ContractionTree
/// One issue of generating a contraction path for a subtree is that tensor ids do not follow a strict
/// ordering. Hence, a reindexing is required to find the replace contraction path. This function can
/// return the replace contraction path if `replace` is set to true.
pub(super) fn subtree_contraction_path(
    subtree_leaf_nodes: Vec<usize>,
    tn: &Tensor,
    contraction_tree: &mut ContractionTree,
    new_max: &mut f64,
    replace: bool,
) -> (Vec<usize>, Vec<ContractionIndex>) {
    // Obtain the flattened list of Tensors corresponding to `indices`. Introduces a new indexing to find the replace contraction path.
    let (indices, tensors): (Vec<_>, Vec<_>) = subtree_leaf_nodes
        .iter()
        .map(|&e| {
            (
                e,
                tn.nested_tensor(contraction_tree.node(e).tensor_index.as_ref().unwrap())
                    .clone(),
            )
        })
        .unzip();
    // Obtain tensor network corresponding to subtree
    let tn_subtree = create_tensor_network(tensors, &tn.bond_dims(), None);

    let mut opt = Greedy::new(&tn_subtree, CostType::Flops);
    opt.optimize_path();
    if opt.get_best_flops() > *new_max {
        *new_max = opt.get_best_flops();
    }
    let path_smaller_subtree = if replace {
        opt.get_best_replace_path()
    } else {
        opt.get_best_path().clone()
    };

    let updated_smaller_path = path_smaller_subtree
        .iter()
        .map(|e| {
            if let ContractionIndex::Pair(v1, v2) = e {
                pair!(indices[*v1], indices[*v2])
            } else {
                panic!("Should only produce Pairs!")
            }
        })
        .collect_vec();
    (indices, updated_smaller_path)
}

/// Calculate partition contraction cast of all children at `rebalance_depth`
/// Returns a vector of tuples, where each tuple has the tensor_id of the child and its contraction cost.
/// tensor_id is required to identify the partition if it is sorted.
pub(super) fn calculate_partition_costs(
    contraction_tree: &ContractionTree,
    rebalance_depth: usize,
    tensor: &Tensor,
    sort: bool,
) -> Vec<(usize, f64)> {
    // Create binary contraction tree.
    let children = &contraction_tree.partitions()[&rebalance_depth];
    assert!(children.len() > 1);

    // Identify the contraction cost of each partition
    let mut partition_costs = children
        .iter()
        .map(|child| {
            (
                *child,
                tree_contraction_cost(contraction_tree, *child, tensor).0,
            )
        })
        .collect_vec();
    if sort {
        partition_costs.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
    }

    partition_costs
}
