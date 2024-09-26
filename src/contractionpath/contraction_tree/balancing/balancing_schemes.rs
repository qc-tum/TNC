use rustc_hash::FxHashMap;

use super::{find_rebalance_node, PartitionData};

use crate::contractionpath::contraction_tree::{
    populate_leaf_node_tensor_map, populate_subtree_tensor_map, ContractionTree,
};

use crate::tensornetwork::tensor::Tensor;

#[derive(Debug, Clone, Copy)]
pub enum BalancingScheme {
    /// Moves a tensor from the slowest subtree to the fastest subtree each time.
    BestWorst,

    /// Identifies the tensor in the slowest subtree and passes it to the subtree with
    /// largest memory reduction.
    Tensor,

    /// Identifies the tensor in the slowest subtree and passes it to the subtree with
    /// largest memory reduction. Then identifies the tensor with the largest memory
    /// reduction when passed to the fastest subtree. Both slowest and fastest
    /// subtrees are updated.
    Tensors,

    Configuration,
}

/// Balancing scheme that moves a tensor from the slowest subtree to the fastest subtree each time.
/// Chosen tensor maximizes the greedy_cost_function, which is typically memory reduction.
pub(crate) fn best_worst_balancing(
    partition_data: &mut [PartitionData],
    contraction_tree: &mut ContractionTree,
    random_balance: Option<usize>,
    greedy_cost_function: fn(&Tensor, &Tensor) -> f64,
    tensor: &Tensor,
) -> Vec<(usize, usize, Vec<usize>)> {
    // Obtain most expensive and cheapest partitions
    let PartitionData {
        id: larger_subtree_id,
        ..
    } = *partition_data.last().unwrap();
    let PartitionData {
        id: smaller_subtree_id,
        ..
    } = *partition_data.first().unwrap();

    let mut larger_subtree_nodes = FxHashMap::default();
    populate_leaf_node_tensor_map(
        contraction_tree,
        larger_subtree_id,
        &mut larger_subtree_nodes,
        tensor,
    );

    let mut smaller_subtree_nodes = FxHashMap::default();
    populate_subtree_tensor_map(
        contraction_tree,
        smaller_subtree_id,
        &mut smaller_subtree_nodes,
        tensor,
        None,
    );

    let (rebalanced_node, _) = find_rebalance_node(
        random_balance,
        &larger_subtree_nodes,
        &smaller_subtree_nodes,
        greedy_cost_function,
    );
    let rebalanced_leaf_ids = contraction_tree.leaf_ids(rebalanced_node);
    vec![(larger_subtree_id, smaller_subtree_id, rebalanced_leaf_ids)]
}

/// Balancing scheme that identifies the tensor in the slowest subtree and passes it to the subtree with largest memory reduction.
/// Chosen tensor maximizes the greedy_cost_function, which is typically memory reduction.
pub(crate) fn best_tensor_balancing(
    partition_data: &mut [PartitionData],
    contraction_tree: &mut ContractionTree,
    random_balance: Option<usize>,
    greedy_cost_function: fn(&Tensor, &Tensor) -> f64,
    tensor: &Tensor,
) -> Vec<(usize, usize, Vec<usize>)> {
    // Obtain most expensive partitions
    let PartitionData {
        id: larger_subtree_id,
        ..
    } = *partition_data.last().unwrap();

    let mut larger_subtree_nodes = FxHashMap::default();
    populate_leaf_node_tensor_map(
        contraction_tree,
        larger_subtree_id,
        &mut larger_subtree_nodes,
        tensor,
    );
    // Find the subtree shift that results in the largest memory savings
    let (smaller_subtree_id, rebalanced_node, _) = partition_data
        .iter()
        .take(partition_data.len() - 1)
        .map(
            |PartitionData {
                 id: smaller_subtree_id,
                 ..
             }| {
                let mut smaller_subtree_nodes = FxHashMap::default();
                populate_subtree_tensor_map(
                    contraction_tree,
                    *smaller_subtree_id,
                    &mut smaller_subtree_nodes,
                    tensor,
                    None,
                );
                let (rebalanced_node, cost) = find_rebalance_node(
                    random_balance,
                    &larger_subtree_nodes,
                    &smaller_subtree_nodes,
                    greedy_cost_function,
                );
                (smaller_subtree_id, rebalanced_node, cost)
            },
        )
        .max_by(|a, b| a.2.total_cmp(&b.2))
        .unwrap();

    let rebalanced_leaf_ids = contraction_tree.leaf_ids(rebalanced_node);
    vec![(larger_subtree_id, *smaller_subtree_id, rebalanced_leaf_ids)]
}

/// Balancing scheme that identifies the tensor in the slowest subtree and passes it to the subtree with largest memory reduction.
/// Then identifies the tensor with the largest memory reduction when passed to the fastest subtree. Both slowest and fastest subtrees are updated.
pub(crate) fn best_tensors_balancing(
    partition_data: &[PartitionData],
    contraction_tree: &mut ContractionTree,
    random_balance: Option<usize>,
    greedy_cost_function: fn(&Tensor, &Tensor) -> f64,
    tensor: &Tensor,
) -> Vec<(usize, usize, Vec<usize>)> {
    // Obtain most expensive and cheapest partitions
    let PartitionData {
        id: larger_subtree_id,
        ..
    } = *partition_data.last().unwrap();

    let mut larger_subtree_nodes = FxHashMap::default();
    populate_leaf_node_tensor_map(
        contraction_tree,
        larger_subtree_id,
        &mut larger_subtree_nodes,
        tensor,
    );

    // Find the subtree shift that results in the largest memory savings
    let (smaller_subtree_id, rebalanced_node, _) = partition_data
        .iter()
        .take(partition_data.len() - 1)
        .map(
            |PartitionData {
                 id: smaller_subtree_id,
                 ..
             }| {
                let mut smaller_subtree_nodes = FxHashMap::default();
                populate_subtree_tensor_map(
                    contraction_tree,
                    *smaller_subtree_id,
                    &mut smaller_subtree_nodes,
                    tensor,
                    None,
                );
                let (rebalanced_node, cost) = find_rebalance_node(
                    random_balance,
                    &larger_subtree_nodes,
                    &smaller_subtree_nodes,
                    greedy_cost_function,
                );
                (smaller_subtree_id, rebalanced_node, cost)
            },
        )
        .max_by(|a, b| a.2.total_cmp(&b.2))
        .unwrap();
    let rebalanced_leaf_ids = contraction_tree.leaf_ids(rebalanced_node);
    let mut shift = vec![(larger_subtree_id, *smaller_subtree_id, rebalanced_leaf_ids)];

    let PartitionData {
        id: smaller_subtree_id,
        ..
    } = *partition_data.first().unwrap();

    let mut smaller_subtree_nodes = FxHashMap::default();
    populate_subtree_tensor_map(
        contraction_tree,
        smaller_subtree_id,
        &mut smaller_subtree_nodes,
        tensor,
        None,
    );

    let (larger_subtree_id, rebalanced_node, _) = partition_data
        .iter()
        .skip(1)
        .map(
            |PartitionData {
                 id: larger_subtree_id,
                 ..
             }| {
                let mut larger_subtree_nodes = FxHashMap::default();
                populate_leaf_node_tensor_map(
                    contraction_tree,
                    *larger_subtree_id,
                    &mut larger_subtree_nodes,
                    tensor,
                );
                let (rebalanced_node, cost) = find_rebalance_node(
                    random_balance,
                    &larger_subtree_nodes,
                    &smaller_subtree_nodes,
                    greedy_cost_function,
                );

                (larger_subtree_id, rebalanced_node, cost)
            },
        )
        .max_by(|(_, _, cost_a), (_, _, cost_b)| cost_a.total_cmp(cost_b))
        .unwrap();

    let rebalanced_leaf_ids = contraction_tree.leaf_ids(rebalanced_node);
    shift.push((*larger_subtree_id, smaller_subtree_id, rebalanced_leaf_ids));
    shift
}

/// Balancing scheme that identifies the tensor in the slowest subtree and passes it to the subtree with largest memory reduction.
/// Then identifies the tensor with the largest memory reduction when passed to the fastest subtree. Both slowest and fastest subtrees are updated.
pub(crate) fn best_intermediate_tensors_balancing(
    partition_data: &[PartitionData],
    contraction_tree: &mut ContractionTree,
    random_balance: Option<usize>,
    greedy_cost_function: fn(&Tensor, &Tensor) -> f64,
    tensor: &Tensor,
    height_limit: usize,
) -> Vec<(usize, usize, Vec<usize>)> {
    // Obtain most expensive and cheapest partitions
    let PartitionData {
        id: larger_subtree_id,
        ..
    } = *partition_data.last().unwrap();

    let mut larger_subtree_nodes = FxHashMap::default();
    populate_subtree_tensor_map(
        contraction_tree,
        larger_subtree_id,
        &mut larger_subtree_nodes,
        tensor,
        Some(height_limit),
    );

    // Find the subtree shift that results in the largest memory savings
    let (smaller_subtree_id, first_rebalanced_node, _) = partition_data
        .iter()
        .take(partition_data.len() - 1)
        .map(
            |PartitionData {
                 id: smaller_subtree_id,
                 ..
             }| {
                let mut smaller_subtree_nodes = FxHashMap::default();
                populate_subtree_tensor_map(
                    contraction_tree,
                    *smaller_subtree_id,
                    &mut smaller_subtree_nodes,
                    tensor,
                    None,
                );
                let (rebalanced_node, cost) = find_rebalance_node(
                    random_balance,
                    &larger_subtree_nodes,
                    &smaller_subtree_nodes,
                    greedy_cost_function,
                );
                (smaller_subtree_id, rebalanced_node, cost)
            },
        )
        .max_by(|a, b| a.2.total_cmp(&b.2))
        .unwrap();
    let rebalanced_leaf_ids = contraction_tree.leaf_ids(first_rebalanced_node);
    let mut shift = vec![(larger_subtree_id, *smaller_subtree_id, rebalanced_leaf_ids)];

    let PartitionData {
        id: smaller_subtree_id,
        ..
    } = *partition_data.first().unwrap();

    let mut smaller_subtree_nodes = FxHashMap::default();
    populate_subtree_tensor_map(
        contraction_tree,
        smaller_subtree_id,
        &mut smaller_subtree_nodes,
        tensor,
        None,
    );

    let (larger_subtree_id, second_rebalanced_node, _) = partition_data
        .iter()
        .skip(1)
        .take(partition_data.len() - 2)
        .map(
            |PartitionData {
                 id: larger_subtree_id,
                 ..
             }| {
                let mut larger_subtree_nodes = FxHashMap::default();
                populate_subtree_tensor_map(
                    contraction_tree,
                    *larger_subtree_id,
                    &mut larger_subtree_nodes,
                    tensor,
                    Some(height_limit),
                );
                let (rebalanced_node, cost) = find_rebalance_node(
                    random_balance,
                    &larger_subtree_nodes,
                    &smaller_subtree_nodes,
                    greedy_cost_function,
                );

                (larger_subtree_id, rebalanced_node, cost)
            },
        )
        .max_by(|(_, _, cost_a), (_, _, cost_b)| cost_a.total_cmp(cost_b))
        .unwrap();

    let rebalanced_leaf_ids = contraction_tree.leaf_ids(second_rebalanced_node);
    shift.push((*larger_subtree_id, smaller_subtree_id, rebalanced_leaf_ids));
    shift
}
