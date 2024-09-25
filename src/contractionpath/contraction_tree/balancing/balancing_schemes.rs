use super::rebalance_node_largest_overlap;

use crate::contractionpath::contraction_tree::utils::{
    calculate_partition_costs, subtensor_network, subtree_contraction_path,
};
use crate::contractionpath::contraction_tree::ContractionTree;
use crate::types::ContractionIndex;

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
    partition_costs: &[(usize, f64)],
    contraction_tree: &mut ContractionTree,
    random_balance: bool,
    greedy_cost_function: fn(&Tensor, &Tensor) -> f64,
    tensor: &Tensor,
    rebalance_depth: usize,
) -> (f64, Vec<Tensor>, Vec<ContractionIndex>) {
    // Obtain most expensive and cheapest partitions
    let (larger_subtree_id, _) = *partition_costs.last().unwrap();
    let (smaller_subtree_id, _) = *partition_costs.first().unwrap();

    let (rebalanced_node, _) = rebalance_node_largest_overlap(
        random_balance,
        contraction_tree,
        &contraction_tree.leaf_ids(larger_subtree_id),
        smaller_subtree_id,
        greedy_cost_function,
        tensor,
    );

    // Obtain the new maximum cost other than smaller/larger subtrees for next balancing iteration
    let mut new_max = partition_costs
        .iter()
        .filter(|&&(subtree_id, _)| {
            subtree_id != smaller_subtree_id && subtree_id != larger_subtree_id
        })
        .map(|(_, cost)| *cost)
        .last()
        .unwrap_or_default();

    let (_, smaller_subtree_cost, _, larger_subtree_cost) = shift_node_between_subtrees(
        contraction_tree,
        smaller_subtree_id,
        larger_subtree_id,
        rebalance_depth,
        rebalanced_node,
        tensor,
    );

    let children = &contraction_tree.partitions()[&rebalance_depth];
    let bond_dims = tensor.bond_dims();
    // Generate new rebalanced path with updated subtree paths
    let (partition_tensors, rebalanced_path) = children
        .iter()
        .enumerate()
        .map(|(i, node_id)| {
            let (tensors, local_path) = subtensor_network(contraction_tree, *node_id, tensor);
            let mut tensor = Tensor::default();
            tensor.push_tensors(tensors, Some(&bond_dims), None);
            (tensor, ContractionIndex::Path(i, local_path))
        })
        .collect::<(Vec<_>, Vec<_>)>();

    new_max = new_max.max(smaller_subtree_cost.max(larger_subtree_cost));
    (new_max, partition_tensors, rebalanced_path)
}

/// Balancing scheme that identifies the tensor in the slowest subtree and passes it to the subtree with largest memory reduction.
/// Chosen tensor maximizes the greedy_cost_function, which is typically memory reduction.
pub(crate) fn best_tensor_balancing(
    partition_costs: &[(usize, f64)],
    contraction_tree: &mut ContractionTree,
    random_balance: bool,
    greedy_cost_function: fn(&Tensor, &Tensor) -> f64,
    tensor: &Tensor,
    rebalance_depth: usize,
) -> (f64, Vec<Tensor>, Vec<ContractionIndex>) {
    // Obtain most expensive and cheapest partitions
    let (larger_subtree_id, _) = *partition_costs.last().unwrap();

    // Find the leaf node in the smaller subtree that causes the biggest memory reduction in the bigger subtree
    let (smaller_subtree_id, rebalanced_node, _) = partition_costs
        .iter()
        .take(partition_costs.len() - 1)
        .map(|(subtree_root_id, _)| {
            let (potential_node, cost) = rebalance_node_largest_overlap(
                random_balance,
                contraction_tree,
                &contraction_tree.leaf_ids(larger_subtree_id),
                *subtree_root_id,
                greedy_cost_function,
                tensor,
            );
            (*subtree_root_id, potential_node, cost)
        })
        .max_by(|(_, _, cost_a), (_, _, cost_b)| cost_a.total_cmp(cost_b))
        .unwrap();

    // Obtain the new maximum cost other than smaller/larger subtrees for next balancing iteration
    let mut new_max = partition_costs
        .iter()
        .filter(|&&(subtree_id, _)| {
            subtree_id != smaller_subtree_id && subtree_id != larger_subtree_id
        })
        .map(|(_, cost)| *cost)
        .last()
        .unwrap_or_default();

    let (_, smaller_subtree_cost, _, larger_subtree_cost) = shift_node_between_subtrees(
        contraction_tree,
        smaller_subtree_id,
        larger_subtree_id,
        rebalance_depth,
        rebalanced_node,
        tensor,
    );

    let children = &contraction_tree.partitions()[&rebalance_depth];
    let bond_dims = tensor.bond_dims();
    // Generate new rebalanced path with updated subtree paths
    let (partition_tensors, rebalanced_path) = children
        .iter()
        .enumerate()
        .map(|(i, node_id)| {
            let (tensors, local_path) = subtensor_network(contraction_tree, *node_id, tensor);
            let mut tensor = Tensor::default();
            tensor.push_tensors(tensors, Some(&bond_dims), None);
            (tensor, ContractionIndex::Path(i, local_path))
        })
        .collect::<(Vec<_>, Vec<_>)>();

    new_max = new_max.max(smaller_subtree_cost.max(larger_subtree_cost));
    (new_max, partition_tensors, rebalanced_path)
}

/// Balancing scheme that identifies the tensor in the slowest subtree and passes it to the subtree with largest memory reduction.
/// Then identifies the tensor with the largest memory reduction when passed to the fastest subtree. Both slowest and fastest subtrees are updated.
pub(crate) fn best_tensors_balancing(
    partition_costs: &[(usize, f64)],
    contraction_tree: &mut ContractionTree,
    random_balance: bool,
    greedy_cost_function: fn(&Tensor, &Tensor) -> f64,
    tensor: &Tensor,
    rebalance_depth: usize,
) -> (f64, Vec<Tensor>, Vec<ContractionIndex>) {
    // Obtain most expensive and cheapest partitions
    let (larger_subtree_id, _) = *partition_costs.last().unwrap();

    // Find the leaf node in the smaller subtree that causes the biggest memory reduction in the bigger subtree
    let (smaller_subtree_id, rebalanced_node, _) = partition_costs
        .iter()
        .take(partition_costs.len() - 1)
        .map(|(subtree_root_id, _)| {
            let (potential_node, cost) = rebalance_node_largest_overlap(
                random_balance,
                contraction_tree,
                &contraction_tree.leaf_ids(larger_subtree_id),
                *subtree_root_id,
                greedy_cost_function,
                tensor,
            );
            (*subtree_root_id, potential_node, cost)
        })
        .max_by(|(_, _, cost_a), (_, _, cost_b)| cost_a.total_cmp(cost_b))
        .unwrap();

    // Obtain the new maximum cost other than smaller/larger subtrees for next balancing iteration
    let _new_max = partition_costs
        .iter()
        .filter(|&&(subtree_id, _)| {
            subtree_id != smaller_subtree_id && subtree_id != larger_subtree_id
        })
        .map(|(_, cost)| *cost)
        .last()
        .unwrap_or_default();

    let _ = shift_node_between_subtrees(
        contraction_tree,
        smaller_subtree_id,
        larger_subtree_id,
        rebalance_depth,
        rebalanced_node,
        tensor,
    );

    let partition_costs =
        calculate_partition_costs(contraction_tree, rebalance_depth, tensor, true);

    // Obtain most expensive and cheapest partitions
    let (smaller_subtree_id, _) = *partition_costs.first().unwrap();

    // Find the leaf node in the smaller subtree that causes the biggest memory reduction in the bigger subtree
    let (larger_subtree_id, rebalanced_node, _) = partition_costs
        .iter()
        .skip(1)
        .map(|(subtree_root_id, _)| {
            let (potential_node, cost) = rebalance_node_largest_overlap(
                random_balance,
                contraction_tree,
                &contraction_tree.leaf_ids(*subtree_root_id),
                smaller_subtree_id,
                greedy_cost_function,
                tensor,
            );
            (*subtree_root_id, potential_node, cost)
        })
        .max_by(|(_, _, cost_a), (_, _, cost_b)| cost_a.total_cmp(cost_b))
        .unwrap();

    // Obtain the new maximum cost other than smaller/larger subtrees for next balancing iteration
    let mut new_max = partition_costs
        .iter()
        .filter(|&&(subtree_id, _)| {
            subtree_id != smaller_subtree_id && subtree_id != larger_subtree_id
        })
        .map(|(_, cost)| *cost)
        .last()
        .unwrap_or_default();

    let (_, smaller_subtree_cost, _, larger_subtree_cost) = shift_node_between_subtrees(
        contraction_tree,
        smaller_subtree_id,
        larger_subtree_id,
        rebalance_depth,
        rebalanced_node,
        tensor,
    );

    let children = &contraction_tree.partitions()[&rebalance_depth];
    let bond_dims = tensor.bond_dims();
    // Generate new rebalanced path with updated subtree paths
    let (partition_tensors, rebalanced_path) = children
        .iter()
        .enumerate()
        .map(|(i, node_id)| {
            let (tensors, local_path) = subtensor_network(contraction_tree, *node_id, tensor);
            let mut tensor = Tensor::default();
            tensor.push_tensors(tensors, Some(&bond_dims), None);
            (tensor, ContractionIndex::Path(i, local_path))
        })
        .collect::<(Vec<_>, Vec<_>)>();

    new_max = new_max.max(smaller_subtree_cost.max(larger_subtree_cost));
    (new_max, partition_tensors, rebalanced_path)
}

/// Shifts `rebalance_node` from the larger subtree to the smaller subtree
/// Updates partition tensor ids after subtrees are updated and a new contraction order is found.
fn shift_node_between_subtrees(
    contraction_tree: &mut ContractionTree,
    smaller_subtree_id: usize,
    larger_subtree_id: usize,
    rebalance_depth: usize,
    rebalanced_node: usize,
    tensor: &Tensor,
) -> (usize, f64, usize, f64) {
    // Obtain parents of the two subtrees that are being updated.
    let smaller_subtree_parent_id = contraction_tree
        .node(smaller_subtree_id)
        .parent_id()
        .unwrap();
    let larger_subtree_parent_id = contraction_tree
        .node(larger_subtree_id)
        .parent_id()
        .unwrap();

    // Remove the updated subtrees from the `ContractionTree` partitions member as the intermediate tensor id will be updated.
    contraction_tree
        .partitions
        .get_mut(&rebalance_depth)
        .unwrap()
        .retain(|&e| e != smaller_subtree_id && e != larger_subtree_id);

    let mut larger_subtree_leaf_nodes = contraction_tree.leaf_ids(larger_subtree_id);
    let mut smaller_subtree_leaf_nodes = contraction_tree.leaf_ids(smaller_subtree_id);
    // Always check that a node can be moved over.
    assert!(!smaller_subtree_leaf_nodes.contains(&rebalanced_node));
    assert!(larger_subtree_leaf_nodes.contains(&rebalanced_node));

    // Remove selected tensor from bigger subtree. Add it to the smaller subtree
    smaller_subtree_leaf_nodes.push(rebalanced_node);
    larger_subtree_leaf_nodes.retain(|&leaf| leaf != rebalanced_node);

    // Run Greedy on the two updated subtrees
    let (updated_smaller_path, smaller_cost) =
        subtree_contraction_path(&smaller_subtree_leaf_nodes, tensor, contraction_tree, true);

    let (updated_larger_path, larger_cost) =
        subtree_contraction_path(&larger_subtree_leaf_nodes, tensor, contraction_tree, true);

    // Remove smaller subtree
    contraction_tree.remove_subtree(smaller_subtree_id);
    // Add new subtree, keep track of updated root id
    let new_smaller_subtree_id = contraction_tree.add_subtree(
        &updated_smaller_path,
        smaller_subtree_parent_id,
        &smaller_subtree_leaf_nodes,
    );
    // Update contraction tree partitions with new subtree root id
    contraction_tree
        .partitions
        .get_mut(&rebalance_depth)
        .unwrap()
        .push(new_smaller_subtree_id);

    // Remove larger subtree
    contraction_tree.remove_subtree(larger_subtree_id);
    // Add new subtree, keep track of updated root id
    let new_larger_subtree_id = contraction_tree.add_subtree(
        &updated_larger_path,
        larger_subtree_parent_id,
        &larger_subtree_leaf_nodes,
    );
    // Update contraction tree partitions with new subtree root id
    contraction_tree
        .partitions
        .get_mut(&rebalance_depth)
        .unwrap()
        .push(new_larger_subtree_id);

    (
        new_smaller_subtree_id,
        smaller_cost,
        new_larger_subtree_id,
        larger_cost,
    )
}
