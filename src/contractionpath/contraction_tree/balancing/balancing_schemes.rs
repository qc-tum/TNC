use rand::Rng;

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

    /// Identifies the intermediate tensor in the slowest subtree and passes it to the subtree with
    /// largest memory reduction. Then identifies the intermediate tensor with the largest memory
    /// reduction when passed to the fastest subtree. Both slowest and fastest
    /// subtrees are updated. The usize parameter indicates the `height` up the contraction tree
    /// we look when passing intermediate tensors between partitions.
    /// A value of `1` allows intermediate tensors that are a product of at most 1 contraction process
    /// Using the value of `0` is then equivalent to the `Tensors` method.
    IntermediateTensors(usize),

    Configuration,
}

/// Balancing scheme that moves a tensor from the slowest subtree to the fastest subtree each time.
/// Chosen tensor maximizes the `objective_function`, which is typically memory reduction.
pub(crate) fn best_worst_balancing<R>(
    partition_data: &mut [PartitionData],
    contraction_tree: &mut ContractionTree,
    random_balance: &mut Option<(usize, R)>,
    objective_function: fn(&Tensor, &Tensor) -> f64,
    tensor: &Tensor,
) -> Vec<(usize, usize, Vec<usize>)>
where
    R: Sized + Rng,
{
    // Obtain most expensive and cheapest partitions
    let larger_subtree_id = partition_data.last().unwrap().id;
    let smaller_subtree_id = partition_data.first().unwrap().id;

    let larger_subtree_leaf_nodes =
        populate_leaf_node_tensor_map(contraction_tree, larger_subtree_id, tensor);

    let smaller_subtree_leaf_nodes =
        populate_leaf_node_tensor_map(contraction_tree, smaller_subtree_id, tensor);

    let (rebalanced_node, _) = find_rebalance_node(
        random_balance,
        &larger_subtree_leaf_nodes,
        &smaller_subtree_leaf_nodes,
        objective_function,
    );
    let rebalanced_leaf_ids = contraction_tree.leaf_ids(rebalanced_node);
    vec![(larger_subtree_id, smaller_subtree_id, rebalanced_leaf_ids)]
}

/// Balancing scheme that identifies the tensor in the slowest subtree and passes it to the subtree with largest memory reduction.
/// Chosen tensor maximizes the `objective_function`, which is typically memory reduction.
pub(crate) fn best_tensor_balancing<R>(
    partition_data: &mut [PartitionData],
    contraction_tree: &mut ContractionTree,
    random_balance: &mut Option<(usize, R)>,
    objective_function: fn(&Tensor, &Tensor) -> f64,
    tensor: &Tensor,
) -> Vec<(usize, usize, Vec<usize>)>
where
    R: Sized + Rng,
{
    // Obtain most expensive partitions
    let larger_subtree_id = partition_data.last().unwrap().id;

    let larger_subtree_leaf_nodes =
        populate_leaf_node_tensor_map(contraction_tree, larger_subtree_id, tensor);
    // Find the subtree shift that results in the largest memory savings
    let (smaller_subtree_id, rebalanced_node, _) = partition_data
        .iter()
        .take(partition_data.len() - 1)
        .map(|smaller| {
            let smaller_subtree_nodes =
                populate_subtree_tensor_map(contraction_tree, smaller.id, tensor, None);
            let (rebalanced_node, objective) = find_rebalance_node(
                random_balance,
                &larger_subtree_leaf_nodes,
                &smaller_subtree_nodes,
                objective_function,
            );
            (smaller.id, rebalanced_node, objective)
        })
        .max_by(|a, b| a.2.total_cmp(&b.2))
        .unwrap();

    let rebalanced_leaf_ids = contraction_tree.leaf_ids(rebalanced_node);
    vec![(larger_subtree_id, smaller_subtree_id, rebalanced_leaf_ids)]
}

/// Balancing scheme that identifies the tensor in the slowest subtree and passes it to the subtree with largest memory reduction.
/// Then identifies the tensor with the largest memory reduction when passed to the fastest subtree. Both slowest and fastest subtrees are updated.
pub(crate) fn best_tensors_balancing<R>(
    partition_data: &[PartitionData],
    contraction_tree: &mut ContractionTree,
    random_balance: &mut Option<(usize, R)>,
    objective_function: fn(&Tensor, &Tensor) -> f64,
    tensor: &Tensor,
) -> Vec<(usize, usize, Vec<usize>)>
where
    R: Sized + Rng,
{
    // Obtain most expensive and cheapest partitions
    let larger_subtree_id = partition_data.last().unwrap().id;

    // let mut larger_subtree_leaf_nodes = FxHashMap::default();
    let larger_subtree_leaf_nodes =
        populate_leaf_node_tensor_map(contraction_tree, larger_subtree_id, tensor);

    // Find the subtree shift that results in the largest memory savings
    let (smaller_subtree_id, rebalanced_node, _) = partition_data
        .iter()
        .take(partition_data.len() - 1)
        .map(|smaller| {
            let smaller_subtree_nodes =
                populate_subtree_tensor_map(contraction_tree, smaller.id, tensor, None);
            let (rebalanced_node, objective) = find_rebalance_node(
                random_balance,
                &larger_subtree_leaf_nodes,
                &smaller_subtree_nodes,
                objective_function,
            );
            (smaller.id, rebalanced_node, objective)
        })
        .max_by(|a, b| a.2.total_cmp(&b.2))
        .unwrap();
    let rebalanced_leaf_ids = contraction_tree.leaf_ids(rebalanced_node);
    let mut shift = vec![(larger_subtree_id, smaller_subtree_id, rebalanced_leaf_ids)];

    let smaller_subtree_id = partition_data.first().unwrap().id;

    let smaller_subtree_nodes =
        populate_subtree_tensor_map(contraction_tree, smaller_subtree_id, tensor, None);

    let (larger_subtree_id, rebalanced_node, _) = partition_data
        .iter()
        .skip(1)
        .take(partition_data.len() - 2)
        .map(|larger| {
            let larger_subtree_nodes =
                populate_leaf_node_tensor_map(contraction_tree, larger.id, tensor);
            let (rebalanced_node, objective) = find_rebalance_node(
                random_balance,
                &larger_subtree_nodes,
                &smaller_subtree_nodes,
                objective_function,
            );

            (larger.id, rebalanced_node, objective)
        })
        .max_by(|(_, _, obj_a), (_, _, obj_b)| obj_a.total_cmp(obj_b))
        .unwrap();

    let rebalanced_leaf_ids = contraction_tree.leaf_ids(rebalanced_node);
    shift.push((larger_subtree_id, smaller_subtree_id, rebalanced_leaf_ids));
    shift
}

/// Balancing scheme that identifies the tensor in the slowest subtree and passes it to the subtree with largest memory reduction.
/// Then identifies the tensor with the largest memory reduction when passed to the fastest subtree. Both slowest and fastest subtrees are updated.
pub(crate) fn best_intermediate_tensors_balancing<R>(
    partition_data: &[PartitionData],
    contraction_tree: &mut ContractionTree,
    random_balance: &mut Option<(usize, R)>,
    objective_function: fn(&Tensor, &Tensor) -> f64,
    tensor: &Tensor,
    height_limit: usize,
) -> Vec<(usize, usize, Vec<usize>)>
where
    R: Sized + Rng,
{
    // Obtain most expensive and cheapest partitions
    let larger_subtree_id = partition_data.last().unwrap().id;
    // Obtain all intermediate nodes up to height `height_limit` in larger subtree
    let larger_subtree_nodes = populate_subtree_tensor_map(
        contraction_tree,
        larger_subtree_id,
        tensor,
        Some(height_limit),
    );

    // Find the subtree shift that results in the largest memory savings
    let (smaller_subtree_id, first_rebalanced_node, _) = partition_data
        .iter()
        .take(partition_data.len() - 1)
        .map(|smaller| {
            let smaller_subtree_nodes =
                populate_subtree_tensor_map(contraction_tree, smaller.id, tensor, None);
            let (rebalanced_node, objective) = find_rebalance_node(
                random_balance,
                &larger_subtree_nodes,
                &smaller_subtree_nodes,
                objective_function,
            );
            (smaller.id, rebalanced_node, objective)
        })
        .max_by(|a, b| a.2.total_cmp(&b.2))
        .unwrap();
    let rebalanced_leaf_ids = contraction_tree.leaf_ids(first_rebalanced_node);
    let mut shift = vec![(larger_subtree_id, smaller_subtree_id, rebalanced_leaf_ids)];

    let smaller_subtree_id = partition_data.first().unwrap().id;

    let smaller_subtree_nodes =
        populate_subtree_tensor_map(contraction_tree, smaller_subtree_id, tensor, None);

    let (larger_subtree_id, second_rebalanced_node, _) = partition_data
        .iter()
        .skip(1)
        .take(partition_data.len() - 2)
        .map(|larger| {
            let larger_subtree_nodes = populate_subtree_tensor_map(
                contraction_tree,
                larger.id,
                tensor,
                Some(height_limit),
            );
            let (rebalanced_node, objective) = find_rebalance_node(
                random_balance,
                &larger_subtree_nodes,
                &smaller_subtree_nodes,
                objective_function,
            );

            (larger.id, rebalanced_node, objective)
        })
        .max_by(|(_, _, obj_a), (_, _, obj_b)| obj_a.total_cmp(obj_b))
        .unwrap();

    let rebalanced_leaf_ids = contraction_tree.leaf_ids(second_rebalanced_node);
    shift.push((larger_subtree_id, smaller_subtree_id, rebalanced_leaf_ids));
    shift
}
