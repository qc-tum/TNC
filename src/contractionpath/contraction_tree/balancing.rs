use std::{cmp::minmax, collections::HashMap};

use itertools::Itertools;
use rand::{
    distributions::{Distribution, WeightedIndex},
    thread_rng,
};

use crate::{
    contractionpath::{
        contraction_cost::contract_cost_tensors,
        contraction_tree::utils::{subtensor_network, subtree_contraction_path},
    },
    pair,
    tensornetwork::{partitioning::communication_partitioning, tensor::Tensor},
    types::ContractionIndex,
};

use super::{populate_subtree_tensor_map, ContractionTree};

/// Uses recursive bipartitioning to identify a communication scheme for final tensors
/// Returns root id of subtree, parallel contraction cost as f64, resultant tensor and prior contraction sequence
fn tensor_bipartition_recursive(
    children_tensor: &[(usize, Tensor)],
    bond_dims: &HashMap<usize, u64>,
) -> (usize, f64, Tensor, Vec<ContractionIndex>) {
    let k = 2;
    let min = true;
    let config_file = String::from("tests/cut_kKaHyPar_sea20.ini");

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

    let partitioning = communication_partitioning(children_tensor, bond_dims, k, config_file, min);

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
pub(super) fn tensor_bipartition(
    children_tensor: &[(usize, Tensor)],
    bond_dims: &HashMap<usize, u64>,
) -> (f64, Vec<ContractionIndex>) {
    let (_, contraction_cost, _, contraction_path) =
        tensor_bipartition_recursive(children_tensor, bond_dims);
    (contraction_cost, contraction_path)
}

pub(super) fn balance_partitions(
    tn: &Tensor,
    contraction_tree: &mut ContractionTree,
    random_balance: bool,
    rebalance_depth: usize,
    partition_costs: &[(usize, f64)],
    greedy_cost_fn: fn(&Tensor, &Tensor) -> f64,
) -> (f64, Vec<ContractionIndex>, Tensor) {
    // If there are less than 3 tensors in the tn, rebalancing will not make sense.
    if tn.total_num_tensors() < 3 {
        // TODO: should not panic, but handle gracefully
        panic!("No rebalancing undertaken, as tn is too small (< 3 tensors)");
    }

    // Use memory reduction to identify which leaf node to shift between partitions.
    let bond_dims = tn.bond_dims();
    // Obtain most expensive and cheapest partitions
    let (larger_subtree_id, _) = *partition_costs.last().unwrap();
    let larger_subtree_parent_id = contraction_tree
        .node(larger_subtree_id)
        .parent_id()
        .unwrap();

    // Get bigger subtree leaf nodes
    let mut larger_subtree_leaf_nodes = contraction_tree.leaf_ids(larger_subtree_id);
    // Find the leaf node in the smaller subtree that causes the biggest memory reduction in the bigger subtree
    let (smaller_subtree_id, rebalanced_node, _) = partition_costs
        .iter()
        .take(partition_costs.len() - 1)
        .map(|(subtree_root_id, _)| {
            let (potential_node, cost) = rebalance_node_largest_overlap(
                random_balance,
                contraction_tree,
                &larger_subtree_leaf_nodes,
                *subtree_root_id,
                greedy_cost_fn,
                tn,
            );
            (*subtree_root_id, potential_node, cost)
        })
        .max_by(|(_, _, cost_a), (_, _, cost_b)| cost_a.total_cmp(cost_b))
        .unwrap();

    let mut new_max = partition_costs
        .iter()
        .filter(|&&(subtree_id, _)| {
            subtree_id != smaller_subtree_id && subtree_id != larger_subtree_id
        })
        .map(|(_, cost)| *cost)
        .last()
        .unwrap_or_default();

    // Remove the updated partitions from the `ContractionTree`
    contraction_tree
        .partitions
        .get_mut(&rebalance_depth)
        .unwrap()
        .retain(|&e| e != smaller_subtree_id && e != larger_subtree_id);

    // Obtain parent nodes of root of most expensive and cheapest partitions.
    let smaller_subtree_parent_id = contraction_tree
        .node(smaller_subtree_id)
        .parent_id()
        .unwrap();

    // Get smaller subtree leaf nodes
    let mut smaller_subtree_leaf_nodes = contraction_tree.leaf_ids(smaller_subtree_id);

    // Always check that a node can be moved over.
    assert!(!smaller_subtree_leaf_nodes.contains(&rebalanced_node));
    assert!(larger_subtree_leaf_nodes.contains(&rebalanced_node));

    // Remove selected tensor from bigger subtree. Add it to the smaller subtree
    smaller_subtree_leaf_nodes.push(rebalanced_node);
    larger_subtree_leaf_nodes.retain(|&leaf| leaf != rebalanced_node);

    let (smaller_indices, updated_smaller_path) = subtree_contraction_path(
        smaller_subtree_leaf_nodes,
        tn,
        contraction_tree,
        &mut new_max,
    );

    let (larger_indices, updated_larger_path) = subtree_contraction_path(
        larger_subtree_leaf_nodes,
        tn,
        contraction_tree,
        &mut new_max,
    );

    contraction_tree.remove_subtree(smaller_subtree_id);
    let smaller_partition_root = contraction_tree.add_subtree(
        &updated_smaller_path,
        smaller_subtree_parent_id,
        &smaller_indices,
    );

    contraction_tree
        .partitions
        .get_mut(&rebalance_depth)
        .unwrap()
        .push(smaller_partition_root);

    contraction_tree.remove_subtree(larger_subtree_id);
    let larger_partition_root = contraction_tree.add_subtree(
        &updated_larger_path,
        larger_subtree_parent_id,
        &larger_indices,
    );

    contraction_tree
        .partitions
        .get_mut(&rebalance_depth)
        .unwrap()
        .push(larger_partition_root);

    // Generate new paths based on greedy paths
    let children = &contraction_tree.partitions()[&rebalance_depth];

    let (partition_tensors, rebalanced_path) = children
        .iter()
        .enumerate()
        .map(|(i, node_id)| {
            let (tensors, local_path) = subtensor_network(contraction_tree, *node_id, tn);
            let mut tensor = Tensor::default();
            tensor.push_tensors(tensors, Some(&bond_dims), None);
            (tensor, ContractionIndex::Path(i, local_path))
        })
        .collect::<(Vec<_>, Vec<_>)>();

    let mut updated_tn = Tensor::default();
    updated_tn.push_tensors(partition_tensors, Some(&bond_dims), None);
    (new_max, rebalanced_path, updated_tn)
}

/// Computes a hashmap that maps the node id to its weight. The weight is the maximum
/// memory reduction that can be achieved if it is shifted to the other subtree.
pub(super) fn find_potential_nodes(
    contraction_tree: &ContractionTree,
    bigger_subtree_leaf_nodes: &[usize],
    smaller_subtree_root: usize,
    tn: &Tensor,
    cost_function: fn(&Tensor, &Tensor) -> f64,
) -> HashMap<usize, f64> {
    // Get a map that maps nodes to their tensors.
    let mut node_tensor_map = HashMap::new();
    populate_subtree_tensor_map(
        contraction_tree,
        smaller_subtree_root,
        &mut node_tensor_map,
        tn,
    );

    (bigger_subtree_leaf_nodes.iter().map(|leaf_index| {
        let t1 = tn.nested_tensor(
            contraction_tree
                .node(*leaf_index)
                .tensor_index
                .as_ref()
                .unwrap(),
        );
        node_tensor_map
            .iter()
            .map(|(&index, tensor)| (index, cost_function(tensor, t1)))
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap()
    }))
    .collect()
}

pub(super) fn rebalance_node_largest_overlap(
    random_balance: bool,
    contraction_tree: &ContractionTree,
    larger_subtree_leaf_nodes: &[usize],
    smaller_subtree_id: usize,
    greedy_cost_fn: fn(&Tensor, &Tensor) -> f64,
    tn: &Tensor,
) -> (usize, f64) {
    let (rebalanced_node, max_cost) = if random_balance {
        // Randomly select one of the top n nodes to rebalance.
        let top_n = 3;
        let rebalanced_node_weights = find_potential_nodes(
            contraction_tree,
            larger_subtree_leaf_nodes,
            smaller_subtree_id,
            tn,
            greedy_cost_fn,
        );

        let mut keys: Vec<(usize, f64)> = rebalanced_node_weights
            .iter()
            .map(|(a, b)| (*a, *b))
            .collect_vec();
        keys.sort_by(|(a, _), (b, _)| {
            rebalanced_node_weights[a].total_cmp(&rebalanced_node_weights[b])
        });
        if keys.len() < top_n {
            panic!("Error rebalance_path: Not enough nodes in the bigger subtree to select the top {top_n} from!");
        } else {
            // Sample randomly from the top n nodes. Use softmax probabilities.
            let top_n_nodes = keys.into_iter().take(top_n).collect_vec();

            // Subtract max val after inverting for numerical stability.
            let l2_norm = top_n_nodes
                .iter()
                .map(|(_idx, weight)| weight.powi(2))
                .sum::<f64>()
                .sqrt();
            let top_n_exp = top_n_nodes
                .iter()
                .map(|(_idx, weight)| (-1.0 * *weight / l2_norm).exp())
                .collect_vec();

            let sum_exp = top_n_exp.iter().sum::<f64>();
            let top_n_prob = top_n_exp.iter().map(|&exp| (exp / sum_exp)).collect_vec();

            // Sample index based on its probability
            let dist = WeightedIndex::new(top_n_prob).unwrap();
            let mut rng = thread_rng();
            let rand_idx = dist.sample(&mut rng);

            top_n_nodes[rand_idx]
        }
    } else {
        let (best_node, max_cost) = larger_subtree_leaf_nodes
            .iter()
            .map(|larger_node| {
                (
                    *larger_node,
                    contraction_tree
                        .max_match_by(*larger_node, smaller_subtree_id, tn, greedy_cost_fn)
                        .unwrap()
                        .1,
                )
            })
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap();
        (best_node, max_cost)
    };
    (rebalanced_node, max_cost)
}
