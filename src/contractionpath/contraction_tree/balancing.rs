use balancing_schemes::BalancingScheme;
use itertools::Itertools;
use rand::{
    distributions::{Distribution, WeightedIndex},
    thread_rng,
};
use rustc_hash::FxHashMap;

use crate::{tensornetwork::tensor::Tensor, types::ContractionIndex};

use super::{populate_subtree_tensor_map, ContractionTree};

pub mod balancing_schemes;
pub mod communication_schemes;

pub(super) fn balance_partitions(
    tn: &Tensor,
    contraction_tree: &mut ContractionTree,
    random_balance: bool,
    rebalance_depth: usize,
    partition_costs: &[(usize, f64)],
    greedy_cost_fn: fn(&Tensor, &Tensor) -> f64,
    balancing_scheme: BalancingScheme,
) -> (f64, Vec<ContractionIndex>, Tensor) {
    // If there are less than 3 tensors in the tn, rebalancing will not make sense.
    if tn.total_num_tensors() < 3 {
        // TODO: should not panic, but handle gracefully
        panic!("No rebalancing undertaken, as tn is too small (< 3 tensors)");
    }
    // Will cause strange errors (picking of same partition multiple times if this is not true.Better to panic here.)
    assert!(partition_costs.len() > 1);
    // Use memory reduction to identify which leaf node to shift between partitions.
    let bond_dims = tn.bond_dims();
    let (new_max, partition_tensors, rebalanced_path) = match balancing_scheme {
        BalancingScheme::BestWorst => balancing_schemes::best_worst_balancing(
            partition_costs,
            contraction_tree,
            random_balance,
            greedy_cost_fn,
            tn,
            rebalance_depth,
        ),
        BalancingScheme::Tensor => balancing_schemes::best_tensor_balancing(
            partition_costs,
            contraction_tree,
            random_balance,
            greedy_cost_fn,
            tn,
            rebalance_depth,
        ),
        BalancingScheme::Tensors => balancing_schemes::best_tensors_balancing(
            partition_costs,
            contraction_tree,
            random_balance,
            greedy_cost_fn,
            tn,
            rebalance_depth,
        ),
        _ => Default::default(),
    };

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
) -> FxHashMap<usize, f64> {
    // Get a map that maps nodes to their tensors.
    let mut node_tensor_map = FxHashMap::default();
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
