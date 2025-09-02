use rand::Rng;
use rustc_hash::FxHashMap;

use crate::contractionpath::contraction_tree::balancing::{find_rebalance_node, PartitionData};
use crate::contractionpath::contraction_tree::{
    populate_leaf_node_tensor_map, populate_subtree_tensor_map, ContractionTree,
};
use crate::tensornetwork::tensor::Tensor;

#[derive(Debug, Clone, Copy)]
pub enum BalancingScheme {
    /// Moves a tensor from the slowest subtree to the fastest subtree each time.
    BestWorst,

    /// Identifies the tensor in the slowest subtree and passes it to the subtree
    /// with largest memory reduction.
    Tensor,

    /// Identifies the tensor in the slowest subtree and passes it to the subtree
    /// with largest memory reduction. Then identifies the tensor with the largest
    /// memory reduction when passed to the fastest subtree. Both slowest and fastest
    /// subtrees are updated.
    Tensors,

    /// Identifies the tensor in the slowest subtree and passes it to the subtree
    /// with largest memory reduction for odd iterations or the tensor with the largest
    /// memory reduction when passed to the fastest subtree for even iterations.
    AlternatingTensors,

    /// Identifies the intermediate tensor in the slowest subtree and passes it to
    /// the subtree with largest memory reduction. Then identifies the intermediate
    /// tensor with the largest memory reduction when passed to the fastest subtree.
    /// Both slowest and fastest subtrees are updated.
    IntermediateTensors {
        /// The `height` up the contraction tree we look when passing intermediate
        /// tensors between partitions. A value of `Some(1)` allows intermediate tensors
        /// that are a product of at most 1 contraction process. Using the value of
        /// `Some(0)` is then equivalent to the `Tensors` method. Setting it to `None`
        /// imposes no height limit.`
        height_limit: Option<usize>,
    },

    /// Identifies the intermediate tensor in the slowest subtree and passes it to
    /// the subtree with largest memory reduction for odd iterations. Identifies the intermediate
    /// tensor with the largest memory reduction when passed to the fastest subtree for
    /// odd iterations.
    AlternatingIntermediateTensors {
        /// The `height` up the contraction tree we look when passing intermediate
        /// tensors between partitions. A value of `Some(1)` allows intermediate tensors
        /// that are a product of at most 1 contraction process. Using the value of
        /// `Some(0)` is then equivalent to the `Tensors` method. Setting it to `None`
        /// imposes no height limit.
        height_limit: Option<usize>,
    },

    /// Identifies the intermediate tensor in the slowest subtree and passes it to
    /// the subtree with largest memory reduction for odd iterations. Identifies the intermediate
    /// tensor with the largest memory reduction when passed to the fastest subtree for
    /// odd iterations.
    AlternatingTreeTensors {
        /// The `height` up the contraction tree we look when passing intermediate
        /// tensors between partitions. A value of `1` allows intermediate tensors
        /// that are a product of at most 1 contraction process. Using the value of
        /// `0` is then equivalent to the `Tensors` method.
        height_limit: usize,
    },
}

/// Shift of tensors between partitions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Shift {
    /// Id of the source partition.
    pub from_subtree_id: usize,
    /// Id of the destination partition.
    pub to_subtree_id: usize,
    /// Ids of the leaf nodes that are moved.
    pub moved_leaf_ids: Vec<usize>,
}

/// Balancing scheme that moves a tensor from the slowest subtree to the fastest subtree each time.
/// Chosen tensor maximizes the `objective_function`, which is typically memory reduction.
pub(super) fn best_worst<R>(
    partition_data: &[PartitionData],
    contraction_tree: &ContractionTree,
    random_balance: &mut Option<(usize, R)>,
    objective_function: fn(&Tensor, &Tensor) -> f64,
    tensor: &Tensor,
) -> Vec<Shift>
where
    R: Rng,
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
    vec![Shift {
        from_subtree_id: larger_subtree_id,
        to_subtree_id: smaller_subtree_id,
        moved_leaf_ids: rebalanced_leaf_ids,
    }]
}

/// Balancing scheme that identifies the tensor in the slowest subtree and passes it to the subtree with largest memory reduction.
/// Chosen tensor maximizes the `objective_function`, which is typically memory reduction.
pub(super) fn best_tensor<R>(
    partition_data: &[PartitionData],
    contraction_tree: &ContractionTree,
    random_balance: &mut Option<(usize, R)>,
    objective_function: fn(&Tensor, &Tensor) -> f64,
    tensor: &Tensor,
) -> Vec<Shift>
where
    R: Rng,
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
    vec![Shift {
        from_subtree_id: larger_subtree_id,
        to_subtree_id: smaller_subtree_id,
        moved_leaf_ids: rebalanced_leaf_ids,
    }]
}

/// Balancing scheme that identifies the tensor in the slowest subtree and passes it to the subtree with largest memory reduction.
/// Then identifies the tensor with the largest memory reduction when passed to the fastest subtree. Both slowest and fastest subtrees are updated.
pub(super) fn best_tensors<R>(
    partition_data: &[PartitionData],
    contraction_tree: &ContractionTree,
    random_balance: &mut Option<(usize, R)>,
    objective_function: fn(&Tensor, &Tensor) -> f64,
    tensor: &Tensor,
) -> Vec<Shift>
where
    R: Rng,
{
    // Obtain most expensive and cheapest partitions
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

    let mut shifts = Vec::with_capacity(2);
    shifts.push(Shift {
        from_subtree_id: larger_subtree_id,
        to_subtree_id: smaller_subtree_id,
        moved_leaf_ids: rebalanced_leaf_ids,
    });

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
    shifts.push(Shift {
        from_subtree_id: larger_subtree_id,
        to_subtree_id: smaller_subtree_id,
        moved_leaf_ids: rebalanced_leaf_ids,
    });
    shifts
}

/// Balancing scheme that identifies the tensor in the slowest subtree and passes it to the subtree with largest memory reduction.
pub(super) fn tensors_odd<R>(
    partition_data: &[PartitionData],
    contraction_tree: &ContractionTree,
    random_balance: &mut Option<(usize, R)>,
    objective_function: fn(&Tensor, &Tensor) -> f64,
    tensor: &Tensor,
) -> Vec<Shift>
where
    R: Rng,
{
    // Obtain most expensive partition
    let larger_subtree_id = partition_data.last().unwrap().id;

    let larger_subtree_leaf_nodes =
        populate_leaf_node_tensor_map(contraction_tree, larger_subtree_id, tensor);

    // Find the subtree shift that results in the largest memory savings
    let (smaller_subtree_id, rebalanced_leaf_node, _) = partition_data
        .iter()
        .take(partition_data.len() - 1)
        .map(|smaller| {
            let smaller_subtree_nodes = FxHashMap::from_iter([(0, smaller.local_tensor.clone())]);
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

    let rebalanced_leaf_id = contraction_tree.leaf_ids(rebalanced_leaf_node);
    vec![Shift {
        from_subtree_id: larger_subtree_id,
        to_subtree_id: smaller_subtree_id,
        moved_leaf_ids: rebalanced_leaf_id,
    }]
}

/// Balancing scheme that identifies the tensor with the largest memory reduction when passed to the fastest subtree.
pub(super) fn tensors_even<R>(
    partition_data: &[PartitionData],
    contraction_tree: &ContractionTree,
    random_balance: &mut Option<(usize, R)>,
    objective_function: fn(&Tensor, &Tensor) -> f64,
    tensor: &Tensor,
) -> Vec<Shift>
where
    R: Rng,
{
    let smaller_subtree_id = partition_data.first().unwrap().id;

    let smaller_subtree_nodes =
        FxHashMap::from_iter([(0, partition_data.first().unwrap().local_tensor.clone())]);

    let (larger_subtree_id, rebalanced_leaf_node, _) = partition_data
        .iter()
        .skip(1)
        .map(|larger| {
            let larger_subtree_leaf_nodes =
                populate_leaf_node_tensor_map(contraction_tree, larger.id, tensor);
            let (rebalanced_node, objective) = find_rebalance_node(
                random_balance,
                &larger_subtree_leaf_nodes,
                &smaller_subtree_nodes,
                objective_function,
            );

            (larger.id, rebalanced_node, objective)
        })
        .max_by(|a, b| a.2.total_cmp(&b.2))
        .unwrap();

    let rebalanced_leaf_id = contraction_tree.leaf_ids(rebalanced_leaf_node);
    vec![Shift {
        from_subtree_id: larger_subtree_id,
        to_subtree_id: smaller_subtree_id,
        moved_leaf_ids: rebalanced_leaf_id,
    }]
}

/// Balancing scheme that identifies the tensor in the slowest subtree and passes it to the subtree with largest memory reduction.
/// Then identifies the tensor with the largest memory reduction when passed to the fastest subtree. Both slowest and fastest subtrees are updated.
pub(super) fn best_intermediate_tensors<R>(
    partition_data: &[PartitionData],
    contraction_tree: &ContractionTree,
    random_balance: &mut Option<(usize, R)>,
    objective_function: fn(&Tensor, &Tensor) -> f64,
    tensor: &Tensor,
    height_limit: Option<usize>,
) -> Vec<Shift>
where
    R: Rng,
{
    // Obtain most expensive partition
    let larger_subtree_id = partition_data.last().unwrap().id;

    // Obtain all intermediate nodes up to height `height_limit` in larger subtree
    let mut larger_subtree_nodes =
        populate_subtree_tensor_map(contraction_tree, larger_subtree_id, tensor, height_limit);
    larger_subtree_nodes.remove(&larger_subtree_id);

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

    let mut shifts = Vec::with_capacity(2);
    shifts.push(Shift {
        from_subtree_id: larger_subtree_id,
        to_subtree_id: smaller_subtree_id,
        moved_leaf_ids: rebalanced_leaf_ids,
    });

    let smaller_subtree_id = partition_data.first().unwrap().id;

    let smaller_subtree_nodes =
        populate_subtree_tensor_map(contraction_tree, smaller_subtree_id, tensor, None);

    let (larger_subtree_id, second_rebalanced_node, _) = partition_data
        .iter()
        .skip(1)
        .take(partition_data.len() - 2)
        .map(|larger| {
            let mut larger_subtree_nodes =
                populate_subtree_tensor_map(contraction_tree, larger.id, tensor, height_limit);
            larger_subtree_nodes.remove(&larger.id);
            let (rebalanced_node, objective) = find_rebalance_node(
                random_balance,
                &larger_subtree_nodes,
                &smaller_subtree_nodes,
                objective_function,
            );

            (larger.id, rebalanced_node, objective)
        })
        .max_by(|a, b| a.2.total_cmp(&b.2))
        .unwrap();

    let rebalanced_leaf_ids = contraction_tree.leaf_ids(second_rebalanced_node);
    shifts.push(Shift {
        from_subtree_id: larger_subtree_id,
        to_subtree_id: smaller_subtree_id,
        moved_leaf_ids: rebalanced_leaf_ids,
    });
    shifts
}

/// Balancing scheme that identifies the tensor in the slowest subtree and passes it to the subtree with largest memory reduction.
pub(super) fn intermediate_tensors_odd<R>(
    partition_data: &[PartitionData],
    contraction_tree: &ContractionTree,
    random_balance: &mut Option<(usize, R)>,
    objective_function: fn(&Tensor, &Tensor) -> f64,
    tensor: &Tensor,
    height_limit: Option<usize>,
) -> Vec<Shift>
where
    R: Rng,
{
    // Obtain most expensive partition
    let larger_subtree_id = partition_data.last().unwrap().id;

    // Obtain all intermediate nodes up to height `height_limit` in larger subtree
    let mut larger_subtree_nodes =
        populate_subtree_tensor_map(contraction_tree, larger_subtree_id, tensor, height_limit);
    larger_subtree_nodes.remove(&larger_subtree_id);

    // Find the subtree shift that results in the largest memory savings
    let (smaller_subtree_id, rebalanced_node, _) = partition_data
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

    let rebalanced_leaf_ids = contraction_tree.leaf_ids(rebalanced_node);
    vec![Shift {
        from_subtree_id: larger_subtree_id,
        to_subtree_id: smaller_subtree_id,
        moved_leaf_ids: rebalanced_leaf_ids,
    }]
}

/// Balancing scheme that identifies the intermediate tensor with the largest memory reduction when passed to the fastest subtree.
pub(super) fn intermediate_tensors_even<R>(
    partition_data: &[PartitionData],
    contraction_tree: &ContractionTree,
    random_balance: &mut Option<(usize, R)>,
    objective_function: fn(&Tensor, &Tensor) -> f64,
    tensor: &Tensor,
    height_limit: Option<usize>,
) -> Vec<Shift>
where
    R: Rng,
{
    let smaller_subtree_id = partition_data.first().unwrap().id;

    let smaller_subtree_nodes =
        populate_subtree_tensor_map(contraction_tree, smaller_subtree_id, tensor, None);

    let (larger_subtree_id, rebalanced_node, _) = partition_data
        .iter()
        .skip(1)
        .filter_map(|larger| {
            let mut larger_subtree_nodes =
                populate_subtree_tensor_map(contraction_tree, larger.id, tensor, height_limit);
            if larger_subtree_nodes.len() == 1 {
                return None;
            }
            larger_subtree_nodes.remove(&larger.id);
            let (rebalanced_node, objective) = find_rebalance_node(
                random_balance,
                &larger_subtree_nodes,
                &smaller_subtree_nodes,
                objective_function,
            );

            Some((larger.id, rebalanced_node, objective))
        })
        .max_by(|a, b| a.2.total_cmp(&b.2))
        .unwrap();

    let rebalanced_leaf_ids = contraction_tree.leaf_ids(rebalanced_node);
    vec![Shift {
        from_subtree_id: larger_subtree_id,
        to_subtree_id: smaller_subtree_id,
        moved_leaf_ids: rebalanced_leaf_ids,
    }]
}

/// Balancing scheme that identifies the tensor in the slowest subtree and passes it to the subtree with largest memory reduction.
pub(super) fn tree_tensors_odd(
    partition_data: &[PartitionData],
    contraction_tree: &ContractionTree,
    objective_function: fn(&Tensor, &Tensor) -> f64,
    tensor: &Tensor,
    height_limit: usize,
) -> Vec<Shift> {
    // Obtain most expensive partition
    let larger_subtree_id = partition_data.last().unwrap().id;

    // Obtain all intermediate nodes up to height `height_limit` in larger subtree
    let mut larger_subtree_nodes = populate_subtree_tensor_map(
        contraction_tree,
        larger_subtree_id,
        tensor,
        Some(height_limit),
    );
    larger_subtree_nodes.remove(&larger_subtree_id);

    // Find the subtree shift that results in the largest memory savings
    let (smaller_subtree_id, rebalanced_node, _) = partition_data
        .iter()
        .take(partition_data.len() - 1)
        .map(|smaller| {
            let PartitionData {
                local_tensor, id, ..
            } = smaller;
            let mut objective = 0.;
            let mut rebalanced_node = None;
            for (node_id, node) in &larger_subtree_nodes {
                let new_obj = objective_function(node, local_tensor);
                if new_obj > objective {
                    objective = new_obj;
                    rebalanced_node = Some(*node_id);
                }
            }
            (*id, rebalanced_node, objective)
        })
        .max_by(|a, b| a.2.total_cmp(&b.2))
        .unwrap();
    if let Some(rebalanced_node) = rebalanced_node {
        let rebalanced_leaf_ids = contraction_tree.leaf_ids(rebalanced_node);
        vec![Shift {
            from_subtree_id: larger_subtree_id,
            to_subtree_id: smaller_subtree_id,
            moved_leaf_ids: rebalanced_leaf_ids,
        }]
    } else {
        Vec::new()
    }
}

/// Balancing scheme that identifies the intermediate tensor with the largest memory reduction when passed to the fastest subtree.
pub(super) fn tree_tensors_even(
    partition_data: &[PartitionData],
    contraction_tree: &ContractionTree,
    objective_function: fn(&Tensor, &Tensor) -> f64,
    tensor: &Tensor,
    height_limit: usize,
) -> Vec<Shift> {
    let smaller_subtree_id = partition_data.first().unwrap().id;

    // let smaller_subtree_nodes =
    //     populate_subtree_tensor_map(contraction_tree, smaller_subtree_id, tensor, None);
    let PartitionData {
        local_tensor: smaller_tensor,
        ..
    } = partition_data.first().unwrap();

    let (larger_subtree_id, rebalanced_node, _) = partition_data
        .iter()
        .skip(1)
        .filter_map(|larger| {
            let mut larger_subtree_nodes = populate_subtree_tensor_map(
                contraction_tree,
                larger.id,
                tensor,
                Some(height_limit),
            );
            if larger_subtree_nodes.len() == 1 {
                return None;
            }
            larger_subtree_nodes.remove(&larger.id);
            let mut objective = 0.;
            let mut rebalanced_node = None;
            for (node_id, node) in &larger_subtree_nodes {
                let new_obj = objective_function(node, smaller_tensor);
                if new_obj > objective {
                    objective = new_obj;
                    rebalanced_node = Some(*node_id);
                }
            }

            Some((larger.id, rebalanced_node, objective))
        })
        .max_by(|a, b| a.2.total_cmp(&b.2))
        .unwrap();

    if let Some(rebalanced_node) = rebalanced_node {
        let rebalanced_leaf_ids = contraction_tree.leaf_ids(rebalanced_node);
        vec![Shift {
            from_subtree_id: larger_subtree_id,
            to_subtree_id: smaller_subtree_id,
            moved_leaf_ids: rebalanced_leaf_ids,
        }]
    } else {
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use rand::rngs::StdRng;
    use rustc_hash::FxHashMap;

    use crate::{
        contractionpath::contraction_tree::{balancing::PartitionData, ContractionTree},
        path,
        tensornetwork::tensor::Tensor,
    };

    fn setup_simple_partition_data() -> Vec<PartitionData> {
        let bond_dims = FxHashMap::from_iter([
            (0, 2),
            (1, 2),
            (2, 2),
            (3, 2),
            (4, 2),
            (5, 2),
            (6, 2),
            (7, 2),
            (8, 2),
            (9, 2),
            (10, 2),
        ]);
        vec![
            PartitionData {
                id: 2,
                flop_cost: 1.,
                mem_cost: 0.,
                contraction: Default::default(),
                local_tensor: Tensor::new_from_map(vec![7, 9, 10], &bond_dims),
            },
            PartitionData {
                id: 7,
                flop_cost: 2.,
                mem_cost: 0.,
                contraction: Default::default(),
                local_tensor: Tensor::new_from_map(vec![0, 1, 5, 7], &bond_dims),
            },
            PartitionData {
                id: 14,
                flop_cost: 3.,
                mem_cost: 0.,
                contraction: Default::default(),
                local_tensor: Tensor::new_from_map(vec![0, 1, 2, 5, 10], &bond_dims),
            },
        ]
    }

    /// Tensor ids in contraction tree included in variable name for easy tracking
    fn setup_simple() -> (ContractionTree, Tensor) {
        let bond_dims = FxHashMap::from_iter([
            (0, 2),
            (1, 2),
            (2, 2),
            (3, 2),
            (4, 2),
            (5, 2),
            (6, 2),
            (7, 2),
            (8, 2),
            (9, 2),
            (10, 2),
        ]);

        let tensor0 = Tensor::new_from_map(vec![7, 8], &bond_dims);
        let tensor1 = Tensor::new_from_map(vec![8, 9, 10], &bond_dims);

        let tensor3 = Tensor::new_from_map(vec![0, 6], &bond_dims);
        let tensor4 = Tensor::new_from_map(vec![1, 6], &bond_dims);
        let tensor5 = Tensor::new_from_map(vec![5, 7], &bond_dims);

        let tensor8 = Tensor::new_from_map(vec![0, 1], &bond_dims);
        let tensor9 = Tensor::new_from_map(vec![2, 3], &bond_dims);
        let tensor10 = Tensor::new_from_map(vec![3, 4], &bond_dims);
        let tensor11 = Tensor::new_from_map(vec![4, 5, 10], &bond_dims);

        let intermediate_tensor2 = Tensor::new_composite(vec![tensor0, tensor1]);

        let intermediate_tensor7 = Tensor::new_composite(vec![tensor3, tensor4, tensor5]);

        let intermediate_tensor14 =
            Tensor::new_composite(vec![tensor8, tensor9, tensor10, tensor11]);

        let tensor15 = Tensor::new_composite(vec![
            intermediate_tensor2,
            intermediate_tensor7,
            intermediate_tensor14,
        ]);

        let contraction_path = path![
            (0, [(0, 1)]),
            (1, [(0, 1), (0, 2)]),
            (2, [(0, 3), (2, 1), (0, 2)]),
            (0, 1),
            (0, 2)
        ];

        (
            ContractionTree::from_contraction_path(&tensor15, contraction_path),
            tensor15,
        )
    }

    fn custom_cost_function(a: &Tensor, b: &Tensor) -> f64 {
        (a & b).legs().len() as f64
    }

    #[test]
    fn test_best_worst_balancing() {
        let partition_data = setup_simple_partition_data();
        let (contraction_tree, tensor) = setup_simple();

        let output = best_worst::<StdRng>(
            &partition_data,
            &contraction_tree,
            &mut None,
            custom_cost_function,
            &tensor,
        );

        let ref_output = vec![Shift {
            from_subtree_id: 14,
            to_subtree_id: 2,
            moved_leaf_ids: vec![11],
        }];
        assert_eq!(output, ref_output);
    }

    #[test]
    fn test_tensor_balancing() {
        let partition_data = setup_simple_partition_data();
        let (contraction_tree, tensor) = setup_simple();

        let output = best_tensor::<StdRng>(
            &partition_data,
            &contraction_tree,
            &mut None,
            custom_cost_function,
            &tensor,
        );

        let ref_output = vec![Shift {
            from_subtree_id: 14,
            to_subtree_id: 7,
            moved_leaf_ids: vec![8],
        }];
        assert_eq!(output, ref_output);
    }

    #[test]
    fn test_tensors_balancing() {
        let partition_data = setup_simple_partition_data();
        let (contraction_tree, tensor) = setup_simple();

        let output = best_tensors::<StdRng>(
            &partition_data,
            &contraction_tree,
            &mut None,
            custom_cost_function,
            &tensor,
        );

        let ref_output = vec![
            Shift {
                from_subtree_id: 14,
                to_subtree_id: 7,
                moved_leaf_ids: vec![8],
            },
            Shift {
                from_subtree_id: 7,
                to_subtree_id: 2,
                moved_leaf_ids: vec![5],
            },
        ];
        assert_eq!(output, ref_output);
    }

    #[test]
    fn test_alternating_tensors_balancing_odd() {
        let partition_data = setup_simple_partition_data();
        let (contraction_tree, tensor) = setup_simple();

        let output = tensors_odd::<StdRng>(
            &partition_data,
            &contraction_tree,
            &mut None,
            custom_cost_function,
            &tensor,
        );

        // Shift tensor11 = Tensor::new(vec![4, 5, 10]);
        // Max overlap is tensor1 = Tensor::new(vec![8, 9, 10]);
        let ref_output = vec![Shift {
            from_subtree_id: 14,
            to_subtree_id: 7,
            moved_leaf_ids: vec![8],
        }];
        assert_eq!(output, ref_output);
    }

    #[test]
    fn test_alternating_tensors_balancing_even() {
        let partition_data = setup_simple_partition_data();
        let (contraction_tree, tensor) = setup_simple();

        let output = tensors_even::<StdRng>(
            &partition_data,
            &contraction_tree,
            &mut None,
            custom_cost_function,
            &tensor,
        );
        // Shift tensor8 = Tensor::new(vec![0, 1]);
        // Max overlap is tensor3 = Tensor::new(vec![0, 6]);
        let ref_output = vec![Shift {
            from_subtree_id: 14,
            to_subtree_id: 2,
            moved_leaf_ids: vec![11],
        }];
        assert_eq!(output, ref_output);
    }

    #[test]
    fn test_intermediate_tensors_balancing() {
        let partition_data = setup_simple_partition_data();
        let (contraction_tree, tensor) = setup_simple();

        let output = best_intermediate_tensors::<StdRng>(
            &partition_data,
            &contraction_tree,
            &mut None,
            custom_cost_function,
            &tensor,
            Some(1),
        );

        let ref_output = vec![
            Shift {
                from_subtree_id: 14,
                to_subtree_id: 7,
                moved_leaf_ids: vec![8, 11],
            },
            Shift {
                from_subtree_id: 7,
                to_subtree_id: 2,
                moved_leaf_ids: vec![5],
            },
        ];
        assert_eq!(output, ref_output);
    }
}
