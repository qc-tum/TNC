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

    /// Identifies the tensor in the slowest subtree and passes it to the subtree
    /// with largest memory reduction.
    Tensor,

    /// Identifies the tensor in the slowest subtree and passes it to the subtree
    /// with largest memory reduction. Then identifies the tensor with the largest
    /// memory reduction when passed to the fastest subtree. Both slowest and fastest
    /// subtrees are updated.
    Tensors,

    /// Identifies the intermediate tensor in the slowest subtree and passes it to
    /// the subtree with largest memory reduction. Then identifies the intermediate
    /// tensor with the largest memory reduction when passed to the fastest subtree.
    /// Both slowest and fastest subtrees are updated.
    IntermediateTensors {
        /// The `height` up the contraction tree we look when passing intermediate
        /// tensors between partitions. A value of `1` allows intermediate tensors
        /// that are a product of at most 1 contraction process. Using the value of
        /// `0` is then equivalent to the `Tensors` method.
        height_limit: usize,
    },

    Configuration,
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
/// Then identifies the tensor with the largest memory reduction when passed to the fastest subtree. Both slowest and fastest subtrees are updated.
pub(super) fn best_intermediate_tensors<R>(
    partition_data: &[PartitionData],
    contraction_tree: &ContractionTree,
    random_balance: &mut Option<(usize, R)>,
    objective_function: fn(&Tensor, &Tensor) -> f64,
    tensor: &Tensor,
    height_limit: usize,
) -> Vec<Shift>
where
    R: Sized + Rng,
{
    // Obtain most expensive and cheapest partitions
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
    shifts.push(Shift {
        from_subtree_id: larger_subtree_id,
        to_subtree_id: smaller_subtree_id,
        moved_leaf_ids: rebalanced_leaf_ids,
    });
    shifts
}

#[cfg(test)]
mod tests {
    use rand::rngs::StdRng;
    use rustc_hash::FxHashMap;

    use crate::{
        contractionpath::contraction_tree::{
            balancing::{
                balancing_schemes::{
                    best_intermediate_tensors, best_tensor, best_tensors, best_worst, Shift,
                },
                PartitionData,
            },
            ContractionTree,
        },
        path,
        tensornetwork::tensor::Tensor,
    };

    fn setup_simple_partition_data() -> Vec<PartitionData> {
        vec![
            PartitionData {
                id: 2,
                cost: 1f64,
                contraction: Vec::new(),
                local_tensor: Tensor::default(),
            },
            PartitionData {
                id: 7,
                cost: 2f64,
                contraction: Vec::new(),
                local_tensor: Tensor::default(),
            },
            PartitionData {
                id: 14,
                cost: 3f64,
                contraction: Vec::new(),
                local_tensor: Tensor::default(),
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

        let tensor0 = Tensor::new(vec![7, 8]);
        let tensor1 = Tensor::new(vec![8, 9, 10]);

        let tensor3 = Tensor::new(vec![0, 6]);
        let tensor4 = Tensor::new(vec![1, 6]);
        let tensor5 = Tensor::new(vec![5, 7]);

        let tensor8 = Tensor::new(vec![0, 1]);
        let tensor9 = Tensor::new(vec![2, 3]);
        let tensor10 = Tensor::new(vec![3, 4]);
        let tensor11 = Tensor::new(vec![4, 5, 10]);

        let mut intermediate_tensor2 = Tensor::default();
        intermediate_tensor2.push_tensors(vec![tensor0, tensor1], Some(&bond_dims), None);

        let mut intermediate_tensor7 = Tensor::default();
        intermediate_tensor7.push_tensors(vec![tensor3, tensor4, tensor5], Some(&bond_dims), None);

        let mut intermediate_tensor14 = Tensor::default();
        intermediate_tensor14.push_tensors(
            vec![tensor8, tensor9, tensor10, tensor11],
            Some(&bond_dims),
            None,
        );

        let mut tensor15 = Tensor::default();
        tensor15.push_tensors(
            vec![
                intermediate_tensor2,
                intermediate_tensor7,
                intermediate_tensor14,
            ],
            Some(&bond_dims),
            None,
        );

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
    fn test_intermediate_tensors_balancing() {
        let partition_data = setup_simple_partition_data();
        let (contraction_tree, tensor) = setup_simple();

        let output = best_intermediate_tensors::<StdRng>(
            &partition_data,
            &contraction_tree,
            &mut None,
            custom_cost_function,
            &tensor,
            1,
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
