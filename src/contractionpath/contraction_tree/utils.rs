use itertools::Itertools;
use rustc_hash::FxHashMap;

use crate::{
    contractionpath::{
        contraction_cost::{contract_cost_tensors, contract_path_cost, contract_size_tensors},
        paths::{greedy::Greedy, CostType, OptimizePath},
    },
    pair,
    tensornetwork::{create_tensor_network, tensor::Tensor},
    types::ContractionIndex,
};

use super::{balancing::PartitionData, ContractionTree};

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
    tensor_network: &Tensor,
) -> (f64, f64, Tensor) {
    let left_child_id = contraction_tree.node(node_id).left_child_id();
    let right_child_id = contraction_tree.node(node_id).right_child_id();
    if let (Some(left_child_id), Some(right_child_id)) = (left_child_id, right_child_id) {
        let (left_op_cost, left_mem_cost, t1) =
            parallel_tree_contraction_cost(contraction_tree, left_child_id, tensor_network);
        let (right_op_cost, right_mem_cost, t2) =
            parallel_tree_contraction_cost(contraction_tree, right_child_id, tensor_network);
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
        let tensor = tensor_network.nested_tensor(&tensor_id).clone();
        (0.0, tensor.size() as f64, tensor)
    }
}

/// Identifies the contraction path designated by subtree rooted at `node_id` in contraction tree. Allows for Tensor to have a different structure than
/// ContractionTree as long as leaf_ids in ContractionTree match the Tensor
pub(super) fn subtree_tensor_network(
    node_id: usize,
    contraction_tree: &ContractionTree,
    tensor_network: &Tensor,
) -> (Vec<Tensor>, Vec<ContractionIndex>) {
    let leaf_ids = contraction_tree.leaf_ids(node_id);
    let local_tensors = leaf_ids
        .iter()
        .map(|&id| {
            tensor_network.nested_tensor(contraction_tree.node(id).tensor_index.as_ref().unwrap())
        })
        .cloned()
        .collect_vec();
    let local_mapping = leaf_ids
        .iter()
        .enumerate()
        .map(|(local_idx, leaf_id)| (leaf_id, local_idx))
        .collect::<FxHashMap<_, _>>();

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

/// Generates a local contraction path for a subtree in a ContractionTree, returns the local contraction path with tree node indices, the local contraction path with local indexing and the cost of contracting.
/// One issue of generating a contraction path for a subtree is that tensor ids do not follow a strict ordering. Hence, a re-indexing is required to find the replace contraction path. This function can return the replace contraction path if `replace` is set to true.
pub(super) fn subtree_contraction_path(
    subtree_leaf_nodes: &[usize],
    contraction_tree: &ContractionTree,
    tensor_network: &Tensor,
) -> (Vec<ContractionIndex>, Vec<ContractionIndex>, f64) {
    // Obtain the flattened list of Tensors corresponding to `indices`. Introduces a new indexing to find the replace contraction path.
    let tensors = subtree_leaf_nodes
        .iter()
        .map(|&e| {
            tensor_network
                .nested_tensor(contraction_tree.node(e).tensor_index.as_ref().unwrap())
                .clone()
        })
        .collect();
    // Obtain tensor network corresponding to subtree
    let subtree_tensor_network = create_tensor_network(tensors, &tensor_network.bond_dims(), None);

    let mut opt = Greedy::new(&subtree_tensor_network, CostType::Flops);
    opt.optimize_path();

    let smaller_path_new_index = opt.get_best_replace_path();

    let smaller_path_node_index = smaller_path_new_index
        .iter()
        .map(|e| {
            if let ContractionIndex::Pair(v1, v2) = e {
                pair!(subtree_leaf_nodes[*v1], subtree_leaf_nodes[*v2])
            } else {
                panic!("Should only produce Pairs!")
            }
        })
        .collect_vec();

    (
        smaller_path_node_index,
        smaller_path_new_index,
        opt.get_best_flops(),
    )
}

/// Calculate local contraction path and corresponding cost at `rebalance_depth`
/// Returns a vector of tuples, where each tuple has the tensor_id of the child and its contraction cost.
/// tensor_id is required to identify the partition if it is sorted.
pub(super) fn characterize_partition(
    contraction_tree: &ContractionTree,
    rebalance_depth: usize,
    tensor_network: &Tensor,
    sort: bool,
) -> Vec<PartitionData> {
    let children = &contraction_tree.partitions()[&rebalance_depth];

    // Identify the contraction cost of each partition
    let mut partition_costs = children
        .iter()
        .map(|child| {
            let (local_tensors, local_contraction_path) =
                subtree_tensor_network(*child, contraction_tree, tensor_network);

            let mut new_tensor = Tensor::default();
            new_tensor.insert_bond_dims(&tensor_network.bond_dims());
            PartitionData {
                id: *child,
                cost: contract_path_cost(&local_tensors, &local_contraction_path, true).0,
                contraction: local_contraction_path,
                local_tensor: local_tensors.iter().fold(new_tensor, |a, b| &a ^ b),
            }
        })
        .collect_vec();
    if sort {
        partition_costs.sort_unstable_by(|a, b| a.cost.total_cmp(&b.cost));
    }

    partition_costs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::path;

    use rustc_hash::FxHashMap;

    fn setup_simple() -> (Tensor, Vec<ContractionIndex>) {
        (
            create_tensor_network(
                vec![
                    Tensor::new(vec![4, 3, 2]),
                    Tensor::new(vec![0, 1, 3, 2]),
                    Tensor::new(vec![4, 5, 6]),
                ],
                &FxHashMap::from_iter([(0, 5), (1, 2), (2, 6), (3, 8), (4, 1), (5, 3), (6, 4)]),
                None,
            ),
            path![(0, 1), (2, 0)].to_vec(),
        )
    }

    fn setup_complex() -> (Tensor, Vec<ContractionIndex>) {
        (
            create_tensor_network(
                vec![
                    Tensor::new(vec![4, 3, 2]),
                    Tensor::new(vec![0, 1, 3, 2]),
                    Tensor::new(vec![4, 5, 6]),
                    Tensor::new(vec![6, 8, 9]),
                    Tensor::new(vec![10, 8, 9]),
                    Tensor::new(vec![5, 1, 0]),
                ],
                &FxHashMap::from_iter([
                    (0, 27),
                    (1, 18),
                    (2, 12),
                    (3, 15),
                    (4, 5),
                    (5, 3),
                    (6, 18),
                    (7, 22),
                    (8, 45),
                    (9, 65),
                    (10, 5),
                ]),
                None,
            ),
            path![(1, 5), (0, 1), (3, 4), (2, 3), (0, 2)].to_vec(),
        )
    }

    fn setup_double_nested() -> (Tensor, Vec<ContractionIndex>) {
        let bond_dims = FxHashMap::from_iter([
            (0, 27),
            (1, 18),
            (2, 12),
            (3, 15),
            (4, 5),
            (5, 3),
            (6, 18),
            (7, 22),
            (8, 45),
            (9, 65),
            (10, 5),
            (11, 17),
        ]);

        let t0 = Tensor::new(vec![4, 3, 2]);
        let t1 = Tensor::new(vec![0, 1, 3, 2]);
        let t2 = Tensor::new(vec![4, 5, 6]);
        let t3 = Tensor::new(vec![6, 8, 9]);
        let t4 = Tensor::new(vec![5, 1, 0]);
        let t5 = Tensor::new(vec![10, 8, 9]);

        let mut t01 = Tensor::default();
        t01.push_tensors(vec![t0, t1], Some(&bond_dims), None);

        let mut t012 = Tensor::default();
        t012.push_tensors(vec![t01, t2], Some(&bond_dims), None);

        let mut t34 = Tensor::default();
        t34.push_tensors(vec![t3, t4], Some(&bond_dims), None);

        let mut t345 = Tensor::default();
        t345.push_tensors(vec![t34, t5], Some(&bond_dims), None);

        let mut tensor_network = Tensor::default();
        tensor_network.push_tensors(vec![t012, t345], Some(&bond_dims), None);
        (
            tensor_network,
            path![
                (0, [(0, [(0, 1)]), (0, 1)]),
                (1, [(0, [(0, 1)]), (0, 1)]),
                (0, 1)
            ]
            .to_vec(),
        )
    }

    fn setup_nested() -> (Tensor, Vec<ContractionIndex>) {
        let bond_dims = FxHashMap::from_iter([
            (0, 4),
            (1, 6),
            (2, 2),
            (3, 3),
            (4, 2),
            (5, 4),
            (6, 7),
            (7, 5),
            (8, 2),
        ]);

        let t0 = Tensor::new(vec![0, 1]);
        let t1 = Tensor::new(vec![0, 2]);
        let t2 = Tensor::new(vec![3]);
        let t3 = Tensor::new(vec![2, 4]);
        let t4 = Tensor::new(vec![1, 3, 5, 8]);
        let t5 = Tensor::new(vec![4, 7, 8]);
        let t6 = Tensor::new(vec![5, 6, 7]);
        let t7 = Tensor::new(vec![6]);

        let mut t012 = Tensor::default();
        t012.push_tensors(vec![t0, t1, t2], Some(&bond_dims), None);

        let mut t345 = Tensor::default();
        t345.push_tensors(vec![t3, t4, t5], Some(&bond_dims), None);

        let mut t67 = Tensor::default();
        t67.push_tensors(vec![t6, t7], Some(&bond_dims), None);

        let mut tensor_network = Tensor::default();
        tensor_network.push_tensors(vec![t012, t345, t67], Some(&bond_dims), None);
        (
            tensor_network,
            path![
                (0, [(0, 1), (0, 2)]),
                (1, [(0, 1), (0, 2)]),
                (2, [(0, 1)]),
                (0, 1),
                (0, 2)
            ]
            .to_vec(),
        )
    }

    #[test]
    fn test_parallel_tree_contraction_path() {
        let (tensor, ref_path) = setup_simple();
        let tree = ContractionTree::from_contraction_path(&tensor, &ref_path);

        let (op_cost, mem_cost, _) =
            parallel_tree_contraction_cost(&tree, tree.root_id().unwrap(), &tensor);

        assert_eq!(op_cost, 4540f64);
        assert_eq!(mem_cost, 538f64);
    }

    #[test]
    fn test_parallel_tree_contraction_path_complex() {
        let (tensor, ref_path) = setup_complex();
        let tree = ContractionTree::from_contraction_path(&tensor, &ref_path);

        let (op_cost, mem_cost, _) =
            parallel_tree_contraction_cost(&tree, tree.root_id().unwrap(), &tensor);

        assert_eq!(op_cost, 2120600f64);
        assert_eq!(mem_cost, 89478f64);
    }

    #[test]
    fn test_subtree_network() {
        let (tensor, ref_path) = setup_double_nested();
        let contraction_tree = ContractionTree::from_contraction_path(&tensor, &ref_path);
        let subtree_leaf_nodes = vec![0, 1, 3, 5];
        let (tree_contraction_path, local_contraction_path, cost) =
            subtree_contraction_path(&subtree_leaf_nodes, &contraction_tree, &tensor);

        assert_eq!(
            path![(1, 0), (5, 3), (1, 5)].to_vec(),
            tree_contraction_path
        );

        assert_eq!(
            path![(1, 0), (3, 2), (1, 3)].to_vec(),
            local_contraction_path
        );

        assert_eq!(171781290f64, cost);
    }

    impl PartialEq for PartitionData {
        fn eq(&self, other: &Self) -> bool {
            self.id == other.id
                && self.cost == other.cost
                && self.contraction == other.contraction
                && self.local_tensor.legs() == other.local_tensor.legs()
        }
    }

    #[test]
    fn test_characterize_partitions() {
        let (tensor, ref_path) = setup_nested();
        let mut contraction_tree = ContractionTree::from_contraction_path(&tensor, &ref_path);
        contraction_tree.partitions.insert(1, vec![4, 9, 12]);
        let rebalance_depth = 1;
        let partition_data =
            characterize_partition(&contraction_tree, rebalance_depth, &tensor, false);
        let ref_partition_data = vec![
            PartitionData {
                id: 4,
                cost: 84f64,
                contraction: path![(0, 1), (0, 2)].to_vec(),
                local_tensor: Tensor::new(vec![1, 2, 3]),
            },
            PartitionData {
                id: 9,
                cost: 3456f64,
                contraction: path![(0, 1), (0, 2)].to_vec(),
                local_tensor: Tensor::new(vec![2, 1, 3, 5, 7]),
            },
            PartitionData {
                id: 12,
                cost: 140f64,
                contraction: path![(0, 1)].to_vec(),
                local_tensor: Tensor::new(vec![5, 7]),
            },
        ];
        assert_eq!(ref_partition_data, partition_data);
    }

    #[test]
    fn test_characterize_partitions_sorted() {
        let (tensor, ref_path) = setup_nested();
        let mut contraction_tree = ContractionTree::from_contraction_path(&tensor, &ref_path);
        contraction_tree.partitions.insert(1, vec![4, 9, 12]);
        let rebalance_depth = 1;
        let partition_data =
            characterize_partition(&contraction_tree, rebalance_depth, &tensor, true);
        let ref_partition_data = vec![
            PartitionData {
                id: 4,
                cost: 84f64,
                contraction: path![(0, 1), (0, 2)].to_vec(),
                local_tensor: Tensor::new(vec![1, 2, 3]),
            },
            PartitionData {
                id: 12,
                cost: 140f64,
                contraction: path![(0, 1)].to_vec(),
                local_tensor: Tensor::new(vec![5, 7]),
            },
            PartitionData {
                id: 9,
                cost: 3456f64,
                contraction: path![(0, 1), (0, 2)].to_vec(),
                local_tensor: Tensor::new(vec![2, 1, 3, 5, 7]),
            },
        ];
        assert_eq!(ref_partition_data, partition_data);
    }
}
