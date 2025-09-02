use itertools::Itertools;
use rustc_hash::FxHashMap;

use crate::{
    contractionpath::{
        contraction_cost::contract_path_cost,
        contraction_tree::{balancing::PartitionData, ContractionTree},
        paths::{
            cotengrust::{Cotengrust, OptMethod},
            OptimizePath,
        },
    },
    pair,
    tensornetwork::tensor::Tensor,
    types::ContractionIndex,
};

/// Identifies the contraction path designated by subtree rooted at `node_id` in contraction tree. Allows for Tensor to have a different structure than
/// ContractionTree as long as `tensor_index` in ContractionTree match the Tensor
pub(super) fn subtree_tensor_network(
    node_id: usize,
    contraction_tree: &ContractionTree,
    tensor_network: &Tensor,
) -> (Vec<Tensor>, Vec<ContractionIndex>) {
    let leaf_ids = contraction_tree.leaf_ids(node_id);
    let local_tensors = leaf_ids
        .iter()
        .map(|&id| {
            tensor_network.nested_tensor(contraction_tree.node(id).tensor_index().as_ref().unwrap())
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
pub(super) fn subtree_contraction_path(
    subtree_leaf_nodes: &[usize],
    contraction_tree: &ContractionTree,
    tensor_network: &Tensor,
) -> (Vec<ContractionIndex>, Vec<ContractionIndex>, f64, f64) {
    // Obtain the flattened list of Tensors corresponding to `indices`. Introduces a new indexing to find the replace contraction path.
    let tensors = subtree_leaf_nodes
        .iter()
        .map(|&e| {
            tensor_network
                .nested_tensor(contraction_tree.node(e).tensor_index().as_ref().unwrap())
                .clone()
        })
        .collect_vec();

    // Obtain tensor network corresponding to subtree
    let subtree_tensor_network = Tensor::new_composite(tensors);

    let mut opt = Cotengrust::new(&subtree_tensor_network, OptMethod::Greedy);
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
        opt.get_best_size(),
    )
}

/// Calculate local contraction path and corresponding cost at `rebalance_depth`.
pub(super) fn characterize_partition(
    contraction_tree: &ContractionTree,
    rebalance_depth: usize,
    tensor_network: &Tensor,
) -> Vec<PartitionData> {
    let children = &contraction_tree.partitions()[&rebalance_depth];

    // Identify the contraction cost of each partition
    let partition_data = children
        .iter()
        .map(|child| {
            let (local_tensors, local_contraction_path) =
                subtree_tensor_network(*child, contraction_tree, tensor_network);

            let (flop_cost, mem_cost) =
                contract_path_cost(&local_tensors, &local_contraction_path, true);
            PartitionData {
                id: *child,
                flop_cost,
                mem_cost,
                contraction: local_contraction_path,
                local_tensor: local_tensors.iter().fold(Tensor::default(), |a, b| &a ^ b),
            }
        })
        .collect_vec();

    partition_data
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::iter::zip;

    use rustc_hash::FxHashMap;

    use crate::{path, types::EdgeIndex};

    fn setup_complex() -> (Tensor, Vec<ContractionIndex>, FxHashMap<EdgeIndex, u64>) {
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
        ]);
        (
            Tensor::new_composite(vec![
                Tensor::new_from_map(vec![4, 3, 2], &bond_dims),
                Tensor::new_from_map(vec![0, 1, 3, 2], &bond_dims),
                Tensor::new_from_map(vec![4, 5, 6], &bond_dims),
                Tensor::new_from_map(vec![6, 8, 9], &bond_dims),
                Tensor::new_from_map(vec![10, 8, 9], &bond_dims),
                Tensor::new_from_map(vec![5, 1, 0], &bond_dims),
            ]),
            path![(1, 5), (0, 1), (3, 4), (2, 3), (0, 2)].to_vec(),
            bond_dims,
        )
    }

    fn setup_double_nested() -> (Tensor, Vec<ContractionIndex>) {
        let bond_dims = FxHashMap::from_iter([
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 8),
            (8, 9),
            (9, 10),
            (10, 11),
            (11, 12),
        ]);

        let t0 = Tensor::new_from_map(vec![4, 3, 2], &bond_dims);
        let t1 = Tensor::new_from_map(vec![0, 1, 3, 2], &bond_dims);
        let t2 = Tensor::new_from_map(vec![4, 5, 6], &bond_dims);
        let t3 = Tensor::new_from_map(vec![6, 8, 9], &bond_dims);
        let t4 = Tensor::new_from_map(vec![5, 1, 0], &bond_dims);
        let t5 = Tensor::new_from_map(vec![10, 8, 9], &bond_dims);

        let t01 = Tensor::new_composite(vec![t0, t1]);
        let t012 = Tensor::new_composite(vec![t01, t2]);
        let t34 = Tensor::new_composite(vec![t3, t4]);
        let t345 = Tensor::new_composite(vec![t34, t5]);
        let tensor_network = Tensor::new_composite(vec![t012, t345]);
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

    fn setup_nested() -> (Tensor, Vec<ContractionIndex>, FxHashMap<EdgeIndex, u64>) {
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

        let t0 = Tensor::new_from_map(vec![0, 1], &bond_dims);
        let t1 = Tensor::new_from_map(vec![0, 2], &bond_dims);
        let t2 = Tensor::new_from_map(vec![3], &bond_dims);
        let t3 = Tensor::new_from_map(vec![2, 4], &bond_dims);
        let t4 = Tensor::new_from_map(vec![1, 3, 5, 8], &bond_dims);
        let t5 = Tensor::new_from_map(vec![4, 7, 8], &bond_dims);
        let t6 = Tensor::new_from_map(vec![5, 6, 7], &bond_dims);
        let t7 = Tensor::new_from_map(vec![6], &bond_dims);

        let t012 = Tensor::new_composite(vec![t0, t1, t2]);
        let t345 = Tensor::new_composite(vec![t3, t4, t5]);
        let t67 = Tensor::new_composite(vec![t6, t7]);
        let tensor_network = Tensor::new_composite(vec![t012, t345, t67]);
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
            bond_dims,
        )
    }

    #[test]
    fn test_subtree_contraction_path() {
        let (tensor, ref_path) = setup_double_nested();
        let contraction_tree = ContractionTree::from_contraction_path(&tensor, &ref_path);
        // Subtree tensors:
        // 0: [4, 3, 2]
        // 1: [0, 1, 3, 2]
        // 3: [4, 5, 6]
        // 5: [10, 8, 9]
        let subtree_leaf_nodes = vec![0, 1, 3, 5];
        let (tree_contraction_path, local_contraction_path, flop_cost, mem_cost) =
            subtree_contraction_path(&subtree_leaf_nodes, &contraction_tree, &tensor);

        assert_eq!(
            tree_contraction_path,
            path![(0, 1), (0, 3), (0, 5)].to_vec(),
        );

        assert_eq!(
            local_contraction_path,
            path![(0, 1), (0, 2), (0, 3)].to_vec(),
        );

        assert_eq!(flop_cost, 8100.); // 120 + 420 + 7560
        assert_eq!(mem_cost, 1794.); // 84 + 630 +1080
    }

    #[test]
    fn test_subtree_tensor_network() {
        let (tensor, ref_path, bond_dims) = setup_complex();
        let contraction_tree = ContractionTree::from_contraction_path(&tensor, &ref_path);
        let node_id = 7;
        let (subtree_tensors, contraction_path) =
            subtree_tensor_network(node_id, &contraction_tree, &tensor);

        let tensor0 = Tensor::new_from_map(vec![4, 3, 2], &bond_dims);
        let tensor1 = Tensor::new_from_map(vec![0, 1, 3, 2], &bond_dims);
        let tensor2 = Tensor::new_from_map(vec![4, 5, 6], &bond_dims);
        let tensor3 = Tensor::new_from_map(vec![6, 8, 9], &bond_dims);
        let tensor4 = Tensor::new_from_map(vec![10, 8, 9], &bond_dims);
        let tensor5 = Tensor::new_from_map(vec![5, 1, 0], &bond_dims);

        let subtree7 = vec![tensor0, tensor1, tensor5];
        for (tensor, ref_tensor) in zip(subtree_tensors, subtree7) {
            assert_eq!(tensor.legs(), ref_tensor.legs());
        }

        assert_eq!(contraction_path, path![(1, 2), (0, 1)]);

        let node_id = 9;
        let (subtree_tensors, contraction_path) =
            subtree_tensor_network(node_id, &contraction_tree, &tensor);

        let subtree9 = vec![tensor2, tensor3, tensor4];
        for (tensor, ref_tensor) in zip(subtree_tensors, subtree9) {
            assert_eq!(tensor.legs(), ref_tensor.legs());
        }

        assert_eq!(contraction_path, path![(1, 2), (0, 1)]);
    }

    impl PartialEq for PartitionData {
        fn eq(&self, other: &Self) -> bool {
            self.id == other.id
                && self.flop_cost == other.flop_cost
                && self.contraction == other.contraction
                && self.local_tensor.legs() == other.local_tensor.legs()
        }
    }

    #[test]
    fn test_characterize_partitions() {
        let (tensor, ref_path, bond_dims) = setup_nested();
        let mut contraction_tree = ContractionTree::from_contraction_path(&tensor, &ref_path);
        contraction_tree.partitions.insert(1, vec![4, 9, 12]);
        let rebalance_depth = 1;
        let partition_data = characterize_partition(&contraction_tree, rebalance_depth, &tensor);
        let ref_partition_data = vec![
            PartitionData {
                id: 4,
                flop_cost: 84., // (0, 1, 2) + (1, 2, 3) = 84
                mem_cost: 44.,  // (0, 1) + (0, 2) + (0, 1, 2) = 44
                contraction: path![(0, 1), (0, 2)].to_vec(),
                local_tensor: Tensor::new_from_map(vec![1, 2, 3], &bond_dims),
            },
            PartitionData {
                id: 9,
                flop_cost: 3456., // (1, 2, 3, 4, 5, 8) + (1, 2, 3, 4, 5, 7, 8) = 3456
                mem_cost: 864.,   // (1, 2, 3, 4, 5, 8) + (4, 7, 8) + (1, 2, 3, 5) = 864
                contraction: path![(0, 1), (0, 2)].to_vec(),
                local_tensor: Tensor::new_from_map(vec![2, 1, 3, 5, 7], &bond_dims),
            },
            PartitionData {
                id: 12,
                flop_cost: 140., // (5, 6, 7) = 140
                mem_cost: 167.,  // (5, 6, 7) + (6) + (5, 7) = 167
                contraction: path![(0, 1)].to_vec(),
                local_tensor: Tensor::new_from_map(vec![5, 7], &bond_dims),
            },
        ];
        assert_eq!(partition_data, ref_partition_data);
    }
}
