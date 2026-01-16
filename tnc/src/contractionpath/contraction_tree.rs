use std::cell::Ref;
use std::rc::Rc;

use itertools::Itertools;
use rustc_hash::FxHashMap;

use crate::contractionpath::contraction_tree::node::{
    child_node, parent_node, Node, NodeRef, WeakNodeRef,
};
use crate::contractionpath::paths::validate_path;
use crate::contractionpath::{ContractionPath, SimplePath, SimplePathRef};
use crate::tensornetwork::tensor::Tensor;

pub mod balancing;
pub mod export;
pub mod import;
mod node;
mod utils;

/// Struct representing the full contraction path of a given [`Tensor`] object.
#[derive(Default, Debug, Clone)]
pub struct ContractionTree {
    nodes: FxHashMap<usize, NodeRef>,
    partitions: FxHashMap<usize, Vec<usize>>,
    root: WeakNodeRef,
}

impl ContractionTree {
    /// Returns a reference to the node with the given `node_id`.
    pub fn node(&self, node_id: usize) -> Ref<'_, Node> {
        let borrow = &self.nodes[&node_id];
        borrow.as_ref().borrow()
    }

    /// Returns the node id of the root node, if any.
    pub fn root_id(&self) -> Option<usize> {
        self.root.upgrade().map(|node| node.borrow().id())
    }

    pub const fn partitions(&self) -> &FxHashMap<usize, Vec<usize>> {
        &self.partitions
    }

    /// Populates `nodes` and `partitions` with the tree structure of the contraction
    /// `path`.
    fn from_contraction_path_recurse(
        tensor: &Tensor,
        path: &ContractionPath,
        nodes: &mut FxHashMap<usize, NodeRef>,
        partitions: &mut FxHashMap<usize, Vec<usize>>,
        prefix: &[usize],
    ) {
        let mut scratch = FxHashMap::default();

        // Obtain tree structure from composite tensors
        for (path_id, path) in path.nested.iter().sorted_by_key(|&(path_id, _)| *path_id) {
            let composite_tensor = tensor.tensor(*path_id);
            let mut new_prefix = prefix.to_owned();
            new_prefix.push(*path_id);
            Self::from_contraction_path_recurse(
                composite_tensor,
                path,
                nodes,
                partitions,
                &new_prefix,
            );
            scratch.insert(*path_id, Rc::clone(&nodes[&(nodes.len() - 1)]));
        }

        // Add nodes for leaf tensors
        for (tensor_idx, tensor) in tensor.tensors().iter().enumerate() {
            if tensor.is_leaf() {
                let mut nested_tensor_idx = prefix.to_owned();
                nested_tensor_idx.push(tensor_idx);
                let new_node = child_node(nodes.len(), nested_tensor_idx);
                scratch.insert(tensor_idx, Rc::clone(&new_node));
                nodes.insert(nodes.len(), new_node);
            }
        }

        // Build tree based on contraction path
        for (i_path, j_path) in &path.toplevel {
            let i = &scratch[i_path];
            let j = &scratch[j_path];
            let parent = parent_node(nodes.len(), i, j);

            scratch.insert(*i_path, Rc::clone(&parent));
            nodes.insert(nodes.len(), parent);
            scratch.remove(j_path);
        }
        partitions
            .entry(prefix.len())
            .or_default()
            .push(nodes.len() - 1);
    }

    /// Creates a `ContractionTree` from `tensor` and contract `path`. The tree
    /// represents all intermediate tensors and costs of given contraction path and
    /// tensor network.
    #[must_use]
    pub fn from_contraction_path(tensor: &Tensor, path: &ContractionPath) -> Self {
        validate_path(path);
        let mut nodes = FxHashMap::default();
        let mut partitions = FxHashMap::default();
        Self::from_contraction_path_recurse(tensor, path, &mut nodes, &mut partitions, &[]);
        let root = Rc::downgrade(&nodes[&(nodes.len() - 1)]);
        Self {
            nodes,
            partitions,
            root,
        }
    }

    fn leaf_ids_recurse(node: &Node, leaf_indices: &mut Vec<usize>) {
        if node.is_leaf() {
            leaf_indices.push(node.id());
        } else {
            Self::leaf_ids_recurse(&node.left_child().unwrap().as_ref().borrow(), leaf_indices);
            Self::leaf_ids_recurse(&node.right_child().unwrap().as_ref().borrow(), leaf_indices);
        }
    }

    /// Returns the id of all leaf nodes in subtree with root at `node_id`.
    pub fn leaf_ids(&self, node_id: usize) -> Vec<usize> {
        let mut leaf_indices = Vec::new();
        let node = self.node(node_id);
        Self::leaf_ids_recurse(&node, &mut leaf_indices);
        leaf_indices
    }

    /// Removes subtree with root at `node_id`.
    fn remove_subtree_recurse(&mut self, node_id: usize) {
        if self.node(node_id).is_leaf() {
            // Leaf nodes are not removed. We need to manually clear parent/children relations
            let node = &self.nodes[&node_id];
            if let Some(parent_id) = node.borrow().parent_id() {
                if self.nodes.contains_key(&parent_id) {
                    self.nodes[&parent_id]
                        .borrow_mut()
                        .remove_child(node.borrow().id());
                }
            }
            node.borrow_mut().remove_parent();
            return;
        }

        let node = self.nodes.remove(&node_id).unwrap();
        let node = node.borrow();

        if let Some(id) = node.left_child_id() {
            self.remove_subtree_recurse(id);
        }
        if let Some(id) = node.right_child_id() {
            self.remove_subtree_recurse(id);
        }
    }

    /// Removes subtree with root at `node_id`.
    pub(crate) fn remove_subtree(&mut self, node_id: usize) {
        self.remove_subtree_recurse(node_id);
    }

    /// Converts a contraction path into a ContractionTree, then attaches this as a subtree at `parent_id`
    /// The ContractionTree should already contain the leaf nodes of the
    pub(crate) fn add_path_as_subtree(
        &mut self,
        path: &ContractionPath,
        parent_id: usize,
        leaf_tensor_indices: &[usize],
    ) -> usize {
        validate_path(path);
        assert!(self.nodes.contains_key(&parent_id));

        let mut index = 0;
        // Utilize a scratch hashmap to store intermediate tensor information
        let mut scratch = FxHashMap::default();

        // Fill scratch with leaf tensors, these should already be present in self.nodes.
        for &tensor_index in leaf_tensor_indices {
            scratch.insert(tensor_index, Rc::clone(&self.nodes[&tensor_index]));
        }

        // Generate intermediate tensors by looping over contraction operations, fill and update scratch as needed.
        assert!(
            path.is_simple(),
            "Constructor not implemented for nested Tensors"
        );
        for (i_path, j_path) in &path.toplevel {
            // Always keep track of latest added tensor. Last index will be the root of the subtree.
            index = self.next_id(index);
            let i = &scratch[i_path];
            let j = &scratch[j_path];

            // Ensure that we are not reusing nodes that are already in another contraction path
            assert!(
                i.borrow().parent_id().is_none(),
                "Tensor {i_path} is already used in another contraction"
            );
            assert!(
                j.borrow().parent_id().is_none(),
                "Tensor {j_path} is already used in another contraction"
            );

            let parent = parent_node(index, i, j);
            scratch.insert(*i_path, Rc::clone(&parent));
            scratch.remove(j_path);
            // Ensure that intermediate tensor information is stored in internal HashMap for reference
            self.nodes.insert(index, parent);
        }

        // Add the root of the subtree to the indicated node `parent_id` in larger contraction tree.
        let new_parent = &self.nodes[&parent_id];
        new_parent
            .borrow_mut()
            .add_child(Rc::downgrade(&self.nodes[&index]));

        let new_child = &self.nodes[&index];
        new_child.borrow_mut().set_parent(Rc::downgrade(new_parent));

        index
    }

    fn remove_communication_path(&mut self, partition_ids: &[usize]) {
        for partition_id in partition_ids {
            let mut parent_id = self.node(*partition_id).parent_id();
            while let Some(tensor_id) = parent_id {
                parent_id = self.node(tensor_id).parent_id();
                self.nodes.remove(&tensor_id);
            }
        }
    }

    fn replace_communication_path(
        &mut self,
        partition_ids: Vec<usize>,
        communication_path: SimplePathRef,
    ) {
        // Remove all nodes involved in communication path
        self.remove_communication_path(&partition_ids);

        // Rebuild the communication-part of the tree
        let mut communication_ids = partition_ids;
        let mut next_id = self.next_id(0);
        for (i, j) in communication_path {
            let left_child = communication_ids[*i];
            let right_child = communication_ids[*j];
            let new_parent =
                parent_node(next_id, &self.nodes[&left_child], &self.nodes[&right_child]);
            self.nodes.insert(next_id, new_parent);

            communication_ids[*i] = next_id;
            next_id = self.next_id(next_id);
        }

        // Update root
        self.root = Rc::downgrade(self.nodes.iter().max_by_key(|entry| entry.0).unwrap().1);
    }

    fn tree_weights_recurse(
        node: &Node,
        tn: &Tensor,
        weights: &mut FxHashMap<usize, f64>,
        scratch: &mut FxHashMap<usize, Tensor>,
        cost_function: fn(&Tensor, &Tensor) -> f64,
    ) {
        if node.is_leaf() {
            let Some(tensor_index) = &node.tensor_index() else {
                panic!("All leaf nodes should have a tensor index")
            };
            weights.insert(node.id(), 0f64);
            scratch.insert(node.id(), tn.nested_tensor(tensor_index).clone());
            return;
        }

        let left_child = &node.left_child().unwrap();
        let right_child = &node.right_child().unwrap();
        let left_ref = left_child.as_ref().borrow();
        let right_ref = right_child.as_ref().borrow();

        // Recurse first because weights of leaves are needed for further computation.
        Self::tree_weights_recurse(&left_ref, tn, weights, scratch, cost_function);
        Self::tree_weights_recurse(&right_ref, tn, weights, scratch, cost_function);

        let t1 = &scratch[&left_ref.id()];
        let t2 = &scratch[&right_ref.id()];

        let cost = weights[&left_ref.id()] + weights[&right_ref.id()] + cost_function(t1, t2);

        weights.insert(node.id(), cost);
        scratch.insert(node.id(), t1 ^ t2);
    }

    /// Returns `HashMap` storing resultant tensor and its respective contraction costs calculated via `cost_function`.
    ///
    /// # Arguments
    /// * `node_id` - root of Node to start calculating contraction costs
    /// * `tn` - [`Tensor`] object containing bond dimension and leaf node information
    /// * `cost_function` - cost function returning contraction cost
    pub fn tree_weights(
        &self,
        node_id: usize,
        tn: &Tensor,
        cost_function: fn(&Tensor, &Tensor) -> f64,
    ) -> FxHashMap<usize, f64> {
        let mut weights = FxHashMap::default();
        let mut scratch = FxHashMap::default();
        let node = self.node(node_id);
        Self::tree_weights_recurse(&node, tn, &mut weights, &mut scratch, cost_function);
        weights
    }

    /// Populates given vector with contractions path of contraction tree starting at `node`.
    ///
    /// # Arguments
    /// * `node` - pointer to Node object
    /// * `path` - vec to store contraction path in
    /// * `replace` - if set to `true` returns replace path, otherwise, returns in SSA format
    fn to_contraction_path_recurse(node: &Node, path: &mut SimplePath, replace: bool) -> usize {
        if node.is_leaf() {
            return node.id();
        }

        // Get children
        let (Some(left_child), Some(right_child)) = (node.left_child(), node.right_child()) else {
            panic!("All parents should have two children")
        };

        // Get right and left child tensor ids
        let mut t1_id =
            Self::to_contraction_path_recurse(&left_child.as_ref().borrow(), path, replace);
        let mut t2_id =
            Self::to_contraction_path_recurse(&right_child.as_ref().borrow(), path, replace);
        if t2_id < t1_id {
            (t1_id, t2_id) = (t2_id, t1_id);
        }

        // Add pair to path
        path.push((t1_id, t2_id));

        // Return id of contracted tensor
        if replace {
            t1_id
        } else {
            node.id()
        }
    }

    /// Populates given vector with contractions path of contraction tree starting at `node_id`.
    /// # Arguments
    /// * `node` - pointer to Node object
    /// * `replace` - if set to `true` returns replace path, otherwise, returns in SSA format
    pub fn to_flat_contraction_path(&self, node_id: usize, replace: bool) -> SimplePath {
        let node = self.node(node_id);
        let mut path = Vec::new();
        Self::to_contraction_path_recurse(&node, &mut path, replace);
        path
    }

    fn next_id(&self, mut init: usize) -> usize {
        while self.nodes.contains_key(&init) {
            init += 1;
        }
        init
    }

    /// Returns intermediate [`Tensor`] object corresponding to `node_id`.
    ///
    /// # Arguments
    /// * `node_id` - id of Node corresponding to [`Tensor`] of interest
    /// * `tensor` - tensor containing bond dimension and leaf node information
    ///
    /// # Returns
    /// Empty tensor with legs (dimensions) of data after fully contracted.
    pub fn tensor(&self, node_id: usize, tensor: &Tensor) -> Tensor {
        let leaf_nodes = self.leaf_ids(node_id);
        let mut new_tensor = Tensor::default();

        for leaf_id in leaf_nodes {
            new_tensor = &new_tensor
                ^ tensor.nested_tensor(self.node(leaf_id).tensor_index().as_ref().unwrap());
        }
        new_tensor
    }
}

fn populate_subtree_tensor_map_recursive(
    contraction_tree: &ContractionTree,
    node_id: usize,
    node_tensor_map: &mut FxHashMap<usize, Tensor>,
    tensor_network: &Tensor,
    height_limit: Option<usize>,
) -> (Tensor, usize) {
    let node = contraction_tree.node(node_id);

    if node.is_leaf() {
        let tensor_index = node.tensor_index().as_ref().unwrap();
        let t = tensor_network.nested_tensor(tensor_index);
        node_tensor_map.insert(node.id(), t.clone());
        (t.clone(), 0)
    } else {
        let (t1, new_height1) = populate_subtree_tensor_map_recursive(
            contraction_tree,
            node.left_child_id().unwrap(),
            node_tensor_map,
            tensor_network,
            height_limit,
        );
        let (t2, new_height2) = populate_subtree_tensor_map_recursive(
            contraction_tree,
            node.right_child_id().unwrap(),
            node_tensor_map,
            tensor_network,
            height_limit,
        );
        let t12 = &t1 ^ &t2;
        if let Some(height_limit) = height_limit {
            if new_height1 < height_limit && new_height2 < height_limit {
                node_tensor_map.insert(node.id(), t12.clone());
            }
        } else {
            node_tensor_map.insert(node.id(), t12.clone());
        }

        (t12, new_height1.max(new_height2) + 1)
    }
}

/// Populates `node_tensor_map` with all intermediate and leaf node ids and corresponding [`Tensor`] object, with root at `node_id`.
/// Only inserts Tensors with up to `height_limit` number of contractions.
///
/// # Arguments
/// * `contraction_tree` - [`ContractionTree`] object
/// * `node_id` - root of subtree to examine
/// * `node_tensor_map` - empty HashMap to populate
/// * `tensor_network` - [`Tensor`] object containing bond dimension and leaf node information
///
///
/// # Returns
/// Populated HashMap mapping intermediate node ids up to `height_limit` to Tensor objects.
fn populate_subtree_tensor_map(
    contraction_tree: &ContractionTree,
    node_id: usize,
    tensor_network: &Tensor,
    height_limit: Option<usize>,
) -> FxHashMap<usize, Tensor> {
    let mut node_tensor_map = FxHashMap::default();
    let _ = populate_subtree_tensor_map_recursive(
        contraction_tree,
        node_id,
        &mut node_tensor_map,
        tensor_network,
        height_limit,
    );
    node_tensor_map
}

/// Populates `node_tensor_map` with all leaf node ids and corresponding [`Tensor`] object, with root at `node_id`.
///
/// # Arguments
/// * `contraction_tree` - [`ContractionTree`] object
/// * `node_id` - root of subtree to examine
/// * `tensor_network` - [`Tensor`] object containing bond dimension and leaf node information
///
/// # Returns
/// Populated HashMap mapping leaf node ids to Tensor objects.
fn populate_leaf_node_tensor_map(
    contraction_tree: &ContractionTree,
    node_id: usize,
    tensor_network: &Tensor,
) -> FxHashMap<usize, Tensor> {
    let mut node_tensor_map = FxHashMap::default();
    for leaf_node_id in contraction_tree.leaf_ids(node_id) {
        node_tensor_map.insert(
            leaf_node_id,
            contraction_tree.tensor(leaf_node_id, tensor_network),
        );
    }
    node_tensor_map
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::cell::RefCell;
    use std::iter::zip;
    use std::rc::Weak;

    use itertools::Itertools;

    use crate::contractionpath::contraction_cost::contract_cost_tensors;
    use crate::contractionpath::contraction_tree::node::{child_node, parent_node};
    use crate::contractionpath::contraction_tree::{ContractionTree, Node};
    use crate::contractionpath::ssa_replace_ordering;
    use crate::path;
    use crate::tensornetwork::tensor::{EdgeIndex, Tensor};

    fn setup_simple() -> (Tensor, ContractionPath, FxHashMap<EdgeIndex, u64>) {
        let bond_dims =
            FxHashMap::from_iter([(0, 5), (1, 2), (2, 6), (3, 8), (4, 1), (5, 3), (6, 4)]);
        (
            Tensor::new_composite(vec![
                Tensor::new_from_map(vec![4, 3, 2], &bond_dims),
                Tensor::new_from_map(vec![0, 1, 3, 2], &bond_dims),
                Tensor::new_from_map(vec![4, 5, 6], &bond_dims),
            ]),
            path![(0, 1), (2, 0)],
            bond_dims,
        )
    }

    fn setup_complex() -> (Tensor, ContractionPath, FxHashMap<EdgeIndex, u64>) {
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
            path![(1, 5), (0, 1), (3, 4), (2, 3), (0, 2)],
            bond_dims,
        )
    }

    fn setup_unbalanced() -> (Tensor, ContractionPath) {
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
        (
            Tensor::new_composite(vec![
                Tensor::new_from_map(vec![4, 3, 2], &bond_dims),
                Tensor::new_from_map(vec![0, 1, 3, 2], &bond_dims),
                Tensor::new_from_map(vec![4, 5, 6], &bond_dims),
                Tensor::new_from_map(vec![6, 8, 9], &bond_dims),
                Tensor::new_from_map(vec![10, 8, 9], &bond_dims),
                Tensor::new_from_map(vec![5, 1, 0], &bond_dims),
            ]),
            path![(0, 1), (2, 0), (3, 2), (4, 3), (5, 4)],
        )
    }

    fn setup_nested() -> (Tensor, ContractionPath) {
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

        let t0 = Tensor::new_from_map(vec![4, 3, 2], &bond_dims);
        let t1 = Tensor::new_from_map(vec![0, 1, 3, 2], &bond_dims);
        let t2 = Tensor::new_from_map(vec![4, 5, 6], &bond_dims);
        let t3 = Tensor::new_from_map(vec![6, 8, 9], &bond_dims);
        let t4 = Tensor::new_from_map(vec![5, 1, 0], &bond_dims);
        let t5 = Tensor::new_from_map(vec![10, 8, 9], &bond_dims);

        let t01 = Tensor::new_composite(vec![t0, t1]);
        let t23 = Tensor::new_composite(vec![t2, t3]);
        let t45 = Tensor::new_composite(vec![t4, t5]);
        let tensor_network = Tensor::new_composite(vec![t01, t23, t45]);
        (
            tensor_network,
            path![{(0, [(0, 1)]), (1, [(0, 1)]), (2, [(0, 1)])}, (0, 1), (0, 2)],
        )
    }

    fn setup_double_nested() -> (Tensor, ContractionPath) {
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
                {
                (0, [{(0, [(0, 1)])}, (0, 1)]),
                (1, [{(0, [(0, 1)])}, (0, 1)]),
                },
                (0, 1)
            ],
        )
    }

    impl PartialEq for Node {
        fn eq(&self, other: &Self) -> bool {
            self.id() == other.id()
                && self.left_child_id() == other.left_child_id()
                && self.right_child_id() == other.right_child_id()
                && self.parent_id() == other.parent_id()
                && self.tensor_index() == other.tensor_index()
        }
    }

    #[test]
    fn test_from_contraction_path_simple() {
        let (tensor, path, _) = setup_simple();
        let ContractionTree { nodes, root, .. } =
            ContractionTree::from_contraction_path(&tensor, &path);

        let node0 = child_node(0, vec![0]);
        let node1 = child_node(1, vec![1]);
        let node2 = child_node(2, vec![2]);

        let node3 = parent_node(3, &node0, &node1);
        let node4 = parent_node(4, &node2, &node3);

        let ref_root = Rc::clone(&node4);
        let ref_nodes = [node0, node1, node2, node3, node4];

        for (key, ref_node) in ref_nodes.iter().enumerate() {
            let node = &nodes[&key];
            assert_eq!(node, ref_node);
        }
        assert_eq!(root.upgrade().unwrap(), ref_root);
    }

    #[test]
    fn test_from_contraction_path_complex() {
        let (tensor, path, _) = setup_complex();
        let ContractionTree { nodes, root, .. } =
            ContractionTree::from_contraction_path(&tensor, &path);

        let node0 = child_node(0, vec![0]);
        let node1 = child_node(1, vec![1]);
        let node2 = child_node(2, vec![2]);
        let node3 = child_node(3, vec![3]);
        let node4 = child_node(4, vec![4]);
        let node5 = child_node(5, vec![5]);

        let node6 = parent_node(6, &node1, &node5);
        let node7 = parent_node(7, &node0, &node6);
        let node8 = parent_node(8, &node3, &node4);
        let node9 = parent_node(9, &node2, &node8);
        let node10 = parent_node(10, &node7, &node9);

        let ref_root = Rc::clone(&node10);
        let ref_nodes = [
            node0, node1, node2, node3, node4, node5, node6, node7, node8, node9, node10,
        ];

        for (key, ref_node) in ref_nodes.iter().enumerate() {
            let node = &nodes[&key];
            assert_eq!(node, ref_node);
        }
        assert_eq!(root.upgrade().unwrap(), ref_root);
    }

    #[test]
    fn test_from_contraction_path_nested() {
        let (tensor, path) = setup_nested();
        let ContractionTree { nodes, root, .. } =
            ContractionTree::from_contraction_path(&tensor, &path);

        let node0 = child_node(0, vec![0, 0]);
        let node1 = child_node(1, vec![0, 1]);
        let node3 = child_node(3, vec![1, 0]);
        let node4 = child_node(4, vec![1, 1]);
        let node6 = child_node(6, vec![2, 0]);
        let node7 = child_node(7, vec![2, 1]);

        let node2 = parent_node(2, &node0, &node1);
        let node5 = parent_node(5, &node3, &node4);
        let node8 = parent_node(8, &node6, &node7);
        let node9 = parent_node(9, &node2, &node5);
        let node10 = parent_node(10, &node9, &node8);

        let ref_root = Rc::clone(&node10);
        let ref_nodes = [
            node0, node1, node2, node3, node4, node5, node6, node7, node8, node9, node10,
        ];

        for (key, ref_node) in ref_nodes.iter().enumerate() {
            let node = &nodes[&key];
            assert_eq!(node, ref_node);
        }
        assert_eq!(root.upgrade().unwrap(), ref_root);
    }

    #[test]
    fn test_from_contraction_path_double_nested() {
        let (tensor, path) = setup_double_nested();
        let ContractionTree { nodes, root, .. } =
            ContractionTree::from_contraction_path(&tensor, &path);

        let node0 = child_node(0, vec![0, 0, 0]);
        let node1 = child_node(1, vec![0, 0, 1]);
        let node3 = child_node(3, vec![0, 1]);
        let node5 = child_node(5, vec![1, 0, 0]);
        let node6 = child_node(6, vec![1, 0, 1]);
        let node8 = child_node(8, vec![1, 1]);

        let node2 = parent_node(2, &node0, &node1);
        let node4 = parent_node(4, &node2, &node3);
        let node7 = parent_node(7, &node5, &node6);
        let node9 = parent_node(9, &node7, &node8);
        let node10 = parent_node(10, &node4, &node9);

        let ref_root = Rc::clone(&node10);
        let ref_nodes = [
            node0, node1, node2, node3, node4, node5, node6, node7, node8, node9, node10,
        ];

        for (key, ref_node) in ref_nodes.iter().enumerate() {
            let node = &nodes[&key];
            assert_eq!(node, ref_node);
        }
        assert_eq!(root.upgrade().unwrap(), ref_root);
    }

    #[test]
    fn test_leaf_ids_simple() {
        let (tn, path, _) = setup_simple();
        let tree = ContractionTree::from_contraction_path(&tn, &path);

        assert_eq!(tree.leaf_ids(4), vec![2, 0, 1]);
        assert_eq!(tree.leaf_ids(3), vec![0, 1]);
        assert_eq!(tree.leaf_ids(2), vec![2]);
    }

    #[test]
    fn test_leaf_ids_complex() {
        let (tn, path, _) = setup_complex();
        let tree = ContractionTree::from_contraction_path(&tn, &path);

        assert_eq!(tree.leaf_ids(10), vec![0, 1, 5, 2, 3, 4]);
        assert_eq!(tree.leaf_ids(9), vec![2, 3, 4]);
        assert_eq!(tree.leaf_ids(8), vec![3, 4]);
        assert_eq!(tree.leaf_ids(7), vec![0, 1, 5]);
        assert_eq!(tree.leaf_ids(6), vec![1, 5]);
        assert_eq!(tree.leaf_ids(3), vec![3]);
    }

    #[test]
    fn test_leaf_ids_nested() {
        let (tn, path) = setup_nested();
        let tree = ContractionTree::from_contraction_path(&tn, &path);
        assert_eq!(tree.leaf_ids(10), vec![0, 1, 3, 4, 6, 7]);
        assert_eq!(tree.leaf_ids(9), vec![0, 1, 3, 4]);
        assert_eq!(tree.leaf_ids(8), vec![6, 7]);
        assert_eq!(tree.leaf_ids(5), vec![3, 4]);
        assert_eq!(tree.leaf_ids(2), vec![0, 1]);
    }

    #[test]
    fn test_leaf_ids_double_nested() {
        let (tn, path) = setup_double_nested();
        let tree = ContractionTree::from_contraction_path(&tn, &path);

        assert_eq!(tree.leaf_ids(10), vec![0, 1, 3, 5, 6, 8]);
        assert_eq!(tree.leaf_ids(9), vec![5, 6, 8]);
        assert_eq!(tree.leaf_ids(7), vec![5, 6]);
        assert_eq!(tree.leaf_ids(4), vec![0, 1, 3]);
        assert_eq!(tree.leaf_ids(2), vec![0, 1]);
    }

    #[test]
    fn test_remove_subtree() {
        let (tn, path) = setup_nested();
        let mut tree = ContractionTree::from_contraction_path(&tn, &path);

        tree.remove_subtree(8);

        let ContractionTree { nodes, root, .. } = tree;

        let node0 = child_node(0, vec![0, 0]);
        let node1 = child_node(1, vec![0, 1]);
        let node3 = child_node(3, vec![1, 0]);
        let node4 = child_node(4, vec![1, 1]);
        let node6 = child_node(6, vec![2, 0]);
        let node7 = child_node(7, vec![2, 1]);
        let node2 = parent_node(2, &node0, &node1);
        let node5 = parent_node(5, &node3, &node4);
        let node9 = parent_node(9, &node2, &node5);
        let node10 = Rc::new(RefCell::new(Node::new(
            10,
            Rc::downgrade(&node9),
            Weak::new(),
            Weak::new(),
            None,
        )));
        node9.borrow_mut().set_parent(Rc::downgrade(&node10));

        let ref_root = Rc::clone(&node10);
        let ref_nodes = [
            node0, node1, node2, node3, node4, node5, node6, node7, node9, node10,
        ];
        let mut range = (0..8).collect_vec();
        range.extend(9..11);
        for (key, ref_node) in zip(range.iter(), ref_nodes.iter()) {
            let node = &nodes[key];
            assert_eq!(node, ref_node);
        }
        assert_eq!(root.upgrade().unwrap(), ref_root);
    }

    #[test]
    fn test_remove_trivial_subtree() {
        let (tensor, path) = setup_nested();
        let mut tree = ContractionTree::from_contraction_path(&tensor, &path);

        tree.remove_subtree(7);

        let ContractionTree { nodes, root, .. } = tree;

        let node0 = child_node(0, vec![0, 0]);
        let node1 = child_node(1, vec![0, 1]);
        let node3 = child_node(3, vec![1, 0]);
        let node4 = child_node(4, vec![1, 1]);
        let node6 = child_node(6, vec![2, 0]);
        let node7 = child_node(7, vec![2, 1]);
        let node2 = parent_node(2, &node0, &node1);
        let node5 = parent_node(5, &node3, &node4);
        let node8 = Rc::new(RefCell::new(Node::new(
            8,
            Rc::downgrade(&node6),
            Weak::new(),
            Weak::new(),
            None,
        )));
        let node9 = parent_node(9, &node2, &node5);
        let node10 = parent_node(10, &node9, &node8);
        node6.borrow_mut().set_parent(Rc::downgrade(&node8));

        let ref_root = Rc::clone(&node10);
        let ref_nodes = [
            node0, node1, node2, node3, node4, node5, node6, node7, node8, node9, node10,
        ];

        for (key, ref_node) in ref_nodes.iter().enumerate() {
            let node = &nodes[&key];
            assert_eq!(node, ref_node);
        }
        assert_eq!(root.upgrade().unwrap(), ref_root);
    }

    #[test]
    fn test_tree_weights_simple() {
        let (tensor, path, _) = setup_simple();
        let tree = ContractionTree::from_contraction_path(&tensor, &path);
        let ref_weights = FxHashMap::from_iter([(1, 0.), (0, 0.), (2, 0.), (3, 3820.), (4, 4540.)]);
        let weights = tree.tree_weights(4, &tensor, contract_cost_tensors);

        assert_eq!(weights, ref_weights);
        let ref_weights = FxHashMap::from_iter([(1, 0.), (0, 0.), (3, 3820.)]);
        let weights = tree.tree_weights(3, &tensor, contract_cost_tensors);
        assert_eq!(weights, ref_weights);

        assert_eq!(weights, ref_weights);
        let ref_weights = FxHashMap::from_iter([(2, 0.)]);
        let weights = tree.tree_weights(2, &tensor, contract_cost_tensors);
        assert_eq!(weights, ref_weights);
    }

    #[test]
    fn test_tree_weights_complex() {
        let (tensor, path, _) = setup_complex();
        let tree = ContractionTree::from_contraction_path(&tensor, &path);
        let ref_weights = FxHashMap::from_iter([
            (0, 0.),
            (1, 0.),
            (2, 0.),
            (3, 0.),
            (4, 0.),
            (5, 0.),
            (6, 2098440.),
            (7, 2120010.),
            (8, 2105820.),
            (9, 2116470.),
            (10, 4237070.),
        ]);
        let weights = tree.tree_weights(10, &tensor, contract_cost_tensors);

        assert_eq!(weights, ref_weights);
    }

    #[test]
    fn test_to_contraction_path_simple() {
        let (tensor, ref_path, _) = setup_simple();
        let tree = ContractionTree::from_contraction_path(&tensor, &ref_path);
        let path = tree.to_flat_contraction_path(4, false);
        let path = ssa_replace_ordering(&ContractionPath::simple(path));
        assert_eq!(path, ref_path);
    }

    #[test]
    fn test_to_contraction_path_complex() {
        let (tensor, ref_path, _) = setup_complex();
        let tree = ContractionTree::from_contraction_path(&tensor, &ref_path);
        let path = tree.to_flat_contraction_path(10, false);
        let path = ssa_replace_ordering(&ContractionPath::simple(path));
        assert_eq!(path, ref_path);
    }

    #[test]
    fn test_to_contraction_path_unbalanced() {
        let (tensor, ref_path) = setup_unbalanced();
        let tree = ContractionTree::from_contraction_path(&tensor, &ref_path);
        let path = tree.to_flat_contraction_path(10, false);
        let path = ssa_replace_ordering(&ContractionPath::simple(path));
        assert_eq!(path, ref_path);
    }

    #[test]
    fn test_populate_subtree_tensor_map_simple() {
        let (tensor, ref_path, bond_dims) = setup_simple();
        let tree = ContractionTree::from_contraction_path(&tensor, &ref_path);
        let mut node_tensor_map = FxHashMap::default();
        populate_subtree_tensor_map_recursive(&tree, 4, &mut node_tensor_map, &tensor, None);

        let ref_node_tensor_map = FxHashMap::from_iter([
            (0, Tensor::new_from_map(vec![4, 3, 2], &bond_dims)),
            (1, Tensor::new_from_map(vec![0, 1, 3, 2], &bond_dims)),
            (2, Tensor::new_from_map(vec![4, 5, 6], &bond_dims)),
            (3, Tensor::new_from_map(vec![4, 0, 1], &bond_dims)),
            (4, Tensor::new_from_map(vec![5, 6, 0, 1], &bond_dims)),
        ]);

        for (key, value) in ref_node_tensor_map {
            assert_eq!(node_tensor_map[&key].legs(), value.legs());
        }
    }

    #[test]
    fn test_populate_subtree_tensor_map_complex() {
        let (tensor, ref_path, bond_dims) = setup_complex();
        let tree = ContractionTree::from_contraction_path(&tensor, &ref_path);
        let mut node_tensor_map = FxHashMap::default();
        populate_subtree_tensor_map_recursive(&tree, 10, &mut node_tensor_map, &tensor, None);

        let ref_node_tensor_map = FxHashMap::from_iter([
            (0, Tensor::new_from_map(vec![4, 3, 2], &bond_dims)),
            (1, Tensor::new_from_map(vec![0, 1, 3, 2], &bond_dims)),
            (2, Tensor::new_from_map(vec![4, 5, 6], &bond_dims)),
            (3, Tensor::new_from_map(vec![6, 8, 9], &bond_dims)),
            (4, Tensor::new_from_map(vec![10, 8, 9], &bond_dims)),
            (5, Tensor::new_from_map(vec![5, 1, 0], &bond_dims)),
            (6, Tensor::new_from_map(vec![3, 2, 5], &bond_dims)),
            (7, Tensor::new_from_map(vec![4, 5], &bond_dims)),
            (8, Tensor::new_from_map(vec![6, 10], &bond_dims)),
            (9, Tensor::new_from_map(vec![4, 5, 10], &bond_dims)),
            (10, Tensor::new_from_map(vec![10], &bond_dims)),
        ]);

        for (key, value) in ref_node_tensor_map {
            assert_eq!(node_tensor_map[&key].legs(), value.legs());
        }
    }

    #[test]
    fn test_populate_subtree_tensor_map_height_limit() {
        let (tensor, ref_path, bond_dims) = setup_complex();
        let tree = ContractionTree::from_contraction_path(&tensor, &ref_path);
        let node_tensor_map = populate_subtree_tensor_map(&tree, 10, &tensor, Some(1));

        let ref_node_tensor_map = FxHashMap::from_iter([
            (0, Tensor::new_from_map(vec![4, 3, 2], &bond_dims)),
            (1, Tensor::new_from_map(vec![0, 1, 3, 2], &bond_dims)),
            (2, Tensor::new_from_map(vec![4, 5, 6], &bond_dims)),
            (3, Tensor::new_from_map(vec![6, 8, 9], &bond_dims)),
            (4, Tensor::new_from_map(vec![10, 8, 9], &bond_dims)),
            (5, Tensor::new_from_map(vec![5, 1, 0], &bond_dims)),
            (6, Tensor::new_from_map(vec![3, 2, 5], &bond_dims)),
            (8, Tensor::new_from_map(vec![6, 10], &bond_dims)),
        ]);

        for (key, value) in ref_node_tensor_map {
            assert_eq!(node_tensor_map[&key].legs(), value.legs());
        }
    }

    #[test]
    fn test_populate_leaf_node_tensor_map_simple() {
        let (tensor, ref_path, bond_dims) = setup_simple();
        let tree = ContractionTree::from_contraction_path(&tensor, &ref_path);

        let node_tensor_map = populate_leaf_node_tensor_map(&tree, 4, &tensor);

        let ref_node_tensor_map = FxHashMap::from_iter([
            (0, Tensor::new_from_map(vec![4, 3, 2], &bond_dims)),
            (1, Tensor::new_from_map(vec![0, 1, 3, 2], &bond_dims)),
            (2, Tensor::new_from_map(vec![4, 5, 6], &bond_dims)),
        ]);

        for (key, value) in ref_node_tensor_map {
            assert_eq!(node_tensor_map[&key].legs(), value.legs());
        }
    }

    #[test]
    fn test_populate_leaf_node_tensor_map_complex() {
        let (tensor, ref_path, bond_dims) = setup_complex();
        let tree = ContractionTree::from_contraction_path(&tensor, &ref_path);
        let node_tensor_map = populate_subtree_tensor_map(&tree, 10, &tensor, None);

        let ref_node_tensor_map = FxHashMap::from_iter([
            (0, Tensor::new_from_map(vec![4, 3, 2], &bond_dims)),
            (1, Tensor::new_from_map(vec![0, 1, 3, 2], &bond_dims)),
            (2, Tensor::new_from_map(vec![4, 5, 6], &bond_dims)),
            (3, Tensor::new_from_map(vec![6, 8, 9], &bond_dims)),
            (4, Tensor::new_from_map(vec![10, 8, 9], &bond_dims)),
        ]);

        for (key, value) in ref_node_tensor_map {
            assert_eq!(node_tensor_map[&key].legs(), value.legs());
        }
    }

    #[test]
    fn test_add_path_as_subtree() {
        let (tensor, path, _) = setup_complex();

        let mut complex_tree = ContractionTree::from_contraction_path(&tensor, &path);
        complex_tree.remove_subtree(9);
        let new_path = path![(4, 2), (4, 3)];

        complex_tree.add_path_as_subtree(&new_path, 10, &[3, 4, 2]);

        let ContractionTree { nodes, root, .. } = complex_tree;

        let node0 = child_node(0, vec![0]);
        let node1 = child_node(1, vec![1]);
        let node2 = child_node(2, vec![2]);
        let node3 = child_node(3, vec![3]);
        let node4 = child_node(4, vec![4]);
        let node5 = child_node(5, vec![5]);
        let node6 = parent_node(6, &node1, &node5);
        let node7 = parent_node(7, &node0, &node6);
        let node8 = parent_node(8, &node4, &node2);
        let node9 = parent_node(9, &node8, &node3);
        let node10 = parent_node(10, &node7, &node9);

        let ref_root = Rc::clone(&node10);
        let ref_nodes = [
            node0, node1, node2, node3, node4, node5, node6, node7, node8, node9, node10,
        ];

        for (key, ref_node) in ref_nodes.iter().enumerate() {
            let node = &nodes[&key];
            assert_eq!(node, ref_node);
        }
        assert_eq!(root.upgrade().unwrap(), ref_root);
    }

    #[test]
    #[should_panic = "Tensor 2 is already used in another contraction"]
    fn test_add_path_as_subtree_invalid_path() {
        let (tensor, path, _) = setup_complex();

        let mut complex_tree = ContractionTree::from_contraction_path(&tensor, &path);
        complex_tree.remove_subtree(8);
        let new_path = path![(4, 2), (4, 3)];

        complex_tree.add_path_as_subtree(&new_path, 9, &[3, 4, 2]);
    }

    #[test]
    fn test_remove_communication_path() {
        let (tensor, path) = setup_nested();
        let mut complex_tree = ContractionTree::from_contraction_path(&tensor, &path);
        let partition_ids = vec![2, 5, 8];
        complex_tree.remove_communication_path(&partition_ids);
        assert!(!complex_tree.nodes.contains_key(&9));
        assert!(!complex_tree.nodes.contains_key(&10));
        assert!(complex_tree.root_id().is_none());
    }

    #[test]
    fn test_replace_communication_path() {
        let (tensor, path) = setup_nested();
        let mut complex_tree = ContractionTree::from_contraction_path(&tensor, &path);
        let partition_ids = vec![2, 5, 8];
        complex_tree.replace_communication_path(partition_ids, &[(0, 2), (1, 0)]);

        let ContractionTree { nodes, root, .. } = complex_tree;

        let node2 = child_node(2, vec![]);
        let node5 = child_node(5, vec![]);
        let node8 = child_node(8, vec![]);
        let node9 = parent_node(9, &node2, &node8);
        let node10 = parent_node(10, &node5, &node9);

        assert_eq!(nodes[&9], node9);
        assert_eq!(nodes[&10], node10);
        assert_eq!(root.upgrade().unwrap(), node10);
    }
}
