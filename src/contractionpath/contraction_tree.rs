use crate::pair;
use crate::tensornetwork::tensor::Tensor;
use crate::types::ContractionIndex;
use node::{Node, NodeRef, WeakNodeRef};
use rustc_hash::FxHashMap;
use std::cell::{Ref, RefCell};
use std::rc::{Rc, Weak};
use std::sync::Arc;

use super::paths::validate_path;

pub mod balancing;
pub mod export;
pub mod import;
mod node;
mod utils;

/// Struct representing the full contraction path of a given Tensor object
#[derive(Default, Debug, Clone)]
pub struct ContractionTree {
    nodes: FxHashMap<usize, NodeRef>,
    partitions: FxHashMap<usize, Vec<usize>>,
    root: WeakNodeRef,
}

impl ContractionTree {
    pub fn node(&self, node_id: usize) -> Ref<Node> {
        let borrow = &self.nodes[&node_id];
        borrow.as_ref().borrow()
    }

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
        path: &[ContractionIndex],
        nodes: &mut FxHashMap<usize, NodeRef>,
        partitions: &mut FxHashMap<usize, Vec<usize>>,
        prefix: &[usize],
    ) {
        let mut scratch = FxHashMap::default();

        // Obtain tree structure from composite tensors
        for contr in path {
            if let ContractionIndex::Path(path_id, path) = contr {
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
        }

        // Add nodes for leaf tensors
        for (tensor_idx, tensor) in tensor.tensors().iter().enumerate() {
            if tensor.is_leaf() {
                let mut nested_tensor_idx = prefix.to_owned();
                nested_tensor_idx.push(tensor_idx);
                let new_node = Node::new(
                    nodes.len(),
                    Weak::new(),
                    Weak::new(),
                    Weak::new(),
                    Some(nested_tensor_idx),
                );
                let new_node = Rc::new(RefCell::new(new_node));
                scratch.insert(tensor_idx, Rc::clone(&new_node));
                nodes.insert(nodes.len(), new_node);
            }
        }

        // Build tree based on contraction path
        for contr in path {
            if let ContractionIndex::Pair(i_path, j_path) = contr {
                let i = &scratch[i_path];
                let j = &scratch[j_path];
                let parent = Node::new(
                    nodes.len(),
                    Rc::downgrade(i),
                    Rc::downgrade(j),
                    Weak::new(),
                    None,
                );

                let parent = Rc::new(RefCell::new(parent));
                i.borrow_mut().set_parent(Rc::downgrade(&parent));
                j.borrow_mut().set_parent(Rc::downgrade(&parent));
                scratch.insert(*i_path, Rc::clone(&parent));
                nodes.insert(nodes.len(), parent);
                scratch.remove(j_path);
            }
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
    pub fn from_contraction_path(tensor: &Tensor, path: &[ContractionIndex]) -> Self {
        validate_path(path);
        let mut nodes = FxHashMap::default();
        let mut partitions = FxHashMap::default();
        Self::from_contraction_path_recurse(tensor, path, &mut nodes, &mut partitions, &Vec::new());
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
        path: &[ContractionIndex],
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
        for contr in path {
            if let ContractionIndex::Pair(i_path, j_path) = contr {
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
                let parent =
                    Node::new(index, Rc::downgrade(i), Rc::downgrade(j), Weak::new(), None);

                let parent = Rc::new(RefCell::new(parent));
                i.borrow_mut().set_parent(Rc::downgrade(&parent));
                j.borrow_mut().set_parent(Rc::downgrade(&parent));
                scratch.insert(*i_path, Rc::clone(&parent));
                scratch.remove(j_path);
                // Ensure that intermediate tensor information is stored in internal HashMap for reference
                self.nodes.insert(index, parent);
            } else {
                panic!("Constructor not implemented for nested Tensors")
            }
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
    /// * `hierarchy` - if set to `true` returns a nested contraction path, otherwise returns a flat contraction path
    fn to_contraction_path_recurse(
        node: &Node,
        path: &mut Vec<ContractionIndex>,
        replace: bool,
        hierarchy: bool,
    ) -> usize {
        if node.is_leaf() {
            if hierarchy {
                let tn_index = node.tensor_index().as_ref().unwrap();
                return *tn_index.last().unwrap();
            } else {
                return node.id();
            }
        }

        // Get children
        let (Some(left_child), Some(right_child)) = (node.left_child(), node.right_child()) else {
            panic!("All parents should have two children")
        };

        // Get right and left child tensor ids
        let mut t1_id = Self::to_contraction_path_recurse(
            &left_child.as_ref().borrow(),
            path,
            replace,
            hierarchy,
        );
        let mut t2_id = Self::to_contraction_path_recurse(
            &right_child.as_ref().borrow(),
            path,
            replace,
            hierarchy,
        );
        if t2_id < t1_id {
            (t1_id, t2_id) = (t2_id, t1_id);
        }

        // Add pair to path
        path.push(pair!(t1_id, t2_id));

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
    /// * `hierarchy` - if set to `true` returns a nested contraction path, otherwise returns a flat contraction path
    pub fn to_flat_contraction_path(&self, node_id: usize, replace: bool) -> Vec<ContractionIndex> {
        let node = self.node(node_id);
        let mut path = Vec::new();
        Self::to_contraction_path_recurse(&node, &mut path, replace, false);
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
        let mut new_tensor = Tensor::new_with_bonddims(Vec::new(), Arc::clone(&tensor.bond_dims));

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
            if new_height1 <= height_limit && new_height2 <= height_limit {
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
    use std::iter::zip;

    use crate::contractionpath::contraction_cost::contract_cost_tensors;
    use crate::contractionpath::contraction_tree::{ContractionTree, Node};
    use crate::contractionpath::ssa_replace_ordering;
    use crate::path;
    use crate::tensornetwork::create_tensor_network;
    use crate::tensornetwork::tensor::Tensor;
    use crate::types::ContractionIndex;

    use super::*;

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

    fn setup_unbalanced() -> (Tensor, Vec<ContractionIndex>) {
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
                    (11, 17),
                ]),
                None,
            ),
            path![(0, 1), (2, 0), (3, 2), (4, 3), (5, 4)].to_vec(),
        )
    }

    fn setup_nested() -> (Tensor, Vec<ContractionIndex>) {
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

        let mut t23 = Tensor::default();
        t23.push_tensors(vec![t2, t3], Some(&bond_dims), None);

        let mut t45 = Tensor::default();
        t45.push_tensors(vec![t4, t5], Some(&bond_dims), None);

        let mut tensor_network = Tensor::default();
        tensor_network.push_tensors(vec![t01, t23, t45], Some(&bond_dims), None);
        (
            tensor_network,
            path![(0, [(0, 1)]), (1, [(0, 1)]), (2, [(0, 1)]), (0, 1), (0, 2)].to_vec(),
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
        let (tensor, path) = setup_simple();
        let ContractionTree { nodes, root, .. } =
            ContractionTree::from_contraction_path(&tensor, &path);

        let node0 = Rc::new(RefCell::new(Node::new(
            0,
            Weak::new(),
            Weak::new(),
            Weak::new(),
            Some(vec![0]),
        )));
        let node1 = Rc::new(RefCell::new(Node::new(
            1,
            Weak::new(),
            Weak::new(),
            Weak::new(),
            Some(vec![1]),
        )));
        let node2 = Rc::new(RefCell::new(Node::new(
            2,
            Weak::new(),
            Weak::new(),
            Weak::new(),
            Some(vec![2]),
        )));
        let node3 = Rc::new(RefCell::new(Node::new(
            3,
            Rc::downgrade(&node0),
            Rc::downgrade(&node1),
            Weak::new(),
            None,
        )));
        let node4 = Rc::new(RefCell::new(Node::new(
            4,
            Rc::downgrade(&node2),
            Rc::downgrade(&node3),
            Weak::new(),
            None,
        )));
        node0.borrow_mut().set_parent(Rc::downgrade(&node3));
        node1.borrow_mut().set_parent(Rc::downgrade(&node3));
        node2.borrow_mut().set_parent(Rc::downgrade(&node4));
        node3.borrow_mut().set_parent(Rc::downgrade(&node4));

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
        let (tensor, path) = setup_complex();
        let ContractionTree { nodes, root, .. } =
            ContractionTree::from_contraction_path(&tensor, &path);

        let node0 = Rc::new(RefCell::new(Node::new(
            0,
            Weak::new(),
            Weak::new(),
            Weak::new(),
            Some(vec![0]),
        )));
        let node1 = Rc::new(RefCell::new(Node::new(
            1,
            Weak::new(),
            Weak::new(),
            Weak::new(),
            Some(vec![1]),
        )));
        let node2 = Rc::new(RefCell::new(Node::new(
            2,
            Weak::new(),
            Weak::new(),
            Weak::new(),
            Some(vec![2]),
        )));
        let node3 = Rc::new(RefCell::new(Node::new(
            3,
            Weak::new(),
            Weak::new(),
            Weak::new(),
            Some(vec![3]),
        )));
        let node4 = Rc::new(RefCell::new(Node::new(
            4,
            Weak::new(),
            Weak::new(),
            Weak::new(),
            Some(vec![4]),
        )));
        let node5 = Rc::new(RefCell::new(Node::new(
            5,
            Weak::new(),
            Weak::new(),
            Weak::new(),
            Some(vec![5]),
        )));
        let node6 = Rc::new(RefCell::new(Node::new(
            6,
            Rc::downgrade(&node1),
            Rc::downgrade(&node5),
            Weak::new(),
            None,
        )));
        let node7 = Rc::new(RefCell::new(Node::new(
            7,
            Rc::downgrade(&node0),
            Rc::downgrade(&node6),
            Weak::new(),
            None,
        )));
        let node8 = Rc::new(RefCell::new(Node::new(
            8,
            Rc::downgrade(&node3),
            Rc::downgrade(&node4),
            Weak::new(),
            None,
        )));
        let node9 = Rc::new(RefCell::new(Node::new(
            9,
            Rc::downgrade(&node2),
            Rc::downgrade(&node8),
            Weak::new(),
            None,
        )));
        let node10 = Rc::new(RefCell::new(Node::new(
            10,
            Rc::downgrade(&node7),
            Rc::downgrade(&node9),
            Weak::new(),
            None,
        )));
        node0.borrow_mut().set_parent(Rc::downgrade(&node7));
        node1.borrow_mut().set_parent(Rc::downgrade(&node6));
        node2.borrow_mut().set_parent(Rc::downgrade(&node9));
        node3.borrow_mut().set_parent(Rc::downgrade(&node8));
        node4.borrow_mut().set_parent(Rc::downgrade(&node8));
        node5.borrow_mut().set_parent(Rc::downgrade(&node6));
        node6.borrow_mut().set_parent(Rc::downgrade(&node7));
        node7.borrow_mut().set_parent(Rc::downgrade(&node10));
        node8.borrow_mut().set_parent(Rc::downgrade(&node9));
        node9.borrow_mut().set_parent(Rc::downgrade(&node10));

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

        let node0 = Rc::new(RefCell::new(Node::new(
            0,
            Weak::new(),
            Weak::new(),
            Weak::new(),
            Some(vec![0, 0]),
        )));
        let node1 = Rc::new(RefCell::new(Node::new(
            1,
            Weak::new(),
            Weak::new(),
            Weak::new(),
            Some(vec![0, 1]),
        )));
        let node3 = Rc::new(RefCell::new(Node::new(
            3,
            Weak::new(),
            Weak::new(),
            Weak::new(),
            Some(vec![1, 0]),
        )));
        let node4 = Rc::new(RefCell::new(Node::new(
            4,
            Weak::new(),
            Weak::new(),
            Weak::new(),
            Some(vec![1, 1]),
        )));
        let node6 = Rc::new(RefCell::new(Node::new(
            6,
            Weak::new(),
            Weak::new(),
            Weak::new(),
            Some(vec![2, 0]),
        )));
        let node7 = Rc::new(RefCell::new(Node::new(
            7,
            Weak::new(),
            Weak::new(),
            Weak::new(),
            Some(vec![2, 1]),
        )));
        let node2 = Rc::new(RefCell::new(Node::new(
            2,
            Rc::downgrade(&node0),
            Rc::downgrade(&node1),
            Weak::new(),
            None,
        )));
        let node5 = Rc::new(RefCell::new(Node::new(
            5,
            Rc::downgrade(&node3),
            Rc::downgrade(&node4),
            Weak::new(),
            None,
        )));
        let node8 = Rc::new(RefCell::new(Node::new(
            8,
            Rc::downgrade(&node6),
            Rc::downgrade(&node7),
            Weak::new(),
            None,
        )));
        let node9 = Rc::new(RefCell::new(Node::new(
            9,
            Rc::downgrade(&node2),
            Rc::downgrade(&node5),
            Weak::new(),
            None,
        )));
        let node10 = Rc::new(RefCell::new(Node::new(
            10,
            Rc::downgrade(&node9),
            Rc::downgrade(&node8),
            Weak::new(),
            None,
        )));
        node0.borrow_mut().set_parent(Rc::downgrade(&node2));
        node1.borrow_mut().set_parent(Rc::downgrade(&node2));
        node2.borrow_mut().set_parent(Rc::downgrade(&node9));
        node3.borrow_mut().set_parent(Rc::downgrade(&node5));
        node4.borrow_mut().set_parent(Rc::downgrade(&node5));
        node5.borrow_mut().set_parent(Rc::downgrade(&node9));
        node6.borrow_mut().set_parent(Rc::downgrade(&node8));
        node7.borrow_mut().set_parent(Rc::downgrade(&node8));
        node8.borrow_mut().set_parent(Rc::downgrade(&node10));
        node9.borrow_mut().set_parent(Rc::downgrade(&node10));

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

        let node0 = Rc::new(RefCell::new(Node::new(
            0,
            Weak::new(),
            Weak::new(),
            Weak::new(),
            Some(vec![0, 0, 0]),
        )));
        let node1 = Rc::new(RefCell::new(Node::new(
            1,
            Weak::new(),
            Weak::new(),
            Weak::new(),
            Some(vec![0, 0, 1]),
        )));
        let node3 = Rc::new(RefCell::new(Node::new(
            3,
            Weak::new(),
            Weak::new(),
            Weak::new(),
            Some(vec![0, 1]),
        )));
        let node5 = Rc::new(RefCell::new(Node::new(
            5,
            Weak::new(),
            Weak::new(),
            Weak::new(),
            Some(vec![1, 0, 0]),
        )));
        let node6 = Rc::new(RefCell::new(Node::new(
            6,
            Weak::new(),
            Weak::new(),
            Weak::new(),
            Some(vec![1, 0, 1]),
        )));
        let node8 = Rc::new(RefCell::new(Node::new(
            8,
            Weak::new(),
            Weak::new(),
            Weak::new(),
            Some(vec![1, 1]),
        )));
        let node2 = Rc::new(RefCell::new(Node::new(
            2,
            Rc::downgrade(&node0),
            Rc::downgrade(&node1),
            Weak::new(),
            None,
        )));
        let node4 = Rc::new(RefCell::new(Node::new(
            4,
            Rc::downgrade(&node2),
            Rc::downgrade(&node3),
            Weak::new(),
            None,
        )));
        let node7 = Rc::new(RefCell::new(Node::new(
            7,
            Rc::downgrade(&node5),
            Rc::downgrade(&node6),
            Weak::new(),
            None,
        )));
        let node9 = Rc::new(RefCell::new(Node::new(
            9,
            Rc::downgrade(&node7),
            Rc::downgrade(&node8),
            Weak::new(),
            None,
        )));
        let node10 = Rc::new(RefCell::new(Node::new(
            10,
            Rc::downgrade(&node4),
            Rc::downgrade(&node9),
            Weak::new(),
            None,
        )));
        node0.borrow_mut().set_parent(Rc::downgrade(&node2));
        node1.borrow_mut().set_parent(Rc::downgrade(&node2));
        node2.borrow_mut().set_parent(Rc::downgrade(&node4));
        node3.borrow_mut().set_parent(Rc::downgrade(&node4));
        node4.borrow_mut().set_parent(Rc::downgrade(&node10));
        node5.borrow_mut().set_parent(Rc::downgrade(&node7));
        node6.borrow_mut().set_parent(Rc::downgrade(&node7));
        node7.borrow_mut().set_parent(Rc::downgrade(&node9));
        node8.borrow_mut().set_parent(Rc::downgrade(&node9));
        node9.borrow_mut().set_parent(Rc::downgrade(&node10));

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
        let (tn, path) = setup_simple();
        let tree = ContractionTree::from_contraction_path(&tn, &path);

        assert_eq!(tree.leaf_ids(4), vec![2, 0, 1]);
        assert_eq!(tree.leaf_ids(3), vec![0, 1]);
        assert_eq!(tree.leaf_ids(2), vec![2]);
    }

    #[test]
    fn test_leaf_ids_complex() {
        let (tn, path) = setup_complex();
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

        let node0 = Rc::new(RefCell::new(Node::new(
            0,
            Weak::new(),
            Weak::new(),
            Weak::new(),
            Some(vec![0, 0]),
        )));
        let node1 = Rc::new(RefCell::new(Node::new(
            1,
            Weak::new(),
            Weak::new(),
            Weak::new(),
            Some(vec![0, 1]),
        )));
        let node3 = Rc::new(RefCell::new(Node::new(
            3,
            Weak::new(),
            Weak::new(),
            Weak::new(),
            Some(vec![1, 0]),
        )));
        let node4 = Rc::new(RefCell::new(Node::new(
            4,
            Weak::new(),
            Weak::new(),
            Weak::new(),
            Some(vec![1, 1]),
        )));
        let node6 = Rc::new(RefCell::new(Node::new(
            6,
            Weak::new(),
            Weak::new(),
            Weak::new(),
            Some(vec![2, 0]),
        )));
        let node7 = Rc::new(RefCell::new(Node::new(
            7,
            Weak::new(),
            Weak::new(),
            Weak::new(),
            Some(vec![2, 1]),
        )));
        let node2 = Rc::new(RefCell::new(Node::new(
            2,
            Rc::downgrade(&node0),
            Rc::downgrade(&node1),
            Weak::new(),
            None,
        )));
        let node5 = Rc::new(RefCell::new(Node::new(
            5,
            Rc::downgrade(&node3),
            Rc::downgrade(&node4),
            Weak::new(),
            None,
        )));

        let node9 = Rc::new(RefCell::new(Node::new(
            9,
            Rc::downgrade(&node2),
            Rc::downgrade(&node5),
            Weak::new(),
            None,
        )));
        let node10 = Rc::new(RefCell::new(Node::new(
            10,
            Rc::downgrade(&node9),
            Default::default(),
            Weak::new(),
            None,
        )));
        node0.borrow_mut().set_parent(Rc::downgrade(&node2));
        node1.borrow_mut().set_parent(Rc::downgrade(&node2));
        node2.borrow_mut().set_parent(Rc::downgrade(&node9));
        node3.borrow_mut().set_parent(Rc::downgrade(&node5));
        node4.borrow_mut().set_parent(Rc::downgrade(&node5));
        node5.borrow_mut().set_parent(Rc::downgrade(&node9));
        node9.borrow_mut().set_parent(Rc::downgrade(&node10));

        let ref_root = Rc::clone(&node10);
        let ref_nodes = [
            node0, node1, node2, node3, node4, node5, node6, node7, node9, node10,
        ];
        let mut range = (0..8).collect::<Vec<usize>>();
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

        let node0 = Rc::new(RefCell::new(Node::new(
            0,
            Weak::new(),
            Weak::new(),
            Weak::new(),
            Some(vec![0, 0]),
        )));
        let node1 = Rc::new(RefCell::new(Node::new(
            1,
            Weak::new(),
            Weak::new(),
            Weak::new(),
            Some(vec![0, 1]),
        )));
        let node3 = Rc::new(RefCell::new(Node::new(
            3,
            Weak::new(),
            Weak::new(),
            Weak::new(),
            Some(vec![1, 0]),
        )));
        let node4 = Rc::new(RefCell::new(Node::new(
            4,
            Weak::new(),
            Weak::new(),
            Weak::new(),
            Some(vec![1, 1]),
        )));
        let node6 = Rc::new(RefCell::new(Node::new(
            6,
            Weak::new(),
            Weak::new(),
            Weak::new(),
            Some(vec![2, 0]),
        )));
        let node7 = Rc::new(RefCell::new(Node::new(
            7,
            Weak::new(),
            Weak::new(),
            Weak::new(),
            Some(vec![2, 1]),
        )));
        let node2 = Rc::new(RefCell::new(Node::new(
            2,
            Rc::downgrade(&node0),
            Rc::downgrade(&node1),
            Weak::new(),
            None,
        )));
        let node5 = Rc::new(RefCell::new(Node::new(
            5,
            Rc::downgrade(&node3),
            Rc::downgrade(&node4),
            Weak::new(),
            None,
        )));
        let node8 = Rc::new(RefCell::new(Node::new(
            8,
            Rc::downgrade(&node6),
            Default::default(),
            Weak::new(),
            None,
        )));
        let node9 = Rc::new(RefCell::new(Node::new(
            9,
            Rc::downgrade(&node2),
            Rc::downgrade(&node5),
            Weak::new(),
            None,
        )));
        let node10 = Rc::new(RefCell::new(Node::new(
            10,
            Rc::downgrade(&node9),
            Rc::downgrade(&node8),
            Weak::new(),
            None,
        )));
        node0.borrow_mut().set_parent(Rc::downgrade(&node2));
        node1.borrow_mut().set_parent(Rc::downgrade(&node2));
        node2.borrow_mut().set_parent(Rc::downgrade(&node9));
        node3.borrow_mut().set_parent(Rc::downgrade(&node5));
        node4.borrow_mut().set_parent(Rc::downgrade(&node5));
        node5.borrow_mut().set_parent(Rc::downgrade(&node9));
        node6.borrow_mut().set_parent(Rc::downgrade(&node8));
        node8.borrow_mut().set_parent(Rc::downgrade(&node10));
        node9.borrow_mut().set_parent(Rc::downgrade(&node10));

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
        let (tensor, path) = setup_simple();
        let tree = ContractionTree::from_contraction_path(&tensor, &path);
        let ref_weights =
            FxHashMap::from_iter([(1, 0f64), (0, 0f64), (2, 0f64), (3, 3820f64), (4, 4540f64)]);
        let weights = tree.tree_weights(4, &tensor, contract_cost_tensors);

        assert_eq!(weights, ref_weights);
        let ref_weights = FxHashMap::from_iter([(1, 0f64), (0, 0f64), (3, 3820f64)]);
        let weights = tree.tree_weights(3, &tensor, contract_cost_tensors);
        assert_eq!(weights, ref_weights);

        assert_eq!(weights, ref_weights);
        let ref_weights = FxHashMap::from_iter([(2, 0f64)]);
        let weights = tree.tree_weights(2, &tensor, contract_cost_tensors);
        assert_eq!(weights, ref_weights);
    }

    #[test]
    fn test_tree_weights_complex() {
        let (tensor, path) = setup_complex();
        let tree = ContractionTree::from_contraction_path(&tensor, &path);
        let ref_weights = FxHashMap::from_iter([
            (0, 0f64),
            (1, 0f64),
            (2, 0f64),
            (3, 0f64),
            (4, 0f64),
            (5, 0f64),
            (6, 2098440f64),
            (7, 2120010f64),
            (8, 2105820f64),
            (9, 2116470f64),
            (10, 4237070f64),
        ]);
        let weights = tree.tree_weights(10, &tensor, contract_cost_tensors);

        assert_eq!(weights, ref_weights);
    }

    #[test]
    fn test_to_contraction_path_simple() {
        let (tensor, ref_path) = setup_simple();
        let tree = ContractionTree::from_contraction_path(&tensor, &ref_path);
        let path = tree.to_flat_contraction_path(4, false);
        let path = ssa_replace_ordering(&path, 3);
        assert_eq!(path, ref_path);
    }

    #[test]
    fn test_to_contraction_path_complex() {
        let (tensor, ref_path) = setup_complex();
        let tree = ContractionTree::from_contraction_path(&tensor, &ref_path);
        let path = tree.to_flat_contraction_path(10, false);
        let path = ssa_replace_ordering(&path, 6);
        assert_eq!(path, ref_path);
    }

    #[test]
    fn test_to_contraction_path_unbalanced() {
        let (tensor, ref_path) = setup_unbalanced();
        let tree = ContractionTree::from_contraction_path(&tensor, &ref_path);
        let path = tree.to_flat_contraction_path(10, false);
        let path = ssa_replace_ordering(&path, 6);
        assert_eq!(path, ref_path);
    }

    #[test]
    fn test_populate_subtree_tensor_map_simple() {
        let (tensor, ref_path) = setup_simple();
        let tree = ContractionTree::from_contraction_path(&tensor, &ref_path);
        let mut node_tensor_map = FxHashMap::default();
        populate_subtree_tensor_map_recursive(&tree, 4, &mut node_tensor_map, &tensor, None);

        let ref_node_tensor_map = FxHashMap::from_iter([
            (0, Tensor::new(vec![4, 3, 2])),
            (1, Tensor::new(vec![0, 1, 3, 2])),
            (2, Tensor::new(vec![4, 5, 6])),
            (3, Tensor::new(vec![4, 0, 1])),
            (4, Tensor::new(vec![5, 6, 0, 1])),
        ]);

        for (key, value) in ref_node_tensor_map {
            assert_eq!(node_tensor_map[&key].legs(), value.legs());
        }
    }

    #[test]
    fn test_populate_subtree_tensor_map_complex() {
        let (tensor, ref_path) = setup_complex();
        let tree = ContractionTree::from_contraction_path(&tensor, &ref_path);
        let mut node_tensor_map = FxHashMap::default();
        populate_subtree_tensor_map_recursive(&tree, 10, &mut node_tensor_map, &tensor, None);

        let ref_node_tensor_map = FxHashMap::from_iter([
            (0, Tensor::new(vec![4, 3, 2])),
            (1, Tensor::new(vec![0, 1, 3, 2])),
            (2, Tensor::new(vec![4, 5, 6])),
            (3, Tensor::new(vec![6, 8, 9])),
            (4, Tensor::new(vec![10, 8, 9])),
            (5, Tensor::new(vec![5, 1, 0])),
            (6, Tensor::new(vec![3, 2, 5])),
            (7, Tensor::new(vec![4, 5])),
            (8, Tensor::new(vec![6, 10])),
            (9, Tensor::new(vec![4, 5, 10])),
            (10, Tensor::new(vec![10])),
        ]);

        for (key, value) in ref_node_tensor_map {
            assert_eq!(node_tensor_map[&key].legs(), value.legs());
        }
    }

    #[test]
    fn test_populate_leaf_node_tensor_map_simple() {
        let (tensor, ref_path) = setup_simple();
        let tree = ContractionTree::from_contraction_path(&tensor, &ref_path);

        let node_tensor_map = populate_leaf_node_tensor_map(&tree, 4, &tensor);

        let ref_node_tensor_map = FxHashMap::from_iter([
            (0, Tensor::new(vec![4, 3, 2])),
            (1, Tensor::new(vec![0, 1, 3, 2])),
            (2, Tensor::new(vec![4, 5, 6])),
        ]);

        for (key, value) in ref_node_tensor_map {
            assert_eq!(node_tensor_map[&key].legs(), value.legs());
        }
    }

    #[test]
    fn test_populate_leaf_node_tensor_map_complex() {
        let (tensor, ref_path) = setup_complex();
        let tree = ContractionTree::from_contraction_path(&tensor, &ref_path);
        let node_tensor_map = populate_subtree_tensor_map(&tree, 10, &tensor, None);

        let ref_node_tensor_map = FxHashMap::from_iter([
            (0, Tensor::new(vec![4, 3, 2])),
            (1, Tensor::new(vec![0, 1, 3, 2])),
            (2, Tensor::new(vec![4, 5, 6])),
            (3, Tensor::new(vec![6, 8, 9])),
            (4, Tensor::new(vec![10, 8, 9])),
        ]);

        for (key, value) in ref_node_tensor_map {
            assert_eq!(node_tensor_map[&key].legs(), value.legs());
        }
    }

    #[test]
    fn test_add_path_as_subtree() {
        let (tensor, path) = setup_complex();

        let mut complex_tree = ContractionTree::from_contraction_path(&tensor, &path);
        complex_tree.remove_subtree(9);
        let new_path = path![(4, 2), (4, 3)];

        complex_tree.add_path_as_subtree(new_path, 10, &[3, 4, 2]);

        let ContractionTree { nodes, root, .. } = complex_tree;

        let node0 = Rc::new(RefCell::new(Node::new(
            0,
            Weak::new(),
            Weak::new(),
            Weak::new(),
            Some(vec![0]),
        )));
        let node1 = Rc::new(RefCell::new(Node::new(
            1,
            Weak::new(),
            Weak::new(),
            Weak::new(),
            Some(vec![1]),
        )));
        let node2 = Rc::new(RefCell::new(Node::new(
            2,
            Weak::new(),
            Weak::new(),
            Weak::new(),
            Some(vec![2]),
        )));
        let node3 = Rc::new(RefCell::new(Node::new(
            3,
            Weak::new(),
            Weak::new(),
            Weak::new(),
            Some(vec![3]),
        )));
        let node4 = Rc::new(RefCell::new(Node::new(
            4,
            Weak::new(),
            Weak::new(),
            Weak::new(),
            Some(vec![4]),
        )));
        let node5 = Rc::new(RefCell::new(Node::new(
            5,
            Weak::new(),
            Weak::new(),
            Weak::new(),
            Some(vec![5]),
        )));
        let node6 = Rc::new(RefCell::new(Node::new(
            6,
            Rc::downgrade(&node1),
            Rc::downgrade(&node5),
            Weak::new(),
            None,
        )));
        let node7 = Rc::new(RefCell::new(Node::new(
            7,
            Rc::downgrade(&node0),
            Rc::downgrade(&node6),
            Weak::new(),
            None,
        )));
        let node8 = Rc::new(RefCell::new(Node::new(
            8,
            Rc::downgrade(&node4),
            Rc::downgrade(&node2),
            Weak::new(),
            None,
        )));
        let node9 = Rc::new(RefCell::new(Node::new(
            9,
            Rc::downgrade(&node8),
            Rc::downgrade(&node3),
            Weak::new(),
            None,
        )));
        let node10 = Rc::new(RefCell::new(Node::new(
            10,
            Rc::downgrade(&node7),
            Rc::downgrade(&node9),
            Weak::new(),
            None,
        )));
        node0.borrow_mut().set_parent(Rc::downgrade(&node7));
        node1.borrow_mut().set_parent(Rc::downgrade(&node6));
        node2.borrow_mut().set_parent(Rc::downgrade(&node8));
        node3.borrow_mut().set_parent(Rc::downgrade(&node9));
        node4.borrow_mut().set_parent(Rc::downgrade(&node8));
        node5.borrow_mut().set_parent(Rc::downgrade(&node6));
        node6.borrow_mut().set_parent(Rc::downgrade(&node7));
        node7.borrow_mut().set_parent(Rc::downgrade(&node10));
        node8.borrow_mut().set_parent(Rc::downgrade(&node9));
        node9.borrow_mut().set_parent(Rc::downgrade(&node10));

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
        let (tensor, path) = setup_complex();

        let mut complex_tree = ContractionTree::from_contraction_path(&tensor, &path);
        complex_tree.remove_subtree(8);
        let new_path = path![(4, 2), (4, 3)];

        complex_tree.add_path_as_subtree(new_path, 9, &[3, 4, 2]);
    }
}
