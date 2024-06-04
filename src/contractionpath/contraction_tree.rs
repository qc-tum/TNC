use rand::distributions::{Distribution, WeightedIndex};
use rand::thread_rng;

use crate::contractionpath::paths::{
    greedy::Greedy,
    {CostType, OptimizePath},
};
use crate::contractionpath::ssa_replace_ordering;
use crate::pair;
use crate::tensornetwork::{create_tensor_network, tensor::Tensor};
use crate::types::ContractionIndex;
use std::cell::{Ref, RefCell, RefMut};
use std::cmp::{self, max, min_by_key};
use std::collections::HashMap;
use std::ptr;
use std::rc::Rc;

use super::contraction_cost::contract_path_cost;
use super::paths::validate_path;

type NodeRef = Rc<RefCell<Node>>;

/// Node in ContractionTree, represents a contraction of Tensor with position `left_child` and position `right_child` to obtain Tensor at position `parent`.
#[derive(Debug, Clone)]
struct Node {
    id: usize,
    left_child: *mut Node,
    right_child: *mut Node,
    parent: *mut Node,
    tensor_index: Option<usize>,
}

impl Node {
    fn new(
        id: usize,
        left_child: *mut Node,
        right_child: *mut Node,
        parent: *mut Node,
        tensor_index: Option<usize>,
    ) -> Self {
        Self {
            id,
            left_child,
            right_child,
            parent,
            tensor_index,
        }
    }
    fn left_child(&self) -> *mut Node {
        self.left_child
    }

    fn right_child(&self) -> *mut Node {
        self.right_child
    }

    fn set_left_child(&mut self, child: *mut Node) {
        self.left_child = child;
    }

    fn set_right_child(&mut self, child: *mut Node) {
        self.right_child = child;
    }

    fn parent(&self) -> *mut Node {
        self.parent
    }

    fn set_parent(&mut self, parent: *mut Node) {
        self.parent = parent;
    }

    fn remove_parent(&mut self) {
        self.parent = ptr::null_mut();
    }

    fn is_leaf(&self) -> bool {
        self.left_child.is_null() && self.right_child.is_null()
    }

    fn add_child(&mut self, child: *mut Node) {
        if self.left_child.is_null() {
            self.left_child = child;
        }
        if self.right_child.is_null() {
            self.right_child = child;
        }

        panic!("Unable to add child, Node already has two children");
    }

    fn remove_child(&mut self, child: *mut Node) {
        let left_child_id;
        let right_child_id;
        let child_id;

        unsafe {
            left_child_id = (*self.left_child).id;
            right_child_id = (*self.right_child).id;
            child_id = (*child).id;
        }

        if left_child_id == child_id {
            self.left_child = ptr::null_mut();
        } else if right_child_id == child_id {
            self.right_child = ptr::null_mut();
        } else {
            panic!("Child {:?} not found in Node", child);
        }
    }

    fn get_other_child(&self, child: *mut Node) -> *mut Node {
        let left_child_id;
        let right_child_id;
        let child_id;

        unsafe {
            left_child_id = (*self.left_child).id;
            right_child_id = (*self.right_child).id;
            child_id = (*child).id;
        }

        if left_child_id == child_id {
            self.right_child
        } else if right_child_id == child_id {
            self.left_child
        } else {
            panic!("Child {:?} not found in Node", child);
        }
    }

    fn replace_child(&mut self, child: *mut Node, repl_node: *mut Node) {
        let left_child_id;
        let right_child_id;
        let child_id;

        unsafe {
            left_child_id = (*self.left_child).id;
            right_child_id = (*self.right_child).id;
            child_id = (*child).id;
        }

        if left_child_id == child_id {
            self.right_child = repl_node;
        } else if right_child_id == child_id {
            self.left_child = repl_node;
        } else {
            panic!("Child {:?} not found in Node", child);
        }
    }

    fn deprecate(&mut self) {
        self.left_child = ptr::null_mut();
        self.right_child = ptr::null_mut();
        self.parent = ptr::null_mut();
    }
}

/// Struct representing the full contraction path of a given Tensor object
#[derive(Debug)]
struct ContractionTree {
    nodes: HashMap<usize, NodeRef>,
    root: *mut Node,
}

impl Default for ContractionTree {
    fn default() -> Self {
        Self {
            nodes: Default::default(),
            root: ptr::null_mut(),
        }
    }
}

impl ContractionTree {
    fn add_node(&mut self, mut node: Node) -> usize {
        let index = self.nodes.len();
        node.id = index;
        self.nodes
            .entry(index)
            .or_insert(Rc::new(RefCell::new(node)));
        index
    }

    pub fn node(&self, tensor_id: usize) -> Ref<Node> {
        let borrow = self.nodes.get(&tensor_id).unwrap();
        borrow.as_ref().borrow()
    }

    pub fn mut_node(&mut self, tensor_id: usize) -> RefMut<Node> {
        self.nodes.get_mut(&tensor_id).unwrap().borrow_mut()
    }

    pub fn node_ptr(&self, tensor_id: usize) -> *mut Node {
        self.nodes.get(&tensor_id).unwrap().as_ptr()
    }

    /// Removes node from HashMap. Warning! As HashMap stores Node data, removing a Node here can result in invalid references.
    unsafe fn remove_node(&mut self, node_id: usize) {
        self.nodes.remove(&node_id);
    }

    #[must_use]
    /// Creates a [`ContractionTree`] object from a [`Tensor`] and a [`Vec<ContractionIndex>`].
    ///
    /// # Arguments
    ///
    /// * `tn` - [`Tensor`] providing topographic and geometric information.
    /// * `contract_path` - slice of [`ContractionIndex`], indicating contraction path.
    /// # Returns
    /// Constructed [`ContractionTree`] that represents all intermediate tensors and costs of given contraction path and tensor network.
    pub fn from_contraction_path(tn: &Tensor, path: &[ContractionIndex]) -> Self {
        validate_path(path);
        // let mut tree = ContractionTree::default();
        let mut nodes = HashMap::new();
        let mut scratch = HashMap::new();
        for (tensor_idx, _) in tn.tensors().iter().enumerate() {
            let new_node = Node::new(
                nodes.len(),
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
                Some(tensor_idx),
            );
            nodes.insert(tensor_idx, Rc::new(RefCell::new(new_node)));
            scratch.insert(tensor_idx, Rc::clone(nodes.get(&tensor_idx).unwrap()));
        }

        for contr in path.iter() {
            match contr {
                ContractionIndex::Pair(i_path, j_path) => {
                    // Destructure contraction index. Check if contracted tensor already moved there.
                    let i = &scratch[i_path];
                    let j = &scratch[j_path];
                    let parent =
                        Node::new(nodes.len(), i.as_ptr(), j.as_ptr(), ptr::null_mut(), None);

                    nodes.insert(nodes.len(), Rc::new(RefCell::new(parent)));
                    i.borrow_mut().parent = nodes.get(&(nodes.len() - 1)).unwrap().as_ptr();
                    j.borrow_mut().parent = nodes.get(&(nodes.len() - 1)).unwrap().as_ptr();
                    scratch.insert(*i_path, Rc::clone(nodes.get(&(nodes.len() - 1)).unwrap()));
                    scratch.remove(j_path);
                }
                _ => {
                    panic!("Constructor not implemented for nested Tensors")
                }
            }
        }
        let parent = nodes.get(&(nodes.len() - 1)).unwrap().clone();
        ContractionTree {
            nodes,
            root: parent.as_ptr(),
        }
    }

    fn tree_depth_recurse(node: *mut Node) -> usize {
        let is_leaf;
        unsafe {
            is_leaf = (*node).is_leaf();
        }
        if is_leaf {
            0
        } else {
            1 + max(
                unsafe { ContractionTree::tree_depth_recurse((*node).left_child) },
                unsafe { ContractionTree::tree_depth_recurse((*node).right_child) },
            )
        }
    }

    /// Returns the depth of subtree in [`ContractionTree`] object starting from a given `node_index`. The depth of a [`Node`] object with no children is 0.
    ///
    /// # Arguments
    /// * `node_index` - `id` attribute of starting [`Node`]
    pub fn tree_depth(&self, node_index: usize) -> usize {
        ContractionTree::tree_depth_recurse(self.node_ptr(node_index))
    }

    fn leaf_count_recurse(node: *mut Node) -> usize {
        if node.is_null() {
            panic!("All non-leaf nodes should have two children in a contraction tree")
        }
        unsafe {
            if (*node).is_leaf() {
                1
            } else {
                ContractionTree::leaf_count_recurse((*node).left_child)
                    + ContractionTree::leaf_count_recurse((*node).right_child)
            }
        }
    }

    /// Returns the number of leaf nodes in subtree of [`ContractionTree`] object starting from a given `node_index`. Returns 1 if `node_index` points to a leaf node.
    ///
    /// # Arguments
    /// * `node_index` - `id` attribute of starting [`Node`]
    pub fn leaf_count(&self, node_index: usize) -> usize {
        ContractionTree::leaf_count_recurse(self.node_ptr(node_index))
    }

    fn leaf_ids_recurse(node: *mut Node, leaf_indices: &mut Vec<usize>) {
        let id;
        let left_child;
        let right_child;
        let is_leaf;

        unsafe {
            id = (*node).id;
            left_child = (*node).left_child;
            right_child = (*node).right_child;
            is_leaf = (*node).is_leaf();
        }
        if is_leaf {
            leaf_indices.push(id);
        } else {
            ContractionTree::leaf_ids_recurse(left_child, leaf_indices);
            ContractionTree::leaf_ids_recurse(right_child, leaf_indices);
        }
    }

    /// Populates `leaf_indices` with `id`` attribute of all leaf nodes in subtree with root at `node_index`.
    ///
    /// # Arguments
    /// * `node_index` - `id` attribute of starting [`Node`]
    /// * `leaf_indices` - mutable Vec that stores `id` attributes
    pub fn leaf_ids(&self, node_index: usize, leaf_indices: &mut Vec<usize>) {
        let node = self.node_ptr(node_index);
        ContractionTree::leaf_ids_recurse(node, leaf_indices);
    }

    /// Populates `children` with pointer to [`Node`] objects at depth  attribute of all leaf nodes in subtree with root at `node_index`.
    ///
    /// # Arguments
    /// * `node` - mutable pointer to [`Node`] object
    /// * `depth` - depth to find children of `node`
    /// * `children` - mutable vector storing found children
    pub fn nodes_at_depth(node: *mut Node, depth: usize, children: &mut Vec<*mut Node>) {
        if node.is_null() {
        } else if depth == 0 {
            children.push(node);
        } else {
            unsafe {
                ContractionTree::nodes_at_depth((*node).left_child, depth - 1, children);
                ContractionTree::nodes_at_depth((*node).right_child, depth - 1, children);
            }
        }
    }

    pub fn remove_subtree_recurse(&mut self, node: *mut Node) {
        if node.is_null() {
        } else {
            unsafe {
                self.remove_subtree_recurse((*node).left_child);
                self.remove_subtree_recurse((*node).right_child);
                self.remove_node((*node).id);
                (*node).deprecate();
            }
        }
    }

    pub fn remove_subtree(&mut self, node_id: usize) {
        self.remove_subtree_recurse(self.node_ptr(node_id));
    }

    pub fn remove_leaf_node(&mut self, leaf_id: usize) {
        assert!(self.node(leaf_id).is_leaf());
        let leaf = self.node_ptr(leaf_id);
        if leaf.is_null() {
            return;
        }
        unsafe {
            let leaf_parent = (*leaf).parent;
            let leaf_parent_id;
            if !leaf_parent.is_null() {
                leaf_parent_id = (*leaf_parent).id;
            } else {
                return;
            };

            let leaf_parent_parent = (*leaf_parent).parent;
            if leaf_parent_parent.is_null() {
                (*leaf_parent).remove_child(leaf);
            } else {
                let replacement_node = (*leaf_parent).get_other_child(leaf);
                (*leaf_parent_parent).replace_child(leaf_parent, replacement_node);
                (*replacement_node).set_parent(leaf_parent_parent);
                (*leaf_parent).deprecate();
                self.remove_node(leaf_parent_id);
            }
            (*leaf).deprecate();
        }
        unsafe {
            self.remove_node(leaf_id);
        }
    }

    fn replace_node(&mut self, node_index: usize, node: Node) {
        self.nodes.insert(node_index, Rc::new(RefCell::new(node)));
    }

    pub fn add_subtree(
        &mut self,
        tn: &Tensor,
        path: &[ContractionIndex],
        parent_id: usize,
        tensor_indices: Option<Vec<usize>>,
    ) {
        validate_path(path);
        assert!(self.nodes.contains_key(&parent_id));
        let mut index = 0;
        let mut scratch = HashMap::new();
        let tensor_indices = if let Some(tensor_indices) = tensor_indices {
            tensor_indices
        } else {
            (0..tn.tensors().len()).collect()
        };

        for tensor_idx in tensor_indices {
            scratch.insert(tensor_idx, Rc::clone(self.nodes.get(&tensor_idx).unwrap()));
        }

        for contr in path.iter() {
            match contr {
                ContractionIndex::Pair(i_path, j_path) => {
                    index = self.next_id(index);
                    // Destructure contraction index. Check if contracted tensor already moved there.
                    let i = &scratch[i_path];
                    let j = &scratch[j_path];
                    let parent = Node::new(index, i.as_ptr(), j.as_ptr(), ptr::null_mut(), None);

                    self.nodes.insert(index, Rc::new(RefCell::new(parent)));
                    i.borrow_mut().parent = self.nodes.get(&index).unwrap().as_ptr();
                    j.borrow_mut().parent = self.nodes.get(&index).unwrap().as_ptr();
                    scratch.insert(*i_path, Rc::clone(self.nodes.get(&index).unwrap()));
                    scratch.remove(j_path);
                }
                _ => {
                    panic!("Constructor not implemented for nested Tensors")
                }
            }
        }
        let new_parent = self.node_ptr(parent_id);
        unsafe {
            (*new_parent).add_child(self.node_ptr(index));
        }
    }

    fn tree_weights_recurse(
        &self,
        node: *mut Node,
        tn: &Tensor,
        weights: &mut HashMap<usize, u64>,
        scratch: &mut HashMap<usize, Tensor>,
        cost_function: fn(&Tensor, &Tensor) -> u64,
    ) {
        unsafe {
            if (*node).is_leaf() {
                weights.entry((*node).id).or_insert(0u64);
                let Some(tensor_index) = (*node).tensor_index else {
                    panic!("All leaf nodes should have a tensor index")
                };
                scratch
                    .entry((*node).id)
                    .or_insert(tn.tensor(tensor_index).clone());
                return;
            }
        }

        let left_child;
        let right_child;
        let left_child_id;
        let right_child_id;
        unsafe {
            left_child = (*node).left_child;
            right_child = (*node).right_child;
            left_child_id = (*left_child).id;
            right_child_id = (*right_child).id;
        }
        self.tree_weights_recurse(left_child, tn, weights, scratch, cost_function);
        self.tree_weights_recurse(right_child, tn, weights, scratch, cost_function);
        let t1 = scratch.get(&left_child_id).unwrap();
        let t2 = scratch.get(&right_child_id).unwrap();

        let cost = weights.get(&left_child_id).unwrap()
            + weights.get(&right_child_id).unwrap()
            + cost_function(t1, t2);

        unsafe {
            weights.insert((*node).id, cost);
            scratch.insert((*node).id, t1 ^ t2);
        }
    }

    /// Returns HashMap storing resultant tensor and its respective contraction costs calculated via `cost_function`.
    ///
    /// # Arguments
    /// * `node_id` - root of Node to start calculating contraction costs
    /// * `tn` - [`Tensor`] object containing bond dimension and leaf node information
    /// * `cost_function` - cost function taking two [`Tensor`] objects and returning contraction cost as u64
    fn tree_weights(
        &self,
        node_id: usize,
        tn: &Tensor,
        cost_function: fn(&Tensor, &Tensor) -> u64,
    ) -> HashMap<usize, u64> {
        let mut weights = HashMap::new();
        let mut scratch = HashMap::new();
        self.tree_weights_recurse(
            self.node_ptr(node_id),
            tn,
            &mut weights,
            &mut scratch,
            cost_function,
        );
        weights
    }

    /// Given a specific tensor at leaf node "n1" with id `node_index`, identifies tensor at node "n2" in ContractionTree subtree rooted at `subtree_root`, such that (n1, n2) maximizes provided cost function `cost_function`.
    ///
    /// # Arguments
    /// * `node_id` - leaf node used to calculation cost function, must be disjoint from subtree rooted at `subtree_root`
    /// * `subtree_root` - identifies root of subtree to be considered
    /// * `tn` - [`Tensor`] object containing bond dimension and leaf node information
    /// * `cost_function` - cost function taking two [`Tensor`] objects and returning a value as i64
    ///
    /// #Returns
    /// * option of node id (not necessarily a leaf node) in subtree that maximizes `cost_function`.
    pub fn max_match_by(
        &self,
        node_index: usize,
        subtree_root: usize,
        tn: &Tensor,
        cost_function: fn(&Tensor, &Tensor) -> i64,
    ) -> Option<usize> {
        assert!(self.nodes[&node_index].borrow().is_leaf());
        // Get a map that maps leaf nodes to corresponding [`Tensor`] objects.
        let mut node_tensor_map: HashMap<usize, Tensor> = HashMap::new();
        populate_subtree_tensor_map(self, subtree_root, &mut node_tensor_map, tn);

        let t1 = tn.tensor(self.nodes[&node_index].borrow().tensor_index.unwrap());

        // Find the tensor that maximizes cost function.
        let (node, _) = node_tensor_map
            .iter()
            .map(|(id, tensor)| (id, cost_function(tensor, t1)))
            .reduce(|node1, node2| cmp::max_by_key(node1, node2, |&a| a.1))?;
        Some(*node)
    }

    /// Given a list of leaf nodes ids `leaf_node_indices`, identifies leaf node id that maximizes `cost_function.
    ///
    /// # Arguments
    /// * `leaf_node_indices` - vector of lead node ids
    /// * `tn` - [`Tensor`] object containing bond dimension and leaf node information
    /// * `cost_function` - cost function taking one [`Tensor`] objects and returning a value as i64
    ///
    /// # Return
    /// * option of leaf node id that maximizes cost function
    pub fn max_leaf_node_by(
        &self,
        leaf_node_indices: &[usize],
        tn: &Tensor,
        cost_function: fn(&Tensor) -> u64,
    ) -> Option<usize> {
        let (node_index, _) = leaf_node_indices
            .iter()
            .map(|&node| (node, tn.tensor(self.node(node).tensor_index.unwrap())))
            .reduce(|node1, node2| cmp::max_by_key(node1, node2, |a| cost_function(a.1)))?;
        Some(node_index)
    }

    /// Populates given vector with contractions path of contraction tree starting at `node`.
    ///
    /// # Arguments
    /// * `node` - pointer to [`Node`] object
    /// * `path` - empty Vec<ContractionIndex> to store contraction path
    fn to_contraction_path_recurse(node: *mut Node, path: &mut Vec<ContractionIndex>) -> usize {
        unsafe {
            if (*node).is_leaf() {
                return (*node).id;
            }
        }
        let left_child;
        let right_child;
        unsafe {
            left_child = (*node).left_child;
            right_child = (*node).right_child;
        }
        if !left_child.is_null() && !right_child.is_null() {
            let mut t1_id = ContractionTree::to_contraction_path_recurse(left_child, path);
            let mut t2_id = ContractionTree::to_contraction_path_recurse(right_child, path);
            if t2_id < t1_id {
                (t1_id, t2_id) = (t2_id, t1_id);
            }
            path.push(pair!(t1_id, t2_id));
            t1_id
        } else {
            panic!("All parents should have two children")
        }
    }

    /// Populates given vector with contractions path of contraction tree starting at `node`.
    ///
    /// # Arguments
    /// * `node_id` - id of root of tree
    /// * `path` - empty Vec<ContractionIndex> to store contraction path
    pub fn to_contraction_path(&self, node_index: usize, path: &mut Vec<ContractionIndex>) {
        ContractionTree::to_contraction_path_recurse(self.node_ptr(node_index), path);
    }

    fn next_id(&self, mut init: usize) -> usize {
        while self.nodes.contains_key(&init) {
            init += 1;
        }
        init
    }
}

/// Populates HashMap<usize, Tensor> `node_tensor_map`  with all intermediate and leaf node ids and corresponding [`Tensor`] object, with root at `node_id`.
///
/// # Arguments
/// * `contraction_tree` - [`ContractionTree`] object
/// * `node_id` - root of subtree to examine
/// * `node_tensor_map` - empty HashMap to populate
/// * `tn` - [`Tensor`] object containing bond dimension and leaf node information
///
/// # Returns
/// final tensor after fully contracted.
fn populate_subtree_tensor_map(
    contraction_tree: &ContractionTree,
    node_id: usize,
    node_tensor_map: &mut HashMap<usize, Tensor>,
    tn: &Tensor,
) -> Tensor {
    let node = contraction_tree.node(node_id);

    if node.is_leaf() {
        let t = tn.tensor(node.tensor_index.unwrap());
        node_tensor_map.insert(node.id, t.clone());
        t.clone()
    } else {
        let t1;
        let t2;
        unsafe {
            t1 = populate_subtree_tensor_map(
                contraction_tree,
                (*node.left_child).id,
                node_tensor_map,
                tn,
            );
            t2 = populate_subtree_tensor_map(
                contraction_tree,
                (*node.right_child).id,
                node_tensor_map,
                tn,
            );
        }
        let t12 = &t1 ^ &t2;
        node_tensor_map.insert(node.id, t12.clone());
        t12
    }
}

/// Returns contraction cost of subtree in [`ContractionTree`] object.
///
/// # Arguments
/// * `contraction_tree` - [`ContractionTree`] object
/// * `node_id` - root of subtree to examine
/// * `tn` - [`Tensor`] object containing bond dimension and leaf node information
///
/// # Returns
/// Total op cost and maximum memory required of fully contracting subtree rooted at `node_id`
fn tree_contraction_cost(
    contraction_tree: &ContractionTree,
    node_index: usize,
    tn: &Tensor,
) -> (u64, u64) {
    let mut contraction_path = vec![];
    contraction_tree.to_contraction_path(node_index, &mut contraction_path);
    let contraction_path = ssa_replace_ordering(&contraction_path, tn.tensors().len());
    contract_path_cost(tn.tensors(), &contraction_path)
}

fn find_potential_nodes(
    contraction_tree: &ContractionTree,
    bigger_subtree_leaf_nodes: &[usize],
    smaller_subtree_root: usize,
    tn: &Tensor,
    cost_function: fn(&Tensor, &Tensor) -> i64,
) -> HashMap<usize, i64> {
    // This hashmap maps the node idx to it's weight. The weight is the
    // maximum memory reduction that can be achieved if it is shifted to the
    // other subtree.

    // Get a map that maps nodes to their tensors.
    let mut node_tensor_map: HashMap<usize, Tensor> = HashMap::new();
    populate_subtree_tensor_map(
        contraction_tree,
        smaller_subtree_root,
        &mut node_tensor_map,
        tn,
    );
    HashMap::from_iter(bigger_subtree_leaf_nodes.iter().map(|leaf_index| {
        let t1 = tn.tensor(contraction_tree.node(*leaf_index).tensor_index.unwrap());
        node_tensor_map
            .iter()
            .map(|(&index, tensor)| (index, cost_function(tensor, t1)))
            .reduce(|a, b| min_by_key(a, b, |a| a.1))
            .unwrap()
    }))
}

// fn find_matching_nodes_in_subtrees(
//     &mut self,
//     bigger_subtree_leaf_nodes: &Vec<usize>,
//     smaller_subtree_root: usize,
//     tn: &Tensor,
// ) -> (usize, usize) {
//     // Get a map that maps nodes to their tensors.
//     let mut node_tensor_map: HashMap<usize, Tensor> = HashMap::new();
//     self.get_subtree_tensors(smaller_subtree_root, &mut node_tensor_map, &tn);
//     let mut cur_tensor_mem: i64 = 0;
//     let mut max_tensor_mem: i64 = std::i64::MAX;
//     let mut best_leaf: Option<usize> = None;
//     let mut best_smaller_subtree_node: Option<usize> = None;
//     for &leaf_idx in bigger_subtree_leaf_nodes {
//         let t1: Tensor = tn.tensors()[self.nodes[leaf_idx].tensor_idx.unwrap()].clone();
//         for (&cur_node_idx, t) in &node_tensor_map {
//             let t2: Tensor = t.clone();
//             let t12 = &t1 ^ &t2;
//             // println!("t12.size(): {}", t12.size() as i64);
//             // println!("t1.size(): {}", t1.size() as i64);
//             // println!("t2.size(): {}\n", t2.size() as i64);
//             cur_tensor_mem = (t12.size() as i64) - (t1.size() as i64) - (t2.size() as i64);
//             // cur_tensor_mem = t12.size() as i64;
//             // if cur_tensor_mem >= max_tensor_mem {
//             if cur_tensor_mem <= max_tensor_mem {
//                 max_tensor_mem = cur_tensor_mem;
//                 best_leaf = Some(leaf_idx);
//                 best_smaller_subtree_node = Some(cur_node_idx);
//             }
//         }
//     }
//     // println!("Tensor memory reduction: {:?}", max_tensor_mem);
//     (best_leaf.unwrap(), best_smaller_subtree_node.unwrap())
// }

fn greedy_cost_fn(t1: &Tensor, t2: &Tensor) -> i64 {
    ((t1 ^ t2).size() as i64) - (t1.size() as i64) - (t2.size() as i64)
}

pub fn rebalance_path(
    tn: &Tensor,
    path: &[ContractionIndex],
    _orig_greedy: bool,
    rebal_random: bool,
    depth: usize,
) -> Vec<ContractionIndex> {
    // If there are less than 3 tensors in the tn, rebalancing will not make sense.
    if tn.tensors().len() < 3 {
        println!("No rebalancing undertaken, as tn is too small (< 3 tensors)");
        return path.to_vec();
    }

    // 1. Create binary contraction tree. It details how tensors are contracted via the given path.
    let mut tree = ContractionTree::from_contraction_path(tn, path);
    // let json = tree.to_json().unwrap();
    // write_to_file(&json, "contr-tree-original.json");

    let mut children = vec![];
    ContractionTree::nodes_at_depth(tree.root, depth, &mut children);

    // 2. Select the bigger subtree. Based on maximum contraction op cost.
    // let (l_op_cost, _r_op_cost) = tree_contraction_cost(&tree, tree.root.unwrap(), tn);
    let (smaller_subtree, larger_subtree) = find_min_max_subtree(children, &tree, tn);

    // 3. Find the tensor to be rebalanced to the bigger subtree.
    // Get smaller subtree leaf nodes
    let mut smaller_subtree_leaf_nodes = Vec::new();
    ContractionTree::leaf_ids_recurse(smaller_subtree, &mut smaller_subtree_leaf_nodes);

    // Get bigger subtree leaf nodes
    let mut larger_subtree_leaf_nodes = Vec::new();
    ContractionTree::leaf_ids_recurse(larger_subtree, &mut larger_subtree_leaf_nodes);

    // /* 3.3 Select the leaf node in the smaller subtree that causes the biggest memory reduction in the bigger subtree
    let larger_subtree_id;
    let smaller_subtree_id;
    unsafe {
        larger_subtree_id = (*larger_subtree).id;
        smaller_subtree_id = (*smaller_subtree).id;
    }
    let rebal_node = if rebal_random {
        // Randomly select one of the top n nodes to rebal.

        let top_n = 5;
        let rebal_nodes_weight = find_potential_nodes(
            &tree,
            &larger_subtree_leaf_nodes,
            smaller_subtree_id,
            tn,
            greedy_cost_fn,
        );

        let mut keys = rebal_nodes_weight.keys().cloned().collect::<Vec<usize>>();
        keys.sort_by_key(|&key| rebal_nodes_weight[&key]);
        if keys.len() < top_n {
            println!("Error rebalance_path: Not enough nodes in the bigger subtree to select the top {} from!", top_n);
            tree.max_match_by(larger_subtree_id, smaller_subtree_id, tn, greedy_cost_fn)
                .unwrap()
        } else {
            // Sample randomly from the top n nodes. Use softmax probabilities.
            let top_n_nodes = keys.iter().take(top_n).cloned().collect::<Vec<usize>>();
            let top_n_weights: Vec<i64> = top_n_nodes
                .iter()
                .map(|idx| rebal_nodes_weight[idx])
                .collect();

            // Subtract max val after inverting for numerical stability.
            let l2_norm: f64 = top_n_nodes
                .iter()
                .map(|idx| rebal_nodes_weight[idx] as f64)
                .map(|weight| weight * weight)
                .sum::<f64>()
                .sqrt();
            let top_n_exp: Vec<f64> = top_n_nodes
                .iter()
                .map(|idx| ((-rebal_nodes_weight[idx] as f64) / l2_norm).exp())
                .collect();

            let sum_exp: f64 = top_n_exp.iter().sum();
            let top_n_prob: Vec<f64> = top_n_exp.iter().map(|&exp| (exp / sum_exp)).collect();

            // Debug
            println!("top_n_nodes: {:?}", top_n_nodes);
            println!("top_n_weights: {:?}", top_n_weights);
            println!("top_n_prob: {:?}", top_n_prob);

            // Sample index based on its probability
            let dist = WeightedIndex::new(&top_n_prob).unwrap();
            let mut rng = thread_rng();
            let rand_idx = dist.sample(&mut rng);
            // Debug
            println!("rand_idx: {:?}", rand_idx);
            top_n_nodes[rand_idx]
        }
    } else {
        tree.max_match_by(larger_subtree_id, smaller_subtree_id, tn, greedy_cost_fn)
            .unwrap()
    };

    // 4. Remove selected tensor from bigger subtree. Add it to the smaller subtree
    smaller_subtree_leaf_nodes.push(rebal_node);
    larger_subtree_leaf_nodes.retain(|&leaf| leaf != rebal_node);

    // 5. Rerun Greedy algorithm on smaller subtree

    // Delete edge between root and smaller subtree. Will use greedy path instead
    let (index, smaller_tensors): (Vec<usize>, Vec<Tensor>) = smaller_subtree_leaf_nodes
        .iter()
        .map(|&e| (e, tn.tensor(tree.node(e).tensor_index.unwrap()).clone()))
        .unzip();

    let tn_smaller_subtree = create_tensor_network(smaller_tensors, &tn.bond_dims(), None);

    let mut opt = Greedy::new(&tn_smaller_subtree, CostType::Flops);
    opt.optimize_path();
    let path_smaller_subtree = opt.get_best_replace_path();
    let updated_path = path_smaller_subtree
        .iter()
        .map(|e| match e {
            ContractionIndex::Pair(v1, v2) => pair!(index[*v1], index[*v2]),
            _ => panic!("Should only produce Pairs!"),
        })
        .collect::<Vec<ContractionIndex>>();

    // Remove smaller subtree from contraction_path
    let parent_id;
    unsafe {
        parent_id = (*(*smaller_subtree).parent).id;
    }
    tree.remove_subtree_recurse(smaller_subtree);
    tree.add_subtree(tn, &updated_path, parent_id, Some(index));

    // 6. Rerun Greedy algorithm on bigger subtree
    // /*
    let (index, larger_tensors): (Vec<usize>, Vec<Tensor>) = larger_subtree_leaf_nodes
        .iter()
        .map(|&e| (e, tn.tensor(tree.node(e).tensor_index.unwrap()).clone()))
        .unzip();

    let tn_larger_subtree = create_tensor_network(larger_tensors, &tn.bond_dims(), None);

    let mut opt = Greedy::new(&tn_larger_subtree, CostType::Flops);
    opt.optimize_path();
    let path_larger_subtree = opt.get_best_replace_path();
    let updated_path = path_larger_subtree
        .iter()
        .map(|e| match e {
            ContractionIndex::Pair(v1, v2) => pair!(index[*v1], index[*v2]),
            _ => panic!("Should only produce Pairs!"),
        })
        .collect::<Vec<ContractionIndex>>();

    // Remove smaller subtree from contraction_path
    let parent_id;
    unsafe {
        parent_id = (*(*larger_subtree).parent).id;
    }
    tree.remove_subtree_recurse(larger_subtree);
    tree.add_subtree(tn, &updated_path, parent_id, Some(index));

    // 7. Generate new path based on greedy paths
    let mut rebal_path = Vec::new();
    ContractionTree::to_contraction_path_recurse(tree.root, &mut rebal_path);
    rebal_path
}

fn find_min_max_subtree(
    children: Vec<*mut Node>,
    tree: &ContractionTree,
    tn: &Tensor,
) -> (*mut Node, *mut Node) {
    let mut min_cost = std::u64::MAX;
    let mut max_cost = 0;

    let mut smaller_subtree = children[0];
    let mut bigger_subtree = children[0];

    for (child_id, (op_cost, _mem_cost)) in children
        .iter()
        .map(|&a| unsafe { (a, tree_contraction_cost(tree, (*a).id, tn)) })
    {
        if op_cost > max_cost {
            max_cost = op_cost;
            bigger_subtree = child_id;
        }
        if op_cost < min_cost {
            min_cost = op_cost;
            smaller_subtree = child_id;
        }
    }
    (smaller_subtree, bigger_subtree)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::ptr;

    use crate::contractionpath::contraction_cost::contract_cost_tensors;
    use crate::contractionpath::contraction_tree::{ContractionTree, Node};
    use crate::path;
    use crate::tensornetwork::create_tensor_network;
    use crate::tensornetwork::tensor::Tensor;
    use crate::types::ContractionIndex;

    use super::populate_subtree_tensor_map;

    fn setup_simple() -> (Tensor, Vec<ContractionIndex>) {
        (
            create_tensor_network(
                vec![
                    Tensor::new(vec![4, 3, 2]),
                    Tensor::new(vec![0, 1, 3, 2]),
                    Tensor::new(vec![4, 5, 6]),
                ],
                &[(0, 5), (1, 2), (2, 6), (3, 8), (4, 1), (5, 3), (6, 4)].into(),
                None,
            ),
            path![(0, 1), (0, 2)].to_vec(),
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
                &[
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
                ]
                .into(),
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
                &[
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
                ]
                .into(),
                None,
            ),
            path![(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)].to_vec(),
        )
    }

    #[test]
    fn test_constructor_simple() {
        let (tensor, path) = setup_simple();
        let ContractionTree { nodes, root } =
            ContractionTree::from_contraction_path(&tensor, &path);

        let mut node0 = Node::new(
            0,
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            Some(0),
        );
        let mut node1 = Node::new(
            1,
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            Some(1),
        );
        let mut node2 = Node::new(
            2,
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            Some(2),
        );
        let mut node3 = Node::new(3, &mut node0, &mut node1, ptr::null_mut(), None);
        let mut node4 = Node::new(4, &mut node3, &mut node2, ptr::null_mut(), None);
        node0.set_parent(&mut node3);
        node1.set_parent(&mut node3);
        node2.set_parent(&mut node4);
        node3.set_parent(&mut node4);

        let ref_root = node4.clone();
        let ref_nodes = vec![node0, node1, node2, node3, node4];

        for (key, ref_node) in ref_nodes.iter().enumerate().rev() {
            let node = nodes.get(&key).unwrap().borrow();

            let Node {
                id: ref_id,
                left_child: ref_left_child,
                right_child: ref_right_child,
                parent: ref_parent,
                tensor_index: ref_tensor_idx,
            } = ref_node;
            assert_eq!(node.id, *ref_id);
            assert_eq!(node.left_child.is_null(), (*ref_left_child).is_null());
            if !node.left_child.is_null() {
                unsafe {
                    assert_eq!((*node.left_child).id, (**ref_left_child).id);
                }
            }
            assert_eq!(node.right_child.is_null(), (*ref_right_child).is_null());
            if !node.right_child.is_null() {
                unsafe {
                    assert_eq!((*node.right_child).id, (**ref_right_child).id);
                }
            }

            assert_eq!(node.parent.is_null(), (*ref_parent).is_null());
            if !node.parent.is_null() {
                unsafe {
                    assert_eq!((*node.parent).id, (*(*ref_parent)).id);
                }
            }

            assert_eq!(node.tensor_index, *ref_tensor_idx);
        }
        unsafe {
            assert_eq!((*root).id, ref_root.id);
        }
    }

    #[test]
    fn test_constructor_complex() {
        let (tensor, path) = setup_complex();
        let ContractionTree { nodes, root } =
            ContractionTree::from_contraction_path(&tensor, &path);

        let mut node0 = Node::new(
            0,
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            Some(0),
        );
        let mut node1 = Node::new(
            1,
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            Some(1),
        );
        let mut node2 = Node::new(
            2,
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            Some(2),
        );
        let mut node3 = Node::new(
            3,
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            Some(3),
        );
        let mut node4 = Node::new(
            4,
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            Some(4),
        );

        let mut node5 = Node::new(
            5,
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            Some(5),
        );

        let mut node6 = Node::new(6, &mut node1, &mut node5, ptr::null_mut(), None);
        let mut node7 = Node::new(7, &mut node0, &mut node6, ptr::null_mut(), None);
        let mut node8 = Node::new(8, &mut node3, &mut node4, ptr::null_mut(), None);
        let mut node9 = Node::new(9, &mut node2, &mut node8, ptr::null_mut(), None);
        let mut node10 = Node::new(10, &mut node7, &mut node9, ptr::null_mut(), None);
        node0.set_parent(&mut node7);
        node1.set_parent(&mut node6);
        node2.set_parent(&mut node9);
        node3.set_parent(&mut node8);
        node4.set_parent(&mut node8);
        node5.set_parent(&mut node6);
        node6.set_parent(&mut node7);
        node7.set_parent(&mut node10);
        node8.set_parent(&mut node9);
        node9.set_parent(&mut node10);

        let ref_root = node10.clone();
        let ref_nodes = vec![
            node0, node1, node2, node3, node4, node5, node6, node7, node8, node9, node10,
        ];

        for (key, ref_node) in ref_nodes.iter().enumerate() {
            let node = nodes.get(&key).unwrap().borrow();
            let Node {
                id: ref_id,
                left_child: ref_left_child,
                right_child: ref_right_child,
                parent: ref_parent,
                tensor_index: ref_tensor_idx,
            } = ref_node;
            assert_eq!(node.id, *ref_id);
            assert_eq!(node.left_child.is_null(), (*ref_left_child).is_null());
            if !node.left_child.is_null() {
                unsafe {
                    assert_eq!((*node.left_child).id, (**ref_left_child).id);
                }
            }
            assert_eq!(node.right_child.is_null(), (*ref_right_child).is_null());
            if !node.right_child.is_null() {
                unsafe {
                    assert_eq!((*node.right_child).id, (**ref_right_child).id);
                }
            }

            assert_eq!(node.parent.is_null(), (*ref_parent).is_null());
            if !node.parent.is_null() {
                unsafe {
                    assert_eq!((*node.parent).id, (*(*ref_parent)).id);
                }
            }

            assert_eq!(node.tensor_index, *ref_tensor_idx);
        }
        unsafe {
            assert_eq!((*root).id, ref_root.id);
        }
    }

    #[test]
    fn test_tree_depth_simple() {
        let (tn, path) = setup_simple();
        let tree = ContractionTree::from_contraction_path(&tn, &path);
        assert_eq!(tree.tree_depth(4), 2);
        assert_eq!(tree.tree_depth(3), 1);
        assert_eq!(tree.tree_depth(2), 0);
        assert_eq!(tree.tree_depth(1), 0);
        assert_eq!(tree.tree_depth(0), 0);
    }

    #[test]
    fn test_tree_depth_complex() {
        let (tn, path) = setup_complex();
        let tree = ContractionTree::from_contraction_path(&tn, &path);
        assert_eq!(tree.tree_depth(10), 3);
        assert_eq!(tree.tree_depth(9), 2);
        assert_eq!(tree.tree_depth(8), 1);
        assert_eq!(tree.tree_depth(7), 2);
        assert_eq!(tree.tree_depth(6), 1);
        assert_eq!(tree.tree_depth(5), 0);
        assert_eq!(tree.tree_depth(4), 0);
        assert_eq!(tree.tree_depth(3), 0);
        assert_eq!(tree.tree_depth(2), 0);
        assert_eq!(tree.tree_depth(1), 0);
        assert_eq!(tree.tree_depth(0), 0);
    }

    #[test]
    fn test_leaf_count_simple() {
        let (tn, path) = setup_simple();
        let tree = ContractionTree::from_contraction_path(&tn, &path);
        assert_eq!(tree.leaf_count(4), 3);
        assert_eq!(tree.leaf_count(3), 2);
        assert_eq!(tree.leaf_count(2), 1);
        assert_eq!(tree.leaf_count(1), 1);
        assert_eq!(tree.leaf_count(0), 1);
    }

    #[test]
    fn test_leaf_count_complex() {
        let (tn, path) = setup_complex();
        let tree = ContractionTree::from_contraction_path(&tn, &path);
        assert_eq!(tree.leaf_count(10), 6);
        assert_eq!(tree.leaf_count(9), 3);
        assert_eq!(tree.leaf_count(8), 2);
        assert_eq!(tree.leaf_count(7), 3);
        assert_eq!(tree.leaf_count(6), 2);
        assert_eq!(tree.leaf_count(5), 1);
        assert_eq!(tree.leaf_count(4), 1);
        assert_eq!(tree.leaf_count(3), 1);
        assert_eq!(tree.leaf_count(2), 1);
        assert_eq!(tree.leaf_count(1), 1);
        assert_eq!(tree.leaf_count(0), 1);
    }

    #[test]
    fn test_leaf_ids_simple() {
        let (tn, path) = setup_simple();
        let tree = ContractionTree::from_contraction_path(&tn, &path);
        let mut leaf_ids = Vec::new();
        tree.leaf_ids(4, &mut leaf_ids);
        assert_eq!(leaf_ids, vec![0, 1, 2]);

        leaf_ids = Vec::new();
        tree.leaf_ids(3, &mut leaf_ids);
        assert_eq!(leaf_ids, vec![0, 1]);

        leaf_ids = Vec::new();
        tree.leaf_ids(2, &mut leaf_ids);
        assert_eq!(leaf_ids, vec![2]);
    }

    #[test]
    fn test_leaf_ids_complex() {
        let (tn, path) = setup_complex();
        let tree = ContractionTree::from_contraction_path(&tn, &path);
        let mut leaf_ids = Vec::new();
        tree.leaf_ids(10, &mut leaf_ids);
        leaf_ids.sort();
        assert_eq!(leaf_ids, vec![0, 1, 2, 3, 4, 5]);

        leaf_ids = Vec::new();
        tree.leaf_ids(9, &mut leaf_ids);
        assert_eq!(leaf_ids, vec![2, 3, 4]);

        leaf_ids = Vec::new();
        tree.leaf_ids(8, &mut leaf_ids);
        assert_eq!(leaf_ids, vec![3, 4]);

        leaf_ids = Vec::new();
        tree.leaf_ids(7, &mut leaf_ids);
        assert_eq!(leaf_ids, vec![0, 1, 5]);

        leaf_ids = Vec::new();
        tree.leaf_ids(6, &mut leaf_ids);
        assert_eq!(leaf_ids, vec![1, 5]);

        leaf_ids = Vec::new();
        tree.leaf_ids(3, &mut leaf_ids);
        assert_eq!(leaf_ids, vec![3]);
    }

    #[test]
    fn test_nodes_at_depth() {
        let (tensor, path) = setup_simple();
        let tree = ContractionTree::from_contraction_path(&tensor, &path);

        let mut leaves = vec![];
        ContractionTree::nodes_at_depth(tree.node_ptr(4), 0, &mut leaves);

        unsafe {
            assert_eq!((*leaves[0]).id, 4);
        }

        leaves = vec![];
        ContractionTree::nodes_at_depth(tree.node_ptr(4), 1, &mut leaves);

        unsafe {
            assert_eq!((*leaves[0]).id, 3);
            assert_eq!((*leaves[1]).id, 2);
        }

        leaves = vec![];
        ContractionTree::nodes_at_depth(tree.node_ptr(4), 2, &mut leaves);
        unsafe {
            assert_eq!((*leaves[0]).id, 0);
            assert_eq!((*leaves[1]).id, 1);
        }
    }

    #[test]
    fn test_nodes_at_depth_complex() {
        let (tensor, path) = setup_complex();
        let tree = ContractionTree::from_contraction_path(&tensor, &path);

        let mut leaves = vec![];
        ContractionTree::nodes_at_depth(tree.node_ptr(10), 0, &mut leaves);
        unsafe {
            assert_eq!((*leaves[0]).id, 10);
        }

        leaves = vec![];
        ContractionTree::nodes_at_depth(tree.node_ptr(10), 1, &mut leaves);
        unsafe {
            assert_eq!((*leaves[0]).id, 7);
            assert_eq!((*leaves[1]).id, 9);
        }

        leaves = vec![];
        ContractionTree::nodes_at_depth(tree.node_ptr(10), 2, &mut leaves);
        unsafe {
            assert_eq!((*leaves[0]).id, 0);
            assert_eq!((*leaves[1]).id, 6);
            assert_eq!((*leaves[2]).id, 2);
            assert_eq!((*leaves[3]).id, 8);
        }

        leaves = vec![];
        ContractionTree::nodes_at_depth(tree.node_ptr(10), 3, &mut leaves);
        unsafe {
            assert_eq!((*leaves[0]).id, 1);
            assert_eq!((*leaves[1]).id, 5);
            assert_eq!((*leaves[2]).id, 3);
            assert_eq!((*leaves[3]).id, 4);
        }
    }

    #[test]
    fn test_tree_weights_simple() {
        let (tensor, path) = setup_simple();
        let tree = ContractionTree::from_contraction_path(&tensor, &path);
        let ref_weights = HashMap::from([(1, 0), (0, 0), (2, 0), (3, 480), (4, 600)]);
        let weights = tree.tree_weights(4, &tensor, contract_cost_tensors);

        assert_eq!(weights, ref_weights);
        let ref_weights = HashMap::from([(1, 0), (0, 0), (3, 480)]);
        let weights = tree.tree_weights(3, &tensor, contract_cost_tensors);
        assert_eq!(weights, ref_weights);

        assert_eq!(weights, ref_weights);
        let ref_weights = HashMap::from([(2, 0)]);
        let weights = tree.tree_weights(2, &tensor, contract_cost_tensors);
        assert_eq!(weights, ref_weights);
    }

    #[test]
    fn test_tree_weights_complex() {
        let (tensor, path) = setup_complex();
        let tree = ContractionTree::from_contraction_path(&tensor, &path);
        let ref_weights = HashMap::from([
            (0, 0),
            (1, 0),
            (2, 0),
            (3, 0),
            (4, 0),
            (5, 0),
            (6, 262440),
            (7, 265140),
            (8, 263250),
            (9, 264600),
            (10, 529815),
        ]);
        let weights = tree.tree_weights(10, &tensor, contract_cost_tensors);

        assert_eq!(weights, ref_weights);
    }

    #[test]
    fn test_max_match_by_complex() {
        let (tensor, path) = setup_complex();
        let tree = ContractionTree::from_contraction_path(&tensor, &path);

        fn greedy_cost_fn(t1: &Tensor, t2: &Tensor) -> i64 {
            (t1.size() as i64) + (t2.size() as i64) - ((t1 ^ t2).size() as i64)
        }
        let max_match = tree.max_match_by(2, 7, &tensor, greedy_cost_fn).unwrap();

        assert_eq!(max_match, 7);

        fn max_memory_cost_fn(t1: &Tensor, t2: &Tensor) -> i64 {
            (t1 ^ t2).size() as i64
        }

        let max_match = tree
            .max_match_by(2, 7, &tensor, max_memory_cost_fn)
            .unwrap();
        assert_eq!(max_match, 1);
    }

    #[test]
    fn test_match_leaf_by_complex() {
        let (tensor, path) = setup_complex();
        let tree = ContractionTree::from_contraction_path(&tensor, &path);
        let leaf_node_indices = vec![0, 1, 2, 3, 4, 5];

        fn greedy_cost_fn(t1: &Tensor) -> u64 {
            t1.size()
        }
        let max_match = tree
            .max_leaf_node_by(&leaf_node_indices, &tensor, greedy_cost_fn)
            .unwrap();

        assert_eq!(max_match, 1);

        fn min_greedy_cost_fn(t1: &Tensor) -> u64 {
            std::u64::MAX - t1.size()
        }

        let max_match = tree
            .max_leaf_node_by(&leaf_node_indices, &tensor, min_greedy_cost_fn)
            .unwrap();
        assert_eq!(max_match, 2);
    }

    #[test]
    fn test_to_contraction_path_simple() {
        let (tensor, ref_path) = setup_simple();
        let tree = ContractionTree::from_contraction_path(&tensor, &ref_path);
        let mut path = Vec::new();
        tree.to_contraction_path(4, &mut path);
        let path = ssa_replace_ordering(&path, 3);
        assert_eq!(path, ref_path);
    }

    #[test]
    fn test_to_contraction_path_complex() {
        let (tensor, ref_path) = setup_complex();
        let tree = ContractionTree::from_contraction_path(&tensor, &ref_path);
        let mut path = Vec::new();
        tree.to_contraction_path(10, &mut path);
        let path = ssa_replace_ordering(&path, 6);
        assert_eq!(path, ref_path);
    }

    #[test]
    fn test_to_contraction_path_unbalanced() {
        let (tensor, ref_path) = setup_unbalanced();
        let tree = ContractionTree::from_contraction_path(&tensor, &ref_path);
        let mut path = Vec::new();
        tree.to_contraction_path(10, &mut path);
        let path = ssa_replace_ordering(&path, 6);
        assert_eq!(path, ref_path);
    }

    #[test]
    fn test_populate_subtree_tensor_map_simple() {
        let (tensor, ref_path) = setup_simple();
        let tree = ContractionTree::from_contraction_path(&tensor, &ref_path);
        let mut node_tensor_map = HashMap::new();
        populate_subtree_tensor_map(&tree, 4, &mut node_tensor_map, &tensor);

        let ref_node_tensor_map = HashMap::from([
            (0, Tensor::new(vec![4, 3, 2])),
            (1, Tensor::new(vec![0, 1, 3, 2])),
            (2, Tensor::new(vec![4, 5, 6])),
            (3, Tensor::new(vec![4, 0, 1])),
            (4, Tensor::new(vec![0, 1, 5, 6])),
        ]);

        for (key, value) in ref_node_tensor_map.iter() {
            assert_eq!(node_tensor_map[key].legs(), value.legs());
        }
    }

    #[test]
    fn test_populate_subtree_tensor_map_complex() {
        let (tensor, ref_path) = setup_complex();
        let tree = ContractionTree::from_contraction_path(&tensor, &ref_path);
        let mut node_tensor_map = HashMap::new();
        populate_subtree_tensor_map(&tree, 10, &mut node_tensor_map, &tensor);

        let ref_node_tensor_map = HashMap::from([
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

        for (key, value) in ref_node_tensor_map.iter() {
            assert_eq!(node_tensor_map[key].legs(), value.legs());
        }
    }
}
