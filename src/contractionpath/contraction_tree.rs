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

    ///
    pub fn mut_node(&mut self, tensor_id: usize) -> RefMut<Node> {
        self.nodes.get_mut(&tensor_id).unwrap().borrow_mut()
    }

    pub fn node(&self, tensor_id: usize) -> Ref<Node> {
        self.nodes.get(&tensor_id).unwrap().borrow()
    }

    pub fn node_ptr(&self, tensor_id: usize) -> *mut Node {
        self.nodes.get(&tensor_id).unwrap().as_ptr()
    }

    /// Removes node from HashMap. Warning! As HashMap stores Node data, removing a Node here can result in invalid references.
    unsafe fn remove_node(&mut self, node_id: usize) {
        self.nodes.remove(&node_id);
    }

    #[must_use]
    /// Constructor from Replace path
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
        if node.is_null() {
            0
        } else {
            1 + max(
                unsafe { ContractionTree::tree_depth_recurse((*node).left_child) },
                unsafe { ContractionTree::tree_depth_recurse((*node).right_child) },
            )
        }
    }

    fn tree_depth(&self, node_index: usize) -> usize {
        ContractionTree::tree_depth_recurse(self.node_ptr(node_index))
    }

    fn leaf_count_recurse(node: *mut Node) -> usize {
        if node.is_null() {
            return 0;
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

    fn leaf_count(&self, node_index: usize) -> usize {
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

    fn leaf_ids(&self, node_index: usize, leaf_indices: &mut Vec<usize>) {
        let node = self.node_ptr(node_index);
        ContractionTree::leaf_ids_recurse(node, leaf_indices);
    }

    fn nodes_at_depth(node: *mut Node, depth: usize, children: &mut Vec<*mut Node>) {
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

    /// Identifies node in subtree that maximizes `cost_function` with a specific node at `node_index`.
    fn max_match_by(
        &mut self,
        node_index: usize,
        subtree_root: usize,
        tn: &Tensor,
        cost_function: fn(&Tensor, &Tensor) -> i64,
    ) -> Option<usize> {
        // Get a map that maps nodes to their tensors.
        let mut node_tensor_map: HashMap<usize, Tensor> = HashMap::new();
        populate_subtree_tensor_map(self, subtree_root, &mut node_tensor_map, tn);

        // Find the tensor that removes the most memory when contracting.
        let t1 = tn.tensor(self.nodes[&node_index].borrow().tensor_index.unwrap());

        let (node, _) = node_tensor_map
            .iter()
            .map(|(id, tensor)| (id, cost_function(tensor, t1)))
            .reduce(|node1, node2| cmp::max_by_key(node1, node2, |&a| a.1))?;
        Some(*node)
    }

    fn max_leaf_node_by(
        &self,
        leaf_node_indices: &[usize],
        tn: &Tensor,
        cost: fn(u64) -> u64,
    ) -> Option<usize> {
        let (node_index, _) = leaf_node_indices
            .iter()
            .map(|&node| {
                (
                    node,
                    tn.tensor(self.node(node).tensor_index.unwrap()).size(),
                )
            })
            .reduce(|node1, node2| cmp::max_by_key(node1, node2, |a| cost(a.1)))?;
        Some(node_index)
    }

    fn to_contraction_path_recurse(node: *mut Node, path: &mut Vec<ContractionIndex>) -> usize {
        let left_child;
        let right_child;
        unsafe {
            left_child = (*node).left_child;
            right_child = (*node).right_child;
        }
        if !left_child.is_null() && right_child.is_null() {
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
                (*node.left_child).id,
                node_tensor_map,
                tn,
            );
        }
        let t12 = &t1 ^ &t2;
        node_tensor_map.insert(node.id, t12.clone());
        t12
    }
}

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
    let (smaller_subtree, larger_subtree) = find_minmax_subtree(children, &tree, tn);

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

fn find_minmax_subtree(
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
    use std::ptr;

    use crate::contractionpath::contraction_tree::{ContractionTree, Node};
    use crate::path;
    use crate::tensornetwork::create_tensor_network;
    use crate::tensornetwork::tensor::Tensor;
    use crate::types::ContractionIndex;

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
                    (11, 17),
                ]
                .into(),
                None,
            ),
            path![(1, 5), (3, 4), (0, 1), (2, 3), (0, 2)].to_vec(),
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
    fn test_simple() {
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
            println!("Ref_node id: {:?}", ref_node.id);

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
    fn test_complex() {
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
        let mut node7 = Node::new(7, &mut node3, &mut node4, ptr::null_mut(), None);
        let mut node8 = Node::new(8, &mut node0, &mut node6, ptr::null_mut(), None);
        let mut node9 = Node::new(9, &mut node2, &mut node7, ptr::null_mut(), None);
        let mut node10 = Node::new(10, &mut node8, &mut node9, ptr::null_mut(), None);
        node0.set_parent(&mut node8);
        node1.set_parent(&mut node6);
        node2.set_parent(&mut node9);
        node3.set_parent(&mut node7);
        node4.set_parent(&mut node7);
        node5.set_parent(&mut node6);
        node6.set_parent(&mut node8);
        node7.set_parent(&mut node9);
        node8.set_parent(&mut node10);
        node9.set_parent(&mut node10);

        let ref_root = node10.clone();
        let ref_nodes = vec![
            node0, node1, node2, node3, node4, node5, node6, node7, node8, node9, node10,
        ];

        for (key, ref_node) in ref_nodes.iter().enumerate() {
            println!("Key: {:?}", key);
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
}
