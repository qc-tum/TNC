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
#[derive(Debug, Serialize, Deserialize, Default)]
struct ContractionTree {
    nodes: Vec<Node>,
    root: Option<usize>,
}

impl ContractionTree {
    fn add_node(&mut self, node: Node) -> usize {
        let index = self.nodes.len();
        self.nodes.push(node);
        index
    }

    fn mut_node(&mut self, tensor_id: usize) -> &mut Node {
        self.nodes.get_mut(tensor_id).unwrap()
    }

    fn node(&self, tensor_id: usize) -> &Node {
        self.nodes.get(tensor_id).unwrap()
    }

    #[must_use]
    pub fn from_contraction_path(tn: &Tensor, path: &[ContractionIndex]) -> Self {
        validate_path(path);
        let mut tree = ContractionTree::default();
        let mut moved_tensors: HashMap<usize, usize> = HashMap::new();
        for (tensor_idx, _) in tn.tensors().iter().enumerate() {
            tree.add_node(Node::new(
                tensor_idx.to_string(),
                None,
                None,
                None,
                Some(tensor_idx),
            ));
            moved_tensors.insert(tensor_idx, tensor_idx);
        }

        for contr in path.iter() {
            match contr {
                ContractionIndex::Pair(i_path, j_path) => {
                    // If the path contains an instruction to contract a tensor with itself, don't do it.
                    if *i_path == *j_path {
                        continue;
                    }

                    // Destructure contraction index. Check if contracted tensor already moved there.
                    let i = *moved_tensors.get(i_path).unwrap();
                    let j = *moved_tensors.get(j_path).unwrap();

                    tree.add_node(Node::new(
                        format!("({},{})", tree.nodes[i].id, tree.nodes[j].id),
                        Some(i),
                        Some(j),
                        None,
                        None,
                    ));
                    let cur: usize = tree.nodes.len() - 1;
                    tree.mut_node(i).set_parent(cur);
                    tree.mut_node(j).set_parent(cur);
                    moved_tensors.insert(*i_path, cur);
                    moved_tensors.remove(j_path);
                }
                _ => {
                    panic!("Rebalancing not implemented for nested Tensors")
                }
            }
        }
        tree.root = Some(tree.nodes.len() - 1);
        tree
    }

    fn leaf_node_indices(&self, node_index: usize, leaf_indices: &mut Vec<usize>) {
        if self.node(node_index).is_leaf() {
            leaf_indices.push(node_index);
        } else {
            if let Some(left_child) = self.node(node_index).left_child {
                self.leaf_node_indices(left_child, leaf_indices);
            }
            if let Some(right_child) = self.node(node_index).right_child {
                self.leaf_node_indices(right_child, leaf_indices);
            }
        }
    }

    fn max_leaf_node_by(
        &self,
        leaf_nodes: &[usize],
        tn: &Tensor,
        cost: fn(u64) -> u64,
    ) -> Option<usize> {
        let (node, _) = leaf_nodes
            .iter()
            .map(|&node| {
                (
                    node,
                    tn.tensor(self.node(node).tensor_index.unwrap()).size(),
                )
            })
            .reduce(|node1, node2| cmp::max_by_key(node1, node2, |a| cost(a.1)))?;
        Some(node)
    }

    // fn biggest_leaf_node(&self, leaf_nodes: &[usize], tn: &Tensor) -> Option<usize> {
    //     let mut cur_tensor_mem: u64 = 0;
    //     let mut max_tensor_mem: u64 = 0;
    //     let mut max_leaf_idx: Option<usize> = None;
    //     for &leaf_idx in leaf_nodes {
    //         cur_tensor_mem = tn.tensors()[self.nodes[leaf_idx].tensor_index.unwrap()].size();
    //         if cur_tensor_mem >= max_tensor_mem {
    //             max_tensor_mem = cur_tensor_mem;
    //             max_leaf_idx = Some(leaf_idx);
    //         }
    //     }
    //     max_leaf_idx
    // }

    // fn smallest_leaf_node(&self, leaf_nodes: &[usize], tn: &Tensor) -> Option<usize> {
    //     let mut cur_tensor_mem: u64 = 0;
    //     let mut min_tensor_mem: u64 = std::u64::MAX;
    //     let mut min_leaf_idx: Option<usize> = None;
    //     for &leaf_idx in leaf_nodes {
    //         cur_tensor_mem = tn.tensors()[self.nodes[leaf_idx].tensor_index.unwrap()].size();
    //         if cur_tensor_mem < min_tensor_mem {
    //             min_tensor_mem = cur_tensor_mem;
    //             min_leaf_idx = Some(leaf_idx);
    //         }
    //     }
    //     min_leaf_idx
    // }

    fn tree_depth(&self, node_index: Option<usize>) -> usize {
        match node_index {
            Some(node_index) => {
                1 + max(
                    self.tree_depth(self.node(node_index).left_child),
                    self.tree_depth(self.node(node_index).right_child),
                )
            }
            None => 0,
        }
    }

    fn children_at_depth(&self, root: Option<usize>, depth: usize, children: &mut Vec<usize>) {
        if let Some(tensor_id) = root {
            if depth == 0 {
                children.push(tensor_id);
            } else {
                self.children_at_depth(self.node(tensor_id).left_child(), depth - 1, children);
                self.children_at_depth(self.node(tensor_id).right_child(), depth - 1, children);
            }
        }
    }

    /// Not used
    fn get_children_depth(&self, node_index: usize) -> (usize, usize) {
        (
            self.tree_depth(self.nodes[node_index].left_child),
            self.tree_depth(self.nodes[node_index].right_child),
        )
    }

    fn remove_leaf_node(&mut self, leaf_id: usize) -> usize {
        assert!(self.node(leaf_id).is_leaf());
        let leaf_parent = self.node(leaf_id).parent().unwrap();
        let leaf_parent_parent = self.node(leaf_parent).parent().unwrap();

        let repl_node = self.node(leaf_parent).get_other_child(leaf_id);
        self.mut_node(leaf_parent_parent)
            .replace_child(leaf_parent, Some(repl_node));
        self.mut_node(repl_node).set_parent(leaf_parent_parent);
        self.mut_node(leaf_id).remove_parent();
        self.mut_node(leaf_parent).deprecate();

        leaf_parent
    }

    fn replace_node(
        &mut self,
        node_index: usize,
        left_child_index: usize,
        right_child_index: usize,
        parent_index: usize,
    ) {
        *self.mut_node(node_index) = Node {
            id: "INSERTED".to_string(),
            left_child: Some(left_child_index),
            right_child: Some(right_child_index),
            parent: Some(parent_index),
            tensor_index: None,
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
        let t1 = tn.tensor(self.nodes[node_index].tensor_index.unwrap());

        let (node, _) = node_tensor_map
            .iter()
            .map(|(id, tensor)| (id, cost_function(tensor, t1)))
            .reduce(|node1, node2| cmp::max_by_key(node1, node2, |&a| a.1))?;
        Some(*node)
    }

    fn add_contraction(&self, node: &Node, path: &mut Vec<ContractionIndex>) {
        match (node.left_child(), node.right_child()) {
            (Some(left_index), Some(right_index)) => {
                self.add_contraction(self.node(left_index), path);
                self.add_contraction(self.node(right_index), path)
            }
            _ => {}
        };
    }

    pub fn to_contraction_path(&self, node_index: usize) -> Vec<ContractionIndex> {
        let mut contraction_path = vec![];
        self.add_contraction(self.node(node_index), &mut contraction_path);
        contraction_path
    }
}

fn populate_subtree_tensor_map(
    contraction_tree: &ContractionTree,
    node_index: usize,
    node_tensor_map: &mut HashMap<usize, Tensor>,
    tn: &Tensor,
) -> Tensor {
    if contraction_tree.node(node_index).is_leaf() {
        let t = tn.tensor(contraction_tree.node(node_index).tensor_index.unwrap());
        node_tensor_map.insert(node_index, t.clone());
        t.clone()
    } else {
        let t1 = populate_subtree_tensor_map(
            contraction_tree,
            contraction_tree.node(node_index).left_child.unwrap(),
            node_tensor_map,
            tn,
        );
        let t2 = populate_subtree_tensor_map(
            contraction_tree,
            contraction_tree.node(node_index).right_child.unwrap(),
            node_tensor_map,
            tn,
        );
        let t12 = &t1 ^ &t2;
        node_tensor_map.insert(node_index, t12.clone());
        t12
    }
}

fn tree_contraction_cost(
    contraction_tree: &ContractionTree,
    node_index: usize,
    tn: &Tensor,
) -> (u64, u64) {
    let contraction_path = ssa_replace_ordering(
        &contraction_tree.to_contraction_path(node_index),
        tn.tensors().len(),
    );
    contract_path_cost(tn.tensors(), &contraction_path)
}

fn find_potential_nodes(
    contraction_tree: &ContractionTree,
    bigger_subtree_leaf_nodes: &Vec<usize>,
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
        &tn,
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

#[cfg(test)]
mod tests {
    use std::iter::zip;

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

    #[test]
    fn test_simple() {
        let (tensor, path) = setup_simple();
        let ContractionTree { nodes, root } =
            ContractionTree::from_contraction_path(&tensor, &path);
        println!("Nodes: {:?}", nodes);
        println!("Root: {:?}", root);
        let ref_nodes = vec![
            Node::new(String::from("0"), None, None, Some(3), Some(0)),
            Node::new(String::from("1"), None, None, Some(3), Some(1)),
            Node::new(String::from("2"), None, None, Some(4), Some(2)),
            Node::new(String::from("(0,1)"), Some(0), Some(1), Some(4), None),
            Node::new(String::from("(2,(0,1))"), Some(2), Some(3), None, None),
        ];
        let ref_root = Some(4);

        for (node, ref_node) in zip(nodes.iter(), ref_nodes.iter()) {
            let Node {
                id,
                left_child,
                right_child,
                parent,
                tensor_index: tensor_idx,
            } = node;

            let Node {
                id: ref_id,
                left_child: ref_left_child,
                right_child: ref_right_child,
                parent: ref_parent,
                tensor_index: ref_tensor_idx,
            } = ref_node;

            assert_eq!(id, ref_id);
            assert_eq!(left_child, ref_left_child);
            assert_eq!(right_child, ref_right_child);
            assert_eq!(parent, ref_parent);
            assert_eq!(tensor_idx, ref_tensor_idx);
        }
        assert_eq!(root, ref_root);
    }

    #[test]
    fn test_complex() {
        let (tensor, path) = setup_complex();
        let ContractionTree { nodes, root } =
            ContractionTree::from_contraction_path(&tensor, &path);

        let ref_nodes = vec![
            Node::new(String::from("0"), None, None, Some(8), Some(0)),
            Node::new(String::from("1"), None, None, Some(6), Some(1)),
            Node::new(String::from("2"), None, None, Some(9), Some(2)),
            Node::new(String::from("3"), None, None, Some(7), Some(3)),
            Node::new(String::from("4"), None, None, Some(7), Some(4)),
            Node::new(String::from("5"), None, None, Some(6), Some(5)),
            Node::new(String::from("(1,5)"), Some(1), Some(5), Some(8), None),
            Node::new(String::from("(3,4)"), Some(3), Some(4), Some(9), None),
            Node::new(String::from("(0,(1,5))"), Some(0), Some(6), Some(10), None),
            Node::new(String::from("(2,(3,4))"), Some(2), Some(7), Some(10), None),
            Node::new(
                String::from("((0,(1,5)),(2,(3,4)))"),
                Some(8),
                Some(9),
                None,
                None,
            ),
        ];
        let ref_root = Some(10);

        for (node, ref_node) in zip(nodes.iter(), ref_nodes.iter()) {
            let Node {
                id,
                left_child,
                right_child,
                parent,
                tensor_index: tensor_idx,
            } = node;

            let Node {
                id: ref_id,
                left_child: ref_left_child,
                right_child: ref_right_child,
                parent: ref_parent,
                tensor_index: ref_tensor_idx,
            } = ref_node;
            assert_eq!(id, ref_id);
            assert_eq!(left_child, ref_left_child);
            assert_eq!(right_child, ref_right_child);
            assert_eq!(parent, ref_parent);
            assert_eq!(tensor_idx, ref_tensor_idx);
        }
        assert_eq!(root, ref_root);
    }
}
