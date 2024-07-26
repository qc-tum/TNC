use crate::contractionpath::paths::{greedy::Greedy, CostType, OptimizePath};
use crate::mpi::communication::CommunicationScheme;
use crate::pair;
use crate::tensornetwork::tensor::Tensor;
use crate::types::ContractionIndex;
use balancing::{balance_partitions, tensor_bipartition};
use export::to_dendogram;
use itertools::Itertools;
use std::cell::{Ref, RefCell};
use std::collections::HashMap;
use std::rc::{Rc, Weak};
use utils::{calculate_partition_costs, parallel_tree_contraction_cost};

use super::contraction_cost::contract_path_cost;
use super::paths::validate_path;

mod balancing;
pub mod export;
mod utils;

type NodeRef = Rc<RefCell<Node>>;
type WeakNodeRef = Weak<RefCell<Node>>;

/// Node in [`ContractionTree`], represents a contraction of [`Tensor`] with position
/// `left_child` and position `right_child` to obtain [`Tensor`] at position
/// `parent`.
#[derive(Debug, Clone)]
pub struct Node {
    id: usize,
    left_child: WeakNodeRef,
    right_child: WeakNodeRef,
    parent: WeakNodeRef,
    tensor_index: Option<Vec<usize>>,
}

impl Node {
    fn new(
        id: usize,
        left_child: WeakNodeRef,
        right_child: WeakNodeRef,
        parent: WeakNodeRef,
        tensor_index: Option<Vec<usize>>,
    ) -> Self {
        Self {
            id,
            left_child,
            right_child,
            parent,
            tensor_index,
        }
    }

    pub fn left_child_id(&self) -> Option<usize> {
        self.left_child.upgrade().map(|node| node.borrow().id)
    }

    pub fn right_child_id(&self) -> Option<usize> {
        self.right_child.upgrade().map(|node| node.borrow().id)
    }

    pub const fn id(&self) -> usize {
        self.id
    }

    pub fn parent_id(&self) -> Option<usize> {
        self.parent.upgrade().map(|node| node.borrow().id)
    }

    fn is_leaf(&self) -> bool {
        self.left_child.upgrade().is_none() && self.right_child.upgrade().is_none()
    }

    fn add_child(&mut self, child: WeakNodeRef) {
        assert!(child.upgrade().is_some(), "Child is already deallocated");
        if self.left_child.upgrade().is_none() {
            self.left_child = child;
        } else if self.right_child.upgrade().is_none() {
            self.right_child = child;
        } else {
            panic!("Parent already has two children");
        }
    }
}

/// Struct representing the full contraction path of a given Tensor object
#[derive(Default, Debug, Clone)]
pub struct ContractionTree {
    nodes: HashMap<usize, NodeRef>,
    partitions: HashMap<usize, Vec<usize>>,
    root: WeakNodeRef,
}

impl ContractionTree {
    pub fn node(&self, tensor_id: usize) -> Ref<Node> {
        let borrow = self.nodes.get(&tensor_id).unwrap();
        borrow.as_ref().borrow()
    }

    pub fn root_id(&self) -> Option<usize> {
        self.root.upgrade().map(|node| node.borrow().id)
    }

    pub const fn partitions(&self) -> &HashMap<usize, Vec<usize>> {
        &self.partitions
    }

    /// Populates `nodes` and `partitions` with the tree structure of the contraction
    /// `path`.
    fn from_contraction_path_recurse(
        tensor: &Tensor,
        path: &[ContractionIndex],
        nodes: &mut HashMap<usize, NodeRef>,
        partitions: &mut HashMap<usize, Vec<usize>>,
        prefix: &[usize],
    ) {
        let mut scratch = HashMap::new();

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
                i.borrow_mut().parent = Rc::downgrade(&parent);
                j.borrow_mut().parent = Rc::downgrade(&parent);
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
        let mut nodes = HashMap::new();
        let mut partitions = HashMap::new();
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
            leaf_indices.push(node.id);
        } else {
            Self::leaf_ids_recurse(
                &node.left_child.upgrade().unwrap().as_ref().borrow(),
                leaf_indices,
            );
            Self::leaf_ids_recurse(
                &node.right_child.upgrade().unwrap().as_ref().borrow(),
                leaf_indices,
            );
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
    pub fn remove_subtree(&mut self, node_id: usize) {
        self.remove_subtree_recurse(node_id);
    }

    /// Converts a contraction path into a ContractionTree, then attaches this as a subtree at "parent_id"
    pub fn add_subtree(
        &mut self,
        path: &[ContractionIndex],
        parent_id: usize,
        tensor_indices: &[usize],
    ) -> usize {
        validate_path(path);
        assert!(self.nodes.contains_key(&parent_id));
        let mut index = 0;
        // Utilize a scratch hashmap to store intermediate tensor information
        let mut scratch = HashMap::new();

        // Fill scratch with initial tensor inputs
        for &tensor_index in tensor_indices {
            scratch.insert(tensor_index, Rc::clone(&self.nodes[&tensor_index]));
        }

        // Generate intermediate tensors by looping over contraction operations, fill and update scratch as needed.
        for contr in path {
            if let ContractionIndex::Pair(i_path, j_path) = contr {
                // Always keep track of latest added tensor. Last index will be the root of the subtree.
                index = self.next_id(index);
                let i = &scratch[i_path];
                let j = &scratch[j_path];
                let parent =
                    Node::new(index, Rc::downgrade(i), Rc::downgrade(j), Weak::new(), None);

                let parent = Rc::new(RefCell::new(parent));
                i.borrow_mut().parent = Rc::downgrade(&parent);
                j.borrow_mut().parent = Rc::downgrade(&parent);
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
        new_child.borrow_mut().parent = Rc::downgrade(new_parent);

        index
    }

    fn tree_weights_recurse(
        node: &Node,
        tn: &Tensor,
        weights: &mut HashMap<usize, f64>,
        scratch: &mut HashMap<usize, Tensor>,
        cost_function: fn(&Tensor, &Tensor) -> f64,
    ) {
        if node.is_leaf() {
            let Some(tensor_index) = &node.tensor_index else {
                panic!("All leaf nodes should have a tensor index")
            };
            weights.insert(node.id, 0f64);
            scratch.insert(node.id, tn.nested_tensor(tensor_index).clone());
            return;
        }

        let left_child = &node.left_child.upgrade().unwrap();
        let right_child = &node.right_child.upgrade().unwrap();
        let left_ref = left_child.as_ref().borrow();
        let right_ref = right_child.as_ref().borrow();

        // Recurse first because weights of leaves are needed for further computation.
        Self::tree_weights_recurse(&left_ref, tn, weights, scratch, cost_function);
        Self::tree_weights_recurse(&right_ref, tn, weights, scratch, cost_function);

        let t1 = &scratch[&left_ref.id];
        let t2 = &scratch[&right_ref.id];

        let cost = weights[&left_ref.id] + weights[&right_ref.id] + cost_function(t1, t2);

        weights.insert(node.id, cost);
        scratch.insert(node.id, t1 ^ t2);
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
    ) -> HashMap<usize, f64> {
        let mut weights = HashMap::new();
        let mut scratch = HashMap::new();
        let node = self.node(node_id);
        Self::tree_weights_recurse(&node, tn, &mut weights, &mut scratch, cost_function);
        weights
    }

    /// Given a specific tensor at leaf node "n1" with id `node_index`, identifies tensor at node "n2" in `ContractionTree` subtree rooted at `subtree_root`, such that (n1, n2) maximizes provided cost function `cost_function`.
    ///
    /// # Arguments
    /// * `node_id` - leaf node used to calculation cost function, must be disjoint from subtree rooted at `subtree_root`
    /// * `subtree_root` - identifies root of subtree to be considered
    /// * `tn` - [`Tensor`] object containing bond dimension and leaf node information
    /// * `cost_function` - cost function of contracting the tensors
    ///
    /// # Returns
    /// * option of node id (not necessarily a leaf node) in subtree that maximizes `cost_function`.
    pub fn max_match_by(
        &self,
        node_id: usize,
        subtree_root: usize,
        tn: &Tensor,
        cost_function: fn(&Tensor, &Tensor) -> f64,
    ) -> Option<(usize, f64)> {
        assert!(self.node(node_id).is_leaf());

        // Get a map that maps leaf nodes to corresponding tensor objects.
        let mut node_tensor_map = HashMap::new();
        populate_subtree_tensor_map(self, subtree_root, &mut node_tensor_map, tn);
        let node = self.node(node_id);
        let tensor_index = node.tensor_index.as_ref().unwrap();
        let t1 = tn.nested_tensor(tensor_index);

        // Find the tensor that maximizes cost function.
        let (node, cost) = node_tensor_map
            .iter()
            .map(|(id, tensor)| (id, cost_function(tensor, t1)))
            .max_by(|a, b| a.1.total_cmp(&b.1))?;
        Some((*node, cost))
    }

    /// Populates given vector with contractions path of contraction tree starting at `node`.
    ///
    /// # Arguments
    /// * `node` - pointer to [`Node`] object
    /// * `path` - vec to store contraction path in
    fn to_contraction_path_recurse(
        node: &Node,
        path: &mut Vec<ContractionIndex>,
        replace: bool,
        hierarchy: bool,
    ) -> usize {
        if node.is_leaf() {
            if hierarchy {
                let tn_index = node.tensor_index.as_ref().unwrap();
                return *tn_index.last().unwrap();
            } else {
                return node.id;
            }
        }

        // Get children
        let (Some(left_child), Some(right_child)) =
            (node.left_child.upgrade(), node.right_child.upgrade())
        else {
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
            node.id
        }
    }

    /// Populates given vector with contractions path of contraction tree starting at `node_id`.
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

    fn tensor_recursive(node: &Node, tn: &Tensor) -> Tensor {
        if node.is_leaf() {
            let tensor_index = node.tensor_index.as_ref().unwrap();
            tn.nested_tensor(tensor_index).clone()
        } else {
            let left =
                Self::tensor_recursive(&node.left_child.upgrade().unwrap().as_ref().borrow(), tn);
            let right =
                Self::tensor_recursive(&node.right_child.upgrade().unwrap().as_ref().borrow(), tn);
            &left ^ &right
        }
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
        let node = self.node(node_id);
        Self::tensor_recursive(&node, tensor)
    }
}

/// Populates `node_tensor_map` with all intermediate and leaf node ids and corresponding [`Tensor`] object, with root at `node_id`.
///
/// # Arguments
/// * `contraction_tree` - [`ContractionTree`] object
/// * `node_id` - root of subtree to examine
/// * `node_tensor_map` - empty HashMap to populate
/// * `tn` - [`Tensor`] object containing bond dimension and leaf node information
///
/// # Returns
/// Empty [`Tensor`] object with legs (dimensions) of data after fully contracted.
fn populate_subtree_tensor_map(
    contraction_tree: &ContractionTree,
    node_id: usize,
    node_tensor_map: &mut HashMap<usize, Tensor>,
    tn: &Tensor,
) -> Tensor {
    let node = contraction_tree.node(node_id);

    if node.is_leaf() {
        let tensor_index = node.tensor_index.as_ref().unwrap();
        let t = tn.nested_tensor(tensor_index);
        node_tensor_map.insert(node.id, t.clone());
        t.clone()
    } else {
        let t1 = populate_subtree_tensor_map(
            contraction_tree,
            node.left_child_id().unwrap(),
            node_tensor_map,
            tn,
        );
        let t2 = populate_subtree_tensor_map(
            contraction_tree,
            node.right_child_id().unwrap(),
            node_tensor_map,
            tn,
        );
        let t12 = &t1 ^ &t2;
        node_tensor_map.insert(node.id, t12.clone());
        t12
    }
}

pub struct BalanceSettings {
    pub random_balance: bool,
    pub rebalance_depth: usize,
    pub iterations: usize,
    pub output_file: String,
    pub dendogram_cost_function: fn(&Tensor, &Tensor) -> f64,
    pub greedy_cost_function: fn(&Tensor, &Tensor) -> f64,
    pub communication_scheme: CommunicationScheme,
}

pub fn balance_partitions_iter(
    tensor: &Tensor,
    path: &[ContractionIndex],
    BalanceSettings {
        random_balance,
        rebalance_depth,
        iterations,
        output_file,
        dendogram_cost_function,
        greedy_cost_function,
        communication_scheme,
    }: BalanceSettings,
) -> (usize, Tensor, Vec<ContractionIndex>, Vec<f64>) {
    let bond_dims = tensor.bond_dims();
    let mut contraction_tree = ContractionTree::from_contraction_path(tensor, path);
    let mut path = path.to_owned();
    let final_contraction = path
        .iter()
        .filter(|&e| matches!(e, ContractionIndex::Pair(..)))
        .cloned()
        .collect_vec();
    let mut partition_costs =
        calculate_partition_costs(&contraction_tree, rebalance_depth, tensor, true);

    assert!(partition_costs.len() > 1);
    let partition_number = partition_costs.len();

    let (_, mut max_cost) = partition_costs.last().unwrap();

    let children = &contraction_tree.partitions()[&rebalance_depth];

    let mut children_tensors = children
        .iter()
        .map(|e| contraction_tree.tensor(*e, tensor))
        .collect_vec();

    let (final_op_cost, _) = contract_path_cost(&children_tensors, &final_contraction);
    let mut max_costs = Vec::with_capacity(iterations + 1);
    max_costs.push(max_cost + final_op_cost);

    to_dendogram(
        &contraction_tree,
        tensor,
        dendogram_cost_function,
        output_file.clone() + "_0",
    );

    let mut new_tn;
    let mut best_contraction = 0;
    let mut best_contraction_path = path.clone();
    let mut best_cost = max_cost + final_op_cost;

    let mut best_tn = tensor.clone();

    for i in 1..=iterations {
        (max_cost, path, new_tn) = balance_partitions(
            tensor,
            &mut contraction_tree,
            random_balance,
            rebalance_depth,
            &partition_costs,
            greedy_cost_function,
        );
        assert_eq!(partition_number, path.len(), "Tensors lost!");
        validate_path(&path);

        partition_costs =
            calculate_partition_costs(&contraction_tree, rebalance_depth, tensor, true);
        children_tensors = new_tn
            .tensors()
            .iter()
            .map(|tensor| {
                let ext_edges = tensor.external_edges();
                let mut new_tensor = Tensor::new(ext_edges);
                new_tensor.insert_bond_dims(&tensor.bond_dims());
                new_tensor
            })
            .collect_vec();

        let (final_op_cost, final_contraction) = match communication_scheme {
            CommunicationScheme::Greedy => {
                let mut communication_tensors = Tensor::default();
                communication_tensors.push_tensors(
                    children_tensors.clone(),
                    Some(&bond_dims),
                    None,
                );

                let mut opt = Greedy::new(&communication_tensors, CostType::Flops);
                opt.optimize_path();
                let final_contraction = opt.get_best_replace_path();
                let contraction_tree = ContractionTree::from_contraction_path(
                    &communication_tensors,
                    &final_contraction,
                );
                // let (final_op_cost, _) = contract_path_cost(&children_tensors, &final_contraction);
                let (final_op_cost, _, _) = parallel_tree_contraction_cost(
                    &contraction_tree,
                    contraction_tree.root_id().unwrap(),
                    &communication_tensors,
                );
                (final_op_cost, final_contraction)
            }
            CommunicationScheme::Bipartition => {
                let children_tensors = children_tensors.iter().cloned().enumerate().collect_vec();
                let (_, final_op_cost, _, final_contraction) =
                    tensor_bipartition(&children_tensors, &bond_dims);

                (final_op_cost, final_contraction)
            }
        };

        path.extend(final_contraction.clone());
        let new_max_cost = max_cost + final_op_cost;

        max_costs.push(new_max_cost);

        if new_max_cost < best_cost {
            best_cost = new_max_cost;
            best_contraction = i;
            best_tn = new_tn.clone();
            best_contraction_path = path.clone();
        }

        to_dendogram(
            &contraction_tree,
            tensor,
            dendogram_cost_function,
            output_file.clone() + &format!("_{}", i),
        );
    }

    (best_contraction, best_tn, best_contraction_path, max_costs)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use utils::tree_contraction_cost;

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

    impl PartialEq for Node {
        fn eq(&self, other: &Self) -> bool {
            self.id == other.id
                && self.left_child_id() == other.left_child_id()
                && self.right_child_id() == other.right_child_id()
                && self.parent_id() == other.parent_id()
                && self.tensor_index == other.tensor_index
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
            Rc::downgrade(&node3),
            Rc::downgrade(&node2),
            Weak::new(),
            None,
        )));
        node0.borrow_mut().parent = Rc::downgrade(&node3);
        node1.borrow_mut().parent = Rc::downgrade(&node3);
        node2.borrow_mut().parent = Rc::downgrade(&node4);
        node3.borrow_mut().parent = Rc::downgrade(&node4);

        let ref_root = Rc::clone(&node4);
        let ref_nodes = [node0, node1, node2, node3, node4];

        for (key, ref_node) in ref_nodes.iter().enumerate().rev() {
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
        node0.borrow_mut().parent = Rc::downgrade(&node7);
        node1.borrow_mut().parent = Rc::downgrade(&node6);
        node2.borrow_mut().parent = Rc::downgrade(&node9);
        node3.borrow_mut().parent = Rc::downgrade(&node8);
        node4.borrow_mut().parent = Rc::downgrade(&node8);
        node5.borrow_mut().parent = Rc::downgrade(&node6);
        node6.borrow_mut().parent = Rc::downgrade(&node7);
        node7.borrow_mut().parent = Rc::downgrade(&node10);
        node8.borrow_mut().parent = Rc::downgrade(&node9);
        node9.borrow_mut().parent = Rc::downgrade(&node10);

        let ref_root = Rc::clone(&node10);
        let ref_nodes = [
            node0, node1, node2, node3, node4, node5, node6, node7, node8, node9, node10,
        ];

        for (key, ref_node) in ref_nodes.iter().enumerate().rev() {
            let node = &nodes[&key];
            assert_eq!(node, ref_node);
        }
        assert_eq!(root.upgrade().unwrap(), ref_root);
    }

    #[test]
    fn test_leaf_ids_simple() {
        let (tn, path) = setup_simple();
        let tree = ContractionTree::from_contraction_path(&tn, &path);

        assert_eq!(tree.leaf_ids(4), vec![0, 1, 2]);
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
    fn test_tree_weights_simple() {
        let (tensor, path) = setup_simple();
        let tree = ContractionTree::from_contraction_path(&tensor, &path);
        let ref_weights =
            HashMap::from([(1, 0f64), (0, 0f64), (2, 0f64), (3, 3820f64), (4, 4540f64)]);
        let weights = tree.tree_weights(4, &tensor, contract_cost_tensors);

        assert_eq!(weights, ref_weights);
        let ref_weights = HashMap::from([(1, 0f64), (0, 0f64), (3, 3820f64)]);
        let weights = tree.tree_weights(3, &tensor, contract_cost_tensors);
        assert_eq!(weights, ref_weights);

        assert_eq!(weights, ref_weights);
        let ref_weights = HashMap::from([(2, 0f64)]);
        let weights = tree.tree_weights(2, &tensor, contract_cost_tensors);
        assert_eq!(weights, ref_weights);
    }

    #[test]
    fn test_tree_weights_complex() {
        let (tensor, path) = setup_complex();
        let tree = ContractionTree::from_contraction_path(&tensor, &path);
        let ref_weights = HashMap::from([
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
    fn test_max_match_by_complex() {
        let (tensor, path) = setup_complex();
        let tree = ContractionTree::from_contraction_path(&tensor, &path);

        fn greedy_cost_fn(t1: &Tensor, t2: &Tensor) -> f64 {
            (t1.size() as f64) + (t2.size() as f64) - ((t1 ^ t2).size() as f64)
        }
        let (max_match, _) = tree.max_match_by(2, 7, &tensor, greedy_cost_fn).unwrap();

        assert_eq!(max_match, 7);

        fn max_memory_cost_fn(t1: &Tensor, t2: &Tensor) -> f64 {
            (t1 ^ t2).size() as f64
        }

        let (max_match, _) = tree
            .max_match_by(2, 7, &tensor, max_memory_cost_fn)
            .unwrap();
        assert_eq!(max_match, 1);
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
        let mut node_tensor_map = HashMap::new();
        populate_subtree_tensor_map(&tree, 4, &mut node_tensor_map, &tensor);

        let ref_node_tensor_map = HashMap::from([
            (0, Tensor::new(vec![4, 3, 2])),
            (1, Tensor::new(vec![0, 1, 3, 2])),
            (2, Tensor::new(vec![4, 5, 6])),
            (3, Tensor::new(vec![4, 0, 1])),
            (4, Tensor::new(vec![0, 1, 5, 6])),
        ]);

        for (key, value) in ref_node_tensor_map {
            assert_eq!(node_tensor_map[&key].legs(), value.legs());
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

        for (key, value) in ref_node_tensor_map {
            assert_eq!(node_tensor_map[&key].legs(), value.legs());
        }
    }

    #[test]
    fn test_tree_contraction_path() {
        let (tensor, ref_path) = setup_simple();
        let tree = ContractionTree::from_contraction_path(&tensor, &ref_path);
        let (op_cost, mem_cost) = tree_contraction_cost(&tree, tree.root_id().unwrap(), &tensor);

        assert_eq!(op_cost, 4540f64);
        assert_eq!(mem_cost, 538f64);
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
    fn test_tree_contraction_path_complex() {
        let (tensor, ref_path) = setup_complex();
        let tree = ContractionTree::from_contraction_path(&tensor, &ref_path);
        let (op_cost, mem_cost) = tree_contraction_cost(&tree, tree.root_id().unwrap(), &tensor);

        assert_eq!(op_cost, 4237070f64);
        assert_eq!(mem_cost, 89478f64);
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
}
