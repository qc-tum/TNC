use std::rc::Rc;

use balancing_schemes::BalancingScheme;
use communication_schemes::{
    bipartition_communication_scheme, greedy_communication_scheme,
    weighted_branchbound_communication_scheme, CommunicationScheme,
};
use itertools::Itertools;
use log::info;
use rand::{seq::SliceRandom, thread_rng};
use rustc_hash::FxHashMap;

use crate::{
    contractionpath::{
        contraction_cost::contract_path_cost,
        contraction_tree::{
            export::{to_dendogram_format, to_pdf},
            utils::{characterize_partition, subtree_contraction_path},
        },
        paths::validate_path,
    },
    mpi::communication::extract_communication_path,
    tensornetwork::tensor::Tensor,
    types::ContractionIndex,
};

use super::{export::DendogramSettings, ContractionTree};

pub mod balancing_schemes;
pub mod communication_schemes;

#[derive(Debug, Clone, Copy)]
pub struct BalanceSettings {
    // if not None, randomly chooses from top `usize` options
    // random choice is weighted by objective outcome
    pub random_balance: Option<usize>,
    pub rebalance_depth: usize,
    pub iterations: usize,
    pub objective_function: fn(&Tensor, &Tensor) -> f64,
    pub communication_scheme: CommunicationScheme,
    pub balancing_scheme: BalancingScheme,
}

#[derive(Debug, Clone)]
pub struct PartitionData {
    pub id: usize,
    pub cost: f64,
    pub contraction: Vec<ContractionIndex>,
    pub local_tensor: Tensor,
}

pub fn balance_partitions_iter(
    tensor_network: &Tensor,
    path: &[ContractionIndex],
    balance_settings: BalanceSettings,
    dendogram_settings: Option<&DendogramSettings>,
) -> (usize, Tensor, Vec<ContractionIndex>, Vec<f64>) {
    let mut contraction_tree = ContractionTree::from_contraction_path(tensor_network, path);

    let communication_path = extract_communication_path(path);
    let BalanceSettings {
        rebalance_depth,
        iterations,
        balancing_scheme,
        communication_scheme,
        ..
    } = balance_settings;
    let mut partition_data =
        characterize_partition(&contraction_tree, rebalance_depth, tensor_network, true);

    assert!(partition_data.len() > 1);

    let partition_number = partition_data.len();

    let intermediate_cost = partition_data.last().unwrap().cost;

    let children_tensors = partition_data
        .iter()
        .map(|PartitionData { local_tensor, .. }| local_tensor.clone())
        .collect_vec();

    let (communication_cost, _) = contract_path_cost(&children_tensors, &communication_path, true);
    let mut max_costs = Vec::with_capacity(iterations + 1);
    max_costs.push(intermediate_cost + communication_cost);

    print_dendogram(dendogram_settings, &contraction_tree, tensor_network, 0);

    let mut best_iteration = 0;
    let mut best_contraction_path = path.to_owned();
    let mut best_cost = intermediate_cost + communication_cost;

    let mut best_tn = tensor_network.clone();

    for i in 1..=iterations {
        info!("Balancing iteration {i} with balancing scheme {balancing_scheme:?}, communication scheme {communication_scheme:?}");

        // Balances and updates partitions
        let (largest_local_cost, mut intermediate_path, new_tensor_network) = balance_partitions(
            &mut partition_data,
            &mut contraction_tree,
            tensor_network,
            balance_settings,
        );

        assert_eq!(intermediate_path.len(), partition_number, "Tensors lost!");
        validate_path(&intermediate_path);

        // Ensures that children tensors are mapped to their respective partition costs
        let (fan_in_cost, communication_path) = communicate_partitions(
            &partition_data,
            &contraction_tree,
            &new_tensor_network,
            balance_settings,
        );

        intermediate_path.extend(communication_path);
        let new_cost = largest_local_cost + fan_in_cost;

        max_costs.push(new_cost);

        if new_cost < best_cost {
            best_cost = new_cost;
            best_iteration = i;
            best_tn = new_tensor_network;
            best_contraction_path = intermediate_path;
        }
        print_dendogram(dendogram_settings, &contraction_tree, tensor_network, i);
    }

    (best_iteration, best_tn, best_contraction_path, max_costs)
}

fn print_dendogram(
    dendogram_settings: Option<&DendogramSettings>,
    contraction_tree: &ContractionTree,
    tensor_network: &Tensor,
    iteration: usize,
) {
    if let Some(settings) = dendogram_settings {
        let dendogram_entries = to_dendogram_format(
            contraction_tree,
            tensor_network,
            settings.objective_function,
        );
        to_pdf(
            &format!("{}_{}", settings.output_file, iteration),
            &dendogram_entries,
            None,
        );
    }
}

pub(super) fn communicate_partitions(
    partition_data: &[PartitionData],
    _contraction_tree: &ContractionTree,
    tensor_network: &Tensor,
    balance_settings: BalanceSettings,
) -> (f64, Vec<ContractionIndex>) {
    let communication_scheme = balance_settings.communication_scheme;
    let bond_dims = tensor_network.bond_dims();
    let children_tensors = tensor_network
        .tensors()
        .iter()
        .map(|t| {
            let mut tc = Tensor::new(t.external_edges());
            tc.bond_dims = t.bond_dims.clone();
            tc
        })
        .collect_vec();

    let (communication_cost, communication_path) = match communication_scheme {
        CommunicationScheme::Greedy => greedy_communication_scheme(&children_tensors, &bond_dims),
        CommunicationScheme::Bipartition => {
            bipartition_communication_scheme(&children_tensors, &bond_dims)
        }
        CommunicationScheme::WeightedBranchBound => {
            let latency_map = partition_data
                .iter()
                .enumerate()
                .map(|(i, partition)| (i, partition.cost))
                .collect::<FxHashMap<usize, f64>>();
            weighted_branchbound_communication_scheme(&children_tensors, &bond_dims, latency_map)
        }
    };
    (communication_cost, communication_path)
}

pub(super) fn balance_partitions(
    partition_data: &mut [PartitionData],
    contraction_tree: &mut ContractionTree,
    tensor_network: &Tensor,
    balance_settings: BalanceSettings,
) -> (f64, Vec<ContractionIndex>, Tensor) {
    let BalanceSettings {
        random_balance,
        rebalance_depth,
        objective_function,
        balancing_scheme,
        ..
    } = balance_settings;
    // If there are less than 3 tensors in the tn, rebalancing will not make sense.
    if tensor_network.total_num_tensors() < 3 {
        // TODO: should not panic, but handle gracefully
        panic!("No rebalancing undertaken, as tn is too small (< 3 tensors)");
    }
    // Will cause strange errors (picking of same partition multiple times if this is not true.Better to panic here.)
    assert!(partition_data.len() > 1);
    // Use memory reduction to identify which leaf node to shift between partitions.
    let bond_dims = tensor_network.bond_dims();
    let shifted_nodes = match balancing_scheme {
        BalancingScheme::BestWorst => balancing_schemes::best_worst_balancing(
            partition_data,
            contraction_tree,
            random_balance,
            objective_function,
            tensor_network,
        ),
        BalancingScheme::Tensor => balancing_schemes::best_tensor_balancing(
            partition_data,
            contraction_tree,
            random_balance,
            objective_function,
            tensor_network,
        ),
        BalancingScheme::Tensors => balancing_schemes::best_tensors_balancing(
            partition_data,
            contraction_tree,
            random_balance,
            objective_function,
            tensor_network,
        ),
        BalancingScheme::IntermediateTensors(height_limit) => {
            balancing_schemes::best_intermediate_tensors_balancing(
                partition_data,
                contraction_tree,
                random_balance,
                objective_function,
                tensor_network,
                height_limit,
            )
        }
        _ => panic!("Balancing Scheme not implemented"),
    };
    let mut shifted_indices = FxHashMap::default();
    for (from_subtree_id, to_subtree_id, rebalanced_nodes) in shifted_nodes {
        let shifted_from_id = *shifted_indices
            .get(&from_subtree_id)
            .unwrap_or(&from_subtree_id);

        let shifted_to_id = if shifted_indices.contains_key(&to_subtree_id) {
            *shifted_indices.get(&to_subtree_id).unwrap()
        } else {
            to_subtree_id
        };
        let (
            larger_id,
            larger_contraction,
            larger_subtree_cost,
            smaller_id,
            smaller_contraction,
            smaller_subtree_cost,
        ) = shift_node_between_subtrees(
            contraction_tree,
            rebalance_depth,
            shifted_from_id,
            shifted_to_id,
            rebalanced_nodes,
            tensor_network,
        );
        shifted_indices.insert(from_subtree_id, larger_id);
        shifted_indices.insert(to_subtree_id, smaller_id);

        let larger_tensor = contraction_tree.tensor(larger_id, tensor_network);
        let smaller_tensor = contraction_tree.tensor(smaller_id, tensor_network);

        // Update partition data based on shift
        for PartitionData {
            id,
            cost,
            contraction: subtree_contraction,
            local_tensor,
        } in partition_data.iter_mut()
        {
            if *id == shifted_from_id {
                *id = larger_id;
                *subtree_contraction = larger_contraction.clone();
                *cost = larger_subtree_cost;
                *local_tensor = larger_tensor.clone();
            } else if *id == shifted_to_id {
                *id = smaller_id;
                *subtree_contraction = smaller_contraction.clone();
                *cost = smaller_subtree_cost;
                *local_tensor = smaller_tensor.clone();
            }
        }
    }

    partition_data.sort_unstable_by(
        |PartitionData { cost: cost_a, .. }, PartitionData { cost: cost_b, .. }| {
            cost_a.total_cmp(cost_b)
        },
    );

    let mut new_max = 0.0;
    let mut rebalanced_path = Vec::new();
    let (partition_tensors, partition_ids): (Vec<_>, Vec<_>) = partition_data
        .iter()
        .enumerate()
        .map(
            |(
                i,
                PartitionData {
                    id,
                    cost,
                    contraction: subtree_contraction,
                    ..
                },
            )| {
                if *cost > new_max {
                    new_max = *cost;
                }
                rebalanced_path.push(ContractionIndex::Path(i, subtree_contraction.clone()));
                let mut child_tensor = Tensor::default();
                let leaf_ids = contraction_tree.leaf_ids(*id);
                let leaf_tensors = leaf_ids
                    .iter()
                    .map(|node_id| {
                        let nested_indices = contraction_tree
                            .node(*node_id)
                            .tensor_index()
                            .clone()
                            .unwrap();
                        tensor_network.nested_tensor(&nested_indices).clone()
                    })
                    .collect_vec();
                child_tensor.push_tensors(leaf_tensors, Some(&bond_dims), None);
                (child_tensor, *id)
            },
        )
        .collect();

    contraction_tree
        .partitions
        .insert(rebalance_depth, partition_ids);

    let mut updated_tn = Tensor::default();
    updated_tn.push_tensors(partition_tensors, Some(&bond_dims), None);
    (new_max, rebalanced_path, updated_tn)
}

/// Takes two hashmaps that contain node information. Identifies which pair of nodes from larger and smaller hashmaps maximizes the greedy cost function and returns the node from the `larger_subtree_nodes`.
/// #arguments
/// * `random_balance` - Allows for random selection of balanced node. If not None, identifies the best `usize` options and randomly selects one by weighted choice.
/// * `larger_subtree_nodes` - A set of nodes used in comparison. Only the id from the larger subtree is returned.
/// * `smaller_subtree_nodes` - A set of nodes used in comparison.
/// * `objective_function` - Cost function that takes in two tensors and returns an f64 cost.
pub(super) fn find_rebalance_node(
    random_balance: Option<usize>,
    larger_subtree_nodes: &FxHashMap<usize, Tensor>,
    smaller_subtree_nodes: &FxHashMap<usize, Tensor>,
    objective_function: fn(&Tensor, &Tensor) -> f64,
) -> (usize, f64) {
    let node_comparison = larger_subtree_nodes
        .iter()
        .cartesian_product(smaller_subtree_nodes.iter())
        .map(|((larger_node_id, larger_tensor), (_, smaller_tensor))| {
            (
                *larger_node_id,
                objective_function(larger_tensor, smaller_tensor),
            )
        });
    if let Some(options_considered) = random_balance {
        let mut rng = thread_rng();
        let node_options = node_comparison
            .sorted_unstable_by(|a, b| a.1.total_cmp(&b.1))
            .take(options_considered)
            .collect::<Vec<(usize, f64)>>();
        let max = node_options.first().unwrap().1;
        // Initial division done here as sum of weights can cause overflow before normalization.
        *node_options
            .choose_weighted(&mut rng, |node_option| node_option.1 / max)
            .unwrap()
    } else {
        node_comparison.max_by(|a, b| a.1.total_cmp(&b.1)).unwrap()
    }
}

/// Shifts `rebalance_node` from the larger subtree to the smaller subtree
/// Updates partition tensor ids after subtrees are updated and a new contraction order is found.
fn shift_node_between_subtrees(
    contraction_tree: &mut ContractionTree,
    rebalance_depth: usize,
    larger_subtree_id: usize,
    smaller_subtree_id: usize,
    rebalanced_nodes: Vec<usize>,
    tensor_network: &Tensor,
) -> (
    usize,
    Vec<ContractionIndex>,
    f64,
    usize,
    Vec<ContractionIndex>,
    f64,
) {
    // Obtain parents of the two subtrees that are being updated.
    let larger_subtree_parent_id = contraction_tree
        .node(larger_subtree_id)
        .parent_id()
        .unwrap();
    let smaller_subtree_parent_id = contraction_tree
        .node(smaller_subtree_id)
        .parent_id()
        .unwrap();

    let mut larger_subtree_leaf_nodes = contraction_tree.leaf_ids(larger_subtree_id);
    let mut smaller_subtree_leaf_nodes = contraction_tree.leaf_ids(smaller_subtree_id);
    // Always check that a node can be moved over.

    assert!(rebalanced_nodes
        .iter()
        .all(|node| !smaller_subtree_leaf_nodes.contains(node)));
    assert!(rebalanced_nodes
        .iter()
        .all(|node| larger_subtree_leaf_nodes.contains(node)));

    // Remove selected tensors from bigger subtree. Add it to the smaller subtree
    larger_subtree_leaf_nodes.retain(|leaf| !rebalanced_nodes.contains(leaf));
    smaller_subtree_leaf_nodes.extend(rebalanced_nodes);

    // Balancing step should not fully merge two partitions leaving one partition empty.
    // As this affects the partitioning structure it is prevented
    assert!(
        !larger_subtree_leaf_nodes.is_empty(),
        "Currently, passing all leaf nodes from larger to smaller results is undefined"
    );

    // Run Greedy on the two updated subtrees
    let (updated_larger_path, local_larger_path, larger_cost) =
        subtree_contraction_path(&larger_subtree_leaf_nodes, contraction_tree, tensor_network);

    let (updated_smaller_path, local_smaller_path, smaller_cost) = subtree_contraction_path(
        &smaller_subtree_leaf_nodes,
        contraction_tree,
        tensor_network,
    );

    // Remove larger subtree and add new subtree, keep track of updated root id
    contraction_tree.remove_subtree(larger_subtree_id);

    let new_larger_subtree_id = if updated_larger_path.is_empty() {
        // In this case, there is only one node left.
        contraction_tree.nodes[&larger_subtree_leaf_nodes[0]]
            .borrow_mut()
            .set_parent(Rc::downgrade(
                &contraction_tree.nodes[&larger_subtree_parent_id],
            ));
        contraction_tree.nodes[&larger_subtree_parent_id]
            .borrow_mut()
            .add_child(Rc::downgrade(
                &contraction_tree.nodes[&larger_subtree_leaf_nodes[0]],
            ));
        larger_subtree_leaf_nodes[0]
    } else {
        contraction_tree.add_path_as_subtree(
            &updated_larger_path,
            larger_subtree_parent_id,
            &larger_subtree_leaf_nodes,
        )
    };

    // Remove smaller subtree
    contraction_tree.remove_subtree(smaller_subtree_id);
    // Add new subtree, keep track of updated root id
    let new_smaller_subtree_id = contraction_tree.add_path_as_subtree(
        &updated_smaller_path,
        smaller_subtree_parent_id,
        &smaller_subtree_leaf_nodes,
    );

    // Remove the old partition ids from the `ContractionTree` partitions member as the intermediate tensor id will be updated and then add the updated partition numbers.`
    let partition = contraction_tree
        .partitions
        .get_mut(&rebalance_depth)
        .unwrap();
    partition.retain(|&e| e != smaller_subtree_id && e != larger_subtree_id);
    partition.push(new_larger_subtree_id);
    partition.push(new_smaller_subtree_id);

    (
        new_larger_subtree_id,
        local_larger_path,
        larger_cost,
        new_smaller_subtree_id,
        local_smaller_path,
        smaller_cost,
    )
}

#[cfg(test)]
mod tests {
    use std::{
        cell::RefCell,
        rc::{Rc, Weak},
    };

    use rustc_hash::FxHashMap;

    use crate::{
        contractionpath::contraction_tree::{node::Node, ContractionTree},
        path,
        tensornetwork::{create_tensor_network, tensor::Tensor},
    };

    use super::shift_node_between_subtrees;

    fn setup_complex() -> (ContractionTree, Tensor) {
        let (tensor, contraction_path) = (
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
        );
        (
            ContractionTree::from_contraction_path(&tensor, &contraction_path),
            tensor,
        )
    }

    #[test]
    fn test_shift_leaf_node_between_subtrees() {
        let (mut tree, tensor) = setup_complex();
        tree.partitions.entry(1).or_insert(vec![9, 7]);
        shift_node_between_subtrees(&mut tree, 1, 9, 7, vec![3], &tensor);

        let ContractionTree { nodes, root, .. } = tree;

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
            Rc::downgrade(&node2),
            Rc::downgrade(&node4),
            Weak::new(),
            None,
        )));
        let node9 = Rc::new(RefCell::new(Node::new(
            9,
            Rc::downgrade(&node3),
            Rc::downgrade(&node7),
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
        node0.borrow_mut().set_parent(Rc::downgrade(&node7));
        node1.borrow_mut().set_parent(Rc::downgrade(&node6));
        node2.borrow_mut().set_parent(Rc::downgrade(&node8));
        node3.borrow_mut().set_parent(Rc::downgrade(&node9));
        node4.borrow_mut().set_parent(Rc::downgrade(&node8));
        node5.borrow_mut().set_parent(Rc::downgrade(&node6));
        node6.borrow_mut().set_parent(Rc::downgrade(&node7));
        node7.borrow_mut().set_parent(Rc::downgrade(&node9));
        node8.borrow_mut().set_parent(Rc::downgrade(&node10));
        node9.borrow_mut().set_parent(Rc::downgrade(&node10));

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
    fn test_shift_subtree_between_subtrees() {
        let (mut tree, tensor) = setup_complex();
        tree.partitions.entry(1).or_insert(vec![9, 7]);
        shift_node_between_subtrees(&mut tree, 1, 9, 7, vec![2, 3], &tensor);

        let ContractionTree { nodes, root, .. } = tree;

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
            Rc::downgrade(&node3),
            Rc::downgrade(&node2),
            Weak::new(),
            None,
        )));
        let node8 = Rc::new(RefCell::new(Node::new(
            8,
            Rc::downgrade(&node0),
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
            Rc::downgrade(&node9),
            Rc::downgrade(&node4),
            Weak::new(),
            None,
        )));
        node0.borrow_mut().set_parent(Rc::downgrade(&node8));
        node1.borrow_mut().set_parent(Rc::downgrade(&node6));
        node2.borrow_mut().set_parent(Rc::downgrade(&node7));
        node3.borrow_mut().set_parent(Rc::downgrade(&node7));
        node4.borrow_mut().set_parent(Rc::downgrade(&node10));
        node5.borrow_mut().set_parent(Rc::downgrade(&node6));
        node6.borrow_mut().set_parent(Rc::downgrade(&node8));
        node7.borrow_mut().set_parent(Rc::downgrade(&node9));
        node8.borrow_mut().set_parent(Rc::downgrade(&node9));
        node9.borrow_mut().set_parent(Rc::downgrade(&node10));

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
    #[should_panic = "Currently, passing all leaf nodes from larger to smaller results is undefined"]
    fn test_shift_entire_subtree_between_subtrees() {
        let (mut tree, tensor) = setup_complex();
        tree.partitions.entry(1).or_insert(vec![9, 7]);
        shift_node_between_subtrees(&mut tree, 1, 9, 7, vec![2, 3, 4], &tensor);
    }
}
