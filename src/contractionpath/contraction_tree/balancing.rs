use std::{collections::HashSet, rc::Rc};

use itertools::Itertools;
use log::info;
use rand::{rngs::StdRng, seq::SliceRandom, Rng};
use rustc_hash::FxHashMap;

use crate::{
    contractionpath::{
        contraction_cost::communication_path_cost,
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

mod balancing_schemes;
mod communication_schemes;

pub use balancing_schemes::BalancingScheme;
pub use communication_schemes::CommunicationScheme;

#[derive(Debug, Clone, Copy)]
pub struct BalanceSettings<R>
where
    R: Sized + Rng,
{
    /// If not None, randomly chooses from top `usize` options. Random choice is
    /// weighted by objective outcome.
    pub random_balance: Option<(usize, R)>,
    pub rebalance_depth: usize,
    pub iterations: usize,
    pub objective_function: fn(&Tensor, &Tensor) -> f64,
    pub communication_scheme: CommunicationScheme,
    pub balancing_scheme: BalancingScheme,
}

impl BalanceSettings<StdRng> {
    pub fn new(
        rebalance_depth: usize,
        iterations: usize,
        objective_function: fn(&Tensor, &Tensor) -> f64,
        communication_scheme: CommunicationScheme,
        balancing_scheme: BalancingScheme,
    ) -> Self {
        BalanceSettings::<StdRng> {
            random_balance: None,
            rebalance_depth,
            iterations,
            objective_function,
            communication_scheme,
            balancing_scheme,
        }
    }
}

impl<R> BalanceSettings<R>
where
    R: Sized + Rng,
{
    pub fn new_random(
        random_balance: Option<(usize, R)>,
        rebalance_depth: usize,
        iterations: usize,
        objective_function: fn(&Tensor, &Tensor) -> f64,
        communication_scheme: CommunicationScheme,
        balancing_scheme: BalancingScheme,
    ) -> Self {
        BalanceSettings::<R> {
            random_balance,
            rebalance_depth,
            iterations,
            objective_function,
            communication_scheme,
            balancing_scheme,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PartitionData {
    pub id: usize,
    pub cost: f64,
    pub contraction: Vec<ContractionIndex>,
    pub local_tensor: Tensor,
}

pub fn balance_partitions_iter<R>(
    tensor_network: &Tensor,
    path: &[ContractionIndex],
    mut balance_settings: BalanceSettings<R>,
    dendogram_settings: Option<&DendogramSettings>,
) -> (usize, Tensor, Vec<ContractionIndex>, Vec<f64>)
where
    R: Sized + Rng,
{
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

    let children_tensors = partition_data
        .iter()
        .map(|PartitionData { local_tensor, .. }| local_tensor.clone())
        .collect_vec();

    let partition_tensor_costs = partition_data
        .iter()
        .enumerate()
        .map(|(i, partition)| (i, partition.cost))
        .collect();
    let (mut best_cost, _) = communication_path_cost(
        &children_tensors,
        &communication_path,
        true,
        Some(&partition_tensor_costs),
    );
    let mut intermediate_cost = f64::INFINITY;
    let mut contracted_tensors = HashSet::new();
    for contraction in communication_path {
        if let ContractionIndex::Pair(i, j) = contraction {
            if !contracted_tensors.contains(&i) && !contracted_tensors.contains(&j) {
                let local_max = partition_data[i].cost.max(partition_data[j].cost);
                if local_max < intermediate_cost {
                    intermediate_cost = local_max;
                }
            }
            contracted_tensors.insert(i);
            contracted_tensors.insert(j);
        }
    }
    let mut max_costs = Vec::with_capacity(iterations + 1);
    max_costs.push(best_cost);

    print_dendogram(dendogram_settings, &contraction_tree, tensor_network, 0);

    let mut best_iteration = 0;
    let mut best_contraction_path = path.to_owned();

    let mut best_tn = tensor_network.clone();

    for i in 1..=iterations {
        info!("Balancing iteration {i} with balancing scheme {balancing_scheme:?}, communication scheme {communication_scheme:?}");

        // Balances and updates partitions
        let (_, mut intermediate_path, new_tensor_network) = balance_partitions(
            &mut partition_data,
            &mut contraction_tree,
            tensor_network,
            &mut balance_settings,
        );

        assert_eq!(intermediate_path.len(), partition_number, "Tensors lost!");
        validate_path(&intermediate_path);

        // Ensures that children tensors are mapped to their respective partition costs
        // Communication costs include intermediate costs
        let (new_cost, communication_path) = communicate_partitions(
            &partition_data,
            &contraction_tree,
            &new_tensor_network,
            &balance_settings,
        );

        let mut intermediate_cost = f64::INFINITY;
        let mut contracted_tensors = HashSet::new();
        for contraction in communication_path.iter() {
            if let ContractionIndex::Pair(i, j) = contraction {
                if !contracted_tensors.contains(&i) && !contracted_tensors.contains(&j) {
                    let local_max = partition_data[*i].cost.max(partition_data[*j].cost);
                    if local_max < intermediate_cost {
                        intermediate_cost = local_max;
                    }
                }
                contracted_tensors.insert(i);
                contracted_tensors.insert(j);
            }
        }

        intermediate_path.extend(communication_path);

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

fn communicate_partitions<R>(
    partition_data: &[PartitionData],
    _contraction_tree: &ContractionTree,
    tensor_network: &Tensor,
    balance_settings: &BalanceSettings<R>,
) -> (f64, Vec<ContractionIndex>)
where
    R: Sized + Rng,
{
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
    let latency_map = partition_data
        .iter()
        .enumerate()
        .map(|(i, partition)| (i, partition.cost))
        .collect::<FxHashMap<usize, f64>>();
    let (communication_cost, communication_path) = match communication_scheme {
        CommunicationScheme::Greedy => {
            greedy_communication_scheme(&children_tensors, &bond_dims, &latency_map)
        }
        CommunicationScheme::Bipartition => {
            bipartition_communication_scheme(&children_tensors, &bond_dims, &latency_map)
        }
        CommunicationScheme::WeightedBranchBound => {
            weighted_branchbound_communication_scheme(&children_tensors, &bond_dims, &latency_map)
        }
    };
    (communication_cost, communication_path)
}

fn balance_partitions<R>(
    partition_data: &mut [PartitionData],
    contraction_tree: &mut ContractionTree,
    tensor_network: &Tensor,
    balance_settings: &mut BalanceSettings<R>,
) -> (f64, Vec<ContractionIndex>, Tensor)
where
    R: Sized + Rng,
{
    let BalanceSettings {
        ref mut random_balance,
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

    let bond_dims = tensor_network.bond_dims();
    let shifted_nodes = match balancing_scheme {
        BalancingScheme::BestWorst => balancing_schemes::best_worst_balancing(
            partition_data,
            contraction_tree,
            random_balance,
            *objective_function,
            tensor_network,
        ),
        BalancingScheme::Tensor => balancing_schemes::best_tensor_balancing(
            partition_data,
            contraction_tree,
            random_balance,
            *objective_function,
            tensor_network,
        ),
        BalancingScheme::Tensors => balancing_schemes::best_tensors_balancing(
            partition_data,
            contraction_tree,
            random_balance,
            *objective_function,
            tensor_network,
        ),
        BalancingScheme::IntermediateTensors { height_limit } => {
            balancing_schemes::best_intermediate_tensors_balancing(
                partition_data,
                contraction_tree,
                random_balance,
                *objective_function,
                tensor_network,
                *height_limit,
            )
        }
        _ => panic!("Balancing Scheme not implemented"),
    };
    let mut shifted_indices = FxHashMap::default();
    for shift in shifted_nodes {
        let shifted_from_id = *shifted_indices
            .get(&shift.from_subtree_id)
            .unwrap_or(&shift.from_subtree_id);

        let shifted_to_id = *shifted_indices
            .get(&shift.to_subtree_id)
            .unwrap_or(&shift.to_subtree_id);

        let (
            larger_id,
            larger_contraction,
            larger_subtree_cost,
            smaller_id,
            smaller_contraction,
            smaller_subtree_cost,
        ) = shift_node_between_subtrees(
            contraction_tree,
            *rebalance_depth,
            shifted_from_id,
            shifted_to_id,
            shift.moved_leaf_ids,
            tensor_network,
        );
        shifted_indices.insert(shift.from_subtree_id, larger_id);
        shifted_indices.insert(shift.to_subtree_id, smaller_id);

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
        .insert(*rebalance_depth, partition_ids);

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
fn find_rebalance_node<R>(
    random_balance: &mut Option<(usize, R)>,
    larger_subtree_nodes: &FxHashMap<usize, Tensor>,
    smaller_subtree_nodes: &FxHashMap<usize, Tensor>,
    objective_function: fn(&Tensor, &Tensor) -> f64,
) -> (usize, f64)
where
    R: Sized + Rng,
{
    let node_comparison = larger_subtree_nodes
        .iter()
        .cartesian_product(smaller_subtree_nodes.iter())
        .map(|((larger_node_id, larger_tensor), (_, smaller_tensor))| {
            (
                *larger_node_id,
                objective_function(larger_tensor, smaller_tensor),
            )
        });
    if let Some((options_considered, ref mut rng)) = random_balance {
        let node_options = node_comparison
            .sorted_unstable_by(|a, b| b.1.total_cmp(&a.1))
            .take(*options_considered)
            .collect::<Vec<(usize, f64)>>();
        let max = node_options.first().unwrap().1;
        // Initial division done here as sum of weights can cause overflow before normalization.
        *node_options
            .choose_weighted(rng, |node_option| node_option.1 / max)
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
    use std::rc::Rc;

    use rand::{rngs::StdRng, SeedableRng};
    use rustc_hash::FxHashMap;

    use crate::{
        contractionpath::contraction_tree::{
            balancing::find_rebalance_node,
            node::{child_node, parent_node},
            ContractionTree,
        },
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

        let node0 = child_node(0, vec![0]);
        let node1 = child_node(1, vec![1]);
        let node2 = child_node(2, vec![2]);
        let node3 = child_node(3, vec![3]);
        let node4 = child_node(4, vec![4]);
        let node5 = child_node(5, vec![5]);

        let node6 = parent_node(6, &node1, &node5);
        let node7 = parent_node(7, &node0, &node6);
        let node8 = parent_node(8, &node2, &node4);
        let node9 = parent_node(9, &node3, &node7);
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
    fn test_shift_subtree_between_subtrees() {
        let (mut tree, tensor) = setup_complex();
        tree.partitions.entry(1).or_insert(vec![9, 7]);
        shift_node_between_subtrees(&mut tree, 1, 9, 7, vec![2, 3], &tensor);

        let ContractionTree { nodes, root, .. } = tree;

        let node0 = child_node(0, vec![0]);
        let node1 = child_node(1, vec![1]);
        let node2 = child_node(2, vec![2]);
        let node3 = child_node(3, vec![3]);
        let node4 = child_node(4, vec![4]);
        let node5 = child_node(5, vec![5]);

        let node6 = parent_node(6, &node1, &node5);
        let node7 = parent_node(7, &node3, &node2);
        let node8 = parent_node(8, &node0, &node6);
        let node9 = parent_node(9, &node7, &node8);
        let node10 = parent_node(10, &node9, &node4);

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
    #[should_panic = "Currently, passing all leaf nodes from larger to smaller results is undefined"]
    fn test_shift_entire_subtree_between_subtrees() {
        let (mut tree, tensor) = setup_complex();
        tree.partitions.entry(1).or_insert(vec![9, 7]);
        shift_node_between_subtrees(&mut tree, 1, 9, 7, vec![2, 3, 4], &tensor);
    }

    fn custom_weight_function(a: &Tensor, b: &Tensor) -> f64 {
        (a & b).legs().len() as f64
    }

    #[test]
    fn test_find_rebalance_node() {
        let larger_hash = FxHashMap::from_iter([
            (0, Tensor::new(vec![0, 1, 2])),
            (1, Tensor::new(vec![1, 2, 3])),
            (2, Tensor::new(vec![3, 4, 5])),
        ]);

        let smaller_hash = FxHashMap::from_iter([(3, Tensor::new(vec![4, 5, 6]))]);

        let ref_balanced_node = 2;
        let (node_id, cost) = find_rebalance_node::<StdRng>(
            &mut None,
            &larger_hash,
            &smaller_hash,
            custom_weight_function,
        );
        assert_eq!(2f64, cost);
        assert_eq!(ref_balanced_node, node_id);
    }

    #[test]
    fn test_find_random_rebalance_node() {
        let larger_hash = FxHashMap::from_iter([
            (0, Tensor::new(vec![0, 1, 2])),
            (1, Tensor::new(vec![1, 2, 6])),
            (2, Tensor::new(vec![3, 4, 5])),
        ]);

        let smaller_hash = FxHashMap::from_iter([(3, Tensor::new(vec![4, 5, 6]))]);

        let ref_balanced_node = 1;
        let (node_id, cost) = find_rebalance_node(
            &mut Some((2, StdRng::seed_from_u64(1))),
            &larger_hash,
            &smaller_hash,
            custom_weight_function,
        );
        assert_eq!(1f64, cost);
        assert_eq!(ref_balanced_node, node_id);
    }
}
