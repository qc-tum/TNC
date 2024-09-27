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

#[derive(Debug)]
pub struct BalanceSettings {
    pub random_balance: Option<usize>,
    pub rebalance_depth: usize,
    pub iterations: usize,
    pub greedy_cost_function: fn(&Tensor, &Tensor) -> f64,
    pub communication_scheme: CommunicationScheme,
    pub balancing_scheme: BalancingScheme,
}

#[derive(Debug)]
pub struct PartitionData {
    pub id: usize,
    pub cost: f64,
    pub contraction: Vec<ContractionIndex>,
    pub local_tensor: Tensor,
}

pub fn balance_partitions_iter(
    tensor_network: &Tensor,
    path: &[ContractionIndex],
    BalanceSettings {
        random_balance,
        rebalance_depth,
        iterations,
        greedy_cost_function,
        communication_scheme,
        balancing_scheme,
    }: BalanceSettings,
    dendogram_settings: &Option<DendogramSettings>,
) -> (usize, Tensor, Vec<ContractionIndex>, Vec<f64>) {
    let mut contraction_tree = ContractionTree::from_contraction_path(tensor_network, path);
    let mut path = path.to_owned();
    let final_contraction = extract_communication_path(&path);

    let mut partition_data =
        characterize_partition(&contraction_tree, rebalance_depth, tensor_network, true);

    assert!(partition_data.len() > 1);

    let partition_number = partition_data.len();

    let PartitionData {
        cost: mut max_cost, ..
    } = partition_data.last().unwrap();

    let children_tensors = partition_data
        .iter()
        .map(
            |PartitionData {
                 local_tensor: tensor,
                 ..
             }| tensor.clone(),
        )
        .collect_vec();

    let (final_op_cost, _) = contract_path_cost(&children_tensors, &final_contraction);
    let mut max_costs = Vec::with_capacity(iterations + 1);
    max_costs.push(max_cost + final_op_cost);

    if let Some(settings) = dendogram_settings {
        let dendogram_entries =
            to_dendogram_format(&contraction_tree, tensor_network, settings.cost_function);
        to_pdf(
            &format!("{}_0", settings.output_file),
            &dendogram_entries,
            None,
        );
    }

    let mut new_tensor;
    let mut best_contraction = 0;
    let mut best_contraction_path = path.clone();
    let mut best_cost = max_cost + final_op_cost;

    let mut best_tn = tensor_network.clone();

    for i in 1..=iterations {
        info!("Balancing iteration {i} with balancing scheme {balancing_scheme:?}, communication scheme {communication_scheme:?}");

        // Balances and updates partitions
        (max_cost, path, new_tensor) = balance_partitions(
            tensor_network,
            &mut contraction_tree,
            rebalance_depth,
            random_balance,
            &mut partition_data,
            greedy_cost_function,
            balancing_scheme,
        );

        assert_eq!(partition_number, path.len(), "Tensors lost!");
        validate_path(&path);

        // Ensures that children tensors are mapped to their respective partition costs
        let (final_op_cost, final_contraction) = communicate_partitions(
            &partition_data,
            &contraction_tree,
            &new_tensor,
            &communication_scheme,
        );

        path.extend(final_contraction);
        let new_max_cost = max_cost + final_op_cost;

        max_costs.push(new_max_cost);

        if new_max_cost < best_cost {
            best_cost = new_max_cost;
            best_contraction = i;
            best_tn = new_tensor;
            best_contraction_path = path;
        }

        if let Some(settings) = dendogram_settings {
            let dendogram_entries =
                to_dendogram_format(&contraction_tree, tensor_network, settings.cost_function);
            to_pdf(
                &format!("{}_{i}", settings.output_file),
                &dendogram_entries,
                None,
            );
        }
    }

    (best_contraction, best_tn, best_contraction_path, max_costs)
}

pub(super) fn communicate_partitions(
    partition_costs: &[PartitionData],
    _contraction_tree: &ContractionTree,
    tensor_network: &Tensor,
    communication_scheme: &CommunicationScheme,
) -> (f64, Vec<ContractionIndex>) {
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

    let (final_op_cost, final_contraction) = match communication_scheme {
        CommunicationScheme::Greedy => greedy_communication_scheme(&children_tensors, &bond_dims),
        CommunicationScheme::Bipartition => {
            bipartition_communication_scheme(&children_tensors, &bond_dims)
        }
        CommunicationScheme::WeightedBranchBound => {
            let partition_iter = partition_costs
                .iter()
                .enumerate()
                .map(|(i, PartitionData { cost, .. })| (i, *cost));
            let latency_map = FxHashMap::from_iter(partition_iter);
            weighted_branchbound_communication_scheme(&children_tensors, &bond_dims, latency_map)
        }
    };
    (final_op_cost, final_contraction)
}

pub(super) fn balance_partitions(
    tensor_network: &Tensor,
    contraction_tree: &mut ContractionTree,
    rebalance_depth: usize,
    random_balance: Option<usize>,
    partition_data: &mut [PartitionData],
    greedy_cost_function: fn(&Tensor, &Tensor) -> f64,
    balancing_scheme: BalancingScheme,
) -> (f64, Vec<ContractionIndex>, Tensor) {
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
            greedy_cost_function,
            tensor_network,
        ),
        BalancingScheme::Tensor => balancing_schemes::best_tensor_balancing(
            partition_data,
            contraction_tree,
            random_balance,
            greedy_cost_function,
            tensor_network,
        ),
        BalancingScheme::Tensors => balancing_schemes::best_tensors_balancing(
            partition_data,
            contraction_tree,
            random_balance,
            greedy_cost_function,
            tensor_network,
        ),
        BalancingScheme::IntermediateTensors(height_limit) => {
            balancing_schemes::best_intermediate_tensors_balancing(
                partition_data,
                contraction_tree,
                random_balance,
                greedy_cost_function,
                tensor_network,
                height_limit,
            )
        }
        _ => panic!("Balancing Scheme not implemented"),
    };
    let mut shifted_indices = FxHashMap::default();
    for (from_subtree_id, to_subtree_id, rebalanced_nodes) in shifted_nodes {
        let shifted_from_id = if shifted_indices.contains_key(&from_subtree_id) {
            *shifted_indices.get(&from_subtree_id).unwrap()
        } else {
            from_subtree_id
        };

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
            }
            if *id == shifted_to_id {
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
                            .tensor_index
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

/// Takes two hashmaps that contain node information. Identifies which pair of nodes from larger and smaller hashmaps maximizes the greedy cost function
pub(super) fn find_rebalance_node(
    random_balance: Option<usize>,
    larger_subtree_nodes: &FxHashMap<usize, Tensor>,
    smaller_subtree_nodes: &FxHashMap<usize, Tensor>,
    greedy_cost_function: fn(&Tensor, &Tensor) -> f64,
) -> (usize, f64) {
    let node_comparison = larger_subtree_nodes
        .iter()
        .cartesian_product(smaller_subtree_nodes.iter())
        .map(|((larger_node_id, larger_tensor), (_, smaller_tensor))| {
            (
                *larger_node_id,
                greedy_cost_function(larger_tensor, smaller_tensor),
            )
        });
    if let Some(options_considered) = random_balance {
        let mut rng = thread_rng();
        let node_options = node_comparison
            .sorted_by(|a, b| a.1.total_cmp(&b.1))
            .take(options_considered)
            .collect::<Vec<(usize, f64)>>();
        let max = node_options.first().unwrap().1;
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

    // Remove the updated subtrees from the `ContractionTree` partitions member as the intermediate tensor id will be updated.
    contraction_tree
        .partitions
        .get_mut(&rebalance_depth)
        .unwrap()
        .retain(|&e| e != smaller_subtree_id && e != larger_subtree_id);

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

    // Run Greedy on the two updated subtrees
    let (updated_larger_path, local_larger_path, larger_cost) =
        subtree_contraction_path(&larger_subtree_leaf_nodes, contraction_tree, tensor_network);

    let (updated_smaller_path, local_smaller_path, smaller_cost) = subtree_contraction_path(
        &smaller_subtree_leaf_nodes,
        contraction_tree,
        tensor_network,
    );

    // Remove larger subtree
    contraction_tree.remove_subtree(larger_subtree_id);
    // Add new subtree, keep track of updated root id
    let new_larger_subtree_id = contraction_tree.add_subtree(
        &updated_larger_path,
        larger_subtree_parent_id,
        &larger_subtree_leaf_nodes,
    );

    // Remove smaller subtree
    contraction_tree.remove_subtree(smaller_subtree_id);

    // Add new subtree, keep track of updated root id
    let new_smaller_subtree_id = contraction_tree.add_subtree(
        &updated_smaller_path,
        smaller_subtree_parent_id,
        &smaller_subtree_leaf_nodes,
    );

    (
        new_larger_subtree_id,
        local_larger_path,
        larger_cost,
        new_smaller_subtree_id,
        local_smaller_path,
        smaller_cost,
    )
}
