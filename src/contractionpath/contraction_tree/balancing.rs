use balancing_schemes::BalancingScheme;
use communication_schemes::{
    bipartition_communication_scheme, greedy_communication_scheme,
    weighted_branchbound_communication_scheme, CommunicationScheme,
};
use itertools::Itertools;
use log::info;
use rand::{
    distributions::{Distribution, WeightedIndex},
    thread_rng,
};
use rustc_hash::FxHashMap;

use crate::{
    contractionpath::{
        contraction_cost::contract_path_cost,
        contraction_tree::{
            export::{to_dendogram_format, to_pdf},
            utils::calculate_partition_costs,
        },
        paths::validate_path,
    },
    mpi::communication::extract_communication_path,
    tensornetwork::tensor::Tensor,
    types::ContractionIndex,
};

use super::{export::DendogramSettings, populate_subtree_tensor_map, ContractionTree};

pub mod balancing_schemes;
pub mod communication_schemes;

#[derive(Debug)]
pub struct BalanceSettings {
    pub random_balance: bool,
    pub rebalance_depth: usize,
    pub iterations: usize,
    pub greedy_cost_function: fn(&Tensor, &Tensor) -> f64,
    pub communication_scheme: CommunicationScheme,
    pub balancing_scheme: BalancingScheme,
}

pub fn balance_partitions_iter(
    tensor: &Tensor,
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
    let mut contraction_tree = ContractionTree::from_contraction_path(tensor, path);
    let mut path = path.to_owned();
    let final_contraction = extract_communication_path(&path);
    let mut partition_costs =
        calculate_partition_costs(&contraction_tree, rebalance_depth, tensor, true);

    assert!(partition_costs.len() > 1);
    let partition_number = partition_costs.len();

    let (_, mut max_cost) = partition_costs.last().unwrap();

    let children = &contraction_tree.partitions()[&rebalance_depth];

    let children_tensors = children
        .iter()
        .map(|e| contraction_tree.tensor(*e, tensor))
        .collect_vec();

    let (final_op_cost, _) = contract_path_cost(&children_tensors, &final_contraction);
    let mut max_costs = Vec::with_capacity(iterations + 1);
    max_costs.push(max_cost + final_op_cost);

    if let Some(settings) = dendogram_settings {
        let dendogram_entries =
            to_dendogram_format(&contraction_tree, tensor, settings.cost_function);
        to_pdf(
            &format!("{}_0", settings.output_file),
            &dendogram_entries,
            None,
        );
    }

    let mut new_tn;
    let mut best_contraction = 0;
    let mut best_contraction_path = path.clone();
    let mut best_cost = max_cost + final_op_cost;

    let mut best_tn = tensor.clone();

    for i in 1..=iterations {
        info!("Balancing iteration {i} with balancing scheme {balancing_scheme:?}, communication scheme {communication_scheme:?}");
        (max_cost, path, new_tn) = balance_partitions(
            tensor,
            &mut contraction_tree,
            random_balance,
            rebalance_depth,
            &partition_costs,
            greedy_cost_function,
            balancing_scheme,
        );
        assert_eq!(partition_number, path.len(), "Tensors lost!");
        validate_path(&path);

        partition_costs =
            calculate_partition_costs(&contraction_tree, rebalance_depth, tensor, true);

        // Ensures that children tensors are mapped to their respective partition costs
        let (final_op_cost, final_contraction) = communicate_partitions(
            &partition_costs,
            &contraction_tree,
            &new_tn,
            communication_scheme,
        );

        path.extend(final_contraction);
        let new_max_cost = max_cost + final_op_cost;

        max_costs.push(new_max_cost);

        if new_max_cost < best_cost {
            best_cost = new_max_cost;
            best_contraction = i;
            best_tn = new_tn;
            best_contraction_path = path;
        }

        if let Some(settings) = dendogram_settings {
            let dendogram_entries =
                to_dendogram_format(&contraction_tree, tensor, settings.cost_function);
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
    partition_costs: &[(usize, f64)],
    _contraction_tree: &ContractionTree,
    tensor: &Tensor,
    communication_scheme: CommunicationScheme,
) -> (f64, Vec<ContractionIndex>) {
    let bond_dims = tensor.bond_dims();
    let children_tensors = tensor
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
            let latency_map = partition_costs.iter().copied().collect();
            weighted_branchbound_communication_scheme(&children_tensors, &bond_dims, latency_map)
        }
    };
    (final_op_cost, final_contraction)
}

pub(super) fn balance_partitions(
    tn: &Tensor,
    contraction_tree: &mut ContractionTree,
    random_balance: bool,
    rebalance_depth: usize,
    partition_costs: &[(usize, f64)],
    greedy_cost_function: fn(&Tensor, &Tensor) -> f64,
    balancing_scheme: BalancingScheme,
) -> (f64, Vec<ContractionIndex>, Tensor) {
    // If there are less than 3 tensors in the tn, rebalancing will not make sense.
    if tn.total_num_tensors() < 3 {
        // TODO: should not panic, but handle gracefully
        panic!("No rebalancing undertaken, as tn is too small (< 3 tensors)");
    }
    // Will cause strange errors (picking of same partition multiple times if this is not true.Better to panic here.)
    assert!(partition_costs.len() > 1);
    // Use memory reduction to identify which leaf node to shift between partitions.
    let bond_dims = tn.bond_dims();
    let (new_max, partition_tensors, rebalanced_path) = match balancing_scheme {
        BalancingScheme::BestWorst => balancing_schemes::best_worst_balancing(
            partition_costs,
            contraction_tree,
            random_balance,
            greedy_cost_function,
            tn,
            rebalance_depth,
        ),
        BalancingScheme::Tensor => balancing_schemes::best_tensor_balancing(
            partition_costs,
            contraction_tree,
            random_balance,
            greedy_cost_function,
            tn,
            rebalance_depth,
        ),
        BalancingScheme::Tensors => balancing_schemes::best_tensors_balancing(
            partition_costs,
            contraction_tree,
            random_balance,
            greedy_cost_function,
            tn,
            rebalance_depth,
        ),
        _ => panic!("Balancing Scheme not implemented"),
    };

    let mut updated_tn = Tensor::default();
    updated_tn.push_tensors(partition_tensors, Some(&bond_dims), None);
    (new_max, rebalanced_path, updated_tn)
}

/// Computes a hashmap that maps the node id to its weight. The weight is the maximum
/// memory reduction that can be achieved if it is shifted to the other subtree.
pub(super) fn find_potential_nodes(
    contraction_tree: &ContractionTree,
    bigger_subtree_leaf_nodes: &[usize],
    smaller_subtree_root: usize,
    tn: &Tensor,
    cost_function: fn(&Tensor, &Tensor) -> f64,
) -> FxHashMap<usize, f64> {
    // Get a map that maps nodes to their tensors.
    let mut node_tensor_map = FxHashMap::default();
    populate_subtree_tensor_map(
        contraction_tree,
        smaller_subtree_root,
        &mut node_tensor_map,
        tn,
    );

    (bigger_subtree_leaf_nodes.iter().map(|leaf_index| {
        let t1 = tn.nested_tensor(
            contraction_tree
                .node(*leaf_index)
                .tensor_index
                .as_ref()
                .unwrap(),
        );
        node_tensor_map
            .iter()
            .map(|(&index, tensor)| (index, cost_function(tensor, t1)))
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap()
    }))
    .collect()
}

/// Takes two hashmaps that contain node information. Identifies which pair of nodes from larger and smaller hashmaps maximizes the greedy cost function
pub(super) fn find_rebalance_node(
    random_balance: Option<usize>,
    larger_subtree_nodes: &FxHashMap<usize, Tensor>,
    smaller_subtree_nodes: &FxHashMap<usize, Tensor>,
    greedy_cost_fn: fn(&Tensor, &Tensor) -> f64,
) -> (usize, f64) {
    let node_comparison = larger_subtree_nodes
        .iter()
        .cartesian_product(smaller_subtree_nodes.iter())
        .map(|((larger_node_id, larger_tensor), (_, smaller_tensor))| {
            (
                *larger_node_id,
                greedy_cost_fn(larger_tensor, smaller_tensor),
            )
        });
    if let Some(options_considered) = random_balance {
        let mut rng = thread_rng();
        let node_options = node_comparison
            .sorted_by(|a, b| a.1.total_cmp(&b.1))
            .take(options_considered)
            .collect::<Vec<(usize, f64)>>();
        *node_options
            .choose_weighted(&mut rng, |node_option| (-1.0 * node_option.1).exp())
            .unwrap()
    } else {
        node_comparison.max_by(|a, b| a.1.total_cmp(&b.1)).unwrap()
    }
}
