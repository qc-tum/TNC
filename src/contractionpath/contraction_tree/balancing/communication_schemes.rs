use itertools::Itertools;
use rustc_hash::FxHashMap;

use crate::contractionpath::contraction_cost::contract_cost_tensors;
use crate::contractionpath::contraction_tree::utils::parallel_tree_contraction_cost;
use crate::contractionpath::contraction_tree::ContractionTree;
use crate::contractionpath::paths::greedy::Greedy;
use crate::contractionpath::paths::weighted_branchbound::WeightedBranchBound;
use crate::contractionpath::paths::{CostType, OptimizePath};
use crate::pair;
use crate::tensornetwork::partitioning::communication_partitioning;
use crate::tensornetwork::partitioning::partition_config::PartitioningStrategy;

use std::sync::RwLockReadGuard;

use crate::types::ContractionIndex;

use crate::tensornetwork::tensor::Tensor;

#[derive(Debug, Copy, Clone)]
pub enum CommunicationScheme {
    /// Uses Greedy scheme to find contraction path for communication
    Greedy,
    /// Uses repeated bipartitioning to identify communication path
    Bipartition,
    /// Uses a filtered search that considered time to intermediate tensor
    WeightedBranchBound,
}

pub(super) fn greedy_communication_scheme(
    children_tensors: &[Tensor],
    bond_dims: &RwLockReadGuard<FxHashMap<usize, u64>>,
) -> (f64, Vec<ContractionIndex>) {
    let mut communication_tensors = Tensor::default();
    communication_tensors.push_tensors(children_tensors.to_vec(), Some(bond_dims), None);

    let mut opt = Greedy::new(&communication_tensors, CostType::Flops);
    opt.optimize_path();
    let final_contraction = opt.get_best_replace_path();
    let contraction_tree =
        ContractionTree::from_contraction_path(&communication_tensors, &final_contraction);
    // let (final_op_cost, _) = contract_path_cost(&children_tensors, &final_contraction);
    let (final_op_cost, _, _) = parallel_tree_contraction_cost(
        &contraction_tree,
        contraction_tree.root_id().unwrap(),
        &communication_tensors,
    );
    (final_op_cost, final_contraction)
}

pub(super) fn bipartition_communication_scheme(
    children_tensors: &[Tensor],
    bond_dims: &RwLockReadGuard<FxHashMap<usize, u64>>,
) -> (f64, Vec<ContractionIndex>) {
    let children_tensors = children_tensors.iter().cloned().enumerate().collect_vec();
    let (final_op_cost, final_contraction) = tensor_bipartition(&children_tensors, bond_dims);

    (final_op_cost, final_contraction)
}

pub(super) fn weighted_branchbound_communication_scheme(
    children_tensors: &[Tensor],
    bond_dims: &RwLockReadGuard<FxHashMap<usize, u64>>,
    latency_map: FxHashMap<usize, f64>,
) -> (f64, Vec<ContractionIndex>) {
    let mut communication_tensors = Tensor::default();
    communication_tensors.push_tensors(children_tensors.to_vec(), Some(bond_dims), None);

    let mut opt = WeightedBranchBound::new(
        &communication_tensors,
        None,
        2f64,
        latency_map,
        CostType::Flops,
    );
    opt.optimize_path();
    let final_contraction = opt.get_best_replace_path();
    let contraction_tree =
        ContractionTree::from_contraction_path(&communication_tensors, &final_contraction);
    let (final_op_cost, _, _) = parallel_tree_contraction_cost(
        &contraction_tree,
        contraction_tree.root_id().unwrap(),
        &communication_tensors,
    );
    (final_op_cost, final_contraction)
}

/// Uses recursive bipartitioning to identify a communication scheme for final tensors
/// Returns root id of subtree, parallel contraction cost as f64, resultant tensor and prior contraction sequence
pub fn tensor_bipartition_recursive(
    children_tensor: &[(usize, Tensor)],
    bond_dims: &FxHashMap<usize, u64>,
) -> (usize, f64, Tensor, Vec<ContractionIndex>) {
    let k = 2;
    let min = true;

    if children_tensor.len() == 1 {
        return (
            children_tensor[0].0,
            0.0,
            children_tensor[0].1.clone(),
            Vec::new(),
        );
    }
    if children_tensor.len() == 2 {
        // Always ensure that the larger tensor size is on the left.
        let (t1, t2) = if children_tensor[1].1.size() > children_tensor[0].1.size() {
            (children_tensor[1].0, children_tensor[0].0)
        } else {
            (children_tensor[0].0, children_tensor[1].0)
        };
        let tensor = &children_tensor[0].1 ^ &children_tensor[1].1;
        let contraction_cost = contract_cost_tensors(&children_tensor[0].1, &children_tensor[1].1);
        return (t1, contraction_cost, tensor, vec![pair!(t1, t2)]);
    }

    let partitioning = communication_partitioning(
        children_tensor,
        bond_dims,
        k,
        PartitioningStrategy::MinCut,
        min,
    );

    let mut partition_iter = partitioning.iter();
    let (children_1, children_2): (Vec<_>, Vec<_>) = children_tensor
        .iter()
        .cloned()
        .partition(|_| partition_iter.next() == Some(&0));

    let (id_1, cost_1, t1, mut contraction_1) =
        tensor_bipartition_recursive(&children_1, bond_dims);

    let (id_2, cost_2, t2, mut contraction_2) =
        tensor_bipartition_recursive(&children_2, bond_dims);

    let cost = cost_1.max(cost_2) + contract_cost_tensors(&t1, &t2);
    let tensor = &t1 ^ &t2;

    contraction_1.append(&mut contraction_2);
    let (id_1, id_2) = if t2.size() > t1.size() {
        (id_2, id_1)
    } else {
        (id_1, id_2)
    };

    contraction_1.push(pair!(id_1, id_2));
    (id_1, cost, tensor, contraction_1)
}

/// Repeatedly bipartitions tensor network to obtain communication scheme
/// Assumes that all tensors contracted do so in parallel
pub fn tensor_bipartition(
    children_tensor: &[(usize, Tensor)],
    bond_dims: &FxHashMap<usize, u64>,
) -> (f64, Vec<ContractionIndex>) {
    let (_, contraction_cost, _, contraction_path) =
        tensor_bipartition_recursive(children_tensor, bond_dims);
    (contraction_cost, contraction_path)
}

#[cfg(test)]
mod tests {
    use std::{
        borrow::Borrow,
        sync::{Arc, RwLock},
    };

    use rand::rngs::StdRng;
    use rustc_hash::FxHashMap;

    use crate::{
        contractionpath::contraction_tree::{
            balancing::{
                communication_schemes::{
                    bipartition_communication_scheme, greedy_communication_scheme,
                    weighted_branchbound_communication_scheme,
                },
                PartitionData,
            },
            ContractionTree,
        },
        path,
        tensornetwork::tensor::Tensor,
    };

    fn setup_simple_partition_data() -> FxHashMap<usize, f64> {
        FxHashMap::from_iter([(0, 0f64), (1, 40f64), (2, 170f64)])
    }

    /// Tensor ids in contraction tree included in variable name for easy tracking
    fn setup_simple() -> (Vec<Tensor>, Arc<RwLock<FxHashMap<usize, u64>>>) {
        let bond_dims = Arc::new(RwLock::new(FxHashMap::from_iter([
            (0, 2),
            (1, 2),
            (2, 2),
            (3, 2),
            (4, 2),
            (5, 2),
            (6, 2),
            (7, 2),
            (8, 2),
            (9, 2),
            (10, 2),
        ])));

        let tensor2 = Tensor::new_with_bonddims(vec![7, 9, 10], Arc::clone(&bond_dims));

        let tensor7 = Tensor::new_with_bonddims(vec![0, 1, 5, 7], Arc::clone(&bond_dims));

        let tensor14 = Tensor::new_with_bonddims(vec![0, 1, 2, 5, 10], Arc::clone(&bond_dims));

        (vec![tensor2, tensor7, tensor14], bond_dims)
    }

    fn custom_cost_function(a: &Tensor, b: &Tensor) -> f64 {
        (a & b).legs().len() as f64
    }

    #[test]
    fn test_greedy_communication() {
        let (tensors, bond_dims) = setup_simple();

        let (cost, communication_scheme) =
            greedy_communication_scheme(&tensors, &bond_dims.read().unwrap());

        assert_eq!(616f64, cost);
        assert_eq!(path![(2, 1), (0, 2)].to_vec(), communication_scheme);
    }

    #[test]
    fn test_bipartition_communication() {
        let (tensors, bond_dims) = setup_simple();

        let (cost, communication_scheme) =
            bipartition_communication_scheme(&tensors, &bond_dims.read().unwrap());

        assert_eq!(952f64, cost);
        assert_eq!(path![(1, 0), (2, 1)].to_vec(), communication_scheme);
    }

    #[test]
    fn test_weighted_communication() {
        let latency_map = setup_simple_partition_data();
        let (tensors, bond_dims) = setup_simple();

        let (cost, communication_scheme) = weighted_branchbound_communication_scheme(
            &tensors,
            &bond_dims.read().unwrap(),
            latency_map,
        );

        assert_eq!(952f64, cost);
        assert_eq!(path![(1, 0), (1, 2)].to_vec(), communication_scheme);
    }
}
