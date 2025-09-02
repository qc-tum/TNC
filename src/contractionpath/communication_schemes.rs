use std::fmt;

use itertools::Itertools;
use rand::distributions::Uniform;
use rand::Rng;
use rustc_hash::FxHashMap;

use crate::contractionpath::contraction_cost::communication_path_cost;
use crate::contractionpath::paths::cotengrust::{Cotengrust, OptMethod};
use crate::contractionpath::paths::weighted_branchbound::WeightedBranchBound;
use crate::contractionpath::paths::{CostType, OptimizePath};
use crate::pair;
use crate::tensornetwork::partitioning::communication_partitioning;
use crate::tensornetwork::partitioning::partition_config::PartitioningStrategy;
use crate::tensornetwork::tensor::Tensor;
use crate::types::ContractionIndex;

#[derive(Debug, Copy, Clone)]
pub enum CommunicationScheme {
    /// Uses Greedy scheme to find contraction path for communication
    Greedy,
    /// Uses a randomized greedy approach
    RandomGreedy,
    /// Uses repeated bipartitioning to identify communication path
    Bipartition,
    /// Uses repeated bipartitioning to identify communication path
    BipartitionSweep,
    /// Uses a filtered search that considered time to intermediate tensor
    WeightedBranchBound,
    /// Uses a filtered search
    BranchBound,
}

impl fmt::Display for CommunicationScheme {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let comm_str = match self {
            CommunicationScheme::Greedy => "greedy",
            CommunicationScheme::RandomGreedy => "random_greedy",
            CommunicationScheme::Bipartition => "bipartition",
            CommunicationScheme::BipartitionSweep => "bipartition_sweep",
            CommunicationScheme::WeightedBranchBound => "weightedbranchbound",
            CommunicationScheme::BranchBound => "branchbound",
        };
        write!(f, "{}", comm_str)
    }
}

impl CommunicationScheme {
    pub(crate) fn communication_path<R>(
        &self,
        children_tensors: &[Tensor],
        latency_map: &FxHashMap<usize, f64>,
        rng: Option<&mut R>,
    ) -> Vec<ContractionIndex>
    where
        R: ?Sized + Rng,
    {
        match self {
            CommunicationScheme::Greedy => greedy(children_tensors, latency_map),
            CommunicationScheme::RandomGreedy => random_greedy(children_tensors),
            CommunicationScheme::Bipartition => bipartition(children_tensors, latency_map),
            CommunicationScheme::BipartitionSweep => {
                let Some(rng) = rng else {
                    panic!("BipartitionSweep requires a random number generator")
                };
                bipartition_sweep(children_tensors, latency_map, rng)
            }

            CommunicationScheme::WeightedBranchBound => {
                weighted_branchbound(children_tensors, latency_map)
            }
            CommunicationScheme::BranchBound => branchbound(children_tensors),
        }
    }
}

fn greedy(
    children_tensors: &[Tensor],
    _latency_map: &FxHashMap<usize, f64>,
) -> Vec<ContractionIndex> {
    let communication_tensors = Tensor::new_composite(children_tensors.to_vec());
    let mut opt = Cotengrust::new(&communication_tensors, OptMethod::Greedy);
    opt.optimize_path();
    opt.get_best_replace_path()
}

fn bipartition(
    children_tensors: &[Tensor],
    _latency_map: &FxHashMap<usize, f64>,
) -> Vec<ContractionIndex> {
    let children_tensors = children_tensors.iter().cloned().enumerate().collect_vec();
    let imbalance = 0.03;
    tensor_bipartition(&children_tensors, imbalance)
}

fn bipartition_sweep<R>(
    children_tensors: &[Tensor],
    latency_map: &FxHashMap<usize, f64>,
    rng: &mut R,
) -> Vec<ContractionIndex>
where
    R: ?Sized + Rng,
{
    let tensors = children_tensors.iter().cloned().enumerate().collect_vec();
    let mut best_flops = f64::INFINITY;
    let mut best_path = vec![];
    let partition_latencies = latency_map
        .iter()
        .sorted_by_key(|(k, _)| **k)
        .map(|(_, v)| *v)
        .collect::<Vec<_>>();
    for _ in 0..20 {
        let imbalance = rng.sample(Uniform::new(0.01, 0.5));
        let path = tensor_bipartition(&tensors, imbalance);
        let (flops, _) = communication_path_cost(
            children_tensors,
            &path,
            true,
            true,
            Some(&partition_latencies),
        );
        if flops < best_flops {
            best_flops = flops;
            best_path = path;
        }
    }
    best_path
}

// pub(crate) fn bipartition_k_sweep(
//     children_tensors: &[Tensor],
//     latency_map: &FxHashMap<usize, f64>,
// ) -> Vec<ContractionIndex> {
//     let children_tensors = children_tensors.iter().cloned().enumerate().collect_vec();
//     let imbalance = 0.03;
//     let k = children_tensors.len();

//     for i in 0..50 {
//         let imbalance = 0.01 * i as f64;
//         tensor_bipartition(&children_tensors, imbalance);
//     }
// }

fn weighted_branchbound(
    children_tensors: &[Tensor],
    latency_map: &FxHashMap<usize, f64>,
) -> Vec<ContractionIndex> {
    let communication_tensors = Tensor::new_composite(children_tensors.to_vec());

    let mut opt = WeightedBranchBound::new(
        &communication_tensors,
        Some(10),
        5.,
        latency_map.clone(),
        CostType::Flops,
    );
    opt.optimize_path();
    opt.get_best_replace_path()
}

fn branchbound(children_tensors: &[Tensor]) -> Vec<ContractionIndex> {
    let communication_tensors = Tensor::new_composite(children_tensors.to_vec());
    let latency_map = FxHashMap::from_iter((0..children_tensors.len()).map(|i| (i, 0.0)));

    let mut opt = WeightedBranchBound::new(
        &communication_tensors,
        Some(10),
        5.,
        latency_map,
        CostType::Flops,
    );
    opt.optimize_path();
    opt.get_best_replace_path()
}

/// Uses recursive bipartitioning to identify a communication scheme for final tensors
/// Returns root id of subtree, parallel contraction cost as f64, resultant tensor and prior contraction sequence
fn tensor_bipartition_recursive(
    children_tensor: &[(usize, Tensor)],
    imbalance: f64,
) -> (usize, Tensor, Vec<ContractionIndex>) {
    let k = 2;
    let min = true;

    // Composite tensor contracts with a single leaf tensor
    if children_tensor.len() == 1 {
        return (
            children_tensor[0].0,
            children_tensor[0].1.clone(),
            Vec::new(),
        );
    }

    // Only occurs when there is a subset of 2 tensors
    if children_tensor.len() == 2 {
        // Always ensure that the larger tensor size is on the left.
        let (t1, t2) = if children_tensor[1].1.size() > children_tensor[0].1.size() {
            (children_tensor[1].0, children_tensor[0].0)
        } else {
            (children_tensor[0].0, children_tensor[1].0)
        };
        let tensor = &children_tensor[0].1 ^ &children_tensor[1].1;

        return (t1, tensor, vec![pair!(t1, t2)]);
    }

    let partitioning = communication_partitioning(
        children_tensor,
        k,
        imbalance,
        PartitioningStrategy::MinCut,
        min,
    );

    let mut partition_iter = partitioning.iter();
    let (children_1, children_2): (Vec<_>, Vec<_>) = children_tensor
        .iter()
        .cloned()
        .partition(|_| partition_iter.next() == Some(&0));

    let (id_1, t1, mut contraction_1) = tensor_bipartition_recursive(&children_1, imbalance);

    let (id_2, t2, mut contraction_2) = tensor_bipartition_recursive(&children_2, imbalance);

    let tensor = &t1 ^ &t2;

    contraction_1.append(&mut contraction_2);
    let (id_1, id_2) = if t2.size() > t1.size() {
        (id_2, id_1)
    } else {
        (id_1, id_2)
    };

    contraction_1.push(pair!(id_1, id_2));
    (id_1, tensor, contraction_1)
}

/// Repeatedly bipartitions tensor network to obtain communication scheme
/// Assumes that all tensors contracted do so in parallel
fn tensor_bipartition(
    children_tensor: &[(usize, Tensor)],
    imbalance: f64,
) -> Vec<ContractionIndex> {
    let (_, _, contraction_path) = tensor_bipartition_recursive(children_tensor, imbalance);
    contraction_path
}

fn random_greedy(children_tensors: &[Tensor]) -> Vec<ContractionIndex> {
    let communication_tensors = Tensor::new_composite(children_tensors.to_vec());

    let mut opt = Cotengrust::new(&communication_tensors, OptMethod::RandomGreedy(100));
    opt.optimize_path();
    opt.get_best_replace_path()
}

#[cfg(test)]
mod tests {
    use super::*;

    use itertools::Itertools;
    use rustc_hash::FxHashMap;

    use crate::{
        contractionpath::contraction_cost::communication_path_cost, path,
        tensornetwork::tensor::Tensor,
    };

    fn setup_simple_partition_data() -> FxHashMap<usize, f64> {
        FxHashMap::from_iter([(0, 40.), (1, 30.), (2, 50.)])
    }

    /// Tensor ids in contraction tree included in variable name for easy tracking
    /// This example prioritizes contracting tensor1 & tensor 2 using the greedy cost function
    /// However, the partition cost of tensor 2 is very high, which makes contracting it later more attractive by reducing wait-time
    fn setup_simple() -> Vec<Tensor> {
        let bond_dims =
            FxHashMap::from_iter([(0, 2), (1, 2), (2, 2), (3, 2), (4, 2), (5, 2), (6, 2)]);

        let tensor0 = Tensor::new_from_map(vec![3, 4, 5], &bond_dims);
        let tensor1 = Tensor::new_from_map(vec![0, 1, 3, 4], &bond_dims);
        let tensor2 = Tensor::new_from_map(vec![0, 1, 2, 5, 6], &bond_dims);
        vec![tensor0, tensor1, tensor2]
    }

    #[test]
    fn test_greedy_communication() {
        let tensors = setup_simple();
        let latency_map = setup_simple_partition_data();
        let communication_scheme = greedy(&tensors, &latency_map);

        assert_eq!(&communication_scheme, path![(0, 1), (0, 2)]);
        let tensor_costs = (0..tensors.len()).map(|i| latency_map[&i]).collect_vec();
        let (flop_cost, mem_cost) = communication_path_cost(
            &tensors,
            &communication_scheme,
            true,
            true,
            Some(&tensor_costs),
        );
        assert_eq!(flop_cost, 104.);
        assert_eq!(mem_cost, 44.);
    }

    #[test]
    fn test_weighted_communication() {
        let tensors = setup_simple();
        let latency_map = setup_simple_partition_data();

        let communication_scheme = weighted_branchbound(&tensors, &latency_map);

        assert_eq!(&communication_scheme, path![(1, 0), (2, 1)]);
        // Flop Cost: (1, 0) = 32 , Tensor cost = 40, Total = 72
        // Flop Cost: (2, 1) = 32, Tensor cost = 50
        // max(72, 50) + 32 = 104
        // Mem Cost: (2, 1) = 2^3 + 2^5 + 2^2 = 44
        let tensor_costs = (0..tensors.len()).map(|i| latency_map[&i]).collect_vec();
        let (flop_cost, mem_cost) = communication_path_cost(
            &tensors,
            &communication_scheme,
            true,
            true,
            Some(&tensor_costs),
        );

        assert_eq!(flop_cost, 104.);
        assert_eq!(mem_cost, 44.);
    }

    #[test]
    fn test_bi_partition_communication() {
        let tensors = setup_simple();
        let latency_map = setup_simple_partition_data();

        let communication_scheme = bipartition(&tensors, &latency_map);

        assert_eq!(&communication_scheme, path![(2, 1), (2, 0)]);

        // Flop Cost: (2, 1) = 128, Tensor cost = 50, Total = 178
        // Flop Cost: (2, 0) = 32 , Tensor cost = 40
        // max(178, 40) + 32 = 210
        // Mem Cost: (2, 1) = 2^4 + 2^5 + 2^5 = 80
        let tensor_costs = (0..tensors.len()).map(|i| latency_map[&i]).collect_vec();
        let (flop_cost, mem_cost) = communication_path_cost(
            &tensors,
            &communication_scheme,
            true,
            true,
            Some(&tensor_costs),
        );

        assert_eq!(flop_cost, 210.);
        assert_eq!(mem_cost, 80.);
    }
}
