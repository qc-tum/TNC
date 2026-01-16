use std::iter::zip;

use itertools::Itertools;
use kahypar::{partition, KaHyParContext};
use rustc_hash::FxHashMap;

use crate::tensornetwork::partitioning::partition_config::PartitioningStrategy;
use crate::tensornetwork::tensor::Tensor;

pub mod partition_config;

/// The scale factor for the log weights used in the partitioning.
/// This is used to convert the log weights to integer weights for KaHyPar.
/// The current value allows for bond dimensions up to 144497 before two consecutive
/// bond dimensions are rounded to the same weight. At the same time, it allows for
/// the sum of ~150000 such bond dims before an i32 overflow occurs.
const LOG_SCALE_FACTOR: f64 = 1e5;

/// Partitions input tensor network using `KaHyPar` library.
///
/// Returns a `Vec<usize>` of length equal to the number of input tensors storing final partitioning results.
/// The usize associated with each Tensor indicates its partionining.
///
/// # Arguments
///
/// * `tensor_network` - [`Tensor`] to be partitionined
/// * `k` - imbalance parameter for `KaHyPar`
/// * `partition_strategy` - The strategy to pass to `KaHyPar`
/// * `min` - if `true` performs `min_cut` to partition tensor network, if `false`, uses `max_cut`
///
pub fn find_partitioning(
    tensor_network: &Tensor,
    k: i32,
    partitioning_strategy: PartitioningStrategy,
    min: bool,
) -> Vec<usize> {
    if k == 1 {
        return vec![0; tensor_network.tensors().len()];
    }

    let num_vertices = tensor_network.tensors().len() as u32;
    let mut context = KaHyParContext::new();
    partitioning_strategy.apply(&mut context);

    let x = if min { 1 } else { -1 };

    let imbalance = 0.03;
    let mut objective = 0;
    let mut hyperedge_weights = vec![];
    let mut hyperedge_indices = vec![0];
    let mut hyperedges = vec![];

    let mut edge_dict = FxHashMap::default();
    for (tensor_id, tensor) in tensor_network.tensors().iter().enumerate() {
        for (leg, dim) in tensor.edges() {
            edge_dict
                .entry(leg)
                .and_modify(|entry| {
                    hyperedges.push(*entry as u32);
                    hyperedges.push(tensor_id as u32);
                    hyperedge_indices.push(hyperedge_indices.last().unwrap() + 2);
                    // Use log weights, because KaHyPar minimizes the sum of weights while we need the product.
                    // Since it accepts only integer weights, we scale the log values before rounding.
                    let weight = LOG_SCALE_FACTOR * (*dim as f64).log2();
                    hyperedge_weights.push(x * weight as i32);
                    *entry = tensor_id;
                })
                .or_insert(tensor_id);
        }
    }

    let mut partitioning = vec![-1; num_vertices as usize];
    partition(
        num_vertices,
        hyperedge_weights.len() as u32,
        imbalance,
        k,
        None,
        Some(hyperedge_weights),
        &hyperedge_indices,
        hyperedges.as_slice(),
        &mut objective,
        &mut context,
        &mut partitioning,
    );
    partitioning.iter().map(|e| *e as usize).collect()
}

/// Repeatedly partitions a tensor network to identify a communication scheme.
/// Returns a `Vec<usize>` of length equal to the number of input tensors minus one, acts as a communication scheme.
///
/// # Arguments
///
/// * `tensors` - &[(usize, `Tensor`)] to be partitioned. each tuple contains the intermediate contraction cost and intermediate tensor for communication.
/// * `bond_dims` - bond_dims for tensors
/// * `k` - number of partitions
/// * `partitioning_strategy` - The strategy to pass to `KaHyPar`
/// * `min` - if `true` performs `min_cut` to partition tensor network, if `false`, uses `max_cut`
///
pub fn communication_partitioning(
    tensors: &[(usize, Tensor)],
    k: i32,
    imbalance: f64,
    partitioning_strategy: PartitioningStrategy,
    min: bool,
) -> Vec<usize> {
    assert!(k > 1, "Partitioning only valid for more than one process");
    let num_vertices = tensors.len() as u32;
    let mut context = KaHyParContext::new();
    partitioning_strategy.apply(&mut context);

    let x = if min { 1 } else { -1 };

    let mut objective = 0;
    let mut hyperedge_weights = vec![];

    let mut hyperedge_indices = vec![0];
    let mut hyperedges = vec![];

    // Bidirectional mapping to a new index as KaHyPar indexes from 0.
    // let mut edge_to_virtual_edge = FxHashMap::default();
    // New index that starts from 0
    // let mut edge_count = 0;
    let mut edge_dict = FxHashMap::default();
    for (tensor_id, (_, tensor)) in tensors.iter().enumerate() {
        for (leg, dim) in tensor.edges() {
            edge_dict
                .entry(leg)
                .and_modify(|entry| {
                    hyperedges.push(*entry as u32);
                    hyperedges.push(tensor_id as u32);
                    hyperedge_indices.push(hyperedge_indices.last().unwrap() + 2);
                    // Use log weights, because KaHyPar minimizes the sum of weights while we need the product.
                    // Since it accepts only integer weights, we scale the log values before rounding.
                    let weight = LOG_SCALE_FACTOR * (*dim as f64).log2();
                    hyperedge_weights.push(x * weight as i32);
                    *entry = tensor_id;
                })
                .or_insert(tensor_id);
        }
    }

    let mut partitioning = vec![-1; num_vertices as usize];
    partition(
        num_vertices,
        hyperedge_weights.len() as u32,
        imbalance,
        k,
        None,
        Some(hyperedge_weights),
        &hyperedge_indices,
        hyperedges.as_slice(),
        &mut objective,
        &mut context,
        &mut partitioning,
    );

    // partitioning
    partitioning.iter().map(|e| *e as usize).collect()
}

/// Partitions the tensor network based on the `partitioning` vector that assigns
/// each vector to a partition.
pub fn partition_tensor_network(tn: Tensor, partitioning: &[usize]) -> Tensor {
    let partition_ids = partitioning.iter().unique().copied().collect_vec();
    let partition_dict =
        zip(partition_ids.iter().copied(), 0..partition_ids.len()).collect::<FxHashMap<_, _>>();

    let mut partitions = vec![Tensor::default(); partition_ids.len()];
    for (partition_id, tensor) in zip(partitioning, tn.tensors) {
        partitions[partition_dict[partition_id]].push_tensor(tensor);
    }
    Tensor::new_composite(partitions)
}

#[cfg(test)]
mod tests {
    use super::*;

    use float_cmp::assert_approx_eq;
    use rustc_hash::FxHashMap;

    use crate::tensornetwork::partitioning::partition_config::PartitioningStrategy;
    use crate::tensornetwork::tensor::{EdgeIndex, Tensor};

    fn setup_complex() -> (Tensor, FxHashMap<EdgeIndex, u64>) {
        let bond_dims = FxHashMap::from_iter([
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
        ]);
        (
            Tensor::new_composite(vec![
                Tensor::new_from_map(vec![4, 3, 2], &bond_dims),
                Tensor::new_from_map(vec![0, 1, 3, 2], &bond_dims),
                Tensor::new_from_map(vec![4, 5, 6], &bond_dims),
                Tensor::new_from_map(vec![6, 8, 9], &bond_dims),
                Tensor::new_from_map(vec![10, 8, 9], &bond_dims),
                Tensor::new_from_map(vec![5, 1, 0], &bond_dims),
            ]),
            bond_dims,
        )
    }

    #[test]
    fn test_simple_partitioning() {
        let (tn, bond_dims) = setup_complex();
        let ref_tensor_1 = Tensor::new_composite(vec![
            Tensor::new_from_map(vec![6, 8, 9], &bond_dims),
            Tensor::new_from_map(vec![10, 8, 9], &bond_dims),
        ]);
        let ref_tensor_2 = Tensor::new_composite(vec![
            Tensor::new_from_map(vec![0, 1, 3, 2], &bond_dims),
            Tensor::new_from_map(vec![5, 1, 0], &bond_dims),
        ]);
        let ref_tensor_3 = Tensor::new_composite(vec![
            Tensor::new_from_map(vec![4, 3, 2], &bond_dims),
            Tensor::new_from_map(vec![4, 5, 6], &bond_dims),
        ]);
        let partitioning = find_partitioning(&tn, 3, PartitioningStrategy::MinCut, true);
        assert_eq!(partitioning, [2, 1, 2, 0, 0, 1]);
        let partitioned_tn = partition_tensor_network(tn, partitioning.as_slice());
        assert_eq!(partitioned_tn.tensors().len(), 3);

        assert_approx_eq!(&Tensor, partitioned_tn.tensor(2), &ref_tensor_1);
        assert_approx_eq!(&Tensor, partitioned_tn.tensor(1), &ref_tensor_2);
        assert_approx_eq!(&Tensor, partitioned_tn.tensor(0), &ref_tensor_3);
    }

    #[test]
    fn test_single_partition() {
        let (tn, _) = setup_complex();
        let partitioning = find_partitioning(&tn, 1, PartitioningStrategy::MinCut, true);
        assert_eq!(partitioning, [0, 0, 0, 0, 0, 0]);
    }
}
