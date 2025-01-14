use itertools::Itertools;
use partition_config::PartitioningStrategy;
use rustc_hash::FxHashMap;
use std::iter::zip;

use super::tensor::Tensor;
use kahypar_sys::{partition, KaHyParContext};

pub mod partition_config;

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

    let imbalance: f64 = 0.03;
    let mut objective = 0;
    let mut hyperedge_weights = vec![];
    let mut hyperedge_indices = vec![0];
    let mut hyperedges = vec![];
    let bond_dims = tensor_network.bond_dims();

    let mut edge_dict = FxHashMap::default();
    for (tensor_id, tensor) in tensor_network.tensors().iter().enumerate() {
        for leg in tensor.legs() {
            edge_dict
                .entry(leg)
                .and_modify(|entry| {
                    hyperedges.push(*entry as u32);
                    hyperedges.push(tensor_id as u32);
                    hyperedge_indices.push(hyperedge_indices.last().unwrap() + 2);
                    hyperedge_weights.push(x * bond_dims[leg] as i32);
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
    partitioning
        .iter()
        .map(|e| *e as usize)
        .collect::<Vec<usize>>()
}

/// Repeatedly partitions a tensor network to identify a communication scheme.
/// Returns a `Vec<ContractionIndex>` of length equal to the number of input tensors minus one, acts as a communication scheme.
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
    bond_dims: &FxHashMap<usize, u64>,
    k: i32,
    partitioning_strategy: PartitioningStrategy,
    min: bool,
) -> Vec<usize> {
    assert!(k > 1, "Partitioning only valid for more than one process");
    let num_vertices = tensors.len() as u32;
    let mut context = KaHyParContext::new();
    partitioning_strategy.apply(&mut context);

    let x = if min { 1 } else { -1 };

    let imbalance: f64 = 0.03;
    let mut objective = 0;
    let mut hyperedge_weights = vec![];

    let mut hyperedge_indices = vec![0];
    let mut hyperedges = vec![];
    // Bidirectional mapping to a new index as KaHyPar indexes from 0.
    // let mut edge_to_virtual_edge = FxHashMap::default();
    // New index that starts from 0
    // let mut edge_count = 0;

    let mut edge_dict = FxHashMap::default();
    for (tensor_id, tensor) in tensors {
        for leg in tensor.legs() {
            edge_dict
                .entry(leg)
                .and_modify(|entry| {
                    hyperedges.push(*entry as u32);
                    hyperedges.push(*tensor_id as u32);
                    hyperedge_indices.push(hyperedge_indices.last().unwrap() + 2);
                    hyperedge_weights.push(x * bond_dims[leg] as i32);
                    *entry = *tensor_id;
                })
                .or_insert(*tensor_id);
        }
    }

    let mut intermediate_tensor = Tensor::default();

    let tensors = tensors
        .iter()
        .map(|(_, b)| b.clone())
        .collect::<Vec<Tensor>>();
    intermediate_tensor.push_tensors(tensors, Some(bond_dims));

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
    partitioning
        .iter()
        .map(|e| *e as usize)
        .collect::<Vec<usize>>()
}

pub fn partition_tensor_network(tn: &Tensor, partitioning: &[usize]) -> Tensor {
    let partition_ids = partitioning
        .iter()
        .unique()
        .copied()
        .collect::<Vec<usize>>();
    let partition_dict =
        zip(partition_ids.iter().copied(), 0..partition_ids.len()).collect::<FxHashMap<_, _>>();
    let mut partitions = vec![Tensor::default(); partition_ids.len()];

    for (partition_id, tensor) in zip(partitioning.iter(), tn.tensors.iter()) {
        partitions[partition_dict[partition_id]]
            .push_tensor(tensor.clone(), Some(&tensor.bond_dims()));
    }
    let mut partitioned_tn = Tensor::default();
    partitioned_tn.push_tensors(partitions, Some(&*tn.bond_dims()));
    partitioned_tn
}

#[cfg(test)]
mod tests {

    use rustc_hash::FxHashMap;

    use crate::tensornetwork::tensor::Tensor;
    use crate::tensornetwork::{
        create_tensor_network, partitioning::partition_config::PartitioningStrategy,
    };

    use super::{find_partitioning, partition_tensor_network};

    fn setup_complex() -> Tensor {
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
                (11, 17),
            ]),
        )
    }

    #[test]
    fn test_simple_partitioning() {
        let tn = setup_complex();
        let mut ref_tensor_1 = Tensor::default();
        let mut ref_tensor_2 = Tensor::default();
        let mut ref_tensor_3 = Tensor::default();
        ref_tensor_1.push_tensors(
            vec![Tensor::new(vec![6, 8, 9]), Tensor::new(vec![10, 8, 9])],
            Some(&tn.bond_dims()),
        );
        ref_tensor_2.push_tensors(
            vec![Tensor::new(vec![0, 1, 3, 2]), Tensor::new(vec![5, 1, 0])],
            Some(&tn.bond_dims()),
        );
        ref_tensor_3.push_tensors(
            vec![Tensor::new(vec![4, 3, 2]), Tensor::new(vec![4, 5, 6])],
            Some(&tn.bond_dims()),
        );
        let partitioning = find_partitioning(&tn, 3, PartitioningStrategy::MinCut, true);
        assert_eq!(partitioning, [2, 1, 2, 0, 0, 1]);
        let partitioned_tn = partition_tensor_network(&tn, partitioning.as_slice());
        assert_eq!(partitioned_tn.tensors().len(), 3);

        assert_eq!(partitioned_tn.tensors()[0].legs(), ref_tensor_3.legs());
        assert_eq!(partitioned_tn.tensors()[1].legs(), ref_tensor_2.legs());
        assert_eq!(partitioned_tn.tensors()[2].legs(), ref_tensor_1.legs());
    }
}
