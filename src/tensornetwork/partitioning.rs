use itertools::Itertools;
use std::collections::HashMap;
use std::ffi::CString;
use std::iter::zip;

use super::tensor::Tensor;
use crate::types::Vertex;

use kahypar_sys;
use kahypar_sys::{partition, KaHyParContext};

pub fn find_partitioning(tn: &Tensor, k: i32, config_file: String, min: bool) -> Vec<usize> {
    let config_file = CString::new(config_file).unwrap();
    let num_vertices = tn.get_tensors().len() as u32;
    let mut num_hyperedges = 0;
    let mut context = KaHyParContext::new();
    context.configure(config_file);

    let x = if min { 1 } else { -1 };

    let imbalance: f64 = 0.03;
    let mut objective = 0;
    let mut hyperedge_weights = vec![];
    let mut hyperedge_indices = vec![0];
    let mut hyperedges = vec![];
    let bond_dims = tn.get_bond_dims();
    for (edges, tensor_ids) in tn.get_edges() {
        let mut length = 0;
        // Don't add edges that connect only one vertex
        if tensor_ids.len() == 2 && tensor_ids.contains(&Vertex::Open) {
            continue;
        }
        hyperedge_weights.push(x * bond_dims[&edges] as i32);

        for id in tensor_ids {
            match id {
                Vertex::Closed(id) => {
                    hyperedges.push(*id as u32);
                    length += 1;
                }
                Vertex::Open => continue,
            }
        }
        hyperedge_indices.push(hyperedge_indices.last().unwrap() + length);
        num_hyperedges += 1;
    }

    let mut partitioning = vec![-1; num_vertices as usize];
    partition(
        num_vertices,
        num_hyperedges,
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

pub fn partition_tensor_network(tn: &Tensor, partitioning: &[usize]) -> Tensor {
    let partition_ids = partitioning
        .iter()
        .cloned()
        .unique()
        .collect::<Vec<usize>>();
    let partition_dict = HashMap::<usize, usize>::from_iter(zip(
        partition_ids.iter().cloned(),
        0..partition_ids.len(),
    ));
    let mut partitions = vec![Tensor::default(); partition_ids.len()];

    for (partition_id, tensor) in zip(partitioning.iter(), tn.tensors.iter()) {
        partitions[partition_dict[partition_id]].push_tensor(
            tensor.clone(),
            Some(&tensor.get_bond_dims()),
            None,
        );
    }
    let mut partitioned_tn = Tensor::default();
    partitioned_tn.push_tensors(partitions, Some(&*tn.get_bond_dims()), None);
    partitioned_tn
}

#[cfg(test)]
mod tests {
    use std::env;
    use std::path::PathBuf;

    use crate::tensornetwork::create_tensor_network;
    use crate::tensornetwork::tensor::Tensor;

    use super::{find_partitioning, partition_tensor_network};

    fn get_current_working_dir() -> std::io::Result<PathBuf> {
        env::current_dir()
    }

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
            Some(&tn.get_bond_dims()),
            None,
        );
        ref_tensor_2.push_tensors(
            vec![Tensor::new(vec![0, 1, 3, 2]), Tensor::new(vec![5, 1, 0])],
            Some(&tn.get_bond_dims()),
            None,
        );
        ref_tensor_3.push_tensors(
            vec![Tensor::new(vec![4, 3, 2]), Tensor::new(vec![4, 5, 6])],
            Some(&tn.get_bond_dims()),
            None,
        );
        let partitioning = find_partitioning(&tn, 3, String::from("tests/km1"), true);
        assert_eq!(partitioning, [2, 1, 2, 0, 0, 1]);
        let partitioned_tn = partition_tensor_network(&tn, partitioning.as_slice());
        assert_eq!(partitioned_tn.get_tensors().len(), 3);

        assert_eq!(partitioned_tn.get_tensors()[0], ref_tensor_3);
        assert_eq!(partitioned_tn.get_tensors()[1], ref_tensor_2);
        assert_eq!(partitioned_tn.get_tensors()[2], ref_tensor_1);
    }
}
