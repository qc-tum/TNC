use super::TensorNetwork;
use kahypar_sys;
use kahypar_sys::{partition, KaHyParContext};
use std::ffi::CString;

/// Partitions input tensor network using KaHyPar library.
///
/// # Arguments
///
/// * `partitioning` - &mut [i32] to store final partitioning results
/// * `tn` - [`TensorNetwork`] to be partitionined
/// * `k` - imbalance parameter for KaHyPar
/// * `config_file` - KaHyPar config file name
pub fn partition_tn(partitioning: &mut [i32], tn: &mut TensorNetwork, k: i32, config_file: String) {
    let config_file = CString::new(config_file).unwrap();
    let num_vertices = tn.get_tensors().len() as u32;
    assert!(partitioning.len() == num_vertices as usize);
    let num_hyperedges = tn.get_edges().len() as u32;
    let mut context = KaHyParContext::new();
    context.configure(config_file);

    let imbalance: f64 = 0.03;
    let mut objective = 0;
    let mut hyperedge_weights = vec![];
    let mut hyperedge_indices = vec![0];
    let mut hyperedges = vec![];
    let bond_dims = tn.get_bond_dims();
    for (edges, tensor_ids) in tn.get_edges().clone() {
        hyperedge_weights.push(bond_dims[&edges] as i32);
        let mut length = 0;
        for id in tensor_ids {
            if id.is_some() {
                hyperedges.push(id.unwrap() as u32);
                length += 1;
            }
        }
        hyperedge_indices.push(hyperedge_indices.last().unwrap() + length);
    }

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
        partitioning,
    );
    tn.partitioning = partitioning.to_vec();
}

#[cfg(test)]
mod tests {
    use std::env;
    use std::path::PathBuf;

    use crate::tensornetwork::tensor::Tensor;
    use crate::tensornetwork::TensorNetwork;

    use super::partition_tn;

    fn get_current_working_dir() -> std::io::Result<PathBuf> {
        env::current_dir()
    }

    fn setup_complex() -> TensorNetwork {
        TensorNetwork::from_vector(
            vec![
                Tensor::new(vec![4, 3, 2]),
                Tensor::new(vec![0, 1, 3, 2]),
                Tensor::new(vec![4, 5, 6]),
                Tensor::new(vec![6, 8, 9]),
                Tensor::new(vec![10, 8, 9]),
                Tensor::new(vec![5, 1, 0]),
            ],
            vec![27, 18, 12, 15, 5, 3, 18, 22, 45, 65, 5, 17],
            None,
        )
    }

    #[test]
    fn test_simple_partitioning() {
        let mut tn = setup_complex();
        let mut partitioning: Vec<i32> = vec![-1; tn.get_tensors().len() as usize];
        partition_tn(&mut partitioning, &mut tn, 2, String::from("test/km1"));
        assert_eq!(tn.partitioning, [1, 1, 0, 0, 0, 1]);
    }
}
