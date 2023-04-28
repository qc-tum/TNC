use crate::tensornetwork::tensor::Tensor;
use crate::tensornetwork::TensorNetwork;
use hdf5::{File, Result};
use num_complex::Complex64;
use std::collections::HashMap;
use std::iter::zip;
use tetra::Layout;
use tetra::Tensor as DataTensor;

pub fn open_hdf5(file: &str) -> Result<(TensorNetwork, Vec<DataTensor>)> {
    let file = File::open(file)?;
    let gr = file.group("/tensors")?;
    let tensor_names = gr.member_names()?;
    let mut t_tensors = Vec::new();
    let mut d_tensors = Vec::new();
    let mut bond_dims = HashMap::<usize, u64>::new();

    // Outuput tensor is always labelled as -1
    let out_tensor = gr.dataset("-1")?;
    let out_tensor_bids = out_tensor.attr("bids")?;
    let out_bond_ids = out_tensor_bids.read_1d::<usize>()?;

    for tensor_name in tensor_names {
        if tensor_name == "-1" {
            continue;
        }
        let tensor = gr.dataset(&tensor_name)?;

        let bond_ids = tensor.attr("bids").unwrap().read_1d::<usize>()?;
        let tensor_dataset = gr.dataset(&tensor_name).unwrap().read_dyn::<Complex64>()?;
        let tensor_shape = tensor_dataset.shape().to_vec();
        t_tensors.push(Tensor::new(bond_ids.to_vec()));
        let d_tensor = DataTensor::new_from_flat(
            &tensor_shape.iter().map(|&e| e as u32).collect::<Vec<u32>>(),
            tensor_dataset.into_raw_vec(),
            Some(Layout::RowMajor),
        );

        d_tensors.push(d_tensor);
        for (bond_id, bond_dim) in zip(bond_ids, tensor_shape) {
            bond_dims.entry(bond_id).or_insert(bond_dim as u64);
        }
    }

    Ok((
        TensorNetwork::new(t_tensors, bond_dims, Some(&out_bond_ids.to_vec())),
        d_tensors,
    ))
}

#[cfg(test)]
mod tests {
    use super::open_hdf5;
    use num_complex::Complex64;
    use std::collections::HashMap;

    #[test]
    fn test_open_hdf5() {
        let (r_tn, d_tn) = open_hdf5("tests/bell_circuit_tensornet.hdf5").unwrap();

        let ref_hadamard = vec![
            Complex64::new(0.7071067811865475, 0.0),
            Complex64::new(0.7071067811865475, 0.0),
            Complex64::new(0.7071067811865475, 0.0),
            Complex64::new(-0.7071067811865475, 0.0),
        ];

        let ref_cnot = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];
        assert_eq!(d_tn[0].get_raw_data().to_vec(), ref_hadamard);
        assert_eq!(d_tn[1].get_raw_data().to_vec(), ref_cnot);

        let tensors = r_tn.get_tensors();

        assert_eq!(tensors[0].get_legs(), &vec![2, 0]);
        assert_eq!(tensors[1].get_legs(), &vec![2, 3, 1]);

        let bond_dims = HashMap::from([(0, 2), (1, 2), (2, 2), (3, 2)]);
        assert_eq!(r_tn.get_bond_dims(), &bond_dims);

        let edges = HashMap::from([
            (2, vec![Some(0), Some(1), None]),
            (0, vec![Some(0), None]),
            (1, vec![Some(1), None]),
            (3, vec![Some(1), None]),
        ]);
        assert_eq!(r_tn.get_edges(), &edges);
    }
}
