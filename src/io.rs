use crate::tensornetwork::tensor::Tensor;
use crate::tensornetwork::TensorNetwork;
use hdf5::{File, Result};
use num_complex::Complex64;
use std::collections::HashMap;
use std::iter::zip;
use tetra::Layout;
use tetra::Tensor as TetraTensor;

pub fn open_hdf5(file: &str) -> Result<(TensorNetwork, Vec<TetraTensor>)> {
    let file = File::open(file)?;
    let gr = file.group("/tensors")?;
    let tensor_names = gr.member_names()?;
    let mut t_tensors = Vec::new();
    let mut d_tensors = Vec::new();
    let mut bond_dims = HashMap::<usize, u64>::new();

    let out_tensor = gr.dataset("-1")?;
    let out_bond_ids = out_tensor.attr("bids").unwrap().read_1d::<usize>()?;
    for tensor_name in tensor_names {
        if tensor_name == "-1" {
            continue;
        }
        let tensor = gr.dataset(&tensor_name)?;

        let bond_ids = tensor.attr("bids").unwrap().read_1d::<usize>()?;
        let tensor_dataset = gr.dataset(&tensor_name).unwrap().read_dyn::<Complex64>()?;
        let tensor_shape = tensor_dataset.shape().to_vec();
        t_tensors.push(Tensor::new(bond_ids.to_vec()));
        let d_tensor = TetraTensor::new_from_flat(
            &tensor_shape.iter().map(|&e| e as i32).collect::<Vec<i32>>(),
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
