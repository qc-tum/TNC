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
        d_tensors
    ))
}

#[cfg(test)]
mod tests {
    use crate::contractionpath::paths::{BranchBound, BranchBoundType, OptimizePath};
    use crate::io::open_hdf5;
    use crate::tensornetwork::contraction::tn_contract;
    use itertools::Itertools;
    use tetra::{Layout, Tensor as TetraTensor};
    use num_complex::Complex64;

    #[test]
    fn test_open_hdf5() {
        let (r_tn, d_tn) = open_hdf5("bell_circuit_tensornet.hdf5").unwrap();
        
        let mut opt = BranchBound::new(r_tn.clone(), None, 20, BranchBoundType::Flops);
        opt.optimize_path(None);
        let contract_path = opt.get_best_replace_path();
        let (r_tn, d_tn) = tn_contract(r_tn, d_tn, &contract_path);
        let mut tn_sol = TetraTensor::new_from_flat(&[2, 2, 2, 2],
            [0.7071067811865475, 0.7071067811865475, 0.0, 0.0,
            0.0, 0.0, 0.7071067811865475, 0.7071067811865475,
            0.0, 0.0, 0.7071067811865475, -0.7071067811865475, 
            0.7071067811865475, -0.7071067811865475, 0.0, 0.0,].iter().map(|&e| Complex64::new(e, 0.0)).collect_vec(),
            Some(Layout::RowMajor),
        );

        assert_eq!(tn_sol, d_tn[0]);
    }
}
