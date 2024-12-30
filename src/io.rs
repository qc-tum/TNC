use std::path::Path;

use hdf5::{File, Result};
use itertools::Itertools;
use ndarray::Array;
use num_complex::Complex64;

use rustc_hash::FxHashMap;
use tetra::Tensor as DataTensor;

use crate::tensornetwork::{tensor::Tensor, tensordata::TensorData};

/// Loads a tensor network from a HDF5 file.
pub fn load_tensor<P>(filename: P) -> Result<Tensor>
where
    P: AsRef<Path>,
{
    let file = File::open(filename)?;
    read_tensor(&file)
}

/// Loads a single tensor from a HDF5 file.
pub fn load_data<P>(filename: P) -> Result<DataTensor>
where
    P: AsRef<Path>,
{
    let file = File::open(filename)?;
    read_data(&file)
}

/// Stores a single tensor in a HDF5 file.
pub fn store_data<P>(filename: P, tensor: &DataTensor) -> Result<()>
where
    P: AsRef<Path>,
{
    let file = File::create(filename)?;
    write_data(&file, tensor)
}

fn read_tensor(file: &File) -> Result<Tensor> {
    let gr = file.group("/tensors")?;
    let tensor_names = gr.member_names()?;

    // Outuput tensor is always labelled as -1
    let out_tensor = gr.dataset("-1")?;
    let out_tensor_bids = out_tensor.attr("bids")?;
    let out_bond_ids = out_tensor_bids.read_1d::<usize>()?;

    let mut new_tensor_network = Tensor::default();

    for tensor_name in tensor_names {
        if tensor_name == "-1" {
            continue;
        }
        let tensor = gr.dataset(&tensor_name)?;
        let bond_ids = tensor.attr("bids").unwrap().read_1d::<usize>()?;
        let tensor_dataset = gr.dataset(&tensor_name).unwrap().read_dyn::<Complex64>()?;
        let tensor_shape = tensor_dataset.shape().to_vec();
        let mut bond_dims = FxHashMap::default();
        for (&bond_id, &bond_dim) in std::iter::zip(&bond_ids, &tensor_shape) {
            bond_dims.entry(bond_id).or_insert(bond_dim as u64);
        }

        let mut new_tensor = Tensor::new(bond_ids.to_vec());
        new_tensor.set_tensor_data(TensorData::Matrix(DataTensor::new_from_flat(
            &tensor_shape,
            tensor_dataset.into_raw_vec(),
            None,
        )));
        new_tensor_network.push_tensor(new_tensor, Some(&bond_dims));
    }
    new_tensor_network.set_legs(out_bond_ids.to_vec());

    Ok(new_tensor_network)
}

fn read_data(file: &File) -> Result<DataTensor> {
    let gr = file.group("/tensors")?;
    let tensor_name = gr.member_names()?;

    let tensor_dataset = gr
        .dataset(&tensor_name[0])
        .unwrap()
        .read_dyn::<Complex64>()?;
    let tensor_shape = tensor_dataset.shape().to_vec();
    Ok(DataTensor::new_from_flat(
        &tensor_shape,
        tensor_dataset.into_raw_vec(),
        None,
    ))
}

fn write_data(file: &File, tensor: &DataTensor) -> Result<()> {
    let gr = file.create_group("/tensors")?;
    let data = tensor.elements().into_owned();
    let data = Array::from(data);
    let shape = tensor
        .shape()
        .into_iter()
        .map(|e| TryInto::<usize>::try_into(e).unwrap())
        .collect_vec();
    let data = data.into_shape(shape)?;
    let tensor_dataset = gr.new_dataset_builder().with_data(&data);
    tensor_dataset.create("-1")?;
    file.flush()
}

#[cfg(test)]
mod tests {
    use std::iter::zip;

    use float_cmp::assert_approx_eq;
    use hdf5::{AttributeBuilder, File, Result};
    use ndarray::array;
    use num_complex::Complex64;
    use rand::{
        distributions::{Alphanumeric, DistString},
        thread_rng,
    };
    use rustc_hash::FxHashMap;

    use crate::tensornetwork::{tensor::Tensor, tensordata::TensorData};

    use tetra::{all_close, Tensor as DataTensor};

    use super::{read_data, read_tensor, write_data};

    /// Creates a new HDF5 file in memory.
    /// This method is taken from the hdf5 crate integration tests:
    /// <https://github.com/aldanor/hdf5-rust/blob/694e900972fbf5ffbdd1a2294f57a2cc3a91c994/hdf5/tests/common/util.rs#L7>.
    fn new_in_memory_file() -> Result<File> {
        let random_filename = Alphanumeric.sample_string(&mut thread_rng(), 8);
        File::with_options()
            .with_access_plist(|p| p.core_filebacked(false))
            .create(random_filename)
    }

    fn create_hdf5_tensor() -> Result<File> {
        let new_file = new_in_memory_file()?;
        let tensor_group = new_file.create_group("./tensors")?;
        let dataset_builder = tensor_group.new_dataset_builder();
        let dataset = dataset_builder.empty::<Complex64>().create("-1")?;
        let attribute = AttributeBuilder::new(&dataset);
        let bid = array![0, 1];
        let attribute = attribute.with_data(&bid);
        attribute.create("bids")?;

        let data = array![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 2.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(0.0, 1.0),
        ];
        let data = data.into_shape((2, 2))?;
        let dataset_builder2 = tensor_group.new_dataset_builder();
        let dataset_data_builder2 = dataset_builder2.with_data(&data);
        let dataset2 = dataset_data_builder2.create("0")?;
        let attribute2 = AttributeBuilder::new(&dataset2);
        let bid2 = array![0, 1];
        let attribute2 = attribute2.with_data(&bid2);
        attribute2.create("bids")?;

        new_file.flush()?;
        Ok(new_file)
    }

    fn create_hdf5_data() -> Result<File> {
        let new_file = new_in_memory_file()?;
        let tensor_group = new_file.create_group("./tensors")?;
        let dataset_builder = tensor_group.new_dataset_builder();
        let data = array![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 2.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(0.0, 1.0),
        ];
        let data = data.into_shape((2, 2))?;
        let dataset_data_builder = dataset_builder.with_data(&data);
        dataset_data_builder.create("-1")?;
        new_file.flush()?;
        Ok(new_file)
    }

    #[test]
    fn test_load_data() {
        let file = create_hdf5_data().unwrap();
        let tensor_data = read_data(&file).unwrap();

        let ref_data = array![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 2.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(0.0, 1.0),
        ];
        for (u, v) in zip(ref_data.iter(), tensor_data.elements().iter()) {
            assert_approx_eq!(f64, u.re, v.re, epsilon = 1e-8);
            assert_approx_eq!(f64, u.im, v.im, epsilon = 1e-8);
        }
    }

    #[test]
    fn test_load_tensor() {
        let file = create_hdf5_tensor().unwrap();
        let tensor = read_tensor(&file).unwrap();

        let mut ref_tn = Tensor::default();
        let mut ref_tensor = Tensor::new(vec![0, 1]);
        ref_tensor.set_tensor_data(TensorData::new_from_data(
            &[2, 2],
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 2.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(0.0, 1.0),
            ],
            None,
        ));
        ref_tn.push_tensor(ref_tensor, Some(&FxHashMap::from_iter([(0, 2), (1, 2)])));
        ref_tn.set_legs(vec![0, 1]);
        assert!(tensor.approx_eq(&ref_tn, 1e-12));
    }

    #[test]
    fn test_write_read() {
        let file = new_in_memory_file().unwrap();
        let data = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, -2.0),
            Complex64::new(-3.0, 0.0),
            Complex64::new(-2.0, -1.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.5, 2.0),
        ];
        let tensor = DataTensor::new_from_flat(&[2, 3], data, None);

        write_data(&file, &tensor).unwrap();
        let read = read_data(&file).unwrap();

        assert!(all_close(&tensor, &read, 1e-10));
    }
}
