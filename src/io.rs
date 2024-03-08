use std::{collections::HashMap, path::PathBuf};

use hdf5::{File, Result};
use num_complex::Complex64;

use tetra::Tensor as DataTensor;

use crate::tensornetwork::{tensor::Tensor, tensordata::TensorData};

pub fn load_tensor(filename: &PathBuf) -> Result<Tensor> {
    let file = File::open(PathBuf::from(filename))?;
    let gr = file.group("/tensors")?;
    let tensor_names = gr.member_names()?;
    // let mut bond_dims = HashMap::<usize, u64>::new();

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
        let mut bond_dims = HashMap::<usize, u64>::new();
        for (&bond_id, &bond_dim) in std::iter::zip(&bond_ids, &tensor_shape) {
            bond_dims.entry(bond_id).or_insert(bond_dim as u64);
        }

        let tensor_dataset = gr.dataset(&tensor_name).unwrap().read_dyn::<Complex64>()?;
        let mut new_tensor = Tensor::new(bond_ids.to_vec());
        new_tensor.set_tensor_data(TensorData::Matrix(DataTensor::new_from_flat(
            tensor_shape
                .into_iter()
                .map(|e| e as u32)
                .collect::<Vec<u32>>()
                .as_slice(),
            tensor_dataset.into_raw_vec(),
            None,
        )));
        new_tensor_network.push_tensor(new_tensor, Some(&bond_dims));
    }
    new_tensor_network.set_legs(out_bond_ids.to_vec());

    Ok(new_tensor_network)
}

pub fn load_data(filename: &PathBuf) -> Result<DataTensor> {
    let file = File::open(filename)?;
    let gr = file.group("/tensors")?;
    let tensor_name = gr.member_names()?;

    let tensor_dataset = gr
        .dataset(&tensor_name[0])
        .unwrap()
        .read_dyn::<Complex64>()?;
    let tensor_shape = tensor_dataset
        .shape()
        .to_vec()
        .iter()
        .map(|e| *e as u32)
        .collect::<Vec<u32>>();
    Ok(DataTensor::new_from_flat(
        tensor_shape.as_slice(),
        tensor_dataset.into_raw_vec(),
        None,
    ))
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashMap,
        fs,
        iter::zip,
        panic,
        path::{Path, PathBuf},
    };

    use float_cmp::assert_approx_eq;
    use hdf5::{AttributeBuilder, Error, File};
    use ndarray::array;
    use num_complex::Complex64;

    use crate::{
        io::{load_data, load_tensor},
        tensornetwork::{tensor::Tensor, tensordata::TensorData},
    };
    static TENSOR_TEST_FILE: &str = "./tests/tensor_test.hdf5";
    static DATA_TEST_FILE: &str = "./tests/data_test.hdf5";

    fn write_hdf5_tensor() -> Result<File, Error> {
        let new_file = File::create(TENSOR_TEST_FILE)?;
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

    fn write_hdf5_data() -> Result<File, Error> {
        let new_file = File::create(DATA_TEST_FILE)?;
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

    fn run_test<T>(test: T)
    where
        T: FnOnce() + panic::UnwindSafe,
    {
        let result = panic::catch_unwind(test);

        assert!(result.is_ok())
    }

    #[test]
    fn test_load_data() {
        run_test(|| {
            load_data_test();
        });
        if Path::new(DATA_TEST_FILE).exists() {
            fs::remove_file(DATA_TEST_FILE).expect("could not remove file");
        }
    }

    #[test]
    fn test_load_tensor() {
        run_test(|| {
            load_tensor_test();
        });
        if Path::new(TENSOR_TEST_FILE).exists() {
            fs::remove_file(TENSOR_TEST_FILE).expect("could not remove file");
        }
    }

    fn load_data_test() {
        let _ = write_hdf5_data();
        let tensor_data = load_data(&PathBuf::from(DATA_TEST_FILE)).unwrap();
        let ref_data = array![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 2.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(0.0, 1.0),
        ];
        for (u, v) in zip(ref_data.iter(), tensor_data.get_raw_data().iter()) {
            assert_approx_eq!(f64, u.re, v.re, epsilon = 1e-8);
            assert_approx_eq!(f64, u.im, v.im, epsilon = 1e-8);
        }
    }

    fn load_tensor_test() {
        let _ = write_hdf5_tensor();
        let mut ref_tn = Tensor::default();
        let mut ref_tensor = Tensor::new(vec![0, 1]);

        ref_tensor.set_tensor_data(TensorData::new_from_data(
            vec![2, 2],
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 2.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(0.0, 1.0),
            ],
            None,
        ));
        ref_tn.push_tensor(ref_tensor, Some(&HashMap::from([(0, 2), (1, 2)])));
        ref_tn.set_legs(vec![0, 1]);
        let tensor = load_tensor(&PathBuf::from(TENSOR_TEST_FILE)).unwrap();
        assert!(tensor.approx_eq(&ref_tn, 1e-12));
    }
}
