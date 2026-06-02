use std::path::PathBuf;

use approx::AbsDiffEq;
use ndarray::ArrayD;
use num_complex::Complex64;
use serde::{Deserialize, Serialize};

use crate::{
    gates::{load_gate, load_gate_adjoint, matrix_adjoint_inplace},
    io::hdf5::load_data,
};

pub type DataTensor = ArrayD<Complex64>;

/// The data of a tensor.
#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TensorData {
    /// No data.
    #[default]
    None,
    /// The data is loaded from a HDF5 file.
    File((PathBuf, bool)),
    /// A quantum gate. The name must be registered in the gates module.
    Gate((String, Vec<f64>, bool)),
    /// A raw vec of complex numbers.
    Matrix(DataTensor),
}

impl TensorData {
    /// Creates a new tensor from raw (flat) data.
    #[must_use]
    pub fn new_from_data(dimensions: &[usize], data: Vec<Complex64>) -> Self {
        Self::Matrix(ArrayD::from_shape_vec(dimensions, data).unwrap())
    }

    /// Consumes the tensor data and returns the contained tensor.
    pub fn into_data(self) -> DataTensor {
        match self {
            TensorData::None => panic!("Cannot convert uncontracted tensor to data"),
            TensorData::File((filename, adjoint)) => {
                let mut data = load_data(filename).unwrap();
                if adjoint {
                    matrix_adjoint_inplace(&mut data);
                }
                data
            }
            TensorData::Gate((gatename, angles, adjoint)) => {
                if adjoint {
                    load_gate_adjoint(&gatename, &angles)
                } else {
                    load_gate(&gatename, &angles)
                }
            }
            TensorData::Matrix(tensor) => tensor,
        }
    }

    /// Returns the adjoint of this data.
    pub fn adjoint(self) -> Self {
        match self {
            TensorData::None => TensorData::None,
            TensorData::File((filename, adjoint)) => TensorData::File((filename, !adjoint)),
            TensorData::Gate((name, params, adjoint)) => TensorData::Gate((name, params, !adjoint)),
            TensorData::Matrix(mut tensor) => {
                matrix_adjoint_inplace(&mut tensor);
                TensorData::Matrix(tensor)
            }
        }
    }
}

impl AbsDiffEq for TensorData {
    type Epsilon = f64;

    fn default_epsilon() -> Self::Epsilon {
        f64::EPSILON
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        match (self, other) {
            (TensorData::File(l0), TensorData::File(r0)) => l0 == r0,
            (
                TensorData::Gate((name_l, angles_l, adjoint_l)),
                TensorData::Gate((name_r, angles_r, adjoint_r)),
            ) => {
                name_l == name_r
                    && adjoint_l == adjoint_r
                    && angles_l
                        .iter()
                        .zip(angles_r)
                        .all(|(l, r)| f64::abs_diff_eq(l, r, epsilon))
            }
            (TensorData::Matrix(l0), TensorData::Matrix(r0)) => {
                DataTensor::abs_diff_eq(l0, r0, epsilon)
            }
            (TensorData::None, TensorData::None) => true,
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;

    #[test]
    #[should_panic(expected = "assert_abs_diff_eq!")]
    fn gates_eq_different_name() {
        let g1 = TensorData::Gate((String::from("cx"), vec![], false));
        let g2 = TensorData::Gate((String::from("CX"), vec![], false));
        assert_abs_diff_eq!(&g1, &g2);
    }

    #[test]
    #[should_panic(expected = "assert_abs_diff_eq!")]
    fn gates_eq_adjoint() {
        let g1 = TensorData::Gate((String::from("h"), vec![], false));
        let g2 = TensorData::Gate((String::from("h"), vec![], true));
        assert_abs_diff_eq!(&g1, &g2);
    }

    #[test]
    #[should_panic(expected = "assert_abs_diff_eq!")]
    fn gates_eq_different_angles() {
        let g1 = TensorData::Gate((String::from("u"), vec![1.4, 2.0, -3.0], false));
        let g2 = TensorData::Gate((String::from("u"), vec![1.4, -2.0, -3.0], false));
        assert_abs_diff_eq!(&g1, &g2);
    }

    #[test]
    #[should_panic(expected = "assert_abs_diff_eq!")]
    fn eq_different_data() {
        let g1 = TensorData::Gate((String::from("u"), vec![1.4, 2.0, -3.0], false));
        let g2 = TensorData::new_from_data(&[], vec![Complex64::ONE]);
        assert_abs_diff_eq!(&g1, &g2);
    }
}
