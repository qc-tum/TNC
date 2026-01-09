use std::path::PathBuf;

use float_cmp::{ApproxEq, F64Margin};
use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use tetra::{Layout, Tensor as DataTensor};

use crate::{
    gates::{load_gate, load_gate_adjoint, matrix_adjoint_inplace},
    hdf5::load_data,
};

/// The data of a tensor.
#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub enum TensorData {
    /// This is for composite tensors that have not been contracted yet, as well as
    /// empty tensors in general.
    #[default]
    Uncontracted,
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
    pub fn new_from_data(
        dimensions: &[usize],
        data: Vec<Complex64>,
        layout: Option<Layout>,
    ) -> Self {
        Self::Matrix(DataTensor::new_from_flat(dimensions, data, layout))
    }

    /// Consumes the tensor data and returns the contained tensor.
    pub fn into_data(self) -> DataTensor {
        match self {
            TensorData::Uncontracted => panic!("Cannot convert uncontracted tensor to data"),
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
            TensorData::Uncontracted => TensorData::Uncontracted,
            TensorData::File((filename, adjoint)) => TensorData::File((filename, !adjoint)),
            TensorData::Gate((name, params, adjoint)) => TensorData::Gate((name, params, !adjoint)),
            TensorData::Matrix(mut tensor) => {
                matrix_adjoint_inplace(&mut tensor);
                TensorData::Matrix(tensor)
            }
        }
    }
}

impl ApproxEq for &TensorData {
    type Margin = F64Margin;

    fn approx_eq<M: Into<Self::Margin>>(self, other: Self, margin: M) -> bool {
        let margin = margin.into();
        match (self, other) {
            (TensorData::File(l0), TensorData::File(r0)) => l0 == r0,
            (
                TensorData::Gate((name_l, angles_l, adjoint_l)),
                TensorData::Gate((name_r, angles_r, adjoint_r)),
            ) => name_l == name_r && adjoint_l == adjoint_r && angles_l.approx_eq(angles_r, margin),
            (TensorData::Matrix(l0), TensorData::Matrix(r0)) => l0.approx_eq(r0, margin),
            (TensorData::Uncontracted, TensorData::Uncontracted) => true,
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use float_cmp::assert_approx_eq;

    use super::*;

    #[test]
    #[should_panic(expected = "assertion failed: `(left approx_eq right)`")]
    fn gates_eq_different_name() {
        let g1 = TensorData::Gate((String::from("cx"), vec![], false));
        let g2 = TensorData::Gate((String::from("CX"), vec![], false));
        assert_approx_eq!(&TensorData, &g1, &g2);
    }

    #[test]
    #[should_panic(expected = "assertion failed: `(left approx_eq right)`")]
    fn gates_eq_adjoint() {
        let g1 = TensorData::Gate((String::from("h"), vec![], false));
        let g2 = TensorData::Gate((String::from("h"), vec![], true));
        assert_approx_eq!(&TensorData, &g1, &g2);
    }

    #[test]
    #[should_panic(expected = "assertion failed: `(left approx_eq right)`")]
    fn gates_eq_different_angles() {
        let g1 = TensorData::Gate((String::from("u"), vec![1.4, 2.0, -3.0], false));
        let g2 = TensorData::Gate((String::from("u"), vec![1.4, -2.0, -3.0], false));
        assert_approx_eq!(&TensorData, &g1, &g2);
    }

    #[test]
    #[should_panic(expected = "assertion failed: `(left approx_eq right)`")]
    fn eq_different_data() {
        let g1 = TensorData::Gate((String::from("u"), vec![1.4, 2.0, -3.0], false));
        let g2 = TensorData::new_from_data(&[], vec![Complex64::ONE], None);
        assert_approx_eq!(&TensorData, &g1, &g2);
    }
}
