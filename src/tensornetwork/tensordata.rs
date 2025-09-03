use std::iter::zip;
use std::path::PathBuf;

use float_cmp::approx_eq;
use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use tetra::{Layout, Tensor as DataTensor};

use crate::{
    gates::{load_gate, load_gate_adjoint},
    io::load_data,
};

/// The data of a tensor.
#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub enum TensorData {
    /// This is for composite tensors that have not been contracted yet, as well as
    /// empty tensors in general.
    #[default]
    Uncontracted,
    /// The data is loaded from a HDF5 file.
    File(PathBuf),
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

    /// Checks for equality of two tensor sources. Does not check between different
    /// types (e.g. `File` and `Matrix`).
    pub fn approx_eq(&self, other: &Self, epsilon: f64) -> bool {
        match (self, other) {
            (Self::File(l0), Self::File(r0)) => l0 == r0,
            (Self::Gate((l0, angles_l, adjoint_l)), Self::Gate((r0, angles_r, adjoint_r))) => {
                if adjoint_l != adjoint_r {
                    return false;
                }
                if l0 != r0 {
                    return false;
                }
                for (angle1, angle2) in zip(angles_l.iter(), angles_r.iter()) {
                    if !approx_eq!(f64, *angle1, *angle2, epsilon = epsilon) {
                        return false;
                    }
                }
                true
            }
            (Self::Matrix(l0), Self::Matrix(r0)) => {
                approx_eq!(&DataTensor, l0, r0, epsilon = epsilon)
            }
            (Self::Uncontracted, Self::Uncontracted) => true,
            _ => false,
        }
    }

    /// Consumes the tensor data and returns the contained tensor.
    pub fn into_data(self) -> DataTensor {
        match self {
            TensorData::Uncontracted => panic!("Cannot convert uncontracted tensor to data"),
            TensorData::File(filename) => load_data(filename).unwrap(),
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gates_eq_different_name() {
        let g1 = TensorData::Gate((String::from("cx"), vec![], false));
        let g2 = TensorData::Gate((String::from("CX"), vec![], false));
        assert!(!g1.approx_eq(&g2, 1e-8));
    }

    #[test]
    fn gates_eq_adjoint() {
        let g1 = TensorData::Gate((String::from("h"), vec![], false));
        let g2 = TensorData::Gate((String::from("h"), vec![], true));
        assert!(!g1.approx_eq(&g2, 1e-8));
    }

    #[test]
    fn gates_eq_different_angles() {
        let g1 = TensorData::Gate((String::from("u"), vec![1.4, 2.0, -3.0], false));
        let g2 = TensorData::Gate((String::from("u"), vec![1.4, -2.0, -3.0], false));
        assert!(!g1.approx_eq(&g2, 1e-8));
    }

    #[test]
    fn eq_different_data() {
        let g1 = TensorData::Gate((String::from("u"), vec![1.4, 2.0, -3.0], false));
        let g2 = TensorData::new_from_data(&[], vec![Complex64::ONE], None);
        assert!(!g1.approx_eq(&g2, 1e-8));
    }
}
