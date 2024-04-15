use float_cmp::approx_eq;
use std::iter::zip;
use std::path::PathBuf;

use num_complex::Complex64;
use tetra::{all_close, Layout, Tensor as DataTensor};

#[derive(Debug, Clone)]
pub enum TensorData {
    File(PathBuf),
    Gate((String, Vec<f64>)),
    Matrix(DataTensor),
    Uncontracted,
}

impl TensorData {
    /// Creates a new tensor from raw (flat) data.
    pub fn new_from_data(
        dimensions: Vec<u64>,
        data: Vec<Complex64>,
        layout: Option<Layout>,
    ) -> Self {
        Self::Matrix(DataTensor::new_from_flat(
            dimensions
                .iter()
                .map(|e| *e as u32)
                .collect::<Vec<u32>>()
                .as_slice(),
            data,
            layout,
        ))
    }

    pub fn approx_eq(&self, other: &TensorData, epsilon: f64) -> bool {
        match (self, other) {
            (Self::File(l0), Self::File(r0)) => l0 == r0,
            (Self::Gate((l0, angles_l)), Self::Gate((r0, angles_r))) => {
                if l0.to_lowercase() != r0.to_lowercase() {
                    return false;
                }
                for (angle1, angle2) in zip(angles_l.iter(), angles_r.iter()) {
                    if !approx_eq!(f64, *angle1, *angle2, epsilon = epsilon) {
                        return false;
                    }
                }
                true
            }
            (Self::Matrix(l0), Self::Matrix(r0)) => all_close(l0, r0, epsilon),
            (Self::Uncontracted, Self::Uncontracted) => true,
            _ => false,
        }
    }
}
