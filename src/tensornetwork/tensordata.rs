use std::iter::zip;

use float_cmp::approx_eq;
use itertools::Itertools;
use num_complex::Complex64;
use tetra::{Layout, Tensor as DataTensor};

#[derive(Debug, Clone)]
pub enum TensorData {
    File(String),
    Gate((&'static str, Vec<f64>)),
    Matrix(DataTensor),
    Empty,
}

impl Eq for TensorData {}

impl PartialEq for TensorData {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::File(l0), Self::File(r0)) => l0.to_lowercase() == r0.to_lowercase(),
            (Self::Gate((l0, angles_l)), Self::Gate((r0, angles_r))) => {
                if l0.to_lowercase() != r0.to_lowercase() {
                    return false;
                }
                for (angle1, angle2) in zip(angles_l.iter(), angles_r.iter()) {
                    if angle1 != angle2 {
                        return false;
                    }
                }
                true
            }
            (Self::Matrix(l0), Self::Matrix(r0)) => {
                let range = l0.shape().iter().map(|e| 0..*e).multi_cartesian_product();

                for index in range {
                    if !approx_eq!(f64, l0.get(&index).im, r0.get(&index).im, epsilon = 1e-8) {
                        return false;
                    }
                    if !approx_eq!(f64, l0.get(&index).re, r0.get(&index).re, epsilon = 1e-8) {
                        return false;
                    }
                }
                true
            }
            (Self::Empty, Self::Empty) => true,
            _ => false,
        }
    }
}

impl TensorData {
    pub fn new_from_flat(
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
}
