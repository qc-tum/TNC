use itertools::Itertools;
use ndarray::Dim;
use num_complex::Complex64;
use rand::distr::Uniform;
use rand::Rng;

use crate::tensornetwork::tensordata::{DataTensor, TensorData};

/// Generates random sparse [`DataTensor`] object.
/// Fills in sparse tensor based on `sparsity` value (defaults to `0.5`).
///
/// # Examples
/// ```
/// # use tnc::builders::tensorgeneration::random_sparse_tensor_data_with_rng;
/// let shape = vec![5, 4, 3];
/// random_sparse_tensor_data_with_rng(&shape, None, &mut rand::thread_rng());
/// ```
pub fn random_sparse_tensor_data_with_rng<R>(
    dims: &[usize],
    sparsity: Option<f32>,
    rng: &mut R,
) -> TensorData
where
    R: Rng,
{
    let sparsity = if let Some(sparsity) = sparsity {
        assert!((0.0..=1.0).contains(&sparsity));
        sparsity
    } else {
        0.5
    };

    let ranges = dims
        .iter()
        .map(|i| Uniform::new(0, *i).unwrap())
        .collect_vec();
    let size = dims.iter().product::<usize>();
    let mut tensor = DataTensor::zeros(dims);

    let mut nnz = 0;
    while (nnz as f32 / size as f32) < sparsity {
        let loc = ranges.iter().map(|r| rng.sample(r)).collect_vec();
        let val = Complex64::new(rng.random(), rng.random());
        let dim = Dim(loc);
        let elem = tensor.get_mut(dim).unwrap();
        if *elem != Complex64::ZERO {
            continue; // Skip if the location is already non-zero
        }
        *elem = val;
        nnz += 1;
    }

    TensorData::Matrix(tensor)
}

/// Generates random sparse [`DataTensor`] object.
/// Fills in sparse tensor based on `sparsity` value (defaults to `0.5`). Uses the thread-local random number generator.
///
/// # Examples
/// ```
/// # use tnc::builders::tensorgeneration::random_sparse_tensor_data;
/// let shape = vec![5,4,3];
/// let r_tensor = random_sparse_tensor_data(&shape, None);
/// ```
#[must_use]
pub fn random_sparse_tensor_data(shape: &[usize], sparsity: Option<f32>) -> TensorData {
    random_sparse_tensor_data_with_rng(shape, sparsity, &mut rand::rng())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_sparse_tensor_data() {
        let shape = vec![5, 4, 3];
        let sparsity = 0.3;
        let tensor_data = random_sparse_tensor_data(&shape, Some(sparsity));
        let TensorData::Matrix(tensor) = tensor_data else {
            panic!("Expected TensorData::Matrix variant");
        };
        let total_elements = shape.iter().product::<usize>();
        let non_zero_elements = tensor.iter().filter(|&&x| x != Complex64::ZERO).count();
        let actual_sparsity = non_zero_elements as f32 / total_elements as f32;
        assert!(
            actual_sparsity >= sparsity,
            "Expected sparsity around {sparsity}, but got {actual_sparsity}",
        );
    }
}
