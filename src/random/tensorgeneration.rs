use itertools::Itertools;
use num_complex::Complex64;
use rand::distributions::Uniform;
use rand::Rng;
use tetra::Tensor as DataTensor;

use crate::tensornetwork::tensordata::TensorData;

/// Generates random sparse [`DataTensor`] object.
/// Fills in sparse tensor based on `sparsity` value (defaults to `0.5`).
///
/// # Examples
/// ```
/// # use tensorcontraction::random::tensorgeneration::random_sparse_tensor_data_with_rng;
/// let shape = vec![5, 4, 3];
/// random_sparse_tensor_data_with_rng(&shape, None, &mut rand::thread_rng());
/// ```
pub fn random_sparse_tensor_data_with_rng<R>(
    dims: &[usize],
    sparsity: Option<f32>,
    rng: &mut R,
) -> TensorData
where
    R: Rng + ?Sized,
{
    let sparsity = if let Some(sparsity) = sparsity {
        assert!((0.0..=1.0).contains(&sparsity));
        sparsity
    } else {
        0.5
    };

    let ranges = dims.iter().map(|i| Uniform::new(0, *i)).collect_vec();
    let size = dims.iter().product::<usize>();
    let mut tensor = DataTensor::new(dims);

    let mut nnz = 0;
    let mut loc = Vec::new();
    while (nnz as f32 / size as f32) < sparsity {
        for r in &ranges {
            loc.push(rng.sample(r));
        }
        let val = Complex64::new(rng.gen(), rng.gen());
        tensor.set(&loc, val);
        loc.clear();
        nnz += 1;
    }

    TensorData::Matrix(tensor)
}

/// Generates random sparse [`DataTensor`] object.
/// Fills in sparse tensor based on `sparsity` value (defaults to `0.5`). Uses the thread-local random number generator.
///
/// # Examples
/// ```
/// # use tensorcontraction::random::tensorgeneration::random_sparse_tensor_data;
/// let shape = vec![5,4,3];
/// let r_tensor = random_sparse_tensor_data(&shape, None);
/// ```
#[must_use]
pub fn random_sparse_tensor_data(shape: &[usize], sparsity: Option<f32>) -> TensorData {
    random_sparse_tensor_data_with_rng(shape, sparsity, &mut rand::thread_rng())
}
