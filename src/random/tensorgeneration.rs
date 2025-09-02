use itertools::Itertools;
use num_complex::Complex64;
use rand::distributions::{Distribution, Uniform};
use rand::prelude::SliceRandom;
use rand::Rng;
use rustc_hash::FxHashMap;
use tetra::Tensor as DataTensor;

use crate::tensornetwork::tensor::Tensor;
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

/// Generates random [`Tensor`] objects based on a quantum circuit with `n` qubits and `cycles` layers of
/// randomly generated 1- or 2-qubit gates.
///
///
/// # Arguments
///
/// * `n` - Number of qubits in quantum circuit
/// * `cycles` - Number of layers of gates
/// * `rng` - The random number generator to use.
///
/// # Examples
/// ```
/// # use tensorcontraction::random::tensorgeneration::random_tensor_network_with_rng;
/// let r_tn = random_tensor_network_with_rng(4, 5, &mut rand::thread_rng());
///
/// ```
pub fn random_tensor_network_with_rng<R>(n: usize, cycles: usize, rng: &mut R) -> Tensor
where
    R: Rng + ?Sized,
{
    let mut tensors = Vec::new();
    // counter for indices in tensor network
    let mut index = n - 1;
    // keeps track of which edge is one a specific wire
    let wires: Vec<usize> = (0..n).collect();
    let mut wire_indices = wires.clone();
    // Allow a 10% chance to not place any gates and continue to next layer early.
    let die = Uniform::from(0..10);

    // Will generate gates for multiple cycles
    for _i in 0..cycles {
        let mut w = wires.clone();
        w.shuffle(rng);
        while die.sample(rng) != 0 {
            if w.is_empty() {
                break;
            }
            if rng.gen() {
                if w.len() > 2 {
                    let l1 = w.pop().unwrap();
                    let l2 = w.pop().unwrap();
                    tensors.push(vec![
                        wire_indices[l1],
                        wire_indices[l2],
                        index + 1,
                        index + 2,
                    ]);
                    wire_indices[l1] = index + 1;
                    wire_indices[l2] = index + 2;
                    index += 2;
                }
            } else {
                let l1 = w.pop().unwrap();
                tensors.push(vec![wire_indices[l1], index + 1]);
                wire_indices[l1] = index + 1;
                index += 1;
            }
        }
    }
    if tensors.is_empty() {
        return random_tensor_network_with_rng(n, cycles, rng);
    }
    let bond_die = Uniform::from(2..4);
    let bond_dims = rng
        .sample_iter(bond_die)
        .take(index + 1)
        .enumerate()
        .collect::<FxHashMap<_, _>>();

    let mut t = Tensor::new_composite(
        tensors
            .into_iter()
            .map(|legs| Tensor::new_from_map(legs, &bond_dims))
            .collect(),
    );
    for tensor in &mut t.tensors {
        tensor.set_tensor_data(random_sparse_tensor_data(&tensor.shape().unwrap(), None));
    }
    t
}

/// Generates random [`Tensor`] objects based on a quantum circuit with `n` qubits and `cycles` layers of
/// randomly generated 1- or 2-qubit gates. Uses the thread-local random number generator.
///
/// # Arguments
/// * `n` - Number of qubits in quantum circuit
/// * `cycles` - Number of layers of gates
///
/// # Examples
/// ```
/// # use tensorcontraction::random::tensorgeneration::random_tensor_network;
/// let r_tn = random_tensor_network(4, 5);
///
/// ```
#[must_use]
pub fn random_tensor_network(n: usize, cycles: usize) -> Tensor {
    random_tensor_network_with_rng(n, cycles, &mut rand::thread_rng())
}

#[must_use]
pub fn create_filled_tensor_network<R>(tensors: Vec<Tensor>, rng: &mut R) -> Tensor
where
    R: Rng + ?Sized,
{
    let mut tn = Tensor::new_composite(tensors);
    for child_tensor in &mut tn.tensors {
        child_tensor.set_tensor_data(random_sparse_tensor_data_with_rng(
            &child_tensor.shape().unwrap(),
            None,
            rng,
        ));
    }
    tn
}
