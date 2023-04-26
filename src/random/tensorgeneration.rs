use crate::tensornetwork::tensor::Tensor;
use crate::tensornetwork::TensorNetwork;
use itertools::Itertools;
use num_complex::Complex64;
use rand::distributions::{Distribution, Uniform};
use rand::prelude::SliceRandom;
use rand::Rng;
use std::collections::HashMap;
use std::ops::RangeInclusive;
use tetra::Tensor as DataTensor;

/// Generates random Tensor object with `n` dimensions and corresponding `bond_dims` HashMap,
/// bond dimensions are uniformly distributed between 1 and 20.
///
/// # Arguments
///
/// * `n` - Sets number of dimensions in random tensor
/// * `rng` - The random number generator to use.
///
/// # Examples
/// ```
/// # use tensorcontraction::tensornetwork::tensor::Tensor;
/// # use tensorcontraction::random::tensorgeneration::random_tensor_with_rng;
/// let legs = 4;
/// let (tensor, hs) = random_tensor_with_rng(legs, &mut rand::thread_rng());
/// assert_eq!(tensor.get_legs().len(), legs);
/// ```
pub fn random_tensor_with_rng<R>(n: usize, rng: &mut R) -> (Tensor, HashMap<usize, u64>)
where
    R: Rng + ?Sized,
{
    let range = Uniform::new(1u64, 21);
    let bond_dims = (0..n).map(|_| rng.sample(range));
    let edges = 0..n;
    let hs = edges.zip(bond_dims).collect_vec();
    let mut bond_dims = HashMap::new();
    for (i, j) in hs {
        bond_dims.insert(i, j);
    }
    (Tensor::new((0..n).collect()), bond_dims)
}

/// Generates random Tensor object with `n` dimensions and corresponding `bond_dims` HashMap,
/// bond dimensions are uniformly distributed between 1 and 20. Uses the thread-local random number generator.
///
/// # Arguments
///
/// * `n` - Sets number of dimensions in random tensor
///
/// # Examples
/// ```
/// # use tensorcontraction::tensornetwork::tensor::Tensor;
/// # use tensorcontraction::random::tensorgeneration::random_tensor;
/// let legs = 4;
/// let (tensor, hs) = random_tensor(legs);
/// assert_eq!(tensor.get_legs().len(), legs);
/// ```
pub fn random_tensor(n: usize) -> (Tensor, HashMap<usize, u64>) {
    random_tensor_with_rng(n, &mut rand::thread_rng())
}

/// Generates random sparse DataTensor object with same dimenions as Tensor object `t`
/// Fills in sparse tensor based on `sparsity` value.
///
/// # Arguments
///
/// * `t` - Tensor object, random DataTensor will have same dimensions
/// * `sparsity` - an optional fraction between 0 and 1 denoting the sparsity of the output DataTensor.
///                 used to fill in entries in DataTensor at random. If no value is provided, defaults to 0.50
/// * `rng` - The random number generator to use.
///
/// # Examples
/// ```
/// # use tensorcontraction::tensornetwork::tensor::Tensor;
/// # use tensorcontraction::random::tensorgeneration::{random_tensor, random_sparse_tensor_with_rng};
/// # use std::collections::HashMap;
/// let legs = 4;
/// let tensor = random_tensor(legs);
/// let bond_dims = HashMap::from([
/// (0, 17), (1, 19), (2, 12), (3, 12)
/// ]);
///
/// let r_tensor = random_sparse_tensor_with_rng(tensor.0, &bond_dims, None, &mut rand::thread_rng());
///
/// ```
pub fn random_sparse_tensor_with_rng<R>(
    t: &Tensor,
    bond_dims: &HashMap<usize, u64>,
    sparsity: Option<f32>,
    rng: &mut R,
) -> DataTensor
where
    R: Rng + ?Sized,
{
    let sparsity = if let Some(sparsity) = sparsity {
        assert!(RangeInclusive::new(0.0, 1.0).contains(&sparsity));
        sparsity
    } else {
        0.5
    };

    let dims = t
        .get_legs()
        .iter()
        .map(|e| *(bond_dims.get(e).unwrap()) as u32)
        .collect::<Vec<u32>>();
    let ranges: Vec<Uniform<u32>> = dims.iter().map(|i| Uniform::new(0, *i)).collect();
    let size = dims.iter().product::<u32>();
    let mut tensor = DataTensor::new(&dims);

    let mut nnz = 0;
    let mut loc = Vec::<u32>::new();
    while (nnz as f32 / size as f32) < sparsity {
        for r in ranges.iter() {
            loc.push(rng.sample(r));
        }
        let val = Complex64::new(rng.gen(), rng.gen());
        tensor.insert(&loc, val);
        loc.clear();
        nnz += 1;
    }

    tensor
}

/// Generates random sparse DataTensor object with same dimenions as Tensor object `t`
/// Fills in sparse tensor based on `sparsity` value. Uses the thread-local random number generator.
///
/// # Arguments
///
/// * `t` - Tensor object, random DataTensor will have same dimensions
/// * `sparsity` - an optional fraction between 0 and 1 denoting the sparsity of the output DataTensor.
///                 used to fill in entries in DataTensor at random. If no value is provided, defaults to 0.50
///
/// # Examples
/// ```
/// # use tensorcontraction::tensornetwork::tensor::Tensor;
/// # use tensorcontraction::random::tensorgeneration::{random_tensor, random_sparse_tensor};
/// # use std::collections::HashMap;
/// let legs = 4;
/// let tensor = random_tensor(legs);
/// let bond_dims = HashMap::from([
/// (0, 17), (1, 19), (2, 12), (3, 12)
/// ]);
///
/// let r_tensor = random_sparse_tensor(tensor.0, &bond_dims, None);
///
/// ```
pub fn random_sparse_tensor(
    t: &Tensor,
    bond_dims: &HashMap<usize, u64>,
    sparsity: Option<f32>,
) -> DataTensor {
    random_sparse_tensor_with_rng(t, bond_dims, sparsity, &mut rand::thread_rng())
}

/// Generates random [TensorNetwork] objects based on a quantum circuit with `n` qubits and `cycles` layers of
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
/// # use tensorcontraction::tensornetwork::tensor::Tensor;
/// # use tensorcontraction::random::tensorgeneration::random_tensor_network_with_rng;
///
/// let r_tn = random_tensor_network_with_rng(4, 5, &mut rand::thread_rng());
///
/// ```
pub fn random_tensor_network_with_rng<R>(n: usize, cycles: usize, rng: &mut R) -> TensorNetwork
where
    R: Rng + ?Sized,
{
    let mut tensors = Vec::new();
    // counter for indices in tensor network
    let mut index = 3;
    // keeps track of which edge is one a specific wire

    let wires: Vec<usize> = (0..n).collect();
    let mut wire_indices = wires.clone();
    let die = Uniform::from(0..n);

    // Will generate gates for multiple cycles
    for _i in 0..cycles {
        let mut w = wires.clone();
        while die.sample(rng) != 0 {
            if w.len() < 2 {
                break;
            }
            if rng.gen() {
                w.shuffle(rng);
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
            } else {
                w.shuffle(rng);
                let l1 = w.pop().unwrap();
                tensors.push(vec![wire_indices[l1], index + 1]);
                wire_indices[l1] = index + 1;
                index += 1;
            }
        }
    }
    let bond_die = Uniform::from(2..4);
    let mut bond_dims = HashMap::new();
    for i in 0..index + 1 {
        bond_dims.entry(i).or_insert(bond_die.sample(rng));
    }

    let t = TensorNetwork::new(
        tensors.into_iter().map(Tensor::new).collect(),
        bond_dims,
        None,
    );
    if t.is_empty() {
        return random_tensor_network_with_rng(n, cycles, rng);
    }
    t
}

/// Generates random [TensorNetwork] objects based on a quantum circuit with `n` qubits and `cycles` layers of
/// randomly generated 1- or 2-qubit gates. Uses the thread-local random number generator.
///
///
/// # Arguments
///
/// * `n` - Number of qubits in quantum circuit
/// * `cycles` - Number of layers of gates
///
/// # Examples
/// ```
/// # use tensorcontraction::tensornetwork::tensor::Tensor;
/// # use tensorcontraction::random::tensorgeneration::{random_tensor_network};
///
/// let r_tn = random_tensor_network(4, 5);
///
/// ```
pub fn random_tensor_network(n: usize, cycles: usize) -> TensorNetwork {
    random_tensor_network_with_rng(n, cycles, &mut rand::thread_rng())
}
