use crate::tensornetwork::tensor::Tensor;
use crate::tensornetwork::TensorNetwork;
use itertools::Itertools;
use rand::distributions::{Distribution, Uniform};
use rand::prelude::SliceRandom;
use rand::Rng;
use std::collections::HashMap;
use taco_sys::Tensor as _TacoTensor;

/// Generates random Tensor object with `n` dimensions and corresponding `bond_dims` HashMap,
/// bond dimensions are uniformly distributed between 1 and 20.
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
pub fn random_tensor(n: usize) -> (Tensor, HashMap<i32, u64>) {
    let mut rng = rand::thread_rng();
    let mut tensor_size = Vec::with_capacity(n);
    tensor_size.resize(n, 0);
    let bond_dims = tensor_size
        .iter()
        .map(|_| rng.gen_range(1u64..20))
        .collect::<Vec<u64>>();
    let edges = (0i32..n as i32).collect::<Vec<i32>>();
    let hs = edges.iter().zip(bond_dims.iter()).collect_vec();
    let mut bond_dims = HashMap::new();
    for (i, j) in hs {
        bond_dims.insert(*i, *j);
    }
    (Tensor::new((0i32..n as i32).collect()), bond_dims)
}

