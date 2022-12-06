use crate::tensornetwork::tensor::Tensor;
use crate::tensornetwork::TensorNetwork;
use std::collections::hash_map::RandomState;
use std::collections::{HashMap, HashSet};
use std::iter::FromIterator;

/// Returns Schroedinger contraction time complexity of contracting Tensor objects at indices `i` and `j`.
///
/// # Arguments
///
/// * `tn` - Reference to TensorNetwork object.
/// * `i`  - Index of first tensor to contract.
/// * `j`  - Index of second tensor to contract.
///
///
/// # Examples
/// ```
/// # use tensorcontraction::tensornetwork::tensor::Tensor;
/// # use tensorcontraction::tensornetwork::TensorNetwork;
/// # use tensorcontraction::contractionpath::contraction_cost::contract_cost;
/// let vec1 = Vec::from([0,1,2]);
/// let vec2 = Vec::from([2,3,4]);
/// let tn = TensorNetwork::from_vector(vec![Tensor::new(vec1), Tensor::new(vec2)], vec![5,7,9,11,13]);
/// assert_eq!(contract_cost(&tn, 0, 1), 45045);
/// ```
pub fn contract_cost(tn: &TensorNetwork, i: usize, j: usize) -> u64 {
    // let kept_dims = tensor_a.symmetric_difference(&tensor_b);
    _contract_cost(tn[i].clone(), tn[j].clone(), tn.get_bond_dims())
    // shared_dims.into_iter().map(|e| tn.get_bond_dims()[e]).product()
}

/// Returns Schroedinger contraction time complexity of contracting two Tensor objects.
///
/// # Arguments
///
/// * `t_1` - First tensor to determine contraction cost.
/// * `t_2`  - First tensor to determine contraction cost.
/// * `bond_dims`- Dict of bond dimensions.
///
///
/// # Examples
/// ```
/// # use tensorcontraction::tensornetwork::tensor::Tensor;
/// # use tensorcontraction::tensornetwork::TensorNetwork;
/// # use tensorcontraction::contractionpath::contraction_cost::contract_cost;
/// let vec1 = Vec::from([0,1,2]);
/// let vec2 = Vec::from([2,3,4]);
/// let tn = TensorNetwork::from_vector(vec![Tensor::new(vec1), Tensor::new(vec2)], vec![5,7,9,11,13]);
/// assert_eq!(contract_cost(&tn, 0, 1), 45045);
/// ```
pub fn _contract_cost(t_1: Tensor, t_2: Tensor, bond_dims: &HashMap<i32, u64>) -> u64 {
    let tensor_a: HashSet<&i32, RandomState> = HashSet::from_iter(t_1.iter());
    let tensor_b: HashSet<&i32, RandomState> = HashSet::from_iter(t_2.iter());

    let shared_dims = tensor_a.union(&tensor_b);

    // let kept_dims = tensor_a.symmetric_difference(&tensor_b);

    shared_dims.into_iter().map(|e| bond_dims[*e]).product()
}

/// Returns Schroedinger contraction space complexity of contracting Tensor objects at indices `i` and `j` and the output
/// tensor as a Tensor object.
///
/// # Arguments
///
/// * `tn` - Reference to TensorNetwork object.
/// * `i`  - Index of first tensor to contract.
/// * `j`  - Index of second tensor to contract.
///
///
/// # Examples
/// ```
/// # use tensorcontraction::tensornetwork::tensor::Tensor;
/// # use tensorcontraction::tensornetwork::TensorNetwork;
/// # use tensorcontraction::contractionpath::contraction_cost::contract_size;
/// let vec1 = Vec::from([0,1,2]);
/// let vec2 = Vec::from([2,3,4]);
/// let tn = TensorNetwork::from_vector(vec![Tensor::new(vec1), Tensor::new(vec2)], vec![5,7,9,11,13]);
/// assert_eq!(contract_size(&tn, 0, 1), (Tensor::new(vec![0,1,3,4]), 6607));
/// ```
pub fn contract_size(tn: &TensorNetwork, i: usize, j: usize) -> (Tensor, u64) {
    _contract_size(tn[i].clone(), tn[j].clone(), tn.get_bond_dims())
}

/// Returns Schroedinger contraction space complexity of contracting two Tensor objects
///
/// # Arguments
///
/// * `t_1` - First tensor to determine contraction cost.
/// * `t_2`  - First tensor to determine contraction cost.
/// * `bond_dims`- Dict of bond dimensions.
///
///
/// # Examples
/// ```
/// # use tensorcontraction::tensornetwork::tensor::Tensor;
/// # use tensorcontraction::tensornetwork::TensorNetwork;
/// # use tensorcontraction::contractionpath::contraction_cost::contract_size;
/// let vec1 = Vec::from([0,1,2]);
/// let vec2 = Vec::from([2,3,4]);
/// let tn = TensorNetwork::from_vector(vec![Tensor::new(vec1), Tensor::new(vec2)], vec![5,7,9,11,13]);
/// assert_eq!(contract_size(&tn, 0, 1), (Tensor::new(vec![0,1,3,4]), 6607));
/// ```
pub fn _contract_size(t_1: Tensor, t_2: Tensor, bond_dims: &HashMap<i32, u64>) -> (Tensor, u64) {
    let legs_i = t_1.get_legs();
    let legs_j = t_2.get_legs();
    let mut diff = Vec::new();
    for leg_i in legs_i.iter() {
        if !legs_j.contains(leg_i) {
            diff.push(*leg_i);
        }
    }

    for leg_j in legs_j.iter() {
        if !legs_i.contains(leg_j) {
            diff.push(*leg_j);
        }
    }

    let cost = diff.iter().map(|e| bond_dims[e]).product::<u64>()
        + t_1.size(bond_dims)
        + t_2.size(bond_dims);
    (Tensor::new(diff), cost)
}

/// Returns number of elements in a given Tensor.
///
/// # Arguments
///
/// * `tn` - Reference to TensorNetwork object.
/// * `i`  - Index of Tensor
///
///
/// # Examples
/// ```
/// # use tensorcontraction::tensornetwork::tensor::Tensor;
/// # use tensorcontraction::tensornetwork::TensorNetwork;
/// # use tensorcontraction::contractionpath::contraction_cost::size;
/// let vec1 = Vec::from([0,1,2]);
/// let tn = TensorNetwork::from_vector(vec![Tensor::new(vec1)], vec![5,7,9]);
/// assert_eq!(size(&tn, 0), 315);
/// ```
pub fn size(tn: &TensorNetwork, i: usize) -> u64 {
    tn[i]
        .get_legs()
        .iter()
        .map(|e| tn.get_bond_dims()[e])
        .product::<u64>()
}
