use crate::tensornetwork::TensorNetwork;
use std::collections::hash_map::RandomState;
use std::collections::HashSet;
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
/// # use tensorcontraction::tensornetwork::contractionpath::contract_cost;
/// let vec1 = Vec::from([0,1,2]);
/// let vec2 = Vec::from([2,3,4]);
/// let tn = TensorNetwork::from_vector(vec![Tensor::new(vec1), Tensor::new(vec2)], vec![5,7,9,11,13]);
/// assert_eq!(contract_cost(&tn, 0, 1), 45045);
/// ```
pub fn contract_cost(tn: &TensorNetwork, i: usize, j: usize) -> u32 {
    let tensor_a: HashSet<&i32, RandomState> = HashSet::from_iter(tn[i].iter());
    let tensor_b: HashSet<&i32, RandomState> = HashSet::from_iter(tn[j].iter());

    let shared_dims = tensor_a.union(&tensor_b);
    // let kept_dims = tensor_a.symmetric_difference(&tensor_b);

    shared_dims.into_iter().map(|e| tn.bond_dims[e]).product()
}

/// Returns Schroedinger contraction space complexity of contracting Tensor objects at indices `i` and `j`.
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
/// # use tensorcontraction::tensornetwork::contractionpath::contract_size;
/// let vec1 = Vec::from([0,1,2]);
/// let vec2 = Vec::from([2,3,4]);
/// let tn = TensorNetwork::from_vector(vec![Tensor::new(vec1), Tensor::new(vec2)], vec![5,7,9,11,13]);
/// assert_eq!(contract_size(&tn, 0, 1), 6607);
/// ```
pub fn contract_size(tn: &TensorNetwork, i: usize, j: usize) -> u32 {
    let legs_i = tn[i].get_legs();
    let legs_j = tn[j].get_legs();
    let mut diff = Vec::new();
    for leg_i in legs_i.iter() {
        if !legs_j.contains(leg_i) {
            diff.push(leg_i);
        }
    }

    for leg_j in legs_j.iter() {
        if !legs_i.contains(leg_j) {
            diff.push(leg_j);
        }
    }

    print!("{:?}", diff);
    diff.iter().map(|e| tn.bond_dims[e]).product::<u32>()
        + tn[j].size(&tn.bond_dims)
        + tn[i].size(&tn.bond_dims)
}
