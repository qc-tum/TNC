use itertools::Itertools;

use crate::tensornetwork::tensor::Tensor;
use crate::tensornetwork::TensorNetwork;
use std::cmp::max;
use std::collections::HashMap;

/// Returns Schroedinger contraction time complexity of contracting [Tensor] objects at indices `i` and `j`.
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
/// let tn = TensorNetwork::from_vector(vec![Tensor::new(vec1), Tensor::new(vec2)], vec![5,7,9,11,13], None);
/// assert_eq!(contract_cost(&tn, 0, 1), 45045);
/// ```
pub fn contract_cost(tn: &TensorNetwork, i: usize, j: usize) -> u64 {
    // let kept_dims = tensor_a.symmetric_difference(&tensor_b);
    _contract_cost(&tn[i], &tn[j], tn.get_bond_dims())
}

/// Returns Schroedinger contraction time complexity of contracting two [Tensor] objects.
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
/// # use tensorcontraction::contractionpath::contraction_cost::_contract_cost;
/// let vec1 = Vec::from([0,1,2]);
/// let vec2 = Vec::from([2,3,4]);
/// let tn = TensorNetwork::from_vector(vec![Tensor::new(vec1), Tensor::new(vec2)], vec![5,7,9,11,13], None);
///
/// assert_eq!(_contract_cost(&tn[0], &tn[1], tn.get_bond_dims()), 45045);
/// ```
pub fn _contract_cost(t_1: &Tensor, t_2: &Tensor, bond_dims: &HashMap<usize, u64>) -> u64 {
    let shared_dims = t_1 | t_2;
    shared_dims.iter().map(|e| bond_dims[e]).product()
}

/// Returns Schroedinger contraction space complexity of contracting [Tensor] objects at indices `i` and `j` and the output
/// tensor as a [Tensor] object.
///
/// # Arguments
///
/// * `tn` - Reference to [TensorNetwork] object.
/// * `i`  - Index of first tensor to contract.
/// * `j`  - Index of second tensor to contract.
///
///
/// # Examples
/// ```
/// # use tensorcontraction::tensornetwork::tensor::Tensor;
/// # use tensorcontraction::tensornetwork::TensorNetwork;
/// # use tensorcontraction::contractionpath::contraction_cost::contract_size_in_tn;
/// let vec1 = Vec::from([0,1,2]);
/// let vec2 = Vec::from([2,3,4]);
/// let tn = TensorNetwork::from_vector(vec![Tensor::new(vec1), Tensor::new(vec2)], vec![5,7,9,11,13], None);
/// assert_eq!(contract_size_in_tn(&tn, 0, 1), (Tensor::new(vec![0,1,3,4]), 6607));
/// ```
pub fn contract_size_in_tn(tn: &TensorNetwork, i: usize, j: usize) -> (Tensor, u64) {
    contract_size_tensors(&tn[i], &tn[j], tn.get_bond_dims())
}

/// Returns Schroedinger contraction space complexity of contracting two [Tensor] objects
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
/// # use tensorcontraction::contractionpath::contraction_cost::contract_size_tensors;
/// let vec1 = Vec::from([0,1,2]);
/// let vec2 = Vec::from([2,3,4]);
/// let tn = TensorNetwork::from_vector(vec![Tensor::new(vec1), Tensor::new(vec2)], vec![5,7,9,11,13], None);
/// assert_eq!(contract_size_tensors(&tn[0], &tn[1], tn.get_bond_dims()), (Tensor::new(vec![0,1,3,4]), 6607));
/// ```
pub fn contract_size_tensors(
    t_1: &Tensor,
    t_2: &Tensor,
    bond_dims: &HashMap<usize, u64>,
) -> (Tensor, u64) {
    let diff = t_1 ^ t_2;

    let cost = diff.iter().map(|e| bond_dims[e]).product::<u64>()
        + t_1.size(bond_dims)
        + t_2.size(bond_dims);

    (diff, cost)
}

/// Returns Schroedinger contraction space complexity of contracting two [Tensor] objects
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
/// # use tensorcontraction::contractionpath::contraction_cost::contract_size_in_tn;
/// let vec1 = Vec::from([0,1,2]);
/// let vec2 = Vec::from([2,3,4]);
/// let tn = TensorNetwork::from_vector(vec![Tensor::new(vec1), Tensor::new(vec2)], vec![5,7,9,11,13], None);
/// assert_eq!(contract_size_in_tn(&tn, 0, 1), (Tensor::new(vec![0,1,3,4]), 6607));
/// ```
pub fn contract_path_cost(
    inputs: &[Tensor],
    ssa_path: &[(usize, usize)],
    bond_dims: &HashMap<usize, u64>,
) -> (u64, u64) {
    let mut op_cost = 0;
    let mut mem_cost = 0;
    let mut inputs = inputs.to_vec();
    for &(i, j) in ssa_path.iter() {
        op_cost += _contract_cost(&inputs[i], &inputs[j], bond_dims);
        let (k12, new_mem_cost) = contract_size_tensors(&inputs[i], &inputs[j], bond_dims);
        mem_cost = max(mem_cost, new_mem_cost);
        inputs[i] = k12;
    }

    (op_cost, mem_cost)
}
