use itertools::Itertools;

use crate::tensornetwork::tensor::Tensor;
use crate::types::ContractionIndex;
use std::cmp::max;
use std::collections::HashMap;

/// Returns Schroedinger contraction time complexity of contracting [Tensor] objects at indices `i` and `j`.
///
/// # Arguments
///
/// * `tn` - Reference to Tensor object.
/// * `i`  - Index of first tensor to contract.
/// * `j`  - Index of second tensor to contract.
///
///
/// # Examples
/// ```
/// # use tensorcontraction::tensornetwork::tensor::Tensor;
/// # use tensorcontraction::tensornetwork::create_tensor_network;
/// # use tensorcontraction::contractionpath::contraction_cost::contract_cost;
/// # use std::collections::HashMap;
/// let vec1 = Vec::from([0,1,2]);
/// let vec2 = Vec::from([2,3,4]);
/// let bond_dims = HashMap::<usize, u64>::from([(0, 5),(1, 7), (2, 9), (3, 11), (4, 13)]);
/// let tn = create_tensor_network(vec![Tensor::new(vec1), Tensor::new(vec2)], &bond_dims, None);
/// assert_eq!(contract_cost(&tn, 0, 1), 45045);
/// ```
pub fn contract_cost(tn: &Tensor, i: usize, j: usize) -> u64 {
    // let kept_dims = tensor_a.symmetric_difference(&tensor_b);
    _contract_cost(tn.get_tensor(i), tn.get_tensor(j), &tn.get_bond_dims())
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
/// # use tensorcontraction::tensornetwork::create_tensor_network;
/// # use tensorcontraction::contractionpath::contraction_cost::_contract_cost;
/// # use std::collections::HashMap;
/// let vec1 = Vec::from([0,1,2]);
/// let vec2 = Vec::from([2,3,4]);
/// let bond_dims = HashMap::<usize, u64>::from([(0, 5),(1, 7), (2, 9), (3, 11), (4, 13)]);
/// let tn = create_tensor_network(vec![Tensor::new(vec1), Tensor::new(vec2)], &bond_dims, None);
/// assert_eq!(_contract_cost(&tn.get_tensor(0), &tn.get_tensor(1), &tn.get_bond_dims()), 45045);
/// ```
pub fn _contract_cost(t_1: &Tensor, t_2: &Tensor, bond_dims: &HashMap<usize, u64>) -> u64 {
    let shared_dims = t_1 | t_2;
    shared_dims
        .get_legs()
        .iter()
        .map(|e| bond_dims[e])
        .product()
}

/// Returns Schroedinger contraction space complexity of contracting [Tensor] objects at indices `i` and `j` and the output
/// tensor as a [Tensor] object.
///
/// # Arguments
///
/// * `tn` - Reference to [Tensor] object.
/// * `i`  - Index of first tensor to contract.
/// * `j`  - Index of second tensor to contract.
///
///
/// # Examples
/// ```
/// # use tensorcontraction::tensornetwork::tensor::Tensor;
/// # use tensorcontraction::tensornetwork::create_tensor_network;
/// # use tensorcontraction::contractionpath::contraction_cost::contract_size;
/// # use std::collections::HashMap;

/// let vec1 = Vec::from([0,1,2]);
/// let vec2 = Vec::from([2,3,4]);
/// let bond_dims = HashMap::<usize, u64>::from([(0, 5),(1, 7), (2, 9), (3, 11), (4, 13)]);
/// let tn = create_tensor_network(vec![Tensor::new(vec1), Tensor::new(vec2)], &bond_dims, None);
/// assert_eq!(contract_size(&tn, 0, 1), (Tensor::new(vec![0,1,3,4]), 6607));
/// ```
pub fn contract_size(tn: &Tensor, i: usize, j: usize) -> (Tensor, u64) {
    _contract_size(tn.get_tensor(i), tn.get_tensor(j), &tn.get_bond_dims())
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
/// # use tensorcontraction::tensornetwork::create_tensor_network;
/// # use tensorcontraction::contractionpath::contraction_cost::_contract_size;
/// # use std::collections::HashMap;
/// let vec1 = Vec::from([0,1,2]);
/// let vec2 = Vec::from([2,3,4]);
/// let bond_dims = HashMap::<usize, u64>::from([(0, 5),(1, 7), (2, 9), (3, 11), (4, 13)]);
/// let tn = create_tensor_network(vec![Tensor::new(vec1), Tensor::new(vec2)], &bond_dims, None);
/// assert_eq!(_contract_size(tn.get_tensor(0), tn.get_tensor(1), &tn.get_bond_dims()), (Tensor::new(vec![0,1,3,4]), 6607));
/// ```
pub fn _contract_size(
    t_1: &Tensor,
    t_2: &Tensor,
    bond_dims: &HashMap<usize, u64>,
) -> (Tensor, u64) {
    let diff = t_1 ^ t_2;

    let cost = diff
        .get_legs()
        .iter()
        .map(|e| bond_dims[e])
        .product::<u64>()
        + t_1.get_legs().iter().map(|e| bond_dims[e]).product::<u64>()
        + t_2.get_legs().iter().map(|e| bond_dims[e]).product::<u64>();

    (
        Tensor::new(diff.get_legs().iter().cloned().collect_vec()),
        cost,
    )
}

/// Returns number of elements in a given [Tensor].
///
/// # Arguments
///
/// * `tensor` - Reference to [Tensor] object.
/// * `bond_dims`- Dict of bond dimensions.
///
/// # Examples
/// ```
/// # use tensorcontraction::tensornetwork::tensor::Tensor;
/// # use tensorcontraction::contractionpath::contraction_cost::_tensor_size;
/// # use tensorcontraction::tensornetwork::create_tensor_network;
/// # use std::collections::HashMap;
/// let vec1 = Vec::from([0,1,2]);
/// let bond_dims = HashMap::<usize, u64>::from([(0, 5),(1, 7), (2, 9), (3, 11), (4, 13)]);
/// let tn = create_tensor_network(vec![Tensor::new(vec1)], &bond_dims, None);
/// assert_eq!(_tensor_size(&tn.get_tensor(0), &tn.get_bond_dims()), 315);
/// ```
pub fn _tensor_size(tensor: &Tensor, bond_dims: &HashMap<usize, u64>) -> u64 {
    tensor
        .get_legs()
        .iter()
        .map(|e| bond_dims[e])
        .product::<u64>()
}

/// Returns number of elements in a given [Tensor].
///
/// # Arguments
///
/// * `tn` - Reference to [Tensor] object.
/// * `i`  - Index of [Tensor]
///
///
/// # Examples
/// ```
/// # use tensorcontraction::tensornetwork::tensor::Tensor;
/// # use tensorcontraction::contractionpath::contraction_cost::size;
/// # use tensorcontraction::tensornetwork::create_tensor_network;
/// # use std::collections::HashMap;
/// let vec1 = Vec::from([0, 1, 2]);
/// let bond_dims = HashMap::<usize, u64>::from([(0, 5), (1, 7), (2, 9)]);
/// let tn = create_tensor_network(vec![Tensor::new(vec1)], &bond_dims, None);
/// assert_eq!(size(&tn, 0), 315);
/// ```
pub fn size(tn: &Tensor, i: usize) -> u64 {
    tn.get_tensor(i)
        .get_legs()
        .iter()
        .map(|e| tn.get_bond_dims()[e])
        .product::<u64>()
}

/// Returns Schroedinger contraction space complexity of fully contracting a nested [Tensor] object
///
/// # Arguments
///
/// * `inputs` - First tensor to determine contraction cost.
/// * `ssa_path`  - Contraction order as SSA path
/// * `bond_dims`- Dict of bond dimensions.
pub fn _contract_path_cost(
    inputs: &[Tensor],
    ssa_path: &[ContractionIndex],
    bond_dims: &HashMap<usize, u64>,
) -> (u64, u64) {
    let mut op_cost = 0;
    let mut mem_cost = 0;
    let mut inputs = inputs.to_vec();
    let mut costs = (0, 0);
    ssa_path.iter().cloned().for_each(|index| match index {
        ContractionIndex::Pair(i, j) => {
            op_cost += _contract_cost(&inputs[i], &inputs[j], bond_dims);
            let (k12, new_mem_cost) = _contract_size(&inputs[i], &inputs[j], bond_dims);
            mem_cost = max(mem_cost, new_mem_cost);
            inputs[i] = k12;
        }
        ContractionIndex::Path(i, path) => {
            costs = _contract_path_cost(inputs[i].get_tensors(), &path, bond_dims);
            op_cost += costs.0;
            mem_cost += costs.1;
            inputs[i] = inputs[i].get_tensor(0).clone();
        }
    });

    (op_cost, mem_cost)
}
