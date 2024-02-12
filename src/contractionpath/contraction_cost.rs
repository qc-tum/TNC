use crate::tensornetwork::tensor::Tensor;
use crate::types::ContractionIndex;
use std::cmp::max;

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
/// # use tensorcontraction::contractionpath::contraction_cost::contract_cost_in_tn;
/// # use std::collections::HashMap;
/// let vec1 = Vec::from([0,1,2]);
/// let vec2 = Vec::from([2,3,4]);
/// let bond_dims = HashMap::<usize, u64>::from([(0, 5),(1, 7), (2, 9), (3, 11), (4, 13)]);
/// let tn = create_tensor_network(vec![Tensor::new(vec1), Tensor::new(vec2)], &bond_dims, None);
/// assert_eq!(contract_cost_in_tn(&tn, 0, 1), 45045);
/// ```
pub fn contract_cost_in_tn(tn: &Tensor, i: usize, j: usize) -> u64 {
    // let kept_dims = tensor_a.symmetric_difference(&tensor_b);
    contract_cost_tensors(tn.get_tensor(i), tn.get_tensor(j))
}

/// Returns Schroedinger contraction time complexity of contracting two [Tensor] objects
///
/// # Arguments
///
/// * `t_1` - First tensor to determine contraction cost.
/// * `t_2` - First tensor to determine contraction cost.
///
///
/// # Examples
/// ```
/// # use tensorcontraction::tensornetwork::tensor::Tensor;
/// # use tensorcontraction::tensornetwork::create_tensor_network;
/// # use tensorcontraction::contractionpath::contraction_cost::contract_cost_tensors;
/// # use std::collections::HashMap;
/// let vec1 = Vec::from([0,1,2]);
/// let vec2 = Vec::from([2,3,4]);
/// let bond_dims = HashMap::<usize, u64>::from([(0, 5),(1, 7), (2, 9), (3, 11), (4, 13)]);
/// let tn = create_tensor_network(vec![Tensor::new(vec1), Tensor::new(vec2)], &bond_dims, None);
/// assert_eq!(contract_cost_tensors(&tn.get_tensor(0), &tn.get_tensor(1)), 45045);
/// ```
pub fn contract_cost_tensors(t_1: &Tensor, t_2: &Tensor) -> u64 {
    let shared_dims = t_1 | t_2;
    shared_dims
        .legs_iter()
        .map(|e| t_1.get_bond_dims()[e])
        .product()
}

/// Returns Schroedinger contraction space complexity of contracting two [Tensor] objects
///
/// # Arguments
///
/// * `t_1` - First tensor to determine contraction cost.
/// * `t_2` - First tensor to determine contraction cost.
///
///
/// # Examples
/// ```
/// # use tensorcontraction::tensornetwork::tensor::Tensor;
/// # use tensorcontraction::tensornetwork::create_tensor_network;
/// # use tensorcontraction::contractionpath::contraction_cost::contract_size_in_tn;
/// # use std::collections::HashMap;
/// let vec1 = Vec::from([0,1,2]);
/// let vec2 = Vec::from([2,3,4]);
/// let bond_dims = HashMap::<usize, u64>::from([(0, 5),(1, 7), (2, 9), (3, 11), (4, 13)]);
/// let tn = create_tensor_network(vec![Tensor::new(vec1), Tensor::new(vec2)], &bond_dims, None);
/// assert_eq!(contract_size_in_tn(&tn, 0, 1), 6607);
/// ```
pub fn contract_size_in_tn(tn: &Tensor, i: usize, j: usize) -> u64 {
    contract_size_tensors(tn.get_tensor(i), tn.get_tensor(j))
}

/// Returns Schroedinger contraction space complexity of contracting two [Tensor] objects
///
/// # Arguments
///
/// * `t_1` - First tensor to determine contraction cost.
/// * `t_2` - First tensor to determine contraction cost.
///
///
/// # Examples
/// ```
/// # use tensorcontraction::tensornetwork::tensor::Tensor;
/// # use tensorcontraction::tensornetwork::create_tensor_network;
/// # use tensorcontraction::contractionpath::contraction_cost::contract_size_tensors;
/// # use std::collections::HashMap;
/// let vec1 = Vec::from([0,1,2]);
/// let vec2 = Vec::from([2,3,4]);
/// let bond_dims = HashMap::<usize, u64>::from([(0, 5),(1, 7), (2, 9), (3, 11), (4, 13)]);
/// let tn = create_tensor_network(vec![Tensor::new(vec1), Tensor::new(vec2)], &bond_dims, None);
/// assert_eq!(contract_size_tensors(&tn.get_tensor(0), &tn.get_tensor(1)), 6607);
/// ```
pub fn contract_size_tensors(t_1: &Tensor, t_2: &Tensor) -> u64 {
    let diff = t_1 ^ t_2;

    diff.size() + t_1.size() + t_2.size()
}

/// Returns Schroedinger contraction space complexity of fully contracting a nested [Tensor] object
///
/// # Arguments
///
/// * `inputs` - First tensor to determine contraction cost.
/// * `ssa_path`  - Contraction order as SSA path
/// * `bond_dims`- Dict of bond dimensions.
pub fn contract_path_cost(inputs: &[Tensor], ssa_path: &[ContractionIndex]) -> (u64, u64) {
    let mut op_cost = 0;
    let mut mem_cost = 0;
    let mut inputs = inputs.to_vec();
    let mut costs = (0, 0);
    ssa_path.iter().cloned().for_each(|index| match index {
        ContractionIndex::Pair(i, j) => {
            op_cost += contract_cost_tensors(&inputs[i], &inputs[j]);
            let k12 = &inputs[i] ^ &inputs[j];
            let new_mem_cost = contract_size_tensors(&inputs[i], &inputs[j]);
            mem_cost = max(mem_cost, new_mem_cost);
            inputs[i] = k12;
        }
        ContractionIndex::Path(i, path) => {
            costs = contract_path_cost(inputs[i].get_tensors(), &path);
            op_cost += costs.0;
            mem_cost += costs.1;
            inputs[i] = inputs[i].get_tensor(0).clone();
        }
    });

    (op_cost, mem_cost)
}
