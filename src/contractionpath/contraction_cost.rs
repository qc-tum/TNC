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
    contract_cost_tensors(tn.tensor(i), tn.tensor(j))
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
/// assert_eq!(contract_cost_tensors(&tn.tensor(0), &tn.tensor(1)), 45045);
/// ```
pub fn contract_cost_tensors(t_1: &Tensor, t_2: &Tensor) -> u64 {
    let shared_dims = t_1 | t_2;
    let bond_dims = t_1.bond_dims();
    shared_dims.legs_iter().map(|e| bond_dims[e]).product()
}

/// Returns Schroedinger contraction space complexity of contracting two [Tensor] objects
///
/// # Arguments
///
/// * `tn` - Tensor containing contracted tensors at positions i and j.
/// * `i` - Position of first Tensor to be contracted
/// * `j` - Position of second Tensor to be contracted
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
    contract_size_tensors(tn.tensor(i), tn.tensor(j))
}

/// Returns Schroedinger contraction space complexity of contracting two [Tensor] objects
///
/// # Arguments
///
/// * `tn` - Tensor containing contracted tensors at positions i and j.
/// * `i` - Position of first Tensor to be contracted
/// * `j` - Position of second Tensor to be contracted
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
/// assert_eq!(contract_size_tensors(&tn.tensor(0), &tn.tensor(1)), 6607);
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
/// * `ssa_path`  - Contraction order as replacement path
/// * `bond_dims`- Dict of bond dimensions.
pub fn contract_path_cost(inputs: &[Tensor], contract_path: &[ContractionIndex]) -> (u64, u64) {
    let mut op_cost = 0;
    let mut mem_cost = 0;
    let mut inputs = inputs.to_vec();

    for index in contract_path {
        match *index {
            ContractionIndex::Pair(i, j) => {
                op_cost += contract_cost_tensors(&inputs[i], &inputs[j]);
                let k12 = &inputs[i] ^ &inputs[j];
                let new_mem_cost = contract_size_tensors(&inputs[i], &inputs[j]);
                mem_cost = max(mem_cost, new_mem_cost);
                inputs[i] = k12;
            }
            ContractionIndex::Path(i, ref path) => {
                let costs = contract_path_cost(inputs[i].tensors(), path);
                op_cost += costs.0;
                mem_cost += costs.1;
                inputs[i] = std::mem::take(&mut inputs[i].tensors[0]);
            }
        }
    }

    (op_cost, mem_cost)
}

#[cfg(test)]
mod tests {
    use crate::contractionpath::contraction_cost::contract_path_cost;
    use crate::path;
    use crate::tensornetwork::create_tensor_network;
    use crate::tensornetwork::tensor::Tensor;

    fn setup_simple() -> Tensor {
        create_tensor_network(
            vec![
                Tensor::new(vec![4, 3, 2]),
                Tensor::new(vec![0, 1, 3, 2]),
                Tensor::new(vec![4, 5, 6]),
            ],
            &[(0, 5), (1, 2), (2, 6), (3, 8), (4, 1), (5, 3), (6, 4)].into(),
            None,
        )
    }

    #[test]
    fn test_contract_path_cost() {
        let tn = setup_simple();
        let (op_cost, mem_cost) = contract_path_cost(tn.tensors(), path![(0, 1), (0, 2)]);
        assert_eq!(op_cost, 600);
        assert_eq!(mem_cost, 538);
        let (op_cost, mem_cost) = contract_path_cost(tn.tensors(), path![(0, 2), (0, 1)]);
        assert_eq!(op_cost, 6336);
        assert_eq!(mem_cost, 1176);
    }
}
