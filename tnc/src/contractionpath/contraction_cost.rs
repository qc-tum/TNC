//! Different methods to compute the computational and memory cost of contraction
//! paths.

use num_complex::Complex64;

use crate::{
    contractionpath::{ContractionPath, SimplePathRef},
    tensornetwork::tensor::{EdgeIndex, Tensor},
};

/// Returns Schroedinger contraction time complexity of contracting two [`Tensor`]
/// objects. Considers cost of complex operations.
///
/// # Examples
/// ```
/// # use tnc::tensornetwork::tensor::Tensor;
/// # use tnc::contractionpath::contraction_cost::contract_cost_tensors;
/// # use rustc_hash::FxHashMap;
/// let bond_dims = FxHashMap::from_iter([(0, 5),(1, 7), (2, 9), (3, 11), (4, 13)]);
/// let tensor1 = Tensor::new_from_map(vec![0, 1, 2], &bond_dims);
/// let tensor2 = Tensor::new_from_map(vec![2, 3, 4], &bond_dims);
/// // result = [0, 1, 2, 3, 4] // cost of (9-1)*54*5005 = 350350;
/// let tn = Tensor::new_composite(vec![tensor1, tensor2]);
/// assert_eq!(contract_cost_tensors(&tn.tensor(0), &tn.tensor(1)), 350350.);
/// ```
pub fn contract_cost_tensors(t_1: &Tensor, t_2: &Tensor) -> f64 {
    let final_dims = t_1 ^ t_2;
    let shared_dims = t_1 & t_2;

    let single_loop_cost = shared_dims.size();
    (single_loop_cost - 1f64).mul_add(2f64, single_loop_cost * 6f64) * final_dims.size()
}

/// Returns Schroedinger contraction time complexity of contracting two [`Tensor`]
/// objects. Naive op cost, does not consider costs of multiplication.
///
/// # Examples
/// ```
/// # use tnc::tensornetwork::tensor::Tensor;
/// # use tnc::contractionpath::contraction_cost::contract_op_cost_tensors;
/// # use rustc_hash::FxHashMap;
/// let bond_dims = FxHashMap::from_iter([(0, 5),(1, 7), (2, 9), (3, 11), (4, 13)]);
/// let tensor1 = Tensor::new_from_map(vec![0, 1, 2], &bond_dims);
/// let tensor2 = Tensor::new_from_map(vec![2, 3, 4], &bond_dims);
/// // result = [0, 1, 2, 3, 4] // cost of 5*7*9*11*13 = 45045;
/// let tn = Tensor::new_composite(vec![tensor1, tensor2]);
/// assert_eq!(contract_op_cost_tensors(&tn.tensor(0), &tn.tensor(1)), 45045.);
/// ```
#[inline]
pub fn contract_op_cost_tensors(t_1: &Tensor, t_2: &Tensor) -> f64 {
    let all_dims = t_1 | t_2;
    all_dims.size()
}

/// Returns Schroedinger contraction space complexity of contracting two [`Tensor`]
/// objects.
///
/// # Examples
/// ```
/// # use tnc::tensornetwork::tensor::Tensor;
/// # use tnc::contractionpath::contraction_cost::contract_size_tensors;
/// # use rustc_hash::FxHashMap;
/// let bond_dims = FxHashMap::from_iter([(0, 5),(1, 7), (2, 9), (3, 11), (4, 13)]);
/// let tensor1 = Tensor::new_from_map(vec![0, 1, 2], &bond_dims); // 315 entries
/// let tensor2 = Tensor::new_from_map(vec![2, 3, 4], &bond_dims); // 1287 entries
/// // result = [0, 1, 3, 4] //  5005 entries -> total 6607 entries
/// let tn = Tensor::new_composite(vec![tensor1, tensor2]);
/// assert_eq!(contract_size_tensors(&tn.tensor(0), &tn.tensor(1)), 6607.);
/// ```
#[inline]
pub fn contract_size_tensors(t_1: &Tensor, t_2: &Tensor) -> f64 {
    let diff = t_1 ^ t_2;
    diff.size() + t_1.size() + t_2.size()
}

/// Returns a rather exact estimate of the memory requirements for
/// contracting tensors `i` and `j`.
///
/// This takes into account if tensors need to be transposed (which doubles the
/// required memory). It does not include memory of additional data like shape,
/// bonddims, legs, etc..
///
/// # Examples
/// ```
/// # use tnc::tensornetwork::tensor::Tensor;
/// # use tnc::contractionpath::contraction_cost::contract_size_tensors_exact;
/// # use rustc_hash::FxHashMap;
/// let bond_dims = FxHashMap::from_iter([(0, 5),(1, 7), (2, 9), (3, 11)]);
/// let tensor1 = Tensor::new_from_map(vec![0, 1, 2], &bond_dims); // requires 5040 bytes
/// let tensor2 = Tensor::new_from_map(vec![3, 2], &bond_dims);    // requires 1584 bytes
/// // result = [0, 1, 3], requires 6160 bytes
/// let tn = Tensor::new_composite(vec![tensor1, tensor2]);
/// assert_eq!(contract_size_tensors_exact(&tn.tensor(0), &tn.tensor(1)), 12784.);
/// ```
pub fn contract_size_tensors_exact(i: &Tensor, j: &Tensor) -> f64 {
    /// Checks if `prefix` is a prefix of `list`.
    #[inline]
    fn is_prefix(prefix: &[EdgeIndex], list: &[EdgeIndex]) -> bool {
        if prefix.len() > list.len() {
            return false;
        }
        list.iter().zip(prefix.iter()).all(|(a, b)| a == b)
    }

    /// Checks if `suffix` is a suffix of `list`.
    #[inline]
    fn is_suffix(suffix: &[EdgeIndex], list: &[EdgeIndex]) -> bool {
        if suffix.len() > list.len() {
            return false;
        }
        list.iter()
            .rev()
            .zip(suffix.iter().rev())
            .all(|(a, b)| a == b)
    }

    let ij = i ^ j;
    let contracted_legs = i & j;
    let i_needs_transpose = !is_suffix(contracted_legs.legs(), i.legs());
    let j_needs_transpose = !is_prefix(contracted_legs.legs(), j.legs());

    let i_size = i.size();
    let j_size = j.size();
    let ij_size = ij.size();

    let elements = match (i_needs_transpose, j_needs_transpose) {
        (true, true) => (2.0 * i_size + j_size)
            .max(i_size + 2.0 * j_size)
            .max(i_size + j_size + ij_size),
        (true, false) => (2.0 * i_size + j_size).max(i_size + j_size + ij_size),
        (false, true) => (i_size + 2.0 * j_size).max(i_size + j_size + ij_size),
        (false, false) => i_size + j_size + ij_size,
    };

    elements * std::mem::size_of::<Complex64>() as f64
}

/// Returns Schroedinger contraction time and space complexity of fully contracting
/// the input tensors.
///
/// # Arguments
/// * `inputs` - Tensors to contract
/// * `contract_path`  - Contraction order (replace path)
/// * `only_count_ops` - If `true`, ignores cost of complex multiplication and addition and only counts number of operations
#[inline]
pub fn contract_path_cost(
    inputs: &[Tensor],
    contract_path: &ContractionPath,
    only_count_ops: bool,
) -> (f64, f64) {
    let cost_function = if only_count_ops {
        contract_op_cost_tensors
    } else {
        contract_cost_tensors
    };
    contract_path_custom_cost(inputs, contract_path, cost_function, contract_size_tensors)
}

/// Returns Schroedinger contraction time and space complexity of fully contracting
/// the input tensors.
///
/// # Arguments
/// * `inputs` - Tensors to contract
/// * `contract_path`  - Contraction order (replace path)
/// * `cost_function` - Function to calculate cost of contracting two tensors
fn contract_path_custom_cost(
    inputs: &[Tensor],
    contract_path: &ContractionPath,
    cost_function: fn(&Tensor, &Tensor) -> f64,
    size_function: fn(&Tensor, &Tensor) -> f64,
) -> (f64, f64) {
    let mut op_cost = 0f64;
    let mut mem_cost = 0f64;
    let mut inputs = inputs.to_vec();

    for (i, path) in &contract_path.nested {
        let costs =
            contract_path_custom_cost(inputs[*i].tensors(), path, cost_function, size_function);
        op_cost += costs.0;
        mem_cost = mem_cost.max(costs.1);
        inputs[*i] = inputs[*i].external_tensor();
    }

    for &(i, j) in &contract_path.toplevel {
        op_cost += cost_function(&inputs[i], &inputs[j]);
        let ij = &inputs[i] ^ &inputs[j];
        let new_mem_cost = size_function(&inputs[i], &inputs[j]);
        mem_cost = mem_cost.max(new_mem_cost);
        inputs[i] = ij;
    }

    (op_cost, mem_cost)
}

/// Returns Schroedinger contraction time complexity using the critical path metric
/// and using the sum metric. Additionally returns the space complexity.
#[inline]
pub fn communication_path_op_costs(
    inputs: &[Tensor],
    contract_path: SimplePathRef,
    only_count_ops: bool,
    tensor_cost: Option<&[f64]>,
) -> ((f64, f64), f64) {
    let (parallel_cost, _) =
        communication_path_cost(inputs, contract_path, only_count_ops, true, tensor_cost);
    let (serial_cost, mem_cost) =
        communication_path_cost(inputs, contract_path, only_count_ops, false, tensor_cost);
    ((parallel_cost, serial_cost), mem_cost)
}

/// Returns Schroedinger contraction time and space complexity of fully contracting
/// the input tensors assuming all operations occur in parallel.
///
/// # Arguments
/// * `inputs` - Tensors to contract
/// * `contract_path`  - Contraction order (replace path)
/// * `only_count_ops` - If `true`, ignores cost of complex multiplication and addition and only counts number of operations
/// * `only_circital_path` - If `true`, only counts the cost along the critical path, otherwise the sum of all costs
/// * `tensor_costs` - Initial cost for each tensor
pub fn communication_path_cost(
    inputs: &[Tensor],
    contract_path: SimplePathRef,
    only_count_ops: bool,
    only_critical_path: bool,
    tensor_cost: Option<&[f64]>,
) -> (f64, f64) {
    let cost_function = if only_count_ops {
        contract_op_cost_tensors
    } else {
        contract_cost_tensors
    };
    let tensor_cost = if let Some(tensor_cost) = tensor_cost {
        assert_eq!(inputs.len(), tensor_cost.len());
        tensor_cost
    } else {
        &vec![0f64; inputs.len()]
    };
    if inputs.len() == 1 {
        return (tensor_cost[0], tensor_cost[0]);
    }

    communication_path_custom_cost(
        inputs,
        contract_path,
        cost_function,
        only_critical_path,
        tensor_cost,
    )
}

/// Returns Schroedinger contraction time and space complexity of fully contracting
/// the input tensors assuming all operations occur in parallel.
///
/// # Arguments
/// * `inputs` - Tensors to contract
/// * `contract_path`  - Contraction order (replace path)
/// * `cost_function` - Function to calculate cost of contracting two tensors
/// * `tensor_costs` - Initial cost for each tensor
fn communication_path_custom_cost(
    inputs: &[Tensor],
    contract_path: SimplePathRef,
    cost_function: fn(&Tensor, &Tensor) -> f64,
    only_critical_path: bool,
    tensor_cost: &[f64],
) -> (f64, f64) {
    let mut op_cost = 0f64;
    let mut mem_cost = 0f64;
    let mut inputs = inputs.to_vec();
    let mut tensor_cost = tensor_cost.to_vec();

    for &(i, j) in contract_path {
        let ij = &inputs[i] ^ &inputs[j];
        let new_mem_cost = contract_size_tensors(&inputs[i], &inputs[j]);
        mem_cost = mem_cost.max(new_mem_cost);

        op_cost = if only_critical_path {
            cost_function(&inputs[i], &inputs[j]) + tensor_cost[i].max(tensor_cost[j])
        } else {
            cost_function(&inputs[i], &inputs[j]) + tensor_cost[i] + tensor_cost[j]
        };
        tensor_cost[i] = op_cost;
        inputs[i] = ij;
    }

    (op_cost, mem_cost)
}

/// Computes the max memory requirements for contracting the tensor network using the
/// given path. Uses `memory_estimator` to compute the memory required to contract
/// two tensors.
///
/// Candidates for `memory_estimator` are e.g.:
/// - [`contract_size_tensors`]
/// - [`contract_size_tensors_exact`]
#[inline]
pub fn compute_memory_requirements(
    inputs: &[Tensor],
    contract_path: &ContractionPath,
    memory_estimator: fn(&Tensor, &Tensor) -> f64,
) -> f64 {
    fn id(_: &Tensor, _: &Tensor) -> f64 {
        0.0
    }
    let (_, mem) = contract_path_custom_cost(inputs, contract_path, id, memory_estimator);
    mem
}

#[cfg(test)]
mod tests {
    use super::*;

    use rustc_hash::FxHashMap;

    use crate::path;
    use crate::tensornetwork::tensor::Tensor;

    fn setup_simple() -> Tensor {
        let bond_dims =
            FxHashMap::from_iter([(0, 5), (1, 2), (2, 6), (3, 8), (4, 1), (5, 3), (6, 4)]);
        Tensor::new_composite(vec![
            Tensor::new_from_map(vec![4, 3, 2], &bond_dims),
            Tensor::new_from_map(vec![0, 1, 3, 2], &bond_dims),
            Tensor::new_from_map(vec![4, 5, 6], &bond_dims),
        ])
    }

    fn setup_complex() -> Tensor {
        let bond_dims = FxHashMap::from_iter([
            (0, 5),
            (1, 2),
            (2, 6),
            (3, 8),
            (4, 1),
            (5, 3),
            (6, 4),
            (7, 3),
            (8, 2),
            (9, 2),
        ]);
        let t1_tensors = vec![
            Tensor::new_from_map(vec![4, 3, 2], &bond_dims),
            Tensor::new_from_map(vec![0, 1, 3, 2], &bond_dims),
            Tensor::new_from_map(vec![4, 5, 6], &bond_dims),
        ];
        let t1 = Tensor::new_composite(t1_tensors);

        let t2_tensors = vec![
            Tensor::new_from_map(vec![5, 6, 8], &bond_dims),
            Tensor::new_from_map(vec![7, 8, 9], &bond_dims),
        ];
        let t2 = Tensor::new_composite(t2_tensors);
        Tensor::new_composite(vec![t1, t2])
    }

    fn setup_parallel() -> Tensor {
        let bond_dims =
            FxHashMap::from_iter([(0, 5), (1, 2), (2, 6), (3, 8), (4, 1), (5, 3), (6, 4)]);
        Tensor::new_composite(vec![
            Tensor::new_from_map(vec![4, 3, 2], &bond_dims),
            Tensor::new_from_map(vec![0, 1, 3, 2], &bond_dims),
            Tensor::new_from_map(vec![4, 5, 6], &bond_dims),
            Tensor::new_from_map(vec![5, 6], &bond_dims),
        ])
    }

    #[test]
    fn test_contract_path_cost() {
        let tn = setup_simple();
        let (op_cost, mem_cost) = contract_path_cost(tn.tensors(), &path![(0, 1), (0, 2)], false);
        assert_eq!(op_cost, 4540.);
        assert_eq!(mem_cost, 538.);
        let (op_cost, mem_cost) = contract_path_cost(tn.tensors(), &path![(0, 2), (0, 1)], false);
        assert_eq!(op_cost, 49296.);
        assert_eq!(mem_cost, 1176.);
    }

    #[test]
    fn test_contract_complex_path_cost() {
        let tn = setup_complex();
        let (op_cost, mem_cost) = contract_path_cost(
            tn.tensors(),
            &path![{(0, [(0, 1), (0, 2)]), (1, [(0, 1)])}, (0, 1)],
            false,
        );
        assert_eq!(op_cost, 11188.);
        assert_eq!(mem_cost, 538.);
    }

    #[test]
    fn test_contract_path_cost_only_ops() {
        let tn = setup_simple();
        let (op_cost, mem_cost) = contract_path_cost(tn.tensors(), &path![(0, 1), (0, 2)], true);
        assert_eq!(op_cost, 600.);
        assert_eq!(mem_cost, 538.);
        let (op_cost, mem_cost) = contract_path_cost(tn.tensors(), &path![(0, 2), (0, 1)], true);
        assert_eq!(op_cost, 6336.);
        assert_eq!(mem_cost, 1176.);
    }

    #[test]
    fn test_contract_path_complex_cost_only_ops() {
        let tn = setup_complex();
        let (op_cost, mem_cost) = contract_path_cost(
            tn.tensors(),
            &path![{(0, [(0, 1), (0, 2)]), (1, [(0, 1)])}, (0, 1)],
            true,
        );
        assert_eq!(op_cost, 1464.);
        assert_eq!(mem_cost, 538.);
    }

    #[test]
    fn test_communication_path_cost_only_ops() {
        let tn = setup_parallel();
        let (op_cost, mem_cost) =
            communication_path_cost(tn.tensors(), &[(0, 1), (2, 3), (0, 2)], true, true, None);
        assert_eq!(op_cost, 490.);
        assert_eq!(mem_cost, 538.);
    }

    #[test]
    fn test_communication_path_cost() {
        let tn = setup_parallel();
        let (op_cost, mem_cost) =
            communication_path_cost(tn.tensors(), &[(0, 1), (2, 3), (0, 1)], false, true, None);
        assert_eq!(op_cost, 7564.);
        assert_eq!(mem_cost, 538.);
    }

    #[test]
    fn test_communication_path_cost_only_ops_with_partition_cost() {
        let tn = setup_parallel();
        let tensor_cost = vec![20., 30., 80., 10.];
        let (op_cost, mem_cost) = communication_path_cost(
            tn.tensors(),
            &[(0, 1), (2, 3), (0, 2)],
            true,
            true,
            Some(&tensor_cost),
        );
        assert_eq!(op_cost, 520.);
        assert_eq!(mem_cost, 538.);
    }

    #[test]
    fn test_communication_path_cost_with_partition_cost() {
        let tn = setup_parallel();
        let tensor_cost = vec![20., 30., 80., 10.];
        let (op_cost, mem_cost) = communication_path_cost(
            tn.tensors(),
            &[(0, 1), (2, 3), (0, 1)],
            false,
            true,
            Some(&tensor_cost),
        );
        assert_eq!(op_cost, 7594.);
        assert_eq!(mem_cost, 538.);
    }
}
