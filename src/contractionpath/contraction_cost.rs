use std::sync::Arc;

use crate::tensornetwork::tensor::Tensor;
use crate::types::{ContractionIndex, EdgeIndex, SlicingPlan};

/// Returns Schroedinger contraction time complexity of contracting two [`Tensor`]
/// objects. Considers cost of complex operations.
///
/// # Examples
/// ```
/// # use tensorcontraction::tensornetwork::tensor::Tensor;
/// # use tensorcontraction::tensornetwork::create_tensor_network;
/// # use tensorcontraction::contractionpath::contraction_cost::contract_cost_tensors;
/// # use rustc_hash::FxHashMap;
/// let vec1 = vec![0, 1, 2];
/// let vec2 = vec![2, 3, 4];
/// let bond_dims = FxHashMap::from_iter([(0, 5),(1, 7), (2, 9), (3, 11), (4, 13)]);
/// let tn = create_tensor_network(vec![Tensor::new(vec1), Tensor::new(vec2)], &bond_dims);
/// assert_eq!(contract_cost_tensors(&tn.tensor(0), &tn.tensor(1), None), 350350f64);
/// ```
pub fn contract_cost_tensors(t_1: &Tensor, t_2: &Tensor, _: Option<&SlicingPlan>) -> f64 {
    let final_dims = t_1 ^ t_2;
    let shared_dims = t_1 & t_2;
    let bond_dims = t_1.bond_dims();
    let single_loop_cost = shared_dims
        .legs()
        .iter()
        .map(|e| bond_dims[e] as f64)
        .product::<f64>();

    (single_loop_cost - 1f64).mul_add(2f64, single_loop_cost * 6f64)
        * final_dims
            .legs()
            .iter()
            .map(|e| bond_dims[e] as f64)
            .product::<f64>()
}

/// Returns Schroedinger contraction time complexity of contracting two [`Tensor`]
/// objects. Naive op cost, does not consider costs of multiplication.
///
/// # Examples
/// ```
/// # use tensorcontraction::tensornetwork::tensor::Tensor;
/// # use tensorcontraction::tensornetwork::create_tensor_network;
/// # use tensorcontraction::contractionpath::contraction_cost::contract_op_cost_tensors;
/// # use rustc_hash::FxHashMap;
/// let vec1 = vec![0, 1, 2];
/// let vec2 = vec![2, 3, 4];
/// let bond_dims = FxHashMap::from_iter([(0, 5),(1, 7), (2, 9), (3, 11), (4, 13)]);
/// let tn = create_tensor_network(vec![Tensor::new(vec1), Tensor::new(vec2)], &bond_dims);
/// assert_eq!(contract_op_cost_tensors(&tn.tensor(0), &tn.tensor(1), None), 45045f64);
/// ```
pub fn contract_op_cost_tensors(t_1: &Tensor, t_2: &Tensor, _: Option<&SlicingPlan>) -> f64 {
    let all_dims = t_1 | t_2;
    let bond_dims = t_1.bond_dims();

    all_dims
        .legs()
        .iter()
        .map(|e| bond_dims[e] as f64)
        .product::<f64>()
}

/// Returns Schroedinger contraction time complexity of contracting two [`Tensor`]
/// objects. Considers cost of complex operations and the usage of slicing.
///
/// # Examples
/// ```
/// # use tensorcontraction::tensornetwork::tensor::Tensor;
/// # use tensorcontraction::tensornetwork::create_tensor_network;
/// # use tensorcontraction::contractionpath::contraction_cost::contract_cost_tensors_slicing;
/// # use tensorcontraction::types::SlicingPlan;
/// # use rustc_hash::FxHashMap;
/// let vec1 = vec![0, 1, 2];
/// let vec2 = vec![2, 3, 4];
/// let bond_dims = FxHashMap::from_iter([(0, 5),(1, 7), (2, 9), (3, 11), (4, 13)]);
/// let tn = create_tensor_network(vec![Tensor::new(vec1), Tensor::new(vec2)], &bond_dims);
/// let slicing_plan = SlicingPlan{ slices: vec![2, 3] };
/// assert_eq!(contract_cost_tensors_slicing(&tn.tensor(0), &tn.tensor(1), Some(&slicing_plan)), 35945f64);
/// ```
pub fn contract_cost_tensors_slicing(
    t_1: &Tensor,
    t_2: &Tensor,
    slicing: Option<&SlicingPlan>,
) -> f64 {
    let final_dims = t_1 ^ t_2;
    let shared_dims = t_1 & t_2;
    let bond_dims = t_1.bond_dims();
    let mut sliced_internal_legs = Vec::new();
    let slicing = if let Some(slicing_plan) = slicing {
        &slicing_plan.slices
    } else {
        &Vec::new()
    };
    let single_loop_cost = shared_dims
        .legs()
        .iter()
        .filter(|e| {
            let internal_slice = slicing.contains(e);
            if internal_slice {
                sliced_internal_legs.push(**e)
            };
            internal_slice
        })
        .map(|e| bond_dims[e] as f64)
        .product::<f64>();

    ((single_loop_cost - 1f64).mul_add(2f64, single_loop_cost * 6f64)
        + sliced_internal_legs
            .iter()
            .map(|e| bond_dims[e] as f64)
            .sum::<f64>())
        * final_dims
            .legs()
            .iter()
            .filter(|e| !slicing.contains(e))
            .map(|e| bond_dims[e] as f64)
            .product::<f64>()
}

/// Returns Schroedinger contraction time complexity of contracting two [`Tensor`]
/// objects. Naive op cost, does not consider costs of multiplication, but considers the effects of slicing.
///
/// # Examples
/// ```
/// # use tensorcontraction::tensornetwork::tensor::Tensor;
/// # use tensorcontraction::tensornetwork::create_tensor_network;
/// # use tensorcontraction::contractionpath::contraction_cost::contract_op_cost_tensors_slicing;
/// # use tensorcontraction::types::SlicingPlan;
/// # use rustc_hash::FxHashMap;
/// let vec1 = vec![0, 1, 2];
/// let vec2 = vec![2, 3, 4];
/// let bond_dims = FxHashMap::from_iter([(0, 5),(1, 7), (2, 9), (3, 11), (4, 13)]);
/// let slicing_plan = SlicingPlan{ slices: vec![2, 3] };
/// let tn = create_tensor_network(vec![Tensor::new(vec1), Tensor::new(vec2)], &bond_dims);
/// assert_eq!(contract_op_cost_tensors_slicing(&tn.tensor(0), &tn.tensor(1), Some(&slicing_plan)), 455f64);
/// ```
pub fn contract_op_cost_tensors_slicing(
    t_1: &Tensor,
    t_2: &Tensor,
    slicing: Option<&SlicingPlan>,
) -> f64 {
    let all_dims = t_1 | t_2;
    let slicing = if let Some(slicing_plan) = slicing {
        &slicing_plan.slices
    } else {
        &Vec::new()
    };
    all_dims
        .sliced_size(slicing)
}

/// Returns Schroedinger contraction space complexity of contracting two [`Tensor`]
/// objects.
///
/// # Examples
/// ```
/// # use tensorcontraction::tensornetwork::tensor::Tensor;
/// # use tensorcontraction::tensornetwork::create_tensor_network;
/// # use tensorcontraction::contractionpath::contraction_cost::contract_size_tensors;
/// # use rustc_hash::FxHashMap;
/// let vec1 = vec![0, 1, 2];
/// let vec2 = vec![2, 3, 4];
/// let bond_dims = FxHashMap::from_iter([(0, 5),(1, 7), (2, 9), (3, 11), (4, 13)]);
/// let tn = create_tensor_network(vec![Tensor::new(vec1), Tensor::new(vec2)], &bond_dims);
/// assert_eq!(contract_size_tensors(&tn.tensor(0), &tn.tensor(1), None), 6607f64);
/// ```
#[inline]
pub fn contract_size_tensors(t_1: &Tensor, t_2: &Tensor, _: Option<&SlicingPlan>) -> f64 {
    let diff = t_1 ^ t_2;
    diff.size() + t_1.size() + t_2.size()
}

/// Returns Schroedinger contraction space complexity of contracting two [`Tensor`]
/// objects. Considers the effects of slicing.
///
/// # Examples
/// ```
/// # use tensorcontraction::tensornetwork::tensor::Tensor;
/// # use tensorcontraction::tensornetwork::create_tensor_network;
/// # use tensorcontraction::contractionpath::contraction_cost::contract_size_tensors_slicing;
/// # use tensorcontraction::types::SlicingPlan;
/// # use rustc_hash::FxHashMap;
/// let vec1 = vec![0, 1, 2];
/// let vec2 = vec![2, 3, 4];
/// let bond_dims = FxHashMap::from_iter([(0, 5),(1, 7), (2, 9), (3, 11), (4, 13)]);
/// let slicing_plan = SlicingPlan{ slices: vec![2, 3] };
/// let tn = create_tensor_network(vec![Tensor::new(vec1), Tensor::new(vec2)], &bond_dims);
/// assert_eq!(contract_size_tensors_slicing(&tn.tensor(0), &tn.tensor(1), Some(&slicing_plan)), 503f64);
/// ```
#[inline]
pub fn contract_size_tensors_slicing(
    t_1: &Tensor,
    t_2: &Tensor,
    slicing: Option<&SlicingPlan>,
) -> f64 {
    let slicing = if let Some(slicing_plan) = slicing {
        &slicing_plan.slices
    } else {
        &Vec::new()
    };
    let diff = t_1 ^ t_2;
    diff.sliced_size(slicing) + t_1.sliced_size(slicing) + t_2.sliced_size(slicing)
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
/// # use tensorcontraction::tensornetwork::tensor::Tensor;
/// # use tensorcontraction::tensornetwork::create_tensor_network;
/// # use tensorcontraction::contractionpath::contraction_cost::contract_size_tensors_exact;
/// # use rustc_hash::FxHashMap;
/// let vec1 = vec![0, 1, 2]; // requires 5040 bytes
/// let vec2 = vec![3, 2]; // requires 1584 bytes
/// // result = [0, 1, 3], requires 6160 bytes
/// let bond_dims = FxHashMap::from_iter([(0, 5),(1, 7), (2, 9), (3, 11)]);
/// let tn = create_tensor_network(vec![Tensor::new(vec1), Tensor::new(vec2)], &bond_dims);
/// assert_eq!(contract_size_tensors_exact(&tn.tensor(0), &tn.tensor(1)), 799f64);
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

    match (i_needs_transpose, j_needs_transpose) {
        (true, true) => (2.0 * i_size + j_size)
            .max(i_size + 2.0 * j_size)
            .max(i_size + j_size + ij_size),
        (true, false) => (2.0 * i_size + j_size).max(i_size + j_size + ij_size),
        (false, true) => (i_size + 2.0 * j_size).max(i_size + j_size + ij_size),
        (false, false) => i_size + j_size + ij_size,
    }
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
    contract_path: &[ContractionIndex],
    only_count_ops: bool,
) -> (f64, f64) {
    let cost_function = if only_count_ops {
        contract_op_cost_tensors
    } else {
        contract_cost_tensors
    };
    contract_path_custom_cost(
        inputs,
        contract_path,
        cost_function,
        contract_size_tensors,
        None,
    )
}

/// Returns Schroedinger contraction time and space complexity of fully contracting
/// the input tensors. Considers the effects of slicing.
///
/// # Arguments
/// * `inputs` - Tensors to contract
/// * `contract_path`  - Contraction order (replace path)
/// * `slicing_plan` - Associated slicing plan
/// * `only_count_ops` - If `true`, ignores cost of complex multiplication and addition and only counts number of operations
#[inline]
pub fn contract_path_cost_slicing(
    inputs: &[Tensor],
    contract_path: &[ContractionIndex],
    slicing_plan: Option<&SlicingPlan>,
    only_count_ops: bool,
) -> (f64, f64) {
    let cost_function = if only_count_ops {
        contract_op_cost_tensors_slicing
    } else {
        contract_cost_tensors_slicing
    };
    contract_path_custom_cost(
        inputs,
        contract_path,
        cost_function,
        contract_size_tensors_slicing,
        slicing_plan,
    )
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
    contract_path: &[ContractionIndex],
    cost_function: fn(&Tensor, &Tensor, Option<&SlicingPlan>) -> f64,
    size_function: fn(&Tensor, &Tensor, Option<&SlicingPlan>) -> f64,
    slicing_plan: Option<&SlicingPlan>,
) -> (f64, f64) {
    let mut op_cost = 0f64;
    let mut mem_cost = 0f64;
    let mut inputs = inputs.to_vec();

    for index in contract_path {
        match index {
            ContractionIndex::Pair(i, j) => {
                op_cost += cost_function(&inputs[*i], &inputs[*j], slicing_plan);
                let ij = &inputs[*i] ^ &inputs[*j];
                let new_mem_cost = size_function(&inputs[*i], &inputs[*j], slicing_plan);
                mem_cost = mem_cost.max(new_mem_cost);
                inputs[*i] = ij;
            }
            ContractionIndex::Path(i, slicing, ref path) => {
                let costs = contract_path_custom_cost(
                    inputs[*i].tensors(),
                    path,
                    cost_function,
                    size_function,
                    slicing.as_ref(),
                );
                op_cost += costs.0;
                mem_cost = mem_cost.max(costs.1);
                let intermediate_tensor = Tensor::new_with_bonddims(
                    inputs[*i].external_edges(),
                    Arc::clone(&inputs[*i].bond_dims),
                );
                inputs[*i] = intermediate_tensor;
            }
        }
    }

    (op_cost, mem_cost)
}

/// Returns Schroedinger contraction time and space complexity of fully contracting
/// the input tensors assuming all operations occur in parallel.
///
/// # Arguments
/// * `inputs` - Tensors to contract
/// * `contract_path`  - Contraction order (replace path)
/// * `only_count_ops` - If `true`, ignores cost of complex multiplication and addition and only counts number of operations
/// * `tensor_costs` - Initial cost for each tensor
pub fn communication_path_cost(
    inputs: &[Tensor],
    contract_path: &[ContractionIndex],
    only_count_ops: bool,
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

    communication_path_custom_cost(inputs, contract_path, cost_function, tensor_cost)
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
    contract_path: &[ContractionIndex],
    cost_function: fn(&Tensor, &Tensor, Option<&SlicingPlan>) -> f64,
    tensor_cost: &[f64],
) -> (f64, f64) {
    let mut op_cost = 0f64;
    let mut mem_cost = 0f64;
    let mut inputs = inputs.to_vec();
    let mut tensor_cost = tensor_cost.to_vec();

    for index in contract_path {
        match *index {
            ContractionIndex::Pair(i, j) => {
                let ij = &inputs[i] ^ &inputs[j];
                let new_mem_cost = contract_size_tensors(&inputs[i], &inputs[j], None);
                mem_cost = mem_cost.max(new_mem_cost);

                op_cost = cost_function(&inputs[i], &inputs[j], None)
                    + tensor_cost[i].max(tensor_cost[j]);
                tensor_cost[i] = op_cost;
                inputs[i] = ij;
            }
            ContractionIndex::Path(..) => {
                panic!("Nested paths not supported for contracting communication path");
            }
        }
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
pub fn compute_memory_requirements(
    inputs: &[Tensor],
    contract_path: &[ContractionIndex],
    memory_estimator: fn(&Tensor, &Tensor) -> f64,
) -> f64 {
    let mut inputs = inputs.to_vec();
    let mut max_size = 0.0f64;
    for index in contract_path {
        match *index {
            ContractionIndex::Pair(i, j) => {
                let contracted = &inputs[i] ^ &inputs[j];
                let size = memory_estimator(&inputs[i], &inputs[j]);
                max_size = max_size.max(size);
                inputs[i] = contracted;
            }
            ContractionIndex::Path(i, ref _slicing, ref path) => {
                // TODO: consider slicing
                let max_child_size =
                    compute_memory_requirements(inputs[i].tensors(), path, memory_estimator);
                max_size = max_size.max(max_child_size);
                let tn = {
                    let bond_dims = inputs[i].bond_dims.clone();
                    Tensor::new_with_bonddims(inputs[i].external_edges(), bond_dims)
                };
                inputs[i] = tn;
            }
        }
    }
    max_size
}

#[cfg(test)]
mod tests {

    use rustc_hash::FxHashMap;

    use crate::contractionpath::contraction_cost::{communication_path_cost, contract_path_cost};
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
            &FxHashMap::from_iter([(0, 5), (1, 2), (2, 6), (3, 8), (4, 1), (5, 3), (6, 4)]),
        )
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
        let mut t1 = Tensor::default();
        let t1_tensors = vec![
            Tensor::new(vec![4, 3, 2]),
            Tensor::new(vec![0, 1, 3, 2]),
            Tensor::new(vec![4, 5, 6]),
        ];
        t1.push_tensors(t1_tensors, Some(&bond_dims));

        let mut t2 = Tensor::default();
        let t2_tensors = vec![Tensor::new(vec![5, 6, 8]), Tensor::new(vec![7, 8, 9])];
        t2.push_tensors(t2_tensors, Some(&bond_dims));
        create_tensor_network(vec![t1, t2], &bond_dims)
    }

    fn setup_parallel() -> Tensor {
        create_tensor_network(
            vec![
                Tensor::new(vec![4, 3, 2]),
                Tensor::new(vec![0, 1, 3, 2]),
                Tensor::new(vec![4, 5, 6]),
                Tensor::new(vec![5, 6]),
            ],
            &FxHashMap::from_iter([(0, 5), (1, 2), (2, 6), (3, 8), (4, 1), (5, 3), (6, 4)]),
        )
    }

    #[test]
    fn test_contract_path_cost() {
        let tn = setup_simple();
        let (op_cost, mem_cost) = contract_path_cost(tn.tensors(), path![(0, 1), (0, 2)], false);
        assert_eq!(op_cost, 4540f64);
        assert_eq!(mem_cost, 538f64);
        let (op_cost, mem_cost) = contract_path_cost(tn.tensors(), path![(0, 2), (0, 1)], false);
        assert_eq!(op_cost, 49296f64);
        assert_eq!(mem_cost, 1176f64);
    }

    #[test]
    fn test_contract_complex_path_cost() {
        let tn = setup_complex();
        let (op_cost, mem_cost) = contract_path_cost(
            tn.tensors(),
            path![(0, [(0, 1), (0, 2)]), (1, [(0, 1)]), (0, 1)],
            false,
        );
        assert_eq!(op_cost, 11188f64);
        assert_eq!(mem_cost, 538f64);
    }

    #[test]
    fn test_contract_path_cost_only_ops() {
        let tn = setup_simple();
        let (op_cost, mem_cost) = contract_path_cost(tn.tensors(), path![(0, 1), (0, 2)], true);
        assert_eq!(op_cost, 600f64);
        assert_eq!(mem_cost, 538f64);
        let (op_cost, mem_cost) = contract_path_cost(tn.tensors(), path![(0, 2), (0, 1)], true);
        assert_eq!(op_cost, 6336f64);
        assert_eq!(mem_cost, 1176f64);
    }

    #[test]
    fn test_contract_path_complex_cost_only_ops() {
        let tn = setup_complex();
        let (op_cost, mem_cost) = contract_path_cost(
            tn.tensors(),
            path![(0, [(0, 1), (0, 2)]), (1, [(0, 1)]), (0, 1)],
            true,
        );
        assert_eq!(op_cost, 1464f64);
        assert_eq!(mem_cost, 538f64);
    }

    #[test]
    fn test_communication_path_cost_only_ops() {
        let tn = setup_parallel();
        let (op_cost, mem_cost) =
            communication_path_cost(tn.tensors(), path![(0, 1), (2, 3), (0, 2)], true, None);
        assert_eq!(op_cost, 490f64);
        assert_eq!(mem_cost, 538f64);
    }

    #[test]
    fn test_communication_path_cost() {
        let tn = setup_parallel();
        let (op_cost, mem_cost) =
            communication_path_cost(tn.tensors(), path![(0, 1), (2, 3), (0, 1)], false, None);
        assert_eq!(op_cost, 7564f64);
        assert_eq!(mem_cost, 538f64);
    }

    #[test]
    fn test_communication_path_cost_only_ops_with_partition_cost() {
        let tn = setup_parallel();
        let tensor_cost = vec![20f64, 30f64, 80f64, 10f64];
        let (op_cost, mem_cost) = communication_path_cost(
            tn.tensors(),
            path![(0, 1), (2, 3), (0, 2)],
            true,
            Some(&tensor_cost),
        );
        assert_eq!(op_cost, 520f64);
        assert_eq!(mem_cost, 538f64);
    }

    #[test]
    fn test_communication_path_cost_with_partition_cost() {
        let tn = setup_parallel();
        let tensor_cost = vec![20f64, 30f64, 80f64, 10f64];
        let (op_cost, mem_cost) = communication_path_cost(
            tn.tensors(),
            path![(0, 1), (2, 3), (0, 1)],
            false,
            Some(&tensor_cost),
        );
        assert_eq!(op_cost, 7594f64);
        assert_eq!(mem_cost, 538f64);
    }
}
