use std::sync::Arc;

use crate::tensornetwork::tensor::Tensor;
use crate::types::ContractionIndex;

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
/// let tn = create_tensor_network(vec![Tensor::new(vec1), Tensor::new(vec2)], &bond_dims, None);
/// assert_eq!(contract_cost_tensors(&tn.tensor(0), &tn.tensor(1)), 350350f64);
/// ```
pub fn contract_cost_tensors(t_1: &Tensor, t_2: &Tensor) -> f64 {
    let final_dims = t_1 ^ t_2;
    let shared_dims = t_1 & t_2;
    let bond_dims = t_1.bond_dims();
    let single_loop_cost = shared_dims
        .legs
        .iter()
        .map(|e| bond_dims[e] as f64)
        .product::<f64>();

    (single_loop_cost - 1f64).mul_add(2f64, single_loop_cost * 6f64)
        * final_dims
            .legs
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
/// let tn = create_tensor_network(vec![Tensor::new(vec1), Tensor::new(vec2)], &bond_dims, None);
/// assert_eq!(contract_op_cost_tensors(&tn.tensor(0), &tn.tensor(1)), 45045f64);
/// ```
pub fn contract_op_cost_tensors(t_1: &Tensor, t_2: &Tensor) -> f64 {
    let all_dims = t_1 | t_2;
    let bond_dims = t_1.bond_dims();

    all_dims
        .legs
        .iter()
        .map(|e| bond_dims[e] as f64)
        .product::<f64>()
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
/// let tn = create_tensor_network(vec![Tensor::new(vec1), Tensor::new(vec2)], &bond_dims, None);
/// assert_eq!(contract_size_tensors(&tn.tensor(0), &tn.tensor(1)), 6607f64);
/// ```
#[inline]
pub fn contract_size_tensors(t_1: &Tensor, t_2: &Tensor) -> f64 {
    let diff = t_1 ^ t_2;
    diff.size() as f64 + t_1.size() as f64 + t_2.size() as f64
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
    contract_path_custom_cost(inputs, contract_path, cost_function)
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
    cost_function: fn(&Tensor, &Tensor) -> f64,
) -> (f64, f64) {
    let mut op_cost = 0f64;
    let mut mem_cost = 0f64;
    let mut inputs = inputs.to_vec();

    for index in contract_path {
        match *index {
            ContractionIndex::Pair(i, j) => {
                op_cost += cost_function(&inputs[i], &inputs[j]);
                let ij = &inputs[i] ^ &inputs[j];
                let new_mem_cost = contract_size_tensors(&inputs[i], &inputs[j]);
                mem_cost = mem_cost.max(new_mem_cost);
                inputs[i] = ij;
            }
            ContractionIndex::Path(i, ref path) => {
                let costs = contract_path_custom_cost(inputs[i].tensors(), path, cost_function);
                op_cost += costs.0;
                mem_cost = mem_cost.max(costs.1);
                let intermediate_tensor = Tensor::new_with_bonddims(
                    inputs[i].external_edges(),
                    Arc::clone(&inputs[i].bond_dims),
                );
                inputs[i] = intermediate_tensor;
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
    cost_function: fn(&Tensor, &Tensor) -> f64,
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
                let new_mem_cost = contract_size_tensors(&inputs[i], &inputs[j]);
                mem_cost = mem_cost.max(new_mem_cost);

                op_cost =
                    cost_function(&inputs[i], &inputs[j]) + tensor_cost[i].max(tensor_cost[j]);
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
            None,
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
        t1.push_tensors(t1_tensors, Some(&bond_dims), None);

        let mut t2 = Tensor::default();
        let t2_tensors = vec![Tensor::new(vec![5, 6, 8]), Tensor::new(vec![7, 8, 9])];
        t2.push_tensors(t2_tensors, Some(&bond_dims), None);
        create_tensor_network(vec![t1, t2], &bond_dims, None)
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
            None,
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
    fn test_contract_path_op_cost() {
        let tn = setup_simple();
        let (op_cost, mem_cost) = contract_path_cost(tn.tensors(), path![(0, 1), (0, 2)], true);
        assert_eq!(op_cost, 600f64);
        assert_eq!(mem_cost, 538f64);
        let (op_cost, mem_cost) = contract_path_cost(tn.tensors(), path![(0, 2), (0, 1)], true);
        assert_eq!(op_cost, 6336f64);
        assert_eq!(mem_cost, 1176f64);
    }

    #[test]
    fn test_contract_path_complex_op_cost() {
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
    fn test_contract_parallel_path_op_cost() {
        let tn = setup_parallel();
        let (op_cost, mem_cost) =
            communication_path_cost(tn.tensors(), path![(0, 1), (2, 3), (0, 2)], true, None);
        assert_eq!(op_cost, 490f64);
        assert_eq!(mem_cost, 538f64);
    }

    #[test]
    fn test_contract_parallel_path_cost() {
        let tn = setup_parallel();
        let (op_cost, mem_cost) =
            communication_path_cost(tn.tensors(), path![(0, 1), (2, 3), (0, 1)], false, None);
        assert_eq!(op_cost, 7564f64);
        assert_eq!(mem_cost, 538f64);
    }

    #[test]
    fn test_contract_parallel_path_op_cost_with_partition_cost() {
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
    fn test_contract_parallel_path_cost_with_partition_cost() {
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
