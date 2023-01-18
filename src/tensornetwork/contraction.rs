use crate::tensornetwork::TensorNetwork;
use itertools::Itertools;
use taco_sys::{contract, multicontract, Tensor as _TacoTensor};

/// Fully contracts a list of [_TacoTensor] objects based on a given contraction path using repeated SSA format.
///
/// # Arguments
///
/// * `tn` - [TensorNetwork] to be contracted
/// * `d_tn` - [Vector] of [_TacoTensor] objects containing data of [TensorNetwork]
/// * `contract_path` - [Vector] of [(usize, usize)], indicating contraction path. See [BranchBound] for details on `contract_path` format.
///
/// # Examples
///
/// ```
/// # extern crate tensorcontraction;
/// # use tensorcontraction::{
///     contractionpath::paths::{BranchBound, BranchBoundType, OptimizePath},
///     random::tensorgeneration::{random_sparse_tensor, random_tensor_network},
///     tensornetwork::{tensor::Tensor, TensorNetwork},
///     tensornetwork::contraction::tn_contract,
/// };
/// 
/// let r_tn = random_tensor_network(2, 3);
/// let mut d_tn = Vec::new();
/// for r_t in r_tn.get_tensors() {
///     d_tn.push(random_sparse_tensor(
///         r_t.clone(),
///         &r_tn.get_bond_dims(),
///         None,
///    ));
/// }
/// let mut opt = BranchBound::new(r_tn.clone(), None, 20, BranchBoundType::Flops);
/// opt._optimize_path(None);
/// let opt_path = opt.get_best_replace_path();
/// tn_contract(r_tn, d_tn, &opt_path);
/// ```
pub fn tn_contract(
    mut tn: TensorNetwork,
    mut d_tn: Vec<_TacoTensor>,
    contract_path: &Vec<(usize, usize)>,
) -> (TensorNetwork, Vec<_TacoTensor>) {
    let mut last_index = 0;
    for (i, j) in contract_path {
        let a_legs = tn[*i].get_legs().clone();
        let b_legs = tn[*j].get_legs().clone();
        let (_tensor_intersection, tensor_difference) = tn._contraction(i, j);
        let bond_dims = tn.get_bond_dims().clone();
        let out_dims = tensor_difference
            .clone()
            .iter()
            .map(|e| bond_dims[e] as i32)
            .collect::<Vec<i32>>();

        let mut new_tensor = _TacoTensor::new(&out_dims.clone());

        contract(
            &tensor_difference,
            &mut new_tensor,
            &a_legs,
            &d_tn[*i],
            &b_legs,
            &d_tn[*j],
        );
        d_tn[*i] = new_tensor;
        d_tn[*j] = _TacoTensor::new(&[1]);
        last_index = *i;
    }
    d_tn.swap(0, last_index);
    d_tn.drain(1..d_tn.len());
    (tn, d_tn)
}

/// Fully contracts a list of [_TacoTensor] objects using the `taco-sys` [multicontract] function.
///
/// # Arguments
///
/// * `tn` - [TensorNetwork] to be contracted
/// * `d_tn` - [Vector] of [_TacoTensor] objects containing data of [TensorNetwork]
///
/// # Examples
///
/// ```
/// # extern crate tensorcontraction;
/// # use tensorcontraction::{
///     contractionpath::paths::{BranchBound, BranchBoundType, OptimizePath},
///     random::tensorgeneration::{random_sparse_tensor, random_tensor_network},
///     tensornetwork::{tensor::Tensor, TensorNetwork},
///     tensornetwork::contraction::tn_multicontract,
/// };
/// let r_tn = random_tensor_network(2, 3);
/// let mut d_tn = Vec::new();
/// for r_t in r_tn.get_tensors() {
///     d_tn.push(random_sparse_tensor(
///         r_t.clone(),
///         &r_tn.get_bond_dims(),
///         None,
///    ));
/// }
/// tn_multicontract(r_tn, d_tn);
/// ```
pub fn tn_multicontract(
    tn: TensorNetwork,
    mut d_tn: Vec<_TacoTensor>,
) -> (TensorNetwork, Vec<_TacoTensor>) {
    let input_indices = tn
        .get_tensors()
        .iter()
        .map(|e| e.get_legs().as_slice())
        .collect_vec();

    let mut output_indices = Vec::new();
    for (i, j) in tn.get_edges() {
        if (*j).1.is_none() {
            output_indices.push(*i);
        }
    }
    let output_indices_size = output_indices
        .iter()
        .map(|e| *(tn.get_bond_dims().get(e).unwrap()) as i32)
        .collect_vec();

    let mut new_tensor = _TacoTensor::new(&output_indices_size);

    multicontract(
        &output_indices,
        &mut new_tensor,
        input_indices.as_slice(),
        d_tn.iter().collect_vec().as_slice(),
    );
    // d_tn.swap(0, last_index);
    d_tn[0] = new_tensor;
    d_tn.drain(1..d_tn.len());
    (tn, d_tn)
}
