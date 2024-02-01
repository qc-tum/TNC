use std::collections::HashMap;
pub mod candidates;
pub mod contraction_cost;
pub mod paths;
pub mod random_paths;
// pub mod optimizer;

/// The contraction ordering labels [Tensor] objects from each possible contraction with a
/// unique identifier in ssa format. As only a subset of these [Tensor] objects are seen in
/// a contraction path, the tensors in the optimal path search are not sequential. This converts
/// the output to strictly obey an ssa format.
///
/// # Arguments
///
/// * `path` - Output path as Vec<(usize, usize, usize)> after an [optimize_path] call.
/// * `n` - Number of initial input tensors.
/// # Returns
///
/// Identical path using ssa format
fn ssa_ordering(path: &Vec<(usize, usize, usize)>, mut n: usize) -> Vec<(usize, usize)> {
    let mut ssa_path = Vec::with_capacity(path.len());
    let mut hs = HashMap::new();
    let path_len = n;
    for (u1, u2, u3) in path {
        let t1 = if *u1 > path_len { hs[u1] } else { *u1 };
        let t2 = if *u2 > path_len { hs[u2] } else { *u2 };
        hs.entry(*u3).or_insert(n);
        n += 1;
        ssa_path.push((t1, t2));
    }
    ssa_path
}

/// Accepts a contraction path that is in ssa format and returns a contraction path
/// assuming that all contracted tensors replace the left input tensor.
///
/// # Arguments
///
/// * `path` - Output path as Vec<(usize, usize)> that is in ssa format.
/// * `n` - Number of initial input tensors.

/// # Returns
///
/// Identical path that replaces the left input tensor upon contraction
pub(super) fn ssa_replace_ordering(
    path: &Vec<(usize, usize)>,
    mut n: usize,
) -> Vec<(usize, usize)> {
    let mut ssa_path = Vec::with_capacity(path.len());
    let mut hs = HashMap::new();
    for tup in path.iter() {
        // let mut tup = path[i];
        let mut new_tup = *tup;
        if hs.contains_key(&tup.0) {
            new_tup.0 = hs[&tup.0];
        }
        if hs.contains_key(&tup.1) {
            new_tup.1 = hs[&tup.1];
        }
        hs.entry(n).or_insert(new_tup.0);
        ssa_path.push(new_tup);
        n += 1;
    }
    ssa_path
}
