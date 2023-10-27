use std::collections::HashMap;

use crate::{pair, types::ContractionIndex};
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
/// * `path` - Output path as Vec<(usize, usize)> after an [optimize_path] call.
/// # Returns
///
/// Identical path using ssa format
fn ssa_ordering(path: &Vec<(usize, usize, usize)>, mut n: usize) -> Vec<ContractionIndex> {
    let mut ssa_path = Vec::with_capacity(path.len());
    let mut hs = HashMap::new();
    let path_len = n;
    for (u1, u2, u3) in path {
        let t1 = if *u1 > path_len { hs[u1] } else { *u1 };
        let t2 = if *u2 > path_len { hs[u2] } else { *u2 };
        hs.entry(*u3).or_insert(n);
        n += 1;
        ssa_path.push(pair!(t1, t2));
    }
    ssa_path
}

/// Accepts a contraction path that is in ssa format and returns a contraction path
/// assuming that all contracted tensors replace the left input tensor.
///
/// # Arguments
///
/// * `path` - Output path as Vec<(usize, usize)> that is in ssa format.
/// # Returns
///
/// Identical path that replaces the left input tensor upon contraction
pub(super) fn ssa_replace_ordering(
    path: &Vec<ContractionIndex>,
    mut n: usize,
) -> Vec<ContractionIndex> {
    let mut ssa_path = Vec::with_capacity(path.len());
    let mut hs = HashMap::new();
    for tup in path.iter() {
        match tup {
            ContractionIndex::Pair(tup0, tup1) => {
                let mut new_tup0 = tup0;
                let mut new_tup1 = tup1;
                if hs.contains_key(tup0) {
                    new_tup0 = hs[tup0];
                }
                if hs.contains_key(tup1) {
                    new_tup1 = hs[tup1];
                }
                hs.entry(n).or_insert(new_tup0);
                ssa_path.push(pair!(*new_tup0, *new_tup1));
                n += 1;
            }
            ContractionIndex::Path(path) => {
                ssa_path.push(ContractionIndex::Path(path.clone()));
            }
        }
    }
    ssa_path
}
