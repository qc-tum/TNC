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
/// * `path` - Output path as Vec<(usize, usize, usize)> after an [optimize_path] call.
/// * `n` - Number of initial input tensors.
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
/// * `n` - Number of initial input tensors.

/// # Returns
///
/// Identical path that replaces the left input tensor upon contraction
pub(super) fn ssa_replace_ordering(
    path: &[ContractionIndex],
    mut n: usize,
) -> Vec<ContractionIndex> {
    let mut ssa_path = Vec::with_capacity(path.len());
    let mut hs = HashMap::new();
    for tup in path.iter() {
        match tup {
            ContractionIndex::Pair(tup0, tup1) => {
                let new_tup0 = *hs.get(tup0).unwrap_or(tup0);
                let new_tup1 = *hs.get(tup1).unwrap_or(tup1);

                hs.entry(n).or_insert(new_tup0);
                ssa_path.push(pair!(new_tup0, new_tup1));
                n += 1;
            }
            ContractionIndex::Path(index, path) => {
                ssa_path.push(ContractionIndex::Path(*index, path.clone()));
            }
        }
    }
    ssa_path
}

#[cfg(test)]
mod tests {

    use crate::{
        contractionpath::{ssa_ordering, ssa_replace_ordering},
        path,
    };

    #[test]
    fn test_ssa_ordering() {
        let path = vec![
            (0, 3, 15),
            (1, 2, 44),
            (6, 4, 8),
            (5, 15, 22),
            (8, 44, 12),
            (12, 22, 99),
        ];
        let new_path = ssa_ordering(&path, 7);

        assert_eq!(
            new_path,
            path![(0, 3), (1, 2), (6, 4), (5, 7), (9, 8), (11, 10)]
        )
    }

    #[test]
    fn test_ssa_replace_ordering() {
        let path = path![(0, 3), (1, 2), (6, 4), (5, 7), (9, 8), (11, 10)];
        let new_path = ssa_replace_ordering(&path, 7);

        assert_eq!(
            new_path,
            path![(0, 3), (1, 2), (6, 4), (5, 0), (6, 1), (6, 5)]
        )
    }
}
