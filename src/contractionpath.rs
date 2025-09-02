use rustc_hash::FxHashMap;

use crate::pair;
use crate::types::ContractionIndex;
use crate::utils::traits::{HashMapInsertNew, WithCapacity};
mod candidates;
pub mod communication_schemes;
pub mod contraction_cost;
pub mod contraction_tree;
pub mod paths;
pub mod repartitioning;

/// The contraction ordering labels [`Tensor`] objects from each possible contraction with a
/// unique identifier in SSA format. As only a subset of these [`Tensor`] objects are seen in
/// a contraction path, the tensors in the optimal path search are not sequential. This converts
/// the output to strictly obey an SSA format.
///
/// # Arguments
/// * `path` - Output path as `&[(usize, usize, usize)]` after an `optimize_path` call.
/// * `n` - Number of initial input tensors.
///
/// # Returns
/// Identical path using SSA format
fn ssa_ordering(path: &[(usize, usize, usize)], mut n: usize) -> Vec<ContractionIndex> {
    let mut ssa_path = Vec::with_capacity(path.len());
    let mut hs = FxHashMap::with_capacity(path.len());
    let path_len = n;
    for (u1, u2, u3) in path {
        let t1 = if *u1 >= path_len { hs[u1] } else { *u1 };
        let t2 = if *u2 >= path_len { hs[u2] } else { *u2 };
        hs.entry(*u3).or_insert(n);
        n += 1;
        ssa_path.push(pair!(t1, t2));
    }
    ssa_path
}

/// Accepts a contraction `path` that is in SSA format (with `n` being the number of
/// initial input tensors) and returns a contraction path assuming that all
/// contracted tensors replace the left input tensor and no tensor is popped.
pub(super) fn ssa_replace_ordering(
    path: &[ContractionIndex],
    mut n: usize,
) -> Vec<ContractionIndex> {
    let mut replace_path = Vec::with_capacity(path.len());
    let mut hs = FxHashMap::with_capacity(path.len());
    for tup in path {
        match tup {
            ContractionIndex::Pair(t0, t1) => {
                let new_t0 = *hs.get(t0).unwrap_or(t0);
                let new_t1 = *hs.get(t1).unwrap_or(t1);

                hs.insert_new(n, new_t0);
                replace_path.push(pair!(new_t0, new_t1));
                n += 1;
            }
            ContractionIndex::Path(index, path) => {
                let k = path
                    .iter()
                    .filter(|n| matches!(n, ContractionIndex::Pair(_, _)))
                    .count()
                    + 1;
                let ssa_path = ssa_replace_ordering(path, k);
                replace_path.push(ContractionIndex::Path(*index, ssa_path));
            }
        }
    }
    replace_path
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::path;

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
        );
    }

    #[test]
    fn test_ssa_replace_ordering() {
        let path = path![(0, 3), (1, 2), (6, 4), (5, 7), (9, 8), (11, 10)];
        let new_path = ssa_replace_ordering(path, 7);

        assert_eq!(
            new_path,
            path![(0, 3), (1, 2), (6, 4), (5, 0), (6, 1), (6, 5)]
        );
    }

    #[test]
    fn test_ssa_replace_ordering_nested() {
        let path = path![
            (0, 3),
            (1, [(2, 1), (0, 3)]),
            (1, 2),
            (6, [(0, 2), (1, 3), (4, 5)]),
            (6, 4),
            (5, 7),
            (9, 8),
            (11, 10)
        ];

        let new_path = ssa_replace_ordering(path, 7);

        assert_eq!(
            new_path,
            path![
                (0, 3),
                (1, [(2, 1), (0, 2)]),
                (1, 2),
                (6, [(0, 2), (1, 3), (0, 1)]),
                (6, 4),
                (5, 0),
                (6, 1),
                (6, 5)
            ]
        );
    }
}
