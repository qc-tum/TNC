use serde::{Deserialize, Serialize};

/// Unique index of a leg.
pub type EdgeIndex = usize;

/// Index of a tensor in a tensor network.
pub type TensorIndex = usize;

/// Element of a contraction path.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub enum ContractionIndex {
    /// A top-level contraction between two tensors, with the result replacing the
    /// left tensor.
    Pair(TensorIndex, TensorIndex),
    /// A nested contraction path for the specified composite tensor. The composite
    /// tensor is replaced with the tensor resulting from contracting the composite
    /// tensor with the given path.
    Path(TensorIndex, Vec<ContractionIndex>),
}

/// Macro to create (nested) contraction paths, assuming the left tensor is replaced
/// in each contraction.
///
/// For instance, `path![(0, 1), (2, [(0, 2), (0, 1)]), (0, 2)]` creates a nested
/// contraction path that
/// - contracts tensors 0 and 1, replacing tensor 0 with the result
/// - recursively contracts the composite tensor 2 with the contraction path `[(0, 2), (0, 1)]`
/// - contracts tensors 0 and (now contracted) tensor 2, replacing tensor 0 with the result
#[macro_export]
macro_rules! path {
    ($(($index:expr $(,$tokens:tt)*)),*) => {
        &[$(path![$index $(,$tokens)*]),*]
    };
    ($index:expr, []) => {
        $crate::types::ContractionIndex::Path($index, vec![])
    };
    ($index:expr, [$($tokens:tt),+]) => {
        $crate::types::ContractionIndex::Path($index, path![$($tokens),+].to_vec())
    };
    ($e:expr, $p:expr) => {
        $crate::types::ContractionIndex::Pair($e, $p)
    };
}

#[macro_export]
macro_rules! pair {
    ($e:expr, $p:expr) => {
        $crate::types::ContractionIndex::Pair($e, $p)
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_path_simple_macro() {
        assert_eq!(
            path![(0, 1), (2, [(1, 2), (1, 3)]), (2, [])],
            &[
                ContractionIndex::Pair(0, 1),
                ContractionIndex::Path(
                    2,
                    vec![ContractionIndex::Pair(1, 2), ContractionIndex::Pair(1, 3)]
                ),
                ContractionIndex::Path(2, vec![]),
            ]
        );
    }

    #[test]
    fn test_path_macro() {
        assert_eq!(
            path![
                (0, 1),
                (2, [(1, 2), (1, 3)]),
                (4, [(2, [(1, 2), (1, 3)]), (1, 3)]),
                (5, [(1, 2), (1, 3)]),
                (0, 2),
                (6, []),
                (3, [(4, 1), (3, 4), (3, 5)]),
                (0, 3)
            ],
            &[
                ContractionIndex::Pair(0, 1),
                ContractionIndex::Path(
                    2,
                    vec![ContractionIndex::Pair(1, 2), ContractionIndex::Pair(1, 3)]
                ),
                ContractionIndex::Path(
                    4,
                    vec![
                        ContractionIndex::Path(
                            2,
                            vec![ContractionIndex::Pair(1, 2), ContractionIndex::Pair(1, 3)]
                        ),
                        ContractionIndex::Pair(1, 3)
                    ]
                ),
                ContractionIndex::Path(
                    5,
                    vec![ContractionIndex::Pair(1, 2), ContractionIndex::Pair(1, 3)]
                ),
                ContractionIndex::Pair(0, 2),
                ContractionIndex::Path(6, vec![]),
                ContractionIndex::Path(
                    3,
                    vec![
                        ContractionIndex::Pair(4, 1),
                        ContractionIndex::Pair(3, 4),
                        ContractionIndex::Pair(3, 5)
                    ]
                ),
                ContractionIndex::Pair(0, 3),
            ]
        );
    }
}
