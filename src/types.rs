use serde::{Deserialize, Serialize};

pub type EdgeIndex = usize;
pub type TensorIndex = usize;

#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub enum ContractionIndex {
    Pair(TensorIndex, TensorIndex),
    Path(TensorIndex, Vec<ContractionIndex>),
}

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
    use crate::types::ContractionIndex;

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
