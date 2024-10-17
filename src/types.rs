use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use serde::{Deserialize, Serialize};
// use std::hash::DefaultHasher;

pub type EdgeIndex = usize;
pub type TensorIndex = usize;

#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd)]
pub enum Vertex {
    Open,
    Closed(TensorIndex),
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub enum ContractionIndex {
    Pair(usize, usize),
    Path(usize, Vec<ContractionIndex>),
}

#[macro_export]
macro_rules! path {
    ($index:expr, [$(($l:expr, $r:expr)),*]) => {
        $crate::types::ContractionIndex::Path($index, vec![$($crate::pair![$l, $r]),*])
    };
    ($(($index:expr, $($tokens:tt),*)),*) => {
        &[$(path![$index, $($tokens),*]),*]
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

pub(crate) fn calculate_hash<T: Hash>(t: &T) -> u64 {
    let mut s = DefaultHasher::new();
    t.hash(&mut s);
    s.finish()
}

#[cfg(test)]
mod tests {
    use crate::types::ContractionIndex;

    #[test]
    fn test_path_macro() {
        assert_eq!(
            path![
                (0, 1),
                (2, [(1, 2), (1, 3)]),
                (0, 2),
                (3, [(4, 1), (3, 4), (3, 5)]),
                (0, 3)
            ],
            &[
                ContractionIndex::Pair(0, 1),
                ContractionIndex::Path(
                    2,
                    vec![ContractionIndex::Pair(1, 2), ContractionIndex::Pair(1, 3)]
                ),
                ContractionIndex::Pair(0, 2),
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
