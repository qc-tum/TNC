use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use serde::{Deserialize, Serialize};

use crate::tensornetwork::tensor::Tensor;
// use std::hash::DefaultHasher;

pub type EdgeIndex = usize;
pub type TensorIndex = usize;

/// Information about which legs are to be sliced.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct Slicing {
    pub slices: Vec<EdgeIndex>,
}

impl Slicing {
    /// Computes the size of the slice when applied to the `target` tensor.
    /// This is the product of all sliced legs.
    pub fn size(&self, target: &Tensor) -> u64 {
        self.slices
            .iter()
            .map(|leg| target.bond_dims()[leg])
            .product()
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub enum ContractionIndex {
    Pair(TensorIndex, TensorIndex),
    Path(TensorIndex, Option<Slicing>, Vec<ContractionIndex>),
}

#[macro_export]
macro_rules! path {
    ($(($index:expr, $($tokens:tt),*)),*) => {
        &[$(path![$index, $($tokens),*]),*]
    };
    ($index:expr, [$($tokens:tt),+]) => {
        $crate::types::ContractionIndex::Path($index, None, path![$($tokens),+].to_vec())
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
                (4, [(2, [(1, 2), (1, 3)]), (1, 3)]),
                (0, 2),
                (3, [(4, 1), (3, 4), (3, 5)]),
                (0, 3)
            ],
            &[
                ContractionIndex::Pair(0, 1),
                ContractionIndex::Path(
                    2,
                    None,
                    vec![ContractionIndex::Pair(1, 2), ContractionIndex::Pair(1, 3)]
                ),
                ContractionIndex::Path(
                    4,
                    None,
                    vec![
                        ContractionIndex::Path(
                            2,
                            None,
                            vec![ContractionIndex::Pair(1, 2), ContractionIndex::Pair(1, 3)]
                        ),
                        ContractionIndex::Pair(1, 3)
                    ]
                ),
                ContractionIndex::Pair(0, 2),
                ContractionIndex::Path(
                    3,
                    None,
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
