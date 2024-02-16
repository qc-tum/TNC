use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
// use std::hash::DefaultHasher;

pub type EdgeIndex = usize;
pub type TensorIndex = usize;

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Vertex {
    Open,
    Closed(TensorIndex),
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum ContractionIndex {
    Pair(usize, usize),
    Path(usize, Vec<ContractionIndex>),
}

impl ContractionIndex {
    pub fn get_data(self) -> Vec<ContractionIndex> {
        match self {
            ContractionIndex::Pair(a, b) => vec![ContractionIndex::Pair(a, b)],
            ContractionIndex::Path(_a, b) => b,
        }
    }
}

#[macro_export]
macro_rules! path {
    ($index:expr, [$(($l:expr, $r:expr)),*]) => {
        $crate::types::ContractionIndex::Path($index, vec![$(pair![$l, $r]),*])
    };
    ($(($index:expr, $($tokens:tt),*)),*) => {
        vec![$(path![$index, $($tokens),*]),*]
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
            vec![
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
