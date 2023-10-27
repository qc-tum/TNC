extern crate num;

use std::convert::From;

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
    Path((usize, Vec<ContractionIndex>)),
}

impl From<(i32, i32)> for ContractionIndex {
    fn from(value: (i32, i32)) -> Self {
        ContractionIndex::Pair(value.0 as usize, value.1 as usize)
    }
}

#[macro_export]
macro_rules! path {
    ($index:expr, [$(($l:expr, $r:expr)),*]) => {
        ContractionIndex::Path(($index, vec![$(pair![$l, $r]),*]))
    };
    ($(($index:expr, $($tokens:tt),*)),*) => {
        vec![$(path![$index, $($tokens),*]),*]
    };
    ($e:expr, $p:expr) => {
        ContractionIndex::Pair($e, $p)
    };
}

#[macro_export]
macro_rules! pair {
    ($e:expr, $p:expr) => {
        ContractionIndex::Pair($e, $p)
    };
}

#[cfg(test)]
mod tests {
    use crate::types::ContractionIndex;

    #[test]
    fn test_path_macro() {
        assert_eq!(
            vec![
                ContractionIndex::Pair(0, 1),
                ContractionIndex::Path((
                    2,
                    vec![ContractionIndex::Pair(1, 2), ContractionIndex::Pair(1, 3)]
                )),
                ContractionIndex::Pair(0, 2),
                ContractionIndex::Path((
                    3,
                    vec![
                        ContractionIndex::Pair(4, 1),
                        ContractionIndex::Pair(3, 4),
                        ContractionIndex::Pair(3, 5)
                    ]
                )),
                ContractionIndex::Pair(0, 3),
            ],
            path![
                (0, 1),
                (2, [(1, 2), (1, 3)]),
                (0, 2),
                (3, [(4, 1), (3, 4), (3, 5)]),
                (0, 3)
            ]
        );
    }
}
