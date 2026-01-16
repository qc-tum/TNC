use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

use crate::tensornetwork::tensor::TensorIndex;
use crate::utils::traits::{HashMapInsertNew, WithCapacity};

mod candidates;
pub mod communication_schemes;
pub mod contraction_cost;
pub mod contraction_tree;
pub mod paths;
pub mod repartitioning;

/// A simple, flat contraction path. If you only need a reference, prefer
/// [`SimplePathRef`].
pub type SimplePath = Vec<(TensorIndex, TensorIndex)>;

/// Reference to a [`SimplePath`].
pub type SimplePathRef<'a> = &'a [(TensorIndex, TensorIndex)];

/// A (possibly nested) contraction path. It specifies the overall contraction path
/// to contract a tensor network, but also allows to specify additional contraction
/// paths for each tensor, in order to deal with composite tensors that have to be
/// contracted first.
#[derive(Debug, Clone, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct ContractionPath {
    /// Nested contraction paths for composite tensors.
    pub nested: FxHashMap<TensorIndex, ContractionPath>,
    /// The top-level contraction path for the tensor network itself.
    pub toplevel: SimplePath,
}

impl ContractionPath {
    /// Creates a contraction path with nested paths.
    #[inline]
    pub fn nested(
        nested: Vec<(TensorIndex, ContractionPath)>,
        toplevel: Vec<(TensorIndex, TensorIndex)>,
    ) -> Self {
        Self {
            nested: FxHashMap::from_iter(nested),
            toplevel,
        }
    }

    /// Creates a plain contraction path without nested paths.
    ///
    /// # Examples
    /// ```
    /// # use tnc::contractionpath::{ContractionPath, SimplePath};
    /// let path: SimplePath = vec![(0, 1), (0, 2), (0, 3)];
    /// let contraction_path = ContractionPath::simple(path.clone());
    /// assert!(contraction_path.is_simple());
    /// assert_eq!(contraction_path.toplevel, path);
    /// ```
    #[inline]
    pub fn simple(path: SimplePath) -> Self {
        Self {
            nested: FxHashMap::default(),
            toplevel: path,
        }
    }

    /// Creates a contraction path from a single contraction of two tensors.
    ///
    /// # Examples
    /// ```
    /// # use tnc::contractionpath::ContractionPath;
    /// let contraction_path = ContractionPath::single(0, 1);
    /// assert!(contraction_path.is_simple());
    /// assert_eq!(contraction_path.toplevel, vec![(0, 1)]);
    /// ```
    #[inline]
    pub fn single(a: TensorIndex, b: TensorIndex) -> Self {
        Self::simple(vec![(a, b)])
    }

    /// The length of the contraction path, that is, the number of top-level
    /// contractions.
    ///
    /// # Examples
    /// ```
    /// # use tnc::contractionpath::ContractionPath;
    /// let contraction_path = ContractionPath::simple(vec![(0, 1), (0, 2), (0, 3)]);
    /// assert_eq!(contraction_path.len(), 3);
    /// ```
    #[inline]
    pub fn len(&self) -> usize {
        self.toplevel.len()
    }

    /// Whether there are any top-level contractions in this contraction path.
    ///
    /// # Examples
    /// ```
    /// # use tnc::contractionpath::ContractionPath;
    /// assert!(ContractionPath::default().is_empty());
    /// assert!(!ContractionPath::simple(vec![(0, 1)]).is_empty());
    /// ```
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.toplevel.is_empty()
    }

    /// Returns whether this path has no nested paths.
    ///
    /// # Examples
    /// ```
    /// # use tnc::contractionpath::ContractionPath;
    /// # use tnc::path;
    /// let simple_path = path![(0, 1), (0, 2), (0, 3)];
    /// assert!(simple_path.is_simple());
    /// let nested_path = path![{(2, [(0, 2), (0, 1)])}, (0, 1), (0, 2)];
    /// assert!(!nested_path.is_simple());
    /// ```
    #[inline]
    pub fn is_simple(&self) -> bool {
        self.nested.is_empty()
    }

    /// Converts this path to its toplevel component.
    ///
    /// # Panics
    /// - Panics when this path has nested components
    ///
    /// # Examples
    /// ```
    /// # use tnc::contractionpath::ContractionPath;
    /// # use tnc::path;
    /// let contractions = vec![(0, 1), (0, 2), (0, 3)];
    /// let simple_path = ContractionPath::simple(contractions.clone());
    /// assert_eq!(simple_path.into_simple(), contractions);
    /// ```
    #[inline]
    pub fn into_simple(self) -> SimplePath {
        assert!(self.is_simple());
        self.toplevel
    }
}

/// Macro to create (nested) contraction paths, assuming the left tensor is replaced
/// in each contraction.
///
/// For instance, `path![{(2, [(0, 2), (0, 1)])}, (0, 1), (0, 2)]` creates a nested
/// contraction path that
/// - recursively contracts the composite tensor 2 with the contraction path `[(0, 2), (0, 1)]`
/// - contracts tensors 0 and 1, replacing tensor 0 with the result
/// - contracts tensors 0 and (now contracted) tensor 2, replacing tensor 0 with the result
#[macro_export]
macro_rules! path {
    [] => {
        $crate::contractionpath::ContractionPath::default()
    };
    [$( ($t0:expr, $t1:expr) ),*] => {
        $crate::contractionpath::ContractionPath::simple(vec![$( ($t0, $t1) ),*])
    };
    [ { $( ( $index:expr, [ $( $tok:tt )* ] ) ),* $(,)? } $(, ($t0:expr, $t1:expr) )* ] => {
        $crate::contractionpath::ContractionPath::nested(
            vec![ $( ($index, path![ $( $tok )* ]) ),* ],
            vec![ $( ($t0, $t1) ),* ]
        )
    };
}

/// The contraction ordering labels [`Tensor`] objects from each possible contraction with a
/// unique identifier in SSA format. As only a subset of these [`Tensor`] objects are seen in
/// a contraction path, the tensors in the optimal path search are not sequential. This converts
/// the output to strictly obey an SSA format.
///
/// # Arguments
/// * `path` - Output path as `&[(usize, usize, usize)]` after an `find_path` call.
/// * `n` - Number of initial input tensors.
///
/// # Returns
/// Identical path using SSA format
fn ssa_ordering(path: &[(usize, usize, usize)], mut n: usize) -> ContractionPath {
    let mut ssa_path = Vec::with_capacity(path.len());
    let mut hs = FxHashMap::with_capacity(path.len());
    let path_len = n;
    for (u1, u2, u3) in path {
        let t1 = if *u1 >= path_len { hs[u1] } else { *u1 };
        let t2 = if *u2 >= path_len { hs[u2] } else { *u2 };
        hs.entry(*u3).or_insert(n);
        n += 1;
        ssa_path.push((t1, t2));
    }
    ContractionPath::simple(ssa_path)
}

/// Accepts a contraction `path` that is in SSA format and returns a contraction path
/// assuming that all contracted tensors replace the left input tensor and no tensor
/// is popped.
pub(super) fn ssa_replace_ordering(path: &ContractionPath) -> ContractionPath {
    let nested = path
        .nested
        .iter()
        .map(|(index, local_path)| (*index, ssa_replace_ordering(local_path)))
        .collect();

    let mut hs = FxHashMap::with_capacity(path.len());
    let mut toplevel = Vec::with_capacity(path.len());
    let mut n = path.len() + 1;
    for (t0, t1) in &path.toplevel {
        let new_t0 = *hs.get(t0).unwrap_or(t0);
        let new_t1 = *hs.get(t1).unwrap_or(t1);

        hs.insert_new(n, new_t0);
        toplevel.push((new_t0, new_t1));
        n += 1;
    }

    ContractionPath { nested, toplevel }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_path_simple_macro() {
        assert_eq!(
            path![{ (2, [(1, 2), (1, 3)]) }, (0, 1)],
            ContractionPath {
                nested: FxHashMap::from_iter([(2, ContractionPath::simple(vec![(1, 2), (1, 3)]))]),
                toplevel: vec![(0, 1)]
            }
        );
    }

    #[test]
    fn test_path_macro() {
        assert_eq!(
            path![
                {
                (2, [(1, 2), (1, 3)]),
                (4, [{(2, [(1, 2), (1, 3)])}, (1, 3)]),
                (5, [(1, 2), (1, 3)]),
                (3, [(4, 1), (3, 4), (3, 5)]),
                },
                (0, 1),
                (0, 2),
                (0, 3)
            ],
            ContractionPath {
                nested: FxHashMap::from_iter([
                    (2, ContractionPath::simple(vec![(1, 2), (1, 3)])),
                    (3, ContractionPath::simple(vec![(4, 1), (3, 4), (3, 5)])),
                    (
                        4,
                        ContractionPath {
                            nested: FxHashMap::from_iter([(
                                2,
                                ContractionPath::simple(vec![(1, 2), (1, 3)])
                            )]),
                            toplevel: vec![(1, 3)]
                        }
                    ),
                    (5, ContractionPath::simple(vec![(1, 2), (1, 3)]))
                ]),
                toplevel: vec![(0, 1), (0, 2), (0, 3)]
            }
        );
    }

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
        let new_path = ssa_replace_ordering(&path);

        assert_eq!(
            new_path,
            path![(0, 3), (1, 2), (6, 4), (5, 0), (6, 1), (6, 5)]
        );
    }

    #[test]
    fn test_ssa_replace_ordering_nested() {
        let path = path![
            {
            (1, [(2, 1), (0, 3)]),
            (6, [(0, 2), (1, 3), (4, 5)])
            },
            (0, 3),
            (1, 2),
            (6, 4),
            (5, 7),
            (9, 8),
            (11, 10)
        ];

        let new_path = ssa_replace_ordering(&path);

        assert_eq!(
            new_path,
            path![
                {
                (1, [(2, 1), (0, 2)]),
                (6, [(0, 2), (1, 3), (0, 1)])
                },
                (0, 3),
                (1, 2),
                (6, 4),
                (5, 0),
                (6, 1),
                (6, 5)
            ]
        );
    }
}
