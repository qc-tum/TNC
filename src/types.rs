use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use itertools::Itertools;
use serde::{Deserialize, Serialize};

use crate::tensornetwork::tensor::Tensor;

pub type EdgeIndex = usize;
pub type TensorIndex = usize;

/// Concrete information about which index to select from each sliced legs.
/// For each [`SlicingPlan`], there are multiple concrete [`SlicingTask`]s.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SlicingTask {
    pub slices: Vec<(EdgeIndex, usize)>,
}

/// Information about which legs are to be sliced.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SlicingPlan {
    pub slices: Vec<EdgeIndex>,
}

impl SlicingPlan {
    /// Gets the dimensions of the specified legs in the `target` (composite) tensor.
    fn get_bond_dims(target: &Tensor, searched_legs: &[EdgeIndex]) -> Vec<u64> {
        assert!(target.legs().is_empty());

        let mut dimensions = vec![None; searched_legs.len()];
        for tensor in target.tensors() {
            for (i, &searched_leg) in searched_legs.iter().enumerate() {
                if let Some((_, dim)) = tensor.edges().find(|(&leg, _)| leg == searched_leg) {
                    dimensions[i] = Some(*dim);
                }
            }
        }
        dimensions.into_iter().flatten().collect()
    }

    /// Computes the number of slices when applied to the `target` tensor.
    /// This is the product of all sliced legs' dimensions.
    ///
    /// # Examples
    /// ```
    /// # use tensorcontraction::types::SlicingPlan;
    /// # use tensorcontraction::tensornetwork::tensor::Tensor;
    /// # use rustc_hash::FxHashMap;
    /// let bond_dims = FxHashMap::from_iter([(0, 2), (1, 3)]);
    /// let t1 = Tensor::new_from_map(vec![0, 1], &bond_dims);
    /// let t2 = Tensor::new_from_map(vec![0, 1], &bond_dims);
    /// let mut tc = Tensor::new_composite(vec![t1, t2]);
    ///
    /// let plan = SlicingPlan { slices: vec![0, 1] };
    /// assert_eq!(plan.task_count(&tc), 6);
    /// ```
    pub fn task_count(&self, target: &Tensor) -> u64 {
        let sizes = SlicingPlan::get_bond_dims(target, &self.slices);
        sizes.into_iter().product()
    }

    /// Gets the specified [`SlicingTask`] from this plan. Each combination of
    /// indices of the sliced legs is a separate task. `task_index` has to be less
    /// than [`SlicingPlan::task_count`].
    ///
    /// # Examples
    /// ```
    /// # use tensorcontraction::types::SlicingPlan;
    /// # use tensorcontraction::tensornetwork::tensor::Tensor;
    /// # use rustc_hash::FxHashMap;
    /// let bond_dims = FxHashMap::from_iter([(0, 2), (1, 3)]);
    /// let t1 = Tensor::new_from_map(vec![0, 1], &bond_dims);
    /// let t2 = Tensor::new_from_map(vec![0, 1], &bond_dims);
    /// let mut tc = Tensor::new_composite(vec![t1, t2]);
    ///
    /// let plan = SlicingPlan { slices: vec![0, 1] };
    /// let task = plan.get_task(&tc, 0);
    /// assert_eq!(task.slices, vec![(0, 0), (1, 0)]);
    /// let task = plan.get_task(&tc, 2);
    /// assert_eq!(task.slices, vec![(0, 0), (1, 2)]);
    /// ```
    pub fn get_task(&self, target: &Tensor, task_index: usize) -> SlicingTask {
        let sizes = SlicingPlan::get_bond_dims(target, &self.slices);
        sizes
            .into_iter()
            .map(|dim| 0..dim as usize)
            .multi_cartesian_product()
            .nth(task_index)
            .map(|indices| SlicingTask {
                slices: self.slices.iter().copied().zip(indices).collect(),
            })
            .unwrap()
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub enum ContractionIndex {
    Pair(TensorIndex, TensorIndex),
    Path(TensorIndex, Option<SlicingPlan>, Vec<ContractionIndex>),
}

#[macro_export]
macro_rules! path {
    ($(($index:expr $(,$tokens:tt)*)),*) => {
        &[$(path![$index $(,$tokens)*]),*]
    };
    ($index:tt, []) => {
        $crate::types::ContractionIndex::Path($index, None, vec![])
    };
    ($index:expr, $slicing:expr, [$($tokens:tt),+]) => {
        $crate::types::ContractionIndex::Path($index, Some($crate::types::SlicingPlan {
            slices: $slicing.to_vec(),
        }), path![$($tokens),+].to_vec())
    };
    ($index:expr, [$($tokens:tt),+]) => {
        $crate::types::ContractionIndex::Path($index, None, path![$($tokens),+].to_vec())
    };
    ($e:expr, $p:expr) => {
        $crate::types::ContractionIndex::Pair($e, $p)
    };
    ($index:tt) => {
        $crate::types::ContractionIndex::Path($index, None, vec![])
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
    use crate::types::{ContractionIndex, SlicingPlan};

    #[test]
    fn test_path_simple_macro() {
        assert_eq!(
            path![(0, 1), (2, [(1, 2), (1, 3)]), (2, [])],
            &[
                ContractionIndex::Pair(0, 1),
                ContractionIndex::Path(
                    2,
                    None,
                    vec![ContractionIndex::Pair(1, 2), ContractionIndex::Pair(1, 3)]
                ),
                ContractionIndex::Path(2, None, vec![]),
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
                (5, [2, 3], [(1, 2), (1, 3)]),
                (0, 2),
                (6, []),
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
                ContractionIndex::Path(
                    5,
                    Some(SlicingPlan { slices: vec![2, 3] }),
                    vec![ContractionIndex::Pair(1, 2), ContractionIndex::Pair(1, 3)]
                ),
                ContractionIndex::Pair(0, 2),
                ContractionIndex::Path(6, None, vec![]),
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
