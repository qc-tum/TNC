use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use itertools::Itertools;
use serde::{Deserialize, Serialize};

use crate::tensornetwork::tensor::Tensor;
// use std::hash::DefaultHasher;

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
    /// Computes the size of the slice when applied to the `target` tensor.
    /// This is the product of all sliced legs.
    ///
    /// # Examples
    /// ```
    /// # use tensorcontraction::types::SlicingPlan;
    /// # use tensorcontraction::tensornetwork::tensor::Tensor;
    /// # use rustc_hash::FxHashMap;
    /// let bond_dims = FxHashMap::from_iter([(0, 2), (1, 3)]);
    /// let t1 = Tensor::new(vec![0, 1]);
    /// let t2 = Tensor::new(vec![0, 1]);
    /// let mut tc = Tensor::default();
    /// tc.push_tensors(vec![t1, t2], Some(&bond_dims));
    ///
    /// let plan = SlicingPlan { slices: vec![0, 1] };
    /// assert_eq!(plan.size(&tc), 6);
    /// ```
    pub fn size(&self, target: &Tensor) -> u64 {
        self.slices
            .iter()
            .map(|leg| target.bond_dims()[leg])
            .product()
    }

    /// Gets the specified [`SlicingTask`] from this plan. Each combination of
    /// indices of the sliced legs is a separate task. `task_index` has to be less
    /// than [`SlicingPlan::size`].
    ///
    /// # Examples
    /// ```
    /// # use tensorcontraction::types::SlicingPlan;
    /// # use tensorcontraction::tensornetwork::tensor::Tensor;
    /// # use rustc_hash::FxHashMap;
    /// let bond_dims = FxHashMap::from_iter([(0, 2), (1, 3)]);
    /// let t1 = Tensor::new(vec![0, 1]);
    /// let t2 = Tensor::new(vec![0, 1]);
    /// let mut tc = Tensor::default();
    /// tc.push_tensors(vec![t1, t2], Some(&bond_dims));
    ///
    /// let plan = SlicingPlan { slices: vec![0, 1] };
    /// let task = plan.get_task(&tc, 0);
    /// assert_eq!(task.slices, vec![(0, 0), (1, 0)]);
    /// ```
    pub fn get_task(&self, target: &Tensor, task_index: usize) -> SlicingTask {
        self.slices
            .iter()
            .map(|leg| target.bond_dims()[leg] as usize)
            .map(|dim| 0..dim)
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
