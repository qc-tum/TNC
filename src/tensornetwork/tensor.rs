use std::hash::{Hash, Hasher};
use std::iter::zip;
use std::num::TryFromIntError;
use std::ops::{BitAnd, BitOr, BitXor, BitXorAssign, Sub};

use float_cmp::approx_eq;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

use crate::tensornetwork::tensordata::TensorData;
use crate::types::{EdgeIndex, TensorIndex};
use crate::utils::datastructures::UnionFind;

/// Abstract representation of a tensor.
#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct Tensor {
    /// The inner tensors that make up this tensor. If non-empty, this tensor is
    /// called a *composite* tensor.
    pub(crate) tensors: Vec<Tensor>,

    /// The legs of the tensor. Each leg is an index to an edge.
    pub(crate) legs: Vec<EdgeIndex>,

    /// The bond dimensions, same length as `legs`.
    pub(crate) bond_dims: Vec<u64>,

    /// The data of the tensor.
    pub(crate) tensordata: TensorData,
}

impl Hash for Tensor {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.legs.hash(state);
    }
}

impl Tensor {
    /// Constructs a Tensor object with the given `legs` (edge ids) and corresponding
    /// `bond_dims`. The tensor doesn't have underlying data.
    #[inline]
    pub(crate) fn new(legs: Vec<EdgeIndex>, bond_dims: Vec<u64>) -> Self {
        assert_eq!(legs.len(), bond_dims.len());
        Self {
            legs,
            bond_dims,
            ..Default::default()
        }
    }

    /// Constructs a Tensor using with the given edge ids and a mapping of edge ids
    /// to corresponding bond dimension.
    ///
    /// # Examples
    /// ```
    /// # use tensorcontraction::tensornetwork::tensor::Tensor;
    /// # use rustc_hash::FxHashMap;
    /// let bond_dims = FxHashMap::from_iter([(1, 2), (2, 4), (3, 6)]);
    /// let tensor = Tensor::new_from_map(vec![1, 2, 3], &bond_dims);
    /// assert_eq!(tensor.legs(), &[1, 2, 3]);
    /// assert_eq!(tensor.bond_dims(), &[2, 4, 6]);
    /// ```
    #[inline]
    pub fn new_from_map(legs: Vec<EdgeIndex>, bond_dims_map: &FxHashMap<EdgeIndex, u64>) -> Self {
        let bond_dims = legs.iter().map(|l| bond_dims_map[l]).collect();
        Self::new(legs, bond_dims)
    }

    /// Constructs a Tensor with the given edge ids and the same bond dimension for
    /// all edges.
    ///
    /// # Examples
    /// ```
    /// # use tensorcontraction::tensornetwork::tensor::Tensor;
    /// let tensor = Tensor::new_from_const(vec![1, 2, 3], 2);
    /// assert_eq!(tensor.legs(), &[1, 2, 3]);
    /// assert_eq!(tensor.bond_dims(), &[2, 2, 2]);
    /// ```
    #[inline]
    pub fn new_from_const(legs: Vec<EdgeIndex>, bond_dim: u64) -> Self {
        let bond_dims = vec![bond_dim; legs.len()];
        Self::new(legs, bond_dims)
    }

    /// Creates a new composite tensor with the given nested tensors.
    #[inline]
    pub fn new_composite(tensors: Vec<Self>) -> Self {
        Self {
            tensors,
            ..Default::default()
        }
    }

    /// Returns edge ids of Tensor object.
    ///
    /// # Examples
    /// ```
    /// # use tensorcontraction::tensornetwork::tensor::Tensor;
    /// let tensor = Tensor::new_from_const(vec![1, 2, 3], 3);
    /// assert_eq!(tensor.legs(), &[1, 2, 3]);
    /// ```
    #[inline]
    pub fn legs(&self) -> &Vec<EdgeIndex> {
        &self.legs
    }

    /// Internal method to set legs. Needs pub(crate) for contraction order finding for hierarchies.
    #[inline]
    pub(crate) fn set_legs(&mut self, legs: Vec<EdgeIndex>) {
        self.legs = legs;
    }

    /// Returns an iterator of tuples of leg ids and their corresponding bond size.
    #[inline]
    pub fn edges(&self) -> impl Iterator<Item = (&EdgeIndex, &u64)> + '_ {
        std::iter::zip(&self.legs, &self.bond_dims)
    }

    /// Returns the nested tensors of a composite tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// # use tensorcontraction::tensornetwork::tensor::Tensor;
    /// # use rustc_hash::FxHashMap;
    /// let bond_dims = FxHashMap::from_iter([(0, 17), (1, 19), (2, 8)]);
    /// let v1 = Tensor::new_from_map(vec![0, 1], &bond_dims);
    /// let v2 = Tensor::new_from_map(vec![1, 2], &bond_dims);
    /// let tn = Tensor::new_composite(vec![v1.clone(), v2.clone()]);
    /// for (tensor, ref_tensor) in std::iter::zip(tn.tensors(), vec![v1, v2]){
    ///    assert_eq!(tensor.legs(), ref_tensor.legs());
    /// }
    /// ```
    #[inline]
    pub fn tensors(&self) -> &Vec<Self> {
        &self.tensors
    }

    /// Gets a nested `Tensor` based on the `nested_indices` which specify the index
    /// of the tensor at each level of the hierarchy.
    ///
    /// # Examples
    /// ```
    /// # use tensorcontraction::tensornetwork::tensor::Tensor;
    /// # use rustc_hash::FxHashMap;
    /// let bond_dims = FxHashMap::from_iter([(0, 17), (1, 19), (2, 8), (3, 2), (4, 1)]);
    /// let mut v1 = Tensor::new_from_map(vec![0, 1], &bond_dims);
    /// let mut v2 = Tensor::new_from_map(vec![1, 2], &bond_dims);
    /// let mut v3 = Tensor::new_from_map(vec![2, 3], &bond_dims);
    /// let mut v4 = Tensor::new_from_map(vec![3, 4], &bond_dims);
    /// let tn1 = Tensor::new_composite(vec![v1, v2]);
    /// let tn2 = Tensor::new_composite(vec![v3.clone(), v4]);
    /// let nested_tn = Tensor::new_composite(vec![tn1, tn2]);
    ///
    /// assert_eq!(nested_tn.nested_tensor(&[1, 0]).legs(), v3.legs());
    /// ```
    pub fn nested_tensor(&self, nested_indices: &[usize]) -> &Tensor {
        let mut tensor = self;
        for index in nested_indices {
            tensor = tensor.tensor(*index);
        }
        tensor
    }

    /// Returns the total number of leaf tensors in the hierarchy.
    pub fn total_num_tensors(&self) -> usize {
        if self.is_composite() {
            self.tensors.iter().map(Self::total_num_tensors).sum()
        } else {
            1
        }
    }

    /// Get ith Tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// # use tensorcontraction::tensornetwork::tensor::Tensor;
    /// # use rustc_hash::FxHashMap;
    /// let bond_dims = FxHashMap::from_iter([(0, 17), (1, 19), (2, 8)]);
    /// let v1 = Tensor::new_from_map(vec![0, 1], &bond_dims);
    /// let v2 = Tensor::new_from_map(vec![1, 2], &bond_dims);
    /// let tn = Tensor::new_composite(vec![v1.clone(), v2]);
    /// assert_eq!(tn.tensor(0).legs(), v1.legs());
    /// ```
    #[inline]
    pub fn tensor(&self, i: TensorIndex) -> &Self {
        &self.tensors[i]
    }

    /// Getter for bond dimensions.
    #[inline]
    pub fn bond_dims(&self) -> &Vec<u64> {
        assert!(self.is_leaf());
        &self.bond_dims
    }

    /// Returns the shape of tensor. This is the same as the bond dimensions, but as
    /// `usize`. The conversion can fail, hence a [`Result`] is returned.
    pub fn shape(&self) -> Result<Vec<usize>, TryFromIntError> {
        self.bond_dims.iter().map(|&dim| dim.try_into()).collect()
    }

    /// Returns the number of dimensions.
    ///
    /// # Examples
    /// ```
    /// # use tensorcontraction::tensornetwork::tensor::Tensor;
    /// # use rustc_hash::FxHashMap;
    /// let bond_dims = FxHashMap::from_iter([(1, 4), (2, 6), (3, 2)]);
    /// let tensor = Tensor::new_from_map(vec![1, 2, 3], &bond_dims);
    /// assert_eq!(tensor.dims(), 3);
    /// ```
    #[inline]
    pub fn dims(&self) -> usize {
        self.legs.len()
    }

    /// Returns the number of elements. This is a f64 to avoid overflow in large
    /// tensors.
    ///
    /// # Examples
    /// ```
    /// # use tensorcontraction::tensornetwork::tensor::Tensor;
    /// # use rustc_hash::FxHashMap;
    /// let bond_dims = FxHashMap::from_iter([(1, 5), (2, 15), (3, 8)]);
    /// let tensor = Tensor::new_from_map(vec![1, 2, 3], &bond_dims);
    /// assert_eq!(tensor.size(), 600.0);
    /// ```
    #[inline]
    pub fn size(&self) -> f64 {
        self.bond_dims.iter().map(|v| *v as f64).product()
    }

    /// Returns true if Tensor is a leaf tensor, without any nested tensors.
    ///
    /// # Examples
    /// ```
    /// # use tensorcontraction::tensornetwork::tensor::Tensor;
    /// # use rustc_hash::FxHashMap;
    /// let bond_dims = FxHashMap::from_iter([(1, 2), (2, 4), (3, 6)]);
    /// let tensor = Tensor::new_from_map(vec![1, 2, 3], &bond_dims);
    /// assert_eq!(tensor.is_leaf(), true);
    /// let comp = Tensor::new_composite(vec![tensor]);
    /// assert_eq!(comp.is_leaf(), false);
    /// ```
    #[inline]
    pub fn is_leaf(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Returns true if Tensor is composite.
    ///
    /// # Examples
    /// ```
    /// # use tensorcontraction::tensornetwork::tensor::Tensor;
    /// # use rustc_hash::FxHashMap;
    /// let bond_dims = FxHashMap::from_iter([(1, 2), (2, 4), (3, 6)]);
    /// let tensor = Tensor::new_from_map(vec![1, 2, 3], &bond_dims);
    /// assert_eq!(tensor.is_composite(), false);
    /// let comp = Tensor::new_composite(vec![tensor]);
    /// assert_eq!(comp.is_composite(), true);
    /// ```
    #[inline]
    pub fn is_composite(&self) -> bool {
        !self.tensors.is_empty()
    }

    /// Returns true if Tensor is empty. This means, it doesn't have any subtensors,
    /// has no legs and is doesn't have any data (e.g., is not a scalar).
    ///
    /// # Examples
    /// ```
    /// # use tensorcontraction::tensornetwork::tensor::Tensor;
    /// let tensor = Tensor::default();
    /// assert_eq!(tensor.is_empty(), true);
    /// ```
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
            && self.legs.is_empty()
            && matches!(*self.tensor_data(), TensorData::Uncontracted)
    }

    /// Comparison of two Tensors, returns true if Tensor objects are equivalent up to `epsilon` precision.
    /// Considers `legs`, `bond_dims` and `tensordata`.
    pub fn approx_eq(&self, other: &Self, epsilon: f64) -> bool {
        if self.tensors.len() != other.tensors.len() {
            return false;
        }
        if self.legs != other.legs {
            return false;
        }
        if self.bond_dims != other.bond_dims {
            return false;
        }

        for (tensor, other_tensor) in zip(&self.tensors, &other.tensors) {
            if !tensor.approx_eq(other_tensor, epsilon) {
                return false;
            }
        }

        approx_eq!(
            &TensorData,
            &self.tensordata,
            &other.tensordata,
            epsilon = epsilon
        )
    }

    /// Pushes additional `tensor` into this tensor, which must be a composite tensor.
    #[inline]
    pub fn push_tensor(&mut self, tensor: Self) {
        assert!(
            self.legs.is_empty() && matches!(self.tensordata, TensorData::Uncontracted),
            "Cannot push tensors into a leaf tensor"
        );
        self.tensors.push(tensor);
    }

    /// Pushes additional `tensors` into this tensor, which must be a composite tensor.
    #[inline]
    pub fn push_tensors(&mut self, mut tensors: Vec<Self>) {
        assert!(
            self.legs.is_empty() && matches!(self.tensordata, TensorData::Uncontracted),
            "Cannot push tensors into a leaf tensor"
        );
        self.tensors.append(&mut tensors);
    }

    /// Getter for tensor data.
    #[inline]
    pub fn tensor_data(&self) -> &TensorData {
        &self.tensordata
    }

    /// Setter for tensor data.
    ///
    /// # Examples
    ///
    /// ```
    /// # use tensorcontraction::tensornetwork::tensor::Tensor;
    /// # use tensorcontraction::tensornetwork::tensordata::TensorData;
    /// let mut tensor = Tensor::new_from_const(vec![0, 1], 2);
    /// let tensordata = TensorData::Gate((String::from("x"), vec![], false));
    /// tensor.set_tensor_data(tensordata);
    /// ```
    #[inline]
    pub fn set_tensor_data(&mut self, tensordata: TensorData) {
        assert!(
            self.is_leaf() || matches!(tensordata, TensorData::Uncontracted),
            "Cannot add data to composite tensor"
        );
        self.tensordata = tensordata;
    }

    /// Returns whether all tensors inside this tensor are connected.
    /// This only checks the top-level, not recursing into composite tensors.
    ///
    /// # Examples
    /// ```
    /// # use tensorcontraction::tensornetwork::tensor::Tensor;
    /// # use rustc_hash::FxHashMap;
    /// // Create a tensor network with two connected tensors
    /// let bond_dims = FxHashMap::from_iter([(0, 17), (1, 19), (2, 8), (3, 5)]);
    /// let v1 = Tensor::new_from_map(vec![0, 1], &bond_dims);
    /// let v2 = Tensor::new_from_map(vec![1, 2], &bond_dims);
    /// let mut tn = Tensor::new_composite(vec![v1, v2]);
    /// assert!(tn.is_connected());
    ///
    /// // Introduce a new tensor that is not connected
    /// let v3 = Tensor::new_from_map(vec![3], &bond_dims);
    /// tn.push_tensor(v3);
    /// assert!(!tn.is_connected());
    /// ```
    pub fn is_connected(&self) -> bool {
        let num_tensors = self.tensors.len();
        let mut uf = UnionFind::new(num_tensors);

        for t1_id in 0..num_tensors - 1 {
            for t2_id in (t1_id + 1)..num_tensors {
                let t1 = &self.tensors[t1_id];
                let t2 = &self.tensors[t2_id];
                if !(t1 & t2).legs.is_empty() {
                    uf.union(t1_id, t2_id);
                }
            }
        }

        uf.count_sets() == 1
    }

    /// Returns `Tensor` with legs in `self` that are not in `other`.
    ///
    /// # Examples
    /// ```
    /// # use tensorcontraction::tensornetwork::tensor::Tensor;
    /// # use rustc_hash::FxHashMap;
    /// let bond_dims = FxHashMap::from_iter([(1, 2), (2, 4), (3, 6), (4, 3), (5, 9)]);
    /// let tensor1 = Tensor::new_from_map(vec![1, 2, 3], &bond_dims);
    /// let tensor2 = Tensor::new_from_map(vec![4, 2, 5], &bond_dims);
    /// let diff_tensor = &tensor1 - &tensor2;
    /// assert_eq!(diff_tensor.legs(), &[1, 3]);
    /// assert_eq!(diff_tensor.bond_dims(), &[2, 6]);
    /// ```
    #[must_use]
    pub fn difference(&self, other: &Self) -> Self {
        let mut new_legs = Vec::with_capacity(self.legs.len());
        let mut new_bond_dims = Vec::with_capacity(new_legs.capacity());
        for (leg, dim) in self.edges() {
            if !other.legs.contains(leg) {
                new_legs.push(*leg);
                new_bond_dims.push(*dim);
            }
        }
        Self::new(new_legs, new_bond_dims)
    }

    /// Returns `Tensor` with union of legs in both `self` and `other`.
    ///
    /// # Examples
    /// ```
    /// # use tensorcontraction::tensornetwork::tensor::Tensor;
    /// # use rustc_hash::FxHashMap;
    /// let bond_dims = FxHashMap::from_iter([(1, 2), (2, 4), (3, 6), (4, 3), (5, 9)]);
    /// let tensor1 = Tensor::new_from_map(vec![1, 2, 3], &bond_dims);
    /// let tensor2 = Tensor::new_from_map(vec![4, 2, 5], &bond_dims);
    /// let union_tensor = &tensor1 | &tensor2;
    /// assert_eq!(union_tensor.legs(), &[1, 2, 3, 4, 5]);
    /// assert_eq!(union_tensor.bond_dims(), &[2, 4, 6, 3, 9]);
    /// ```
    #[must_use]
    pub fn union(&self, other: &Self) -> Self {
        let mut new_legs = Vec::with_capacity(self.legs.len() + other.legs.len());
        let mut new_bond_dims = Vec::with_capacity(new_legs.capacity());
        new_legs.extend_from_slice(&self.legs);
        new_bond_dims.extend_from_slice(&self.bond_dims);
        for (leg, dim) in other.edges() {
            if !self.legs.contains(leg) {
                new_legs.push(*leg);
                new_bond_dims.push(*dim);
            }
        }
        Self::new(new_legs, new_bond_dims)
    }

    /// Returns `Tensor` with intersection of legs in `self` and `other`.
    ///
    /// # Examples
    /// ```
    /// # use tensorcontraction::tensornetwork::tensor::Tensor;
    /// # use rustc_hash::FxHashMap;
    /// let bond_dims = FxHashMap::from_iter([(1, 2), (2, 4), (3, 6), (4, 3), (5, 9)]);
    /// let tensor1 = Tensor::new_from_map(vec![1, 2, 3], &bond_dims);
    /// let tensor2 = Tensor::new_from_map(vec![4, 2, 5], &bond_dims);
    /// let intersection_tensor = &tensor1 & &tensor2;
    /// assert_eq!(intersection_tensor.legs(), &[2]);
    /// assert_eq!(intersection_tensor.bond_dims(), &[4]);
    /// ```
    #[must_use]
    pub fn intersection(&self, other: &Self) -> Self {
        let mut new_legs = Vec::with_capacity(self.legs.len().min(other.legs.len()));
        let mut new_bond_dims = Vec::with_capacity(new_legs.capacity());
        for (leg, dim) in self.edges() {
            if other.legs.contains(leg) {
                new_legs.push(*leg);
                new_bond_dims.push(*dim);
            }
        }
        Self::new(new_legs, new_bond_dims)
    }

    /// Returns `Tensor` with symmetrical difference of legs in `self` and `other`.
    ///
    /// # Examples
    /// ```
    /// # use tensorcontraction::tensornetwork::tensor::Tensor;
    /// # use rustc_hash::FxHashMap;
    /// let bond_dims = FxHashMap::from_iter([(1, 2), (2, 4), (3, 6), (4, 3), (5, 9)]);
    /// let tensor1 = Tensor::new_from_map(vec![1, 2, 3], &bond_dims);
    /// let tensor2 = Tensor::new_from_map(vec![4, 2, 5], &bond_dims);
    /// let sym_dif_tensor = &tensor1 ^ &tensor2;
    /// assert_eq!(sym_dif_tensor.legs(), &[1, 3, 4, 5]);
    /// assert_eq!(sym_dif_tensor.bond_dims(), &[2, 6, 3, 9]);
    /// ```
    #[must_use]
    pub fn symmetric_difference(&self, other: &Self) -> Self {
        let mut new_legs = Vec::with_capacity(self.legs.len() + other.legs.len());
        let mut new_bond_dims = Vec::with_capacity(new_legs.capacity());
        for (leg, dim) in self.edges() {
            if !other.legs.contains(leg) {
                new_legs.push(*leg);
                new_bond_dims.push(*dim);
            }
        }
        for (leg, dim) in other.edges() {
            if !self.legs.contains(leg) {
                new_legs.push(*leg);
                new_bond_dims.push(*dim);
            }
        }
        Self::new(new_legs, new_bond_dims)
    }

    /// Get output legs after tensor contraction
    pub fn external_tensor(&self) -> Tensor {
        if self.is_leaf() {
            return self.clone();
        }

        let mut ext_tensor = Self::default();
        for tensor in &self.tensors {
            let new_tensor = if tensor.is_composite() {
                &tensor.external_tensor()
            } else {
                tensor
            };
            ext_tensor = &ext_tensor ^ new_tensor;
        }

        ext_tensor
    }
}

impl BitOr for &Tensor {
    type Output = Tensor;
    #[inline]
    fn bitor(self, rhs: &Tensor) -> Tensor {
        self.union(rhs)
    }
}

impl BitAnd for &Tensor {
    type Output = Tensor;
    #[inline]
    fn bitand(self, rhs: &Tensor) -> Tensor {
        self.intersection(rhs)
    }
}

impl BitXor for &Tensor {
    type Output = Tensor;
    #[inline]
    fn bitxor(self, rhs: &Tensor) -> Tensor {
        self.symmetric_difference(rhs)
    }
}

impl Sub for &Tensor {
    type Output = Tensor;
    #[inline]
    fn sub(self, rhs: &Tensor) -> Tensor {
        self.difference(rhs)
    }
}

impl BitXorAssign<&Tensor> for Tensor {
    #[inline]
    fn bitxor_assign(&mut self, rhs: &Tensor) {
        *self = self.symmetric_difference(rhs);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::{assert_matches::assert_matches, iter::zip};

    use rustc_hash::FxHashMap;

    use crate::tensornetwork::tensordata::TensorData;

    #[test]
    fn test_empty_tensor() {
        let tensor = Tensor::default();
        assert!(tensor.tensors.is_empty());
        assert!(tensor.legs.is_empty());
        assert!(tensor.bond_dims.is_empty());
        assert!(tensor.is_empty());
    }

    #[test]
    fn test_new() {
        let tensor = Tensor::new(vec![2, 4, 5], vec![4, 2, 6]);
        assert_eq!(tensor.legs(), &[2, 4, 5]);
        assert_eq!(tensor.bond_dims(), &[4, 2, 6]);
        assert_matches!(tensor.tensor_data(), TensorData::Uncontracted);
    }

    #[test]
    fn test_new_from_map() {
        let bond_dims = FxHashMap::from_iter([(1, 1), (2, 4), (3, 7), (4, 2), (5, 6)]);
        let tensor = Tensor::new_from_map(vec![2, 4, 5], &bond_dims);
        assert_eq!(tensor.legs(), &[2, 4, 5]);
        assert_eq!(tensor.bond_dims(), &[4, 2, 6]);
        assert_matches!(tensor.tensor_data(), TensorData::Uncontracted);
    }

    #[test]
    fn test_new_from_const() {
        let tensor = Tensor::new_from_const(vec![9, 2, 5, 1], 3);
        assert_eq!(tensor.legs(), &[9, 2, 5, 1]);
        assert_eq!(tensor.bond_dims(), &[3, 3, 3, 3]);
        assert_matches!(tensor.tensor_data(), TensorData::Uncontracted);
    }

    #[test]
    fn test_external_tensor() {
        let bond_dims = FxHashMap::from_iter([
            (2, 2),
            (3, 4),
            (4, 6),
            (5, 8),
            (6, 10),
            (7, 12),
            (8, 14),
            (9, 16),
        ]);
        let tensor_1 = Tensor::new_from_map(vec![2, 3, 4], &bond_dims);
        let tensor_2 = Tensor::new_from_map(vec![2, 3, 5], &bond_dims);
        let tensor_12 = Tensor::new_composite(vec![tensor_1, tensor_2]);

        let tensor_3 = Tensor::new_from_map(vec![6, 7, 8], &bond_dims);
        let tensor_4 = Tensor::new_from_map(vec![6, 8, 9], &bond_dims);
        let tensor_34 = Tensor::new_composite(vec![tensor_3, tensor_4]);

        let tensor_1234 = Tensor::new_composite(vec![tensor_12, tensor_34]);

        let external = tensor_1234.external_tensor();
        assert_eq!(external.legs(), &[4, 5, 7, 9]);
        assert_eq!(external.bond_dims(), &[6, 8, 12, 16]);
    }

    #[test]
    fn test_push_tensor() {
        let bond_dims =
            FxHashMap::from_iter([(2, 17), (3, 1), (4, 11), (8, 3), (9, 20), (7, 7), (10, 14)]);
        let ref_tensor_1 = Tensor::new_from_map(vec![8, 4, 9], &bond_dims);
        let ref_tensor_2 = Tensor::new_from_map(vec![7, 10, 2], &bond_dims);

        let mut tensor = Tensor::default();

        // Push tensor 1
        let tensor_1 = Tensor::new_from_map(vec![8, 4, 9], &bond_dims);
        tensor.push_tensor(tensor_1);

        for (sub_tensor, ref_tensor) in zip(tensor.tensors(), [&ref_tensor_1]) {
            assert_eq!(sub_tensor.legs(), ref_tensor.legs());
            assert_eq!(sub_tensor.bond_dims(), ref_tensor.bond_dims());
        }

        // Push tensor 2
        let tensor_2 = Tensor::new_from_map(vec![7, 10, 2], &bond_dims);
        tensor.push_tensor(tensor_2);

        for (sub_tensor, ref_tensor) in zip(tensor.tensors(), [&ref_tensor_1, &ref_tensor_2]) {
            assert_eq!(sub_tensor.legs(), ref_tensor.legs());
            assert_eq!(sub_tensor.bond_dims(), ref_tensor.bond_dims());
        }

        // Test that other fields are unchanged
        assert_matches!(tensor.tensor_data(), TensorData::Uncontracted);
        assert!(tensor.legs().is_empty());
    }

    #[test]
    #[should_panic(expected = "Cannot push tensors into a leaf tensor")]
    fn test_push_tensor_to_leaf() {
        let bond_dims =
            FxHashMap::from_iter([(2, 17), (3, 1), (4, 11), (8, 3), (9, 20), (7, 7), (10, 14)]);
        let mut leaf_tensor = Tensor::new_from_map(vec![4, 3, 2], &bond_dims);
        let pushed_tensor = Tensor::new_from_map(vec![8, 4, 9], &bond_dims);
        leaf_tensor.push_tensor(pushed_tensor);
    }

    #[test]
    fn test_push_tensors() {
        let bond_dims =
            FxHashMap::from_iter([(2, 17), (3, 1), (4, 11), (8, 3), (9, 20), (7, 7), (10, 14)]);
        let ref_tensor_1 = Tensor::new_from_map(vec![4, 3, 2], &bond_dims);
        let ref_tensor_2 = Tensor::new_from_map(vec![8, 4, 9], &bond_dims);
        let ref_tensor_3 = Tensor::new_from_map(vec![7, 10, 2], &bond_dims);

        let tensor_1 = Tensor::new_from_map(vec![4, 3, 2], &bond_dims);
        let tensor_2 = Tensor::new_from_map(vec![8, 4, 9], &bond_dims);
        let tensor_3 = Tensor::new_from_map(vec![7, 10, 2], &bond_dims);
        let mut tensor = Tensor::default();
        tensor.push_tensors(vec![tensor_1, tensor_2, tensor_3]);

        assert_matches!(tensor.tensor_data(), TensorData::Uncontracted);

        for (sub_tensor, ref_tensor) in zip(
            tensor.tensors(),
            &vec![ref_tensor_1, ref_tensor_2, ref_tensor_3],
        ) {
            assert_eq!(sub_tensor.legs(), ref_tensor.legs());
            assert_eq!(sub_tensor.bond_dims(), ref_tensor.bond_dims());
        }
    }

    #[test]
    #[should_panic(expected = "Cannot push tensors into a leaf tensor")]
    fn test_push_tensors_to_leaf() {
        let bond_dims =
            FxHashMap::from_iter([(2, 17), (3, 1), (4, 11), (8, 3), (9, 20), (7, 7), (10, 14)]);
        let mut leaf_tensor = Tensor::new_from_map(vec![4, 3, 2], &bond_dims);
        let pushed_tensor_1 = Tensor::new_from_map(vec![8, 4, 9], &bond_dims);
        let pushed_tensor_2 = Tensor::new_from_map(vec![7, 10, 2], &bond_dims);

        leaf_tensor.push_tensors(vec![pushed_tensor_1, pushed_tensor_2]);
    }
}
