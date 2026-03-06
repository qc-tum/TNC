use std::iter::zip;
use std::num::TryFromIntError;
use std::ops::{BitAnd, BitOr, BitXor, BitXorAssign, Sub};

use bytemuck::{TransparentWrapper, TransparentWrapperAlloc};
use float_cmp::{ApproxEq, F64Margin};
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

use crate::tensornetwork::tensordata::TensorData;
use crate::utils::datastructures::UnionFind;

/// Unique index of a leg.
pub type EdgeIndex = usize;

/// Index of a tensor in a tensor network.
pub type TensorIndex = usize;

/// An abstract tensor. This can either be a [`CompositeTensor`] or a [`LeafTensor`].
#[derive(Debug, Clone, TransparentWrapper, Serialize, Deserialize)]
#[repr(transparent)]
pub struct Tensor(TensorRepr);

/// A composite tensor that has other tensors as children, similar to a tensor
/// network.
#[derive(Debug, Clone, TransparentWrapper, Serialize, Deserialize)]
#[repr(transparent)]
pub struct CompositeTensor(TensorRepr);

/// A single leaf tensor.
#[derive(Debug, Clone, TransparentWrapper, Serialize, Deserialize)]
#[repr(transparent)]
pub struct LeafTensor(TensorRepr);

/// The type of a tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TensorType {
    Composite,
    Leaf,
}

/// Abstract representation of a tensor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorRepr {
    /// The type of this tensor.
    kind: TensorType,

    /// The inner tensors that make up this tensor. If non-empty, this tensor is
    /// called a *composite* tensor.
    tensors: Vec<Tensor>,

    /// The legs of the tensor. Each leg should have a unique id. Connected tensors
    /// are recognized by having at least one leg id in common.
    legs: Vec<EdgeIndex>,

    /// The bond dimensions of the legs, same length and order as `legs`. It is
    /// assumed (but not checked!) that the bond dimensions of different tensors that
    /// connect to the same leg match.
    bond_dims: Vec<u64>,

    /// The data of the tensor.
    tensordata: TensorData,
}

impl Tensor {
    /// Returns the kind of this tensor.
    #[inline]
    pub fn kind(&self) -> TensorType {
        self.0.kind
    }

    /// Returns true if the tensor is a leaf tensor, without any nested tensors.
    ///
    /// # Examples
    /// ```
    /// # use tnc::tensornetwork::tensor::{CompositeTensor, LeafTensor, Tensor};
    /// # use rustc_hash::FxHashMap;
    /// let bond_dims = FxHashMap::from_iter([(1, 2), (2, 4), (3, 6)]);
    /// let leaf: Tensor = LeafTensor::new_from_map(vec![1, 2, 3], &bond_dims).into();
    /// assert_eq!(leaf.is_leaf(), true);
    /// let comp: Tensor = CompositeTensor::new(vec![leaf]).into();
    /// assert_eq!(comp.is_leaf(), false);
    /// ```
    #[inline]
    pub fn is_leaf(&self) -> bool {
        self.kind() == TensorType::Leaf
    }

    /// Returns a reference to this tensor as a [`LeafTensor`] if it is one.
    #[inline]
    pub fn as_leaf(&self) -> Option<&LeafTensor> {
        self.is_leaf()
            .then(|| LeafTensor::wrap_ref(Self::peel_ref(self)))
    }

    /// Returns this tensor as a [`LeafTensor`] if it is one.
    #[inline]
    pub fn into_leaf(self) -> Option<LeafTensor> {
        self.is_leaf().then(|| LeafTensor::wrap(Self::peel(self)))
    }

    /// Returns true if the tensor is composite.
    ///
    /// # Examples
    /// ```
    /// # use tnc::tensornetwork::tensor::{CompositeTensor, LeafTensor, Tensor};
    /// # use rustc_hash::FxHashMap;
    /// let bond_dims = FxHashMap::from_iter([(1, 2), (2, 4), (3, 6)]);
    /// let leaf: Tensor = LeafTensor::new_from_map(vec![1, 2, 3], &bond_dims).into();
    /// assert_eq!(leaf.is_composite(), false);
    /// let comp: Tensor = CompositeTensor::new(vec![leaf]).into();
    /// assert_eq!(comp.is_composite(), true);
    /// ```
    #[inline]
    pub fn is_composite(&self) -> bool {
        self.kind() == TensorType::Composite
    }

    /// Returns a reference to this tensor as a [`CompositeTensor`] if it is one.
    #[inline]
    pub fn as_composite(&self) -> Option<&CompositeTensor> {
        self.is_composite()
            .then(|| CompositeTensor::wrap_ref(Self::peel_ref(self)))
    }

    /// Returns this tensor as a [`CompositeTensor`] if it is one.
    #[inline]
    pub fn into_composite(self) -> Option<CompositeTensor> {
        self.is_composite()
            .then(|| CompositeTensor::wrap(Self::peel(self)))
    }
}

pub trait TensorList {
    fn into_tensors(self) -> Vec<Tensor>;
    fn as_tensors(&self) -> &[Tensor];
}

impl TensorList for Vec<Tensor> {
    #[inline]
    fn into_tensors(self) -> Vec<Tensor> {
        self
    }

    #[inline]
    fn as_tensors(&self) -> &[Tensor] {
        self
    }
}

impl From<LeafTensor> for Tensor {
    #[inline]
    fn from(value: LeafTensor) -> Self {
        Self::wrap(LeafTensor::peel(value))
    }
}

impl TensorList for Vec<LeafTensor> {
    #[inline]
    fn into_tensors(self) -> Vec<Tensor> {
        Tensor::wrap_vec(LeafTensor::peel_vec(self))
    }

    #[inline]
    fn as_tensors(&self) -> &[Tensor] {
        Tensor::wrap_slice(LeafTensor::peel_slice(self))
    }
}

impl From<CompositeTensor> for Tensor {
    #[inline]
    fn from(value: CompositeTensor) -> Self {
        Self::wrap(CompositeTensor::peel(value))
    }
}

impl TensorList for Vec<CompositeTensor> {
    #[inline]
    fn into_tensors(self) -> Vec<Tensor> {
        Tensor::wrap_vec(CompositeTensor::peel_vec(self))
    }

    #[inline]
    fn as_tensors(&self) -> &[Tensor] {
        Tensor::wrap_slice(CompositeTensor::peel_slice(self))
    }
}

impl CompositeTensor {
    /// Creates a new composite tensor with the given nested tensors.
    #[inline]
    pub fn new<T>(tensors: T) -> Self
    where
        T: TensorList,
    {
        Self(TensorRepr {
            kind: TensorType::Composite,
            tensors: tensors.into_tensors(),
            legs: Vec::new(),
            bond_dims: Vec::new(),
            tensordata: TensorData::None,
        })
    }

    /// Returns a reference to this composite tensor as a [`Tensor`].
    #[inline]
    pub fn as_tensor(&self) -> &Tensor {
        Tensor::wrap_ref(Self::peel_ref(self))
    }

    /// Returns the children tensors.
    ///
    /// # Examples
    /// ```
    /// # use tnc::tensornetwork::tensor::{CompositeTensor, LeafTensor, Tensor};
    /// # use rustc_hash::FxHashMap;
    /// # use float_cmp::assert_approx_eq;
    /// let bond_dims = FxHashMap::from_iter([(0, 17), (1, 19), (2, 8)]);
    /// let v1 = LeafTensor::new_from_map(vec![0, 1], &bond_dims);
    /// let v2 = LeafTensor::new_from_map(vec![1, 2], &bond_dims);
    /// let tn = CompositeTensor::new(vec![v1.clone(), v2.clone()]);
    /// for (tensor, ref_tensor) in std::iter::zip(tn.tensors(), vec![v1, v2]){
    ///    assert_approx_eq!(&Tensor, tensor, ref_tensor.as_tensor());
    /// }
    /// ```
    #[inline]
    pub fn tensors(&self) -> &Vec<Tensor> {
        &self.0.tensors
    }

    /// Get the ith tensor.
    ///
    /// # Examples
    /// ```
    /// # use tnc::tensornetwork::tensor::{CompositeTensor, LeafTensor, Tensor};
    /// # use rustc_hash::FxHashMap;
    /// # use float_cmp::assert_approx_eq;
    /// let bond_dims = FxHashMap::from_iter([(0, 17), (1, 19), (2, 8)]);
    /// let v1 = LeafTensor::new_from_map(vec![0, 1], &bond_dims);
    /// let v2 = LeafTensor::new_from_map(vec![1, 2], &bond_dims);
    /// let tn = CompositeTensor::new(vec![v1.clone(), v2]);
    /// assert_approx_eq!(&Tensor, tn.tensor(0), v1.as_tensor());
    /// ```
    #[inline]
    pub fn tensor(&self, i: TensorIndex) -> &Tensor {
        &self.0.tensors[i]
    }

    /// Converts this tensor into the vec of its children.
    #[inline]
    pub fn into_tensors(self) -> Vec<Tensor> {
        self.0.tensors
    }

    /// Returns true if the tensor is empty. This means, it doesn't have any
    /// children.
    ///
    /// # Examples
    /// ```
    /// # use tnc::tensornetwork::tensor::CompositeTensor;
    /// let tensor = CompositeTensor::default();
    /// assert_eq!(tensor.is_empty(), true);
    /// ```
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.0.tensors.is_empty()
    }

    /// Returns the number of direct children of this tensor.
    ///
    /// # Examples
    /// ```
    /// # use tnc::tensornetwork::tensor::CompositeTensor;
    /// let tensor = CompositeTensor::default();
    /// assert_eq!(tensor.len(), 0);
    /// ```
    #[inline]
    pub fn len(&self) -> usize {
        self.0.tensors.len()
    }

    /// Gets a nested [`Tensor`] based on the `nested_indices` which specify the
    /// index of the tensor at each level of the hierarchy.
    ///
    /// # Examples
    /// ```
    /// # use tnc::tensornetwork::tensor::{CompositeTensor, LeafTensor};
    /// # use rustc_hash::FxHashMap;
    /// let bond_dims = FxHashMap::from_iter([(0, 17), (1, 19), (2, 8), (3, 2), (4, 1)]);
    /// let mut v1 = LeafTensor::new_from_map(vec![0, 1], &bond_dims);
    /// let mut v2 = LeafTensor::new_from_map(vec![1, 2], &bond_dims);
    /// let mut v3 = LeafTensor::new_from_map(vec![2, 3], &bond_dims);
    /// let mut v4 = LeafTensor::new_from_map(vec![3, 4], &bond_dims);
    /// let tn1 = CompositeTensor::new(vec![v1, v2]);
    /// let tn2 = CompositeTensor::new(vec![v3.clone(), v4]);
    /// let nested_tn = CompositeTensor::new(vec![tn1, tn2]);
    ///
    /// let found = nested_tn.nested_tensor(&[1, 0]);
    /// assert!(found.is_leaf());
    /// let found = found.as_leaf().unwrap();
    /// assert_eq!(found.legs(), v3.legs());
    /// ```
    pub fn nested_tensor(&self, nested_indices: &[usize]) -> &Tensor {
        let mut tensor = self.as_tensor();
        for index in nested_indices {
            tensor = tensor.as_composite().unwrap().tensor(*index);
        }
        tensor
    }

    /// Returns the total number of leaf tensors in the hierarchy.
    pub fn total_num_tensors(&self) -> usize {
        self.0
            .tensors
            .iter()
            .map(|t| match t.kind() {
                TensorType::Composite => t.as_composite().unwrap().total_num_tensors(),
                TensorType::Leaf => 1,
            })
            .sum()
    }

    /// Pushes additional `tensor` into this composite tensor.
    #[inline]
    pub fn push_tensor<T>(&mut self, tensor: T)
    where
        T: Into<Tensor>,
    {
        self.0.tensors.push(tensor.into());
    }

    /// Pushes additional `tensors` into this composite tensor.
    #[inline]
    pub fn push_tensors<T>(&mut self, tensors: T)
    where
        T: TensorList,
    {
        let mut tensors = tensors.into_tensors();
        self.0.tensors.append(&mut tensors);
    }

    /// Reserves space for at least `additional` more tensors to be pushed to this
    /// composite tensor without reallocation.
    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        self.0.tensors.reserve(additional);
    }

    /// Returns whether all tensors inside this tensor are connected. This currently
    /// requires all children to be leaf tensors.
    ///
    /// # Examples
    /// ```
    /// # use tnc::tensornetwork::tensor::{CompositeTensor, LeafTensor};
    /// # use rustc_hash::FxHashMap;
    /// // Create a tensor network with two connected tensors
    /// let bond_dims = FxHashMap::from_iter([(0, 17), (1, 19), (2, 8), (3, 5)]);
    /// let v1 = LeafTensor::new_from_map(vec![0, 1], &bond_dims);
    /// let v2 = LeafTensor::new_from_map(vec![1, 2], &bond_dims);
    /// let mut tn = CompositeTensor::new(vec![v1, v2]);
    /// assert!(tn.is_connected());
    ///
    /// // Introduce a new tensor that is not connected
    /// let v3 = LeafTensor::new_from_map(vec![3], &bond_dims);
    /// tn.push_tensor(v3);
    /// assert!(!tn.is_connected());
    /// ```
    pub fn is_connected(&self) -> bool {
        let num_tensors = self.len();
        let mut uf = UnionFind::new(num_tensors);

        for t1_id in 0..num_tensors {
            for t2_id in (t1_id + 1)..num_tensors {
                let t1 = self
                    .tensor(t1_id)
                    .as_leaf()
                    .expect("Expected all children to be leaves");
                let t2 = self
                    .tensor(t2_id)
                    .as_leaf()
                    .expect("Expected all children to be leaves");
                if !(t1 & t2).legs().is_empty() {
                    uf.union(t1_id, t2_id);
                }
            }
        }

        uf.count_sets() == 1
    }

    /// Get output legs after tensor contraction.
    pub fn external_tensor(&self) -> LeafTensor {
        self.tensors()
            .iter()
            .fold(LeafTensor::default(), |acc, tensor| {
                let tensor = match tensor.kind() {
                    TensorType::Composite => &tensor.as_composite().unwrap().external_tensor(),
                    TensorType::Leaf => tensor.as_leaf().unwrap(),
                };
                &acc ^ tensor
            })
    }
}

impl Default for CompositeTensor {
    fn default() -> Self {
        Self(TensorRepr {
            kind: TensorType::Composite,
            tensors: Vec::new(),
            legs: Vec::new(),
            bond_dims: Vec::new(),
            tensordata: TensorData::None,
        })
    }
}

impl ApproxEq for &CompositeTensor {
    type Margin = F64Margin;

    fn approx_eq<M: Into<Self::Margin>>(self, other: Self, margin: M) -> bool {
        let margin = margin.into();
        if self.len() != other.len() {
            return false;
        }
        for (tensor, other_tensor) in zip(self.tensors(), other.tensors()) {
            if !tensor.approx_eq(other_tensor, margin) {
                return false;
            }
        }
        true
    }
}

impl LeafTensor {
    /// Constructs a leaf tensor object with the given `legs` (edge ids) and
    /// corresponding `bond_dims`. The tensor doesn't have underlying data.
    #[inline]
    pub(crate) fn new(legs: Vec<EdgeIndex>, bond_dims: Vec<u64>) -> Self {
        Self::new_with_data(legs, bond_dims, TensorData::None)
    }

    /// Constructs a leaf tensor object with the given `legs` (edge ids),
    /// corresponding `bond_dims` and `data`.
    #[inline]
    pub(crate) fn new_with_data(
        legs: Vec<EdgeIndex>,
        bond_dims: Vec<u64>,
        data: TensorData,
    ) -> Self {
        assert_eq!(legs.len(), bond_dims.len());
        Self(TensorRepr {
            kind: TensorType::Leaf,
            legs,
            tensors: Vec::new(),
            bond_dims,
            tensordata: data,
        })
    }

    /// Constructs a leaf tensor with the given edge ids and a mapping of edge ids
    /// to corresponding bond dimensions.
    ///
    /// # Examples
    /// ```
    /// # use tnc::tensornetwork::tensor::LeafTensor;
    /// # use rustc_hash::FxHashMap;
    /// let bond_dims = FxHashMap::from_iter([(1, 2), (2, 4), (3, 6)]);
    /// let tensor = LeafTensor::new_from_map(vec![1, 2, 3], &bond_dims);
    /// assert_eq!(tensor.legs(), &[1, 2, 3]);
    /// assert_eq!(tensor.bond_dims(), &[2, 4, 6]);
    /// ```
    #[inline]
    pub fn new_from_map(legs: Vec<EdgeIndex>, bond_dims_map: &FxHashMap<EdgeIndex, u64>) -> Self {
        let bond_dims = legs.iter().map(|l| bond_dims_map[l]).collect();
        Self::new(legs, bond_dims)
    }

    /// Constructs a leaf tensor with the given edge ids and the same bond dimension
    /// for all edges.
    ///
    /// # Examples
    /// ```
    /// # use tnc::tensornetwork::tensor::LeafTensor;
    /// let tensor = LeafTensor::new_from_const(vec![1, 2, 3], 2);
    /// assert_eq!(tensor.legs(), &[1, 2, 3]);
    /// assert_eq!(tensor.bond_dims(), &[2, 2, 2]);
    /// ```
    #[inline]
    pub fn new_from_const(legs: Vec<EdgeIndex>, bond_dim: u64) -> Self {
        let bond_dims = vec![bond_dim; legs.len()];
        Self::new(legs, bond_dims)
    }

    /// Returns a reference to this leaf tensor as a [`Tensor`].
    #[inline]
    pub fn as_tensor(&self) -> &Tensor {
        Tensor::wrap_ref(Self::peel_ref(self))
    }

    /// Returns edge ids of the tensor.
    ///
    /// # Examples
    /// ```
    /// # use tnc::tensornetwork::tensor::LeafTensor;
    /// let tensor = LeafTensor::new_from_const(vec![1, 2, 3], 3);
    /// assert_eq!(tensor.legs(), &[1, 2, 3]);
    /// ```
    #[inline]
    pub fn legs(&self) -> &Vec<EdgeIndex> {
        &self.0.legs
    }

    /// Returns an iterator of tuples of leg ids and their corresponding bond size.
    #[inline]
    pub fn edges(&self) -> impl Iterator<Item = (&EdgeIndex, &u64)> + '_ {
        std::iter::zip(&self.0.legs, &self.0.bond_dims)
    }

    /// Getter for bond dimensions.
    #[inline]
    pub fn bond_dims(&self) -> &Vec<u64> {
        &self.0.bond_dims
    }

    /// Returns the shape of tensor. This is the same as the bond dimensions, but as
    /// `usize`. The conversion can fail, hence a [`Result`] is returned.
    pub fn shape(&self) -> Result<Vec<usize>, TryFromIntError> {
        self.0.bond_dims.iter().map(|&dim| dim.try_into()).collect()
    }

    /// Returns the number of dimensions.
    ///
    /// # Examples
    /// ```
    /// # use tnc::tensornetwork::tensor::LeafTensor;
    /// # use rustc_hash::FxHashMap;
    /// let bond_dims = FxHashMap::from_iter([(1, 4), (2, 6), (3, 2)]);
    /// let tensor = LeafTensor::new_from_map(vec![1, 2, 3], &bond_dims);
    /// assert_eq!(tensor.dims(), 3);
    /// ```
    #[inline]
    pub fn dims(&self) -> usize {
        self.0.legs.len()
    }

    /// Returns the number of elements. This is a f64 to avoid overflow in large
    /// tensors.
    ///
    /// # Examples
    /// ```
    /// # use tnc::tensornetwork::tensor::LeafTensor;
    /// # use rustc_hash::FxHashMap;
    /// let bond_dims = FxHashMap::from_iter([(1, 5), (2, 15), (3, 8)]);
    /// let tensor = LeafTensor::new_from_map(vec![1, 2, 3], &bond_dims);
    /// assert_eq!(tensor.size(), 600.0);
    /// ```
    #[inline]
    pub fn size(&self) -> f64 {
        self.0.bond_dims.iter().map(|v| *v as f64).product()
    }

    /// Converts this tensor into the leg ids and bond dimensions it is made of.
    #[inline]
    pub fn into_legs(self) -> (Vec<EdgeIndex>, Vec<u64>) {
        (self.0.legs, self.0.bond_dims)
    }

    /// Converts this tensor into the data it contains.
    #[inline]
    pub fn into_data(self) -> TensorData {
        self.0.tensordata
    }

    /// Converts this tensor into the leg ids, bond dimensions and data it is made
    /// of.
    #[inline]
    pub fn into_inner(self) -> (Vec<EdgeIndex>, Vec<u64>, TensorData) {
        (self.0.legs, self.0.bond_dims, self.0.tensordata)
    }

    /// Getter for tensor data.
    #[inline]
    pub fn tensor_data(&self) -> &TensorData {
        &self.0.tensordata
    }

    /// Setter for tensor data.
    ///
    /// # Examples
    ///
    /// ```
    /// # use tnc::tensornetwork::tensor::LeafTensor;
    /// # use tnc::tensornetwork::tensordata::TensorData;
    /// let mut tensor = LeafTensor::new_from_const(vec![0, 1], 2);
    /// let tensordata = TensorData::Gate((String::from("x"), vec![], false));
    /// tensor.set_tensor_data(tensordata);
    /// ```
    #[inline]
    pub fn set_tensor_data(&mut self, tensordata: TensorData) {
        self.0.tensordata = tensordata;
    }

    /// Returns the tensor with legs in `self` that are not in `other`.
    ///
    /// # Examples
    /// ```
    /// # use tnc::tensornetwork::tensor::LeafTensor;
    /// # use rustc_hash::FxHashMap;
    /// let bond_dims = FxHashMap::from_iter([(1, 2), (2, 4), (3, 6), (4, 3), (5, 9)]);
    /// let tensor1 = LeafTensor::new_from_map(vec![1, 2, 3], &bond_dims);
    /// let tensor2 = LeafTensor::new_from_map(vec![4, 2, 5], &bond_dims);
    /// let diff_tensor = &tensor1 - &tensor2;
    /// assert_eq!(diff_tensor.legs(), &[1, 3]);
    /// assert_eq!(diff_tensor.bond_dims(), &[2, 6]);
    /// ```
    #[must_use]
    pub fn difference(&self, other: &Self) -> Self {
        let mut new_legs = Vec::with_capacity(self.legs().len());
        let mut new_bond_dims = Vec::with_capacity(new_legs.capacity());
        for (leg, dim) in self.edges() {
            if !other.legs().contains(leg) {
                new_legs.push(*leg);
                new_bond_dims.push(*dim);
            }
        }
        Self::new(new_legs, new_bond_dims)
    }

    /// Returns the tensor with union of legs in both `self` and `other`.
    ///
    /// # Examples
    /// ```
    /// # use tnc::tensornetwork::tensor::LeafTensor;
    /// # use rustc_hash::FxHashMap;
    /// let bond_dims = FxHashMap::from_iter([(1, 2), (2, 4), (3, 6), (4, 3), (5, 9)]);
    /// let tensor1 = LeafTensor::new_from_map(vec![1, 2, 3], &bond_dims);
    /// let tensor2 = LeafTensor::new_from_map(vec![4, 2, 5], &bond_dims);
    /// let union_tensor = &tensor1 | &tensor2;
    /// assert_eq!(union_tensor.legs(), &[1, 2, 3, 4, 5]);
    /// assert_eq!(union_tensor.bond_dims(), &[2, 4, 6, 3, 9]);
    /// ```
    #[must_use]
    pub fn union(&self, other: &Self) -> Self {
        let mut new_legs = Vec::with_capacity(self.legs().len() + other.legs().len());
        let mut new_bond_dims = Vec::with_capacity(new_legs.capacity());
        new_legs.extend_from_slice(self.legs());
        new_bond_dims.extend_from_slice(self.bond_dims());
        for (leg, dim) in other.edges() {
            if !self.legs().contains(leg) {
                new_legs.push(*leg);
                new_bond_dims.push(*dim);
            }
        }
        Self::new(new_legs, new_bond_dims)
    }

    /// Returns the tensor with intersection of legs in `self` and `other`.
    ///
    /// # Examples
    /// ```
    /// # use tnc::tensornetwork::tensor::LeafTensor;
    /// # use rustc_hash::FxHashMap;
    /// let bond_dims = FxHashMap::from_iter([(1, 2), (2, 4), (3, 6), (4, 3), (5, 9)]);
    /// let tensor1 = LeafTensor::new_from_map(vec![1, 2, 3], &bond_dims);
    /// let tensor2 = LeafTensor::new_from_map(vec![4, 2, 5], &bond_dims);
    /// let intersection_tensor = &tensor1 & &tensor2;
    /// assert_eq!(intersection_tensor.legs(), &[2]);
    /// assert_eq!(intersection_tensor.bond_dims(), &[4]);
    /// ```
    #[must_use]
    pub fn intersection(&self, other: &Self) -> Self {
        let mut new_legs = Vec::with_capacity(self.legs().len().min(other.legs().len()));
        let mut new_bond_dims = Vec::with_capacity(new_legs.capacity());
        for (leg, dim) in self.edges() {
            if other.legs().contains(leg) {
                new_legs.push(*leg);
                new_bond_dims.push(*dim);
            }
        }
        Self::new(new_legs, new_bond_dims)
    }

    /// Returns the tensor with symmetrical difference of legs in `self` and `other`.
    ///
    /// # Examples
    /// ```
    /// # use tnc::tensornetwork::tensor::LeafTensor;
    /// # use rustc_hash::FxHashMap;
    /// let bond_dims = FxHashMap::from_iter([(1, 2), (2, 4), (3, 6), (4, 3), (5, 9)]);
    /// let tensor1 = LeafTensor::new_from_map(vec![1, 2, 3], &bond_dims);
    /// let tensor2 = LeafTensor::new_from_map(vec![4, 2, 5], &bond_dims);
    /// let sym_dif_tensor = &tensor1 ^ &tensor2;
    /// assert_eq!(sym_dif_tensor.legs(), &[1, 3, 4, 5]);
    /// assert_eq!(sym_dif_tensor.bond_dims(), &[2, 6, 3, 9]);
    /// ```
    #[must_use]
    pub fn symmetric_difference(&self, other: &Self) -> Self {
        let mut new_legs = Vec::with_capacity(self.legs().len() + other.legs().len());
        let mut new_bond_dims = Vec::with_capacity(new_legs.capacity());
        for (leg, dim) in self.edges() {
            if !other.legs().contains(leg) {
                new_legs.push(*leg);
                new_bond_dims.push(*dim);
            }
        }
        for (leg, dim) in other.edges() {
            if !self.legs().contains(leg) {
                new_legs.push(*leg);
                new_bond_dims.push(*dim);
            }
        }
        Self::new(new_legs, new_bond_dims)
    }
}

impl Default for LeafTensor {
    fn default() -> Self {
        Self(TensorRepr {
            kind: TensorType::Leaf,
            tensors: Vec::new(),
            legs: Vec::new(),
            bond_dims: Vec::new(),
            tensordata: TensorData::None,
        })
    }
}

impl BitOr for &LeafTensor {
    type Output = LeafTensor;
    #[inline]
    fn bitor(self, rhs: Self) -> Self::Output {
        self.union(rhs)
    }
}

impl BitAnd for &LeafTensor {
    type Output = LeafTensor;
    #[inline]
    fn bitand(self, rhs: Self) -> Self::Output {
        self.intersection(rhs)
    }
}

impl BitXor for &LeafTensor {
    type Output = LeafTensor;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self::Output {
        self.symmetric_difference(rhs)
    }
}

impl Sub for &LeafTensor {
    type Output = LeafTensor;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        self.difference(rhs)
    }
}

impl BitXorAssign<&LeafTensor> for LeafTensor {
    #[inline]
    fn bitxor_assign(&mut self, rhs: &Self) {
        *self = self.symmetric_difference(rhs);
    }
}

impl ApproxEq for &LeafTensor {
    type Margin = F64Margin;

    fn approx_eq<M: Into<Self::Margin>>(self, other: Self, margin: M) -> bool {
        let margin = margin.into();
        if self.legs() != other.legs() {
            return false;
        }
        if self.bond_dims() != other.bond_dims() {
            return false;
        }

        self.tensor_data().approx_eq(other.tensor_data(), margin)
    }
}

impl ApproxEq for &Tensor {
    type Margin = F64Margin;

    fn approx_eq<M: Into<Self::Margin>>(self, other: Self, margin: M) -> bool {
        match (self.kind(), other.kind()) {
            (TensorType::Leaf, TensorType::Leaf) => self
                .as_leaf()
                .unwrap()
                .approx_eq(other.as_leaf().unwrap(), margin),
            (TensorType::Composite, TensorType::Composite) => self
                .as_composite()
                .unwrap()
                .approx_eq(other.as_composite().unwrap(), margin),
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::iter::zip;

    use rustc_hash::FxHashMap;

    use crate::tensornetwork::tensordata::TensorData;

    macro_rules! assert_matches {
        ($left:expr, $pattern:pat) => {
            match $left {
                $pattern => (),
                _ => panic!(
                    "Expected pattern {} but got {:?}",
                    stringify!($pattern),
                    $left
                ),
            }
        };
    }

    mod leaf {
        use super::*;

        #[test]
        fn default() {
            let tensor = LeafTensor::default();
            assert!(tensor.legs().is_empty());
            assert!(tensor.bond_dims().is_empty());
            assert_matches!(tensor.tensor_data(), TensorData::None);
        }

        #[test]
        fn new() {
            let tensor = LeafTensor::new(vec![2, 4, 5], vec![4, 2, 6]);
            assert_eq!(tensor.legs(), &[2, 4, 5]);
            assert_eq!(tensor.bond_dims(), &[4, 2, 6]);
            assert_matches!(tensor.tensor_data(), TensorData::None);
        }

        #[test]
        fn new_from_map() {
            let bond_dims = FxHashMap::from_iter([(1, 1), (2, 4), (3, 7), (4, 2), (5, 6)]);
            let tensor = LeafTensor::new_from_map(vec![2, 4, 5], &bond_dims);
            assert_eq!(tensor.legs(), &[2, 4, 5]);
            assert_eq!(tensor.bond_dims(), &[4, 2, 6]);
            assert_matches!(tensor.tensor_data(), TensorData::None);
        }

        #[test]
        fn new_from_const() {
            let tensor = LeafTensor::new_from_const(vec![9, 2, 5, 1], 3);
            assert_eq!(tensor.legs(), &[9, 2, 5, 1]);
            assert_eq!(tensor.bond_dims(), &[3, 3, 3, 3]);
            assert_matches!(tensor.tensor_data(), TensorData::None);
        }
    }

    mod composite {
        use super::*;

        #[test]
        fn default() {
            let tensor = CompositeTensor::default();
            assert!(tensor.is_empty());
            assert_eq!(tensor.len(), 0);
        }

        #[test]
        fn external_tensor() {
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
            let tensor_1 = LeafTensor::new_from_map(vec![2, 3, 4], &bond_dims);
            let tensor_2 = LeafTensor::new_from_map(vec![2, 3, 5], &bond_dims);
            let tensor_12 = CompositeTensor::new(vec![tensor_1, tensor_2]);

            let tensor_3 = LeafTensor::new_from_map(vec![6, 7, 8], &bond_dims);
            let tensor_4 = LeafTensor::new_from_map(vec![6, 8, 9], &bond_dims);
            let tensor_34 = CompositeTensor::new(vec![tensor_3, tensor_4]);

            let tensor_1234 = CompositeTensor::new(vec![tensor_12, tensor_34]);

            let external = tensor_1234.external_tensor();
            assert_eq!(external.legs(), &[4, 5, 7, 9]);
            assert_eq!(external.bond_dims(), &[6, 8, 12, 16]);
        }

        #[test]
        fn test_push_tensor() {
            let bond_dims =
                FxHashMap::from_iter([(2, 17), (3, 1), (4, 11), (8, 3), (9, 20), (7, 7), (10, 14)]);
            let ref_tensor_1 = LeafTensor::new_from_map(vec![8, 4, 9], &bond_dims);
            let ref_tensor_2 = LeafTensor::new_from_map(vec![7, 10, 2], &bond_dims);

            let mut tensor = CompositeTensor::default();

            // Push tensor 1
            let tensor_1 = LeafTensor::new_from_map(vec![8, 4, 9], &bond_dims);
            tensor.push_tensor(tensor_1);

            for (sub_tensor, ref_tensor) in zip(tensor.tensors(), [&ref_tensor_1]) {
                let sub_tensor = sub_tensor.as_leaf().unwrap();
                assert_eq!(sub_tensor.legs(), ref_tensor.legs());
                assert_eq!(sub_tensor.bond_dims(), ref_tensor.bond_dims());
            }

            // Push tensor 2
            let tensor_2 = LeafTensor::new_from_map(vec![7, 10, 2], &bond_dims);
            tensor.push_tensor(tensor_2);

            for (sub_tensor, ref_tensor) in zip(tensor.tensors(), [&ref_tensor_1, &ref_tensor_2]) {
                let sub_tensor = sub_tensor.as_leaf().unwrap();
                assert_eq!(sub_tensor.legs(), ref_tensor.legs());
                assert_eq!(sub_tensor.bond_dims(), ref_tensor.bond_dims());
            }
        }

        #[test]
        fn test_push_tensors() {
            let bond_dims =
                FxHashMap::from_iter([(2, 17), (3, 1), (4, 11), (8, 3), (9, 20), (7, 7), (10, 14)]);
            let ref_tensor_1 = LeafTensor::new_from_map(vec![4, 3, 2], &bond_dims);
            let ref_tensor_2 = LeafTensor::new_from_map(vec![8, 4, 9], &bond_dims);
            let ref_tensor_3 = LeafTensor::new_from_map(vec![7, 10, 2], &bond_dims);

            let tensor_1 = LeafTensor::new_from_map(vec![4, 3, 2], &bond_dims);
            let tensor_2 = LeafTensor::new_from_map(vec![8, 4, 9], &bond_dims);
            let tensor_3 = LeafTensor::new_from_map(vec![7, 10, 2], &bond_dims);
            let mut tensor = CompositeTensor::default();
            tensor.push_tensors(vec![tensor_1, tensor_2, tensor_3]);

            for (sub_tensor, ref_tensor) in zip(
                tensor.tensors(),
                &vec![ref_tensor_1, ref_tensor_2, ref_tensor_3],
            ) {
                let sub_tensor = sub_tensor.as_leaf().unwrap();
                assert_eq!(sub_tensor.legs(), ref_tensor.legs());
                assert_eq!(sub_tensor.bond_dims(), ref_tensor.bond_dims());
            }
        }
    }
}
