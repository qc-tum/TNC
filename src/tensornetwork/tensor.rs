use core::ops::{BitAnd, BitOr, BitXor, Sub};
use std::hash::{Hash, Hasher};
use std::iter::zip;
use std::sync::{Arc, RwLock, RwLockReadGuard};

use itertools::Itertools;
use rustc_hash::FxHashMap;

use crate::types::{EdgeIndex, Vertex};
use crate::utils::datastructures::UnionFind;

use super::tensordata::TensorData;

/// Abstract representation of a tensor.
#[derive(Default, Debug, Clone)]
pub struct Tensor {
    /// The inner tensors that make up this tensor. If non-empty, this tensor is
    /// called a *composite* tensor.
    pub(crate) tensors: Vec<Tensor>,

    /// The legs of the tensor. Each leg is an index to an edge.
    pub(crate) legs: Vec<EdgeIndex>,

    /// The shared bond dimensions. Maps an edge index to the bond dimension.
    pub(crate) bond_dims: Arc<RwLock<FxHashMap<EdgeIndex, u64>>>,

    /// All edges of the tensor. Maps an edge index to the vertices that make up the
    /// edge.
    pub(crate) edges: FxHashMap<EdgeIndex, Vec<Vertex>>,

    /// Maps an edge index which corresponds to a hyperedge to the number of external
    /// tensors that it connects to.
    external_hyperedge: FxHashMap<EdgeIndex, usize>,

    /// The data of the tensor.
    pub(crate) tensordata: TensorData,
}

impl Hash for Tensor {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.legs.hash(state);
    }
}

impl Tensor {
    /// Constructs a Tensor object without underlying data.
    ///
    /// # Arguments
    ///
    /// * `legs` - A vector of usize containing edge ids.
    ///
    /// # Examples
    /// ```
    /// # use tensorcontraction::tensornetwork::tensor::Tensor;
    /// let tensor = Tensor::new(vec![1, 2, 3]);
    /// assert_eq!(tensor.legs(), &[1, 2, 3]);
    /// ```
    #[inline]
    pub fn new(legs: Vec<EdgeIndex>) -> Self {
        Self {
            legs,
            ..Default::default()
        }
    }

    /// Constructs a Tensor object with bond dimensions without underlying data.
    ///
    /// # Examples
    /// ```
    /// # use tensorcontraction::tensornetwork::tensor::Tensor;
    /// # use rustc_hash::FxHashMap;
    /// # use std::sync::{Arc, RwLock};
    /// let bond_dims = FxHashMap::from_iter([(0, 17), (1, 19), (2, 8)]);
    /// let tensor = Tensor::new_with_bonddims(vec![1, 2, 3], Arc::new(RwLock::new(bond_dims.clone())));
    /// assert_eq!(tensor.legs(), &[1, 2, 3]);
    /// assert_eq!(*tensor.bond_dims(), bond_dims);
    /// ```
    #[inline]
    pub fn new_with_bonddims(
        legs: Vec<EdgeIndex>,
        bond_dims: Arc<RwLock<FxHashMap<EdgeIndex, u64>>>,
    ) -> Self {
        Self {
            legs,
            bond_dims,
            ..Default::default()
        }
    }

    /// Returns edge ids of Tensor object.
    ///
    /// # Examples
    /// ```
    /// # use tensorcontraction::tensornetwork::tensor::Tensor;
    /// let vec = vec![1, 2, 3];
    /// let tensor = Tensor::new(vec.clone()) ;
    /// assert_eq!(tensor.legs(), &vec);
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

    /// Getter for list of Tensor objects.
    ///
    /// # Examples
    ///
    /// ```
    /// # use tensorcontraction::tensornetwork::tensor::Tensor;
    /// # use tensorcontraction::tensornetwork::tensordata::TensorData;
    /// # use rustc_hash::FxHashMap;
    /// let mut v1 = Tensor::new(vec![0, 1]);
    /// let mut v2 = Tensor::new(vec![1, 2]);
    /// let bond_dims = FxHashMap::from_iter([
    /// (0, 17), (1, 19), (2, 8)
    /// ]);
    /// let mut tn = Tensor::default();
    /// tn.push_tensors(vec![v1.clone(), v2.clone()], Some(&bond_dims), None);
    /// v1.insert_bond_dims(&bond_dims);
    /// v2.insert_bond_dims(&bond_dims);
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
    /// # use tensorcontraction::tensornetwork::tensordata::TensorData;
    /// # use rustc_hash::FxHashMap;
    /// let mut v1 = Tensor::new(vec![0, 1]);
    /// let mut v2 = Tensor::new(vec![1, 2]);
    /// let mut v3 = Tensor::new(vec![2, 3]);
    /// let mut v4 = Tensor::new(vec![3, 4]);
    /// let bond_dims = FxHashMap::from_iter([
    /// (0, 17), (1, 19), (2, 8), (3, 2), (4, 1)
    /// ]);
    /// let mut tn1 = Tensor::default();
    /// tn1.push_tensors(vec![v1, v2], Some(&bond_dims), None);
    /// let mut tn2 = Tensor::default();
    /// tn2.push_tensors(vec![v3.clone(), v4], Some(&bond_dims), None);
    /// let mut nested_tn = Tensor::default();
    /// nested_tn.push_tensors(vec![tn1, tn2], Some(&bond_dims), None);
    /// v3.insert_bond_dims(&bond_dims);
    ///
    /// assert_eq!(nested_tn.nested_tensor(&[1, 0]).legs(), v3.legs());
    ///
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
    /// # use tensorcontraction::tensornetwork::tensordata::TensorData;
    /// # use rustc_hash::FxHashMap;
    /// let mut v1 = Tensor::new(vec![0, 1]);
    /// let mut v2 = Tensor::new(vec![1, 2]);
    /// let bond_dims = FxHashMap::from_iter([
    /// (0, 17), (1, 19), (2, 8)
    /// ]);
    /// let mut tn = Tensor::default();
    /// tn.push_tensors(vec![v1, v2], Some(&bond_dims), None);
    /// tn.insert_bond_dims(&bond_dims);
    /// let mut ref_tensor = Tensor::new(vec![0, 1]);
    /// ref_tensor.insert_bond_dims(&bond_dims);
    /// assert_eq!(tn.tensor(0).legs(), ref_tensor.legs());
    /// ```
    #[inline]
    pub fn tensor(&self, i: usize) -> &Self {
        &self.tensors[i]
    }

    /// Getter for bond dimensions.
    ///
    /// # Examples
    ///
    /// ```
    /// # use tensorcontraction::tensornetwork::tensor::Tensor;
    /// # use tensorcontraction::tensornetwork::create_tensor_network;
    /// # use rustc_hash::FxHashMap;
    /// let v1 = Tensor::new(vec![0, 1]);
    /// let v2 = Tensor::new(vec![1, 2]);
    /// let bond_dims = FxHashMap::from_iter([
    /// (0, 17), (1, 19), (2, 8)
    /// ]);
    /// let tn = create_tensor_network(vec![v1, v2], &bond_dims, None);
    /// assert_eq!(*tn.bond_dims(), bond_dims);
    /// ```
    #[inline]
    pub fn bond_dims(&self) -> RwLockReadGuard<FxHashMap<EdgeIndex, u64>> {
        self.bond_dims.read().unwrap()
    }

    /// Setter for multiple bond dimensions. Overwrites existing bond dimensions.
    ///
    /// # Examples
    ///
    /// ```
    /// # use tensorcontraction::tensornetwork::tensor::Tensor;
    /// # use tensorcontraction::tensornetwork::create_tensor_network;
    /// # use rustc_hash::FxHashMap;
    /// let v1 = Tensor::new(vec![0, 1]);
    /// let v2 = Tensor::new(vec![1, 2]);
    /// let bond_dims = FxHashMap::from_iter([
    /// (0, 17), (1, 19), (2, 8)
    /// ]);
    /// let mut tn = create_tensor_network(vec![v1, v2], &bond_dims, None);
    /// tn.insert_bond_dims(&FxHashMap::from_iter([(1, 12), (0, 5)]));
    /// assert_eq!(*tn.bond_dims(), FxHashMap::from_iter([(0, 5), (1, 12), (2, 8)]) );
    /// ```
    pub fn insert_bond_dims(&mut self, bond_dims: &FxHashMap<EdgeIndex, u64>) {
        let mut own_bond_dims = self.bond_dims.write().unwrap();
        for (k, v) in bond_dims {
            own_bond_dims.insert(*k, *v);
        }
    }

    /// Getter for edges
    ///
    /// # Examples
    ///
    /// ```
    /// # use tensorcontraction::tensornetwork::tensor::Tensor;
    /// # use tensorcontraction::tensornetwork::create_tensor_network;
    /// # use tensorcontraction::types::*;
    /// # use rustc_hash::FxHashMap;
    /// let v1 = Tensor::new(vec![0, 1]);
    /// let v2 = Tensor::new(vec![1, 2]);
    /// let bond_dims = FxHashMap::from_iter([
    /// (0, 17), (1, 19), (2, 8)
    /// ]);
    /// let mut tn = Tensor::default();
    /// tn.push_tensors(vec![v1, v2], Some(&bond_dims), None);
    /// assert_eq!(tn.edges(), &FxHashMap::from_iter(
    /// [
    /// (0, vec![Vertex::Closed(0), Vertex::Open]),
    /// (1, vec![Vertex::Closed(0), Vertex::Closed(1)]),
    /// (2, vec![Vertex::Closed(1), Vertex::Open])
    /// ]));
    /// ```
    #[inline]
    pub fn edges(&self) -> &FxHashMap<EdgeIndex, Vec<Vertex>> {
        &self.edges
    }

    /// Returns the shape.
    ///
    /// # Examples
    /// ```
    /// # use tensorcontraction::tensornetwork::tensor::Tensor;
    /// # use rustc_hash::FxHashMap;
    /// let legs = vec![0, 1, 2];
    /// let bond_dims = FxHashMap::from_iter([
    /// (0, 17), (1, 19), (2, 8)
    /// ]);
    /// let mut tensor = Tensor::new(legs);
    /// tensor.insert_bond_dims(&bond_dims);
    ///
    /// assert_eq!(tensor.shape(), vec![17, 19, 8]);
    /// ```
    #[inline]
    pub fn shape(&self) -> Vec<usize> {
        let bond_dims = self.bond_dims();
        self.legs.iter().map(|e| bond_dims[e] as usize).collect()
    }

    /// Returns the number of dimensions.
    ///
    /// # Examples
    /// ```
    /// # use tensorcontraction::tensornetwork::tensor::Tensor;
    /// let legs = vec![1, 2, 3];
    /// let tensor = Tensor::new(legs);
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
    /// # use tensorcontraction::tensornetwork::tensordata::TensorData;
    /// # use rustc_hash::FxHashMap;
    /// let mut tensor = Tensor::new(vec![1, 2, 3]);
    /// let bond_dims = FxHashMap::from_iter([(1, 5), (2, 15), (3, 8)]);
    /// tensor.insert_bond_dims(&bond_dims);
    /// assert_eq!(tensor.size(), 600.0);
    /// ```
    #[inline]
    pub fn size(&self) -> f64 {
        let bond_dims = self.bond_dims();
        self.legs.iter().map(|e| bond_dims[e] as f64).product()
    }

    /// Returns true if Tensor contains `leg_id`.
    ///
    /// # Arguments
    ///
    /// * `leg_id` - `usize` referencing specific leg
    ///
    /// # Examples
    /// ```
    /// # use tensorcontraction::tensornetwork::tensor::Tensor;
    /// let tensor = Tensor::new(vec![1, 2, 3]);
    /// assert_eq!(tensor.contains_leg(2), true);
    /// assert_eq!(tensor.contains_leg(4), false);
    /// ```
    #[inline]
    pub fn contains_leg(&self, leg_id: EdgeIndex) -> bool {
        self.legs.contains(&leg_id)
    }

    /// Returns true if Tensor is a leaf tensor, without any nested tensors.
    ///
    /// # Examples
    /// ```
    /// # use tensorcontraction::tensornetwork::tensor::Tensor;
    /// let tensor = Tensor::new(vec![1, 2, 3]);
    /// assert_eq!(tensor.is_leaf(), true);
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
    /// let mut tensor = Tensor::new(vec![1, 2, 3]);
    /// assert_eq!(tensor.is_composite(), false);
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
    /// Considers `legs`, `bond_dims`, `external_hyperedges` and `tensordata`.
    /// `edges` are not compared as different contraction ordering will result in
    /// different edges even though the tensors are otherwise identical.
    pub fn approx_eq(&self, other: &Self, epsilon: f64) -> bool {
        if self.tensors.len() != other.tensors.len() {
            return false;
        }
        if self.legs != other.legs {
            return false;
        }
        if *self.bond_dims() != *other.bond_dims() {
            return false;
        }
        if self.external_hyperedge != other.external_hyperedge {
            return false;
        }
        for (tensor, other_tensor) in zip(&self.tensors, &other.tensors) {
            if !tensor.approx_eq(other_tensor, epsilon) {
                return false;
            }
        }

        self.tensordata.approx_eq(&other.tensordata, epsilon)
    }

    /// Pushes additional tensor into Tensor object. If self is a leaf tensor, clone it and push it into itself.
    ///
    /// # Arguments
    /// * `tensor` - new `Tensor` to be added
    /// * `bond_dims` - `FxHashMap<usize, u64>` mapping edge id to bond dimension
    pub fn push_tensor(&mut self, mut tensor: Self, bond_dims: Option<&FxHashMap<usize, u64>>) {
        // In the case of pushing to an empty tensor, avoid unnecessary hierarchies
        if self.is_empty() {
            let Self {
                legs,
                tensors: _,
                bond_dims: _,
                edges: _,
                external_hyperedge,
                tensordata,
            } = tensor;
            self.set_legs(legs);
            self.set_tensor_data(tensordata);

            if let Some(bond_dims) = bond_dims {
                self.add_bond_dims(bond_dims);
            }
            for (&key, &value) in &external_hyperedge {
                self.external_hyperedge
                    .entry(key)
                    .and_modify(|old_val| *old_val += value)
                    .or_insert_with(|| value);
            }
            self.update_external_edges(&external_hyperedge);

            return;
        }

        if self.is_leaf() {
            let old_self = self.clone();
            // Only update legs once contraction is complete to keep track of data permutation
            self.legs = Vec::new();
            self.update_tensor_edges(&old_self);
            self.set_tensor_data(TensorData::Uncontracted);
            self.tensors.push(old_self);
        }

        if let Some(bond_dims) = bond_dims {
            self.add_bond_dims(bond_dims);
        };

        tensor.bond_dims = Arc::clone(&self.bond_dims);
        self.update_tensor_edges(&tensor);
        self.update_external_edges(&tensor.external_hyperedge);

        self.tensors.push(tensor);
    }

    /// Pushes additional tensor into Tensor object. If self is a leaf tensor, clone it and push it into itself.
    /// Assumes that added tensors do not have external hyperedges, which is fed as an optional argument instead
    ///
    /// # Arguments
    /// * `tensors` - `Vec<Tensor>` to be added
    /// * `bond_dims` - `FxHashMap<usize, u64>` mapping edge id to bond dimension
    /// * `external_hyperedge` - Optional `FxHashMap<EdgeIndex, usize>` of external hyperedges, mapping the edge index to count of external hyperedges.
    pub fn push_tensors(
        &mut self,
        tensors: Vec<Self>,
        bond_dims: Option<&FxHashMap<EdgeIndex, u64>>,
        external_hyperedge: Option<&FxHashMap<EdgeIndex, usize>>,
    ) {
        // If self is a leaf tensor but not empty (i.e. it has legs or data), need to preserve it
        if !self.is_empty() && self.is_leaf() {
            let old_self = self.clone();
            // Only update legs once contraction is complete to keep track of data permutation
            self.legs = Vec::new();
            self.update_tensor_edges(&old_self);
            self.set_tensor_data(TensorData::Uncontracted);
            self.tensors.push(old_self);
        }

        if let Some(bond_dims) = bond_dims {
            self.add_bond_dims(bond_dims);
        };

        self.tensors.reserve(tensors.len());
        for mut tensor in tensors {
            tensor.bond_dims = Arc::clone(&self.bond_dims);
            self.update_tensor_edges(&tensor);
            self.tensors.push(tensor);
        }

        if let Some(external_hyperedge) = external_hyperedge {
            self.update_external_edges(external_hyperedge);
        };
    }

    /// Internal method to update bond dimensions based on `bond_dims`. Only incorporates missing dimensions,
    /// existing keys are not changed.
    fn add_bond_dims(&mut self, bond_dims: &FxHashMap<EdgeIndex, u64>) {
        let mut shared_bond_dims = self.bond_dims.write().unwrap();
        for (key, value) in bond_dims {
            shared_bond_dims
                .entry(*key)
                .and_modify(|e| assert_eq!(e, value, "Updating bond dims will overwrite entry at key {key} with value {e} with new value of {value}"))
                .or_insert(*value);
        }
    }

    /// Internal method to update hyperedges in edge `FxHashMap`. Adds an additional open vertex to each indicated edge if they are missing
    pub(super) fn update_external_edges(
        &mut self,
        external_hyperedge: &FxHashMap<EdgeIndex, usize>,
    ) {
        for (&edge_index, &count) in external_hyperedge {
            self.edges.entry(edge_index).and_modify(|edge| {
                let missing_hyperedges =
                    edge.iter().filter(|e| e == &&Vertex::Open).count() - count;
                edge.append(&mut vec![Vertex::Open; missing_hyperedges]);
            });
        }
    }

    /// Internal method to update edges in tensornetwork after new tensor is added.
    /// If existing edges are introduced, assume that a contraction occurs between them
    /// Otherwise, introduce a new open vertex in edges
    pub(super) fn update_tensor_edges(&mut self, tensor: &Self) {
        let shared_bond_dims = self.bond_dims.read().unwrap();

        // Index is current length as tensor is pushed after.
        let index = self.tensors.len();
        for &leg in &tensor.legs {
            assert!(
                shared_bond_dims.contains_key(&leg),
                "Leg {leg} bond dimension is not defined"
            );

            // Never introduces a Vertex::Open as this is handled in `[update_external_hyperedge]`
            self.edges
                .entry(leg)
                .and_modify(|edge| {
                    // New tensor contracts on a previous external leg
                    if let Some(pos) = edge.iter().position(|e| e == &Vertex::Open) {
                        edge[pos] = Vertex::Closed(index);
                        // Leg is no longer external as it contracts with new tensor
                    } else {
                        // New tensor adds a hyper edge to graph
                        edge.push(Vertex::Closed(index));
                    }
                })
                .or_insert_with(|| vec![Vertex::Closed(index), Vertex::Open]);
        }
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
    /// let mut tensor = Tensor::new(vec![0, 1]);
    /// let tensordata = TensorData::Gate((String::from("X"), vec![]));
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
    ///
    /// ```
    /// # use tensorcontraction::tensornetwork::tensor::Tensor;
    /// # use tensorcontraction::tensornetwork::create_tensor_network;
    /// # use rustc_hash::FxHashMap;
    /// // Create a tensor network with two connected tensors
    /// let v1 = Tensor::new(vec![0, 1]);
    /// let v2 = Tensor::new(vec![1, 2]);
    /// let bond_dims = FxHashMap::from_iter([
    /// (0, 17), (1, 19), (2, 8), (3, 5)
    /// ]);
    /// let mut tn = create_tensor_network(vec![v1, v2], &bond_dims, None);
    /// assert!(tn.is_connected());
    ///
    /// // Introduce a new tensor that is not connected
    /// let v3 = Tensor::new(vec![3]);
    /// tn.push_tensor(v3, None);
    /// assert!(!tn.is_connected());
    /// ```
    pub fn is_connected(&self) -> bool {
        let mut uf = UnionFind::new(self.tensors.len());
        for edge in self.edges.values() {
            edge.iter()
                .filter_map(|v| match v {
                    Vertex::Closed(i) => Some(*i),
                    Vertex::Open => None,
                })
                .tuple_windows()
                .for_each(|(ta, tb)| uf.union(ta, tb));
        }
        uf.count_sets() == 1
    }

    /// Returns `Tensor` with legs in `self` that are not in `other`.
    ///
    /// # Examples
    /// ```
    /// # use tensorcontraction::tensornetwork::tensor::Tensor;
    /// let tensor1 = Tensor::new(vec![1, 2, 3]);
    /// let tensor2 = Tensor::new(vec![4, 2, 5]);
    /// let diff_tensor = &tensor1 - &tensor2;
    /// assert_eq!(diff_tensor.legs(), &[1, 3]);
    /// ```
    #[must_use]
    pub fn difference(&self, other: &Self) -> Self {
        let mut new_legs = Vec::with_capacity(self.legs.len());
        for &i in &self.legs {
            if !other.contains_leg(i) {
                new_legs.push(i);
            }
        }
        Self::new_with_bonddims(new_legs, Arc::clone(&self.bond_dims))
    }

    /// Returns `Tensor` with union of legs in both `self` and `other`.
    ///
    /// # Examples
    /// ```
    /// # use tensorcontraction::tensornetwork::tensor::Tensor;
    /// let tensor1 = Tensor::new(vec![1, 2, 3]);
    /// let tensor2 = Tensor::new(vec![4, 2, 5]);
    /// let union_tensor = &tensor1 | &tensor2;
    /// assert_eq!(union_tensor.legs(), &[1, 2, 3, 4, 5]);
    /// ```
    #[must_use]
    pub fn union(&self, other: &Self) -> Self {
        let mut new_legs = Vec::with_capacity(self.legs.len() + other.legs.len());
        new_legs.extend_from_slice(&self.legs);
        for &i in &other.legs {
            if !self.contains_leg(i) {
                new_legs.push(i);
            }
        }
        Self::new_with_bonddims(new_legs, Arc::clone(&self.bond_dims))
    }

    /// Returns `Tensor` with intersection of legs in `self` and `other`.
    ///
    /// # Examples
    /// ```
    /// # use tensorcontraction::tensornetwork::tensor::Tensor;
    /// let tensor1 = Tensor::new(vec![1, 2, 3]);
    /// let tensor2 = Tensor::new(vec![4, 2, 5]);
    /// let intersection_tensor = &tensor1 & &tensor2;
    /// assert_eq!(intersection_tensor.legs(), &[2]);
    /// ```
    #[must_use]
    pub fn intersection(&self, other: &Self) -> Self {
        let mut new_legs = Vec::with_capacity(self.legs.len().min(other.legs.len()));
        for &i in &self.legs {
            if other.contains_leg(i) {
                new_legs.push(i);
            }
        }
        Self::new_with_bonddims(new_legs, Arc::clone(&self.bond_dims))
    }

    /// Returns `Tensor` with symmetrical difference of legs in `self` and `other`.
    ///
    /// # Examples
    /// ```
    /// # use tensorcontraction::tensornetwork::tensor::Tensor;
    /// let tensor1 = Tensor::new(vec![1, 2, 3]);
    /// let tensor2 = Tensor::new(vec![4, 2, 5]);
    /// let sym_dif_tensor = &tensor1 ^ &tensor2;
    /// assert_eq!(sym_dif_tensor.legs(), &[1, 3, 4, 5]);
    /// ```
    #[must_use]
    pub fn symmetric_difference(&self, other: &Self) -> Self {
        let mut new_legs = Vec::with_capacity(self.legs.len() + other.legs.len());
        for &i in &self.legs {
            if !other.contains_leg(i) {
                new_legs.push(i);
            }
        }
        for &i in &other.legs {
            if !self.contains_leg(i) {
                new_legs.push(i);
            }
        }
        Self::new_with_bonddims(new_legs, Arc::clone(&self.bond_dims))
    }

    /// Get output legs after tensor contraction
    pub fn external_edges(&self) -> Vec<EdgeIndex> {
        if self.is_leaf() {
            return self.legs.clone();
        }

        let mut ext_edges = Self::default();
        for tensor in &self.tensors {
            let new_tensor = if tensor.is_composite() {
                &Tensor::new(tensor.external_edges())
            } else {
                tensor
            };
            ext_edges = &ext_edges ^ new_tensor;
        }
        let mut ext_edges = std::mem::take(&mut ext_edges.legs);
        for (&edge_index, &count) in &self.external_hyperedge {
            ext_edges.append(&mut vec![edge_index; count]);
        }
        ext_edges
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

#[cfg(test)]
mod tests {
    use std::iter::zip;

    use rustc_hash::FxHashMap;

    use crate::{tensornetwork::tensordata::TensorData, types::Vertex};

    use super::Tensor;

    #[test]
    fn test_empty_tensor() {
        let tensor = Tensor::default();
        assert!(tensor.tensors.is_empty());
        assert!(tensor.bond_dims().is_empty());
    }

    #[test]
    fn test_new() {
        let tensor = Tensor::new(vec![2, 4, 5]);
        assert_eq!(tensor.legs(), &vec![2, 4, 5]);
        assert_eq!(tensor.dims(), 3);
        assert!(tensor
            .tensor_data()
            .approx_eq(&TensorData::Uncontracted, 1e-12));
    }

    #[test]
    #[should_panic(
        expected = "assertion `left == right` failed: Updating bond dims will overwrite entry at key 2 with value 5 with new value of 17\n  left: 5\n right: 17"
    )]
    fn test_add_bond_dims() {
        let mut tensor = Tensor::new(vec![2, 4, 5]);

        let bond_dims = FxHashMap::from_iter([(2, 5), (7, 24), (9, 2)]);
        tensor.add_bond_dims(&bond_dims);

        let bond_dims = FxHashMap::from_iter([(2, 17), (4, 11), (5, 14)]);
        tensor.add_bond_dims(&bond_dims);
    }

    #[test]
    fn test_external_legs() {
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
        let mut tensor_1234 = Tensor::default();
        let mut tensor_12 = Tensor::default();

        let tensor_1 = Tensor::new(vec![2, 3, 4]);
        let tensor_2 = Tensor::new(vec![2, 3, 5]);
        tensor_12.push_tensors(vec![tensor_1, tensor_2], Some(&bond_dims), None);

        let mut tensor_34 = Tensor::default();
        let tensor_3 = Tensor::new(vec![6, 7, 8]);
        let tensor_4 = Tensor::new(vec![6, 8, 9]);
        tensor_34.push_tensors(vec![tensor_3, tensor_4], Some(&bond_dims), None);

        tensor_1234.push_tensors(vec![tensor_12, tensor_34], Some(&bond_dims), None);

        assert_eq!(vec![4, 5, 7, 9], tensor_1234.external_edges());
    }

    #[test]
    fn test_push_tensor() {
        let reference_bond_dims_1 = FxHashMap::from_iter([(2, 17), (3, 1), (4, 11)]);
        let reference_bond_dims_2 =
            FxHashMap::from_iter([(2, 17), (3, 1), (4, 11), (8, 3), (9, 20)]);
        let reference_bond_dims_3 =
            FxHashMap::from_iter([(2, 17), (3, 1), (4, 11), (8, 3), (9, 20), (7, 7), (10, 14)]);

        let mut ref_tensor_1 = Tensor::new(vec![4, 3, 2]);
        ref_tensor_1.insert_bond_dims(&reference_bond_dims_1);

        let mut ref_tensor_2 = Tensor::new(vec![8, 4, 9]);
        ref_tensor_2.insert_bond_dims(&reference_bond_dims_2);

        let mut ref_tensor_3 = Tensor::new(vec![7, 10, 2]);
        ref_tensor_3.insert_bond_dims(&reference_bond_dims_3);

        let mut tensor = ref_tensor_1.clone();

        let tensor_2 = Tensor::new(vec![8, 4, 9]);
        let bond_dims_2 = FxHashMap::from_iter([(8, 3), (9, 20)]);
        tensor.push_tensor(tensor_2, Some(&bond_dims_2));

        assert!(tensor
            .tensor_data()
            .approx_eq(&TensorData::Uncontracted, 1e-12),);
        for (key, value) in tensor.bond_dims().iter() {
            assert_eq!(reference_bond_dims_2[key], *value);
        }

        for (tensor_legs, other_tensor_legs) in zip(
            tensor.tensors(),
            &vec![ref_tensor_1.clone(), ref_tensor_2.clone()],
        ) {
            assert_eq!(tensor_legs.legs(), other_tensor_legs.legs());
        }

        assert_eq!(tensor.legs(), &Vec::<usize>::new());

        let tensor_3 = Tensor::new(vec![7, 10, 2]);
        let bond_dims_3 = FxHashMap::from_iter([(7, 7), (10, 14)]);

        tensor.push_tensor(tensor_3, Some(&bond_dims_3));
        for (key, value) in tensor.bond_dims().iter() {
            assert_eq!(reference_bond_dims_3[key], *value);
        }
        ref_tensor_2.insert_bond_dims(&reference_bond_dims_3);

        for (tensor_legs, other_tensor_legs) in zip(
            tensor.tensors(),
            &vec![ref_tensor_1, ref_tensor_2, ref_tensor_3],
        ) {
            assert_eq!(tensor_legs.legs(), other_tensor_legs.legs());
        }

        assert_eq!(
            tensor.edges(),
            &FxHashMap::from_iter([
                (2, vec![Vertex::Closed(0), Vertex::Closed(2)]),
                (3, vec![Vertex::Closed(0), Vertex::Open]),
                (4, vec![Vertex::Closed(0), Vertex::Closed(1)]),
                (7, vec![Vertex::Closed(2), Vertex::Open]),
                (8, vec![Vertex::Closed(1), Vertex::Open]),
                (9, vec![Vertex::Closed(1), Vertex::Open]),
                (10, vec![Vertex::Closed(2), Vertex::Open]),
            ])
        );
    }

    #[test]
    fn test_push_tensors() {
        let reference_bond_dims_1 = FxHashMap::from_iter([(2, 17), (3, 1), (4, 11)]);
        let reference_bond_dims_3 =
            FxHashMap::from_iter([(2, 17), (3, 1), (4, 11), (8, 3), (9, 20), (7, 7), (10, 14)]);

        let mut ref_tensor_1 = Tensor::new(vec![4, 3, 2]);
        ref_tensor_1.insert_bond_dims(&reference_bond_dims_1);

        let mut ref_tensor_2 = Tensor::new(vec![8, 4, 9]);
        ref_tensor_2.insert_bond_dims(&reference_bond_dims_3);

        let mut ref_tensor_3 = Tensor::new(vec![7, 10, 2]);
        ref_tensor_3.insert_bond_dims(&reference_bond_dims_3);

        let mut tensor = ref_tensor_1.clone();

        let tensor_2 = Tensor::new(vec![8, 4, 9]);
        let tensor_3 = Tensor::new(vec![7, 10, 2]);
        tensor.push_tensors(vec![tensor_2, tensor_3], Some(&reference_bond_dims_3), None);

        assert!(tensor
            .tensor_data()
            .approx_eq(&TensorData::Uncontracted, 1e-12));

        for (tensor, other_tensor) in zip(
            tensor.tensors(),
            &vec![ref_tensor_1, ref_tensor_2, ref_tensor_3],
        ) {
            assert_eq!(tensor.legs(), other_tensor.legs());
        }

        assert_eq!(*tensor.bond_dims(), reference_bond_dims_3);
        assert_eq!(
            tensor.edges(),
            &FxHashMap::from_iter([
                (2, vec![Vertex::Closed(0), Vertex::Closed(2)]),
                (3, vec![Vertex::Closed(0), Vertex::Open]),
                (4, vec![Vertex::Closed(0), Vertex::Closed(1)]),
                (7, vec![Vertex::Closed(2), Vertex::Open]),
                (8, vec![Vertex::Closed(1), Vertex::Open]),
                (9, vec![Vertex::Closed(1), Vertex::Open]),
                (10, vec![Vertex::Closed(2), Vertex::Open]),
            ])
        );
    }

    #[test]
    fn test_push_tensor_hyperedges() {
        let reference_bond_dims_0 = FxHashMap::from_iter([(2, 17), (3, 1), (4, 11)]);
        let reference_bond_dims_1 = FxHashMap::from_iter([(2, 17), (3, 1), (4, 11), (9, 20)]);
        let reference_bond_dims_2 = FxHashMap::from_iter([(2, 17), (3, 1), (4, 11), (9, 20)]);

        let reference_bond_dims_3 =
            FxHashMap::from_iter([(2, 17), (3, 1), (4, 11), (5, 2), (6, 6), (9, 20)]);

        let mut ref_tensor_0 = Tensor::new(vec![4, 3, 2]);
        ref_tensor_0.insert_bond_dims(&reference_bond_dims_0);

        let mut ref_tensor_1 = Tensor::new(vec![3, 4, 9]);
        ref_tensor_1.insert_bond_dims(&reference_bond_dims_1);

        let mut ref_tensor_2 = Tensor::new(vec![4, 9, 2]);
        ref_tensor_2.insert_bond_dims(&reference_bond_dims_2);

        let mut ref_tensor_3 = Tensor::new(vec![5, 6, 6]);
        ref_tensor_3.insert_bond_dims(&reference_bond_dims_3);

        let mut tensor = ref_tensor_0.clone();

        let tensor_1 = Tensor::new(vec![3, 4, 9]);
        let bond_dims_1 = FxHashMap::from_iter([(9, 20)]);
        tensor.push_tensor(tensor_1, Some(&bond_dims_1));

        assert_eq!(reference_bond_dims_1, *tensor.bond_dims());
        assert_eq!(tensor.legs(), &Vec::<usize>::new());

        let tensor_2 = Tensor::new(vec![4, 9, 2]);

        tensor.push_tensor(tensor_2, None);
        assert_eq!(reference_bond_dims_2, *tensor.bond_dims());

        let tensor_3 = Tensor::new(vec![5, 6, 6]);
        let bond_dims_3 = FxHashMap::from_iter([(5, 2), (6, 6)]);
        tensor.push_tensor(tensor_3, Some(&bond_dims_3));
        assert_eq!(reference_bond_dims_3, *tensor.bond_dims());

        for (tensor_legs, other_tensor_legs) in zip(
            tensor.tensors(),
            &vec![ref_tensor_0, ref_tensor_1, ref_tensor_2, ref_tensor_3],
        ) {
            assert_eq!(tensor_legs.legs(), other_tensor_legs.legs());
        }

        assert_eq!(
            tensor.edges(),
            &FxHashMap::from_iter([
                (2, vec![Vertex::Closed(0), Vertex::Closed(2)]),
                (3, vec![Vertex::Closed(0), Vertex::Closed(1)]),
                (
                    4,
                    vec![Vertex::Closed(0), Vertex::Closed(1), Vertex::Closed(2)]
                ),
                (5, vec![Vertex::Closed(3), Vertex::Open]),
                (6, vec![Vertex::Closed(3), Vertex::Closed(3)]),
                (9, vec![Vertex::Closed(1), Vertex::Closed(2)]),
            ])
        );
    }
}
