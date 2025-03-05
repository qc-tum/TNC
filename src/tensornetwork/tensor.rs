use core::ops::{BitAnd, BitOr, BitXor, Sub};
use std::hash::{Hash, Hasher};
use std::iter::zip;
use std::ops::BitXorAssign;
use std::sync::{Arc, RwLock, RwLockReadGuard};

use rustc_hash::FxHashMap;

use crate::types::{EdgeIndex, SlicingTask, TensorIndex};
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
    /// tn.push_tensors(vec![v1.clone(), v2.clone()], Some(&bond_dims));
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
    /// tn1.push_tensors(vec![v1, v2], Some(&bond_dims));
    /// let mut tn2 = Tensor::default();
    /// tn2.push_tensors(vec![v3.clone(), v4], Some(&bond_dims));
    /// let mut nested_tn = Tensor::default();
    /// nested_tn.push_tensors(vec![tn1, tn2], Some(&bond_dims));
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
    /// tn.push_tensors(vec![v1, v2], Some(&bond_dims));
    /// tn.insert_bond_dims(&bond_dims);
    /// let mut ref_tensor = Tensor::new(vec![0, 1]);
    /// ref_tensor.insert_bond_dims(&bond_dims);
    /// assert_eq!(tn.tensor(0).legs(), ref_tensor.legs());
    /// ```
    #[inline]
    pub fn tensor(&self, i: TensorIndex) -> &Self {
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
    /// let tn = create_tensor_network(vec![v1, v2], &bond_dims);
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
    /// let mut tn = create_tensor_network(vec![v1, v2], &bond_dims);
    /// tn.insert_bond_dims(&FxHashMap::from_iter([(1, 12), (0, 5)]));
    /// assert_eq!(*tn.bond_dims(), FxHashMap::from_iter([(0, 5), (1, 12), (2, 8)]) );
    /// ```
    pub fn insert_bond_dims(&mut self, bond_dims: &FxHashMap<EdgeIndex, u64>) {
        let mut own_bond_dims = self.bond_dims.write().unwrap();
        own_bond_dims.extend(bond_dims);
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

    /// Returns the number of elements ignoring sliced edges. This is a f64 to avoid overflow in large
    /// tensors.
    ///
    /// # Examples
    /// ```
    /// # use tensorcontraction::tensornetwork::tensor::Tensor;
    /// # use tensorcontraction::tensornetwork::tensordata::TensorData;
    /// # use rustc_hash::FxHashMap;
    /// let mut tensor = Tensor::new(vec![1, 2, 3]);
    /// let bond_dims = FxHashMap::from_iter([(1, 5), (2, 15), (3, 8)]);
    /// let sliced_edges = vec![2];
    /// tensor.insert_bond_dims(&bond_dims);
    /// assert_eq!(tensor.sliced_size(&sliced_edges), 40.0);
    /// ```
    #[inline]
    pub fn sliced_size(&self, slicing: &[EdgeIndex]) -> f64 {
        let bond_dims = self.bond_dims();
        self.legs
            .iter()
            .filter(|e| !slicing.contains(e))
            .map(|e| bond_dims[e] as f64)
            .product()
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
    /// Considers `legs`, `bond_dims` and `tensordata`.
    pub fn approx_eq(&self, other: &Self, epsilon: f64) -> bool {
        if self.tensors.len() != other.tensors.len() {
            return false;
        }
        if self.legs != other.legs {
            return false;
        }
        if !Arc::ptr_eq(&self.bond_dims, &other.bond_dims)
            && *self.bond_dims() != *other.bond_dims()
        {
            return false;
        }

        for (tensor, other_tensor) in zip(&self.tensors, &other.tensors) {
            if !tensor.approx_eq(other_tensor, epsilon) {
                return false;
            }
        }

        self.tensordata.approx_eq(&other.tensordata, epsilon)
    }

    /// Pushes additional `tensor` into this tensor, which must be a composite tensor.
    pub fn push_tensor(&mut self, mut tensor: Self, bond_dims: Option<&FxHashMap<EdgeIndex, u64>>) {
        assert!(
            self.legs.is_empty() && matches!(self.tensordata, TensorData::Uncontracted),
            "Cannot push tensors into a leaf tensor"
        );

        if let Some(bond_dims) = bond_dims {
            self.add_bond_dims(bond_dims);
        }
        self.add_bond_dims(&tensor.bond_dims());
        tensor.bond_dims = Arc::clone(&self.bond_dims);

        self.tensors.push(tensor);
    }

    /// Pushes additional `tensors` into this tensor, which must be a composite tensor.
    pub fn push_tensors(
        &mut self,
        mut tensors: Vec<Self>,
        bond_dims: Option<&FxHashMap<EdgeIndex, u64>>,
    ) {
        assert!(
            self.legs.is_empty() && matches!(self.tensordata, TensorData::Uncontracted),
            "Cannot push tensors into a leaf tensor"
        );

        if let Some(bond_dims) = bond_dims {
            self.add_bond_dims(bond_dims);
        };

        for tensor in &mut tensors {
            tensor.bond_dims = Arc::clone(&self.bond_dims);
        }
        self.tensors.append(&mut tensors);
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

    /// Returns an edge by its id. The edge is a tuple of the two tensors it connects,
    /// or a single tensor if it is an unbound leg.
    fn get_edge(&self, leg_id: EdgeIndex) -> (TensorIndex, Option<TensorIndex>) {
        for (t1_id, t1) in self.tensors.iter().enumerate() {
            if !t1.contains_leg(leg_id) {
                continue;
            }

            for (t2_id, t2) in self.tensors.iter().enumerate().skip(t1_id + 1) {
                if t2.contains_leg(leg_id) {
                    return (t1_id, Some(t2_id));
                }
            }
            return (t1_id, None);
        }
        panic!("Edge {leg_id} not found in tensor");
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
    /// let mut tn = create_tensor_network(vec![v1, v2], &bond_dims);
    /// assert!(tn.is_connected());
    ///
    /// // Introduce a new tensor that is not connected
    /// let v3 = Tensor::new(vec![3]);
    /// tn.push_tensor(v3, None);
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

    /// Applies a slicing to the tensor. The sliced legs are removed from the tensor
    /// and the data of the affected tensors is sliced accordingly.
    pub fn apply_slicing(&mut self, slicing: &SlicingTask) {
        for (leg, index) in &slicing.slices {
            let (t1, Some(t2)) = self.get_edge(*leg) else {
                panic!("Sliced legs must be bound")
            };

            // Replace with sliced data
            let t1 = &mut self.tensors[t1];
            let t1_leg_index = t1.legs.iter().position(|l| l == leg).unwrap();
            let t1_data = std::mem::take(&mut t1.tensordata);
            t1.legs.remove(t1_leg_index);
            t1.set_tensor_data(t1_data.into_sliced(t1_leg_index, *index));

            let t2 = &mut self.tensors[t2];
            let t2_leg_index = t2.legs.iter().position(|l| l == leg).unwrap();
            let t2_data = std::mem::take(&mut t2.tensordata);
            t2.legs.remove(t2_leg_index);
            t2.set_tensor_data(t2_data.into_sliced(t2_leg_index, *index));
        }
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
        let bond_dims = if self.bond_dims().is_empty() {
            other.bond_dims.clone()
        } else {
            self.bond_dims.clone()
        };
        Self::new_with_bonddims(new_legs, bond_dims)
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
        let bond_dims = if self.bond_dims().is_empty() {
            other.bond_dims.clone()
        } else {
            self.bond_dims.clone()
        };
        Self::new_with_bonddims(new_legs, bond_dims)
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
        let bond_dims = if self.bond_dims().is_empty() {
            other.bond_dims.clone()
        } else {
            self.bond_dims.clone()
        };
        Self::new_with_bonddims(new_legs, bond_dims)
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
        let bond_dims = if self.bond_dims().is_empty() {
            other.bond_dims.clone()
        } else {
            self.bond_dims.clone()
        };
        Self::new_with_bonddims(new_legs, bond_dims)
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

        std::mem::take(&mut ext_edges.legs)
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
    use std::iter::zip;

    use num_complex::c64;
    use rustc_hash::FxHashMap;

    use crate::{tensornetwork::tensordata::TensorData, types::SlicingTask};

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
        tensor_12.push_tensors(vec![tensor_1, tensor_2], Some(&bond_dims));

        let mut tensor_34 = Tensor::default();
        let tensor_3 = Tensor::new(vec![6, 7, 8]);
        let tensor_4 = Tensor::new(vec![6, 8, 9]);
        tensor_34.push_tensors(vec![tensor_3, tensor_4], Some(&bond_dims));

        tensor_1234.push_tensors(vec![tensor_12, tensor_34], Some(&bond_dims));

        assert_eq!(tensor_1234.external_edges(), vec![4, 5, 7, 9]);
    }

    #[test]
    fn test_push_tensor() {
        let reference_bond_dims_1 =
            FxHashMap::from_iter([(2, 17), (3, 1), (4, 11), (8, 3), (9, 20)]);
        let reference_bond_dims_2 =
            FxHashMap::from_iter([(2, 17), (3, 1), (4, 11), (8, 3), (9, 20), (7, 7), (10, 14)]);

        let mut ref_tensor_1 = Tensor::new(vec![8, 4, 9]);
        ref_tensor_1.insert_bond_dims(&reference_bond_dims_1);

        let mut ref_tensor_2 = Tensor::new(vec![7, 10, 2]);
        ref_tensor_2.insert_bond_dims(&reference_bond_dims_2);

        let mut tensor = Tensor::default();

        // Push tensor 1
        let tensor_1 = Tensor::new(vec![8, 4, 9]);
        let bond_dims_1 = FxHashMap::from_iter([(8, 3), (9, 20)]);
        tensor.push_tensor(tensor_1, Some(&bond_dims_1));

        assert!(tensor
            .tensor_data()
            .approx_eq(&TensorData::Uncontracted, 1e-12));
        for (key, value) in tensor.bond_dims().iter() {
            assert_eq!(*value, reference_bond_dims_1[key]);
        }

        for (tensor_legs, ref_tensor_legs) in zip(tensor.tensors(), [&ref_tensor_1]) {
            assert_eq!(tensor_legs.legs(), ref_tensor_legs.legs());
        }

        assert_eq!(tensor.legs(), &Vec::<usize>::new());

        // Push tensor 2
        let mut tensor_2 = Tensor::new(vec![7, 10, 2]);
        let bond_dims_2 = FxHashMap::from_iter([(7, 7), (10, 14)]);
        tensor_2.insert_bond_dims(&bond_dims_2);

        tensor.push_tensor(tensor_2, None);
        for (key, value) in tensor.bond_dims().iter() {
            assert_eq!(*value, reference_bond_dims_2[key]);
        }

        for (tensor_legs, ref_tensor_legs) in zip(tensor.tensors(), [&ref_tensor_1, &ref_tensor_2])
        {
            assert_eq!(tensor_legs.legs(), ref_tensor_legs.legs());
        }
    }

    #[test]
    #[should_panic(expected = "Cannot push tensors into a leaf tensor")]
    fn test_push_tensor_to_leaf() {
        let reference_bond_dims =
            FxHashMap::from_iter([(2, 17), (3, 1), (4, 11), (8, 3), (9, 20), (7, 7), (10, 14)]);
        let mut leaf_tensor = Tensor::new(vec![4, 3, 2]);
        leaf_tensor.insert_bond_dims(&reference_bond_dims);

        let mut pushed_tensor = Tensor::new(vec![8, 4, 9]);
        pushed_tensor.insert_bond_dims(&reference_bond_dims);

        leaf_tensor.push_tensor(pushed_tensor, None);
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

        let mut tensor = Tensor::default();

        let tensor_1 = Tensor::new(vec![4, 3, 2]);
        let tensor_2 = Tensor::new(vec![8, 4, 9]);
        let tensor_3 = Tensor::new(vec![7, 10, 2]);
        tensor.push_tensors(
            vec![tensor_1, tensor_2, tensor_3],
            Some(&reference_bond_dims_3),
        );

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
    }

    #[test]
    #[should_panic(expected = "Cannot push tensors into a leaf tensor")]
    fn test_push_tensors_to_leaf() {
        let reference_bond_dims =
            FxHashMap::from_iter([(2, 17), (3, 1), (4, 11), (8, 3), (9, 20), (7, 7), (10, 14)]);
        let mut leaf_tensor = Tensor::new(vec![4, 3, 2]);
        leaf_tensor.insert_bond_dims(&reference_bond_dims);

        let mut pushed_tensor_1 = Tensor::new(vec![8, 4, 9]);
        pushed_tensor_1.insert_bond_dims(&reference_bond_dims);

        let mut pushed_tensor_2 = Tensor::new(vec![7, 10, 2]);
        pushed_tensor_2.insert_bond_dims(&reference_bond_dims);

        leaf_tensor.push_tensors(vec![pushed_tensor_1, pushed_tensor_2], None);
    }

    #[test]
    fn test_apply_slicing() {
        let bond_dims = FxHashMap::from_iter([(0, 2), (1, 2), (2, 2)]);
        let mut t0 = Tensor::new(vec![0, 1]);
        let mut t1 = Tensor::new(vec![1, 2]);
        let mut t2 = Tensor::new(vec![0, 2]);
        t0.set_tensor_data(TensorData::new_from_data(
            &[2, 2],
            vec![c64(0, 0), c64(1, 0), c64(2, 0), c64(3, 0)],
            None,
        ));
        t1.set_tensor_data(TensorData::new_from_data(
            &[2, 2],
            vec![c64(4, 0), c64(5, 0), c64(6, 0), c64(7, 0)],
            None,
        ));
        t2.set_tensor_data(TensorData::new_from_data(
            &[2, 2],
            vec![c64(8, 0), c64(9, 0), c64(10, 0), c64(11, 0)],
            None,
        ));

        let mut tensor = Tensor::default();
        tensor.push_tensors(vec![t0, t1, t2], Some(&bond_dims));

        let slicing_task = SlicingTask {
            slices: vec![(0, 1), (2, 0)],
        };

        tensor.apply_slicing(&slicing_task);

        assert_eq!(tensor.tensor(0).legs(), &[1]);
        assert_eq!(tensor.tensor(1).legs(), &[1]);
        assert!(tensor.tensor(2).legs().is_empty());

        assert!(tensor.tensor(0).tensor_data().approx_eq(
            &TensorData::new_from_data(&[2], vec![c64(1, 0), c64(3, 0)], None),
            1e-12
        ));
        assert!(tensor.tensor(1).tensor_data().approx_eq(
            &TensorData::new_from_data(&[2], vec![c64(4, 0), c64(5, 0)], None),
            1e-12
        ));
        assert!(tensor.tensor(2).tensor_data().approx_eq(
            &TensorData::new_from_data(&[], vec![c64(9, 0)], None),
            1e-12
        ));
    }
}
