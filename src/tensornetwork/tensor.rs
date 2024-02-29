use array_tool::vec::{Intersect, Union};
use core::ops::{BitAnd, BitOr, BitXor, Sub};
use itertools::Itertools;
use std::cell::{Ref, RefCell};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::ops::Index;
use std::sync::Arc;

use crate::types::{EdgeIndex, Vertex};

use super::tensordata::TensorData;

#[derive(Debug, Eq, Clone)]
/// Abstract representation of a tensor.
pub struct Tensor {
    pub(crate) tensors: Vec<Tensor>,
    pub(crate) legs: Vec<EdgeIndex>,
    pub(crate) bond_dims: Arc<RefCell<HashMap<EdgeIndex, u64>>>,
    pub(crate) edges: HashMap<EdgeIndex, Vec<Vertex>>,
    tensordata: RefCell<TensorData>,
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        if *self.get_bond_dims() != *other.get_bond_dims() {
            return false;
        }
        if *self.get_legs() != *other.get_legs() {
            return false;
        }
        if *self.get_tensor_data() != *other.get_tensor_data() {
            return false;
        }
        let other_edges = other.get_edges();
        for (k, v) in self.get_edges().iter() {
            if !(other_edges[k].iter().sorted().eq(v.iter().sorted())) {
                return false;
            }
        }
        true
    }
}

impl Hash for Tensor {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.legs.hash(state);
    }
}

impl Default for Tensor {
    /// Constructs an empty Tensor object
    ///
    ///
    /// # Examples
    /// ```
    /// use tensorcontraction::tensornetwork::tensor::Tensor;
    /// let tensor = Tensor::default();
    /// ```
    fn default() -> Self {
        Self {
            tensors: Vec::new(),
            legs: Vec::new(),
            bond_dims: Arc::new(RefCell::new(HashMap::new())),
            edges: HashMap::new(),
            tensordata: RefCell::new(TensorData::Uncontracted),
        }
    }
}

impl Tensor {
    /// Constructs a Tensor object without underlying data
    ///
    /// # Arguments
    ///
    /// * `legs` - A vector of usize containing edge ids.
    ///
    /// # Examples
    /// ```
    /// use tensorcontraction::tensornetwork::tensor::Tensor;
    /// let vec = Vec::from([1,2,3]);
    /// let tensor = Tensor::new(vec);
    /// ```
    pub fn new(legs: Vec<EdgeIndex>) -> Self {
        Self {
            tensors: Vec::new(),
            legs,
            bond_dims: Arc::new(RefCell::new(HashMap::new())),
            edges: HashMap::new(),
            tensordata: RefCell::new(TensorData::Uncontracted),
        }
    }

    /// Returns edge ids of Tensor object
    ///
    /// # Examples
    /// ```
    /// use tensorcontraction::tensornetwork::tensor::Tensor;
    /// let vec = Vec::from([1,2,3]);
    /// let tensor = Tensor::new(vec.clone()) ;
    /// assert_eq!(*tensor.get_legs(), vec);
    /// ```
    pub fn get_legs(&self) -> &Vec<EdgeIndex> {
        &self.legs
    }

    /// Returns iterator over leg ids of Tensor object
    ///
    /// # Examples
    /// ```
    /// use tensorcontraction::tensornetwork::tensor::Tensor;
    /// let vec = Vec::from([1,2,3]);
    /// let tensor = Tensor::new(vec.clone()) ;
    /// assert!(tensor.legs_iter().eq(vec.iter()));
    /// ```
    pub fn legs_iter(&self) -> impl Iterator<Item = &EdgeIndex> {
        self.legs.iter()
    }

    /// Internal method to set legs
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
    /// # use std::collections::HashMap;
    /// let mut v1 = Tensor::new(vec![0,1]);
    /// let mut v2 = Tensor::new(vec![1,2]);
    /// let bond_dims = HashMap::from([
    /// (0, 17), (1, 19), (2, 8)
    /// ]);
    /// let mut tn = Tensor::default();
    /// tn.push_tensors(vec![v1.clone(), v2.clone()], Some(&bond_dims), None);
    /// v1.insert_bond_dims(&bond_dims);
    /// v2.insert_bond_dims(&bond_dims);
    /// assert_eq!(*tn.get_tensors(), vec![v1, v2]);
    /// ```
    pub fn get_tensors(&self) -> &Vec<Tensor> {
        &self.tensors
    }

    /// Get ith Tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// # use tensorcontraction::tensornetwork::tensor::Tensor;
    /// # use tensorcontraction::tensornetwork::tensordata::TensorData;
    /// # use std::collections::HashMap;
    /// let mut v1 = Tensor::new(vec![0,1]);
    /// let mut v2 = Tensor::new(vec![1,2]);
    /// let bond_dims = HashMap::from([
    /// (0, 17), (1, 19), (2, 8)
    /// ]);
    /// let mut tn = Tensor::default();
    /// tn.push_tensors(vec![v1.clone(), v2.clone()], Some(&bond_dims), None);
    /// tn.insert_bond_dims(&bond_dims);
    /// let mut ref_tensor = Tensor::new(vec![0,1]);
    /// ref_tensor.insert_bond_dims(&bond_dims);
    /// assert_eq!(*tn.get_tensor(0), ref_tensor);
    /// ```
    pub fn get_tensor(&self, i: usize) -> &Tensor {
        &self.tensors[i]
    }

    /// Getter for iterator over Tensor objects.
    ///
    /// # Examples
    ///
    /// ```
    /// # use tensorcontraction::tensornetwork::tensor::Tensor;
    /// # use tensorcontraction::tensornetwork::tensordata::TensorData;
    /// # use std::collections::HashMap;
    /// let mut v1 = Tensor::new(vec![0,1]);
    /// let mut v2 = Tensor::new(vec![1,2]);
    /// let bond_dims = HashMap::from([
    /// (0, 17), (1, 19), (2, 8)
    /// ]);
    /// let mut tn = Tensor::default();
    /// tn.push_tensors(vec![v1.clone(), v2.clone()], Some(&bond_dims), None);
    /// v1.insert_bond_dims(&bond_dims);
    /// v2.insert_bond_dims(&bond_dims);
    /// assert!(tn.tensor_iter().eq(vec![v1, v2].iter()));
    /// ```
    pub fn tensor_iter(&self) -> impl Iterator<Item = &Tensor> {
        self.get_tensors().iter()
    }

    /// Getter for bond dimensions.
    ///
    /// # Examples
    ///
    /// ```
    /// # use tensorcontraction::tensornetwork::tensor::Tensor;
    /// # use tensorcontraction::tensornetwork::create_tensor_network;
    /// # use std::collections::HashMap;
    /// let v1 = Tensor::new(vec![0,1]);
    /// let v2 = Tensor::new(vec![1,2]);
    /// let bond_dims = HashMap::from([
    /// (0, 17), (1, 19), (2, 8)
    /// ]);
    /// let tn = create_tensor_network(vec![v1,v2], &bond_dims, None);
    /// assert_eq!(*tn.get_bond_dims(), bond_dims);
    /// ```
    pub fn get_bond_dims(&self) -> std::cell::Ref<HashMap<EdgeIndex, u64>> {
        self.bond_dims.borrow()
    }

    /// Setter for single bond dimension.
    ///
    /// # Examples
    ///
    /// ```
    /// # use tensorcontraction::tensornetwork::tensor::Tensor;
    /// # use tensorcontraction::tensornetwork::create_tensor_network;
    /// # use std::collections::HashMap;
    /// let v1 = Tensor::new(vec![0,1]);
    /// let v2 = Tensor::new(vec![1,2]);
    /// let bond_dims = HashMap::from([
    /// (0, 17), (1, 19), (2, 8)
    /// ]);
    /// let mut tn = Tensor::default();
    /// tn.push_tensors(vec![v1,v2], Some(&bond_dims), None);
    /// assert_eq!(*tn.get_bond_dims(), bond_dims);
    /// tn.insert_bond_dim(1, 12);
    /// assert_ne!(*tn.get_bond_dims(), bond_dims);
    /// ```
    pub fn insert_bond_dim(&mut self, k: EdgeIndex, v: u64) {
        self.bond_dims
            .borrow_mut()
            .entry(k)
            .and_modify(|e| {
                *e = v;
            })
            .or_insert(v);
    }

    /// Setter for multiple bond dimensions.
    ///
    /// # Examples
    ///
    /// ```
    /// # use tensorcontraction::tensornetwork::tensor::Tensor;
    /// # use tensorcontraction::tensornetwork::create_tensor_network;
    /// # use std::collections::HashMap;
    /// let v1 = Tensor::new(vec![0,1]);
    /// let v2 = Tensor::new(vec![1,2]);
    /// let bond_dims = HashMap::from([
    /// (0, 17), (1, 19), (2, 8)
    /// ]);
    /// let mut tn = create_tensor_network(vec![v1,v2], &bond_dims, None);
    /// tn.insert_bond_dims(&HashMap::from([(1, 12), (0, 5)]));
    /// assert_eq!(*tn.get_bond_dims(), HashMap::from([(0, 5), (1, 12), (2, 8)]) );
    /// ```
    pub fn insert_bond_dims(&mut self, bond_dims: &HashMap<EdgeIndex, u64>) {
        for (k, v) in bond_dims {
            self.bond_dims
                .borrow_mut()
                .entry(*k)
                .and_modify(|e| {
                    *e = *v;
                })
                .or_insert(*v);
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
    /// # use std::collections::HashMap;
    /// let v1 = Tensor::new(vec![0,1]);
    /// let v2 = Tensor::new(vec![1,2]);
    /// let bond_dims = HashMap::from([
    /// (0, 17), (1, 19), (2, 8)
    /// ]);
    /// let mut tn = Tensor::default();
    /// tn.push_tensors(vec![v1, v2], Some(&bond_dims), None);
    /// assert_eq!(tn.get_edges(), &HashMap::from(
    /// [
    /// (0, vec![Vertex::Closed(0), Vertex::Open]),
    /// (1, vec![Vertex::Closed(0), Vertex::Closed(1)]),
    /// (2, vec![Vertex::Closed(1), Vertex::Open])
    /// ]));
    /// ```
    pub fn get_edges(&self) -> &HashMap<EdgeIndex, Vec<Vertex>> {
        &self.edges
    }

    /// Returns number of dimensions of Tensor object
    ///
    /// # Examples
    /// ```
    /// # use tensorcontraction::tensornetwork::tensor::Tensor;
    /// # use std::collections::HashMap;
    /// let vec = Vec::from([0, 1, 2]);
    /// let bond_dims = HashMap::from([
    /// (0, 17), (1, 19), (2, 8)
    /// ]);
    /// let mut tensor = Tensor::new(vec.clone()) ;
    /// tensor.insert_bond_dims(&bond_dims);
    ///
    /// assert_eq!(tensor.shape(), vec![17, 19, 8]);
    /// ```
    pub fn shape(&self) -> Vec<u64> {
        self.legs
            .iter()
            .map(|e| self.bond_dims.borrow()[e])
            .collect::<Vec<u64>>()
    }

    /// Returns shape of Tensor object
    ///
    /// # Examples
    /// ```
    /// use tensorcontraction::tensornetwork::tensor::Tensor;
    /// let vec = Vec::from([1,2,3]);
    /// let tensor = Tensor::new(vec.clone()) ;
    /// assert_eq!(tensor.dims(), 3);
    /// ```
    pub fn dims(&self) -> usize {
        self.legs.len()
    }

    /// Returns product of leg sizes based on input Hashmap. Returns the number of elements in a tensor
    ///
    /// # Arguments
    ///
    /// * `bond_dim` - Reference to hashmap mapping edge ID to bond dimension size
    ///
    /// # Examples
    /// ```
    /// use tensorcontraction::tensornetwork::tensor::Tensor;
    /// use tensorcontraction::tensornetwork::tensordata::TensorData;
    /// use std::collections::HashMap;
    /// let mut tensor = Tensor::new(Vec::from([1,2,3]));
    /// let bond_dims = HashMap::from([(1, 5),
    /// (2, 15),
    /// (3, 8)]);
    /// tensor.insert_bond_dims(&bond_dims);
    /// assert_eq!(tensor.size(), 600);
    /// ```
    pub fn size(&self) -> u64 {
        self.legs.iter().map(|e| self.get_bond_dims()[e]).product()
    }

    /// Returns true if Tensor contains leg_id
    ///
    /// # Arguments
    ///
    /// * `leg_id` - `usize` referencing specific leg
    ///
    /// # Examples
    /// ```
    /// use tensorcontraction::tensornetwork::tensor::Tensor;
    /// let tensor = Tensor::new(Vec::from([1,2,3]));
    /// assert_eq!(tensor.contains_leg(2), true);
    /// assert_eq!(tensor.contains_leg(4), false);
    /// ```
    pub fn contains_leg(&self, leg_id: EdgeIndex) -> bool {
        self.legs.contains(&leg_id)
    }

    /// Returns true if Tensor is not a tensornetwork
    ///
    /// # Examples
    /// ```
    /// use tensorcontraction::tensornetwork::tensor::Tensor;
    /// let tensor = Tensor::new(Vec::from([1,2,3]));
    /// assert_eq!(tensor.is_tensornetwork(), true);
    /// ```
    pub fn is_tensornetwork(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Returns true if Tensor is composite
    ///
    /// # Examples
    /// ```
    /// use tensorcontraction::tensornetwork::tensor::Tensor;
    /// let mut tensor = Tensor::new(Vec::from([1,2,3]));
    /// assert_eq!(tensor.is_composite(), false);
    /// ```
    pub fn is_composite(&self) -> bool {
        self.get_legs().is_empty()
    }

    /// Pushes additional tensor into Tensor object. If self is a leaf tensor, clone it and push it into itself.
    /// # Arguments
    ///
    /// * `tensor` - new `Tensor` to be added
    /// ```
    pub fn push_tensor(
        &mut self,
        mut tensor: Tensor,
        bond_dims: Option<&HashMap<usize, u64>>,
        external_hyperedge: Option<&Vec<usize>>,
    ) {
        // In the case of pushing to an empty tensor, avoid unnecessary heirarchies
        if self.get_tensors().is_empty() && self.get_legs().is_empty() {
            self.set_legs(tensor.get_legs().clone());
            self.set_tensor_data(tensor.get_tensor_data().clone());
            if let Some(bond_dims) = bond_dims {
                self.update_bond_dims(bond_dims);
            };
            if let Some(external_hyperedge) = external_hyperedge {
                self.update_external_edges(external_hyperedge);
            };
            return;
        }

        if self.get_tensors().is_empty() && !self.get_legs().is_empty() {
            let mut new_self = self.clone();
            // Only update legs once contraction is complete to keep track of data permutation
            self.legs = Vec::new();
            // Don't clone large data is needed.
            self.update_tensor_edges(&mut new_self);
            self.set_tensor_data(TensorData::Uncontracted);
            self.tensors.push(new_self);
        }
        // Ensure that external legs are cleared each time a new tensor is pushed
        if !self.get_legs().is_empty() {
            self.set_legs(vec![]);
        }
        if let Some(bond_dims) = bond_dims {
            self.update_bond_dims(bond_dims);
        };
        if let Some(external_hyperedge) = external_hyperedge {
            self.update_external_edges(external_hyperedge);
        };

        self.update_tensor_edges(&mut tensor);
        self.tensors.push(tensor)
    }

    /// Pushes additional tensor into Tensor object. If self is a leaf tensor, clone it and push it into itself.
    /// # Arguments
    ///
    /// * `tensors` - `Vec<Tensor>` to be added
    /// ```
    pub fn push_tensors(
        &mut self,
        mut tensors: Vec<Tensor>,
        bond_dims: Option<&HashMap<usize, u64>>,
        external_hyperedge: Option<&Vec<usize>>,
    ) {
        // Case that tensor is not empty and has no subtensors.
        if self.get_tensors().is_empty() && !self.get_legs().is_empty() {
            let mut new_self = self.clone();
            // Only update legs once contraction is complete to keep track of data permutation
            self.legs = Vec::new();
            // Don't clone large data is needed.
            self.update_tensor_edges(&mut new_self);
            self.set_tensor_data(TensorData::Uncontracted);
            self.tensors.push(new_self);
        }
        if let Some(bond_dims) = bond_dims {
            self.update_bond_dims(bond_dims);
        };
        if let Some(external_hyperedge) = external_hyperedge {
            self.update_external_edges(external_hyperedge);
        };
        for tensor in tensors.iter_mut() {
            self.update_tensor_edges(tensor);
            self.tensors.push(tensor.clone());
        }
    }

    // Internal method to update bond dimensions based on `bond_dims`. Only incorporates missing dimensions,
    // existing keys are not changed.
    fn update_bond_dims(&mut self, bond_dims: &HashMap<EdgeIndex, u64>) {
        let mut shared_bond_dims = self.bond_dims.borrow_mut();
        for (key, value) in bond_dims.iter() {
            shared_bond_dims.entry(*key).or_insert(*value);
        }
    }

    // Internal method to update hyperedges in edge HashMap. Adds an additional open vertex to each indicated
    // edge
    pub(crate) fn update_external_edges(&mut self, external_hyperedge: &Vec<usize>) {
        for i in external_hyperedge {
            self.edges
                .entry(*i)
                .and_modify(|edge| edge.push(Vertex::Open));
        }
    }

    // Internal method to update edges in tensornetwork after new tensor is added.
    // If existing edges are introduced, assume that a contraction occurs between them
    // Otherwise, introduce a new open vertex in edges
    pub(super) fn update_tensor_edges(&mut self, tensor: &mut Tensor) {
        tensor.bond_dims = Arc::clone(&self.bond_dims);
        let shared_bond_dims = self.bond_dims.borrow();

        // Index is current length as tensor is pushed after.
        let index = self.get_tensors().len();
        for &leg in tensor.get_legs() {
            if !shared_bond_dims.contains_key(&leg) {
                panic!("Leg {leg} bond dimension is not defined");
            }
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
                .or_insert(vec![Vertex::Closed(index), Vertex::Open]);
        }
        // Add new external edges to TensorNetwork
    }

    /// Getter for tensor data.
    ///
    /// # Examples
    ///
    /// ```
    /// # use tensorcontraction::tensornetwork::tensor::Tensor;
    /// # use tensorcontraction::tensornetwork::tensordata::TensorData;
    /// # use std::collections::HashMap;
    /// let tensor = Tensor::new(vec![0,1]);
    /// assert_eq!(*tensor.get_tensor_data(), TensorData::Uncontracted);
    /// ```
    pub fn get_tensor_data(&self) -> Ref<'_, TensorData> {
        self.tensordata.borrow()
    }

    /// Setter for tensor data.
    ///
    /// # Examples
    ///
    /// ```
    /// # use tensorcontraction::tensornetwork::tensor::Tensor;
    /// # use tensorcontraction::tensornetwork::tensordata::TensorData;
    /// # use tensorcontraction::gates::PAULIX;
    /// let mut tensor = Tensor::new(vec![0,1]);
    /// let tensordata = TensorData::Gate(("X", vec![]));
    /// tensor.set_tensor_data(tensordata);
    /// assert_eq!(*tensor.get_tensor_data(), PAULIX);
    /// ```
    pub fn set_tensor_data(&mut self, tensordata: TensorData) {
        assert_eq!(
            self.get_tensors().len(),
            0,
            "Cannot add data to Tensor object with multiple child Tensors"
        );
        let mut td = self.tensordata.borrow_mut();
        *td = tensordata;
    }

    /// Returns Tensor with legs in `self` that are not in `other`.
    ///
    /// # Arguments
    ///
    /// * `other` - Tensor with legs to remove
    ///
    /// # Examples
    /// ```
    /// use tensorcontraction::tensornetwork::tensor::Tensor;
    /// use std::collections::HashMap;
    /// let tensor1 = Tensor::new(vec![1,2,3]);
    /// let tensor2 = Tensor::new(vec![4,2,5]);
    /// let diff_tensor = &tensor1 - &tensor2;
    /// assert_eq!(diff_tensor, Tensor::new(vec![1,3]));
    /// ```
    pub fn difference(&self, other: &Tensor) -> Tensor {
        let mut new_legs = Vec::new();
        for &i in self.legs_iter() {
            if !other.contains_leg(i) {
                new_legs.push(i);
            }
        }
        Tensor::new(new_legs)
    }

    /// Returns Tensor with union of legs in both `self` and `other`.
    ///
    /// # Arguments
    ///
    /// * `other` - Tensor with legs to join
    ///
    /// # Examples
    /// ```
    /// use tensorcontraction::tensornetwork::tensor::Tensor;
    /// use std::collections::HashMap;
    /// let tensor1 = Tensor::new(vec![1,2,3]);
    /// let tensor2 = Tensor::new(vec![4,2,5]);
    /// let union_tensor = &tensor1 | &tensor2;
    /// assert_eq!(union_tensor, Tensor::new(vec![1,2,3,4,5]));
    /// ```
    pub fn union(&self, other: &Tensor) -> Tensor {
        let mut new_tn = Tensor::new(self.get_legs().union(other.get_legs().clone()));
        new_tn.insert_bond_dims(&self.get_bond_dims());
        new_tn
    }

    /// Returns Tensor with intersection of legs in `self` and `other`.
    ///
    /// # Arguments
    ///
    /// * `other` - Tensor with legs to intersect
    ///
    /// # Examples
    /// ```
    /// use tensorcontraction::tensornetwork::tensor::Tensor;
    /// use std::collections::HashMap;
    /// let tensor1 = Tensor::new(vec![1,2,3]);
    /// let tensor2 = Tensor::new(vec![4,2,5]);
    /// let intersection_tensor = &tensor1 & &tensor2;
    /// assert_eq!(intersection_tensor, Tensor::new(vec![2]));
    /// ```
    pub fn intersection(&self, other: &Tensor) -> Tensor {
        let mut new_tn = Tensor::new(self.get_legs().intersect(other.get_legs().clone()));
        new_tn.insert_bond_dims(&self.get_bond_dims());
        new_tn
    }

    /// Returns Tensor with symmetrical difference of legs in `self` and `other`.
    ///
    /// # Arguments
    ///
    /// * `other` - Tensor with legs to intersect
    ///
    /// # Examples
    /// ```
    /// use tensorcontraction::tensornetwork::tensor::Tensor;
    /// use std::collections::HashMap;
    /// let tensor1 = Tensor::new(vec![1,2,3]);
    /// let tensor2 = Tensor::new(vec![4,2,5]);
    /// let sym_dif_tensor = &tensor1 ^ &tensor2;
    /// assert_eq!(sym_dif_tensor, Tensor::new(vec![1,3,4,5]));
    /// ```
    pub fn symmetric_difference(&self, other: &Tensor) -> Tensor {
        let mut new_legs = Vec::new();
        for &i in self.legs_iter() {
            if !other.contains_leg(i) {
                new_legs.push(i);
            }
        }
        for &i in other.get_legs().iter() {
            if !self.contains_leg(i) {
                new_legs.push(i);
            }
        }
        let mut new_tn = Tensor::new(new_legs);
        new_tn.insert_bond_dims(&self.get_bond_dims());
        new_tn
    }

    /// Get output after tensor contraction
    pub fn get_external_edges(&self) -> Vec<usize> {
        if !self.get_legs().is_empty() {
            return self.get_legs().clone();
        }

        let mut ext_edges = Tensor::new(Vec::new());
        for tensor in self.tensors.iter() {
            let tensor_legs = Tensor::new(tensor.get_external_edges());
            let tensor_union = &ext_edges | &tensor_legs;
            let counter = tensor_union.get_legs().iter().counts();
            ext_edges = &ext_edges ^ &tensor_legs;
            for leg in tensor_union.get_legs().iter() {
                // Check if hyperedges are being contracted, if so, only append once to output tensor
                let mut i = 1;
                while self.edges.contains_key(leg) && self.edges[leg].len() > (counter[leg] + i) {
                    i += 1;
                    ext_edges.legs.push(*leg);
                }
            }
        }
        ext_edges.get_legs().clone()
    }
}

/// Implementation of indexing for Tensor.
impl Index<usize> for Tensor {
    type Output = usize;

    fn index(&self, index: usize) -> &Self::Output {
        &self.legs[index]
    }
}

impl BitOr for &Tensor {
    type Output = Tensor;
    fn bitor(self, rhs: &Tensor) -> Tensor {
        self.union(rhs)
    }
}

impl BitAnd for &Tensor {
    type Output = Tensor;
    fn bitand(self, rhs: &Tensor) -> Tensor {
        self.intersection(rhs)
    }
}

impl BitXor for &Tensor {
    type Output = Tensor;
    fn bitxor(self, rhs: &Tensor) -> Tensor {
        self.symmetric_difference(rhs)
    }
}

impl Sub for &Tensor {
    type Output = Tensor;
    fn sub(self, rhs: &Tensor) -> Tensor {
        self.difference(rhs)
    }
}

#[cfg(test)]
mod tests {
    use num_complex::Complex64;
    use std::collections::HashMap;

    use crate::{tensornetwork::tensordata::TensorData, types::Vertex};

    use super::Tensor;

    #[test]
    fn test_empty_tensor() {
        let tensor = Tensor::default();
        assert!(tensor.tensors.is_empty());
        assert!(tensor.get_bond_dims().is_empty());
    }

    #[test]
    fn test_new() {
        let tensor = Tensor::new(vec![2, 4, 5]);
        assert_eq!(tensor.get_legs(), &vec![2, 4, 5]);
        assert_eq!(tensor.dims(), 3);
        assert_eq!(*tensor.get_tensor_data(), TensorData::Uncontracted);
    }

    #[test]
    fn test_push_tensor() {
        let reference_bond_dims_1 = HashMap::<usize, u64>::from([(2, 17), (3, 1), (4, 11)]);
        let reference_bond_dims_2 =
            HashMap::<usize, u64>::from([(2, 17), (3, 1), (4, 11), (8, 3), (9, 20)]);
        let reference_bond_dims_3 = HashMap::<usize, u64>::from([
            (2, 17),
            (3, 1),
            (4, 11),
            (8, 3),
            (9, 20),
            (7, 7),
            (10, 14),
        ]);

        let mut ref_tensor_1 = Tensor::new(vec![4, 3, 2]);
        ref_tensor_1.insert_bond_dims(&reference_bond_dims_1);
        ref_tensor_1.set_tensor_data(TensorData::new_from_data(
            ref_tensor_1.shape(),
            vec![Complex64::new(5.0, 3.0); 187],
            None,
        ));

        let mut ref_tensor_2 = Tensor::new(vec![8, 4, 9]);
        ref_tensor_2.insert_bond_dims(&reference_bond_dims_2);

        let mut ref_tensor_3 = Tensor::new(vec![7, 10, 2]);
        ref_tensor_3.insert_bond_dims(&reference_bond_dims_3);

        let mut tensor = ref_tensor_1.clone();

        let tensor_2 = Tensor::new(vec![8, 4, 9]);
        let bond_dims_2 = HashMap::from([(8, 3), (9, 20)]);
        tensor.push_tensor(tensor_2, Some(&bond_dims_2), None);

        assert_eq!(*tensor.get_tensor_data(), TensorData::Uncontracted);
        for (key, value) in tensor.get_bond_dims().iter() {
            assert_eq!(reference_bond_dims_2[key], *value);
        }
        assert_eq!(
            tensor.get_tensors(),
            &vec![ref_tensor_1.clone(), ref_tensor_2.clone()]
        );
        assert_eq!(tensor.get_legs(), &Vec::<usize>::new());

        let tensor_3 = Tensor::new(vec![7, 10, 2]);
        let bond_dims_3 = HashMap::from([(7, 7), (10, 14)]);

        tensor.push_tensor(tensor_3, Some(&bond_dims_3), None);
        for (key, value) in tensor.get_bond_dims().iter() {
            assert_eq!(reference_bond_dims_3[key], *value);
        }
        ref_tensor_2.insert_bond_dims(&reference_bond_dims_3);
        assert_eq!(
            tensor.get_tensors(),
            &vec![ref_tensor_1, ref_tensor_2, ref_tensor_3]
        );
        assert_eq!(
            tensor.get_edges(),
            &HashMap::from([
                (2, vec![Vertex::Closed(0), Vertex::Closed(2)]),
                (3, vec![Vertex::Closed(0), Vertex::Open]),
                (4, vec![Vertex::Closed(0), Vertex::Closed(1)]),
                (7, vec![Vertex::Closed(2), Vertex::Open]),
                (8, vec![Vertex::Closed(1), Vertex::Open]),
                (9, vec![Vertex::Closed(1), Vertex::Open]),
                (10, vec![Vertex::Closed(2), Vertex::Open]),
            ])
        )
    }

    #[test]
    fn test_push_tensors() {
        let reference_bond_dims_1 = HashMap::<usize, u64>::from([(2, 17), (3, 1), (4, 11)]);
        let reference_bond_dims_3 = HashMap::<usize, u64>::from([
            (2, 17),
            (3, 1),
            (4, 11),
            (8, 3),
            (9, 20),
            (7, 7),
            (10, 14),
        ]);

        let mut ref_tensor_1 = Tensor::new(vec![4, 3, 2]);
        ref_tensor_1.insert_bond_dims(&reference_bond_dims_1);
        ref_tensor_1.set_tensor_data(TensorData::new_from_data(
            ref_tensor_1.shape(),
            vec![Complex64::new(5.0, 3.0); 187],
            None,
        ));

        let mut ref_tensor_2 = Tensor::new(vec![8, 4, 9]);
        ref_tensor_2.insert_bond_dims(&reference_bond_dims_3);

        let mut ref_tensor_3 = Tensor::new(vec![7, 10, 2]);
        ref_tensor_3.insert_bond_dims(&reference_bond_dims_3);

        let mut tensor = ref_tensor_1.clone();

        let tensor_2 = Tensor::new(vec![8, 4, 9]);
        let tensor_3 = Tensor::new(vec![7, 10, 2]);
        tensor.push_tensors(vec![tensor_2, tensor_3], Some(&reference_bond_dims_3), None);

        assert_eq!(*tensor.get_tensor_data(), TensorData::Uncontracted);
        assert_eq!(
            tensor.get_tensors(),
            &vec![ref_tensor_1, ref_tensor_2, ref_tensor_3]
        );
        assert_eq!(*tensor.get_bond_dims(), reference_bond_dims_3);
        assert_eq!(
            tensor.get_edges(),
            &HashMap::from([
                (2, vec![Vertex::Closed(0), Vertex::Closed(2)]),
                (3, vec![Vertex::Closed(0), Vertex::Open]),
                (4, vec![Vertex::Closed(0), Vertex::Closed(1)]),
                (7, vec![Vertex::Closed(2), Vertex::Open]),
                (8, vec![Vertex::Closed(1), Vertex::Open]),
                (9, vec![Vertex::Closed(1), Vertex::Open]),
                (10, vec![Vertex::Closed(2), Vertex::Open]),
            ])
        )
    }
}
