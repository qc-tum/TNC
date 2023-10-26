use core::ops::{BitAnd, BitOr, BitXor, Sub};
use std::cell::{Ref, RefCell};
use std::collections::HashMap;
use std::fmt;
use std::hash::Hash;
use std::iter::Iterator;
use std::ops::{Index, IndexMut, RangeBounds};
use std::rc::Rc;

use array_tool::vec::Uniq;
use tetra::{contract, Tensor as DataTensor};

use crate::gates::*;
use crate::io::load_data;
use crate::types::*;

use super::contraction::contract_tensor_network;
use super::tensordata::TensorData;

#[derive(Debug, Eq)]
/// Abstract representation of a tensor.
pub struct Tensor {
    pub(crate) tensors: Vec<Tensor>,
    legs: Vec<EdgeIndex>,
    bond_dims: Rc<RefCell<HashMap<EdgeIndex, u64>>>,
    edges: HashMap<EdgeIndex, Vec<Vertex>>,
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
            if !(other_edges[k].iter().eq(v.iter())) {
                return false;
            }
        }
        true
    }
}

impl Hash for Tensor {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.tensors.hash(state);
        self.legs.hash(state);
    }
}

impl Clone for Tensor {
    fn clone(&self) -> Self {
        Self {
            tensors: self.tensors.clone(),
            legs: self.legs.clone(),
            //Ensure only pointer is cloned
            bond_dims: Rc::clone(&self.bond_dims),
            //Ensure only pointer is cloned
            edges: self.edges.clone(),
            tensordata: self.tensordata.clone(),
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
            bond_dims: Rc::new(RefCell::new(HashMap::new())),
            edges: HashMap::new(),
            tensordata: RefCell::new(TensorData::Empty),
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
    /// v1.set_bond_dims(&bond_dims);
    /// v2.set_bond_dims(&bond_dims);
    /// assert_eq!(*tn.get_tensors(), vec![v1, v2]);
    /// ```
    pub fn get_tensors(&self) -> &Vec<Tensor> {
        &self.tensors
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
    /// tn.set_bond_dims(&bond_dims);
    /// let mut ref_tensor = Tensor::new(vec![0,1]);
    /// ref_tensor.set_bond_dims(&bond_dims);
    /// assert_eq!(*tn.get_tensor(0), ref_tensor);
    /// ```
    pub fn get_tensor(&self, i: usize) -> &Tensor {
        &self.tensors[i]
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
    pub fn get_bond_dims(&self) -> std::cell::Ref<'_, std::collections::HashMap<EdgeIndex, u64>> {
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
    /// tn.set_bond_dim(1, 12);
    /// assert_ne!(*tn.get_bond_dims(), bond_dims);
    /// ```
    pub fn set_bond_dim(&mut self, k: EdgeIndex, v: u64) {
        self.bond_dims
            .borrow_mut()
            .entry(k)
            .and_modify(|e| {
                *e = v;
            })
            .or_insert(v);
    }

    /// Setter for bond dimensions.
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
    /// tn.set_bond_dims(&HashMap::from([(1, 12), (0, 5)]));
    /// assert_eq!(*tn.get_bond_dims(), HashMap::from([(0, 5), (1, 12), (2, 8)]) );
    /// ```
    pub fn set_bond_dims(&mut self, bond_dims: &HashMap<EdgeIndex, u64>) {
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

    ///Setter for edges
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

    pub(crate) fn get_mut_edges(&mut self) -> &mut HashMap<EdgeIndex, Vec<Vertex>> {
        &mut self.edges
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
    /// tensor.set_bond_dims(&bond_dims);
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
    /// tensor.set_bond_dims(&bond_dims);
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
    /// assert_eq!(tensor.is_empty(), true);
    /// ```
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
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
        if self.get_tensors().is_empty() && !self.get_legs().is_empty() {
            let mut new_self = self.clone();
            // Only update legs once contraction is complete to keep track of data permutation
            self.legs = Vec::new();
            // Don't clone large data is needed.
            self._update_tensor(&mut new_self);
            self.tensors.push(new_self);
            self.set_tensor_data(TensorData::Empty);
        }
        if let Some(bond_dims) = bond_dims {
            self._update_bond_dims(bond_dims);
        };
        if let Some(external_hyperedge) = external_hyperedge {
            self._update_external_edges(external_hyperedge);
        };

        self._update_tensor(&mut tensor);
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
        if self.get_tensors().is_empty() && !self.get_legs().is_empty() {
            let mut new_self = self.clone();
            // Only update legs once contraction is complete to keep track of data permutation
            self.legs = Vec::new();
            // Don't clone large data is needed.
            self._update_tensor(&mut new_self);
            self.tensors.push(new_self);
            self.set_tensor_data(TensorData::Empty);
        }
        if let Some(bond_dims) = bond_dims {
            self._update_bond_dims(bond_dims);
        };
        if let Some(external_hyperedge) = external_hyperedge {
            self._update_external_edges(external_hyperedge);
        };
        for tensor in tensors.iter_mut() {
            self._update_tensor(tensor);
            self.tensors.push(tensor.clone());
        }
    }

    fn _update_bond_dims(&mut self, bond_dims: &HashMap<EdgeIndex, u64>) {
        let mut shared_bond_dims = self.bond_dims.borrow_mut();
        for (key, value) in bond_dims.iter() {
            shared_bond_dims.entry(*key).or_insert(*value);
        }
    }

    fn _update_external_edges(&mut self, external_hyperedge: &Vec<usize>) {
        for i in external_hyperedge {
            self.edges
                .entry(*i)
                .and_modify(|edge| edge.push(Vertex::Open));
        }
    }

    /// Updates edges in tensornetwork after adding new tensors. DOes
    fn _update_tensor(&mut self, tensor: &mut Tensor) {
        tensor.bond_dims = Rc::clone(&self.bond_dims);
        let shared_bond_dims = self.bond_dims.borrow();

        // Index is current length as tensor is pushed after.
        let index = self.get_tensors().len();
        for leg in tensor.get_legs() {
            if !shared_bond_dims.contains_key(leg) {
                panic!("Leg {leg} bond dimension is not defined");
            }
            self.edges
                .entry(*leg)
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
    /// assert_eq!(*tensor.get_tensor_data(), TensorData::Empty);
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
    /// # use tensorcontraction::tensornetwork::tensordata::{TensorData, PAULIX};
    /// let mut tensor = Tensor::new(vec![0,1]);
    /// let tensordata = TensorData::Gate("X");
    /// tensor.set_tensor_data(tensordata);
    /// assert_eq!(*tensor.get_tensor_data(), PAULIX);
    /// ```
    pub fn set_tensor_data(&self, tensordata: TensorData) {
        assert!(
            self.get_tensors().len() <= 1,
            "Cannot add data to Tensor object with multiple child Tensors"
        );
        let mut td = self.tensordata.borrow_mut();
        *td = tensordata;
    }

    /// Getter for underlying raw data
    pub(crate) fn get_data(&self) -> DataTensor {
        match self.get_tensor_data().clone() {
            TensorData::File(filename) => load_data(&filename).unwrap(),
            TensorData::Gate(gatename) => load_gate(gatename), // load_gate[gatename.to_lowercase()],
            TensorData::Matrix(rawdata) => rawdata.clone(),
            TensorData::Empty => DataTensor::new(&[]),
        }
    }

    /// Getter for underlying raw data
    pub(crate) fn drain<R>(&mut self, range: R)
    where
        R: RangeBounds<usize>,
    {
        self.tensors.drain(range);
    }

    /// Partitions tensor network using the provided partitioning vector
    /// Only allows single layer of partitioning
    pub fn partition(&mut self, partitioning: &Vec<usize>) {
        assert!(partitioning.len() == self.tensors.len());
        let mut partitions = partitioning.clone();

        partitions.dedup();
        let partition_map: HashMap<&usize, usize> =
            HashMap::from_iter(std::iter::zip(partitions.iter(), 0..self.tensors.len()));
        let mut new_tensors = vec![Tensor::default(); self.tensors.len()];
        for (partition, tensor) in
            std::iter::zip(partitioning.iter().rev(), self.tensors.iter().rev())
        {
            new_tensors[partition_map[partition]].push_tensor(tensor.clone(), None, None);
        }

        self.tensors = new_tensors;
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
        for i in self.get_legs().iter().cloned() {
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
        let mut new_legs = self.legs.clone();
        for i in other.get_legs().iter().cloned() {
            if !self.contains_leg(i) {
                new_legs.push(i);
            }
        }
        Tensor::new(new_legs)
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
        let mut new_legs = Vec::new();
        for i in self.get_legs().iter().cloned() {
            if other.contains_leg(i) {
                new_legs.push(i);
            }
        }
        Tensor::new(new_legs)
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
    /// let sym_dif_tensor = &tensor1 ^ &tensor2;
    /// assert_eq!(sym_dif_tensor, Tensor::new(vec![1,3,4,5]));
    /// ```
    pub fn symmetric_difference(&self, other: &Tensor) -> Tensor {
        let mut new_legs = Vec::new();
        for i in self.get_legs().iter().cloned() {
            if !other.contains_leg(i) {
                new_legs.push(i);
            }
        }
        for i in other.get_legs().iter().cloned() {
            if !self.contains_leg(i) {
                new_legs.push(i);
            }
        }
        Tensor::new(new_legs)
    }

    pub(crate) fn swap_tensor(&mut self, i: usize, j: usize) {
        self.tensors.swap(i, j);
    }

    pub fn get_external_edges(&self) -> Vec<usize> {
        if !self.get_legs().is_empty() {
            return self.get_legs().clone();
        }
        let mut ext_edges = Tensor::new(Vec::<usize>::new());
        for tensor in self.tensors.iter() {
            let tensor_union = &ext_edges | tensor;
            let counter = count_edges(tensor_union.get_legs().iter());
            ext_edges = &ext_edges ^ tensor;
            for leg in tensor_union.get_legs().iter() {
                // Check if hyperedges are being contracted, if so, only append once to output tensor
                let mut i = 0;
                while self.edges[leg].len() > (counter[leg] + i) {
                    i += 1;
                    ext_edges.legs.push(*leg);
                }
            }
        }
        ext_edges.get_legs().clone()
    }

    pub fn contract_tensors(&mut self, tensor_a_loc: usize, tensor_b_loc: usize) {
        let tensor_a = self.clone().get_tensor(tensor_a_loc).clone();
        let tensor_b = self.clone().get_tensor(tensor_b_loc).clone();
        let tensor_a_legs = tensor_a.get_legs();
        let tensor_b_legs = tensor_b.get_legs();
        let tensor_union = &tensor_b | &tensor_a;
        let mut tensor_symmetric_difference = &tensor_b ^ &tensor_a;
        let counter = count_edges(tensor_union.get_legs().iter());

        let edges = self.get_mut_edges();
        for leg in tensor_union.get_legs().unique().iter() {
            // Check if hyperedges are being contracted, if so, only append once to output tensor
            let mut i = 0;
            while edges[leg].len() - 1 > (counter[leg] + i) {
                i += 1;
                tensor_symmetric_difference.legs.push(*leg);
            }
        }
        // Update internal edges HashMap to point tensor b legs to new contracted tensor
        for leg in tensor_b_legs.iter() {
            edges.entry(*leg).and_modify(|e| {
                e.retain(|e| {
                    if let Vertex::Closed(edge) = e {
                        *edge != tensor_a_loc
                    } else {
                        true
                    }
                });
                for edge in &mut e.iter_mut() {
                    if let Vertex::Closed(tensor_loc) = edge {
                        if *tensor_loc == tensor_b_loc {
                            *edge = Vertex::Closed(tensor_a_loc);
                        }
                    }
                }
            });
        }
        let mut new_tensor = Tensor::new(tensor_symmetric_difference.get_legs().clone());
        new_tensor.bond_dims = Rc::clone(&self.bond_dims);
        new_tensor.set_tensor_data(TensorData::Matrix(contract(
            tensor_symmetric_difference
                .get_legs()
                .iter()
                .map(|e| *e as u32)
                .collect::<Vec<u32>>()
                .as_slice(),
            tensor_a_legs
                .iter()
                .map(|e| *e as u32)
                .collect::<Vec<u32>>()
                .as_slice(),
            &self.get_tensor(tensor_a_loc).get_data(),
            tensor_b_legs
                .iter()
                .map(|e| *e as u32)
                .collect::<Vec<u32>>()
                .as_slice(),
            &self.get_tensor(tensor_b_loc).get_data(),
        )));
        self.tensors[tensor_a_loc] = new_tensor;
        // remove old tensor
        self.tensors[tensor_b_loc] = Tensor::new(Vec::new());
        // (tensor_intersect, tensor_difference)
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
            bond_dims: Rc::new(RefCell::new(HashMap::new())),
            edges: HashMap::new(),
            tensordata: RefCell::new(TensorData::Empty),
        }
    }
}

/// Implementation of printing for Tensor. Simply prints the legs as a vector
impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self.legs)
    }
}

/// Implementation of indexing for Tensor.
impl Index<usize> for Tensor {
    type Output = usize;

    fn index(&self, index: usize) -> &Self::Output {
        &self.legs[index]
    }
}

/// Implementation of indexing of mutable Tensor object.
impl IndexMut<usize> for Tensor {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.legs[index]
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

fn count_edges<I>(it: I) -> HashMap<I::Item, usize>
where
    I: IntoIterator,
    I::Item: Eq + core::hash::Hash,
{
    let mut result = HashMap::new();

    for item in it {
        *result.entry(item).or_insert(0) += 1;
    }

    result
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
        assert_eq!(*tensor.get_tensor_data(), TensorData::Empty);
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
        ref_tensor_1.set_bond_dims(&reference_bond_dims_1);
        ref_tensor_1.set_tensor_data(TensorData::new_from_flat(
            ref_tensor_1.shape(),
            vec![Complex64::new(5.0, 3.0); 187],
            None,
        ));

        let mut ref_tensor_2 = Tensor::new(vec![8, 4, 9]);
        ref_tensor_2.set_bond_dims(&reference_bond_dims_2);

        let mut ref_tensor_3 = Tensor::new(vec![7, 10, 2]);
        ref_tensor_3.set_bond_dims(&reference_bond_dims_3);

        let mut tensor = ref_tensor_1.clone();

        let tensor_2 = Tensor::new(vec![8, 4, 9]);
        let bond_dims_2 = HashMap::from([(8, 3), (9, 20)]);
        tensor.push_tensor(tensor_2, Some(&bond_dims_2), None);

        assert_eq!(*tensor.get_tensor_data(), TensorData::Empty);
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
        ref_tensor_2.set_bond_dims(&reference_bond_dims_3);
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
        ref_tensor_1.set_bond_dims(&reference_bond_dims_1);
        ref_tensor_1.set_tensor_data(TensorData::new_from_flat(
            ref_tensor_1.shape(),
            vec![Complex64::new(5.0, 3.0); 187],
            None,
        ));

        let mut ref_tensor_2 = Tensor::new(vec![8, 4, 9]);
        ref_tensor_2.set_bond_dims(&reference_bond_dims_3);

        let mut ref_tensor_3 = Tensor::new(vec![7, 10, 2]);
        ref_tensor_3.set_bond_dims(&reference_bond_dims_3);

        let mut tensor = ref_tensor_1.clone();

        let tensor_2 = Tensor::new(vec![8, 4, 9]);
        let tensor_3 = Tensor::new(vec![7, 10, 2]);
        tensor.push_tensors(vec![tensor_2, tensor_3], Some(&reference_bond_dims_3), None);

        assert_eq!(*tensor.get_tensor_data(), TensorData::Empty);
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
