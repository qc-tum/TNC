#[allow(unused_imports)]
use array_tool::vec::{Intersect, Union};
use std::collections::HashMap;
use std::fmt;
use std::fmt::Write;
use std::ops::{Index, IndexMut};

pub mod contraction;
pub mod tensor;
pub mod tacotensor;

use tensor::Tensor;

/// Helper function that returns the largest edge id.
pub trait MaximumLeg {
    fn max_leg(&self) -> i32;
}

#[derive(Debug, Clone, PartialEq)]
/// Abstract representation of a tensor network. Stores a vector of [`Tensor`] objects connected
/// by edges. The edges are stored in a HashMap and edge dimensions are stored in `bond_dim`.
pub struct TensorNetwork {
    /// Vector of Tensor objects in tensor network
    tensors: Vec<Tensor>,
    /// Returns bond dimension of edge based on edge id.
    bond_dims: HashMap<i32, u64>,
    /// Hashmap for easy lookup of edge connectivity based on edge id.
    edges: HashMap<i32, (Option<i32>, Option<i32>)>,
}

/// Helper function that returns the largest edge id of all Tensors in a TensorNetwork.
impl MaximumLeg for Vec<Tensor> {
    fn max_leg(&self) -> i32 {
        let mut m = self[0].iter().max();
        for tensor in self.iter() {
            let n = tensor.iter().max();
            if n > m {
                m = n;
            }
        }
        *m.unwrap_or_else(|| &0)
    }
}

impl Index<usize> for TensorNetwork {
    type Output = Tensor;

    fn index(&self, index: usize) -> &Self::Output {
        &self.tensors[index]
    }
}

impl IndexMut<usize> for TensorNetwork {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.tensors[index]
    }
}



impl TensorNetwork {
    /// Creates an empty TensorNetwork
    ///
    /// # Examples
    ///
    /// ```
    /// # use tensorcontraction::tensornetwork::TensorNetwork;
    /// let tn = TensorNetwork::empty_tensor_network();
    /// ```
    pub fn empty_tensor_network() -> Self {
        Self {
            tensors: Vec::<Tensor>::new(),
            bond_dims: HashMap::new(),
            edges: HashMap::new(),
        }
    }

    /// Getter for edge HashMap.
    ///
    /// # Examples
    ///
    /// ```
    /// # use tensorcontraction::tensornetwork::TensorNetwork;
    /// let tn = TensorNetwork::empty_tensor_network();
    /// let edges = tn.get_edges();
    /// assert_eq!(edges.is_empty(), true);
    /// ```
    pub fn get_edges(&self) -> &HashMap<i32, (Option<i32>, Option<i32>)> {
        &self.edges
    }

    /// Getter for list of Tensor objects.
    ///
    /// # Examples
    ///
    /// ```
    /// # use tensorcontraction::tensornetwork::TensorNetwork;
    /// let tn = TensorNetwork::empty_tensor_network();
    /// let tensors = tn.get_tensors();
    /// assert_eq!(tensors.is_empty(), true);
    /// ```
    pub fn get_tensors(&self) -> &Vec<Tensor> {
        &self.tensors
    }

    /// Getter for bond dimensions.
    ///
    /// # Examples
    ///
    /// ```
    /// # use tensorcontraction::tensornetwork::TensorNetwork;
    /// let tn = TensorNetwork::empty_tensor_network();
    /// let tensors = tn.get_tensors();
    /// assert_eq!(tensors.is_empty(), true);
    /// ```
    pub fn get_bond_dims(&self) -> &HashMap<i32, u64> {
        &self.bond_dims
    }

    /// Returns true if tensor network is empty
    ///
    /// # Examples
    ///
    /// ```
    /// # use tensorcontraction::tensornetwork::TensorNetwork;
    /// let tn = TensorNetwork::empty_tensor_network();
    /// assert_eq!(tn.is_empty(), true);
    /// ```
    pub fn is_empty(&self) -> bool{
        self.tensors.is_empty()
    }

    /// Constructs a TensorNetwork object based on an input Vector of Tensors and a list of bond
    /// dimensions.  Edge ids in the list of Tensors are assumed to be sequential starting from 0.
    /// Each edge id must have an accompanying bond dimension.
    /// Thus, it is assumed that `bond_dim.len()` is g.e. to `tensor.max_leg()`.
    /// # Arguments
    ///
    /// * `tensors` - A Vector of Tensor objects
    /// * `bond_dims` - A Vector of u32 bond dimensions corresponding to edge ids in `tensors`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use tensorcontraction::tensornetwork::TensorNetwork;
    /// # use tensorcontraction::tensornetwork::tensor::Tensor;
    /// let tn = TensorNetwork::empty_tensor_network();
    /// let tensors = tn.get_tensors();
    /// assert_eq!(tensors.is_empty(), true);
    /// ```
    /// # Panics
    ///
    /// Panics when a bond dimension is not defined.
    ///
    /// ```should_panic
    /// # use tensorcontraction::tensornetwork::TensorNetwork;
    /// # use tensorcontraction::tensornetwork::tensor::Tensor;
    /// let v1 = Tensor::new(Vec::from([2,8,7]));
    /// let bond_dims = vec![17, 19];
    /// let tn = TensorNetwork::from_vector(vec![v1], bond_dims);
    /// ```
    pub fn from_vector(tensors: Vec<Tensor>, bond_dims: Vec<u64>) -> Self {
        assert!(tensors.max_leg() < bond_dims.len() as i32);
        let mut edges: HashMap<i32, (Option<i32>, Option<i32>)> = HashMap::new();
        for index in 0usize..tensors.len() {
            for leg in tensors[index].get_legs() {
                edges
                    .entry(*leg)
                    .and_modify(|edge| edge.1 = Some(index as i32))
                    .or_insert((Some(index as i32), None));
            }
        }
        Self {
            tensors,
            bond_dims: (0i32..).zip(bond_dims).collect(),
            edges,
        }
    }

    /// Constructs a TensorNetwork object based on an input Vector of Tensors and a HashMap mapping edge ids to bond
    /// dimensions.  All edge ids in the list of Tensors must have an accompanying bond dimension.
    ///
    /// # Arguments
    ///
    /// * `tensors` - A Vector of Tensor objects
    /// * `bond_dims` - A HashMap taking using edge ids as keys and returning the corresponding bond dimension.
    ///
    /// # Examples
    ///
    /// ```
    /// # use tensorcontraction::tensornetwork::TensorNetwork;
    /// # use tensorcontraction::tensornetwork::tensor::Tensor;
    /// # use std::collections::HashMap;
    /// let v1 = Tensor::new(vec![2,1,0]);
    /// let v2 = Tensor::new(vec![2,3,4]);
    /// let bond_dims = HashMap::from([
    /// (0, 17), (1, 19), (2, 8), (3, 12), (4, 12)
    /// ]);
    /// let mut tn = TensorNetwork::new(vec![v1,v2], bond_dims);
    /// assert_eq!(tn.get_bond_dims()[&0], 17);
    /// assert_eq!(tn.get_bond_dims()[&1], 19);
    /// assert_eq!(tn.get_tensors().len(), 2);
    /// ```
    /// # Panics
    ///
    /// Panics when a bond dimension is not defined.
    ///
    /// ```should_panic
    /// # use tensorcontraction::tensornetwork::TensorNetwork;
    /// # use tensorcontraction::tensornetwork::tensor::Tensor;
    /// # use std::collections::HashMap;
    /// let v1 = Tensor::new(vec![2,1,0]);
    /// let v2 = Tensor::new(vec![2,3,4]);
    /// let bond_dims = HashMap::from([
    /// (0, 17), (1, 19), (3, 12), (4, 12) // edge id `2` does not have bond dimension defined
    /// ]);
    /// let tn = TensorNetwork::new(vec![v1, v2], bond_dims);
    /// ```    
    pub fn new(tensors: Vec<Tensor>, bond_dims: HashMap<i32, u64>) -> Self {
        let mut edges: HashMap<i32, (Option<i32>, Option<i32>)> = HashMap::new();
        for index in 0usize..tensors.len() {
            for leg in tensors[index].get_legs() {
                if !bond_dims.contains_key(&leg) {
                    panic!("Leg {} bond dimension is not defined", leg);
                }
                edges
                    .entry(*leg)
                    .and_modify(|edge| edge.1 = Some(index as i32))
                    .or_insert((Some(index as i32), None));
            }
        }
        Self {
            tensors,
            bond_dims,
            edges,
        }
    }

    /// Private function that updates `TensorNetwork::edges`. Used to modify edge
    /// connections after contraction or when new Tensor objects are appended
    /// Does not perform checks to ensure that each new edge id has a corresponding
    /// bond_dim entry.
    ///
    /// # Arguments
    ///
    /// * `tensors` - A Vector of Tensor objects
    ///
    /// # Panics
    ///
    /// Panics when an edge id appears in more than two Tensor objects.
    fn update_edges(&mut self, tensors: &Vec<Tensor>) {
        // Always push tensor after updating edges
        let start = self.tensors.len();
        for index in start..(tensors.len() + start) {
            for leg in tensors[index].get_legs() {
                self.edges
                    .entry(*leg)
                    .and_modify(|edge|
                        if edge.1.is_some(){
                            panic!(
                                "edge {leg} connects Tensor {t1} and {t2}. attempting to connect to third Tensor {t3}", 
                                leg=leg, t1=edge.0.unwrap(), t2=edge.1.unwrap(), t3=index);
                        } else{
                        edge.1 = Some(index as i32);
                    })
                    .or_insert((Some(index as i32), None));
            }
        }
    }

    /// Private function that updates `TensorNetwork::edges`. Used to modify edge
    /// connections after contraction or when a new Tensor object is appended
    /// Does not perform checks to ensure that each new edge id has a corresponding
    /// bond_dim entry.
    ///
    /// # Arguments
    ///
    /// * `tensors` - A Vector of Tensor objects
    ///
    /// # Panics
    ///
    /// Panics when an edge id appears in more than two Tensor objects.
    fn update_edge(&mut self, tensor: &Tensor) {
        // Always push tensor after updating edges
        let index = self.tensors.len();
        for leg in tensor.get_legs() {
            self.edges
                .entry(*leg)
                .and_modify(|edge|
                    if edge.1.is_some(){
                        panic!(
                            "edge {leg} connects Tensor {t1} and {t2}. attempting to connect to third Tensor {t3}", 
                            leg=leg, t1=edge.0.unwrap(), t2=edge.1.unwrap(), t3=index);
                    } else{
                    edge.1 = Some(index as i32);
                })
                .or_insert((Some(index as i32), None));
        }
    }

    /// Appends a new Tensor object to TensorNetwork object. Optionally, accepts a HashMap of bond dimensions
    /// if edge ids in new Tensor are not defined in `Tensor::bond_dims`. This function only updates edge ids in
    /// new Tensor object and ignores any other edge ids in HashMap.
    ///
    /// # Arguments
    ///
    /// * `tensor` - A Tensor object
    /// * `bond_dims` - - A HashMap taking using edge ids as keys and returning the corresponding bond dimension.
    ///
    /// # Examples
    ///
    /// ```
    /// # use tensorcontraction::tensornetwork::TensorNetwork;
    /// # use tensorcontraction::tensornetwork::tensor::Tensor;
    /// # use std::collections::HashMap;
    /// let v1 = Tensor::new(vec![2,1,0]);
    /// let v2 = Tensor::new(vec![2,3,4]);
    /// let v3 = Tensor::new(vec![4,5,6]);
    /// let bond_dims = HashMap::from([(0, 17), (1, 19), (2, 8), (3, 12), (4, 12)]);
    /// let mut tn = TensorNetwork::new(vec![v1,v2], bond_dims);
    /// let bond_dims_new = HashMap::from([(5, 17), (6, 19)]);
    /// tn.push_tensor(v3, Some(bond_dims_new));
    /// assert_eq!(tn.get_bond_dims()[&4], 12);
    /// assert_eq!(tn.get_bond_dims()[&5], 17);
    /// assert_eq!(tn.get_bond_dims()[&6], 19);
    /// assert_eq!(tn.get_tensors().len(), 3);
    /// ```
    /// # Panics
    ///
    /// Panics when a bond dimension is not defined.
    ///
    /// ```should_panic
    /// # use tensorcontraction::tensornetwork::TensorNetwork;
    /// # use tensorcontraction::tensornetwork::tensor::Tensor;
    /// # use std::collections::HashMap;
    /// # let v1 = Tensor::new(vec![2,1,0]);
    /// # let v2 = Tensor::new(vec![2,3,4]);
    /// # let v3 = Tensor::new(vec![4,5,6]);
    /// # let bond_dims = HashMap::from([(0, 17), (1, 19), (2, 8), (3, 12), (4, 12)]);
    /// let mut tn = TensorNetwork::new(vec![v1,v2], bond_dims);
    /// let bond_dims_new = HashMap::from([(5, 17)]);
    /// tn.push_tensor(v3, Some(bond_dims_new));
    /// ```
    ///
    /// Panics when an existing bond dimension is redefined
    ///
    /// ```should_panic
    /// # use tensorcontraction::tensornetwork::TensorNetwork;
    /// # use tensorcontraction::tensornetwork::tensor::Tensor;
    /// # use std::collections::HashMap;
    /// # let v1 = Tensor::new(vec![2,1,0]);
    /// # let v2 = Tensor::new(vec![2,3,4]);
    /// # let v3 = Tensor::new(vec![4,5,6]);
    /// # let bond_dims = HashMap::from([(0, 17), (1, 19), (2, 8), (3, 12), (4, 12)]);
    /// let mut tn = TensorNetwork::new(vec![v1,v2], bond_dims);
    /// let bond_dims_new = HashMap::from([(4, 3), (5, 17), (6,19)]);
    /// tn.push_tensor(v3, Some(bond_dims_new));
    /// ```
    pub fn push_tensor(&mut self, tensor: Tensor, bond_dims: Option<HashMap<i32, u64>>) {
        if bond_dims.is_none() {
            for leg in tensor.get_legs() {
                if !self.bond_dims.contains_key(leg) {
                    panic!(
                        "Input {:?} contains leg {}, with unknown bond dimension.",
                        tensor, leg
                    );
                }
            }
        } else {
            let bond_dims = bond_dims.unwrap();
            for leg in tensor.get_legs().iter() {
                if !self.bond_dims.contains_key(leg) {
                    if !bond_dims.contains_key(leg) {
                        panic!("Edge id {} bond dimension is not defined.", leg);
                    }
                    self.bond_dims.entry(*leg).or_insert(bond_dims[&leg]);
                } else if bond_dims.contains_key(leg)
                    && *self.bond_dims.get(leg).unwrap() != bond_dims[leg]
                {
                    panic!(
                        "Attempt to update bond {} with value: {}, previous value: {}",
                        leg,
                        &bond_dims[&leg],
                        self.bond_dims.get(leg).unwrap()
                    )
                }
            }
        }
        //ensure that new tensor is connected
        self.update_edge(&tensor);
        self.tensors.push(tensor);
    }

    /// Updates TensorNetwork object by contracting two tensors, replacing the first contracted tensor with the
    /// resulting tensor. `tn.edges` is then updated replacing all connections to the second tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor_a_loc` - Index of first Tensor to be contracted
    /// * `tensor_b_loc` - Index of second Tensor to be contracted
    fn _contraction(&mut self, tensor_a_loc: usize, tensor_b_loc: usize) -> (Vec<i32>, Vec<i32>) {
        let tensor_a_legs = self.tensors[tensor_a_loc].get_legs();
        let tensor_b_legs = self.tensors[tensor_b_loc].get_legs();

        let tensor_union = tensor_b_legs.union(tensor_a_legs.to_vec());
        let tensor_intersect = tensor_b_legs.intersect(tensor_a_legs.to_vec());

        let mut tensor_difference: Vec<i32> = Vec::new();
        for leg in tensor_union.iter() {
            if !tensor_intersect.iter().any(|&i| i == *leg) {
                tensor_difference.push(*leg);
            }
        }

        for leg in tensor_b_legs.iter() {
            if self.edges[&leg].0.unwrap_or_default() == tensor_b_loc as i32 {
                self.edges
                    .entry(*leg)
                    .and_modify(|e| e.0 = Some(tensor_a_loc as i32));
            } else {
                self.edges
                    .entry(*leg)
                    .and_modify(|e| e.1 = Some(tensor_a_loc as i32));
            }
        }
        self.tensors[tensor_a_loc] = Tensor::new(tensor_difference.clone());

        (tensor_intersect, tensor_difference)
    }


    /// Constructs Graphviz code showing the tensor network as a graph. The tensor numbering corresponds to their
    /// tensor index (i.e., their position in the tensors vector). The edges are annotated with the bond dims,
    /// as well as the edge id in smaller font.
    pub fn to_graphviz(&self) -> String {
        let mut out = String::new();
        let mut invis_counter = 0u32;
        out.push_str("graph tn {\n");

        for (i, tensor) in self.tensors.iter().enumerate() {
            for leg in tensor.get_legs() {
                let connection = self.edges[leg];

                if let (Some(idx1), Some(_)) = connection {
                    if idx1 == i as i32 {
                        // prevent each edge being added twice, by only considering
                        // edges where this tensor is in the first place
                        continue;
                    }
                }

                // Get tensor1 name (or create an invisible node if None)
                let t1 = if let Some(idx) = connection.0 {
                    format!("t{}", idx)
                } else {
                    let name = format!("i{}", invis_counter);
                    writeln!(out, "\t{} [style=\"invis\"];", name).unwrap();
                    invis_counter += 1;
                    name
                };

                // Get tensor2 name (or create an invisible node if None)
                let t2 = if let Some(idx) = connection.1 {
                    format!("t{}", idx)
                } else {
                    let name = format!("i{}", invis_counter);
                    writeln!(out, "\t{} [style=\"invis\"];", name).unwrap();
                    invis_counter += 1;
                    name
                };

                // Write edge between tensors
                writeln!(out, "\t{} -- {} [label=\"{}\", taillabel=\"{}\", headlabel=\"{}\", labelfontsize=\"8pt\"];", t1, t2, self.bond_dims[leg], leg, leg).unwrap();
            }
        }

        out.push_str("}");
        out
    }
}

/// Implementation of printing for TensorNetwork. Simply prints the Tensor objects in TensorNetwork
impl fmt::Display for TensorNetwork {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for (key, value) in &self.bond_dims {
            println!("{}: {}", key, value);
        }
        write!(f, "Tensor: {:?}", self.tensors)
    }
}

impl Default for TensorNetwork {
    fn default() -> Self {
        Self {
            tensors: Vec::<Tensor>::new(),
            bond_dims: HashMap::new(),
            edges: HashMap::new(),
        }
    }
}


#[cfg(test)]
mod tests {
    use crate::tensornetwork::tensor::Tensor;
    use crate::tensornetwork::MaximumLeg;
    use crate::tensornetwork::TensorNetwork;
    use std::collections::HashMap;

    fn setup() -> TensorNetwork {
        TensorNetwork::from_vector(
            vec![Tensor::new(vec![4, 3, 2]), Tensor::new(vec![0, 1, 3, 2])],
            vec![17, 18, 19, 12, 22],
        )
    }
    #[test]
    fn test_empty_tensor_network() {
        let t = TensorNetwork::default();
        assert!(t.tensors.is_empty());
        assert!(t.bond_dims.is_empty());
    }
    #[test]
    fn test_new() {
        let tensors = vec![Tensor::new(vec![4, 3, 2]), Tensor::new(vec![0, 1, 3, 2])];
        let mut edge_sol = HashMap::<i32, (Option<i32>, Option<i32>)>::new();
        edge_sol.entry(0).or_insert((Some(1), None));
        edge_sol.entry(1).or_insert((Some(1), None));
        edge_sol.entry(2).or_insert((Some(0), Some(1)));
        edge_sol.entry(3).or_insert((Some(0), Some(1)));
        edge_sol.entry(4).or_insert((Some(0), None));
        let bond_dims = vec![17, 18, 19, 12, 22];
        let t = TensorNetwork::from_vector(tensors, bond_dims.clone());
        for leg in 0..t.tensors.max_leg() as usize {
            assert_eq!(t.bond_dims[&(leg as i32)], bond_dims[leg]);
        }
        for edge_key in 0i32..4 {
            assert_eq!(edge_sol[&edge_key], t.get_edges()[&edge_key]);
        }
    }

    #[test]
    fn test_push_tensor_good() {
        //TODO: Add test to check for edge update
        let mut t = setup();
        let good_tensor = Tensor::new(vec![0, 1, 4]);
        t.push_tensor(good_tensor, None);
        let mut edge_sol = HashMap::<i32, (Option<i32>, Option<i32>)>::new();
        edge_sol.entry(0).or_insert((Some(1), Some(2)));
        edge_sol.entry(1).or_insert((Some(1), Some(2)));
        edge_sol.entry(2).or_insert((Some(0), Some(1)));
        edge_sol.entry(3).or_insert((Some(0), Some(1)));
        edge_sol.entry(4).or_insert((Some(0), Some(2)));
        let bond_dims = vec![17, 18, 19, 12, 22];

        for leg in 0..t.tensors.max_leg() as usize {
            assert_eq!(t.bond_dims[&(leg as i32)], bond_dims[leg]);
        }

        for edge_key in 0i32..4 {
            assert_eq!(edge_sol[&edge_key], t.get_edges()[&edge_key]);
        }
    }

    #[test]
    fn test_push_tensor_good_newlegs() {
        let mut t = setup();
        let good_tensor = Tensor::new(vec![7, 9, 12]);
        let good_bond_dims = HashMap::from([(7, 55), (9, 5), (12, 6)]);
        println!("{:?}", good_bond_dims);
        t.push_tensor(good_tensor.clone(), Some(good_bond_dims.clone()));
        for legs in good_tensor.get_legs() {
            assert_eq!(good_bond_dims[&legs], t.bond_dims[legs]);
        }
        let mut edge_sol = HashMap::<i32, (Option<i32>, Option<i32>)>::new();
        edge_sol.entry(0).or_insert((Some(1), None));
        edge_sol.entry(1).or_insert((Some(1), None));
        edge_sol.entry(2).or_insert((Some(0), Some(1)));
        edge_sol.entry(3).or_insert((Some(0), Some(1)));
        edge_sol.entry(4).or_insert((Some(0), None));
        edge_sol.entry(7).or_insert((Some(2), None));
        edge_sol.entry(9).or_insert((Some(3), None));
        edge_sol.entry(12).or_insert((Some(4), None));
        let bond_dims = vec![55, 5, 6];
        let mut x = bond_dims.iter();
        for leg in good_tensor.get_legs() {
            assert_eq!(t.bond_dims[leg], *x.next().unwrap());
        }

        for edge_key in 0i32..4 {
            assert_eq!(edge_sol[&edge_key], t.get_edges()[&edge_key]);
        }
    }

    #[test]
    #[should_panic(
        expected = "Input Tensor { legs: [0, 5, 4] } contains leg 5, with unknown bond dimension."
    )]
    fn test_push_tensor_bad() {
        let mut t = setup();
        let bad_tensor = Tensor::new(vec![0, 5, 4]);
        t.push_tensor(bad_tensor, None);
    }

    #[test]
    #[should_panic(expected = "Attempt to update bond 0 with value: 12, previous value: 17")]
    fn test_push_tensor_bad_rewrite() {
        let mut t = setup();
        let bad_tensor = Tensor::new(vec![0, 1, 4]);
        let bad_bond_dims = HashMap::from([(0, 12), (1, 32), (4, 2)]);
        t.push_tensor(bad_tensor, Some(bad_bond_dims));
    }

    #[test]
    fn test_tensor_contraction_good() {
        let mut t = setup();
        let (tensor_intersect, _tensor_difference) = t._contraction(0, 1);
        // contraction should maintain leg order
        let vec_sol = vec![0, 1, 4];
        let tensor_sol = Tensor::new(vec_sol.clone());
        let mut edge_sol = HashMap::<i32, (Option<i32>, Option<i32>)>::new();
        edge_sol.entry(0).or_insert((Some(0), None));
        edge_sol.entry(1).or_insert((Some(0), None));
        edge_sol.entry(2).or_insert((Some(0), Some(0)));
        edge_sol.entry(3).or_insert((Some(0), Some(0)));
        edge_sol.entry(4).or_insert((Some(0), None));

        assert_eq!(t.get_tensors()[0], tensor_sol);
        for edge_key in 0i32..4 {
            assert_eq!(edge_sol[&edge_key], t.get_edges()[&edge_key]);
        }

        assert_eq!(tensor_intersect, vec![3, 2]);
        assert_eq!(_tensor_difference, vec_sol);
    }
}
