#[allow(unused_imports)]
use array_tool::vec::{Intersect, Union};
use std::collections::HashMap;
use std::fmt;
use std::ops::{Index, IndexMut};

pub mod contraction;
pub mod tacotensor;
pub mod tensor;

use tensor::Tensor;

/// Helper function that returns the largest edge id.
pub trait MaximumLeg {
    fn max_leg(&self) -> i32;
}

#[derive(Debug, Clone, PartialEq, Default)]
/// Abstract representation of a tensor network. Stores a vector of [`Tensor`] objects connected
/// by edges. The edges are stored in a HashMap and edge dimensions are stored in `bond_dim`.
pub struct TensorNetwork {
    /// Vector of Tensor objects in tensor network
    tensors: Vec<Tensor>,
    /// Returns bond dimension of edge based on edge id.
    bond_dims: HashMap<i32, u64>,
    /// Hashmap for easy lookup of edge connectivity based on edge id.
    edges: HashMap<i32, Vec<Option<i32>>>,
    /// List of external dimensions that remain after contraction.
    ext_edges: Vec<i32>,
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
        *m.unwrap_or(&0)
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
            ext_edges: Vec::new(),
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
    pub fn get_edges(&self) -> &HashMap<i32, Vec<Option<i32>>> {
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
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Constructs a TensorNetwork object based on an input Vector of Tensors and a list of bond
    /// dimensions.  Edge ids in the list of Tensors are assumed to be sequential starting from 0.
    /// Each edge id must have an accompanying bond dimension.
    /// Thus, it is assumed that `bond_dim.len()` is g.e. to `tensor.max_leg()`.
    /// Allows an `ext` argument to specify external edges after contraction. This is only required
    /// when there are external edges that are part of hyperedges.
    ///
    /// # Arguments
    ///
    /// * `tensors` - A Vector of Tensor objects
    /// * `bond_dims` - A Vector of u32 bond dimensions corresponding to edge ids in `tensors`.
    /// * `ext` - An optional Vector of i32 edge IDs, indicates which edges are external edges after full contraction.
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
    /// let tn = TensorNetwork::from_vector(vec![v1], bond_dims, None);
    /// ```
    pub fn from_vector(tensors: Vec<Tensor>, bond_dims: Vec<u64>, ext: Option<&Vec<i32>>) -> Self {
        assert!(tensors.max_leg() < bond_dims.len() as i32);
        let mut edges: HashMap<i32, Vec<Option<i32>>> = HashMap::new();
        for (index, tensor) in tensors.iter().enumerate() {
            for leg in tensor.get_legs() {
                edges
                    .entry(*leg)
                    .and_modify(|edge| edge.push(Some(index as i32)))
                    .or_insert(vec![Some(index as i32)]);
            }
        }
        let mut ext_edges: Vec<i32> = if let Some(ext_edges) = ext {
            for i in ext_edges {
                edges.entry(*i).and_modify(|edge| edge.push(None));
            }
            ext_edges.clone()
        } else {
            Vec::new()
        };
        for (index, edge) in &mut edges{
                if edge.len() == 1 {
                    edge.push(None);
                ext_edges.push(*index);
                }
        }

        Self {
            tensors,
            bond_dims: (0i32..).zip(bond_dims).collect(),
            edges,
            ext_edges,
        }
    }

    // TODO: Add hyperedge example
    /// Constructs a TensorNetwork object based on an input Vector of Tensors and a HashMap mapping edge ids to bond
    /// dimensions.  All edge ids in the list of Tensors must have an accompanying bond dimension.
    /// Allows an `ext` argument to specify external edges after contraction. This is only required
    /// when there are external edges that are part of hyperedges.
    ///
    /// # Arguments
    ///
    /// * `tensors` - A Vector of Tensor objects
    /// * `bond_dims` - A HashMap taking using edge ids as keys and returning the corresponding bond dimension.
    /// * `ext` - An optional Vector of i32 edge IDs, indicates which edges are external edges after full contraction.
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
    /// let mut tn = TensorNetwork::new(vec![v1,v2], bond_dims, None);
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
    /// let tn = TensorNetwork::new(vec![v1, v2], bond_dims, None);
    /// ```    
    pub fn new(tensors: Vec<Tensor>, bond_dims: HashMap<i32, u64>, ext: Option<&Vec<i32>>) -> Self {
        let mut edges: HashMap<i32, Vec<Option<i32>>> = HashMap::new();
        for (index, tensor) in tensors.iter().enumerate() {
            for leg in tensor.get_legs() {
                if !bond_dims.contains_key(leg) {
                    panic!("Leg {leg} bond dimension is not defined");
                }
                edges
                    .entry(*leg)
                    .and_modify(|edge| edge.push(Some(index as i32)))
                    .or_insert(vec![Some(index as i32)]);
            }
        }
        let ext_edges: Vec<i32> = if let Some(ext_edges) = ext {
            for i in ext_edges {
                edges.entry(*i).and_modify(|edge| edge.push(None));
            }
            ext_edges.clone()
        } else {
            let mut ext_edges = Vec::new();
            for i in 0..edges.len() {
                edges.entry(i as i32).and_modify(|edge| {
                    if edge.len() == 1 {
                        edge.push(None);
                        ext_edges.push(i as i32);
                    }
                });
            }
            ext_edges
        };
        Self {
            tensors,
            bond_dims,
            edges,
            ext_edges,
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
    /// * `ext` - An optional Vector of i32 edge IDs, indicates which edges are external edges after full contraction.
    ///
    /// # Panics
    ///
    /// Panics when an edge id appears in more than two Tensor objects.
    fn update_edges(&mut self, tensors: &Vec<Tensor>, ext: Option<&Vec<i32>>) {
        // Always push tensor after updating edges
        let start = self.tensors.len();
        // for (index, tensor) in start..(tensors.len() + start) {
        for (index, tensor) in tensors.iter().enumerate().skip(start).take(tensors.len()) {
            for leg in tensor.get_legs() {
                self.edges
                    .entry(*leg)
                    .and_modify(|edge| {
                        // New tensor contracts on a previous external leg
                        if let Some(pos) = edge.iter().position(|e| e.is_none()) {
                            edge[pos] = Some(index as i32);
                            // Leg is no longer external as it contracts with new tensor
                            if let Some(pos_ext) = self.ext_edges.iter().position(|e| e == leg) {
                                self.ext_edges.remove(pos_ext);
                            }
                        } else {
                            // New Tensor connects with existing leg
                            edge.push(Some(index as i32));
                        }
                    })
                    // Inserts new edge
                    .or_insert(vec![Some(index as i32), None]);
            }
        }
        // Add new external edges to TensorNetwork
        if let Some(ext_edges) = ext {
            for i in ext_edges {
                self.edges.entry(*i).and_modify(|edge| edge.push(None));
                self.ext_edges.push(*i);
            }
        };
    }

    /// Private function that updates `TensorNetwork::edges`. Used to modify edge
    /// connections after contraction or when a new Tensor object is appended
    /// Does not perform checks to ensure that each new edge id has a corresponding
    /// bond_dim entry.
    ///
    /// # Arguments
    ///
    /// * `tensor` - A Tensor object
    /// * `ext` - An optional Vector of i32 edge IDs, indicates which edges are external edges after full contraction.
    ///
    fn update_edge(&mut self, tensor: &Tensor, ext: Option<&Vec<i32>>) {
        let index = self.tensors.len();
        for leg in tensor.get_legs() {
            self.edges
                .entry(*leg)
                .and_modify(|edge| {
                    // New tensor contracts on a previous external leg
                    if let Some(pos) = edge.iter().position(|e| e.is_none()) {
                        edge[pos] = Some(index as i32);
                        // Leg is no longer external as it contracts with new tensor
                        if let Some(pos_ext) = self.ext_edges.iter().position(|e| e == leg) {
                            self.ext_edges.remove(pos_ext);
                        }
                    } else {
                        // New Tensor connects with existing leg
                        edge.push(Some(index as i32));
                    }
                })
                // Inserts new edge
                .or_insert(vec![Some(index as i32), None]);
        }

        // Add new external edges to TensorNetwork
        if let Some(ext_edges) = ext {
            for i in ext_edges {
                self.edges.entry(*i).and_modify(|edge| edge.push(None));
                self.ext_edges.push(*i);
            }
        };
    }

    /// Appends a new Tensor object to TensorNetwork object. Optionally, accepts a HashMap of bond dimensions
    /// if edge ids in new Tensor are not defined in `TensorNetwork::bond_dims`. This updates edge ids in `TensorNetwork::edges`
    /// if they are in the new Tensor object via a call to [update_edge].
    ///
    /// # Arguments
    ///
    /// * `tensor` - A Tensor object
    /// * `bond_dims` - A HashMap taking using edge ids as keys and returning the corresponding bond dimension.
    /// * `ext` -
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
    /// let mut tn = TensorNetwork::new(vec![v1,v2], bond_dims, None);
    /// let bond_dims_new = HashMap::from([(5, 17), (6, 19)]);
    /// tn.push_tensor(v3, Some(&bond_dims_new), None);
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
    /// let mut tn = TensorNetwork::new(vec![v1,v2], bond_dims, None);
    /// let bond_dims_new = HashMap::from([(5, 17)]);
    /// tn.push_tensor(v3, Some(&bond_dims_new), None);
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
    /// let mut tn = TensorNetwork::new(vec![v1,v2], bond_dims, None);
    /// let bond_dims_new = HashMap::from([(4, 3), (5, 17), (6,19)]);
    /// tn.push_tensor(v3, Some(&bond_dims_new), None);
    /// ```
    pub fn push_tensor(
        &mut self,
        tensor: Tensor,
        bond_dims: Option<&HashMap<i32, u64>>,
        ext: Option<&Vec<i32>>,
    ) {
        if let Some(bond_dims) = bond_dims {
            for leg in tensor.get_legs().iter() {
                if !self.bond_dims.contains_key(leg) {
                    if !bond_dims.contains_key(leg) {
                        panic!("Edge id {leg} bond dimension is not defined.");
                    }
                    self.bond_dims.entry(*leg).or_insert(bond_dims[leg]);
                } else if bond_dims.contains_key(leg)
                    && *self.bond_dims.get(leg).unwrap() != bond_dims[leg]
                {
                    panic!(
                        "Attempt to update bond {} with value: {}, previous value: {}",
                        leg,
                        &bond_dims[leg],
                        self.bond_dims.get(leg).unwrap()
                    )
                }
            }
        } else {
            for leg in tensor.get_legs() {
                if !self.bond_dims.contains_key(leg) {
                    panic!("Input {tensor:?} contains leg {leg}, with unknown bond dimension.");
                }
            }
        }
        //ensure that new tensor is connected
        self.update_edge(&tensor, ext);
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
            // Check for legs that are not shared between the two contracted tensors
            if !tensor_intersect.iter().any(|&i| i == *leg) {
                tensor_difference.push(*leg);
            }
            // Check if hyperedges are being contracted, if so, only append once to output tensor
            if self.edges[leg].len() > 2 && !tensor_difference.iter().any(|&i| i == *leg) {
                {
                    tensor_difference.push(*leg);
                }
            }
        }

        for leg in tensor_b_legs.iter() {
            self.edges.entry(*leg).and_modify(|e| {
                e.drain_filter(|e| {
                    if let Some(edge) = e {
                        *edge == tensor_a_loc as i32
                    } else {
                        false
                    }
                });
                for edge in &mut e.iter_mut() {
                    if let Some(tensor_loc) = edge {
                        if *tensor_loc as usize == tensor_b_loc {
                            *edge = Some(tensor_a_loc as i32);
                        }
                    }
                }
            });
        }
        self.tensors[tensor_a_loc] = Tensor::new(tensor_difference.clone());
        (tensor_intersect, tensor_difference)
    }
}

// Constructs Graphviz code showing the tensor network as a graph. The tensor numbering corresponds to their
// tensor index (i.e., their position in the tensors vector). The edges are annotated with the bond dims,
// as well as the edge id in smaller font.
// pub fn to_graphviz(&self) -> String {
//     let mut out = String::new();
//     let mut invis_counter = 0u32;
//     out.push_str("graph tn {\n");

//     for (i, tensor) in self.tensors.iter().enumerate() {
//         for leg in tensor.get_legs() {
//             let connection = self.edges[leg];

//             if let (Some(idx1), Some(_)) = connection {
//                 if idx1 == i as i32 {
//                     // prevent each edge being added twice, by only considering
//                     // edges where this tensor is in the first place
//                     continue;
//                 }
//             }

//             // Get tensor1 name (or create an invisible node if None)
//             let t1 = if let Some(idx) = connection.0 {
//                 format!("t{}", idx)
//             } else {
//                 let name = format!("i{}", invis_counter);
//                 writeln!(out, "\t{} [style=\"invis\"];", name).unwrap();
//                 invis_counter += 1;
//                 name
//             };

//             // Get tensor2 name (or create an invisible node if None)
//             let t2 = if let Some(idx) = connection.1 {
//                 format!("t{}", idx)
//             } else {
//                 let name = format!("i{}", invis_counter);
//                 writeln!(out, "\t{} [style=\"invis\"];", name).unwrap();
//                 invis_counter += 1;
//                 name
//             };

//             // Write edge between tensors
//             writeln!(out, "\t{} -- {} [label=\"{}\", taillabel=\"{}\", headlabel=\"{}\", labelfontsize=\"8pt\"];", t1, t2, self.bond_dims[leg], leg, leg).unwrap();
//         }
//     }

// Write edge between tensors
// writeln!(out, "\t{} -- {} [label=\"{}\", taillabel=\"{}\", headlabel=\"{}\", labelfontsize=\"8pt\"];", t1, t2, self.bond_dims[leg], leg, leg).unwrap();

/// Implementation of printing for TensorNetwork. Simply prints the Tensor objects in TensorNetwork
impl fmt::Display for TensorNetwork {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for (key, value) in &self.bond_dims {
            writeln!(f, "{key}: {value}")?;
        }
        write!(f, "Tensor: {:?}", self.tensors)
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
            None,
        )
    }

    fn setup_hyperedge() -> TensorNetwork {
        TensorNetwork::from_vector(
            vec![Tensor::new(vec![4, 3, 2]), Tensor::new(vec![0, 1, 3, 2])],
            vec![17, 18, 19, 12, 22],
            Some(&vec![2]),
        )
    }

    fn setup_hyperedge_complex() -> TensorNetwork {
        TensorNetwork::from_vector(
            vec![
                Tensor::new(vec![0, 1, 2]),
                Tensor::new(vec![1, 4]),
                Tensor::new(vec![1, 2, 3, 4, 5]),
                Tensor::new(vec![5, 6]),
            ],
            vec![5, 2, 4, 6, 8, 3, 7],
            Some(&vec![2]),
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
        let mut edge_sol = HashMap::<i32, Vec<Option<i32>>>::new();
        edge_sol.entry(0).or_insert(vec![Some(1), None]);
        edge_sol.entry(1).or_insert(vec![Some(1), None]);
        edge_sol.entry(2).or_insert(vec![Some(0), Some(1)]);
        edge_sol.entry(3).or_insert(vec![Some(0), Some(1)]);
        edge_sol.entry(4).or_insert(vec![Some(0), None]);
        let bond_dims = vec![17, 18, 19, 12, 22];
        let t = TensorNetwork::from_vector(tensors, bond_dims.clone(), None);
        for (index, leg) in t.bond_dims.iter().take(t.tensors.max_leg() as usize) {
            assert_eq!(*leg, bond_dims[*index as usize]);
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
        t.push_tensor(good_tensor, None, None);
        let mut edge_sol = HashMap::<i32, Vec<Option<i32>>>::new();
        edge_sol.entry(0).or_insert(vec![Some(1), Some(2)]);
        edge_sol.entry(1).or_insert(vec![Some(1), Some(2)]);
        edge_sol.entry(2).or_insert(vec![Some(0), Some(1)]);
        edge_sol.entry(3).or_insert(vec![Some(0), Some(1)]);
        edge_sol.entry(4).or_insert(vec![Some(0), Some(2)]);
        let bond_dims = vec![17, 18, 19, 12, 22];

        for (index, leg) in t.bond_dims.iter().take(t.tensors.max_leg() as usize) {
            assert_eq!(*leg, bond_dims[*index as usize]);
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
        t.push_tensor(good_tensor.clone(), Some(&good_bond_dims), None);
        for legs in good_tensor.get_legs() {
            assert_eq!(good_bond_dims[legs], t.bond_dims[legs]);
        }
        let mut edge_sol = HashMap::<i32, Vec<Option<i32>>>::new();
        edge_sol.entry(0).or_insert(vec![Some(1), None]);
        edge_sol.entry(1).or_insert(vec![Some(1), None]);
        edge_sol.entry(2).or_insert(vec![Some(0), Some(1)]);
        edge_sol.entry(3).or_insert(vec![Some(0), Some(1)]);
        edge_sol.entry(4).or_insert(vec![Some(0), None]);
        edge_sol.entry(7).or_insert(vec![Some(2), None]);
        edge_sol.entry(9).or_insert(vec![Some(3), None]);
        edge_sol.entry(12).or_insert(vec![Some(4), None]);
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
        t.push_tensor(bad_tensor, None, None);
    }

    #[test]
    #[should_panic(expected = "Attempt to update bond 0 with value: 12, previous value: 17")]
    fn test_push_tensor_bad_rewrite() {
        let mut t = setup();
        let bad_tensor = Tensor::new(vec![0, 1, 4]);
        let bad_bond_dims = HashMap::from([(0, 12), (1, 32), (4, 2)]);
        t.push_tensor(bad_tensor, Some(&bad_bond_dims), None);
    }

    #[test]
    fn test_tensor_contraction_good() {
        let mut t = setup();
        let (tensor_intersect, _tensor_difference) = t._contraction(0, 1);
        // contraction should maintain leg order
        let vec_sol = vec![0, 1, 4];
        let tensor_sol = Tensor::new(vec_sol.clone());
        let mut edge_sol = HashMap::<i32, Vec<Option<i32>>>::new();
        edge_sol.entry(0).or_insert(vec![Some(0), None]);
        edge_sol.entry(1).or_insert(vec![Some(0), None]);
        edge_sol.entry(2).or_insert(vec![Some(0)]);
        edge_sol.entry(3).or_insert(vec![Some(0)]);
        edge_sol.entry(4).or_insert(vec![Some(0), None]);

        assert_eq!(t.get_tensors()[0], tensor_sol);
        for edge_key in 0i32..4 {
            assert_eq!(edge_sol[&edge_key], t.get_edges()[&edge_key]);
        }

        assert_eq!(tensor_intersect, vec![3, 2]);
        assert_eq!(_tensor_difference, vec_sol);
    }

    #[test]
    fn test_tensor_hyperedge_contraction_good() {
        let mut t = setup_hyperedge();
        let (tensor_intersect, tensor_difference) = t._contraction(0, 1);
        // contraction should maintain leg order
        let tensor_intersect_sol = vec![3, 2];
        let tensor_difference_sol = vec![0, 1, 2, 4];
        let tensor_sol = Tensor::new(tensor_difference_sol.clone());
        let mut edge_sol = HashMap::<i32, Vec<Option<i32>>>::new();
        edge_sol.entry(0).or_insert(vec![Some(0), None]);
        edge_sol.entry(1).or_insert(vec![Some(0), None]);
        edge_sol.entry(2).or_insert(vec![Some(0), None]);
        edge_sol.entry(3).or_insert(vec![Some(0)]);
        edge_sol.entry(4).or_insert(vec![Some(0), None]);

        assert_eq!(t.get_tensors()[0], tensor_sol);
        for edge_key in 0i32..4 {
            assert_eq!(edge_sol[&edge_key], t.get_edges()[&edge_key]);
        }

        assert_eq!(tensor_intersect, tensor_intersect_sol);
        assert_eq!(tensor_difference, tensor_difference_sol);
    }

    #[test]
    fn test_update_edge() {
        let mut t = setup_hyperedge();
        let tensor = Tensor::new(vec![4, 5, 6]);
        // let bond_dims = vec![22, 5, 3];
        t.update_edge(&tensor, Some(&vec![5]));

        let mut edge_sol = HashMap::<i32, Vec<Option<i32>>>::new();
        edge_sol.entry(0).or_insert(vec![Some(1), None]);
        edge_sol.entry(1).or_insert(vec![Some(1), None]);
        edge_sol.entry(2).or_insert(vec![Some(0), Some(1), None]);
        edge_sol.entry(3).or_insert(vec![Some(0), Some(1)]);
        edge_sol.entry(4).or_insert(vec![Some(0), Some(2)]);
        edge_sol.entry(5).or_insert(vec![Some(2), None, None]);
        edge_sol.entry(6).or_insert(vec![Some(2), None]);


        for edge_key in 0i32..7 {
            assert_eq!(edge_sol[&edge_key], t.get_edges()[&edge_key]);
        }
    }

    #[test]
    fn test_tensor_complex_hyperedge_contraction_good() {
        let mut t = setup_hyperedge_complex();
        let mut edge_sol = HashMap::<i32, Vec<Option<i32>>>::new();
        edge_sol.entry(0).or_insert(vec![Some(0), None]);
        edge_sol.entry(1).or_insert(vec![Some(0), Some(1), Some(2)]);
        edge_sol.entry(2).or_insert(vec![Some(0), Some(2), None]);
        edge_sol.entry(3).or_insert(vec![Some(2), None]);
        edge_sol.entry(4).or_insert(vec![Some(1), Some(2)]);
        edge_sol.entry(5).or_insert(vec![Some(2), Some(3)]);
        edge_sol.entry(6).or_insert(vec![Some(3), None]);

        for edge_key in 0i32..7 {
            assert_eq!(edge_sol[&edge_key], t.get_edges()[&edge_key]);
        }
        edge_sol.clear();

        let ext_sol = vec![2, 0, 3, 6];
        assert_eq!(ext_sol, t.ext_edges);

        let (tensor_intersect, tensor_difference) = t._contraction(0, 1);
        // contraction should maintain leg order
        let tensor_intersect_sol = vec![1];
        let tensor_difference_sol = vec![1, 4, 0, 2];
        let tensor_sol = Tensor::new(tensor_difference_sol.clone());
        assert_eq!(t.get_tensors()[0], tensor_sol);
        assert_eq!(tensor_intersect, tensor_intersect_sol);
        assert_eq!(tensor_difference, tensor_difference_sol);

        edge_sol.entry(0).or_insert(vec![Some(0), None]);
        edge_sol.entry(1).or_insert(vec![Some(0), Some(2)]);
        edge_sol.entry(2).or_insert(vec![Some(0), Some(2), None]);
        edge_sol.entry(3).or_insert(vec![Some(2), None]);
        edge_sol.entry(4).or_insert(vec![Some(0), Some(2)]);
        edge_sol.entry(5).or_insert(vec![Some(2), Some(3)]);
        edge_sol.entry(6).or_insert(vec![Some(3), None]);

        for edge_key in 0i32..7 {
            assert_eq!(edge_sol[&edge_key], t.get_edges()[&edge_key]);
        }
        edge_sol.clear();

        let (tensor_intersect, tensor_difference) = t._contraction(0, 2);
        // contraction should maintain leg order
        let tensor_intersect_sol = vec![1, 2, 4];
        let tensor_difference_sol = vec![2, 3, 5, 0];
        let tensor_sol = Tensor::new(tensor_difference_sol.clone());
        assert_eq!(tensor_intersect, tensor_intersect_sol);
        assert_eq!(tensor_difference, tensor_difference_sol);
        assert_eq!(t.get_tensors()[0], tensor_sol);

        edge_sol.entry(0).or_insert(vec![Some(0), None]);
        edge_sol.entry(1).or_insert(vec![Some(0)]);
        edge_sol.entry(2).or_insert(vec![Some(0), None]);
        edge_sol.entry(3).or_insert(vec![Some(0), None]);
        edge_sol.entry(4).or_insert(vec![Some(0)]);
        edge_sol.entry(5).or_insert(vec![Some(0), Some(3)]);
        edge_sol.entry(6).or_insert(vec![Some(3), None]);

        for edge_key in 0i32..7 {
            assert_eq!(edge_sol[&edge_key], t.get_edges()[&edge_key]);
        }
        edge_sol.clear();

        let (tensor_intersect, tensor_difference) = t._contraction(0, 3);
        // contraction should maintain leg order
        let tensor_intersect_sol = vec![5];
        let tensor_difference_sol = vec![6, 2, 3, 0];
        let tensor_sol = Tensor::new(tensor_difference_sol.clone());
        assert_eq!(tensor_intersect, tensor_intersect_sol);
        assert_eq!(tensor_difference, tensor_difference_sol);
        assert_eq!(t.get_tensors()[0], tensor_sol);

        edge_sol.entry(0).or_insert(vec![Some(0), None]);
        edge_sol.entry(1).or_insert(vec![Some(0)]);
        edge_sol.entry(2).or_insert(vec![Some(0), None]);
        edge_sol.entry(3).or_insert(vec![Some(0), None]);
        edge_sol.entry(4).or_insert(vec![Some(0)]);
        edge_sol.entry(5).or_insert(vec![Some(0)]);
        edge_sol.entry(6).or_insert(vec![Some(0), None]);

        for edge_key in 0i32..7 {
            assert_eq!(edge_sol[&edge_key], t.get_edges()[&edge_key]);
        }
        edge_sol.clear();
    }
}
