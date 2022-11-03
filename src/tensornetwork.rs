use array_tool::vec::{Intersect, Union};
use std::collections::HashMap;
use std::fmt;

pub mod contraction;
pub mod tensor;

use crate::tensornetwork::tensor::Tensor;

#[path = "tensornetwork_tests.rs"]
mod tensornetwork_tests;

pub trait MaximumLeg {
    fn max_leg(&self) -> i32;
}

#[derive(Debug, Clone, PartialEq)]
pub struct TensorNetwork {
    tensors: Vec<Tensor>,
    bond_dims: HashMap<i32, u32>,
    edges: HashMap<i32, (Option<i32>, Option<i32>)>,
}

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

impl TensorNetwork {
    // Create empty TensorNetwork
    pub fn empty_tensor_network() -> Self {
        Self {
            tensors: Vec::<Tensor>::new(),
            bond_dims: HashMap::new(),
            edges: HashMap::new(),
        }
    }

    pub fn get_edges(&self) -> &HashMap<i32, (Option<i32>, Option<i32>)> {
        &self.edges
    }

    pub fn get_tensors(&self) -> &Vec<Tensor> {
        &self.tensors
    }

    // Creating custom implementation that accepts list of bond_dims
    pub fn new(tensors: Vec<Tensor>, bond_dims: Vec<u32>) -> Self {
        assert!(tensors.max_leg() < bond_dims.capacity() as i32);
        let mut edges: HashMap<i32, (Option<i32>, Option<i32>)> = HashMap::new();
        for index in 0usize..tensors.capacity() {
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

    // Adding a single tensor to the tensor network
    pub fn push_tensor(&mut self, tensor: Tensor, bond_dims: Option<Vec<u32>>) {
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
            for (index, leg) in (0usize..).zip(tensor.get_legs()) {
                if self.bond_dims.get(leg).is_none() {
                    self.bond_dims
                        .entry(*leg)
                        .or_insert(bond_dims.as_ref().unwrap()[index]);
                } else if *self.bond_dims.get(leg).unwrap() != bond_dims.as_ref().unwrap()[index] {
                    panic!(
                        "Attempt to update bond {} with value: {}, previous value: {}",
                        leg,
                        &bond_dims.as_ref().unwrap()[index],
                        self.bond_dims.get(leg).unwrap()
                    )
                }
            }
        }

        self.tensors.push(tensor);
    }

    //implementation for Tensor as vec<i32>
    pub fn contraction(&mut self, tensor_a_loc: usize, tensor_b_loc: usize) -> (u32, u32) {
        let tensor_a_legs = self.tensors[tensor_a_loc].get_legs();
        let tensor_b_legs = self.tensors[tensor_b_loc].get_legs();

        let tensor_union = tensor_a_legs.union(tensor_b_legs.to_vec());
        let tensor_intersect = tensor_a_legs.intersect(tensor_b_legs.to_vec());

        let mut tensor_difference: Vec<i32> = Vec::new();
        for leg in tensor_union.iter() {
            if !tensor_intersect.iter().any(|&i| i == *leg) {
                tensor_difference.push(*leg);
            }
        }

        let time_complexity = tensor_union
            .iter()
            .map(|x| self.bond_dims.get(x).unwrap())
            .product();
        let space_complexity : u32= tensor_a_legs
            .iter()
            .map(|x| self.bond_dims.get(x).unwrap())
            .product::<u32>()
            + tensor_b_legs
                .iter()
                .map(|x| self.bond_dims.get(x).unwrap())
                .product::<u32>()
            + tensor_difference
                .iter()
                .map(|x| self.bond_dims.get(x).unwrap())
                .product::<u32>();

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
        self.tensors[tensor_a_loc] = Tensor::new(tensor_difference);

        (time_complexity, space_complexity)
    }

    //implementation for Tensor as vec<i32>
    pub fn contraction_hash(&mut self, tensor_a: Tensor, tensor_b: Tensor) -> (i32, i32) {
        let tensor_union = tensor_a.get_legs().union(tensor_b.get_legs().to_vec());
        let tensor_intersect = tensor_a.get_legs().intersect(tensor_b.get_legs().to_vec());
        let mut tensor_difference = Vec::new();
        for leg in tensor_union {
            if tensor_intersect.iter().any(|&i| i == leg) {
                tensor_difference.push(leg);
            }
        }
        (3, 2)
    }
}

impl fmt::Display for TensorNetwork {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for (key, value) in &self.bond_dims {
            println!("{}: {}", key, value);
        }
        write!(f, "Tensor: {:?}", self.tensors)
    }
}
