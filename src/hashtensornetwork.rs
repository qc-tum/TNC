use array_tool::vec::{Intersect, Union};
use std::collections::HashMap;
use std::fmt;

pub mod hashtensor;

use crate::hashtensornetwork::hashtensor::HashTensor;
use crate::MaximumLeg;

#[path = "hashtensornetwork_tests.rs"]
mod hashtensornetwork_tests;



#[derive(Debug, Clone, PartialEq)]
pub struct HashTensorNetwork {
    tensors: Vec<HashTensor>,
    bond_dims: HashMap<i32, u32>,
    edges: HashMap<i32, (Option<i32>, Option<i32>)>,
}

impl MaximumLeg for Vec<HashTensor> {
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


impl HashTensorNetwork {
    // Create empty TensorNetwork
    pub fn empty_tensor_network() -> Self {
        Self {
            tensors: Vec::<HashTensor>::new(),
            bond_dims: HashMap::new(),
            edges: HashMap::new(),
        }
    }

    pub fn get_edges(&self) -> &HashMap<i32, (Option<i32>, Option<i32>)> {
        &self.edges
    }

    pub fn get_tensors(&self) -> &Vec<HashTensor> {
        &self.tensors
    }

    // Creating custom implementation that accepts list of bond_dims
    pub fn new(tensors: Vec<HashTensor>, bond_dims: Vec<u32>) -> Self {
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
    pub fn push_tensor(&mut self, tensor: HashTensor, bond_dims: Option<Vec<u32>>) {
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

        let tensor_union = tensor_a_legs.union(tensor_b_legs);
        let tensor_difference = tensor_a_legs.symmetric_difference(tensor_b_legs);

        let time_complexity = tensor_union
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
                .clone()
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
        self.tensors[tensor_a_loc] = HashTensor::new(tensor_difference.into_iter().map(|e| *e).collect());

        (time_complexity, space_complexity)
    }

}

impl fmt::Display for HashTensorNetwork {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for (key, value) in &self.bond_dims {
            println!("{}: {}", key, value);
        }
        write!(f, "Tensor: {:?}", self.tensors)
    }
}

