use std::collections::HashMap;
use std::fmt;
use array_tool::vec::Union;


pub mod contraction;
pub mod tensor;

use crate::tensornetwork::tensor::Tensor;

#[path = "tensornetwork_tests.rs"]
mod tensornetwork_tests;

pub trait Maximum {
    fn maximum(&self) -> i32;
}

#[derive(Debug, Clone, PartialEq)]
pub struct TensorNetwork{
    tensors: Vec<Tensor>,
    bond_dims: HashMap<i32, u32>,
    edges: HashMap<i32, (Option<i32>, Option<i32>)>,
}

impl Maximum for Vec<Tensor> {
    fn maximum(&self) -> i32 {
        let mut m = self[0].maximum();
        for tensor in self.iter() {
            let n = tensor.maximum();
            if n > m {
                m = n;
            }
        }
        m
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

    // Creating custom implementation that accepts list of bond_dims
    pub fn new(tensors: Vec<Tensor>, bond_dims: Vec<u32>) -> Self {
        assert!(tensors.maximum() < bond_dims.capacity() as i32);
        let mut edges: HashMap<i32, (Option<i32>, Option<i32>)> = HashMap::new();
        for index in 0usize..tensors.capacity(){
            for leg in tensors[index].get_legs(){
                edges.entry(*leg)
                .and_modify(|&mut mut edge| {edge.1 = Some(index as i32)})
                .or_insert((Some(index as i32), None));
            }
        }
        Self {
            tensors,
            bond_dims: (0i32..).zip(bond_dims).collect(),
            edges
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


}

impl fmt::Display for TensorNetwork {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for (key, value) in &self.bond_dims {
            println!("{}: {}", key, value);
        }
        write!(f, "Tensor: {:?}", self.tensors)
    }
}
