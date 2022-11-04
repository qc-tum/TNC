#![feature(test)]
pub mod tensornetwork;

use crate::tensornetwork::tensor::Tensor;
use crate::tensornetwork::TensorNetwork;

pub trait MaximumLeg {
    fn max_leg(&self) -> i32;
}
fn main() {
    let tn = TensorNetwork::new(
        vec![Tensor::new(vec![4, 3, 2]), Tensor::new(vec![0, 1, 3, 2])],
        vec![17, 18, 19, 12, 22],
    );
    for edge in tn.get_edges(){
        println!("{:}->({:}, {:})", edge.0, (*edge.1).0.unwrap_or_else(|| -1) , (*edge.1).1.unwrap_or_else(|| -1));
    }
}
