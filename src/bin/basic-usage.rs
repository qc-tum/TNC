
extern crate tensorcontraction;
use tensorcontraction::tensornetwork::{TensorNetwork, tensor::Tensor};


fn main() {
    let tn = TensorNetwork::from_vector(
        vec![Tensor::new(vec![4, 3, 2]), Tensor::new(vec![0, 1, 3, 2])],
        vec![17, 18, 19, 12, 22],
    );
    for edge in tn.get_edges(){
        println!("{:}->({:}, {:})", edge.0, (*edge.1).0.unwrap_or_else(|| -1) , (*edge.1).1.unwrap_or_else(|| -1));
    }
}
