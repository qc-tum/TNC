pub mod tensornetwork;

use crate::tensornetwork::tensor::Tensor;
use crate::tensornetwork::TensorNetwork;

fn main() {
    let mut t= TensorNetwork::new(
        vec![
            Tensor::new(vec![4, 3, 2]),
            Tensor::new(vec![0, 1, 3, 2]),
        ],
        vec![17, 18, 19, 12, 22],
    );
    let bad_tensor = Tensor::new(vec![0, 1, 4]);
    let bad_bond_dims = vec![12, 32, 2];
    t.push_tensor(bad_tensor, Some(bad_bond_dims));

    println!("My Tensornetwork\n{}!", t);
}
