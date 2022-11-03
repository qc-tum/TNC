#![feature(test)]

extern crate test;

#[cfg(test)]
mod tests {
    use rand::distributions::{Distribution, Uniform};
    // TODO: Use random tensors
    use crate::tensornetwork::TensorNetwork;
    use crate::tensornetwork::Maximum;
    use crate::tensornetwork::tensor::Tensor;
    use super::test::Bencher;

    fn setup() -> TensorNetwork {
        TensorNetwork::new(
            vec![
                Tensor::new(vec![4, 3, 2]),
                Tensor::new(vec![0, 1, 3, 2]),
            ],
            vec![17, 18, 19, 12, 22],
        )
    }

    #[test]
    fn test_empty_tensor_network() {
        let t = TensorNetwork::empty_tensor_network();
        assert!(t.tensors.is_empty());
        assert!(t.bond_dims.is_empty());
    }
    #[test]
    fn test_new() {
        let tensors = vec![
            Tensor::new(vec![4, 3, 2]),
            Tensor::new(vec![0, 1, 3, 2]),
        ];
        let bond_dims = vec![17, 18, 19, 12, 22];
        let t = TensorNetwork::new(tensors, bond_dims.clone());
        for leg in 0..t.tensors.maximum() as usize {
            assert_eq!(t.bond_dims[&(leg as i32)], bond_dims[leg]);
        }
    }

    #[test]
    fn test_push_tensor_good() {
        let mut t = setup();
        let good_tensor = Tensor::new(vec![0, 1, 4]);
        t.push_tensor(good_tensor, None);
    }

    #[test]
    fn test_push_tensor_good_newlegs() {
        let mut t = setup();    
        let good_tensor = Tensor::new(vec![7, 9, 12]);
        let good_bond_dims = vec![55, 5, 6];
        t.push_tensor(good_tensor.clone(), Some(good_bond_dims.clone()));
        for (index, legs) in (0usize..).zip(good_tensor.get_legs()){
            assert_eq!(good_bond_dims[index], t.bond_dims[legs]);
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
    #[should_panic(
        expected = "Attempt to update bond 0 with value: 12, previous value: 17"
    )]
    fn test_push_tensor_bad_rewrite() {
        let mut t = setup();
        let bad_tensor = Tensor::new(vec![0, 1, 4]);
        let bad_bond_dims = vec![12, 32, 2];
        t.push_tensor(bad_tensor, Some(bad_bond_dims));
    }

    #[bench]
    fn build_tensor(b: &mut Bencher) {
        b.iter(||{
            let tensors = vec![
            Tensor::new(vec![4, 3, 2]),
            Tensor::new(vec![0, 1, 3, 2]),
        ];
        let bond_dims = vec![17, 18, 19, 12, 22];
        let t = TensorNetwork::new(tensors, bond_dims.clone());
        for leg in 0..t.tensors.maximum() as usize {
            assert_eq!(t.bond_dims[&(leg as i32)], bond_dims[leg]);
        }
        } )
    }

}
