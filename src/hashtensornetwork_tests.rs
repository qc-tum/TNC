extern crate test;

#[cfg(test)]
mod tests {
    // use rand::distributions::{Distribution, Uniform};
    // TODO: Use random tensors
    use std::collections::HashMap;
    use linked_hash_set::LinkedHashSet;
    use crate::hashtensornetwork::HashTensorNetwork;
    use crate::hashtensornetwork::hashtensor::HashTensor;
    use crate::hashtensornetwork::MaximumLeg;
    use super::test::Bencher;

    fn setup() -> HashTensorNetwork {
        let mut tensor_a = LinkedHashSet::new();
        tensor_a.insert(4);
        tensor_a.insert(3);
        tensor_a.insert(2);

        let mut tensor_b = LinkedHashSet::new();
        tensor_b.insert(0);
        tensor_b.insert(1);
        tensor_b.insert(3);
        tensor_b.insert(2);
        HashTensorNetwork::new(
            vec![
                HashTensor::new(tensor_a),
                HashTensor::new(tensor_b),
            ],
            vec![17, 18, 19, 12, 22],
        )
    }

    #[test]
    fn test_empty_tensor_network() {
        let t = HashTensorNetwork::empty_tensor_network();
        assert!(t.tensors.is_empty());
        assert!(t.bond_dims.is_empty());
    }
    #[test]
    fn test_new() {
        let mut tensor_a = LinkedHashSet::new();
        tensor_a.insert(4);
        tensor_a.insert(3);
        tensor_a.insert(2);

        let mut tensor_b = LinkedHashSet::new();
        tensor_b.insert(0);
        tensor_b.insert(1);
        tensor_b.insert(3);
        tensor_b.insert(2);
        let t = HashTensorNetwork::new(
            vec![
                HashTensor::new(tensor_a),
                HashTensor::new(tensor_b),
            ],
            vec![17, 18, 19, 12, 22],
        );

        let mut edge_sol = HashMap::<i32, (Option<i32>, Option<i32>)>::new();
        edge_sol.entry(0).or_insert((Some(1), None));
        edge_sol.entry(1).or_insert((Some(1), None));
        edge_sol.entry(2).or_insert((Some(0), Some(1)));
        edge_sol.entry(3).or_insert((Some(0), Some(1)));
        edge_sol.entry(4).or_insert((Some(0), None));
        let bond_dims = vec![17, 18, 19, 12, 22];

        for leg in 0..t.tensors.max_leg() as usize {
            assert_eq!(t.bond_dims[&(leg as i32)], bond_dims[leg]);
        }

        for edge_key in 0i32..4{
            assert_eq!(edge_sol[&edge_key], t.get_edges()[&edge_key]);
        }

    }

    #[test]
    fn test_push_tensor_good() {
        //TODO: Add test to check for edge update
        let mut t = setup();

        let mut tensor_new = LinkedHashSet::new();
        tensor_new.insert(0);
        tensor_new.insert(1);
        tensor_new.insert(4);

        let good_tensor = HashTensor::new(tensor_new);
        t.push_tensor(good_tensor, None);
    }

    #[test]
    fn test_push_tensor_good_newlegs() {
        let mut t = setup();

        let mut tensor_new = LinkedHashSet::new();
        tensor_new.insert(7);
        tensor_new.insert(9);
        tensor_new.insert(12);
 
        let good_tensor = HashTensor::new(tensor_new);
        let good_bond_dims = vec![55, 5, 6];
        t.push_tensor(good_tensor.clone(), Some(good_bond_dims.clone()));
        for (index, legs) in (0usize..).zip(good_tensor.get_legs()){
            assert_eq!(good_bond_dims[index], t.bond_dims[legs]);
        }
    }

    #[test]
    #[should_panic(
        expected = "Input HashTensor { legs: {0, 5, 4} } contains leg 5, with unknown bond dimension."
    )]
    fn test_push_tensor_bad() {
        let mut t = setup();
        let mut tensor_new = LinkedHashSet::new();
        tensor_new.insert(0);
        tensor_new.insert(5);
        tensor_new.insert(4);

        let bad_tensor = HashTensor::new(tensor_new);
        t.push_tensor(bad_tensor, None);
    }

    #[test]
    #[should_panic(
        expected = "Attempt to update bond 0 with value: 12, previous value: 17"
    )]
    fn test_push_tensor_bad_rewrite() {
        let mut t = setup();

        let mut tensor_new = LinkedHashSet::new();
        tensor_new.insert(0);
        tensor_new.insert(1);
        tensor_new.insert(4);

        let bad_tensor = HashTensor::new(tensor_new);
        let bad_bond_dims = vec![12, 32, 2];
        t.push_tensor(bad_tensor, Some(bad_bond_dims));
    }

    #[test]
    fn test_tensor_contraction_good() {
        let mut t = setup();
        let (time_complexity, space_complexity) = t.contraction(0, 1);
        // contraction should maintain leg order

        let mut tensor_new = LinkedHashSet::new();
        tensor_new.insert(0);
        tensor_new.insert(1);
        tensor_new.insert(4);

        let tensor_sol = HashTensor::new(tensor_new);
        let mut edge_sol = HashMap::<i32, (Option<i32>, Option<i32>)>::new();
        edge_sol.entry(0).or_insert((Some(0), None));
        edge_sol.entry(1).or_insert((Some(0), None));
        edge_sol.entry(2).or_insert((Some(0), Some(0)));
        edge_sol.entry(3).or_insert((Some(0), Some(0)));
        edge_sol.entry(4).or_insert((Some(0), None));

        assert_eq!(t.get_tensors()[0], tensor_sol);
        for edge_key in 0i32..4{
            assert_eq!(edge_sol[&edge_key], t.get_edges()[&edge_key]);
        }

        assert_eq!(time_complexity, 1534896);
        assert_eq!(space_complexity, 81516);
    }

    #[bench]
    fn build_tensor(b: &mut Bencher) {
        b.iter(||{
            setup();
        } )
    }
    #[bench]
    fn contract_tensor(b: &mut Bencher) {
        b.iter(||{
            let mut t = setup();
            t.contraction(0, 1);
        } )
    }

}

