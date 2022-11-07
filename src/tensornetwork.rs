use array_tool::vec::{Intersect, Union};
use std::collections::HashMap;
use std::fmt;

pub mod contraction;
pub mod tensor;

use tensor::Tensor;

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
        //ensure that new tensor is connected
        self.update_edge(&tensor);
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
        let space_complexity: u32 = tensor_a_legs
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
}

impl fmt::Display for TensorNetwork {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for (key, value) in &self.bond_dims {
            println!("{}: {}", key, value);
        }
        write!(f, "Tensor: {:?}", self.tensors)
    }
}

mod tests {
    use rand::distributions::{Distribution, Uniform};
    // TODO: Use random tensors
    use crate::tensornetwork::tensor::Tensor;
    use crate::tensornetwork::MaximumLeg;
    use crate::tensornetwork::TensorNetwork;
    use std::collections::HashMap;

    // fn generate_random_tensor() -> (Tensor, u32) {
    //     let tensor_size = Uniform::from(3..1000);
    //     let rng = rand::thread_rng();
    //     let mut tensor_legs = Vec::new();
    //     for _i in 0i32..tensor_size.sample(&mut rng.clone()) {
    //         tensor_legs.push(tensor_size.sample(&mut rng.clone()));
    //     }
    //     let new_tensor = Tensor::new(tensor_legs);
    //     let size = new_tensor.get_legs().len();
    //     (new_tensor, size as u32)
    // }

    fn setup() -> TensorNetwork {
        TensorNetwork::new(
            vec![Tensor::new(vec![4, 3, 2]), Tensor::new(vec![0, 1, 3, 2])],
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
        let tensors = vec![Tensor::new(vec![4, 3, 2]), Tensor::new(vec![0, 1, 3, 2])];
        let mut edge_sol = HashMap::<i32, (Option<i32>, Option<i32>)>::new();
        edge_sol.entry(0).or_insert((Some(1), None));
        edge_sol.entry(1).or_insert((Some(1), None));
        edge_sol.entry(2).or_insert((Some(0), Some(1)));
        edge_sol.entry(3).or_insert((Some(0), Some(1)));
        edge_sol.entry(4).or_insert((Some(0), None));
        let bond_dims = vec![17, 18, 19, 12, 22];
        let t = TensorNetwork::new(tensors, bond_dims.clone());
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
        let good_bond_dims = vec![55, 5, 6];
        t.push_tensor(good_tensor.clone(), Some(good_bond_dims.clone()));
        for (index, legs) in (0usize..).zip(good_tensor.get_legs()) {
            assert_eq!(good_bond_dims[index], t.bond_dims[legs]);
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
            assert_eq!(t.bond_dims[leg], *x.next().unwrap() as u32);
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
        let bad_bond_dims = vec![12, 32, 2];
        t.push_tensor(bad_tensor, Some(bad_bond_dims));
    }

    #[test]
    fn test_tensor_contraction_good() {
        let mut t = setup();
        let (time_complexity, space_complexity) = t.contraction(0, 1);
        // contraction should maintain leg order
        let tensor_sol = Tensor::new(vec![4, 0, 1]);
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

        assert_eq!(time_complexity, 1534896);
        assert_eq!(space_complexity, 81516);
    }
}
