use std::collections::HashMap;
use std::iter::zip;

use mpi::topology::SimpleCommunicator;
use mpi::traits::Equivalence;
use mpi::traits::*;
use num_complex::Complex64;

use crate::tensornetwork::contraction::{contract_tensor_network, TensorContraction};
use crate::tensornetwork::tensor::Tensor;
use crate::tensornetwork::tensordata::TensorData;
use crate::types::{ContractionIndex, EdgeIndex};

#[derive(Default, Clone, Equivalence)]
struct BondDim {
    bond_id: usize,
    bond_size: u64,
}

impl From<(usize, u64)> for BondDim {
    fn from(value: (usize, u64)) -> Self {
        Self {
            bond_id: value.0,
            bond_size: value.1,
        }
    }
}

fn string_to_static_str(s: String) -> &'static str {
    Box::leak(s.into_boxed_str())
}

pub fn scatter_tensor_network(
    r_tn: Tensor,
    path: &[ContractionIndex],
    rank: i32,
    size: i32,
    world: &SimpleCommunicator,
) -> (Tensor, Vec<ContractionIndex>) {
    let mut bond_num = 0;
    let root_process = world.process_at_rank(0);
    // Distribute bond_dims
    let mut bond_vec = if rank == 0 {
        bond_num = r_tn.get_bond_dims().len();
        root_process.broadcast_into(&mut bond_num);
        r_tn.get_bond_dims()
            .iter()
            .map(|(bond_id, bond_size)| BondDim {
                bond_id: *bond_id,
                bond_size: *bond_size,
            })
            .collect::<Vec<BondDim>>()
    } else {
        root_process.broadcast_into(&mut bond_num);
        vec![BondDim::default(); bond_num]
    };
    root_process.broadcast_into(&mut bond_vec);
    world.barrier();
    let bond_dims = HashMap::from_iter(bond_vec.iter().map(|e| (e.bond_id, e.bond_size)));

    let (local_tn, local_path) = if rank == 0 {
        let local_path = path[0].clone().get_data();
        let local_tn = r_tn.get_tensor(0).clone();
        for (i, contraction_path) in zip(1..size, path[1..size as usize].iter()) {
            match contraction_path {
                ContractionIndex::Path(_, local) => {
                    world.process_at_rank(i).send(local);
                }
                ContractionIndex::Pair(_, _) => panic!("Requires path"),
            }
        }

        for (i, tensor) in zip(1..size, r_tn.get_tensors()[1..size as usize].iter()) {
            let num_tensors = tensor.get_tensors().len();
            world.process_at_rank(i).send(&num_tensors);
            for inner_tensor in tensor.get_tensors() {
                let legs = inner_tensor.get_legs().clone();
                world.process_at_rank(i).send(&legs);

                let tensor_data = inner_tensor.get_tensor_data().clone();
                match tensor_data {
                    TensorData::Empty => {
                        world.process_at_rank(i).send(&0_i32);
                    }
                    TensorData::File(file_name) => {
                        world.process_at_rank(i).send(&1_i32);
                        world.process_at_rank(i).send(file_name.as_bytes());
                    }
                    TensorData::Gate((gate_name, angles)) => {
                        world.process_at_rank(i).send(&2_i32);
                        world.process_at_rank(i).send(gate_name.as_bytes());
                        world.process_at_rank(i).send(&angles);
                    }
                    TensorData::Matrix(matrix_data) => {
                        world.process_at_rank(i).send(&3_i32);
                        world.process_at_rank(i).send(&(*matrix_data.shape()));
                        world
                            .process_at_rank(i)
                            .send(&(*matrix_data.get_raw_data()));
                    }
                }
            }
        }
        (local_tn, local_path)
    } else {
        let (path, _status) = world.any_process().receive_vec::<ContractionIndex>();
        let local_path = path;
        let mut local_tn = Tensor::default();
        let (num_tensors, _status) = world.any_process().receive::<usize>();

        for _ in 0..num_tensors {
            // First send tensor legs
            let (legs, _status) = world.any_process().receive_vec::<EdgeIndex>();
            // Then determine data type of sent tensor data
            let (tensor_data_type, _status) = world.any_process().receive::<i32>();
            let mut tensor_data = TensorData::Empty;
            match tensor_data_type {
                0 => {}
                1 => {
                    let (file_name, _status) = world.any_process().receive::<u8>();
                    tensor_data = TensorData::File(file_name.to_string());
                }
                2 => {
                    let (gate_name, _status) = world.any_process().receive_vec::<u8>();
                    let gate_name = String::from_utf8(gate_name).unwrap();
                    let (gate_angles, _status) = world.any_process().receive_vec::<f64>();
                    tensor_data =
                        TensorData::Gate(((string_to_static_str(gate_name)), gate_angles));
                }
                3 => {
                    let (shape, _status) = world.any_process().receive_vec::<u32>();
                    let shape = shape.iter().map(|e| *e as u64).collect::<Vec<u64>>();
                    let (data, _status) = world.any_process().receive_vec::<Complex64>();
                    tensor_data = TensorData::new_from_flat(shape, data, None);
                }
                _ => {
                    panic!("Unrecognized data type");
                }
            }
            let new_tensor = Tensor::new(legs);
            new_tensor.set_tensor_data(tensor_data);
            local_tn.push_tensor(new_tensor, Some(&bond_dims), None);
        }
        (local_tn, local_path)
    };
    world.barrier();
    (local_tn, local_path)
}

pub fn intermediate_reduce_tensor_network(
    local_tn: &mut Tensor,
    path: &[ContractionIndex],
    rank: i32,
    _size: i32,
    world: &SimpleCommunicator,
) {
    let mut final_rank = 0;
    path.iter().for_each(|i| match i {
        ContractionIndex::Pair(x, y) => {
            let receiver = *x as i32;
            let sender = *y as i32;
            final_rank = receiver;
            if receiver == rank {
                let (legs, _status) = world.process_at_rank(sender).receive_vec::<EdgeIndex>();
                let returned_tensor = Tensor::new(legs);
                let (shape, _status) = world.process_at_rank(sender).receive_vec::<u32>();
                let shape = shape.iter().map(|e| *e as u64).collect::<Vec<u64>>();
                let (data, _status) = world.process_at_rank(sender).receive_vec::<Complex64>();
                let tensor_data = TensorData::new_from_flat(shape, data, None);
                returned_tensor.set_tensor_data(tensor_data);
                local_tn.push_tensor(returned_tensor, None, None);
                contract_tensor_network(local_tn, &[ContractionIndex::Pair(0, 1)]);
            }
            if sender == rank {
                let legs = local_tn.get_legs().clone();
                world.process_at_rank(receiver).send(&legs);
                let local_tensor = local_tn.get_data();
                world
                    .process_at_rank(receiver)
                    .send(&(*local_tensor.shape()));
                world
                    .process_at_rank(receiver)
                    .send(&(*local_tensor.get_raw_data()));
                std::mem::drop(local_tensor);
            }
        }
        ContractionIndex::Path(..) => (),
    });

    // Only runs if the final contracted process is not process 0
    if final_rank != 0 {
        if rank == 0 {
            let (legs, _status) = world.process_at_rank(final_rank).receive_vec::<EdgeIndex>();
            let returned_tensor = Tensor::new(legs);
            let (shape, _status) = world.process_at_rank(final_rank).receive_vec::<u32>();
            let shape = shape.iter().map(|e| *e as u64).collect::<Vec<u64>>();
            let (data, _status) = world.process_at_rank(final_rank).receive_vec::<Complex64>();
            let tensor_data = TensorData::new_from_flat(shape, data, None);
            returned_tensor.set_tensor_data(tensor_data);
            // return returned_tensor;
            *local_tn = returned_tensor;
        }
        if rank == final_rank {
            let legs = local_tn.get_legs().clone();
            world.process_at_rank(0).send(&legs);
            let local_tensor = local_tn.get_data();
            world.process_at_rank(0).send(&(*local_tensor.shape()));
            world
                .process_at_rank(0)
                .send(&(*local_tensor.get_raw_data()));
        }
    }
}

pub fn naive_reduce_tensor_network(
    local_tn: &mut Tensor,
    path: &[ContractionIndex],
    rank: i32,
    size: i32,
    world: &SimpleCommunicator,
) {
    if rank == 0 {
        for i in 1..size {
            let (legs, _status) = world.process_at_rank(i).receive_vec::<EdgeIndex>();
            let returned_tensor = Tensor::new(legs);
            let (shape, _status) = world.process_at_rank(i).receive_vec::<u32>();
            let shape = shape.iter().map(|e| *e as u64).collect::<Vec<u64>>();
            let (data, _status) = world.process_at_rank(i).receive_vec::<Complex64>();
            let tensor_data = TensorData::new_from_flat(shape, data, None);
            returned_tensor.set_tensor_data(tensor_data);
            local_tn.push_tensor(returned_tensor, None, None);
        }
    } else {
        let legs = local_tn.get_legs().clone();
        world.process_at_rank(0).send(&legs);
        let local_tensor = local_tn.get_data();
        world.process_at_rank(0).send(&(*local_tensor.shape()));
        world
            .process_at_rank(0)
            .send(&(*local_tensor.get_raw_data()));
    }
    world.barrier();
    if rank == 0 {
        contract_tensor_network(local_tn, &path[(size as usize)..path.len()]);
    }
}
