[![check](https://github.com/qc-tum/TNC/actions/workflows/check.yml/badge.svg)](https://github.com/qc-tum/TNC/actions/workflows/check.yml)
[![test](https://github.com/qc-tum/TNC/actions/workflows/test.yml/badge.svg)](https://github.com/qc-tum/TNC/actions/workflows/test.yml)

# Tensornetwork Contraction Library

This library is designed to perform tensor network contractions, using partitioning of the network as parallelization strategy.
To this end, it ships multiple methods to partition a tensor network for lowest contraction cost, for example, based on simulated annealing.
The partitionings can then be contracted in parallel on a distributed-memory system, as common in high-performance computing.
Local contractions on a single system are also possible.

## Crate

### Installation
The library requires a few dependencies to be installed on the system.
Those can be installed with
```shell
sudo apt install libhdf5-dev openmpi-bin libopenmpi-dev libboost-program-options-dev
```

Additionally, to run the `HyperOptimizer` of cotengra, Python is required.
The following Python packages have to be installed:
```shell
pip install cotengra kahypar optuna
```

Finally, the crate currently requires the nightly toolchain.
To tell Rust to use nightly for this library, run from anywhere inside the library directory the following:
```shell
rustup override set nightly
```

### Features
- `cotengra`: Enables Rust bindings to the tree annealing, tree reconfiguration and tree tempering methods of cotengra

## Example

```rust
use std::fs;

use mpi::{topology::SimpleCommunicator, traits::Communicator};
use num_complex::Complex64;
use tensorcontraction::{
    contractionpath::paths::{
        cotengrust::{Cotengrust, OptMethod},
        OptimizePath,
    },
    mpi::communication::{
        broadcast_path, extract_communication_path, intermediate_reduce_tensor_network,
        scatter_tensor_network,
    },
    qasm::qasm_to_tensornetwork::create_tensornetwork,
    tensornetwork::{
        contraction::contract_tensor_network,
        partitioning::{
            find_partitioning, partition_config::PartitioningStrategy, partition_tensor_network,
        },
        tensor::Tensor,
        tensordata::TensorData,
    },
};


fn main() {
    // Read from file
    let tensor = read_qasm("foo.qasm");

    // Set up MPI
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank();
    let size = world.size();

    // Perform the contraction
    let result = if size == 1 {
        local_contraction(tensor)
    } else {
        distributed_contraction(tensor, &world)
    };

    // Print the result
    if rank == 0 {
        println!("{result:?}");
    }
}


fn read_qasm(file: &str) -> Tensor {
    let source = fs::read_to_string(file).unwrap();
    let (mut tensor, open_legs) = create_tensornetwork(source);

    // Add bras to each open leg to compute a single amplitude
    for leg in open_legs {
        // Create a new tensor with a single leg of dimension 2
        let mut bra = Tensor::new_from_const(vec![leg], 2);
        // Set the actual data of the tensor
        bra.set_tensor_data(TensorData::new_from_data(
            &[2],
            vec![Complex64::ONE, Complex64::ZERO],
            None,
        ));
        // Add the bra to our tensor network
        tensor.push_tensor(bra);
    }
    tensor
}


fn local_contraction(tensor: Tensor) -> Tensor {
    // Find a contraction path for the whole network
    let mut opt = Cotengrust::new(&tensor, OptMethod::Greedy);
    opt.optimize_path();
    let contract_path = opt.get_best_replace_path();

    // Contract the whole tensor network on this single node
    contract_tensor_network(tensor, &contract_path)
}


fn distributed_contraction(tensor: Tensor, world: &SimpleCommunicator) -> Tensor {
    let rank = world.rank();
    let size = world.size();
    let root = world.process_at_rank(0);

    let (partitioned_tn, path) = if rank == 0 {
        // Find a partitioning for the tensor network
        let partitioning = find_partitioning(&tensor, size, PartitioningStrategy::MinCut, true);
        let partitioned_tn = partition_tensor_network(tensor, &partitioning);

        // Find a contraction path for the individual partitions and the final fan-in
        let mut opt = Cotengrust::new(&partitioned_tn, OptMethod::Greedy);
        opt.optimize_path();
        let path = opt.get_best_replace_path();

        (partitioned_tn, path)
    } else {
        Default::default()
    };

    // Distribute partitions to ranks
    let (mut local_tn, local_path, comm) =
        scatter_tensor_network(&partitioned_tn, &path, rank, size, &world);

    // Contract the partitions on each rank
    local_tn = contract_tensor_network(local_tn, &local_path);

    // Get the part of the path that describes the final fan-in between ranks
    let mut communication_path = if rank == 0 {
        extract_communication_path(&path)
    } else {
        Default::default()
    };
    broadcast_path(&mut communication_path, &root);

    // Perform the final fan-in, sending tensors between ranks and contracting them
    // until there is only the final tensor left, which will end up on rank 0.
    intermediate_reduce_tensor_network(&mut local_tn, &communication_path, rank, &world, &comm);

    local_tn
}
```

## Publications
- <i>Optimizing Tensor Network Partitioning using Simulated Annealing</i>, Geiger et al. (2025): <https://arxiv.org/abs/2507.20667>

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed under the terms of both the Apache License, Version 2.0 and the MIT license without any additional terms or conditions.
