[![check](https://github.com/qc-tum/TNC/actions/workflows/check.yml/badge.svg)](https://github.com/qc-tum/TNC/actions/workflows/check.yml)
[![test](https://github.com/qc-tum/TNC/actions/workflows/test.yml/badge.svg)](https://github.com/qc-tum/TNC/actions/workflows/test.yml)

[**Documentation**](https://qc-tum.github.io/TNC/)

# Tensornetwork Contraction Library

This library is designed to perform tensor network contractions, using partitioning of the network as parallelization strategy.
To this end, it ships multiple methods to partition a tensor network for lowest contraction cost, for example, based on simulated annealing.
The partitionings can then be contracted in parallel on a distributed-memory system, as common in high-performance computing.
Local contractions on a single system are also possible.

## Crate

### Requirements
The library requires a few dependencies to be installed on the system.
Those can be installed with
```shell
sudo apt install libhdf5-dev openmpi-bin libopenmpi-dev libboost-program-options-dev
```
Furthermore, for building the library, `cmake` and a C++ compiler must be installed on the system.

Additionally, to run the `HyperOptimizer` of [cotengra](https://github.com/jcmgray/cotengra), Python is required.
The following Python packages have to be installed (in a virtual environment if preferred):
```shell
pip install cotengra kahypar optuna
```

### Usage
The library can be added to an existing Rust project with e.g.
```shell
cargo add --git https://github.com/qc-tum/TNC.git --features cotengra tnc
```
or you can run the examples of this library by e.g.
```shell
cargo run --example local_contraction
```

### Features
- `cotengra`: Enables Rust bindings to the tree annealing, tree reconfiguration and tree tempering methods of cotengra

## Getting started

To familiarize yourself with the code, it is recommended to look at the examples and the documentation.
Some aspects of the library are also covered in more detail in the [Tutorial](https://qc-tum.github.io/TNC/tnc/_tutorial/index.html).
If you want to contribute, please take a loot at the Contribution guide.

## Example

```rust
use tnc::{
    contractionpath::paths::{
        cotengrust::{Cotengrust, OptMethod},
        FindPath,
    },
    qasm::import_qasm,
    tensornetwork::contraction::contract_tensor_network,
};

fn main() {
    // The QASM code prepares a GHZ state
    let code = "\
OPENQASM 2.0;
include \"qelib1.inc\";

qreg q[3];
h q[0];
cx q[0], q[1];
cx q[1], q[2];
";

    // Create a Circuit instance out of the code
    let circuit = import_qasm(code);

    // Create a tensor network that computes the full state vector.
    // This also returns a permutator which we need to apply to the final tensor to
    // make sure the entries are in the expected order.
    let (tensor_network, permutator) = circuit.into_statevector_network();

    // Find a contraction path to contract the tensor network.
    // We use a greedy path finder here.
    let mut opt = Cotengrust::new(&tensor_network, OptMethod::Greedy);
    opt.find_path();
    let path = opt.get_best_replace_path();

    // Contract the tensor network locally
    let final_tensor = contract_tensor_network(tensor_network, &path);

    // Apply the permutator to make sure the data is in the expected order
    let statevector = permutator.apply(final_tensor);

    // Get the data vector. Don't worry, the clone does not clone the data itself.
    let data = statevector.tensor_data().clone().into_data();

    // Print the data
    println!("Resulting statevector is: {:?}", data.elements());
}
```

## Publications
- <i>Optimizing Tensor Network Partitioning using Simulated Annealing</i>, Geiger et al. (2025): <https://arxiv.org/abs/2507.20667>
