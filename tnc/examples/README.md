# Examples

This folder contains examples of the usage of the library.
They can be executed with
```bash
cargo run --example name
```
The necessary features (if any) will be automatically activated.

## Overview
Here is an overview of the examples in this folder:

`distributed_contraction`:
Distributed contraction of partitioned tensor networks using MPI.
- Creates a small [Sycamore](https://arxiv.org/pdf/1910.11333)-like circuit
- Creates a tensor network that computes a single amplitude
- Partitions the tensor network into multiple tensor networks
- Finds a contraction path
- Distributes the tensor network
- Contracts the tensor networks on each compute node (intra-node)
- Contracts the results (inter-node) to get the final result

`local_contraction`:
Local contraction of a tensor network on one machine.
- Loads a small circuit (QASM code)
- Creates a tensor network that computes the state vector
- Finds a contraction path
- Contracts the tensor network locally (no MPI / distributed memory)