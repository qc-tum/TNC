# Examples

This folder contains examples of the usage of the library.
They can be executed with
```bash
cargo run --example name
```
The necessary features (if any) will be automatically activated.

## Overview
Here is an overview of the examples in this folder:

### `distributed_contraction`
Distributed contraction of partitioned tensor networks using MPI. Execute with `mpirun` to actually run multiple processes (see the tutorial on running on HPC).
- Creates a small [Sycamore](https://arxiv.org/pdf/1910.11333)-like circuit
- Creates a tensor network that computes a single amplitude
- Partitions the tensor network into multiple tensor networks
- Finds a contraction path
- Distributes the tensor network
- Contracts the tensor networks on each compute node (intra-node)
- Contracts the results (inter-node) to get the final result

### `local_contraction`
Local contraction of a tensor network on one machine.
- Loads a small circuit (QASM code)
- Creates a tensor network that computes the state vector
- Finds a contraction path
- Contracts the tensor network locally (no MPI / distributed memory)

### `repartitioning`
Using the simulated annealing algorithm of the library to improve the initial partitioning of a tensor network for lower contraction cost.
- Creates a small Sycamore-like circuit
- Creates a tensor network that computes the expectation value
- Finds a naive partitioning (using KaHyPar)
- Computes the contraction cost when using this partitioning
- Uses simulated annealing to find an improved partitioning
- Computes again the contraction cost