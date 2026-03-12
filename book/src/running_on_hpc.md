{{#include links.md}}

# Running on HPC

## Running with MPI
The library can run on multiple connected compute nodes such as in a HPC setting.
It uses MPI for communication between nodes.
To run the code with MPI, compile it (e.g. using `cargo build -r`) and then execute the binary found in the `target`
folder using an MPI launcher (such as `mpirun`).
For example:
```shell
cargo build -r --example distributed_contraction
mpirun -n 4 target/release/examples/distributed_contraction
```
This command runs the executable on 4 nodes in parallel.
While the nodes could be on the same physical device, this would make memory limitations even more severe.
Instead, you usually want to do this with distributed nodes, where each node has its own memory.
This is common in HPC centers.

## Running on a cluster
Usually, HPC clusters use a cluster management system, most commonly SLURM.
Here, the steps to running this library are similar to:
1. Load modules for the required dependencies (HDF5, Boost, Python, ...)
2. Optionally create a virtual Python environment and install the Python dependencies for cotengra
3. Build the code using `cargo build -r` (assuming the login nodes have the same architecture as the compute nodes)
4. Write a SLURM script that calls `mpirun` on the compiled binary
5. Submit the job