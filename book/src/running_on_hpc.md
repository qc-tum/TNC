{{#include links.md}}

# Running on HPC

## Running with MPI
The library can run on multiple connected compute nodes such as in a HPC setting.
It uses MPI for communication between nodes.
To run the code with MPI, compile it (e.g. using `cargo build -r`) and then execute the binary found in the `target/release`
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

## Tips
- For max performance:
    - Use `export RUSTFLAGS='-Ctarget-cpu=native'` when building (assuming the login nodes have the same hardware as the compute nodes)
    - Add the following to your `Cargo.toml`:
        ```toml
        [profile.release]
        codegen-units = 1 # don't split crate into multiple compilation units
        lto = true # enable link time optimization
        ```
- There might be problems with finding or loading dependencies if they come from modules.
    - You can write a custom [build script](https://doc.rust-lang.org/cargo/reference/build-scripts.html) to specify further linker args.
- MPI might not terminate if one of the MPI ranks panics, making the job hang until it hits the time limit.
    - Use `panic = "abort"` in the `Cargo.toml` file (see [here](https://doc.rust-lang.org/book/ch09-01-unrecoverable-errors-with-panic.html)). The library doesn't rely on unwinding panics, so aborting is fine.
- Use `export RUST_BACKTRACE=full` to get a stacktrace in case of errors