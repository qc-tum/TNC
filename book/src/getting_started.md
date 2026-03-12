# Getting Started

TNC is not yet on [crates.io](https://crates.io/).
Instead, it can be added to your project by running
```bash
cargo add --git https://github.com/qc-tum/TNC tnc
```
or by directly modifying your `Cargo.toml` file to include
```toml
[dependencies]
tnc = { git = "https://github.com/qc-tum/TNC.git" }
```

## Features
There are no default features activated.
The list of optional features to include is:
- `cotengra`: Enables Rust bindings to the tree annealing, tree reconfiguration and tree tempering methods of cotengra (for improving contraction paths)
- `mkl`: Uses the Intel Math Kernel Library (MKL) for performing tensor contractions. Otherwise, a pure Rust implementation is used

## System Requirements
As noted in the README, the library relies on system dependencies.
To install them, run
```bash
sudo apt install libhdf5-dev openmpi-bin libopenmpi-dev libboost-program-options-dev
```
Furthermore, the library relies on C++ libraries that are being built when building the library.
For this, `cmake` and a C++ compiler are required on the system.

Optionally, to use the `cotengra` feature, Python must be installed and the following Python packages must be installed:
```bash
pip install cotengra kahypar optuna
```