//! # High-Performance Computing (HPC)
//!
//! The library can run on multiple connected compute nodes such as in a HPC setting.
//! It uses MPI for communication between nodes. To run the code with MPI, compile it
//! (e.g. using `cargo build -r`) and then execute the binary found in the `target`
//! folder using an MPI launcher (such as `mpirun`). For example:
//! ```shell
//! cargo build -r --example basic_usage
//! mpirun -n 4 target/release/examples/basic_usage
//! ```
//!
//! ## Parallelization
//!
//! ### Distributed memory parallelism
//! The parallelization strategy currently is partitioning: Given a tensor network,
//! it can be partitioned into multiple networks using the [`find_partitioning`]
//! function which makes use of the hypergraph partitioning library KaHyPar. Then,
//! the partitioned tensor network can be distributed to individual nodes using
//! [`scatter_tensor_network`]. Each node can then indepently contract its part of
//! the tensor network. No communication is needed during this time. Finally, the
//! results are gathered in a parallel reduce operation, where the tensors are sent
//! between nodes according to the contraction path, contracted locally and sent
//! again, until the final contraction, which is guaranteed to happen on rank 0.
//!
//! ### Shared memory parallelism
//! Given the large tensor sizes that can occur during contraction, we do not
//! partition tensor networks further than the node level. Instead, we use the
//! avilable cores to parallelize the individual tensor tensor contractions.
//!
//! ### What about slicing?
//! Slicing is currently not supported, as it is not easy to combine it with
//! partitioning. We hope to implement it at a later point.
//!
//! ## Dealing with memory limits
//! Unfortunately, the high memory requirements of tensor network contraction are a
//! general problem. One thing to try is to use less or more partitions, as there can
//! be sweet spots -- more is not always better. In particular, the memory
//! requirements can already be computed theoretically (with the functions in
//! [`contraction_cost`]) before doing any actual run on a compute cluster. Other
//! than that, the library unfortunately currently lacks support for slicing, which
//! would allow trading compute time for lower memory requirements.
#![allow(unused_imports)]
use crate::contractionpath::contraction_cost;
use crate::mpi::communication::scatter_tensor_network;
use crate::tensornetwork::partitioning::find_partitioning;
use crate::tensornetwork::tensor::Tensor;

pub use crate::_tutorial as table_of_contents;
