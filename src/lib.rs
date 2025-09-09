//! This library is designed to perform tensor network contractions, using partitioning of the network as parallelization strategy.
//! To this end, it ships multiple methods to partition a tensor network for lowest contraction cost, for example, based on simulated annealing.
//! The partitionings can then be contracted in parallel on a distributed-memory system, as common in high-performance computing.
//! Local contractions on a single system are also possible.

extern crate jemallocator;

#[global_allocator]
static GLOBAL: jemallocator::Jemalloc = jemallocator::Jemalloc;

pub mod builders;
pub mod contractionpath;
pub mod gates;
pub mod io;
pub mod mpi;
pub mod qasm;
pub mod tensornetwork;
pub mod types;
mod utils;
