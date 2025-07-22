#![feature(map_try_insert)]
#![feature(assert_matches)]
#![feature(binary_heap_into_iter_sorted)]

extern crate jemallocator;

#[global_allocator]
static GLOBAL: jemallocator::Jemalloc = jemallocator::Jemalloc;

pub mod contractionpath;
pub mod gates;
pub mod io;
pub mod mpi;
pub mod networks;
pub mod qasm;
pub mod random;
pub mod tensornetwork;
pub mod types;
mod utils;
