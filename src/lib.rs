#![feature(stmt_expr_attributes)]
#![feature(entry_insert)]
#![allow(dead_code)]
#![feature(slice_pattern)]
#![feature(map_many_mut)]
#![feature(const_trait_impl)]

#[cfg(feature = "hdf5")]
pub mod io;

pub mod circuits;
pub mod contractionpath;
pub mod gates;
pub mod mpi;
pub mod qasm;
pub mod random;
pub mod tensornetwork;
pub mod types;
