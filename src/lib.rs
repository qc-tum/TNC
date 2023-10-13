#![feature(stmt_expr_attributes)]
#![feature(entry_insert)]
#![allow(dead_code)]
#![feature(slice_pattern)]
#![feature(map_many_mut)]

#[cfg(feature = "hdf5")]
pub mod io;

pub mod contractionpath;
pub mod qasm;
pub mod random;
pub mod tensornetwork;
