#![allow(dead_code)]
#![feature(const_trait_impl)]
#![feature(cmp_minmax)]
#![feature(map_many_mut)]
#![feature(map_try_insert)]
#![feature(slice_pattern)]
#![feature(stmt_expr_attributes)]
#![feature(vec_into_raw_parts)]
#![feature(pointer_is_aligned_to)]

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
