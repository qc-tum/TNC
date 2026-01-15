//! # Pathfinding and Contraction
//!
//! ## Pathfinding
//! To contract a tensor network, we need to specify the order in which the
//! contractions should appear. This is passed as a list of pairs of tensor indices.
//! There are different formats / interpretions of contraction paths; this library
//! uses what we call the "replace left" format:
//!
//! | Format       | Action for a single contraction (i, j)                                | Example for 4 tensors                   |
//! |--------------|-----------------------------------------------------------------------|-----------------------------------------|
//! | SSA          | Contract tensor i and j, then append resulting tensor at end          | `[(0, 2), (1, 3), (4, 5)]`, result in 6 |
//! | opt-einsum   | Pop tensors i and j, then contract and append resulting tensor at end | `[(0, 2), (0, 2), (0, 1)]`, result in 0 |
//! | Replace left | Contract tensor i and j, then replace i by the resulting tensor       | `[(0, 2), (1, 3), (0, 1)]`, result in 0 |
//!
//! The rationale behind this format is that it can operate in-place without modifying the size of
//! the tensors list, hence avoiding moving of elements and reallocations.
//!
//! To find contraction paths for a given tensor network, the library offers various contraction
//! path finders. The best choice is usually to use [`Cotengrust`], which integrates a Rust port of
//! path finders from the Python library *cotengra*. It has three variants specified by
//! [`OptMethod`].
//!
//! ## Contraction
//! To contract the tensor network locally, just use [`contract_tensor_network`] and pass in the
//! tensor network and a contraction path.
#![allow(unused_imports)]
use crate::contractionpath::paths::cotengrust::{Cotengrust, OptMethod};
use crate::tensornetwork::contraction::contract_tensor_network;

pub use crate::_tutorial as table_of_contents;
