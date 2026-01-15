//! # Constructing tensor networks
//!
//! There are multiple ways to construct tensors and tensor networks.
//!
//! ## Quantum
//!
//! ### OpenQASM2 code
//! If the goal is to clasically simulate quantum circuits, one can directly load
//! OpenQASM2 code and construct a [`Circuit`] out of it using [`import_qasm`].
//!
//! <div class="warning">
//! This library implements many standard gates and when it encounters one in the
//! QASM code, it will not look for a gate definition; only when it doesn't know the
//! gate, it will decompose the gate using an earlier gate definition in the QASM
//! code.
//! </div>
//!
//! From the circuit, we can then construct different tensor networks, depending on
//! what we want to compute:
//! - [`into_amplitude_network`] creates a tensor network that computes the
//!   amplitude(s) to one or more states.
//! - [`into_statevector_network`] creates a tensor network that computes the full
//!   statevector
//! - [`into_expectation_value_network`]: creates a tensor network that computes the
//!   expectation value of the circuit with respect to `Z` observables on each qubit
//!
//! [`into_amplitude_network`]: Circuit::into_amplitude_network
//! [`into_statevector_network`]: Circuit::into_statevector_network
//! [`into_expectation_value_network`]: Circuit::into_expectation_value_network
//!
//! ### Circuit builder
//! Similar to importing QASM2 code, the [`Circuit`] struct can also directly be used
//! to construct tensor networks that simulate quantum circuits.
//!
//! ### Sycamore circuit
//! There are special methods to construct tensor networks corresponding to the
//! quantum circuits of the Sycamore experiment ([Quantum supremacy using a
//! programmable superconducting processor](
//! https://www.nature.com/articles/s41586-019-1666-5) (Arute et al.)). See the
//! [`sycamore_circuit`] method.
//!
//! ## HDF5 files
//! Tensors and tensor networks can also be saved and loaded from HDF5 files, see the
//! functions in [`hdf5`]. The structure of the files is:
//! ```text
//! [Group name="tensors"]
//!     [Dataset name="tensorA" datatype=double complex tensor]
//!         [Attribute name="bids" datatype=int list]
//!     [Dataset name="tensorB" datatype=double complex tensor]
//!         [Attribute name="bids" datatype=int list]
//!     ...
//! ```
//! where `bids` are the leg IDs.
//!
//! ## General tensor networks
//! The [`Tensor`] struct can be used to directly construct arbitrary tensors and
//! tensor networks. Tensors are created from a list of leg IDs and the corresponding
//! dimensions of these legs. Connected tensors are identified by having at least one
//! leg ID in common. The corresponding bond dimensions have to match.
//!
//! Tensors without data can already be used for e.g. finding a contraction path, but
//! if you want to actually contract a tensor network, the tensors need data. For
//! this, there is the [`set_tensor_data`] method which takes a variant of
//! [`TensorData`].
//!
//! [`set_tensor_data`]: Tensor::set_tensor_data
//!
//! A normal tensor network is a list of tensors. However, this library also supports
//! hierarchical tensor network structures, which are detailed in another tutorial.
#![allow(unused_imports)]
use crate::builders::circuit_builder::Circuit;
use crate::builders::sycamore_circuit::sycamore_circuit;
use crate::hdf5;
use crate::qasm::import_qasm;
use crate::tensornetwork::tensor::Tensor;
use crate::tensornetwork::tensordata::TensorData;

pub use crate::_tutorial as table_of_contents;
