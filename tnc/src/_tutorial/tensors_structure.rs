//! # Structure of tensors
//!
//! In contrast to the common design of having tensors and tensor networks, this
//! library allows arbitrary nesting of tensors. This means that the children of
//! tensor networks can themselves be tensor networks again. We call tensors with
//! children "composite tensors" and those without children "leaf tensors".
//!
//! ## Composite tensors
//! Composite tensors are the equivalent to tensor networks, as they own a list of
//! tensors that are their children. However, these child tensors can also be
//! composite, allowing for a hierarchical tree structure. The outer legs of a
//! composite tensor can be obtained by the [`external_tensor`] method.
//!
//! ## Leaf tensors
//! Leaf tensors have legs with corresponding bond dimensions and can optionally have
//! data.
//!
//! ## Rationale
//! The idea for this recursive design is natural: When looking only at the outer
//! (i.e. open) legs of a tensor network, it can be seen as a plain tensor again. It
//! has legs with sizes and it has data (that is obtained by contracting the tensor
//! network).
//!
//! In addition, this format is useful for multi-level parallelization based on
//! partitioning the network: For example, the top level could be a composite tensor,
//! where each children is assigned to one compute node. Each children is also again
//! a composite tensor, where each children is assigned to one core. Finally, each of
//! those children is again a composite tensor that is an actual tensor network
//! (i.e., with leaf tensors as children).
//!
//! [`external_tensor`]: Tensor::external_tensor
//!
#![allow(unused_imports)]
use crate::tensornetwork::tensor::Tensor;

pub use crate::_tutorial as table_of_contents;
