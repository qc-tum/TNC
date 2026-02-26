//! # Partitioning
//!
//! To partition a tensor network, use [`find_partitioning`] and specify how many
//! partitions should be created. This will use the Hypergraph partitioner KaHyPar to
//! find roughly equally sized partitions with minimial cost for the legs between the
//! partitions.
//!
//! To actually get a partitioned tensor network, use
//! [`partition_tensor_network`] and pass the partitioning (you can of course also
//! provide your own). The partitioning just specifies for each tensor to which
//! partition it should belong to.
//!
//! This initial partitioning can be suboptimal for contraction, though. For this
//! reason, we provide multiple methods to iteratively refine the partitioning for
//! lower time-to-solution. The best method is simulated annealing with the
//! [`IntermediatePartitioningModel`]. For details on the method, see publication
//! [1].
//!
//! [1]: https://arxiv.org/abs/2507.20667
#![allow(unused_imports)]
use crate::contractionpath::repartitioning::simulated_annealing::IntermediatePartitioningModel;
use crate::tensornetwork::partitioning::{find_partitioning, partition_tensor_network};

pub use crate::_tutorial as table_of_contents;
