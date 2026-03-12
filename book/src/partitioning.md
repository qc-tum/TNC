{{#include links.md}}

# Partitioning

The library uses partitiioning of tensor networks to parallelize the contraction.
For example, given the following tensor network

![A tensor network](assets/tensor_network.svg)

one can partition it into two separate tensor networks like this:

![A tensor network split into two networks, separated by a dashed line.](assets/partitioning.svg)

To partition a tensor network, use [`find_partitioning`] and specify how many partitions should be created.
This will use the Hypergraph partitioner KaHyPar to find roughly equally sized partitions with minimial cost for the legs between the partitions.

To actually get a partitioned tensor network, use [`partition_tensor_network`] and pass the partitioning (you can of course also provide your own).
The partitioning just specifies for each tensor to which partition it should belong to.

This initial partitioning can be suboptimal for contraction, though.
For this reason, we provide multiple methods to iteratively refine the partitioning for lower time-to-solution.
The best method is simulated annealing with the [`IntermediatePartitioningModel`].
For details on the method, see [Optimizing Tensor Network Partitioning using Simulated Annealing](https://arxiv.org/abs/2507.20667) (Geiger et al.).
