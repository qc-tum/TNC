{{#include links.md}}

# Pathfinding and Contraction

## Contraction Paths
To contract a tensor network, we need to specify the order in which the contractions should appear.
This is passed as a list of pairs of tensor indices.
There are different formats / interpretions of contraction paths; this library uses what we call the "replace left" format:

| Format       | Action for a single contraction (i, j)                                | Example for 4 tensors                         |
|--------------|-----------------------------------------------------------------------|-----------------------------------------------|
| SSA          | Contract tensor i and j, then append resulting tensor at end          | `[(0, 2), (1, 3), (4, 5)]`, result at index 6 |
| opt-einsum   | Pop tensors i and j, then contract and append resulting tensor at end | `[(0, 2), (0, 2), (0, 1)]`, result at index 0 |
| Replace left | Contract tensor i and j, then replace i by the resulting tensor       | `[(0, 2), (1, 3), (0, 1)]`, result at index 0 |

The rationale behind this format is that it can operate in-place without modifying the size of the tensors list, hence avoiding moving of elements and reallocations.

## Hierarchical Paths
Since tensor networks can be nested to form a tree-like structure (see [Tensor Structure](./tensor_structure.md)), contraction paths can likewise be nested, specifying sub-paths for sub-networks.
A contraction path hence specifies the contraction paths for all composite children of a composite tensors, and a top-level path to use after all children have been contracted to single tensors.
An abstract example could look like this: `[{0: [(0, 1), (0, 2)], 2: [(0, 1)]}, (2, 1), (0, 2)]`.
This specifies the paths to contract composite tensors `0` and `2`, followed by the top-level path to contract the resulting tensor networks.
Note that the tensor numbering is not globally unique: In every composite tensor, we start labeling from 0 again.

## Pathfinding
To find contraction paths for a given tensor network, the library offers various contraction path finders.
The best choice is usually to use [`Cotengrust`], which integrates a Rust port of path finders from the Python library *cotengra*.
It has three variants specified by [`OptMethod`].

## Contraction
To contract the tensor network locally, just use [`contract_tensor_network`] and pass in the tensor network and a contraction path.