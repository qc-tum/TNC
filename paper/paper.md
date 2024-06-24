---
title: 'TBD: Efficient Tensor Network Contractions in Rust'
tags:
  - Rust
  - tensor networks
  - quantum computing
authors:
  - name: Manuel Geiger
    orcid: 0000-0003-3514-8657
    equal-contrib: true
    affiliation: '1'
  - name: Qunsheng Huang
    orcid: 0000-0002-1289-6559
    equal-contrib: true
    affiliation: '1'
affiliations:
  - name: School for Computation, Information and Technology, Technical University of Munich, Germany
    index: 1
date: 22.01.2024
bibliography: paper.bib
---

# Summary

TBD is a library for contracting large tensor networks written in Rust.
Its primary focus is on performance.
While it can run on a single node, we particularily target multi-node scenarios as found in supercompters.
To this end, we employ partitioning techniques to contract tensor networks in parallel.
Additionally, slicing of tensors allows to contract tensor networks that would otherwise not fit into memory, by trading for an increased computation effort.
The local contractions use the efficient BLAS routines and an efficient C++ library for transposition of data.
Rust as language of choice allows for high-performance code while guaranteeing memory safety.

# Statement of need

With the rapid advancements in quantum computing, ever larger quantum circuits are explored which require more resources.
Since real quantum hardware is often not yet available and has to fight problems such as high error rates, research on efficient classical simulation methods is crucial to bridge this gap.
Up to now, every claim of quantum advantage (the realization of an algorithm on real quantum hardware which would be infeasable to simulate on classical hardware) has been disproven [@Pan2022].
The main technique used for classical simulations are tensor networks.

TBD allows the efficient contraction of large tensor networks and hence enables the classical simulation of large quantum circuits. The library development was guided by benchmarks run on an actual supercomputer. Dedicated features for the use of the library for quantum computing were developed: An import of QASM2 files allow for easy construction of tensor networks corresponding to quantum circuits. Furthermore, the library allows hyperedges between tensors. These can be used for a more efficient representation of CNOT gates [@Gray2021].

# Acknowledgements

The research is part of the Munich Quantum Valley (MQV), which is supported by the Bavarian state government with funds from the Hightech Agenda Bayern Plus.

# References