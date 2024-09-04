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

TBD is a library for the efficient contraction of large tensor networks, focusing on a distributed-memory (i.e., multi-node HPC) setting.
To this end, we employ partitioning techniques to contract tensor networks in parallel with similar time-to-solution.
The local contractions use MKL matrix-matrix multiplications and the HPTT library for efficient transposition of data.
TBD is written in Rust, allowing for high-performance code while guaranteeing memory safety.

# Statement of need

With the rapid advancements in quantum computing, ever larger quantum circuits are explored which require more computing resources.
Since real quantum hardware is often not yet available and has to fight problems such as high error rates, research on efficient classical simulation methods is crucial to bridge this gap.
Tensor networks have shown to be a viable tool for the classical simulation, disproving even two major claims of quantum advantage (the realization of an algorithm on real quantum hardware which would be infeasable to simulate on classical hardware) [@Pan2022;@Patra2024].

TBD allows the efficient contraction of large tensor networks and hence enables the classical simulation of large quantum circuits. The library development was guided by benchmarks run on an actual supercomputer. Dedicated features for the use of the library for quantum computing were developed: An import of circuits given in the OpenQASM 2 language allows for easy construction of tensor networks corresponding to quantum circuits. Furthermore, the library allows hyperedges between tensors. These can be used for a more efficient representation of CNOT gates [@Gray2021].

# Acknowledgements

The research is part of the Munich Quantum Valley (MQV), which is supported by the Bavarian state government with funds from the Hightech Agenda Bayern Plus.

# References