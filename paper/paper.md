---
title: 'TNC: Distributed Tensor Network Contractions in Rust'
tags:
  - tensor networks
  - quantum computing
  - high-performance computing
  - Rust
authors:
  - name: Manuel Geiger
    orcid: 0000-0003-3514-8657
    affiliation: '1'
  - name: Qunsheng Huang
    orcid: 0000-0002-1289-6559
    affiliation: '1'
affiliations:
  - name: School for Computation, Information and Technology, Technical University of Munich, Germany
    index: 1
date: 8 September 2025
bibliography: paper.bib
---

# Summary

TNC is a Rust-based library for efficient contraction of tensor networks, a key technique for classical simulation of quantum circuits.
Designed for distributed-memory environments common in high-performance computing (HPC), TNC partitions tensor networks to enable parallel contraction across multiple nodes.
While optimized for quantum circuit simulation, TNC is general enough to handle arbitrary tensor networks, combining high performance with Rust's memory safety guarantees.

# Statement of need

With the rapid advancements in quantum computing, ever larger quantum circuits are being explored, which require more computing resources.
Since real quantum hardware is often not yet available and has to fight problems such as high error rates, research on efficient classical simulation methods is crucial to bridge this gap.
Tensor networks have been shown to be a viable tool for classical simulation, disproving even two major claims of quantum advantage, i.e., the realization of an algorithm on real quantum hardware which would be infeasible to simulate on classical hardware [@Pan2022;@Patra2024].

However, existing libraries for tensor network contractions often only consider shared-memory parallelism [@Pfeifer2015], are not open-source [@Bayraktar2023], or are Python-based [@Gray2021] and hence limited in performance.
Furthermore, parallelization is usually done by a technique called *slicing*, which incurs computational overhead.
TNC addresses these limitations by targeting distributed-memory systems, being fully open-source, implemented in Rust, and employing partitioning rather than slicing for parallelization.

# Features

Quantum circuits can be constructed using the library's API or imported from code written in the widely-used OpenQASM2 language [@Cross2017].
From a given circuit, tensor networks can be constructed that compute the expectation value, the amplitudes to specific measurement outcomes, or the full state vector.
Furthermore, random tensor networks can be created as well, e.g., in the form of the original Sycamore experiment [@Arute2019].

For contraction, tensor networks are partitioned based on the number of available MPI ranks.
The initial partitioning is done using the multi-graph partitioning library KaHyPar [@Andre2018].
Since the resulting partitionings are often not optimal in terms of the computational cost required to contract them, our library features different algorithms to improve this initial partitioning, such as greedy rebalancing, simulated annealing, and a genetic algorithm.

Contraction paths, which specify the order of contraction operations, are found using methods from the cotengra library [@Gray2021].
The individual contractions are then performed using MKL for matrix-matrix multiplications and the hptt library [@Springer2017] for data transposition.

In a recent publication, we showed that partitionings optimized with the simulated annealing method have contraction costs that are competitive with state-of-the-art methods [@Geiger2025].

# Acknowledgements

We acknowledge the support and funding received for the MUNIQC-SC initiative under funding number 13N16191 from the VDI technology center as part of the German BMBF program.

# References