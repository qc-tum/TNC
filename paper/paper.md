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
  - name: Christian B. Mendl
    orcid: 0000-0002-6386-0230
    affiliation: '1, 2'
affiliations:
  - name: School for Computation, Information and Technology, Technical University of Munich, Germany
    index: 1
  - name: Institute for Advanced Study, Technical University of Munich, Germany
    index: 2
date: 9 October 2025
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

However, existing libraries for tensor network contractions often only consider shared-memory parallelism [@Pfeifer2015], are not open-source [@Bayraktar2023], or are written in Python [@Gray2021] and hence less suited for HPC.
Furthermore, parallelization is usually done by a technique called *slicing*, which incurs computational overhead.
TNC addresses these limitations by targeting distributed-memory systems, being fully open-source, implemented in Rust, and employing partitioning rather than slicing for parallelization.

# Features

The user can construct quantum circuits using the library's API or import them from code written in the widely-used OpenQASM2 language [@Cross2017].
From a given circuit, the user can then generate tensor networks that compute the expectation value, the amplitudes to specific measurement outcomes, or the full state vector.
Furthermore, random tensor networks can be created as well, e.g., in the form of the original Sycamore experiment [@Arute2019].

For contraction, the library partitions tensor networks based on the number of available MPI ranks.
It obtains an initial partitioning using the multi-graph partitioning library KaHyPar [@Andre2018].
Since the resulting partitionings are often not optimal regarding the computational cost required to contract them, our library features different algorithms to improve this initial partitioning, such as greedy rebalancing, simulated annealing, and a genetic algorithm.

The library can find contraction paths, which specify the order of contraction operations, using different methods from the cotengra library [@Gray2021].
It then performs the individual contractions using MKL for matrix-matrix multiplications and the hptt library [@Springer2017] for data transposition.

In a recent publication, we showed that partitionings optimized with the simulated annealing method have contraction costs that are competitive with state-of-the-art methods [@Geiger2025].

# Acknowledgements

We acknowledge the funding received for the MUNIQC-SC initiative under funding number 13N16191 from the VDI technology center as part of the German BMBF program.
The work is also supported by the Bavarian state government via the BayQS project with funds from the Hightech Agenda Bayern, and via the Munich Quantum Valley, section K7, with funds from the Hightech Agenda Bayern Plus.

# References