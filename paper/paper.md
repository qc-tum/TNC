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

Since real quantum computers continue to have high error rates, classical error-free simulation of circuits is essential for verification, benchmarking, and algorithm design.
A commonly used tool for the simulation are tensor networks, which represent a quantum circuit as network of multi-dimensional arrays that must be multiplied ("contracted") to compute the result of the circuit.
TNC is a Rust-based library for efficiently contracting large tensor networks.
Designed for distributed-memory environments common in high-performance computing (HPC), TNC partitions tensor networks to enable parallel contraction across multiple compute nodes.
While optimized for quantum circuit simulation, TNC is general enough to contract arbitrary tensor networks, combining high performance with Rust's memory safety guarantees.

# Statement of need

With the rapid advancements in quantum computing, ever larger quantum circuits are being explored, which require more computing resources.
Real quantum hardware is often not yet available, however, has limited qubit counts, and has high error rates due to imperfect gates and noise induced by the environment.
Hence, research on efficient classical simulation methods is crucial to bridge this gap.
Tensor networks have been shown to be a viable tool for classical simulation, even disproving two major claims of quantum advantage, i.e., the realization of an algorithm on real quantum hardware which would be infeasible to simulate on classical hardware [@Pan2022;@Patra2024].
Apart from quantum simulation, they are also used in other areas, such as machine learning, quantum optimization, quantum chemistry, and quantum error correction [@Garcia2024].

However, existing libraries for tensor network contractions often only consider shared-memory parallelism [@Pfeifer2015], are not open-source [@Bayraktar2023], or are written in Python [@QTensor] and hence less suited for HPC.
Furthermore, parallelization is usually done by a technique called *slicing*, which incurs computational overhead.
Popular libraries using slicing include ExaTN [@ExaTN] and Jet [@Jet].
TNC addresses these limitations by targeting distributed-memory systems, being fully open-source, implemented in Rust, and employing partitioning rather than slicing for parallelization.
The use of Rust further provides strong guarantees of memory safety.

# Features

The user can construct quantum circuits using the library's API or import them from code written in the widely-used OpenQASM2 language [@Cross2017].
From a given circuit, the user can then generate tensor networks that compute the expectation value, the amplitudes to specific measurement outcomes, or the full state vector.
Furthermore, random tensor networks can be created as well, for instance, in the form of the original Sycamore experiment [@Arute2019].
The library supports serialization of tensors and tensor networks to and from HDF5 files.

Tensors in TNC can be defined hierarchically, enabling tensor networks in which vertices may represent either raw tensors or nested subnetworks.
This hierarchical representation naturally expresses partitioning‑based parallelization and supports multiple parallelization levels.

For contraction, the library partitions tensor networks based on the number of available compute nodes.
It obtains an initial partitioning using the hypergraph partitioning library KaHyPar [@Andre2018].
However, these initial partitionings are often not optimal in terms of contraction cost.
Therefore, our library features different algorithms to improve an initial partitioning, such as greedy rebalancing, simulated annealing, and a genetic algorithm.
As a cost function, we use a parallelism-aware computation cost metric.
The final partitioning thus achieves better time-to-solution by better load balancing between compute nodes.
In a recent publication, we showed that partitionings optimized with the simulated annealing method have contraction costs that are competitive with state-of-the-art methods [@Geiger2025].

The library can find contraction paths, which specify the order of contraction operations, using different methods from the cotengra library [@Gray2021].
It then performs the individual contractions using matrix-matrix multiplications via MKL [@MKL] or a pure Rust backend, with tensor transpositions performed using the hptt library [@Springer2017].

To execute distributed contractions, TNC performs the contractions of different partitions in parallel across compute nodes.
Communication and data exchange between nodes are coordinated using MPI, enabling scalable execution on distributed‑memory HPC systems.

# Acknowledgements

We acknowledge the funding received for the MUNIQC-SC initiative under funding number 13N16191 from the VDI technology center as part of the German BMBF program.
The work is also supported by the Bavarian state government via the BayQS project with funds from the Hightech Agenda Bayern, and via the Munich Quantum Valley, section K7, with funds from the Hightech Agenda Bayern Plus.

# References