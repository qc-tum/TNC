---
title: 'TNC: Distributed Tensor Network Contractions in Rust'
tags:
  - Rust
  - tensor networks
  - quantum computing
  - high-performance computing
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

TNC is a library for the efficient contraction of general tensor networks, focusing on a distributed-memory setting, as commonly found in high-performance computing (HPC) centers.
To this end, we employ partitioning techniques to contract a tensor network in parallel on multiple nodes.
TNC is written in Rust, allowing for high performance while guaranteeing memory safety.
While the library is focused on the classical simulation of quantum circuits, it can likewise be used to contract general tensor networks.

# Statement of need

With the rapid advancements in quantum computing, ever larger quantum circuits are explored which require more computing resources.
Since real quantum hardware is often not yet available and has to fight problems such as high error rates, research on efficient classical simulation methods is crucial to bridge this gap.
Tensor networks have shown to be a viable tool for the classical simulation, disproving even two major claims of quantum advantage, i.e., the realization of an algorithm on real quantum hardware which would be infeasable to simulate on classical hardware [@Pan2022;@Patra2024].

However, existing libraries for tensor network contractions often only consider shared-memory parallelism [@Pfeifer2015], are not open-source [@Bayraktar2023], or are Python-based [@Gray2021] and hence lack performance.
Furthermore, parallelization is usually done by a technique called *slicing* which, however, incurs computational overhead.

TNC allows the efficient contraction of tensor networks and hence enables the classical simulation of large quantum circuits on HPC systems or a single node.
Quantum circuits can either be constructed using the library API or imported from code in the common OpenQASM2 language [@Cross2017].
From the circuit, tensor networks can be constructed that compute the expectation value, the amplitudes to specific measurement outcomes or the full statevector.
Furthermore, random tensors networks can be created as well, e.g., in the form of the original Sycamore experiment [@Arute2019].
For contraction, tensor networks are partitioned based on the number of available MPI ranks.
The initial partitioning is done using the multi-graph partitioning library KaHyPar [@Andre2018].
Since the resulting partitionings are not optimal in terms of the computational cost required to contract them, our library features different algorithms to improve this initial partitioning, such as greedy rebalancing, simulated annealing and a genetic algorithm.
Contraction paths for the partitions which dictate the order of contractions are found by methods from the cotengra library [@Gray2021].
The inidivudal contractions are then performed using MKL for matrix-matrix multiplications and the hptt library [@Springer2017] for data transposition.

In a recent publication, we showed that partitionings optimized with the simulated annealing method have contraction costs that are competitive with state-of-the-art methods [@Geiger2025].


# Acknowledgements

We acknowledge the support and funding received for the MUNIQC-SC initiative under funding number 13N16191 from the VDI technology center as part of the German BMBF program.

# References