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
To this end, we employ partitioning techniques to contract a tensor network in parallel with similar time-to-solution.
TBD is written in Rust, allowing for high-performance while guaranteeing memory safety.
The library can for instance be used to classically simulate quantum circuits.

# Statement of need

With the rapid advancements in quantum computing, ever larger quantum circuits are explored which require more computing resources.
Since real quantum hardware is often not yet available and has to fight problems such as high error rates, research on efficient classical simulation methods is crucial to bridge this gap.
Tensor networks have shown to be a viable tool for the classical simulation, disproving even two major claims of quantum advantage (the realization of an algorithm on real quantum hardware which would be infeasable to simulate on classical hardware) [@Pan2022;@Patra2024].

However, existing libraries for tensor network contractions either only consider shared-memory parallelism (TODO: example), are not open-source (TODO: cuTensor), or are unmaintained (TODO: example).

TBD allows the efficient contraction of large tensor networks and hence enables the classical simulation of large quantum circuits.
The library can construct tensor networks from quantum circuits given in the common OpenQASM 2 language [@Cross2017].
For contraction, tensor networks are partitioned based on the number of available MPI ranks.
The partitioning is done using KaHyPar [@Andre2018].
The library features different algorithms to improve this initial partitioning, for instance using a simulated annealing approach, for faster time-to-solution.
Contraction paths for the partitions which dictate the order of contractions are found by methods from cotengra [@Gray2021].
The contraction is done using MKL for matrix-matrix multiplications and hptt [@Springer2017] for data transposition.


# Acknowledgements

The research is part of the Munich Quantum Valley (MQV), which is supported by the Bavarian state government with funds from the Hightech Agenda Bayern Plus. Moreover, this project is also supported by the German Federal Ministry of Education and Research(BMBF) through the MUNIQC-SC project.

# References