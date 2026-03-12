# TNC Guide

This guide acts as a starting point to get familiar with the library and its features.
It is intended to be more abstract than the [documentation](https://qc-tum.github.io/TNC/doc), while providing more info and background than the [examples](https://github.com/qc-tum/TNC/tree/main/tnc/examples).
If anything is missing, feel free to open an [issue](https://github.com/qc-tum/TNC/issues) on Github.

## About
TNC is a library for general tensor network contraction, with a focus on classical simulation of quantum circuits.

It supports the local contraction of tensor networks as well as parallel contraction on distributed-memory systems (HPC).
To this end, we employ partitioning of tensor networks to contract the individual partitions in parallel on different compute nodes.
Different methods for optimizing a partitioning before contraction are available.

The library allows to build and import tensor networks and offers methods to find contraction paths.