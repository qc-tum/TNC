# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.1] - 2026-05-26

### Fixed
- A wrong leg order bug in `CircuitBuilder` that could lead to wrong order of statevectors

## [1.0.0] - 2026-05-08

Initial release, including
- QASM2 importing
- Amplitude, statevector and expectation value computation
- Tensor network contraction
- Tensor network partitioning
- Rebalancing of partitioned tensor network for better time-to-solution
- Distributed contraction of partitioned tensor networks via MPI

[unreleased]: https://github.com/qc-tum/TNC/compare/v1.0.1...HEAD
[1.0.1]: https://github.com/qc-tum/TNC/releases/tag/v1.0.1
[1.0.0]: https://github.com/qc-tum/TNC/releases/tag/v1.0.0