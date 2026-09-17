# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Function `contract_size_tensors_bytes` to use as memory estimator. Computes the memory requirement for a contraction.
- Function `into_tensor_data` to get owned data from a tensor

### Changed
- Use TBLIS instead of HPTT + MKL for contraction. This means contractions are faster and often require only half as much memory. Also, build time decreased.
- Use ndarray instead of own implementation for tensors. This means more features (slicing, arbitrary memory layout, ...) and interoperability.
- Implement `approx` instead of `float-cmp` for tensors and tensor data
- `find_path` now takes the tensor network as argument and returns a result struct
- `into_expectation_network` now takes a custom observable

### Fixed
- QASM2 import now uses float division and power instead of integer division and bitxor
- Fixed wrong adjoints of single qubit gates in `random_circuit_with_(set_)observable`

### Removed
- Function `contract_size_tensors_exact` (since there is no explicit transpose, the normal contraction size estimate is sufficient)
- Functions `get_best_path`, `get_best_flops`, ... from `FindPath` trait. They exist on the struct returned by `find_path` now
- Greedy balancing (inferior to other methods like simulated annealing)

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