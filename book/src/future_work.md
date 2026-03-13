# Future Work

There are a few things that would be nice to have for the library.
The following list of features and ideas is sorted somewhat by priority.
We can not guarantee that these points will ever be implemented by us, but if you feel like working on something, check out the Contribution guide in the repository.
1. **Tensor Network Optimization**:
It is common to perform optimizations before contracting tensor networks.
For instance, by performing trivial contractions in advance, the search space for the pathfinding can be reduced.
2. **Slicing**:
Currently, the library can face memory limitations with large tensor networks.
We want to use slicing to reduce the required memory in such cases, at the cost of having to do additional computations.
3. **GPU Support**:
GPUs are undoubtedly superior to CPU at matrix calculations.
Having support for GPUs could speed up contractions tremendously.
4. **Support for shared-memory parallelism**:
While we already use multithreading for single contractions, this will only be worth it for large tensors.
It would be nice to utilize the full system at all times, even when the individual tensor contractions are not so large.
5. **Replacing the C++ dependencies**:
The installation of the library is cumbersome, particularily due to the C++ dependencies.
If we found pure Rust replacements, the installation would be easier, build times probably faster, and we could likely support more plattforms.
6. **More tensor operations**:
Having e.g. options for singular value decomposition or other operations could make the library more useful for other use cases.