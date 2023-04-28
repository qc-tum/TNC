# HDF5 Support

This library supports the HDF5 output of the `qib` library. This follows the structure:

tensors/
    tensor: n-dimensional dataset 
        attrs: 
            - bids
            - tids

There is a single `tensors/` group containing multiple tensor datasets. Each `tensor` is a flattened tensor with dimensions `shape`. The `tid` is the unique positive integer used to identify each tensor, with the output tensor, identified by `-1`, containing output bond dimensions and no tensor data. The `bids` are a list of integers corresponding to the bond ids of in each tensor. 