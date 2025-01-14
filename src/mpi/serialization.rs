use rustc_hash::FxHashMap;

use crate::{
    mpi::mpi_types::MessageBinaryBlob,
    tensornetwork::{tensor::Tensor, tensordata::TensorData},
    types::EdgeIndex,
};

/// Serializes data to a byte array.
pub fn serialize<S>(value: &S) -> Vec<u8>
where
    S: serde::Serialize,
{
    bincode::serialize(value).unwrap()
}

/// Serializes data into a writer.
pub fn serialize_into<W, S>(writer: W, value: &S)
where
    W: std::io::Write,
    S: serde::Serialize,
{
    bincode::serialize_into(writer, value).unwrap();
}

/// Returns the serialized size of the data (i.e., the number of bytes).
pub fn serialized_size<S>(value: &S) -> u64
where
    S: serde::Serialize,
{
    bincode::serialized_size(value).unwrap()
}

/// Deserializes data from a byte array.
pub fn deserialize<D>(data: &[u8]) -> D
where
    D: serde::de::DeserializeOwned,
{
    bincode::deserialize(data).unwrap()
}

/// Deserializes data from a reader.
pub fn deserialize_from<R, D>(reader: R) -> D
where
    R: std::io::Read,
    D: serde::de::DeserializeOwned,
{
    bincode::deserialize_from(reader).unwrap()
}

/// Gets the serialized size of a tensor.
fn serialized_tensor_size(tensor: &Tensor) -> u64 {
    // Get own size
    let mut total_size = serialized_size(&tensor.legs());
    total_size += serialized_size(&tensor.tensor_data());

    // Add size for children count
    let num_children = tensor.tensors().len();
    total_size += serialized_size(&num_children);

    // Add size for children
    for child in tensor.tensors() {
        assert!(
            child.is_leaf(),
            "Tensor serialization only supports one level of nesting"
        );
        total_size += serialized_size(&child.legs());
        total_size += serialized_size(&child.tensor_data());
    }
    total_size
}

/// Serializes a tensor into a byte array.
fn serialize_tensor_inner(writer: &mut &mut [u8], tensor: &Tensor) {
    // Serialize the tensor itself
    serialize_into(&mut *writer, &tensor.legs());
    serialize_into(&mut *writer, &tensor.tensor_data());

    // Serialize the child count
    let num_children = tensor.tensors().len();
    serialize_into(&mut *writer, &num_children);

    // Serialize the child tensors
    for child in tensor.tensors() {
        assert!(
            child.is_leaf(),
            "Tensor serialization only supports one level of nesting"
        );
        serialize_into(&mut *writer, &child.legs());
        serialize_into(&mut *writer, &child.tensor_data());
    }
}

/// Serializes `tensor` (and its child tensors if any) into a vector of binary blobs.
///
/// MPI uses an `i32` to store the number of elements in a buffer. This means we can
/// send at most `i32::MAX * sizeof(datatype)`. Hence, if we used a `Vec<u8>`, we
/// could send at most `i32::MAX` bytes (~2 GB) which is not enough. Instead, we
/// interpret the byte arrays as arrays of an artificial, larger data type, which
/// allows us to send more bytes in total.
pub fn serialize_tensor(tensor: &Tensor) -> Vec<MessageBinaryBlob> {
    // Get the total message size in bytes
    let total_size = serialized_tensor_size(tensor);
    let total_size: usize = total_size.try_into().unwrap();

    // Allocate a buffer of blobs
    let element_size = std::mem::size_of::<MessageBinaryBlob>();
    let elements = total_size.div_ceil(element_size);
    let mut buffer = Vec::<MessageBinaryBlob>::with_capacity(elements);

    // Get a bytes view of the buffer
    let mut write_view = unsafe {
        std::slice::from_raw_parts_mut(
            buffer.as_mut_ptr().cast::<u8>(),
            buffer.capacity() * element_size,
        )
    };

    // Serialize legs and data into the buffer
    serialize_tensor_inner(&mut write_view, tensor);

    // Update the buffer length
    unsafe { buffer.set_len(buffer.capacity()) };

    buffer
}

/// Deserializes a tensor from a byte array.
fn deserialize_tensor_inner(
    reader: &mut &[u8],
    bond_dims: Option<&FxHashMap<EdgeIndex, u64>>,
) -> Tensor {
    let mut tensor = deserialize_leaf_tensor_inner(reader);
    let num_children: usize = deserialize_from(&mut *reader);
    for _ in 0..num_children {
        let child = deserialize_leaf_tensor_inner(&mut *reader);
        tensor.push_tensor(child, bond_dims);
    }
    tensor
}

/// Deserializes a leaf tensor from a byte array.
fn deserialize_leaf_tensor_inner(reader: &mut &[u8]) -> Tensor {
    // Deserialize the legs and tensor data
    let legs: Vec<EdgeIndex> = deserialize_from(&mut *reader);
    let tensor_data: TensorData = deserialize_from(&mut *reader);

    // Create the tensor
    let mut tensor = Tensor::new(legs);
    tensor.set_tensor_data(tensor_data);

    tensor
}

/// Deserializes a tensor from a array of binary blobs. See [`serialize_tensor`] for
/// more info. Requires `bond_dims` for building the composite tensor.
pub fn deserialize_tensor(
    data: &[MessageBinaryBlob],
    bond_dims: Option<&FxHashMap<EdgeIndex, u64>>,
) -> Tensor {
    // Get a bytes view of the buffer
    let size_in_bytes = std::mem::size_of_val(data);
    let mut read_buffer =
        unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), size_in_bytes) };

    // Deserialize the tensor
    deserialize_tensor_inner(&mut read_buffer, bond_dims)
}

#[cfg(test)]
mod tests {
    use rustc_hash::FxHashMap;

    use crate::{
        mpi::serialization::{deserialize_tensor, serialize_tensor},
        tensornetwork::tensor::Tensor,
    };

    #[test]
    fn test_serialize_deserialize_tensor_roundtrip() {
        let bond_dims = FxHashMap::from_iter([(1, 2), (2, 2), (3, 2), (4, 2), (5, 2)]);
        let mut ta = Tensor::default();
        let t2 = Tensor::new(vec![1, 2, 3]);
        let t3 = Tensor::new(vec![2, 3, 4]);
        let t4 = Tensor::new(vec![4, 5]);
        ta.push_tensors(vec![t2, t3, t4], Some(&bond_dims));
        let serialized = serialize_tensor(&ta);
        let deserialized = deserialize_tensor(&serialized, Some(&bond_dims));
        assert!(Tensor::approx_eq(&ta, &deserialized, 1e-10));
    }
}
