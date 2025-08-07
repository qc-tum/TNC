use bincode::{
    config::{self, Configuration},
    enc::write::SizeWriter,
    serde::{
        decode_from_slice, decode_from_std_read, encode_into_std_write, encode_into_writer,
        encode_to_vec,
    },
};

use crate::{mpi::mpi_types::MessageBinaryBlob, tensornetwork::tensor::Tensor};

static BINCODE_CONFIG: Configuration = config::standard();

/// Serializes data to a byte array.
pub fn serialize<S>(value: &S) -> Vec<u8>
where
    S: serde::Serialize,
{
    encode_to_vec(value, BINCODE_CONFIG).unwrap()
}

/// Serializes data into a writer.
pub fn serialize_into<W, S>(mut writer: W, value: &S)
where
    W: std::io::Write,
    S: serde::Serialize,
{
    encode_into_std_write(value, &mut writer, BINCODE_CONFIG).unwrap();
}

/// Returns the serialized size of the data (i.e., the number of bytes).
pub fn serialized_size<S>(value: &S) -> usize
where
    S: serde::Serialize,
{
    let mut size_writer = SizeWriter::default();
    encode_into_writer(value, &mut size_writer, BINCODE_CONFIG).unwrap();
    size_writer.bytes_written
}

/// Deserializes data from a byte array.
pub fn deserialize<D>(data: &[u8]) -> D
where
    D: serde::de::DeserializeOwned,
{
    decode_from_slice(data, BINCODE_CONFIG)
        .map(|(data, _size)| data)
        .unwrap()
}

/// Deserializes data from a reader.
pub fn deserialize_from<R, D>(mut reader: R) -> D
where
    R: std::io::Read,
    D: serde::de::DeserializeOwned,
{
    decode_from_std_read(&mut reader, BINCODE_CONFIG).unwrap()
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
    let total_size = serialized_size(tensor);
    let total_size: usize = total_size.try_into().unwrap();

    // Allocate a buffer of blobs
    let element_size = std::mem::size_of::<MessageBinaryBlob>();
    let elements = total_size.div_ceil(element_size);
    let mut buffer = Vec::<MessageBinaryBlob>::with_capacity(elements);

    // Get a bytes view of the buffer
    let write_view = unsafe {
        std::slice::from_raw_parts_mut(
            buffer.as_mut_ptr().cast::<u8>(),
            buffer.capacity() * element_size,
        )
    };

    // Serialize legs and data into the buffer
    serialize_into(write_view, tensor);

    // Update the buffer length
    unsafe { buffer.set_len(buffer.capacity()) };

    buffer
}

/// Deserializes a tensor from a array of binary blobs. See [`serialize_tensor`] for
/// more info. Requires `bond_dims` for building the composite tensor.
pub fn deserialize_tensor(data: &[MessageBinaryBlob]) -> Tensor {
    // Get a bytes view of the buffer
    let size_in_bytes = std::mem::size_of_val(data);
    let read_buffer =
        unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), size_in_bytes) };

    // Deserialize the tensor
    deserialize_from(read_buffer)
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
        let t2 = Tensor::new_from_map(vec![1, 2, 3], &bond_dims);
        let t3 = Tensor::new_from_map(vec![2, 3, 4], &bond_dims);
        let t4 = Tensor::new_from_map(vec![4, 5], &bond_dims);
        let ta = Tensor::new_composite(vec![t2, t3, t4]);
        let serialized = serialize_tensor(&ta);
        let deserialized = deserialize_tensor(&serialized);
        assert!(Tensor::approx_eq(&ta, &deserialized, 1e-10));
    }
}
