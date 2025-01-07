use rustc_hash::FxHashMap;
use serde::{de::Visitor, ser::SerializeStruct, Deserialize, Serialize};

use crate::{
    tensornetwork::{tensor::Tensor, tensordata::TensorData},
    types::EdgeIndex,
};

const NAME: &'static str = "Tensor";
const FIELDS: &[&str] = &["legs", "data", "tensors"];

/// Serializes data to a byte array.
fn serialize<S>(value: &S) -> Vec<u8>
where
    S: serde::Serialize,
{
    bincode::serialize(value).unwrap()
}

/// Serializes data into a writer.
fn serialize_into<W, S>(writer: W, value: &S)
where
    W: std::io::Write,
    S: serde::Serialize,
{
    bincode::serialize_into(writer, value).unwrap();
}

/// Returns the serialized size of the data (i.e., the number of bytes).
fn serialized_size<S>(value: &S) -> u64
where
    S: serde::Serialize,
{
    bincode::serialized_size(value).unwrap()
}

/// Deserializes data from a byte array.
fn deserialize<D>(data: &[u8]) -> D
where
    D: serde::de::DeserializeOwned,
{
    bincode::deserialize(data).unwrap()
}

/// Deserializes data from a reader.
fn deserialize_from<R, D>(reader: R) -> D
where
    R: std::io::Read,
    D: serde::de::DeserializeOwned,
{
    bincode::deserialize_from(reader).unwrap()
}

impl Serialize for Tensor {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut state = serializer.serialize_struct(NAME, FIELDS.len())?;
        state.serialize_field("legs", self.legs())?;
        state.serialize_field("data", self.tensor_data())?;
        state.serialize_field("tensors", self.tensors())?;
        state.end()
    }
}

/// A helper struct for building a composite tensor. This is required as directly
/// deserializing a composite Tensor does not work when we don't have the bond
/// dimensions given. Currently, the bond dimensions are not serialized with the
/// tensor, and there is no way to provide them during deserialization.
pub(super) struct TensorBuilder {
    legs: Vec<EdgeIndex>,
    data: TensorData,
    children: Vec<TensorBuilder>,
}

impl TensorBuilder {
    /// Builds the composite tensor from the builder.
    pub fn build(self, bond_dims: &FxHashMap<EdgeIndex, u64>) -> Tensor {
        let mut tensor = Tensor::new(self.legs);
        if !self.children.is_empty() {
            tensor.push_tensors(
                self.children
                    .into_iter()
                    .map(|t| t.build(bond_dims))
                    .collect(),
                Some(bond_dims),
                None,
            );
        }
        tensor.set_tensor_data(self.data);
        tensor
    }
}

impl<'de> Deserialize<'de> for TensorBuilder {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct TensorVisitor;

        impl<'de> Visitor<'de> for TensorVisitor {
            type Value = TensorBuilder;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("struct ")?;
                formatter.write_str(NAME)
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: serde::de::SeqAccess<'de>,
            {
                let legs: Vec<EdgeIndex> = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(0, &self))?;
                let data: TensorData = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(1, &self))?;
                let tensors: Vec<TensorBuilder> = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(2, &self))?;

                let tensor_builder = TensorBuilder {
                    legs,
                    data,
                    children: tensors,
                };
                Ok(tensor_builder)
            }
        }

        deserializer.deserialize_struct(NAME, FIELDS, TensorVisitor)
    }
}

#[cfg(test)]
mod tests {
    use rustc_hash::FxHashMap;

    use crate::{
        mpi::serialization::{deserialize, serialize, TensorBuilder},
        tensornetwork::tensor::Tensor,
    };

    #[test]
    fn test_serialize_deserialize_tensor_roundtrip() {
        let bond_dims = FxHashMap::from_iter([(1, 2), (2, 2), (3, 2), (4, 2), (5, 2), (6, 2)]);
        let mut ta = Tensor::default();
        let t2 = Tensor::new(vec![1, 2, 3]);
        let t3 = Tensor::new(vec![2, 3, 4]);
        let t4 = Tensor::new(vec![4, 5]);
        ta.push_tensors(vec![t2, t3, t4], Some(&bond_dims), None);
        let mut tb = Tensor::default();
        let t5 = Tensor::new(vec![5, 6]);
        let t6 = Tensor::new(vec![6]);
        tb.push_tensors(vec![t5, t6], Some(&bond_dims), None);
        let mut tc = Tensor::default();
        tc.push_tensors(vec![ta, tb], Some(&bond_dims), None);
        let serialized = serialize(&tc);
        let deserialized: TensorBuilder = deserialize(&serialized);
        let tensor = deserialized.build(&bond_dims);
        assert!(Tensor::approx_eq(&tc, &tensor, 1e-10));
    }
}
