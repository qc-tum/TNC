use mpi::{traits::Equivalence, Rank};
use serde::{Deserialize, Serialize};

use crate::types::TensorIndex;

/// A bidirectional (1:1) mapping between MPI ranks and composite tensors.
///
/// Each valid tensor index should map to a rank, but not every rank has to map to a
/// tensor.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RankTensorMapping(Vec<(Rank, TensorIndex)>);

impl RankTensorMapping {
    pub fn with_capacity(capacity: usize) -> Self {
        Self(Vec::with_capacity(capacity))
    }

    /// Inserts a new mapping between an MPI rank and a tensor.
    pub fn insert(&mut self, rank: Rank, tensor: TensorIndex) {
        assert!(
            self.tensor(rank).is_none(),
            "Rank {rank} is already associated with a tensor",
        );
        assert!(
            self.rank_opt(tensor).is_none(),
            "Tensor {tensor} is already associated with a rank",
        );

        self.0.push((rank, tensor));
    }

    /// Gets the MPI rank associated with the given tensor (if any).
    #[inline]
    fn rank_opt(&self, tensor: TensorIndex) -> Option<Rank> {
        self.0.iter().find(|(_, t)| *t == tensor).map(|(r, _)| *r)
    }

    /// Gets the MPI rank associated with the given tensor. Each tensor should have
    /// a unique rank associated with it.
    pub fn rank(&self, tensor: TensorIndex) -> Rank {
        self.rank_opt(tensor).unwrap()
    }

    /// Gets the tensor associated with the given MPI rank (if any). As there can be
    /// more ranks than tensors, not every rank has to be associated with a tensor.
    pub fn tensor(&self, rank: Rank) -> Option<TensorIndex> {
        self.0.iter().find(|(r, _)| *r == rank).map(|(_, t)| *t)
    }

    /// Gets the number of mappings. This is equivalent to the number of MPI ranks
    /// needed and the number of composite tensors.
    #[inline]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Checks if there are no mappings.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl<'a> IntoIterator for &'a RankTensorMapping {
    type Item = &'a (Rank, TensorIndex);
    type IntoIter = std::slice::Iter<'a, (Rank, TensorIndex)>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

/// An artificial data type to send more data in a single MPI message.
///
/// MPI messages can only contain [`i32::MAX`] elements. To send more data, we can
/// increase the size of an element while keeping the total number of elements the
/// same.
///
/// The size of this type is 192 bytes, which allows for messages of
/// `192 B * 2^31 = 412 GB`.
#[derive(Debug, Clone, PartialEq, Eq, Equivalence)]
#[repr(transparent)]
pub struct MessageBinaryBlob([u8; 192]);

impl Default for MessageBinaryBlob {
    fn default() -> Self {
        Self([0; 192])
    }
}

#[cfg(test)]
mod tests {
    use super::RankTensorMapping;

    #[test]
    fn test_tensor_mapping() {
        let mut tensor_mapping = RankTensorMapping::default();

        assert_eq!(tensor_mapping.tensor(2), None);
        assert_eq!(tensor_mapping.tensor(3), None);

        tensor_mapping.insert(2, 4);

        assert_eq!(tensor_mapping.rank(4), 2);
        assert_eq!(tensor_mapping.tensor(2), Some(4));
        assert_eq!(tensor_mapping.tensor(3), None);

        tensor_mapping.insert(3, 0);

        assert_eq!(tensor_mapping.rank(4), 2);
        assert_eq!(tensor_mapping.tensor(2), Some(4));
        assert_eq!(tensor_mapping.rank(0), 3);
        assert_eq!(tensor_mapping.tensor(3), Some(0));
    }

    #[test]
    #[should_panic(expected = "Rank 2 is already associated with a tensor")]
    fn test_tensor_mapping_insert_rank_twice() {
        let mut tensor_mapping = RankTensorMapping::default();

        tensor_mapping.insert(2, 4);
        tensor_mapping.insert(3, 0);
        tensor_mapping.insert(2, 5);
    }

    #[test]
    #[should_panic(expected = "Tensor 4 is already associated with a rank")]
    fn test_tensor_mapping_insert_tensor_twice() {
        let mut tensor_mapping = RankTensorMapping::default();

        tensor_mapping.insert(2, 4);
        tensor_mapping.insert(3, 5);
        tensor_mapping.insert(4, 4);
    }
}
