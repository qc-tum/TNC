use mpi::{traits::Equivalence, Rank};
use serde::{Deserialize, Serialize};

use crate::types::TensorIndex;

/// A bidirectional mapping between MPI ranks and composite tensors.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RankTensorMapping(Vec<(Rank, TensorIndex)>);

impl RankTensorMapping {
    pub fn with_capacity(capacity: usize) -> Self {
        Self(Vec::with_capacity(capacity))
    }

    pub fn insert(&mut self, rank: Rank, tensor: TensorIndex) {
        self.0.push((rank, tensor));
    }

    /// Gets the MPI rank associated with the given tensor.
    pub fn rank(&self, tensor: TensorIndex) -> Rank {
        self.0
            .iter()
            .find(|(_, t)| *t == tensor)
            .map(|(r, _)| *r)
            .unwrap()
    }

    /// Gets the tensor associated with the given MPI rank (if any).
    pub fn tensor(&self, rank: Rank) -> Option<TensorIndex> {
        self.0.iter().find(|(r, _)| *r == rank).map(|(_, t)| *t)
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
