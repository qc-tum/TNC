use mpi::traits::Equivalence;

#[derive(Default, Debug, Clone, Equivalence, PartialEq)]
pub(super) struct BondDim {
    pub bond_id: usize,
    pub bond_size: u64,
}

impl BondDim {
    pub fn new(bond_id: usize, bond_size: u64) -> Self {
        Self { bond_id, bond_size }
    }
}

/// An artificial data type to send more data in a single MPI message.
///
/// MPI messages can only contain [`i32::MAX`] elements. To send more data, we can
/// increase the size of an element while keeping the total number of elements the
/// same.
#[derive(Debug, Clone, PartialEq, Eq, Equivalence)]
#[repr(transparent)]
pub struct MessageBinaryBlob([u8; 128]);

#[cfg(test)]
mod tests {
    use crate::mpi::mpi_types::BondDim;
    use mpi::traits::{Communicator, Root};
    use mpi_test::mpi_test;

    #[mpi_test(2)]
    fn test_sendrecv_bond_dims_need_mpi() {
        let universe = mpi::initialize().unwrap();
        let world = universe.world();
        // let size = world.size();
        let rank = world.rank();
        let root_process = world.process_at_rank(0);

        let bond_dims = if rank == 0 {
            let mut bond_dims = vec![
                BondDim::new(10, 24),
                BondDim::new(31, 55),
                BondDim::new(27, 126),
            ];
            root_process.broadcast_into(&mut bond_dims);
            bond_dims
        } else {
            let mut bond_dims = vec![BondDim::default(); 3];
            root_process.broadcast_into(&mut bond_dims);
            bond_dims
        };
        let bond_dims_ref = vec![
            BondDim::new(10, 24),
            BondDim::new(31, 55),
            BondDim::new(27, 126),
        ];

        assert_eq!(bond_dims, bond_dims_ref);
    }
}
