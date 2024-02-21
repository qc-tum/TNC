use mpi::{
    datatype::{UncommittedUserDatatype, UserDatatype},
    traits::Equivalence,
    Address,
};

use crate::types::ContractionIndex;

#[derive(Default, Debug, Clone, Equivalence, PartialEq)]
pub(crate) struct BondDim {
    pub(crate) bond_id: usize,
    pub(crate) bond_size: u64,
}

impl BondDim {
    pub fn new(bond_id: usize, bond_size: u64) -> Self {
        Self { bond_id, bond_size }
    }
}

unsafe impl Equivalence for ContractionIndex {
    type Out = UserDatatype;

    fn equivalent_datatype() -> Self::Out {
        UserDatatype::structured(
            &[2],
            &[0 as Address],
            &[UncommittedUserDatatype::structured(
                &[2],
                &[0 as Address],
                &[usize::equivalent_datatype()],
            )],
        )
    }
}

#[cfg(test)]
mod tests {

    use crate::mpi::mpi_types::BondDim;
    use crate::types::ContractionIndex;
    use crate::{mpi_test, path};
    use mpi::traits::*;

    mpi_test!(
        2,
        fn test_sendrecv_contraction_index_need_mpi() {
            let universe = mpi::initialize().unwrap();
            let world = universe.world();
            // let size = world.size();
            let rank = world.rank();
            let root_process = world.process_at_rank(0);

            let contraction_indices = if rank == 0 {
                let mut contraction_indices = path![(0, 4), (1, 5), (2, 6)];
                root_process.broadcast_into(&mut contraction_indices);
                contraction_indices
            } else {
                let mut contraction_indices = vec![ContractionIndex::Pair(0, 0); 3];
                root_process.broadcast_into(&mut contraction_indices);
                contraction_indices
            };
            assert_eq!(contraction_indices, path![(0, 4), (1, 5), (2, 6)]);
        }
    );

    mpi_test!(
        2,
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
    );
}
