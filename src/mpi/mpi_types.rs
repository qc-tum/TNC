use mpi::{
    datatype::{UncommittedUserDatatype, UserDatatype},
    traits::Equivalence,
    Address,
};

use crate::types::ContractionIndex;

#[derive(Default, Debug, Clone, Equivalence, PartialEq)]
pub(super) struct BondDim {
    pub(super) bond_id: usize,
    pub(super) bond_size: u64,
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
            &[1, 1],
            &[0 as Address, 24 as Address],
            &[
                UncommittedUserDatatype::contiguous(3, &usize::equivalent_datatype()).as_ref(),
                usize::equivalent_datatype().into(),
            ],
        )
    }
}

#[cfg(test)]
mod tests {
    use std::iter::zip;

    use crate::mpi::mpi_types::BondDim;
    use crate::path;
    use crate::types::ContractionIndex;
    use mpi::traits::*;
    use mpi_test::mpi_test;

    mpi_test!(
        2,
        fn test_sendrecv_contraction_index() {
            let universe = mpi::initialize().unwrap();
            let world = universe.world();
            // let size = world.size();
            let rank = world.rank();
            let root_process = world.process_at_rank(0);
            let max = usize::MAX;

            let contraction_indices = if rank == 0 {
                let mut contraction_indices = path![
                    (0, 4),
                    (1, 5),
                    (2, 16),
                    (7, max),
                    (max, 5),
                    (64, 2),
                    (4, 55),
                    (81, 21),
                    (2, 72),
                    (23, 3),
                    (40, 5),
                    (2, 26)
                ]
                .to_vec();
                root_process.broadcast_into(&mut contraction_indices);
                contraction_indices
            } else {
                let mut contraction_indices = vec![ContractionIndex::Pair(0, 0); 12];
                root_process.broadcast_into(&mut contraction_indices);
                contraction_indices
            };
            let ref_contraction_indices = path![
                (0, 4),
                (1, 5),
                (2, 16),
                (7, max),
                (max, 5),
                (64, 2),
                (4, 55),
                (81, 21),
                (2, 72),
                (23, 3),
                (40, 5),
                (2, 26)
            ]
            .to_vec();
            assert_eq!(contraction_indices, ref_contraction_indices);
            // Note that rust fills empty space in enum instance with garbage
            for (ref_data, data) in zip(ref_contraction_indices, contraction_indices) {
                assert_eq!(get_memory(&ref_data)[0..24], get_memory(&data)[0..24]);
            }
        }
    );
    fn get_memory<'a, T>(input: &'a T) -> &'a [u8] {
        unsafe {
            std::slice::from_raw_parts(input as *const _ as *const u8, std::mem::size_of::<T>())
        }
    }

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
