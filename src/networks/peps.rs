use itertools::iproduct;

use crate::tensornetwork::tensor::Tensor;

/// Generate the initial state for an `length` x `depth` PEPS.
///
/// # Arguments
/// * `length` - length of the PEPS
/// * `depth` - depth of the PEPS
/// * `physical_dim` - physical dimension of the PEPs
/// * `virtual_dim` - virtual bond dimension between lattice sites in the PEPs
///
/// # Returns
/// [`Tensor`] of initial state of PEPS
fn peps_init(length: usize, depth: usize, physical_dim: u64, virtual_dim: u64) -> Tensor {
    let mut pep = Tensor::default();

    let physical_up = length * depth;
    let virtual_vertical = (length - 1) * depth;
    let virtual_horizontal = (depth - 1) * length;
    let total_edges = physical_up + virtual_vertical + virtual_horizontal;

    let mut tensors = vec![Tensor::default(); length * depth];

    // Consider the corners
    tensors[0] = Tensor::new(
        vec![0, physical_up, physical_up + virtual_vertical],
        vec![physical_dim, virtual_dim, virtual_dim],
    );
    tensors[length - 1] = Tensor::new(
        vec![
            length - 1,
            physical_up + length - 2,
            physical_up + virtual_vertical + length - 1,
        ],
        vec![physical_dim, virtual_dim, virtual_dim],
    );
    tensors[length * (depth - 1)] = Tensor::new(
        vec![
            length * (depth - 1),
            physical_up + (length - 1) * (depth - 1),
            physical_up + virtual_vertical + length * (depth - 2),
        ],
        vec![physical_dim, virtual_dim, virtual_dim],
    );
    tensors[length * depth - 1] = Tensor::new(
        vec![
            length * depth - 1,
            physical_up + (length - 1) * depth - 1,
            physical_up + virtual_vertical + length * (depth - 1) - 1,
        ],
        vec![physical_dim, virtual_dim, virtual_dim],
    );

    // Consider the horizontal edges
    for j in 1..(length - 1) {
        tensors[j] = Tensor::new(
            vec![
                j,
                physical_up + j - 1,
                physical_up + j,
                physical_up + virtual_vertical + j,
            ],
            vec![physical_dim, virtual_dim, virtual_dim, virtual_dim],
        );

        tensors[physical_up - j - 1] = Tensor::new(
            vec![
                physical_up - j - 1,
                physical_up + virtual_vertical - j - 1,
                physical_up + virtual_vertical - j,
                total_edges - j - 1,
            ],
            vec![physical_dim, virtual_dim, virtual_dim, virtual_dim],
        );
    }

    // Consider the vertical edges
    for i in 1..(depth - 1) {
        tensors[i * length] = Tensor::new(
            vec![
                i * length,
                physical_up + i * (length - 1),
                physical_up + virtual_vertical + (i - 1) * length,
                physical_up + virtual_vertical + i * length,
            ],
            vec![physical_dim, virtual_dim, virtual_dim, virtual_dim],
        );

        tensors[(i + 1) * length - 1] = Tensor::new(
            vec![
                (i + 1) * length - 1,
                physical_up + (i + 1) * (length - 1) - 1,
                physical_up + virtual_vertical + i * length - 1,
                physical_up + virtual_vertical + (i + 1) * length - 1,
            ],
            vec![physical_dim, virtual_dim, virtual_dim, virtual_dim],
        );
    }

    // Consider the remaining bulk not on the edges
    for (i, j) in iproduct!(1..(depth - 1), 1..(length - 1)) {
        let index = i * length + j;
        tensors[index] = Tensor::new(
            vec![
                index,
                physical_up + i * (length - 1) + j - 1,
                physical_up + i * (length - 1) + j,
                physical_up + virtual_vertical + (i - 1) * length + j,
                physical_up + virtual_vertical + i * length + j,
            ],
            vec![
                physical_dim,
                virtual_dim,
                virtual_dim,
                virtual_dim,
                virtual_dim,
            ],
        );
    }

    pep.push_tensors(tensors);
    pep
}

/// Generate an intermediate PEP-operator an n x n PEPs with shared bond dimension of `dimension`.
///
/// # Arguments
/// * `peps` - mutable [`Tensor`] representing initial PEPS and successive PEPOs
/// * `length` - length of the PEPS
/// * `depth` - depth of the PEPS
/// * `layer` - layer of PEPO to be applied
/// * `physical_dim` - physical dimension of the PEPs
/// * `virtual_dim` - virtual bond dimension between lattice sites in the PEPs
///
/// # Returns
/// Updated PEPS [`Tensor`] with `layer`th PEPO applied.
fn pepo(
    mut peps: Tensor,
    length: usize,
    depth: usize,
    layer: usize,
    physical_dim: u64,
    virtual_dim: u64,
) -> Tensor {
    let physical_up = length * depth;
    let virtual_vertical = (length - 1) * depth;
    let virtual_horizontal = (depth - 1) * length;
    let total_edges = physical_up + virtual_vertical + virtual_horizontal;
    let last = total_edges * layer;
    let start = total_edges * (layer + 1);

    let mut tensors = vec![Tensor::default(); length * depth];

    // Consider the corners
    tensors[0] = Tensor::new(
        vec![
            last,
            start,
            start + physical_up,
            start + physical_up + virtual_vertical,
        ],
        vec![physical_dim, physical_dim, virtual_dim, virtual_dim],
    );
    tensors[length - 1] = Tensor::new(
        vec![
            last + length - 1,
            start + length - 1,
            start + physical_up + length - 2,
            start + physical_up + virtual_vertical + length - 1,
        ],
        vec![physical_dim, physical_dim, virtual_dim, virtual_dim],
    );
    tensors[length * (depth - 1)] = Tensor::new(
        vec![
            last + length * (depth - 1),
            start + length * (depth - 1),
            start + physical_up + (length - 1) * (depth - 1),
            start + physical_up + virtual_vertical + length * (depth - 2),
        ],
        vec![physical_dim, physical_dim, virtual_dim, virtual_dim],
    );
    tensors[length * depth - 1] = Tensor::new(
        vec![
            last + length * depth - 1,
            start + length * depth - 1,
            start + physical_up + (length - 1) * depth - 1,
            start + physical_up + virtual_vertical + length * (depth - 1) - 1,
        ],
        vec![physical_dim, physical_dim, virtual_dim, virtual_dim],
    );

    // Consider the horizontal edges
    for j in 1..(length - 1) {
        tensors[j] = Tensor::new(
            vec![
                last + j,
                start + j,
                start + physical_up + j - 1,
                start + physical_up + j,
                start + physical_up + virtual_vertical + j,
            ],
            vec![
                physical_dim,
                physical_dim,
                virtual_dim,
                virtual_dim,
                virtual_dim,
            ],
        );

        tensors[physical_up - j - 1] = Tensor::new(
            vec![
                last + physical_up - j - 1,
                start + physical_up - j - 1,
                start + physical_up + virtual_vertical - j - 1,
                start + physical_up + virtual_vertical - j,
                start + total_edges - j - 1,
            ],
            vec![
                physical_dim,
                physical_dim,
                virtual_dim,
                virtual_dim,
                virtual_dim,
            ],
        );
    }

    // Consider the vertical edges
    for i in 1..(depth - 1) {
        tensors[i * length] = Tensor::new(
            vec![
                last + i * length,
                start + i * length,
                start + physical_up + i * (length - 1),
                start + physical_up + virtual_vertical + (i - 1) * length,
                start + physical_up + virtual_vertical + i * length,
            ],
            vec![
                physical_dim,
                physical_dim,
                virtual_dim,
                virtual_dim,
                virtual_dim,
            ],
        );

        tensors[(i + 1) * length - 1] = Tensor::new(
            vec![
                last + (i + 1) * length - 1,
                start + (i + 1) * length - 1,
                start + physical_up + (i + 1) * (length - 1) - 1,
                start + physical_up + virtual_vertical + i * length - 1,
                start + physical_up + virtual_vertical + (i + 1) * length - 1,
            ],
            vec![
                physical_dim,
                physical_dim,
                virtual_dim,
                virtual_dim,
                virtual_dim,
            ],
        );
    }

    // Consider the remaining bulk not on the edges
    for (i, j) in iproduct!(1..(depth - 1), 1..(length - 1)) {
        let index = i * length + j;
        tensors[index] = Tensor::new(
            vec![
                last + index,
                start + index,
                start + physical_up + i * (length - 1) + j - 1,
                start + physical_up + i * (length - 1) + j,
                start + physical_up + virtual_vertical + (i - 1) * length + j,
                start + physical_up + virtual_vertical + i * length + j,
            ],
            vec![
                physical_dim,
                physical_dim,
                virtual_dim,
                virtual_dim,
                virtual_dim,
                virtual_dim,
            ],
        );
    }
    peps.push_tensors(tensors);
    peps
}

/// Applies the final state in a PEPS.
///
/// # Arguments
/// * `peps` - mutable [`Tensor`] representing initial PEPS and successive PEPOs
/// * `length` - length of the PEPS
/// * `depth` - depth of the PEPS
/// * `physical_dim` - physical dimension of the PEPs
/// * `virtual_dim` - virtual bond dimension between lattice sites in the PEPs
///
/// # Returns
/// Updated PEPS [`Tensor`] with final PEPS applied
fn peps_final(
    mut peps: Tensor,
    length: usize,
    depth: usize,
    physical_dim: u64,
    virtual_dim: u64,
    layers: usize,
) -> Tensor {
    let physical_up = length * depth;
    let virtual_vertical = (length - 1) * depth;
    let virtual_horizontal = (depth - 1) * length;
    let mut total_edges = physical_up + virtual_vertical + virtual_horizontal;
    let last = total_edges * layers;
    let start = total_edges * (layers + 1);
    total_edges -= physical_up;

    let mut tensors = vec![Tensor::default(); length * depth];

    // Consider the corners
    tensors[0] = Tensor::new(
        vec![last, start, start + virtual_vertical],
        vec![physical_dim, virtual_dim, virtual_dim],
    );
    tensors[length - 1] = Tensor::new(
        vec![
            last + length - 1,
            start + length - 2,
            start + virtual_vertical + length - 1,
        ],
        vec![physical_dim, virtual_dim, virtual_dim],
    );
    tensors[length * (depth - 1)] = Tensor::new(
        vec![
            last + length * (depth - 1),
            start + (length - 1) * (depth - 1),
            start + virtual_vertical + length * (depth - 2),
        ],
        vec![physical_dim, virtual_dim, virtual_dim],
    );
    tensors[length * depth - 1] = Tensor::new(
        vec![
            last + length * depth - 1,
            start + (length - 1) * depth - 1,
            start + virtual_vertical + length * (depth - 1) - 1,
        ],
        vec![physical_dim, virtual_dim, virtual_dim],
    );

    // Consider the horizontal edges
    for j in 1..(length - 1) {
        tensors[j] = Tensor::new(
            vec![
                last + j,
                start + j - 1,
                start + j,
                start + virtual_vertical + j,
            ],
            vec![physical_dim, virtual_dim, virtual_dim, virtual_dim],
        );

        tensors[physical_up - j - 1] = Tensor::new(
            vec![
                last + physical_up - j - 1,
                start + virtual_vertical - j - 1,
                start + virtual_vertical - j,
                start + total_edges - j - 1,
            ],
            vec![physical_dim, virtual_dim, virtual_dim, virtual_dim],
        );
    }

    // Consider the vertical edges
    for i in 1..(depth - 1) {
        tensors[i * length] = Tensor::new(
            vec![
                last + i * length,
                start + i * (length - 1),
                start + virtual_vertical + (i - 1) * length,
                start + virtual_vertical + i * length,
            ],
            vec![physical_dim, virtual_dim, virtual_dim, virtual_dim],
        );

        tensors[(i + 1) * length - 1] = Tensor::new(
            vec![
                last + (i + 1) * length - 1,
                start + (i + 1) * (length - 1) - 1,
                start + virtual_vertical + i * length - 1,
                start + virtual_vertical + (i + 1) * length - 1,
            ],
            vec![physical_dim, virtual_dim, virtual_dim, virtual_dim],
        );
    }

    // Consider the remaining bulk not on the edges
    for (i, j) in iproduct!(1..(depth - 1), 1..(length - 1)) {
        let index = i * length + j;
        tensors[index] = Tensor::new(
            vec![
                last + index,
                start + i * (length - 1) + j - 1,
                start + i * (length - 1) + j,
                start + virtual_vertical + (i - 1) * length + j,
                start + virtual_vertical + i * length + j,
            ],
            vec![
                physical_dim,
                virtual_dim,
                virtual_dim,
                virtual_dim,
                virtual_dim,
            ],
        );
    }
    peps.push_tensors(tensors);
    peps
}

/// Generates the structure for a PEPS with `length` x `depth` dimensions and with `layers` layers.
///
/// The `EdgeIndex` in the PEPS `bond_dims` for each layer is ordered as `physical bonds to previous layer,
/// physical bonds to next layer, vertical virtual bonds, horizontal virtual bonds`.
///
/// Each new layer other than the final layer adds:
/// * p = length * depth physical bonds
/// * vv = (length - 1) * depth vertical virtual bonds
/// * vh = (depth - 1) * length horizontal virtual bonds
///
/// Bonds for the `k`th layer, where k is not the initial or final layer, then run from:
/// * physical bonds to previous layer: (k-1) * (p+vv+vh): (kp) + (k-1) * (vv+vh)
/// * physical bonds to next layer: k * (p+vv+vh)
///
/// # Arguments
/// * `length` - length of the PEPS
/// * `depth` - depth of the PEPS
/// * `physical_dim` - physical dimension of the PEPs
/// * `virtual_dim` - virtual bond dimension between lattice sites in the PEPs
/// * `layers` - number of operator layers in PEPS, 0 layers returns an inner product of two states.
///
/// # Returns
/// [`Tensor`] representing a PEPS.
///
/// # Panics
/// Panics if `length < 2` or `depth < 2`.
#[must_use]
pub fn peps(
    length: usize,
    depth: usize,
    physical_dim: u64,
    virtual_dim: u64,
    layers: usize,
) -> Tensor {
    assert!(length > 1, "PEPS should have length greater than 1");
    assert!(depth > 1, "PEPS should have depth greater than 1");
    let mut new_peps = peps_init(length, depth, physical_dim, virtual_dim);
    for layer in 0..layers {
        new_peps = pepo(new_peps, length, depth, layer, physical_dim, virtual_dim);
    }
    peps_final(new_peps, length, depth, physical_dim, virtual_dim, layers)
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::iter::zip;

    use rustc_hash::FxHashMap;

    use crate::tensornetwork::tensor::Tensor;

    #[test]
    fn test_pep_init() {
        let length = 3;
        let depth = 3;
        let physical_dim = 4;
        let virtual_dim = 10;

        let bond_dims = FxHashMap::from_iter([
            (0, 4),
            (1, 4),
            (2, 4),
            (3, 4),
            (4, 4),
            (5, 4),
            (6, 4),
            (7, 4),
            (8, 4),
            (9, 10),
            (10, 10),
            (11, 10),
            (12, 10),
            (13, 10),
            (14, 10),
            (15, 10),
            (16, 10),
            (17, 10),
            (18, 10),
            (19, 10),
            (20, 10),
        ]);
        let tensors = vec![
            Tensor::new_from_map(vec![0, 9, 15], &bond_dims),
            Tensor::new_from_map(vec![1, 9, 10, 16], &bond_dims),
            Tensor::new_from_map(vec![2, 10, 17], &bond_dims),
            Tensor::new_from_map(vec![3, 11, 15, 18], &bond_dims),
            Tensor::new_from_map(vec![4, 11, 12, 16, 19], &bond_dims),
            Tensor::new_from_map(vec![5, 12, 17, 20], &bond_dims),
            Tensor::new_from_map(vec![6, 13, 18], &bond_dims),
            Tensor::new_from_map(vec![7, 13, 14, 19], &bond_dims),
            Tensor::new_from_map(vec![8, 14, 20], &bond_dims),
        ];
        let ref_tensor = Tensor::new_composite(tensors);

        let new_peps = peps_init(length, depth, physical_dim, virtual_dim);
        for (t1, t2) in zip(new_peps.tensors().iter(), ref_tensor.tensors().iter()) {
            assert_eq!(t1.legs(), t2.legs());
            assert_eq!(t1.bond_dims(), t2.bond_dims());
        }
    }

    #[test]
    fn test_pepo() {
        let length = 3;
        let depth = 3;
        let physical_dim = 4;
        let virtual_dim = 10;
        let layers = 1;

        let bond_dims = FxHashMap::from_iter([
            (0, 4),
            (1, 4),
            (2, 4),
            (3, 4),
            (4, 4),
            (5, 4),
            (6, 4),
            (7, 4),
            (8, 4),
            (9, 10),
            (10, 10),
            (11, 10),
            (12, 10),
            (13, 10),
            (14, 10),
            (15, 10),
            (16, 10),
            (17, 10),
            (18, 10),
            (19, 10),
            (20, 10),
            (21, 4),
            (22, 4),
            (23, 4),
            (24, 4),
            (25, 4),
            (26, 4),
            (27, 4),
            (28, 4),
            (29, 4),
            (30, 10),
            (31, 10),
            (32, 10),
            (33, 10),
            (34, 10),
            (35, 10),
            (36, 10),
            (37, 10),
            (38, 10),
            (39, 10),
            (40, 10),
            (41, 10),
        ]);
        let tensors = vec![
            Tensor::new_from_map(vec![0, 9, 15], &bond_dims),
            Tensor::new_from_map(vec![1, 9, 10, 16], &bond_dims),
            Tensor::new_from_map(vec![2, 10, 17], &bond_dims),
            Tensor::new_from_map(vec![3, 11, 15, 18], &bond_dims),
            Tensor::new_from_map(vec![4, 11, 12, 16, 19], &bond_dims),
            Tensor::new_from_map(vec![5, 12, 17, 20], &bond_dims),
            Tensor::new_from_map(vec![6, 13, 18], &bond_dims),
            Tensor::new_from_map(vec![7, 13, 14, 19], &bond_dims),
            Tensor::new_from_map(vec![8, 14, 20], &bond_dims),
            Tensor::new_from_map(vec![0, 21, 30, 36], &bond_dims),
            Tensor::new_from_map(vec![1, 22, 30, 31, 37], &bond_dims),
            Tensor::new_from_map(vec![2, 23, 31, 38], &bond_dims),
            Tensor::new_from_map(vec![3, 24, 32, 36, 39], &bond_dims),
            Tensor::new_from_map(vec![4, 25, 32, 33, 37, 40], &bond_dims),
            Tensor::new_from_map(vec![5, 26, 33, 38, 41], &bond_dims),
            Tensor::new_from_map(vec![6, 27, 34, 39], &bond_dims),
            Tensor::new_from_map(vec![7, 28, 34, 35, 40], &bond_dims),
            Tensor::new_from_map(vec![8, 29, 35, 41], &bond_dims),
        ];
        let ref_tensor = Tensor::new_composite(tensors);

        let mut new_peps = peps_init(length, depth, physical_dim, virtual_dim);
        for layer in 0..layers {
            new_peps = pepo(new_peps, length, depth, layer, physical_dim, virtual_dim);
        }

        for (t1, t2) in zip(new_peps.tensors().iter(), ref_tensor.tensors().iter()) {
            assert_eq!(t1.legs(), t2.legs());
            assert_eq!(t1.bond_dims(), t2.bond_dims());
        }
    }

    #[test]
    fn test_peps_final() {
        let length = 3;
        let depth = 3;
        let physical_dim = 4;
        let virtual_dim = 10;
        let layers = 1;

        let bond_dims = FxHashMap::from_iter([
            (0, 4),
            (1, 4),
            (2, 4),
            (3, 4),
            (4, 4),
            (5, 4),
            (6, 4),
            (7, 4),
            (8, 4),
            (9, 10),
            (10, 10),
            (11, 10),
            (12, 10),
            (13, 10),
            (14, 10),
            (15, 10),
            (16, 10),
            (17, 10),
            (18, 10),
            (19, 10),
            (20, 10),
            (21, 4),
            (22, 4),
            (23, 4),
            (24, 4),
            (25, 4),
            (26, 4),
            (27, 4),
            (28, 4),
            (29, 4),
            (30, 10),
            (31, 10),
            (32, 10),
            (33, 10),
            (34, 10),
            (35, 10),
            (36, 10),
            (37, 10),
            (38, 10),
            (39, 10),
            (40, 10),
            (41, 10),
            (42, 10),
            (43, 10),
            (44, 10),
            (45, 10),
            (46, 10),
            (47, 10),
            (48, 10),
            (49, 10),
            (50, 10),
            (51, 10),
            (52, 10),
            (53, 10),
        ]);
        let tensors = vec![
            Tensor::new_from_map(vec![0, 9, 15], &bond_dims),
            Tensor::new_from_map(vec![1, 9, 10, 16], &bond_dims),
            Tensor::new_from_map(vec![2, 10, 17], &bond_dims),
            Tensor::new_from_map(vec![3, 11, 15, 18], &bond_dims),
            Tensor::new_from_map(vec![4, 11, 12, 16, 19], &bond_dims),
            Tensor::new_from_map(vec![5, 12, 17, 20], &bond_dims),
            Tensor::new_from_map(vec![6, 13, 18], &bond_dims),
            Tensor::new_from_map(vec![7, 13, 14, 19], &bond_dims),
            Tensor::new_from_map(vec![8, 14, 20], &bond_dims),
            Tensor::new_from_map(vec![0, 21, 30, 36], &bond_dims),
            Tensor::new_from_map(vec![1, 22, 30, 31, 37], &bond_dims),
            Tensor::new_from_map(vec![2, 23, 31, 38], &bond_dims),
            Tensor::new_from_map(vec![3, 24, 32, 36, 39], &bond_dims),
            Tensor::new_from_map(vec![4, 25, 32, 33, 37, 40], &bond_dims),
            Tensor::new_from_map(vec![5, 26, 33, 38, 41], &bond_dims),
            Tensor::new_from_map(vec![6, 27, 34, 39], &bond_dims),
            Tensor::new_from_map(vec![7, 28, 34, 35, 40], &bond_dims),
            Tensor::new_from_map(vec![8, 29, 35, 41], &bond_dims),
            Tensor::new_from_map(vec![21, 42, 48], &bond_dims),
            Tensor::new_from_map(vec![22, 42, 43, 49], &bond_dims),
            Tensor::new_from_map(vec![23, 43, 50], &bond_dims),
            Tensor::new_from_map(vec![24, 44, 48, 51], &bond_dims),
            Tensor::new_from_map(vec![25, 44, 45, 49, 52], &bond_dims),
            Tensor::new_from_map(vec![26, 45, 50, 53], &bond_dims),
            Tensor::new_from_map(vec![27, 46, 51], &bond_dims),
            Tensor::new_from_map(vec![28, 46, 47, 52], &bond_dims),
            Tensor::new_from_map(vec![29, 47, 53], &bond_dims),
        ];
        let ref_tensor = Tensor::new_composite(tensors);

        let mut new_peps = peps_init(length, depth, physical_dim, virtual_dim);
        for layer in 0..layers {
            new_peps = pepo(new_peps, length, depth, layer, physical_dim, virtual_dim);
        }
        let new_peps = peps_final(new_peps, length, depth, physical_dim, virtual_dim, layers);
        for (t1, t2) in zip(new_peps.tensors().iter(), ref_tensor.tensors().iter()) {
            assert_eq!(t1.legs(), t2.legs());
            assert_eq!(t1.bond_dims(), t2.bond_dims());
        }
    }

    #[test]
    fn test_peps() {
        let length = 3;
        let depth = 3;
        let physical_dim = 4;
        let virtual_dim = 10;
        let layers = 1;

        let bond_dims = FxHashMap::from_iter([
            (0, 4),
            (1, 4),
            (2, 4),
            (3, 4),
            (4, 4),
            (5, 4),
            (6, 4),
            (7, 4),
            (8, 4),
            (9, 10),
            (10, 10),
            (11, 10),
            (12, 10),
            (13, 10),
            (14, 10),
            (15, 10),
            (16, 10),
            (17, 10),
            (18, 10),
            (19, 10),
            (20, 10),
            (21, 4),
            (22, 4),
            (23, 4),
            (24, 4),
            (25, 4),
            (26, 4),
            (27, 4),
            (28, 4),
            (29, 4),
            (30, 10),
            (31, 10),
            (32, 10),
            (33, 10),
            (34, 10),
            (35, 10),
            (36, 10),
            (37, 10),
            (38, 10),
            (39, 10),
            (40, 10),
            (41, 10),
            (42, 10),
            (43, 10),
            (44, 10),
            (45, 10),
            (46, 10),
            (47, 10),
            (48, 10),
            (49, 10),
            (50, 10),
            (51, 10),
            (52, 10),
            (53, 10),
        ]);
        let tensors = vec![
            Tensor::new_from_map(vec![0, 9, 15], &bond_dims),
            Tensor::new_from_map(vec![1, 9, 10, 16], &bond_dims),
            Tensor::new_from_map(vec![2, 10, 17], &bond_dims),
            Tensor::new_from_map(vec![3, 11, 15, 18], &bond_dims),
            Tensor::new_from_map(vec![4, 11, 12, 16, 19], &bond_dims),
            Tensor::new_from_map(vec![5, 12, 17, 20], &bond_dims),
            Tensor::new_from_map(vec![6, 13, 18], &bond_dims),
            Tensor::new_from_map(vec![7, 13, 14, 19], &bond_dims),
            Tensor::new_from_map(vec![8, 14, 20], &bond_dims),
            Tensor::new_from_map(vec![0, 21, 30, 36], &bond_dims),
            Tensor::new_from_map(vec![1, 22, 30, 31, 37], &bond_dims),
            Tensor::new_from_map(vec![2, 23, 31, 38], &bond_dims),
            Tensor::new_from_map(vec![3, 24, 32, 36, 39], &bond_dims),
            Tensor::new_from_map(vec![4, 25, 32, 33, 37, 40], &bond_dims),
            Tensor::new_from_map(vec![5, 26, 33, 38, 41], &bond_dims),
            Tensor::new_from_map(vec![6, 27, 34, 39], &bond_dims),
            Tensor::new_from_map(vec![7, 28, 34, 35, 40], &bond_dims),
            Tensor::new_from_map(vec![8, 29, 35, 41], &bond_dims),
            Tensor::new_from_map(vec![21, 42, 48], &bond_dims),
            Tensor::new_from_map(vec![22, 42, 43, 49], &bond_dims),
            Tensor::new_from_map(vec![23, 43, 50], &bond_dims),
            Tensor::new_from_map(vec![24, 44, 48, 51], &bond_dims),
            Tensor::new_from_map(vec![25, 44, 45, 49, 52], &bond_dims),
            Tensor::new_from_map(vec![26, 45, 50, 53], &bond_dims),
            Tensor::new_from_map(vec![27, 46, 51], &bond_dims),
            Tensor::new_from_map(vec![28, 46, 47, 52], &bond_dims),
            Tensor::new_from_map(vec![29, 47, 53], &bond_dims),
        ];
        let ref_tensor = Tensor::new_composite(tensors);

        let new_peps = peps(length, depth, physical_dim, virtual_dim, layers);
        for (t1, t2) in zip(new_peps.tensors().iter(), ref_tensor.tensors().iter()) {
            assert_eq!(t1.legs(), t2.legs());
            assert_eq!(t1.bond_dims(), t2.bond_dims());
        }
    }

    #[test]
    fn test_inner_product() {
        let length = 2;
        let depth = 2;
        let physical_dim = 4;
        let virtual_dim = 10;
        let layers = 0;

        let bond_dims = FxHashMap::from_iter([
            (0, 4),
            (1, 4),
            (2, 4),
            (3, 4),
            (4, 10),
            (5, 10),
            (6, 10),
            (7, 10),
            (8, 10),
            (9, 10),
            (10, 10),
            (11, 10),
        ]);
        let tensors = vec![
            Tensor::new_from_map(vec![0, 4, 6], &bond_dims),
            Tensor::new_from_map(vec![1, 4, 7], &bond_dims),
            Tensor::new_from_map(vec![2, 5, 6], &bond_dims),
            Tensor::new_from_map(vec![3, 5, 7], &bond_dims),
            Tensor::new_from_map(vec![0, 8, 10], &bond_dims),
            Tensor::new_from_map(vec![1, 8, 11], &bond_dims),
            Tensor::new_from_map(vec![2, 9, 10], &bond_dims),
            Tensor::new_from_map(vec![3, 9, 11], &bond_dims),
        ];
        let ref_tensor = Tensor::new_composite(tensors);

        let mut new_peps = peps_init(length, depth, physical_dim, virtual_dim);
        for layer in 0..layers {
            new_peps = pepo(new_peps, length, depth, layer, physical_dim, virtual_dim);
        }
        let new_peps = peps_final(new_peps, length, depth, physical_dim, virtual_dim, layers);
        for (t1, t2) in zip(new_peps.tensors().iter(), ref_tensor.tensors().iter()) {
            assert_eq!(t1.legs(), t2.legs());
            assert_eq!(t1.bond_dims(), t2.bond_dims());
        }
    }

    #[test]
    #[should_panic(expected = "PEPS should have length greater than 1")]
    fn test_mps() {
        let length = 1;
        let depth = 2;
        let physical_dim = 4;
        let virtual_dim = 10;
        let layers = 0;

        let bond_dims = FxHashMap::from_iter([(0, 4), (1, 4), (2, 10)]);
        let tensors = vec![
            Tensor::new_from_map(vec![0, 2], &bond_dims),
            Tensor::new_from_map(vec![1, 2], &bond_dims),
        ];
        let ref_tensor = Tensor::new_composite(tensors);

        let new_peps = peps(length, depth, physical_dim, virtual_dim, layers);
        for (t1, t2) in zip(new_peps.tensors().iter(), ref_tensor.tensors().iter()) {
            assert_eq!(t1.legs(), t2.legs());
            assert_eq!(t1.bond_dims(), t2.bond_dims());
        }
    }
}
