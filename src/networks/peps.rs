use std::collections::HashMap;

use itertools::iproduct;

use crate::tensornetwork::tensor::Tensor;

// fn generate_row(length: usize, depth: usize, start: usize, operator: bool) {
//     let mut tensors: Vec<usize> = (start..n + start).collect();
//     tensors.iter().map(|a| )
// }

/// Generate an n x n PEPs with shared bond dimension of `dimension`
fn peps_init(length: usize, depth: usize, physical_dim: u64, virtual_dim: u64) -> Tensor {
    let mut pep = Tensor::default();

    let physical_up = length * depth;
    let virtual_vertical = (length - 1) * depth;
    let virtual_horizontal = (depth - 1) * length;
    let total_edges = physical_up + virtual_vertical + virtual_horizontal;
    let new_iter = iproduct!(0..physical_up, physical_dim..=physical_dim).chain(iproduct!(
        physical_up..total_edges,
        virtual_dim..=virtual_dim
    ));
    let bond_dims: HashMap<usize, u64> = HashMap::from_iter(new_iter);

    let mut tensors = vec![Tensor::default(); length * depth];

    // Consider the corners
    tensors[0] = Tensor::new(vec![0, physical_up, physical_up + virtual_vertical]);
    tensors[length - 1] = Tensor::new(vec![
        length - 1,
        physical_up + length - 2,
        physical_up + virtual_vertical + length - 1,
    ]);
    tensors[length * (depth - 1)] = Tensor::new(vec![
        length * (depth - 1),
        physical_up + (length - 1) * (depth - 1),
        physical_up + virtual_vertical + length * (depth - 2),
    ]);
    tensors[length * depth - 1] = Tensor::new(vec![
        length * depth - 1,
        physical_up + (length - 1) * depth - 1,
        physical_up + virtual_vertical + length * (depth - 1) - 1,
    ]);

    // Consider the horizontal edges
    for (j, tensor) in tensors.iter_mut().enumerate().take(length - 1).skip(1) {
        *tensor = Tensor::new(vec![
            j,
            physical_up + j - 1,
            physical_up + j,
            physical_up + virtual_vertical + j,
        ])
    }
    for (j, tensor) in tensors
        .iter_mut()
        .rev()
        .enumerate()
        .take(length - 1)
        .skip(1)
    {
        *tensor = Tensor::new(vec![
            physical_up - j - 1,
            physical_up + virtual_vertical - j - 1,
            physical_up + virtual_vertical - j,
            total_edges - j - 1,
        ])
    }

    // Consider the vertical edges
    for i in 1..(depth - 1) {
        tensors[i * length] = Tensor::new(vec![
            i * length,
            physical_up + i * (length - 1),
            physical_up + virtual_vertical + (i - 1) * length,
            physical_up + virtual_vertical + i * length,
        ]);

        tensors[(i + 1) * length - 1] = Tensor::new(vec![
            (i + 1) * length - 1,
            physical_up + (i + 1) * (length - 1) - 1,
            physical_up + virtual_vertical + i * length - 1,
            physical_up + virtual_vertical + (i + 1) * length - 1,
        ]);
    }

    for (i, j) in iproduct!(1..(depth - 1), 1..(length - 1)) {
        let index = i * length + j;
        tensors[index] = Tensor::new(vec![
            index,
            physical_up + i * (length - 1) + j - 1,
            physical_up + i * (length - 1) + j,
            physical_up + virtual_vertical + (i - 1) * length + j,
            physical_up + virtual_vertical + i * length + j,
        ])
    }

    pep.push_tensors(tensors, Some(&bond_dims), None);
    pep
}

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

    let new_iter =
        iproduct!(start..(start + physical_up), physical_dim..=physical_dim).chain(iproduct!(
            (start + physical_up)..(start + total_edges),
            virtual_dim..=virtual_dim
        ));
    let bond_dims: HashMap<usize, u64> = HashMap::from_iter(new_iter);

    let mut tensors = vec![Tensor::default(); length * depth];

    // Consider the corners
    tensors[0] = Tensor::new(vec![
        last,
        start,
        start + physical_up,
        start + physical_up + virtual_vertical,
    ]);
    tensors[length - 1] = Tensor::new(vec![
        last + length - 1,
        start + length - 1,
        start + physical_up + length - 2,
        start + physical_up + virtual_vertical + length - 1,
    ]);
    tensors[length * (depth - 1)] = Tensor::new(vec![
        last + length * (depth - 1),
        start + length * (depth - 1),
        start + physical_up + (length - 1) * (depth - 1),
        start + physical_up + virtual_vertical + length * (depth - 2),
    ]);
    tensors[length * depth - 1] = Tensor::new(vec![
        last + length * depth - 1,
        start + length * depth - 1,
        start + physical_up + (length - 1) * depth - 1,
        start + physical_up + virtual_vertical + length * (depth - 1) - 1,
    ]);

    // Consider the edges
    for (j, tensor) in tensors.iter_mut().enumerate().take(length - 1).skip(1) {
        *tensor = Tensor::new(vec![
            last + j,
            start + j,
            start + physical_up + j - 1,
            start + physical_up + j,
            start + physical_up + virtual_vertical + j,
        ])
    }
    for (j, tensor) in tensors
        .iter_mut()
        .rev()
        .enumerate()
        .take(length - 1)
        .skip(1)
    {
        *tensor = Tensor::new(vec![
            last + physical_up - j - 1,
            start + physical_up - j - 1,
            start + physical_up + virtual_vertical - j - 1,
            start + physical_up + virtual_vertical - j,
            start + total_edges - j - 1,
        ])
    }

    for i in 1..(depth - 1) {
        tensors[i * length] = Tensor::new(vec![
            last + i * length,
            start + i * length,
            start + physical_up + i * (length - 1),
            start + physical_up + virtual_vertical + (i - 1) * length,
            start + physical_up + virtual_vertical + i * length,
        ]);

        tensors[(i + 1) * length - 1] = Tensor::new(vec![
            last + (i + 1) * length - 1,
            start + (i + 1) * length - 1,
            start + physical_up + (i + 1) * (length - 1) - 1,
            start + physical_up + virtual_vertical + i * length - 1,
            start + physical_up + virtual_vertical + (i + 1) * length - 1,
        ]);
    }

    for (i, j) in iproduct!(1..(depth - 1), 1..(length - 1)) {
        let index = i * length + j;
        tensors[index] = Tensor::new(vec![
            last + index,
            start + index,
            start + physical_up + i * (length - 1) + j - 1,
            start + physical_up + i * (length - 1) + j,
            start + physical_up + virtual_vertical + (i - 1) * length + j,
            start + physical_up + virtual_vertical + i * length + j,
        ])
    }
    peps.push_tensors(tensors, Some(&bond_dims), None);
    peps
}

fn peps_final(
    mut peps: Tensor,
    length: usize,
    depth: usize,
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

    let new_iter = iproduct!(start..(start + total_edges), virtual_dim..=virtual_dim);
    let bond_dims: HashMap<usize, u64> = HashMap::from_iter(new_iter);

    let mut tensors = vec![Tensor::default(); length * depth];

    // Consider the corners
    tensors[0] = Tensor::new(vec![last, start, start + virtual_vertical]);
    tensors[length - 1] = Tensor::new(vec![
        last + length - 1,
        start + length - 2,
        start + virtual_vertical + length - 1,
    ]);
    tensors[length * (depth - 1)] = Tensor::new(vec![
        last + length * (depth - 1),
        start + (length - 1) * (depth - 1),
        start + virtual_vertical + length * (depth - 2),
    ]);
    tensors[length * depth - 1] = Tensor::new(vec![
        last + length * depth - 1,
        start + (length - 1) * depth - 1,
        start + virtual_vertical + length * (depth - 1) - 1,
    ]);

    // Consider the edges
    for (j, tensor) in tensors.iter_mut().enumerate().take(length - 1).skip(1) {
        *tensor = Tensor::new(vec![
            last + j,
            start + j - 1,
            start + j,
            start + virtual_vertical + j,
        ])
    }
    for (j, tensor) in tensors
        .iter_mut()
        .rev()
        .enumerate()
        .take(length - 1)
        .skip(1)
    {
        *tensor = Tensor::new(vec![
            last + physical_up - j - 1,
            start + virtual_vertical - j - 1,
            start + virtual_vertical - j,
            start + total_edges - j - 1,
        ])
    }

    for i in 1..(depth - 1) {
        tensors[i * length] = Tensor::new(vec![
            last + i * length,
            start + i * (length - 1),
            start + virtual_vertical + (i - 1) * length,
            start + virtual_vertical + i * length,
        ]);

        tensors[(i + 1) * length - 1] = Tensor::new(vec![
            last + (i + 1) * length - 1,
            start + (i + 1) * (length - 1) - 1,
            start + virtual_vertical + i * length - 1,
            start + virtual_vertical + (i + 1) * length - 1,
        ]);
    }

    for (i, j) in iproduct!(1..(depth - 1), 1..(length - 1)) {
        let index = i * length + j;
        tensors[index] = Tensor::new(vec![
            last + index,
            start + i * (length - 1) + j - 1,
            start + i * (length - 1) + j,
            start + virtual_vertical + (i - 1) * length + j,
            start + virtual_vertical + i * length + j,
        ])
    }
    peps.push_tensors(tensors, Some(&bond_dims), None);
    peps
}

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
    peps_final(new_peps, length, depth, virtual_dim, layers)
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, iter::zip};

    use crate::tensornetwork::tensor::Tensor;

    use super::{pepo, peps, peps_final, peps_init};

    #[test]
    fn test_pep_init() {
        let length = 2;
        let depth = 2;
        let physical_dim = 4;
        let virtual_dim = 10;

        let mut ref_tensor = Tensor::default();
        let tensors = vec![
            Tensor::new(vec![0, 4, 6]),
            Tensor::new(vec![1, 4, 7]),
            Tensor::new(vec![2, 5, 6]),
            Tensor::new(vec![3, 5, 7]),
        ];
        let bond_dims = HashMap::from([
            (3, 4),
            (4, 10),
            (1, 4),
            (5, 10),
            (7, 10),
            (2, 4),
            (0, 4),
            (6, 10),
        ]);
        ref_tensor.push_tensors(tensors, Some(&bond_dims), None);

        let new_peps = peps_init(length, depth, physical_dim, virtual_dim);
        for (t1, t2) in zip(new_peps.tensors().iter(), ref_tensor.tensors().iter()) {
            assert_eq!(t1.legs(), t2.legs());
        }
        assert_eq!(*new_peps.bond_dims(), *ref_tensor.bond_dims());
    }

    #[test]
    fn test_pepo() {
        let length = 2;
        let depth = 2;
        let physical_dim = 4;
        let virtual_dim = 10;
        let layers = 1;

        let mut ref_tensor = Tensor::default();
        let tensors = vec![
            Tensor::new(vec![0, 4, 6]),
            Tensor::new(vec![1, 4, 7]),
            Tensor::new(vec![2, 5, 6]),
            Tensor::new(vec![3, 5, 7]),
            Tensor::new(vec![0, 8, 12, 14]),
            Tensor::new(vec![1, 9, 12, 15]),
            Tensor::new(vec![2, 10, 13, 14]),
            Tensor::new(vec![3, 11, 13, 15]),
        ];
        let bond_dims = HashMap::from([
            (0, 4),
            (1, 4),
            (2, 4),
            (3, 4),
            (4, 10),
            (5, 10),
            (6, 10),
            (7, 10),
            (8, 4),
            (9, 4),
            (10, 4),
            (11, 4),
            (12, 10),
            (13, 10),
            (14, 10),
            (15, 10),
        ]);
        ref_tensor.push_tensors(tensors, Some(&bond_dims), None);

        let mut new_peps = peps_init(length, depth, physical_dim, virtual_dim);
        for layer in 0..layers {
            new_peps = pepo(new_peps, length, depth, layer, physical_dim, virtual_dim);
        }

        for (t1, t2) in zip(new_peps.tensors().iter(), ref_tensor.tensors().iter()) {
            assert_eq!(t1.legs(), t2.legs());
        }
        assert_eq!(*new_peps.bond_dims(), *ref_tensor.bond_dims());
    }

    #[test]
    fn test_peps_final() {
        let length = 2;
        let depth = 2;
        let physical_dim = 4;
        let virtual_dim = 10;
        let layers = 1;

        let mut ref_tensor = Tensor::default();
        let tensors = vec![
            Tensor::new(vec![0, 4, 6]),
            Tensor::new(vec![1, 4, 7]),
            Tensor::new(vec![2, 5, 6]),
            Tensor::new(vec![3, 5, 7]),
            Tensor::new(vec![0, 8, 12, 14]),
            Tensor::new(vec![1, 9, 12, 15]),
            Tensor::new(vec![2, 10, 13, 14]),
            Tensor::new(vec![3, 11, 13, 15]),
            Tensor::new(vec![8, 16, 18]),
            Tensor::new(vec![9, 16, 19]),
            Tensor::new(vec![10, 17, 18]),
            Tensor::new(vec![11, 17, 19]),
        ];
        let bond_dims = HashMap::from([
            (0, 4),
            (1, 4),
            (2, 4),
            (3, 4),
            (4, 10),
            (5, 10),
            (6, 10),
            (7, 10),
            (8, 4),
            (9, 4),
            (10, 4),
            (11, 4),
            (12, 10),
            (13, 10),
            (14, 10),
            (15, 10),
            (16, 10),
            (17, 10),
            (18, 10),
            (19, 10),
        ]);
        ref_tensor.push_tensors(tensors, Some(&bond_dims), None);

        let mut new_peps = peps_init(length, depth, physical_dim, virtual_dim);
        for layer in 0..layers {
            new_peps = pepo(new_peps, length, depth, layer, physical_dim, virtual_dim);
        }
        let new_peps = peps_final(new_peps, length, depth, virtual_dim, layers);
        for (t1, t2) in zip(new_peps.tensors().iter(), ref_tensor.tensors().iter()) {
            assert_eq!(t1.legs(), t2.legs());
        }
        assert_eq!(*new_peps.bond_dims(), *ref_tensor.bond_dims());
    }

    #[test]
    fn test_peps() {
        let length = 2;
        let depth = 2;
        let physical_dim = 4;
        let virtual_dim = 10;
        let layers = 1;

        let mut ref_tensor = Tensor::default();
        let tensors = vec![
            Tensor::new(vec![0, 4, 6]),
            Tensor::new(vec![1, 4, 7]),
            Tensor::new(vec![2, 5, 6]),
            Tensor::new(vec![3, 5, 7]),
            Tensor::new(vec![0, 8, 12, 14]),
            Tensor::new(vec![1, 9, 12, 15]),
            Tensor::new(vec![2, 10, 13, 14]),
            Tensor::new(vec![3, 11, 13, 15]),
            Tensor::new(vec![8, 16, 18]),
            Tensor::new(vec![9, 16, 19]),
            Tensor::new(vec![10, 17, 18]),
            Tensor::new(vec![11, 17, 19]),
        ];
        let bond_dims = HashMap::from([
            (0, 4),
            (1, 4),
            (2, 4),
            (3, 4),
            (4, 10),
            (5, 10),
            (6, 10),
            (7, 10),
            (8, 4),
            (9, 4),
            (10, 4),
            (11, 4),
            (12, 10),
            (13, 10),
            (14, 10),
            (15, 10),
            (16, 10),
            (17, 10),
            (18, 10),
            (19, 10),
        ]);
        ref_tensor.push_tensors(tensors, Some(&bond_dims), None);

        let mut new_peps = peps(length, depth, physical_dim, virtual_dim, layers);
        for (t1, t2) in zip(new_peps.tensors().iter(), ref_tensor.tensors().iter()) {
            assert_eq!(t1.legs(), t2.legs());
        }
        assert_eq!(*new_peps.bond_dims(), *ref_tensor.bond_dims());
    }

    #[test]
    fn test_inner_product() {
        let length = 2;
        let depth = 2;
        let physical_dim = 4;
        let virtual_dim = 10;
        let layers = 0;

        let mut ref_tensor = Tensor::default();
        let tensors = vec![
            Tensor::new(vec![0, 4, 6]),
            Tensor::new(vec![1, 4, 7]),
            Tensor::new(vec![2, 5, 6]),
            Tensor::new(vec![3, 5, 7]),
            Tensor::new(vec![0, 8, 10]),
            Tensor::new(vec![1, 8, 11]),
            Tensor::new(vec![2, 9, 10]),
            Tensor::new(vec![3, 9, 11]),
        ];
        let bond_dims = HashMap::from([
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
        ref_tensor.push_tensors(tensors, Some(&bond_dims), None);

        let mut new_peps = peps_init(length, depth, physical_dim, virtual_dim);
        for layer in 0..layers {
            new_peps = pepo(new_peps, length, depth, layer, physical_dim, virtual_dim);
        }
        let new_peps = peps_final(new_peps, length, depth, virtual_dim, layers);
        for (t1, t2) in zip(new_peps.tensors().iter(), ref_tensor.tensors().iter()) {
            assert_eq!(t1.legs(), t2.legs());
        }
        assert_eq!(*new_peps.bond_dims(), *ref_tensor.bond_dims());
    }

    #[test]
    #[should_panic(expected = "PEPS should have length greater than 1")]
    fn test_mps() {
        let length = 1;
        let depth = 2;
        let physical_dim = 4;
        let virtual_dim = 10;
        let layers = 0;

        let mut ref_tensor = Tensor::default();
        let tensors = vec![Tensor::new(vec![0, 2]), Tensor::new(vec![1, 2])];
        let bond_dims = HashMap::from([(0, 4), (1, 4), (2, 10)]);
        ref_tensor.push_tensors(tensors, Some(&bond_dims), None);

        let new_peps = peps(length, depth, physical_dim, virtual_dim, layers);
        for (t1, t2) in zip(new_peps.tensors().iter(), ref_tensor.tensors().iter()) {
            assert_eq!(t1.legs(), t2.legs());
        }
        assert_eq!(*new_peps.bond_dims(), *ref_tensor.bond_dims());
    }
}
