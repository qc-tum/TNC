use std::collections::HashMap;

use itertools::iproduct;

use crate::tensornetwork::tensor::Tensor;

// fn generate_row(length: usize, depth: usize, start: usize, operator: bool) {
//     let mut tensors: Vec<usize> = (start..n + start).collect();
//     tensors.iter().map(|a| )
// }

/// Generate an n x n PEPs with shared bond dimension of `dimension`
pub fn peps_init(length: usize, depth: usize, physical_dim: u64, virtual_dim: u64) -> Tensor {
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

pub fn pepo(
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

pub fn peps_final(
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
