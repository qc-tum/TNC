use std::{
    cmp::{max, min},
    collections::{BinaryHeap, HashMap, HashSet},
};

use itertools::Itertools;
use rand::{rngs::StdRng, Rng, SeedableRng};

use crate::{
    contractionpath::{
        candidates::Candidate, contraction_cost::contract_path_cost, paths::RNGChooser,
        ssa_ordering, ssa_replace_ordering,
    },
    tensornetwork::tensor::Tensor,
    types::{calculate_hash, ContractionIndex},
};

use super::{validate_path, CostFnType, CostType, OptimizePath};

pub struct Greedy<'a> {
    pub(crate) tn: &'a Tensor,
    pub(crate) minimize: CostType,
    pub(crate) best_flops: u64,
    pub(crate) best_size: u64,
    pub(crate) best_path: Vec<ContractionIndex>,
    best_progress: HashMap<usize, u64>,
}

struct SimpleChooser;

impl RNGChooser for SimpleChooser {
    fn choose<R: Rng + ?Sized>(
        &self,
        queue: &mut BinaryHeap<Candidate>,
        remaining_tensors: &HashMap<u64, usize>,
        _nbranch: usize,
        mut _temperature: f64,
        _rel_temperature: bool,
        _rng: &mut R,
    ) -> Option<Candidate> {
        if let Some(Candidate {
            flop_cost,
            size_cost,
            parent_ids: (k1, k2),
            child_id,
        }) = queue.pop()
        {
            if !remaining_tensors.values().any(|&x| x == k1)
                && !remaining_tensors.values().any(|&x| x == k2)
            {
                return None;
            }
            return Some(Candidate {
                flop_cost,
                size_cost,
                parent_ids: (k1, k2),
                child_id,
            });
        }
        None
    }
}

impl<'a> Greedy<'a> {
    pub fn new(tn: &'a Tensor, minimize: CostType) -> Self {
        Self {
            tn,
            minimize,
            best_flops: u64::MAX,
            best_size: u64::MAX,
            best_path: Vec::new(),
            best_progress: HashMap::<usize, u64>::new(),
        }
    }

    /// The default heuristic cost, corresponding to the total reduction in
    /// memory of performing a contraction.
    pub(crate) fn cost_memory_removed(
        size12: i64,
        size1: i64,
        size2: i64,
        _k12: &Tensor,
        _k1: &Tensor,
        _k2: &Tensor,
    ) -> i64 {
        size12 - size1 - size2
    }

    /// Con cost, corresponding to the total reduction in
    /// memory of performing a contraction.
    pub(crate) fn cost_communication(
        _size12: i64,
        size1: i64,
        _size2: i64,
        _k12: &Tensor,
        _k1: &Tensor,
        _k2: &Tensor,
    ) -> i64 {
        size1
    }

    /// Returns Tensor obtained after contracting k1 and k2.
    fn get_candidate(
        output: &Tensor,
        edge_tensor_counts: &HashMap<usize, HashSet<usize>>,
        k1: &Tensor,
        k2: &Tensor,
    ) -> Tensor {
        let either = k1 | k2;
        let two = k1 & k2;
        let one = &either - &two;

        let ref3 = if let Some(ref_count_3) = edge_tensor_counts.get(&3) {
            Tensor::new(ref_count_3.iter().cloned().collect_vec())
        } else {
            Tensor::new(vec![])
        };

        let ref2 = if let Some(ref_count_2) = edge_tensor_counts.get(&2) {
            Tensor::new(ref_count_2.iter().cloned().collect_vec())
        } else {
            Tensor::new(vec![])
        };
        // Don't consider uncontracted dimensions
        &(&(&either & output) | &(&two & &ref3)) | &(&one & &ref2)
    }

    fn update_ref_counts(
        dim_to_tensors: &HashMap<usize, Vec<Tensor>>,
        dim_tensor_counts: &mut HashMap<usize, HashSet<usize>>,
        dims: &Tensor,
    ) {
        for &dim in dims.legs().iter() {
            let count = dim_to_tensors[&dim].len();
            if count <= 1 {
                dim_tensor_counts.entry(2).and_modify(|e| {
                    e.remove(&dim);
                });
                dim_tensor_counts.entry(3).and_modify(|e| {
                    e.remove(&dim);
                });
            } else if count == 2 {
                dim_tensor_counts.entry(2).and_modify(|e| {
                    e.insert(dim);
                });
                dim_tensor_counts.entry(3).and_modify(|e| {
                    e.remove(&dim);
                });
            } else {
                dim_tensor_counts.entry(2).and_modify(|e| {
                    e.insert(dim);
                });
                dim_tensor_counts.entry(3).and_modify(|e| {
                    e.insert(dim);
                });
            }
        }
    }

    /// Greedily finds cheapest contractions based on input `choice_fn` and `cost_fn`.
    /// This function relies on the fact that [`Tensor`] hash depends only on leg ids
    pub(crate) fn ssa_greedy_optimize<R>(
        &self,
        inputs: &[Tensor],
        output_dims: &Tensor,
        choice_fn: impl RNGChooser,
        cost_fn: Box<CostFnType>,
        rng: &mut R,
    ) -> Vec<ContractionIndex>
    where
        R: ?Sized + Rng,
    {
        const TEMPERATURE: f64 = 0.3;
        const NBRANCH: usize = 5;
        const REL_TEMPERATURE: bool = true;
        // Keeps track of remaining vectors, mapping between Vector of tensor leg ids to ssa number
        let mut next_ssa_id = inputs.len();
        let (
            mut remaining_tensors,
            mut ssa_id_to_tensor,
            mut scalar_tensors,
            mut ssa_path,
            mut next_ssa_id,
        ) = populate_remaining_tensors(inputs, &mut next_ssa_id);

        let mut edge_to_tensors = populate_edge_to_tensors(inputs, &remaining_tensors, output_dims);
        let mut edge_tensor_counts = populate_edge_tensor_counts(&edge_to_tensors);

        // Maps tensor ssa_id to size
        let mut tensor_mem_size: HashMap<u64, u64> = HashMap::from_iter(
            ssa_id_to_tensor
                .values()
                .map(|tensor| (calculate_hash(tensor), tensor.size())),
        );

        let mut queue = BinaryHeap::new();
        // Fill queue with all possible contraction combinations of contractions
        for (_edge, connected_tensors) in edge_to_tensors.iter_mut() {
            connected_tensors.sort_unstable_by_key(|a| a.legs().len());
            // Loop over all but the last entry
            for (i, k1) in connected_tensors[0..connected_tensors.len() - 1]
                .iter()
                .enumerate()
            {
                let k1_hash = calculate_hash(k1);
                // Get all possible unconsidered combinations
                let k2s = connected_tensors[(i + 1)..].iter();
                for k2 in k2s {
                    let k2_hash = calculate_hash(k2);
                    let k12 = Greedy::get_candidate(output_dims, &edge_tensor_counts, k1, k2);
                    let k12_hash = calculate_hash(&k12);
                    tensor_mem_size.entry(k12_hash).or_insert(k12.size());
                    ssa_id_to_tensor.entry(next_ssa_id).or_insert(k12.clone());

                    let size_cost = cost_fn(
                        tensor_mem_size[&k12_hash] as i64,
                        tensor_mem_size[&k1_hash] as i64,
                        tensor_mem_size[&k2_hash] as i64,
                        &k12,
                        k1,
                        k2,
                    );

                    let mut id1 = remaining_tensors[&k1_hash];
                    let mut id2 = remaining_tensors[&k2_hash];

                    if id1 > id2 {
                        (id1, id2) = (id2, id1);
                    }

                    queue.push(Candidate {
                        flop_cost: 0,
                        size_cost,
                        parent_ids: (id1, id2),
                        child_id: next_ssa_id,
                    });
                    next_ssa_id += 1;
                }
            }
        }

        // Start going through all possible contraction combinations
        while !queue.is_empty() {
            // Choose a candidate with lowest cost
            let candidate = choice_fn.choose(
                &mut queue,
                &remaining_tensors,
                NBRANCH,
                TEMPERATURE,
                REL_TEMPERATURE,
                rng,
            );
            let Some(Candidate {
                flop_cost: 0,
                size_cost: _cost,
                parent_ids: (id1, id2),
                child_id,
            }) = candidate
            else {
                continue;
            };

            // Get k1, k2 and k12 tensors if possible, removing them from ssa_id_to_tensor
            // If either k1 or k2 is not present, the contraction is invalid and avoided.
            let (k1_hash, k1) = if let Some(k1) = ssa_id_to_tensor.remove(&id1) {
                (calculate_hash(&k1), k1)
            } else {
                continue;
            };
            let (k2_hash, k2) = if let Some(k2) = ssa_id_to_tensor.remove(&id2) {
                (calculate_hash(&k2), k2)
            } else {
                // If k1 is present, add it back to ssa_id_tensor as contraction is invalid
                ssa_id_to_tensor.entry(id1).or_insert(k1);
                continue;
            };

            // k12 must be in ssa_id_to_tensor if both k1 and k2 are present
            let k12 = ssa_id_to_tensor.remove(&child_id).unwrap();
            let k12_hash = calculate_hash(&k12);

            // Removing k1 from edge_to_tensors
            for &dim in (&k1 - output_dims).legs().iter() {
                edge_to_tensors.entry(dim).and_modify(|e| {
                    if let Some(index) = e.iter().position(|x| x.legs() == k1.legs()) {
                        e.remove(index);
                    }
                });
            }

            // Removing k2 from edge_to_tensors
            for &dim in (&k2 - output_dims).legs().iter() {
                edge_to_tensors.entry(dim).and_modify(|e| {
                    if let Some(index) = e.iter().position(|x| x.legs() == k2.legs()) {
                        e.remove(index);
                    }
                });
            }

            // remove contracted tensors
            remaining_tensors.remove(&k1_hash);
            remaining_tensors.remove(&k2_hash);

            ssa_path.push((id1, id2, child_id));

            // Hash only considers legs, this actively finds an inner product
            if let Some(x) = remaining_tensors.remove(&k12_hash) {
                // Greedily perform inner products first
                ssa_path.push((min(x, child_id), max(x, child_id), next_ssa_id));
                scalar_tensors.push(next_ssa_id);
                next_ssa_id += 1;
                continue;
            } else {
                for &dim in (&k12 - output_dims).legs().iter() {
                    edge_to_tensors
                        .entry(dim)
                        .and_modify(|e| e.push(k12.clone()));
                }
            }

            // add newly output tensor to remaining tensors
            remaining_tensors
                .entry(calculate_hash(&k12))
                .or_insert_with(|| child_id);

            Greedy::update_ref_counts(
                &edge_to_tensors,
                &mut edge_tensor_counts,
                &(&(&k1 | &k2) - output_dims),
            );

            tensor_mem_size
                .entry(k12_hash)
                .or_insert_with(|| k12.size());

            //Find new candidate contractions.
            let k1 = k12;
            let k1_hash = k12_hash;

            let mut k2s = Vec::new();

            // for each dimension in output tensor that will be contracted in the future, find respective contracted tensors.
            for dim in (&k1 - output_dims).legs().iter() {
                for k2 in edge_to_tensors[dim].iter() {
                    // do not consider contracting with self. Inner products already removed
                    if calculate_hash(&k2) != calculate_hash(&k1) {
                        k2s.push(k2);
                    }
                }
            }
            if !k2s.is_empty() {
                for k2 in k2s {
                    let k2_hash = calculate_hash(k2);
                    let k12 = Greedy::get_candidate(output_dims, &edge_tensor_counts, &k1, k2);
                    let k12_hash = calculate_hash(&k12);
                    tensor_mem_size
                        .entry(k12_hash)
                        .or_insert_with(|| k12.size());

                    let size_cost = cost_fn(
                        tensor_mem_size[&k12_hash] as i64,
                        tensor_mem_size[&k1_hash] as i64,
                        tensor_mem_size[&k2_hash] as i64,
                        &k12,
                        &k1,
                        k2,
                    );
                    ssa_id_to_tensor.entry(next_ssa_id).or_insert_with(|| k12);

                    let mut id1 = remaining_tensors[&k1_hash];
                    let mut id2 = remaining_tensors[&k2_hash];

                    if id1 > id2 {
                        (id2, id1) = (id1, id2);
                    }

                    queue.push(Candidate {
                        flop_cost: 0,
                        size_cost,
                        parent_ids: (id1, id2),
                        child_id: next_ssa_id,
                    });
                    next_ssa_id += 1;
                }
            }
            ssa_id_to_tensor.entry(child_id).or_insert_with(|| k1);
        }
        assert!(queue.is_empty());
        for (_key, ssa_id) in remaining_tensors {
            let k12_tensor = ssa_id_to_tensor[&ssa_id].clone();
            let tensor_size = (&k12_tensor & output_dims).size() as i64;
            if tensor_size > 0 {
                let candidate = Candidate {
                    flop_cost: 0,
                    size_cost: tensor_size,
                    parent_ids: (ssa_id, 0),
                    child_id: 0,
                };
                queue.push(candidate);
            }
        }

        while !queue.is_empty() {
            if queue.len() == 1 {
                break;
            }
            let Some(Candidate {
                flop_cost: _flop_cost,
                size_cost: _cost,
                parent_ids: (ssa_id1, _id1),
                child_id: _child_id,
            }) = queue.pop()
            else {
                continue;
            };
            let Some(Candidate {
                flop_cost: _flop_cost,
                size_cost: _cost,
                parent_ids: (ssa_id2, _id2),
                child_id: _child_id,
            }) = queue.pop()
            else {
                continue;
            };
            let k1 = ssa_id_to_tensor.remove(&ssa_id1).unwrap();
            let k2 = ssa_id_to_tensor.remove(&ssa_id2).unwrap();
            ssa_path.push((min(ssa_id1, ssa_id2), max(ssa_id1, ssa_id2), next_ssa_id));
            let k12 = &(&k1 | &k2) & output_dims;
            let cost = k12.size() as i64;

            queue.push(Candidate {
                flop_cost: 0,
                size_cost: cost,
                parent_ids: (next_ssa_id, 0),
                child_id: 0,
            });
            ssa_id_to_tensor.try_insert(next_ssa_id, k12).unwrap();
            next_ssa_id += 1;
        }
        if !scalar_tensors.is_empty() {
            let mut latest_scalar = scalar_tensors[0];
            // let last_tensor_position = ssa_path.len() - 1;
            // Multiply the various scalar results together
            for &scalar_id in scalar_tensors[1..].iter() {
                ssa_path.push((
                    min(latest_scalar, scalar_id),
                    max(latest_scalar, scalar_id),
                    next_ssa_id,
                ));
                latest_scalar = next_ssa_id;
                next_ssa_id += 1;
            }
            // Perform final scalar multiplication with final tensor
            let Some(Candidate {
                flop_cost: _flop_cost,
                size_cost: _cost,
                parent_ids: (last_tensor, _id1),
                child_id: _child_id,
            }) = queue.pop()
            else {
                let ssa_path = ssa_ordering(&ssa_path, inputs.len());
                validate_path(&ssa_path);
                return ssa_path;
            };
            if !scalar_tensors.contains(&last_tensor) {
                ssa_path.push((
                    min(last_tensor, latest_scalar),
                    max(last_tensor, latest_scalar),
                    next_ssa_id,
                ));
            }
        }
        let ssa_path = ssa_ordering(&ssa_path, inputs.len());
        validate_path(&ssa_path);

        ssa_path
    }
}

fn populate_remaining_tensors(
    inputs: &[Tensor],
    next_ssa_id: &mut usize,
) -> (
    HashMap<u64, usize>,
    HashMap<usize, Tensor>,
    Vec<usize>,
    Vec<(usize, usize, usize)>,
    usize,
) {
    let mut ssa_path = Vec::new();
    let mut remaining_tensors = HashMap::new();
    let mut ssa_id_to_tensor = HashMap::new();

    let mut scalar_tensors = Vec::new();

    for (ssa_id, v) in inputs.iter().enumerate() {
        let tensor_hash = calculate_hash(v);
        ssa_id_to_tensor.entry(ssa_id).or_insert_with(|| v.clone());
        // Greedily perform inner products first
        if let Some(x) = remaining_tensors.remove(&tensor_hash) {
            ssa_path.push((x, ssa_id, *next_ssa_id));
            scalar_tensors.push(*next_ssa_id);
            ssa_id_to_tensor.remove(&x);
            ssa_id_to_tensor.remove(&ssa_id);
            *next_ssa_id += 1;
        } else {
            remaining_tensors.insert(tensor_hash, ssa_id);
        }
    }
    (
        remaining_tensors,
        ssa_id_to_tensor,
        scalar_tensors,
        ssa_path,
        *next_ssa_id,
    )
}

fn populate_edge_to_tensors(
    inputs: &[Tensor],
    remaining_tensors: &HashMap<u64, usize>,
    output_dims: &Tensor,
) -> HashMap<usize, Vec<Tensor>> {
    // Dictionary that maps leg id to tensor
    let remaining_inputs = remaining_tensors
        .values()
        .map(|&e| inputs.get(e).unwrap())
        .collect::<Vec<_>>();
    let mut bond_dim_to_tensors = HashMap::<usize, Vec<Tensor>>::new();
    for key in remaining_inputs.into_iter() {
        for dim in (key - output_dims).legs().iter() {
            bond_dim_to_tensors
                .entry(*dim)
                .or_default()
                .push(key.clone());
        }
    }
    bond_dim_to_tensors
}

fn populate_edge_tensor_counts(
    bond_dim_to_tensors: &HashMap<usize, Vec<Tensor>>,
) -> HashMap<usize, HashSet<usize>> {
    // Get dims that are contracted
    let mut bond_dim_tensor_counts: HashMap<usize, HashSet<usize>> = HashMap::new();
    for i in 2..=3 {
        for (bond_dim, tensor_legs) in bond_dim_to_tensors.iter() {
            if tensor_legs.len() >= i {
                bond_dim_tensor_counts
                    .entry(i)
                    .or_default()
                    .insert(*bond_dim);
            }
        }
    }
    bond_dim_tensor_counts
}

// Assume one-level of parallelism
impl<'a> OptimizePath for Greedy<'a> {
    fn optimize_path(&mut self) {
        if self.tn.tensors().len() == 1 {
            // Perform a single contraction to match output shape.
            self.best_flops = 0;
            self.best_size = 0;
            self.best_path = vec![];
            return;
        }
        let mut inputs: Vec<Tensor> = self.tn.tensors().clone();
        let mut rng: StdRng = StdRng::seed_from_u64(24);
        for (index, input_tensor) in inputs.iter_mut().enumerate() {
            if input_tensor.is_composite() {
                let external_legs = input_tensor.external_edges();
                let path = self.ssa_greedy_optimize(
                    input_tensor.tensors(),
                    &Tensor::new(external_legs.clone()),
                    SimpleChooser,
                    Box::new(&Greedy::cost_memory_removed),
                    &mut rng,
                );
                if !path.is_empty() {
                    let ssa_path = ssa_replace_ordering(&path, input_tensor.tensors().len());
                    self.best_path.push(ContractionIndex::Path(index, ssa_path));
                }
                input_tensor.set_legs(external_legs);
            }
        }

        // Vector of output leg ids
        let output_dims = Tensor::new(self.tn.external_edges().clone());
        // Start considering communication here!
        self.best_path.append(&mut self.ssa_greedy_optimize(
            &inputs,
            &output_dims,
            SimpleChooser,
            Box::new(&Greedy::cost_memory_removed),
            &mut rng,
        ));
        let (op_cost, mem_cost) =
            contract_path_cost(self.tn.tensors(), &self.get_best_replace_path());
        self.best_size = mem_cost;
        self.best_flops = op_cost;
    }

    fn get_best_flops(&self) -> u64 {
        self.best_flops
    }

    fn get_best_size(&self) -> u64 {
        self.best_size
    }

    fn get_best_path(&self) -> &Vec<ContractionIndex> {
        &self.best_path
    }

    fn get_best_replace_path(&self) -> Vec<ContractionIndex> {
        ssa_replace_ordering(&self.best_path, self.tn.tensors().len())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::hash::Hash;
    use std::iter::zip;

    use crate::contractionpath::paths::greedy::Greedy;
    use crate::contractionpath::paths::CostType;
    use crate::contractionpath::paths::OptimizePath;
    use crate::path;
    use crate::tensornetwork::create_tensor_network;
    use crate::tensornetwork::tensor::Tensor;

    use super::populate_edge_to_tensors;
    use super::populate_remaining_tensors;

    fn setup_simple() -> Tensor {
        create_tensor_network(
            vec![
                Tensor::new(vec![4, 3, 2]),
                Tensor::new(vec![0, 1, 3, 2]),
                Tensor::new(vec![4, 5, 6]),
            ],
            &[(0, 5), (1, 2), (2, 6), (3, 8), (4, 1), (5, 3), (6, 4)].into(),
            None,
        )
    }

    fn setup_complex() -> Tensor {
        create_tensor_network(
            vec![
                Tensor::new(vec![4, 3, 2]),
                Tensor::new(vec![0, 1, 3, 2]),
                Tensor::new(vec![4, 5, 6]),
                Tensor::new(vec![6, 8, 9]),
                Tensor::new(vec![10, 8, 9]),
                Tensor::new(vec![5, 1, 0]),
            ],
            &[
                (0, 27),
                (1, 18),
                (2, 12),
                (3, 15),
                (4, 5),
                (5, 3),
                (6, 18),
                (7, 22),
                (8, 45),
                (9, 65),
                (10, 5),
                (11, 17),
            ]
            .into(),
            None,
        )
    }

    fn setup_complex_simple() -> Tensor {
        create_tensor_network(
            vec![
                Tensor::new(vec![4, 3, 2]),
                Tensor::new(vec![0, 1, 3, 2]),
                Tensor::new(vec![4, 5, 6]),
                Tensor::new(vec![6, 8, 9]),
                Tensor::new(vec![10, 8, 9]),
                Tensor::new(vec![5, 1, 0]),
            ],
            &[
                (0, 5),
                (1, 2),
                (2, 6),
                (3, 8),
                (4, 1),
                (5, 3),
                (6, 4),
                (7, 22),
                (8, 45),
                (9, 65),
                (10, 5),
                (11, 17),
            ]
            .into(),
            None,
        )
    }

    fn setup_simple_inner_product() -> Tensor {
        create_tensor_network(
            vec![
                Tensor::new(vec![4, 3, 2]),
                Tensor::new(vec![4, 3, 2]),
                Tensor::new(vec![0, 1, 5]),
                Tensor::new(vec![1, 6]),
            ],
            &[(0, 5), (1, 2), (2, 6), (3, 8), (4, 1), (5, 3), (6, 4)].into(),
            None,
        )
    }

    fn setup_simple_outer_product() -> Tensor {
        create_tensor_network(
            vec![
                Tensor::new(vec![0]),
                Tensor::new(vec![1]),
                Tensor::new(vec![2]),
            ],
            &[(0, 3), (1, 2), (2, 2)].into(),
            None,
        )
    }

    fn setup_complex_outer_product() -> Tensor {
        create_tensor_network(
            vec![
                Tensor::new(vec![0]),
                Tensor::new(vec![0]),
                Tensor::new(vec![1]),
                Tensor::new(vec![1]),
            ],
            &[(0, 5), (1, 4)].into(),
            None,
        )
    }

    fn map_zip<'a, K, V, T>(
        a: &'a HashMap<K, V>,
        b: &'a HashMap<K, T>,
    ) -> impl Iterator<Item = (&'a K, (&'a V, &'a T))>
    where
        K: Eq + Hash,
    {
        assert_eq!(a.len(), b.len());
        a.iter().map(|(k, v)| (k, (v, &b[k])))
    }

    #[test]
    fn test_populate_remaining_tensors() {
        let tn = setup_simple_inner_product();
        let tensors = tn.tensors();
        let mut next_ssa_id = tensors.len();
        let (remaining_tensors, ssa_id_to_tensor, scalar_tensors, ssa_path, next_ssa_id) =
            populate_remaining_tensors(tensors, &mut next_ssa_id);
        let bond_dims = HashMap::from([(0, 5), (6, 4), (3, 8), (2, 6), (1, 2), (5, 3), (4, 1)]);
        let ref_remaining_tensors =
            HashMap::from([(8653979201402620513, 3), (13850888498708788536, 2)]);
        let mut t1 = Tensor::new(vec![0, 1, 5]);
        let mut t2 = Tensor::new(vec![1, 6]);
        t1.insert_bond_dims(&bond_dims);
        t2.insert_bond_dims(&bond_dims);
        let ref_ssa_id_to_tensor = HashMap::from([(2, t1), (3, t2)]);
        let ref_scalar_tensors = vec![4];
        let ref_ssa_path = vec![(0, 1, 4)];
        let ref_next_ssa_id = 5;

        assert_eq!(remaining_tensors, ref_remaining_tensors);
        for (_, (t1, t2)) in map_zip(&ssa_id_to_tensor, &ref_ssa_id_to_tensor) {
            assert_eq!(t1.legs(), t2.legs());
        }
        assert_eq!(scalar_tensors, ref_scalar_tensors);
        assert_eq!(ssa_path, ref_ssa_path);
        assert_eq!(next_ssa_id, ref_next_ssa_id);
    }

    #[test]
    fn test_populate_edge_to_tensors() {
        let tn = setup_simple_inner_product();
        let tensors = tn.tensors();

        let remaining_tensors =
            HashMap::from([(8653979201402620513, 3), (13850888498708788536, 2)]);
        let output_dims = Tensor::default();
        let edge_to_tensors = populate_edge_to_tensors(tensors, &remaining_tensors, &output_dims);

        let bond_dims = HashMap::from([(0, 5), (6, 4), (3, 8), (2, 6), (1, 2), (5, 3), (4, 1)]);
        let mut t1 = Tensor::new(vec![0, 1, 5]);
        let mut t2 = Tensor::new(vec![1, 6]);
        t1.insert_bond_dims(&bond_dims);
        t2.insert_bond_dims(&bond_dims);

        let ref_edge_to_tensors = HashMap::from([
            (0, vec![&t1]),
            (1, vec![&t1, &t2]),
            (5, vec![&t1]),
            (6, vec![&t2]),
        ]);
        for (_, (t1, t2)) in map_zip(&edge_to_tensors, &ref_edge_to_tensors) {
            let mut t1 = t1.clone();
            let mut t2 = t2.clone();
            t1.sort_by_key(|a| a.legs().len());
            t2.sort_by_key(|a| a.legs().len());
            for (legs1, legs2) in zip(&t1, t2) {
                assert_eq!(legs1.legs(), legs2.legs());
            }
        }
    }

    #[test]
    fn test_contract_order_greedy_simple() {
        let tn = setup_simple();
        let mut opt = Greedy::new(&tn, CostType::Flops);
        opt.optimize_path();

        assert_eq!(opt.best_flops, 600);
        assert_eq!(opt.best_size, 538);
        assert_eq!(opt.best_path, path![(0, 1), (2, 3)]);
        assert_eq!(opt.get_best_replace_path(), path![(0, 1), (2, 0)]);
    }

    #[test]
    fn test_contract_order_greedy_simple_inner() {
        let tn = setup_simple_inner_product();
        let mut opt = Greedy::new(&tn, CostType::Flops);
        opt.optimize_path();

        assert_eq!(opt.best_flops, 228);
        assert_eq!(opt.best_size, 121);
        assert_eq!(opt.best_path, path![(0, 1), (2, 3), (4, 5)]);
        assert_eq!(opt.get_best_replace_path(), path![(0, 1), (2, 3), (0, 2)]);
    }

    #[test]
    fn test_contract_order_greedy_simple_outer() {
        let tn = setup_simple_outer_product();
        let mut opt = Greedy::new(&tn, CostType::Flops);
        opt.optimize_path();

        assert_eq!(opt.best_flops, 16);
        assert_eq!(opt.best_size, 19);
        assert_eq!(opt.best_path, path![(1, 2), (0, 3)]);
        assert_eq!(opt.get_best_replace_path(), path![(1, 2), (0, 1)]);
    }

    #[test]
    fn test_contract_order_greedy_complex_outer() {
        let tn = setup_complex_outer_product();
        let mut opt = Greedy::new(&tn, CostType::Flops);
        opt.optimize_path();

        assert_eq!(opt.best_flops, 10);
        assert_eq!(opt.best_size, 11);
        assert_eq!(opt.best_path, path![(0, 1), (2, 3), (4, 5)]);
        assert_eq!(opt.get_best_replace_path(), path![(0, 1), (2, 3), (0, 2)]);
    }

    #[test]
    fn test_contract_order_greedy_complex() {
        let tn = setup_complex();
        let mut opt = Greedy::new(&tn, CostType::Flops);
        opt.optimize_path();

        assert_eq!(opt.best_flops, 529815);
        assert_eq!(opt.best_size, 89478);
        assert_eq!(opt.best_path, path![(1, 5), (3, 4), (0, 6), (2, 7), (8, 9)]);
        assert_eq!(
            opt.get_best_replace_path(),
            path![(1, 5), (3, 4), (0, 1), (2, 3), (0, 2)]
        );
    }
}
