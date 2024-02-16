use std::{
    cmp::{max, min},
    collections::{BinaryHeap, HashMap, HashSet},
};

use itertools::Itertools;
use rand::{rngs::StdRng, Rng, SeedableRng};

use crate::{
    contractionpath::{
        candidates::Candidate, contraction_cost::contract_path_cost, ssa_replace_ordering,
    },
    pair,
    tensornetwork::tensor::Tensor,
    types::{calculate_hash, ContractionIndex},
};

use super::{validate_path, ChoiceFnType, CostFnType, CostType, OptimizePath};

pub struct Greedy<'a> {
    pub(crate) tn: &'a Tensor,
    minimize: CostType,
    pub(crate) best_flops: u64,
    pub(crate) best_size: u64,
    pub(crate) best_path: Vec<ContractionIndex>,
    best_progress: HashMap<usize, u64>,
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

    pub(crate) fn _simple_chooser<R: Rng + ?Sized>(
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
            parent_ids,
            parent_tensors: Some((k1, k2)),
            child_id,
            child_tensor,
        }) = queue.pop()
        {
            if !remaining_tensors.contains_key(&calculate_hash::<Tensor>(&k1))
                || !remaining_tensors.contains_key(&calculate_hash::<Tensor>(&k2))
            {
                return None;
            }
            return Some(Candidate {
                flop_cost,
                size_cost,
                parent_ids,
                parent_tensors: Some((k1, k2)),
                child_id,
                child_tensor,
            });
        }
        None
    }

    /// The default heuristic cost, corresponding to the total reduction in
    /// memory of performing a contraction.
    pub(crate) fn _cost_memory_removed(
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

    #[allow(clippy::too_many_arguments)]
    fn push_candidate(
        output: &Tensor,
        remaining_tensors: &HashMap<u64, usize>,
        tensor_mem_size: &HashMap<u64, u64>,
        dim_tensor_counts: &HashMap<usize, HashSet<usize>>,
        k1: &Tensor,
        k2s: Vec<&Tensor>,
        queue: &mut BinaryHeap<Candidate>,
        cost_function: &CostFnType,
    ) {
        let mut candidates = Vec::new();
        for k2 in k2s {
            candidates.push(Greedy::get_candidate(
                output,
                remaining_tensors,
                tensor_mem_size,
                dim_tensor_counts,
                k1,
                k2,
                cost_function,
            ));
        }
        for candidate in candidates {
            queue.push(candidate);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn get_candidate<'b>(
        output: &Tensor,
        remaining_tensors: &HashMap<u64, usize>,
        tensor_mem_size: &HashMap<u64, u64>,
        dim_tensor_counts: &HashMap<usize, HashSet<usize>>,
        mut k1: &'b Tensor,
        mut k2: &'b Tensor,
        cost_function: &CostFnType,
    ) -> Candidate {
        let k1_hash = calculate_hash(k1);
        let k2_hash = calculate_hash(k2);

        let either = k1 | k2;
        let two = k1 & k2;
        let one = &either - &two;

        let ref3 = if let Some(ref_count_3) = dim_tensor_counts.get(&3) {
            Tensor::new(ref_count_3.iter().cloned().collect_vec())
        } else {
            Tensor::new(vec![])
        };

        let ref2 = if let Some(ref_count_2) = dim_tensor_counts.get(&2) {
            Tensor::new(ref_count_2.iter().cloned().collect_vec())
        } else {
            Tensor::new(vec![])
        };
        // Don't consider uncontracted dimensions
        let k12 = &(&(&either & output) | &(&two & &ref3)) | &(&one & &ref2);
        let size_k12 = k12.size();

        let cost = cost_function(
            size_k12 as i64,
            tensor_mem_size[&k1_hash] as i64,
            tensor_mem_size[&k2_hash] as i64,
            &k12,
            k1,
            k2,
        );
        let mut id1 = remaining_tensors[&k1_hash];
        let mut id2 = remaining_tensors[&k2_hash];

        if id1 > id2 {
            (k1, k2) = (k2, k1);
            (id1, id2) = (id2, id1);
        }

        Candidate {
            flop_cost: 0,
            size_cost: cost,
            parent_ids: (id1, id2),
            parent_tensors: Some((k1.clone(), k2.clone())),
            child_id: 0,
            child_tensor: Some(k12),
        }
    }

    fn update_ref_counts(
        dim_to_tensors: &HashMap<usize, Vec<Tensor>>,
        dim_tensor_counts: &mut HashMap<usize, HashSet<usize>>,
        dims: &Tensor,
    ) {
        for &dim in dims.get_legs().iter() {
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

    /// Greedily finds cheapest contractions based on input choice_fn and cost_fn.
    /// This function relies on the fact that 'Tensor' hash depends only on leg ids
    pub(crate) fn ssa_greedy_optimize(
        &self,
        inputs: &[Tensor],
        output_dims: &Tensor,
        choice_fn: Box<ChoiceFnType>,
        cost_fn: Box<CostFnType>,
    ) -> Vec<ContractionIndex> {
        let mut ssa_path = Vec::new();

        // Keeps track of remaining vectors, mapping between Vector of tensor leg ids to ssa number
        // Clone here to avoid mutating HashMap keys
        let mut next_ssa_id: usize = inputs.len();
        let mut remaining_tensors = HashMap::new();
        let mut hash_to_tensor = HashMap::new();

        populate_remaining_tensors(
            inputs,
            &mut remaining_tensors,
            &mut hash_to_tensor,
            &mut ssa_path,
            &mut next_ssa_id,
        );

        let mut dim_to_tensors = populate_dim_to_tensors(inputs, output_dims);
        let mut dim_tensor_counts = populate_dim_tensor_counts(&dim_to_tensors);

        // Maps tensor legs to size
        let mut tensor_mem_size = HashMap::from_iter(inputs.iter().map(|legs| {
            let size = legs.size();
            (calculate_hash(legs), size)
        }));

        let mut queue = BinaryHeap::new();
        for (_dim, keys) in dim_to_tensors.iter_mut() {
            keys.sort_by_key(|a| a.get_legs().len());
            // Loop over all but the last entry
            for (i, k1) in keys[0..keys.len()].iter().enumerate() {
                // Get all possible unconsidered combinations
                let k2s = keys[(i + 1)..keys.len()].iter().collect_vec();
                Greedy::push_candidate(
                    output_dims,
                    &remaining_tensors,
                    &tensor_mem_size,
                    &dim_tensor_counts,
                    k1,
                    k2s,
                    &mut queue,
                    &Greedy::_cost_memory_removed,
                );
            }
        }

        while !queue.is_empty() {
            // Choose a candidate with lowest cost
            let candidate = choice_fn(
                &mut queue,
                &remaining_tensors,
                5,
                0.3,
                true,
                &mut StdRng::from_entropy(),
            );
            let Some(Candidate {
                flop_cost: 0,
                size_cost: _cost,
                parent_ids: (_id1, _id2),
                parent_tensors: Some((k1, k2)),
                child_id: 0,
                child_tensor: Some(k12),
            }) = candidate
            else {
                continue;
            };
            let k1_hash = calculate_hash(&k1);
            let k2_hash = calculate_hash(&k2);

            let Some(ssa_id1) = remaining_tensors.get(&k1_hash) else {
                panic!("SSA ID '{:?}' missing", k1)
            };

            let Some(ssa_id2) = remaining_tensors.get(&k2_hash) else {
                panic!("SSA ID '{:?}' missing", k2)
            };

            let k12_hash = calculate_hash(&k12);

            for &dim in (&k1 - output_dims).get_legs().iter() {
                dim_to_tensors.entry(dim).and_modify(|e| {
                    if let Some(index) = e.iter().position(|x| *x == k1) {
                        e.remove(index);
                    }
                });
            }

            for &dim in (&k2 - output_dims).get_legs().iter() {
                dim_to_tensors.entry(dim).and_modify(|e| {
                    if let Some(index) = e.iter().position(|x| *x == k2) {
                        e.remove(index);
                    }
                });
            }
            ssa_path.push(pair!(*ssa_id1, *ssa_id2));

            remaining_tensors.remove(&k1_hash);
            remaining_tensors.remove(&k2_hash);
            hash_to_tensor.remove(&k1_hash);
            hash_to_tensor.remove(&k2_hash);

            if remaining_tensors.contains_key(&k12_hash) {
                // Actively perform inner products first
                ssa_path.push(pair!(remaining_tensors[&k12_hash], next_ssa_id));
                next_ssa_id += 1;
            } else {
                for &dim in (&k12 - output_dims).get_legs().iter() {
                    dim_to_tensors
                        .entry(dim)
                        .and_modify(|e| e.push(k12.clone()));
                }
            }
            remaining_tensors
                .entry(calculate_hash(&k12))
                .or_insert_with(|| next_ssa_id);
            hash_to_tensor
                .entry(calculate_hash(&k12))
                .or_insert_with(|| k12.clone());
            next_ssa_id += 1;

            Greedy::update_ref_counts(
                &dim_to_tensors,
                &mut dim_tensor_counts,
                &(&(&k1 | &k2) - output_dims),
            );

            tensor_mem_size
                .entry(k12_hash)
                .or_insert_with(|| k12.size());

            //Find new candidate contractions.
            let k1 = k12;

            let mut k2s = Vec::new();
            for dim in (&k1 - output_dims).get_legs().iter() {
                for k2 in dim_to_tensors[dim].iter() {
                    if calculate_hash(&k2) != calculate_hash(&k1) {
                        k2s.push(k2);
                    }
                }
            }
            if !k2s.is_empty() {
                Greedy::push_candidate(
                    output_dims,
                    &remaining_tensors,
                    &tensor_mem_size,
                    &dim_tensor_counts,
                    &k1,
                    k2s,
                    &mut queue,
                    &cost_fn,
                );
            }
        }

        for (key, ssa_id) in remaining_tensors {
            let k12_tensor = hash_to_tensor[&key].clone();
            let tensor_size = (&k12_tensor & output_dims).size() as i64;
            if tensor_size > 0 {
                let candidate = Candidate {
                    flop_cost: 0,
                    size_cost: tensor_size,
                    parent_ids: (ssa_id, 0),
                    parent_tensors: Some((k12_tensor, Tensor::default())),
                    child_id: 0,
                    child_tensor: None,
                };
                queue.push(candidate);
            }
        }

        while !queue.is_empty() {
            let Some(Candidate {
                flop_cost: _flop_cost,
                size_cost: _cost,
                parent_ids: (ssa_id1, _id2),
                parent_tensors: Some((k1, _k2)),
                child_id: _child_id,
                child_tensor: _child_tensor,
            }) = queue.pop()
            else {
                continue;
            };
            let Some(Candidate {
                flop_cost: _flop_cost,
                size_cost: _cost,
                parent_ids: (ssa_id2, _id2),
                parent_tensors: Some((k2, _k2)),
                child_id: _child_id,
                child_tensor: _child_tensor,
            }) = queue.pop()
            else {
                continue;
            };
            ssa_path.push(pair!(min(ssa_id1, ssa_id2), max(ssa_id1, ssa_id2)));
            let k12 = &(&k1 | &k2) & output_dims;

            let cost = k12.size() as i64;
            queue.push(Candidate {
                flop_cost: 0,
                size_cost: cost,
                parent_ids: (min(ssa_id1, ssa_id2), 0),
                parent_tensors: Some((k1.clone(), k2)),
                child_id: 0,
                child_tensor: None,
            });
        }
        validate_path(&ssa_path);

        ssa_path
    }
}

fn populate_remaining_tensors(
    inputs: &[Tensor],
    remaining_tensors: &mut HashMap<u64, usize>,
    hash_to_tensor: &mut HashMap<u64, Tensor>,
    ssa_path: &mut Vec<ContractionIndex>,
    next_ssa_id: &mut usize,
) {
    for (ssa_id, v) in inputs.iter().enumerate() {
        let tensor_hash = calculate_hash(v);
        hash_to_tensor
            .entry(tensor_hash)
            .or_insert_with(|| v.clone());
        // greedily calculate inner products
        let entry = remaining_tensors
            .entry(tensor_hash)
            .and_modify(|e| {
                *e = *next_ssa_id;
                *next_ssa_id += 1;
            })
            .or_insert_with(|| ssa_id);
        if *entry != ssa_id {
            ssa_path.push(pair!(*entry, ssa_id))
        }
    }
}

fn populate_dim_to_tensors(inputs: &[Tensor], output_dims: &Tensor) -> HashMap<usize, Vec<Tensor>> {
    // Dictionary that maps leg id to tensor
    let mut dim_to_tensors = HashMap::<usize, Vec<Tensor>>::new();
    for key in inputs.iter() {
        for dim in (key - output_dims).get_legs().iter() {
            dim_to_tensors.entry(*dim).or_default().push(key.clone());
        }
    }
    dim_to_tensors
}

fn populate_dim_tensor_counts(
    dim_to_tensors: &HashMap<usize, Vec<Tensor>>,
) -> HashMap<usize, HashSet<usize>> {
    // Get dims that are contracted
    let mut dim_tensor_counts = HashMap::<usize, HashSet<usize>>::new();
    for i in 2..=3 {
        for (dim, tensor_legs) in dim_to_tensors.iter() {
            if tensor_legs.len() >= i {
                dim_tensor_counts.entry(i).or_default().insert(*dim);
            }
        }
    }
    dim_tensor_counts
}

// Assume one-level of parallelism
impl<'a> OptimizePath for Greedy<'a> {
    fn optimize_path(&mut self) {
        if self.tn.get_tensors().len() == 1 {
            // Perform a single contraction to match output shape.
            self.best_flops = 0;
            self.best_size = 0;
            self.best_path = vec![];
            return;
        }
        let mut inputs: Vec<Tensor> = self.tn.get_tensors().clone();
        for (index, input_tensor) in inputs.iter_mut().enumerate() {
            if input_tensor.get_legs().is_empty() {
                let external_legs = input_tensor.get_external_edges().clone();
                let path = self.ssa_greedy_optimize(
                    input_tensor.get_tensors(),
                    &Tensor::new(external_legs.clone()),
                    Box::new(&Greedy::_simple_chooser),
                    Box::new(&Greedy::_cost_memory_removed),
                );
                if !path.is_empty() {
                    let ssa_path = ssa_replace_ordering(&path, input_tensor.get_tensors().len());
                    self.best_path.push(ContractionIndex::Path(index, ssa_path));
                }
                input_tensor.set_legs(external_legs);
            }
        }

        // Vector of output leg ids
        let output_dims = Tensor::new(self.tn.get_external_edges().clone());
        // Start considering communication here!
        self.best_path.append(&mut self.ssa_greedy_optimize(
            &inputs,
            &output_dims,
            Box::new(&Greedy::_simple_chooser),
            Box::new(&Greedy::_cost_memory_removed),
        ));
        let (op_cost, mem_cost) =
            contract_path_cost(self.tn.get_tensors(), &self.get_best_replace_path());
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
        ssa_replace_ordering(&self.best_path, self.tn.get_tensors().len())
    }
}

#[cfg(test)]
mod tests {
    use crate::contractionpath::paths::greedy::Greedy;
    use crate::contractionpath::paths::CostType;
    use crate::contractionpath::paths::OptimizePath;
    use crate::path;
    use crate::tensornetwork::create_tensor_network;
    use crate::tensornetwork::tensor::Tensor;

    // use rand::rngs::StdRng;
    // use rand::SeedableRng;

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
