use std::{
    cmp::{max, min},
    collections::{hash_map::Entry, BinaryHeap, HashMap, HashSet},
};

use itertools::Itertools;
use rand::{rngs::StdRng, Rng, SeedableRng};

use crate::{
    contractionpath::{
        candidates::Candidate,
        contraction_cost::{_contract_path_cost, _tensor_size},
        ssa_replace_ordering,
    },
    pair,
    tensornetwork::tensor::Tensor,
    types::ContractionIndex,
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
        remaining_tensors: &HashMap<Tensor, usize>,
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
            if !remaining_tensors.contains_key(&k1) || !remaining_tensors.contains_key(&k2) {
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
        _bond_dims: &HashMap<usize, u64>,
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
    pub(crate) fn _cost_communication(
        _bond_dims: &HashMap<usize, u64>,
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
    fn _push_candidate(
        output: &Tensor,
        bond_dims: &HashMap<usize, u64>,
        remaining: &HashMap<Tensor, usize>,
        footprints: &HashMap<Tensor, u64>,
        dim_ref_counts: &HashMap<usize, HashSet<usize>>,
        k1: &Tensor,
        k2s: Vec<&Tensor>,
        queue: &mut BinaryHeap<Candidate>,
        cost_function: &CostFnType,
    ) {
        let mut candidates = Vec::new();
        for k2 in k2s {
            candidates.push(Greedy::_get_candidate(
                output,
                bond_dims,
                remaining,
                footprints,
                dim_ref_counts,
                k1,
                k2,
                cost_function,
            ));
        }
        if true {
            for candidate in candidates {
                queue.push(candidate);
            }
        } else if let Some(min_value) = candidates.iter().cloned().min() {
            queue.push(min_value);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn _get_candidate<'b>(
        output: &Tensor,
        bond_dims: &HashMap<usize, u64>,
        remaining_tensors: &HashMap<Tensor, usize>,
        tensor_mem_size: &HashMap<Tensor, u64>,
        dim_tensor_counts: &HashMap<usize, HashSet<usize>>,
        mut k1: &'b Tensor,
        mut k2: &'b Tensor,
        cost_function: &CostFnType,
    ) -> Candidate {
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

        let k12 = &(&(&either & output) | &(&two & &ref3)) | &(&one & &ref2);

        let size_k12 = _tensor_size(&k12, bond_dims);

        let cost = cost_function(
            bond_dims,
            size_k12 as i64,
            tensor_mem_size[k1] as i64,
            tensor_mem_size[k2] as i64,
            &k12,
            k1,
            k2,
        );
        let mut id1 = remaining_tensors[k1];
        let mut id2 = remaining_tensors[k2];

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

    fn _update_ref_counts(
        dim_to_tensors: &HashMap<usize, Vec<Tensor>>,
        dim_tensor_counts: &mut HashMap<usize, HashSet<usize>>,
        dims: &Tensor,
    ) {
        for dim in dims.get_legs().iter().cloned() {
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

    pub(crate) fn _ssa_greedy_optimize(
        &self,
        inputs: &Vec<Tensor>,
        output_dims: &Tensor,
        bond_dims: &HashMap<usize, u64>,
        choice_fn: Box<ChoiceFnType>,
        cost_fn: Box<CostFnType>,
    ) -> Vec<ContractionIndex> {
        let mut ssa_path = Vec::new();

        // Keeps track of remaining vectors, mapping between Vector of tensor leg ids to ssa number
        // Clone here to avoid mutating HashMap keys
        let mut remaining_tensors: HashMap<Tensor, usize> = HashMap::<Tensor, usize>::new();
        let mut next_ssa_id: usize = inputs.len();

        for (ssa_id, v) in inputs.iter().enumerate() {
            if remaining_tensors.contains_key(v) {
                // greedily compute inner products
                ssa_path.push(pair!(remaining_tensors[v], ssa_id));
                remaining_tensors.entry(v.clone()).or_insert(next_ssa_id);
                next_ssa_id += 1;
            } else {
                remaining_tensors.entry(v.clone()).or_insert(ssa_id);
            }
        }

        // Dictionary that maps leg id to tensor
        let mut dim_to_tensors = HashMap::<usize, Vec<Tensor>>::new();
        for key in remaining_tensors.keys() {
            for dim in (key - output_dims).get_legs().iter() {
                dim_to_tensors
                    .entry(*dim)
                    .and_modify(|entry| entry.push(key.clone()))
                    .or_insert(vec![key.clone()]);
            }
        }

        // Get dims that are contracted
        let mut dim_tensor_counts = HashMap::<usize, HashSet<usize>>::new();
        for i in 2..=3 {
            for (dim, tensor_legs) in dim_to_tensors.iter() {
                if tensor_legs.len() >= i && !output_dims.get_legs().contains(dim) {
                    dim_tensor_counts
                        .entry(i)
                        .and_modify(|entry| {
                            entry.insert(*dim);
                        })
                        .or_default();
                }
            }
        }

        // Maps tensor legs to size
        // Clone here to avoid mutating HashMap keys
        let mut tensor_mem_size = HashMap::from_iter(inputs.iter().map(|legs| {
            let size = _tensor_size(legs, bond_dims);
            (legs.clone(), size)
        }));

        let mut queue = BinaryHeap::new();
        for (_dim, key) in dim_to_tensors.iter() {
            let mut new_keys = key.clone();
            new_keys.sort_by_key(|a| a.get_legs().len());
            for (i, k1) in new_keys[0..new_keys.len()].iter().enumerate() {
                let k2s = new_keys[(i + 1)..new_keys.len()].iter().collect_vec();
                Greedy::_push_candidate(
                    output_dims,
                    bond_dims,
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
            let candidate = choice_fn(
                &mut queue,
                &remaining_tensors,
                0,
                0.0,
                false,
                &mut StdRng::seed_from_u64(42),
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
            let Some(ssa_id1) = remaining_tensors.get(&k1) else {
                panic!("SSA ID '{:?}' missing", k1)
            };

            let Some(ssa_id2) = remaining_tensors.get(&k2) else {
                panic!("SSA ID '{:?}' missing", k2)
            };
            // already_contracted.push(ssa_id2);

            for dim in (&k1 - output_dims).get_legs().iter().cloned() {
                dim_to_tensors.entry(dim).and_modify(|e| {
                    let index = e.iter().position(|x| *x == k1);
                    if let Some(index) = index {
                        e.remove(index);
                    }
                });
            }

            for dim in (&k2 - output_dims).get_legs().iter().cloned() {
                dim_to_tensors.entry(dim).and_modify(|e| {
                    let index = e.iter().position(|x| *x == k2);
                    if let Some(index) = index {
                        e.remove(index);
                    }
                });
            }
            ssa_path.push(pair!(*ssa_id1, *ssa_id2));

            if let Entry::Occupied(o) = remaining_tensors.entry(k1.clone()) {
                o.remove_entry();
            }
            if let Entry::Occupied(o) = remaining_tensors.entry(k2.clone()) {
                o.remove_entry();
            }

            if remaining_tensors.contains_key(&k12) {
                ssa_path.push(pair!(remaining_tensors[&k12], next_ssa_id));
                next_ssa_id += 1;
            } else {
                for dim in (&k12 - output_dims).get_legs().iter().cloned() {
                    dim_to_tensors
                        .entry(dim)
                        .and_modify(|e| e.push(k12.clone()));
                }
            }
            remaining_tensors.entry(k12.clone()).or_insert(next_ssa_id);
            next_ssa_id += 1;

            Greedy::_update_ref_counts(
                &dim_to_tensors,
                &mut dim_tensor_counts,
                &(&(&k1 | &k2) - output_dims),
            );

            tensor_mem_size
                .entry(k12.clone())
                .or_insert(_tensor_size(&k12, bond_dims));

            //Find new candidate contractions.
            let k1 = k12;

            let mut k2s = Vec::new();
            for dim in (&k1 - output_dims).get_legs().iter() {
                for k2 in dim_to_tensors[dim].iter() {
                    if k2 != &k1 {
                        k2s.push(k2);
                    }
                }
            }
            if !k2s.is_empty() {
                Greedy::_push_candidate(
                    output_dims,
                    bond_dims,
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
        // println!("Remaining tensors: {:?}",);

        let mut heapq = BinaryHeap::new();
        for (key, ssa_id) in remaining_tensors {
            let tensor_size = _tensor_size(&(&key & output_dims), bond_dims) as i64;
            if tensor_size > 0 {
                let candidate = Candidate {
                    flop_cost: 0,
                    size_cost: tensor_size,
                    parent_ids: (ssa_id, 0),
                    parent_tensors: Some((key, Tensor::default())),
                    child_id: 0,
                    child_tensor: None,
                };
                heapq.push(candidate);
            }
        }

        while !heapq.is_empty() {
            let Some(Candidate {
                flop_cost: _flop_cost,
                size_cost: _cost,
                parent_ids: (ssa_id1, _id2),
                parent_tensors: Some((k1, _k2)),
                child_id: _child_id,
                child_tensor: _child_tensor,
            }) = heapq.pop()
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
            }) = heapq.pop()
            else {
                continue;
            };
            ssa_path.push(pair!(min(ssa_id1, ssa_id2), max(ssa_id1, ssa_id2)));
            let k12 = &(&k1 | &k2) & output_dims;

            let cost = _tensor_size(&k12, bond_dims) as i64;
            heapq.push(Candidate {
                flop_cost: 0,
                size_cost: cost,
                parent_ids: (min(ssa_id1, ssa_id2), 0),
                parent_tensors: Some((k1.clone(), k2)),
                child_id: 0,
                child_tensor: None,
            });
        }
        let _ = validate_path(&ssa_path);
        ssa_path
    }
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
                let path = self._ssa_greedy_optimize(
                    input_tensor.get_tensors(),
                    &Tensor::new(external_legs.clone()),
                    &input_tensor.get_bond_dims(),
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
        // Dictionary that maps leg id to bond dimension
        let bond_dims = self.tn.get_bond_dims();
        // Start considering communication here!
        self.best_path.append(&mut self._ssa_greedy_optimize(
            &inputs,
            &output_dims,
            &bond_dims,
            Box::new(&Greedy::_simple_chooser),
            Box::new(&Greedy::_cost_memory_removed),
        ));
        let (op_cost, mem_cost) = _contract_path_cost(
            self.tn.get_tensors(),
            &self.get_best_replace_path(),
            &self.tn.get_bond_dims(),
        );
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

    use rand::rngs::StdRng;
    use rand::SeedableRng;

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
