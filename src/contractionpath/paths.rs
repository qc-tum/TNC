use itertools::Itertools;
use rand::rngs::ThreadRng;
use rand::thread_rng;
use rand::Rng;
use std::cmp::min;
use std::collections::hash_map::Entry;
use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::option::Option;
use std::{cmp::max, collections::HashSet};

use crate::contractionpath::{
    candidates::Candidate,
    contraction_cost::{_contract_cost, _contract_path_cost, _contract_size, _tensor_size, size},
    ssa_ordering, ssa_replace_ordering,
};
use crate::tensornetwork::tensor::Tensor;
use crate::tensornetwork::TensorNetwork;

pub trait OptimizePath {
    fn optimize_path(&mut self);

    fn get_best_path(&self) -> &Vec<(usize, usize)>;
    fn get_best_replace_path(&self) -> Vec<(usize, usize)>;
    fn get_best_flops(&self) -> u64;
    fn get_best_size(&self) -> u64;
}

pub enum CostType {
    Flops = 0,
    Size = 1,
}

/// A struct with an OptimizePath implementation that explores possible pair contractions in a depth-first manner.
pub struct BranchBound<'a> {
    tn: &'a TensorNetwork,
    nbranch: Option<u32>,
    cutoff_flops_factor: u64,
    minimize: CostType,
    best_flops: u64,
    best_size: u64,
    best_path: Vec<(usize, usize)>,
    best_progress: HashMap<usize, u64>,
    result_cache: HashMap<Vec<usize>, usize>,
    flop_cache: HashMap<usize, u64>,
    size_cache: HashMap<usize, u64>,
    tensor_cache: HashMap<usize, Tensor>,
}

impl<'a> BranchBound<'a> {
    pub fn new(
        tn: &'a TensorNetwork,
        nbranch: Option<u32>,
        cutoff_flops_factor: u64,
        minimize: CostType,
    ) -> Self {
        Self {
            tn,
            nbranch,
            cutoff_flops_factor,
            minimize,
            best_flops: u64::MAX,
            best_size: u64::MAX,
            best_path: Vec::new(),
            best_progress: HashMap::<usize, u64>::new(),
            result_cache: HashMap::<Vec<usize>, usize>::new(),
            flop_cache: HashMap::<usize, u64>::new(),
            size_cache: HashMap::<usize, u64>::new(),
            tensor_cache: HashMap::<usize, Tensor>::new(),
        }
    }

    /// Explores possible pair contractions in a depth-first
    /// recursive manner like the `optimal` approach, but with extra heuristic early pruning of branches
    /// as well sieving by `memory_limit` and the best path found so far. A rust implementation of
    /// the Python based `opt_einsum` implementation. Found at github.com/dgasmith/opt_einsum.
    fn _branch_iterate(
        &mut self,
        path: Vec<(usize, usize, usize)>,
        remaining: Vec<u32>,
        flops: u64,
        size: u64,
    ) {
        if remaining.len() == 1 {
            match self.minimize {
                CostType::Flops => {
                    if self.best_flops > flops {
                        self.best_flops = flops;
                        self.best_size = size;
                        self.best_path = ssa_ordering(&path, self.tn.get_tensors().len());
                    }
                }
                CostType::Size => {
                    if self.best_size > size {
                        self.best_flops = flops;
                        self.best_size = size;
                        self.best_path = ssa_ordering(&path, self.tn.get_tensors().len());
                    }
                }
            }
            return;
        }

        let mut assess_candidate = |i: usize, j: usize| -> Option<Candidate> {
            let flops_12: u64;
            let size_12: u64;
            let k12: usize;
            let k12_tensor: Tensor;
            let mut current_flops = flops;
            let mut current_size = size;
            if self.result_cache.contains_key(&vec![i, j]) {
                k12 = self.result_cache[&vec![i, j]];
                flops_12 = self.flop_cache[&k12];
                size_12 = self.size_cache[&k12];
                k12_tensor = self.tensor_cache[&k12].clone();
            } else {
                k12 = self.tensor_cache.len();
                flops_12 = _contract_cost(
                    &self.tensor_cache[&i],
                    &self.tensor_cache[&j],
                    self.tn.get_bond_dims(),
                );
                (k12_tensor, size_12) = _contract_size(
                    &self.tensor_cache[&i],
                    &self.tensor_cache[&j],
                    self.tn.get_bond_dims(),
                );
                self.result_cache.entry(vec![i, j]).or_insert(k12);
                self.flop_cache.entry(k12).or_insert(flops_12);
                self.size_cache.entry(k12).or_insert(size_12);
                self.tensor_cache.entry(k12).or_insert(k12_tensor.clone());
            }
            current_flops += flops_12;
            current_size = max(current_size, size_12);

            if current_flops > self.best_flops && current_size > self.best_size {
                return None;
            }
            let best_flops: u64;
            if self.best_progress.contains_key(&remaining.len()) {
                best_flops = self.best_progress[&remaining.len()];
            } else {
                best_flops = current_flops;
                self.best_progress
                    .entry(remaining.len())
                    .or_insert(current_flops);
            }

            if current_flops < best_flops {
                self.best_progress
                    .entry(remaining.len())
                    .insert_entry(current_flops);
            } else if current_flops
                > self.cutoff_flops_factor * self.best_progress[&remaining.len()]
            {
                return None;
            }

            Some(Candidate {
                flop_cost: current_flops as i64,
                size_cost: current_size as i64,
                parent_ids: (i, j),
                parent_tensors: None,
                child_id: k12,
                child_tensor: Some(k12_tensor),
            })
        };

        let mut candidates = BinaryHeap::<Candidate>::new();
        for i in remaining.iter().cloned().combinations(2) {
            let candidate = assess_candidate(i[0] as usize, i[1] as usize);
            if let Some(new_candidate) = candidate {
                candidates.push(new_candidate)
            } else {
                continue;
            }
        }
        let bi = 0;
        let mut new_remaining;
        let mut new_path: Vec<(usize, usize, usize)>;
        while self.nbranch.is_none() || bi < self.nbranch.unwrap() {
            let Some(Candidate {
                flop_cost,
                size_cost,
                parent_ids,
                parent_tensors: _parent_tensor,
                child_id,
                child_tensor: _child_tensor,
            }) = candidates.pop()
            else {
                break;
            };
            new_remaining = remaining.clone();
            new_remaining.retain(|e| *e != parent_ids.0 as u32);
            new_remaining.retain(|e| *e != parent_ids.1 as u32);
            new_remaining.insert(new_remaining.len(), child_id as u32);
            new_path = path.clone();
            new_path.push((parent_ids.0, parent_ids.1, child_id));
            BranchBound::_branch_iterate(
                self,
                new_path,
                new_remaining,
                flop_cost as u64,
                size_cost as u64,
            );
        }
    }
}

impl<'a> OptimizePath for BranchBound<'a> {
    fn optimize_path(&mut self) {
        let tensors = self.tn.get_tensors();
        if self.tn.is_empty() {
            return;
        }

        self.flop_cache.clear();
        self.size_cache.clear();

        // Get the initial space requirements for uncontracted tensors
        for (index, tensor) in tensors.iter().enumerate() {
            self.size_cache.entry(index).or_insert(size(self.tn, index));
            self.tensor_cache.entry(index).or_insert(tensor.clone());
        }

        let remaining: Vec<u32> = (0u32..self.tn.get_tensors().len() as u32).collect();
        BranchBound::_branch_iterate(self, vec![], remaining, 0, 0);
    }

    fn get_best_flops(&self) -> u64 {
        self.best_flops
    }

    fn get_best_size(&self) -> u64 {
        self.best_size
    }

    fn get_best_path(&self) -> &Vec<(usize, usize)> {
        &self.best_path
    }

    fn get_best_replace_path(&self) -> Vec<(usize, usize)> {
        ssa_replace_ordering(&self.best_path, self.tn.get_tensors().len())
    }
}

type CostFnType = dyn Fn(&HashMap<usize, u64>, i64, i64, i64, &Tensor, &Tensor, &Tensor) -> i64;
type ChoiceFnType = dyn for<'b, 'c> Fn(
    &'b mut std::collections::BinaryHeap<Candidate>,
    &'c std::collections::HashMap<Tensor, usize>,
    usize,
    f64,
    bool,
    &mut ThreadRng,
) -> Option<Candidate>;

pub struct Greedy<'a> {
    pub(crate) tn: &'a TensorNetwork,
    minimize: CostType,
    pub(crate) best_flops: u64,
    pub(crate) best_size: u64,
    pub(crate) best_path: Vec<(usize, usize)>,
    best_progress: HashMap<usize, u64>,
}

impl<'a> Greedy<'a> {
    pub fn new(tn: &'a TensorNetwork, minimize: CostType) -> Self {
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
        let out = &either & output;

        let ref2 = if let Some(ref_count_3) = dim_tensor_counts.get(&3) {
            &out | &(&two & &Tensor::new(ref_count_3.iter().cloned().collect_vec()))
        } else {
            out
        };

        let k12 = if let Some(ref_count_2) = dim_tensor_counts.get(&2) {
            &ref2 | &(&one & &Tensor::new(ref_count_2.iter().cloned().collect_vec()))
        } else {
            ref2
        };

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
        for dim in dims.iter().cloned() {
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
        &mut self,
        inputs: &Vec<Tensor>,
        output_dims: &Tensor,
        bond_dims: &HashMap<usize, u64>,
        choice_fn: Box<ChoiceFnType>,
        cost_fn: Box<CostFnType>,
    ) -> Vec<(usize, usize)> {
        let mut ssa_path = Vec::new();
        // Keeps track of remaining vectors, mapping between Vector of tensor leg ids to ssa number
        let mut remaining_tensors = HashMap::<Tensor, usize>::new();
        let mut next_ssa_id: usize = inputs.len();
        for (ssa_id, v) in inputs.iter().enumerate() {
            if remaining_tensors.contains_key(v) {
                // greedily compute inner products
                ssa_path.push((remaining_tensors[v], ssa_id));
                remaining_tensors.entry(v.clone()).or_insert(next_ssa_id);
                next_ssa_id += 1;
            } else {
                remaining_tensors.entry(v.clone()).or_insert(ssa_id);
            }
        }

        // Dictionary that maps leg id to tensor
        let mut dim_to_tensors = HashMap::<usize, Vec<Tensor>>::new();
        for key in remaining_tensors.keys() {
            for dim in (key - output_dims).iter() {
                // for dim in key.iter().filter(|e| !output.contains(e)) {
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
                if tensor_legs.len() >= i {
                    dim_tensor_counts
                        .entry(i)
                        .and_modify(|entry| {
                            entry.insert(*dim);
                        })
                        .or_insert(HashSet::new());
                }
            }
        }

        // Maps tensor to size
        let mut tensor_mem_size = HashMap::from_iter(inputs.iter().map(|legs| {
            let size = _tensor_size(legs, bond_dims);
            (legs.clone(), size)
        }));

        let mut queue = BinaryHeap::new();
        for (_dim, key) in dim_to_tensors.iter() {
            let mut new_keys = key.clone();
            new_keys.sort_by_key(|a| a.get_legs().len());
            for (i, k1) in new_keys[0..new_keys.len() - 1].iter().enumerate() {
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
                &mut thread_rng(),
            );
            let Some(Candidate {
                    flop_cost: 0,
                    size_cost: _cost,
                    parent_ids: (_id1, _id2),
                    parent_tensors: Some((k1, k2)),
                    child_id: 0,
                    child_tensor: Some(k12)
                }) = candidate else{
                    continue;
                };

            let Some(ssa_id1) = remaining_tensors.get(&k1) else{
                panic!("SSA ID '{:?}' missing", k1)
            };

            let Some(ssa_id2) = remaining_tensors.get(&k2) else{
                panic!("SSA ID '{:?}' missing", k2)
            };

            for dim in (&k1 - output_dims).iter().cloned() {
                dim_to_tensors.entry(dim).and_modify(|e| {
                    let index = e.iter().position(|x| *x == k1);
                    if let Some(index) = index {
                        e.remove(index);
                    }
                });
            }

            for dim in (&k2 - output_dims).iter().cloned() {
                dim_to_tensors.entry(dim).and_modify(|e| {
                    let index = e.iter().position(|x| *x == k2);
                    if let Some(index) = index {
                        e.remove(index);
                    }
                });
            }
            ssa_path.push((*ssa_id1, *ssa_id2));

            if let Entry::Occupied(o) = remaining_tensors.entry(k1.clone()) {
                o.remove_entry();
            }
            if let Entry::Occupied(o) = remaining_tensors.entry(k2.clone()) {
                o.remove_entry();
            }

            if remaining_tensors.contains_key(&k12) {
                ssa_path.push((remaining_tensors[&k12], next_ssa_id));
                next_ssa_id += 1;
            } else {
                for dim in (&k12 - output_dims).iter().cloned() {
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
            for dim in (&k1 - output_dims).iter() {
                for k2 in dim_to_tensors[dim].iter() {
                    if k2 != &k1 {
                        k2s.push(k2);
                    }
                }
            }
            if k2.dims() > 0 {
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

        let mut heap = BinaryHeap::new();
        for (key, ssa_id) in remaining_tensors {
            let candidate = Candidate {
                flop_cost: 0,
                size_cost: _tensor_size(&(&key & output_dims), bond_dims) as i64,
                parent_ids: (ssa_id, 0),
                parent_tensors: Some((key, Tensor::default())),
                child_id: 0,
                child_tensor: None,
            };
            heap.push(candidate);
        }

        let Some(Candidate {
            flop_cost: 0,
            size_cost: _cost,
            parent_ids: (ssa_id1, _id2),
            parent_tensors: Some((k1, _k2)),
            child_id: 0,
            child_tensor: None,
        }) = queue.pop() else{
            return ssa_path;
        };

        while !queue.is_empty() {
            let Some(Candidate {
                flop_cost: _flop_cost,
                size_cost: _cost,
                parent_ids: (ssa_id2, _id2),
                parent_tensors: Some((k2, _k2)),
                child_id: _child_id,
                child_tensor: _child_tensor,
            }) = queue.pop() else{
                continue;
            };

            ssa_path.push((min(ssa_id1, ssa_id2), max(ssa_id1, ssa_id2)));
            let k12 = &(&k1 | &k2) & output_dims;
            let cost = _tensor_size(&k12, bond_dims) as i64;
            queue.push(Candidate {
                flop_cost: 0,
                size_cost: cost,
                parent_ids: (ssa_id1, 0),
                parent_tensors: Some((k1.clone(), _k2)),
                child_id: 0,
                child_tensor: None,
            });
            let Some(Candidate {
                flop_cost: _flop_cost,
                size_cost: _cost,
                parent_ids: (ssa_id1, _id2),
                parent_tensors: Some((k1, _k2)),
                child_id: _child_id,
                child_tensor: _child_tensor,
            }) = queue.pop() else{
                continue
            };
        }
        ssa_path
    }
}

impl<'a> OptimizePath for Greedy<'a> {
    fn optimize_path(&mut self) {
        if self.tn.get_tensors().len() == 1 {
            // Perform a single contraction to match output shape.
            self.best_flops = 0;
            self.best_size = 0;
            self.best_path = vec![];
            return;
        }
        let inputs: Vec<Tensor> = self.tn.get_tensors().clone();

        // Vector of output leg ids
        let output_dims = Tensor::new(self.tn.get_ext_edges().clone());
        // Dictionary that maps leg id to bond dimension
        let bond_dims = self.tn.get_bond_dims();
        self.best_path = self._ssa_greedy_optimize(
            &inputs,
            &output_dims,
            bond_dims,
            Box::new(&Greedy::_simple_chooser),
            Box::new(&Greedy::_cost_memory_removed),
        );
        let (op_cost, mem_cost) = _contract_path_cost(
            self.tn.get_tensors(),
            &self.get_best_replace_path(),
            self.tn.get_bond_dims(),
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

    fn get_best_path(&self) -> &Vec<(usize, usize)> {
        &self.best_path
    }

    fn get_best_replace_path(&self) -> Vec<(usize, usize)> {
        ssa_replace_ordering(&self.best_path, self.tn.get_tensors().len())
    }
}

#[cfg(test)]
mod tests {
    // use rand::distributions::{Distribution, Uniform};
    // TODO: Use random tensors
    use super::ssa_ordering;
    use super::ssa_replace_ordering;
    use crate::contractionpath::paths::BranchBound;
    use crate::contractionpath::paths::CostType;
    use crate::contractionpath::paths::Greedy;
    use crate::contractionpath::paths::OptimizePath;
    use crate::tensornetwork::tensor::Tensor;
    use crate::tensornetwork::TensorNetwork;

    fn setup_simple() -> TensorNetwork {
        TensorNetwork::from_vector(
            vec![
                Tensor::new(vec![4, 3, 2]),
                Tensor::new(vec![0, 1, 3, 2]),
                Tensor::new(vec![4, 5, 6]),
            ],
            vec![5, 2, 6, 8, 1, 3, 4],
            None,
        )
    }

    fn setup_complex() -> TensorNetwork {
        TensorNetwork::from_vector(
            vec![
                Tensor::new(vec![4, 3, 2]),
                Tensor::new(vec![0, 1, 3, 2]),
                Tensor::new(vec![4, 5, 6]),
                Tensor::new(vec![6, 8, 9]),
                Tensor::new(vec![10, 8, 9]),
                Tensor::new(vec![5, 1, 0]),
            ],
            vec![27, 18, 12, 15, 5, 3, 18, 22, 45, 65, 5, 17],
            None,
        )
    }

    #[test]
    fn test_ssa_ordering() {
        let path = vec![
            (0, 3, 15),
            (1, 2, 44),
            (6, 4, 8),
            (5, 15, 22),
            (8, 44, 12),
            (12, 22, 99),
        ];
        let new_path = ssa_ordering(&path, 7);

        assert_eq!(
            new_path,
            vec![(0, 3), (1, 2), (6, 4), (5, 7), (9, 8), (11, 10)]
        )
    }

    #[test]
    fn test_ssa_replace_ordering() {
        let path = vec![(0, 3), (1, 2), (6, 4), (5, 7), (9, 8), (11, 10)];
        let new_path = ssa_replace_ordering(&path, 7);

        assert_eq!(
            new_path,
            vec![(0, 3), (1, 2), (6, 4), (5, 0), (6, 1), (6, 5)]
        )
    }

    #[test]
    fn test_contract_order_simple() {
        let tn = setup_simple();
        let mut opt = BranchBound::new(&tn, None, 20, CostType::Flops);
        opt.optimize_path();

        assert_eq!(opt.best_flops, 600);
        assert_eq!(opt.best_size, 538);
        assert_eq!(opt.get_best_path(), &vec![(0, 1), (2, 3)]);
        assert_eq!(opt.get_best_replace_path(), vec![(0, 1), (2, 0)]);
    }

    #[test]
    fn test_contract_order_complex() {
        let tn = setup_complex();
        let mut opt = BranchBound::new(&tn, None, 20, CostType::Flops);
        opt.optimize_path();

        assert_eq!(opt.best_flops, 332685);
        assert_eq!(opt.best_size, 89478);
        assert_eq!(opt.best_path, vec![(1, 5), (0, 6), (2, 7), (3, 8), (4, 9)]);
        assert_eq!(
            opt.get_best_replace_path(),
            vec![(1, 5), (0, 1), (2, 0), (3, 2), (4, 3)]
        );
    }

    #[test]
    fn test_contract_order_greedy_simple() {
        let tn = setup_simple();
        let mut opt = Greedy::new(&tn, CostType::Flops);
        opt.optimize_path();

        assert_eq!(opt.best_flops, 600);
        assert_eq!(opt.best_size, 538);
        assert_eq!(opt.best_path, vec![(0, 1), (2, 3)]);
        assert_eq!(opt.get_best_replace_path(), vec![(0, 1), (2, 0)]);
    }
    #[test]
    fn test_contract_order_greedy_complex() {
        let tn = setup_complex();
        let mut opt = Greedy::new(&tn, CostType::Flops);
        opt.optimize_path();

        assert_eq!(opt.best_flops, 529815);
        assert_eq!(opt.best_size, 89478);
        assert_eq!(opt.best_path, vec![(1, 5), (3, 4), (0, 6), (2, 7), (8, 9)]);
        assert_eq!(
            opt.get_best_replace_path(),
            vec![(1, 5), (3, 4), (0, 1), (2, 3), (0, 2)]
        );
    }
}
