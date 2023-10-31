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
    contraction_cost::{_contract_cost, _contract_path_cost, _contract_size, _tensor_size},
    ssa_ordering, ssa_replace_ordering,
};
use crate::pair;
use crate::tensornetwork::tensor::Tensor;
use crate::types::ContractionIndex;

pub trait OptimizePath {
    fn optimize_path(&mut self);
    // fn optimize_partitioned_path(&mut self, partition: Vec<usize>);

    fn get_best_path(&self) -> &Vec<ContractionIndex>;
    fn get_best_replace_path(&self) -> Vec<ContractionIndex>;
    // fn get_best_partition_replace_path(&self, k: usize) -> Vec<(usize, usize)>;
    fn get_best_flops(&self) -> u64;
    fn get_best_size(&self) -> u64;
}

#[derive(Debug, Clone)]
pub enum CostType {
    Flops = 0,
    Size = 1,
}

/// A struct with an OptimizePath implementation that explores possible pair contractions in a depth-first manner.
pub struct BranchBound<'a> {
    tn: &'a Tensor,
    nbranch: Option<u32>,
    cutoff_flops_factor: u64,
    minimize: CostType,
    best_flops: u64,
    best_size: u64,
    best_path: Vec<ContractionIndex>,
    best_progress: HashMap<usize, u64>,
    result_cache: HashMap<Vec<usize>, usize>,
    flop_cache: HashMap<usize, u64>,
    size_cache: HashMap<usize, u64>,
    tensor_cache: HashMap<usize, Tensor>,
}

impl<'a> BranchBound<'a> {
    pub fn new(
        tn: &'a Tensor,
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
                    &self.tn.get_bond_dims(),
                );
                (k12_tensor, size_12) = _contract_size(
                    &self.tensor_cache[&i],
                    &self.tensor_cache[&j],
                    &self.tn.get_bond_dims(),
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
        if self.tn.is_empty() {
            return;
        }
        let mut tensors = self.tn.get_tensors().clone();
        self.flop_cache.clear();
        self.size_cache.clear();
        let mut sub_tensor_contraction = Vec::new();
        // Get the initial space requirements for uncontracted tensors
        for (index, tensor) in tensors.iter_mut().enumerate() {
            // Check that tensor has sub-tensors and doesn't have external legs set
            if !tensor.get_tensors().is_empty() && tensor.get_legs().is_empty() {
                let mut bb = BranchBound::new(
                    tensor,
                    self.nbranch,
                    self.cutoff_flops_factor,
                    self.minimize.clone(),
                );
                bb.optimize_path();
                sub_tensor_contraction
                    .push(ContractionIndex::Path((index, bb.get_best_path().clone())));
                tensor.set_legs(tensor.get_external_edges());
            }
            self.size_cache
                .entry(index)
                .or_insert(tensor.shape().iter().product::<u64>());

            self.tensor_cache.entry(index).or_insert(tensor.clone());
        }
        let remaining: Vec<u32> = (0u32..self.tn.get_tensors().len() as u32).collect();
        BranchBound::_branch_iterate(self, vec![], remaining, 0, 0);
        sub_tensor_contraction.append(&mut self.best_path.clone());
        self.best_path = sub_tensor_contraction;
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
        if false {
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
                if tensor_legs.len() >= i {
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
                &mut thread_rng(),
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

        for (key, ssa_id) in remaining_tensors {
            let candidate = Candidate {
                flop_cost: 0,
                size_cost: _tensor_size(&(&key & output_dims), bond_dims) as i64,
                parent_ids: (ssa_id, 0),
                parent_tensors: Some((key, Tensor::default())),
                child_id: 0,
                child_tensor: None,
            };
            queue.push(candidate);
        }

        let Some(Candidate {
            flop_cost: 0,
            size_cost: _cost,
            parent_ids: (ssa_id1, _id2),
            parent_tensors: Some((k1, _k2)),
            child_id: 0,
            child_tensor: None,
        }) = queue.pop()
        else {
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
            }) = queue.pop()
            else {
                continue;
            };
            ssa_path.push(pair!(min(ssa_id1, ssa_id2), max(ssa_id1, ssa_id2)));
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
                parent_ids: (_ssa_id1, _id2),
                parent_tensors: Some((_k1, _k2)),
                child_id: _child_id,
                child_tensor: _child_tensor,
            }) = queue.pop()
            else {
                continue;
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
                    self.best_path
                        .push(ContractionIndex::Path((index, ssa_path)));
                }
                input_tensor.set_legs(external_legs);
            }
        }

        // Vector of output leg ids
        let output_dims = Tensor::new(self.tn.get_external_edges().clone());
        // Dictionary that maps leg id to bond dimension
        let bond_dims = self.tn.get_bond_dims();
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
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    use super::ssa_ordering;
    use super::ssa_replace_ordering;
    use crate::contractionpath::paths::BranchBound;
    use crate::contractionpath::paths::CostType;
    use crate::contractionpath::paths::Greedy;
    use crate::contractionpath::paths::OptimizePath;
    use crate::path;
    use crate::tensornetwork::create_tensor_network;
    use crate::tensornetwork::tensor::Tensor;
    use crate::types::ContractionIndex;

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
            path![(0, 3), (1, 2), (6, 4), (5, 7), (9, 8), (11, 10)]
        )
    }

    #[test]
    fn test_ssa_replace_ordering() {
        let path = path![(0, 3), (1, 2), (6, 4), (5, 7), (9, 8), (11, 10)];
        let new_path = ssa_replace_ordering(&path, 7);

        assert_eq!(
            new_path,
            path![(0, 3), (1, 2), (6, 4), (5, 0), (6, 1), (6, 5)]
        )
    }

    #[test]
    fn test_contract_order_simple() {
        let tn = setup_simple();
        let mut opt = BranchBound::new(&tn, None, 20, CostType::Flops);
        opt.optimize_path();

        assert_eq!(opt.best_flops, 600);
        assert_eq!(opt.best_size, 538);
        assert_eq!(opt.get_best_path(), &path![(0, 1), (2, 3)]);
        assert_eq!(opt.get_best_replace_path(), path![(0, 1), (2, 0)]);
    }

    #[test]
    fn test_contract_order_complex() {
        let tn = setup_complex();
        let mut opt = BranchBound::new(&tn, None, 20, CostType::Flops);
        opt.optimize_path();

        assert_eq!(opt.best_flops, 332685);
        assert_eq!(opt.best_size, 89478);
        assert_eq!(opt.best_path, path![(1, 5), (0, 6), (2, 7), (3, 8), (4, 9)]);
        assert_eq!(
            opt.get_best_replace_path(),
            path![(1, 5), (0, 1), (2, 0), (3, 2), (4, 3)]
        );
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
        let mut _rng = StdRng::seed_from_u64(52);
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
