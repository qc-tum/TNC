use std::{
    cmp::{max, min},
    collections::{BinaryHeap, HashSet, VecDeque},
};

use itertools::Itertools;
use rand::{rngs::StdRng, Rng, SeedableRng};
use rustc_hash::FxHashMap;

use crate::{
    contractionpath::{
        candidates::Candidate,
        contraction_cost::contract_path_cost_slicing,
        paths::{greedy::populate_remaining_tensors, RNGChooser},
        ssa_ordering, ssa_replace_ordering,
    },
    tensornetwork::tensor::Tensor,
    types::{calculate_hash, ContractionIndex, EdgeIndex, SlicingPlan},
    utils::traits::HashMapInsertNew,
};

use super::{greedy::SimpleChooser, validate_path, CostFnType, CostType, OptimizePath};

pub struct GreedySlice<'a> {
    pub(crate) tn: &'a Tensor,
    pub(crate) minimize: CostType,
    pub(crate) best_flops: f64,
    pub(crate) best_size: f64,
    pub(crate) best_path: Vec<ContractionIndex>,
    pub(crate) slicing: HashSet<EdgeIndex>,
    max_memory: f64,
    best_progress: FxHashMap<usize, f64>,
}

impl<'a> GreedySlice<'a> {
    pub fn new(tn: &'a Tensor, max_memory: f64, minimize: CostType) -> Self {
        Self {
            tn,
            minimize,
            best_flops: f64::INFINITY,
            best_size: f64::INFINITY,
            best_path: Vec::new(),
            slicing: HashSet::new(),
            max_memory,
            best_progress: FxHashMap::default(),
        }
    }

    /// The default heuristic cost, corresponding to the total reduction in
    /// memory of performing a contraction.
    pub(crate) fn cost_memory_removed(
        size12: f64,
        size1: f64,
        size2: f64,
        _k12: &Tensor,
        _k1: &Tensor,
        _k2: &Tensor,
    ) -> f64 {
        size12 - size1 - size2
    }

    /// Con cost, corresponding to the total reduction in
    /// memory of performing a contraction.
    pub(crate) fn cost_communication(
        _size12: f64,
        size1: f64,
        _size2: f64,
        _k12: &Tensor,
        _k1: &Tensor,
        _k2: &Tensor,
    ) -> f64 {
        size1
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
    ) -> (Vec<ContractionIndex>, Vec<EdgeIndex>)
    where
        R: ?Sized + Rng,
    {
        const TEMPERATURE: f64 = 0.3;
        const NBRANCH: usize = 5;
        const REL_TEMPERATURE: bool = true;
        let mut slicing = Vec::new();
        let bond_dims = inputs[0].bond_dims();
        // Keeps track of remaining vectors, mapping between Vector of tensor leg ids to ssa number
        let (
            mut remaining_tensors,
            mut ssa_id_to_tensor,
            mut edge_to_tensors,
            mut scalar_tensors,
            mut ssa_path,
            mut next_ssa_id,
        ) = populate_remaining_tensors(inputs, output_dims);
        
        // Maps tensor ssa_id to size
        let mut tensor_mem_size = ssa_id_to_tensor
            .values()
            .map(|tensor| (calculate_hash(tensor), tensor.size()))
            .collect::<FxHashMap<_, _>>();

        let mut queue = BinaryHeap::new();
        // Fill queue with all possible contraction combinations of contractions
        for connected_tensors in edge_to_tensors.values_mut() {
            connected_tensors
                .sort_unstable_by_key(|a| ssa_id_to_tensor.get(a).unwrap().legs().len());
            // Loop over all but the last entry
            for (i, k1_id) in connected_tensors[0..connected_tensors.len() - 1]
                .iter()
                .enumerate()
            {
                // Get all possible unconsidered combinations
                for k2_id in &connected_tensors[(i + 1)..] {
                    let (k12, size_cost, k1_hash, k2_hash) = {
                        let k1 = ssa_id_to_tensor.get(k1_id).unwrap();
                        let k1_hash = calculate_hash(k1);
                        let k2 = ssa_id_to_tensor.get(k2_id).unwrap();
                        let k2_hash = calculate_hash(k2);
                        let k12 = k1 ^ k2;
                        let k12_hash = calculate_hash(&k12);
                        tensor_mem_size
                            .entry(k12_hash)
                            .or_insert_with(|| k12.size());

                        let size_cost = cost_fn(
                            tensor_mem_size[&k12_hash],
                            tensor_mem_size[&k1_hash],
                            tensor_mem_size[&k2_hash],
                            &k12,
                            k1,
                            k2,
                        );
                        (k12, size_cost, k1_hash, k2_hash)
                    };

                    ssa_id_to_tensor
                        .entry(next_ssa_id)
                        .or_insert_with(|| k12.clone());

                    let mut id1 = remaining_tensors[&k1_hash];
                    let mut id2 = remaining_tensors[&k2_hash];

                    if tensor_mem_size[&k2_hash] > tensor_mem_size[&k1_hash] {
                        (id1, id2) = (id2, id1);
                    }

                    queue.push(Candidate {
                        flop_cost: 0f64,
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
                parent_ids: (id1, id2),
                child_id,
                ..
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
            for &leg in (&k1 - output_dims).legs() {
                edge_to_tensors
                    .entry(leg)
                    .and_modify(|e| e.retain(|&x| x != id1));
            }

            // Removing k2 from edge_to_tensors
            for &leg in (&k2 - output_dims).legs() {
                edge_to_tensors
                    .entry(leg)
                    .and_modify(|e| e.retain(|&x| x != id2));
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
                for &dim in (&k12 - output_dims).legs() {
                    edge_to_tensors.entry(dim).and_modify(|e| e.push(child_id));
                }
            }

            let mut mem_size = k12.size();

            if mem_size > self.max_memory {
                let mut sorted_legs = k12
                    .legs()
                    .iter()
                    .filter(|leg| !output_dims.legs().contains(leg))
                    .sorted_unstable_by_key(|leg| bond_dims[leg]);

                while mem_size > self.max_memory {
                    let first_leg = sorted_legs.next().unwrap();

                    slicing.push(*first_leg);
                    mem_size /= bond_dims[first_leg] as f64;
                }
            }

            // add newly output tensor to remaining tensors
            remaining_tensors
                .entry(calculate_hash(&k12))
                .or_insert_with(|| child_id);

            tensor_mem_size
                .entry(k12_hash)
                .or_insert_with(|| k12.sliced_size(&slicing));

            //Find new candidate contractions.
            let k1 = k12;
            let k1_hash = k12_hash;

            let mut k2s = Vec::new();

            // for each dimension in output tensor that will be contracted in the future, find respective contracted tensors.
            for dim in (&k1 - output_dims).legs() {
                for k2_id in &edge_to_tensors[dim] {
                    // do not consider contracting with self. Inner products already removed
                    if *k2_id != child_id {
                        k2s.push(k2_id);
                    }
                }
            }

            if !k2s.is_empty() {
                for k2_id in k2s {
                    let k2 = ssa_id_to_tensor.get(k2_id).unwrap();
                    let k2_hash = calculate_hash(k2);
                    let k12 = &k1 ^ k2;
                    let k12_hash = calculate_hash(&k12);
                    tensor_mem_size
                        .entry(k12_hash)
                        .or_insert_with(|| k12.size());

                    let size_cost = cost_fn(
                        tensor_mem_size[&k12_hash],
                        tensor_mem_size[&k1_hash],
                        tensor_mem_size[&k2_hash],
                        &k12,
                        &k1,
                        k2,
                    );
                    ssa_id_to_tensor.entry(next_ssa_id).or_insert_with(|| k12);

                    let mut id1 = remaining_tensors[&k1_hash];
                    let mut id2 = remaining_tensors[&k2_hash];

                    if tensor_mem_size[&k2_hash] > tensor_mem_size[&k1_hash] {
                        (id2, id1) = (id1, id2);
                    }

                    queue.push(Candidate {
                        flop_cost: 0f64,
                        size_cost,
                        parent_ids: (id1, id2),
                        child_id: next_ssa_id,
                    });
                    next_ssa_id += 1;
                }
            }
            ssa_id_to_tensor.entry(child_id).or_insert_with(|| k1);
        }

        for (_key, ssa_id) in remaining_tensors {
            let k12_tensor = &ssa_id_to_tensor[&ssa_id];
            let tensor_size = (k12_tensor & output_dims).size();
            if tensor_size > 0f64 {
                let candidate = Candidate {
                    flop_cost: 0f64,
                    size_cost: tensor_size,
                    parent_ids: (ssa_id, 0),
                    child_id: 0,
                };
                queue.push(candidate);
            }
        }

        while queue.len() >= 2 {
            let Candidate {
                flop_cost: _flop_cost,
                size_cost: _cost,
                parent_ids: (ssa_id1, _id1),
                child_id: _child_id,
            } = queue.pop().unwrap();
            let Candidate {
                flop_cost: _flop_cost,
                size_cost: _cost,
                parent_ids: (ssa_id2, _id2),
                child_id: _child_id,
            } = queue.pop().unwrap();
            let k1 = ssa_id_to_tensor.remove(&ssa_id1).unwrap();
            let k2 = ssa_id_to_tensor.remove(&ssa_id2).unwrap();
            ssa_path.push((min(ssa_id1, ssa_id2), max(ssa_id1, ssa_id2), next_ssa_id));
            let k12 = &k1 ^ &k2;
            let cost = k12.size();

            queue.push(Candidate {
                flop_cost: 0f64,
                size_cost: cost,
                parent_ids: (next_ssa_id, 0),
                child_id: 0,
            });
            ssa_id_to_tensor.insert_new(next_ssa_id, k12);
            next_ssa_id += 1;
        }
        if !scalar_tensors.is_empty() {
            let mut latest_scalar = scalar_tensors[0];
            // Multiply the various scalar results together
            for &scalar_id in &scalar_tensors[1..] {
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
                parent_ids: (last_tensor, _id1),
                ..
            }) = queue.pop()
            else {
                let ssa_path = ssa_ordering(&ssa_path, inputs.len());
                validate_path(&ssa_path);
                return (ssa_path, slicing);
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

        (ssa_path, slicing)
    }

    fn get_slicing(&self) -> &HashSet<EdgeIndex> {
        &self.slicing
    }
}

// Assume one-level of parallelism
impl OptimizePath for GreedySlice<'_> {
    fn optimize_path(&mut self) {
        if self.tn.tensors().len() == 1 {
            // Perform a single contraction to match output shape.
            self.best_flops = 0f64;
            self.best_size = 0f64;
            self.best_path = vec![];
            return;
        }
        let mut inputs = self.tn.tensors().clone();
        let mut rng = StdRng::seed_from_u64(24);
        for (index, input_tensor) in inputs.iter_mut().enumerate() {
            if input_tensor.is_composite() {
                let external_legs = input_tensor.external_edges();
                let (path, slicing) = self.ssa_greedy_optimize(
                    input_tensor.tensors(),
                    &Tensor::new(external_legs.clone()),
                    SimpleChooser,
                    Box::new(&GreedySlice::cost_memory_removed),
                    &mut rng,
                );
                self.slicing.extend(slicing);
                if !path.is_empty() {
                    self.best_path
                        .push(ContractionIndex::Path(index, None, path));
                }
                input_tensor.set_legs(external_legs);
            }
        }

        // Vector of output leg ids
        let output_dims = Tensor::new(self.tn.external_edges());
        // Start considering communication here!
        let (mut path, slicing) = self.ssa_greedy_optimize(
            &inputs,
            &output_dims,
            SimpleChooser,
            Box::new(&GreedySlice::cost_memory_removed),
            &mut rng,
        );

        self.best_path.append(&mut path);
        self.slicing.extend(slicing);

        let slicing_plan = SlicingPlan {
            slices: self.slicing.iter().copied().collect::<Vec<_>>(),
        };

        let (op_cost, mem_cost) = contract_path_cost_slicing(
            self.tn.tensors(),
            &self.get_best_replace_path(),
            Some(&slicing_plan),
            true,
        );
        self.best_size = mem_cost;
        self.best_flops = op_cost;
    }

    fn get_best_flops(&self) -> f64 {
        self.best_flops
    }

    fn get_best_size(&self) -> f64 {
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
    use std::collections::HashSet;
    use std::hash::Hash;

    use rustc_hash::FxHashMap;

    use crate::contractionpath::paths::greedy_slice::GreedySlice;
    use crate::contractionpath::paths::CostType;
    use crate::contractionpath::paths::OptimizePath;
    use crate::path;
    use crate::tensornetwork::create_tensor_network;
    use crate::tensornetwork::tensor::Tensor;

    fn setup_simple() -> Tensor {
        create_tensor_network(
            vec![
                Tensor::new(vec![4, 3, 2]),
                Tensor::new(vec![0, 1, 3, 2]),
                Tensor::new(vec![4, 5, 6]),
            ],
            &FxHashMap::from_iter([(0, 2), (1, 2), (2, 12), (3, 8), (4, 8), (5, 3), (6, 2)]),
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
            &FxHashMap::from_iter([
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
            ]),
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
            &FxHashMap::from_iter([
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
            ]),
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
            &FxHashMap::from_iter([(0, 5), (1, 2), (2, 6), (3, 8), (4, 1), (5, 3), (6, 4)]),
        )
    }

    fn setup_simple_outer_product() -> Tensor {
        create_tensor_network(
            vec![
                Tensor::new(vec![0]),
                Tensor::new(vec![1]),
                Tensor::new(vec![2]),
            ],
            &FxHashMap::from_iter([(0, 3), (1, 2), (2, 2)]),
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
            &FxHashMap::from_iter([(0, 5), (1, 4)]),
        )
    }

    fn map_zip<'a, K, V, T>(
        a: &'a FxHashMap<K, V>,
        b: &'a FxHashMap<K, T>,
    ) -> impl Iterator<Item = (&'a K, (&'a V, &'a T))>
    where
        K: Eq + Hash,
    {
        assert_eq!(a.len(), b.len());
        a.iter().map(|(k, v)| (k, (v, &b[k])))
    }

    #[test]
    fn test_contract_order_greedy_slicing_simple() {
        let tn = setup_simple();

        let mut opt = GreedySlice::new(&tn, 24f64, CostType::Flops);
        opt.optimize_path();
        assert_eq!(opt.slicing, HashSet::from([4]));
        assert_eq!(opt.best_flops, 408f64);
        assert_eq!(opt.best_size, 484f64);
        assert_eq!(opt.best_path, path![(0, 1), (2, 3)]);
        assert_eq!(opt.get_best_replace_path(), path![(0, 1), (2, 0)]);
    }

    #[test]
    fn test_contract_order_greedy_slicing_simple_inner() {
        let tn = setup_simple_inner_product();
        let mut opt = GreedySlice::new(&tn, 60f64, CostType::Flops);
        opt.optimize_path();

        assert_eq!(opt.slicing, HashSet::from([]));
        assert_eq!(opt.best_flops, 228f64);
        assert_eq!(opt.best_size, 121f64);
        assert_eq!(opt.best_path, path![(0, 1), (2, 3), (4, 5)]);
        assert_eq!(opt.get_best_replace_path(), path![(0, 1), (2, 3), (0, 2)]);
    }

    #[test]
    fn test_contract_order_greedy_slicing_simple_outer() {
        let tn = setup_simple_outer_product();
        let mut opt = GreedySlice::new(&tn, 24f64, CostType::Flops);
        opt.optimize_path();

        assert_eq!(opt.slicing, HashSet::from([]));
        assert_eq!(opt.best_flops, 16f64);
        assert_eq!(opt.best_size, 19f64);
        assert_eq!(opt.best_path, path![(1, 2), (0, 3)]);
        assert_eq!(opt.get_best_replace_path(), path![(1, 2), (0, 1)]);
    }

    #[test]
    fn test_contract_order_greedy_slicing_complex_outer() {
        let tn = setup_complex_outer_product();
        let mut opt = GreedySlice::new(&tn, 200f64, CostType::Flops);
        opt.optimize_path();

        assert_eq!(opt.slicing, HashSet::from([]));
        assert_eq!(opt.best_flops, 10f64);
        assert_eq!(opt.best_size, 11f64);
        assert_eq!(opt.best_path, path![(0, 1), (2, 3), (4, 5)]);
        assert_eq!(opt.get_best_replace_path(), path![(0, 1), (2, 3), (0, 2)]);
    }

    #[test]
    fn test_contract_order_greedy_slicing_complex() {
        let tn = setup_complex();
        let mut opt = GreedySlice::new(&tn, 200f64, CostType::Flops);
        opt.optimize_path();

        assert_eq!(opt.slicing, HashSet::from([5]));
        assert_eq!(opt.best_flops, 352105f64);
        assert_eq!(opt.best_size, 88146f64);
        assert_eq!(opt.best_path, path![(1, 5), (3, 4), (0, 6), (2, 7), (9, 8)]);
        assert_eq!(
            opt.get_best_replace_path(),
            path![(1, 5), (3, 4), (0, 1), (2, 3), (2, 0)]
        );
    }

    #[test]
    fn test_contract_order_greedy_two_slicing_complex() {
        let tn = setup_complex();
        let mut opt = GreedySlice::new(&tn, 150f64, CostType::Flops);
        opt.optimize_path();

        assert_eq!(opt.slicing, HashSet::from([5, 2]));
        assert_eq!(opt.best_flops, 271090f64);
        assert_eq!(opt.best_size, 67365f64);
        assert_eq!(opt.best_path, path![(1, 5), (3, 4), (0, 6), (2, 7), (9, 8)]);
        assert_eq!(
            opt.get_best_replace_path(),
            path![(1, 5), (3, 4), (0, 1), (2, 3), (2, 0)]
        );
    }
}
