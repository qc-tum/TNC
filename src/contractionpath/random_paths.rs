use rand::{distributions::WeightedIndex, prelude::*};
use std::{
    cmp::max,
    collections::{BinaryHeap, HashMap},
};

use crate::tensornetwork::tensor::Tensor;

use super::{
    candidates::Candidate, contraction_cost::_contract_path_cost, paths::Greedy,
    ssa_replace_ordering,
};

pub trait RandomOptimizePath {
    fn random_optimize_path<R>(&mut self, trials: usize, rng: &mut R)
    where
        R: ?Sized + Rng;
}

// __all__ = ["RandomGreedy", "random_greedy", "random_greedy_128"]
impl<'a> Greedy<'a> {
    pub(crate) fn _thermal_chooser<R: Rng + ?Sized>(
        queue: &mut BinaryHeap<Candidate>,
        remaining_tensors: &HashMap<Tensor, usize>,
        nbranch: usize,
        mut temperature: f64,
        rel_temperature: bool,
        mut rng: &mut R,
    ) -> Option<Candidate> {
        let mut n = 0;
        let mut choices = Vec::new();
        while !queue.is_empty() && n <= nbranch {
            let candidate = queue.pop();
            if let Some(Candidate {
                flop_cost,
                size_cost,
                parent_ids,
                parent_tensors: Some((k1, k2)),
                child_id,
                child_tensor,
            }) = candidate
            {
                if !remaining_tensors.contains_key(&k1) || !remaining_tensors.contains_key(&k2) {
                    continue;
                }
                choices.push(Candidate {
                    flop_cost,
                    size_cost,
                    parent_ids,
                    parent_tensors: Some((k1, k2)),
                    child_id,
                    child_tensor,
                });
                n += 1;
            }
        }

        if n == 0 {
            return None;
        }
        if n == 1 {
            return Some(choices[0].clone());
        }

        let costs = choices.iter().map(|e| e.size_cost).collect::<Vec<i64>>();
        let cmin = costs[0];

        // adjust by the overall scale to account for fluctuating absolute costs
        if rel_temperature {
            temperature *= max(1, cmin.abs()) as f64;
        }

        // compute relative probability for each potential contraction
        let mut weights = Vec::new();
        if temperature == 0.0 {
            weights = vec![0.0; costs.len()];
            weights[0] = 1.0;
        } else {
            for c in costs {
                weights.push((-(-c - cmin) as f64).exp());
            }
        }
        let dist = WeightedIndex::new(&weights).unwrap();
        let chosen = dist.sample(&mut rng);
        let candidate = choices.get(chosen);
        for (index, other) in choices.iter().enumerate() {
            if index != chosen {
                queue.push(other.clone());
            }
        }
        candidate.cloned()
    }
}

impl<'a> RandomOptimizePath for Greedy<'a> {
    fn random_optimize_path<R>(&mut self, trials: usize, rng: &mut R)
    where
        R: ?Sized + Rng,
    {
        let inputs: Vec<Tensor> = self.tn.get_tensors().clone();

        let output_dims = Tensor::new(self.tn.get_external_edges().clone());

        // Dictionary that maps leg id to bond dimension
        let bond_dims = self.tn.get_bond_dims();
        for _ in 0..trials {
            let ssa_path = self._ssa_greedy_optimize(
                &inputs,
                &output_dims,
                &bond_dims,
                Box::new(&Greedy::_thermal_chooser),
                Box::new(&Greedy::_cost_memory_removed),
            );
            let (cost, size) = _contract_path_cost(
                &inputs,
                &ssa_replace_ordering(&ssa_path, inputs.len()),
                &bond_dims,
            );

            if cost < self.best_flops {
                self.best_flops = cost;
                self.best_size = size;
                self.best_path = ssa_path;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::rngs::StdRng;
    use rand::thread_rng;
    use rand::SeedableRng;

    use crate::contractionpath::paths::CostType;
    // use rand::distributions::{Distribution, Uniform};
    // TODO: Use random tensors
    use crate::contractionpath::paths::Greedy;
    use crate::contractionpath::paths::OptimizePath;
    use crate::contractionpath::random_paths::RandomOptimizePath;
    use crate::tensornetwork::create_tensor_network;
    use crate::tensornetwork::tensor::Tensor;

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

    #[test]
    fn test_contract_order_greedy_simple() {
        let tn = setup_simple();
        let mut opt = Greedy::new(&tn, CostType::Flops);
        opt.random_optimize_path(32, &mut StdRng::seed_from_u64(42));
        assert_eq!(opt.best_flops, 600);
        assert_eq!(opt.best_size, 538);
        assert_eq!(opt.best_path, vec![(0, 1), (2, 3)]);
        assert_eq!(opt.get_best_replace_path(), vec![(0, 1), (2, 0)]);
    }
    #[test]
    fn test_contract_order_greedy_complex() {
        let tn = setup_complex();
        let mut opt = Greedy::new(&tn, CostType::Flops);
        opt.random_optimize_path(32, &mut StdRng::seed_from_u64(42));

        assert_eq!(opt.best_flops, 528750);
        assert_eq!(opt.best_size, 89478);
        assert_eq!(opt.best_path, vec![(1, 5), (3, 4), (0, 6), (2, 8), (7, 9)]);
        assert_eq!(
            opt.get_best_replace_path(),
            vec![(1, 5), (3, 4), (0, 1), (2, 0), (3, 2)]
        );
    }
}

// fn _trial_greedy_ssa_path_and_cost(r : R,
//      inputs: Vec<(usize,usize)>,
//      output: Vec<usize>,
//      bond_dims: HashMap<usize, u64>){

//     let ssa_path = ssa_greedy_optimize(inputs, output, bond_dims, choose_fn, cost_fn)
//     cost, size = ssa_path_compute_cost(ssa_path, inputs, output, size_dict)

//     return ssa_path, cost, size
//      }

// class RandomOptimizer(paths.PathOptimizer):
//     """Base class for running any random path finder that benefits
//     from repeated calling, possibly in a parallel fashion. Custom random
//     optimizers should subclass this, and the ``setup`` method should be
//     implemented with the following signature::

//         def setup(self, inputs, output, size_dict):
//             # custom preparation here ...
//             return trial_fn, trial_args

//     Where ``trial_fn`` itself should have the signature::

//         def trial_fn(r, *trial_args):
//             # custom computation of path here
//             return ssa_path, cost, size

//     Where ``r`` is the run number and could for example be used to seed a
//     random number generator. See ``RandomGreedy`` for an example.

//     Parameters
//     ----------
//     max_repeats : int, optional
//         The maximum number of repeat trials to have.
//     max_time : float, optional
//         The maximum amount of time to run the algorithm for.
//     minimize : {'flops', 'size'}, optional
//         Whether to favour paths that minimize the total estimated flop-count or
//         the size of the largest intermediate created.
//     parallel : {bool, int, or executor-pool like}, optional
//         Whether to parallelize the random trials, by default ``False``. If
//         ``True``, use a ``concurrent.futures.ProcessPoolExecutor`` with the same
//         number of processes as cores. If an integer is specified, use that many
//         processes instead. Finally, you can supply a custom executor-pool which
//         should have an API matching that of the python 3 standard library
//         module ``concurrent.futures``. Namely, a ``submit`` method that returns
//         ``Future`` objects, themselves with ``result`` and ``cancel`` methods.
//     pre_dispatch : int, optional
//         If running in parallel, how many jobs to pre-dispatch so as to avoid
//         submitting all jobs at once. Should also be more than twice the number
//         of workers to avoid under-subscription. Default: 128.

//     Attributes
//     ----------
//     path : list[tuple[int]]
//         The best path found so far.
//     costs : list[int]
//         The list of each trial's costs found so far.
//     sizes : list[int]
//         The list of each trial's largest intermediate size so far.

//     See Also
//     --------
//     RandomGreedy
//     """
//     def __init__(self, max_repeats=32, max_time=None, minimize='flops', parallel=False, pre_dispatch=128):

//         if minimize not in ('flops', 'size'):
//             raise ValueError("`minimize` should be one of {'flops', 'size'}.")

//         self.max_repeats = max_repeats
//         self.max_time = max_time
//         self.minimize = minimize
//         self.better = paths.get_better_fn(minimize)
//         self.parallel = parallel
//         self.pre_dispatch = pre_dispatch

//         self.costs = []
//         self.sizes = []
//         self.best = {'flops': float('inf'), 'size': float('inf')}

//         self._repeats_start = 0

//     @property
//     def path(self):
//         """The best path found so far.
//         """
//         return paths.ssa_to_linear(self.best['ssa_path'])

//     @property
//     def parallel(self):
//         return self._parallel

//     @parallel.setter
//     def parallel(self, parallel):
//         # shutdown any previous executor if we are managing it
//         if getattr(self, '_managing_executor', False):
//             self._executor.shutdown()

//         self._parallel = parallel
//         self._managing_executor = False

//         if parallel is False:
//             self._executor = None
//             return

//         if parallel is True:
//             from concurrent.futures import ProcessPoolExecutor
//             self._executor = ProcessPoolExecutor()
//             self._managing_executor = True
//             return

//         if isinstance(parallel, numbers.Number):
//             from concurrent.futures import ProcessPoolExecutor
//             self._executor = ProcessPoolExecutor(parallel)
//             self._managing_executor = True
//             return

//         # assume a pool-executor has been supplied
//         self._executor = parallel

//     def _gen_results_parallel(self, repeats, trial_fn, args):
//         """Lazily generate results from an executor without submitting all jobs at once.
//         """
//         self._futures = deque()

//         # the idea here is to submit at least ``pre_dispatch`` jobs *before* we
//         # yield any results, then do both in tandem, before draining the queue
//         for r in repeats:
//             if len(self._futures) < self.pre_dispatch:
//                 self._futures.append(self._executor.submit(trial_fn, r, *args))
//                 continue
//             yield self._futures.popleft().result()

//         while self._futures:
//             yield self._futures.popleft().result()

//     def _cancel_futures(self):
//         if self._executor is not None:
//             for f in self._futures:
//                 f.cancel()

//     def setup(self, inputs, output, size_dict):
//         raise NotImplementedError

//     def __call__(self, inputs, output, size_dict, memory_limit):
//         self._check_args_against_first_call(inputs, output, size_dict)

//         # start a timer?
//         if self.max_time is not None:
//             t0 = time.time()

//         trial_fn, trial_args = self.setup(inputs, output, size_dict)

//         r_start = self._repeats_start + len(self.costs)
//         r_stop = r_start + self.max_repeats
//         repeats = range(r_start, r_stop)

//         # create the trials lazily
//         if self._executor is not None:
//             trials = self._gen_results_parallel(repeats, trial_fn, trial_args)
//         else:
//             trials = (trial_fn(r, *trial_args) for r in repeats)

//         # assess the trials
//         for ssa_path, cost, size in trials:

//             # keep track of all costs and sizes
//             self.costs.append(cost)
//             self.sizes.append(size)

//             # check if we have found a new best
//             found_new_best = self.better(cost, size, self.best['flops'], self.best['size'])

//             if found_new_best:
//                 self.best['flops'] = cost
//                 self.best['size'] = size
//                 self.best['ssa_path'] = ssa_path

//             # check if we have run out of time
//             if (self.max_time is not None) and (time.time() > t0 + self.max_time):
//                 break

//         self._cancel_futures()
//         return self.path

//     def __del__(self):
//         # if we created the parallel pool-executor, shut it down
//         if getattr(self, '_managing_executor', False):
//             self._executor.shutdown()

// def thermal_chooser(queue, remaining, nbranch=8, temperature=1, rel_temperature=True):
//     """A contraction 'chooser' that weights possible contractions using a
//     Boltzmann distribution. Explicitly, given costs ``c_i`` (with ``c_0`` the
//     smallest), the relative weights, ``w_i``, are computed as:

//         w_i = exp( -(c_i - c_0) / temperature)

//     Additionally, if ``rel_temperature`` is set, scale ``temperature`` by
//     ``abs(c_0)`` to account for likely fluctuating cost magnitudes during the
//     course of a contraction.

//     Parameters
//     ----------
//     queue : list
//         The heapified list of candidate contractions.
//     remaining : dict[str, int]
//         Mapping of remaining inputs' indices to the ssa id.
//     temperature : float, optional
//         When choosing a possible contraction, its relative probability will be
//         proportional to ``exp(-cost / temperature)``. Thus the larger
//         ``temperature`` is, the further random paths will stray from the normal
//         'greedy' path. Conversely, if set to zero, only paths with exactly the
//         same cost as the best at each step will be explored.
//     rel_temperature : bool, optional
//         Whether to normalize the ``temperature`` at each step to the scale of
//         the best cost. This is generally beneficial as the magnitude of costs
//         can vary significantly throughout a contraction.
//     nbranch : int, optional
//         How many potential paths to calculate probability for and choose from
//         at each step.

//     Returns
//     -------
//     cost, k1, k2, k12
//     """
//     n = 0
//     choices = []
//     while queue and n < nbranch:
//         cost, k1, k2, k12 = heapq.heappop(queue)
//         if k1 not in remaining or k2 not in remaining:
//             continue  # candidate is obsolete
//         choices.append((cost, k1, k2, k12))
//         n += 1

//     if n == 0:
//         return None
//     if n == 1:
//         return choices[0]

//     costs = [choice[0][0] for choice in choices]
//     cmin = costs[0]

//     # adjust by the overall scale to account for fluctuating absolute costs
//     if rel_temperature:
//         temperature *= max(1, abs(cmin))

//     # compute relative probability for each potential contraction
//     if temperature == 0.0:
//         energies = [1 if c == cmin else 0 for c in costs]
//     else:
//         # shift by cmin for numerical reasons
//         energies = [math.exp(-(c - cmin) / temperature) for c in costs]

//     # randomly choose a contraction based on energies
//     chosen, = random_choices(range(n), weights=energies)
//     cost, k1, k2, k12 = choices.pop(chosen)

//     # put the other choice back in the heap
//     for other in choices:
//         heapq.heappush(queue, other)

//     return cost, k1, k2, k12

// fn ssa_path_compute_cost(ssa_path: Vec<(usize, usize)>, mut inputs : Vec<Vec<usize>>, output: Vec<usize>, bond_dims: HashMap<usize, u64>) -> (u64, u64){
//     let total_cost = 0
//     let max_size = 0
//     for (i ,j) in ssa_path{
//         let flops_12 = _contract_cost(
//             &inputs[&i],
//             &inputs[&j],
//             self.tn.get_bond_dims(),
//         );
//         let (k12_tensor, size_12) = _contract_size(
//             &inputs[&i],
//             &inputs[&j]
//             self.tn.get_bond_dims(),
//         );
//         *inputs.entry(i).or_insert(k12_tensor) = k12_tensor;
//         total_cost += flops_12;
//         max_size = max(max_size, size_12);
//     }
//     (total_cost, max_size)
// }

// def _trial_greedy_ssa_path_and_cost(r, inputs, output, size_dict, choose_fn, cost_fn):
//     """A single, repeatable, greedy trial run. Returns ``ssa_path`` and cost.
//     """
//     if r == 0:
//         # always start with the standard greedy approach
//         choose_fn = None

//     random_seed(r)

//     ssa_path = paths.ssa_greedy_optimize(inputs, output, size_dict, choose_fn, cost_fn)
//     cost, size = ssa_path_compute_cost(ssa_path, inputs, output, size_dict)

//     return ssa_path, cost, size

// struct RandomGreedy{

// }

// class RandomGreedy(RandomOptimizer):
//     """

//     Parameters
//     ----------
//     cost_fn : callable, optional
//         A function that returns a heuristic 'cost' of a potential contraction
//         with which to sort candidates. Should have signature
//         ``cost_fn(size12, size1, size2, k12, k1, k2)``.
//     temperature : float, optional
//         When choosing a possible contraction, its relative probability will be
//         proportional to ``exp(-cost / temperature)``. Thus the larger
//         ``temperature`` is, the further random paths will stray from the normal
//         'greedy' path. Conversely, if set to zero, only paths with exactly the
//         same cost as the best at each step will be explored.
//     rel_temperature : bool, optional
//         Whether to normalize the ``temperature`` at each step to the scale of
//         the best cost. This is generally beneficial as the magnitude of costs
//         can vary significantly throughout a contraction. If False, the
//         algorithm will end up branching when the absolute cost is low, but
//         stick to the 'greedy' path when the cost is high - this can also be
//         beneficial.
//     nbranch : int, optional
//         How many potential paths to calculate probability for and choose from
//         at each step.
//     kwargs
//         Supplied to RandomOptimizer.

//     See Also
//     --------
//     RandomOptimizer
//     """
//     def __init__(self, cost_fn='memory-removed-jitter', temperature=1.0, rel_temperature=True, nbranch=8, **kwargs):
//         self.cost_fn = cost_fn
//         self.temperature = temperature
//         self.rel_temperature = rel_temperature
//         self.nbranch = nbranch
//         super().__init__(**kwargs)

//     @property
//     def choose_fn(self):
//         """The function that chooses which contraction to take - make this a
//         property so that ``temperature`` and ``nbranch`` etc. can be updated
//         between runs.
//         """
//         if self.nbranch == 1:
//             return None

//         return functools.partial(thermal_chooser,
//                                  temperature=self.temperature,
//                                  nbranch=self.nbranch,
//                                  rel_temperature=self.rel_temperature)

//     def setup(self, inputs, output, size_dict):
//         fn = _trial_greedy_ssa_path_and_cost
//         args = (inputs, output, size_dict, self.choose_fn, self.cost_fn)
//         return fn, args

// def random_greedy(inputs, output, idx_dict, memory_limit=None, **optimizer_kwargs):
//     """
//     """
//     optimizer = RandomGreedy(**optimizer_kwargs)
//     return optimizer(inputs, output, idx_dict, memory_limit)

// random_greedy_128 = functools.partial(random_greedy, max_repeats=128)
