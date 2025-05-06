use std::cell::RefCell;
use std::fs::{self};
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use clap::Parser;
use cli::{Cli, Mode};
use flate2::read::ZlibDecoder;
use flate2::write::ZlibEncoder;
use flate2::Compression;
use itertools::{iproduct, Itertools};
use log::info;
use mpi::topology::SimpleCommunicator;
use mpi::traits::{Communicator, CommunicatorCollectives};
use num_complex::Complex64;
use ordered_float::NotNan;
use protocol::Protocol;
use rand::distributions::Standard;
use rand::rngs::StdRng;
use rand::{Rng, RngCore, SeedableRng};
use results::{OptimizationResult, RunResult, Writer};
use tensorcontraction::contractionpath::contraction_cost::{
    compute_memory_requirements, contract_path_cost, contract_size_tensors_exact,
};
use tensorcontraction::contractionpath::contraction_tree::balancing::{
    balance_partitions_iter, BalanceSettings, BalancingScheme, CommunicationScheme,
};
use tensorcontraction::contractionpath::contraction_tree::repartitioning::simulated_annealing::{
    IntermediatePartitioningModel, LeafPartitioningModel, NaivePartitioningModel,
};
use tensorcontraction::contractionpath::contraction_tree::repartitioning::simulated_annealing::{
    Metric, TerminationCondition,
};
use tensorcontraction::contractionpath::contraction_tree::repartitioning::{
    compute_solution, simulated_annealing,
};
use tensorcontraction::contractionpath::contraction_tree::ContractionTree;
use tensorcontraction::contractionpath::paths::cotengrust::{Cotengrust, OptMethod};
use tensorcontraction::contractionpath::paths::tree_annealing::TreeAnnealing;
use tensorcontraction::contractionpath::paths::tree_reconfiguration::TreeReconfigure;
use tensorcontraction::contractionpath::paths::tree_tempering::TreeTempering;
use tensorcontraction::contractionpath::paths::{CostType, OptimizePath};
use tensorcontraction::mpi::communication::{
    broadcast_path, broadcast_serializing, extract_communication_path,
    intermediate_reduce_tensor_network, scatter_tensor_network,
};
use tensorcontraction::qasm::qasm_to_tensornetwork::create_tensornetwork;
use tensorcontraction::tensornetwork::contraction::contract_tensor_network;
use tensorcontraction::tensornetwork::partitioning::find_partitioning;
use tensorcontraction::tensornetwork::partitioning::partition_config::PartitioningStrategy;
use tensorcontraction::tensornetwork::tensor::Tensor;
use tensorcontraction::tensornetwork::tensordata::TensorData;
use tensorcontraction::types::ContractionIndex;
use utils::{hash_str, parse_range_list, setup_logging_mpi};

mod cli;
mod protocol;
mod results;
mod utils;

const TERMINATION: TerminationCondition = TerminationCondition::Iterations {
    n_iter: 1000,
    patience: 500,
};

const ANNEAL_ITERATIONS: TerminationCondition = TerminationCondition::Iterations {
    n_iter: 300,
    patience: 100,
};

const TEMPER_ITERATIONS: TerminationCondition = TerminationCondition::Iterations {
    n_iter: 300,
    patience: 100,
};

/// Reads a circuit from the given qasm file.
fn read_circuit(file: &str) -> Tensor {
    static LAST_RETURN: Mutex<Option<(String, Tensor)>> = Mutex::new(None);
    let mut last_values = LAST_RETURN.lock().unwrap();
    if let Some((arg, out)) = &*last_values {
        if arg == file {
            return out.clone();
        }
    }
    let source = fs::read_to_string(file).unwrap();
    let (mut tensor, open_legs) = create_tensornetwork(source);

    // Add bras to each open leg
    for leg in open_legs {
        let mut bra = Tensor::new_from_const(vec![leg], 2);
        bra.set_tensor_data(TensorData::new_from_data(
            &[2],
            vec![Complex64::ONE, Complex64::ONE],
            None,
        ));
        tensor.push_tensor(bra);
    }

    last_values.replace((file.into(), tensor.clone()));
    tensor
}

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let size = world.size();
    let rank = world.rank();
    let root = world.process_at_rank(0);
    setup_logging_mpi(rank);
    info!(rank, size; "Logging setup");

    let args = Cli::parse();

    let mut writer = if rank == 0 {
        Writer::new(&args.out_file)
    } else {
        Writer::default()
    };

    // Set the parameters
    let partition_range = args.partitions;
    let seed_index_range = 0..args.num_seeds;
    assert!(
        args.include.is_empty() || args.exclude.is_empty(),
        "Can not pass 'include' and 'exclude' parameters at the same time"
    );
    let includes = parse_range_list(&args.include);
    let excludes = parse_range_list(&args.exclude);
    let communication_scheme = CommunicationScheme::RandomGreedy;
    let mode = args.mode;
    let cache_dir = args.cache_dir;

    // Create the cache dir
    fs::create_dir_all(&cache_dir).unwrap();

    // Read the protocol and broadcast it to all ranks
    let protocol = if rank == 0 {
        if let Ok(log_file) = fs::File::open("protocol.json") {
            Protocol::from_file(log_file)
        } else {
            Default::default()
        }
    } else {
        Default::default()
    };
    let mut protocol = broadcast_serializing(protocol, &root);

    // Define the methods to run
    let methods: Vec<Rc<dyn MethodRun>> = vec![
        Rc::new(InitialProblem),
        // Rc::new(GreedyBalance {
        //     iterations: 40,
        //     balancing_scheme: BalancingScheme::AlternatingIntermediateTensors { height_limit: 8 },
        // }),
        // Rc::new(GreedyBalance {
        //     iterations: 40,
        //     balancing_scheme: BalancingScheme::AlternatingTreeTensors { height_limit: 8 },
        // }),
        // Rc::new(Sa),
        // Rc::new(Sad),
        Rc::new(Iad),
        //Rc::new(Cotengra::default()),
        //Rc::new(CotengraAnneal::default()),
        //Rc::new(CotengraTempering::default()),
    ];

    // Read the circuit directory
    let folder = PathBuf::from("circuits/");
    let files = fs::read_dir(&folder).unwrap();
    let files = files
        .map(|entry| {
            entry
                .unwrap()
                .path()
                .into_os_string()
                .into_string()
                .unwrap()
        })
        .collect_vec();
    let file_range = 0..files.len();

    let scenarios = iproduct!(file_range, seed_index_range, partition_range, methods);

    // Run the scenarios
    let past_protocol = protocol.clone();
    for (id, scenario) in scenarios
        .enumerate()
        .filter(|(i, _)| includes.is_empty() || includes.contains(i))
        .filter(|(i, _)| !excludes.contains(i))
        .filter(|(i, _)| !past_protocol.contains(i))
    {
        let (file_index, seed_index, num_partitions, method) = scenario;
        let file = &files[file_index];
        let file_hash = hash_str(file);
        let rng = StdRng::seed_from_u64(file_hash);
        let seed = rng.sample_iter(Standard).nth(seed_index).unwrap();
        info!(file=file, seed, num_partitions, method=method.name(); "Doing run");

        let key = format!(
            "{communication_scheme:?}_{file_hash}_{seed}_{num_partitions}_{}",
            method.name(),
        );

        if rank != 0 {
            // Other ranks are just for contraction
            perform_contraction(&Default::default(), Default::default(), &world);
        } else {
            protocol.write_trying(id);

            match mode {
                Mode::Sweep => do_sweep(
                    file,
                    &cache_dir,
                    &key,
                    &method,
                    num_partitions,
                    communication_scheme,
                    seed,
                    &mut writer,
                ),
                Mode::Run => do_run(
                    file,
                    &cache_dir,
                    &key,
                    &method,
                    num_partitions,
                    seed,
                    &world,
                    &mut writer,
                ),
            }

            protocol.write_done(id);
        }
    }
}

fn write_to_cache(
    directory: &str,
    key: &str,
    partitioned_tensor: &Tensor,
    contraction_path: &[ContractionIndex],
) {
    let file = fs::File::create(format!("{directory}/{key}")).unwrap();
    let stream = ZlibEncoder::new(file, Compression::default());
    let serializable = (partitioned_tensor, contraction_path);
    bincode::serialize_into(stream, &serializable).unwrap();
}

fn read_from_cache(directory: &str, key: &str) -> (Tensor, Vec<ContractionIndex>) {
    let file = fs::File::open(format!("{directory}/{key}")).unwrap();
    let stream = ZlibDecoder::new(file);
    let (deserializable, path): (Tensor, Vec<ContractionIndex>) =
        bincode::deserialize_from(stream).unwrap();
    (deserializable, path)
}

/// Computes the flops and memory when using Greedy without partitioning.
/// Uses the file as a cache key.
fn serial_cost(tensor: &Tensor, file: &str) -> (f64, f64) {
    static LAST_RETURN: Mutex<Option<(String, (f64, f64))>> = Mutex::new(None);
    let mut last_values = LAST_RETURN.lock().unwrap();
    if let Some((arg, out)) = &*last_values {
        if arg == file {
            return out.clone();
        }
    }
    let mut opt = Cotengrust::new(tensor, OptMethod::Greedy);
    opt.optimize_path();
    let cost = opt.get_best_flops();
    let memory = compute_memory_requirements(
        tensor.tensors(),
        &opt.get_best_replace_path(),
        contract_size_tensors_exact,
    );
    last_values.replace((file.into(), (cost, memory)));
    (cost, memory)
}

#[allow(clippy::too_many_arguments)]
fn do_sweep(
    file: &str,
    cache_dir: &str,
    key: &str,
    method: &Rc<dyn MethodRun>,
    num_partitions: i32,
    communication_scheme: CommunicationScheme,
    seed: u64,
    writer: &mut Writer,
) {
    // Generate circuit
    let tensor = read_circuit(file);

    // Compute the serial cost
    let (greedy_flops, greedy_memory) = serial_cost(&tensor, file);

    // Get initial partitioning
    let initial_partitioning =
        find_partitioning(&tensor, num_partitions, PartitioningStrategy::MinCut, true);

    // Run method
    let ((partitioned_tensor, contraction_path, parallel_flops, serial_flops), optimization_time) =
        method.timed_run(
            &tensor,
            num_partitions,
            &initial_partitioning,
            communication_scheme,
            &mut StdRng::seed_from_u64(seed),
        );

    // Compute the memory
    let memory = compute_memory_requirements(
        partitioned_tensor.tensors(),
        &contraction_path,
        contract_size_tensors_exact,
    );

    // Store the partitioning and contraction path
    write_to_cache(cache_dir, key, &partitioned_tensor, &contraction_path);

    // Write the results
    writer.write(OptimizationResult {
        file: file.into(),
        seed,
        partitions: num_partitions,
        actual_partitions: method.actual_num_partitions().unwrap_or(num_partitions),
        method: method.name(),
        serial_flops: greedy_flops,
        serial_memory: greedy_memory,
        flops_sum: serial_flops,
        flops: parallel_flops,
        memory,
        optimization_time,
    });
}

#[allow(clippy::too_many_arguments)]
fn do_run(
    file: &str,
    cache_dir: &str,
    key: &str,
    method: &Rc<dyn MethodRun>,
    num_partitions: i32,
    seed: u64,
    world: &SimpleCommunicator,
    writer: &mut Writer,
) {
    // Load partitioned tensor and contraction path
    let (partitioned_tensor, contraction_path) = read_from_cache(cache_dir, key);

    // Perform the actual contraction
    let (final_tensor, time_to_solution) =
        perform_contraction(&partitioned_tensor, &contraction_path, world);

    // Write the results
    writer.write(RunResult {
        file: file.into(),
        seed,
        partitions: num_partitions,
        method: method.name(),
        time_to_solution,
    });

    std::hint::black_box(final_tensor);
}

fn perform_contraction(
    partitioned_tensor: &Tensor,
    contraction_path: &[ContractionIndex],
    world: &SimpleCommunicator,
) -> (Tensor, Duration) {
    let rank = world.rank();
    let root = world.process_at_rank(0);
    world.barrier();

    let t0 = Instant::now();

    // Distribute the communication path.
    // This is done first, because the ranks are still synchronized.
    // Doing it after contraction would be an implicit barrier, so all the ranks
    // would have to wait for the slowest rank to finish the contraction.
    let mut communication_path = if rank == 0 {
        extract_communication_path(contraction_path)
    } else {
        Default::default()
    };
    broadcast_path(&mut communication_path, &root);

    // Distribute tensor network
    let (mut local_tn, local_path, comm) = scatter_tensor_network(
        partitioned_tensor,
        contraction_path,
        rank,
        world.size(),
        world,
    );

    // Contract the local tensor network
    local_tn = contract_tensor_network(local_tn, &local_path);

    // Reduce the tensor network
    intermediate_reduce_tensor_network(&mut local_tn, &communication_path, rank, world, &comm);

    let duration = t0.elapsed();

    (local_tn, duration)
}

trait MethodRun {
    /// Returns the name of the method.
    fn name(&self) -> String;

    /// Returns the actual number of partitions used by the method. This is useful
    /// for methods that dynamically determine the number of partitions.
    ///
    /// If the method did not use a different number of partitions, return `None`.
    fn actual_num_partitions(&self) -> Option<i32> {
        None
    }

    /// Runs the method on the given tensor, returning the partitioned tensor, the
    /// contraction path and the number of operations it takes to constract.
    fn run(
        &self,
        tensor: &Tensor,
        num_partitions: i32,
        initial_partitioning: &[usize],
        communication_scheme: CommunicationScheme,
        rng: &mut StdRng,
    ) -> (Tensor, Vec<ContractionIndex>, f64, f64);

    /// Runs the method on the given tensor, returning the partitioned tensor, the
    /// contraction path, the number of operations it takes to constract and the
    /// time it took to run the method.
    fn timed_run(
        &self,
        tensor: &Tensor,
        num_partitions: i32,
        initial_partitioning: &[usize],
        communication_scheme: CommunicationScheme,
        rng: &mut StdRng,
    ) -> ((Tensor, Vec<ContractionIndex>, f64, f64), Duration) {
        let t0 = Instant::now();
        let result = self.run(
            tensor,
            num_partitions,
            initial_partitioning,
            communication_scheme,
            rng,
        );
        let duration = t0.elapsed();
        (result, duration)
    }
}

#[derive(Debug, Clone)]
struct InitialProblem;
impl MethodRun for InitialProblem {
    fn name(&self) -> String {
        "Generic".into()
    }

    fn run(
        &self,
        tensor: &Tensor,
        _num_partitions: i32,
        initial_partitioning: &[usize],
        communication_scheme: CommunicationScheme,
        rng: &mut StdRng,
    ) -> (Tensor, Vec<ContractionIndex>, f64, f64) {
        let (
            initial_partitioned_tensor,
            initial_contraction_path,
            original_parallel_flops,
            original_sum_flops,
        ) = compute_solution(
            tensor,
            initial_partitioning,
            communication_scheme,
            Some(rng),
        );
        (
            initial_partitioned_tensor,
            initial_contraction_path,
            original_parallel_flops,
            original_sum_flops,
        )
    }
}

#[derive(Debug, Clone)]
struct Sa;
impl MethodRun for Sa {
    fn name(&self) -> String {
        "SA".into()
    }

    fn run(
        &self,
        tensor: &Tensor,
        num_partitions: i32,
        initial_partitioning: &[usize],
        communication_scheme: CommunicationScheme,
        rng: &mut StdRng,
    ) -> (Tensor, Vec<ContractionIndex>, f64, f64) {
        let (partitioning, _): (Vec<usize>, NotNan<f64>) = simulated_annealing::balance_partitions(
            NaivePartitioningModel {
                tensor,
                num_partitions: num_partitions as _,
                communication_scheme,
                memory_limit: None,
            },
            initial_partitioning.to_vec(),
            rng,
            &TERMINATION,
        );

        let (partitioned_tensor, contraction_path, parallel_flops, serial_flops) =
            compute_solution(tensor, &partitioning, communication_scheme, Some(rng));
        (
            partitioned_tensor,
            contraction_path,
            parallel_flops,
            serial_flops,
        )
    }
}

#[derive(Debug, Clone)]
struct Iad;
impl MethodRun for Iad {
    fn name(&self) -> String {
        "IAD".into()
    }

    fn run(
        &self,
        tensor: &Tensor,
        num_partitions: i32,
        initial_partitioning: &[usize],
        communication_scheme: CommunicationScheme,
        rng: &mut StdRng,
    ) -> (Tensor, Vec<ContractionIndex>, f64, f64) {
        let mut intermediate_tensors = vec![Tensor::default(); num_partitions as usize];
        for (index, partition) in initial_partitioning.iter().enumerate() {
            intermediate_tensors[*partition] ^= tensor.tensor(index);
        }

        let (_, initial_contraction_path, _, _) = compute_solution(
            tensor,
            initial_partitioning,
            communication_scheme,
            Some(&mut rng.clone()),
        );

        let mut initial_contractions = Vec::new();
        for contraction_path in initial_contraction_path {
            if let ContractionIndex::Path(_, path) = contraction_path {
                initial_contractions.push(path);
            }
        }

        let (solution, _) = simulated_annealing::balance_partitions(
            IntermediatePartitioningModel {
                tensor,
                communication_scheme,
                memory_limit: None,
                metric: Metric::ParallelWithTieBreaking,
            },
            (
                initial_partitioning.to_vec(),
                intermediate_tensors.to_vec(),
                initial_contractions,
            ),
            rng,
            &TERMINATION,
        );
        let (partitioning, ..) = solution;

        let (partitioned_tensor, contraction_path, parallel_flops, sum_flops) =
            compute_solution(tensor, &partitioning, communication_scheme, Some(rng));
        (
            partitioned_tensor,
            contraction_path,
            parallel_flops,
            sum_flops,
        )
    }
}

// #[derive(Debug, Clone)]
// struct Sad;
// impl MethodRun for Sad {
//     fn name(&self) -> String {
//         "SAD".into()
//     }

//     fn run(
//         &self,
//         tensor: &Tensor,
//         num_partitions: i32,
//         initial_partitioning: &[usize],
//         communication_scheme: CommunicationScheme,
//         rng: &mut StdRng,
//     ) -> (Tensor, Vec<ContractionIndex>, f64) {
//         let mut intermediate_tensors = vec![Tensor::default(); num_partitions as usize];
//         for (index, partition) in initial_partitioning.iter().enumerate() {
//             intermediate_tensors[*partition] ^= tensor.tensor(index);
//         }

//         let (solution, _) = simulated_annealing::balance_partitions::<_, LeafPartitioningModel>(
//             tensor,
//             num_partitions as usize,
//             (initial_partitioning.to_vec(), intermediate_tensors),
//             communication_scheme,
//             rng,
//             None,
//             &TERMINATION,
//         );
//         let (partitioning, ..) = solution;

//         let (partitioned_tensor, contraction_path, flops) =
//             compute_solution(tensor, &partitioning, communication_scheme, Some(rng));
//         (partitioned_tensor, contraction_path, flops)
//     }
// }

// #[derive(Debug, Clone)]
// struct GreedyBalance {
//     iterations: usize,
//     balancing_scheme: BalancingScheme,
// }
// fn objective_function(a: &Tensor, b: &Tensor) -> f64 {
//     a.size() + b.size() - (a ^ b).size()
// }
// impl MethodRun for GreedyBalance {
//     fn name(&self) -> String {
//         match &self.balancing_scheme {
//             BalancingScheme::AlternatingIntermediateTensors { .. } => "GreedyIntermediate".into(),
//             BalancingScheme::AlternatingTreeTensors { .. } => "GreedyTree".into(),
//             _ => panic!(),
//         }
//     }

//     fn run(
//         &self,
//         tensor: &Tensor,
//         _num_partitions: i32,
//         initial_partitioning: &[usize],
//         communication_scheme: CommunicationScheme,
//         rng: &mut StdRng,
//     ) -> (Tensor, Vec<ContractionIndex>, f64) {
//         let (initial_partitioned_tensor, initial_contraction_path, _) = compute_solution(
//             tensor,
//             initial_partitioning,
//             communication_scheme,
//             Some(&mut rng.clone()),
//         );

//         let balance_settings = BalanceSettings::new(
//             1,
//             self.iterations,
//             objective_function,
//             communication_scheme,
//             self.balancing_scheme,
//             None,
//         );
//         let (best_iteration, partitioned_tensor, contraction_path, max_costs) =
//             balance_partitions_iter(
//                 &initial_partitioned_tensor,
//                 &initial_contraction_path,
//                 balance_settings,
//                 None,
//                 rng,
//             );
//         let flops = max_costs[best_iteration];
//         (partitioned_tensor, contraction_path, flops)
//     }
// }

// #[derive(Debug, Clone, Default)]
// struct Cotengra {
//     last_used_num_partitions: RefCell<i32>,
// }
// impl MethodRun for Cotengra {
//     fn name(&self) -> String {
//         "Cotengra".into()
//     }

//     fn actual_num_partitions(&self) -> Option<i32> {
//         Some(*self.last_used_num_partitions.borrow())
//     }

//     fn run(
//         &self,
//         tensor: &Tensor,
//         num_partitions: i32,
//         _initial_partitioning: &[usize],
//         communication_scheme: CommunicationScheme,
//         rng: &mut StdRng,
//     ) -> (Tensor, Vec<ContractionIndex>, f64) {
//         let num_partitions = num_partitions as usize;
//         let mut tree = TreeReconfigure::new(tensor, 8, CostType::Flops);
//         tree.optimize_path();
//         let best_path = tree.get_best_replace_path();

//         let contraction_tree = ContractionTree::from_contraction_path(tensor, &best_path);

//         let tree_root = contraction_tree.root_id().unwrap();
//         let mut leaves = vec![];
//         let mut traversal = vec![tree_root];
//         while (leaves.len() + traversal.len()) < num_partitions && !traversal.is_empty() {
//             let node_id = traversal.pop().unwrap();
//             let node = contraction_tree.node(node_id);
//             if node.is_leaf() {
//                 leaves.push(node_id);
//             } else {
//                 traversal.push(node.left_child_id().unwrap());
//                 traversal.push(node.right_child_id().unwrap());
//             }
//         }
//         traversal.append(&mut leaves);
//         let num_partitions = traversal.len();
//         let mut partitioning = vec![0; tensor.tensors().len()];
//         for (i, partition_root) in traversal.iter().enumerate() {
//             let leaf_tensors = contraction_tree.leaf_ids(*partition_root);
//             for leaf in leaf_tensors {
//                 partitioning[leaf] = i;
//             }
//         }

//         let (partitioned_tensor, contraction_path, flops) =
//             compute_solution(tensor, &partitioning, communication_scheme, Some(rng));
//         self.last_used_num_partitions.replace(num_partitions as i32);
//         (partitioned_tensor, contraction_path, flops)
//     }
// }

// #[derive(Debug, Clone, Default)]
// struct CotengraTempering {
//     last_used_num_partitions: RefCell<i32>,
// }
// impl MethodRun for CotengraTempering {
//     fn name(&self) -> String {
//         "CotengraTempering".into()
//     }

//     fn actual_num_partitions(&self) -> Option<i32> {
//         Some(*self.last_used_num_partitions.borrow())
//     }

//     fn run(
//         &self,
//         tensor: &Tensor,
//         num_partitions: i32,
//         _initial_partitioning: &[usize],
//         communication_scheme: CommunicationScheme,
//         rng: &mut StdRng,
//     ) -> (Tensor, Vec<ContractionIndex>, f64) {
//         let seed = rng.next_u64();
//         let num_partitions = num_partitions as usize;
//         let mut tree = TreeTempering::new(tensor, Some(seed), CostType::Flops, TEMPER_ITERATIONS);
//         tree.optimize_path();
//         let best_path = tree.get_best_replace_path();

//         let contraction_tree = ContractionTree::from_contraction_path(tensor, &best_path);

//         let tree_root = contraction_tree.root_id().unwrap();
//         let mut leaves = vec![];
//         let mut traversal = vec![tree_root];
//         while (leaves.len() + traversal.len()) < num_partitions && !traversal.is_empty() {
//             let node_id = traversal.pop().unwrap();
//             let node = contraction_tree.node(node_id);
//             if node.is_leaf() {
//                 leaves.push(node_id);
//             } else {
//                 traversal.push(node.left_child_id().unwrap());
//                 traversal.push(node.right_child_id().unwrap());
//             }
//         }
//         traversal.append(&mut leaves);
//         let num_partitions = traversal.len();
//         let mut partitioning = vec![0; tensor.tensors().len()];
//         for (i, partition_root) in traversal.iter().enumerate() {
//             let leaf_tensors = contraction_tree.leaf_ids(*partition_root);
//             for leaf in leaf_tensors {
//                 partitioning[leaf] = i;
//             }
//         }

//         let (partitioned_tensor, contraction_path, flops) =
//             compute_solution(tensor, &partitioning, communication_scheme, Some(rng));
//         self.last_used_num_partitions.replace(num_partitions as i32);
//         (partitioned_tensor, contraction_path, flops)
//     }
// }

// #[derive(Debug, Clone, Default)]
// struct CotengraAnneal {
//     last_used_num_partitions: RefCell<i32>,
// }
// impl MethodRun for CotengraAnneal {
//     fn name(&self) -> String {
//         "CotengraAnneal".into()
//     }

//     fn actual_num_partitions(&self) -> Option<i32> {
//         Some(*self.last_used_num_partitions.borrow())
//     }

//     fn run(
//         &self,
//         tensor: &Tensor,
//         num_partitions: i32,
//         _initial_partitioning: &[usize],
//         communication_scheme: CommunicationScheme,
//         rng: &mut StdRng,
//     ) -> (Tensor, Vec<ContractionIndex>, f64) {
//         let seed = rng.next_u64();
//         let num_partitions = num_partitions as usize;
//         let mut tree = TreeAnnealing::new(tensor, Some(seed), CostType::Flops, ANNEAL_ITERATIONS);
//         tree.optimize_path();
//         let best_path = tree.get_best_replace_path();

//         let contraction_tree = ContractionTree::from_contraction_path(tensor, &best_path);

//         let tree_root = contraction_tree.root_id().unwrap();
//         let mut leaves = vec![];
//         let mut traversal = vec![tree_root];
//         while (leaves.len() + traversal.len()) < num_partitions && !traversal.is_empty() {
//             let node_id = traversal.pop().unwrap();
//             let node = contraction_tree.node(node_id);
//             if node.is_leaf() {
//                 leaves.push(node_id);
//             } else {
//                 traversal.push(node.left_child_id().unwrap());
//                 traversal.push(node.right_child_id().unwrap());
//             }
//         }
//         traversal.append(&mut leaves);
//         let num_partitions = traversal.len();
//         let mut partitioning = vec![0; tensor.tensors().len()];
//         for (i, partition_root) in traversal.iter().enumerate() {
//             let leaf_tensors = contraction_tree.leaf_ids(*partition_root);
//             for leaf in leaf_tensors {
//                 partitioning[leaf] = i;
//             }
//         }

//         let (partitioned_tensor, contraction_path, flops) =
//             compute_solution(tensor, &partitioning, communication_scheme, Some(rng));
//         self.last_used_num_partitions.replace(num_partitions as i32);
//         (partitioned_tensor, contraction_path, flops)
//     }
// }
