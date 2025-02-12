use std::cell::RefCell;
use std::collections::HashSet;
use std::fs;
use std::io::Write;
use std::rc::Rc;
use std::time::{Duration, Instant};

use clap::{command, Parser};
use flexi_logger::{json_format, Duplicate, FileSpec, Logger};
use itertools::{iproduct, Itertools};
use log::{info, LevelFilter};
use mpi::topology::SimpleCommunicator;
use mpi::traits::{Communicator, CommunicatorCollectives};
use mpi::Rank;
use ordered_float::NotNan;
use rand::distributions::Standard;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use tensorcontraction::contractionpath::contraction_tree::balancing::{
    balance_partitions_iter, BalanceSettings, BalancingScheme, CommunicationScheme,
};
use tensorcontraction::contractionpath::contraction_tree::repartitioning::simulated_annealing::{
    IntermediatePartitioningModel, LeafPartitioningModel, NaivePartitioningModel,
};
use tensorcontraction::contractionpath::contraction_tree::repartitioning::{
    compute_solution, simulated_annealing,
};
use tensorcontraction::contractionpath::contraction_tree::ContractionTree;
use tensorcontraction::contractionpath::paths::tree_reconfiguration::TreeReconfigure;
use tensorcontraction::contractionpath::paths::{CostType, OptimizePath};
use tensorcontraction::mpi::communication::{
    broadcast_path, broadcast_serializing, extract_communication_path,
    intermediate_reduce_tensor_network, scatter_tensor_network,
};
use tensorcontraction::networks::connectivity::ConnectivityLayout;
use tensorcontraction::networks::random_circuit::random_circuit;
use tensorcontraction::tensornetwork::contraction::contract_tensor_network;
use tensorcontraction::tensornetwork::partitioning::find_partitioning;
use tensorcontraction::tensornetwork::partitioning::partition_config::PartitioningStrategy;
use tensorcontraction::tensornetwork::tensor::Tensor;
use tensorcontraction::types::ContractionIndex;

#[derive(Serialize, Deserialize, Debug)]
struct RunResult {
    seed: u64,
    num_qubits: usize,
    circuit_depth: usize,
    partitions: i32,
    method: String,
    optimization_time: Duration,
    time_to_solution: Duration,
    flops: f64,
}

#[derive(Debug, Parser)]
#[command(version, about, long_about=None)]
struct Cli {
    single_qubit_probability: f64,
    two_qubit_probability: f64,
    #[arg(short, long, value_delimiter = ',', num_args = 1..)]
    qubits: Vec<usize>,
    #[arg(short, long, value_delimiter = ',', num_args = 1..)]
    depths: Vec<usize>,
    #[arg(short, long, value_delimiter = ',', num_args = 1..)]
    partitions: Vec<i32>,
    #[arg(short, long, value_delimiter = ',', num_args = 1..)]
    include: Vec<String>,
    #[arg(short, long, value_delimiter = ',', num_args = 1..)]
    exclude: Vec<String>,
    #[arg(short, long, default_value_t = 10)]
    num_seeds: usize,
    out_file: String,
}

/// Sets up logging for rank `rank`. Each rank logs to a separate file and to stdout.
fn setup_logging_mpi(rank: Rank) {
    let _logger = Logger::with(LevelFilter::Debug)
        .format(json_format)
        .log_to_file(
            FileSpec::default()
                .discriminant(format!("rank{rank}"))
                .suppress_timestamp()
                .suffix("log.json"),
        )
        .duplicate_to_stdout(Duplicate::Info)
        .start()
        .unwrap();
}

#[derive(Default)]
struct Writer(Option<fs::File>);

impl Writer {
    fn new(filename: &str) -> Self {
        let file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(filename)
            .unwrap();
        Self(Some(file))
    }

    fn write(&mut self, result: RunResult) {
        if let Some(file) = &mut self.0 {
            serde_json::to_writer(file, &[result]).unwrap();
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
enum LogEntry {
    Trying(usize),
    Done(usize),
    Error(usize),
}

#[derive(Clone, Default, Debug, Serialize, Deserialize)]
struct Protocol(Vec<LogEntry>);

impl Protocol {
    fn from_file(file: fs::File) -> Self {
        let mut protocol: Protocol = serde_json::from_reader(file).unwrap();
        protocol.0.iter_mut().for_each(|entry| {
            if let LogEntry::Trying(id) = entry {
                *entry = LogEntry::Error(*id);
            }
        });
        protocol
    }

    fn write(&self, filename: &str) {
        let mut file = fs::File::create(filename).unwrap();
        serde_json::to_writer(&mut file, &self).unwrap();
        file.flush().unwrap();
    }

    fn contains(&self, id: &usize) -> bool {
        self.0.iter().any(|entry| match entry {
            LogEntry::Done(x) => x == id,
            LogEntry::Error(x) => x == id,
            LogEntry::Trying(_) => panic!("Trying entry should not be in the protocol"),
        })
    }

    fn write_trying(&mut self, id: usize) {
        self.0.push(LogEntry::Trying(id));
        self.write("protocol.json");
    }

    fn write_done(&mut self, id: usize) {
        let LogEntry::Trying(x) = self.0.pop().unwrap() else {
            panic!("Expected a trying entry when writing a done entry")
        };
        assert_eq!(id, x);
        self.0.push(LogEntry::Done(id));
        self.write("protocol.json");
    }
}

/// Parses a list of numbers and ranges into a set of numbers.
fn parse_range_list(entries: &[String]) -> HashSet<usize> {
    let mut out = HashSet::new();
    for entry in entries {
        if entry.contains("-") {
            // An inclusive range
            let parts = entry.split("-").collect_vec();
            assert_eq!(parts.len(), 2);
            let start = parts[0].parse().unwrap();
            let end = parts[1].parse().unwrap();
            for i in start..=end {
                out.insert(i);
            }
        } else {
            // A single number
            let number = entry.parse().unwrap();
            out.insert(number);
        }
    }
    out
}

/// Gets the main RNG used to generate the list of seeds.
fn get_main_rng(qubits: u64, depth: u64) -> StdRng {
    let seed = (qubits << 32) | depth;
    StdRng::seed_from_u64(seed)
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

    let single_qubit_probability = args.single_qubit_probability;
    let two_qubit_probability = args.two_qubit_probability;
    let connectivity = ConnectivityLayout::Sycamore;

    let qubit_range = args.qubits;
    let circuit_depth_range = args.depths;
    let partition_range = args.partitions;
    let seed_index_range = 0..args.num_seeds;
    assert!(
        args.include.is_empty() || args.exclude.is_empty(),
        "Can not pass 'include' and 'exclude' parameters at the same time"
    );
    let includes = parse_range_list(&args.include);
    let excludes = parse_range_list(&args.exclude);
    let communication_scheme = CommunicationScheme::RandomGreedyLatency;

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

    let methods: Vec<Rc<dyn MethodRun>> = vec![
        Rc::new(InitialProblem),
        Rc::new(GreedyBalance {
            iterations: 40,
            balancing_scheme: BalancingScheme::AlternatingIntermediateTensors { height_limit: 8 },
        }),
        Rc::new(Sa),
        Rc::new(Sad),
        Rc::new(Iad),
        Rc::new(Cotengra::default()),
    ];

    let scenarios = iproduct!(
        qubit_range,
        circuit_depth_range,
        seed_index_range,
        partition_range,
        methods,
    );

    let past_protocol = protocol.clone();
    for (id, scenario) in scenarios
        .enumerate()
        .filter(|(i, _)| includes.is_empty() || includes.contains(i))
        .filter(|(i, _)| !excludes.contains(i))
        .filter(|(i, _)| !past_protocol.contains(i))
    {
        let (num_qubits, circuit_depth, seed_index, num_partitions, method) = scenario;
        let rng = get_main_rng(num_qubits as u64, circuit_depth as u64);
        let seed = rng.sample_iter(Standard).nth(seed_index).unwrap();
        info!(num_qubits, circuit_depth, seed, num_partitions, single_qubit_probability, two_qubit_probability, connectivity:?, method=method.name(); "Doing run");

        if rank != 0 {
            // Other ranks are just for contraction
            perform_contraction(&Default::default(), Default::default(), &world);
        } else {
            // Generate circuit
            protocol.write_trying(id);
            let tensor = random_circuit(
                num_qubits,
                circuit_depth,
                single_qubit_probability,
                two_qubit_probability,
                &mut StdRng::seed_from_u64(seed),
                connectivity,
            );

            // Get initial partitioning
            let initial_partitioning =
                find_partitioning(&tensor, num_partitions, PartitioningStrategy::MinCut, true);

            // Run method
            let (partitioned_tensor, contraction_path, flops, optimization_time) = method
                .timed_run(
                    &tensor,
                    num_partitions,
                    &initial_partitioning,
                    communication_scheme,
                    &mut StdRng::seed_from_u64(seed),
                );

            // Perform the actual contraction
            let (final_tensor, time_to_solution) =
                perform_contraction(&partitioned_tensor, &contraction_path, &world);

            // Write the results
            writer.write(RunResult {
                seed,
                num_qubits,
                circuit_depth,
                partitions: method.actual_num_partitions().unwrap_or(num_partitions),
                method: method.name().into(),
                flops,
                optimization_time,
                time_to_solution,
            });

            std::hint::black_box(final_tensor);
            protocol.write_done(id);
        }
    }
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
    fn name(&self) -> &'static str;

    fn actual_num_partitions(&self) -> Option<i32> {
        None
    }

    fn run(
        &self,
        tensor: &Tensor,
        num_partitions: i32,
        initial_partitioning: &[usize],
        communication_scheme: CommunicationScheme,
        rng: &mut StdRng,
    ) -> (Tensor, Vec<ContractionIndex>, f64);

    fn timed_run(
        &self,
        tensor: &Tensor,
        num_partitions: i32,
        initial_partitioning: &[usize],
        communication_scheme: CommunicationScheme,
        rng: &mut StdRng,
    ) -> (Tensor, Vec<ContractionIndex>, f64, Duration) {
        let t0 = Instant::now();
        let result = self.run(
            tensor,
            num_partitions,
            initial_partitioning,
            communication_scheme,
            rng,
        );
        let duration = t0.elapsed();
        (result.0, result.1, result.2, duration)
    }
}

#[derive(Debug, Clone)]
struct InitialProblem;
impl MethodRun for InitialProblem {
    fn name(&self) -> &'static str {
        "Generic"
    }

    fn run(
        &self,
        tensor: &Tensor,
        _num_partitions: i32,
        initial_partitioning: &[usize],
        communication_scheme: CommunicationScheme,
        rng: &mut StdRng,
    ) -> (Tensor, Vec<ContractionIndex>, f64) {
        let (initial_partitioned_tensor, initial_contraction_path, original_flops) =
            compute_solution(
                tensor,
                initial_partitioning,
                communication_scheme,
                Some(rng),
            );
        (
            initial_partitioned_tensor,
            initial_contraction_path,
            original_flops,
        )
    }
}

#[derive(Debug, Clone)]
struct Sa;
impl MethodRun for Sa {
    fn name(&self) -> &'static str {
        "SA"
    }

    fn run(
        &self,
        tensor: &Tensor,
        num_partitions: i32,
        initial_partitioning: &[usize],
        communication_scheme: CommunicationScheme,
        rng: &mut StdRng,
    ) -> (Tensor, Vec<ContractionIndex>, f64) {
        let (partitioning, _): (Vec<usize>, NotNan<f64>) =
            simulated_annealing::balance_partitions::<_, NaivePartitioningModel>(
                tensor,
                num_partitions as usize,
                initial_partitioning.to_vec(),
                communication_scheme,
                rng,
                None,
            );

        let (partitioned_tensor, contraction_path, flops) =
            compute_solution(tensor, &partitioning, communication_scheme, Some(rng));
        (partitioned_tensor, contraction_path, flops)
    }
}

#[derive(Debug, Clone)]
struct Iad;
impl MethodRun for Iad {
    fn name(&self) -> &'static str {
        "IAD"
    }

    fn run(
        &self,
        tensor: &Tensor,
        num_partitions: i32,
        initial_partitioning: &[usize],
        communication_scheme: CommunicationScheme,
        rng: &mut StdRng,
    ) -> (Tensor, Vec<ContractionIndex>, f64) {
        let mut intermediate_tensors = vec![Tensor::default(); num_partitions as usize];
        for (index, partition) in initial_partitioning.iter().enumerate() {
            intermediate_tensors[*partition] ^= tensor.tensor(index);
        }

        let (_, initial_contraction_path, _) = compute_solution(
            tensor,
            initial_partitioning,
            communication_scheme,
            Some(rng),
        );

        let mut initial_contractions = Vec::new();
        for contraction_path in initial_contraction_path {
            if let ContractionIndex::Path(_, path) = contraction_path {
                initial_contractions.push(path);
            }
        }

        let (solution, _) =
            simulated_annealing::balance_partitions::<_, IntermediatePartitioningModel>(
                tensor,
                num_partitions as usize,
                (
                    initial_partitioning.to_vec(),
                    intermediate_tensors.to_vec(),
                    initial_contractions,
                ),
                communication_scheme,
                rng,
                None,
            );
        let (partitioning, ..) = solution;

        let (partitioned_tensor, contraction_path, flops) =
            compute_solution(tensor, &partitioning, communication_scheme, Some(rng));
        (partitioned_tensor, contraction_path, flops)
    }
}

#[derive(Debug, Clone)]
struct Sad;
impl MethodRun for Sad {
    fn name(&self) -> &'static str {
        "SAD"
    }

    fn run(
        &self,
        tensor: &Tensor,
        num_partitions: i32,
        initial_partitioning: &[usize],
        communication_scheme: CommunicationScheme,
        rng: &mut StdRng,
    ) -> (Tensor, Vec<ContractionIndex>, f64) {
        let mut intermediate_tensors = vec![Tensor::default(); num_partitions as usize];
        for (index, partition) in initial_partitioning.iter().enumerate() {
            intermediate_tensors[*partition] ^= tensor.tensor(index);
        }

        let (solution, _) = simulated_annealing::balance_partitions::<_, LeafPartitioningModel>(
            tensor,
            num_partitions as usize,
            (initial_partitioning.to_vec(), intermediate_tensors),
            communication_scheme,
            rng,
            None,
        );
        let (partitioning, ..) = solution;

        let (partitioned_tensor, contraction_path, flops) =
            compute_solution(tensor, &partitioning, communication_scheme, Some(rng));
        (partitioned_tensor, contraction_path, flops)
    }
}

#[derive(Debug, Clone)]
struct GreedyBalance {
    iterations: usize,
    balancing_scheme: BalancingScheme,
}
fn objective_function(a: &Tensor, b: &Tensor) -> f64 {
    a.size() + b.size() - (a ^ b).size()
}
impl MethodRun for GreedyBalance {
    fn name(&self) -> &'static str {
        "GreedyBalance"
    }

    fn run(
        &self,
        tensor: &Tensor,
        _num_partitions: i32,
        initial_partitioning: &[usize],
        communication_scheme: CommunicationScheme,
        rng: &mut StdRng,
    ) -> (Tensor, Vec<ContractionIndex>, f64) {
        let (initial_partitioned_tensor, initial_contraction_path, _) = compute_solution(
            tensor,
            initial_partitioning,
            communication_scheme,
            Some(rng),
        );

        let balance_settings = BalanceSettings::new(
            1,
            self.iterations,
            objective_function,
            communication_scheme,
            self.balancing_scheme,
            None,
        );
        let (best_iteration, partitioned_tensor, contraction_path, max_costs) =
            balance_partitions_iter(
                &initial_partitioned_tensor,
                &initial_contraction_path,
                balance_settings,
                None,
                rng,
            );
        let flops = max_costs[best_iteration];
        (partitioned_tensor, contraction_path, flops)
    }
}

#[derive(Debug, Clone, Default)]
struct Cotengra {
    last_used_num_partitions: RefCell<i32>,
}
impl MethodRun for Cotengra {
    fn name(&self) -> &'static str {
        "Cotengra"
    }

    fn actual_num_partitions(&self) -> Option<i32> {
        Some(*self.last_used_num_partitions.borrow())
    }

    fn run(
        &self,
        tensor: &Tensor,
        num_partitions: i32,
        _initial_partitioning: &[usize],
        communication_scheme: CommunicationScheme,
        rng: &mut StdRng,
    ) -> (Tensor, Vec<ContractionIndex>, f64) {
        let num_partitions = num_partitions as usize;
        let mut tree = TreeReconfigure::new(tensor, 4, CostType::Flops);
        tree.optimize_path();
        let best_path = tree.get_best_replace_path();

        let contraction_tree = ContractionTree::from_contraction_path(tensor, &best_path);

        let tree_root = contraction_tree.root_id().unwrap();
        let mut leaves = vec![];
        let mut traversal = vec![tree_root];
        while (leaves.len() + traversal.len()) < num_partitions && !traversal.is_empty() {
            let node_id = traversal.pop().unwrap();
            let node = contraction_tree.node(node_id);
            if node.is_leaf() {
                leaves.push(node_id);
            } else {
                traversal.push(node.left_child_id().unwrap());
                traversal.push(node.right_child_id().unwrap());
            }
        }
        traversal.append(&mut leaves);
        let num_partitions = traversal.len();
        let mut partitioning = vec![0; tensor.tensors().len()];
        for (i, partition_root) in traversal.iter().enumerate() {
            let leaf_tensors = contraction_tree.leaf_ids(*partition_root);
            for leaf in leaf_tensors {
                partitioning[leaf] = i;
            }
        }

        let (partitioned_tensor, contraction_path, flops) =
            compute_solution(tensor, &partitioning, communication_scheme, Some(rng));
        self.last_used_num_partitions.replace(num_partitions as i32);
        (partitioned_tensor, contraction_path, flops)
    }
}
