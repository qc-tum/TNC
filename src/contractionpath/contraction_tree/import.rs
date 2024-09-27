use chrono::{DateTime, Utc};
use itertools::Itertools;
use regex::RegexSet;
use rustc_hash::FxHashMap;
use std::{
    cell::RefCell,
    fs,
    rc::{Rc, Weak},
};

use crate::{
    contractionpath::contraction_tree::{export::to_pdf, node::Node},
    types::ContractionIndex,
    utils::traits::HashMapInsertNew,
};

use super::{
    export::{DendogramEntry, COLORS, COMMUNICATION_COLOR},
    ContractionTree,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Direction {
    Send,
    Recv,
}

pub fn logs_to_pdf(filename: &str, suffix: &str, ranks: usize, output: &str) {
    let height = 120f64;
    let width = 80f64;
    let LogsToTreeResult {
        tree: contraction_tree,
        tensor_cost: tensor_position,
        tensor_color,
        communication_data: mut communication_logging,
    } = logs_to_tree(filename, suffix, ranks);

    let flat_contraction_path =
        contraction_tree.to_flat_contraction_path(contraction_tree.root_id().unwrap(), false);

    let leaf_nodes = contraction_tree.leaf_ids(contraction_tree.root_id().unwrap());

    let num_leaf_nodes = leaf_nodes.len();
    let mut dendogram_entries = Vec::new();

    let mut next_x = 0f64;
    let mut tensor_x_position = FxHashMap::default();

    for contraction_index in &flat_contraction_path {
        if let ContractionIndex::Pair(i, j) = contraction_index {
            if contraction_tree.node(*i).is_leaf() {
                tensor_x_position.insert_new(*i, next_x);
                dendogram_entries.push(DendogramEntry {
                    id: *i,
                    x: tensor_x_position[i],
                    y: 0f64,
                    cost: 0f64,
                    color: tensor_color[i].clone(),
                    children: None,
                });
                next_x += 1f64;
            }

            if contraction_tree.node(*j).is_leaf() {
                tensor_x_position.insert_new(*j, next_x);
                dendogram_entries.push(DendogramEntry {
                    id: *j,
                    x: tensor_x_position[j],
                    y: 0f64,
                    cost: 0f64,
                    color: tensor_color[j].clone(),
                    children: None,
                });
                next_x += 1f64;
            }

            let x1 = tensor_x_position[i];
            let x2 = tensor_x_position[j];
            let new_x = (x1 + x2) / 2f64;
            if let Some(parent_id) = contraction_tree.node(*i).parent_id() {
                let y = tensor_position[&parent_id];
                let color = tensor_color[&parent_id].clone();
                dendogram_entries.push(DendogramEntry {
                    id: parent_id,
                    x: new_x,
                    y,
                    cost: y,
                    color,
                    children: Some((*i, *j)),
                });
                tensor_x_position.insert_new(parent_id, new_x);
            }
        }
    }

    let width_scaling = width / num_leaf_nodes as f64;
    let height_scaling = height / dendogram_entries.last().unwrap().cost;

    for dendogram_entry in &mut dendogram_entries {
        dendogram_entry.x *= width_scaling;
        dendogram_entry.y *= height_scaling;
    }
    for (start, end) in communication_logging.values_mut() {
        *start *= height_scaling;
        *end *= height_scaling;
    }
    to_pdf(output, &dendogram_entries, Some(communication_logging));
}

#[derive(Debug, Clone)]
pub struct LogsToTreeResult {
    /// The reconstructed contraction tree
    tree: ContractionTree,
    /// Maps node id in contraction tree to a position in pdf
    tensor_cost: FxHashMap<usize, f64>,
    /// Maps node id in contraction tree to a color
    tensor_color: FxHashMap<usize, String>,
    /// Stores communications between two ranks and their start and end times
    communication_data: FxHashMap<(Direction, usize, usize), (f64, f64)>,
}

/// Reads logging results and reconstructs construction operations, indicating timings for contraction and communication.
/// Prints "filename" pdf with dendogram of the overall contraction operation.
pub fn logs_to_tree(filename: &str, suffix: &str, ranks: usize) -> LogsToTreeResult {
    // Maps node id in contraction tree to a position in pdf
    let mut tensor_cost = FxHashMap::default();
    // Counter to keep track of total number of intermediate + leaf nodes between ranks.
    let mut tensor_count = 0;

    // Tracks the root node of each rank's partition, ordered by rank by construction
    let mut partition_root_nodes = Vec::new();

    // Hashmap of all nodes
    let mut remaining_nodes = FxHashMap::default();
    // Hashmap of all communication timestamps
    let mut communication_logging = FxHashMap::default();
    // Tracks all communication taking place, only lists receiving rank for each instance to prevent doubles.
    let mut communication_path = Vec::new();

    // Tracks the color of each node
    let mut tensor_color = FxHashMap::default();
    let mut logging_start = DateTime::fixed_offset(&Utc::now());

    for (rank, color) in (0..ranks).zip(COLORS.iter().cycle()) {
        let LogToSubtreeResult {
            nodes,
            mut local_communication_path,
            communication_timestamps,
            contraction_start,
        } = log_to_subtree(
            &format!("{filename}_rank{rank}.{suffix}"),
            &mut tensor_cost,
            &mut tensor_count,
            rank,
        );
        logging_start = logging_start.min(contraction_start);
        for &key in nodes.keys() {
            tensor_color.insert(key, String::from(*color));
        }
        communication_logging.extend(communication_timestamps);
        remaining_nodes.extend(nodes);
        communication_path.append(&mut local_communication_path);
        partition_root_nodes.push(Rc::clone(&remaining_nodes[&(tensor_count - 1)]));
    }

    let mut tensor_cost = tensor_cost
        .iter()
        .map(|(k, v)| (*k, (*v - logging_start).num_microseconds().unwrap() as f64))
        .collect::<FxHashMap<_, _>>();

    let partition_tensor_ids = partition_root_nodes
        .iter()
        .map(|node| node.borrow().id)
        .collect::<Vec<_>>();

    // Sort communication by time to ensure no violation of data dependencies
    let communication_path = communication_path
        .iter_mut()
        .sorted_unstable()
        .collect::<Vec<_>>();

    let mut communication_data = FxHashMap::default();
    for (rank1, rank2, timestamp) in communication_path {
        let left_child = Rc::clone(&partition_root_nodes[*rank1]);
        let right_child = Rc::clone(&partition_root_nodes[*rank2]);

        let send_timestamps = communication_logging.remove(&(Direction::Send, *rank1, *rank2));
        if let Some((timestamp1, timestamp2)) = send_timestamps {
            communication_data.insert(
                (
                    Direction::Send,
                    left_child.as_ref().borrow().id(),
                    right_child.as_ref().borrow().id(),
                ),
                (
                    (timestamp1 - logging_start).num_microseconds().unwrap() as f64,
                    (timestamp2 - logging_start).num_microseconds().unwrap() as f64,
                ),
            );
        };

        let recv_timestamps = communication_logging.remove(&(Direction::Recv, *rank1, *rank2));
        if let Some((timestamp1, timestamp2)) = recv_timestamps {
            communication_data.insert(
                (
                    Direction::Recv,
                    left_child.as_ref().borrow().id(),
                    right_child.as_ref().borrow().id(),
                ),
                (
                    (timestamp1 - logging_start).num_microseconds().unwrap() as f64,
                    (timestamp2 - logging_start).num_microseconds().unwrap() as f64,
                ),
            );
        };

        let new_node = Node::new(
            tensor_count,
            Rc::downgrade(&left_child),
            Rc::downgrade(&right_child),
            Weak::new(),
            None,
        );

        let new_node_ref = Rc::new(RefCell::new(new_node));

        left_child
            .borrow_mut()
            .set_parent(Rc::downgrade(&new_node_ref));

        right_child
            .borrow_mut()
            .set_parent(Rc::downgrade(&new_node_ref));

        remaining_nodes.insert_new(tensor_count, Rc::clone(&new_node_ref));
        let cost = (*timestamp - logging_start).num_microseconds().unwrap() as f64;
        tensor_cost.insert_new(tensor_count, cost);
        tensor_color.insert_new(tensor_count, String::from(COMMUNICATION_COLOR));
        partition_root_nodes[*rank1] = Rc::clone(&new_node_ref);
        tensor_count += 1;
    }

    let root = Rc::downgrade(&remaining_nodes[&(tensor_count - 1)]);
    LogsToTreeResult {
        tree: ContractionTree {
            nodes: remaining_nodes,
            partitions: FxHashMap::from_iter([(0, partition_tensor_ids)]),
            root,
        },
        tensor_cost,
        tensor_color,
        communication_data,
    }
}

pub type CommunicationEvent = (Direction, usize, usize);
pub type TimeRange = (DateTime<chrono::FixedOffset>, DateTime<chrono::FixedOffset>);

#[derive(Debug, Clone)]
struct LogToSubtreeResult {
    /// Dict of remaining nodes to process, keeps track of intermediate tensors
    nodes: FxHashMap<usize, Rc<RefCell<Node>>>,
    /// Keeps track of communication with time stamps
    local_communication_path: Vec<(usize, usize, DateTime<chrono::FixedOffset>)>,
    /// Keeps track of communication time stamps
    communication_timestamps: FxHashMap<CommunicationEvent, TimeRange>,
    /// Start of contraction for reference
    contraction_start: DateTime<chrono::FixedOffset>,
}

/// Processes the log of a single rank. Extracts subtree information corresponding to the single rank and returns it
/// as a LogToSubtreeResult object.
fn log_to_subtree(
    filename: &str,
    tensor_cost: &mut FxHashMap<usize, DateTime<chrono::FixedOffset>>,
    tensor_count: &mut usize,
    rank: usize,
) -> LogToSubtreeResult {
    let file = fs::read_to_string(filename).expect("Should have been able to read the file");
    // Keeps track of nodes representing intermediate tensors when going through logs.
    let mut remaining_nodes = FxHashMap::default();

    // Hashmap that maps local tensor id to node id in overall contraction tree
    let mut replace_to_ssa = FxHashMap::default();

    // Hashmap that stores communication
    let mut communication_timestamps = FxHashMap::default();

    // Stores starting time for contraction
    let mut contraction_start = DateTime::<chrono::FixedOffset>::default();

    // Keeps track of communication between partitions in form (receiver, sender, time contraction ends)
    let mut communication_path = Vec::new();

    // Identify contraction operations, beginning and ending of contractions
    let patterns = [
        // Start of contraction
        r"Start contracting tensor network",
        // Point that a particular contraction finishes
        r"Finished contracting tensors",
        // When all local contractions are done
        r"Completed tensor network contraction",
        // Identifies start of receiving tensor
        r"Start receiving tensor",
        // Identifies end of receiving tensor
        r"Finish receiving tensor",
        // Identifies start of sending tensor
        r"Start sending tensor",
        // Identifies end of sending tensor
        r"Finish sending tensor",
    ];
    // true while local contractions are occurring
    let mut is_local_contraction = true;
    // Compile a set matching any of our patterns.
    let set = RegexSet::new(patterns).unwrap();

    // Counter to track where tensors are being sent from.
    let mut sender = 0;
    let mut receiver = 0;
    let mut send_start = Default::default();
    let mut recv_start = Default::default();

    for line in file.split('\n') {
        if line.is_empty() {
            break;
        }
        let json_value: serde_json::Value = serde_json::from_str(line).unwrap();

        let log = json_value["text"].as_str().unwrap();
        let matches: Vec<_> = set.matches(log).into_iter().collect();

        match matches[..] {
            [0] => {
                // Tracks contraction starting from first local contraction
                // Does no reset timer when communicating.
                if is_local_contraction {
                    contraction_start = parse_timestamp(&json_value);
                }
            }
            [1] => {
                // Tracking contractions due to communication here
                if !is_local_contraction {
                    let contraction_timestamp = parse_timestamp(&json_value);
                    // Don't create any new nodes, simply track communication and contraction time
                    communication_path.push((rank, sender, contraction_timestamp));
                    continue;
                }

                let contraction_timestamp = parse_timestamp(&json_value);
                // Tracking local contractions here.
                let ij: Vec<usize> = ["i", "j"]
                    .iter()
                    .map(|key| {
                        json_value["kv"]
                            .get(key)
                            .unwrap()
                            .to_string()
                            .parse::<usize>()
                            .expect("Unable to parse contracted tensor")
                    })
                    .collect::<Vec<_>>();

                for &tensor_id in &ij {
                    if !replace_to_ssa.contains_key(&tensor_id) {
                        let leaf_node =
                            Node::new(*tensor_count, Weak::new(), Weak::new(), Weak::new(), None);
                        let leaf_node_ref = Rc::new(RefCell::new(leaf_node));

                        replace_to_ssa.insert_new(tensor_id, *tensor_count);
                        tensor_cost.insert_new(*tensor_count, DateTime::default());
                        remaining_nodes.insert_new(*tensor_count, Rc::clone(&leaf_node_ref));
                        *tensor_count += 1;
                    }
                }

                // Store contraction time of resultant tensor
                tensor_cost.insert_new(*tensor_count, contraction_timestamp);

                let intermediate_node_ref =
                    new_intermediate_node(&remaining_nodes, &replace_to_ssa, &ij, *tensor_count);
                // New tensor replaces tensor at position i
                replace_to_ssa.insert(ij[0], *tensor_count);
                // Tensor in j should never be referenced again
                replace_to_ssa.remove(&ij[1]);

                remaining_nodes.insert_new(*tensor_count, intermediate_node_ref);

                *tensor_count += 1;
            }
            [2] => {
                // No longer local contraction after this point
                is_local_contraction = false;
            }
            [3] => {
                // Parse sending rank from log
                sender = json_value["kv"]
                    .get("sender")
                    .unwrap()
                    .to_string()
                    .parse::<usize>()
                    .unwrap();
                recv_start = parse_timestamp(&json_value);
            }
            [4] => {
                let recv_end = parse_timestamp(&json_value);
                communication_timestamps
                    .insert((Direction::Recv, rank, sender), (recv_start, recv_end));
            }
            [5] => {
                // Parse sending rank from log
                receiver = json_value["kv"]
                    .get("receiver")
                    .unwrap()
                    .to_string()
                    .parse::<usize>()
                    .unwrap();
                send_start = parse_timestamp(&json_value);
            }
            [6] => {
                let send_end = parse_timestamp(&json_value);
                communication_timestamps
                    .insert((Direction::Send, receiver, rank), (send_start, send_end));
            }
            _ => {}
        }
    }

    LogToSubtreeResult {
        nodes: remaining_nodes,
        local_communication_path: communication_path,
        communication_timestamps,
        contraction_start,
    }
}

fn new_intermediate_node(
    remaining_nodes: &FxHashMap<usize, Rc<RefCell<Node>>>,
    replace_to_ssa: &FxHashMap<usize, usize>,
    ij: &[usize],
    tensor_count: usize,
) -> Rc<RefCell<Node>> {
    let left_child = Rc::clone(&remaining_nodes[&replace_to_ssa[&ij[0]]]);
    let right_child = Rc::clone(&remaining_nodes[&replace_to_ssa[&ij[1]]]);
    let intermediate_node = Node::new(
        tensor_count,
        Rc::downgrade(&left_child),
        Rc::downgrade(&right_child),
        Weak::new(),
        None,
    );

    let intermediate_node_ref = Rc::new(RefCell::new(intermediate_node));
    left_child
        .borrow_mut()
        .set_parent(Rc::downgrade(&intermediate_node_ref));
    right_child
        .borrow_mut()
        .set_parent(Rc::downgrade(&intermediate_node_ref));
    intermediate_node_ref
}

fn contraction_timing(
    json_value: &serde_json::Value,
    contraction_start: DateTime<chrono::FixedOffset>,
) -> f64 {
    let timestamp = parse_timestamp(json_value);
    (timestamp - contraction_start).num_microseconds().unwrap() as f64
}

/// Parses the timestamp of a log entry given as json.
fn parse_timestamp(json_value: &serde_json::Value) -> DateTime<chrono::FixedOffset> {
    DateTime::parse_from_str(
        json_value["timestamp"].as_str().unwrap(),
        "%Y-%m-%d %H:%M:%S%.6f %z",
    )
    .unwrap()
}
