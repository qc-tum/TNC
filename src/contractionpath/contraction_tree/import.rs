use chrono::DateTime;
use itertools::Itertools;
use regex::RegexSet;
use std::{
    cell::RefCell,
    collections::HashMap,
    fs::{self},
    rc::{Rc, Weak},
};

use crate::{
    contractionpath::contraction_tree::{export::to_pdf, Node},
    types::ContractionIndex,
};

use super::{export::DendogramEntry, ContractionTree};

const COLORS: [&str; 19] = [
    "black",
    "blue",
    "brown",
    "cyan",
    "green",
    "lightgray",
    "lime",
    "magenta",
    "olive",
    "orange",
    "pink",
    "purple",
    "red",
    "teal",
    "violet",
    "white",
    "yellow",
    "darkgray",
    "gray",
];

pub fn logs_to_pdf(filename: String, suffix: String, ranks: usize) {
    let height = 120f64;
    let width = 80f64;

    let (contraction_tree, tensor_position, tensor_color) = logs_to_tree(filename, suffix, ranks);

    let flat_contraction_path =
        contraction_tree.to_flat_contraction_path(contraction_tree.root_id().unwrap(), false);
    let leaf_nodes = contraction_tree.leaf_ids(contraction_tree.root_id().unwrap());
    let num_leaf_nodes = leaf_nodes.len();
    let mut dendogram_entries = Vec::new();

    let mut next_x = 0f64;
    let mut tensor_x_position = HashMap::new();

    for contraction_index in flat_contraction_path.iter() {
        if let ContractionIndex::Pair(i, j) = contraction_index {
            if contraction_tree.node(*i).is_leaf() {
                tensor_x_position
                    .try_insert(*i, next_x)
                    .expect("Tensor {i} already there.");
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
                tensor_x_position
                    .try_insert(*j, next_x)
                    .expect("Tensor {j} already there.");
                dendogram_entries.push(DendogramEntry {
                    id: *j,
                    x: tensor_x_position[j],
                    y: 0f64,
                    cost: 0f64,
                    color: tensor_color[i].clone(),
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
                tensor_x_position
                    .try_insert(parent_id, new_x)
                    .expect("Tensor {parent_id} already there.");
            } else {
                println!(
                    "Final contraction: {:?}",
                    contraction_tree.node(*i).parent_id()
                );
                println!(
                    "Final contraction: {:?}",
                    contraction_tree.node(*j).parent_id()
                );
            }
        }
    }

    let width_scaling = width / num_leaf_nodes as f64;
    let height_scaling = height / dendogram_entries.last().unwrap().cost;

    for dendogram_entry in dendogram_entries.iter_mut() {
        dendogram_entry.x *= width_scaling;
        dendogram_entry.y *= height_scaling;
    }
    println!("Dendogram entries: {:?}", dendogram_entries);
    to_pdf("logs_to_pdf", &dendogram_entries);
}

pub fn logs_to_tree(
    filename: String,
    suffix: String,
    ranks: usize,
) -> (ContractionTree, HashMap<usize, f64>, HashMap<usize, String>) {
    // Maps node id in contraction tree to a position in pdf
    let mut tensor_cost = HashMap::new();
    // Counter to keep track of total number of intermediate + leaf nodes between ranks.
    let mut tensor_count = 0;

    // Tracks the root node of each rank's partition, ordered by rank by construction
    let mut partition_root_nodes = Vec::new();

    // Hashmap of all nodes
    let mut remaining_nodes = HashMap::new();
    // Tracks all communication taking place, only lists receiving rank for each instance to prevent doubles.
    let mut communication_path = Vec::new();

    // Tracks the color of each node
    let mut tensor_color: HashMap<usize, String> = HashMap::new();

    for rank in 0..ranks {
        let LogToSubtreeResult(nodes, mut local_communication_path) = log_to_subtree(
            filename.clone() + &format!("_rank{}.", rank) + &suffix,
            &mut tensor_cost,
            &mut tensor_count,
            rank,
        );
        let color = COLORS[rank + 1];
        nodes.keys().for_each(|&key| {
            tensor_color.insert(key, String::from(color));
        });

        remaining_nodes.extend(nodes);
        communication_path.append(&mut local_communication_path);
        partition_root_nodes.push(Rc::clone(&remaining_nodes[&(tensor_count - 1)]));
    }

    let partition_tensor_ids = partition_root_nodes
        .iter()
        .map(|node| node.borrow().id)
        .collect::<Vec<usize>>();

    // Sort communication by time to ensure no violation of data dependencies
    let communication_path = communication_path
        .iter_mut()
        .sorted_by(|pair1, pair2| pair1.2.partial_cmp(&pair2.2).unwrap())
        .collect::<Vec<_>>();

    for (rank1, rank2, cost) in communication_path.iter() {
        let left_child = Rc::clone(&partition_root_nodes[*rank1]);
        let right_child = Rc::clone(&partition_root_nodes[*rank2]);
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
            .add_parent(Rc::downgrade(&new_node_ref));

        right_child
            .borrow_mut()
            .add_parent(Rc::downgrade(&new_node_ref));

        remaining_nodes
            .try_insert(tensor_count, Rc::clone(&new_node_ref))
            .expect("SSA {tensor_count} already exists");

        tensor_cost
            .try_insert(tensor_count, *cost)
            .expect("SSA {tensor_count} already exists");
        tensor_color
            .try_insert(tensor_count, String::from(COLORS[0]))
            .expect("Tensor count already in dict");
        partition_root_nodes[*rank1] = Rc::clone(&new_node_ref);
        tensor_count += 1;
    }

    let root = Rc::downgrade(&remaining_nodes[&(tensor_count - 1)]);
    (
        ContractionTree {
            nodes: remaining_nodes,
            partitions: HashMap::from([(0, partition_tensor_ids)]),
            root,
        },
        tensor_cost,
        tensor_color,
    )
}

struct LogToSubtreeResult(HashMap<usize, Rc<RefCell<Node>>>, Vec<(usize, usize, f64)>);

fn log_to_subtree(
    filename: String,
    tensor_cost: &mut HashMap<usize, f64>,
    tensor_count: &mut usize,
    rank: usize,
) -> LogToSubtreeResult {
    let file = fs::read_to_string(filename).expect("Should have been able to read the file");
    // Keeps track of nodes representing intermediate tensors when going through logs.
    let mut remaining_nodes = HashMap::new();

    // Hashmap that maps local tensor id to node id in overall contraction tree
    let mut replace_to_ssa = HashMap::new();

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
        // Identifies communication between partitions
        r"Receiving tensor",
        r".*",
    ];
    // true while local contractions are occurring
    let mut is_local_contraction = true;
    // Compile a set matching any of our patterns.
    let set = RegexSet::new(patterns).unwrap();

    // Counter to track where tensors are being sent from.
    let mut sender = 0;

    for line in file.split("\n") {
        if line.is_empty() {
            break;
        }
        let json_value: serde_json::Value = serde_json::from_str(line).unwrap();

        let log = json_value["text"].as_str().unwrap();
        let matches: Vec<_> = set.matches(log).into_iter().collect();

        match matches[0] {
            0 => {
                // Tracks contraction starting from first local contraction
                // Does no reset timer when communicating.
                if is_local_contraction {
                    contraction_start = DateTime::parse_from_str(
                        json_value["timestamp"].as_str().unwrap(),
                        "%Y-%m-%d %H:%M:%S%.6f %z",
                    )
                    .unwrap();
                    println!("Contraction start: {:?}", contraction_start)
                }
            }
            1 => {
                let contraction_time = contraction_timing(&json_value, contraction_start);
                // Tracking contractions due to communication here
                if !is_local_contraction {
                    // Don't create any new nodes, simply track communication and contraction time
                    communication_path.push((rank, sender, contraction_time));
                    continue;
                }
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

                for tensor_id in ij.iter().cloned() {
                    if !replace_to_ssa.contains_key(&tensor_id) {
                        let leaf_node =
                            Node::new(*tensor_count, Weak::new(), Weak::new(), Weak::new(), None);
                        let leaf_node_ref = Rc::new(RefCell::new(leaf_node));

                        replace_to_ssa
                            .try_insert(tensor_id, *tensor_count)
                            .expect("SSA {i} already exists");
                        tensor_cost
                            .try_insert(*tensor_count, 0f64)
                            .expect("SSA {i} already exists");
                        remaining_nodes
                            .try_insert(*tensor_count, Rc::clone(&leaf_node_ref))
                            .expect("SSA {i} already exists");
                        *tensor_count += 1;
                    }
                }

                // Store contraction time of resultant tensor
                tensor_cost
                    .try_insert(*tensor_count, contraction_time)
                    .unwrap_or_else(|_| panic!("Tensor {} already inserted", *tensor_count));

                let intermediate_node_ref =
                    new_intermediate_node(&remaining_nodes, &replace_to_ssa, &ij, tensor_count);
                // New tensor replaces tensor at position i
                replace_to_ssa.insert(ij[0], *tensor_count);
                // Tensor in j should never be referenced again
                replace_to_ssa.remove(&ij[1]);

                remaining_nodes
                    .try_insert(*tensor_count, Rc::clone(&intermediate_node_ref))
                    .expect("SSA {tensor_count} already exists");

                *tensor_count += 1;
            }
            2 => {
                // No longer local contraction after this point
                is_local_contraction = false;
            }
            3 => {
                // Parse sending rank from log
                sender = json_value["kv"]
                    .get("sender")
                    .unwrap()
                    .to_string()
                    .parse::<usize>()
                    .unwrap();
            }
            _ => {}
        }
    }

    LogToSubtreeResult(remaining_nodes, communication_path)
}

fn new_intermediate_node(
    remaining_nodes: &HashMap<usize, Rc<RefCell<Node>>>,
    replace_to_ssa: &HashMap<usize, usize>,
    ij: &[usize],
    tensor_count: &usize,
) -> Rc<RefCell<Node>> {
    let left_child = Rc::clone(&remaining_nodes[&replace_to_ssa[&ij[0]]]);
    let right_child = Rc::clone(&remaining_nodes[&replace_to_ssa[&ij[1]]]);
    let intermediate_node = Node::new(
        *tensor_count,
        Rc::downgrade(&left_child),
        Rc::downgrade(&right_child),
        Weak::new(),
        None,
    );

    let intermediate_node_ref = Rc::new(RefCell::new(intermediate_node));
    left_child
        .borrow_mut()
        .add_parent(Rc::downgrade(&intermediate_node_ref));
    right_child
        .borrow_mut()
        .add_parent(Rc::downgrade(&intermediate_node_ref));
    intermediate_node_ref
}

fn contraction_timing(
    json_value: &serde_json::Value,
    contraction_start: DateTime<chrono::FixedOffset>,
) -> f64 {
    let timestamp = DateTime::parse_from_str(
        json_value["timestamp"].as_str().unwrap(),
        "%Y-%m-%d %H:%M:%S%.6f %z",
    )
    .expect("Invalid contraction time");

    (timestamp - contraction_start).num_nanoseconds().unwrap() as f64
}
