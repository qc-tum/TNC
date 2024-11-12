use std::{
    fs,
    process::{Command, Stdio},
};

use itertools::Itertools;
use rustc_hash::FxHashMap;

use crate::{
    tensornetwork::tensor::Tensor, types::ContractionIndex, utils::traits::HashMapInsertNew,
};

use super::{
    import::{CommunicationEvent, Direction},
    ContractionTree,
};

pub(super) const COMMUNICATION_COLOR: &str = "black";
pub(super) const COLORS: [&str; 18] = [
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

#[derive(Debug)]
pub struct DendogramSettings {
    pub output_file: String,
    pub objective_function: fn(&Tensor, &Tensor) -> f64,
}

#[derive(Debug)]
pub struct DendogramEntry {
    pub id: usize,
    pub x: f64,
    pub y: f64,
    pub cost: f64,
    pub color: String,
    pub children: Option<(usize, usize)>,
}

pub fn to_dendogram_format(
    contraction_tree: &ContractionTree,
    tensor_network: &Tensor,
    objective_function: fn(&Tensor, &Tensor) -> f64,
) -> Vec<DendogramEntry> {
    let length = 80f64;
    let height = 60f64;

    let x_spacing = length / tensor_network.total_num_tensors() as f64;
    let mut next_leaf_x = x_spacing;

    let mut node_to_position = FxHashMap::default();

    let root_id = contraction_tree.root_id().unwrap();
    let path = contraction_tree.to_flat_contraction_path(root_id, false);
    let mut path_iter = path.iter();

    let partition_nodes = contraction_tree.partitions.get(&1).unwrap();
    let partitions = partition_nodes
        .iter()
        .map(|subtree_root_id| contraction_tree.leaf_ids(*subtree_root_id))
        .collect_vec();

    let mut id_to_partition = FxHashMap::default();
    let mut partition_color = FxHashMap::default();
    let communication_color = String::from(COMMUNICATION_COLOR);
    let mut intermediate_tensors = FxHashMap::default();

    let mut dendogram_entries = Vec::new();
    let mut tree_weights = FxHashMap::default();

    for (i, (partition, color)) in partitions.iter().zip(COLORS.iter().cycle()).enumerate() {
        partition_color.insert_new(i, String::from(*color));
        for &leaf_id in partition {
            id_to_partition.insert_new(leaf_id, i);
            tree_weights.insert_new(leaf_id, 0f64);
            intermediate_tensors.insert_new(leaf_id, {
                tensor_network
                    .nested_tensor(
                        contraction_tree
                            .node(leaf_id)
                            .tensor_index()
                            .as_ref()
                            .unwrap(),
                    )
                    .clone()
            });
        }
    }

    let mut get_coordinates = |node_id,
                               node_map: &mut FxHashMap<usize, (f64, f64)>,
                               dendogram_entries: &mut Vec<DendogramEntry>,
                               id_to_partition: &FxHashMap<usize, usize>|
     -> (f64, f64) {
        if let Some((x, y)) = node_map.get(&node_id) {
            (*x, *y)
        } else {
            assert!(
                contraction_tree.node(node_id).is_leaf(),
                "Contraction relies on Node id {node_id} but it does not yet exist in tree",
            );

            let (x, y) = (next_leaf_x, 0f64);
            node_map.insert_new(node_id, (x, y));
            dendogram_entries.push(DendogramEntry {
                id: node_id,
                x,
                y,
                cost: 0f64,
                color: partition_color[&id_to_partition[&node_id]].clone(),
                children: None,
            });
            next_leaf_x += x_spacing;
            (x, y)
        }
    };

    let mut update = |&node_1_id,
                      &node_2_id,
                      dendogram_entries: &mut Vec<DendogramEntry>,
                      id_to_partition: &mut FxHashMap<usize, usize>| {
        let (x1, _) = get_coordinates(
            node_1_id,
            &mut node_to_position,
            dendogram_entries,
            id_to_partition,
        );
        let (x2, _) = get_coordinates(
            node_2_id,
            &mut node_to_position,
            dendogram_entries,
            id_to_partition,
        );

        let parent_id = contraction_tree.node(node_1_id).parent_id().unwrap();
        let parent_tensor = &intermediate_tensors[&node_1_id] ^ &intermediate_tensors[&node_2_id];
        let mut parent_cost = objective_function(
            &intermediate_tensors[&node_1_id],
            &intermediate_tensors[&node_2_id],
        );
        // Check that child tensors both exist in partitions and they are in the same partitions
        let color = match (
            id_to_partition.get(&node_1_id),
            id_to_partition.get(&node_2_id),
        ) {
            (Some(&partition_1), Some(&partition_2)) if partition_1 == partition_2 => {
                // If both child node are present in one partition, this happens in serial
                parent_cost += tree_weights[&node_1_id];
                parent_cost += tree_weights[&node_2_id];
                // Attribute this intermediate node to particular partition.
                id_to_partition.insert_new(parent_id, partition_1);
                partition_color[&partition_1].clone()
            }
            _ => {
                // Otherwise, this happens in parallel
                let child_cost = tree_weights[&node_1_id].max(tree_weights[&node_2_id]);
                parent_cost += child_cost;
                communication_color.clone()
            }
        };
        node_to_position.insert_new(parent_id, ((x1 + x2) / 2f64, parent_cost));
        dendogram_entries.push(DendogramEntry {
            id: parent_id,
            x: (x1 + x2) / 2f64,
            y: 0f64,
            cost: parent_cost,
            color,
            children: Some((node_1_id, node_2_id)),
        });
        tree_weights.insert_new(parent_id, parent_cost);
        intermediate_tensors.insert_new(parent_id, parent_tensor);
    };

    while let Some(ContractionIndex::Pair(i, j)) = path_iter.next() {
        update(i, j, &mut dendogram_entries, &mut id_to_partition);
    }

    let scaling_factor = height / dendogram_entries.last().unwrap().cost;
    for entry in &mut dendogram_entries {
        entry.y = entry.cost * scaling_factor;
    }

    dendogram_entries
}

pub fn to_pdf(
    pdf_name: &str,
    dendogram_entries: &[DendogramEntry],
    communication_logging: Option<FxHashMap<CommunicationEvent, (f64, f64)>>,
) {
    let communication_logging = communication_logging.unwrap_or_default();
    let mut tikz_picture = String::from(
        r#"% tikzpic.tex
\documentclass[crop,tikz]{standalone}% 'crop' is the default for v1.0, before it was 'preview'
%\usetikzlibrary{...}% tikz package already loaded by 'tikz' option
\begin{document}
\begin{tikzpicture}[scale=.7]
"#,
    );

    let mut id_position = FxHashMap::default();
    for DendogramEntry {
        id,
        x,
        y,
        cost,
        color,
        children,
    } in dendogram_entries
    {
        id_position.insert_new(id, (x, y));

        if let Some((node_1_id, node_2_id)) = children {
            let (x1, _) = id_position[node_1_id];
            let (x2, _) = id_position[node_2_id];
            let recv_timestamps =
                communication_logging.get(&(Direction::Recv, *node_1_id, *node_2_id));
            let send_timestamps =
                communication_logging.get(&(Direction::Send, *node_1_id, *node_2_id));

            if let (Some(&(send_start, send_end)), Some(&(recv_start, recv_end))) =
                (send_timestamps, recv_timestamps)
            {
                tikz_picture.push_str(&format!(
                r#"    \path[draw, color=magenta, line width=0.5mm] ({x1}, {recv_start}) -- ({x1}, {recv_end});
    "#,
                    ));
                tikz_picture.push_str(&format!(
                r#"    \path[draw, color=magenta, line width=0.5mm] ({x2}, {send_start}) -- ({x2}, {send_end});
    "#,
                        ));
                let latest_start = recv_start.max(send_start);
                tikz_picture.push_str(&format!(
                r#"    \path[draw, color=orange, line width=1mm] ({x1}, {latest_start}) -- ({x1}, {recv_end});
    "#,         
    ));
            }

            tikz_picture.push_str(&format!(
                r#"    \node[circle, scale=0.3, fill={color}, label={{[shift={{(-0.4,-0.1)}}]{cost}}}, label=below:{{{id}}}] at ({x}, {y}) ({id}) {{}};
    "#,
            ));
            tikz_picture.push_str(&format!(
                r#"    \path[draw, color={color}] ({node_1_id}.north) -- ({x1}, {y}) -- ({id}.center);
    "#,
            ));
            tikz_picture.push_str(&format!(
                r#"    \path[draw, color={color}] ({node_2_id}.north) -- ({x2}, {y}) -- ({id}.center);
    "#,
            ));
        } else {
            tikz_picture.push_str(&format!(
                r#"    \node[circle, scale=0.3, fill={color}, label=below:{{{id}}}] at ({x}, {y}) ({id}) {{}};
            "#,
            ));
        }
    }
    tikz_picture.push_str(
        r#"\end{tikzpicture}
\end{document}
"#,
    );
    compile_tex(pdf_name, &tikz_picture);
}

pub fn to_dendogram(
    contraction_tree: &ContractionTree,
    tn: &Tensor,
    cost_function: fn(&Tensor, &Tensor) -> f64,
    pdf_name: &str,
) {
    let length = 80f64;
    let x_spacing = length / tn.total_num_tensors() as f64;
    let mut last_leaf_x = x_spacing;
    let height = 60f64;
    let mut node_to_position = FxHashMap::default();
    let root_id = contraction_tree.root_id().unwrap();
    let path = contraction_tree.to_flat_contraction_path(root_id, false);

    let tree_weights = contraction_tree.tree_weights(root_id, tn, cost_function);
    let scaling_factor = tree_weights[&root_id];
    let mut tikz_picture = String::from(
        r#"% tikzpic.tex
\documentclass[crop,tikz]{standalone}% 'crop' is the default for v1.0, before it was 'preview'
%\usetikzlibrary{...}% tikz package already loaded by 'tikz' option
\begin{document}
\begin{tikzpicture}[scale=.7]
"#,
    );

    let mut get_coordinates = |node_id,
                               node_map: &mut FxHashMap<usize, (f64, f64)>,
                               tikz_picture: &mut String|
     -> (f64, f64) {
        if let Some((x, y)) = node_map.get(&node_id) {
            (*x, *y)
        } else {
            assert!(
                contraction_tree.node(node_id).is_leaf(),
                "Contraction relies on Node id {node_id} but it does not yet exist in tree",
            );

            let (x, y) = (last_leaf_x, 0f64);
            node_map.insert_new(node_id, (x, y));
            last_leaf_x += x_spacing;
            tikz_picture.push_str(&format!(
                r#"    \node[label=below:{{{node_id}}}] at ({x}, {y}) ({node_id}) {{}};
            "#,
            ));
            (x, y)
        }
    };

    let mut plot = |&node_1_id, &node_2_id, last| {
        let (x1, _) = get_coordinates(node_1_id, &mut node_to_position, &mut tikz_picture);
        let (x2, _) = get_coordinates(node_2_id, &mut node_to_position, &mut tikz_picture);

        let parent_id = contraction_tree.node(node_1_id).parent_id().unwrap();

        let mut parent_cost = tree_weights[&parent_id];
        if last {
            let child_cost = tree_weights[&node_1_id];
            let child_cost = child_cost.min(tree_weights[&node_2_id]);
            parent_cost -= child_cost;
        }
        let scaled_height = parent_cost / scaling_factor * height;
        node_to_position.insert_new(parent_id, ((x1 + x2) / 2f64, scaled_height));
        tikz_picture.push_str(&format!(
            r#"    \node[label={{[shift={{(-0.4,-0.1)}}]{}}}, label=below:{{{parent_id}}}] at ({}, {scaled_height}) ({parent_id}) {{}};
"#,
            tree_weights[&parent_id],
            (x1 + x2) / 2f64,
        ));
        tikz_picture.push_str(&format!(
            r#"    \path[draw] ({node_1_id}.center) -- ({x1}, {scaled_height}) -- ({parent_id}.center);
"#,
        ));
        tikz_picture.push_str(&format!(
            r#"    \path[draw] ({node_2_id}.center) -- ({x2}, {scaled_height}) -- ({parent_id}.center);
"#,
        ));
    };
    let mut node_iter = path.iter().peekable();

    while let Some(ContractionIndex::Pair(i, j)) = node_iter.next() {
        if node_iter.peek().is_none() {
            plot(i, j, true);
        } else {
            plot(i, j, false);
        }
    }

    tikz_picture.push_str(
        r#"\end{tikzpicture}
\end{document}
"#,
    );
    compile_tex(pdf_name, &tikz_picture);
}

/// Compiles the `tex_code` using `pdflatex` and saves the output as `pdf_name`.
fn compile_tex(pdf_name: &str, tex_code: &str) {
    fs::write("final.tex", tex_code).unwrap();

    let compilation_status = Command::new("lualatex")
        .arg(format!("-jobname={pdf_name}"))
        .arg("final.tex")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .unwrap();

    assert!(compilation_status.success());
}
