use std::{
    collections::HashMap,
    fs,
    io::Write,
    process::{Command, Stdio},
};

use crate::{tensornetwork::tensor::Tensor, types::ContractionIndex};

use super::ContractionTree;

pub fn to_dendogram(
    contraction_tree: &ContractionTree,
    tn: &Tensor,
    cost_function: fn(&Tensor, &Tensor) -> f64,
    svg_name: String,
) {
    let length = 80f64;
    let x_spacing = length / tn.total_num_tensors() as f64;
    let mut last_leaf_x = x_spacing;
    let height = 60f64;
    let mut node_to_position: HashMap<usize, (f64, f64)> = HashMap::new();
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
                               node_map: &mut HashMap<usize, (f64, f64)>,
                               tikz_picture: &mut String|
     -> (f64, f64) {
        if let Some((x, y)) = node_map.get(&node_id) {
            (*x, *y)
        } else {
            if !contraction_tree.node(node_id).is_leaf() {
                panic!(
                    "Contraction relies on Node id {node_id:?} but it does not yet exist in tree",
                );
            }
            let (x, y) = (last_leaf_x, 0f64);
            node_map.entry(node_id).or_insert((x, y));
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
        node_to_position
            .entry(parent_id)
            .or_insert(((x1 + x2) / 2f64, scaled_height));
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
    let mut pdf_output = Command::new("pdflatex")
        .arg("-quiet")
        .arg(format!("-jobname={svg_name}"))
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    fs::write("final.txt", tikz_picture.clone()).expect("Unable to write out .gv file");
    let mut pdf_run = pdf_output.stdin.take().expect("Failed to open stdin");
    std::thread::spawn(move || {
        pdf_run
            .write_all(tikz_picture.as_bytes())
            .expect("Failed to write to stdin");
    });
    // fs::write(svg_name.clone(), tikz_picture).expect("Unable to write");
    // Command::new("pdflatex").arg(svg_name).spawn().unwrap();
}
