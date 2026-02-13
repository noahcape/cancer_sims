use std::{fs::File, io::Write, process::Command};

use ndarray::Array2;
use petgraph::{dot::Dot, graph::Graph};

pub fn graph_from_edge_matrix(m: Array2<i32>) -> Graph<usize, i32> {
    // m is a square matrix
    let n = m.nrows();
    let mut g = Graph::<usize, i32>::new();

    // Add nodes
    let nodes: Vec<_> = (0..n).map(|i| g.add_node(i)).collect();

    // Add edges
    for i in 0..n {
        for j in 0..n {
            if i != j && m[[i, j]] > 0 {
                g.add_edge(nodes[i], nodes[j], m[[i, j]]);
            }
        }
    }

    g
}

pub fn save_graph_png(g: &Graph<usize, i32>, out: &str) -> std::io::Result<()> {
    // 1. Write DOT to a temporary file
    let dot = format!("{:?}", Dot::new(g));
    let dot_path = format!("{out}_mig_graph.dot");

    let mut file = File::create(&dot_path)?;
    file.write_all(dot.as_bytes())?;

    // 2. Call Graphviz
    let status = Command::new("dot")
        .args(["-Tpng", &dot_path, "-o", &format!("{out}_migration_graph.png")])
        .status()?;

    if !status.success() {
        panic!("Graphviz failed");
    }

    Ok(())
}