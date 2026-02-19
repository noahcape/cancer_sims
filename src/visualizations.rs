use std::{fs::File, io::Write, process::Command};

use ndarray::Array2;
use petgraph::{dot::Dot, graph::Graph};

use plotters::prelude::*;
use std::error::Error;

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
        .args([
            "-Tpng",
            &dot_path,
            "-o",
            &format!("{out}_migration_graph.png"),
        ])
        .status()?;

    if !status.success() {
        panic!("Graphviz failed");
    }

    Ok(())
}

pub fn save_heatmap(matrix: &Array2<f64>, filename: &str) -> Result<(), Box<dyn Error>> {
    let (rows, cols) = matrix.dim();

    let root = BitMapBackend::new(filename, (600, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .margin(0)
        .build_cartesian_2d(0..cols, 0..rows)?;

    for ((row, col), value) in matrix.indexed_iter() {
        println!("{:?}", value);
        let color = RGBColor((120. * value) as u8, 120, 120);

        chart.draw_series(std::iter::once(Rectangle::new(
            [(row, col), (row + 1, col + 1)],
            color.filled(),
        )))?;
    }

    root.present()?;
    Ok(())
}
