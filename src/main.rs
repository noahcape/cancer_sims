use std::fs::File;

use cancer_migration_sims::{
    simulations::Simulations,
    tree::Phylogeny,
    visualizations::{graph_from_edge_matrix, save_graph_png},
};

use clap::Parser;

/// Simple program to greet a person
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Birth rate for Yule model
    #[arg(short, long, default_value_t = 0.2)]
    birth_rate: f64,

    /// Migration probability between sites
    #[arg(short, long, default_value_t = 0.01)]
    migration_probability: f64,

    /// Generations to simulate
    #[arg(short, long, default_value_t = 10)]
    generations: usize,

    /// Number of sites to simulate migration between
    #[arg(short = 's', long, default_value_t = 6)]
    sites: usize,

    /// Seed for reproducible simulation
    #[arg(short = 'r', long, default_value_t = 42)]
    seed: u64,

    /// File name (no file type)
    #[arg(short, long, default_value = "out")]
    out: String,
}

fn main() {
    let Args {
        birth_rate,
        migration_probability,
        generations,
        sites,
        seed,
        out,
    } = Args::parse();

    let (tree, migration_matrix) =
        Phylogeny::yule_migrations(birth_rate, generations, sites, migration_probability, seed);

    match tree.write_csv(File::create(format!("{out}_edgelist.csv")).unwrap()) {
        Ok(_) => println!("Wrote edgelist to {out}_edgelist.csv"),
        Err(e) => println!("{e}: while writing edgelist"),
    }

    match tree
        .write_csv_vertex_labeling(File::create(format!("{out}_vertex_labeling.csv")).unwrap())
    {
        Ok(_) => println!("Wrote vertex labeling to {out}_vertex_labeling.csv"),
        Err(e) => println!("{e}: while writing vertex labeling"),
    }

    match tree.write_csv_leaf_labeling(File::create(format!("{out}_leaf_labeling.csv")).unwrap()) {
        Ok(_) => println!("Wrote leaf labeling to {out}_leaf_labeling.csv"),
        Err(e) => println!("{e}: while writing leaf labeling"),
    }

    let g = graph_from_edge_matrix(migration_matrix);
    match save_graph_png(&g, &out) {
        Ok(_) => println!("Save to {out}_migration_graph.png"),
        Err(e) => println!("{e}"),
    }
}
