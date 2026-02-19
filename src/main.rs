use std::{
    fs::{self, File},
    process::exit,
};

use cancer_migration_sims::{
    simulations::Simulations,
    tree::Phylogeny,
    visualizations::{graph_from_edge_matrix, save_graph_png},
};

use clap::Parser;
use serde::Serialize;

/// Simple program to greet a person
#[derive(Parser, Debug, Serialize)]
#[command(version, about, long_about = None)]
struct Args {
    /// Birth rate for Yule model
    #[arg(short, long, default_value_t = 0.2)]
    birth_rate: f64,

    /// Migration rate between sites
    #[arg(short, long, default_value_t = 0.02)]
    migration_rate: f64,

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

    /// Temperature to soften bias (0 = no bias, 1 = strong bias)
    #[arg(short, long, default_value_t = 1.)]
    temperature: f64,

    /// Number of SPR perturbations to add to the tree
    #[arg(short, long, default_value = None)]
    perturbations: Option<usize>,
}

fn io(tree: &Phylogeny<usize, usize>, base_fname: &str) -> Result<(), String> {
    match tree.write_csv(File::create(format!("{base_fname}_edgelist.csv")).unwrap()) {
        Ok(_) => println!("Wrote edgelist to {base_fname}_edgelist.csv"),
        Err(e) => return Err(format!("{e}: while writing edgelist")),
    }

    match tree.write_csv_vertex_labeling(
        File::create(format!("{base_fname}_vertex_labeling.csv")).unwrap(),
    ) {
        Ok(_) => println!("Wrote vertex labeling to {base_fname}_vertex_labeling.csv"),
        Err(e) => return Err(format!("{e}: while writing vertex labeling")),
    }

    match tree
        .write_csv_leaf_labeling(File::create(format!("{base_fname}_leaf_labeling.csv")).unwrap())
    {
        Ok(_) => println!("Wrote leaf labeling to {base_fname}_leaf_labeling.csv"),
        Err(e) => return Err(format!("{e}: while writing leaf labeling")),
    }

    Ok(())
}

fn main() {
    let args = Args::parse();

    // write simulation metadata
    match fs::write(
        format!("{}_simulation_metadata.json", args.out),
        serde_json::to_string_pretty(&args).unwrap(),
    ) {
        Ok(_) => println!("Saved metadata at {}_simulation_metadata.json", args.out),
        Err(e) => {
            println!("{e}: while writing simulation metadata");
            exit(1);
        }
    }

    let Args {
        birth_rate,
        migration_rate,
        generations,
        sites,
        seed,
        out,
        perturbations,
        temperature,
    } = args;

    let (tree, migration_matrix) = Phylogeny::yule_migrations(
        birth_rate,
        generations,
        sites,
        migration_rate,
        seed,
        temperature,
    );

    match io(&tree, &out) {
        Ok(_) => (),
        Err(e) => {
            println!("{e}");
            exit(1);
        }
    }

    let g = graph_from_edge_matrix(migration_matrix);
    match save_graph_png(&g, &out) {
        Ok(_) => println!("Save to {out}_migration_graph.png"),
        Err(e) => println!("{e}"),
    }

    if let Some(n) = perturbations {
        let perturbed_tree = tree.perturb(n);
        match io(&perturbed_tree, &format!("{out}_perturbed")) {
            Ok(_) => (),
            Err(e) => {
                println!("{e}");
                exit(1)
            }
        }
    }
}
