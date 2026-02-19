use ndarray::{Array1, Array2};

use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, Exp};

use crate::{
    qmatrix::QMatrix,
    tree::{Node, Phylogeny},
};

pub trait Simulations {
    const BRANCHING: usize = 2;
    fn yule_migrations(
        lambda: f64,
        g: usize,
        n: usize,
        m_prob: f64,
        seed: u64,
        temperature: f64,
    ) -> (Self, Array2<i32>)
    where
        Self: Sized;
}

impl Simulations for Phylogeny<usize, usize> {
    fn yule_migrations(
        lambda: f64,
        g: usize,
        n: usize,
        m_rate: f64,
        seed: u64,
        temperature: f64,
    ) -> (Self, Array2<i32>) {
        let exp_dist = Exp::new(lambda).unwrap();
        let mut rng = StdRng::seed_from_u64(seed);

        let mut migration_matrix: Array2<i32> = Array2::zeros((n, n));

        let mut rmatrix = QMatrix::new_with_initial_conditions(n, m_rate, temperature);

        let root = Node::root(0usize, 0);
        let mut tree: Phylogeny<usize, usize> = Phylogeny::new(root, exp_dist.sample(&mut rng));

        let mut idx = 1usize;
        let mut leaves: Vec<(usize, usize)> = vec![(0, 0)];
        let mut counts = vec![0.0; n];
        counts[0] = 1.0;

        for _ in 0..g {
            rmatrix = rmatrix.rescale_from_frequencies(Array1::from_vec(counts));

            let mut new_counts = vec![0.; n];
            let mut new_leaves = vec![];
            for &(leaf, label) in &leaves {
                for _ in 0..Self::BRANCHING {
                    let branch_length = exp_dist.sample(&mut rng);

                    let pmatrix = rmatrix.to_pmatrix(-branch_length);

                    let next_label = pmatrix.sample(label, &mut rng);
                    new_counts[next_label] += 1.0;

                    tree.add_child(leaf, idx, next_label, branch_length);
                    new_leaves.push((idx, next_label));

                    migration_matrix[[label, next_label]] += 1;
                    idx += 1;
                }
            }
            counts = new_counts;
            leaves = new_leaves;
        }

        (tree, migration_matrix)
    }
}

#[test]
fn test_yule_migrations() {
    use crate::visualizations::{graph_from_edge_matrix, save_graph_png};

    let sites = 6;
    let (_tree, migration_matrix) = Phylogeny::yule_migrations(0.2, 10, sites, 0.015, 42, 1.0);
    for i in 0..sites {
        for j in 0..sites {
            println!("{i} -> {j}: {}", migration_matrix[[i, j]])
        }
    }

    let g = graph_from_edge_matrix(migration_matrix);
    match save_graph_png(&g, "test_mig_graph.png") {
        Ok(_) => println!("Save to test_mig_graph.png"),
        Err(e) => println!("{e}"),
    }
}
