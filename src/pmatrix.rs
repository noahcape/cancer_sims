use std::{
    fmt::Display,
    ops::{Div, Mul},
};

use ndarray::{Array1, Array2, Axis};
use rand::RngCore;
use rand_distr::{Distribution, WeightedIndex};

#[derive(Debug)]
pub struct PMatrix {
    p: Array2<f64>,
}

impl Display for PMatrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.p)
    }
}

impl PMatrix {
    pub fn new(n: usize) -> Self {
        let mut pmatrix = Array2::zeros((n, n));

        for i in 0..n {
            for j in 0..n {
                if i == j {
                    pmatrix[[i, j]] = 0.5;
                } else {
                    pmatrix[[i, j]] = 0.5.div(n as f64 - 1.0)
                }
            }
        }

        PMatrix { p: pmatrix }
    }

    pub fn new_with_initial_conditions(n: usize, migration_probability: f64) -> Self {
        let mut pmatrix = Array2::zeros((n, n));

        let no_mig = 1.0 - migration_probability;

        for i in 0..n {
            for j in 0..n {
                if i == j {
                    pmatrix[[i, j]] = no_mig;
                } else {
                    pmatrix[[i, j]] = migration_probability.div(n as f64 - 1.0)
                }
            }
        }

        PMatrix { p: pmatrix }
    }

    pub fn from_vector(v: Vec<f64>, n: usize) -> Self {
        let p = Array2::from_shape_vec((n, n), v).unwrap();

        Self { p }
    }

    pub fn exponentiate(self, migration_rate: f64, branch_length: f64) -> Self {
        let pmatrix = self.p.mul(migration_rate * branch_length);

        PMatrix { p: pmatrix.exp() }
    }

    // Rescale matrix via Sinkhorn-knupp algorithm - iterative proportion fitting
    fn rescale(&self, iters: usize) -> Self {
        let n = self.p.nrows();
        let mut r = Array1::ones(n);
        let mut c = Array1::ones(n);

        let p = &self.p;
        for _ in 0..iters {
            let row_sums = p.dot(&c);
            r.assign(&row_sums.mapv(|e| 1.0 / e));

            let col_sums = p.t().dot(&r);
            c.assign(&col_sums.mapv(|e| 1.0 / e));
        }

        let q = p * &r.insert_axis(Axis(1)) * &c.insert_axis(Axis(0));

        Self { p: q }
    }

    fn diag_mul(&self, v: Array1<f64>) -> Self {
        Self {
            p: &self.p * &v.insert_axis(Axis(0)),
        }
    }

    // A (close to) doubly stochastic matrix p
    pub fn sample<R: RngCore>(&self, i: usize, rng: &mut R) -> usize {
        let probabilities = self.p.row(i);
        let dist = WeightedIndex::new(&probabilities).unwrap();

        dist.sample(rng)
    }

    pub fn rescale_from_frequencies(self, frequencies: Array1<f64>) -> Self {
        let updated_freqs = frequencies
            .iter()
            //  can either use e^(-f + epsilon) or 1 / (f + eps)
            .map(|f| (-f + f64::EPSILON).exp())
            // .map(|e| 1.0.div(e + f64::EPSILON))
            .collect::<Vec<_>>();
        let sum: f64 = updated_freqs.iter().sum();
        let weights = updated_freqs.iter().map(|e| e / sum).collect();

        self.diag_mul(weights).rescale(3)
    }
}

#[test]
fn test_rescaling() {
    use rand::thread_rng;    

    let p = vec![0.8, 0.1, 0.1, 0.3, 0.4, 0.3, 0.2, 0.1, 0.7];
    let mut rng = thread_rng();

    let pmatrix = PMatrix::from_vector(p, 3);
    let mut count = vec![0; 3];
    for _ in 0..10000 {
        let i = pmatrix.sample(1, &mut rng);
        count[i] += 1;
    }
    println!("{:?}", count);

    let freqs = vec![1.0, 0.0, 0.0];
    let updated_freqs = freqs
        .iter()
        .map(|e| 1.0.div(e + f64::EPSILON as f64))
        .collect::<Vec<_>>();
    let sum: f64 = updated_freqs.iter().sum();
    let weights = updated_freqs.iter().map(|e| e / sum).collect();

    let biased_p = pmatrix.diag_mul(weights);
    let pmatrix = biased_p.rescale(10);

    println!("After rescaling");
    let mut count = vec![0; 3];
    for _ in 0..10000 {
        let i = pmatrix.sample(1, &mut rng);
        count[i] += 1;
    }
    println!("{:?}", count);
}

#[test]
fn test_diag_mul() {
    let m = PMatrix::from_vector(vec![2.0, 3.0, 4.0, 5.0], 2);
    let d = Array1::from_vec(vec![2.0, 1.0]);
    let diag_m = m.diag_mul(d);
    assert_eq!(
        diag_m.p,
        Array2::from_shape_vec((2, 2), vec![4.0, 3.0, 8.0, 5.0]).unwrap()
    );
}
