use std::{
    fmt::Display,
    ops::{Div, Mul, Sub},
};

use ndarray::{Array1, Array2, Axis};
use rand::RngCore;
use rand_distr::{Distribution, WeightedIndex};

#[derive(Debug, Clone)]
pub struct QMatrix {
    m: Array2<f64>,
    temperature: f64,
}

impl Display for QMatrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.m)
    }
}

impl QMatrix {
    pub fn get_matrix(&self) -> Array2<f64> {
        self.m.clone()
    }

    pub fn set_temperate(&mut self, temperature: f64) {
        self.temperature = temperature
    }

    pub fn new(n: usize) -> Self {
        let mut qmatrix = Array2::zeros((n, n));

        for i in 0..n {
            for j in 0..n {
                if i == j {
                    qmatrix[[i, j]] = 0.5;
                } else {
                    qmatrix[[i, j]] = 0.5.div(n as f64 - 1.0)
                }
            }
        }

        QMatrix {
            m: qmatrix,
            temperature: 1.0,
        }
    }

    pub fn new_with_initial_conditions(n: usize, migration_rate: f64, temperature: f64) -> Self {
        let mut qmatrix = Array2::zeros((n, n));

        let no_mig = 1.0 - ((n as f64 - 1.0) * migration_rate);

        for i in 0..n {
            for j in 0..n {
                if i == j {
                    qmatrix[[i, j]] = no_mig;
                } else {
                    qmatrix[[i, j]] = migration_rate
                }
            }
        }

        QMatrix {
            m: qmatrix,
            temperature,
        }
    }

    pub fn from_vector(v: Vec<f64>, n: usize) -> Self {
        let m = Array2::from_shape_vec((n, n), v).unwrap();

        Self {
            m,
            temperature: 1.0,
        }
    }

    pub fn to_pmatrix(&self, branch_length: f64) -> Self {
        let qmatrix = self.m.clone().mul(-branch_length);

        QMatrix {
            m: qmatrix.exp().sub(1.0).abs(),
            temperature: self.temperature,
        }
    }

    // Rescale matrix via Sinkhorn-knupp algorithm - iterative proportion fitting
    fn rescale(&self, iters: usize) -> Self {
        let n = self.m.nrows();
        let mut r = Array1::ones(n);
        let mut c = Array1::ones(n);

        let m = &self.m;
        for _ in 0..iters {
            let row_sums = m.dot(&c);
            r.assign(&row_sums.mapv(|e| 1.0 / e));

            let col_sums = m.t().dot(&r);
            c.assign(&col_sums.mapv(|e| 1.0 / e));
        }

        let q = m * &r.insert_axis(Axis(1)) * &c.insert_axis(Axis(0));

        Self {
            m: q,
            temperature: self.temperature,
        }
    }

    fn diag_mul(&self, v: Array1<f64>) -> Self {
        Self {
            m: &self.m * &v.insert_axis(Axis(0)),
            temperature: self.temperature,
        }
    }

    // A (close to) doubly stochastic matrix p
    pub fn sample<R: RngCore>(&self, i: usize, rng: &mut R) -> usize {
        let probabilities = self.m.row(i);
        let dist = WeightedIndex::new(&probabilities).unwrap();

        dist.sample(rng)
    }

    pub fn rescale_from_frequencies(self, counts: Array1<f64>) -> Self {
        let weights = counts
            .iter()
            .map(|e| (1.0.div(e + f64::EPSILON)).powf(self.temperature))
            .collect();

        self.diag_mul(weights).rescale(3)
    }
}

#[test]
fn test_rate_rescaling() {
    use rand::thread_rng;
    let mut rng = thread_rng();
    let r = vec![0.96, 0.02, 0.02, 0.02, 0.96, 0.02, 0.02, 0.02, 0.96];
    let mut rmatrix = QMatrix::from_vector(r, 3);
    rmatrix.set_temperate(0.5);
    let bl = rand_distr::Exp::new(0.25).unwrap().sample(&mut rng);
    let pmatrix = rmatrix.clone().to_pmatrix(bl);

    // migrations from site 0
    let mut count = vec![0; 3];

    for _ in 0..1000 {
        let i = pmatrix.sample(0, &mut rng);
        count[i] += 1;
    }
    println!("Migration from site 0 to: {:?}", count);

    // let weights: Vec<_> = count.iter().map(|&e| e as f64 / 1000.0).collect();
    let freqs = count
        .iter()
        .map(|&e| 1.0.div(e as f64 + f64::EPSILON as f64))
        .collect();

    let biased_r = rmatrix.diag_mul(freqs);
    let rmatrix = biased_r.rescale(5);
    let bl = rand_distr::Exp::new(0.25).unwrap().sample(&mut rng);
    let pmatrix = rmatrix.to_pmatrix(bl);

    println!("After rescaling");
    let mut count = vec![0; 3];
    for _ in 0..1000 {
        let i = pmatrix.sample(1, &mut rng);
        count[i] += 1;
    }
    println!("Migrations from site 1 to: {:?}", count);

    let mut count = vec![0; 3];
    for _ in 0..1000 {
        let i = pmatrix.sample(2, &mut rng);
        count[i] += 1;
    }
    println!("Migrations from site 2 to: {:?}", count);

    let mut count = vec![0; 3];
    for _ in 0..1000 {
        let i = pmatrix.sample(0, &mut rng);
        count[i] += 1;
    }
    println!("Migrations from site 0 to: {:?}", count);
}

#[test]
fn test_rescaling() {
    use rand::thread_rng;

    let p = vec![0.8, 0.1, 0.1, 0.3, 0.4, 0.3, 0.2, 0.1, 0.7];
    let mut rng = thread_rng();

    let pmatrix = QMatrix::from_vector(p, 3);
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
    let m = QMatrix::from_vector(vec![2.0, 3.0, 4.0, 5.0], 2);
    let d = Array1::from_vec(vec![2.0, 1.0]);
    let diag_m = m.diag_mul(d);
    assert_eq!(
        diag_m.m,
        Array2::from_shape_vec((2, 2), vec![4.0, 3.0, 8.0, 5.0]).unwrap()
    );
}
