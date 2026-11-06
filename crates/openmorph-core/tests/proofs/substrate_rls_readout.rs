// SPDX-License-Identifier: MIT-0
//! substrate_rls_readout — the opt-in RECURSIVE-LEAST-SQUARES readout (`Config::readout_rls`), the online form of
//! ridge regression, on a frozen random feature layer. Claims:
//!   1. EXACT: a linear readout with `λ = 1` holds the closed-form ridge solution `(XᵀX + I/P₀)⁻¹Xᵀy` after the
//!      stream (it is ridge, computed one event at a time).
//!   2. USEFUL: on a smooth 2-D target over a 64-feature `grow_backbone`, RLS reaches a far lower held-out error
//!      than the default delta rule on the same basis and stream, and leaves the basis exactly as it was born.
//!   3. SAFE: the covariance follows `grow`'s births and the economy's culls, and survives a snapshot (the reloaded
//!      organism continues bit for bit).
//!
//!   cargo test -p openmorph-core --test proofs substrate_rls_readout -- --nocapture

use openmorph_core::{Activation, Config, Event, Field};

use crate::common::Rng;

fn rls(lambda: f32) -> Config {
    Config {
        readout_rls: lambda,
        ..Config::default()
    }
}

/// Solve `A·x = b` (A symmetric positive definite, n small) by Gaussian elimination in f64.
fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Vec<f64> {
    let n = b.len();
    for c in 0..n {
        let piv = (c..n)
            .max_by(|&i, &j| a[i][c].abs().total_cmp(&a[j][c].abs()))
            .unwrap();
        a.swap(c, piv);
        b.swap(c, piv);
        let pivot = a[c].clone();
        for r in c + 1..n {
            let m = a[r][c] / pivot[c];
            for (ark, pk) in a[r][c..].iter_mut().zip(&pivot[c..]) {
                *ark -= m * pk;
            }
            b[r] -= m * b[c];
        }
    }
    let mut x = vec![0.0; n];
    for r in (0..n).rev() {
        let s: f64 = (r + 1..n).map(|k| a[r][k] * x[k]).sum();
        x[r] = (b[r] - s) / a[r][r];
    }
    x
}

#[test]
fn a_linear_rls_readout_holds_the_closed_form_ridge_solution() {
    const N_IN: usize = 4;
    let p0 = 10.0f32;
    let mut f = Field::new(N_IN + 1);
    f.configure(Config {
        rls_prior: p0,
        ..rls(1.0)
    });
    f.add_unit((0..N_IN).collect(), vec![0.0; N_IN], 0.0, N_IN);
    f.set_activation(N_IN, Some(Activation::Identity));

    let mut rng = Rng::new(0x51D6E);
    let mut rows: Vec<Vec<f64>> = Vec::new();
    let mut ys: Vec<f64> = Vec::new();
    for _ in 0..300 {
        let x: Vec<f32> = (0..N_IN).map(|_| rng.sym()).collect();
        let y = 0.7 * x[0] - 0.4 * x[1] + 0.2 * x[2] * x[3] + 0.1 + 0.05 * rng.sym();
        let inputs: Vec<(usize, f32)> = x.iter().copied().enumerate().collect();
        f.learn_step(&inputs, N_IN, y, 0.1, 100);
        let mut row: Vec<f64> = x.iter().map(|&v| v as f64).collect();
        row.push(1.0); // the bias read
        rows.push(row);
        ys.push(y as f64);
    }

    // Closed-form ridge with penalty 1/P₀ on every weight and the bias.
    let n = N_IN + 1;
    let mut a = vec![vec![0.0f64; n]; n];
    let mut b = vec![0.0f64; n];
    for (row, &y) in rows.iter().zip(&ys) {
        for i in 0..n {
            b[i] += row[i] * y;
            for j in 0..n {
                a[i][j] += row[i] * row[j];
            }
        }
    }
    for (i, ai) in a.iter_mut().enumerate() {
        ai[i] += 1.0 / p0 as f64;
    }
    let ridge = solve(a, b);

    let u = &f.units[0];
    let got: Vec<f64> = u
        .weights
        .iter()
        .map(|&w| w as f64)
        .chain(core::iter::once(u.bias as f64))
        .collect();
    let gap = got
        .iter()
        .zip(&ridge)
        .map(|(g, r)| (g - r).abs())
        .fold(0.0f64, f64::max);
    eprintln!("[RLS] after 300 events: weights {got:.4?}\n      closed-form ridge {ridge:.4?}  (max gap {gap:.2e})");
    assert!(
        gap < 1e-4,
        "a λ = 1 linear RLS readout is ridge regression: max gap {gap:.2e}"
    );
}

/// NMSE of the backbone regressor on a smooth 2-D target, trained on `n` events of one fixed stream.
fn backbone_nmse(cfg: Config, n: usize) -> (f32, bool) {
    let target = |x: f32, y: f32| 0.8 * (2.0 * x).sin() * (1.5 * y).cos();
    let mut f = Field::new(3);
    f.configure(cfg);
    f.rng_seed(7);
    let feats = f.grow_backbone(&[0, 1], 64, 1.0);
    let mut reads = vec![0, 1];
    reads.extend(&feats);
    f.add_unit(reads.clone(), vec![0.0; reads.len()], 0.0, 2);
    let basis: Vec<(Vec<f32>, f32)> = f.units[..feats.len()]
        .iter()
        .map(|u| (u.weights.clone(), u.bias))
        .collect();

    let mut rng = Rng::new(0xB0B);
    for _ in 0..n {
        let (x, y) = (rng.sym(), rng.sym());
        f.learn_step(&[(0, x), (1, y)], 2, target(x, y), 0.1, 10_000);
    }
    let mut rng = Rng::new(0x7E57);
    let (mut se, mut ss, mut mean) = (0.0f32, 0.0f32, 0.0f32);
    let test: Vec<(f32, f32)> = (0..1000).map(|_| (rng.sym(), rng.sym())).collect();
    for &(x, y) in &test {
        mean += target(x, y) / test.len() as f32;
    }
    for &(x, y) in &test {
        f.stimulate(&[(0, x), (1, y)], 10_000);
        se += (f.cells()[2] - target(x, y)).powi(2);
        ss += (target(x, y) - mean).powi(2);
    }
    let frozen = f.units[..feats.len()]
        .iter()
        .zip(&basis)
        .all(|(u, (w, b))| u.weights == *w && u.bias.to_bits() == b.to_bits());
    (se / ss, frozen)
}

#[test]
fn rls_on_a_frozen_random_basis_beats_the_delta_rule_on_smooth_regression() {
    let (delta, delta_frozen) = backbone_nmse(Config::default(), 3000);
    let (fixed, _) = backbone_nmse(
        Config {
            feature_lr: 0.0,
            ..Config::default()
        },
        3000,
    );
    let (ridge, frozen) = backbone_nmse(rls(1.0), 3000);
    eprintln!(
        "[RLS] 64-feature backbone, 3000 events, held-out NMSE: delta rule {delta:.4} (fixed basis {fixed:.4}), \
         RLS {ridge:.5}"
    );
    assert!(
        !delta_frozen,
        "the default delta rule moves the basis (feature_lr)"
    );
    assert!(
        frozen,
        "an RLS readout leaves its random basis exactly as born"
    );
    assert!(
        ridge < 0.2 * delta.min(fixed),
        "RLS {ridge:.5} vs delta {delta:.4} / fixed-basis delta {fixed:.4}"
    );
}

#[test]
fn an_rls_readout_survives_growth_the_economy_and_a_snapshot() {
    let target = |x: f32, y: f32| 0.8 * (2.0 * x).sin() * (1.5 * y).cos();
    let mut f = Field::new(3);
    f.configure(rls(0.999));
    f.rng_seed(11);
    let mut rng = Rng::new(0x6A0);
    let stream: Vec<(f32, f32)> = (0..2000).map(|_| (rng.sym(), rng.sym())).collect();
    let step = |f: &mut Field, (x, y): (f32, f32)| {
        f.alive(Event::Perceive {
            inputs: &[(0, x), (1, y)],
            target_cell: 2,
            target: target(x, y),
            lr: 0.1,
            budget: 10_000,
        });
    };
    for &s in &stream[..1500] {
        step(&mut f, s);
    }
    let units = f.n_units();
    let mut g = Field::load(&f.snapshot()).expect("an RLS organism snapshots");
    for &s in &stream[1500..] {
        step(&mut f, s);
        step(&mut g, s);
    }
    let same = f
        .cells()
        .iter()
        .zip(g.cells())
        .all(|(a, b)| a.to_bits() == b.to_bits());
    eprintln!(
        "[RLS] alive(Perceive) with an RLS readout: {units} units after 1500 events (grown and culled), \
         surprise {:.4}; snapshot → load → 500 more events bit-identical: {same}",
        f.surprise_of(2)
    );
    assert!(f.cells().iter().all(|c| c.is_finite()));
    assert!(
        same,
        "the reloaded organism continues bit for bit (the covariance is serialized)"
    );
}
