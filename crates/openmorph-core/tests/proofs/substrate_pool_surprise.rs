// SPDX-License-Identifier: MIT-0
//! substrate_pool_surprise — `alive`'s living pool paid by SURPRISE REDUCTION (`Config::pool_surprise`) instead of
//! reliance. A feature earns its end-to-end ablation importance `(∂output/∂c)²·Var(c)` over the pool mean. Claims:
//!   1. PARSIMONY where a few features carry the answer: on a smooth 2-D target the pool is far smaller than under
//!      the reliance economy, at no worse held-out error.
//!   2. THE BOUNDARY, kept as a claim: on a 64-input target whose answer is spread over many weak features, a newborn
//!      feature has too little measured importance when its grace ends, so the pool turns over many times (a birth /
//!      death treadmill the reliance economy does not run).
//!
//!   cargo test -p openmorph-core --test proofs substrate_pool_surprise -- --nocapture

use openmorph_core::{Config, Event, Field, Outcome};

use crate::common::Rng;

/// Train `alive(Perceive)` on `n` events of `f(x)` over `d` inputs; return (held-out NMSE, units, births).
fn run(
    pool_surprise: bool,
    d: usize,
    n: usize,
    target: &dyn Fn(&[f32]) -> f32,
) -> (f32, usize, usize) {
    let mut f = Field::new(d + 1);
    f.configure(Config {
        pool_surprise,
        ..Config::default()
    });
    f.rng_seed(3);
    let mut rng = Rng::new(0x5EED);
    let mut births = 0;
    for _ in 0..n {
        let x: Vec<f32> = (0..d).map(|_| rng.sym()).collect();
        let inputs: Vec<(usize, f32)> = x.iter().copied().enumerate().collect();
        if let Outcome::Perceived { born: true, .. } = f.alive(Event::Perceive {
            inputs: &inputs,
            target_cell: d,
            target: target(&x),
            lr: 0.1,
            budget: 100_000,
        }) {
            births += 1;
        }
    }
    let mut rng = Rng::new(0x7E57);
    let test: Vec<Vec<f32>> = (0..1000)
        .map(|_| (0..d).map(|_| rng.sym()).collect())
        .collect();
    let mean = test.iter().map(|x| target(x)).sum::<f32>() / test.len() as f32;
    let (mut se, mut ss) = (0.0f32, 0.0f32);
    for x in &test {
        let inputs: Vec<(usize, f32)> = x.iter().copied().enumerate().collect();
        f.stimulate(&inputs, 100_000);
        se += (f.cells()[d] - target(x)).powi(2);
        ss += (target(x) - mean).powi(2);
    }
    (se / ss, f.n_units(), births)
}

#[test]
fn a_surprise_paid_pool_is_smaller_where_few_features_carry_the_answer() {
    let smooth = |x: &[f32]| 0.8 * (2.5 * x[0]).sin() * (2.0 * x[1]).cos();
    let (e_rel, u_rel, _) = run(false, 2, 10_000, &smooth);
    let (e_sur, u_sur, _) = run(true, 2, 10_000, &smooth);
    eprintln!(
        "[POOL] smooth 2-D target, 10k alive events: reliance economy {u_rel} units NMSE {e_rel:.4}; \
         surprise economy {u_sur} units NMSE {e_sur:.4}"
    );
    assert!(
        (u_sur as f32) < 0.6 * u_rel as f32,
        "surprise pays only the features that remove error: {u_sur} vs {u_rel} units"
    );
    assert!(
        e_sur <= 1.2 * e_rel,
        "and keeps the error: NMSE {e_sur:.4} vs {e_rel:.4}"
    );
}

#[test]
fn a_surprise_paid_pool_turns_over_where_the_answer_is_spread_wide() {
    // 64 inputs, target = mean of 32 pairwise products: no small set of features carries it.
    let wide = |x: &[f32]| (0..32).map(|k| x[2 * k] * x[2 * k + 1]).sum::<f32>() / 8.0;
    let (_, _, b_rel) = run(false, 64, 4_000, &wide);
    let (_, _, b_sur) = run(true, 64, 4_000, &wide);
    eprintln!(
        "[POOL] 64-input pairwise target, 4k alive events: births under reliance {b_rel}, under surprise {b_sur}"
    );
    assert!(
        b_sur > 3 * b_rel,
        "the surprise economy culls newborns before they earn: {b_sur} vs {b_rel} births"
    );
}
