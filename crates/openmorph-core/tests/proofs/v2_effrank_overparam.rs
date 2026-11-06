// SPDX-License-Identifier: MIT-0
//! v2.0-dev · GATE-4 follow-up (#149): does `effective_rank` beat `parsimony` as a fitness term in the
//! OVER-PARAMETERIZED regime? MorphLab found them tied (r≈−0.22) on an easy, well-fit task; the honest test is
//! where nominal size ≠ used capacity. Here: fill every morph to the SAME high `max_hidden` (birth_gate=0), so
//! `parsimony` (#units) SATURATES (≈constant), while the task complexity k varies — the readout leans on ~k of
//! the many features, so `effective_rank` (used capacity) TRACKS k, and so does held-out MAE. Prediction:
//! `effective_rank` correlates with generalization; `parsimony`, saturated, cannot.
use openmorph_core::substrate::Field;

use crate::common::Rng;

const NIN: usize = 16; // 8 input pairs
const MAXH: usize = 32; // over-parameterized capacity (fixed for ALL morphs)

fn corr(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len() as f32;
    let (ma, mb) = (a.iter().sum::<f32>() / n, b.iter().sum::<f32>() / n);
    let (mut cov, mut va, mut vb) = (0.0f32, 0.0f32, 0.0f32);
    for (x, y) in a.iter().zip(b) {
        cov += (x - ma) * (y - mb);
        va += (x - ma).powi(2);
        vb += (y - mb).powi(2);
    }
    if va <= 1e-9 || vb <= 1e-9 {
        0.0
    } else {
        cov / (va.sqrt() * vb.sqrt())
    }
}

/// target of complexity k: tanh(Σ_{i<k} 1.5·x[2i]·x[2i+1]) — needs ~k product features.
fn target(k: usize, x: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for i in 0..k {
        s += 1.5 * x[2 * i] * x[2 * i + 1];
    }
    s.tanh()
}

/// Grow an OVER-parameterized morph on task k; return (held-out MAE, parsimony, effective_rank).
fn morph(k: usize, seed: u64) -> (f32, f32, f32) {
    let mut f = Field::new(NIN + 1);
    f.config_mut().birth_gate = 0.0; // fill to exactly MAXH ⇒ parsimony saturates
    f.config_mut().max_hidden = MAXH;
    f.config_mut().feature_lr = 0.3;
    let out = NIN;
    let mut rng = Rng::new(seed);
    let build =
        |x: &[f32]| -> Vec<(usize, f32)> { x.iter().enumerate().map(|(i, &v)| (i, v)).collect() };
    for _ in 0..6000 {
        let x: Vec<f32> = (0..NIN).map(|_| rng.sym()).collect();
        f.grow(&build(&x), out, target(k, &x), 0.02, MAXH + 8);
    }
    let mut r2 = Rng::new(seed ^ 0xFFFF);
    let (mut e, n) = (0.0f32, 400);
    for _ in 0..n {
        let x: Vec<f32> = (0..NIN).map(|_| r2.sym()).collect();
        f.stimulate(&build(&x), MAXH + 8);
        e += (f.cells()[out] - target(k, &x)).abs();
    }
    (e / n as f32, f.parsimony() as f32, f.effective_rank())
}

#[test]
fn effective_rank_separates_where_parsimony_saturates() {
    let (mut maes, mut pars, mut ers) = (Vec::new(), Vec::new(), Vec::new());
    eprintln!("[ER] over-parameterized (max_hidden={MAXH}, birth_gate=0): task-k → MAE / parsimony / effective_rank");
    for k in 1..=7usize {
        for &s in &[0x1u64, 0x2] {
            let (mae, par, er) = morph(k, s);
            maes.push(mae);
            pars.push(par);
            ers.push(er);
            eprintln!("[ER]   k={k} seed={s:#x}:  MAE={mae:.3}  parsimony={par:.0}  effective_rank={er:.2}");
        }
    }
    let r_par = corr(&maes, &pars);
    let r_er = corr(&maes, &ers);
    // spread of each predictor (parsimony saturates ⇒ ~0 spread)
    let spread = |v: &[f32]| {
        let m = v.iter().sum::<f32>() / v.len() as f32;
        (v.iter().map(|x| (x - m).powi(2)).sum::<f32>() / v.len() as f32).sqrt()
    };
    eprintln!(
        "[ER] corr(MAE, parsimony) = {r_par:.3} (parsimony spread {:.2})   corr(MAE, effective_rank) = {r_er:.3} (eff-rank spread {:.2})",
        spread(&pars),
        spread(&ers)
    );

    // mechanism: used capacity ≤ nominal size, always
    for (&er, &par) in ers.iter().zip(pars.iter()) {
        assert!(
            er <= par + 1e-3,
            "effective_rank {er:.2} must be ≤ parsimony {par:.0}"
        );
    }
    // parsimony SATURATES (near-constant) in this regime ⇒ it cannot predict the varying MAE
    assert!(
        spread(&pars) < 2.0,
        "parsimony must be ~saturated in the over-parameterized regime (spread {:.2})",
        spread(&pars)
    );
    // effective_rank varies AND predicts generalization better than saturated parsimony
    assert!(
        spread(&ers) > 2.0 && r_er.abs() > r_par.abs() + 0.2,
        "effective_rank must separate + predict MAE better than saturated parsimony (r_er {r_er:.2} vs r_par {r_par:.2})"
    );

    eprintln!("[ER] VERDICT: in the OVER-PARAMETERIZED regime effective_rank IS the better fitness/health term. With every morph filled to max_hidden, parsimony SATURATES (~constant, spread {:.2}) and cannot predict generalization (r {r_par:.2}); effective_rank tracks the USED capacity (~task complexity), varies, and predicts held-out MAE (r {r_er:.2}). This is exactly where MorphLab's easy-task tie was expected to break (#149) — nominal size ≠ used capacity, and Φ should weight effective_rank there. On easy/well-fit tasks (nominal ≈ used) the two agree, so it's a regime-dependent choice, not a blanket swap.", spread(&pars));
}
