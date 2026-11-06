// SPDX-License-Identifier: MIT-0
//! META-DISCOVERY, POINTED AT THE NEW REGIMES — the honest instrument for "did we forget an equation?"
//! (branch #6 of the Field-Theory-of-the-Digital-Organism program; the symbolic-regression / "specify" route
//! of `docs/research/meta-discovery-search.md`, now run IN-CORE against a RUNNING morph, not synthetic data.)
//!
//! THE HYPOTHESIS UNDER TEST. Meta-discovery has so far only REDISCOVERED momentum (`physics_meta_discovery.rs`:
//! a GA over the physics graph turns on the 2nd-order term the hand default leaves off) — weak evidence the
//! current equations are complete. But that search was pointed at the ONE regime already known to have slack.
//! It has never been aimed at the regimes THIS program opened:
//!   (A) the SELF-ANNEALING solver's temperature (`grown_search.rs`: T from the organism's own violation-energy),
//!   (B) the ECONOMY / metabolism (energy over a unit's lifecycle), and
//!   (C) the NOVELTY signal (the grown surprise EMA the discovery frontier reads).
//! Pointed there, does the search FIND an unhypothesized term — a "temperature = f(energy, size)" law, a
//! mass-like extensive conserved quantity, a novelty law — or does it only recover what we wrote?
//!
//! THE METHOD (the "specify" route, faithful to the doc's spike — STLSQ / SINDy, re-implemented dependency-free
//! in the core test so it stays CI-fast and byte-replayable). For each regime: harvest a BEHAVIOUR TRACE from
//! the real substrate, build a small term LIBRARY over the regime's raw observables, and run
//! Sequential-Thresholded Least-Squares — fit, prune below threshold, refit — which recovers BOTH the sparse
//! FORM (which terms survive) AND the COEFFICIENTS. Two controls per regime, both truth-not-trophy:
//!   * a NULL library (missing the true structure) must fit materially WORSE — the recovered structure is
//!     load-bearing, not a lucky over-parameterisation, and
//!   * an EXTENDED library (the true terms PLUS distractors / cross-couplings / higher orders) must prune every
//!     distractor — if a coupling the written equation misses were real, it would SURVIVE here. Nothing extra
//!     surviving is the direct "no forgotten term" instrument.
//!
//! THE VERDICT (truth-not-trophy, stated up-front so the asserts are not a trophy hunt): across all three new
//! regimes the search recovers the WRITTEN law — form and coefficient — from the substrate's own behaviour, and
//! NO unhypothesized term survives. It CORROBORATES, from data not design, that the annealing temperature is
//! INTENSIVE (energy / size — a purely extensive polynomial cannot fit it) and that energy is an EXTENSIVE
//! quantity obeying a linear income−upkeep balance with a unit self-coefficient (accumulation / near-conserved).
//! It finds no mass-like quantity beyond energy and no novel temperature term. Momentum remains the only slack
//! the search has ever found. Honest scope: regime (A)'s temperature is authored from exactly this closed form,
//! so its recovery CONFIRMS the machinery + selects the intensive structure over extensive alternatives — it
//! does not invent a term; the genuine forgotten-term test is regimes (B)/(C), whose traces are the full coupled
//! dynamics of a live `Field`, where a missed coupling would show up as a surviving distractor and does not.
//!
//!   cargo test -p openmorph-core --test research meta_discovery_regimes -- --ignored --nocapture --test-threads=1

use openmorph_core::substrate::Field;
use openmorph_core::Activation;

use crate::common::Rng;

// =============================================================================================================
// STLSQ — Sequential-Thresholded Least SQuares (the SINDy core), f64, dependency-free.
// =============================================================================================================

/// Ordinary least squares over the SELECTED columns `cols` of design matrix `phi` (n×k) against target `y`,
/// by the normal equations `(ΦᵀΦ + λI) c = Φᵀy` with a tiny ridge for numerical stability, solved by Gaussian
/// elimination with partial pivoting. Returns the coefficient for each selected column, in `cols` order.
fn lstsq(phi: &[Vec<f64>], y: &[f64], cols: &[usize]) -> Vec<f64> {
    let k = cols.len();
    let n = phi.len();
    let mut a = vec![vec![0.0f64; k]; k];
    let mut b = vec![0.0f64; k];
    for r in 0..n {
        for (ii, &ci) in cols.iter().enumerate() {
            b[ii] += phi[r][ci] * y[r];
            for (jj, &cj) in cols.iter().enumerate() {
                a[ii][jj] += phi[r][ci] * phi[r][cj];
            }
        }
    }
    // Ridge relative to the mean diagonal — a whisker of regularisation so a near-singular ΦᵀΦ (the NULL
    // libraries are deliberately collinear) stays invertible without moving a well-posed coefficient.
    let tr: f64 = (0..k).map(|i| a[i][i]).sum::<f64>() / k.max(1) as f64;
    let lambda = 1e-9 * tr.max(1e-12);
    for (i, row) in a.iter_mut().enumerate() {
        row[i] += lambda;
    }
    solve(a, b)
}

/// Solve `A x = b` (A is k×k) by Gaussian elimination with partial pivoting.
fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Vec<f64> {
    let k = b.len();
    for col in 0..k {
        // pivot
        let mut piv = col;
        for r in (col + 1)..k {
            if a[r][col].abs() > a[piv][col].abs() {
                piv = r;
            }
        }
        a.swap(col, piv);
        b.swap(col, piv);
        let d = a[col][col];
        if d.abs() < 1e-300 {
            continue;
        }
        for r in (col + 1)..k {
            let f = a[r][col] / d;
            if f == 0.0 {
                continue;
            }
            let pivot_row = a[col].clone();
            for (c, arc) in a[r].iter_mut().enumerate().skip(col) {
                *arc -= f * pivot_row[c];
            }
            b[r] -= f * b[col];
        }
    }
    let mut x = vec![0.0f64; k];
    for i in (0..k).rev() {
        let mut s = b[i];
        for (c, &xc) in x.iter().enumerate().skip(i + 1) {
            s -= a[i][c] * xc;
        }
        x[i] = if a[i][i].abs() < 1e-300 {
            0.0
        } else {
            s / a[i][i]
        };
    }
    x
}

/// STLSQ over the FULL library `phi` (n×k): iterate {fit on kept cols, prune |coef| < thr, refit} to a fixed
/// point. Returns the full-length coefficient vector (0 for pruned terms). This is the transparent sparse
/// regression that identifies the FORM (which terms survive) as well as the coefficients.
fn stlsq(phi: &[Vec<f64>], y: &[f64], thr: f64) -> Vec<f64> {
    let k = phi[0].len();
    let mut keep: Vec<usize> = (0..k).collect();
    let mut coef = vec![0.0f64; k];
    for _ in 0..16 {
        let c = lstsq(phi, y, &keep);
        coef = vec![0.0f64; k];
        for (ii, &ci) in keep.iter().enumerate() {
            coef[ci] = c[ii];
        }
        let newkeep: Vec<usize> = keep
            .iter()
            .copied()
            .filter(|&ci| coef[ci].abs() >= thr)
            .collect();
        if newkeep.len() == keep.len() || newkeep.is_empty() {
            break;
        }
        keep = newkeep;
    }
    coef
}

/// Coefficient of determination R² of a linear model `coef` over `phi` against `y` (fit quality, 1 = exact).
fn r2(phi: &[Vec<f64>], y: &[f64], coef: &[f64]) -> f64 {
    let n = phi.len();
    let mean = y.iter().sum::<f64>() / n as f64;
    let mut ss_res = 0.0;
    let mut ss_tot = 0.0;
    for r in 0..n {
        let pred: f64 = phi[r].iter().zip(coef).map(|(x, c)| x * c).sum();
        ss_res += (y[r] - pred).powi(2);
        ss_tot += (y[r] - mean).powi(2);
    }
    1.0 - ss_res / ss_tot.max(1e-12)
}

/// The surviving-term names of a coefficient vector (|coef| ≥ thr), for the "no forgotten term" report.
fn survivors<'a>(names: &[&'a str], coef: &[f64], thr: f64) -> Vec<&'a str> {
    names
        .iter()
        .zip(coef)
        .filter(|(_, &c)| c.abs() >= thr)
        .map(|(n, _)| *n)
        .collect()
}

const THR: f64 = 0.02;

// =============================================================================================================
// REGIME (A) — the self-annealing solver's TEMPERATURE. Is it INTENSIVE (energy / size)?
// =============================================================================================================
//
// `grown_search.rs`'s nature-based constraint solver anneals at T = T_MIN + T_SCALE·E/E_max, where E is the
// organism's OWN settled violation-energy (read off native `factor` units) and E_max = the edge count (a size /
// extent measure). We harvest (E, size) from the REAL coupling substrate — E is genuinely the organism's
// violation-energy, not a synthetic number — and ask the search to recover the temperature law from raw
// extensive observables {E, size}. The corroboration the field-theory program predicts: temperature is
// INTENSIVE (energy per unit size), so a purely extensive polynomial in {E, size} CANNOT fit it (the NULL),
// while the ratio E/size fits exactly (the FULL) — the data SELECTS the intensive structure.

const V: usize = 10;
const K: usize = 3;
const T_MIN: f64 = 0.05;
const T_SCALE: f64 = 1.5;

fn spin(v: usize, k: usize) -> usize {
    v * K + k
}

/// A tiny Potts coupler (the `grown_search` mechanism, minimal): native `factor` units compute the organism's
/// own violation-energy for a colouring. Returns (violation-energy E, edge count = size) for a random colouring.
struct Coupler {
    f: Field,
    e_cell: Vec<usize>,
    budget: usize,
    edges: usize,
}

impl Coupler {
    fn build(rng: &mut Rng, m: usize) -> Coupler {
        // random simple graph with `m` edges (satisfiability irrelevant here — we only need a spread of E)
        let mut adj = vec![Vec::new(); V];
        let mut nedges = 0;
        let mut tries = 0;
        while nedges < m && tries < 4000 {
            tries += 1;
            let a = rng.below(V);
            let b = rng.below(V);
            if a != b && !adj[a].contains(&b) {
                adj[a].push(b);
                adj[b].push(a);
                nedges += 1;
            }
        }
        let mut f = Field::new(V * K);
        f.config_mut().activation = Activation::Identity; // identity genome ⇒ exact sums / products
        let mut h_cell = vec![0usize; V * K];
        let mut e_cell = vec![0usize; V * K];
        for v in 0..V {
            for k in 0..K {
                let reads: Vec<usize> = adj[v].iter().map(|&u| spin(u, k)).collect();
                let hc = f.len();
                let n = reads.len();
                f.add_unit(reads, vec![1.0; n], 0.0, hc);
                h_cell[v * K + k] = hc;
            }
        }
        for v in 0..V {
            for k in 0..K {
                let hc = h_cell[v * K + k];
                let ec = f.len();
                f.add_unit(vec![hc], vec![1.0], 0.0, ec);
                f.set_factor(ec, Some(spin(v, k)));
                e_cell[v * K + k] = ec;
            }
        }
        let budget = 4 * f.len();
        Coupler {
            f,
            e_cell,
            budget,
            edges: nedges,
        }
    }

    /// Settle a random colouring; return the organism's OWN violation-energy E (= conflict-edge count).
    fn energy_of_random(&mut self, rng: &mut Rng) -> f64 {
        let colour: Vec<usize> = (0..V).map(|_| rng.below(K)).collect();
        let mut inputs = Vec::with_capacity(V * K);
        for (v, &c) in colour.iter().enumerate() {
            for k in 0..K {
                inputs.push((spin(v, k), if k == c { 1.0 } else { 0.0 }));
            }
        }
        self.f.stimulate(&inputs, self.budget);
        let e: f32 = self.e_cell.iter().map(|&c| self.f.cells()[c]).sum();
        (0.5 * e) as f64
    }
}

#[test]
#[ignore = "research: on demand"]
fn regime_a_annealing_temperature_is_intensive() {
    let mut gen = Rng::new(0xA00E);
    // Harvest (E, size, T) across graph SIZES (edge counts) and random colourings, so both E and size vary.
    let mut e_v = Vec::new();
    let mut size_v = Vec::new();
    let mut t_v = Vec::new();
    for &m in &[10usize, 14, 18, 22, 26] {
        let mut c = Coupler::build(&mut gen, m);
        let size = c.edges as f64;
        for _ in 0..40 {
            let e = c.energy_of_random(&mut gen);
            e_v.push(e);
            size_v.push(size);
            // the solver's OWN self-annealing temperature (grown_search.rs::temperature)
            t_v.push(T_MIN + T_SCALE * e / size.max(1.0));
        }
    }
    let n = e_v.len();

    // Library columns: 0:1  1:E  2:size  3:E·size  4:E²  5:E/size  6:1/size  7:size/(E+1)
    let names = [
        "1",
        "E",
        "size",
        "E*size",
        "E^2",
        "E/size",
        "1/size",
        "size/(E+1)",
    ];
    let phi: Vec<Vec<f64>> = (0..n)
        .map(|r| {
            let (e, s) = (e_v[r], size_v[r]);
            vec![1.0, e, s, e * s, e * e, e / s, 1.0 / s, s / (e + 1.0)]
        })
        .collect();

    // NULL — a purely EXTENSIVE polynomial in {E, size} (cols 0..5: 1, E, size, E·size, E²). No ratio.
    let null_cols: Vec<usize> = vec![0, 1, 2, 3, 4];
    let null_coef_sel = lstsq(&phi, &t_v, &null_cols);
    let mut null_coef = vec![0.0; names.len()];
    for (ii, &ci) in null_cols.iter().enumerate() {
        null_coef[ci] = null_coef_sel[ii];
    }
    let null_r2 = r2(&phi, &t_v, &null_coef);

    // FULL — STLSQ over the whole library (the intensive ratio E/size is now available).
    let full_coef = stlsq(&phi, &t_v, THR);
    let full_r2 = r2(&phi, &t_v, &full_coef);
    let surv = survivors(&names, &full_coef, THR);

    eprintln!("[REGIME-A] annealing temperature — {n} samples from the real coupling substrate (E = the organism's own violation-energy)");
    eprintln!(
        "[REGIME-A]   NULL (extensive polynomial in E,size, NO ratio): R² = {null_r2:.4}  ← cannot fit an intensive quantity"
    );
    eprintln!(
        "[REGIME-A]   FULL (STLSQ): T = {:.4}·1 + {:.4}·(E/size)   R² = {full_r2:.6}   survivors {surv:?}",
        full_coef[0], full_coef[5]
    );
    eprintln!(
        "[REGIME-A]   truth: T_MIN = {T_MIN}, T_SCALE = {T_SCALE}   →  recovered |ΔT_MIN| = {:.2e}, |ΔT_SCALE| = {:.2e}",
        (full_coef[0] - T_MIN).abs(),
        (full_coef[5] - T_SCALE).abs()
    );
    eprintln!("[REGIME-A]   CORROBORATION: temperature is INTENSIVE (energy/size); a purely extensive polynomial fails. NO forgotten term (only 1 and E/size survive). CAVEAT: T is authored from this closed form, so recovery CONFIRMS the search + selects the intensive structure — it does not invent a term.");

    // --- asserts ---
    // The intensive ratio recovers the temperature law to precision.
    assert!(
        (full_coef[0] - T_MIN).abs() < 0.01 && (full_coef[5] - T_SCALE).abs() < 0.01,
        "STLSQ must recover T = {T_MIN} + {T_SCALE}·E/size (got {:.4} + {:.4}·E/size)",
        full_coef[0],
        full_coef[5]
    );
    // The FORM is exactly {constant, E/size} — nothing extra survives (the "no forgotten term" instrument).
    let mut surv_sorted = surv.clone();
    surv_sorted.sort_unstable();
    assert_eq!(
        surv_sorted,
        vec!["1", "E/size"],
        "only the constant and the intensive ratio may survive (got {surv:?})"
    );
    // The intensive structure is LOAD-BEARING: the extensive polynomial (NULL) leaves a material residual the
    // ratio removes exactly. (A 4-term polynomial APPROXIMATES 1/size over a bounded size range, so its R² is
    // high but not 1 — the honest gap; the ratio is exact.)
    assert!(
        full_r2 > 0.999 && null_r2 < full_r2 - 0.02,
        "the intensive ratio must fit exactly (R² {full_r2:.6}) where an extensive polynomial leaves residual (R² {null_r2:.4})"
    );
}

// =============================================================================================================
// REGIME (B) — the ECONOMY / metabolism. Is energy an EXTENSIVE quantity on a linear income−upkeep balance?
// =============================================================================================================
//
// A live `Field`'s energy law (substrate.rs::metabolize_cells / metabolize_reliance) is
//   e ← min(e + income·reliance − upkeep, cap).
// We run a REAL economy — a two-layer morph whose readout gives each feature cell a distinct reliance — for a
// lifecycle, logging (e_before, reliance, e_after) per unit each event (BELOW the cap, the linear regime). The
// search recovers e_next = 1·e + income·reliance − upkeep: a UNIT self-coefficient (accumulation / near-
// conservation) plus a linear balance of an EXTENSIVE quantity. The distractor library tests for a forgotten
// coupling (e·reliance, e², reliance²): none survives.
//
// PINNED to `reliance_norm` at its `Config::legacy()` value (`RelianceNorm::Absolute`) — every other lever is the 1.6
// default (the 1.6 learning flip): the law written above is the ABSOLUTE economy's. The 1.6 default runs the
// mean-relative economy, e ← min(e + income·reliance/mean(reliance) − upkeep, cap); on it the search recovers that law
// instead (coefficient 0.0930 = 0.1 / 1.075, the readout's mean |weight|), which passed this proof's 0.01 tolerance on
// INCOME only by coincidence (|0.0930 − 0.1| = 0.007).

#[test]
#[ignore = "research: on demand"]
fn regime_b_economy_energy_is_extensive_balance() {
    const INCOME: f32 = 0.1;
    const UPKEEP: f32 = 0.05;
    const NFEAT: usize = 8;
    const CAP: f32 = 10.0; // Config default energy_cap

    let mut rng = Rng::new(0xEC0);
    let mut f = Field::new(1 + NFEAT + 1); // cell 0 = input, 1..=NFEAT = features, last = readout
    f.config_mut().reliance_norm = openmorph_core::Config::legacy().reliance_norm; // pinned — see above
                                                                                   // Feature units: each reads the input, writes its own feature cell (unit index = feature order).
    let mut feat_cell = Vec::new();
    let mut feat_unit = Vec::new();
    for w in 0..NFEAT {
        let c = 1 + w;
        let idx = f.add_unit(vec![0], vec![rng.sym()], rng.sym(), c);
        feat_cell.push(c);
        feat_unit.push(idx);
    }
    // Readout reads every feature with a DISTINCT weight ⇒ each feature cell gets a distinct reliance
    // (reliance_of(c) = Σ readers |weight on c| = |readout weight on c|), fixed (we never learn).
    let readout_cell = 1 + NFEAT;
    let ro_weights: Vec<f32> = (0..NFEAT).map(|w| 0.2 + 0.25 * w as f32).collect();
    f.add_unit(feat_cell.clone(), ro_weights, 0.0, readout_cell);
    f.stimulate(&[(0, 1.0)], 64); // one settle so reliance/weights are live

    let mut e_before = Vec::new();
    let mut reliance = Vec::new();
    let mut e_after = Vec::new();
    for _ in 0..40 {
        // pre-state per feature unit
        let pre: Vec<f32> = feat_unit.iter().map(|&i| f.energy_of(i)).collect();
        let rel: Vec<f32> = feat_cell.iter().map(|&c| f.reliance_of(c)).collect();
        f.metabolize_reliance(&feat_cell, INCOME, UPKEEP);
        for w in 0..NFEAT {
            let raw = pre[w] + INCOME * rel[w] - UPKEEP;
            // keep only BELOW-CAP samples — the linear regime (the min() clamp is the one nonlinearity, excluded)
            if raw < CAP - 1e-4 {
                e_before.push(pre[w] as f64);
                reliance.push(rel[w] as f64);
                e_after.push(f.energy_of(feat_unit[w]) as f64);
            }
        }
    }
    let n = e_before.len();
    assert!(n > 100, "need a populated economy trace (got {n})");

    // Library: 0:1  1:e  2:reliance  3:e·reliance  4:e²  5:reliance²
    let names = ["1", "e", "reliance", "e*rel", "e^2", "rel^2"];
    let phi: Vec<Vec<f64>> = (0..n)
        .map(|r| {
            let (e, rl) = (e_before[r], reliance[r]);
            vec![1.0, e, rl, e * rl, e * e, rl * rl]
        })
        .collect();

    // NULL — drop the reliance income term (cols 0,1: 1, e only): a bare decay, no earnings.
    let null_cols = vec![0usize, 1];
    let null_sel = lstsq(&phi, &e_after, &null_cols);
    let mut null_coef = vec![0.0; names.len()];
    for (ii, &ci) in null_cols.iter().enumerate() {
        null_coef[ci] = null_sel[ii];
    }
    let null_r2 = r2(&phi, &e_after, &null_coef);

    // FULL — STLSQ over the whole library.
    let full_coef = stlsq(&phi, &e_after, THR);
    let full_r2 = r2(&phi, &e_after, &full_coef);
    let surv = survivors(&names, &full_coef, THR);

    eprintln!("[REGIME-B] economy energy law — {n} below-cap samples from a live Field lifecycle");
    eprintln!(
        "[REGIME-B]   FULL (STLSQ): e_next = {:.4}·e + {:.4}·reliance + {:.4}·1   R² = {full_r2:.6}   survivors {surv:?}",
        full_coef[1], full_coef[2], full_coef[0]
    );
    eprintln!(
        "[REGIME-B]   truth: e_next = 1·e + {INCOME}·reliance − {UPKEEP}   →  |Δself| = {:.2e}, |Δincome| = {:.2e}, |Δ(−upkeep)| = {:.2e}",
        (full_coef[1] - 1.0).abs(),
        (full_coef[2] - INCOME as f64).abs(),
        (full_coef[0] - (-(UPKEEP as f64))).abs()
    );
    eprintln!(
        "[REGIME-B]   NULL (bare e-decay, no earnings term): R² = {null_r2:.4}  (high only because energy is ACCUMULATION-dominated — the income term is small-variance yet recovered EXACTLY; completeness rests on the coefficients + survivor set, not the R² gap)"
    );
    eprintln!("[REGIME-B]   CORROBORATION: energy is EXTENSIVE — a UNIT self-coefficient (accumulation / near-conserved) + a linear income−upkeep balance. NO forgotten coupling survives (e·reliance, e², reliance² all pruned). No mass-like quantity beyond energy itself.");

    // --- asserts (real-morph trace ⇒ a genuine completeness test) ---
    assert!(
        (full_coef[1] - 1.0).abs() < 0.01,
        "energy self-coefficient must be 1 (accumulation), got {:.4}",
        full_coef[1]
    );
    assert!(
        (full_coef[2] - INCOME as f64).abs() < 0.01,
        "income coefficient on reliance must be {INCOME}, got {:.4}",
        full_coef[2]
    );
    assert!(
        (full_coef[0] + UPKEEP as f64).abs() < 0.01,
        "constant must be −upkeep = {}, got {:.4}",
        -(UPKEEP as f64),
        full_coef[0]
    );
    let mut surv_sorted = surv.clone();
    surv_sorted.sort_unstable();
    assert_eq!(
        surv_sorted,
        vec!["1", "e", "reliance"],
        "only the written balance terms may survive — no forgotten coupling (got {surv:?})"
    );
    assert!(
        full_r2 > 0.9999 && full_r2 > null_r2,
        "the FULL balance must fit exactly (R² {full_r2:.6}) and beat the earnings-free NULL (R² {null_r2:.4})"
    );
}

// =============================================================================================================
// REGIME (C) — the NOVELTY signal. Is the grown surprise a first-order EMA of |err|?
// =============================================================================================================
//
// The discovery frontier reads the organism's grown SURPRISE (substrate.rs::grow):
//   surprise[c] ← sd·surprise[c] + (1 − sd)·|err|      (sd = surprise_decay = 0.9).
// We drive a REAL `grow` loop toward a learnable target, and log (s_before, |err|, s_after) — |err| is grow's
// OWN returned drive, s the organism's OWN grown surprise cell. The search recovers the two EMA coefficients
// (sd, 1−sd) and prunes the bias + the s·|err| / s² distractors. This is the doc's "next step 1": SINDy-fit a
// RUNNING morph, confirming the observed novelty dynamics match the written equation with no extra coupling.

#[test]
#[ignore = "research: on demand"]
fn regime_c_novelty_surprise_ema() {
    const SD: f64 = 0.9; // Config default surprise_decay
    let mut rng = Rng::new(0x5017);
    let mut f = Field::new(6); // cells 0..4 inputs, 5 = target head
    let target_cell = 5;
    // a learnable target so |err| SPANS a range (large early, shrinking as grow learns) — identifiable EMA
    let w_true = [0.7f32, -0.4, 0.5, -0.3];

    let mut s_before = Vec::new();
    let mut abserr = Vec::new();
    let mut s_after = Vec::new();
    for _ in 0..600 {
        let x = [rng.sym(), rng.sym(), rng.sym(), rng.sym()];
        let inputs: Vec<(usize, f32)> = (0..4).map(|k| (k, x[k])).collect();
        let target = (0..4).map(|k| w_true[k] * x[k]).sum::<f32>().tanh();
        let s0 = f.surprise_of(target_cell) as f64;
        let (err, _born) = f.grow(&inputs, target_cell, target, 0.1, 64);
        let s1 = f.surprise_of(target_cell) as f64;
        // skip the very first samples where the surprise cell is still cold (0) — the EMA relation holds from
        // the first update on, but discard s0==0 && s1 first-touch to avoid a degenerate all-zeros row.
        s_before.push(s0);
        abserr.push(err as f64);
        s_after.push(s1);
    }
    let n = s_before.len();

    // Library: 0:1  1:s  2:|err|  3:s·|err|  4:s²
    let names = ["1", "s", "|err|", "s*|err|", "s^2"];
    let phi: Vec<Vec<f64>> = (0..n)
        .map(|r| {
            let (s, u) = (s_before[r], abserr[r]);
            vec![1.0, s, u, s * u, s * s]
        })
        .collect();

    // NULL — only the drive term (cols 0,2: 1, |err|): no carried state ⇒ cannot be an EMA.
    let null_cols = vec![0usize, 2];
    let null_sel = lstsq(&phi, &s_after, &null_cols);
    let mut null_coef = vec![0.0; names.len()];
    for (ii, &ci) in null_cols.iter().enumerate() {
        null_coef[ci] = null_sel[ii];
    }
    let null_r2 = r2(&phi, &s_after, &null_coef);

    // FULL — STLSQ.
    let full_coef = stlsq(&phi, &s_after, THR);
    let full_r2 = r2(&phi, &s_after, &full_coef);
    let surv = survivors(&names, &full_coef, THR);

    eprintln!("[REGIME-C] novelty / surprise EMA — {n} samples from a running grow loop (s = the organism's own grown surprise)");
    eprintln!(
        "[REGIME-C]   FULL (STLSQ): s_next = {:.4}·s + {:.4}·|err|  (bias {:+.4})   R² = {full_r2:.6}   survivors {surv:?}",
        full_coef[1], full_coef[2], full_coef[0]
    );
    eprintln!(
        "[REGIME-C]   truth: sd = {SD}, (1−sd) = {:.2}   →  |Δsd| = {:.2e}, |Δ(1−sd)| = {:.2e}",
        1.0 - SD,
        (full_coef[1] - SD).abs(),
        (full_coef[2] - (1.0 - SD)).abs()
    );
    eprintln!("[REGIME-C]   NULL (drive only, no carried state): R² = {null_r2:.4}");
    eprintln!("[REGIME-C]   CORROBORATION: novelty is a first-order EMA of |err| with the substrate's own decay; the CARRIED STATE (s term) is load-bearing. NO forgotten term (s·|err|, s² pruned). Matches the written law on a LIVE morph.");

    // --- asserts (real-morph trace) ---
    assert!(
        (full_coef[1] - SD).abs() < 0.01,
        "surprise decay must recover sd = {SD}, got {:.4}",
        full_coef[1]
    );
    assert!(
        (full_coef[2] - (1.0 - SD)).abs() < 0.01,
        "drive coefficient must recover 1−sd = {:.2}, got {:.4}",
        1.0 - SD,
        full_coef[2]
    );
    let mut surv_sorted = surv.clone();
    surv_sorted.sort_unstable();
    assert_eq!(
        surv_sorted,
        vec!["s", "|err|"],
        "only the EMA terms may survive — bias and cross-terms pruned (got {surv:?})"
    );
    assert!(
        full_r2 > 0.999 && null_r2 < full_r2 - 0.05,
        "the carried state is load-bearing: FULL R² {full_r2:.4} must beat drive-only NULL R² {null_r2:.4}"
    );
}
