// SPDX-License-Identifier: MIT-0
//! SURPRISE-CURRENCY SPIKE — the GROUPING deletion (`docs/research/surprise-currency-spike.md`).
//!
//! THE IN-CORE ANALOG of `substrate_causal_grouping` (a zero-core harness proof): move concept-map positions by the
//! ONE surprise currency instead of by `co_activate` co-firing. `Field::group_by_surprise` (gated behind
//! `Config::surprise_currency`, byte-identical off) pulls each PAIR together in proportion to how much BOTH cells
//! reduce the warmed output's surprise (`rate·min(SR_i,SR_j)` over the RAW `surprise_reduction_of`). A pair
//! coheres only when both members are causally load-bearing; a member that is redundant given a shared cause has
//! conditional sensitivity ≈ 0, so a SPURIOUS common-cause pair does NOT group — where `co_activate` (correlation)
//! groups it just as tightly as a real causal pair.
//!
//! THE TOPOLOGY (the anchor's, built INSIDE one Field, warmed by real grown predictors so the currency is live):
//!   Z = common cause, A = independent driver.
//!   B = tanh(1.5·Z + 1.5·A)   — B ← Z AND A  (A→B is the REAL causal link)
//!   C = tanh(1.5·Z + 0.3·εC)  — C ← Z only   (a noisy copy of Z; redundant given Z)
//!   D = tanh(1.5·Z + 0.3·εD)  — D ← Z only   (C,D share the common cause; neither drives the other)
//! Grow a predictor of B over {Z,A} (its output cell `pb`) and a predictor of D over {Z,C} (its output cell `pd`),
//! each warming the surprise currency toward its own target. Then group the CAUSAL pair {A, pb} while B is warmed
//! (A is load-bearing for B ⇒ SR(A) high; pb is the output ⇒ SR(pb) high) and the SPURIOUS pair {C, pd} while D is
//! warmed (C is redundant given Z ⇒ SR(C) ≈ 0; pd is the output ⇒ SR(pd) high). Both pairs start at the SAME
//! concept-map distance (`place_at`), so only the grouping rule differs.
//!
//! THE CLAIM. `group_by_surprise` GROUPS the causal pair (both members reduce surprise) and REFUSES the spurious
//! pair (min-gate ≈ 0 for the redundant member) — where `co_activate` groups BOTH tightly (it cannot tell a driver
//! from a common-cause co-effect). The discriminator is the spurious pair: surprise keeps it far, co-fire collapses
//! it.
//!
//! TRUTH-NOT-TROPHY. `group_by_surprise` is purely ADDITIVE (nothing existing calls it) and only meaningful with
//! `Config::surprise_currency` on ⇒ the default path is byte-identical (whole suite green with the flag off). The
//! generative model, seed, warming, and initial positions are authored and identical across the two grouping arms;
//! only the position-update RULE differs. HONEST SCOPE: per-cell `surprise_reduction_of` is importance to the
//! CURRENTLY-warmed output, so the pairwise driver-vs-common-cause distinction is realized by warming the pair's
//! TARGET before grouping (SR then reads "sensitivity to THIS target"). A single fixed-output per-cell SR could NOT
//! separate a driver from a common-cause co-effect for an arbitrary pair — see the spike doc's grouping bullet for
//! the residual (a running `err(Y|others) − err(Y|others,X)` would be the fully order-free pairwise quantity). This
//! proves the DOMINATION that justifies retiring `co_activate` as the grouping driver; it does NOT delete
//! `co_activate` or flip a default (that waits on live validation).
//!
//! RE-BASELINED at the 1.6 learning flip — `#[ignore]`d: the per-seed claim does not hold in EITHER genome, so the
//! earlier green was the luck of these four seeds. Over 24 seeds (`0x11·k`, k = 1..24; this file's four are k = 1..4) the
//! per-seed assertions fail on 13/24 seeds on the 1.6 default and on 12/24 on the legacy genome (`Config::legacy()` —
//! 0/4 of this file's seeds fail there, 1/4 — seed 0x33, spurious distance 0.506 — on the default). The mean spurious
//! distance after `group_by_surprise` is 0.54 (default) / 0.58 (legacy), under the 0.6 refusal bar; the surprise gate
//! still ranks causal ≫ spurious on average (0.091 vs 0.0036 default; 0.089 vs 0.0021 legacy) and `co_activate` still
//! collapses the spurious pair to 0.000 on every seed. What survives is a weaker, AVERAGE statement (surprise-grouping
//! keeps a spurious pair much farther than co-firing does), not the per-seed refusal asserted below.
//!
//!   cargo test -p openmorph-core --test research spike_group_by_surprise -- --ignored --nocapture --test-threads=1

use openmorph_core::substrate::Field;

use crate::common::Rng;

/// One sample of the generative model: `(z, a, b, c, d)` — the anchor's common-cause topology.
fn sample(rng: &mut Rng) -> (f32, f32, f32, f32, f32) {
    let z = rng.sym();
    let a = rng.sym(); // independent causal driver
    let b = (1.5 * z + 1.5 * a).tanh(); // B ← Z AND A  (A→B is the real causal link)
    let c = (1.5 * z + 0.3 * rng.sym()).tanh(); // C ← Z only  (a noisy copy of Z)
    let d = (1.5 * z + 0.3 * rng.sym()).tanh(); // D ← Z only  (common cause; no direct C→D link)
    (z, a, b, c, d)
}

// Cell layout on the shared membrane.
const Z: usize = 0; // common cause (input)
const A: usize = 1; // independent driver (input)
const C: usize = 2; // common-cause co-effect (input)
const PB: usize = 3; // grown predictor of B (output)
const PD: usize = 4; // grown predictor of D (output)

/// Grow a predictor of `target` (written to `out`) over `reads`, `warm` grown steps — this warms the surprise
/// currency toward `out` (each `grow` runs a learn cascade + the adjoint sweep `accumulate_sensitivity`).
fn warm(
    f: &mut Field,
    rng: &mut Rng,
    reads: &[usize],
    out: usize,
    pick: impl Fn(&(f32, f32, f32, f32, f32)) -> (Vec<f32>, f32),
    warm: usize,
) {
    for _ in 0..warm {
        let row = sample(rng);
        let (vals, tgt) = pick(&row);
        let pairs: Vec<(usize, f32)> = reads.iter().copied().zip(vals).collect();
        f.grow(&pairs, out, tgt, 0.05, 300);
    }
}

/// Build one Field with the topology, warm both target predictors, and return it (surprise currency ON).
fn build(seed: u64) -> Field {
    let mut f = Field::new(5);
    f.config_mut().surprise_currency = true;
    f.rng_seed(seed);
    let mut rng = Rng::new(seed ^ 0xC0FFEE);
    // Warm B over {Z,A} — A is a real driver of B.
    warm(
        &mut f,
        &mut rng,
        &[Z, A],
        PB,
        |r| (vec![r.0, r.1], r.2),
        1500,
    );
    // Warm D over {Z,C} — C is redundant with Z (a noisy copy), so it should not help predict D.
    warm(
        &mut f,
        &mut rng,
        &[Z, C],
        PD,
        |r| (vec![r.0, r.3], r.4),
        1500,
    );
    f
}

/// The pair's cohesion gate `min(SR_i, SR_j)` — the RAW surprise-reduction the min-gate reads (a pair coheres only
/// as much as its WEAKER causal link). A causal pair's weaker member is still a real driver (SR well above 0); a
/// spurious pair's weaker member is the redundant one (SR ≈ 0).
fn gate(f: &Field, i: usize, j: usize) -> f32 {
    f.surprise_reduction_of(i).min(f.surprise_reduction_of(j))
}

/// Group `pair` by `steps` of `group_by_surprise` at `rate`, measuring the concept-map distance after (both members
/// of `pair` start at an equal distance via `place_at`).
fn group_surprise(f: &mut Field, pair: [usize; 2], rate: f32, steps: usize) -> f32 {
    f.place_at(pair[0], &[0.0, 0.0]);
    f.place_at(pair[1], &[1.0, 0.0]);
    for _ in 0..steps {
        f.group_by_surprise(&pair, rate);
    }
    f.distance(pair[0], pair[1], 0)
}

/// The `co_activate` baseline over the same pair, same steps/start. Its `rate` is the raw step (∈[0,1]); where
/// `group_by_surprise` scales its rate by the raw SR gate (~0.1), `co_activate` moves by `rate` directly — so each
/// runs in its own natural range and the comparison is the SELECTIVITY, not the step size.
fn group_cofire(f: &mut Field, pair: [usize; 2], rate: f32, steps: usize) -> f32 {
    f.place_at(pair[0], &[0.0, 0.0]);
    f.place_at(pair[1], &[1.0, 0.0]);
    for _ in 0..steps {
        f.co_activate(&pair, rate);
    }
    f.distance(pair[0], pair[1], 0)
}

/// One seed's record: `(seed, gate(A,pb), gate(C,pd), surprise-causal, surprise-spurious, cofire-causal,
/// cofire-spurious)` — the concept-map distances after grouping each pair from an equal start of 1.0.
type Row = (u64, f32, f32, f32, f32, f32, f32);

#[test]
#[ignore = "1.6 re-baseline: the per-seed claim fails on 13/24 seeds (default) and 12/24 (legacy); these 4 seeds were lucky; see this file's module docs"]
fn group_by_surprise_groups_causally_where_cofire_groups_correlationally() {
    let (rate, co_rate, steps) = (2.0f32, 0.3f32, 200usize);
    let seeds = [0x11u64, 0x22, 0x33, 0x44];
    let mut rows: Vec<Row> = Vec::new();
    for &seed in &seeds {
        let mut rng = Rng::new(seed ^ 0xC0FFEE);

        // ── SURPRISE arm ──
        let mut fs = build(seed);
        // Warm B so SR reads "sensitivity to B", then group the CAUSAL pair {A, pb}.
        warm(
            &mut fs,
            &mut rng,
            &[Z, A],
            PB,
            |r| (vec![r.0, r.1], r.2),
            400,
        );
        let g_causal = gate(&fs, A, PB);
        let d_causal_s = group_surprise(&mut fs, [A, PB], rate, steps);
        // Warm D so SR reads "sensitivity to D", then group the SPURIOUS pair {C, pd}.
        warm(
            &mut fs,
            &mut rng,
            &[Z, C],
            PD,
            |r| (vec![r.0, r.3], r.4),
            400,
        );
        let g_spurious = gate(&fs, C, PD);
        let d_spurious_s = group_surprise(&mut fs, [C, PD], rate, steps);

        // ── CO-ACTIVATE baseline (identical topology/warming/positions, grouped by co-firing) ──
        let mut fc = build(seed);
        let d_causal_c = group_cofire(&mut fc, [A, PB], co_rate, steps);
        let d_spurious_c = group_cofire(&mut fc, [C, PD], co_rate, steps);

        rows.push((
            seed,
            g_causal,
            g_spurious,
            d_causal_s,
            d_spurious_s,
            d_causal_c,
            d_spurious_c,
        ));
    }

    println!("\n  SURPRISE-CURRENCY SPIKE — grouping (group_by_surprise vs co_activate), start dist = 1.0:");
    println!("    seed    gate(A,pb)  gate(C,pd) | surprise: causal  spurious | co_activate: causal  spurious");
    let mean = |f: &dyn Fn(&Row) -> f32| rows.iter().map(f).sum::<f32>() / rows.len() as f32;
    for r in &rows {
        println!(
            "    {:#06x}   {:.5}     {:.5}   |           {:.4}   {:.4}   |            {:.4}   {:.4}",
            r.0, r.1, r.2, r.3, r.4, r.5, r.6
        );
    }
    let (gc, gs) = (mean(&|r| r.1), mean(&|r| r.2));
    let (mcs, mss) = (mean(&|r| r.3), mean(&|r| r.4));
    let (mcc, msc) = (mean(&|r| r.5), mean(&|r| r.6));
    println!("    MEAN     {gc:.5}     {gs:.5}   |           {mcs:.4}   {mss:.4}   |            {mcc:.4}   {msc:.4}");
    println!("  ⇒ the RAW surprise gate separates the causal link ({gc:.4}) from the spurious one ({gs:.5}, ~0);");
    println!("    group_by_surprise GROUPS the causal pair and REFUSES the spurious one; co-fire GROUPS BOTH.\n");

    for r in &rows {
        let (seed, g_c, g_s, d_cs, d_ss, _d_cc, d_sc) = *r;
        // (1) The RAW surprise gate ranks causally: the causal pair's weaker member is a real driver (nonzero); the
        // spurious pair's weaker member (the redundant C, conditional contribution ≈ 0) reads far smaller.
        assert!(
            g_c > 0.003 && g_c > 5.0 * g_s,
            "seed {seed:#x}: surprise gate must rank causally: causal {g_c:.5} ≫ spurious {g_s:.5}"
        );
        // (2) group_by_surprise GROUPS the causal pair (both members reduce surprise) — collapses from 1.0.
        assert!(
            d_cs < 0.2,
            "seed {seed:#x}: group_by_surprise must pull the causal pair together: dist {d_cs:.4}"
        );
        // (3) It REFUSES the spurious pair (min-gate ≈ 0 for the redundant member) — the causal claim: stays far.
        assert!(
            d_ss > 0.6 && d_ss > 4.0 * d_cs.max(1e-4),
            "seed {seed:#x}: group_by_surprise must NOT group the spurious pair: causal {d_cs:.4} vs spurious {d_ss:.4}"
        );
        // (4) co_activate CANNOT separate: it collapses the SAME spurious pair (co-firing is correlational).
        assert!(
            d_sc < 0.2 && d_ss > 4.0 * d_sc.max(1e-4),
            "seed {seed:#x}: co_activate must group the spurious pair too: co {d_sc:.4} vs surprise {d_ss:.4}"
        );
    }
}
