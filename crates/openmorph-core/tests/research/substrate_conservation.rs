// SPDX-License-Identifier: MIT-0
//! CONSERVATION & SYMMETRY — a Noether probe of the digital organism (research spike, branch #4 of the
//! "Field Theory of the Digital Organism" program; `docs/research/conservation-symmetry.md`).
//!
//! HYPOTHESIS. Real physics CONSERVES quantities, and a continuous SYMMETRY implies a conserved quantity
//! (Noether) that can DERIVE constants we currently hand-set. The organism runs an energy economy — so:
//!   (1) does anything stay INVARIANT across its dynamics (grow / learn / settle / metabolize / cull /
//!       reproduce)? Candidates: total energy `Σ eᵢ`, total contribution `Σ|creditᵢ|`, total reliance.
//!   (2) is there a SYMMETRY (weight×gain rescale, async-clock reparam, cell permutation) under which a
//!       quantity is invariant — and does that invariant DERIVE / CONSTRAIN a constant?
//!
//! TRUTH-NOT-TROPHY. The headline result is a NEGATIVE on (1): the economy conserves NOTHING — energy is
//! MINTED at birth and by income, DESTROYED by upkeep / the cap / cull. It is an openly DRIVEN / DISSIPATIVE
//! system (a metabolism), not a closed Hamiltonian one. The POSITIVE results are on (2): the *dynamics* carry
//! three exact symmetries whose invariants are BEHAVIOURAL (the settled function), and each derives/constrains a
//! constant. Every number below is read from a real run (`--nocapture` to see them).
//!
//! run: cargo test -p openmorph-core --test research substrate_conservation -- --ignored --nocapture --test-threads=1

use openmorph_core::substrate::Field;

use crate::common::Rng;

// ── helpers ─────────────────────────────────────────────────────────────────────────────────────

/// Total energy in the account: `Σ eᵢ` over every grown unit (the candidate "conserved charge").
fn total_energy(f: &Field) -> f32 {
    (0..f.n_units()).map(|i| f.energy_of(i)).sum()
}

/// Total contribution credit `Σ|creditᵢ|` over every write cell (the residual-correlation currency).
fn total_credit(f: &Field) -> f32 {
    write_cells(f).iter().map(|&c| f.credit_of(c)).sum()
}

/// Every cell WRITTEN by a unit (features + outputs), deduped ascending.
fn write_cells(f: &Field) -> Vec<usize> {
    let mut v: Vec<usize> = f.units.iter().map(|u| u.write).collect();
    v.sort_unstable();
    v.dedup();
    v
}

/// Index of the (first) unit writing `cell`.
fn writer_idx(f: &Field, cell: usize) -> usize {
    f.units
        .iter()
        .position(|u| u.write == cell)
        .expect("a writer")
}

fn l2(w: &[f32]) -> f32 {
    w.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Settle to the fixed point for input `(x0,x1)` and read `cell`. A perturbation step FIRST guarantees the
/// input cells CHANGE (the substrate fires ON CHANGE — re-injecting the same value would not recompute), so
/// this reads a freshly-settled value even right after we mutate a weight by hand.
fn settle_out(f: &mut Field, x0: f32, x1: f32, cell: usize) -> f32 {
    f.stimulate(&[(0, x0 + 0.137), (1, x1 - 0.091)], 64); // perturb away
    f.stimulate(&[(0, x0), (1, x1)], 64); // then the real input
    f.cells()[cell]
}

// ── (1) CONSERVATION: energy is MINTED and DESTROYED — nothing is conserved ──────────────────────
//
// The energy law (E1) is `eᵢ ← min(eᵢ + income·φᵢ − upkeep, energy_cap)`, birth seeds `eᵢ = birth_energy`,
// death (cull) deletes the account. None of these is a TRANSFER between units — income is minted per-unit
// out of nothing, upkeep and the cap destroy to nothing, birth creates and cull annihilates. So `Σ eᵢ`
// is not a conserved charge. We show all three leaks with numbers, and FALSIFY the transfer hypothesis.
#[test]
#[ignore = "research: on demand"]
fn energy_is_openly_dissipative_not_conserved() {
    let mut f = Field::new(2); // inputs 0,1; target 2
    let mut rng = Rng::new(0xC0FFEE);

    // GROW a nonlinear task. `grow` births units but NEVER metabolizes, so afterwards every unit sits at
    // exactly `birth_energy` — energy created EX NIHILO at birth.
    for _ in 0..1500 {
        let (x0, x1) = (rng.sym(), rng.sym());
        let t = (1.3 * x0 * x1).tanh();
        f.grow(&[(0, x0), (1, x1)], 2, t, 0.1, 40);
    }
    let birth_energy = f.config().birth_energy;
    let u_born = f.n_units();
    let e_born = total_energy(&f);
    println!(
        "[E] after grow: units={u_born}  ΣE={e_born:.3}  (= units·birth_energy = {:.3})",
        u_born as f32 * birth_energy
    );
    // MINTED: total energy == units · birth_energy, created from nothing (there was 0 before any unit).
    assert!(u_born > 1, "the task should grow a real population");
    assert!(
        (e_born - u_born as f32 * birth_energy).abs() < 1e-3,
        "post-grow energy is purely minted birth reserve"
    );

    // FALSIFY the TRANSFER hypothesis: if the economy conserved energy, income to one unit would be debited
    // from another. Metabolize only a SUBSET; the untouched units must stay EXACTLY at birth_energy (no flow
    // reached them) while the total changes — income is minted per-unit, not transferred.
    let all = write_cells(&f);
    let (subset, rest): (Vec<usize>, Vec<usize>) = all.iter().partition(|&&c| c % 2 == 0);
    let rest_before: f32 = rest.iter().map(|&c| f.energy_of(writer_idx(&f, c))).sum();
    f.metabolize_reliance(&subset, 1.0, 0.5); // income 1.0·reliance − upkeep 0.5, on the subset only
    let rest_after: f32 = rest.iter().map(|&c| f.energy_of(writer_idx(&f, c))).sum();
    let e_meta = total_energy(&f);
    println!(
        "[E] metabolize SUBSET only: ΣE {e_born:.3} → {e_meta:.3}   untouched-subset ΣE {rest_before:.3} → {rest_after:.3}"
    );
    assert!(
        (rest_before - rest_after).abs() < 1e-6,
        "energy did NOT flow to the un-metabolized units — income is minted, not transferred"
    );
    assert!(
        (e_meta - e_born).abs() > 1e-3,
        "the total changed — a source/sink, not a conserved transfer"
    );

    // DESTROYED at death: drain hard, then cull. Total energy drops by the annihilated accounts.
    for _ in 0..12 {
        f.metabolize_reliance(&all, 0.0, 1.0); // pure upkeep — starve everyone toward death
    }
    let e_pre_cull = total_energy(&f);
    let u_pre = f.n_units();
    let culled = f.cull(0.0);
    let e_post_cull = total_energy(&f);
    println!(
        "[E] starve+cull: units {u_pre}→{}  culled={culled}  ΣE {e_pre_cull:.3}→{e_post_cull:.3} (DESTROYED)",
        f.n_units()
    );
    assert!(
        culled > 0 && f.n_units() < u_pre,
        "cull annihilates starved units"
    );
    assert!(
        e_post_cull < e_pre_cull - 1e-4 || f.n_units() == 0,
        "energy is DESTROYED at death — not conserved"
    );

    println!("[E] VERDICT: Σeᵢ is minted at birth, minted/burned by income/upkeep/cap, annihilated at cull → OPEN system, no conserved energy charge.");
}

// ── (1b) CONSERVATION: credit is a LEAKY trace (spent surprise), not an accumulating store ────────
//
// `credit ← credit_decay·credit + (1−credit_decay)·grad` is an EMA of the gradient. It is NOT a running
// integral — when the residual (surprise) goes to zero, credit DISSIPATES geometrically. So `Σ|credit|`
// tracks CURRENT surprise-reduction and decays when learning stops; there is no conserved credit.
#[test]
#[ignore = "research: on demand"]
fn credit_is_a_dissipative_trace_not_conserved() {
    let mut f = Field::new(2);
    f.config_mut().max_hidden = 6;
    let mut rng = Rng::new(0x5EED);

    // Train a nonlinear task so credit accrues on the grown features.
    for _ in 0..1200 {
        let (x0, x1) = (rng.sym(), rng.sym());
        let t = (1.1 * x0 * x1).tanh();
        f.grow(&[(0, x0), (1, x1)], 2, t, 0.1, 40);
    }
    let credit_trained = total_credit(&f);

    // Now feed ZERO-surprise steps: target := the model's OWN current output ⇒ err = 0 ⇒ grad = 0. Credit
    // must decay toward 0 at rate `credit_decay` per step — surprise-reduction that is SPENT, not banked.
    let cd = f.config().credit_decay;
    for _ in 0..1200 {
        let (x0, x1) = (rng.sym(), rng.sym());
        f.stimulate(&[(0, x0), (1, x1)], 40);
        let self_target = f.cells()[2]; // predict yourself → no error
        f.learn_step(&[(0, x0), (1, x1)], 2, self_target, 0.0, 40);
    }
    let credit_settled = total_credit(&f);
    println!(
        "[C] Σ|credit|: trained={credit_trained:.4}  after zero-surprise={credit_settled:.4}  (credit_decay={cd})"
    );
    assert!(credit_trained > 1e-3, "credit accrues while learning");
    assert!(
        credit_settled < 0.25 * credit_trained,
        "credit DISSIPATES once surprise is gone — a leaky trace, not a conserved store"
    );
    println!("[C] VERDICT: credit is an EMA of surprise-reduction — it LEAKS to 0 at rest, never conserved.");
}

// ── (2a) SYMMETRY: the weight×gain GAUGE — an exact continuous redundant DOF (the Noether result) ─
//
// A unit computes `out = tanh(gain·(Σ wₖ xₖ + b))`. The 1-parameter group `(w, b, gain) → (w/α, b/α, α·gain)`
// leaves the pre-activation `gain·(Σ w x + b)` — hence the OUTPUT — EXACTLY invariant. The behaviour is the
// conserved quantity; the overall magnitude `||w||` (equivalently `gain`) is a pure GAUGE coordinate the
// function does not fix. This is the redundant DOF the economy currently tunes twice (heritable `scale` AND
// the learned weights) for ONE physical magnitude.
#[test]
#[ignore = "research: on demand"]
fn weight_gain_rescale_is_an_exact_gauge_symmetry() {
    let mut f = Field::new(2);
    f.config_mut().max_hidden = 0; // one clean writer unit: tanh(gain·(w·x + b))
    let mut rng = Rng::new(0xA11CE);
    for _ in 0..500 {
        let (x0, x1) = (rng.sym(), rng.sym());
        let t = (0.9 * x0 - 0.6 * x1).tanh();
        f.grow(&[(0, x0), (1, x1)], 2, t, 0.1, 40);
    }

    let ui = writer_idx(&f, 2);
    let w_before = f.units[ui].weights.clone();
    let norm_before = l2(&w_before);
    let g0 = f.scale_of(2);
    let out0 = settle_out(&mut f, 0.7, -0.4, 2);

    // Apply the gauge transform with a non-trivial α.
    let alpha = 3.7_f32;
    for w in f.units[ui].weights.iter_mut() {
        *w /= alpha;
    }
    f.units[ui].bias /= alpha;
    f.set_scale(2, alpha * g0);

    let out1 = settle_out(&mut f, 0.7, -0.4, 2);
    let norm_after = l2(&f.units[ui].weights);
    // The behaviour-invariant: the PRODUCT gain·wₖ is unchanged for every k.
    let prod_drift: f32 = w_before
        .iter()
        .zip(f.units[ui].weights.iter())
        .map(|(&wb, &wa)| (g0 * wb - (alpha * g0) * wa).abs())
        .fold(0.0, f32::max);

    println!(
        "[G] output {out0:.6} → {out1:.6}  |Δ|={:.2e}   ||w|| {norm_before:.4} → {norm_after:.4} (×{:.3})   max |Δ(gain·w)|={prod_drift:.2e}",
        (out1 - out0).abs(),
        norm_after / norm_before
    );
    assert!(
        (out1 - out0).abs() < 1e-5,
        "output is EXACTLY gauge-invariant"
    );
    assert!(
        (norm_after - norm_before / alpha).abs() < 1e-4,
        "the magnitude ||w|| is NOT invariant — a redundant gauge coordinate"
    );
    assert!(
        prod_drift < 1e-5,
        "the physical invariant is the product gain·(w,b)"
    );
    println!("[G] VERDICT: a 1-param CONTINUOUS gauge symmetry. Invariant = behaviour (product gain·w). Redundant coord = magnitude → DERIVES the canonical gauge-fix (init_scale=1 + unit-variance weights).");
}

// ── (2b) SYMMETRY: async-clock REPARAM — the fixed point is the invariant; the clock is gauge ─────
//
// The substrate is async / fire-on-change with no global tick. For a MEMORYLESS (feed-forward) organism the
// settled output is a function of the INPUT ALONE — invariant to the number of events (the "clock") spent
// reaching it, and to the PATH taken. The clock is a pure gauge coordinate the answer does not depend on.
// (Deliberately BROKEN by `integrate`/`delay`, which read event-order — that is the sequence-memory faculty.)
#[test]
#[ignore = "research: on demand"]
fn async_clock_reparam_leaves_the_fixed_point_invariant() {
    let mut f = Field::new(2);
    f.config_mut().max_hidden = 4;
    let mut rng = Rng::new(0xF13D);
    for _ in 0..600 {
        let (x0, x1) = (rng.sym(), rng.sym());
        let t = (1.2 * x0 * x1 + 0.3 * x0).tanh();
        f.grow(&[(0, x0), (1, x1)], 2, t, 0.1, 40);
    }

    // Fixed point for xA from a cold settle.
    f.stimulate(&[(0, 0.5), (1, -0.3)], 80);
    let out_a = f.cells()[2];
    let c1 = f.clock();

    // A long DETOUR through other inputs (spends many clock ticks), then RETURN to xA.
    for _ in 0..25 {
        let (x0, x1) = (rng.sym(), rng.sym());
        f.stimulate(&[(0, x0), (1, x1)], 80);
    }
    f.stimulate(&[(0, 0.5), (1, -0.3)], 80);
    let out_a2 = f.cells()[2];
    let c2 = f.clock();

    println!(
        "[T] fixed point for xA: {out_a:.7} (clock {c1}) → after detour {out_a2:.7} (clock {c2})  |Δout|={:.2e}  Δclock={}",
        (out_a2 - out_a).abs(),
        c2 - c1
    );
    assert!(
        (out_a2 - out_a).abs() < 1e-6,
        "the fixed point is path- and clock-INVARIANT (the invariant of the async dynamics)"
    );
    assert!(
        c2 > c1,
        "the clock advanced — a gauge coordinate the answer ignores"
    );
    println!("[T] VERDICT: settle is idempotent up to the fixed point; the event-count is gauge → CONSTRAINS budget/settle_eps as pure COMPUTE knobs, not behavioural ones.");
}

// ── (2c) SYMMETRY: cell PERMUTATION — a graph automorphism leaves behaviour invariant ─────────────
//
// Two hidden features that read the same inputs are interchangeable: relabelling them (swap their genomes +
// swap the readout's weights on them) is a graph AUTOMORPHISM. Behaviour is invariant; the cell INDEX is a
// gauge label. This is why `fingerprint`/`embed` are index-free and the economy selects on reliance/credit
// (both index-free), never on a cell id.
#[test]
#[ignore = "research: on demand"]
fn cell_permutation_is_a_behaviour_preserving_automorphism() {
    let mut f = Field::new(2);
    f.config_mut().max_hidden = 8;
    f.config_mut().birth_gate = 0.02; // encourage several hidden features
    let mut rng = Rng::new(0xBEEF);
    for _ in 0..1500 {
        let (x0, x1) = (rng.sym(), rng.sym());
        let t = (1.4 * x0 * x1).tanh(); // strongly nonlinear ⇒ multiple hidden features born
        f.grow(&[(0, x0), (1, x1)], 2, t, 0.1, 40);
    }

    // Two hidden feature cells (written cells beyond the inputs+target) that both read inputs [0,1].
    let hidden: Vec<usize> = write_cells(&f)
        .into_iter()
        .filter(|&c| c > 2 && f.units[writer_idx(&f, c)].reads == vec![0, 1])
        .collect();
    assert!(
        hidden.len() >= 2,
        "need ≥2 interchangeable hidden features (got {})",
        hidden.len()
    );
    let (h1, h2) = (hidden[0], hidden[1]);

    let out0 = settle_out(&mut f, 0.62, -0.51, 2);

    // The automorphism swapping h1 ↔ h2: swap the two features' genomes (weights/bias/scale) and swap the
    // readout's weight on h1 with its weight on h2.
    let (i1, i2) = (writer_idx(&f, h1), writer_idx(&f, h2));
    let (w1, b1) = (f.units[i1].weights.clone(), f.units[i1].bias);
    let (w2, b2) = (f.units[i2].weights.clone(), f.units[i2].bias);
    f.units[i1].weights = w2;
    f.units[i1].bias = b2;
    f.units[i2].weights = w1;
    f.units[i2].bias = b1;
    let (s1, s2) = (f.scale_of(h1), f.scale_of(h2));
    f.set_scale(h1, s2);
    f.set_scale(h2, s1);
    // swap the readout's weights on the two feature cells
    let r = writer_idx(&f, 2);
    let k1 = f.units[r]
        .reads
        .iter()
        .position(|&c| c == h1)
        .expect("readout reads h1");
    let k2 = f.units[r]
        .reads
        .iter()
        .position(|&c| c == h2)
        .expect("readout reads h2");
    f.units[r].weights.swap(k1, k2);

    let out1 = settle_out(&mut f, 0.62, -0.51, 2);
    println!(
        "[P] swap hidden cells {h1}↔{h2}: output {out0:.6} → {out1:.6}  |Δ|={:.2e}",
        (out1 - out0).abs()
    );
    assert!(
        (out1 - out0).abs() < 1e-5,
        "behaviour is invariant under the relabelling — cell index is a gauge label"
    );
    println!("[P] VERDICT: permutation automorphism ⇒ the organism's identity is its GRAPH up to iso → CONSTRAINS the economy to index-free fitness (reliance/credit), as it already is.");
}

// ── (1c) The OPEN-SYSTEM analog of conservation: FLUX BALANCE (a steady state, not a charge) ──────
//
// An open economy has no conserved sum, but under a stationary task the continuous loop grow → metabolize →
// cull reaches a STEADY STATE where the birth flux ≈ the death flux: the population size stabilises. That
// homeostatic fixed point — not a conserved charge — is the organism's substitute for conservation, and it
// LINKS the income/upkeep ratio to the survival threshold (the derivable constraint).
#[test]
#[ignore = "research: on demand"]
fn open_economy_reaches_flux_balance_steady_state() {
    let mut f = Field::new(2);
    f.config_mut().max_hidden = 40;
    let mut rng = Rng::new(0x0F10);
    let mut sizes = Vec::new();
    for step in 0..3000 {
        let (x0, x1) = (rng.sym(), rng.sym());
        let t = (1.2 * x0 * x1).tanh();
        f.grow(&[(0, x0), (1, x1)], 2, t, 0.1, 40);
        let feats: Vec<usize> = write_cells(&f).into_iter().filter(|&c| c > 2).collect();
        f.metabolize_reliance(&feats, 1.0, 0.4); // income·reliance − upkeep
        f.cull(0.1); // death threshold
        if step >= 2000 && step % 100 == 0 {
            sizes.push(f.n_units());
        }
    }
    let mean = sizes.iter().sum::<usize>() as f32 / sizes.len() as f32;
    let spread = (sizes
        .iter()
        .map(|&s| (s as f32 - mean).powi(2))
        .sum::<f32>()
        / sizes.len() as f32)
        .sqrt();
    println!(
        "[F] steady-state population over the tail: mean={mean:.1} units  std={spread:.2}  (samples={:?})",
        sizes
    );
    // Bounded and stable: birth flux ≈ death flux. NOT a conserved charge — a driven fixed point.
    assert!(
        mean > 0.0 && mean < 40.0,
        "population is bounded (self-sizing), not runaway"
    );
    assert!(
        spread < 0.30 * mean + 3.0,
        "the population size has settled to a flux-balance steady state"
    );
    println!("[F] VERDICT: the OPEN system's analog of conservation is FLUX BALANCE (birth≈death) — a homeostatic steady state linking income/upkeep to the cull threshold, not a conserved sum.");
}
