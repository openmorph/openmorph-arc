// SPDX-License-Identifier: MIT-0
//! THE NOVELTY FIELD — is "novelty" ONE signal the equations forgot, or several faces of `surprise`?
//!
//! RESEARCH SPIKE (branch #1 of the "Field Theory of the Digital Organism" program; truth-not-trophy — a clean
//! falsification IS a result). The organism reinvents a "novelty / curiosity" signal in ≥4 harnesses:
//!   (a) BEHAVIOUR-novelty = a nested cell's cache-MISS (`nested_settles`, `substrate_discover`).
//!   (b) WORLD-surprise    = prediction error, the physics' ONE signal `s=|target−pred|` (`observe`,
//!                           `surprise_of`, `substrate_observe`/`substrate_predict`).
//!   (c) VALUE-uncertainty = ensemble DISAGREEMENT `aggregate().spread` (VoI, `substrate_voi`).
//!   (d) FRONTIER-distance = farthest-point distance on the grown concept-map (`substrate_skillmap`).
//!
//! The physics §0 (`docs/research/organism-differential-equations.md`) names exactly ONE driving signal —
//! `surprise` (the `err` hub). QUESTION: is novelty DERIVABLE from surprise (so 3-of-4 collapse), or is it a
//! genuine SECOND axis the equations forgot?
//!
//! We put all four on the SAME controlled scenarios. The controlled world gives each state TWO orthogonal
//! knobs: EPISTEMIC (visitation count — reducible by data) and ALEATORIC (irreducible target noise). The four
//! measures are read against those knobs. HONEST DECOMPOSITION (the asserts; full write-up in
//! `docs/research/novelty-field.md`):
//!   R1  the surprise LEVEL is NOT novelty — on two equally-visited states, noise alone moves it (a≠b).
//!   R2  ensemble SPREAD isolates the EPISTEMIC part surprise conflates: spread ⟂ noise, ∝ 1/visits;
//!       surprise = epistemic ⊕ aleatoric.
//!   R3  cache-miss (a) and spread (c) COLLAPSE to one latent (occupancy / familiarity): high mutual corr.
//!   R3b frontier-distance (d) is that SAME latent, but ONLY through a GROWN metric (a raw map is blind).
//!   R4  THE UNIFICATION — the missing variable is a MOMENT of surprise, not its level. The time-derivative
//!       `learning_progress` (§0 S4, dS/dt) RANKS a still-learnable clean state ABOVE a noisy-familiar one,
//!       INVERTING the level's ranking. That is why world-surprise-LEVEL curiosity HURT ARC while the
//!       reducible signals (VoI / frontier) HELPED.
//!   R5  HONEST CAVEAT — cache-miss is an EXACT occupancy meter for DISCRETE behaviour, but noise-fooled on
//!       CONTINUOUS input (the same aleatoric contamination the surprise LEVEL suffers).
//!
//!   cargo test -p openmorph-core --test research substrate_novelty_field -- --ignored --nocapture --test-threads=1

use openmorph_core::substrate::Field;

// ---------- The controlled world: N states × {epistemic visits, aleatoric noise} ----------

const N: usize = 16; // distinct states
const E: usize = 6; // ensemble estimators (for the VALUE-disagreement measure c)
const BUD: usize = 64; // settle budget — must exceed the per-event fan-in of the N·E-unit ensemble
const LR: f32 = 0.30;
const NOISE_AMP: f32 = 0.6;

/// Deterministic clean target for state `s` — spread in [-0.8, 0.8].
fn base(s: usize) -> f32 {
    -0.8 + 1.6 * (s as f32 + 0.5) / N as f32
}
/// A state is NOISY iff odd — half carry irreducible (aleatoric) noise, half are clean.
fn is_noisy(s: usize) -> bool {
    s % 2 == 1
}
/// A deterministic, ~zero-mean noise sample for (state, visit) — replayable, no RNG state.
fn noise_sample(s: usize, visit: usize) -> f32 {
    if !is_noisy(s) {
        return 0.0;
    }
    let h = (s as u64)
        .wrapping_mul(0x9E3779B97F4A7C15)
        .wrapping_add((visit as u64).wrapping_mul(0xD1B54A32D192ED03));
    let u = ((h >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0;
    NOISE_AMP * u
}
/// The observed (noisy) target on the `visit`-th presentation of state `s`.
fn observed(s: usize, visit: usize) -> f32 {
    base(s) + noise_sample(s, visit)
}
/// The state one-hot as membrane input (all N cells set each visit ⇒ the input CHANGES ⇒ units fire).
fn onehot(s: usize) -> Vec<(usize, f32)> {
    (0..N)
        .map(|i| (i, if i == s { 1.0 } else { 0.0 }))
        .collect()
}

// ---------- (c) VALUE-disagreement — an ensemble; spread = epistemic uncertainty ----------
//
// Estimator (s,e) reads bit `s`, starts at a VARIED prior (they disagree before data), and is corrected toward
// the SAME observed sample each visit (common-mode world) ⇒ pairwise gap CONTRACTS by (1−LR) each visit —
// decaying with visitation, INDEPENDENT of the noise magnitude (common noise moves the mean, not the spread).
struct Ensemble {
    f: Field,
}
impl Ensemble {
    fn new() -> Self {
        let mut f = Field::new(N);
        for s in 0..N {
            for e in 0..E {
                let prior = ((s * E + e) as f32).sin() * 0.9; // deterministic varied prior ~[-0.9,0.9]
                f.add_unit(vec![s], vec![prior], 0.0, N + s * E + e);
            }
        }
        Ensemble { f }
    }
    fn est_cells(s: usize) -> Vec<usize> {
        (0..E).map(|e| N + s * E + e).collect()
    }
    fn learn(&mut self, s: usize, target: f32) {
        self.learn_lr(s, target, LR);
    }
    fn learn_lr(&mut self, s: usize, target: f32, lr: f32) {
        let inp = onehot(s);
        for c in Self::est_cells(s) {
            self.f.learn_step(&inp, c, target, lr, BUD);
        }
    }
    /// (c) ensemble DISAGREEMENT for state `s` — `aggregate().spread` over its estimators.
    fn spread(&mut self, s: usize) -> f32 {
        self.f.stimulate(&onehot(s), BUD);
        let cells = self.f.cells();
        let vals: Vec<f32> = Self::est_cells(s).iter().map(|&c| cells[c]).collect();
        openmorph_core::grouping::aggregate(&vals).spread
    }
}

// ---------- (b) WORLD-surprise — the organism's OWN grown surprise, per state, via `grow` ----------
//
// One predictor cell per state, grown over the state one-hot; `grow` maintains the surprise EMA and its lagging
// trace, so `surprise_of` (the LEVEL) and `learning_progress` (its normalized DERIVATIVE, §0 S4) are the
// organism's own signals — no re-derivation in the harness.
struct Predictor {
    f: Field,
}
impl Predictor {
    fn new() -> Self {
        Predictor { f: Field::new(N) }
    }
    fn pred_cell(s: usize) -> usize {
        N + s
    }
    fn learn(&mut self, s: usize, target: f32) {
        self.f.grow(&onehot(s), Self::pred_cell(s), target, LR, BUD);
    }
    /// (b) the surprise LEVEL at state `s`.
    fn surprise(&self, s: usize) -> f32 {
        self.f.surprise_of(Self::pred_cell(s))
    }
    /// The reducible part: `learning_progress` = the positive normalized surprise-derivative (§0 S4).
    fn learning_progress(&self, s: usize) -> f32 {
        self.f.learning_progress(Self::pred_cell(s))
    }
}

fn pearson(xs: &[f32], ys: &[f32]) -> f32 {
    let n = xs.len() as f32;
    let mx = xs.iter().sum::<f32>() / n;
    let my = ys.iter().sum::<f32>() / n;
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for (&x, &y) in xs.iter().zip(ys) {
        sxy += (x - mx) * (y - my);
        sxx += (x - mx) * (x - mx);
        syy += (y - my) * (y - my);
    }
    if sxx <= 1e-12 || syy <= 1e-12 {
        return 0.0;
    }
    sxy / (sxx.sqrt() * syy.sqrt())
}

/// SPEARMAN rank correlation — Pearson on the value ranks. The right tool here: every novelty measure is a
/// MONOTONE (but nonlinear) function of visitation, so rank agreement, not linear agreement, is the question.
fn spearman(xs: &[f32], ys: &[f32]) -> f32 {
    let rank = |v: &[f32]| -> Vec<f32> {
        let mut idx: Vec<usize> = (0..v.len()).collect();
        idx.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap());
        let mut r = vec![0.0f32; v.len()];
        for (rank, &i) in idx.iter().enumerate() {
            r[i] = rank as f32;
        }
        r
    };
    pearson(&rank(xs), &rank(ys))
}

/// A frozen leaf skill: `out = tanh(1.0·x)` — a distinct behaviour over one input cell (for the cache-miss meter).
fn leaf() -> Field {
    let mut f = Field::new(2);
    f.add_unit(vec![0], vec![1.0], 0.0, 1);
    f
}

// =====================================================================================================
// R1 — the surprise LEVEL is NOT novelty: noise alone decouples them on two equally-visited states.
// =====================================================================================================
#[test]
#[ignore = "research: on demand"]
fn r1_surprise_level_is_not_novelty_noise_decouples_them() {
    let mut pred = Predictor::new();
    for r in 0..80 {
        for s in 0..N {
            pred.learn(s, observed(s, r));
        }
    }
    let (clean, noisy) = (4usize, 5usize); // both visited 80× ⇒ occupancy-novelty(both)=0
    let (s_clean, s_noisy) = (pred.surprise(clean), pred.surprise(noisy));

    println!("\n[R1] two FULLY-VISITED states (occupancy-novelty = 0 for both):");
    println!("     surprise(clean s{clean}) = {s_clean:.4}   (model learned it ⇒ collapses)");
    println!(
        "     surprise(noisy s{noisy}) = {s_noisy:.4}   (stuck at the aleatoric floor ≈ E|noise| ≈ {:.2})",
        NOISE_AMP / 2.0
    );

    assert!(
        s_clean < 0.08,
        "familiar-clean surprise should collapse, got {s_clean:.4}"
    );
    assert!(
        s_noisy > 3.0 * s_clean.max(0.02),
        "familiar-noisy surprise stays HIGH (aleatoric) though equally visited: {s_noisy:.4} vs {s_clean:.4}"
    );
    // Occupancy-novelty is identical (0) for the two, yet surprise differs >3× ⇒ surprise LEVEL carries an
    // aleatoric component novelty does not. The two are ORTHOGONAL.
}

// =====================================================================================================
// R2 — ensemble SPREAD isolates the epistemic part surprise conflates.
// spread(fresh) ≫ spread(familiar); spread(familiar-clean) ≈ spread(familiar-noisy) (noise-independent),
// while surprise(familiar-noisy) ≫ surprise(familiar-clean). ⇒ spread = epistemic; surprise = epi ⊕ aleatoric.
// =====================================================================================================
#[test]
#[ignore = "research: on demand"]
fn r2_ensemble_spread_isolates_epistemic_surprise_conflates_aleatoric() {
    let mut ens = Ensemble::new();
    let mut pred = Predictor::new();
    let (fresh_clean, fresh_noisy) = (8usize, 9usize); // kept fresh (1 visit)
    for r in 0..80 {
        for s in 0..N {
            if (s == fresh_clean || s == fresh_noisy) && r > 0 {
                continue;
            }
            let t = observed(s, r);
            ens.learn(s, t);
            pred.learn(s, t);
        }
    }
    let (fam_clean, fam_noisy) = (4usize, 5usize);
    let sp_fresh = 0.5 * (ens.spread(fresh_clean) + ens.spread(fresh_noisy));
    let sp_fam_clean = ens.spread(fam_clean);
    let sp_fam_noisy = ens.spread(fam_noisy);
    let su_fam_clean = pred.surprise(fam_clean);
    let su_fam_noisy = pred.surprise(fam_noisy);

    println!("\n[R2] epistemic vs aleatoric decomposition:");
    println!("     spread(FRESH)            = {sp_fresh:.4}  (high epistemic — few visits)");
    println!("     spread(familiar clean)   = {sp_fam_clean:.4}");
    println!("     spread(familiar noisy)   = {sp_fam_noisy:.4}  ≈ clean ⇒ spread ⟂ noise (EPISTEMIC only)");
    println!("     surprise(familiar clean) = {su_fam_clean:.4}");
    println!("     surprise(familiar noisy) = {su_fam_noisy:.4}  ≫ clean ⇒ surprise = epistemic ⊕ ALEATORIC");

    assert!(
        sp_fresh > 5.0 * sp_fam_clean.max(1e-3),
        "spread must fall with visits: fresh {sp_fresh:.4} vs fam {sp_fam_clean:.4}"
    );
    let spread_ratio = sp_fam_noisy.max(1e-4) / sp_fam_clean.max(1e-4);
    assert!(
        spread_ratio < 6.0,
        "spread ~noise-independent, ratio {spread_ratio:.2}"
    );
    assert!(
        su_fam_noisy > 3.0 * su_fam_clean.max(0.02),
        "surprise carries the aleatoric floor: {su_fam_noisy:.4} vs {su_fam_clean:.4}"
    );
}

// =====================================================================================================
// R3 — cache-miss (a) and ensemble-spread (c) COLLAPSE to one latent (occupancy / familiarity).
// On a CLEAN visitation gradient (state k visited k+1 times) both track the truth 1/visits and each other.
// =====================================================================================================
#[test]
#[ignore = "research: on demand"]
fn r3_cache_miss_and_spread_collapse_to_one_occupancy_axis() {
    let clean: Vec<usize> = (0..N).filter(|s| !is_noisy(*s)).collect();
    let visits: Vec<usize> = (0..clean.len()).map(|k| k + 1).collect(); // 1,2,3,..,8 — a LOW gradient so
                                                                        // spread has not fully converged

    // (c) spread per clean state after its visits.
    // a SLOW ensemble lr so spread decays GRADUALLY across 1..8 visits (a strong lr collapses it in ~2 steps,
    // tying the high-visit states and destroying the rank signal — the honest reason for the slow rate here).
    let mut ens = Ensemble::new();
    for (k, &s) in clean.iter().enumerate() {
        for r in 0..visits[k] {
            ens.learn_lr(s, observed(s, r), 0.12); // clean ⇒ observed == base
        }
    }
    let spread: Vec<f32> = clean.iter().map(|&s| ens.spread(s)).collect();

    // (a) cache-miss rate per clean state: replay the visits interleaved (never twice in a row ⇒ the nested
    // input always changes) through ONE frozen skill; miss_rate = misses / presentations.
    let mut host = Field::new(1);
    host.nest(leaf(), &[0], 1, BUD);
    let maxv = *visits.iter().max().unwrap();
    let mut interleaved: Vec<usize> = Vec::new();
    for r in 0..maxv {
        for (k, &v) in visits.iter().enumerate() {
            if r < v {
                interleaved.push(k);
            }
        }
    }
    let mut misses = vec![0usize; clean.len()];
    let mut seen = vec![0usize; clean.len()];
    for &k in &interleaved {
        let before = host.nested_settles();
        host.stimulate(&[(0, base(clean[k]))], BUD); // a stable discrete key per state
        if host.nested_settles() > before {
            misses[k] += 1;
        }
        seen[k] += 1;
    }
    let miss_rate: Vec<f32> = (0..clean.len())
        .map(|k| misses[k] as f32 / seen[k] as f32)
        .collect();

    let truth: Vec<f32> = visits.iter().map(|&v| 1.0 / v as f32).collect(); // occupancy-novelty (↓ with visits)
    let c_spread = spearman(&truth, &spread);
    let c_miss = spearman(&truth, &miss_rate);
    let c_pair = spearman(&spread, &miss_rate);
    // linear (Pearson) too, to show the relation is monotone but nonlinear (why rank is the honest measure).
    let p_spread = pearson(&truth, &spread);

    println!(
        "\n[R3] occupancy collapse (truth = 1/visits) — SPEARMAN rank corr (monotone, nonlinear):"
    );
    println!("     corr(truth, spread   c) = {c_spread:.3}   (Pearson {p_spread:.3} — nonlinear, so rank is honest)");
    println!("     corr(truth, cache-miss a) = {c_miss:.3}");
    println!("     corr(spread, cache-miss)  = {c_pair:.3}  ← two measures are the SAME latent");

    assert!(
        c_spread > 0.9,
        "spread ranks states by visitation-novelty, got {c_spread:.3}"
    );
    assert!(
        c_miss > 0.9,
        "cache-miss ranks states by visitation-novelty, got {c_miss:.3}"
    );
    assert!(
        c_pair > 0.9,
        "spread and cache-miss agree directly, got {c_pair:.3}"
    );
}

// =====================================================================================================
// R3b — frontier-distance (d) is the SAME occupancy latent, but ONLY through a GROWN metric.
// On the RAW map the frontier is blind (no visitation signal); after growing it from co-visitation, an
// under-explored state sits farther from the visited cluster than a well-explored one.
// =====================================================================================================
#[test]
#[ignore = "research: on demand"]
fn r3b_frontier_distance_is_occupancy_only_through_a_grown_metric() {
    // States 0..8 are "explored" (a cluster); 8..16 are "under-explored". Grow the map: pull the explored
    // cluster together by co-activation (the experience of using them together), the substrate_skillmap move.
    let explored: Vec<usize> = (0..8).collect();
    let under: Vec<usize> = (8..16).collect();

    let mut f = Field::new(N);
    // RAW-map baseline: min-distance of each group to the explored set (positions are ungrown scatter).
    let raw_gap = frontier_gap(&f, &explored, &under);
    // GROW the metric: repeatedly co-activate the explored cluster so those cells draw together.
    for _ in 0..200 {
        f.co_activate(&explored, 0.15);
    }
    f.spread_positions(); // keep the map from collapsing (isotropy), the standard follow-up
    let grown_gap = frontier_gap(&f, &explored, &under);

    println!("\n[R3b] frontier-distance as occupancy (mean min-dist to the explored cluster):");
    println!(
        "     RAW  map: explored {:.3}  vs under-explored {:.3}  (gap {:.3} — blind)",
        raw_gap.0,
        raw_gap.1,
        raw_gap.1 - raw_gap.0
    );
    println!(
        "     GROWN map: explored {:.3}  vs under-explored {:.3}  (gap {:.3} — frontier appears)",
        grown_gap.0,
        grown_gap.1,
        grown_gap.1 - grown_gap.0
    );

    // On the grown map, under-explored states are farther from the explored cluster than explored states are
    // from each other — the frontier is now an occupancy signal. On the raw map there is (essentially) no gap.
    assert!(
        grown_gap.1 - grown_gap.0 > 2.0 * (raw_gap.1 - raw_gap.0).abs().max(1e-3),
        "growing the metric must MAKE the frontier: grown gap {:.3} vs raw gap {:.3}",
        grown_gap.1 - grown_gap.0,
        raw_gap.1 - raw_gap.0
    );
    assert!(
        grown_gap.1 > grown_gap.0,
        "under-explored sits at the frontier on the grown map"
    );
}

/// (mean min-distance of the explored set to itself, mean min-distance of the under-explored set to the explored set).
fn frontier_gap(f: &Field, explored: &[usize], under: &[usize]) -> (f32, f32) {
    let min_to = |s: usize, set: &[usize]| {
        set.iter()
            .filter(|&&c| c != s)
            .map(|&c| f.distance(s, c, 0))
            .fold(f32::INFINITY, f32::min)
    };
    let e = explored.iter().map(|&s| min_to(s, explored)).sum::<f32>() / explored.len() as f32;
    let u = under.iter().map(|&s| min_to(s, explored)).sum::<f32>() / under.len() as f32;
    (e, u)
}

// =====================================================================================================
// R4 — THE UNIFICATION: novelty is a MOMENT of surprise (its DERIVATIVE), not its LEVEL.
// The surprise LEVEL ranks a noisy-familiar state ABOVE a still-learnable clean state (the noisy-TV trap);
// `learning_progress` (dS/dt, §0 S4) INVERTS that ranking — flagging the reducible state instead. This is
// exactly why world-surprise-LEVEL curiosity HURT ARC while the reducible signals (VoI/frontier) HELPED.
// =====================================================================================================
#[test]
#[ignore = "research: on demand"]
fn r4_novelty_is_the_derivative_of_surprise_not_its_level() {
    let mut pred = Predictor::new();
    for r in 0..80 {
        for s in 0..N {
            pred.learn(s, observed(s, r));
        }
    }
    let (clean, noisy) = (4usize, 5usize); // both fully visited
    let (lvl_clean, lvl_noisy) = (pred.surprise(clean), pred.surprise(noisy));
    let (lp_clean, lp_noisy) = (pred.learning_progress(clean), pred.learning_progress(noisy));

    println!("\n[R4] the surprise LEVEL vs its DERIVATIVE as the useful curiosity signal:");
    println!("     surprise LEVEL:      clean {lvl_clean:.3}  noisy {lvl_noisy:.3}  → ranks the NOISY state higher (WRONG: nothing to learn there)");
    println!("     learning_progress:   clean {lp_clean:.3}  noisy {lp_noisy:.3}  → ranks the CLEAN/reducible state higher (RIGHT)");

    // The two signals give OPPOSITE rankings; the derivative's is the useful one.
    assert!(
        lvl_noisy > lvl_clean,
        "the LEVEL is fooled: it rates the noisy-TV state as more surprising"
    );
    assert!(
        lp_clean > lp_noisy,
        "learning_progress inverts it — reducible surprise on the clean state, ~0 on the noisy: {lp_clean:.3} vs {lp_noisy:.3}"
    );
    assert!(
        lp_noisy < 0.1,
        "the derivative correctly rejects the irreducible (aleatoric) state: {lp_noisy:.3}"
    );
}

// =====================================================================================================
// R5 — HONEST CAVEAT: cache-miss is EXACT for DISCRETE behaviour but NOISE-FOOLED for CONTINUOUS input.
// The same aleatoric contamination the surprise LEVEL suffers — a symmetry worth naming.
// =====================================================================================================
#[test]
#[ignore = "research: on demand"]
fn r5_cache_miss_exact_for_discrete_but_noise_fooled_for_continuous() {
    // DISCRETE: replay 4 states in a non-repeating interleave, TWICE. pass 1 = all misses; pass 2 = all hits.
    let states = [0.2f32, -0.5, 0.7, -0.1];
    let mut host = Field::new(1);
    host.nest(leaf(), &[0], 1, BUD);
    let mut misses = [0usize; 2];
    for (pass, m) in misses.iter_mut().enumerate() {
        for &v in &states {
            let before = host.nested_settles();
            host.stimulate(&[(0, v)], BUD);
            if host.nested_settles() > before {
                *m += 1;
            }
        }
        let _ = pass;
    }
    println!("\n[R5] cache-miss as an occupancy meter:");
    println!(
        "     DISCRETE:   pass1 misses={} (all novel)  pass2 misses={} (all familiar ⇒ EXACT)",
        misses[0], misses[1]
    );
    assert_eq!(misses[0], states.len(), "first pass = all cache-misses");
    assert_eq!(
        misses[1], 0,
        "second pass = all cache-hits (exact discrete occupancy meter)"
    );

    // CONTINUOUS-noisy: revisit ONE underlying state, fresh sensory noise each time ⇒ the bit-key never
    // repeats ⇒ every visit is a MISS though the behaviour is entirely familiar. Over-counts.
    let mut host2 = Field::new(1);
    host2.nest(leaf(), &[0], 1, BUD);
    let (visits, mut noisy_misses, mut prev) = (12usize, 0usize, f32::NAN);
    for visit in 0..visits {
        let mut v = 0.3 + noise_sample(1, visit); // same base, fresh noise each visit
        if v == prev {
            v += 1e-3; // guarantee the input CHANGES (fire-on-change)
        }
        prev = v;
        let before = host2.nested_settles();
        host2.stimulate(&[(0, v)], BUD);
        if host2.nested_settles() > before {
            noisy_misses += 1;
        }
    }
    println!("     CONTINUOUS: {noisy_misses}/{visits} visits counted 'novel' (noise-fooled — over-counts, like the surprise LEVEL)");
    assert!(
        noisy_misses >= visits - 1,
        "continuous noise defeats the occupancy meter: {noisy_misses}/{visits}"
    );
}
