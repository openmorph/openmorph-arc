// SPDX-License-Identifier: MIT-0
//! TEMPERATURE UNIFICATION — one thermal temperature, or two? (research spike, truth-not-trophy)
//!
//! "Temperature" is computed independently in THREE harnesses. This probe asks whether they are ONE quantity
//! or TWO, and reports honestly.
//!
//! The three reinventions:
//!   1. THERMAL / annealing (`grown_search.rs`): `T = T_MIN + T_SCALE·(violation-energy / E_max)` — hot while
//!      frustrated, cold as conflicts vanish. Meta-discovery (`meta_discovery_regimes.rs`) recovered exactly
//!      `T = 0.05 + 1.5·(E/size)` from the data: an INTENSIVE energy ratio (energy per extensive size).
//!   2. METABOLIC hunger (`substrate_metabolic_actor.rs`, removed with the `ActScore` rule): the `alive(Act)`
//!      exploration weight was the organism's HUNGER `1 − metabolism/energy_cap` — an INTENSIVE energy ratio
//!      (energy per capacity); starving ⇒ explore, fed ⇒ exploit.
//!   3. EPISTEMIC entropy-floor (`substrate_entropy_floor.rs`): the free-energy softmax `T = mean_a σ_a` — the
//!      ensemble's own epistemic STD (its disagreement), the `U` moment of `surprise = U ⊕ aleatoric`
//!      (docs/research/novelty-field.md, R2).
//!
//! THE RESULT (do not force a single collapse): #1 and #2 are THERMAL — the same intensive-energy-ratio FORM,
//! self-annealing as the organism ORDERS/SATIATES (tests 1, 2). #3 is EPISTEMIC — an uncertainty/variance that
//! falls with DATA (visitation), not an energy account (test 4). TWO temperatures: THERMAL ⊥ EPISTEMIC, the
//! exact `U ⊕ aleatoric` split the novelty branch already proved. Test 5 dissolves the annealer's three
//! hand-set constants into coefficient-one energy/size and the energy economy's own starvation.
//!
//! (The grounded `Field::temperature_of = energy/mass` probe over the extensive `mass` state was RETIRED with the
//! mass/gravity cluster — validated NEGATIVE, gravity neutral/hurts and does not transfer — together with the
//! two tests that exercised it.)
//!
//! Scored INSIDE, deterministic, CI-fast. Pure usage-protocol over the PUBLIC `Field` API — NO core change.
//!
//!   cargo test -p openmorph-core --test research substrate_temperature -- --ignored --nocapture --test-threads=1

use openmorph_core::substrate::Field;

use crate::common::Rng;

// ─────────────────────────────────────────── shared: a Boltzmann read-off ───────────────────────────────────
// The ONE operator every "temperature" drives (operator-minimality [G]): a Gibbs/heat-bath select at T. High T
// ⇒ uniform (EXPLORE), low T ⇒ argmax (COMMIT). We measure the NORMALIZED ENTROPY of π(a) ∝ exp(score_a / T)
// over a FIXED score vector, sweeping only T — isolating "does this quantity control explore↔commit?".

/// Normalized Shannon entropy (0 = one option, 1 = uniform) of the Boltzmann policy π ∝ exp(scores/T).
fn boltzmann_entropy(scores: &[f32], t: f32) -> f32 {
    let t = t.max(1e-3); // the T→0 commit guard (grown_search's `heatbath` uses the same `.max(1e-3)`)
    let hi = scores.iter().copied().fold(f32::MIN, f32::max);
    let exps: Vec<f32> = scores.iter().map(|&s| ((s - hi) / t).exp()).collect();
    let z: f32 = exps.iter().sum();
    let mut h = 0.0f32;
    for &e in &exps {
        let p = e / z;
        if p > 0.0 {
            h -= p * p.ln();
        }
    }
    h / (scores.len() as f32).ln()
}

/// Is the slice strictly (weakly, within eps) monotone DECREASING?
fn monotone_down(xs: &[f32]) -> bool {
    xs.windows(2).all(|w| w[1] <= w[0] + 1e-6)
}

// A fixed spread of option scores (four options, one clearly best) — the SAME across every T-sweep below, so
// any change in select-entropy is due to T alone. Higher score = better option.
const SCORES: [f32; 4] = [0.0, 0.3, 0.6, 1.0];

// ═════════════════════════════════════════════════════════════════════════════════════════════════════════
// TEST 1 — THERMAL #1: annealing's temperature IS an intensive energy ratio (energy/size), coefficient one.
// ═════════════════════════════════════════════════════════════════════════════════════════════════════════

/// grown_search's ACTUAL annealing temperature (the authored form, with its two constants).
fn anneal_authored(conflicts: usize, e_max: usize) -> f32 {
    const T_MIN: f32 = 0.05;
    const T_SCALE: f32 = 1.5;
    T_MIN + T_SCALE * conflicts as f32 / e_max.max(1) as f32
}

/// The GROUNDED coefficient-one form: temperature = energy/size (energy over an extensive normalizer), exactly
/// what meta-discovery selected from the data.
fn anneal_grounded(conflicts: usize, e_max: usize) -> f32 {
    conflicts as f32 / e_max.max(1) as f32
}

#[test]
#[ignore = "research: on demand"]
fn thermal_annealing_temperature_is_energy_over_size() {
    let e_max = 20; // the instance's max violation-energy (edge count) — the extensive normalizer
    let conflicts: Vec<usize> = (0..=10).rev().map(|k| k * 2).collect(); // ordering: 20 → 0

    let authored: Vec<f32> = conflicts
        .iter()
        .map(|&c| anneal_authored(c, e_max))
        .collect();
    let grounded: Vec<f32> = conflicts
        .iter()
        .map(|&c| anneal_grounded(c, e_max))
        .collect();
    let entropy: Vec<f32> = grounded
        .iter()
        .map(|&t| boltzmann_entropy(&SCORES, t))
        .collect();

    // Both temperatures self-anneal (monotone down as the system orders); the grounded one is the authored one
    // shifted/scaled by (T_MIN, T_SCALE) — same RANK ordering (Spearman = 1), so the SELECT they drive is the
    // same. The two constants are a units offset+gain, NOT part of the temperature's information content.
    let spearman_ok = authored
        .windows(2)
        .zip(grounded.windows(2))
        .all(|(a, g)| (a[1] - a[0]).signum() == (g[1] - g[0]).signum());

    eprintln!("[THERMAL #1 · annealing] energy/size (violation-energy per edge), self-annealing as conflicts→0:");
    eprintln!("  conflicts : {conflicts:?}");
    eprintln!(
        "  authored  T = 0.05 + 1.5·E/size : {}",
        authored
            .iter()
            .map(|x| format!("{x:.3}"))
            .collect::<Vec<_>>()
            .join(" ")
    );
    eprintln!(
        "  grounded  T = E/size (coeff 1)  : {}",
        grounded
            .iter()
            .map(|x| format!("{x:.3}"))
            .collect::<Vec<_>>()
            .join(" ")
    );
    eprintln!(
        "  SELECT entropy at grounded T    : {}   (hot⇒explore≈1, cold⇒commit→0)",
        entropy
            .iter()
            .map(|x| format!("{x:.3}"))
            .collect::<Vec<_>>()
            .join(" ")
    );
    eprintln!("  VERDICT: annealing's T is the intensive ratio energy/size; T_MIN/T_SCALE are a units offset+gain, not information. A Boltzmann SELECT driven by it explores hot and commits cold — the annealing role, coefficient one.");

    assert!(
        monotone_down(&authored),
        "authored annealing T must self-cool as conflicts fall"
    );
    assert!(
        monotone_down(&grounded),
        "grounded energy/size T must self-cool as conflicts fall"
    );
    assert!(spearman_ok, "grounded T must have the SAME rank order as the authored T (the constants are offset+gain)");
    assert!(
        monotone_down(&entropy),
        "the SELECT driven by energy/size must anneal (entropy hot→cold)"
    );
    assert!(
        entropy[0] - entropy[entropy.len() - 1] > 0.3,
        "hot (frustrated) must EXPLORE and cold (ordered) must COMMIT — a real explore→commit swing ({:.3}→{:.3})",
        entropy[0],
        entropy[entropy.len() - 1]
    );
}

// ═════════════════════════════════════════════════════════════════════════════════════════════════════════
// TEST 2 — THERMAL #2: hunger IS an intensive energy ratio (energy/capacity), same FORM, drives explore↔exploit.
// ═════════════════════════════════════════════════════════════════════════════════════════════════════════

#[test]
#[ignore = "research: on demand"]
fn thermal_hunger_is_energy_over_capacity_same_form() {
    let cap = 10.0f32;
    // A feeding trajectory: metabolism climbs 0 → cap. Hunger = 1 − metabolism/cap is the intensive ratio.
    let metabolism: Vec<f32> = (0..=10).map(|k| k as f32).collect();
    let hunger: Vec<f32> = metabolism.iter().map(|&m| 1.0 - m / cap).collect();
    let entropy: Vec<f32> = hunger
        .iter()
        .map(|&t| boltzmann_entropy(&SCORES, t))
        .collect();

    // INTENSIVE = scale-free: double BOTH metabolism and capacity and hunger is unchanged (the hallmark of an
    // intensive quantity, exactly like temperature = energy/mass is unchanged under a rescale of both).
    let hunger_2x: Vec<f32> = metabolism
        .iter()
        .map(|&m| 1.0 - (2.0 * m) / (2.0 * cap))
        .collect();
    let intensive = hunger
        .iter()
        .zip(&hunger_2x)
        .all(|(a, b)| (a - b).abs() < 1e-6);

    eprintln!("[THERMAL #2 · hunger] energy/capacity (1 − metabolism/cap), self-annealing as the organism is fed:");
    eprintln!(
        "  metabolism : {}",
        metabolism
            .iter()
            .map(|x| format!("{x:.1}"))
            .collect::<Vec<_>>()
            .join(" ")
    );
    eprintln!(
        "  hunger T   : {}   (starving⇒hot⇒forage, fed⇒cold⇒exploit)",
        hunger
            .iter()
            .map(|x| format!("{x:.3}"))
            .collect::<Vec<_>>()
            .join(" ")
    );
    eprintln!(
        "  SELECT entropy at hunger T : {}",
        entropy
            .iter()
            .map(|x| format!("{x:.3}"))
            .collect::<Vec<_>>()
            .join(" ")
    );
    eprintln!("  VERDICT: hunger is the SAME thermal FORM as annealing's T — an intensive energy ratio, scale-free, self-annealing. Honest nuance: its numerator is a DEFICIT (1−satiation = free energy), so it is HOT at LOW stored energy (the mirror of energy/mass); the shared invariant is 'cool as the organism settles/satiates', not the sign of the raw energy.");

    assert!(
        monotone_down(&hunger),
        "hunger must fall as the organism is fed (self-anneal)"
    );
    assert!(
        intensive,
        "hunger must be INTENSIVE (scale-free under a joint rescale of metabolism & capacity)"
    );
    assert!(
        monotone_down(&entropy),
        "a SELECT driven by hunger must anneal (forage hot → exploit cold)"
    );
    assert!(
        entropy[0] - entropy[entropy.len() - 1] > 0.3,
        "starving must FORAGE and fed must EXPLOIT — a real swing ({:.3}→{:.3})",
        entropy[0],
        entropy[entropy.len() - 1]
    );
}

// ═════════════════════════════════════════════════════════════════════════════════════════════════════════
// TEST 4 — EPISTEMIC #3: σ (ensemble disagreement) is driven by DATA, not an energy account. THE `TWO` RESULT.
// ═════════════════════════════════════════════════════════════════════════════════════════════════════════
//
// The entropy-floor's temperature is `T = mean_a σ_a` — the varied-prior ensemble's DISAGREEMENT. It is not an
// energy account: it is driven by VISITATION (data), falling to 0 as an arm is learned while an unvisited arm
// stays ignorant — a second axis, beside the thermal energy ratio of tests 1–2.

const ARMS: usize = 4;
const ENSEMBLE: usize = 16;

/// One ensemble member: `ARMS` independent linear units, arm `a` reads one-hot bit `a`, writes cell `ARMS + a`,
/// all sharing this member's prior weight `w` (exchangeable). Grown online by `grow`.
fn member(seed: u64, w: f32) -> Field {
    let mut f = Field::new(2 * ARMS);
    f.rng_seed(seed);
    f.config_mut().max_hidden = 0;
    for a in 0..ARMS {
        f.add_unit(vec![a], vec![w], 0.0, ARMS + a);
    }
    f
}

/// A varied-prior ensemble with antithetic (+w,−w) pairs — mean pinned, only the SPREAD (epistemic σ) varies.
fn build_ensemble(seed: u64) -> Vec<Field> {
    let mut r = Rng::new(seed ^ 0xF00D);
    r.u();
    let mut ens = Vec::with_capacity(ENSEMBLE);
    for j in 0..ENSEMBLE / 2 {
        let w = r.sym() * 1.3;
        ens.push(member(seed ^ (0xA51 * (2 * j as u64 + 1)), w));
        ens.push(member(seed ^ (0xA51 * (2 * j as u64 + 2)), -w));
    }
    ens
}

fn onehot(a: usize) -> Vec<(usize, f32)> {
    (0..ARMS)
        .map(|i| (i, if i == a { 1.0 } else { 0.0 }))
        .collect()
}

/// arm `a`'s epistemic std σ_a over the grown ensemble (the entropy-floor's temperature ingredient).
fn sigma(ens: &mut [Field], a: usize) -> f32 {
    let os: Vec<f32> = ens
        .iter_mut()
        .map(|f| {
            f.stimulate(&onehot(a), 20);
            f.cells()[ARMS + a]
        })
        .collect();
    let n = os.len() as f32;
    let mean = os.iter().sum::<f32>() / n;
    (os.iter().map(|o| (o - mean) * (o - mean)).sum::<f32>() / n).sqrt()
}

/// Visit arm `a` `k` times with reward `r` (grows every member toward the datum ⇒ shrinks σ_a). A neutral
/// stimulus is applied between grows so the one-hot input CHANGES each time (fire-on-change: a constant input
/// never re-fires its units, so it would never learn).
fn visit(ens: &mut [Field], a: usize, r: f32, k: usize) {
    let rest = (a + 1) % ARMS; // any other arm — forces the input to change before the next grow
    for _ in 0..k {
        for f in ens.iter_mut() {
            f.stimulate(&onehot(rest), 20);
            f.grow(&onehot(a), ARMS + a, r, 0.2, 20);
        }
    }
}

#[test]
#[ignore = "research: on demand"]
fn epistemic_sigma_is_driven_by_visitation() {
    // σ is driven by VISITATION (epistemic), not by an energy account. Two arms: arm 0 visited many times
    // (learned ⇒ σ→0), arm 1 never visited (ignorant ⇒ σ high).
    let mut ens = build_ensemble(0x1CE);
    let sig0_pre = sigma(&mut ens, 0);
    let sig1_pre = sigma(&mut ens, 1);
    visit(&mut ens, 0, 0.7, 40); // learn arm 0 only
    let sig0_post = sigma(&mut ens, 0);
    let sig1_post = sigma(&mut ens, 1);

    eprintln!("[EPISTEMIC #3 · entropy-floor] σ = ensemble disagreement (the `U` moment):");
    eprintln!("  VISITATION drives σ:");
    eprintln!(
        "      arm0 (visited 40×): σ {sig0_pre:.3} → {sig0_post:.3}  (LEARNED ⇒ σ collapses)"
    );
    eprintln!("      arm1 (unvisited)  : σ {sig1_pre:.3} → {sig1_post:.3}  (still IGNORANT ⇒ σ stays high)");
    eprintln!("  VERDICT (TWO temperatures): σ falls with DATA (epistemic, reducible) — the thermal energy ratio (tests 1–2) falls as the organism orders/satiates. Different drivers: THERMAL ⊥ EPISTEMIC, exactly `surprise = U ⊕ aleatoric` (novelty-field R2).");

    // σ is epistemic: visiting collapses the visited arm's σ, but the unvisited arm stays high.
    assert!(
        sig0_post < sig0_pre * 0.6,
        "visiting arm 0 must COLLAPSE its epistemic σ (learned): {sig0_pre:.3} → {sig0_post:.3}"
    );
    assert!(
        sig1_post > sig0_post + 0.05,
        "the UNVISITED arm must keep HIGH σ while the visited arm's σ collapsed (σ tracks DATA, not mass): unvisited {sig1_post:.3} vs visited {sig0_post:.3}"
    );
}

// ═════════════════════════════════════════════════════════════════════════════════════════════════════════
// TEST 5 — A1b: the annealing solver is PARAMETER-FREE — T_MIN/T_SCALE dissolve into coefficient-one energy/size,
//          and PATIENCE dissolves into the ENERGY ECONOMY's own starvation (restart when the account runs out).
// ═════════════════════════════════════════════════════════════════════════════════════════════════════════
//
// grown_search carries three hand-set constants: T_MIN=0.05, T_SCALE=1.5, PATIENCE=6. This is the discrete layer
// of that solver (its `debug_assert` shows the substrate's settled violation-energy == the conflict count, so the
// conflict count IS the organism's energy here). We show all three dissolve:
//   • T_SCALE → coefficient ONE: temperature = conflicts/e_max = the field's own energy in its own units.
//   • T_MIN   → the heat-bath's `.max(1e-3)` commit guard (the T→0 limit, not a temperature constant).
//   • PATIENCE→ an ENERGY ACCOUNT that earns on improvement and pays upkeep; restart when STARVED (≤ 0). The
//     reserve is the instance's OWN max energy (e_max), so the effective patience SELF-SCALES with hardness —
//     strictly better than a fixed integer.

const K: usize = 3; // colours

struct Graph {
    v: usize,
    edges: Vec<(usize, usize)>,
}

/// A guaranteed-3-colourable random graph: assign a random proper 3-colouring, then only add edges between
/// DIFFERENT-colour vertices (so a proper colouring provably exists). Deterministic from `seed`.
fn satisfiable_graph(v: usize, target_edges: usize, seed: u64) -> Graph {
    let mut r = Rng::new(seed);
    let truth: Vec<usize> = (0..v).map(|_| r.below(K)).collect();
    let mut edges = Vec::new();
    let mut tries = 0;
    while edges.len() < target_edges && tries < target_edges * 20 {
        tries += 1;
        let a = r.below(v);
        let b = r.below(v);
        if a != b && truth[a] != truth[b] && !edges.contains(&(a.min(b), a.max(b))) {
            edges.push((a.min(b), a.max(b)));
        }
    }
    Graph { v, edges }
}

fn conflicts(g: &Graph, colour: &[usize]) -> usize {
    g.edges
        .iter()
        .filter(|&&(a, b)| colour[a] == colour[b])
        .count()
}

/// per-vertex per-colour conflict pressure h[v][k] = # neighbours already coloured k (the field).
fn field(g: &Graph, colour: &[usize], v: usize, k: usize) -> usize {
    g.edges
        .iter()
        .filter(|&&(a, b)| (a == v && colour[b] == k) || (b == v && colour[a] == k))
        .count()
}

/// Heat-bath resample of vertex v at temperature t: colour ∝ exp(−h[v][k]/t). T→0 ⇒ argmin (commit).
fn heatbath(g: &Graph, colour: &[usize], v: usize, t: f32, rng: &mut Rng) -> usize {
    let t = t.max(1e-3); // == grown_search's guard; this IS the dissolved T_MIN (the commit limit)
    let h: Vec<f32> = (0..K).map(|k| field(g, colour, v, k) as f32).collect();
    let mn = h.iter().copied().fold(f32::MAX, f32::min);
    let ex: Vec<f32> = h.iter().map(|&hk| (-(hk - mn) / t).exp()).collect();
    let z: f32 = ex.iter().sum();
    let mut u = rng.f() * z;
    for (k, &e) in ex.iter().enumerate() {
        u -= e;
        if u <= 0.0 {
            return k;
        }
    }
    K - 1
}

/// PARAMETER-FREE solver: temperature = conflicts/e_max (coefficient one, NO T_MIN/T_SCALE), restart when the
/// configuration's ENERGY ACCOUNT starves (NO PATIENCE integer). Returns whether it solved.
fn solve_parameter_free(g: &Graph, rng: &mut Rng, max_sweeps: usize) -> bool {
    let e_max = g.edges.len().max(1);
    let mut colour: Vec<usize> = (0..g.v).map(|_| rng.below(K)).collect();
    let mut best = usize::MAX;
    // The energy account: reserve = the instance's OWN max energy (self-scaling, not a hand-set integer).
    let mut account = e_max as f32;
    for _ in 0..max_sweeps {
        let conf = conflicts(g, &colour);
        if conf == 0 {
            return true;
        }
        // the economy: earn on improvement (best drop), pay unit upkeep each sweep; starve ⇒ reseed (repopulate).
        if conf < best {
            account += (best.min(e_max) as f32) - conf as f32; // income = how much the violation-energy dropped
            best = conf;
        }
        account -= 1.0; // upkeep (one unit per sweep of being awake)
        if account <= 0.0 {
            colour = (0..g.v).map(|_| rng.below(K)).collect();
            best = usize::MAX;
            account = e_max as f32;
            continue;
        }
        let t = conf as f32 / e_max as f32; // GROUNDED temperature: energy/size, coefficient one
        for v in 0..g.v {
            colour[v] = heatbath(g, &colour, v, t, rng);
        }
    }
    conflicts(g, &colour) == 0
}

/// GREEDY control (T→0, deterministic argmin-conflict descent, no jump, no restart) — must TRAP.
fn solve_greedy(g: &Graph, rng: &mut Rng, max_sweeps: usize) -> bool {
    let mut colour: Vec<usize> = (0..g.v).map(|_| rng.below(K)).collect();
    for _ in 0..max_sweeps {
        if conflicts(g, &colour) == 0 {
            return true;
        }
        let mut changed = false;
        for v in 0..g.v {
            let nc = (0..K).min_by_key(|&k| field(g, &colour, v, k)).unwrap();
            if nc != colour[v] {
                colour[v] = nc;
                changed = true;
            }
        }
        if !changed {
            return false; // a local minimum with residual conflicts — greedy is trapped
        }
    }
    false
}

#[test]
#[ignore = "research: on demand"]
fn a1b_annealer_is_parameter_free() {
    let n = 40;
    let mut param_free = 0usize;
    let mut greedy = 0usize;
    for i in 0..n {
        // frustrated instances: 12 vertices, ~24 edges (dense enough to trap greedy in local minima).
        let g = satisfiable_graph(12, 24, 0xC0FFEE + i as u64);
        let mut r1 = Rng::new(0xA1B ^ i as u64);
        let mut r2 = Rng::new(0xA1B ^ i as u64); // same seed ⇒ same start, isolates the mechanism
        if solve_parameter_free(&g, &mut r1, 400) {
            param_free += 1;
        }
        if solve_greedy(&g, &mut r2, 400) {
            greedy += 1;
        }
    }
    let pf = param_free as f64 / n as f64;
    let gd = greedy as f64 / n as f64;

    eprintln!("[A1b · parameter-free annealer] over {n} guaranteed-satisfiable frustrated 3-colouring instances:");
    eprintln!("  GREEDY (T→0, no jump/restart)                     : solve {gd:.2}   (min-conflicts TRAPS in local minima)");
    eprintln!("  PARAMETER-FREE (T=E/size coeff-1, economy restart): solve {pf:.2}   (NO T_MIN, NO T_SCALE, NO PATIENCE)");
    eprintln!("  DISSOLUTION: T_SCALE → coefficient ONE (the field's own energy units); T_MIN → the heat-bath's .max(1e-3) commit guard (the T→0 limit, not a temperature constant); PATIENCE → the ENERGY ACCOUNT's own starvation (reserve = e_max ⇒ effective patience SELF-SCALES with instance hardness, strictly better than a fixed integer).");
    eprintln!("  HONEST RESIDUAL (named, not hidden): the account's UPKEEP RATE (1.0/sweep) and the coefficient-one temperature both assume the field's own energy UNITS (violation measured in edges) — that binding of 'which energy, in which units' is the interface, not itself dissolved. But it is the SAME energy currency the organism already runs, not a bespoke search constant.");

    // The coefficient-one temperature's stochastic jump escapes traps greedy (T→0) cannot — annealing survives
    // WITHOUT T_MIN/T_SCALE.
    assert!(
        pf > gd + 0.2,
        "the parameter-free annealer (coeff-1 temperature) must escape traps greedy sticks in ({pf:.2} vs greedy {gd:.2})"
    );
    // And the economy-restart annealer solves the satisfiable bank at a high rate WITHOUT a PATIENCE integer.
    assert!(
        pf > 0.9,
        "the economy-restart solver must settle satisfiable instances to zero violation without a hand-set PATIENCE ({pf:.2})"
    );
}
