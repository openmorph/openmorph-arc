// SPDX-License-Identifier: MIT-0
//! THE GUARANTEED EXPLORATION FLOOR — the free-energy actor must NEVER underperform random on a
//! RANDOM-COMPLETE game (every action equireward, so uniform random already scores the max and a learned
//! actor can only lose by collapsing its action entropy below random). MorphBot measured exactly this
//! failure on cw01/lo01/lo05/rp04: a learned actor that argmax-collapses nets NEGATIVE against random.
//! MorphBot's "optimism" patch also netted negative; this is the PRINCIPLED version.
//!
//! THE FLOOR ALREADY EXISTS IN THE FUNCTIONAL — this test GUARANTEES it. The unified expected-free-energy
//! functional is  G(a) = (μ_a − s*)²  −  σ_a²  (pragmatic minus epistemic, `substrate_free_energy_actor`).
//! The SHIPPED actor picks `argmin G` — a HARD, deterministic choice. That is the bug on a random-complete
//! game: every arm is equireward, so as the grown ensemble converges the epistemic variance σ_a² → 0 on
//! EVERY arm and the pragmatic term (μ_a − s*)² is IDENTICAL on every arm ⇒ G ties ⇒ argmin falls to the
//! tie-break (arm 0) FOREVER ⇒ action entropy → 0 ⇒ it underperforms uniform random on any game whose reward
//! is coverage/diversity. The FIX is not an authored ε-random floor bolted on; it is to read the SAME G
//! through a Boltzmann policy instead of a hard argmin:
//!
//!     π(a) ∝ exp( −G(a) / T ),   T = mean_a σ_a   (the epistemic std — the actor's own uncertainty)
//!
//! Two things then hold FOR FREE, with no ε:
//!   (1) FLOOR — on a random-complete game G(a) is EQUAL across arms (equireward ⇒ equal μ, equal σ), so the
//!       softmax logits are equal ⇒ π is UNIFORM ⇒ action entropy ≥ random, for ANY T. The floor is a
//!       consequence of G-equality, not of a temperature constant. (And spurious finite-ensemble asymmetry
//!       ΔG is O(σ²) while T is O(σ), so ΔG/T → 0 as the ensemble anneals — the spread never sharpens.)
//!   (2) EXPLOITATION — where an arm has a REAL edge, its G is genuinely lower; as the ensemble converges the
//!       temperature T = σ self-ANNEALS toward 0, so the softmax sharpens onto that arm ⇒ it EXPLOITS. The
//!       temperature is the epistemic uncertainty: broad while ignorant (explore/floor), sharp once certain
//!       (exploit). Nothing is a hand-set exploration rate.
//!
//! THREE actors, ONE grown ensemble & world model (identical `grow`, identical priors) — they differ ONLY in
//! the action rule, so any gap isolates the rule:
//!   * VALUE-GREEDY   argmin (μ−s*)²           — epistemic term DELETED; the naive collapse MorphBot saw.
//!   * ARGMIN-G       argmin [(μ−s*)² − σ²]     — the SHIPPED free-energy actor: explores via the epistemic
//!     term early, but COLLAPSES to the tie-break once σ→0 (the converged tail is where it nets negative).
//!   * SOFTMAX-FLOOR  sample π ∝ exp(−G/T)      — the guaranteed floor: uniform when no edge, sharp when edge.
//!
//! Scored INSIDE, net-vs-random, held-out (2nd half), seeded, chance-anchored:
//!   * RANDOM-COMPLETE task (all arms equireward): score = normalized action ENTROPY over the 2nd half
//!     (random = 1.0 by construction). The floor actor must NET ≥ 0 vs random (not collapse); the greedy and
//!     the shipped argmin-G actor must NET < 0 (they collapse) — the failure the floor fixes.
//!   * EDGE-PRESENT task (one arm is the peak): score = mean reward over the 2nd half. The SAME floor actor
//!     must NET > 0 vs random (exploit the peak) — the floor must not destroy exploitation.
//!
//! AUTHORING LEDGER (truth-not-trophy): the floor is the softmax temperature T = mean_a σ_a — the ensemble's
//! own epistemic std, in the functional's own units. There is NO ε, NO fixed exploration rate, NO novelty
//! bonus. The one non-behavioral constant is a numeric guard `T.max(1e-4)` (avoid divide-by-zero once σ
//! anneals to exactly 0); it injects NO randomness and does NOT set a floor on entropy — the entropy floor
//! comes from G-equality, provable at any T. Softmax uses the functional's natural o² scale (coefficient one),
//! same as the shipped `free_energy_score`. Prefer-no-core-change: this actor composes the public `Field` API
//! (`grow`/`stimulate`/`cells`) — the SAME construction as `substrate_free_energy_actor`; promoting the floor
//! into the built-in `alive(Act)` actor would be a small byte-identical-off Config lever (a temperature), not
//! required here.
//!
//!   cargo test -p openmorph-core --test proofs substrate_entropy_floor -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;

use crate::common::Rng;

const ARMS: usize = 8;
const SSTAR: f32 = 0.9; // the preferred observation (kept in the tanh genome's range)
const ENSEMBLE: usize = 24; // varied-prior world models; their DISAGREEMENT is the epistemic variance
const HORIZON: usize = 1000;
const TAIL: usize = 600; // score the held-out 2nd-half steady state (the converged tail is where it collapses)

/// EDGE-PRESENT payoffs: arm 0 DEAD (0.0), arms 1..=6 a middling DECOY (0.45), arm 7 the true PEAK (0.9 ≈ s*).
fn payoff_edge(a: usize) -> f32 {
    match a {
        7 => 0.90,
        1..=6 => 0.45,
        _ => 0.0,
    }
}

/// RANDOM-COMPLETE payoffs: EVERY arm equireward (a flat 0.5). No arm is better than another — the only thing
/// a policy can do is choose HOW to spread its actions, so uniform random already scores the maximum coverage.
fn payoff_flat(_a: usize) -> f32 {
    0.5
}

/// One-hot arm encoding over input cells `0..ARMS` (every cell supplied so the input CHANGES as the arm
/// switches — fire-on-change: a constant input never drives its units).
fn onehot(a: usize) -> Vec<(usize, f32)> {
    (0..ARMS)
        .map(|i| (i, if i == a { 1.0 } else { 0.0 }))
        .collect()
}

/// arm `a`'s predicted-payoff cell — each arm an INDEPENDENT unit (cell `ARMS + a`) so a sampled arm's estimate
/// converges WITHOUT dragging the others (a shared readout would couple them and collapse the epistemic variance
/// of the unvisited arms).
fn arm_cell(a: usize) -> usize {
    ARMS + a
}

/// One ensemble member: `ARMS` INDEPENDENT linear units (max_hidden = 0 — a bandit is linear); arm `a`'s unit
/// reads one-hot bit `a`, writes cell `ARMS + a`. All arms share this member's prior weight `w` (an EXCHANGEABLE
/// prior — before data the arms are indistinguishable). Grown online by `grow`.
fn model(seed: u64, w: f32) -> Field {
    let mut f = Field::new(ARMS);
    f.ensure_cells(2 * ARMS);
    f.rng_seed(seed);
    f.config_mut().max_hidden = 0;
    for a in 0..ARMS {
        f.add_unit(vec![a], vec![w], 0.406, arm_cell(a)); // bias centres pred near the decoy
    }
    f
}

/// The varied-prior ensemble with ANTITHETIC weight pairs (`+w`, `−w`): each pair straddles the centre, so the
/// ensemble MEAN is pinned near `tanh(0.406) ≈ 0.38` on EVERY seed — only the spread (the epistemic variance)
/// varies with the seed (same trick as `substrate_free_energy_actor`).
fn ensemble(seed: u64) -> Vec<Field> {
    let mut r = Rng::new(seed.wrapping_mul(0x9E3779B97F4A7C15) ^ 0xF00D);
    r.u();
    let mut ens = Vec::with_capacity(ENSEMBLE);
    for j in 0..ENSEMBLE / 2 {
        let w = r.sym() * 1.3;
        ens.push(model(seed ^ (0xA51 * (2 * j as u64 + 1)), w));
        ens.push(model(seed ^ (0xA51 * (2 * j as u64 + 2)), -w));
    }
    ens
}

fn pred(f: &mut Field, a: usize) -> f32 {
    f.stimulate(&onehot(a), 20);
    f.cells()[arm_cell(a)]
}

/// arm `a`'s ensemble moments (μ, σ²) over the grown world model — the substrate's OWN disagreement.
fn moments(ens: &mut [Field], a: usize) -> (f32, f32) {
    let os: Vec<f32> = ens.iter_mut().map(|f| pred(f, a)).collect();
    let n = os.len() as f32;
    let mean = os.iter().sum::<f32>() / n;
    let var = os.iter().map(|o| (o - mean) * (o - mean)).sum::<f32>() / n;
    (mean, var)
}

/// The action rule.
#[derive(Clone, Copy, PartialEq)]
enum Actor {
    Greedy,      // argmin (μ−s*)²           — epistemic term DELETED (the naive collapse)
    ArgminG,     // argmin [(μ−s*)² − σ²]     — the SHIPPED free-energy actor (hard argmin)
    SoftmaxG,    // sample π ∝ exp(−G/T)      — the guaranteed floor (T = mean σ)
    RandomFloor, // uniform sample            — the reference the floor must match
}

/// Choose an arm. All three learned actors read the SAME grown G; only the READ-OFF differs. `SoftmaxG` and
/// `RandomFloor` consume the seeded PRNG (deterministic, replayable).
fn choose(actor: Actor, ens: &mut [Field], rng: &mut Rng) -> usize {
    if actor == Actor::RandomFloor {
        return rng.below(ARMS);
    }
    // grown moments per arm
    let m: Vec<(f32, f32)> = (0..ARMS).map(|a| moments(ens, a)).collect();
    let g: Vec<f32> = m
        .iter()
        .map(|&(mean, var)| {
            let pragmatic = (mean - SSTAR) * (mean - SSTAR);
            match actor {
                Actor::Greedy => pragmatic, // epistemic term deleted
                _ => pragmatic - var,       // full expected free energy G
            }
        })
        .collect();

    match actor {
        Actor::Greedy | Actor::ArgminG => (0..ARMS).min_by(|&x, &y| g[x].total_cmp(&g[y])).unwrap(),
        Actor::SoftmaxG => {
            // temperature = the ensemble's OWN epistemic std (mean over arms). Broad while ignorant (σ large ⇒
            // explore/floor), sharp once certain (σ → 0 ⇒ exploit). The `.max(1e-4)` is a numeric divide-by-zero
            // guard only — it sets NO entropy floor (equal G ⇒ uniform at any T).
            let t = (m.iter().map(|&(_, v)| v.sqrt()).sum::<f32>() / ARMS as f32).max(1e-4);
            let logits: Vec<f32> = g.iter().map(|&gv| -gv / t).collect();
            let hi = logits.iter().copied().fold(f32::MIN, f32::max);
            let exps: Vec<f32> = logits.iter().map(|&l| (l - hi).exp()).collect();
            let z: f32 = exps.iter().sum();
            let mut u = rng.f() * z;
            for (a, &e) in exps.iter().enumerate() {
                u -= e;
                if u <= 0.0 {
                    return a;
                }
            }
            ARMS - 1
        }
        Actor::RandomFloor => unreachable!(),
    }
}

/// Normalized Shannon entropy of an action histogram (0 = one arm, 1 = uniform = random).
fn norm_entropy(hist: &[usize]) -> f32 {
    let total: usize = hist.iter().sum();
    if total == 0 {
        return 0.0;
    }
    let mut h = 0.0f32;
    for &c in hist {
        if c > 0 {
            let p = c as f32 / total as f32;
            h -= p * p.ln();
        }
    }
    h / (ARMS as f32).ln()
}

/// Run one actor on one world. Returns (mean 2nd-half REWARD, 2nd-half action ENTROPY). Every learned actor
/// grows the SAME varied-prior ensemble online from the SAME real payoffs; they differ ONLY in `choose`.
fn run(actor: Actor, payoff: fn(usize) -> f32, seed: u64) -> (f32, f32) {
    let mut ens = ensemble(seed);
    let mut rng = Rng::new(seed ^ 0x5EED_F100_0FEE);
    let mut hist = [0usize; ARMS];
    let (mut rew, mut cnt) = (0.0f32, 0usize);
    for step in 0..HORIZON {
        let a = choose(actor, &mut ens, &mut rng);
        let r = payoff(a);
        for f in ens.iter_mut() {
            for _ in 0..2 {
                f.grow(&onehot(a), arm_cell(a), r, 0.15, 20);
            }
        }
        if step >= TAIL {
            rew += r;
            cnt += 1;
            hist[a] += 1;
        }
    }
    (rew / cnt as f32, norm_entropy(&hist))
}

fn avg(actor: Actor, payoff: fn(usize) -> f32, seeds: u64) -> (f32, f32) {
    let (mut r, mut h) = (0.0f32, 0.0f32);
    for s in 0..seeds {
        let (ri, hi) = run(actor, payoff, 0x1CE + s);
        r += ri;
        h += hi;
    }
    (r / seeds as f32, h / seeds as f32)
}

#[test]
fn the_epistemic_floor_guarantees_action_entropy_at_least_random_with_no_epsilon() {
    let seeds = 6;

    // ── RANDOM-COMPLETE: every arm equireward. Score = normalized action entropy over the held-out 2nd half. ──
    let (_, h_rand) = avg(Actor::RandomFloor, payoff_flat, seeds);
    let (_, h_greedy) = avg(Actor::Greedy, payoff_flat, seeds);
    let (_, h_argmin) = avg(Actor::ArgminG, payoff_flat, seeds);
    let (_, h_soft) = avg(Actor::SoftmaxG, payoff_flat, seeds);

    // ── EDGE-PRESENT: arm 7 is the peak. Score = mean reward over the held-out 2nd half. ──
    let (r_rand, _) = avg(Actor::RandomFloor, payoff_edge, seeds);
    let (r_soft, h_soft_edge) = avg(Actor::SoftmaxG, payoff_edge, seeds);
    let (r_argmin, _) = avg(Actor::ArgminG, payoff_edge, seeds);

    eprintln!("[ENTROPY FLOOR] RANDOM-COMPLETE (all arms equireward; random already scores max) — normalized action entropy over the held-out 2nd half (1.0 = uniform = random):");
    eprintln!("[ENTROPY FLOOR]   random (reference)         : {h_rand:.3}");
    eprintln!("[ENTROPY FLOOR]   value-greedy argmin(μ−s*)² : {h_greedy:.3}   net vs random {:+.3}  <- collapses (epistemic term deleted)", h_greedy - h_rand);
    eprintln!("[ENTROPY FLOOR]   SHIPPED argmin G           : {h_argmin:.3}   net vs random {:+.3}  <- collapses in the converged tail (σ→0, G ties, tie-break)", h_argmin - h_rand);
    eprintln!("[ENTROPY FLOOR]   SOFTMAX floor  π∝exp(−G/T) : {h_soft:.3}   net vs random {:+.3}  <- matches random (equal G ⇒ uniform, ANY T)", h_soft - h_rand);
    eprintln!("[ENTROPY FLOOR] EDGE-PRESENT (arm 7 is the peak) — mean reward over the held-out 2nd half (random = mean payoff, peak = 0.90):");
    eprintln!("[ENTROPY FLOOR]   random (reference)         : {r_rand:.3}");
    eprintln!(
        "[ENTROPY FLOOR]   SHIPPED argmin G           : {r_argmin:.3}   net vs random {:+.3}",
        r_argmin - r_rand
    );
    eprintln!("[ENTROPY FLOOR]   SOFTMAX floor              : {r_soft:.3}   net vs random {:+.3}  (action entropy fell to {h_soft_edge:.3} — it CONCENTRATED to exploit)", r_soft - r_rand);
    eprintln!("[ENTROPY FLOOR] VERDICT: the softmax floor reads the SAME grown G as the shipped argmin actor, but π∝exp(−G/T) with T=mean σ is UNIFORM when arms tie (random-complete ⇒ net ≥ 0 vs random) and SHARP when an arm has a real edge (T=σ self-anneals ⇒ exploits ⇒ net > 0). No ε, no fixed exploration rate — the floor is G-equality, the sharpness is the epistemic std. The value-greedy and shipped-argmin actors both net NEGATIVE on the random-complete game (they collapse the action distribution below random) — the failure the floor fixes.");

    // (1) FLOOR — the softmax actor must NOT underperform random on the random-complete game (net ≥ 0).
    assert!(
        h_soft >= h_rand - 0.03,
        "the softmax floor must keep action entropy ≥ random on a random-complete game (soft {h_soft:.3} vs random {h_rand:.3})"
    );
    // (2) The failure it fixes — the hard-argmin actors DO collapse below random here.
    assert!(
        h_greedy < h_rand - 0.3,
        "the value-greedy actor must collapse below random (greedy {h_greedy:.3} vs random {h_rand:.3})"
    );
    assert!(
        h_argmin < h_rand - 0.3,
        "the SHIPPED argmin-G actor must collapse below random in the converged tail (argmin {h_argmin:.3} vs random {h_rand:.3})"
    );
    // (3) EXPLOITATION — the SAME softmax actor must still beat random where an edge exists (floor ≠ blindness).
    assert!(
        r_soft > r_rand + 0.05,
        "the softmax floor must still EXPLOIT a real edge (soft reward {r_soft:.3} vs random {r_rand:.3})"
    );
    // and it exploited by CONCENTRATING — the floor is adaptive, not a fixed ε (entropy fell on the edge task).
    assert!(
        h_soft_edge < h_soft - 0.15,
        "on the edge task the floor must ADAPT DOWN (entropy {h_soft_edge:.3} < its random-complete entropy {h_soft:.3}) — it concentrates to exploit, so the floor is not a fixed exploration rate"
    );
}
