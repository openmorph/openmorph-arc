// SPDX-License-Identifier: MIT-0
//! THE UNIFIED FREE-ENERGY (DYNA) ACTOR — the canonical reference loop, as a usage protocol over the
//! `Field` public API. This is the actor validated by `tests/substrate_free_energy_{actor,policy,dyna}`,
//! extracted so a HARNESS or BRIDGE (e.g. an ARC-3 MORPH stdio agent) can MIRROR the genuine loop rather
//! than re-derive a hand-written proxy over raw frames — a proxy is exactly what reproduces "no better
//! than random" on ARC-3, because crude novelty + premature reward-commitment IS random. Run:
//!
//!   cargo run -p openmorph-core --example free_energy_actor
//!
//! THE FUNCTIONAL (one line, no ε/β/TD/novelty). `G(π) = Σ_t [ (μ_t − s*)² − σ_t² ]` — pragmatic minus
//! epistemic, SAME currency (o²), coefficient ONE. `s*` is the PREFERRED observation (reward = a prior,
//! not a scalar signal); `μ(s)`,`σ²(s)` are the ensemble MEAN and DISAGREEMENT of the grown world model
//! at abstract state `s` (the epistemic term is literally the model's own uncertainty).
//!
//! THE INTERLEAVE (explore + plan in ONE loop). The `−σ²` term folds into PLANNING by choosing the goal
//! `relax_values` spreads its cost-to-go from:
//!   goal = argmin_s g(s) = (μ(s) − s*)² − σ²(s)
//!   * model IGNORANT (σ² large on an un-mapped region) ⇒ the frontier OUTSCORES the local pragmatic
//!     optimum ⇒ the goal is the frontier ⇒ `relax_values` PLANS a path across the valley to it.
//!     Explore-to-find BECOMES plan-to-the-frontier.
//!   * model CERTAIN (σ²→0, self-annealing) ⇒ g collapses to (μ−s*)² ⇒ the discovered preferred state is
//!     the goal ⇒ plan-to-reach-and-hold.
//!
//! Explore→exploit is `argmin g` following σ² down to zero — no switch, no schedule.
//!
//! HOW A BRIDGE PLUGS IN GROWN PERCEPTION (the only missing piece on real games):
//!   1. PERCEIVE — a grown front-end (openmorph-perception + `seed_coord` / the predictive latent) maps a
//!      raw 64×64 frame to an abstract STATE index `s` (a stable, distance-preserving key). THIS is the
//!      open wall on ARC-3 — the acting law below is done; it needs a state code that de-aliases frames.
//!   2. MODEL — grow an ensemble of world models predicting the preferred observation per state (`belief`
//!      below) and an action-conditional transition model action→next-state (the `adjacency`).
//!   3. ACT — call [`free_energy_action`] each step. That is the whole actor.

use openmorph_core::substrate::Field;

// ─────────────────────────────────────────────────────────────────────────────────────────────────────
// THE ACTOR — decoupled from any world. Mirror THIS.
// ─────────────────────────────────────────────────────────────────────────────────────────────────────

/// Pick an action by the free-energy Dyna rule. `mu`/`var` are the ensemble belief per abstract state,
/// `s_star` the preferred observation, `adjacency[s][a]` the imagined next state from `s` under action `a`
/// (the grown action-conditional transition model), `cur` the current state. `epi` weights the epistemic
/// term: `1.0` = the unified actor (explore AND plan), `0.0` = the pure-plan ablation (no exploration).
///
/// Returns the index of the chosen action. The ONLY difference from pure planning is the `−epi·σ²` term
/// in the goal — that single term is what turns a planner into an explorer that plans toward the unknown.
pub fn free_energy_action(
    mu: &[f32],
    var: &[f32],
    s_star: f32,
    adjacency: &[Vec<usize>],
    cur: usize,
    epi: f32,
) -> usize {
    let n = mu.len();
    // 1. the free-energy goal: the state minimising g(s) = (μ−s*)² − epi·σ².
    let goal = (0..n)
        .min_by(|&x, &y| {
            let gx = (mu[x] - s_star).powi(2) - epi * var[x];
            let gy = (mu[y] - s_star).powi(2) - epi * var[y];
            gx.total_cmp(&gy)
        })
        .unwrap();
    // 2. value-iterate the cost-to-go to that goal over the IMAGINED dynamics (planning depth).
    let mut planner = Field::new(n);
    planner.clear_values();
    planner.prefer(goal, 0.0);
    planner.relax_values(adjacency, 1.0, 0);
    // 3. descend it: the action whose imagined next state is closest-to-go.
    let nexts = &adjacency[cur];
    (0..nexts.len())
        .min_by(|&a, &b| {
            planner
                .value_of(nexts[a])
                .total_cmp(&planner.value_of(nexts[b]))
        })
        .unwrap()
}

/// The ensemble's MEAN μ(s) and VARIANCE σ²(s) of the predicted preferred-observation at every state — the
/// pragmatic signal and the epistemic signal, read straight off a set of grown world models. A never-
/// visited state keeps the ensemble's prior DISAGREEMENT (broad σ²) = the frontier the `−σ²` term seeks.
pub fn belief(ens: &mut [Field], n_states: usize, out_base: usize) -> (Vec<f32>, Vec<f32>) {
    let (mut mu, mut var) = (vec![0.0f32; n_states], vec![0.0f32; n_states]);
    for (s, (m, v)) in mu.iter_mut().zip(var.iter_mut()).enumerate() {
        let os: Vec<f32> = ens
            .iter_mut()
            .map(|f| {
                f.stimulate(&onehot(s, n_states), 20);
                f.cells()[out_base + s]
            })
            .collect();
        let k = os.len() as f32;
        *m = os.iter().sum::<f32>() / k;
        *v = os.iter().map(|o| (o - *m) * (o - *m)).sum::<f32>() / k;
    }
    (mu, var)
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────────
// A COMPACT RUNNABLE WORLD — a corridor needing BOTH: an unknown goal ACROSS a known valley. (The rigorous
// multi-seed proof is `tests/substrate_free_energy_dyna`; this is the runnable reference demo.)
// ─────────────────────────────────────────────────────────────────────────────────────────────────────

const N: usize = 16; // corridor states 0..N (same as the proven test)
const SSTAR: f32 = 0.90; // the preferred observation (the global peak's height)
const KNOWN_END: usize = 9; // "prior life" maps 0..=9 (near + valley); 10..N is unknown = the goal region
const ENS: usize = 24; // varied-prior world models; their disagreement is σ²

// height h(s): LOCAL peak at s=2, a long known VALLEY (s=5..14), the GLOBAL peak a NEEDLE at s=15 (≈ s*).
const H: [f32; N] = [
    0.30, 0.45, 0.52, 0.38, 0.24, 0.14, 0.09, 0.07, 0.07, 0.08, 0.10, 0.11, 0.10, 0.12, 0.11, 0.90,
];
const DELTAS: [i32; 2] = [-1, 1]; // action 0 = left, 1 = right

/// xorshift64 — the same generator quality the proven test uses; examples can't share `tests/common`.
/// Research stays byte-replayable (a seeded PRNG, never a wall-clock).
struct Rng(u64);
impl Rng {
    fn new(s: u64) -> Self {
        Rng(s | 1)
    }
    fn u(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn sym(&mut self) -> f32 {
        (self.u() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0 // [-1, 1)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.u() % n as u64) as usize
    }
}

fn onehot(s: usize, n: usize) -> Vec<(usize, f32)> {
    (0..n).map(|i| (i, (i == s) as u8 as f32)).collect()
}
fn clamp_s(s: i32) -> usize {
    s.clamp(0, N as i32 - 1) as usize
}

/// One ensemble member: N independent linear units, unit s reads one-hot bit s and writes cell N+s. Members
/// share a prior bias and differ only in weight (antithetic ±w pairs pin the mean, vary the disagreement).
fn model(w: f32) -> Field {
    let mut f = Field::new(N);
    f.ensure_cells(2 * N);
    f.config_mut().max_hidden = 0;
    for s in 0..N {
        f.add_unit(vec![s], vec![w], 0.406, N + s);
    }
    f
}
fn ensemble(seed: u64) -> Vec<Field> {
    let mut r = Rng::new(seed.wrapping_mul(0x9E3779B97F4A7C15) ^ 0xF00D);
    r.u();
    (0..ENS / 2)
        .flat_map(|_| {
            let w = r.sym() * 1.7;
            [model(w), model(-w)]
        })
        .collect()
}
/// Grow every member toward the observed height at s (the model is GROWN online, never given).
fn observe_state(ens: &mut [Field], s: usize) {
    for f in ens.iter_mut() {
        for _ in 0..3 {
            f.grow(&onehot(s, N), N + s, H[s], 0.2, 20);
        }
    }
}
/// A brief random walk over the KNOWN region (0..=KNOWN_END): σ²→0 there, broad beyond = the unknown goal.
fn prior_life(ens: &mut [Field], seed: u64) {
    let mut r = Rng::new(seed ^ 0xB1A5);
    let mut s: i32 = 0;
    for _ in 0..600 {
        observe_state(ens, s as usize);
        s = (s + DELTAS[r.below(2)]).clamp(0, KNOWN_END as i32);
    }
}

/// Run the interleave loop for `epi` (1.0 unified, 0.0 pure-plan); return (reached-the-goal, terminal dwell).
fn run(seed: u64, epi: f32) -> (bool, f32) {
    let mut ens = ensemble(seed);
    prior_life(&mut ens, seed);
    // the bridge LEARNS this action→next-state map online (see substrate_free_energy_policy); here we use
    // the true corridor map to keep the reference focused on the ACTOR, not the (separately-proven) model.
    let adj: Vec<Vec<usize>> = (0..N)
        .map(|s| DELTAS.iter().map(|&d| clamp_s(s as i32 + d)).collect())
        .collect();
    let mut s: usize = 0;
    let mut traj = Vec::new();
    for _ in 0..96 {
        observe_state(&mut ens, s); // interleave: grow the model at the visited state
        let (mu, var) = belief(&mut ens, N, N);
        let a = free_energy_action(&mu, &var, SSTAR, &adj, s, epi);
        s = clamp_s(s as i32 + DELTAS[a]);
        traj.push(s);
    }
    let reached = traj.contains(&(N - 1));
    let dwell = traj[traj.len() - N..]
        .iter()
        .map(|&s| (H[s] - SSTAR).abs())
        .sum::<f32>()
        / N as f32;
    (reached, dwell)
}

fn main() {
    println!("Unified free-energy (Dyna) actor — explore + plan in one loop, G(π)=Σ[(μ−s*)²−σ²]\n");
    let seeds = 8u64;
    let (mut uni_r, mut plan_r, mut uni_d, mut plan_d) = (0, 0, 0.0f32, 0.0f32);
    for seed in 0..seeds {
        let (ur, ud) = run(0x100 + seed, 1.0); // unified: pragmatic − epistemic
        let (pr, pd) = run(0x100 + seed, 0.0); // pure-plan: no epistemic term
        uni_r += ur as usize;
        plan_r += pr as usize;
        uni_d += ud;
        plan_d += pd;
    }
    let sf = seeds as f32;
    println!(
        "world: local peak s=2, KNOWN valley s=5..14, UNKNOWN needle goal s=15 (across the valley)"
    );
    println!("  pure-plan (no −σ²) : reached {plan_r}/{seeds}  dwell {:.3}  <- plans to best KNOWN, never FINDS the goal", plan_d / sf);
    println!("  unified   (−σ² goal): reached {uni_r}/{seeds}  dwell {:.3}  <- explores to the frontier AND plans across the valley", uni_d / sf);
    println!("\nThe only difference is the −σ² term in the goal. Reward never entered; exploration self-annealed as σ→0.");
    println!("To move real ARC-3 numbers, put a GROWN perceptual state-code under `belief`'s state index (the open wall).");
    assert!(
        uni_r >= 7 && plan_r == 0,
        "reference sanity: unified reaches, pure-plan does not"
    );
}
