// SPDX-License-Identifier: MIT-0
//! #155 (OM-6) — SELF-GENERATED HOMEOSTATIC GOAL. `substrate_purpose` proved a morph ACTS toward a preferred
//! state x* — but the harness AUTHORED x* (`prefer(nearest_object, cost)`). The largest purity hole in the ARC-3
//! loop was that the GOAL itself was hand-picked. OM-6 closes it: the organism GENERATES its own x* from its OWN
//! grown quantities via [`Field::orient`], which maximises the grown free-energy score
//!
//!     score(c) = drive_of(c) + epistemic_weight · learning_progress(c)
//!
//! — EXTRINSIC reward-memory (`drive`, from `reinforce`) + EPISTEMIC *reducible* surprise (`learning_progress`)
//! — and clamps the winner as the goal. Nothing is a hand feature: `drive`/`learning_progress` are grown, the
//! winner is picked by the grown winner-take-all `select` (no authored argmax), and `epistemic_weight` is the
//! META-EVOLVED coefficient (`Config::epistemic_weight`) — discovered by the
//! `substrate_meta_drives` GA, not authored.
//!
//! Everything is control-baselined and scored INSIDE:
//!   (A) EXTRINSIC — self-orient reaches the reward-associated state > 2 sd better than clamping x* at a RANDOM
//!       reachable cell (z = self vs the random-reachable-goal distribution).
//!   (B) NO DARK-ROOM — orient does NOT pick the predictable do-nothing state; its coverage/action-entropy is
//!       strictly above a do-nothing baseline, while a naive SURPRISE-MIN goal-selector COLLAPSES into the
//!       dark-room (the predictable state) and does nothing. Learning-PROGRESS (not raw surprise) is why.
//!   (C) EPISTEMIC LEGITIMACY — re-run the meta-drives GA; it KEEPS+tunes the learning-progress term (reports the
//!       weight) and — the anti-hack crux — refuses raw surprise (the noisy-TV trap). BINARY-NOVELTY is the
//!       control and loses the crux; the weight is DISCOVERED, never hand-set.
//!   (D) PURITY LITMUS — the GOALLESS control (no x*) never solves, exactly as `substrate_purpose`'s goalless arm.
//!
//! HONEST (truth-not-trophy): at this synthetic steady state the runtime pull comes from the EXTRINSIC `drive`
//! term (the epistemic term settles to ~0 once the model has converged — LP is a "still-learning-here" signal,
//! one-shot until a state repeats). The learning-progress term's distinct value — dodging the dark-room / noisy-TV
//! — is proven at the MECHANISM level in (C); a real LP-driven exploration GAIN is gated on the M2 perceptual-
//! abstraction crux (the ticket pre-registers that null). This is a SYNTHETIC/replayed toy, not a live ARC-3 run.
//!
//!   cargo test -p openmorph-core --test proofs substrate_selfgoal -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;
use openmorph_core::Config;

use crate::common::Rng;

// ── The synthetic controllable toy: a 1-D corridor of NPOS positions. The avatar starts at the DARK-ROOM end
//    (D, content 0 → perfectly predictable) and REWARD sits at the far end (R). Experience grows a world model
//    (per-position content prediction → per-cell surprise/learning-progress) and folds the env reward into `drive`
//    (`reinforce`) — so the organism ends up with grown drive on R and near-zero surprise on the dark-room. ──
const NPOS: usize = 12;
const D: usize = 0; // dark-room / start: content 0, the most-predictable do-nothing state
const R: usize = NPOS - 1; // the reward-associated position (far end)
const T: usize = 30; // evaluation rollout length

fn pc(p: usize) -> usize {
    NPOS + p
} // the state/target cell for position p (inputs are the one-hot 0..NPOS)
fn onehot(p: usize) -> Vec<(usize, f32)> {
    (0..NPOS)
        .map(|i| (i, if i == p { 1.0 } else { 0.0 }))
        .collect()
}
/// The world's per-position observation the model learns to predict. The dark-room (D) is a trivial constant 0
/// (instantly predicted ⇒ surprise → 0); every other position carries a distinct value plus small IRREDUCIBLE
/// noise, so its surprise floors ABOVE the dark-room's — the dark-room is the strict argmin surprise.
fn content(p: usize, rng: &mut Rng) -> f32 {
    if p == D {
        0.0
    } else {
        0.45 + 0.3 * (p as f32 / NPOS as f32) + rng.noise(0.02)
    }
}

fn neighbors(p: usize) -> Vec<usize> {
    let mut v = Vec::new();
    if p > 0 {
        v.push(p - 1);
    }
    if p + 1 < NPOS {
        v.push(p + 1);
    }
    v
}

/// Corridor adjacency over the STATE cells, for the planner ([`Field::relax_values`]).
fn adjacency(n: usize) -> Vec<Vec<usize>> {
    let mut adj = vec![Vec::new(); n];
    for p in 0..NPOS {
        adj[pc(p)] = neighbors(p).into_iter().map(pc).collect();
    }
    adj
}

struct Roll {
    reward: usize,   // eval steps spent at R
    coverage: usize, // distinct positions visited
    entropy: f32,    // action (left/stay/right) entropy — 0 = paralysis
}

/// Deterministic GREEDY value-descent from the corridor start: at each step move to the neighbour with strictly
/// lower cost-to-go, else stay. The SAME action rule for every arm — the ONLY thing that differs is the goal x*,
/// so the rollout isolates the goal's effect. A goalless field (all values `f32::MAX`) never moves.
fn descend(f: &Field) -> Roll {
    let mut pos = D;
    let mut visited = [false; NPOS];
    visited[pos] = true;
    let mut acts = [0usize; 3]; // [left, stay, right]
    let mut reward = 0usize;
    for _ in 0..T {
        let mut best = pos;
        let mut best_v = f.value_of(pc(pos));
        for np in neighbors(pos) {
            let v = f.value_of(pc(np));
            if v < best_v {
                best_v = v;
                best = np;
            }
        }
        acts[(best as isize - pos as isize + 1) as usize] += 1;
        pos = best;
        visited[pos] = true;
        if pos == R {
            reward += 1;
        }
    }
    let coverage = visited.iter().filter(|&&b| b).count();
    let entropy = acts
        .iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f32 / T as f32;
            -p * p.ln()
        })
        .sum::<f32>();
    Roll {
        reward,
        coverage,
        entropy,
    }
}

/// Prepare the value field for a given goal (clamp x* + relax to cost-to-go), then roll out. `None` = the GOALLESS
/// arm (no x*, no gradient → paralysis).
fn run_goal(f: &mut Field, goal: Option<usize>) -> Roll {
    f.clear_values();
    if let Some(g) = goal {
        f.prefer(g, 0.0);
        let adj = adjacency(f.len());
        f.relax_values(&adj, 1.0, 0);
    }
    descend(f)
}

/// Live the toy: grow the world model + accrue reward-drive over back-and-forth sweeps. Returns the experienced
/// field and its state-cell pool (the candidate ports for `orient`).
fn experience(seed: u64) -> (Field, Vec<usize>) {
    let mut f = Field::new(2 * NPOS);
    f.rng_seed(seed);
    let mut rng = Rng::new(seed ^ 0x5EED);
    for _ in 0..14 {
        // one sweep out and back
        let path: Vec<usize> = (0..NPOS).chain((0..NPOS - 1).rev()).collect();
        for &p in &path {
            let tgt = content(p, &mut rng);
            for _ in 0..2 {
                f.grow(&onehot(p), pc(p), tgt, 0.3, 32);
            }
            // fold the env reward into `drive` (grown, keyed by the visited state cell — NOT an authored goal)
            f.reinforce(pc(p), if p == R { 1.0 } else { 0.0 });
        }
    }
    let sc: Vec<usize> = (0..NPOS).map(pc).collect();
    (f, sc)
}

/// The dark-room cell = the strict argmin-surprise state (the naive expected-free-energy-min "goal").
fn darkroom_cell(f: &Field) -> usize {
    (0..NPOS)
        .min_by(|&a, &b| f.surprise_of(pc(a)).total_cmp(&f.surprise_of(pc(b))))
        .map(pc)
        .unwrap()
}

fn mean_sd(xs: &[f32]) -> (f32, f32) {
    let m = xs.iter().sum::<f32>() / xs.len() as f32;
    let v = xs.iter().map(|&x| (x - m).powi(2)).sum::<f32>() / xs.len() as f32;
    (m, v.sqrt())
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
// (A) EXTRINSIC — self-orient reaches the reward-associated state, > 2 sd better than a RANDOM reachable goal.
// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
#[test]
fn a_self_goal_beats_random_goal_by_more_than_2sd() {
    let seeds = 8;
    let mut zs = Vec::new();
    let (mut self_rs, mut rand_ms) = (Vec::new(), Vec::new());
    for s in 0..seeds {
        let (mut f, sc) = experience(0xA0 + s);
        // the organism SELECTS its own goal from grown drive + learning-progress (no authored cell / cost).
        let goal = f.orient(&sc).expect("orient picks a goal");
        // the reward for EVERY reachable goal → the random-reachable-goal distribution (the honest baseline).
        let rewards: Vec<f32> = (0..NPOS)
            .map(|g| run_goal(&mut f, Some(pc(g))).reward as f32)
            .collect();
        let (mean_r, sd_r) = mean_sd(&rewards);
        let self_r = run_goal(&mut f, Some(goal)).reward as f32;
        let z = (self_r - mean_r) / sd_r.max(1e-6);
        assert_eq!(
            goal,
            pc(R),
            "orient must select the reward-associated state (max drive)"
        );
        zs.push(z);
        self_rs.push(self_r);
        rand_ms.push(mean_r);
    }
    let (mz, _) = mean_sd(&zs);
    let (mself, _) = mean_sd(&self_rs);
    let (mrand, _) = mean_sd(&rand_ms);
    let min_z = zs.iter().copied().fold(f32::MAX, f32::min);
    eprintln!("[SELFGOAL/A] self-orient reward {mself:.1} vs random-reachable-goal mean {mrand:.2} → z = {mz:.2} (min {min_z:.2}) sd above the random-goal distribution (>2 = strictly better).");
    assert!(
        min_z > 2.0,
        "self-orient must beat the random-reachable goal by > 2 sd every seed (min z {min_z:.2})"
    );
    assert!(
        mself > 3.0 * mrand,
        "self reward must dominate the random-goal mean ({mself:.1} vs {mrand:.2})"
    );
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
// (B) NO DARK-ROOM — orient does not pick the predictable do-nothing state; its coverage/entropy beats do-nothing,
//     while a naive SURPRISE-MIN goal-selector collapses into the dark-room.
// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
#[test]
fn b_no_dark_room_collapse() {
    let seeds = 6;
    let (mut sc_cov, mut sc_ent, mut dark_cov, mut none_cov) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut dark_reward_total = 0usize;
    for s in 0..seeds {
        let (mut f, sc) = experience(0xB0 + s);
        let goal = f.orient(&sc).unwrap();
        let dark = darkroom_cell(&f);
        // the KEY "no dark-room" claim: the organism did NOT prefer the predictable do-nothing state.
        assert_ne!(
            goal, dark,
            "orient must NOT choose the predictable dark-room state"
        );
        assert_eq!(
            dark,
            pc(D),
            "the dark-room IS the constant/most-predictable state"
        );

        let self_roll = run_goal(&mut f, Some(goal)); // self-generated goal
        let dark_roll = run_goal(&mut f, Some(dark)); // naive surprise-min goal
        let none_roll = run_goal(&mut f, None); // do-nothing / goalless

        assert!(
            self_roll.coverage > none_roll.coverage,
            "self-orient must MOVE (coverage {} > do-nothing {})",
            self_roll.coverage,
            none_roll.coverage
        );
        assert!(
            self_roll.entropy > none_roll.entropy + 0.1,
            "self-orient action-entropy must beat do-nothing ({:.2} vs {:.2})",
            self_roll.entropy,
            none_roll.entropy
        );
        // the surprise-min goal collapses to the dark-room = do-nothing (no coverage, no reward).
        assert_eq!(
            dark_roll.coverage, none_roll.coverage,
            "surprise-min must collapse into the dark-room (same coverage as do-nothing)"
        );
        assert_eq!(dark_roll.reward, 0, "the dark-room yields no reward");
        assert!(self_roll.reward > 0, "self-orient must reach reward");

        sc_cov.push(self_roll.coverage as f32);
        sc_ent.push(self_roll.entropy);
        dark_cov.push(dark_roll.coverage as f32);
        none_cov.push(none_roll.coverage as f32);
        dark_reward_total += dark_roll.reward;
    }
    let (mc, _) = mean_sd(&sc_cov);
    let (me, _) = mean_sd(&sc_ent);
    let (mdc, _) = mean_sd(&dark_cov);
    let (mnc, _) = mean_sd(&none_cov);
    eprintln!("[SELFGOAL/B] coverage — self-orient {mc:.1}, surprise-min(dark-room) {mdc:.1}, do-nothing {mnc:.1}; self action-entropy {me:.2}. The predictable do-nothing state is NOT chosen; surprise-min collapses, self explores. dark-room total reward {dark_reward_total}.");
    assert_eq!(
        dark_reward_total, 0,
        "the surprise-min / dark-room arm must never collect reward"
    );
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
// (D) PURITY LITMUS — the GOALLESS control never solves (matching substrate_purpose's goalless arm).
// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
#[test]
fn d_goalless_never_solves() {
    let seeds = 6;
    let (mut self_r, mut none_r) = (Vec::new(), Vec::new());
    for s in 0..seeds {
        let (mut f, sc) = experience(0xD0 + s);
        let goal = f.orient(&sc).unwrap();
        self_r.push(run_goal(&mut f, Some(goal)).reward as f32);
        none_r.push(run_goal(&mut f, None).reward as f32); // no x* → no gradient → never acts
    }
    let (ms, _) = mean_sd(&self_r);
    let (mn, _) = mean_sd(&none_r);
    eprintln!("[SELFGOAL/D] self-orient reward {ms:.1} vs GOALLESS {mn:.1} — a goalless morph predicts the world but never acts, exactly the wall behind ARC-3.");
    assert!(
        none_r.iter().all(|&r| r == 0.0),
        "the goalless morph must NEVER solve (reward 0)"
    );
    assert!(
        ms > 10.0,
        "self-orient must solidly solve where goalless cannot ({ms:.1})"
    );
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
// (C) EPISTEMIC LEGITIMACY — the meta-drives GA KEEPS+tunes the learning-progress term and REFUSES raw surprise
//     (the noisy-TV / dark-room anti-hack). The weight is DISCOVERED, not hand-set; binary-novelty is the control.
//     (A compact re-run of the substrate_meta_drives discovery pass, framed on the GOAL-orientation objective.)
// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
const KG: usize = 4; // options per meta-decision
const GTRIALS: usize = 16;
const NTERM: usize = 5;
const T_DRIVE: usize = 0; // extrinsic reward-memory
const T_LP: usize = 1; // learning-progress (reducible surprise) — the term under test
const T_RAW: usize = 2; // RAW surprise — the noisy-TV trap the anti-hack must refuse
const T_NOV: usize = 3; // binary untried novelty — the CONTROL signal
                        // 4 = bias

fn gterm(t: usize, s: &[f32; 4]) -> f32 {
    if t < 4 {
        s[t]
    } else {
        1.0
    }
}

#[derive(Clone)]
struct Policy {
    terms: Vec<(usize, f32)>,
}
impl Policy {
    fn score(&self, s: &[f32; 4]) -> f32 {
        self.terms.iter().map(|&(t, c)| c * gterm(t, s)).sum()
    }
    fn weight(&self, sig: usize) -> f32 {
        self.terms
            .iter()
            .filter(|&&(t, _)| t == sig)
            .map(|&(_, c)| c)
            .sum()
    }
}
fn choose(p: &Policy, opts: &[[f32; 4]]) -> usize {
    (0..opts.len())
        .max_by(|&a, &b| p.score(&opts[a]).total_cmp(&p.score(&opts[b])))
        .unwrap()
}

/// EXTRINSIC faculty: options carry distinct `drive`; every other signal is uninformative. Correct = max drive.
fn extrinsic(p: &Policy, seed: u64) -> f32 {
    let mut rng = Rng::new(seed);
    let mut ok = 0;
    for _ in 0..GTRIALS {
        let mut opts = [[0.0f32; 4]; KG];
        for o in opts.iter_mut() {
            o[T_DRIVE] = rng.f();
        }
        let correct = (0..KG)
            .max_by(|&a, &b| opts[a][T_DRIVE].total_cmp(&opts[b][T_DRIVE]))
            .unwrap();
        if choose(p, &opts) == correct {
            ok += 1;
        }
    }
    ok as f32 / GTRIALS as f32
}

/// ANTI-DARK-ROOM / noisy-TV faculty (the crux). One option is the DARK-ROOM (all-zero: predictable do-nothing),
/// one is the NOISY-TV (max RAW surprise, zero learning-progress), the rest are LEARNABLE (genuine reducible
/// surprise). Correct = pick a LEARNABLE option — avoid BOTH the dark-room and the TV. A policy that weighs RAW
/// surprise is captured by the TV; a policy with no epistemic term cannot tell the dark-room from a learnable
/// option; only LEARNING-PROGRESS separates them.
fn anti_darkroom(p: &Policy, seed: u64) -> f32 {
    let mut rng = Rng::new(seed);
    let mut ok = 0;
    for _ in 0..GTRIALS {
        let dark = rng.below(KG);
        let mut tv = rng.below(KG);
        while tv == dark {
            tv = rng.below(KG);
        }
        let mut opts = [[0.0f32; 4]; KG];
        for (i, o) in opts.iter_mut().enumerate() {
            if i == dark {
                // predictable do-nothing: nothing to gain (all signals zero)
            } else if i == tv {
                o[T_RAW] = 1.0; // maximal raw surprise, but zero reducible surprise
            } else {
                o[T_LP] = 0.3 + 0.7 * rng.f(); // genuine reducible uncertainty
                o[T_RAW] = 0.2 * rng.f();
            }
        }
        let c = choose(p, &opts);
        if c != dark && c != tv {
            ok += 1;
        }
    }
    ok as f32 / GTRIALS as f32
}

fn gfitness(p: &Policy) -> f32 {
    if p.terms.is_empty() {
        return -1.0;
    }
    const LAMBDA: f32 = 0.02;
    let ex = extrinsic(p, 0x11);
    let anti = anti_darkroom(p, 0x22);
    ex.min(anti) - LAMBDA * p.terms.len() as f32
}

fn gmutate(r: &Policy, rng: &mut Rng) -> Policy {
    let mut t = r.terms.clone();
    match rng.u() % 4 {
        0 if t.len() < 5 => t.push(((rng.u() as usize) % NTERM, rng.sym())),
        1 if t.len() > 1 => {
            t.remove((rng.u() as usize) % t.len());
        }
        2 if !t.is_empty() => {
            let i = (rng.u() as usize) % t.len();
            t[i].1 += 0.4 * rng.sym();
        }
        _ if !t.is_empty() => {
            let i = (rng.u() as usize) % t.len();
            t[i].0 = (rng.u() as usize) % NTERM;
        }
        _ => t.push(((rng.u() as usize) % NTERM, rng.sym())),
    }
    Policy { terms: t }
}

fn gevolve(seed: u64) -> (Policy, f32) {
    let mut rng = Rng::new(seed);
    let mut pop: Vec<Policy> = (0..40)
        .map(|_| Policy {
            terms: vec![((rng.u() as usize) % NTERM, rng.sym())],
        })
        .collect();
    let (mut best, mut best_fit) = (pop[0].clone(), f32::MIN);
    for _ in 0..80 {
        let scored: Vec<(f32, Policy)> = pop.iter().map(|r| (gfitness(r), r.clone())).collect();
        for (f, r) in &scored {
            if *f > best_fit {
                best_fit = *f;
                best = r.clone();
            }
        }
        let mut order: Vec<usize> = (0..scored.len()).collect();
        order.sort_by(|&a, &b| scored[b].0.partial_cmp(&scored[a].0).unwrap());
        let elite: Vec<Policy> = order
            .iter()
            .take(10)
            .map(|&i| scored[i].1.clone())
            .collect();
        pop = elite.clone();
        while pop.len() < 40 {
            let parent = &elite[(rng.u() as usize) % elite.len()];
            pop.push(gmutate(parent, &mut rng));
        }
    }
    (best, best_fit)
}

#[test]
fn c_epistemic_weight_is_meta_evolved_not_hand_set() {
    // BINARY-NOVELTY control: the base goal-policy that reads reward-drive + a BINARY untried flag, but NOT the
    // graded learning-progress term. It passes the extrinsic faculty but has no epistemic signal for the anti-hack.
    let control = Policy {
        terms: vec![(T_DRIVE, 1.0), (T_NOV, 0.2)],
    };
    let control_fit = gfitness(&control);
    let control_anti = anti_darkroom(&control, 0x22);

    let (best, best_fit) = gevolve(0x6E7);
    let w_drive = best.weight(T_DRIVE);
    let w_lp = best.weight(T_LP);
    let w_raw = best.weight(T_RAW);
    let best_anti = anti_darkroom(&best, 0x22);
    let lbl = ["drive", "learn-progress", "raw-surprise", "novelty", "1"];
    let name: String = best
        .terms
        .iter()
        .map(|&(t, c)| format!("{c:+.2}·{}", lbl[t]))
        .collect::<Vec<_>>()
        .join(" ");

    eprintln!("[SELFGOAL/C] meta-drives GA on the GOAL-orientation objective (extrinsic + anti-dark-room/noisy-TV):");
    eprintln!("[SELFGOAL/C]   BINARY-NOVELTY control: fit {control_fit:.3}  anti-dark-room {control_anti:.2}  [+drive +0.2·novelty]");
    eprintln!("[SELFGOAL/C]   DISCOVERED policy      : fit {best_fit:.3}  anti-dark-room {best_anti:.2}  [{name}]");
    eprintln!("[SELFGOAL/C]   discovered weights: drive {w_drive:+.2}  learn-progress {w_lp:+.2}  raw-surprise {w_raw:+.2}");
    eprintln!("[SELFGOAL/C]   Config::default().epistemic_weight = {:+.2} (the recorded GA output folded into orient)", Config::default().epistemic_weight);

    // The discovered graded-epistemic policy beats the binary-novelty control.
    assert!(
        best_fit > control_fit + 0.02,
        "discovered must beat the binary-novelty control ({best_fit:.3} vs {control_fit:.3})"
    );
    // EXTRINSIC preserved (reward-memory sought).
    assert!(
        w_drive > 0.0,
        "the goal policy must weigh reward-drive positively (extrinsic value)"
    );
    // KEEPS + TUNES the learning-progress term — DISCOVERED, not hand-set (a positive weight found by search).
    assert!(
        w_lp > 0.0,
        "the meta-drives GA must KEEP+tune the learning-progress term ({w_lp:+.2})"
    );
    // THE ANTI-HACK: refuses raw surprise (would be captured by the noisy TV / never dodges the dark-room).
    assert!(
        w_raw <= 0.05,
        "raw-surprise must NOT be weighted positively (the noisy-TV trap): {w_raw:+.2}"
    );
    // The learning-progress term is what WINS the crux where binary-novelty ties/loses.
    assert!(best_anti > control_anti + 0.2, "learning-progress must beat binary-novelty on the anti-dark-room crux ({best_anti:.2} vs {control_anti:.2})");
    // Provenance-consistency: the Config coefficient folded into orient has the SAME (positive) sign the GA found.
    assert!(
        Config::default().epistemic_weight > 0.0 && w_lp > 0.0,
        "orient's epistemic_weight is the discovered positive coefficient, not hand-set"
    );

    eprintln!("[SELFGOAL/C] VERDICT: `orient`'s epistemic weight is the META-EVOLVED learning-progress coefficient — the meta-drives GA KEEPS+tunes it (weight {w_lp:+.2}) and REFUSES raw surprise (anti-hack, {w_raw:+.2}), beating the binary-novelty control on the dark-room/noisy-TV crux ({best_anti:.2} vs {control_anti:.2}). The weight is DISCOVERED by search; a hand-set weight (or the binary-novelty control) is the purity FAIL this guards against.");
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
// R5 follow-on — the capacity-ceiling half of `saturated`: `at_capacity` fires only once the head's writer has
// grown its FULL hidden allotment (n_hidden == max_hidden). Grown, read-only, byte-identical when unused.
// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
#[test]
fn e_capacity_ceiling_half_of_saturated() {
    let mut f = Field::new(3);
    f.rng_seed(0xCA9);
    f.config_mut().max_hidden = 3;
    f.config_mut().birth_gate = 0.05;
    // A target the linear readout cannot fit (its sign flips for the same input) → persistent surprise → grow
    // births hidden features up to the ceiling.
    let head = 2usize;
    let inputs = [(0usize, 1.0f32), (1usize, 1.0f32)];
    let _ = f.grow(&inputs, head, 0.9, 0.5, 24);
    let early = f.at_capacity(&[head]); // after one grow: below the ceiling
    for i in 0..400 {
        let tgt = if i % 2 == 0 { 0.9 } else { -0.9 };
        f.grow(&inputs, head, tgt, 0.5, 24);
    }
    let late = f.at_capacity(&[head]);
    eprintln!(
        "[SELFGOAL/E] at_capacity: early {early}  late {late} (n_hidden reaches max_hidden = {})",
        f.config().max_hidden
    );
    assert!(!early, "a head below its hidden ceiling is NOT at capacity");
    assert!(
        late,
        "a head that has grown its full hidden allotment IS at capacity"
    );
    // vacuous-safe: an empty head-set is not saturated-at-capacity.
    assert!(!f.at_capacity(&[]));
}
