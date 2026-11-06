// SPDX-License-Identifier: MIT-0
//! GROWN ACTIVE INFO-GATHERING — an info-gain drive for a FOG / POMDP where the win is SENSING, not reward.
//!
//! MorphBot's A7 cluster (~12 real ARC-3 games: beacon-sweep bn01–bn04, minefield deduction, concealment
//! pk02/ms0x) rewards REDUCING UNCERTAINTY about a HIDDEN state — you must SENSE (probe the right beacon) to
//! deduce where the mine / the concealed target is, and only then can you act correctly. Novelty *nudges* a
//! wanderer toward unseen cells but it does not VALUE information: it is drawn to whatever the world model
//! CANNOT predict — which on a fog world is the irreducible NOISE, the classic "noisy-TV" trap (exactly the
//! world-surprise bonus MorphLab measured to HURT). Reward-seeking has nothing to seek until the hidden state
//! is known. What these games need is a drive that values EXPECTED BELIEF-REDUCTION over the hidden state.
//!
//! This battery grows that drive by COMPOSING two shipped grown primitives — nothing authored about the task's
//! hidden variable:
//!   · the GROWN BELIEF-STATE (`belief_state.rs`): an ENSEMBLE of grown per-state scorers over the accumulated
//!     evidence, whose averaged softmax is a POSTERIOR `q(h)` over the hidden state (the same varied-prior grown
//!     estimators + `aggregate`-style pooling `select_explore`/`substrate_voi` stand on). Trained on random
//!     PARTIAL evidence, so under-determined evidence ⇒ a broad posterior, complete evidence ⇒ a peaked one. The
//!     belief's UNCERTAINTY is the SPREAD of that posterior over the hidden state — its variance `Var_{h∼q}[h]`,
//!     high when many states remain consistent, ≈0 once one state is deduced.
//!   · a GROWN FORWARD MODEL: a grown map hidden-state → expected observation-per-probe. On the informative
//!     probes it learns `obs = f(state)`; on the noisy probes the target is random so it regresses to ≈0 — the
//!     organism DISCOVERS (not authored) that the noisy channels carry no state-signal.
//! The INFO-GAIN of a probe is then `U(q) − E_{h∼q}[ U(q | e with slot p ← forward(h,p)) ]`, `U` = the belief's
//! variance over the hidden state — the expected reduction in the belief's OWN spread, the outcomes imagined
//! through the grown forward model. Because the forward model maps every state to ≈0 on a noisy probe, sensing it
//! leaves the belief unchanged ⇒ info-gain ≈ 0; on an informative probe the imagined outcomes resolve a hidden
//! bit ⇒ info-gain is large. The drive senses to the beacon that most reduces hidden-state uncertainty, distinct
//! both from novelty (world-surprise, maximal on the noise) and from reward-seeking (flat until the state is deduced).
//!
//! ## The FOG world (INSIDE-scored, deterministic, chance-anchored)
//! A hidden state `h ∈ 0..K` (K=4, two bits), uniform. `P` probes: the first `N_INFO`=2 each reveal one bit of
//! `h` (deterministic given `h`); the remaining `N_NOISY`=6 are NOISY-TV channels — a fresh random ±1 every
//! sense, independent of `h` (high world-surprise, ZERO information). An agent may SENSE a probe (fills that
//! slot of its evidence) up to a small BUDGET, then must DECLARE `h`. Reward requires the declaration to be
//! correct — reachable only by sensing BOTH informative probes.
//!
//! ## Arms (same grown belief + forward model; they differ ONLY in the DRIVE that picks the next probe)
//!   REWARD-SEEK — value = expected extrinsic reward. Sensing pays nothing, so it EXPLOITS the prior and
//!                 declares immediately → chance (nothing to seek until the state is known).
//!   NOVELTY     — value = the forward model's own prediction-surprise per probe (go-explore / world-surprise).
//!                 Maximal on the NOISY channels (unpredictable) → senses noise → deduces nothing (the trap).
//!   RANDOM      — sense uniformly at random for the budget (the ablation: sensing WITHOUT the info-gain value).
//!   INFO-GAIN   — value = expected reduction in belief-spread (grown). Senses the two informative probes,
//!                 stops when no probe reduces the belief further, declares → deduces the hidden state.
//!
//! ## THE AUTHORING LEDGER (truth-not-trophy)
//! GIVEN (the sensing INTERFACE, not the answer): which membrane channels are probes and that sensing a probe
//!   fills its evidence slot; the observation of a probe under a hidden state (the sensor's forward law) is
//!   OBSERVED during a learning phase, never told — the agent is never told which probes are noise or what `h` is.
//! GROWN: the belief (the ensemble of grown per-state scorers → a posterior), its uncertainty (the posterior's
//!   variance over the hidden state), the forward model (state → expected observation, which DISCOVERS the noisy
//!   probes are unpredictable), and thus the expected-belief-reduction VALUE. Nothing is a hand-wired entropy
//!   over the task's hidden variable — the uncertainty is a moment of the grown belief, not of the true `h`.
//! FIXED: the substrate laws (`grow`/`stimulate`/the local delta rule).
//! LOOP: the sense-until-uninformative-then-declare protocol is a harness policy over the grown drive.
//! No `openmorph-core` change — public API only.
//!
//!   cargo test -p openmorph-core --test proofs info_gain -- --nocapture

use openmorph_core::substrate::Field;

// ───────────────────────────────────────────── PRNG (byte-replayable xorshift64) ──────────────────────────
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
    /// Random sign −1.0 / +1.0.
    fn bit(&mut self) -> f32 {
        if (self.u() >> 40) & 1 == 0 {
            -1.0
        } else {
            1.0
        }
    }
    fn unit(&mut self) -> f32 {
        (self.u() >> 40) as f32 / (1u64 << 24) as f32
    }
    fn below(&mut self, n: usize) -> usize {
        (self.u() % n as u64) as usize
    }
}

// ─────────────────────────────────────────────── the FOG world ────────────────────────────────────────────
const K: usize = 4; // hidden states (two bits)
const N_INFO: usize = 2; // informative probes — probe p reveals bit p of h
const N_NOISY: usize = 6; // noisy-TV probes — random ±1, independent of h
const P: usize = N_INFO + N_NOISY; // total probes = evidence width
const BUDGET: usize = 4; // sensing steps before a declaration is forced

/// The sensor's forward law: what probe `p` OBSERVES under hidden state `h` (noisy probes draw fresh ±1).
fn observe(p: usize, h: usize, rng: &mut Rng) -> f32 {
    if p < N_INFO {
        if (h >> p) & 1 == 1 {
            1.0
        } else {
            -1.0
        }
    } else {
        rng.bit()
    }
}

// ─────────────────────────────────────── the GROWN belief-state (ensemble) ────────────────────────────────
const M: usize = 4; // ensemble members (varied priors + bootstrap data)
const EST_HIDDEN: usize = 4; // a few grown features per member; the readout still leans on the raw bits
const EST_TRAIN: usize = 4000;
const SCORE: usize = P; // each member's K per-state scorer cells live at P..P+K
const TAU: f32 = 4.0; // softmax temperature turning per-state scores into a posterior

/// Grow `M` belief estimators. Each is a K-way, one-vs-rest scorer of the hidden state over the evidence vector,
/// trained on random PARTIAL evidence (each slot masked to 0 = unsensed with prob 0.5) so that 0 is an
/// in-distribution "unsensed" value everywhere and the members learn a CALIBRATED per-state posterior: under
/// ambiguous evidence the consistent states score alike (a broad posterior), under complete evidence one state
/// wins (a peaked posterior). The noisy slots are uncorrelated with h ⇒ the scorers learn to ignore them.
fn grow_belief(seed: u64) -> Vec<Field> {
    (0..M)
        .map(|j| {
            let mut f = Field::new(P);
            f.config_mut().max_hidden = EST_HIDDEN;
            f.rng_seed(seed ^ (0x9E37_u64.wrapping_mul(j as u64 + 1)));
            f.ensure_cells(SCORE + K);
            let mut rng = Rng::new(seed ^ 0xBEEF ^ ((j as u64) << 20));
            for _ in 0..EST_TRAIN {
                let h = rng.below(K);
                let inp: Vec<(usize, f32)> = (0..P)
                    .map(|p| {
                        let o = observe(p, h, &mut rng);
                        (p, if rng.unit() < 0.5 { 0.0 } else { o }) // MASK to 0 = unsensed
                    })
                    .collect();
                for s in 0..K {
                    let tgt = if s == h { 1.0 } else { -1.0 }; // one-vs-rest membership of state s
                    f.grow(&inp, SCORE + s, tgt, 0.05, 40);
                }
            }
            f
        })
        .collect()
}

/// The belief = a soft posterior over the K hidden states, grown: each member's per-state scores → a softmax,
/// averaged across the ensemble (member disagreement broadens it; a confident consensus sharpens it).
fn belief_posterior(ens: &mut [Field], e: &[f32]) -> [f32; K] {
    let inp: Vec<(usize, f32)> = e.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    let mut q = [0.0f32; K];
    for f in ens.iter_mut() {
        f.stimulate(&inp, 80); // settle fully (all inputs specified) → deterministic regardless of history
        let sc: [f32; K] = core::array::from_fn(|s| f.cells()[SCORE + s]);
        let mx = sc.iter().copied().fold(f32::MIN, f32::max);
        let ex: [f32; K] = core::array::from_fn(|s| ((sc[s] - mx) * TAU).exp());
        let z: f32 = ex.iter().sum::<f32>().max(1e-9);
        for s in 0..K {
            q[s] += ex[s] / z;
        }
    }
    for s in q.iter_mut() {
        *s /= M as f32;
    }
    q
}

/// The belief's UNCERTAINTY over the hidden state = the posterior's VARIANCE over the state — the SPREAD of the
/// belief (high when many states remain consistent, ≈0 once one state is deduced). This is the quantity the
/// info-gain drive reduces.
fn belief_uncertainty(q: &[f32; K]) -> f32 {
    let mu: f32 = (0..K).map(|s| q[s] * s as f32).sum();
    (0..K).map(|s| q[s] * (s as f32 - mu).powi(2)).sum()
}

// ─────────────────────────────────────── the GROWN forward (observation) model ────────────────────────────
const FWD_BASE: usize = K; // outputs live at cells K..K+P; inputs are the one-hot state on 0..K

/// Grow a forward model `state → expected observation per probe`, and MEASURE its per-probe prediction surprise
/// (the novelty signal). Returns (field, per-probe residual EMA). The residual is high on the noisy probes
/// (random target ⇒ unpredictable) and ≈0 on the informative probes — the organism DISCOVERS which channels
/// carry state-signal; nothing is told.
fn grow_forward(seed: u64) -> (Field, [f32; P]) {
    let mut f = Field::new(K);
    f.config_mut().max_hidden = 0; // linear over the one-hot state = the exact per-state mean observation
    f.ensure_cells(FWD_BASE + P);
    let mut resid = [0.0f32; P];
    let mut rng = Rng::new(seed ^ 0xF00D);
    for step in 0..(EST_TRAIN * 2) {
        let h = rng.below(K);
        let onehot: Vec<(usize, f32)> = (0..K)
            .map(|c| (c, if c == h { 1.0 } else { 0.0 }))
            .collect();
        for (p, r) in resid.iter_mut().enumerate() {
            let o = observe(p, h, &mut rng);
            let (err, _) = f.grow(&onehot, FWD_BASE + p, o, 0.05, 20);
            if step > EST_TRAIN {
                // EMA of the residual once the model has converged — the world-surprise per probe.
                *r = 0.99 * *r + 0.01 * err;
            }
        }
    }
    (f, resid)
}

/// The forward model's predicted observation of probe `p` under hypothesis state `h`.
fn forward_predict(fwd: &mut Field, h: usize, p: usize) -> f32 {
    let onehot: Vec<(usize, f32)> = (0..K)
        .map(|c| (c, if c == h { 1.0 } else { 0.0 }))
        .collect();
    fwd.stimulate(&onehot, 20);
    fwd.cells()[FWD_BASE + p]
}

/// EXPECTED BELIEF-REDUCTION for sensing probe `p` given current evidence `e` — the grown info-gain:
/// `U(belief|e) − E_{state∼belief}[ U(belief | e with slot p ← forward(state, p)) ]`, where `U` is the belief's
/// variance over the hidden state. The expectation is EXACT over the K states weighted by the current posterior
/// `q`, and each outcome is imagined through the grown FORWARD MODEL. A noisy probe maps every state to ≈0 ⇒ the
/// imagined evidence is unchanged ⇒ the value is ≈0; an informative probe resolves a hidden bit ⇒ the value is large.
fn info_gain(
    ens: &mut [Field],
    fwd: &mut Field,
    e: &[f32],
    p: usize,
    q: &[f32; K],
    now: f32,
) -> f32 {
    let mut post = 0.0f32;
    for (s, &qs) in q.iter().enumerate() {
        if qs < 1e-4 {
            continue;
        }
        let mut e2 = e.to_vec();
        e2[p] = forward_predict(fwd, s, p);
        post += qs * belief_uncertainty(&belief_posterior(ens, &e2));
    }
    now - post
}

// ───────────────────────────────────────────────── the drives ─────────────────────────────────────────────
#[derive(Clone, Copy, PartialEq)]
enum Drive {
    RewardSeek,
    Novelty,
    Random,
    InfoGain,
}

const EPS: f32 = 0.02; // info-gain stops sensing when no probe reduces the belief by more than this

/// Run one fog episode under one drive: sense probes (per the drive) up to the budget, then declare `h`.
/// Returns (correct, sensing_steps).
fn run_episode(
    ens: &mut [Field],
    fwd: &mut Field,
    resid: &[f32; P],
    drive: Drive,
    true_h: usize,
    rng: &mut Rng,
) -> (bool, usize) {
    let mut e = [0.0f32; P];
    let mut sensed = [false; P];
    let mut steps = 0usize;
    loop {
        let choice: Option<usize> = match drive {
            // Nothing to seek until the state is known → exploit the prior immediately.
            Drive::RewardSeek => None,
            _ if steps >= BUDGET => None,
            Drive::Novelty => {
                // Go-explore / world-surprise: the probe the forward model predicts WORST (→ the noise).
                (0..P)
                    .filter(|&p| !sensed[p])
                    .max_by(|&a, &b| resid[a].total_cmp(&resid[b]))
            }
            Drive::Random => {
                let un: Vec<usize> = (0..P).filter(|&p| !sensed[p]).collect();
                un.get(rng.below(un.len().max(1)).min(un.len().saturating_sub(1)))
                    .copied()
            }
            Drive::InfoGain => {
                let q = belief_posterior(ens, &e);
                let now = belief_uncertainty(&q);
                let best = (0..P)
                    .filter(|&p| !sensed[p])
                    .map(|p| (p, info_gain(ens, fwd, &e, p, &q, now)))
                    .max_by(|a, b| a.1.total_cmp(&b.1));
                match best {
                    Some((p, ig)) if ig > EPS => Some(p),
                    _ => None, // no probe reduces the belief further → declare
                }
            }
        };
        match choice {
            Some(p) => {
                e[p] = observe(p, true_h, rng);
                sensed[p] = true;
                steps += 1;
            }
            None => break,
        }
    }
    let q = belief_posterior(ens, &e);
    let declared = (0..K).max_by(|&a, &b| q[a].total_cmp(&q[b])).unwrap_or(0);
    (declared == true_h, steps)
}

// ─────────────────────────────────────────────────── the battery ──────────────────────────────────────────
struct Arm {
    acc: f64,
    steps: f64,
}

fn run_seed(seed: u64, n_trial: usize) -> ([Arm; 4], f32, f32, f32, f32) {
    let mut ens = grow_belief(seed);
    let (mut fwd, resid) = grow_forward(seed);

    // Distinctness probe: what each drive VALUES, split info-probes vs noisy-probes, on the empty-evidence
    // belief (the first decision). info-gain should rank informative ≫ noisy; novelty the opposite.
    let e0 = vec![0.0f32; P];
    let q0 = belief_posterior(&mut ens, &e0);
    let now0 = belief_uncertainty(&q0);
    let ig: Vec<f32> = (0..P)
        .map(|p| info_gain(&mut ens, &mut fwd, &e0, p, &q0, now0))
        .collect();
    let ig_info = ig[..N_INFO].iter().sum::<f32>() / N_INFO as f32;
    let ig_noisy = ig[N_INFO..].iter().sum::<f32>() / N_NOISY as f32;
    let nov_info = resid[..N_INFO].iter().sum::<f32>() / N_INFO as f32;
    let nov_noisy = resid[N_INFO..].iter().sum::<f32>() / N_NOISY as f32;

    let drives = [
        Drive::RewardSeek,
        Drive::Novelty,
        Drive::Random,
        Drive::InfoGain,
    ];
    let mut arms: [Arm; 4] = core::array::from_fn(|_| Arm {
        acc: 0.0,
        steps: 0.0,
    });
    let mut rng = Rng::new(seed ^ 0x5EED_1234);
    for _ in 0..n_trial {
        let h = rng.below(K);
        for (a, &d) in drives.iter().enumerate() {
            let (ok, steps) = run_episode(&mut ens, &mut fwd, &resid, d, h, &mut rng);
            arms[a].acc += ok as u8 as f64;
            arms[a].steps += steps as f64;
        }
    }
    for a in arms.iter_mut() {
        a.acc /= n_trial as f64;
        a.steps /= n_trial as f64;
    }
    (arms, ig_info, ig_noisy, nov_info, nov_noisy)
}

#[test]
fn grown_info_gain_deduces_the_hidden_state_where_reward_and_novelty_cannot() {
    let seeds = [0xA0u64, 0xA1, 0xA2];
    let n_trial = 150;
    let (mut r, mut n, mut rnd, mut ig) = (
        Arm {
            acc: 0.0,
            steps: 0.0,
        },
        Arm {
            acc: 0.0,
            steps: 0.0,
        },
        Arm {
            acc: 0.0,
            steps: 0.0,
        },
        Arm {
            acc: 0.0,
            steps: 0.0,
        },
    );
    let (mut igi, mut ign, mut nvi, mut nvn) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    for &s in &seeds {
        let (arms, a, b, c, d) = run_seed(s, n_trial);
        let [ra, na, rna, ia] = arms;
        r.acc += ra.acc;
        r.steps += ra.steps;
        n.acc += na.acc;
        n.steps += na.steps;
        rnd.acc += rna.acc;
        rnd.steps += rna.steps;
        ig.acc += ia.acc;
        ig.steps += ia.steps;
        igi += a;
        ign += b;
        nvi += c;
        nvn += d;
    }
    let ns = seeds.len() as f64;
    for x in [&mut r, &mut n, &mut rnd, &mut ig] {
        x.acc /= ns;
        x.steps /= ns;
    }
    let nsf = seeds.len() as f32;
    igi /= nsf;
    ign /= nsf;
    nvi /= nsf;
    nvn /= nsf;
    let chance = 1.0 / K as f64;

    eprintln!(
        "[INFO-GAIN] FOG / POMDP — deduce hidden h∈0..{K} by SENSING ({N_INFO} informative + {N_NOISY} noisy-TV probes, budget {BUDGET}), {} trials × {} seeds. chance {:.0}%:",
        n_trial, seeds.len(), chance * 100.0
    );
    eprintln!(
        "  REWARD-SEEK (declare on the prior — nothing to seek)   deduce {:.0}%  senses {:.1}",
        r.acc * 100.0,
        r.steps
    );
    eprintln!(
        "  NOVELTY     (world-surprise → the noisy-TV trap)       deduce {:.0}%  senses {:.1}",
        n.acc * 100.0,
        n.steps
    );
    eprintln!(
        "  RANDOM      (sense at random — the ablation)           deduce {:.0}%  senses {:.1}",
        rnd.acc * 100.0,
        rnd.steps
    );
    eprintln!(
        "  INFO-GAIN   (expected belief-reduction — GROWN)        deduce {:.0}%  senses {:.1}",
        ig.acc * 100.0,
        ig.steps
    );
    eprintln!(
        "[INFO-GAIN] DISTINCTNESS (values on empty evidence, informative vs noisy probe):  info-gain  informative {igi:.3} ≫ noisy {ign:.3}   |   novelty(world-surprise)  informative {nvi:.2} ≪ noisy {nvn:.2}  — the two drives value OPPOSITE probes."
    );
    eprintln!(
        "[INFO-GAIN] VERDICT: a fog world rewards SENSING, not reward-seeking. Reward-seeking has nothing to seek until the hidden state is known, so it declares on the prior ({:.0}% = chance). Novelty (world-model surprise) is drawn to what it cannot predict — the noisy-TV channels — so it senses noise and deduces nothing ({:.0}%). Even random sensing ({:.0}%) beats novelty, because novelty is actively MISLED. The GROWN info-gain drive values a probe by the EXPECTED REDUCTION in its own grown belief's uncertainty (the variance of the grown posterior over the hidden state, the outcome imagined through the grown forward model that DISCOVERED the noisy channels carry no state-signal). It senses exactly the {N_INFO} informative beacons ({:.1} senses on average, well under the budget) and deduces the hidden state ({:.0}%). Expected-belief-reduction, not novelty, is the drive a POMDP needs. No `openmorph-core` change; the belief, its uncertainty, and the forward model are all grown — see the AUTHORING LEDGER.",
        r.acc * 100.0,
        n.acc * 100.0,
        rnd.acc * 100.0,
        ig.steps,
        ig.acc * 100.0
    );

    // ── Assertions (INSIDE-scored, chance-anchored, 3 seeds) ──
    // INFO-GAIN deduces the hidden state; the others cannot.
    assert!(
        ig.acc > 0.85,
        "info-gain must deduce the hidden state (deduce {:.2})",
        ig.acc
    );
    // Reward-seeking has nothing to seek → at chance, and never senses.
    assert!(
        (r.acc - chance).abs() < 0.13 && r.steps < 0.5,
        "reward-seeking must sit at chance and not sense (deduce {:.2}, senses {:.2})",
        r.acc,
        r.steps
    );
    // Novelty is trapped by the noisy-TV → no better than random, far below info-gain.
    assert!(
        ig.acc > n.acc + 0.40,
        "info-gain must beat novelty (info-gain {:.2} vs novelty {:.2})",
        ig.acc,
        n.acc
    );
    assert!(
        n.acc <= rnd.acc + 0.05,
        "novelty (world-surprise) must be no better than random — the noisy-TV trap (novelty {:.2} vs random {:.2})",
        n.acc,
        rnd.acc
    );
    // The ablation: targeted info-gain ≫ undirected random sensing.
    assert!(
        ig.acc > rnd.acc + 0.25,
        "targeted info-gain must beat random sensing (info-gain {:.2} vs random {:.2})",
        ig.acc,
        rnd.acc
    );
    // Info-gain senses efficiently — the two informative probes, under budget.
    assert!(
        ig.steps > 1.5 && ig.steps < BUDGET as f64,
        "info-gain must sense the informative probes and stop (senses {:.2})",
        ig.steps
    );
    // DISTINCTNESS: info-gain values the informative probes; novelty values the noisy ones — opposite rankings.
    assert!(
        igi > ign + 0.10,
        "info-gain must value informative ≫ noisy probes (info {igi:.3} vs noisy {ign:.3})"
    );
    assert!(
        nvn > nvi + 0.30,
        "novelty (world-surprise) must value noisy ≫ informative probes (noisy {nvn:.2} vs info {nvi:.2})"
    );
}
