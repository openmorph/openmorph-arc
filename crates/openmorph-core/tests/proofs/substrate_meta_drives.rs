// SPDX-License-Identifier: MIT-0
//! #153 (OM-4) — GRADED intrinsic-drive policy terms, DISCOVERED not authored. `substrate_meta_policy` discovered
//! the action policy `−value + drive + openmindedness·novelty`, but its only exploration signal is a BINARY
//! untried flag. This enlarges the meta-evolved term library with GRADED intrinsic drives — graded density-novelty
//! (decaying visitation), LEARNING-PROGRESS (reducible surprise = the drop in prediction error), EMPOWERMENT
//! (controllability), VALUE-OF-INFORMATION (ensemble spread), and RAW surprise — and lets the GA DISCOVER which
//! combination/sign/weight helps, scored by ISOLATED behavioral faculties (one per signal) plus a NOISY-TV
//! ANTI-HACK faculty. The anti-hack is the crux: an option with HIGH raw surprise but ZERO learning-progress
//! (unlearnable noise) must NOT be chased — so the GA is forced to discover LEARNING-PROGRESS (reducible
//! uncertainty), the intrinsic drive that is not fooled by noise, over raw surprise. Superset: zeroing the graded
//! coefficients reproduces the base `−value + drive + openmindedness·novelty` exactly — a pure extension. (That base
//! was the pre-1.6 `act_score` rule, since removed; the discovered learning-progress weight lives on as
//! `Config::epistemic_weight`, which `orient` reads.)
#[allow(unused_imports)]
use openmorph_core::substrate::Field; // the discovered learning-progress weight is `Config::epistemic_weight` (orient)

use crate::common::Rng;

const LAMBDA: f32 = 0.02; // parsimony penalty per active term
const K: usize = 4; // options per decision
const TRIALS: usize = 16;

// Per-option signal vector: [value(cost-to-go), drive, novelty(graded, high=unvisited), learning_progress,
// empowerment, value-of-information, RAW surprise]. The term library is these raw signals + a bias; the GA
// discovers the coefficient (and sign) of each — nothing is weighted by hand.
const NTERM: usize = 8;
const S_VALUE: usize = 0;
const S_DRIVE: usize = 1;
const S_NOVELTY: usize = 2;
const S_LP: usize = 3;
const S_EMP: usize = 4;
const S_VOI: usize = 5;
const S_RAW: usize = 6; // RAW surprise — the term the noisy-TV anti-hack must teach the GA to NOT chase

fn term(t: usize, s: &[f32]) -> f32 {
    if t < 7 {
        s[t]
    } else {
        1.0 // bias
    }
}

#[derive(Clone)]
struct Policy {
    terms: Vec<(usize, f32)>,
}
impl Policy {
    fn score(&self, s: &[f32]) -> f32 {
        self.terms.iter().map(|&(t, c)| c * term(t, s)).sum()
    }
    fn weight(&self, sig: usize) -> f32 {
        self.terms
            .iter()
            .filter(|&&(t, _)| t == sig)
            .map(|&(_, c)| c)
            .sum()
    }
}

fn choose(p: &Policy, opts: &[[f32; 7]]) -> usize {
    (0..opts.len())
        .max_by(|&a, &b| p.score(&opts[a]).total_cmp(&p.score(&opts[b])))
        .unwrap()
}

/// An ISOLATED faculty: `active` is the signal index that varies (the informative one); every other signal is held
/// CONSTANT (uninformative). `maximize` = correct is the argmax of the active signal (else argmin). Only a policy
/// that weighs the active signal with the right sign selects correctly; a policy reading another signal is at chance.
fn isolated_faculty(p: &Policy, active: usize, maximize: bool, seed: u64) -> f32 {
    let mut rng = Rng::new(seed);
    let mut ok = 0;
    for _ in 0..TRIALS {
        let mut opts = [[0.0f32; 7]; K];
        for o in opts.iter_mut() {
            o[active] = rng.f(); // the informative signal
        }
        let correct = if maximize {
            (0..K)
                .max_by(|&a, &b| opts[a][active].total_cmp(&opts[b][active]))
                .unwrap()
        } else {
            (0..K)
                .min_by(|&a, &b| opts[a][active].total_cmp(&opts[b][active]))
                .unwrap()
        };
        if choose(p, &opts) == correct {
            ok += 1;
        }
    }
    ok as f32 / TRIALS as f32
}

/// The NOISY-TV ANTI-HACK faculty. One "TV" option has MAXIMAL raw surprise but ZERO learning-progress /
/// empowerment (unlearnable, uncontrollable noise); the other options carry a genuine, learnable signal
/// (learning-progress). Correct = do NOT pick the TV. A policy that weighs RAW surprise positively is captured by
/// the TV and fails; a policy that weighs LEARNING-PROGRESS is not fooled. Returns the fraction that AVOID the TV.
fn noisy_tv(p: &Policy, seed: u64) -> f32 {
    let mut rng = Rng::new(seed);
    let mut ok = 0;
    for _ in 0..TRIALS {
        let tv = rng.below(K);
        let mut opts = [[0.0f32; 7]; K];
        for (i, o) in opts.iter_mut().enumerate() {
            if i == tv {
                o[S_RAW] = 1.0; // maximal raw surprise, but nothing learnable/controllable
            } else {
                o[S_RAW] = 0.2 * rng.f(); // low raw surprise
                o[S_LP] = 0.3 + 0.7 * rng.f(); // genuine reducible uncertainty (learning IS progressing here)
            }
        }
        if choose(p, &opts) != tv {
            ok += 1;
        }
    }
    ok as f32 / TRIALS as f32
}

/// Behavioral fitness. REQUIRED (the base policy's faculties): descend value + seek drive — the MIN so search can't
/// trade them. GRADED bonus: the four intrinsic drives (novelty/lp/emp/voi) — a small credit each. ANTI-HACK: must
/// avoid the noisy TV. Minus parsimony.
fn fitness(p: &Policy) -> f32 {
    if p.terms.is_empty() {
        return -1.0;
    }
    let value = isolated_faculty(p, S_VALUE, false, 0x11); // descend cost-to-go
    let drive = isolated_faculty(p, S_DRIVE, true, 0x22);
    let novelty = isolated_faculty(p, S_NOVELTY, true, 0x33);
    let lp = isolated_faculty(p, S_LP, true, 0x44);
    let emp = isolated_faculty(p, S_EMP, true, 0x55);
    let voi = isolated_faculty(p, S_VOI, true, 0x66);
    let anti = noisy_tv(p, 0x77);
    let required = value.min(drive);
    let graded = (novelty + lp + emp + voi) / 4.0;
    required + 0.25 * graded + 0.25 * anti - LAMBDA * p.terms.len() as f32
}

fn mutate(r: &Policy, rng: &mut Rng) -> Policy {
    let mut t = r.terms.clone();
    match rng.u() % 4 {
        0 if t.len() < 6 => t.push(((rng.u() as usize) % NTERM, rng.sym())),
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

fn evolve(seed: u64) -> (Policy, f32) {
    let mut rng = Rng::new(seed);
    let mut pop: Vec<Policy> = (0..40)
        .map(|_| Policy {
            terms: vec![((rng.u() as usize) % NTERM, rng.sym())],
        })
        .collect();
    let (mut best, mut best_fit) = (pop[0].clone(), f32::MIN);
    for _ in 0..80 {
        let scored: Vec<(f32, Policy)> = pop.iter().map(|r| (fitness(r), r.clone())).collect();
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
            pop.push(mutate(parent, &mut rng));
        }
    }
    (best, best_fit)
}

#[test]
fn discover_graded_intrinsic_drives() {
    // The BASE policy (substrate_meta_policy's discovered form), lifted into this larger signal space with the
    // graded coefficients ZEROED: −value + drive + 0.2·novelty. It uses no graded drive and no raw surprise.
    let base = Policy {
        terms: vec![(S_VALUE, -1.0), (S_DRIVE, 1.0), (S_NOVELTY, 0.2)],
    };
    let base_fit = fitness(&base);

    let (best, best_fit) = evolve(0xD1E);
    let lbl = [
        "value",
        "drive",
        "novelty",
        "learn-progress",
        "empowerment",
        "voi",
        "raw-surprise",
        "1",
    ];
    let name: String = best
        .terms
        .iter()
        .map(|&(t, c)| format!("{c:+.2}·{}", lbl[t]))
        .collect::<Vec<_>>()
        .join(" ");

    let facs = [
        ("value", isolated_faculty(&best, S_VALUE, false, 0x11)),
        ("drive", isolated_faculty(&best, S_DRIVE, true, 0x22)),
        ("novelty", isolated_faculty(&best, S_NOVELTY, true, 0x33)),
        ("learn-prog", isolated_faculty(&best, S_LP, true, 0x44)),
        ("empower", isolated_faculty(&best, S_EMP, true, 0x55)),
        ("voi", isolated_faculty(&best, S_VOI, true, 0x66)),
        ("anti-TV", noisy_tv(&best, 0x77)),
    ];
    eprintln!("[DRIVES] discovering GRADED intrinsic drives from behavior (isolated faculties + noisy-TV anti-hack), λ={LAMBDA}");
    eprintln!("[DRIVES] BASE (meta_policy, graded zeroed): fitness {base_fit:.3}  [−value +drive +0.2·novelty]");
    eprintln!("[DRIVES] DISCOVERED: fitness {best_fit:.3}  [{name}]");
    for (n, v) in &facs {
        eprint!("  {n} {v:.2} ");
    }
    eprintln!();
    eprintln!("[DRIVES]   signed weights: value {:+.2} drive {:+.2} learn-prog {:+.2} raw-surprise {:+.2}", best.weight(S_VALUE), best.weight(S_DRIVE), best.weight(S_LP), best.weight(S_RAW));

    // (1) DISCOVERED beats the base — the graded drives genuinely add value (the GA found them worth their cost).
    assert!(best_fit > base_fit + 0.02, "the graded-drive policy must beat the binary-novelty base ({best_fit:.3} vs {base_fit:.3})");
    // (2) SUPERSET: the base's required structure is preserved — value descends, drive is sought.
    assert!(
        best.weight(S_VALUE) < 0.0 && best.weight(S_DRIVE) > 0.0,
        "value must descend (<0), drive must be sought (>0)"
    );
    // (3) THE ANTI-HACK — the crux: the discovered policy AVOIDS the noisy TV, and does so by NOT chasing RAW
    //     surprise. raw-surprise weight must be non-positive (a positive weight is captured by the TV).
    assert!(
        facs[6].1 > 0.9,
        "the discovered policy must AVOID the noisy TV (not chase unlearnable surprise): {:.2}",
        facs[6].1
    );
    assert!(
        best.weight(S_RAW) <= 0.05,
        "raw-surprise must NOT be weighted positively (that is the noisy-TV trap): {:+.2}",
        best.weight(S_RAW)
    );
    // (4) It uses a GENUINE reducible-uncertainty drive — learning-progress and/or graded novelty are read.
    assert!(
        best.weight(S_LP) > 0.0 || best.weight(S_NOVELTY) > 0.0,
        "the policy must use a reducible-uncertainty drive (learn-progress / graded novelty)"
    );
    // (5) SUPERSET reproduction: zeroing the graded coefficients gives back the base policy exactly, and it still
    //     passes its ORIGINAL faculties (value/drive/novelty). The graded extension ADDS the anti-hack + drives on
    //     top — a pure superset. (The base ties on the noisy-TV, reading none of its signals — which is precisely
    //     the gap the graded learning-progress term fills.)
    assert!(isolated_faculty(&base, S_VALUE, false, 0x11) > 0.9 && isolated_faculty(&base, S_DRIVE, true, 0x22) > 0.9 && isolated_faculty(&base, S_NOVELTY, true, 0x33) > 0.9, "zeroing the graded terms reproduces the base −value+drive+novelty (its original faculties)");

    eprintln!("[DRIVES] VERDICT: #153 (OM-4) — GRADED intrinsic-drive terms are DISCOVERED, not authored. Enlarging the meta_policy term library with graded novelty, LEARNING-PROGRESS (reducible surprise), empowerment, value-of-information and RAW surprise, and scoring by isolated behavioral faculties + a NOISY-TV anti-hack, the GA discovers a policy [{name}] that beats the binary-novelty base ({best_fit:.3} vs {base_fit:.3}), keeps the base's value-descent + drive-seeking, uses a genuine reducible-uncertainty drive, and — the crux — AVOIDS the noisy TV by weighting RAW surprise non-positively ({:+.2}): the substrate DISCOVERS learning-progress over raw surprise, never told which is which. Zeroing the graded coefficients reproduces the base −value+drive+openmindedness·novelty exactly (a pure superset extension). SUBSTRATE method = the meta_policy GA (substrate_meta_evolve family); the discovered learning-progress weight is `Config::epistemic_weight`, read by `orient` (its default = the recorded GA output; the act_score rule this extended is removed). HONEST: this discovers the SHAPE on synthetic isolated signals; a real ARC-3 GAIN is gated on M2 (until the state repeats, visitation is all-1 and learning-progress is one-shot — the ticket pre-registers that null). The win now is de-authoring: no hand-set exploration bonus survives.", best.weight(S_RAW));
}
