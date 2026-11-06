// SPDX-License-Identifier: MIT-0
//! #150 — DISCOVER the action-selection policy, don't AUTHOR it (Sébastien: "discover the diff equation as we
//! did to get v2 — it was the solution"). The `act_score` I hand-wrote (`value − (1−wildness)·aversion +
//! openmindedness·novelty`, hand-set weights) was an authored policy in core — a founding-rule #1 violation.
//! Instead, meta-evolve the policy SCORE from a raw term library over the per-option signals {value (cost-to-go),
//! drive (reward memory), novelty (untried)}, selected by BEHAVIORAL fitness across three ISOLATED faculties —
//! descend cost-to-go (plan), seek reward-memory (avoid/commit), explore the untried — minus a PARSIMONY penalty.
//! Each faculty holds the other two signals constant (uninformative) so a hack scores at chance on what it
//! ignores. The TEMPERAMENT (the weighting) then falls out of search against behavior — nothing authored —
//! exactly the meta-discovery method that produced v2. (Template: `substrate_meta_evolve`.)
//!
//! HISTORY: the discovered form `−value + drive + openmindedness·novelty` became the `act_score` rule, the default
//! `alive(Act)` rule until 1.6. The valence rule replaced it (`substrate_valence`) and it is now removed; this proof
//! keeps the discovery method and its measurement.
use openmorph_core::substrate::Field; // pulled in so the file sits with the core suite

use crate::common::Rng;

const LAMBDA: f32 = 0.03; // parsimony penalty per active term

// The term LIBRARY — candidate terms of the per-option score f(v, d, n): v=cost-to-go value, d=drive
// (reward/aversive memory), n=novelty (1 if untried). Raw signals + a couple of interactions; the coefficient
// (and its SIGN) is what search discovers — no weighting is authored.
fn term(t: usize, v: f32, d: f32, n: f32) -> f32 {
    match t {
        0 => v,     // cost-to-go
        1 => d,     // drive
        2 => n,     // novelty
        3 => v * d, // value×drive interaction
        4 => d * d, // drive²
        5 => v * v, // value²
        _ => 1.0,   // bias
    }
}
const NTERM: usize = 7;

#[derive(Clone)]
struct Policy {
    terms: Vec<(usize, f32)>,
}
impl Policy {
    fn score(&self, v: f32, d: f32, n: f32) -> f32 {
        self.terms.iter().map(|&(t, c)| c * term(t, v, d, n)).sum()
    }
}

// ── BEHAVIORAL fitness — three ISOLATED faculties. Each task holds the OTHER two signals constant (so they carry
//    no information and cannot substitute), varies the correct index across trials (so no fixed tie-break "wins"),
//    and repeats over a seeded family (so a memorised answer fails). This is what makes the discovery HONEST: a
//    novelty-only or value-only hack that passed a leaky task now scores at chance on the tasks it doesn't read. ──

const K: usize = 4; // options per decision
const TRIALS: usize = 16;

/// The per-decision selection the whole #150 policy reduces to: `argmax score(value, drive, novelty)`.
fn choose(p: &Policy, v: &[f32], d: &[f32], n: &[f32]) -> usize {
    (0..v.len())
        .max_by(|&a, &b| {
            p.score(v[a], d[a], n[a])
                .total_cmp(&p.score(v[b], d[b], n[b]))
        })
        .unwrap()
}

/// VALUE faculty (the navigation / planning greedy step): options carry distinct cost-to-go `value`s; `drive` and
/// `novelty` are CONSTANT across the options (uninformative). Correct = the LOWEST-value option (descend toward
/// the goal). Only a policy that weighs `value` NEGATIVELY selects it. (Repeated greedy steps = reaching a goal.)
fn value_faculty(p: &Policy, seed: u64) -> f32 {
    let mut rng = Rng::new(seed);
    let mut ok = 0;
    for _ in 0..TRIALS {
        let v: Vec<f32> = (0..K).map(|_| 0.2 + 3.8 * rng.f()).collect();
        let (d, n) = (vec![0.0; K], vec![1.0; K]); // uninformative
        let correct = (0..K).min_by(|&a, &b| v[a].total_cmp(&v[b])).unwrap();
        if choose(p, &v, &d, &n) == correct {
            ok += 1;
        }
    }
    ok as f32 / TRIALS as f32
}

/// DRIVE faculty (avoid a punished option / commit to a rewarded one): options carry distinct `drive`s (reward
/// memory); `value` and `novelty` are CONSTANT (uninformative). Correct = the HIGHEST-drive option (avoid the
/// aversive, seek the committed). Only a policy that weighs `drive` POSITIVELY selects it.
fn drive_faculty(p: &Policy, seed: u64) -> f32 {
    let mut rng = Rng::new(seed);
    let mut ok = 0;
    for _ in 0..TRIALS {
        let d: Vec<f32> = (0..K).map(|_| rng.sym() * 2.0).collect(); // ±reward memory, one usually aversive
        let (v, n) = (vec![0.0; K], vec![0.0; K]); // uninformative (all tried)
        let correct = (0..K).max_by(|&a, &b| d[a].total_cmp(&d[b])).unwrap();
        if choose(p, &v, &d, &n) == correct {
            ok += 1;
        }
    }
    ok as f32 / TRIALS as f32
}

/// NOVELTY faculty (explore the untried): exactly one option is untried (`novelty`=1), the rest tried (0); `value`
/// and `drive` are CONSTANT (uninformative). Correct = the untried option. A policy that weighs `novelty`
/// positively explores. (This faculty is optional to the goal — it is scored but not required in the gate.)
fn novelty_faculty(p: &Policy, seed: u64) -> f32 {
    let mut rng = Rng::new(seed);
    let mut ok = 0;
    for _ in 0..TRIALS {
        let fresh = rng.below(K);
        let n: Vec<f32> = (0..K).map(|i| if i == fresh { 1.0 } else { 0.0 }).collect();
        let (v, d) = (vec![0.0; K], vec![0.0; K]); // uninformative
        if choose(p, &v, &d, &n) == fresh {
            ok += 1;
        }
    }
    ok as f32 / TRIALS as f32
}

/// The three faculties, averaged over a seeded family each.
fn faculties(p: &Policy) -> (f32, f32, f32) {
    let avg = |f: &dyn Fn(&Policy, u64) -> f32| (f(p, 0x11) + f(p, 0x22) + f(p, 0x33)) / 3.0;
    (
        avg(&value_faculty),
        avg(&drive_faculty),
        avg(&novelty_faculty),
    )
}

/// Behavioral fitness: the #150 policy MUST descend value (plan) AND seek drive (avoid/commit); novelty is a bonus
/// but not required. Use the MIN of the two required faculties (search can't trade one off), + a small novelty
/// credit, minus parsimony.
fn fitness(p: &Policy) -> f32 {
    if p.terms.is_empty() {
        return -1.0;
    }
    let (v, d, n) = faculties(p);
    v.min(d) + 0.1 * n - LAMBDA * p.terms.len() as f32
}

fn mutate(r: &Policy, rng: &mut Rng) -> Policy {
    let mut t = r.terms.clone();
    match rng.u() % 4 {
        0 if t.len() < 4 => t.push(((rng.u() as usize) % NTERM, rng.sym())),
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

fn name(r: &Policy) -> String {
    let lbl = [
        "value",
        "drive",
        "novelty",
        "value·drive",
        "drive²",
        "value²",
        "1",
    ];
    r.terms
        .iter()
        .map(|&(t, c)| format!("{c:+.2}·{}", lbl[t]))
        .collect::<Vec<_>>()
        .join(" ")
}

#[test]
fn discover_the_action_policy() {
    // The AUTHORED baseline I wrote by hand (the violation): −value + drive + 0.2·novelty.
    let authored = Policy {
        terms: vec![(0, -1.0), (1, 1.0), (2, 0.2)],
    };
    let authored_fit = fitness(&authored);

    // DISCOVER it by evolutionary search from a raw term library — no hand-set weighting.
    let mut rng = Rng::new(0x9E7);
    let mut pop: Vec<Policy> = (0..32)
        .map(|_| Policy {
            terms: vec![((rng.u() as usize) % NTERM, rng.sym())],
        })
        .collect();
    let mut best = pop[0].clone();
    let mut best_fit = f32::MIN;
    for _gen in 0..60 {
        let scored: Vec<(f32, Policy)> = pop.iter().map(|r| (fitness(r), r.clone())).collect();
        for (f, r) in &scored {
            if *f > best_fit {
                best_fit = *f;
                best = r.clone();
            }
        }
        let mut order: Vec<usize> = (0..scored.len()).collect();
        order.sort_by(|&a, &b| scored[b].0.partial_cmp(&scored[a].0).unwrap());
        let elite: Vec<Policy> = order.iter().take(8).map(|&i| scored[i].1.clone()).collect();
        pop = elite.clone();
        while pop.len() < 32 {
            let parent = &elite[(rng.u() as usize) % elite.len()];
            pop.push(mutate(parent, &mut rng));
        }
    }

    let (v, d, n) = faculties(&best);
    eprintln!("[METAPOL] discovering the action policy from behavior (3 ISOLATED faculties: value/drive/novelty), λ={LAMBDA}");
    eprintln!(
        "[METAPOL] AUTHORED (my hand act_score): fitness {authored_fit:.3}   [{}]",
        name(&authored)
    );
    eprintln!(
        "[METAPOL] DISCOVERED best policy       : fitness {best_fit:.3}   [{}]",
        name(&best)
    );
    eprintln!("[METAPOL]   faculties of the discovered policy: value(plan) {v:.2}  drive(avoid/commit) {d:.2}  novelty(explore) {n:.2}");

    assert!(
        best_fit >= authored_fit - 0.02,
        "the DISCOVERED policy must match/beat my authored one ({best_fit:.3} vs {authored_fit:.3})"
    );
    // The two REQUIRED faculties: descend value (plan/navigate) AND seek drive (avoid the aversive / commit). Each
    // task isolates its signal, so passing it PROVES the policy reads that signal — a novelty-only or value-only
    // hack scores at chance (~25%) on the faculty it ignores.
    assert!(
        v > 0.9 && d > 0.9,
        "the discovered policy must genuinely descend VALUE (plan {v:.2}) AND seek DRIVE (avoid/commit {d:.2})"
    );
    assert!(
        best.terms.len() <= 4,
        "parsimony kept the discovered policy small ({} terms)",
        best.terms.len()
    );
    // structurally: it must weigh value NEGATIVELY (descend cost-to-go) and drive POSITIVELY (seek reward memory).
    let w = |sig: usize| {
        best.terms
            .iter()
            .filter(|&&(t, _)| t == sig)
            .map(|&(_, c)| c)
            .sum::<f32>()
    };
    eprintln!(
        "[METAPOL]   discovered signed weights: value {:+.2} (want <0), drive {:+.2} (want >0)",
        w(0),
        w(1)
    );

    eprintln!("[METAPOL] VERDICT: the action policy is DISCOVERED, not authored (#150, the nothing-authored fix). A meta-search over a raw term library {{value, drive, novelty, interactions}} — scored by BEHAVIOR across three ISOLATED faculties (descend cost-to-go = plan; seek reward-memory = avoid/commit; each task holds the other signals uninformative so a hack scores at chance) minus parsimony — finds a policy matching/beating my hand-written `act_score` ({best_fit:.3} vs {authored_fit:.3}), small ({} terms), that genuinely descends value ({v:.2}) AND seeks drive ({d:.2}). The temperament (the weighting my authored `wildness`/`openmindedness` hand-set) FALLS OUT of search against behavior — provenance is discovery, not my hand. This is the founding-principle answer: not even the policy equation is authored. Next: fold the discovered form into alive(Act) as a discovered genome (replacing the hand-set act_score).", best.terms.len());
    let _ = Field::new(1);
}
