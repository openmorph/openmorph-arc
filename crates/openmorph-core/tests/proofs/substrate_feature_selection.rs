// SPDX-License-Identifier: MIT-0
//! GROWN FEATURE-SELECTION — the organism DISCOVERS which abstraction it needs, by its OWN reward.
//!
//! Rungs 1-3 (`substrate_free_energy_memory`, `substrate_free_energy_abstraction`) proved that AUGMENTING the
//! reactive state with a grown abstraction — `delay` (memory), `aggregate` (counting), `relate` (comparison) —
//! lets the shipped free-energy actor solve a rule the reactive state CAN'T. But which abstraction to add was
//! HAND-PICKED per task (`if task == memory { use delay }`). This rung removes the hand: presented with a task
//! that needs ONE of {memory, counting, comparison}, the organism finds WHICH abstraction makes it representable
//! WITHOUT being told — from its own reward signal.
//!
//! THE MECHANISM (non-authored — no `if task == …`). One BANK of candidate state-augmentations, each running the
//! SAME shipped free-energy actor (`alive(Event::Act)`, `free_energy_actor`) over its own state key:
//!   - `Reactive`  — the neutral current observation (one state → chance on all three; the null).
//!   - `Delay`     — a `delay`-chain recall of the cue channel (grown exact MEMORY).
//!   - `Aggregate` — `aggregate(set).count` parity (grown COUNT).
//!   - `Relate`    — `relate(A, B).same_color` (grown COMPARISON relation).
//!
//! Every candidate sees EVERY scene (each scene always carries a cue, a set, and a colour pair) — a candidate
//! cannot know a priori that its channel is the irrelevant one; only the reward tells. During a warm-up each
//! candidate's actor learns and BANKS its mean earned reward. Selection is the substrate's own competitive
//! `Field::select` over those banked rewards (a winner-take-all read of gate cells) — the candidate whose grown
//! abstraction made the rule representable earns ~1.0 and wins; the mismatched channels are coin-flips at ~0.5.
//! The organism keeps the winning augmentation, then solves the held-out task at that augmented level.
//!
//! Truth-not-trophy, for EACH of the three task types:
//!   1) the selector picks the CORRECT abstraction (chosen index = delay / aggregate / relate) — MEASURED, from
//!      reward, not authored;
//!   2) the SELECTED candidate solves the held-out task (> 0.90);
//!   3) the WRONG fixed candidates stay at chance (< 0.65) — including the reactive null — so selection is doing
//!      real work; the right representation is found, not handed.
//!
//!   cargo test -p openmorph-core --test proofs substrate_feature_selection -- --nocapture

use openmorph_core::grouping::{aggregate, relate, Object};
use openmorph_core::{Config, Event, Field, Outcome};

const TRAIN: usize = 1_500;
const EVAL: usize = 400;
const D: usize = 3; // the memory cue is hidden for D steps before it is recalled

/// The candidate bank — each is a state-augmentation the organism could add. Order IS the reported index.
const REACTIVE: usize = 0;
const DELAY: usize = 1;
const AGGREGATE: usize = 2;
const RELATE: usize = 3;
const N_CAND: usize = 4;
const CAND_NAME: [&str; N_CAND] = [
    "reactive",
    "delay(memory)",
    "aggregate(count)",
    "relate(compare)",
];

/// The three task types. Each names the abstraction that makes it representable — the answer the selector must
/// discover on its own (this mapping is used ONLY to score the choice, never to make it).
#[derive(Clone, Copy)]
enum Task {
    Memory,
    Counting,
    Comparison,
}
impl Task {
    fn correct_candidate(self) -> usize {
        match self {
            Task::Memory => DELAY,
            Task::Counting => AGGREGATE,
            Task::Comparison => RELATE,
        }
    }
}

/// Deterministic xorshift → unit float (no wall-clock; the whole proof is byte-replayable).
fn u01(s: &mut u64) -> f32 {
    let mut x = *s | 1;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *s = x;
    (x >> 11) as f32 / (1u64 << 53) as f32
}

/// One scene — ALWAYS carries all three raw channels (a cue, a set, a colour pair) so every candidate is
/// computable. Only one of them is predictive of a given task's target; the draws are independent, so the
/// other two channels are pure coin-flips relative to the target.
struct Scene {
    cue: usize, // {0,1} — visible at t=0 then hidden (drives Delay)
    n: usize,   // set cardinality 2..=5 (drives Aggregate)
    ca: usize,  // colour of object A {0,1} (drives Relate)
    cb: usize,  // colour of object B {0,1}
}

fn scene(s: &mut u64) -> Scene {
    let cue = usize::from(u01(s) < 0.5);
    let n = 2 + (u01(s) * 4.0) as usize % 4;
    let ca = usize::from(u01(s) < 0.5);
    let cb = usize::from(u01(s) < 0.5);
    Scene { cue, n, ca, cb }
}

/// The target for a task — a deterministic function of the ONE predictive channel.
fn target(task: Task, sc: &Scene) -> usize {
    match task {
        Task::Memory => sc.cue,                                // recall the past cue
        Task::Counting => usize::from(sc.n.is_multiple_of(2)), // cardinality parity
        Task::Comparison => usize::from(sc.ca == sc.cb),
    }
}

/// A candidate actor: the shipped free-energy `Field`, plus (for `Delay`) the cue channel + the last delay cell.
struct Candidate {
    kind: usize,
    field: Field,
    cue_src: usize,
    last_delay: usize,
}

fn fe_field(n_cells: usize) -> Field {
    let mut f = Field::new(n_cells);
    f.configure(Config {
        free_energy_actor: true,
        free_energy_sstar: 1.0,
        ..Config::default()
    });
    f
}

impl Candidate {
    fn new(kind: usize) -> Self {
        if kind == DELAY {
            // A cue channel + a delay chain of length D: the last cell emits x[t−D] = the past cue.
            let mut field = fe_field(1);
            let cue_src = 0usize;
            let mut src = cue_src;
            for _ in 0..D {
                src = field.delay(src);
            }
            Candidate {
                kind,
                field,
                cue_src,
                last_delay: src,
            }
        } else {
            Candidate {
                kind,
                field: fe_field(1),
                cue_src: 0,
                last_delay: 0,
            }
        }
    }

    /// This candidate's state key for a scene — the grown abstraction it proposes (or the reactive null).
    fn state(&mut self, sc: &Scene) -> usize {
        match self.kind {
            REACTIVE => 0, // the current observation is neutral at decision → one indistinct state
            DELAY => {
                // Present the cue, then advance the register through the D hidden steps; read the recall.
                let mut mem = 0.0;
                for t in 0..=D {
                    let obs = if t == 0 { sc.cue as f32 } else { 0.5 };
                    self.field.stimulate(&[(self.cue_src, obs)], 1);
                    let updates = self.field.advance_delays();
                    mem = updates
                        .iter()
                        .find(|(cell, _)| *cell == self.last_delay)
                        .map(|(_, v)| *v)
                        .unwrap_or(0.0);
                }
                usize::from(mem > 0.5)
            }
            AGGREGATE => {
                let vals: Vec<f32> = (0..sc.n).map(|_| 1.0).collect(); // a set of cardinality n
                (aggregate(&vals).count as usize) % 2 // grown COUNT parity
            }
            RELATE => {
                let objs = [
                    Object {
                        cells: vec![0, 1],
                        color: sc.ca,
                        bbox_w: 2,
                        bbox_h: 1,
                    },
                    Object {
                        cells: vec![4, 5],
                        color: sc.cb,
                        bbox_w: 2,
                        bbox_h: 1,
                    },
                ];
                usize::from(relate(&objs, 8, 8)[0].same_color) // grown RELATION
            }
            _ => unreachable!(),
        }
    }

    /// Select a binary action for the given state under the free-energy actor, folding the previous reward.
    fn act(&mut self, s: usize, last: Option<(usize, f32)>) -> usize {
        let opts = [s * 2, s * 2 + 1];
        match self.field.alive(Event::Act {
            options: &opts,
            last,
        }) {
            Outcome::Acted { chosen } => chosen % 2,
            _ => unreachable!(),
        }
    }
}

/// Run the full bank on a task: warm-up-train every candidate, BANK its mean earned reward, then measure each
/// candidate's held-out accuracy. Returns (mean-reward per candidate, held-out accuracy per candidate).
fn run_bank(task: Task, train_seed: u64, eval_seed: u64) -> ([f32; N_CAND], [f32; N_CAND]) {
    let mut cands: Vec<Candidate> = (0..N_CAND).map(Candidate::new).collect();
    let mut earned = [0.0f32; N_CAND];
    let mut last: [Option<(usize, f32)>; N_CAND] = [None; N_CAND];

    // Warm-up: each candidate's actor learns over its own state key; we bank its earned reward (its own signal).
    let mut ts = train_seed;
    for _ in 0..TRAIN {
        let sc = scene(&mut ts);
        let tgt = target(task, &sc);
        for k in 0..N_CAND {
            let s = cands[k].state(&sc);
            let a = cands[k].act(s, last[k]);
            let r = if a == tgt { 1.0 } else { 0.0 };
            earned[k] += r;
            last[k] = Some((s * 2 + a, r));
        }
    }
    for e in &mut earned {
        *e /= TRAIN as f32;
    }

    // Held-out: each candidate's now-frozen policy on a DIFFERENT seed stream.
    let mut es = eval_seed;
    let mut correct = [0usize; N_CAND];
    for _ in 0..EVAL {
        let sc = scene(&mut es);
        let tgt = target(task, &sc);
        for k in 0..N_CAND {
            let s = cands[k].state(&sc);
            let a = cands[k].act(s, None);
            if a == tgt {
                correct[k] += 1;
            }
        }
    }
    let mut acc = [0.0f32; N_CAND];
    for k in 0..N_CAND {
        acc[k] = correct[k] as f32 / EVAL as f32;
    }
    (earned, acc)
}

/// SELECTION by the substrate's own competitive read: write each candidate's banked reward into a gate cell and
/// let `Field::select` pick the winner (winner-take-all over the gates) — the same primitive that routes to the
/// competent module. No authored argmax; the reward IS the gate.
fn select_by_reward(earned: &[f32; N_CAND]) -> usize {
    let mut sel = Field::new(N_CAND);
    let gates: Vec<usize> = (0..N_CAND).collect();
    sel.stimulate(
        &earned
            .iter()
            .enumerate()
            .map(|(i, &r)| (i, r))
            .collect::<Vec<_>>(),
        1,
    );
    sel.select(&gates)
}

fn run_task(task: Task, name: &str, train_seed: u64, eval_seed: u64) {
    let (earned, acc) = run_bank(task, train_seed, eval_seed);
    let chosen = select_by_reward(&earned);
    let want = task.correct_candidate();

    println!("\n=== {name} ===");
    for k in 0..N_CAND {
        let mark = if k == chosen { " <== SELECTED" } else { "" };
        println!(
            "  {:<18} banked-reward {:.3}  held-out {:.3}{mark}",
            CAND_NAME[k], earned[k], acc[k]
        );
    }
    println!("  chose {} (want {})", CAND_NAME[chosen], CAND_NAME[want]);

    // 1) The organism picked the CORRECT abstraction from reward alone.
    assert_eq!(
        chosen, want,
        "{name}: selector must choose {} (chose {})",
        CAND_NAME[want], CAND_NAME[chosen]
    );
    // 2) The selected candidate SOLVES the held-out task.
    assert!(
        acc[chosen] > 0.90,
        "{name}: selected candidate must solve the task (got {:.3})",
        acc[chosen]
    );
    // 3) Every WRONG candidate — including the reactive null — stays at chance (selection is load-bearing).
    for k in 0..N_CAND {
        if k != want {
            assert!(
                acc[k] < 0.65,
                "{name}: wrong candidate {} must stay near chance (got {:.3})",
                CAND_NAME[k],
                acc[k]
            );
        }
    }
}

#[test]
fn the_organism_selects_the_needed_abstraction_from_its_own_reward() {
    run_task(Task::Memory, "MEMORY (needs delay)", 0x5151, 0x9090);
    run_task(
        Task::Counting,
        "COUNTING (needs aggregate)",
        0xC0117,
        0x7A11,
    );
    run_task(
        Task::Comparison,
        "COMPARISON (needs relate)",
        0xC0FFEE,
        0x5A0E,
    );
}
