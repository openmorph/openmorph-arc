// SPDX-License-Identifier: MIT-0
//! IDENTIFYING CAUSALITY — grow the state until the world is predictable (MorphLab #115). A cause needs no
//! new vocabulary: a cause is a feature that, TOGETHER WITH THE ACTION, makes the effect predictable. So a
//! sufficient (Markov) state — one where the same state + action always gives the same outcome — is the SAME
//! thing as a causally-creditable state. When perception is LOSSY (an aggregate that collapses genuinely
//! different inputs onto one key), the dynamics look non-deterministic: the same perceived state + action
//! yields different outcomes, reward is credited to a state that did not cause it, and the world model can't
//! predict. The fix is the substrate's own residual-birth: predict the outcome from the RAW input; where the
//! residual is high the state is ambiguous; `grow_deep_step` BIRTHS a feature that splits it — the state
//! refines until the outcome is a function of it. Nothing authored: the birth is driven by the residual.
//!
//! Acceptance is STATE-DETERMINISM (domain-blind, inside-scored): the fraction of (state, action) pairs with
//! a UNIQUE outcome must climb toward 1 as the state grows. Here a lossy aggregate is 50% (aliased); the
//! sufficient state is 100%; and `grow_deep_step` over the raw input recovers the discriminating detail the
//! aggregate discarded — predicting the outcome the lossy state cannot — the causality faculty the primitives
//! already imply.
//!
//!   cargo test -p openmorph-core --test proofs substrate_causal -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;
use std::collections::HashMap;

use crate::common::Rng;

// The world's TRUE dynamics: the action SELECTS which feature is the cause. `action > 0` ⇒ the outcome is
// the hidden `b`; `action < 0` ⇒ the outcome is `a`. A state that keeps only `a` is therefore Markov half
// the time (action < 0) and ALIASED the other half (action > 0, where the outcome is the dropped `b`) —
// exactly the partial-non-determinism MorphLab measured on real ARC-3. `c` is an irrelevant distractor.
fn outcome(a: f32, b: f32, action: f32) -> f32 {
    if action > 0.0 {
        b // the cause is the hidden b
    } else {
        a // the cause is a — the lossy state is sufficient here
    }
}

/// STATE-DETERMINISM (MorphLab #115 acceptance metric) — the fraction of (state-key, action) groups whose
/// outcome is UNIQUE. 1.0 ⇒ the state is Markov (causally sufficient); < 1 ⇒ aliased (a hidden cause). This
/// is domain-blind: it takes only quantized (state, action) → outcome observations.
fn state_determinism(obs: &[(u64, i64, i64)]) -> f64 {
    let mut groups: HashMap<(u64, i64), Vec<i64>> = HashMap::new();
    for &(s, a, o) in obs {
        groups.entry((s, a)).or_default().push(o);
    }
    let unique = groups
        .values()
        .filter(|outs| outs.iter().all(|&o| o == outs[0]))
        .count();
    unique as f64 / groups.len().max(1) as f64
}

fn key2(a: f32, b: f32) -> u64 {
    ((a > 0.0) as u64) | (((b > 0.0) as u64) << 1)
}

#[test]
fn growing_the_state_until_predictable_makes_the_dynamics_markov() {
    let mut rng = Rng::new(0xCA05A1);

    // ── 1. STATE-DETERMINISM detects the non-Markov (aliased) state. ──
    let mut lossy_obs = Vec::new(); // state key = a only (the lossy aggregate drops b)
    let mut suff_obs = Vec::new(); // state key = (a, b) — the sufficient state
    for _ in 0..4000 {
        let (a, b, action) = (rng.bit(), rng.bit(), rng.bit());
        let o = outcome(a, b, action) as i64;
        let ak = (action > 0.0) as i64;
        lossy_obs.push(((a > 0.0) as u64, ak, o));
        suff_obs.push((key2(a, b), ak, o));
    }
    let lossy_det = state_determinism(&lossy_obs);
    let suff_det = state_determinism(&suff_obs);

    // ── 2. The RESIDUAL is the detector; `grow_deep_step` over the RAW input is the fix. ──
    // LOSSY predictor: sees only (a, action) — cannot predict the b-dependent outcome.
    let mut lossy = Field::new(2);
    lossy.rng_seed(0x1005);
    lossy.add_unit(vec![0, 1], vec![0.0, 0.0], 0.0, 2);
    // GROWN predictor: sees the RAW input (a, b, c, action) and grows the depth the product needs.
    let mut grown = Field::new(4);
    grown.rng_seed(0x9701);
    grown.config_mut().birth_gate = 0.3;
    grown.config_mut().max_hidden = 36;
    let gout = 4;

    let mut rng = Rng::new(0x5EED);
    for _ in 0..32000 {
        let (a, b, c, action) = (rng.bit(), rng.bit(), rng.bit(), rng.bit());
        let o = outcome(a, b, action);
        lossy.learn_step(&[(0, a), (1, action)], 2, o, 0.1, 300);
        grown.grow(&[(0, a), (1, b), (2, c), (3, action)], gout, o, 0.1, 300);
    }
    let (mut lossy_ok, mut grown_ok, mut tot) = (0.0f64, 0.0, 0.0);
    for _ in 0..2000 {
        let (a, b, c, action) = (rng.bit(), rng.bit(), rng.bit(), rng.bit());
        let o = outcome(a, b, action);
        lossy.stimulate(&[(0, a), (1, action)], 300);
        if lossy.cells()[2].signum() == o {
            lossy_ok += 1.0;
        }
        grown.stimulate(&[(0, a), (1, b), (2, c), (3, action)], 300);
        if grown.cells()[gout].signum() == o {
            grown_ok += 1.0;
        }
        tot += 1.0;
    }
    let (lossy_acc, grown_acc) = (lossy_ok / tot, grown_ok / tot);

    eprintln!(
        "[CAUSAL] identify causality = grow the state until predictable (MorphLab #115):\n  STATE-DETERMINISM   lossy (aggregate, drops the cause) {lossy_det:.2}  vs  sufficient (a,b) {suff_det:.2}\n  PREDICTS the outcome lossy {:.0}%  vs  grown-from-raw {:.0}%",
        lossy_acc * 100.0,
        grown_acc * 100.0,
    );
    eprintln!("[CAUSAL] VERDICT: a cause is a feature that, with the action, makes the effect predictable — so a Markov state IS a causally-creditable state, and the substrate grows it the same way it grows everything: by residual-birth. State-determinism (domain-blind, inside-scored) detects the aliasing — a lossy aggregate that drops the cause sits at 50%, the sufficient state at 100%. The lossy predictor cannot beat the aliasing; `grow_deep_step` over the RAW input recovers the discriminating detail the aggregate discarded and predicts the outcome — growing the sufficient state, non-authored. Credit assignment, value, and the world model are then all well-posed, in every modality. OPEN (the temporal half): when the disambiguator is in the PAST (true hidden state), the residual must grow MEMORY, not a spatial feature — the recurrent case.");

    // The lossy state aliases ~half the (state,action) groups (the action>0 half where b matters); the
    // sufficient state is fully Markov. The lossy predictor is capped by the aliasing (~75%: right on the
    // action<0 half, chance on the aliased half); growing from raw recovers the hidden b and beats it.
    assert!(
        lossy_det < 0.7 && lossy_det > 0.3,
        "the lossy aggregate must alias about half its groups (non-Markov) — determinism {lossy_det:.2}"
    );
    assert!(
        suff_det > 0.99,
        "the sufficient state must be Markov — determinism {suff_det:.2}"
    );
    assert!(
        lossy_acc < 0.82,
        "the lossy predictor is capped by the aliasing (got {lossy_acc:.2})"
    );
    assert!(
        grown_acc > lossy_acc + 0.12 && grown_acc > 0.9,
        "growing the state from raw must recover predictability the lossy state cannot (grown {grown_acc:.2} vs lossy {lossy_acc:.2})"
    );
}
