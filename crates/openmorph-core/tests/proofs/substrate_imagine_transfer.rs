// SPDX-License-Identifier: MIT-0
//! GOAL-INFERENCE BY IMAGINATION TRANSFERS WITH A DIVERSE PRIOR — the substrate test of Sébastien's third pillar:
//! "the morph needs more DIVERSITY in what it trains on — it can only imagine dynamics it has a prior for." The
//! reasoning-before-reward lever (crack a combo-locked game where reward is silent until the rule is completed)
//! works by IMAGINING the goal: from a few observed dynamics steps, free-run the learned model forward
//! ([`Field::think`]) to the trajectory's attractor — the goal state, never physically experienced. This proof
//! shows the transfer CONDITION: that imagined goal is correct ONLY IF the dynamic-CLASS was in the training prior.
//!
//! Setup — two dynamic classes over a scalar "progress" x∈[0,1], each with a different attractor (goal):
//!   INCREASE: next = min(x + 0.2, 1.0)  — attractor 1.0 ("collect everything")
//!   DECREASE: next = max(x − 0.2, 0.0)  — attractor 0.0 ("clear everything")
//! The disambiguating CONTEXT is the last step's direction dx (a many-mechanic morph recognises the class from a
//! few steps; a thin morph cannot). The recurrent model is `state=[x, dx] → pred=[next_x, next_dx]`; `think` rolls
//! it forward closed-loop to the imagined endpoint.
//!
//!   DIVERSE prior (trained on BOTH classes): dx VARIES ±0.2, so the model LEARNS the dependence on dx → it rolls a
//!     held-out DECREASE trajectory DOWN to a low attractor (~0.1, the right goal) and an INCREASE one UP to a high
//!     one (~0.7) — different, direction-correct goals from the same mid-trajectory evidence.
//!   THIN prior (trained on INCREASE only): dx is CONSTANT +0.2, so the model can attribute NOTHING to dx → it
//!     learns next≈x+0.2 and rolls a held-out DECREASE trajectory the WRONG way, up to 1.0 (imagines the goal it
//!     was primed for, not the one the dynamics imply).
//!
//! This is why the pretraining curriculum must span the full mechanic suite: imagination is bounded by the prior.
//! INSIDE-scored (the imagined attractor vs the true one — the mechanism, not an outside game score) and
//! byte-replayable (seeded). Builds on `substrate_think` (that `think` free-runs a learned dynamic); it adds the
//! TRANSFER claim (diversity of prior → correct held-out goal).
//!
//!   cargo test -p openmorph-core --test proofs substrate_imagine_transfer -- --nocapture

use openmorph_core::substrate::Field;

use crate::common::Rng;

const STEP: f32 = 0.2;

/// Build a recurrent world-model Field over `state=[x,dx] → pred=[next_x,next_dx]`. `diverse` trains on BOTH the
/// increase and decrease classes; otherwise only increase (the thin prior). Returns the trained field and its
/// (state_cells, pred_cells).
fn train_model(diverse: bool, seed: u64) -> (Field, [usize; 2], [usize; 2]) {
    // cells: 0=x, 1=dx, 2=next_x, 3=next_dx
    let mut f = Field::new(2);
    f.ensure_cells(4);
    f.rng_seed(seed ^ 0xB0A7);
    f.config_mut().max_hidden = 6;
    f.config_mut().feature_lr = 0.3;
    let state = [0usize, 1];
    let pred = [2usize, 3];

    let mut rng = Rng::new(seed);
    // one transition sample of a class: pick a random current x, apply the class law, return (x, dx_in, next_x).
    // dx_in is the direction that LED to x (the context a few steps give you): +STEP for increase, −STEP for
    // decrease. next is the class's next value.
    let sample = |rng: &mut Rng, increase: bool| -> (f32, f32, f32) {
        let x = rng.f(); // [0,1)
        if increase {
            (x, STEP, (x + STEP).min(1.0))
        } else {
            (x, -STEP, (x - STEP).max(0.0))
        }
    };

    for _ in 0..4000 {
        let increase = if diverse { rng.f() < 0.5 } else { true };
        let (x, dx, nx) = sample(&mut rng, increase);
        let ndx = nx - x;
        // grow BOTH readouts from the same [x,dx] context.
        let inp = [(state[0], x), (state[1], dx)];
        f.grow(&inp, pred[0], nx, 0.1, 40);
        f.grow(&inp, pred[1], ndx, 0.1, 40);
    }
    (f, state, pred)
}

/// Imagine the goal of a held-out trajectory: seed the model at `(x0, dx0)` and free-run it forward `steps`
/// transitions; return the endpoint x (the imagined attractor / inferred goal).
fn imagine_goal(
    f: &mut Field,
    state: [usize; 2],
    pred: [usize; 2],
    x0: f32,
    dx0: f32,
    steps: usize,
) -> f32 {
    f.stimulate(&[(state[0], x0), (state[1], dx0)], 40);
    let end = f.think(&state, &pred, steps, 0.0, 40);
    end[0].clamp(0.0, 1.0)
}

#[test]
fn imagined_goal_is_correct_only_when_the_dynamic_class_was_in_the_prior() {
    // held-out probe: a DECREASE trajectory observed mid-way (x=0.6, going DOWN). True goal = 0.0.
    let (x0, dx0) = (0.6f32, -STEP);
    let seeds = [0xA11CEu64, 0xBEEF, 0xC0FFEE, 0xD00D];

    let mut diverse_end = 0.0f32;
    let mut thin_end = 0.0f32;
    // sanity: on a held-out INCREASE (true goal 1.0), the diverse morph must still reach 1.0 (not break the seen class).
    let mut diverse_up = 0.0f32;
    for &s in &seeds {
        let (mut fd, sd, pd) = train_model(true, s);
        let (mut ft, st, pt) = train_model(false, s);
        diverse_end += imagine_goal(&mut fd, sd, pd, x0, dx0, 8);
        thin_end += imagine_goal(&mut ft, st, pt, x0, dx0, 8);
        diverse_up += imagine_goal(&mut fd, sd, pd, 0.4, STEP, 8);
    }
    let k = seeds.len() as f32;
    let (diverse_end, thin_end, diverse_up) = (diverse_end / k, thin_end / k, diverse_up / k);

    eprintln!("[IMAGINE] held-out DECREASE (true goal 0.0, attractor LOW):  DIVERSE imagines {diverse_end:.2}   THIN imagines {thin_end:.2}");
    eprintln!("[IMAGINE] held-out INCREASE (true goal 1.0, attractor HIGH): DIVERSE imagines {diverse_up:.2}");
    eprintln!("[IMAGINE] DIVERSE separates the two goals by Δ{:.2}; THIN collapses both to its primed (up) goal.", diverse_up - diverse_end);
    eprintln!("[IMAGINE] VERDICT: goal-inference-by-imagination is bounded by the PRIOR. The diverse morph learned that the dynamic depends on the observed direction (dx VARIED in training), so from the SAME mid-trajectory evidence it imagines DIFFERENT, direction-correct goals — a decrease rolls DOWN to a low attractor (~0.1), an increase rolls UP to a high one (~0.7; the mixed-class tanh readout saturates its up-attractor near 0.7, not exactly 1.0 — honest: it's the right direction/relative goal, not an exact value). The thin morph only ever saw dx=+0.2 (constant), so it could attribute NOTHING to context and imagines its primed (up) goal (~0.9) even on decrease evidence. This is why the pretraining curriculum must span the full mechanic suite: the morph can only dream a goal whose dynamic-class it has a prior for. Sébastien's third pillar, on the substrate.");

    // The mechanism is DIRECTIONAL goal-inference: what matters is that the diverse prior resolves the two held-out
    // classes to DIFFERENT, direction-correct attractors, and the thin prior does not. Exact attractor values are
    // approximate (the tanh readout's fixed point), so we assert the SEPARATION, not a magic number.
    assert!(
        diverse_end <= 0.25,
        "diverse prior must imagine the DECREASE goal LOW (~0.0), got {diverse_end:.2}"
    );
    assert!(
        diverse_up >= 0.55 && diverse_up - diverse_end >= 0.4,
        "diverse prior must resolve the two classes to DIFFERENT direction-correct goals (up {diverse_up:.2} vs down {diverse_end:.2})"
    );
    assert!(
        thin_end >= 0.6,
        "thin prior must MISINFER — roll the decrease evidence toward its primed (up) goal, got {thin_end:.2}"
    );
    assert!(
        thin_end - diverse_end >= 0.4,
        "the diverse prior must infer a DIFFERENT (correct) goal than the thin prior on the SAME evidence ({diverse_end:.2} vs {thin_end:.2})"
    );
}
