// SPDX-License-Identifier: MIT-0
//! PURPOSE — active inference: a morph that ACTS because it wants the world to be a certain way. The wall
//! behind ARC-3 was that the morph was GOALLESS — it predicted the world beautifully but nothing drove it to
//! SOLVE. A living entity has a PREFERRED STATE (a prior over how the world should be) and acts to bring the
//! world into it: it minimises EXPECTED FREE ENERGY — the predicted surprise of each candidate action against
//! the preferred state — over its OWN LEARNED world model. Perception was already here (`observe`/`grow` +
//! surprise); purpose is the preferred prior plus the will to act toward it.
//!
//! Everything is scored INTRINSICALLY — free energy, the divergence between the world and the preferred state
//! (mean |s − s*|) — never an external accuracy label. That is a LIFE metric (is the world how I need it?) and
//! it is dense every moment, which dissolves sparse reward at the root.
//!
//! Truth-not-trophy is built into the world: the sign of the action→effect gain `k` is RANDOMISED and UNKNOWN
//! to the morph, and there is a model-SCRAMBLE control. So the ablation proves the winner is genuinely PLANNING
//! over a learned model, not running a lucky controller:
//!   GOALLESS ......... predicts the world, has no preference → never acts → never solves.
//!   NAIVE controller . hard-coded "move toward the target" (assumes k>0) → catastrophic when the sign flips.
//!   ACTIVE INFERENCE . picks the action whose PREDICTED next state is closest to the preferred → solves on
//!                      BOTH signs, because it selects by the LEARNED model.
//!   SCRAMBLED ........ active inference over an action-BLIND model (the model can't tell the actions apart) →
//!                      collapses back to goalless: no valid model ⇒ no purposeful action. The model IS the load.
//!
//! Two new pieces only (the rest is assembled over existing primitives): a preferred-state prior + the
//! expected-free-energy action rule. Usage protocol today; the `act` step of the alive loop tomorrow.
//!
//!   cargo test -p openmorph-core --test proofs substrate_purpose -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;

use crate::common::Rng;

const SSTAR: f32 = 0.5; // the preferred sensor value — the morph's prior over "how the world should be"
const ACTIONS: [f32; 3] = [-1.0, 0.0, 1.0];
const CLAMP: f32 = 0.8; // keep the sensor off the tanh asymptote

/// A controllable 1-D world `s' = clamp(s + k·a + noise)`. `k`'s SIGN is unknown to the morph; it must be
/// learned. `mode` selects how the morph chooses each action; the world model is grown online in every mode.
fn run(mode: &str, k: f32, seed: u64) -> f32 {
    let mut wm = Field::new(2);
    let out = wm.len();
    wm.ensure_cells(out + 1);
    wm.rng_seed(seed);
    wm.config_mut().max_hidden = 12;
    wm.config_mut().feature_lr = 0.3;
    // action-BLIND for the scramble control: the model never sees which action was taken.
    let ablind = mode == "scrambled";
    let ain =
        |s: f32, a: f32| -> Vec<(usize, f32)> { vec![(0, s), (1, if ablind { 0.0 } else { a })] };

    let mut rng = Rng::new(seed);
    let mut s = 0.0f32;
    let (mut err, mut cnt) = (0.0f32, 0usize);
    for step in 0..600 {
        let a = match mode {
            "goalless" => 0.0, // predicts the world but has NO preference → no reason to act
            "naive" => {
                if s < SSTAR {
                    1.0
                } else {
                    -1.0
                }
            } // hard-coded "move up toward the target" — assumes k>0
            _ => {
                // active inference (and its scrambled control): argmin over actions of the predicted surprise
                // against the preferred state, using the morph's OWN world model.
                *ACTIONS
                    .iter()
                    .min_by(|&&x, &&y| {
                        wm.stimulate(&ain(s, x), 40);
                        let px = (wm.cells()[out] - SSTAR).abs();
                        wm.stimulate(&ain(s, y), 40);
                        let py = (wm.cells()[out] - SSTAR).abs();
                        px.total_cmp(&py)
                    })
                    .unwrap()
            }
        };
        let noise = (rng.f() - 0.5) * 0.05;
        let s_next = (s + k * a + noise).clamp(-CLAMP, CLAMP);
        // learn the world model from the real transition (the model is GROWN, never given)
        for _ in 0..3 {
            wm.grow(&ain(s, a), out, s_next, 0.1, 40);
        }
        s = s_next;
        if step >= 300 {
            err += (s - SSTAR).abs(); // intrinsic free energy vs the preferred state
            cnt += 1;
        }
    }
    err / cnt as f32
}

fn avg(mode: &str, k: f32, seeds: u64) -> f32 {
    (0..seeds).map(|s| run(mode, k, 0x9E + s)).sum::<f32>() / seeds as f32
}

#[test]
fn purpose_makes_a_morph_act_where_a_goalless_one_will_not() {
    let seeds = 6;
    // two worlds: the action→effect sign is positive in one, negative in the other, unknown to the morph.
    let (gp, np, ap, sp) = (
        avg("goalless", 0.3, seeds),
        avg("naive", 0.3, seeds),
        avg("active", 0.3, seeds),
        avg("scrambled", 0.3, seeds),
    );
    let (gn, nn, an, sn) = (
        avg("goalless", -0.3, seeds),
        avg("naive", -0.3, seeds),
        avg("active", -0.3, seeds),
        avg("scrambled", -0.3, seeds),
    );

    eprintln!("[PURPOSE] intrinsic free energy = mean |s − s*| over the 2nd half (lower = the world is how the morph prefers it):");
    eprintln!("[PURPOSE]              goalless   naive-controller   active-inference   scrambled(model-blind)");
    eprintln!(
        "[PURPOSE]   k > 0 :     {gp:.3}         {np:.3}              {ap:.3}              {sp:.3}"
    );
    eprintln!(
        "[PURPOSE]   k < 0 :     {gn:.3}         {nn:.3}              {an:.3}              {sn:.3}"
    );
    eprintln!("[PURPOSE] VERDICT: PURPOSE is a preferred state + the will to act toward it. A GOALLESS morph predicts the world ({gp:.2}) but never acts on it — exactly the wall behind ARC-3. Giving it a preferred state and letting it minimise EXPECTED FREE ENERGY (the predicted surprise of each action vs the preference) over its OWN learned world model drives the world into the preferred state on BOTH signs ({ap:.2}/{an:.2}) — a {:.0}× reduction. Truth-not-trophy: it is genuinely MODEL-BASED planning, not a controller — the hard-coded NAIVE rule works when its sign assumption holds ({np:.2}) but is CATASTROPHIC when the sign flips ({nn:.2}), and a model-BLIND agent (can't tell the actions apart) collapses back to goalless ({sp:.2}/{sn:.2}). No valid model ⇒ no purposeful action. Scored intrinsically (free energy, dense every moment — no sparse reward), which is how a LIVING entity measures success. Two new pieces only: the preferred prior + the free-energy action rule; the world model, surprise, and the energy homeostat already exist. The ARC-3 bridge: preferred = a consistent/solved grid, action = produce the answer, free energy = its inconsistency — the drive the goalless morph lacked.", gp / ap.max(1e-3));

    // purpose makes it act — active inference reaches the preferred state on both signs, crushing goalless
    assert!(
        ap < 0.15 && an < 0.15,
        "active inference must reach the preferred state on both signs ({ap:.3}/{an:.3})"
    );
    assert!(
        ap < gp * 0.4 && an < gn * 0.4,
        "purpose must crush goalless (act vs never-act)"
    );
    // model-based, not a controller — the naive rule is catastrophic when the action-effect sign flips
    assert!(nn > 0.5 && nn > an * 4.0, "the naive controller must fail on the flipped sign where active inference succeeds ({nn:.3} vs {an:.3})");
    // the model IS the load — an action-blind model collapses purpose back to goalless
    assert!(sp > ap * 3.0 && sn > an * 3.0, "a scrambled (action-blind) model must collapse purpose back toward goalless ({sp:.3}/{sn:.3} vs {ap:.3}/{an:.3})");
}
