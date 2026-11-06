// SPDX-License-Identifier: MIT-0
//! SNAPSHOT → LOAD → CONTINUE ≡ CONTINUE — the byte-replay moat carried ACROSS a save. A `Field` persists by
//! `serde` (JSON for the bindings/perception, bincode for the onnx `.morph`); a loaded organism must then live
//! EXACTLY the life the saved one goes on to live. Every state that shapes a FUTURE step must therefore cross the
//! boundary — the free-cell pool a `cull` refills, the surprise-driven plasticity controller, a trace's half-life
//! sensitivity, a resonator's frequency sensitivity, the product co-credit, the fast-weight rule, the dwell cursor, a
//! nested cell's memo — and only pure caches/scratch/diagnostics may stay behind.
//!
//! For the default path and each stateful opt-in lever: run `T` steps, snapshot through every wire format (JSON, bincode, the versioned envelope), load, run
//! `M` more identical steps on the original and on each loaded copy, and require every cell value, every unit's
//! reads/weights/bias, every per-step observable, AND the final serialized bytes to be `.to_bits()`-identical.
//!
//!   cargo test -p openmorph-core --release --test proofs substrate_snapshot_continue -- --nocapture

use openmorph_core::substrate::{Event, Field, Outcome};
use openmorph_core::Config;

/// Every value a caller can SEE after a step: each cell, and each unit's write/reads/weights/bias — as raw bits.
fn state_bits(f: &Field) -> Vec<u32> {
    let mut v: Vec<u32> = f.cells().iter().map(|x| x.to_bits()).collect();
    v.push(u32::MAX); // separator: cells | units
    for u in &f.units {
        v.push(u.write as u32);
        v.extend(u.reads.iter().map(|&c| c as u32));
        v.extend(u.weights.iter().map(|w| w.to_bits()));
        v.push(u.bias.to_bits());
    }
    v
}

fn bits(x: f32) -> u32 {
    x.to_bits()
}

/// Run `t` steps, snapshot (JSON + bincode + envelope), load, then run `m` more steps on the original and every loaded copy.
/// `S` is the caller's own per-organism driver state (e.g. the last action), cloned at the snapshot so each copy
/// continues from the same point. Returns a description of every divergence (empty ⇒ replay holds).
fn divergence<S: Clone>(
    mut f: Field,
    s0: S,
    t: usize,
    m: usize,
    step: impl Fn(&mut Field, &mut S, usize) -> Vec<u32>,
) -> Vec<String> {
    let mut s = s0;
    for i in 0..t {
        step(&mut f, &mut s, i);
    }
    let json = serde_json::to_vec(&f).expect("Field serializes (JSON)");
    let bin = bincode::serialize(&f).expect("Field serializes (bincode)");
    let loaded: Vec<(&str, Field)> = vec![
        (
            "json",
            serde_json::from_slice(&json).expect("Field loads (JSON)"),
        ),
        (
            "bincode",
            bincode::deserialize(&bin).expect("Field loads (bincode)"),
        ),
        (
            "envelope",
            Field::load(&f.snapshot()).expect("Field loads (envelope)"),
        ),
    ];
    let mut copies: Vec<(&str, Field, S, Vec<u32>)> = loaded
        .into_iter()
        .map(|(n, g)| {
            g.validate().expect("a self-produced snapshot validates");
            (n, g, s.clone(), Vec::new())
        })
        .collect();
    let mut obs = Vec::new();
    for i in t..t + m {
        obs.extend(step(&mut f, &mut s, i));
        for (_, g, sg, og) in copies.iter_mut() {
            og.extend(step(g, sg, i));
        }
    }
    let (want_state, want_bytes) = (state_bits(&f), serde_json::to_vec(&f).unwrap());
    let mut diffs = Vec::new();
    for (name, g, _, og) in &copies {
        if g.len() != f.len() || g.n_units() != f.n_units() {
            diffs.push(format!(
                "{name}: shape — cells {} vs {}, units {} vs {}",
                g.len(),
                f.len(),
                g.n_units(),
                f.n_units()
            ));
        }
        if *og != obs {
            let k = og.iter().zip(&obs).position(|(a, b)| a != b);
            diffs.push(format!(
                "{name}: per-step observables diverge (first at {k:?})"
            ));
        }
        let got = state_bits(g);
        if got != want_state {
            let k = got.iter().zip(&want_state).position(|(a, b)| a != b);
            diffs.push(format!("{name}: cell/weight bits diverge (first at {k:?})"));
        }
        if serde_json::to_vec(g).unwrap() != want_bytes {
            diffs.push(format!("{name}: final serialized state diverges"));
        }
    }
    diffs
}

/// Assert replay holds, printing the verdict line for the proof log.
fn assert_replays(lever: &str, diffs: Vec<String>) {
    eprintln!(
        "[SNAPSHOT-CONTINUE] {lever:<22} {}",
        if diffs.is_empty() {
            "byte-identical (json + bincode + envelope)".to_string()
        } else {
            format!("DIVERGED: {}", diffs.join("; "))
        }
    );
    assert!(
        diffs.is_empty(),
        "{lever}: snapshot→load→continue must be byte-identical to continue — {diffs:?}"
    );
}

/// One XOR event on inputs 0/1 → target cell 2 (a grown, non-linear readout).
fn xor(t: usize) -> ([(usize, f32); 2], f32) {
    let a = if t.is_multiple_of(2) { 1.0 } else { -1.0 };
    let b = if (t / 2).is_multiple_of(2) { 1.0 } else { -1.0 };
    ([(0, a), (1, b)], 0.8 * a * b)
}

fn grow_step(f: &mut Field, t: usize) -> Vec<u32> {
    let (inp, y) = xor(t);
    let (e, born) = f.grow(&inp, 2, y, 0.1, 200);
    vec![bits(e), born as u32]
}

fn seeded(n: usize) -> Field {
    let mut f = Field::new(n);
    f.rng_seed(7);
    f
}

#[test]
fn default_grow_path() {
    let mut f = seeded(3);
    f.config_mut().max_hidden = 12;
    assert_replays(
        "default grow",
        divergence(f, (), 150, 80, |f, _, t| grow_step(f, t)),
    );
}

/// The DEFAULT living loop (default `Config`): `alive(Perceive)` grows, meters reliance and CULLS — a culled
/// feature's slot goes to the free-cell pool and the next birth reuses it. Snapshotted mid-churn (births and culls
/// both still happening), a lost pool makes the loaded organism append new cells instead.
#[test]
fn default_alive_living_pool() {
    let f = seeded(3);
    let step = |f: &mut Field, _: &mut (), t: usize| {
        let (inp, y) = xor(t);
        match f.alive(Event::Perceive {
            inputs: &inp,
            target_cell: 2,
            target: y,
            lr: 0.1,
            budget: 200,
        }) {
            Outcome::Perceived { error, born } => vec![bits(error), born as u32, f.len() as u32],
            _ => unreachable!(),
        }
    };
    assert_replays("alive living pool", divergence(f, (), 45, 100, step));
}

/// Explicit `cull` with slot reuse (core-debt E3): starve the grown basis and cull it on the last step before the
/// snapshot (its slots → the free list), then keep growing — every regrown feature should reuse a slot.
#[test]
fn cull_slot_reuse() {
    let mut f = seeded(3);
    f.config_mut().max_hidden = 12;
    let step = |f: &mut Field, _: &mut (), t: usize| {
        let mut o = grow_step(f, t);
        if t == 149 {
            let basis = f.grown_basis();
            f.metabolize_reliance(&basis, 0.0, 10.0); // starve every hidden feature
            o.push(f.cull(0.1) as u32);
        }
        o.push(f.len() as u32);
        o
    };
    assert_replays("cull + slot reuse", divergence(f, (), 150, 120, step));
}

/// `orient` materialises its scores into SCRATCH cells drawn from (and handed back to) the free-cell pool — a
/// default-path, no-lever user of the pool.
#[test]
fn orient_scratch_pool() {
    let f = seeded(8);
    let step = |f: &mut Field, _: &mut (), t: usize| {
        let _ = grow_step(f, t);
        f.reinforce(3 + t % 4, 0.1 * ((t % 3) as f32 - 1.0));
        let goal = f.orient(&[3, 4, 5, 6]).unwrap_or(usize::MAX);
        let _ = f.grow(&[(0, 1.0), (1, -1.0)], 7, 0.3, 0.1, 200); // a birth after orient may reuse a scratch slot
        vec![goal as u32, f.len() as u32]
    };
    assert_replays("orient scratch pool", divergence(f, (), 20, 40, step));
}

/// Surprise-driven plasticity (core-debt E5): the per-cell `err_fast/err_slow/learn_gain` controller.
#[test]
fn plasticity_anneal() {
    let mut f = Field::new(2);
    f.config_mut().max_hidden = 0;
    f.config_mut().plasticity_anneal = 0.05;
    f.add_unit(vec![0], vec![0.0], 0.0, 1);
    let step = |f: &mut Field, _: &mut (), t: usize| {
        let x = if t.is_multiple_of(2) { 1.0 } else { -1.0 };
        let y = if t < 70 { 0.5 * x } else { -0.5 * x }; // a target switch re-opens plasticity after the snapshot
        f.learn_step(&[(0, x)], 1, y, 0.1, 50);
        vec![bits(f.plasticity_gain_of(1))]
    };
    assert_replays("plasticity_anneal", divergence(f, (), 50, 60, step));
}

/// `adapt_scales` descends a trace's half-life along its RTRL sensitivity `Trace::sens` (core-debt E5).
#[test]
fn adapt_scales_traces() {
    let mut f = Field::new(2);
    f.config_mut().max_hidden = 0;
    let tr = f.integrate(0, 3.0);
    let step = move |f: &mut Field, _: &mut (), t: usize| {
        let x = (t as f32 * 0.3).sin();
        let y = 0.8 * ((t as f32 - 5.0) * 0.3).sin();
        let trv = f.cells()[tr];
        f.grow(&[(0, x), (tr, trv)], 1, y, 0.05, 50);
        f.adapt_scales(1, y, 0.5, 0.5, 200.0);
        vec![]
    };
    assert_replays("adapt_scales (traces)", divergence(f, (), 100, 100, step));
}

/// A resonator's frequency self-tunes (`adapt_freq`) along its RTRL sensitivities `Resonator::su/sv`.
#[test]
fn resonators_adapt_freq() {
    let mut f = Field::new(2);
    f.config_mut().max_hidden = 0;
    let rc = f.resonate(0, 0.6, 0.1);
    let step = move |f: &mut Field, _: &mut (), t: usize| {
        let x = (t as f32 * 0.9).sin();
        let (u, v) = (f.cells()[rc], f.cells()[rc + 1]);
        f.grow(&[(0, x), (rc, u), (rc + 1, v)], 1, 0.5 * x, 0.05, 50);
        f.adapt_freq(0.01);
        vec![]
    };
    assert_replays("resonators adapt_freq", divergence(f, (), 60, 60, step));
}

/// `grow_products` accrues the pairwise product co-credit and births a PRODUCT feature on the dominant pair.
#[test]
fn grow_products_co_credit() {
    let mut f = seeded(4);
    f.config_mut().grow_products = true;
    f.config_mut().max_hidden = 6;
    let step = |f: &mut Field, _: &mut (), t: usize| {
        let (inp, y) = xor(t);
        let z = if (t / 4).is_multiple_of(2) { 0.5 } else { -0.5 };
        let (e, born) = f.grow(&[inp[0], inp[1], (2, z)], 3, y, 0.1, 200);
        vec![
            bits(e),
            born as u32,
            bits(f.co_credit_of(0, 1)),
            bits(f.co_credit_of(1, 2)),
        ]
    };
    assert_replays("grow_products", divergence(f, (), 3, 40, step));
}

#[test]
fn feature_lr() {
    let mut f = seeded(3);
    f.config_mut().max_hidden = 8;
    f.config_mut().feature_lr = 0.05;
    assert_replays(
        "feature_lr",
        divergence(f, (), 100, 60, |f, _, t| grow_step(f, t)),
    );
}

#[test]
fn momentum_heavy_ball() {
    let mut f = seeded(3);
    f.config_mut().max_hidden = 8;
    f.config_mut().init_momentum = 0.5;
    assert_replays(
        "momentum (heavy-ball)",
        divergence(f, (), 100, 60, |f, _, t| grow_step(f, t)),
    );
}

#[test]
fn var_gated_norm() {
    let mut f = seeded(4);
    f.config_mut().max_hidden = 8;
    f.config_mut().lr_fanin_norm = true;
    f.config_mut().var_gated_norm = true;
    let step = |f: &mut Field, _: &mut (), t: usize| {
        let (inp, y) = xor(t);
        let (e, born) = f.grow(&[inp[0], inp[1], (2, 1.0)], 3, y, 0.1, 200); // cell 2 is a CONSTANT read
        vec![bits(e), born as u32]
    };
    assert_replays("var_gated_norm", divergence(f, (), 60, 60, step));
}

/// Episodic memory: store → replay → learn from the replayed state.
#[test]
fn episodes_replay() {
    let mut f = seeded(3);
    f.config_mut().max_hidden = 6;
    let step = |f: &mut Field, _: &mut (), t: usize| {
        let mut o = grow_step(f, t);
        if t.is_multiple_of(5) {
            f.store_episode(&[0, 1], &[2]);
        }
        if t.is_multiple_of(7) && !f.episodes().is_empty() {
            o.push(f.replay(t % f.episodes().len(), 100) as u32);
        }
        o
    };
    assert_replays("episodes / replay", divergence(f, (), 40, 40, step));
}

/// The dwell CURSOR (which option the actor is committed to, and for how long) shapes the next choice: a held
/// commitment overrides a fresh decision that has already moved on.
#[test]
fn action_dwell() {
    let mut f = Field::new(4);
    f.config_mut().action_dwell = true;
    let opts = [1usize, 2, 3];
    let step = move |f: &mut Field, last: &mut Option<(usize, f32)>, t: usize| {
        // the planner's goal rotates every 5 steps (cost-to-go 0 on it, 10 elsewhere), so the FRESH decision moves
        // while a grown commitment (every act earns a little, so dwell lengthens) keeps holding the old one
        let goal = opts[(t / 5) % 3];
        for &o in &opts {
            f.prefer(o, if o == goal { 0.0 } else { 10.0 });
        }
        let chosen = match f.alive(Event::Act {
            options: &opts,
            last: *last,
        }) {
            Outcome::Acted { chosen } => chosen,
            _ => unreachable!(),
        };
        *last = Some((chosen, 0.1));
        vec![chosen as u32]
    };
    assert_replays("action_dwell", divergence(f, None, 22, 40, step));
}

/// The fast-weight rule store: demonstrations are bound before the snapshot, the question is answered after.
#[test]
fn bind_complete_rule() {
    let f = Field::new(3);
    let step = |f: &mut Field, _: &mut (), t: usize| {
        if t < 4 {
            let v = t as f32 + 1.0;
            f.alive(Event::Suggest {
                observation: &[(0, v), (1, 9.0 - t as f32), (2, v)],
            });
            return vec![];
        }
        match f.alive(Event::Answer {
            known: &[(0, 7.0 + t as f32), (1, 5.0)],
            unknown: &[2],
        }) {
            Outcome::Answered(a) => a
                .iter()
                .flat_map(|&(c, v)| [c as u32, bits(v)])
                .chain([u32::MAX])
                .collect(),
            _ => unreachable!(),
        }
    };
    assert_replays("bind / complete", divergence(f, (), 4, 4, step));
}

/// A NESTED cell (a banked sub-morph, `nest_multi` — a primary AND an extra vector output) MEMOIZES its output by
/// input. The memo is not a pure cache: a hit returns what the FIRST settle produced, while a re-settle starts from
/// the sub's CURRENT state and can land a different fixed point within `settle_eps`. So a reload with a cold memo
/// re-settles and drifts. Inputs recur (hits) and nearly recur (misses that move the sub's state); the per-step
/// cache-miss count (`nested_settles`) is an observable too.
#[test]
fn nest_memo() {
    let mut sub = Field::new(6);
    sub.add_unit(vec![0, 1], vec![3.0, -2.0], 0.1, 2);
    sub.add_unit(vec![0, 1], vec![-2.5, 4.0], 0.3, 3);
    sub.add_unit(vec![2, 3], vec![2.5, 1.5], -0.2, 4); // sink 4 → the primary output
    sub.add_unit(vec![2, 3], vec![-1.5, 2.0], 0.1, 5); // sink 5 → the extra output
    let mut f = Field::new(4);
    f.nest_multi(sub, &[0, 1], &[2, 3], 50);
    let xs: [f32; 7] = [0.31, 0.3101, -0.7, 1.2, 0.05, 0.0502, -0.33];
    let step = move |f: &mut Field, _: &mut (), t: usize| {
        let (a, b) = (xs[(t * 5 + t / 3) % 7], xs[(t * 3 + 1 + t / 5) % 7]);
        let misses = f.nested_settles();
        f.stimulate(&[(0, a), (1, b)], 200);
        let (out, extra) = (f.cells()[2], f.cells()[3]);
        vec![bits(out), bits(extra), (f.nested_settles() - misses) as u32]
    };
    assert_replays("nest (memo)", divergence(f, (), 60, 60, step));
}

/// The serialized plasticity controller is sized as one; a hostile snapshot that breaks that is rejected by
/// `validate` instead of panicking inside the learn path.
#[test]
fn hostile_controller_lengths_rejected() {
    let mut f = Field::new(2);
    f.config_mut().plasticity_anneal = 0.05;
    f.add_unit(vec![0], vec![0.0], 0.0, 1);
    f.learn_step(&[(0, 1.0)], 1, 0.5, 0.1, 50);
    let mut v: serde_json::Value = serde_json::to_value(&f).unwrap();
    v["err_fast"] = serde_json::json!([0.0]); // shorter than learn_gain
    let hostile: Field = serde_json::from_value(v).unwrap();
    let err = hostile
        .validate()
        .expect_err("mismatched controller lengths must be rejected");
    assert!(format!("{err}").contains("plasticity controller"), "{err}");
}

/// The serialized nest memo is read one output per sink; a hostile entry with too few outputs is rejected by
/// `validate` instead of panicking on the next memo hit.
#[test]
fn hostile_memo_entry_rejected() {
    let mut sub = Field::new(3);
    sub.add_unit(vec![0, 1], vec![1.0, -1.0], 0.0, 2);
    let mut f = Field::new(3);
    f.nest(sub, &[0, 1], 2, 50);
    f.stimulate(&[(0, 0.5), (1, 0.25)], 200);
    let mut v: serde_json::Value = serde_json::to_value(&f).unwrap();
    let memo = &mut v["units"][0]["nested"]["memo"];
    assert_eq!(memo.as_array().map(Vec::len), Some(1), "one input cached");
    memo[0][1] = serde_json::json!([]); // no outputs for a node with one sink
    let hostile: Field = serde_json::from_value(v).unwrap();
    let err = hostile
        .validate()
        .expect_err("a memo entry without one output per sink must be rejected");
    assert!(format!("{err}").contains("nested memo"), "{err}");
}

/// `Field::default()` is the empty organism `Field::new(0)` — one PRNG stream, one metabolism, not two.
#[test]
fn field_default_is_new_zero() {
    let d = serde_json::to_vec(&Field::default()).unwrap();
    let n = serde_json::to_vec(&Field::new(0)).unwrap();
    assert_eq!(
        String::from_utf8(d).unwrap(),
        String::from_utf8(n).unwrap(),
        "Field::default() must equal Field::new(0)"
    );
}

/// A partial (or empty) `Config` JSON loads, filling every absent knob from `Config::default()`.
#[test]
fn config_partial_json_loads_defaults() {
    let empty: Config = serde_json::from_str("{}").expect("an empty Config JSON loads");
    assert_eq!(empty, Config::default());
    let one: Config =
        serde_json::from_str(r#"{"birth_energy": 5.0}"#).expect("a one-knob Config JSON loads");
    let want = Config {
        birth_energy: 5.0,
        ..Config::default()
    };
    assert_eq!(one, want);
}

/// An organism snapshot written before 1.6 has no key for the levers 1.6 added. It used to load them at their NEW
/// defaults next to its stored pre-1.6 values — a genome that is neither (review of #494, core #2). Each absent lever
/// now takes its `Config::legacy()` value; a present key keeps its value; a current snapshot round-trips its genome
/// exactly, in JSON and in bincode.
#[test]
fn an_old_snapshot_keeps_its_genome() {
    let new_keys = [
        "act_rule",
        "stability_cap",
        "fanin_init_gain",
        "reliance_norm",
        "grow_within_budget",
    ];
    let mut f = Field::new(4);
    f.configure(Config::legacy());
    f.grow(&[(0, 0.3), (1, -0.2)], 3, 0.5, 0.1, 32);
    // the pre-1.6 JSON: the same organism without the 1.6 keys (and with `feature_lr` stored, as #488 wrote it)
    let mut v: serde_json::Value = serde_json::to_value(&f).unwrap();
    for k in new_keys {
        v["config"].as_object_mut().unwrap().remove(k);
    }
    let old: Field = serde_json::from_value(v.clone()).unwrap();
    assert_eq!(
        *old.config(),
        Config::legacy(),
        "the old snapshot is the legacy genome"
    );
    // a present key keeps its value
    v["config"]["stability_cap"] = serde_json::json!(0.25);
    let mixed: Field = serde_json::from_value(v.clone()).unwrap();
    assert_eq!(mixed.config().stability_cap, 0.25);
    // no config at all: legacy
    v.as_object_mut().unwrap().remove("config");
    let bare: Field = serde_json::from_value(v).unwrap();
    assert_eq!(*bare.config(), Config::legacy());
    // current snapshots round-trip their genome exactly
    for cfg in [Config::default(), Config::legacy()] {
        let mut g = Field::new(4);
        g.configure(cfg.clone());
        let j: Field = serde_json::from_str(&serde_json::to_string(&g).unwrap()).unwrap();
        assert_eq!(*j.config(), cfg, "JSON");
        let b: Field = bincode::deserialize(&bincode::serialize(&g).unwrap()).unwrap();
        assert_eq!(*b.config(), cfg, "bincode");
    }
}
