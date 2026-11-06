// SPDX-License-Identifier: MIT-0
//! LEGACY REPLAY — `Config::legacy()` must keep reproducing the PRE-FLIP organism (commit `38074e7`) BIT FOR BIT on
//! the learning path. Scope, said plainly: `38074e7` is inside the 2026-09 review pass, AFTER its correctness fixes
//! (the stale-output refresh, one settle per observation, exact `encapsulate`/`nest`, …) landed, and those fixes run
//! on every genome. So this pins "legacy() undoes the default FLIP", not "legacy() is the organism on `main` before
//! the review": that one differs wherever the fixed bugs fired (e.g. a repeated `learn_step` re-applying a frozen
//! error).
//!
//! 1.6 flipped the learning defaults (the step-stability cap, `feature_lr`, the fan-in-scaled feature init and the
//! mean-relative reliance economy). An organism grown before the flip is replayed by building it with
//! [`Config::legacy`], which restores the pre-flip values of every flipped LEARNING lever. (1.6 also replaced the
//! `ActScore` actor; that rule is removed, so `legacy()` acts with the default `Valence` rule and an organism that
//! acts no longer replays its pre-1.6 choices. The stream below never acts, so the pin did not move.)
//! This proof pins that promise to a hash:
//!
//! 1. `Config::legacy()` holds the pre-flip value of every flipped learning lever (and the default actor);
//! 2. a small mixed learning stream — `alive(Perceive)` growth + the reliance economy on two targets, the raw delta
//!    step on a narrow readout, the auto-normalized step on a 130-read readout, a `grow_backbone` readout, `alive`
//!    growth at a settle budget below the pool's cost (truncating under the legacy genome) — run under
//!    `Config::legacy()` hashes (every cell, unit parameter and return value, `.to_bits()`) to the value the
//!    pre-flip default organism produced at commit `38074e7`;
//! 3. the same stream under `Config::default()` hashes differently (the stream really exercises the flipped levers).
//!
//! The pinned hash covers `f32::tanh`, which comes from the platform's libm, so it is asserted on the CI platform
//! (x86-64 Linux) only; elsewhere the test still checks (1) and (3).
//!
//!   cargo test -p openmorph-core --test proofs substrate_legacy_replay -- --nocapture

use openmorph_core::{ActRule, Config, Event, Field, Outcome, RelianceNorm};

/// The stream's hash under the pre-flip default organism (commit `38074e7`, x86-64 Linux, release and debug alike).
const PRE_FLIP_HASH: u64 = 0xacf1_249e_a733_9798;

/// FNV-1a over raw bits.
struct H(u64);
impl H {
    fn u(&mut self, x: u64) {
        for b in x.to_le_bytes() {
            self.0 ^= b as u64;
            self.0 = self.0.wrapping_mul(0x100000001b3);
        }
    }
    fn f(&mut self, x: f32) {
        self.u(x.to_bits() as u64);
    }
    fn field(&mut self, f: &Field) {
        for &c in f.cells() {
            self.f(c);
        }
        for u in &f.units {
            for &c in &u.reads {
                self.u(c as u64);
            }
            for &w in &u.weights {
                self.f(w);
            }
            self.f(u.bias);
            self.u(u.write as u64);
        }
        self.u(f.clock());
        self.u(f.n_units() as u64);
    }
}

/// A deterministic xorshift stream in `[-1, 1]`, independent of the organism's PRNG.
struct Rng(u64);
impl Rng {
    fn f(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }
}

/// Run the mixed learning stream on `genome` and hash everything it produced.
fn stream(genome: &Config) -> u64 {
    let mut h = H(0xcbf29ce484222325);
    let mut r = Rng(0x0DDB_A11C_0FFE_E123);

    // (1) alive(Perceive): growth on surprise + the reliance economy, a nonlinear target and a second skill.
    let n = 6;
    let mut f = Field::new(n + 2);
    f.configure(genome.clone());
    f.rng_seed(7);
    let mut inp: Vec<(usize, f32)> = (0..n).map(|i| (i, 0.0)).collect();
    for _ in 0..600 {
        for p in inp.iter_mut() {
            p.1 = r.f();
        }
        let t = (1.5 * inp[0].1 * inp[1].1 - 0.5 * inp[2].1).tanh();
        for (cell, target) in [(n, t), (n + 1, (inp[3].1 - inp[4].1).tanh())] {
            match f.alive(Event::Perceive {
                inputs: &inp,
                target_cell: cell,
                target,
                lr: 0.1,
                budget: 1_000,
            }) {
                Outcome::Perceived { error, born } => {
                    h.f(error);
                    h.u(born as u64);
                }
                _ => unreachable!("Perceive answers Perceived"),
            }
        }
    }
    h.field(&f);

    // (2) learn_step on a NARROW readout (the raw delta step) and a 130-read one (the auto-normalized step).
    for d in [20usize, 130] {
        let mut g = Field::new(d + 1);
        g.configure(genome.clone());
        g.add_unit((0..d).collect(), vec![0.0; d], 0.0, d);
        for _ in 0..200 {
            let x: Vec<(usize, f32)> = (0..d).map(|k| (k, r.f())).collect();
            let y = (x[0].1 - 0.5 * x[1].1).tanh();
            h.f(g.learn_step(&x, d, y, 0.1, 1_000));
        }
        h.field(&g);
    }

    // (3) a grow_backbone basis + a learned readout over it (feature directions move iff feature_lr > 0).
    let mut b = Field::new(4);
    b.configure(genome.clone());
    b.rng_seed(3);
    let ins = [0usize, 1, 2];
    let feats = b.grow_backbone(&ins, 12, 1.0);
    let mut reads = ins.to_vec();
    reads.extend(feats);
    b.add_unit(reads.clone(), vec![0.0; reads.len()], 0.0, 3);
    for _ in 0..300 {
        let x: Vec<(usize, f32)> = ins.iter().map(|&c| (c, r.f())).collect();
        let y = (2.0 * x[0].1 * x[1].1).tanh();
        h.f(b.learn_step(&x, 3, y, 0.1, 1_000));
    }
    h.field(&b);

    // (4) alive(Perceive) at a settle budget below the grown pool's cost (the legacy pool outgrows it and truncates;
    // the default `grow_within_budget` stops births that would not fit).
    let mut t = Field::new(3);
    t.configure(genome.clone());
    t.rng_seed(11);
    for _ in 0..400 {
        let (a, c) = (r.f(), r.f());
        if let Outcome::Perceived { error, born } = t.alive(Event::Perceive {
            inputs: &[(0, a), (1, c)],
            target_cell: 2,
            target: if a * c > 0.0 { 0.8 } else { -0.8 },
            lr: 0.1,
            budget: 20,
        }) {
            h.f(error);
            h.u(born as u64);
        }
    }
    h.field(&t);
    h.0
}

#[test]
fn legacy_holds_the_pre_flip_value_of_every_flipped_lever() {
    let l = Config::legacy();
    assert_eq!(
        l.act_rule,
        ActRule::Valence,
        "the default actor (the pre-1.6 one is removed)"
    );
    assert_eq!(l.feature_lr, 0.0, "feature directions stay random");
    assert_eq!(l.stability_cap, 0.0, "the raw delta step");
    assert_eq!(
        l.fanin_init_gain, 0.0,
        "the fan-in-blind U[-1,1] feature init"
    );
    assert_eq!(
        l.reliance_norm,
        RelianceNorm::Absolute,
        "absolute reliance income"
    );
    assert!(!l.relative_reliance, "no share-of-total economy");
    assert_eq!(l.width_norm_min, 128, "auto-NLMS at fan-in >= 128");
    assert_eq!(l.pool_grace, 200, "the newborn grace");
    assert!(
        !l.grow_within_budget,
        "births up to max_hidden whatever the settle budget"
    );
    // Everything else is the default genome.
    let d = Config::default();
    let legacy_levers = Config {
        feature_lr: d.feature_lr,
        stability_cap: d.stability_cap,
        fanin_init_gain: d.fanin_init_gain,
        reliance_norm: d.reliance_norm,
        width_norm_min: d.width_norm_min,
        pool_grace: d.pool_grace,
        grow_within_budget: d.grow_within_budget,
        ..l
    };
    assert_eq!(
        legacy_levers, d,
        "legacy() differs from the default only in the flipped levers"
    );
}

#[test]
fn a_legacy_stream_replays_the_pre_flip_organism_bit_for_bit() {
    let legacy = stream(&Config::legacy());
    let default = stream(&Config::default());
    eprintln!("[LEGACY-REPLAY] legacy {legacy:#018x}, default {default:#018x}, pinned {PRE_FLIP_HASH:#018x}");
    assert_ne!(
        legacy, default,
        "the stream must exercise the flipped levers, or it cannot guard legacy replay"
    );
    if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
        assert_eq!(
            legacy, PRE_FLIP_HASH,
            "Config::legacy() no longer replays the pre-flip default organism bit for bit"
        );
    }
}
