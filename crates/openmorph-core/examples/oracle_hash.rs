// SPDX-License-Identifier: MIT-0
//! BYTE-IDENTITY ORACLE for performance work. Runs a long MIXED workload over every hot path an optimisation
//! might touch (alive/grow/cull churn, observe with traces/resonators in context, delays, nest + memo + nest
//! credit, feature-direction learning, factors/connect + cull (non-canonical adjacency), the heartbeat, the
//! evolution economy, maps, planning, acting, binding, encapsulate/detach/merge, the opt-in levers) and prints
//! an FNV-1a hash of the FULL serialized state (serde_json — every serialized field, incl. the scheduler queue,
//! `last_fired`, adjacency) plus the raw `.to_bits()` of every cell and unit parameter, plus every return value,
//! at each stage. Run before and after a change; the printed lines must be identical.
//!
//!   cargo run --release --example oracle_hash -p openmorph-core
//!
//! Two environment variables let it compare organisms ACROSS a genome change:
//! - `ORACLE_GENOME` — the `Config` every stage's field starts from: `default` (the default) or `legacy`
//!   ([`Config::legacy`], the pre-1.6-learning-flip organism; it acts with the default `Valence` rule since
//!   `ActScore` was removed). Stage-specific levers are set on top of it.
//! - `ORACLE_STRIP_CONFIG` — a comma-separated list of `Config` keys dropped from the serialized state before it is
//!   hashed, so a run on a revision that ADDED those keys can be compared with one from before they existed (the
//!   keys' values then have to be proven inert some other way, e.g. by the rest of the hash matching).
//!
//! For example, to replay the pre-flip default organism against a revision from before the flipped levers existed:
//!
//! ```text
//! ORACLE_GENOME=legacy ORACLE_STRIP_CONFIG=stability_cap,fanin_init_gain,reliance_norm,grow_within_budget \
//!   cargo run --release --example oracle_hash -p openmorph-core
//! ```

use openmorph_core::substrate::{Event, Field, Outcome};
use openmorph_core::Config;

/// The genome every stage's field starts from (`ORACLE_GENOME`, see the module docs).
fn genome() -> Config {
    match std::env::var("ORACLE_GENOME").as_deref() {
        Err(_) | Ok("default") => Config::default(),
        Ok("legacy") => Config::legacy(),
        Ok(other) => panic!("ORACLE_GENOME={other}: expected default | legacy"),
    }
}

/// A fresh field of `n` cells on the selected [`genome`].
fn new_field(n: usize) -> Field {
    let mut f = Field::new(n);
    f.configure(genome());
    f
}

/// The serialized state, minus the `Config` keys named in `ORACLE_STRIP_CONFIG` (see the module docs) — from the
/// field's own genome and from every nested sub-morph's (a banked skill carries the genome it grew under).
fn serialized(f: &Field) -> String {
    let strip = std::env::var("ORACLE_STRIP_CONFIG").unwrap_or_default();
    if strip.is_empty() {
        return serde_json::to_string(f).expect("serialize");
    }
    fn walk(v: &mut serde_json::Value, keys: &[&str]) {
        match v {
            serde_json::Value::Object(m) => {
                if let Some(cfg) = m.get_mut("config").and_then(|c| c.as_object_mut()) {
                    for k in keys {
                        cfg.remove(*k);
                    }
                }
                m.values_mut().for_each(|x| walk(x, keys));
            }
            serde_json::Value::Array(a) => a.iter_mut().for_each(|x| walk(x, keys)),
            _ => {}
        }
    }
    let keys: Vec<&str> = strip.split(',').map(str::trim).collect();
    let mut v = serde_json::to_value(f).expect("serialize");
    walk(&mut v, &keys);
    serde_json::to_string(&v).expect("serialize")
}

struct H(u64);
impl H {
    fn new() -> Self {
        H(0xcbf29ce484222325)
    }
    fn u(&mut self, x: u64) {
        for b in x.to_le_bytes() {
            self.0 ^= b as u64;
            self.0 = self.0.wrapping_mul(0x100000001b3);
        }
    }
    fn f(&mut self, x: f32) {
        self.u(x.to_bits() as u64);
    }
    fn s(&mut self, s: &str) {
        for b in s.bytes() {
            self.0 ^= b as u64;
            self.0 = self.0.wrapping_mul(0x100000001b3);
        }
    }
}

struct Rng(u64);
impl Rng {
    fn f(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }
}

/// Hash the whole organism: its full serde state + raw bits of cells and unit params + key counters.
fn state(h: &mut H, f: &Field) {
    h.s(&serialized(f));
    for &c in f.cells() {
        h.f(c);
    }
    for u in &f.units {
        for &c in &u.reads {
            h.u(c as u64);
        }
        for &w in &u.weights {
            h.f(w);
        }
        h.f(u.bias);
        h.u(u.write as u64);
    }
    h.u(f.clock());
    h.u(f.nested_settles());
    h.f(f.dissipated_overflow());
}

fn report(stage: &str, h: &mut H, f: &Field) {
    state(h, f);
    println!(
        "{stage:<28} units={:<5} cells={:<6} clock={:<9} hash={:016x}",
        f.n_units(),
        f.len(),
        f.clock(),
        h.0
    );
}

fn outcome(h: &mut H, o: &Outcome) {
    match o {
        Outcome::Perceived { error, born } => {
            h.f(*error);
            h.u(*born as u64);
        }
        Outcome::Suggested => h.u(1),
        Outcome::Answered(v) => {
            for &(c, x) in v {
                h.u(c as u64);
                h.f(x);
            }
        }
        Outcome::Acted { chosen } => h.u(*chosen as u64),
    }
}

/// (1) alive(Perceive) churn: a nonlinear target on 48 inputs, pool fills, the reliance economy culls/reclaims.
fn stage_alive(h: &mut H) {
    let n = 48;
    let mut f = new_field(n + 2);
    f.rng_seed(42);
    let mut r = Rng(0x1234_5678_9ABC_DEF1);
    let mut inp: Vec<(usize, f32)> = (0..n).map(|i| (i, 0.0)).collect();
    for ev in 0..2500 {
        for p in inp.iter_mut() {
            p.1 = r.f();
        }
        // occasionally a sparse change: only a few inputs move (exercise the no-change path)
        if ev % 7 == 0 {
            for p in inp.iter_mut().skip(5) {
                p.1 = 0.25;
            }
        }
        let t = (0.9 * inp[0].1 * inp[1].1 - 0.4 * inp[2].1 + 0.2 * inp[3].1 * inp[4].1).tanh();
        let o = f.alive(Event::Perceive {
            inputs: &inp,
            target_cell: n,
            target: t,
            lr: 0.08,
            budget: if ev % 11 == 0 { 20 } else { 100_000 }, // budget-truncated settles too
        });
        outcome(h, &o);
        // a second skill on another channel (per-target economy)
        let t2 = (inp[5].1 - inp[6].1).tanh();
        let o2 = f.alive(Event::Perceive {
            inputs: &inp[..12],
            target_cell: n + 1,
            target: t2,
            lr: 0.05,
            budget: 100_000,
        });
        outcome(h, &o2);
        if ev % 97 == 0 {
            f.metabolize(0.5, 0.3); // firing economy reads last_fired
            h.u(f.cull(0.0) as u64);
        }
    }
    report("alive-churn", h, &f);
    // crystal inference path
    f.crystallize();
    let o = f.alive(Event::Perceive {
        inputs: &inp,
        target_cell: n,
        target: 0.0,
        lr: 0.1,
        budget: 1000,
    });
    outcome(h, &o);
    f.thaw();
    // encapsulate + bank + detach + merge
    let banked = f.bank_skill(n + 1, 1000);
    h.u(banked as u64);
    let sub = f.encapsulate(&[n]);
    h.u(sub.n_units() as u64);
    h.u(f.detach(&[n + 1]) as u64);
    let g = f.grown_basis();
    if g.len() >= 2 {
        f.merge(g[0], g[1]);
    }
    let e = f.stimulate(&inp, 100_000);
    h.u(e as u64);
    report("alive-bank-detach-merge", h, &f);
    // heartbeat (reactivate + reliance cull + consolidate)
    let mut fl = f.clone();
    let hb = fl.live(64);
    h.u(hb.ticks as u64);
    h.u(hb.reactivations as u64);
    h.u(hb.consolidated as u64);
    h.u(hb.culled as u64);
    report("alive-live", h, &fl);
}

/// (2) observe / observe_from on a frame with a trace + resonator + recency-like context; backbone predictor.
fn stage_observe(h: &mut H) {
    let d = 40;
    let mut f = new_field(2 * d);
    f.rng_seed(7);
    let obs: Vec<usize> = (0..d).collect();
    let bb = f.grow_backbone(&obs, 24, 1.3);
    for i in 0..d {
        let mut reads = bb.clone();
        reads.push(i);
        let w = vec![0.0; reads.len()];
        f.add_unit(reads, w, 0.0, d + i);
    }
    // plain observe (no temporal structures) — the pure-predictor case
    for t in 0..120 {
        let frame: Vec<f32> = (0..d)
            .map(|i| if (i + t) % 5 == 0 { 1.0 } else { -0.2 })
            .collect();
        for e in f.observe(0, d, d, &frame, 0.05, 100_000) {
            h.f(e);
        }
        if t == 60 {
            // an unfinished (budget-truncated) observe
            for e in f.observe(0, d, d, &frame, 0.05, 7) {
                h.f(e);
            }
        }
    }
    report("observe-plain", h, &f);
    // now add a trace and a resonator into the context (both change the settle semantics)
    let tr = f.integrate(0, 3.0);
    let rs = f.resonate(1, 0.7, 0.1);
    let tr_unread = f.integrate(2, 5.0); // a trace NOBODY reads, but in the context
    let mut ctx: Vec<usize> = obs.clone();
    ctx.push(tr);
    ctx.push(rs);
    ctx.push(tr_unread);
    for t in 0..80 {
        let frame: Vec<f32> = (0..d)
            .map(|i| if (i * 3 + t) % 7 == 0 { 0.8 } else { 0.0 })
            .collect();
        for e in f.observe_from(&ctx, 0, d, d, &frame, 0.04, 100_000) {
            h.f(e);
        }
    }
    f.adapt_scales(d, 0.3, 0.01, 1.0, 50.0);
    f.adapt_freq(0.01);
    report("observe-traces", h, &f);
    // frozen observe
    f.crystallize();
    for e in f.observe(0, d, d, &vec![0.5; d], 0.05, 1000) {
        h.f(e);
    }
    f.thaw();
    report("observe-frozen", h, &f);

    // traces WITHOUT resonators (the quiet-settle fast path is live): one trace read by a predictor, one unread
    // trace in the context, a DUPLICATED context cell, and GROWN cells (the pred band) in the context.
    let d = 24;
    let mut g = new_field(2 * d);
    g.rng_seed(8);
    let obs: Vec<usize> = (0..d).collect();
    let tr_read = g.integrate(3, 2.0);
    let tr_unread = g.integrate(4, 6.0);
    let mut src = obs.clone();
    src.push(tr_read);
    let bb = g.grow_backbone(&src, 16, 1.1);
    for i in 0..d {
        g.add_unit(bb.clone(), vec![0.0; bb.len()], 0.0, d + i);
    }
    let mut ctx: Vec<usize> = obs.clone();
    ctx.push(tr_unread);
    ctx.push(tr_read);
    ctx.push(5); // duplicate
    ctx.push(d + 1); // a grown (pred) cell
    ctx.push(d + 7);
    for t in 0..90 {
        let frame: Vec<f32> = (0..d)
            .map(|i| if (i + 2 * t) % 6 == 0 { 0.9 } else { 0.1 })
            .collect();
        let budget = if t % 13 == 0 { 9 } else { 100_000 };
        for e in g.observe_from(&ctx, 0, d, d, &frame, 0.05, budget) {
            h.f(e);
        }
        if t % 10 == 0 {
            for e in g.observe(0, d, d, &frame, 0.05, 100_000) {
                h.f(e);
            }
        }
    }
    report("observe-trace-only", h, &g);
}

/// (2b) dream: think + consolidate long-range heads + the sleep heartbeat.
fn stage_dream(h: &mut H) {
    let mut f = new_field(8);
    f.rng_seed(21);
    // a small recurrent model: state 0..2 -> pred 2..4 ; long-range heads 4..6
    for _ in 0..300 {
        // black_box: without it LLVM constant-folds `tanh` of the literal targets in release with a different
        // tanh than the runtime libm one, so debug and release builds of THIS HARNESS would hash differently
        // (the substrate itself is debug == release).
        for &(a, b) in std::hint::black_box(&[(0.5f32, -0.5f32), (-0.3, 0.8), (0.9, 0.1)]) {
            f.grow(
                &[(0, a), (1, b)],
                2,
                (0.7 * a + 0.2 * b).tanh(),
                0.1,
                10_000,
            );
            f.grow(
                &[(0, a), (1, b)],
                3,
                (0.4 * b - 0.3 * a).tanh(),
                0.1,
                10_000,
            );
            f.grow(&[(0, a), (1, b)], 4, 0.0, 0.05, 10_000);
            f.grow(&[(0, a), (1, b)], 5, 0.0, 0.05, 10_000);
        }
    }
    let seeds = vec![vec![0.5f32, -0.5], vec![-0.3, 0.8]];
    let mut keep_all = f.clone();
    keep_all.config_mut().pool_death = f32::MIN; // the sleep culls nothing ⇒ the dreamt weights stay hashed
    let hb = keep_all.dream(&[0, 1], &[2, 3], &[4, 5], &seeds, 3, 0.05, 4, 10_000);
    h.u(hb.ticks as u64);
    h.u(hb.consolidated as u64);
    report("dream-keep", h, &keep_all);
    let hb = f.dream(&[0, 1], &[2, 3], &[4, 5], &seeds, 3, 0.05, 4, 10_000);
    h.u(hb.ticks as u64);
    h.u(hb.reactivations as u64);
    h.u(hb.culled as u64);
    report("dream", h, &f);
}

/// (3) delays + traces streamed into grow, plus adapt_scales.
fn stage_stream(h: &mut H) {
    let mut f = new_field(9);
    f.rng_seed(3);
    let mut feats: Vec<usize> = (0..8).collect();
    for c in 0..4 {
        feats.push(f.delay(c));
        let d2 = f.delay(*feats.last().unwrap());
        feats.push(d2);
        feats.push(f.integrate(c, 2.0 + c as f32));
    }
    let target = 8;
    let mut r = Rng(99);
    let mut hist = [0.0f32; 4];
    for _ in 0..1500 {
        let mem = f.advance_delays();
        let x: Vec<f32> = (0..8).map(|_| r.f()).collect();
        let mut inp: Vec<(usize, f32)> = (0..8).map(|i| (i, x[i])).collect();
        inp.extend(mem.iter().copied());
        let t = (0.7 * hist[0] - 0.5 * hist[1]).tanh();
        let (e, born) = f.grow(&inp, target, t, 0.05, 100_000);
        h.f(e);
        h.u(born as u64);
        f.adapt_scales(target, t, 0.02, 0.5, 64.0);
        hist.rotate_right(1);
        hist[0] = x[0];
        let basis = f.target_basis(target);
        f.metabolize_reliance(&basis, 2.0, 0.25);
        h.u(f.cull_after(0.1, 150) as u64);
    }
    report("stream", h, &f);
}

/// (4) nest + memo + nest_credit + multi-output nest.
fn stage_nest(h: &mut H) {
    // grow a small sub skill
    let mut s = new_field(4);
    s.rng_seed(5);
    for _ in 0..400 {
        for &(a, b) in &[(-1.0f32, -1.0f32), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)] {
            s.grow(&[(0, a), (1, b)], 2, (a * b) * 0.8, 0.1, 10_000);
        }
    }
    let sub = s.encapsulate(&[2]);
    let mut host = new_field(6);
    host.rng_seed(6);
    let id = host.nest(sub.clone(), &[0, 1], 2, 1000);
    h.u(id as u64);
    host.add_unit(vec![2, 3], vec![0.5, -0.2], 0.1, 4);
    let mut r = Rng(17);
    for k in 0..600 {
        let (a, b) = if k % 3 == 0 {
            (r.f(), r.f())
        } else {
            ([-1.0, 1.0][k % 2], [1.0, -1.0][(k / 2) % 2])
        };
        let e = host.learn_step(&[(0, a), (1, b), (3, 0.3)], 4, (a * b).tanh(), 0.05, 10_000);
        h.f(e);
    }
    h.u(host.nested_settles());
    report("nest-memo", h, &host);
    host.config_mut().nest_credit = true;
    for k in 0..200 {
        let a = [-1.0f32, 1.0][k % 2];
        let b = [1.0f32, -1.0][(k / 3) % 2];
        let e = host.learn_step(&[(0, a), (1, b), (3, 0.3)], 4, -(a * b) * 0.7, 0.05, 10_000);
        h.f(e);
    }
    report("nest-credit", h, &host);
    // multi-output nest
    let mut s2 = new_field(3);
    s2.add_unit(vec![0], vec![0.7], 0.0, 1);
    s2.add_unit(vec![0], vec![-0.4], 0.1, 2);
    let mut h2 = new_field(4);
    h2.nest_multi(s2, &[0], &[1, 2], 100);
    h2.add_unit(vec![1, 2], vec![1.0, 1.0], 0.0, 3);
    for k in 0..50 {
        let e = h2.stimulate(&[(0, (k % 5) as f32 * 0.2)], 1000);
        h.u(e as u64);
    }
    report("nest-multi", h, &h2);
}

/// (5) opt-in learning levers on a wide readout: feature_lr, momentum, lr/var-gated norm, surprise currency,
/// plasticity anneal, grow_products, gates, factors/connect (non-canonical adjacency) + cull.
fn stage_levers(h: &mut H) {
    let n = 140; // >= width_norm_min on the writer once hidden features are born
    let mut f = new_field(n + 1);
    f.rng_seed(11);
    {
        let c = f.config_mut();
        c.feature_lr = 0.02;
        c.init_momentum = 0.3;
        c.var_gated_norm = true;
        c.surprise_currency = true;
        c.plasticity_anneal = 0.5;
        c.grow_products = true;
        c.max_hidden = 12;
    }
    let mut r = Rng(0xDEAD_BEEF);
    let mut inp: Vec<(usize, f32)> = (0..n).map(|i| (i, 0.0)).collect();
    for ev in 0..400 {
        for (i, p) in inp.iter_mut().enumerate() {
            p.1 = if i % 9 == 0 { 0.5 } else { r.f() }; // some CONSTANT reads for var_gated_norm
        }
        let t = (inp[1].1 * inp[2].1 + 0.3 * inp[3].1).tanh();
        let o = f.alive(Event::Perceive {
            inputs: &inp,
            target_cell: n,
            target: t,
            lr: 0.05,
            budget: 100_000,
        });
        outcome(h, &o);
        if ev == 150 {
            // wire a factor + a lateral connection (adjacency no longer canonical until the next rebuild)
            let g = f.grown_basis();
            if g.len() >= 2 {
                f.set_factor(g[0], Some(g[1]));
                h.u(f.connect(g[1], g[0], 0.05) as u64);
                h.u(f.connect(3, g[0], -0.1) as u64);
            }
        }
        if ev == 250 {
            let g = f.grown_basis();
            f.set_gate(&g[..1.min(g.len())], 7);
            f.metabolize_surprise(&g, 1.0, 0.2);
            f.relax();
        }
    }
    for &c in &f.grown_basis() {
        h.f(f.surprise_reduction_of(c));
        h.f(f.reliance_of(c));
        h.f(f.credit_of(c));
    }
    h.f(f.effective_rank());
    h.f(f.spectral_radius(16));
    report("levers", h, &f);
    // evolution economy: contribution metabolism + repopulate (reproduce) + resample
    f.config_mut().feature_lr = 0.0;
    let basis = f.grown_basis();
    for _ in 0..30 {
        f.metabolize_contribution(&basis, 5.0, 1.0);
        f.repopulate(&basis, 1.5);
        for p in inp.iter_mut() {
            p.1 = r.f();
        }
        let o = f.alive(Event::Perceive {
            inputs: &inp,
            target_cell: n,
            target: 0.2,
            lr: 0.05,
            budget: 100_000,
        });
        outcome(h, &o);
    }
    if let Some(&c) = basis.first() {
        f.resample_feature(c, 0.7);
    }
    report("evolve", h, &f);
}

/// (6) maps, planning, acting, binding, episodes, selection, clustering, routing.
fn stage_misc(h: &mut H) {
    let mut f = new_field(64);
    f.rng_seed(13);
    for k in 0..400usize {
        f.metric_step(&[k % 64, (k * 7 + 3) % 64, (k * 5 + 1) % 64], 0.08, 4);
        if k % 50 == 0 {
            f.co_activate(&[k % 64, (k + 1) % 64], 0.1);
        }
    }
    f.grow_dimension();
    for a in 0..16 {
        h.f(f.distance(a, (a * 3) % 64, 0));
        h.f(f.distance(a, (a * 5) % 64, 2));
    }
    let cl = f.cluster(&(0..64).collect::<Vec<_>>(), 4);
    for c in &cl {
        h.u(c.len() as u64);
        for &x in c {
            h.u(x as u64);
        }
    }
    let idx = f.index_skills(&(0..64).collect::<Vec<_>>(), 4, 3);
    let (best, cmp) = f.route(&idx, &[0.1, -0.2, 0.3], 2, 0);
    h.u(best as u64);
    h.u(cmp as u64);
    // value relaxation on an 8x8 grid + acting
    let side = 8;
    let adj: Vec<Vec<usize>> = (0..side * side)
        .map(|c| {
            let (x, y) = (c % side, c / side);
            let mut v = Vec::new();
            if x > 0 {
                v.push(c - 1);
            }
            if x + 1 < side {
                v.push(c + 1);
            }
            if y > 0 {
                v.push(c - side);
            }
            if y + 1 < side {
                v.push(c + side);
            }
            v
        })
        .collect();
    f.prefer(63, 0.0);
    h.u(f.relax_values(&adj, 0.37, 0) as u64);
    h.u(f.relax_values(&adj, 0.37, 3) as u64);
    for c in 0..64 {
        h.f(f.value_of(c));
    }
    let mut last = None;
    for k in 0..60 {
        let opts = [(k % 60) + 1, (k % 60) + 2, (k % 60) + 3];
        let o = f.alive(Event::Act {
            options: &opts,
            last,
        });
        outcome(h, &o);
        if let Outcome::Acted { chosen } = o {
            last = Some((chosen, if chosen % 2 == 0 { 0.5 } else { -0.3 }));
        }
    }
    // binding
    for v in 0..5 {
        let obs: Vec<(usize, f32)> = (0..10).map(|i| (i, ((i + v) % 3) as f32)).collect();
        let o = f.alive(Event::Suggest { observation: &obs });
        outcome(h, &o);
    }
    let o = f.alive(Event::Answer {
        known: &[(0, 2.0), (1, 1.0)],
        unknown: &[3, 4, 6],
    });
    outcome(h, &o);
    // episodes + replay/imagine on a small grown net
    let mut g = new_field(5);
    g.rng_seed(2);
    for _ in 0..200 {
        for &x in &[-1.0f32, 0.0, 1.0] {
            g.grow(&[(0, x), (1, -x)], 2, (x * 1.2).tanh(), 0.1, 1000);
        }
    }
    let ep = g.store_episode(&[0, 1], &[2]);
    h.u(g.replay(ep, 1000) as u64);
    h.u(g.imagine(ep, 1000) as u64);
    h.u(g.select(&[0, 1, 2]) as u64);
    let opts: [&[usize]; 2] = [&[0, 1], &[2, 3]];
    h.u(g.select_explore(&opts, 0.5) as u64);
    let v = g.think(&[0, 1], &[2, 2], 3, 0.1, 1000);
    for x in v {
        h.f(x);
    }
    let fm = g.fork_mutated();
    report("misc-fork", h, &fm);
    report("misc-map", h, &f);
    report("misc-grown", h, &g);
    // free-energy + valence actors
    let mut a = new_field(16);
    a.config_mut().free_energy_actor = true;
    let mut last = None;
    for k in 0..40 {
        let o = a.alive(Event::Act {
            options: &[3, 4, 5],
            last,
        });
        outcome(h, &o);
        if let Outcome::Acted { chosen } = o {
            last = Some((
                chosen,
                if chosen == 4 {
                    1.0
                } else {
                    (k % 3) as f32 * 0.1
                },
            ));
        }
    }
    h.f(a.g_spread());
    report("act-free-energy", h, &a);
}

fn main() {
    let mut h = H::new();
    stage_alive(&mut h);
    stage_observe(&mut h);
    stage_dream(&mut h);
    stage_stream(&mut h);
    stage_nest(&mut h);
    stage_levers(&mut h);
    stage_misc(&mut h);
    println!("FINAL {:016x}", h.0);
}
