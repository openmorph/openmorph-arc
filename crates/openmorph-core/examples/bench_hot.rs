// SPDX-License-Identifier: MIT-0
//! Perception-scale hot-path bench — the realistic workloads behind the micro `bench` example: a `Field` grown on
//! a WIDE stream (`alive(Perceive)` at 64..4096 input cells), a settle over a wide grown field (dense and sparse
//! input changes), `learn_step` on a wide (NLMS) readout, `observe` on a 256..4096-cell frame, the living-pool
//! economy (`metabolize_reliance` + `cull_after`) over hundreds of units, `relax_values` on a big grid, nested
//! memo hits, and the temporal banks (`integrate`/`delay`). Wall-clock is fine here (an example, not the
//! substrate). Reports the median of `REPS` timed runs.
//!
//!   cargo run --release --example bench_hot -p openmorph-core [-- <filter>]

use openmorph_core::substrate::{Event, Field};
use std::time::Instant;

const REPS: usize = 5;

struct Rng(u64);
impl Rng {
    fn f(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }
}

/// Time `iters` calls of `f` (after `setup` builds fresh state each rep); report the median ns/op.
fn bench<S>(name: &str, iters: usize, mut setup: impl FnMut() -> S, mut f: impl FnMut(&mut S)) {
    let filter = std::env::args().nth(1).unwrap_or_default();
    if !name.contains(&filter) {
        return;
    }
    let mut v = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        let mut s = setup();
        let t = Instant::now();
        for _ in 0..iters {
            f(&mut s);
        }
        v.push(t.elapsed().as_nanos() as f64 / iters as f64);
        std::hint::black_box(&s);
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = v[REPS / 2];
    let (val, unit) = if med >= 1e6 {
        (med / 1e6, "ms")
    } else if med >= 1e3 {
        (med / 1e3, "us")
    } else {
        (med, "ns")
    };
    println!(
        "  {name:<44} {val:>9.2} {unit}/op   (min {:.2} {unit})",
        v[0] / (med / val)
    );
}

/// A field grown by `alive(Perceive)` on `n` inputs until its hidden pool is full (the steady state).
fn grown_alive(n: usize, warm: usize) -> (Field, Rng) {
    let mut f = Field::new(n + 1);
    let mut r = Rng(0x9E37_79B9_7F4A_7C15);
    let mut inp: Vec<(usize, f32)> = (0..n).map(|i| (i, 0.0)).collect();
    for _ in 0..warm {
        for p in inp.iter_mut() {
            p.1 = r.f();
        }
        let t = (0.8 * inp[0].1 * inp[1 % n].1 + 0.3 * inp[2 % n].1).tanh();
        f.alive(Event::Perceive {
            inputs: &inp,
            target_cell: n,
            target: t,
            lr: 0.05,
            budget: 1_000_000,
        });
    }
    (f, r)
}

/// A wide grown field: `n_in` inputs, a `width` random-projection backbone, `n_out` linear readouts over it.
fn wide_field(n_in: usize, width: usize, n_out: usize) -> (Field, Vec<usize>) {
    let mut f = Field::new(n_in);
    f.rng_seed(7);
    let inputs: Vec<usize> = (0..n_in).collect();
    let bb = f.grow_backbone(&inputs, width, 1.0);
    let mut outs = Vec::new();
    for o in 0..n_out {
        let cell = f.len();
        f.ensure_cells(cell + 1);
        let w: Vec<f32> = (0..bb.len())
            .map(|k| (((k * 31 + o * 7) % 17) as f32 / 17.0 - 0.5) * 0.1)
            .collect();
        f.add_unit(bb.clone(), w, 0.0, cell);
        outs.push(cell);
    }
    (f, outs)
}

fn main() {
    println!("openmorph perception-scale hot-path bench (median of {REPS})\n");

    // ── alive(Perceive): grow + learn + reliance economy + cull, per event, on a dense stream ──
    for &(n, iters) in &[(64usize, 400usize), (256, 200), (1024, 60), (4096, 15)] {
        bench(
            &format!("alive(Perceive) n={n} (pool full)"),
            iters,
            || {
                let (f, r) = grown_alive(n, 150);
                let inp: Vec<(usize, f32)> = (0..n).map(|i| (i, 0.0)).collect();
                (f, r, inp)
            },
            |(f, r, inp)| {
                for p in inp.iter_mut() {
                    p.1 = r.f();
                }
                let t = (0.8 * inp[0].1 * inp[1].1 + 0.3 * inp[2].1).tanh();
                f.alive(Event::Perceive {
                    inputs: inp,
                    target_cell: n,
                    target: t,
                    lr: 0.05,
                    budget: 1_000_000,
                });
            },
        );
    }

    // ── settle over a wide grown field: all inputs change (dense) vs ONE input changes (sparse) ──
    for &(n_in, width, n_out) in &[(256usize, 512usize, 16usize), (1024, 1024, 32)] {
        bench(
            &format!("stimulate dense in={n_in} w={width} o={n_out}"),
            50,
            || (wide_field(n_in, width, n_out).0, Rng(3)),
            |(f, r)| {
                let inp: Vec<(usize, f32)> = (0..n_in).map(|i| (i, r.f())).collect();
                f.stimulate(&inp, usize::MAX);
            },
        );
        bench(
            &format!("stimulate 1-input in={n_in} w={width} o={n_out}"),
            500,
            || (wide_field(n_in, width, n_out).0, Rng(5), 0usize),
            |(f, r, k)| {
                *k = (*k + 1) % n_in;
                f.stimulate(&[(*k, r.f())], usize::MAX);
            },
        );
        bench(
            &format!("stimulate no-change in={n_in} w={width} o={n_out}"),
            2000,
            || wide_field(n_in, width, n_out).0,
            |f| {
                let v = f.cells()[0];
                f.stimulate(&[(0, v)], usize::MAX);
            },
        );
    }

    // ── learn_step on a wide readout (fan-in >= width_norm_min ⇒ the NLMS path) ──
    for &n in &[128usize, 1024, 4096] {
        bench(
            &format!("learn_step fan-in={n} (NLMS)"),
            200,
            || {
                let mut f = Field::new(n + 1);
                f.add_unit((0..n).collect(), vec![0.0; n], 0.0, n);
                (f, Rng(11))
            },
            |(f, r)| {
                let inp: Vec<(usize, f32)> = (0..n).map(|i| (i, r.f())).collect();
                let t = inp[0].1 * 0.5;
                f.learn_step(&inp, n, t, 0.1, usize::MAX);
            },
        );
    }

    // ── observe on a frame: obs band [0,D), pred band [D,2D), W-wide backbone over obs, linear pred readouts ──
    for &(d, w, iters) in &[(256usize, 64usize, 20usize), (1024, 64, 4), (4096, 32, 1)] {
        bench(
            &format!("observe frame D={d} backbone={w}"),
            iters,
            || {
                let mut f = Field::new(2 * d);
                f.rng_seed(9);
                let obs: Vec<usize> = (0..d).collect();
                let bb = f.grow_backbone(&obs, w, 1.0);
                for i in 0..d {
                    f.add_unit(bb.clone(), vec![0.0; bb.len()], 0.0, d + i);
                }
                (f, 0usize)
            },
            |(f, t)| {
                *t += 1;
                let frame: Vec<f32> = (0..d)
                    .map(|i| if (i + *t) % 7 == 0 { 1.0 } else { 0.0 })
                    .collect();
                std::hint::black_box(f.observe(0, d, d, &frame, 0.05, usize::MAX));
            },
        );
    }

    // ── living-pool economy over hundreds of units: metabolize_reliance + cull_after (mostly no deaths) ──
    for &units in &[256usize, 1024] {
        bench(
            &format!("metabolize_reliance+cull units={units}"),
            200,
            || {
                let (f, _) = wide_field(64, units, 1);
                let basis = f.target_basis(f.len() - 1);
                (f, basis)
            },
            |(f, basis)| {
                f.metabolize_reliance(basis, 2.0, 0.25);
                f.cull_after(0.1, 200);
            },
        );
    }

    // ── relax_values on a big grid graph (planning) ──
    for &side in &[64usize, 128] {
        let n = side * side;
        let adj: Vec<Vec<usize>> = (0..n)
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
        bench(
            &format!("relax_values grid {side}x{side} (to convergence)"),
            3,
            || {
                let mut f = Field::new(n);
                f.prefer(n - 1, 0.0);
                f
            },
            |f| {
                f.clear_values();
                f.prefer(n - 1, 0.0);
                std::hint::black_box(f.relax_values(&adj, 1.0, 0));
            },
        );
    }

    // ── nested memo hits: a host firing a nested sub-morph over a small recurring input set ──
    bench(
        "nest memo-hit fire (sub 16->64->1)",
        20_000,
        || {
            let (mut sub, _) = wide_field(16, 64, 1);
            sub.rng_seed(1);
            let mut host = Field::new(16);
            let reads: Vec<usize> = (0..16).collect();
            let out = host.len();
            host.nest(sub, &reads, out, 10_000);
            (host, 0usize)
        },
        |(h, k)| {
            *k = (*k + 1) % 4;
            let v = *k as f32 * 0.25;
            let inp: Vec<(usize, f32)> = (0..16).map(|i| (i, v + i as f32 * 0.01)).collect();
            h.stimulate(&inp, 10_000);
        },
    );

    // ── temporal banks: 64 traces + 64 delays over 64 channels, a grown readout, streamed ──
    bench(
        "stream integrate+delay (64ch) + grow",
        500,
        || {
            let mut f = Field::new(65);
            let mut cells: Vec<usize> = (0..64).collect();
            for c in 0..64 {
                cells.push(f.integrate(c, 8.0));
                cells.push(f.delay(c));
            }
            let target = f.len();
            f.ensure_cells(target + 1);
            (f, Rng(21), cells, target)
        },
        |(f, r, cells, target)| {
            let mem = f.advance_delays();
            let mut inp: Vec<(usize, f32)> = (0..64).map(|i| (i, r.f())).collect();
            inp.extend(mem);
            let _ = cells;
            let t = inp[0].1 * 0.3;
            f.grow(&inp, *target, t, 0.05, usize::MAX);
        },
    );
}
