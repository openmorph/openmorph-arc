// SPDX-License-Identifier: MIT-0
//! Profiling driver (for `valgrind --tool=callgrind`): builds ONE workload, then runs its hot call `events`
//! times, so `Ir(events) − Ir(0)` is the instruction cost of the hot call alone — a load-independent metric on a
//! shared machine. `cargo run --release --example prof_hot -- <mode> <n> <events>`; modes: alive, observe,
//! dense, sparse, learn, econ, memo, stream, relax.

use openmorph_core::substrate::{Event, Field};

struct Rng(u64);
impl Rng {
    fn f(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }
}

fn alive_step(f: &mut Field, r: &mut Rng, inp: &mut [(usize, f32)], n: usize) {
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
}

fn wide_field(n_in: usize, width: usize, n_out: usize) -> Field {
    let mut f = Field::new(n_in);
    f.rng_seed(7);
    let inputs: Vec<usize> = (0..n_in).collect();
    let bb = f.grow_backbone(&inputs, width, 1.0);
    for o in 0..n_out {
        let cell = f.len();
        f.ensure_cells(cell + 1);
        let w: Vec<f32> = (0..bb.len())
            .map(|k| (((k * 31 + o * 7) % 17) as f32 / 17.0 - 0.5) * 0.1)
            .collect();
        f.add_unit(bb.clone(), w, 0.0, cell);
    }
    f
}

fn main() {
    let arg = |i: usize, d: usize| {
        std::env::args()
            .nth(i)
            .and_then(|s| s.parse().ok())
            .unwrap_or(d)
    };
    let mode = std::env::args().nth(1).unwrap_or_else(|| "alive".into());
    let n: usize = arg(2, 1024);
    let events: usize = arg(3, 50);
    let mut r = Rng(0x9E37_79B9_7F4A_7C15);
    match mode.as_str() {
        // alive(Perceive) at steady state: 150 warm-up events (pool full + churning), then `events` timed.
        "alive" => {
            let mut f = Field::new(n + 1);
            let mut inp: Vec<(usize, f32)> = (0..n).map(|i| (i, 0.0)).collect();
            for _ in 0..150 {
                alive_step(&mut f, &mut r, &mut inp, n);
            }
            for _ in 0..events {
                alive_step(&mut f, &mut r, &mut inp, n);
            }
            println!("units {} cells {}", f.n_units(), f.len());
        }
        "observe" => {
            let d = n;
            let mut f = Field::new(2 * d);
            f.rng_seed(9);
            let obs: Vec<usize> = (0..d).collect();
            let bb = f.grow_backbone(&obs, 32, 1.0);
            for i in 0..d {
                f.add_unit(bb.clone(), vec![0.0; bb.len()], 0.0, d + i);
            }
            for t in 0..events {
                let frame: Vec<f32> = (0..d)
                    .map(|i| if (i + t) % 7 == 0 { 1.0 } else { 0.0 })
                    .collect();
                std::hint::black_box(f.observe(0, d, d, &frame, 0.05, usize::MAX));
            }
        }
        // settle a wide grown field (n inputs → n-wide backbone → 16 readouts): every input changes
        "dense" => {
            let mut f = wide_field(n, n, 16);
            for _ in 0..events {
                let inp: Vec<(usize, f32)> = (0..n).map(|i| (i, r.f())).collect();
                f.stimulate(&inp, usize::MAX);
            }
        }
        // same field, ONE input changes per settle
        "sparse" => {
            let mut f = wide_field(n, n, 16);
            for k in 0..events {
                f.stimulate(&[(k % n, r.f())], usize::MAX);
            }
        }
        // learn_step on an n-wide readout (NLMS path once n >= 128)
        "learn" => {
            let mut f = Field::new(n + 1);
            f.add_unit((0..n).collect(), vec![0.0; n], 0.0, n);
            for _ in 0..events {
                let inp: Vec<(usize, f32)> = (0..n).map(|i| (i, r.f())).collect();
                let t = inp[0].1 * 0.5;
                f.learn_step(&inp, n, t, 0.1, usize::MAX);
            }
        }
        // living-pool economy over an n-unit backbone read by one readout
        "econ" => {
            let mut f = wide_field(64, n, 1);
            let basis = f.target_basis(f.len() - 1);
            for _ in 0..events {
                f.metabolize_reliance(&basis, 2.0, 0.25);
                f.cull_after(0.1, 200);
            }
        }
        // nested memo hits (a host firing a 16->n->1 sub over 4 recurring inputs)
        "memo" => {
            let sub = wide_field(16, n, 1);
            let mut host = Field::new(16);
            let reads: Vec<usize> = (0..16).collect();
            let out = host.len();
            host.nest(sub, &reads, out, 1_000_000);
            for k in 0..events {
                let v = (k % 4) as f32 * 0.25;
                let inp: Vec<(usize, f32)> = (0..16).map(|i| (i, v + i as f32 * 0.01)).collect();
                host.stimulate(&inp, 1_000_000);
            }
        }
        // temporal banks: n channels, each with a trace and a delay, streamed into grow
        "stream" => {
            let mut f = Field::new(n + 1);
            for c in 0..n {
                f.integrate(c, 8.0);
                let _ = f.delay(c);
            }
            let target = f.len();
            f.ensure_cells(target + 1);
            for _ in 0..events {
                let mem = f.advance_delays();
                let mut inp: Vec<(usize, f32)> = (0..n).map(|i| (i, r.f())).collect();
                inp.extend(mem);
                let t = inp[0].1 * 0.3;
                f.grow(&inp, target, t, 0.05, usize::MAX);
            }
        }
        // feature-direction learning (`feature_lr` on) through a readout over an n-wide backbone of 64 inputs
        "flr" => {
            let mut f = wide_field(64, n, 1);
            f.config_mut().feature_lr = 0.01;
            let out = f.len() - 1;
            for _ in 0..events {
                let inp: Vec<(usize, f32)> = (0..64).map(|i| (i, r.f())).collect();
                let t = inp[0].1 * inp[1].1;
                f.learn_step(&inp, out, t, 0.05, usize::MAX);
            }
        }
        // encapsulate the whole grown backbone+readout (n features over 64 inputs) as a sub-morph
        "encap" => {
            let f = wide_field(64, n, 1);
            let out = f.len() - 1;
            for _ in 0..events {
                std::hint::black_box(f.encapsulate(&[out]));
            }
        }
        "relax" => {
            let side = n;
            let cells = side * side;
            let adj: Vec<Vec<usize>> = (0..cells)
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
            let mut f = Field::new(cells);
            for _ in 0..events {
                f.clear_values();
                f.prefer(cells - 1, 0.0);
                std::hint::black_box(f.relax_values(&adj, 1.0, 0));
            }
        }
        _ => panic!("unknown mode {mode}"),
    }
}
