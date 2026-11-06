// SPDX-License-Identifier: MIT-0
//! Lightweight throughput bench for the hot paths — no dependency, just `std::time` (an example, so wall-clock
//! is fine here; the substrate itself stays deterministic). The rc2 "measure first" harness: run it before and
//! after an optimisation to check the change is evidence, not a hunch.
//!
//!   cargo run --release --example bench -p openmorph-core

use openmorph_core::substrate::{Event, Field};
use std::time::Instant;

fn bench(name: &str, iters: u64, mut f: impl FnMut()) {
    let t = Instant::now();
    for _ in 0..iters {
        f();
    }
    let ns = t.elapsed().as_nanos() as f64 / iters as f64;
    println!(
        "  {name:<28} {ns:>9.1} ns/op   {:>10.2} M ops/s",
        1000.0 / ns
    );
}

fn main() {
    println!("openmorph hot-path bench (release recommended)\n");

    // distance() — the no-alloc hot loop (navigation / retrieval / foraging call it thousands of times)
    {
        let mut f = Field::new(256);
        let mut s = 0u64;
        for _ in 0..20_000 {
            f.metric_step(&[(s as usize) % 256, (s as usize * 7 + 3) % 256], 0.1, 8);
            s = s.wrapping_add(1);
        }
        let mut acc = 0.0f32;
        bench("distance(a,b,0)", 2_000_000, || {
            let a = (s as usize) % 256;
            let b = (s as usize * 13 + 1) % 256;
            acc += f.distance(a, b, 0);
            s = s.wrapping_add(1);
        });
        std::hint::black_box(acc);
    }

    // metric_step() — map training
    {
        let mut f = Field::new(256);
        let mut s = 0u64;
        bench("metric_step(2 cells, n_neg=8)", 200_000, || {
            f.metric_step(&[(s as usize) % 256, (s as usize * 7 + 3) % 256], 0.1, 8);
            s = s.wrapping_add(1);
        });
    }

    // alive(Perceive) — learning at inference (grow + self-size), one call per event
    {
        let mut f = Field::new(4);
        let mut s = 1u64;
        let rnd = |s: &mut u64| {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
            (*s >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
        };
        bench("alive(Perceive) step", 20_000, || {
            let x = [rnd(&mut s), rnd(&mut s), rnd(&mut s)];
            let inp = [(0, x[0]), (1, x[1]), (2, x[2])];
            f.alive(Event::Perceive {
                inputs: &inp,
                target_cell: 3,
                target: (0.6 * x[0] - 0.4 * x[1]).tanh(),
                lr: 0.1,
                budget: 100,
            });
        });
    }

    // bind + complete — the fast few-shot path (ARC / suggestions)
    {
        let mut f = Field::new(16);
        let obs: Vec<(usize, f32)> = (0..16).map(|i| (i, (i % 3) as f32)).collect();
        bench("bind(16-cell grid)", 200_000, || f.bind(&obs));
        bench("complete(cell)", 500_000, || {
            std::hint::black_box(f.complete(0));
        });
    }
}
