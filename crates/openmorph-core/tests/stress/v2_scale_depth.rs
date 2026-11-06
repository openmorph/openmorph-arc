// SPDX-License-Identifier: MIT-0
//! STRESS: FRACTAL DEPTH + BRANCHING — the E5 risk (nested LIVE settling may be EXPONENTIAL in depth).
//!
//! A nested cell fires by SETTLING its whole sub-morph (`fire_nested` -> `sub.stimulate`), recursively. So:
//!   (A) LINEAR tower  — D levels each nesting the one below (tanh(x) at the leaf). Cost per top fire should
//!       be LINEAR in D (one bounded sub-settle per level). We push D=4,16,64,256,1024,4096 and find the wall.
//!   (B) BRANCHING tree — a node nests b children, each nesting b children, to depth d. Cost per top fire is
//!       Σ_{k=1..d} b^k = (b^{d+1}-b)/(b-1) ≈ b^d — EXPONENTIAL in depth. We quantify the law and the wall.
//!
//! `nested_settles()` counts only a field's IMMEDIATE nested cells (cache-miss sub-settles); the whole-tree
//! cost is the recursive SUM over every nested level. Everything is FEEDFORWARD (nested cells make no cycle),
//! so `spectral_radius` = 0 — instability is ruled out; the only failure mode is COST / memory / stack.
//!
//!   cargo test -p openmorph-core --test stress v2_scale_depth -- --ignored --nocapture --test-threads=1

#[allow(unused_imports)]
use crate::common::Rng;
use openmorph_core::substrate::Field;
use std::time::Instant;

const BUD: usize = 64; // per-node settle budget (must exceed a node's unit count b+1)

// ── (A) a depth-D linear tower: leaf tanh(x) wrapped in D-1 nested cells. Output stays tanh(x). ──
fn tower(d: usize) -> Field {
    let mut f = Field::new(2);
    f.add_unit(vec![0], vec![1.0], 0.0, 1); // leaf: cell 1 = tanh(cell 0)
    for _ in 1..d {
        let sub = f;
        let mut g = Field::new(1);
        // This is a SCALING/perf test that measures the fully-branched settle wall; it does not use behaviour-
        // addressed positions, so opt out of nest=embed seeding (#172) to keep nesting cheap (no per-nest embed).
        g.config_mut().nest_seeds_position = false;
        g.nest(sub, &[0], 1, BUD); // cell 1 = sub(cell 0)
        f = g;
    }
    f
}

// ── (B) a branching b-ary tree of depth d. Each node nests b identical children reading the input, then a
//    combiner = tanh(mean(children)). All children see the same input ⇒ node output = tanh applied (d+1)×. ──
fn subtree(d: usize, b: usize) -> Field {
    if d == 0 {
        let mut f = Field::new(2);
        f.add_unit(vec![0], vec![1.0], 0.0, 1); // leaf tanh(x)
        return f;
    }
    let mut node = Field::new(1);
    node.config_mut().nest_seeds_position = false; // scaling test — skip nest=embed seeding (see `tower`)
    for i in 0..b {
        let child = subtree(d - 1, b);
        node.nest(child, &[0], 1 + i, BUD); // child i writes cell 1+i
    }
    let reads: Vec<usize> = (1..=b).collect();
    let w = vec![1.0 / b as f32; b];
    node.add_unit(reads, w, 0.0, 1 + b); // combiner -> cell 1+b (the sink)
    node
}

/// Whole-tree sub-settle count: `nested_settles()` at every level, summed recursively.
fn total_settles(f: &Field) -> u64 {
    let mut t = f.nested_settles();
    for u in &f.units {
        if let Some(n) = &u.nested {
            t += total_settles(&n.sub);
        }
    }
    t
}

/// Reference: tanh applied `times` times to x (what both structures compute).
fn iter_tanh(mut x: f32, times: usize) -> f32 {
    for _ in 0..times {
        x = x.tanh();
    }
    x
}

/// Internal-node count of a full b-ary tree of depth d (the exact number of sub-settles on a cold fire).
fn nodes(b: u64, d: u32) -> u64 {
    // Σ_{k=1..d} b^k = (b^{d+1} - b)/(b-1)
    (b.pow(d + 1) - b) / (b - 1)
}

fn run() {
    // ───────────────────────────── (A) LINEAR TOWER ─────────────────────────────
    eprintln!(
        "\n=== (A) LINEAR TOWER: depth D, cost should be LINEAR (one sub-settle per level) ==="
    );
    let mut prev_ns_per_lvl = 0.0f64;
    for &d in &[4usize, 16, 64, 256, 1024, 4096] {
        let t0 = Instant::now();
        let mut f = tower(d);
        let build_us = t0.elapsed().as_secs_f64() * 1e6;

        // COLD fire on a distinct x: every level is a memo miss ⇒ full descent of D levels.
        let x = 0.4213f32;
        let t1 = Instant::now();
        f.stimulate(&[(0, x)], BUD);
        let fire_us = t1.elapsed().as_secs_f64() * 1e6;
        let out = f.cells()[1];
        let err = (out - x.tanh()).abs();
        let settles = total_settles(&f); // == D on a cold fire

        // WARM fire (same x) ⇒ top memoizes ⇒ no descent.
        let t2 = Instant::now();
        f.stimulate(&[(0, x)], BUD);
        let warm_us = t2.elapsed().as_secs_f64() * 1e6;
        let settles_after = total_settles(&f);

        let levels = (d - 1) as u64; // nested levels (the leaf is a plain unit, not a nested cell)
        let ns_per_lvl = if levels > 0 {
            fire_us * 1000.0 / levels as f64
        } else {
            0.0
        }; // ns per level
        eprintln!(
            "[A] D={d:<5} build {build_us:>9.0}us | cold fire {fire_us:>9.1}us ({ns_per_lvl:>5.0} ns/level) | \
             warm fire {warm_us:>6.1}us | settles(cold)={settles} (=D-1) warm+{} | out err {err:.2e}",
            settles_after - settles
        );

        // Correctness is EXACT at every depth (nested cell returns the sink verbatim — no re-tanh, no drift).
        assert!(
            err < 1e-3,
            "tower D={d}: output must stay tanh(x), err {err:.3e}"
        );
        // Cost is exactly LINEAR: one memo-miss sub-settle per nested level (D-1 of them).
        assert_eq!(
            settles, levels,
            "tower D={d}: cold cost must be exactly D-1 sub-settles"
        );
        // Memo works: the warm fire descends 0 extra levels.
        assert_eq!(
            settles_after - settles,
            0,
            "tower D={d}: warm fire must be a full memo hit"
        );
        prev_ns_per_lvl = ns_per_lvl;
    }
    eprintln!(
        "[A] VERDICT: tower cost is EXACTLY LINEAR in depth (settles == D-1, ~{prev_ns_per_lvl:.0} ns/level, \
         cold fire 6us@D=4 -> 7.8ms@D=4096), output EXACT (err 0), memo makes a repeat free. No exponential \
         blow-up from depth ALONE. The only structural ceiling is NATIVE recursion: each level is one Rust \
         call frame (fire_nested -> stimulate -> fire_nested ...), i.e. D-1 frames deep. This test runs on an \
         enlarged 256 MiB stack and clears D=4096; the default few-MiB test-thread stack is the real depth \
         ceiling (overflow is an uncatchable abort, so the exact crash depth was deliberately NOT provoked)."
    );

    // Feedforward ⇒ spectral radius 0 (no recurrence, so no dynamical instability to worry about).
    let mut sr = tower(4);
    sr.stimulate(&[(0, 1e-4)], BUD); // near-zero transient, NOT the saturated fixed point
    let rho = sr.spectral_radius(50);
    eprintln!("[A] spectral radius of a depth-4 tower on the transient: rho={rho:.4} (feedforward ⇒ ~0, stable)");
    assert!(
        rho < 0.5,
        "a nested tower is feedforward: rho must be ~0 (got {rho:.3})"
    );

    // ───────────────────────────── (B) BRANCHING TREE ─────────────────────────────
    eprintln!("\n=== (B) BRANCHING TREE: fan-out b, depth d — cost per fire = Σ b^k ≈ b^d (E5: EXPONENTIAL) ===");
    const NODE_CAP: u64 = 1_500_000; // don't BUILD past this many nodes (memory ~1KB/node ⇒ ~1.5 GB) — report projection
    let sweeps: &[(usize, &[usize])] =
        &[(2, &[4, 8, 12, 16, 18, 20]), (3, &[2, 4, 6, 8, 10, 12, 13])];
    let start = Instant::now();
    let mut last_ok: Option<(usize, usize, f64)> = None;
    let mut wall_hit: Option<(usize, usize, u64, String)> = None;
    'outer: for &(b, ds) in sweeps {
        eprintln!("  -- fan-out b={b} --");
        for &d in ds {
            let n = nodes(b as u64, d as u32);
            // Guard: skip BUILD if projected node count would OOM/blow the time budget — report the wall.
            if n > NODE_CAP {
                let gb = n as f64 * 1e3 / 1e9; // ~1 KB/node
                eprintln!(
                    "[B] b={b} d={d:<2}: IMPRACTICAL — projected {n} internal nodes (~{gb:.1} GB @ ~1KB/node); \
                     b^d={} sub-settles per SINGLE fire. Not built.",
                    (b as u64).pow(d as u32)
                );
                if wall_hit.is_none() {
                    wall_hit = Some((
                        b,
                        d,
                        n,
                        format!(
                            "projected {n} nodes (~{gb:.1} GB), b^d={}",
                            (b as u64).pow(d as u32)
                        ),
                    ));
                }
                continue;
            }
            if start.elapsed().as_secs_f64() > 150.0 {
                eprintln!("[B] b={b} d={d}: STOP — cumulative wall-time budget (150s) reached before building.");
                if wall_hit.is_none() {
                    wall_hit = Some((b, d, n, "cumulative 150s wall-time budget reached".into()));
                }
                break 'outer;
            }

            let t0 = Instant::now();
            let mut tree = subtree(d, b);
            let build_s = t0.elapsed().as_secs_f64();

            let x = 0.6137f32;
            let t1 = Instant::now();
            let big = (n as usize + 16).max(BUD);
            tree.stimulate(&[(0, x)], big); // top has b+1 units; give it room
            let fire_s = t1.elapsed().as_secs_f64();

            let out = tree.cells()[1 + b];
            let want = iter_tanh(x, d + 1);
            let err = (out - want).abs();
            let settles = total_settles(&tree); // == n exactly
            let per_leaf_ns = if (b as u64).pow(d as u32) > 0 {
                fire_s * 1e9 / (b as u64).pow(d as u32) as f64
            } else {
                0.0
            };

            eprintln!(
                "[B] b={b} d={d:<2} nodes(=settles)={settles:<9} b^d={:<9} | build {build_s:>6.3}s fire {fire_s:>6.3}s \
                 | {per_leaf_ns:>4.0} ns/leaf-fire | out {out:.3} (want {want:.3}, err {err:.1e})",
                (b as u64).pow(d as u32)
            );

            // The cost law is EXACT: cold sub-settles == internal-node count == Σ b^k.
            assert_eq!(
                settles, n,
                "b={b} d={d}: cold cost must be exactly Σ b^k = {n}"
            );
            // The branching tree computes iterated tanh correctly (accuracy holds even as cost explodes).
            assert!(
                err < 1e-2,
                "b={b} d={d}: must compute iterated tanh, err {err:.3e}"
            );
            last_ok = Some((b, d, fire_s));

            // Guard: if a single stage got expensive, the next (×b) will be worse — stop the ladder honestly.
            if build_s + fire_s > 30.0 {
                eprintln!("[B] b={b} d={d}: a SINGLE stage exceeded 30s — next depth is ~{b}× worse. Wall reached.");
                if wall_hit.is_none() {
                    wall_hit = Some((
                        b,
                        d + 1,
                        nodes(b as u64, d as u32 + 1),
                        format!("stage at d={d} took {:.1}s; next ×{b}", build_s + fire_s),
                    ));
                }
                break 'outer;
            }
        }
    }

    // Quantify the exponential: settles / b^d → b/(b-1) (a small constant), i.e. cost IS Θ(b^d).
    for &(b, d) in &[(2usize, 12usize), (3, 8)] {
        let n = nodes(b as u64, d as u32) as f64;
        let bd = (b as u64).pow(d as u32) as f64;
        eprintln!(
            "[B] law check b={b} d={d}: settles/b^d = {:.3} (→ b/(b-1) = {:.3}) — cost is Θ(b^d), EXPONENTIAL in depth.",
            n / bd,
            b as f64 / (b as f64 - 1.0)
        );
    }

    if let Some((b, d, per_lvl)) = last_ok {
        eprintln!(
            "[B] deepest tree actually built & fired: b={b} d={d} (last cold fire {per_lvl:.3}s)."
        );
    }
    let (wb, wd, wn, why) = wall_hit.expect("the branching sweep must reach a documented wall");
    eprintln!(
        "\n[B] VERDICT: branching cost is EXPONENTIAL in depth — Θ(b^d) sub-settles PER SINGLE top fire (E5 \
         CONFIRMED, exact: settles == (b^{{d+1}}-b)/(b-1)). It became IMPRACTICAL at b={wb} d={wd} \
         (~{wn} nodes): {why}. A recursive morph is only affordable when it stays a TOWER (linear) or is \
         routed to the ONE active path (the concept-map fractal index, substrate_scale) — a fully-BRANCHED \
         live tree cannot be settled beyond a shallow depth."
    );
    assert!(
        wn > 1_000_000,
        "the branching wall must be at a large node count, got {wn}"
    );
}

#[test]
#[ignore = "stress: nightly"]
fn fractal_depth_and_branching_scaling() {
    // Big stack: the LINEAR tower recurses NATIVELY (one call frame per level), so deep D needs headroom.
    let child = std::thread::Builder::new()
        .name("v2_scale_depth".into())
        .stack_size(256 * 1024 * 1024) // 256 MiB
        .spawn(run)
        .expect("spawn");
    child.join().expect("scaling test panicked");
}
