// SPDX-License-Identifier: MIT-0
//! PREDICTION = TRAJECTORY EXTRAPOLATION ON THE MANIFOLD — the deepest unification: the concept-map is SPACE,
//! event-order is TIME, and together they are a spatiotemporal manifold on which the WORLD MODEL is just
//! navigation. If the organism moves with MOMENTUM (a walker that tends to keep its heading), then predicting
//! the next state is reading off where the trajectory is HEADING: `next ≈ position + velocity`, where velocity
//! is the last displacement ON THE MAP. This needs a METRIC-faithful map (so a constant real-velocity is a
//! constant map-displacement) — exactly what `metric_step` buys. Perception locates a state on the map;
//! prediction extrapolates the trajectory; both are navigation, one comparison — cell vs where you're headed.
//!
//! World: a momentum walker on a hidden grid (keeps its heading ~80%). The organism grows a metric map of the
//! grid from consecutive-state co-activation (it never sees a coordinate or a velocity), then predicts the next
//! state by `pos[cur] + (pos[cur] − pos[prev])` and reads off the nearest cell. Extrapolation should predict
//! the walker far above a "you'll move to SOME neighbour" baseline — momentum is legible as a map-velocity.
//!
//!   cargo test -p openmorph-core --test proofs substrate_trajectory -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;

use crate::common::Rng;

const SIDE: usize = 9;
const N: usize = SIDE * SIDE;

fn cell(r: i32, c: i32) -> usize {
    (r as usize) * SIDE + c as usize
}
fn dist(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

const DIRS: [(i32, i32); 4] = [(0, 1), (0, -1), (1, 0), (-1, 0)];

/// Generate a MOMENTUM walk on the grid: keep the current heading ~80% of the time (else pick a new legal
/// one). Returns the sequence of visited cells.
fn momentum_walk(steps: usize, rng: &mut Rng) -> Vec<usize> {
    let (mut r, mut c) = (SIDE as i32 / 2, SIDE as i32 / 2);
    let mut dir = rng.below(4);
    let mut seq = vec![cell(r, c)];
    let legal = |r: i32, c: i32, d: usize| {
        let (dr, dc) = DIRS[d];
        let (nr, nc) = (r + dr, c + dc);
        nr >= 0 && nc >= 0 && nr < SIDE as i32 && nc < SIDE as i32
    };
    for _ in 0..steps {
        // keep heading if legal and the momentum coin says so; otherwise pick a legal new heading.
        if !(legal(r, c, dir) && rng.below(100) < 80) {
            let opts: Vec<usize> = (0..4).filter(|&d| legal(r, c, d)).collect();
            dir = opts[rng.below(opts.len())];
        }
        let (dr, dc) = DIRS[dir];
        r += dr;
        c += dc;
        seq.push(cell(r, c));
    }
    seq
}

#[test]
fn prediction_is_trajectory_extrapolation_on_the_grown_manifold() {
    // ── Grow a METRIC map of the grid from consecutive-state co-activation (annealed metric_step). ──
    let mut org = Field::new(N);
    let mut rng = Rng::new(0x007A_7EC0);
    const ITERS: usize = 120000;
    for it in 0..ITERS {
        // short momentum bursts so consecutive states co-activate (the transitions the map learns).
        let (mut r, mut c) = (rng.below(SIDE) as i32, rng.below(SIDE) as i32);
        for _ in 0..6 {
            let opts: Vec<usize> = (0..4)
                .filter(|&d| {
                    let (dr, dc) = DIRS[d];
                    let (nr, nc) = (r + dr, c + dc);
                    nr >= 0 && nc >= 0 && nr < SIDE as i32 && nc < SIDE as i32
                })
                .collect();
            let d = opts[rng.below(opts.len())];
            let (dr, dc) = DIRS[d];
            let (nr, nc) = (r + dr, c + dc);
            let rate = 0.2 * (1.0 - it as f32 / ITERS as f32).max(0.02);
            org.metric_step(&[cell(r, c), cell(nr, nc)], rate, 12);
            r = nr;
            c = nc;
        }
    }
    let pos: Vec<Vec<f32>> = (0..N).map(|i| org.position(i)).collect();

    // ── Predict a MOMENTUM walk by extrapolating the trajectory on the map. ──
    let dims = pos[0].len();
    let mut test = Rng::new(0x7E57);
    let (mut extrap_ok, mut base_ok, mut tot) = (0.0f64, 0.0, 0.0);
    for _ in 0..400 {
        let walk = momentum_walk(30, &mut test);
        for t in 1..walk.len() - 1 {
            let (prev, cur, next) = (walk[t - 1], walk[t], walk[t + 1]);
            // velocity = last map-displacement; predicted next position = cur + velocity.
            let pred: Vec<f32> = (0..dims)
                .map(|k| pos[cur][k] + (pos[cur][k] - pos[prev][k]))
                .collect();
            let extrap = (0..N)
                .filter(|&x| x != cur)
                .min_by(|&a, &b| dist(&pos[a], &pred).total_cmp(&dist(&pos[b], &pred)))
                .unwrap();
            if extrap == next {
                extrap_ok += 1.0;
            }
            // baseline: "you'll move to SOME neighbour" — the nearest cell to cur (no velocity). Best a
            // velocity-free predictor can do: guess an adjacent cell, but not WHICH.
            let base = (0..N)
                .filter(|&x| x != cur)
                .min_by(|&a, &b| dist(&pos[a], &pos[cur]).total_cmp(&dist(&pos[b], &pos[cur])))
                .unwrap();
            if base == next {
                base_ok += 1.0;
            }
            tot += 1.0;
        }
    }
    let (extrap, base) = (100.0 * extrap_ok / tot, 100.0 * base_ok / tot);

    eprintln!(
        "[TRAJECTORY] predicting a momentum walker on the grown manifold (never saw a coordinate or a velocity):\n  TRAJECTORY EXTRAPOLATION (pos + velocity) {extrap:.0}%   vs   velocity-free nearest-neighbour baseline {base:.0}%   (chance ≈ {:.0}%)",
        100.0 / N as f32
    );
    eprintln!("[TRAJECTORY] VERDICT: the world model IS navigation on the manifold. The concept-map is SPACE, event-order is TIME; a momentum walker's next state is just where its trajectory is HEADING — `position + velocity`, velocity being the last displacement ON THE MAP. Because `metric_step` made the map metric-faithful, a constant real-velocity is a constant map-displacement, so extrapolation reads off the next state far above a velocity-free 'you'll move somewhere' baseline. Prediction, perception, and action are one act — locate on the manifold, extrapolate the trajectory, step — the async substrate's two axes (grown concept-space × event-time) fused into a spatiotemporal manifold the organism navigates.");

    assert!(
        extrap > base + 20.0 && extrap > 45.0,
        "trajectory extrapolation must predict the momentum walker far above the velocity-free baseline (extrap {extrap:.0}% vs base {base:.0}%)"
    );
}
