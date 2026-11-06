// SPDX-License-Identifier: MIT-0
//! GRADED VALUE-CELLS with LOCAL correction — the first v1.3.x substrate primitive for purpose/planning, now
//! IN the `Field` (`prefer` / `relax_values` / `value_of`). The investigations settled the representation:
//! value must live PER CELL with LOCAL updates — a grow/`learn_step` readout couples every cell through its one
//! shared bias, so raising the trapped cells raises them all and no gradient forms (`plan-and-correct-are-one-
//! loop.md`). A per-cell value store, relaxed locally over the connectivity the organism experiences, fixes it.
//!
//! Three properties, proven through the real substrate API:
//!   1. NO LOCAL MINIMA — `relax_values` (each cell = cost + min of connected neighbours, source clamped by
//!      `prefer`) makes descent cross a barrier a euclidean heuristic TRAPS on.
//!   2. ONLINE CORRECTION — start from a WRONG (open-world) connectivity; discover each wall by hitting it (the
//!      surprise), drop it, re-relax → the field corrects and the path opens. No pre-built isometric map.
//!   3. LAZY / PER-CELL — an online re-relax touches only the cells whose value actually shifts, and every value
//!      is independent (the shared-bias coupling is gone).
//!
//!   cargo test -p openmorph-core --test proofs substrate_valuefield -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;

const SIDE: usize = 10;
const GAP: usize = 8;
fn blocked(c: usize) -> bool {
    let (r, cc) = (c / SIDE, c % SIDE);
    cc == 5 && r != GAP
}
fn grid_nbrs(c: usize) -> Vec<usize> {
    let (r, cc) = (c / SIDE, c % SIDE);
    let mut v = vec![];
    if r + 1 < SIDE {
        v.push((r + 1) * SIDE + cc)
    }
    if r > 0 {
        v.push((r - 1) * SIDE + cc)
    }
    if cc + 1 < SIDE {
        v.push(r * SIDE + cc + 1)
    }
    if cc > 0 {
        v.push(r * SIDE + cc - 1)
    }
    v
}
fn real_nbrs(c: usize) -> Vec<usize> {
    grid_nbrs(c).into_iter().filter(|&x| !blocked(x)).collect()
}
fn eu(c: usize, g: usize) -> f32 {
    let (r, cc) = ((c / SIDE) as f32, (c % SIDE) as f32);
    let (gr, gc) = ((g / SIDE) as f32, (g % SIDE) as f32);
    ((r - gr).powi(2) + (cc - gc).powi(2)).sqrt()
}
/// Gradient descent on a value function over the REAL neighbours; stuck if no strictly-descending move exists.
fn navigate(val: &dyn Fn(usize) -> f32) -> (bool, usize) {
    let (a, b) = (0usize, SIDE - 1);
    let cap = 25 * SIDE * SIDE;
    let mut cur = a;
    for step in 0..cap {
        if cur == b {
            return (true, step);
        }
        let nb = real_nbrs(cur);
        if nb.is_empty() {
            return (false, step);
        }
        let next = *nb
            .iter()
            .min_by(|&&x, &&y| val(x).total_cmp(&val(y)))
            .unwrap();
        if val(next) >= val(cur) {
            return (false, step);
        }
        cur = next;
    }
    (false, cap)
}

#[test]
fn graded_value_cells_with_local_correction() {
    let n = SIDE * SIDE;
    let b = SIDE - 1;

    // (1) NO LOCAL MINIMA: euclidean heuristic vs the substrate's relaxed value, both on the REAL walled world.
    let (euc_ok, _) = navigate(&|c| eu(c, b));
    let mut f = Field::new(n);
    f.prefer(b, 0.0); // the PREFERRED / goal cell — clamped source
    let real_adj: Vec<Vec<usize>> = (0..n)
        .map(|c| if blocked(c) { vec![] } else { real_nbrs(c) })
        .collect();
    f.relax_values(&real_adj, 1.0, 0);
    let (val_ok, val_steps) = navigate(&|c| f.value_of(c));

    // (2)+(3) ONLINE + LAZY: believe the world is OPEN (walls look passable & get low values → they LURE the
    // planner); discover each wall by hitting it; mark it an obstacle; re-relax only where the value shifts.
    let mut g = Field::new(n);
    g.prefer(b, 0.0);
    let mut adj: Vec<Vec<usize>> = (0..n).map(grid_nbrs).collect(); // wrong: every cell looks passable
    g.relax_values(&adj, 1.0, 0);
    let (mut cur, cap) = (0usize, 25 * n);
    let (mut reached, mut surprises, mut corrected) = (false, 0usize, 0usize);
    for _ in 0..cap {
        if cur == b {
            reached = true;
            break;
        }
        let next = *adj[cur]
            .iter()
            .min_by(|&&x, &&y| g.value_of(x).total_cmp(&g.value_of(y)))
            .unwrap();
        if blocked(next) {
            adj[next].clear();
            for a in adj.iter_mut() {
                a.retain(|&x| x != next);
            }
            corrected += g.relax_values(&adj, 1.0, 0); // returns the correction footprint (lazy)
            surprises += 1;
            continue;
        }
        cur = next;
    }

    eprintln!("[VALUEFIELD] graded value CELLS in the Field (prefer / relax_values / value_of):");
    eprintln!("[VALUEFIELD]   (1) euclidean heuristic (walled) ....... reached {euc_ok}  (TRAPS — a local minimum at the wall)");
    eprintln!("[VALUEFIELD]       relaxed value (cost + min nb) ...... reached {val_ok} in {val_steps} steps  (NO local minima — descent crosses the barrier)");
    eprintln!("[VALUEFIELD]   (2) online from WRONG (open) map ....... reached {reached}  ({surprises} wall-discoveries, corrected {corrected} cell-values total — LAZY)");
    eprintln!("[VALUEFIELD] VERDICT: value/preference as GRADED CELLS with LOCAL correction is now a substrate primitive — `prefer(cell, v)` clamps a preferred/goal cell, `relax_values(adjacency, cost)` corrects each other cell to `cost + min(connected neighbours)`. It is per-cell, so the shared-bias coupling that sank the grow/learn readout is gone; the relaxed field has NO interior local minima, so descent crosses a barrier euclidean traps on; and from a WRONG connectivity, each wall is discovered by hitting it (the surprise) and only the shifted values are re-relaxed (lazy) — planning and correcting as one loop, in the right representation. This is the first v1.3.x purpose step. NEXT (below): drive the relaxation by the SETTLING cascade (diffusion in the membrane) rather than an explicit sweep, and source the adjacency from the organism's own grown connectivity — native, top-down predictive coding.");

    assert!(
        !euc_ok,
        "euclidean heuristic must trap at the barrier (motivating the value field)"
    );
    assert!(val_ok, "the substrate's relaxed value field must have no local minima — descent crosses the barrier");
    assert!(
        reached,
        "online correction from a wrong connectivity must reach the goal"
    );
    assert!(
        surprises > 0 && corrected < n * surprises,
        "correction must be online (walls discovered) and lazy"
    );
}
