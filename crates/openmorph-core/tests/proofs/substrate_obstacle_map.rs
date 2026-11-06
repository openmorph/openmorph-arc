// SPDX-License-Identifier: MIT-0
//! #145 — OBSTACLE-MAP PLANNING: the SUBSTRATE half of the second multiplier MorphLab named (the obstacle map, with
//! the colour-invariant controllable object `transfer_invariant`). A morph can only reach a goal in a structured
//! world if the PLANNER routes around walls. Given the world's TRAVERSABLE CONNECTIVITY (which cell→cell moves
//! actually succeed — an obstacle is a cell with no traversable edge into it), `relax_values` produces a cost-to-go
//! field with no interior minima, so greedy descent detours around the obstacle. A naive open-world planner that
//! assumes every move succeeds walks straight into the wall and live-locks.
//!
//! TRUTH-NOT-TROPHY SCOPE (an early draft overclaimed this and was corrected). This is a PLANNER proof, NOT a
//! growth/perception one. No growth faculty runs — no `grow`, no `learn_step`, no cells. The connectivity is
//! obtained by `enumerate_connectivity`, a HARNESS probe of the transition function (try each action from each
//! cell, keep the moves that succeed — a wall is inferred from a blocked move, never read from `is_wall` into the
//! edges). That is not embodied exploration and not a grown map; the PERCEPTION half — growing the obstacle map
//! from real frames / on-policy experience — stays open (MorphLab's real-frame domain). What IS demonstrated:
//! `relax_values` routes around empty-adjacency obstacles where an open-world plan fails, and it RETURNS the
//! change-footprint of a correction (which cells' cost-to-go shifted when the wall is discovered) — a measured
//! locality of the RESULT, not a compute saving (it still sweeps all cells).
//!
//! SUBSTRATE vs PROTOCOL (ZERO new core). SUBSTRATE: `prefer`/`relax_values`/`value_of` — the value field over the
//! connectivity + its change-footprint. PROTOCOL (harness): the world (a move into a wall/edge is a no-op),
//! enumerating the connectivity, and the greedy descent.
use openmorph_core::Field;

const W: usize = 12;
const H: usize = 9;
const DELTAS: [(i32, i32); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)]; // (drow, dcol)

fn cell_of(p: (i32, i32)) -> usize {
    p.1 as usize * W + p.0 as usize
}

/// The WORLD: a vertical wall at column `wx` blocking every row except a single `gap` row. A move off-grid or into
/// a wall cell is a NO-OP (the body stays) — the only way the morph learns the wall is by trying and not moving.
struct World {
    wx: i32,
    gap: i32,
}
impl World {
    fn is_wall(&self, p: (i32, i32)) -> bool {
        p.0 == self.wx && p.1 != self.gap
    }
    fn on_grid(&self, p: (i32, i32)) -> bool {
        p.0 >= 0 && p.0 < W as i32 && p.1 >= 0 && p.1 < H as i32
    }
    /// Execute action `a` from `p`: move unless the target is off-grid or a wall (then stay).
    fn step(&self, p: (i32, i32), a: usize) -> (i32, i32) {
        let (dr, dc) = DELTAS[a];
        let t = (p.0 + dc, p.1 + dr);
        if self.on_grid(t) && !self.is_wall(t) {
            t
        } else {
            p
        }
    }
}

/// ENUMERATE the world's traversable connectivity (a harness probe of the transition function, NOT a growth
/// faculty): from every occupiable cell, try each action and keep only the moves that actually succeed. A wall is
/// inferred from a blocked move — never read from `is_wall` into an edge. (A full-grid enumeration, not embodied
/// exploration; the perception half — growing this from real frames — is out of scope, see the module note.)
fn enumerate_connectivity(world: &World) -> Vec<Vec<usize>> {
    let mut adj = vec![Vec::new(); W * H];
    for y in 0..H as i32 {
        for x in 0..W as i32 {
            let p = (x, y);
            if world.is_wall(p) {
                continue; // the morph can never stand on a wall cell, so it has no transitions from it
            }
            for a in 0..4 {
                let n = world.step(p, a);
                if n != p {
                    adj[cell_of(p)].push(cell_of(n)); // an OBSERVED successful transition
                }
            }
        }
    }
    adj
}

/// The NAIVE map: assume every move succeeds (full 4-connectivity, walls invisible) — the open-world prior.
fn naive_adjacency() -> Vec<Vec<usize>> {
    let mut adj = vec![Vec::new(); W * H];
    for y in 0..H as i32 {
        for x in 0..W as i32 {
            for &(dr, dc) in &DELTAS {
                let t = (x + dc, y + dr);
                if t.0 >= 0 && t.0 < W as i32 && t.1 >= 0 && t.1 < H as i32 {
                    adj[cell_of((x, y))].push(cell_of(t));
                }
            }
        }
    }
    adj
}

/// Plan on `adj` toward `goal`, then walk in the REAL world (which enforces the walls). The morph descends its own
/// value field, predicting landings with the same map it planned on. Returns (reached, steps).
fn plan_and_walk(
    world: &World,
    adj: &[Vec<usize>],
    knows_walls: bool,
    start: (i32, i32),
    goal: (i32, i32),
) -> (bool, usize) {
    let mut planner = Field::new(W * H);
    planner.prefer(cell_of(goal), 0.0);
    planner.relax_values(adj, 1.0, 0);
    let (mut p, mut steps) = (start, 0usize);
    let budget = 4 * (W + H);
    while p != goal && steps < budget {
        // predict each action's landing with the morph's OWN model: the wall-aware morph predicts a blocked move
        // stays put; the naive morph predicts it passes through (walls invisible).
        let predict = |a: usize| -> (i32, i32) {
            let (dr, dc) = DELTAS[a];
            let t = (p.0 + dc, p.1 + dr);
            let on_grid = t.0 >= 0 && t.0 < W as i32 && t.1 >= 0 && t.1 < H as i32;
            if knows_walls {
                world.step(p, a) // respects walls (stays if blocked)
            } else if on_grid {
                t // naive: assumes any on-grid move succeeds, walls included
            } else {
                p
            }
        };
        let a = (0..4)
            .min_by(|&x, &y| {
                planner
                    .value_of(cell_of(predict(x)))
                    .total_cmp(&planner.value_of(cell_of(predict(y))))
            })
            .unwrap();
        p = world.step(p, a); // REALITY moves the body (walls enforced). If the naive plan fixates on the
                              // through-wall action, reality keeps the body put → it live-locks until the budget.
        steps += 1;
    }
    (p == goal, steps)
}

#[test]
fn obstacle_map_planner_routes_around_walls() {
    // A vertical wall at column 6 with a single gap at row 1 (near the top). Start bottom-left, goal bottom-right —
    // the straight line hits the wall; the only path detours up through the gap and back down.
    let world = World { wx: 6, gap: 1 };
    let (start, goal) = ((2, 7), (9, 7));
    assert!(
        !world.is_wall(start) && !world.is_wall(goal),
        "start/goal must be free"
    );

    let walled = enumerate_connectivity(&world);
    let naive = naive_adjacency();

    // The walled connectivity has NO edge crossing the wall (except at the gap); the naive map has all of them.
    let wall_crossings = |adj: &[Vec<usize>]| -> usize {
        let mut n = 0;
        for y in 0..H as i32 {
            // count edges from column 5 → column 6 (the wall column) that the map believes exist
            let from = cell_of((5, y));
            let to = cell_of((6, y));
            if adj[from].contains(&to) {
                n += 1;
            }
        }
        n
    };
    let (walled_cross, naive_cross) = (wall_crossings(&walled), wall_crossings(&naive));
    assert_eq!(
        naive_cross, H,
        "the naive map believes EVERY row crosses the wall"
    );
    assert_eq!(
        walled_cross, 1,
        "the enumerated connectivity has only the GAP row crossing the wall (a wall = a blocked transition)"
    );

    // PLAN + WALK.
    let (walled_reached, walled_steps) = plan_and_walk(&world, &walled, true, start, goal);
    let (naive_reached, naive_steps) = plan_and_walk(&world, &naive, false, start, goal);

    let manhattan = ((start.0 - goal.0).abs() + (start.1 - goal.1).abs()) as usize;
    eprintln!("[OBSTACLE] wall at col 6, gap row 1; start (2,7) → goal (9,7):");
    eprintln!("[OBSTACLE]   walled connectivity: {walled_cross} wall-crossing edge (the gap); naive open-world: {naive_cross}");
    eprintln!("[OBSTACLE]   PLANNER over the obstacle map: reached={walled_reached} in {walled_steps} steps (detour > Manhattan {manhattan}); NAIVE open-world: reached={naive_reached} in {naive_steps} (live-locked at the wall)");

    assert!(
        walled_reached,
        "relax_values over the obstacle map must ROUTE AROUND the wall and reach the goal"
    );
    assert!(walled_steps > manhattan, "the route must DETOUR through the gap (longer than the straight-line Manhattan {manhattan})");
    assert!(
        !naive_reached,
        "the naive open-world planner must FAIL — it walks straight into the wall and live-locks"
    );

    // ── The re-relax FOOTPRINT: discovering the wall shifts the cost-to-go of only the far side (not the whole
    //    grid). `relax_values` RETURNS this footprint. Honest scope: it still sweeps every cell each pass — this is
    //    a measured locality of the RESULT (which cells re-route), not a compute saving. ──
    let mut planner = Field::new(W * H);
    planner.prefer(cell_of(goal), 0.0);
    planner.relax_values(&naive, 1.0, 0); // first believe the open world
    let footprint = planner.relax_values(&walled, 1.0, 0); // then correct to the walled connectivity
    eprintln!("[OBSTACLE]   re-relax footprint after discovering the wall: {footprint} of {} cells' cost-to-go shifted (only the far side re-routes)", W * H);
    assert!(footprint > 0 && footprint < W * H, "the correction footprint is LOCALIZED — only the cells whose cost-to-go shifts, not the whole grid: {footprint}/{}", W * H);

    eprintln!("[OBSTACLE] VERDICT: #145 obstacle map — the PLANNER half (the substrate half of MorphLab's second multiplier). Given the world's traversable connectivity (a wall = a blocked transition; the enumerated map has only the 1 gap-row crossing of {H}, a wall inferred from a blocked move, never read directly), `relax_values` produces a no-interior-minima cost-to-go field and greedy descent ROUTES AROUND the obstacle to the goal ({walled_steps} steps, a detour past the straight-line Manhattan {manhattan}). A NAIVE open-world planner that assumes every move succeeds heads straight at the wall and LIVE-LOCKS (never reaches). The re-relax footprint of discovering the wall is localized to the far side ({footprint}/{}), which `relax_values` returns — a measured locality of the RESULT, not a compute saving (it still sweeps all cells). TRUTH-NOT-TROPHY (an early draft was corrected): this is a PLANNER proof, NOT growth/perception — no growth faculty runs; the map is a HARNESS enumeration of the transition function, not embodied exploration or a grown perception. SUBSTRATE = prefer/relax_values/value_of (zero new core); PROTOCOL = the world + connectivity enumeration + greedy descent. Composes with `transfer_invariant` (route around obstacles + a colour-invariant controllable object); the OPEN half is GROWING the obstacle map from real frames — MorphLab's real-frame domain.", W * H);
}
