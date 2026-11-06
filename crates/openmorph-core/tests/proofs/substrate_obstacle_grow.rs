// SPDX-License-Identifier: MIT-0
//! #145 — the GROWN obstacle rule (the growth half `substrate_obstacle_map` deliberately did NOT claim). There the
//! obstacle map was a harness enumeration of one world's transitions — no growth faculty. Here the morph `grow`s a
//! world model that learns, FROM BUMPING, WHICH COLOUR blocks it (the wall colour is never authored — it is
//! inferred from moves that failed), and that rule GENERALIZES: applied to a DIFFERENT wall layout it never
//! experienced, it routes around the new walls. A map that MEMORIZED the training layout's positions instead walks
//! straight into the new walls.
//!
//! The generalization is the point: the grown rule is `traversable(colour)`, position-independent, so a wall of the
//! learned colour at a NEW position is recognized as blocked. This is the perceptual growth faculty that turns the
//! enumerated obstacle map into a grown one.
//!
//! HONEST LIMIT (this still feels authored, and it partly is). The CORE is unauthored — `grow`/`relax_values` know
//! nothing of walls, colours, or games. But the SCENARIO is authored end-to-end: the walls are hand-placed, the
//! wall colour is defined, and the A/B layouts are chosen so the contrast lands; and the grown rule is a near-
//! trivial per-colour classifier, so the growth faculty is barely stretched. So this proves the MECHANISM composes
//! (grow a position-independent rule from experience → generalize → route), NOT that the morph handles real
//! perceptual structure. The genuinely-unauthored bar is the same rule over GROWN percepts on REAL ls20 frames,
//! where nothing is hand-placed — MorphLab's domain, and the open frontier.
//!
//! SUBSTRATE (grown): `grow` learns `traversable(colour)` from experienced (colour → moved?) transitions;
//! `relax_values` routes over the adjacency the rule predicts. PROTOCOL (harness): the world, sampling transitions,
//! building the adjacency from the rule's predictions, the greedy descent. Zero new core.
use openmorph_core::Field;

const W: usize = 12;
const H: usize = 9;
const NC: usize = 6; // colours 0..6; 0 = free, WALL = a wall (the morph learns which one blocks it)
const WALL: usize = 3;
const DELTAS: [(i32, i32); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)]; // (drow, dcol)

fn cell_of(p: (i32, i32)) -> usize {
    p.1 as usize * W + p.0 as usize
}
fn on_grid(p: (i32, i32)) -> bool {
    p.0 >= 0 && p.0 < W as i32 && p.1 >= 0 && p.1 < H as i32
}

struct Rng(u64);
impl Rng {
    fn new(s: u64) -> Self {
        Rng(s | 1)
    }
    fn u(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn below(&mut self, n: usize) -> usize {
        (self.u() % n as u64) as usize
    }
}

/// A world = a vertical wall at column `wx` (colour WALL) with a single free `gap` row. Free cells are colour 0.
struct World {
    wx: i32,
    gap: i32,
}
impl World {
    fn is_wall(&self, p: (i32, i32)) -> bool {
        p.0 == self.wx && p.1 != self.gap
    }
    fn colour(&self, p: (i32, i32)) -> usize {
        if self.is_wall(p) {
            WALL
        } else {
            0
        }
    }
    /// Reality: a move into a wall or off-grid is a no-op.
    fn step(&self, p: (i32, i32), a: usize) -> (i32, i32) {
        let (dr, dc) = DELTAS[a];
        let t = (p.0 + dc, p.1 + dr);
        if on_grid(t) && !self.is_wall(t) {
            t
        } else {
            p
        }
    }
}

/// One-hot ±1 encoding of a colour over `NC` input cells (the morph's percept of a cell's colour).
fn enc(c: usize) -> Vec<(usize, f32)> {
    (0..NC)
        .map(|i| (i, if i == c { 1.0 } else { -1.0 }))
        .collect()
}

/// GROW `traversable(colour)`: sample transitions in the TRAINING world and, for each, feed the TARGET cell's
/// colour and whether the body MOVED (moved ⇔ the target was free). The morph learns which colour blocks it — the
/// wall colour is never told to it, only inferred from failed moves. Returns the grown model (writes cell `NC`).
fn grow_traversable(train: &World, seed: u64) -> Field {
    let mut m = Field::new(NC + 1);
    m.config_mut().max_hidden = 0; // colour → traversable is a per-colour readout
    let mut rng = Rng::new(seed);
    for _ in 0..6000 {
        // a random occupiable cell and action → observe the target's colour and whether we moved
        let p = (rng.below(W) as i32, rng.below(H) as i32);
        if train.is_wall(p) {
            continue;
        }
        let a = rng.below(4);
        let (dr, dc) = DELTAS[a];
        let t = (p.0 + dc, p.1 + dr);
        if !on_grid(t) {
            continue; // only learn the colour rule from on-grid targets (edge-blocking is geometry, not colour)
        }
        let moved = train.step(p, a) != p; // moved ⇔ target free
        m.grow(
            &enc(train.colour(t)),
            NC,
            if moved { 1.0 } else { 0.0 },
            0.1,
            20,
        );
    }
    m
}

/// Apply the grown rule: is a cell of this colour traversable? (the morph's learned percept).
fn traversable(model: &Field, colour: usize) -> bool {
    let mut m = model.clone();
    m.stimulate(&enc(colour), 20);
    m.cells()[NC] > 0.5
}

/// Build the adjacency the GROWN RULE predicts for a world it may never have experienced: from each cell, an action
/// is traversable iff on-grid AND the grown rule says the target's colour is free.
fn adjacency_from_rule(world: &World, model: &Field) -> Vec<Vec<usize>> {
    let mut adj = vec![Vec::new(); W * H];
    for y in 0..H as i32 {
        for x in 0..W as i32 {
            if world.is_wall((x, y)) {
                continue;
            }
            for &(dr, dc) in &DELTAS {
                let t = (x + dc, y + dr);
                if on_grid(t) && traversable(model, world.colour(t)) {
                    adj[cell_of((x, y))].push(cell_of(t));
                }
            }
        }
    }
    adj
}

/// The MEMORIZED map: the true traversable adjacency of the TRAINING world (positions, not a rule). Applied to a
/// different world it is wrong.
fn adjacency_memorized(train: &World) -> Vec<Vec<usize>> {
    let mut adj = vec![Vec::new(); W * H];
    for y in 0..H as i32 {
        for x in 0..W as i32 {
            if train.is_wall((x, y)) {
                continue;
            }
            for a in 0..4 {
                let n = train.step((x, y), a);
                if n != (x, y) {
                    adj[cell_of((x, y))].push(cell_of(n));
                }
            }
        }
    }
    adj
}

/// Plan over `adj`, walk in the REAL `world` (which enforces its own walls). The agent predicts landings with the
/// SAME map it planned on (an edge in `adj` ⇒ it believes the move succeeds). Returns whether it reaches the goal.
fn plan_and_walk(world: &World, adj: &[Vec<usize>], start: (i32, i32), goal: (i32, i32)) -> bool {
    let mut planner = Field::new(W * H);
    planner.prefer(cell_of(goal), 0.0);
    planner.relax_values(adj, 1.0, 0);
    let (mut p, mut steps) = (start, 0usize);
    while p != goal && steps < 4 * (W + H) {
        let believes_edge = |a: usize| -> (i32, i32) {
            let (dr, dc) = DELTAS[a];
            let t = (p.0 + dc, p.1 + dr);
            if on_grid(t) && adj[cell_of(p)].contains(&cell_of(t)) {
                t
            } else {
                p
            }
        };
        let a = (0..4)
            .min_by(|&x, &y| {
                planner
                    .value_of(cell_of(believes_edge(x)))
                    .total_cmp(&planner.value_of(cell_of(believes_edge(y))))
            })
            .unwrap();
        p = world.step(p, a); // reality
        steps += 1;
    }
    p == goal
}

#[test]
fn grown_obstacle_rule_generalizes_to_unseen_walls() {
    // TRAIN on world A (wall column 6, gap row 1). TEST on world B (wall column 3, gap row 7) — a wall the morph
    // never experienced, at a new column AND a new gap row, same colour. Start/goal on row 4 (off both gap rows),
    // so the grown rule must detour through B's row-7 gap while the memorized A-map routes via A's row-1 gap —
    // which runs straight into B's col-3 wall.
    let world_a = World { wx: 6, gap: 1 };
    let world_b = World { wx: 3, gap: 7 };
    let (start, goal) = ((1, 4), (10, 4));

    // GROW the colour rule from bumping in A. The wall colour is inferred, never authored.
    let model = grow_traversable(&world_a, 0xB33);
    let free_ok = traversable(&model, 0);
    let wall_ok = traversable(&model, WALL);
    eprintln!("[OBSTGROW] grown rule: traversable(free 0)={free_ok}, traversable(wall {WALL})={wall_ok} (learned WHICH colour blocks, from bumping in A)");
    assert!(free_ok && !wall_ok, "the morph must LEARN the wall colour blocks and free does not (free={free_ok}, wall={wall_ok})");

    // GENERALIZE: apply the grown rule to world B (unseen layout) → adjacency → route around B's new walls.
    let adj_grown_b = adjacency_from_rule(&world_b, &model);
    let reached_grown = plan_and_walk(&world_b, &adj_grown_b, start, goal);

    // MEMORIZED: use A's true adjacency (positions) in world B → walks into B's walls (which A had free).
    let adj_memo_a = adjacency_memorized(&world_a);
    let reached_memo = plan_and_walk(&world_b, &adj_memo_a, start, goal);

    // sanity: the grown rule reproduces the true traversability of B everywhere (generalization is exact here).
    let adj_true_b = adjacency_memorized(&world_b);
    assert_eq!(
        adj_grown_b, adj_true_b,
        "the grown rule must predict world B's true connectivity (colour-general)"
    );

    eprintln!("[OBSTGROW] TEST on unseen world B (wall col 3): grown-rule reached={reached_grown}; memorized-A-map reached={reached_memo}");
    assert!(
        reached_grown,
        "the GROWN colour-rule must GENERALIZE to B's unseen walls and reach the goal"
    );
    assert!(!reached_memo, "the MEMORIZED A-position-map must FAIL in B — it walks into B's new walls (A had them free)");

    eprintln!("[OBSTGROW] VERDICT: #145 — the GROWN obstacle rule GENERALIZES (the growth half substrate_obstacle_map didn't claim). The morph `grow`s traversable(colour) from BUMPING in world A — inferring WHICH colour blocks it (wall {WALL}→blocked, free 0→open), never told the wall colour. That rule, being position-independent, TRANSFERS to world B (a wall at a new column it never experienced): it predicts B's true connectivity exactly and `relax_values` routes around B's walls to the goal (reached={reached_grown}), where a map that MEMORIZED A's wall POSITIONS walks straight into B's new walls and fails (reached={reached_memo}). This is the perceptual growth faculty — grow + generalization — that turns the enumerated obstacle map into a grown one. SUBSTRATE = grow (the colour rule) + relax_values (routing); PROTOCOL = the world + transition sampling + greedy descent (zero new core). The remaining frontier: the same rule over GROWN colour/channel percepts on real ls20 frames, composed with the invariant controllable object (transfer_invariant).");
}
