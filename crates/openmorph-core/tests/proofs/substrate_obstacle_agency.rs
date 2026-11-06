// SPDX-License-Identifier: MIT-0
//! #145 — the COMPOSED reach: BOTH grown-invariant faculties are LOAD-BEARING in ONE game-2 episode.
//!
//! Two component proofs are joined here into a single navigation task where removing EITHER grown faculty breaks
//! it (each ablation fails; only the composition reaches):
//!   - the GROWN INVARIANT OBJECT KEY (`transfer_invariant`): AGENCY — "the object I control", the WITH−WITHOUT-
//!     action move-prediction gap (`substrate_agent`) — binds the controllable body, colour/position-blind by
//!     construction. A colour-slot / position-slot key mis-binds to a distractor placed at the OLD colour+position.
//!   - the GROWN OBSTACLE RULE (`substrate_obstacle_grow`): `grow` `traversable(colour)` from BUMPING in game 1
//!     (infers WHICH colour blocks, never told); it GENERALIZES to an UNSEEN wall layout (new column+gap),
//!     predicting its true connectivity so `relax_values` routes around it. A NAIVE open-world planner walks in.
//!
//! GAME 2 (one deterministic world): a vertical wall of a wall-colour with a single gap at an UNSEEN layout; a
//! RECOLOURED + REPOSITIONED single-cell avatar; a distractor wearing the OLD colour at the OLD position; a goal.
//! Positions are chosen so the only crossing is the gap (the composed path DETOURS) and the goal differs from the
//! avatar on BOTH axes (so a mis-bound constant self-model can never reach). The SAME frozen game-1 movement skill
//! then navigates the bound avatar, walls enforced by the world.
//!
//! THE THREE RUNS (each flips exactly ONE grown faculty relative to the composition):
//!   (c) COMPOSED    = AGENCY binding  + OBSTACLE-aware field (grown rule) → REACHES (detours through the gap).
//!   (a) OBJECT-ABL  = colour/position binding (distractor) + obstacle field → FAILS: the static distractor as
//!                     self-model yields a CONSTANT action → cannot reach a both-axes goal (object faculty is
//!                     load-bearing).
//!   (b) OBSTACLE-ABL= AGENCY binding + NAIVE open-world field → FAILS: heads into a wall and LIVE-LOCKS
//!                     (obstacle faculty is load-bearing).
//!
//! SUBTLETY handled: the avatar MOVES, so the obstacle adjacency is built over the STATIC wall structure
//! (`world.colour`), never the avatar cell; and `segment_objects` is COLOUR-AWARE, so the recoloured avatar never
//! merges with the wall or the distractor.
//!
//! ZERO NEW CORE. SUBSTRATE (grown): `grow` (movement model + `traversable(colour)`) / `encapsulate` (freeze the
//! skill) / `prefer`+`relax_values`+`value_of` (route). PROTOCOL (harness): the world, the agency re-perception
//! probe, the grown-rule adjacency, the greedy descent, the two ablations. What is BY-CONSTRUCTION vs MEASURED is
//! asserted inline.
//!
//! HONEST LIMIT: a SYNTHETIC scenario — hand-placed walls/colours/positions, single-cell objects, one layout, and
//! the two grown faculties read two channels of one world (movable objects for agency, the wall colour for the
//! obstacle rule). The genuinely-unauthored bar is the SAME two faculties over GROWN percepts on REAL ls20 frames,
//! where nothing is hand-placed — MorphLab's domain, the open frontier. This proves the MECHANISMS COMPOSE and are
//! each load-bearing, not that the morph handles real perceptual structure.
use openmorph_core::grouping::segment_objects;
use openmorph_core::Field;

use crate::common::Rng;

const W: usize = 12;
const H: usize = 9;
const NC: usize = 6; // colour channel 0..6 for the obstacle rule; 0 = free, WALL blocks
const WALL: usize = 3;
const C_GOAL: usize = 5;
const DELTAS: [(i32, i32); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)]; // (drow, dcol): up, down, left, right

fn cell_of(p: (i32, i32)) -> usize {
    p.1 as usize * W + p.0 as usize
}
fn on_grid(p: (i32, i32)) -> bool {
    p.0 >= 0 && p.0 < W as i32 && p.1 >= 0 && p.1 < H as i32
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
// THE WORLD — a vertical wall at column `wx` (colour WALL) with one free `gap` row. A move into a wall / off-grid
// is a NO-OP (reality; the only way the morph learns the wall is by bumping). Reused from substrate_obstacle_grow.
// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
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

// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
// GAME-1 FACULTY 1 — the MOVEMENT SKILL (action → Δ), grown then `encapsulate`d FROZEN. (from transfer_invariant)
// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
fn onehot(a: usize, with: bool) -> Vec<(usize, f32)> {
    (0..4)
        .map(|i| (i, if with && i == a { 1.0 } else { -1.0 }))
        .collect()
}
/// Grow an action→Δ model (max_hidden=0, linear in the one-hot), `encapsulate` cells [4,5] as a frozen sub-morph.
fn bank_skill(seed: u64) -> Field {
    let mut mdl = Field::new(6);
    mdl.config_mut().max_hidden = 0;
    let mut rng = Rng::new(seed);
    for _ in 0..8000 {
        let a = rng.below(4);
        let (dr, dc) = DELTAS[a];
        mdl.grow(&onehot(a, true), 4, dr as f32, 0.1, 20);
        mdl.grow(&onehot(a, true), 5, dc as f32, 0.1, 20);
    }
    mdl.encapsulate(&[4, 5])
}
/// Apply the FROZEN skill: predict Δ for action `a`, return the landing from `pos` (clamped to the grid — the
/// movement model itself is wall-blind; reality enforces the walls).
fn apply_skill(skill: &Field, pos: (i32, i32), a: usize) -> (i32, i32) {
    let mut s = skill.clone();
    s.stimulate(&onehot(a, true), 200);
    let (dr, dc) = (s.cells()[4].round() as i32, s.cells()[5].round() as i32);
    (
        (pos.0 + dc).clamp(0, W as i32 - 1),
        (pos.1 + dr).clamp(0, H as i32 - 1),
    )
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
// GAME-1 FACULTY 2 — AGENCY: "the object I control", the WITH−WITHOUT-action move-prediction gap. Only action and
// Δ are inputs; appearance (colour/position) is structurally absent → invariant to both. (from substrate_agent)
// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
fn agency(moves: &[(i32, i32); 4], seed: u64) -> f32 {
    let train = |with: bool| -> Field {
        let mut f = Field::new(6);
        f.config_mut().max_hidden = 0;
        let mut rng = Rng::new(seed);
        for _ in 0..8000 {
            let a = rng.below(4);
            let (dr, dc) = moves[a];
            f.grow(&onehot(a, with), 4, dr as f32, 0.1, 20);
            f.grow(&onehot(a, with), 5, dc as f32, 0.1, 20);
        }
        f
    };
    let acc = |f: &mut Field, with: bool| -> f32 {
        let mut ok = 0;
        for (a, &(dr, dc)) in moves.iter().enumerate() {
            f.stimulate(&onehot(a, with), 20);
            if f.cells()[4].round() as i32 == dr && f.cells()[5].round() as i32 == dc {
                ok += 1;
            }
        }
        ok as f32 / 4.0
    };
    let (mut w, mut wo) = (train(true), train(false));
    (acc(&mut w, true) - acc(&mut wo, false)).max(0.0)
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
// GAME-1 FACULTY 3 — the OBSTACLE RULE: `grow` `traversable(colour)` from BUMPING. The wall colour is never told;
// it is inferred from moves that failed. Position-independent, so it generalizes. (from substrate_obstacle_grow)
// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
fn enc(c: usize) -> Vec<(usize, f32)> {
    (0..NC)
        .map(|i| (i, if i == c { 1.0 } else { -1.0 }))
        .collect()
}
fn grow_traversable(train: &World, seed: u64) -> Field {
    let mut m = Field::new(NC + 1);
    m.config_mut().max_hidden = 0; // colour → traversable is a per-colour readout
    let mut rng = Rng::new(seed);
    for _ in 0..6000 {
        let p = (rng.below(W) as i32, rng.below(H) as i32);
        if train.is_wall(p) {
            continue;
        }
        let a = rng.below(4);
        let (dr, dc) = DELTAS[a];
        let t = (p.0 + dc, p.1 + dr);
        if !on_grid(t) {
            continue; // edge-blocking is geometry, not colour
        }
        let moved = train.step(p, a) != p; // moved ⇔ target free
                                           // Targets are ±0.6 (NOT the ±1/0 tanh asymptote) at lr 0.02: a saturating target under the class imbalance
                                           // (free bumps ≫ wall bumps) is bistable — the readout can pin at +1 for BOTH colours before the rare wall-
                                           // bumps carve out the exception (tanh′ vanishes at the asymptote). Off-asymptote targets + a gentle lr keep
                                           // tanh′ alive, so the per-colour rule converges from UNIFORM random bumping (no authored curriculum, no
                                           // lucky seed). Zero new core — a call-site dial.
        m.grow(
            &enc(train.colour(t)),
            NC,
            if moved { 0.6 } else { -0.6 },
            0.02,
            20,
        );
    }
    m
}
/// Apply the grown rule: is a cell of this colour traversable? (raw readout > 0; free ≈ +0.6, wall ≈ −0.6.)
fn traversable(model: &Field, colour: usize) -> bool {
    let mut m = model.clone();
    m.stimulate(&enc(colour), 20);
    m.cells()[NC] > 0.0
}
/// Adjacency the GROWN rule predicts for a (possibly unseen) world: an edge iff on-grid AND the rule says the
/// target's colour is free. Built over the STATIC wall structure (`world.colour`), never the moving avatar cell.
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
/// The world's TRUE traversable adjacency (positions) — for the generalization equality check.
fn adjacency_true(world: &World) -> Vec<Vec<usize>> {
    let mut adj = vec![Vec::new(); W * H];
    for y in 0..H as i32 {
        for x in 0..W as i32 {
            if world.is_wall((x, y)) {
                continue;
            }
            for a in 0..4 {
                let n = world.step((x, y), a);
                if n != (x, y) {
                    adj[cell_of((x, y))].push(cell_of(n));
                }
            }
        }
    }
    adj
}
/// The NAIVE map: assume every on-grid move succeeds (walls invisible) — the open-world prior (the obstacle ablation).
fn naive_adjacency() -> Vec<Vec<usize>> {
    let mut adj = vec![Vec::new(); W * H];
    for y in 0..H as i32 {
        for x in 0..W as i32 {
            for &(dr, dc) in &DELTAS {
                let t = (x + dc, y + dr);
                if on_grid(t) {
                    adj[cell_of((x, y))].push(cell_of(t));
                }
            }
        }
    }
    adj
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
// THE COMPOSED REACH — the frozen skill navigates the BOUND object over a value field. `field` = a planner whose
// `relax_values` was run over the chosen adjacency; `bound_is_avatar` selects agency (tracks the avatar) vs a
// raw-property key (the fixed distractor). Reality (`world.step`) moves the body, walls enforced.
// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
fn reach(
    world: &World,
    field: &Field,
    skill: &Field,
    start: (i32, i32),
    goal: (i32, i32),
    distractor: (i32, i32),
    bound_is_avatar: bool,
) -> (bool, usize) {
    let budget = 3 * (W + H);
    let (mut av, mut steps) = (start, 0usize);
    while av != goal && steps < budget {
        let my_pos = if bound_is_avatar { av } else { distractor };
        // pick the action whose SELF-MODELLED landing has the lowest cost-to-go (argmin over value_of).
        let a = (0..4)
            .min_by(|&x, &y| {
                field
                    .value_of(cell_of(apply_skill(skill, my_pos, x)))
                    .total_cmp(&field.value_of(cell_of(apply_skill(skill, my_pos, y))))
            })
            .unwrap();
        av = world.step(av, a); // REALITY — walls enforced (a blocked move stays put → live-lock if fixated)
        steps += 1;
    }
    (av == goal, steps)
}

fn planner_over(adj: &[Vec<usize>], goal: (i32, i32)) -> Field {
    let mut p = Field::new(W * H);
    p.prefer(cell_of(goal), 0.0);
    p.relax_values(adj, 1.0, 0);
    p
}

#[test]
fn composed_agency_and_obstacle_rule_both_load_bearing() {
    // ═══ GAME 1 — grow the three faculties (movement skill, obstacle rule, agency mechanism). ═══
    let skill = bank_skill(0xB0D);
    // FROZEN guard (before): the skill's four action-landings are the true deltas — no relearning smuggled later.
    let base_landings: Vec<(i32, i32)> = (0..4).map(|a| apply_skill(&skill, (5, 4), a)).collect();
    assert_eq!(
        base_landings,
        DELTAS
            .iter()
            .map(|&(dr, dc)| (
                (5 + dc).clamp(0, W as i32 - 1),
                (4 + dr).clamp(0, H as i32 - 1)
            ))
            .collect::<Vec<_>>(),
        "the grown movement skill must predict the true action deltas (game 1)"
    );
    // Obstacle rule grown from BUMPING in GAME-1 world A (wall col 5, gap row 4). Colour never authored.
    let world_a = World { wx: 5, gap: 4 };
    let model = grow_traversable(&world_a, 0xB33);
    let (free_ok, wall_ok) = (traversable(&model, 0), traversable(&model, WALL));
    assert!(
        free_ok && !wall_ok,
        "the morph must LEARN the wall colour blocks and free does not (free={free_ok}, wall={wall_ok})"
    );
    // Agency mechanism separates a controllable body from a static object (colour/position never enter).
    let (ag_move, ag_static) = (agency(&DELTAS, 0x51), agency(&[(0, 0); 4], 0x51));
    assert!(
        ag_move > 0.5 && ag_move > ag_static + 0.5,
        "agency must separate controllable ({ag_move:.2}) from static ({ag_static:.2})"
    );

    // ═══ GAME 2 — a NEW deterministic world B: UNSEEN wall layout + recoloured/repositioned avatar + old-disguise
    //     distractor + a both-axes goal. ═══
    let world_b = World { wx: 7, gap: 1 }; // UNSEEN by the obstacle rule (new column AND new gap row)
    let c1 = 1; // GAME-1 avatar colour — becomes the distractor's disguise
    let p1 = (4i32, 5i32); // GAME-1 avatar position — the distractor sits here wearing c1
    let c2 = 2; // GAME-2 avatar recoloured c2 ≠ c1
    let p2 = (2i32, 6i32); // GAME-2 avatar repositioned p2 ≠ p1, LEFT of the wall
    let goal = (10i32, 4i32); // RIGHT of the wall; differs from the avatar on BOTH axes (2≠10, 6≠4)
                              // structural guarantees of the scenario (asserted, not assumed):
    assert!(
        p2.0 != goal.0 && p2.1 != goal.1,
        "goal must differ from the avatar on BOTH axes"
    );
    assert!(
        !world_b.is_wall(p2) && !world_b.is_wall(goal) && !world_b.is_wall(p1),
        "avatar/goal/distractor must all be on free cells in world B"
    );
    assert!(
        p2.0 < world_b.wx && goal.0 > world_b.wx,
        "avatar and goal must be on OPPOSITE sides of the wall"
    );

    // one grid of the movable-object channel (avatar/distractor/goal on the wall-painted world); segment_objects is
    // COLOUR-AWARE so the recoloured avatar never merges with the wall or the distractor.
    let grid_at = |av: (i32, i32)| -> Vec<usize> {
        let mut g = vec![0usize; W * H];
        for y in 0..H as i32 {
            for x in 0..W as i32 {
                if world_b.is_wall((x, y)) {
                    g[cell_of((x, y))] = WALL;
                }
            }
        }
        g[cell_of(p1)] = c1; // distractor: old position, old colour
        g[cell_of(av)] = c2; // avatar: recoloured, repositioned
        g[cell_of(goal)] = C_GOAL;
        g
    };
    let objs = segment_objects(&grid_at(p2), W, H, 0);
    let idx_at = |p: (i32, i32)| {
        objs.iter()
            .position(|o| o.cells.contains(&cell_of(p)))
            .unwrap()
    };
    let (ai, di) = (idx_at(p2), idx_at(p1)); // avatar, distractor object indices

    // RAW-PROPERTY keys — both mis-bind to the distractor (BY CONSTRUCTION — it is the sole object wearing c1 at p1).
    let bind_colour = objs.iter().position(|o| o.color == c1).unwrap();
    let bind_position = idx_at(p1);
    assert_eq!(
        bind_colour, di,
        "colour slot selects the old-colour distractor (by construction — the trap)"
    );
    assert_eq!(
        bind_position, di,
        "position slot selects the old-position distractor (by construction — the trap)"
    );

    // GROWN AGENCY key — for EACH candidate, re-perceive its Δ under each action by re-segmenting (the world moves
    // the BODY; the probe discovers which object responds), grow its agency scalar, bind the argmax. No colour /
    // position is read. (Reproduces transfer_invariant's per-candidate probe.)
    let observe = |target_idx: usize| -> [(i32, i32); 4] {
        let track =
            |ii: usize, av: (i32, i32)| segment_objects(&grid_at(av), W, H, 0)[ii].centroid(W);
        let base = track(target_idx, p2);
        let mut m = [(0, 0); 4];
        for (a, &(dr, dc)) in DELTAS.iter().enumerate() {
            // issue action a: the world moves ONLY the avatar (the body); re-segment and read `target`'s Δ.
            let av2 = (
                (p2.0 + dc).clamp(0, W as i32 - 1),
                (p2.1 + dr).clamp(0, H as i32 - 1),
            );
            let objs2 = segment_objects(&grid_at(av2), W, H, 0);
            let moved = objs2
                .iter()
                .map(|o| o.centroid(W))
                .min_by(|a, b| {
                    let da = (a.0 - base.0).abs() + (a.1 - base.1).abs();
                    let db = (b.0 - base.0).abs() + (b.1 - base.1).abs();
                    da.total_cmp(&db)
                })
                .unwrap();
            m[a] = (
                (moved.1 - base.1).round() as i32,
                (moved.0 - base.0).round() as i32,
            );
        }
        m
    };
    let (ag_avatar, ag_distr) = (agency(&observe(ai), 0x51), agency(&observe(di), 0x51));
    let bind_agency = if ag_avatar >= ag_distr { ai } else { di };
    // MEASURED: agency of the recoloured+repositioned avatar exceeds the distractor's, and the bound object wears
    // the NEW colour — the key demonstrably ignored the changed nuisances.
    assert!(
        ag_avatar > 0.5 && ag_avatar > ag_distr + 0.5,
        "agency must bind the avatar over the static distractor (avatar {ag_avatar:.2} vs distractor {ag_distr:.2})"
    );
    assert!(
        bind_agency == ai,
        "the grown agency key must bind the avatar object"
    );
    assert_eq!(
        objs[bind_agency].color, c2,
        "the agency-bound object wears the NEW colour (colour-blind key)"
    );

    // GROWN OBSTACLE RULE GENERALIZES to world B (never experienced): it predicts B's TRUE connectivity exactly.
    let adj_grown_b = adjacency_from_rule(&world_b, &model);
    assert_eq!(
        adj_grown_b,
        adjacency_true(&world_b),
        "the grown colour-rule must predict world B's true connectivity (unseen layout, colour-general)"
    );
    let adj_naive = naive_adjacency();

    // ═══ THE THREE RUNS — each flips exactly ONE grown faculty. ═══
    let field_obstacle = planner_over(&adj_grown_b, goal); // grown-rule value field (routes around walls)
    let field_naive = planner_over(&adj_naive, goal); // open-world value field (walls invisible)

    // (c) COMPOSED: agency binding + obstacle field → detour through the gap.
    let (reach_composed, steps_composed) =
        reach(&world_b, &field_obstacle, &skill, p2, goal, p1, true);
    // (a) OBJECT-ABLATION: colour/position binding (distractor) + obstacle field → constant action.
    let (reach_object_ablation, _) = reach(&world_b, &field_obstacle, &skill, p2, goal, p1, false);
    // (b) OBSTACLE-ABLATION: agency binding + naive field → walks into the wall.
    let (reach_obstacle_ablation, steps_obstacle) =
        reach(&world_b, &field_naive, &skill, p2, goal, p1, true);

    let manhattan = ((p2.0 - goal.0).abs() + (p2.1 - goal.1).abs()) as usize;
    eprintln!("[COMPOSE] GAME 1 grew: movement skill (Δ={base_landings:?}), obstacle rule (free={free_ok}, wall={wall_ok}), agency (ctrl {ag_move:.2} vs static {ag_static:.2}).");
    eprintln!("[COMPOSE] GAME 2 (wall col {} gap {}, avatar c{c2}@{p2:?}, distractor c{c1}@{p1:?}, goal@{goal:?}):", world_b.wx, world_b.gap);
    eprintln!("[COMPOSE]   binding — colour-slot→distractor {}, position-slot→distractor {}, AGENCY→avatar {} (avatar {ag_avatar:.2} vs distractor {ag_distr:.2}, wears c{c2})", bind_colour == di, bind_position == di, bind_agency == ai);
    eprintln!(
        "[COMPOSE]   obstacle — grown rule predicts world B's unseen walls EXACTLY (adj == true)."
    );
    eprintln!("[COMPOSE]   REACH: (c) COMPOSED reached={reach_composed} in {steps_composed} steps (detour > Manhattan {manhattan}); (a) OBJECT-abl reached={reach_object_ablation} (constant action); (b) OBSTACLE-abl reached={reach_obstacle_ablation} in {steps_obstacle} steps (live-locked at the wall).");

    // ═══ EXACT ASSERTIONS — both faculties load-bearing (each ablation fails; only the composition reaches). ═══
    assert!(
        reach_composed,
        "(c) COMPOSED — agency binding + grown obstacle rule must REACH the goal"
    );
    assert!(
        steps_composed > manhattan,
        "(c) COMPOSED must DETOUR through the gap (steps {steps_composed} > straight-line Manhattan {manhattan})"
    );
    assert!(
        !reach_object_ablation,
        "(a) OBJECT-ABLATION — the colour/position key binds the static distractor → a constant action → cannot reach a both-axes goal (object faculty is load-bearing)"
    );
    assert!(
        !reach_obstacle_ablation,
        "(b) OBSTACLE-ABLATION — the naive open-world field walks the avatar into the wall and LIVE-LOCKS (obstacle faculty is load-bearing)"
    );

    // FROZEN guard (after): the banked skill is UNCHANGED — reused (cloned + stimulated), never re-grown.
    let post_landings: Vec<(i32, i32)> = (0..4).map(|a| apply_skill(&skill, (5, 4), a)).collect();
    assert_eq!(
        post_landings, base_landings,
        "the banked skill must be UNCHANGED after the run (frozen, reused not relearned)"
    );

    eprintln!("[COMPOSE] VERDICT: #145 — BOTH grown-invariant faculties are LOAD-BEARING in ONE game-2 reach. The morph binds the RECOLOURED+REPOSITIONED avatar by GROWN AGENCY (measured {ag_avatar:.2} vs distractor {ag_distr:.2}; the object wears the NEW colour c{c2}), where a colour-slot AND a position-slot key both lock onto the distractor at the old colour+position (by construction, the trap), AND it routes around an UNSEEN wall layout by the GROWN OBSTACLE RULE (grow traversable(colour) from bumping in game 1 → predicts world B's true connectivity EXACTLY → relax_values detours through the gap). The SAME frozen game-1 movement skill then reaches the goal ({steps_composed} steps, a detour past Manhattan {manhattan}). Remove EITHER faculty and it breaks: the colour/position key binds the STATIC distractor → a constant self-model action → cannot reach a both-axes goal (0), and the naive open-world planner heads into the wall and LIVE-LOCKS ({steps_obstacle} steps, never reached). SUBSTRATE = grow/encapsulate + prefer/relax_values/value_of + segment_objects (ZERO new core); PROTOCOL = the world, the agency re-perception probe, the grown-rule adjacency, the greedy descent, the two ablations. HONEST: a SYNTHETIC scenario (hand-placed walls/colours/positions, single-cell objects, one layout, two channels of one world); the raw-key and constant-action failures are BY-CONSTRUCTION; the MEASURED content is the grown agency separation, the exact obstacle-rule generalization to the unseen walls, and the end-to-end composed reach. The genuinely-unauthored bar is the SAME two faculties over GROWN percepts on REAL ls20 frames — MorphLab's domain, the open frontier.");
}
