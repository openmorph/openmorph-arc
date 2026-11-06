// SPDX-License-Identifier: MIT-0
//! #145 — THE COMPOSED PROOF: the two grown-invariant faculties MorphLab named (`transfer_invariant`'s AGENCY-keyed
//! controllable object, `substrate_obstacle_grow`'s GROWN traversable(colour) rule) are BOTH load-bearing in ONE
//! game-2 reach. A single agent must simultaneously (i) BIND the object it controls by AGENCY — the game-2 avatar is
//! recoloured AND repositioned, and a distractor wears the OLD colour at the OLD position (so a colour-slot AND a
//! position-slot key both mis-bind to it) — and (ii) ROUTE AROUND WALLS at an UNSEEN layout via the grown obstacle
//! rule, then navigate the bound avatar to a goal that differs on BOTH axes. Two ablations isolate each faculty:
//!   (a) COLOUR/POSITION-slot object key → binds the STATIC distractor → a constant action → can't reach a both-axes
//!       goal (the OBJECT faculty is load-bearing);
//!   (b) NAIVE open-world obstacle handling (correct AGENCY binding, but walls invisible) → heads into the wall and
//!       LIVE-LOCKS (the OBSTACLE faculty is load-bearing);
//!   (c) COMPOSED (AGENCY binding + grown obstacle rule) → reaches.
//! Remove EITHER grown faculty and the reach breaks; only the composition succeeds.
//!
//! THE MOVING-AVATAR vs STATIC-OBSTACLE SUBTLETY (called out explicitly). The avatar MOVES every step, so the
//! obstacle adjacency must be built over the STATIC WALL STRUCTURE (`World`, which knows only free/wall colours),
//! NOT over the moving avatar cell. The grown `traversable(colour)` rule is queried only with the world's wall/free
//! colour; the avatar/distractor/goal are never walls, so they are traversable and the avatar can move through them.
//! The value field (`relax_values` over the walled adjacency) marks wall cells `f32::MAX`, and reality (`world.step`)
//! enforces the walls — so the plan and the body agree on the STATIC structure while the body moves within it. The
//! AGENCY probe is a separate faculty (identify the body by its action-response in open space): the avatar start is
//! chosen with a wall-free 4-neighbourhood, so the probe's clamp-only move equals the real move (no wall contrivance).
//! `segment_objects` is COLOUR-AWARE, so the distinct-coloured avatar / distractor / goal / walls never merge.
//!
//! SUBSTRATE (zero new core): `grow`/`encapsulate` (skill + agency + traversable rule), `prefer`/`relax_values`/
//! `value_of` (obstacle-aware value field), `segment_objects` (core grouping). PROTOCOL (harness): the two worlds,
//! the keys, the agency re-perception probe, and the reach loop. HONEST LIMIT: a synthetic scenario with hand-placed
//! walls/colours and a near-trivial per-colour rule — the real bar is the SAME composition over GROWN percepts on
//! real ls20 frames (MorphLab's domain). This proves the two grown faculties COMPOSE and are each necessary, not
//! that the morph handles real perceptual structure. Rigor: TWO layouts × THREE growth seeds; the naive ablation is
//! shown to genuinely live-lock (not a lucky miss) and the raw-key ablation to genuinely bind the static distractor.
use openmorph_core::grouping::segment_objects;
use openmorph_core::Field;

const W: usize = 12;
const H: usize = 9;
const NC: usize = 6; // colour channels for the obstacle rule: 0 = free, WALL = wall (the rule learns which blocks)
const WALL: usize = 3;
const C_FREE: usize = 0;
const C_GOAL: usize = 5;
const C1: usize = 1; // the game-1 avatar colour = the distractor's disguise (old colour)
const P1: (i32, i32) = (5, 4); // the game-1 avatar position = the distractor's position (old position)
const DELTAS: [(i32, i32); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)]; // (drow, dcol)

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

fn cell_of(p: (i32, i32)) -> usize {
    p.1 as usize * W + p.0 as usize
}
fn on_grid(p: (i32, i32)) -> bool {
    p.0 >= 0 && p.0 < W as i32 && p.1 >= 0 && p.1 < H as i32
}
fn onehot(a: usize, with: bool) -> Vec<(usize, f32)> {
    (0..4)
        .map(|i| (i, if with && i == a { 1.0 } else { -1.0 }))
        .collect()
}

/// A game-2 trial: a training world A (grow the rule) whose wall differs in column AND gap from the test world B
/// (so the rule must GENERALIZE), plus the avatar start and the both-axes-different goal in B.
struct Layout {
    train: World,
    test: World,
    start: (i32, i32),
    goal: (i32, i32),
}

// ── The STATIC wall structure (game 1 = a training world with walls to bump; game 2 = an UNSEEN layout) ──
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
            C_FREE
        }
    }
    /// Reality: a move into a wall or off-grid is a no-op (the only way to LEARN the wall is to bump it).
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

// ── GAME 1 grows three pieces ──

/// FACULTY 6 — the movement skill: an action→Δ model (linear in the one-hot), `encapsulate`d as a FROZEN sub-morph.
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

/// Apply the FROZEN skill: predict the Δ for action `a`, return the landing from `pos` (grid-clamped; wall-blind —
/// the walls are enforced by the value field (MAX) and by reality, per the moving-avatar/static-obstacle split).
fn apply_skill(skill: &Field, pos: (i32, i32), a: usize) -> (i32, i32) {
    let mut s = skill.clone();
    s.stimulate(&onehot(a, true), 200);
    let (dr, dc) = (s.cells()[4].round() as i32, s.cells()[5].round() as i32);
    (
        (pos.0 + dc).clamp(0, W as i32 - 1),
        (pos.1 + dr).clamp(0, H as i32 - 1),
    )
}

/// FACULTY 3 — AGENCY of an object given its OBSERVED moves under each action (the WITH−WITHOUT-action move-
/// prediction gap; colour/position never enter). Reproduced from `substrate_agent` / `transfer_invariant`.
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

/// One-hot ±1 encoding of a colour over `NC` cells (the morph's percept of a cell's colour).
fn enc(c: usize) -> Vec<(usize, f32)> {
    (0..NC)
        .map(|i| (i, if i == c { 1.0 } else { -1.0 }))
        .collect()
}

/// GROW the obstacle rule `traversable(colour)` from BUMPING in the training world: for random (cell, action), feed
/// the target's colour and whether the body MOVED (moved ⇔ target free). The wall colour is inferred, never told.
fn grow_traversable(train: &World, seed: u64) -> Field {
    let mut m = Field::new(NC + 1);
    m.config_mut().max_hidden = 0; // colour → traversable is a per-colour readout
    let mut rng = Rng::new(seed);
    for _ in 0..8000 {
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
        let moved = train.step(p, a) != p;
        // Targets are ±0.6 (NOT the ±1/0 tanh asymptote) at lr 0.02: a saturating target under class imbalance
        // (free ≫ wall bumps) is bistable — the readout can pin at +1 for BOTH colours before the rare wall-bumps
        // carve out the exception (tanh′ vanishes). Off-asymptote targets + a gentle lr keep tanh′ alive so the
        // per-colour rule converges from UNIFORM random bumping (no authored curriculum) on every seed/world.
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

/// Apply the grown rule: is a cell of this colour traversable? (raw readout > 0; free ≈ +0.6, wall ≈ −0.6).
fn traversable(model: &Field, colour: usize) -> bool {
    traversable_raw(model, colour) > 0.0
}
fn traversable_raw(model: &Field, colour: usize) -> f32 {
    let mut m = model.clone();
    m.stimulate(&enc(colour), 20);
    m.cells()[NC]
}

// ── Adjacencies (over the STATIC wall structure — NOT the moving avatar) ──

/// The adjacency the GROWN RULE predicts for a (possibly unseen) world: from each free cell, an action is
/// traversable iff on-grid AND the grown rule says the target's colour is free. Wall cells get no edges (→ MAX).
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

/// The true walled adjacency of a world (positions) — the oracle the grown rule must reproduce.
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

/// The NAIVE open-world adjacency: assume every on-grid move succeeds (walls invisible).
fn adjacency_naive() -> Vec<Vec<usize>> {
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

/// THE REACH LOOP: plan on `adj`, walk the avatar with the FROZEN skill, reality (`world.step`) enforces the walls.
/// `bound_is_avatar` selects the self-model: the AGENCY-bound avatar (tracks the body) or the mis-bound STATIC
/// distractor (constant). The value field over the walled adjacency marks wall cells MAX so descent avoids them.
fn reach(
    world: &World,
    adj: &[Vec<usize>],
    start: (i32, i32),
    goal: (i32, i32),
    distractor: (i32, i32),
    skill: &Field,
    bound_is_avatar: bool,
) -> bool {
    let mut planner = Field::new(W * H);
    planner.prefer(cell_of(goal), 0.0);
    planner.relax_values(adj, 1.0, 0);
    let mut av = start;
    for _ in 0..4 * (W + H) {
        if av == goal {
            return true;
        }
        let my_pos = if bound_is_avatar { av } else { distractor };
        let a = (0..4)
            .min_by(|&x, &y| {
                planner
                    .value_of(cell_of(apply_skill(skill, my_pos, x)))
                    .total_cmp(&planner.value_of(cell_of(apply_skill(skill, my_pos, y))))
            })
            .unwrap();
        av = world.step(av, a); // reality moves the body; a blocked move is a no-op (walls enforced)
    }
    av == goal
}

/// Build the game-2 grid for segmentation: walls (static structure) + the three coloured objects on free cells.
fn build_grid(
    world: &World,
    avatar: (i32, i32),
    avatar_colour: usize,
    goal: (i32, i32),
) -> Vec<usize> {
    let mut grid = vec![C_FREE; W * H];
    for y in 0..H as i32 {
        for x in 0..W as i32 {
            if world.is_wall((x, y)) {
                grid[cell_of((x, y))] = WALL;
            }
        }
    }
    grid[cell_of(P1)] = C1; // distractor: old position, old colour
    grid[cell_of(avatar)] = avatar_colour; // avatar: recoloured + repositioned
    grid[cell_of(goal)] = C_GOAL;
    grid
}

#[test]
fn both_grown_invariants_are_load_bearing_in_one_reach() {
    // Two GAME-2 layouts (unseen walls, both-axes-different goals, a genuine detour). Each is paired with a game-1
    // TRAINING world whose wall is at a DIFFERENT column+gap, so the grown obstacle rule must GENERALIZE, not recall.
    // (world_a, world_b, avatar_start p2, goal)
    let layouts = [
        Layout {
            train: World { wx: 3, gap: 7 },
            test: World { wx: 6, gap: 1 },
            start: (2, 5),
            goal: (9, 3),
        },
        Layout {
            train: World { wx: 4, gap: 2 },
            test: World { wx: 8, gap: 7 },
            start: (3, 2),
            goal: (10, 5),
        },
    ];

    for (li, lay) in layouts.iter().enumerate() {
        let (world_a, world_b, p2, goal) = (&lay.train, &lay.test, &lay.start, &lay.goal);
        // p2 (avatar) and goal must be off the wall and off the distractor; goal differs on BOTH axes; p2 has a
        // wall-free 4-neighbourhood (so the agency probe's clamp-move == the real move — no wall contrivance).
        assert!(!world_b.is_wall(*p2) && !world_b.is_wall(*goal) && !world_b.is_wall(P1));
        assert!(*p2 != P1 && *goal != P1 && p2.0 != goal.0 && p2.1 != goal.1);
        assert!(
            DELTAS.iter().all(|&(dr, dc)| {
                let n = (p2.0 + dc, p2.1 + dr);
                on_grid(n) && !world_b.is_wall(n)
            }),
            "avatar start must have a wall-free 4-neighbourhood for a clean agency probe"
        );

        for &seed in &[0xB0Du64, 0x5EED, 0xA11CE] {
            // ═══ GAME 1 — grow the three pieces (once per seed; the skill/rule are position-independent) ═══
            let skill = bank_skill(seed);
            // FROZEN guard (before): the four action-landings.
            let base_landings: Vec<(i32, i32)> =
                (0..4).map(|a| apply_skill(&skill, (5, 4), a)).collect();
            let model = grow_traversable(world_a, seed ^ 0xB33);

            // GROWN RULE learned WHICH colour blocks (from bumping in world A), never told.
            let free_ok = traversable(&model, C_FREE);
            let wall_ok = traversable(&model, WALL);
            eprintln!(
                "[COMPOSE] L{li} s{seed:#x}: traversable_raw free={:.3} wall={:.3}",
                traversable_raw(&model, C_FREE),
                traversable_raw(&model, WALL)
            );
            assert!(
                free_ok && !wall_ok,
                "L{li} s{seed:#x}: rule must learn free open / wall blocks (free={free_ok}, wall={wall_ok})"
            );

            // ═══ GAME 2 — one world, both faculties needed ═══
            let c2 = 2 + (seed as usize % 2) * 2; // ∈ {2,4}: new avatar colour ≠ free/C1/WALL/C_GOAL
            assert!(c2 != C_FREE && c2 != C1 && c2 != WALL && c2 != C_GOAL);
            let grid = build_grid(world_b, *p2, c2, *goal);
            let objs = segment_objects(&grid, W, H, C_FREE);
            let idx_at = |p: (i32, i32)| {
                objs.iter()
                    .position(|o| o.cells.contains(&cell_of(p)))
                    .unwrap()
            };
            let (ai, di) = (idx_at(*p2), idx_at(P1)); // avatar, distractor object indices

            // ── (a-trap) RAW-PROPERTY keys mis-bind to the static distractor (by construction — the distractor is
            //    the SOLE object wearing the old colour C1 at the old position P1). ──
            let bind_colour = objs.iter().position(|o| o.color == C1).unwrap();
            let bind_position = idx_at(P1);
            assert_eq!(
                bind_colour, di,
                "L{li} s{seed:#x}: colour slot → old-colour distractor"
            );
            assert_eq!(
                bind_position, di,
                "L{li} s{seed:#x}: position slot → old-position distractor"
            );

            // ── GROWN AGENCY key: re-perceive each candidate's Δ by re-segmenting (the world moves ONLY the body),
            //    grow its agency scalar, bind argmax. Appearance is never an input. ──
            let observe = |target_idx: usize| -> [(i32, i32); 4] {
                let base = segment_objects(&grid, W, H, C_FREE)[target_idx].centroid(W);
                let mut m = [(0, 0); 4];
                for (a, &(dr, dc)) in DELTAS.iter().enumerate() {
                    // issue action a: only the avatar (the body) moves; p2's neighbourhood is wall-free so the
                    // clamp-only move equals the real move. Re-segment and read `target`'s Δ by nearest centroid.
                    let av2 = (
                        (p2.0 + dc).clamp(0, W as i32 - 1),
                        (p2.1 + dr).clamp(0, H as i32 - 1),
                    );
                    let g2 = build_grid(world_b, av2, c2, *goal);
                    let objs2 = segment_objects(&g2, W, H, C_FREE);
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
            let ag_avatar = agency(&observe(ai), 0x51);
            let ag_distr = agency(&observe(di), 0x51);
            let bind_agency = if ag_avatar >= ag_distr { ai } else { di };
            assert!(
                ag_avatar > 0.5 && ag_avatar > ag_distr + 0.5,
                "L{li} s{seed:#x}: agency must separate avatar ({ag_avatar:.2}) from distractor ({ag_distr:.2})"
            );
            assert!(
                bind_agency == ai && objs[ai].color == c2,
                "L{li} s{seed:#x}: agency binds the recoloured (c2={c2}) + repositioned avatar (measured)"
            );

            // ── GROWN RULE predicts world B's UNSEEN connectivity exactly (generalization from bumping in A). ──
            let adj_grown = adjacency_from_rule(world_b, &model);
            assert_eq!(
                adj_grown,
                adjacency_true(world_b),
                "L{li} s{seed:#x}: grown rule must predict world B's true connectivity"
            );
            let adj_naive = adjacency_naive();

            // ═══ THE THREE OUTCOMES ═══
            // (c) COMPOSED — agency binding + grown obstacle rule → reaches.
            let composed = reach(world_b, &adj_grown, *p2, *goal, P1, &skill, true);
            // (a) OBJECT ablation — grown obstacle rule kept, but self-model = the static distractor → constant
            //     action → cannot reach a both-axes goal.
            let object_ablation = reach(world_b, &adj_grown, *p2, *goal, P1, &skill, false);
            // (b) OBSTACLE ablation — correct agency binding, but NAIVE open-world value field → walks into the wall.
            let obstacle_ablation = reach(world_b, &adj_naive, *p2, *goal, P1, &skill, true);

            // The naive ablation must GENUINELY LIVE-LOCK (fixate stuck), not merely run out of budget mid-path:
            // trace it and confirm the body stops moving well before the goal (a fixed point off the goal).
            let mut planner = Field::new(W * H);
            planner.prefer(cell_of(*goal), 0.0);
            planner.relax_values(&adj_naive, 1.0, 0);
            let mut av = *p2;
            let mut stuck_at: Option<(i32, i32)> = None;
            for _ in 0..4 * (W + H) {
                let a = (0..4)
                    .min_by(|&x, &y| {
                        planner
                            .value_of(cell_of(apply_skill(&skill, av, x)))
                            .total_cmp(&planner.value_of(cell_of(apply_skill(&skill, av, y))))
                    })
                    .unwrap();
                let nxt = world_b.step(av, a);
                if nxt == av {
                    stuck_at = Some(av); // reality refused the move → live-lock
                    break;
                }
                av = nxt;
            }
            let live_locked = stuck_at.is_some_and(|s| s != *goal);

            eprintln!(
                "[COMPOSE] L{li} s{seed:#x}: agency avatar {ag_avatar:.2} vs distractor {ag_distr:.2}; \
                 reach composed={composed}, object-ablation={object_ablation}, obstacle-ablation={obstacle_ablation}; \
                 naive live-locked at {stuck_at:?}"
            );

            // ── EXACT ASSERTIONS: only the composition reaches; each ablation fails for its OWN reason. ──
            assert!(
                composed,
                "L{li} s{seed:#x}: COMPOSED (agency + grown obstacle rule) must reach"
            );
            assert!(
                !object_ablation,
                "L{li} s{seed:#x}: OBJECT ablation (distractor self-model) must FAIL — a constant action can't reach a both-axes goal"
            );
            assert!(
                !obstacle_ablation,
                "L{li} s{seed:#x}: OBSTACLE ablation (naive open-world) must FAIL — it walks into the wall"
            );
            assert!(
                live_locked,
                "L{li} s{seed:#x}: the naive ablation must GENUINELY LIVE-LOCK (a fixed point off the goal), not a lucky miss"
            );

            // FROZEN guard (after): the banked skill is unchanged — reused (cloned + stimulated), never re-grown.
            let post_landings: Vec<(i32, i32)> =
                (0..4).map(|a| apply_skill(&skill, (5, 4), a)).collect();
            assert_eq!(
                post_landings, base_landings,
                "L{li} s{seed:#x}: the banked skill must be UNCHANGED (frozen, reused not relearned)"
            );
        }
    }

    eprintln!(
        "[COMPOSE] VERDICT: #145 — BOTH grown-invariant faculties are LOAD-BEARING in ONE game-2 reach, across 2 \
         layouts × 3 seeds. The avatar is recoloured AND repositioned with a distractor at the old colour+position, \
         and the walls are at an UNSEEN layout. Only the COMPOSITION (AGENCY-bound controllable object + grown \
         traversable(colour) rule → relax_values value field) reaches the both-axes goal. Ablating the OBJECT \
         faculty (colour/position slot → the static distractor) yields a constant action that cannot reach; ablating \
         the OBSTACLE faculty (naive open-world) live-locks against the wall. The moving-avatar/static-obstacle split \
         is explicit: the obstacle adjacency is over the STATIC wall structure, walls enforced by the value field \
         (MAX) and by reality, while the body moves within it. SUBSTRATE = grow/encapsulate + prefer/relax_values/\
         value_of + segment_objects (ZERO new core); PROTOCOL = the worlds, keys, agency probe, reach loop. HONEST \
         LIMIT: synthetic, hand-placed walls/colours, a near-trivial per-colour rule — the real bar is this same \
         composition over GROWN percepts on real ls20 frames (MorphLab's domain)."
    );
}
