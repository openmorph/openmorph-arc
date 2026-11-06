// SPDX-License-Identifier: MIT-0
//! #145 — the TWO grown-invariant perception faculties COMPOSED into one agent, BOTH load-bearing. It fuses
//! `transfer_invariant` (the colour/position-invariant controllable object, via AGENCY) and
//! `substrate_obstacle_grow` (the grown obstacle rule that generalizes to unseen walls). In a transfer game where
//! the avatar is recoloured AND repositioned, a distractor wears the old colour, AND the walls are at a layout the
//! morph never experienced, the composed agent (i) binds the controllable object by agency and (ii) routes around
//! the walls by the grown obstacle rule, then navigates to the goal. Removing EITHER grown faculty breaks it: a
//! COLOUR-object key binds the static distractor (old colour) so the self-model never moves → a constant action →
//! fails; a NAIVE-obstacle plan sends the agency-bound avatar into a wall it cannot see → live-locks. Only agency +
//! grown obstacle rule reaches, so both grown invariances are necessary, not decorative.
//!
//! HONEST LIMIT (the standing caveat for these synthetic proofs). The CORE is unauthored (grow/relax_values/agency
//! are task-blind). But the SCENARIO is authored — hand-placed walls, defined colours, positions chosen so the
//! detour and the ablations land. This proves the two grown MECHANISMS COMPOSE and are each load-bearing; it does
//! NOT prove the morph handles real perceptual structure. The genuinely-unauthored bar is the same faculties over
//! GROWN percepts on REAL ls20 frames (MorphLab's domain).
//!
//! SUBSTRATE (grown, zero new core): `grow` (the movement skill + agency predictors + the obstacle rule),
//! `encapsulate` (bank the skill), `prefer`/`relax_values`/`value_of` (the obstacle-aware planner),
//! `segment_objects` (object perception). PROTOCOL (harness): the world, the agency probe, building the adjacency
//! from the grown rule, the greedy descent.
use openmorph_core::grouping::segment_objects;
use openmorph_core::Field;

const W: usize = 14;
const H: usize = 10;
const WALL: usize = 3; // the wall colour the morph must LEARN blocks it
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

fn onehot(a: usize, with: bool) -> Vec<(usize, f32)> {
    (0..4)
        .map(|i| (i, if with && i == a { 1.0 } else { -1.0 }))
        .collect()
}

// ── a world = a vertical wall (colour WALL) at column `wx` with a single free `gap` row ──
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

// ── FACULTY 1: the grown movement skill + AGENCY (colour/position-invariant controllable object) ──

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

fn apply_skill(skill: &Field, pos: (i32, i32), a: usize) -> (i32, i32) {
    let mut s = skill.clone();
    s.stimulate(&onehot(a, true), 200);
    let (dr, dc) = (s.cells()[4].round() as i32, s.cells()[5].round() as i32);
    (
        (pos.0 + dc).clamp(0, W as i32 - 1),
        (pos.1 + dr).clamp(0, H as i32 - 1),
    )
}

// ── FACULTY 2: the grown obstacle rule traversable(colour) ──

const NC: usize = 6; // colour one-hot width
fn enc(c: usize) -> Vec<(usize, f32)> {
    (0..NC)
        .map(|i| (i, if i == c { 1.0 } else { -1.0 }))
        .collect()
}

/// GROW traversable(colour) by bumping in a TRAINING world (a different wall layout) — the wall colour is inferred
/// from failed moves, never told.
fn grow_traversable(train: &World, seed: u64) -> Field {
    let mut m = Field::new(NC + 1);
    m.config_mut().max_hidden = 0;
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
            continue;
        }
        let moved = train.step(p, a) != p;
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
fn traversable(model: &Field, colour: usize) -> bool {
    let mut m = model.clone();
    m.stimulate(&enc(colour), 20);
    m.cells()[NC] > 0.5
}

/// The obstacle-aware adjacency the grown rule predicts over the STATIC wall structure of `world` (NOT the movable
/// avatar): an edge exists iff the target cell's colour is traversable per the grown rule.
fn obstacle_adjacency(world: &World, model: &Field) -> Vec<Vec<usize>> {
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

/// The composed reach: descend `value_field` by the frozen skill applied to the self-model position (the avatar if
/// bound by agency; the static distractor if mis-bound by colour), executing on the TRUE avatar (walls enforced).
/// Returns (reached, steps, live_lock_cell). A live-lock is a GENUINE fixed point: the world is deterministic and
/// (self-model, value field) don't change when a move is blocked, so the same action is re-picked forever — the
/// FIRST no-move step at a non-goal cell is a true stall, not budget wandering.
fn reach(
    world: &World,
    value_field: &Field,
    skill: &Field,
    start: (i32, i32),
    goal: (i32, i32),
    bound_is_avatar: bool,
    distractor: (i32, i32),
) -> (bool, usize, Option<(i32, i32)>) {
    let mut av = start;
    for step in 0..5 * (W + H) {
        if av == goal {
            return (true, step, None);
        }
        let my_pos = if bound_is_avatar { av } else { distractor };
        let a = (0..4)
            .min_by(|&x, &y| {
                value_field
                    .value_of(cell_of(apply_skill(skill, my_pos, x)))
                    .total_cmp(&value_field.value_of(cell_of(apply_skill(skill, my_pos, y))))
            })
            .unwrap();
        let next = world.step(av, a); // reality: the world moves the body, walls enforced
        if next == av {
            return (false, step, Some(av)); // blocked, and (self-model, field) fixed ⇒ a deterministic fixed point
        }
        av = next;
    }
    (
        av == goal,
        5 * (W + H),
        if av == goal { None } else { Some(av) },
    )
}

/// Observe an object's Δ under each action by re-segmenting after the world moves the AVATAR (the body). Returns its
/// per-action move-vector — DELTAS for the avatar (interior), (0,0) for the static distractor.
fn observe(
    grid: &[usize],
    target: (i32, i32),
    avatar: (i32, i32),
    av_col: usize,
    di: (i32, i32),
    di_col: usize,
) -> [(i32, i32); 4] {
    let base = target;
    let mut m = [(0, 0); 4];
    for (a, &(dr, dc)) in DELTAS.iter().enumerate() {
        let av2 = (
            (avatar.0 + dc).clamp(0, W as i32 - 1),
            (avatar.1 + dr).clamp(0, H as i32 - 1),
        );
        // rebuild the grid with the avatar moved, re-segment, track `target` by nearest centroid
        let mut g = grid.to_vec();
        g[cell_of(avatar)] = 0;
        g[cell_of(av2)] = av_col;
        g[cell_of(di)] = di_col;
        let objs = segment_objects(&g, W, H, 0);
        let moved = objs
            .iter()
            .map(|o| o.centroid(W))
            .min_by(|a, b| {
                let da = (a.0 - base.0 as f32).abs() + (a.1 - base.1 as f32).abs();
                let db = (b.0 - base.0 as f32).abs() + (b.1 - base.1 as f32).abs();
                da.total_cmp(&db)
            })
            .unwrap();
        m[a] = (
            (moved.1 - base.1 as f32).round() as i32,
            (moved.0 - base.0 as f32).round() as i32,
        );
    }
    m
}

#[test]
fn two_grown_invariances_compose_and_are_both_load_bearing() {
    // ── GAME 1 (learn + bank): a movement skill, the avatar's agency, and the obstacle rule (trained on wall layout
    //    A — a DIFFERENT column/gap than the test, so the obstacle rule must GENERALIZE). ──
    let skill = bank_skill(0xB0D);
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
        "the grown skill must predict the true action deltas"
    );
    let obstacle = grow_traversable(&World { wx: 4, gap: 1 }, 0xB33);
    assert!(
        traversable(&obstacle, 0) && !traversable(&obstacle, WALL),
        "the obstacle rule must learn WALL blocks, free does not"
    );
    let ag_move = agency(&DELTAS, 0x51);
    let ag_static = agency(&[(0, 0); 4], 0x51);
    assert!(
        ag_move > 0.5 && ag_move > ag_static + 0.5,
        "agency must separate a controllable object from a static one"
    );

    // ── GAME 2 (transfer, everything changed): avatar recoloured c2 at a NEW position; a distractor wears the OLD
    //    colour c1 (off the path); the wall is at an UNSEEN column 7 / gap row 8; goal off both axes. ──
    let world = World { wx: 7, gap: 8 };
    let (c1, c2) = (1usize, 2usize); // c1 = game-1 avatar colour (now the distractor's disguise); c2 = recoloured avatar
    let (start, goal, distractor) = ((2, 2), (11, 6), (11, 1));
    assert!(
        !world.is_wall(start) && !world.is_wall(goal) && start.0 != goal.0 && start.1 != goal.1
    );

    // object perception: segment the scene, find the avatar (c2) and distractor (c1)
    let mut grid = vec![0usize; W * H];
    for y in 0..H as i32 {
        for x in 0..W as i32 {
            if world.is_wall((x, y)) {
                grid[cell_of((x, y))] = WALL;
            }
        }
    }
    grid[cell_of(start)] = c2;
    grid[cell_of(distractor)] = c1;
    let objs = segment_objects(&grid, W, H, 0);
    let idx_col = |c: usize| objs.iter().position(|o| o.color == c).unwrap();
    let ai = idx_col(c2);

    // FACULTY 1 — AGENCY binds the controllable object (colour/position never read). The probe moves the TRUE
    // avatar and observes each candidate's Δ by re-segmenting: the avatar responds, the distractor stays static.
    let ag_avatar = agency(&observe(&grid, start, start, c2, distractor, c1), 0x51);
    let ag_distr = agency(&observe(&grid, distractor, start, c2, distractor, c1), 0x51);
    let bind_agency = if ag_avatar >= ag_distr {
        ai
    } else {
        idx_col(c1)
    }; // the object agency selects
    let bind_colour = idx_col(c1); // the object a colour-slot key (banked colour c1) selects — the distractor
                                   // NON-TAUTOLOGICAL: the two keys bind DIFFERENT objects. Agency picks the RESPONDER (measured), which wears the
                                   // NEW colour c2; the colour slot picks the object still wearing the OLD colour c1.
    assert!(
        bind_agency != bind_colour,
        "agency and the colour slot must bind DIFFERENT objects"
    );
    assert!(
        objs[bind_agency].color == c2,
        "agency binds the recoloured object (c2), ignoring the changed colour"
    );
    assert!(
        objs[bind_colour].color == c1,
        "the colour slot binds the old-colour distractor (c1)"
    );
    let bind_agency_is_avatar = bind_agency == ai;
    let bind_colour_is_avatar = bind_colour == ai; // false

    // FACULTY 2 — the grown obstacle rule predicts the UNSEEN wall's connectivity; relax_values routes around it.
    // (The rule is colour-only by construction, so it is position-independent — its transfer to the new column is
    // structural, not an overfit-avoiding feat; the real content is that it LEARNED the wall colour from bumping.)
    let obs_adj = obstacle_adjacency(&world, &obstacle);
    let mut obs_field = Field::new(W * H);
    obs_field.prefer(cell_of(goal), 0.0);
    obs_field.relax_values(&obs_adj, 1.0, 0);
    let mut naive_field = Field::new(W * H);
    naive_field.prefer(cell_of(goal), 0.0);
    naive_field.relax_values(&naive_adjacency(), 1.0, 0);

    // ── COMPOSED vs the two ablations. The BINDINGS are threaded from the keys above (not hardcoded): the composed
    //    and naive arms use the AGENCY binding; the colour arm uses the COLOUR-slot binding. ──
    let (composed, steps, comp_stuck) = reach(
        &world,
        &obs_field,
        &skill,
        start,
        goal,
        bind_agency_is_avatar,
        distractor,
    );
    let (colour_ok, _, colour_stuck) = reach(
        &world,
        &obs_field,
        &skill,
        start,
        goal,
        bind_colour_is_avatar,
        distractor,
    );
    let (naive_ok, _, naive_stuck) = reach(
        &world,
        &naive_field,
        &skill,
        start,
        goal,
        bind_agency_is_avatar,
        distractor,
    );
    let manhattan = ((start.0 - goal.0).abs() + (start.1 - goal.1).abs()) as usize;

    eprintln!("[COMPOSE] game 2 (avatar recoloured c2 at new pos; distractor wears old c1; wall UNSEEN col 7):");
    eprintln!("[COMPOSE]   agency binds obj#{bind_agency} (c2, responder) vs colour binds obj#{bind_colour} (c1); ag {ag_avatar:.2} vs {ag_distr:.2}");
    eprintln!("[COMPOSE]   COMPOSED reached={composed} in {steps} steps (Manhattan {manhattan}, so a detour); colour stuck={colour_stuck:?}; naive stuck={naive_stuck:?}");

    assert!(
        composed && comp_stuck.is_none(),
        "the COMPOSED agent (agency + grown obstacle rule) must reach"
    );
    assert!(
        steps > manhattan,
        "the composed path must DETOUR around the wall (steps {steps} > Manhattan {manhattan})"
    );
    // Each ablation must GENUINELY live-lock at an off-goal fixed point (not merely exhaust the budget):
    assert!(!colour_ok && matches!(colour_stuck, Some(c) if c != goal), "COLOUR ablation must live-lock off-goal (static self-model → constant action): {colour_stuck:?}");
    assert!(
        !naive_ok && matches!(naive_stuck, Some(c) if c != goal),
        "NAIVE-obstacle ablation must live-lock at the unseen wall: {naive_stuck:?}"
    );

    // FROZEN guard: the banked skill's four action-landings are UNCHANGED after the whole run (reused, never re-grown).
    let post_landings: Vec<(i32, i32)> = (0..4).map(|a| apply_skill(&skill, (5, 4), a)).collect();
    assert_eq!(
        post_landings, base_landings,
        "the banked skill must be UNCHANGED after the run (frozen, reused not relearned)"
    );

    eprintln!("[COMPOSE] VERDICT: the two grown-invariant perception faculties COMPOSE and are BOTH load-bearing. In a transfer game where the avatar is recoloured+repositioned, a distractor wears the old colour, AND the wall is at a layout never experienced, the composed agent binds the controllable object by AGENCY (colour/position-blind: {ag_avatar:.2} vs static {ag_static:.2}) and routes around the wall by the GROWN obstacle rule (WALL learned to block, generalized to the unseen column), reaching the goal. Remove the invariant object (colour key → the static old-colour distractor → a constant action) and it fails; remove the grown obstacle rule (naive open-world → into the unseen wall → live-lock) and it fails. So neither invariance is decorative — the two grown perception multipliers MorphLab named are jointly necessary and jointly sufficient here. SUBSTRATE = grow + encapsulate + relax_values + segment_objects (zero new core); PROTOCOL = the world + agency probe + rule-predicted adjacency + greedy descent. HONEST LIMIT: synthetic scenario (hand-placed walls/colours); the genuinely-unauthored bar is these faculties over GROWN percepts on real ls20 frames.");
}
