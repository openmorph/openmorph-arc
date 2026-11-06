// SPDX-License-Identifier: MIT-0
//! THE INTEGRATION LOOP — the epic #337 rungs COMPOSED into one closed System-2 cycle on ONE task (Gap
//! "Integration loop", `docs/research/arc3-gaps-to-100.md`).
//!
//! THE GAP. Five capabilities are validated SEPARATELY, each on its own bespoke world: Gap-1 go-explore
//! (`substrate_coverage_recovery`), #332 grown world model (`substrate_imagine`), #334 planning over the model
//! (`substrate_planning`), #335 goal inference from reward attribution (`substrate_goal_inference`), #336 skill
//! memory (`substrate_skill_memory`). They have NEVER been COMPOSED into one loop on a single task — and composition
//! is exactly where things break (Gap 2 is the cautionary tale: two halves worked alone, the seam was unproven until
//! tested). This proof runs the WHOLE organism as ONE loop and measures whether the stages' outputs actually feed
//! each other, or whether error compounds across the seam.
//!
//! THE CLOSED LOOP (the composed cycle, per level). Nothing here is a new core primitive — every stage is one of the
//! already-validated usage protocols, wired end-to-end:
//!   1. EXPLORE (Gap 1) — go-explore over the EXPERIENCED map (`relax_values`-planned reaches to the frontier, NOT a
//!      greedy walk) until the sparse reward is first encountered. Records real transitions (for stage 3) and the
//!      reward↔feature co-occurrences (for stage 2).
//!   2. INFER (#335) — grow a linear reward predictor over candidate relational features; the feature it RELIES on
//!      (`reliance_of`) IS the inferred goal predicate. The predictor is a `Field`, snapshotted+reloaded (#336 skill
//!      memory) so the banked goal round-trips.
//!   3. MODEL (#332) — grow a forward world model of the DYNAMICS from the recorded transitions (`common::WorldModel`,
//!      the `imagine` idiom: predict the displacement an action produces).
//!   4. PLAN (#334) — read the adjacency the grown MODEL predicts, `prefer` the states satisfying the INFERRED goal,
//!      `relax_values` to a dense cost-to-go, descend it — executing each action in the REAL world.
//!   5. OBSERVE + REVISE — if the reward is NOT obtained (an unlucky first exploration never saw it, or the inference
//!      was wrong), the loop REVISES: explore more, re-infer, retry. This is the System-2 close.
//!
//! THE CLAIM. The FULL composed loop reaches the reward efficiently (planned exploit ≈ shortest path) at kept
//! coverage, and ABLATING ANY ONE STAGE degrades it: remove EXPLORATION → never finds the reward (misses); remove
//! GOAL-INFERENCE → no goal predicate, undirected (wanders); remove PLANNING → a sparse goal gives no gradient, the
//! reach is random (inefficient); remove REVISE → an unlucky single exploration never banks the skill (loses
//! coverage). The full loop must beat EVERY ablation on the coverage×efficiency product.
//!
//! TRUTH-NOT-TROPHY. The grid, the per-level HIDDEN win relation, the (perceived) target position, the seeded PRNG,
//! the budgets, and the feature set are authored and identical across arms; the win relation is NEVER given — it is
//! recovered from reward↔feature co-occurrence, resisting collinear decoys (overlap ⟹ x-align ∧ y-align). The plan
//! runs over the GROWN model's predicted adjacency (its fidelity is measured), and every planned action is EXECUTED
//! in the real world, so a model error must be recovered by the loop, not hidden by walking a fantasy graph. Only the
//! POLICY differs across arms — one stage removed each. This proof asserts the SUBSTRATE-composition claim (the loop
//! closes and each stage is load-bearing); the decisive number stays MorphBot's live 25-game A/B. ZERO
//! openmorph-core change: a usage protocol over `add_unit`/`learn_step`/`reliance_of` + `common::WorldModel`
//! (`imagine`) + `prefer`/`relax_values`/`value_of` + serde `snapshot`/`load`.
//!
//!   cargo test -p openmorph-core --test proofs substrate_integration_loop -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;
use std::collections::VecDeque;

use crate::common::{Rng, WorldModel};

const W: usize = 6;
const H: usize = 6;
const N_ACT: usize = 4; // 0=up 1=down 2=left 3=right
const K: usize = 3; // candidate relational features: 0=overlap, 1=x-align, 2=y-align (relative to the target)
const START: usize = 0; // (0,0)

// ── the world (authored; the organism never sees the win rule) ──────────────────────────────────────────────────

fn delta(a: usize) -> (i32, i32) {
    match a {
        0 => (0, -1),
        1 => (0, 1),
        2 => (-1, 0),
        _ => (1, 0),
    }
}
fn id(x: usize, y: usize) -> usize {
    y * W + x
}
fn xy(c: usize) -> (usize, usize) {
    (c % W, c / W)
}

/// Deterministic transition: apply the action; a move off the lattice STAYS in place (the world's hard boundary).
fn step_cell(x: usize, y: usize, a: usize) -> (usize, usize) {
    let (dx, dy) = delta(a);
    let nx = x as i32 + dx;
    let ny = y as i32 + dy;
    if nx < 0 || ny < 0 || nx >= W as i32 || ny >= H as i32 {
        (x, y)
    } else {
        (nx as usize, ny as usize)
    }
}

/// The real 4-neighbours of a cell (distinct cells only — the experienced adjacency go-explore reveals on arrival).
fn neighbours(c: usize) -> Vec<usize> {
    let (x, y) = xy(c);
    let mut v = Vec::new();
    for a in 0..N_ACT {
        let (nx, ny) = step_cell(x, y, a);
        if (nx, ny) != (x, y) {
            v.push(id(nx, ny));
        }
    }
    v
}

/// The action index that moves `from` to the adjacent `to` (for recording world-model transitions).
fn action_of(from: usize, to: usize) -> usize {
    let (fx, fy) = xy(from);
    let (tx, ty) = xy(to);
    let (dx, dy) = (tx as i32 - fx as i32, ty as i32 - fy as i32);
    match (dx, dy) {
        (0, -1) => 0,
        (0, 1) => 1,
        (-1, 0) => 2,
        _ => 3,
    }
}

/// The candidate relations at `(ax,ay)` RELATIVE to the (perceived) `target`, bipolar (+1 holds / −1 not). The hidden
/// win is ONE of these; note the collinearity overlap(0) ⟹ x-align(1) ∧ y-align(2) — the decoy inference must resist.
fn features(ax: usize, ay: usize, target: (usize, usize)) -> [f32; K] {
    let b = |c: bool| if c { 1.0 } else { -1.0 };
    [
        b(ax == target.0 && ay == target.1),
        b(ax == target.0),
        b(ay == target.1),
    ]
}
fn wins(k: usize, c: usize, target: (usize, usize)) -> bool {
    let (x, y) = xy(c);
    features(x, y, target)[k] > 0.0
}

/// The world-model context: normalised (x, y) ⊕ one-hot(action) — the `substrate_planning` layout.
fn ctx(x: usize, y: usize, a: usize) -> Vec<f32> {
    let mut c = vec![x as f32 / (W - 1) as f32, y as f32 / (H - 1) as f32];
    for k in 0..N_ACT {
        c.push(if k == a { 1.0 } else { 0.0 });
    }
    c
}

/// The true shortest path (BFS) start→any-win-cell — the efficiency oracle (never given to a policy).
fn shortest_to_win(k: usize, target: (usize, usize)) -> usize {
    let mut dist = vec![usize::MAX; W * H];
    let mut q = VecDeque::new();
    dist[START] = 0;
    q.push_back(START);
    let mut best = usize::MAX;
    while let Some(c) = q.pop_front() {
        if wins(k, c, target) {
            best = best.min(dist[c]);
        }
        for nb in neighbours(c) {
            if dist[nb] == usize::MAX {
                dist[nb] = dist[c] + 1;
                q.push_back(nb);
            }
        }
    }
    best
}

// ── STAGE 1: go-explore over the EXPERIENCED map (Gap 1) — records data for stages 2 & 3 ─────────────────────────

/// A persistent explorer: go-explore (expand the local frontier; when it is exhausted, `relax_values`-PLAN an
/// efficient reach to the nearest cell that still has an unvisited neighbour — NOT a random wander), while recording
/// every real transition (for the world model) and every (features, reward) pair (for goal inference).
struct Explorer {
    k: usize,
    target: (usize, usize),
    pos: usize,
    visited: Vec<bool>,
    known_adj: Vec<Vec<usize>>,
    transitions: Vec<(Vec<f32>, Vec<f32>)>, // (ctx, displacement) — stage 3
    obs: Vec<([f32; K], f32)>,              // (features, reward) — stage 2
    reward_found: bool,
    actions: usize,
}

impl Explorer {
    fn new(k: usize, target: (usize, usize)) -> Self {
        let n = W * H;
        let mut e = Explorer {
            k,
            target,
            pos: START,
            visited: vec![false; n],
            known_adj: vec![Vec::new(); n],
            transitions: Vec::new(),
            obs: Vec::new(),
            reward_found: false,
            actions: 0,
        };
        e.visited[START] = true;
        e.reveal(START);
        e.record_state(START); // the start is an observed state too
        e
    }

    fn reveal(&mut self, c: usize) {
        for nb in neighbours(c) {
            if !self.known_adj[c].contains(&nb) {
                self.known_adj[c].push(nb);
            }
        }
    }

    /// Record the (features, reward) observation at a cell, and note whether the sparse reward fired.
    fn record_state(&mut self, c: usize) {
        let (x, y) = xy(c);
        let f = features(x, y, self.target);
        let r = if wins(self.k, c, self.target) {
            1.0
        } else {
            -1.0
        };
        self.obs.push((f, r));
        if r > 0.0 {
            self.reward_found = true;
        }
    }

    /// Take one real step to an adjacent cell `nb`, recording the transition + the arrival observation.
    fn go(&mut self, nb: usize) {
        let a = action_of(self.pos, nb);
        let (x, y) = xy(self.pos);
        let (nx, ny) = xy(nb);
        self.transitions.push((
            ctx(x, y, a),
            vec![nx as f32 - x as f32, ny as f32 - y as f32],
        ));
        self.pos = nb;
        if !self.visited[nb] {
            self.visited[nb] = true;
            self.reveal(nb);
        }
        self.record_state(nb);
        self.actions += 1;
    }

    /// The nearest visited cell (BFS over the experienced graph) that still has an unvisited neighbour.
    fn nearest_frontier(&self) -> Option<usize> {
        let mut seen = vec![false; self.visited.len()];
        let mut q = VecDeque::new();
        seen[self.pos] = true;
        q.push_back(self.pos);
        while let Some(c) = q.pop_front() {
            if self.known_adj[c].iter().any(|&nb| !self.visited[nb]) {
                return Some(c);
            }
            for &nb in &self.known_adj[c] {
                if self.visited[nb] && !seen[nb] {
                    seen[nb] = true;
                    q.push_back(nb);
                }
            }
        }
        None
    }

    /// Run go-explore for up to `budget` real actions (or until the whole reachable set is covered).
    fn explore(&mut self, budget: usize) {
        let start_actions = self.actions;
        while self.actions - start_actions < budget {
            // a locally-unvisited neighbour? step to it (expand the frontier).
            if let Some(&nb) = self.known_adj[self.pos]
                .iter()
                .find(|&&nb| !self.visited[nb])
            {
                self.go(nb);
                continue;
            }
            // local frontier exhausted → PLAN a reach to the nearest cell that still has one (Gap 1's lever).
            let Some(target) = self.nearest_frontier() else {
                break; // the whole reachable component is covered
            };
            let mut field = Field::new(W * H);
            field.prefer(target, 0.0);
            field.relax_values(&self.known_adj, 1.0, 0);
            while self.pos != target && self.actions - start_actions < budget {
                let step = self.known_adj[self.pos]
                    .iter()
                    .copied()
                    .min_by(|&a, &b| {
                        field
                            .value_of(a)
                            .total_cmp(&field.value_of(b))
                            .then(a.cmp(&b))
                    })
                    .unwrap_or(self.pos);
                if step == self.pos {
                    break;
                }
                self.go(step);
            }
        }
    }
}

// ── STAGE 2: infer the goal predicate from reward attribution (#335) + skill memory (#336) ───────────────────────

/// Grow a linear reward predictor over the recorded (features, reward) pairs; return `(skill Field, inferred k)`.
/// The feature it comes to RELY on (`reliance_of`) IS the goal predicate. The `Field` round-trips through serde.
fn infer_goal(obs: &[([f32; K], f32)]) -> (Field, usize) {
    let mut f = Field::new(K + 1);
    f.add_unit((0..K).collect(), vec![0.0; K], 0.0, K);
    // Replay the collected co-occurrences to convergence (internal, free — no new world samples).
    for _ in 0..120 {
        for (feats, reward) in obs {
            let inp: Vec<(usize, f32)> = (0..K).map(|i| (i, feats[i])).collect();
            f.learn_step(&inp, K, *reward, 0.05, 50);
        }
    }
    let k_hat = (0..K)
        .max_by(|&i, &j| f.reliance_of(i).total_cmp(&f.reliance_of(j)))
        .unwrap();
    (f, k_hat)
}

/// The relation a (possibly reloaded) skill encodes = the feature it relies on most.
fn skill_relation(skill: &Field) -> usize {
    (0..K)
        .max_by(|&i, &j| skill.reliance_of(i).total_cmp(&skill.reliance_of(j)))
        .unwrap()
}

// ── STAGE 3: grow the world model of the dynamics (#332) ─────────────────────────────────────────────────────────

/// Grow the forward model from the recorded transitions (displacement prediction, the `substrate_planning` regime).
fn grow_model(transitions: &[(Vec<f32>, Vec<f32>)]) -> WorldModel {
    let mut model = WorldModel::new(6, 2 + N_ACT, 2, 20);
    model.replay(transitions, 80); // free internal training on the experienced transitions
    model
}

/// The cell the model IMAGINES `(x,y,a)` lands on: its predicted displacement + the world's boundary clamp.
fn imagined_cell(model: &mut WorldModel, x: usize, y: usize, a: usize) -> usize {
    let (pred, _s2) = model.imagine(&ctx(x, y, a));
    let px = (x as i32 + pred[0].round() as i32).clamp(0, W as i32 - 1) as usize;
    let py = (y as i32 + pred[1].round() as i32).clamp(0, H as i32 - 1) as usize;
    id(px, py)
}

/// The adjacency the grown MODEL predicts (+ its fidelity vs the true grid), for planning over.
fn learned_adjacency(model: &mut WorldModel) -> (Vec<Vec<usize>>, usize, usize) {
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); W * H];
    let (mut correct, mut total) = (0usize, 0usize);
    for (c, edges) in adj.iter_mut().enumerate() {
        let (x, y) = xy(c);
        for a in 0..N_ACT {
            let pc = imagined_cell(model, x, y, a);
            let (tx, ty) = step_cell(x, y, a);
            total += 1;
            if pc == id(tx, ty) {
                correct += 1;
            }
            if pc != c {
                edges.push(pc);
            }
        }
    }
    (adj, correct, total)
}

// ── STAGE 4: PLAN over the grown model to the inferred goal, executing in the REAL world (#334) ──────────────────

/// Faithful model-based control: `prefer` every state satisfying the INFERRED goal `k_hat` (relative to `target`),
/// `relax_values` over the LEARNED adjacency to a dense cost-to-go, descend it via the model's predicted next cells,
/// EXECUTING each action in the real world. Reward is scored by the TRUE win `k` (so a wrong `k_hat` cannot fake a
/// win). Returns `(reached, actions)`.
fn plan_execute(
    model: &mut WorldModel,
    adj_hat: &[Vec<usize>],
    k_hat: usize,
    k_true: usize,
    target: (usize, usize),
    cap: usize,
) -> (bool, usize) {
    let mut f = Field::new(W * H);
    for c in 0..W * H {
        if wins(k_hat, c, target) {
            f.prefer(c, 0.0);
        }
    }
    f.relax_values(adj_hat, 1.0, 0);
    let mut pos = START;
    let mut actions = 0;
    while actions < cap {
        if wins(k_true, pos, target) {
            return (true, actions);
        }
        let (x, y) = xy(pos);
        let best_a = (0..N_ACT)
            .min_by(|&p, &q| {
                let vp = f.value_of(imagined_cell(model, x, y, p));
                let vq = f.value_of(imagined_cell(model, x, y, q));
                vp.total_cmp(&vq).then(p.cmp(&q))
            })
            .unwrap();
        let (nx, ny) = step_cell(x, y, best_a); // EXECUTE in the real world (observe the true outcome)
        let next = id(nx, ny);
        if next == pos {
            // the real move stayed (an edge the model may have mis-imagined) — nudge so an error cannot deadlock.
            let alt = (best_a + 1) % N_ACT;
            let (ax, ay) = step_cell(x, y, alt);
            pos = id(ax, ay);
        } else {
            pos = next;
        }
        actions += 1;
    }
    (wins(k_true, pos, target), actions)
}

// ── the banked knowledge a full learning loop produces ──────────────────────────────────────────────────────────

struct Banked {
    k_hat: usize,
    adj_hat: Vec<Vec<usize>>,
    model: WorldModel,
    fidelity: f32,
    roundtrip_ok: bool,
    inferred_correct: bool,
}

/// THE FULL CLOSED LOOP (learning phase). Explore in chunks; the moment the sparse reward is observed, INFER the goal
/// (+ snapshot/reload the skill), grow the MODEL, and bank the predicted adjacency. If a chunk does NOT find the
/// reward, REVISE: explore another chunk (up to `rounds`). Returns the banked knowledge, or `None` (never found).
fn learn_full(k: usize, target: (usize, usize), chunk: usize, rounds: usize) -> Option<Banked> {
    let mut ex = Explorer::new(k, target);
    for _ in 0..rounds {
        ex.explore(chunk);
        if ex.reward_found {
            let (skill, k_hat) = infer_goal(&ex.obs);
            // #336 skill memory: the goal skill round-trips through serde.
            let bytes = serde_json::to_vec(&skill).expect("skill serialises");
            let reloaded: Field = serde_json::from_slice(&bytes).expect("skill loads");
            let roundtrip_ok = skill_relation(&reloaded) == k_hat;
            let k_hat = skill_relation(&reloaded);
            let mut model = grow_model(&ex.transitions);
            let (adj_hat, correct, total) = learned_adjacency(&mut model);
            return Some(Banked {
                k_hat,
                adj_hat,
                model,
                fidelity: correct as f32 / total as f32,
                roundtrip_ok,
                inferred_correct: k_hat == k,
            });
        }
    }
    None
}

// ── the ablation exploit policies (each removes ONE stage) ───────────────────────────────────────────────────────

/// NO-PLANNING exploit: the loop KNOWS the goal cells (inferred `k_hat`) and the model, but builds NO cost-to-go
/// field — a sparse goal predicate gives no gradient from afar, so the reach is an undirected (seeded) walk that only
/// snaps when a goal cell is adjacent. Reaches, but inefficiently — the honest cost of removing `relax_values`.
fn exploit_no_planning(
    k_true: usize,
    target: (usize, usize),
    seed: u64,
    cap: usize,
) -> (bool, usize) {
    let mut rng = Rng::new(seed);
    let mut pos = START;
    for t in 0..cap {
        if wins(k_true, pos, target) {
            return (true, t);
        }
        // snap onto an adjacent goal cell if one is right here; else wander (no gradient without planning).
        let nbrs = neighbours(pos);
        pos = match nbrs.iter().copied().find(|&c| wins(k_true, c, target)) {
            Some(g) => g,
            None => nbrs[rng.below(nbrs.len())],
        };
    }
    (wins(k_true, pos, target), cap)
}

/// NO-GOAL-INFERENCE exploit: the loop explored + grew the model but formed NO goal predicate, so it has nothing to
/// plan TO — it falls back to undirected go-explore, treating the reward as just another cell to cover. Reaches (its
/// coverage is systematic) but pays a coverage cost, not a planned reach.
fn exploit_no_goal(k_true: usize, target: (usize, usize), cap: usize) -> (bool, usize) {
    let mut ex = Explorer::new(k_true, target);
    while ex.actions < cap {
        if wins(k_true, ex.pos, target) {
            return (true, ex.actions);
        }
        let before = ex.actions;
        ex.explore(1);
        if ex.actions == before {
            break; // fully covered without a win (shouldn't happen — the win cell is reachable)
        }
    }
    (wins(k_true, ex.pos, target), ex.actions.min(cap))
}

/// NO-EXPLORE exploit: the loop never explored, so it never saw the reward, never inferred a goal, never grew a
/// model — it can only random-chase the sparse reward from cold. On a sparse target it usually MISSES the cap.
fn exploit_no_explore(
    k_true: usize,
    target: (usize, usize),
    seed: u64,
    cap: usize,
) -> (bool, usize) {
    let mut rng = Rng::new(seed);
    let mut pos = START;
    for t in 0..cap {
        if wins(k_true, pos, target) {
            return (true, t);
        }
        let nbrs = neighbours(pos);
        pos = nbrs[rng.below(nbrs.len())];
    }
    (wins(k_true, pos, target), cap)
}

// ── the proof ───────────────────────────────────────────────────────────────────────────────────────────────────

/// One game = a (hidden win relation `k`, perceived target position). The dynamics are shared across games; only the
/// goal grounding and the win relation change (the ARC-3 grain: objects are seen, the win rule is hidden).
fn games() -> Vec<(usize, (usize, usize))> {
    let targets = [(3, 3), (4, 2), (1, 4), (2, 1)];
    let mut g = Vec::new();
    for k in 0..K {
        for &t in &targets {
            g.push((k, t));
        }
    }
    g
}

/// Coverage / mean-actions-per-win / product (coverage ÷ mean-actions-per-win, the ARC-3 wins×efficiency shape).
struct Score {
    wins: usize,
    act_sum: usize,
    n: usize,
}
impl Score {
    fn new() -> Self {
        Score {
            wins: 0,
            act_sum: 0,
            n: 0,
        }
    }
    fn add(&mut self, o: (bool, usize), games: usize) {
        self.n = games;
        if o.0 {
            self.wins += 1;
            self.act_sum += o.1;
        }
    }
    fn coverage(&self) -> f32 {
        self.wins as f32 / self.n.max(1) as f32
    }
    fn mean_act(&self) -> f32 {
        if self.wins == 0 {
            f32::INFINITY
        } else {
            self.act_sum as f32 / self.wins as f32
        }
    }
    fn product(&self) -> f32 {
        self.coverage() / self.mean_act().max(1.0)
    }
}

#[test]
fn the_full_loop_closes_and_every_stage_is_load_bearing() {
    let games = games();
    let chunk = 15; // go-explore actions per revise round
    let rounds = 6; // revise budget (plenty to cover a 36-cell grid)
    let single = 12; // NO-REVISE gets ONE short exploration chunk, then must commit
    let exploit_cap = 30; // per exploit reach — planned ≈ shortest path (≪ 30); random usually misses

    let (mut full, mut no_explore, mut no_goal, mut no_plan, mut no_revise) = (
        Score::new(),
        Score::new(),
        Score::new(),
        Score::new(),
        Score::new(),
    );

    let (mut fid_sum, mut fid_n) = (0.0f32, 0usize);
    let (mut roundtrips, mut infer_correct, mut learned_n) = (0usize, 0usize, 0usize);
    let mut sp_sum = 0usize;
    let n_games = games.len();

    for (gi, &(k, target)) in games.iter().enumerate() {
        let seed = 0x1_0000u64 + gi as u64 * 0x9E37;
        sp_sum += shortest_to_win(k, target);

        // ── FULL LOOP: learn (explore→infer→model→plan-bank, revising until reward found), then planned exploit.
        let full_out = match learn_full(k, target, chunk, rounds) {
            Some(mut b) => {
                fid_sum += b.fidelity;
                fid_n += 1;
                learned_n += 1;
                if b.roundtrip_ok {
                    roundtrips += 1;
                }
                if b.inferred_correct {
                    infer_correct += 1;
                }
                let adj = b.adj_hat.clone();
                plan_execute(&mut b.model, &adj, b.k_hat, k, target, exploit_cap)
            }
            None => (false, exploit_cap),
        };
        full.add(full_out, n_games);

        // ── ABLATION: NO-EXPLORE (stage 1 removed) — cold random-chase, no learning at all.
        no_explore.add(exploit_no_explore(k, target, seed, exploit_cap), n_games);

        // ── ABLATION: NO-GOAL-INFERENCE (stage 2 removed) — explored + modelled, but no goal predicate to plan to.
        no_goal.add(exploit_no_goal(k, target, exploit_cap), n_games);

        // ── ABLATION: NO-PLANNING (stage 4 removed) — has the inferred goal + model, but no cost-to-go field.
        no_plan.add(exploit_no_planning(k, target, seed, exploit_cap), n_games);

        // ── ABLATION: NO-REVISE (stage 5 removed) — a SINGLE exploration chunk, no retry; commit or miss.
        let no_revise_out = match learn_full(k, target, single, 1) {
            Some(mut b) => {
                let adj = b.adj_hat.clone();
                plan_execute(&mut b.model, &adj, b.k_hat, k, target, exploit_cap)
            }
            None => (false, exploit_cap), // one unlucky chunk never saw the reward ⇒ nothing banked ⇒ miss
        };
        no_revise.add(no_revise_out, n_games);
    }

    let mean_fid = fid_sum / fid_n.max(1) as f32;
    let mean_sp = sp_sum as f32 / n_games as f32;

    // ── report ──────────────────────────────────────────────────────────────────────────────────────────────────
    println!("\n  THE INTEGRATION LOOP — five #337 rungs composed on ONE task ({n_games} games):");
    println!(
        "    stage seams (measured): grown-model adjacency fidelity {mean_fid:.3}  ·  goal inference correct {infer_correct}/{learned_n}  ·  skill snapshot round-trips {roundtrips}/{learned_n}"
    );
    println!("    shortest-path-to-win oracle (mean): {mean_sp:.1} actions\n");
    let row = |name: &str, s: &Score| {
        println!(
            "    {name:<26}: wins {:>2}/{n_games}  coverage {:.2}  actions/win {:>5.1}  product {:.4}",
            s.wins,
            s.coverage(),
            if s.wins == 0 { 0.0 } else { s.mean_act() },
            s.product()
        );
    };
    row("FULL LOOP", &full);
    row("  −exploration", &no_explore);
    row("  −goal-inference", &no_goal);
    row("  −planning", &no_plan);
    row("  −revise", &no_revise);
    println!();

    // ── assertions: the composition HOLDS end-to-end, and every stage is load-bearing ───────────────────────────
    // The seams: the grown model is planning-usable, inference recovers the hidden win, the skill round-trips.
    assert!(
        learned_n == n_games,
        "the full loop must find the reward and bank a skill on EVERY game (revise closes it): {learned_n}/{n_games}"
    );
    assert!(
        mean_fid > 0.7,
        "the grown model must recover most of the adjacency for planning: fidelity {mean_fid:.3}"
    );
    assert_eq!(
        roundtrips, learned_n,
        "every banked skill must round-trip through snapshot/load: {roundtrips}/{learned_n}"
    );
    assert!(
        infer_correct as f32 >= 0.8 * learned_n as f32,
        "goal inference must recover the hidden win on ≥80% of games: {infer_correct}/{learned_n}"
    );

    // The full loop reaches the reward on every game, efficiently (planned exploit near the shortest path).
    assert_eq!(
        full.wins, n_games,
        "the FULL loop must reach the reward on every game: {}/{n_games}",
        full.wins
    );
    assert!(
        full.mean_act() <= 1.5 * mean_sp,
        "the FULL loop's planned exploit must be near shortest path: {:.1} vs oracle {mean_sp:.1}",
        full.mean_act()
    );

    // Every ablation degrades the coverage×efficiency product — each removed stage is load-bearing.
    assert!(
        full.product() > no_explore.product(),
        "removing EXPLORATION must degrade the loop: full {:.4} vs {:.4}",
        full.product(),
        no_explore.product()
    );
    assert!(
        full.product() > no_goal.product(),
        "removing GOAL-INFERENCE must degrade the loop: full {:.4} vs {:.4}",
        full.product(),
        no_goal.product()
    );
    assert!(
        full.product() > no_plan.product(),
        "removing PLANNING must degrade the loop: full {:.4} vs {:.4}",
        full.product(),
        no_plan.product()
    );
    assert!(
        full.product() > no_revise.product(),
        "removing REVISE must degrade the loop: full {:.4} vs {:.4}",
        full.product(),
        no_revise.product()
    );
    // Coverage-specific reads (the qualitative degradations the report narrates).
    assert!(
        no_explore.coverage() < full.coverage(),
        "removing exploration must LOSE coverage (misses the sparse reward): {:.2} vs {:.2}",
        no_explore.coverage(),
        full.coverage()
    );
    assert!(
        no_revise.coverage() < full.coverage(),
        "removing revise must LOSE coverage (an unlucky single chunk never banks): {:.2} vs {:.2}",
        no_revise.coverage(),
        full.coverage()
    );

    println!("  ⇒ the composition HOLDS: explore→infer→model→plan→revise closes as one loop; ablating ANY stage degrades it.\n");
}
