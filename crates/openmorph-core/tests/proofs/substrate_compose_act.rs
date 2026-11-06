// SPDX-License-Identifier: MIT-0
//! #150 faculty 6 — the LIBRARY COMPOSE LOOP, the ACTING-side twin of `substrate_recurse`/`substrate_curriculum`.
//!
//! Reasoning grows its library through VERIFICATION (`recurse`/`curriculum` bank a predicate composition that
//! `imagine` READS to reach a deeper RULE). Acting grows its library through REACHING GOALS: a goal too deep for a
//! BOUNDED planner is reached by BANKING a macro-action — a genuinely `encapsulate`+`nest`ed sub-morph that becomes
//! a planner EDGE and an `alive(Event::Act)` OPTION — and the library GROWS because the banked skill is itself
//! `nest`ed into a DEEPER skill. Reuse IS nesting; the acting twin of reason→bank→reason-deeper.
//!
//! GENUINE NESTING (a real nest-of-nest, verified — an earlier draft was caught overclaiming a cosmetic one).
//! `advance` is a GROWN 1-step transition model (x → x+1, `Identity` genome so an absolute position threads a
//! chain; trained on the VALID transitions only, so it is exact). M1 is a Field that NESTS `advance` (cell i+1 =
//! advance(cell i)); M2 is a Field that NESTS the banked M1. So M2 CONTAINS M1 CONTAINS advance — reading M2
//! settles that whole tree in ONE cascade (`nested_settles` rises), and M2's landing EQUALS M1∘M1 (asserted). The
//! displacement accumulates INSIDE the nested chain (each cell reads the previous), not in a harness loop.
//!
//! THE WALL (dual, so flat-fails rests on no single contrivance):
//!   (i) BUDGET (primary, orientation/tie-break independent): a K-capped episode whose every primitive decision
//!       moves ≤1 cell CANNOT cross distance D>K — the textbook temporal-abstraction argument (a macro crosses m
//!       cells per decision, so ⌈D/m⌉ macro-decisions beat D primitive ones).
//!   (ii) PLANNER-BLIND (secondary): after `relax_values(prim_adj, 1.0, H)` the cost-to-go has propagated only H
//!       hops back from the goal, so `value_of(start) == f32::MAX` when the goal is >H hops away. (Load-bearing:
//!       the goal sits ABOVE the start and `relax_values` is Gauss-Seidel low→high, ~1 hop/sweep — a clean integer
//!       wall. We also assert the world IS solvable at unbounded sweeps, so the HORIZON blocks it, not geometry.)
//!
//! SUBSTRATE vs PROTOCOL (truth-not-trophy, ZERO new core). SUBSTRATE (all shipped): `grow` (the transition
//! model), `encapsulate`+`nest` (bank a sub-morph as a fractal cell fired in the cascade — reuse IS nesting),
//! `prefer`/`relax_values(max_sweeps=H)`/`value_of` (the bounded planner), `alive(Event::Act)` + the default
//! valence rule (selection). PROTOCOL (harness, exactly like deliberate/curriculum): the compose loop, the
//! reachability search that DISCOVERS the smallest-displacement macro that unlocks the goal (parsimony — a
//! deterministic objective, not iteration order), candidate GENERATION (nest each library skill by a factor set),
//! building adjacency, and reading a macro's landing (imagination — what `substrate_planner` does).
//!
//! HONEST scope: the transition model is +1-DIRECTIONAL (a state-DEPENDENT maneuver is future work, still
//! zero-core). The win is that the deeper goal is reachable ONLY after a shallower skill is banked and re-nested.
use openmorph_core::{Activation, Event, Field, Outcome};

const N: usize = 25; // corridor cells 0..=24 — these ARE the value/option/adjacency indices
const H: usize = 3; // the bounded planning horizon (relax_values max_sweeps)
const BUD: usize = 100; // settle budget for the nested towers
const FACTORS: [usize; 3] = [1, 2, 3]; // "nest a banked skill f times" — the candidate generator

fn clampi(x: i32) -> usize {
    x.clamp(0, N as i32 - 1) as usize
}
// Position ↔ model value: normalized and shifted so position 0 is NON-zero (else stimulating cell 0 with 0.0 is a
// no-change under the async fire-on-change rule and the nested model never fires).
fn enc(pos: usize) -> f32 {
    (pos + 1) as f32 / (N + 1) as f32
}
fn dec(v: f32) -> usize {
    ((v * (N + 1) as f32).round() as i32 - 1).clamp(0, N as i32 - 1) as usize
}

/// A banked skill: a Field whose cell 0 (source) is a position-in and whose `sink` cell settles to the position
/// AFTER the maneuver. `disp` = the displacement it applies.
struct Skill {
    field: Field,
    sink: usize,
    disp: i32,
}
impl Skill {
    /// Settle the banked (possibly deeply nested) skill on position `s` and read where it LANDS. Settling fires
    /// the whole nested tree in the active cascade — `nest` is load-bearing (remove it and nothing writes `sink`).
    fn landing(&mut self, s: usize) -> usize {
        self.field.stimulate(&[(0, enc(s))], BUD);
        dec(self.field.cells()[self.sink])
    }
}

/// GROW a 1-step transition model x → x+`dir` (Identity genome → an absolute position, learned not authored).
/// Trained on the VALID transitions only (no clamped self-loop at the boundary), so the linear fit is exact.
fn grow_step(dir: i32) -> Field {
    let mut step = Field::new(2);
    step.config_mut().activation = Activation::Identity;
    step.config_mut().max_hidden = 0;
    let domain: Vec<usize> = if dir > 0 {
        (0..N - 1).collect()
    } else {
        (1..N).collect()
    };
    for _ in 0..4000 {
        for &p in &domain {
            let q = clampi(p as i32 + dir);
            step.grow(&[(0, enc(p))], 1, enc(q), 0.05, 20);
        }
    }
    step.encapsulate(&[1])
}

/// NEST `base` `n` times into a NEW banked skill (cell i+1 = base(cell i)). The genuine fractal composition:
/// `encapsulate` carries nested cells verbatim, so a chain of M1 truly CONTAINS M1 which CONTAINS advance.
fn nest_chain(base: &Field, n: usize, base_disp: i32) -> Skill {
    let mut host = Field::new(n + 1);
    for i in 0..n {
        host.nest(base.clone(), &[i], i + 1, BUD);
    }
    let cells: Vec<usize> = (1..=n).collect();
    Skill {
        field: host.encapsulate(&cells),
        sink: n,
        disp: base_disp * n as i32,
    }
}

fn prim_adj() -> Vec<Vec<usize>> {
    (0..N)
        .map(|s| vec![clampi(s as i32 - 1), clampi(s as i32 + 1)])
        .collect()
}

/// The AUGMENTED adjacency: primitives + one macro EDGE per library skill (s → its landing).
fn aug_adj(library: &mut [Skill]) -> Vec<Vec<usize>> {
    let mut adj = prim_adj();
    for (s, nbrs) in adj.iter_mut().enumerate() {
        for sk in library.iter_mut() {
            nbrs.push(sk.landing(s));
        }
    }
    adj
}

fn options_at(library: &mut [Skill], s: usize) -> Vec<usize> {
    let mut opts = vec![clampi(s as i32 - 1), clampi(s as i32 + 1)];
    for sk in library.iter_mut() {
        opts.push(sk.landing(s));
    }
    opts
}

/// Run one episode through `alive(Event::Act)` under a decision budget K. Returns (reached, decisions, used_macro).
fn run_episode(
    agent: &mut Field,
    library: &mut [Skill],
    start: usize,
    goal: usize,
    k_budget: usize,
) -> (bool, usize, bool) {
    let (mut s, mut decisions, mut used_macro) = (start, 0usize, false);
    while s != goal && decisions < k_budget {
        let opts = options_at(library, s);
        let chosen = match agent.alive(Event::Act {
            options: &opts,
            last: None,
        }) {
            Outcome::Acted { chosen } => chosen,
            _ => unreachable!(),
        };
        if chosen != clampi(s as i32 - 1) && chosen != clampi(s as i32 + 1) {
            used_macro = true;
        }
        s = chosen;
        decisions += 1;
    }
    (s == goal, decisions, used_macro)
}

/// Plan (bounded horizon H) then act (budget K). Returns (reached, decisions, used_macro, value_of(start)).
fn plan_and_act(
    library: &mut [Skill],
    start: usize,
    goal: usize,
    k_budget: usize,
) -> (bool, usize, bool, f32) {
    let mut agent = Field::new(N);
    agent.prefer(goal, 0.0);
    let adj = aug_adj(library);
    agent.relax_values(&adj, 1.0, H); // BOUNDED — only H hops of cost-to-go propagate
    let v_start = agent.value_of(start);
    let (reached, decisions, used_macro) = run_episode(&mut agent, library, start, goal, k_budget);
    (reached, decisions, used_macro, v_start)
}

/// Generate candidates for the current library: NEST each library skill f∈FACTORS times — "compose a banked skill
/// a few times". Every candidate is a real nested chain of a banked skill; no displacement is hand-written.
fn gen_candidates(library: &[Skill]) -> Vec<(Skill, String)> {
    let mut out = Vec::new();
    for (li, sk) in library.iter().enumerate() {
        for &f in &FACTORS {
            out.push((
                nest_chain(&sk.field, f, sk.disp),
                format!("skill{li}(disp{})×{f}", sk.disp),
            ));
        }
    }
    out
}

/// REACHABILITY discovery (the acting twin of `substrate_curriculum`'s subgoal search): among generated candidates,
/// keep the SMALLEST-displacement macro that makes `goal` reachable under the bounded horizon H AND within budget K
/// — parsimony, a deterministic objective (not iteration order). Returns the kept skill + label, or None.
fn discover(
    mut candidates: Vec<(Skill, String)>,
    start: usize,
    goal: usize,
    k_budget: usize,
) -> Option<(Skill, String)> {
    candidates.sort_by_key(|(sk, _)| sk.disp.abs());
    for (mut sk, label) in candidates {
        let (reached, _d, _m, v) =
            plan_and_act(std::slice::from_mut(&mut sk), start, goal, k_budget);
        if v != f32::MAX && reached {
            return Some((sk, label));
        }
    }
    None
}

#[test]
fn faculty6_library_compose_loop() {
    let advance = grow_step(1);
    let retreat = grow_step(-1);

    // SANITY — a GENUINELY nested chain lands correctly and settles the whole tree (nest is load-bearing), and a
    // nest-of-nest (M2 ⊃ M1 ⊃ advance) equals M1∘M1.
    let mut m1 = nest_chain(&advance, 3, 1); // M1 = nest(advance)×3  → +3
    let ns0 = m1.field.nested_settles();
    assert_eq!(m1.landing(0), 3, "M1 = nest(advance)×3 must land +3 from 0");
    assert_eq!(m1.landing(10), 13, "M1 must land +3 from 10");
    assert!(
        m1.field.nested_settles() > ns0,
        "reading M1 must SETTLE its nested advance cells (nest is load-bearing)"
    );
    let mut m2 = nest_chain(&m1.field, 2, 3); // M2 = nest(M1)×2  → +6  (a real nest-of-nest)
    let m1_once = m1.landing(4);
    let m1_twice = m1.landing(m1_once);
    assert_eq!(
        m2.landing(4),
        m1_twice,
        "M2 must equal M1∘M1 (it genuinely nests the banked M1 twice)"
    );
    let ns_m2 = m2.field.nested_settles();
    assert_eq!(m2.landing(0), 6, "M2 = nest(M1)×2 must land +6 from 0");
    // reading M2 must settle its nested M1 cells (which settle THEIR nested advance cells) — a 3-level cascade.
    assert!(
        m2.field.nested_settles() > ns_m2,
        "reading M2 must SETTLE its nested M1 cells (the nest-of-nest is load-bearing)"
    );
    eprintln!("[COMPOSE] genuine nesting: M1=nest(advance)×3 lands +3, M2=nest(M1)×2 lands +6 = M1∘M1 (nest-of-nest, the tree fires in one cascade)");

    // ── TASK A (goal 9, D=9): FLAT fails BOTH ways; DISCOVERY banks +3; COMPOSE reaches. ──
    let (start, goal_a, k_a) = (0usize, 9usize, 5usize); // K=5 < D=9

    let (flat_reached, _, _, flat_v) = plan_and_act(&mut [], start, goal_a, k_a);
    assert_eq!(
        flat_v,
        f32::MAX,
        "PLANNER-BLIND: at H={H} the far goal ({goal_a} hops) leaves value_of(start)=MAX"
    );
    assert!(
        !flat_reached,
        "BUDGET: {k_a} primitive decisions (≤1 cell each) cannot cross D={goal_a}"
    );
    let mut solvable = Field::new(N);
    solvable.prefer(goal_a, 0.0);
    solvable.relax_values(&prim_adj(), 1.0, 0);
    assert!(
        solvable.value_of(start).is_finite(),
        "the world must be solvable at unbounded sweeps (H blocks it, not geometry)"
    );
    eprintln!("[COMPOSE] TASK A flat: value_of(0)=MAX at H={H} AND {k_a} primitive decisions can't cross D={goal_a}; yet solvable unbounded (value_of(0)={:.0})", solvable.value_of(start));

    // DISCOVERY: generate candidates = nest each library skill {advance, retreat} by {1,2,3}; keep the smallest
    // that unlocks the goal. advance×{1,2}=+1/+2 undershoot (reach ≤6<9); retreat go the wrong way; only
    // advance×3=+3 (reach 9) is reachability-sufficient — kept by the mechanism (reachability+parsimony).
    let library0 = vec![
        Skill {
            field: advance.clone(),
            sink: 1,
            disp: 1,
        },
        Skill {
            field: retreat.clone(),
            sink: 1,
            disp: -1,
        },
    ];
    let (kept_a, label_a) = discover(gen_candidates(&library0), start, goal_a, k_a)
        .expect("task A must discover a macro");
    assert_eq!(
        kept_a.disp, 3,
        "reachability must DISCOVER +3 (advance×3), the smallest macro that unlocks D={goal_a}"
    );
    eprintln!("[COMPOSE] TASK A discovery: kept {label_a} (disp {}) by REACHABILITY+parsimony (advance×1/2 undershoot, retreat wrong-way)", kept_a.disp);

    // COMPOSE (same bounded H): with +3 banked, the SAME planner reaches the goal via ⌈D/m⌉=3 macro decisions.
    let mut lib_a = vec![kept_a];
    let (c_reached, c_dec, c_macro, c_v) = plan_and_act(&mut lib_a, start, goal_a, k_a);
    assert!(
        c_v < f32::MAX,
        "COMPOSE: the banked macro makes value_of(start) finite at the SAME H={H}"
    );
    assert!(
        c_reached && c_macro,
        "COMPOSE: the agent must REACH the goal USING the macro"
    );
    assert_eq!(
        c_dec, 3,
        "COMPOSE: exactly ⌈D/m⌉ = 9/3 = 3 macro decisions (0→3→6→9)"
    );
    eprintln!("[COMPOSE] TASK A compose: value_of(0)={c_v:.0} finite, REACHED in {c_dec} macro decisions (0→3→6→9), used_macro={c_macro}");

    // ── LIBRARY GROWS THROUGH ACTING (goal 18, D=18, K=3): the deeper goal is reachable ONLY after M1 is BANKED
    //    and re-nested into a deeper macro (M2 = nest(M1)×2). ──
    let (goal_b, k_b) = (18usize, 3usize);

    // With only the PRIMITIVES, no generated candidate reaches D=18 (advance×3=+3 is the max, reach 9<18).
    let none_b = discover(gen_candidates(&library0), start, goal_b, k_b);
    assert!(
        none_b.is_none(),
        "with only primitives, NO candidate unlocks D={goal_b} (advance×3 reaches 9<18)"
    );

    // BANK M1 into the library, then generate candidates again — now NESTING the banked M1 reaches deeper.
    let m1_banked = nest_chain(&advance, 3, 1); // the skill discovered for task A, banked
    let library1 = vec![
        Skill {
            field: advance.clone(),
            sink: 1,
            disp: 1,
        },
        m1_banked,
    ];
    let (kept_b, label_b) = discover(gen_candidates(&library1), start, goal_b, k_b)
        .expect("after banking M1, task B must be discoverable");
    assert!(
        kept_b.disp >= 6 && kept_b.disp % 3 == 0,
        "task B's macro must be a NEST OF THE BANKED M1 (disp a multiple of 3, ≥6): got {}",
        kept_b.disp
    );
    let mut lib_b = vec![kept_b];
    let (b_reached, b_dec, b_macro, b_v) = plan_and_act(&mut lib_b, start, goal_b, k_b);
    assert!(
        b_v < f32::MAX && b_reached && b_macro,
        "the grown macro (nest of banked M1) must reach D={goal_b}"
    );
    assert!(
        b_dec <= k_b,
        "reached within the tight budget K={k_b}: {b_dec} decisions"
    );
    eprintln!("[COMPOSE] LIBRARY GROWS: D={goal_b} — primitives alone unreachable (advance×3=9<18); after BANKING M1, discovery finds {label_b} (disp {}), REACHED in {b_dec}≤{k_b} decisions (a real nest of the banked M1)", lib_b[0].disp);

    eprintln!("[COMPOSE] VERDICT: #150 faculty 6 — the LIBRARY COMPOSE LOOP, the ACTING-side twin of substrate_recurse/curriculum. A goal too deep for a bounded-horizon (H={H}), bounded-budget (K) alive(Act) planner is REACHED by BANKING a discovered macro (a GENUINELY encapsulate+nested sub-morph: M1 = nest(advance)×3, M2 = nest(M1)×2 — M2 CONTAINS M1 CONTAINS advance, a real nest-of-nest verified by M2==M1∘M1 and rising nested_settles). Flat fails BOTH the budget wall (K<D, orientation-independent) and the planner-blind wall (value_of=MAX); reachability+parsimony DISCOVERS advance×3 among generated candidates; and the SAME planner reaches the goal via macro decisions. The library GROWS THROUGH ACTING: the DEEPER goal (D=18) is reachable ONLY AFTER M1 is banked and re-nested (primitives-only discovery returns None; +banked-M1 finds a nest-of-M1). reuse IS nesting — the acting twin of reason→bank→reason-deeper. SUBSTRATE = grow + encapsulate/nest + planner + the default valence act rule (zero new core); PROTOCOL = the compose loop + generation + reachability/parsimony discovery. Honest: the step model is +1-directional (state-dependent maneuvers are future work), and value_of's cleanliness rests on goal-above-start (the budget wall is the orientation-free primary).");
}
