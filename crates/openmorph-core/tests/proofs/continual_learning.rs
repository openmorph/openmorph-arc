// SPDX-License-Identifier: MIT-0
//! CONTINUAL LEARNING — the Phase-B enabler (MorphBot A8). Train ONE persistent morph across a
//! CURRICULUM of N distinct tasks, SEQUENTIALLY, and keep every earlier skill after all the later ones
//! are learned. This is not a new wall: it is a VALIDATION that the shipped isolation primitives compose
//! into a non-interfering continual-learning library at scale, so transfer ACCUMULATES instead of
//! overwriting the grown value head.
//!
//! The composition (all shipped, all grown):
//!   - PLACEMENT / NESTING (space)  — each task gets its OWN gate+actor cells (`add_unit`); a later
//!     `learn_step` is LOCAL (touches only `writers[out]`), so it cannot reach an earlier module's cells.
//!   - CONSOLIDATION (time + death) — `consolidate(cells)` freezes a matured module (plasticity → 0): a
//!     later `learn_step` / a shared router / a stray update cannot degrade it, AND `cull` spares it, so a
//!     dormant banked skill cannot be forgotten by STARVATION either.
//!   - ARBITRATION (routing)        — `select`/`arbitrate` route each context to the module whose grown
//!     gate (its learned recognition of the context) is highest — the competent module drives, no argmax.
//!   - RECRUITMENT (growth)         — surprise on a NOVEL task under the current library births a fresh
//!     module rather than overwriting a banked one (the surprise-birth law at the module scale).
//!
//! The battery runs the SAME organism through several arms and scores retention INSIDE (MAE per task,
//! chance-anchored by the opposite-slope tasks), deterministic, CI-fast:
//!   - CONSOLIDATED+NESTED+ARBITRATED (the library) — every earlier skill RETAINED across the sequence.
//!   - SHARED-PLASTIC (one readout, no isolation)   — CATASTROPHIC INTERFERENCE (task 1 collapses).
//!   - ablations (skip consolidation → death under dormancy; skip arbitration → mis-route) locate WHICH
//!     isolation is load-bearing for WHICH failure (space collision vs time/death overwrite vs mis-route).
//!
//! AUTHORING LEDGER (truth-not-trophy):
//! GIVEN = the task SEQUENCE + which module is active when (the VALUE HEADS are learned strictly
//! sequentially, one task at a time) + the context prototype codes (the routing key). GROWN = every skill
//! (each gate+actor by `learn_step`), the routing (a gate is just cells, learned over the context
//! prototypes), and the recruitment decision (the organism's own surprise EMA crossing the birth gate).
//! FIXED = the substrate laws, incl. the plasticity/cull immunity a consolidated unit inherits. LOOP = the
//! harness curriculum protocol (train a task, bank it, advance) — a usage protocol over the primitives, NOT
//! a substrate faculty, exactly per "skills are disposable." Retention comes from the GROWN isolation
//! primitives, NOT an authored per-task weight save/restore. (The value heads — the dimension A8 protects
//! against catastrophic overwrite — are the continual part; the router is trained once every context has
//! arrived, since a recognizer over the given context prototypes needs balanced exposure to all of them.)
//!
//!   cargo test -p openmorph-core --test proofs continual_learning -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;

use crate::common::Rng;

const N: usize = 5; // tasks in the curriculum
const C: usize = 6; // context cells (0..C)
const S: usize = C; // the shared STATE cell
const LR: f32 = 0.1;
const BUD: usize = 50;
const STEPS: usize = 4000; // learn steps per task phase
const ROUTE_STEPS: usize = 2000; // balanced router-refinement sweeps once the curriculum is complete
const EVAL: usize = 200; // eval samples per task

/// Per-task policy slopes. Adjacent signs conflict and task 0 vs task N-1 are OPPOSITE, so a shared
/// readout that ends on the last task is maximally wrong on task 0 (chance-anchored collapse).
const W: [f32; N] = [1.6, 2.0, -0.8, 0.8, -1.6];

fn target(i: usize, s: f32) -> f32 {
    (W[i] * s).tanh()
}

/// Context prototype codes: distinct 2-hot ±1 patterns (each task is +1 on a distinct PAIR of cells, −1
/// elsewhere). The pairs OVERLAP — {0,1},{2,3},{4,5},{0,2},{1,3} — so routing is NOT a trivial one-hot
/// (a gate must combine cells), yet each pair shares ≤1 cell with any other, so each context is
/// one-vs-all linearly separable: the grown linear gate can recognize its own context and reject the rest.
const CODE: [[f32; C]; N] = [
    [1.0, 1.0, -1.0, -1.0, -1.0, -1.0], // {0,1}
    [-1.0, -1.0, 1.0, 1.0, -1.0, -1.0], // {2,3}
    [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0], // {4,5}
    [1.0, -1.0, 1.0, -1.0, -1.0, -1.0], // {0,2}
    [-1.0, 1.0, -1.0, 1.0, -1.0, -1.0], // {1,3}
];

fn code(i: usize) -> [f32; C] {
    CODE[i]
}

/// The membrane inputs for task `i` at state `s`: the context code on 0..C, the state on cell S.
fn inp(i: usize, s: f32) -> Vec<(usize, f32)> {
    let mut v: Vec<(usize, f32)> = code(i).iter().enumerate().map(|(j, &x)| (j, x)).collect();
    v.push((S, s));
    v
}

/// Train ONE more task-phase into the persistent library, SEQUENTIALLY. This is the continual part and
/// the dimension A8 protects: the ACTOR (this task's policy over the state — the VALUE HEAD) is learned,
/// then banked when `freeze` (consolidated → immune to later `learn_step` AND to `cull`). Nothing later
/// ever trains it again, so transfer accumulates instead of overwriting. The module's GATE cell is
/// created here but trained later, once every context has arrived (see `refine_router`).
fn phase_train(
    f: &mut Field,
    gates: &mut Vec<usize>,
    actors: &mut Vec<usize>,
    i: usize,
    freeze: bool,
    rng: &mut Rng,
) {
    let gate = f.len();
    f.add_unit((0..C).collect(), vec![0.0; C], 0.0, gate);
    let actor = f.len();
    f.add_unit(vec![S], vec![0.0], 0.0, actor);
    gates.push(gate);
    actors.push(actor);
    for _ in 0..STEPS {
        let s = rng.sym();
        f.learn_step(&inp(i, s), actor, target(i, s), LR, BUD); // the value head — the continual part
    }
    if freeze {
        f.consolidate(&[actor]); // bank the value head as it matures
    }
}

/// Train the ROUTERS: a grown context-recognizer per module, learned with balanced one-vs-all exposure
/// over the context prototypes (each gate: +1 on its OWN context, −1 on every other). The routers are
/// trained once every context has arrived, from fresh weights so the tanh gate never saturates into a
/// wrong, vanishing-gradient corner — a moderate, balanced budget keeps the margin intact for `select` (a
/// full-match preactivation stays above a partial-match one). This touches ONLY the gates (grown
/// recognizers over given context prototypes); the frozen value heads are never re-opened. The routing is
/// grown — a gate is just cells. `ids[m]` is the task-id of module `m` (order may differ after recruitment).
fn refine_router(f: &mut Field, gates: &[usize], ids: &[usize], rng: &mut Rng) {
    for _ in 0..ROUTE_STEPS {
        for (i, &gate) in gates.iter().enumerate() {
            let own = ids[i]; // this module's own context
            for &kid in ids {
                let t = if kid == own { 1.0 } else { -1.0 };
                f.learn_step(&inp(kid, rng.sym()), gate, t, LR, BUD);
            }
        }
    }
}

/// MAE of module `i`'s actor read DIRECTLY (the active module is KNOWN during training — a given).
fn mae_actor(f: &mut Field, actor: usize, i: usize, rng: &mut Rng) -> f32 {
    let mut e = 0.0;
    for _ in 0..EVAL {
        let s = rng.sym();
        f.stimulate(&inp(i, s), BUD);
        e += (f.cells()[actor] - target(i, s)).abs();
    }
    e / EVAL as f32
}

/// MAE of the library's ARBITRATED output on task `i` (routes by grown gate, reads the winning actor).
fn mae_arbitrated(
    f: &mut Field,
    gates: &[usize],
    actors: &[usize],
    i: usize,
    rng: &mut Rng,
) -> f32 {
    let mut e = 0.0;
    for _ in 0..EVAL {
        let s = rng.sym();
        f.stimulate(&inp(i, s), BUD);
        e += (f.arbitrate(gates, actors) - target(i, s)).abs();
    }
    e / EVAL as f32
}

/// Routing accuracy: fraction of contexts across all tasks whose winning gate is the RIGHT module.
fn routing_accuracy(f: &mut Field, gates: &[usize], rng: &mut Rng) -> f32 {
    let mut ok = 0usize;
    let mut total = 0usize;
    for i in 0..gates.len() {
        for _ in 0..EVAL {
            let s = rng.sym();
            f.stimulate(&inp(i, s), BUD);
            if f.select(gates) == i {
                ok += 1;
            }
            total += 1;
        }
    }
    ok as f32 / total as f32
}

/// Build the full library: N modules trained SEQUENTIALLY into one persistent morph. When `freeze`, each
/// value head is banked as it matures and the routers are consolidated after the whole curriculum (a
/// permanent, dormancy-proof library); when not, everything stays plastic (the control). Returns the
/// organism plus the gate/actor cell of every module.
fn build_library(freeze: bool, seed: u64) -> (Field, Vec<usize>, Vec<usize>) {
    let mut f = Field::new(C + 1);
    let mut rng = Rng::new(seed);
    let (mut gates, mut actors) = (Vec::new(), Vec::new());
    for i in 0..N {
        phase_train(&mut f, &mut gates, &mut actors, i, freeze, &mut rng);
    }
    refine_router(
        &mut f,
        &gates,
        &(0..gates.len()).collect::<Vec<_>>(),
        &mut rng,
    ); // balance the routers over all contexts now seen
    if freeze {
        f.consolidate(&gates); // bank the routers once the curriculum is complete
    }
    (f, gates, actors)
}

// ───────────────────────────────────────────────────────────────────────────────────────────────────
// 1) THE LIBRARY — every earlier skill retained after the WHOLE sequence; routing accurate.
// ───────────────────────────────────────────────────────────────────────────────────────────────────
#[test]
fn the_library_retains_every_skill_across_the_sequence() {
    let mut f = Field::new(C + 1);
    let mut rng = Rng::new(0xC0117AC7);
    let (mut gates, mut actors) = (Vec::new(), Vec::new());

    // Train the curriculum ONE task at a time, banking each value head as it matures. Record each task's
    // MAE the moment it finishes training (over the library grown SO FAR) — the "fresh skill" baseline.
    let mut mae_fresh = [0.0f32; N];
    for i in 0..N {
        phase_train(&mut f, &mut gates, &mut actors, i, true, &mut rng);
        mae_fresh[i] = mae_actor(&mut f, actors[i], i, &mut rng); // active module is known during training
    }
    refine_router(
        &mut f,
        &gates,
        &(0..gates.len()).collect::<Vec<_>>(),
        &mut rng,
    ); // balance the routers over all contexts now seen
    f.consolidate(&gates); // bank the routers now the curriculum is complete

    // After the WHOLE sequence: retention MAE on every earlier task, and routing accuracy.
    let retention: Vec<f32> = (0..N)
        .map(|i| mae_arbitrated(&mut f, &gates, &actors, i, &mut rng))
        .collect();
    let route = routing_accuracy(&mut f, &gates, &mut rng);
    let worst = retention.iter().cloned().fold(0.0f32, f32::max);
    let drift0 = (retention[0] - mae_fresh[0]).abs();

    eprintln!("[CONTINUAL] a persistent morph trained across a {N}-task curriculum, each module consolidated+nested and arbitrated by a grown gate:");
    for i in 0..N {
        eprintln!(
            "  task {i} (policy tanh({:+.1}·s)): MAE fresh {:.3}  →  after full sequence {:.3}",
            W[i], mae_fresh[i], retention[i]
        );
    }
    eprintln!(
        "  worst retained MAE = {worst:.3}   routing accuracy = {:.1}%   task-0 drift over the whole run = {drift0:.3}",
        route * 100.0
    );
    eprintln!("[CONTINUAL] VERDICT: the shipped isolation primitives COMPOSE — placement/nesting isolates in space, consolidation freezes in time (and grants cull-immunity), arbitration routes the grown gate. Trained sequentially, every earlier skill is RETAINED after all later ones; transfer ACCUMULATES rather than overwriting. Phase-B-ready.");

    assert!(
        worst < 0.15,
        "every task must stay retained after the full sequence (worst MAE {worst:.3})"
    );
    assert!(
        route > 0.98,
        "the grown gates must route each context to its own module (routing {:.1}%)",
        route * 100.0
    );
    assert!(
        drift0 < 0.05,
        "task 0 must not drift as tasks 1..N are learned (drift {drift0:.3})"
    );
}

// ───────────────────────────────────────────────────────────────────────────────────────────────────
// 2) SHARED-PLASTIC control — one readout, no isolation → CATASTROPHIC INTERFERENCE.
// ───────────────────────────────────────────────────────────────────────────────────────────────────
#[test]
fn shared_plastic_readout_catastrophically_forgets() {
    // ONE actor over the shared state, no gate, no consolidation — MorphBot's measured shared-readout
    // policy. Train the SAME sequential curriculum; a single readout cannot hold N conflicting slopes.
    let mut f = Field::new(C + 1);
    let mut rng = Rng::new(0x5A1AD);
    let actor = f.len();
    f.add_unit(vec![S], vec![0.0], 0.0, actor);

    let mae_shared = |f: &mut Field, i: usize, rng: &mut Rng| -> f32 {
        let mut e = 0.0;
        for _ in 0..EVAL {
            let s = rng.sym();
            f.stimulate(&inp(i, s), BUD);
            e += (f.cells()[actor] - target(i, s)).abs();
        }
        e / EVAL as f32
    };

    let mut task0_fresh = 0.0;
    for i in 0..N {
        for _ in 0..STEPS {
            let s = rng.sym();
            f.learn_step(&inp(i, s), actor, target(i, s), LR, BUD);
        }
        if i == 0 {
            task0_fresh = mae_shared(&mut f, 0, &mut rng); // task 0, right after learning it
        }
    }
    let task0_after = mae_shared(&mut f, 0, &mut rng); // task 0, after the whole sequence

    // For contrast, the library on the same curriculum retains task 0.
    let (mut lib, gates, actors) = build_library(true, 0xC0117AC7);
    let lib_task0 = mae_arbitrated(&mut lib, &gates, &actors, 0, &mut rng);

    eprintln!("[CONTINUAL] SHARED-PLASTIC (one readout, no isolation) vs the LIBRARY, on the SAME curriculum:");
    eprintln!(
        "  shared readout : task-0 MAE {task0_fresh:.3} (fresh)  →  {task0_after:.3} (after tasks 1..{})  ← catastrophically overwritten",
        N - 1
    );
    eprintln!("  library        : task-0 MAE {lib_task0:.3} (after the whole sequence)          ← retained");
    eprintln!("[CONTINUAL] VERDICT: with NO space isolation, later tasks overwrite the shared readout — task 0 collapses to the opposite policy (this reproduces MorphBot's 41/80-rounds-HURT). The library's per-module cells + local learn keep it isolated. SPACE-collision is the shared-readout failure mode.");

    assert!(
        task0_fresh < 0.15,
        "task 0 must be learnable fresh on the shared readout (MAE {task0_fresh:.3})"
    );
    assert!(
        task0_after > 0.6,
        "the shared readout must catastrophically forget task 0 (MAE {task0_after:.3})"
    );
    assert!(
        lib_task0 < 0.15,
        "the library must retain task 0 on the same curriculum (MAE {lib_task0:.3})"
    );
}

// ───────────────────────────────────────────────────────────────────────────────────────────────────
// 3) WHICH ISOLATION IS LOAD-BEARING — space (nesting) vs time/death (consolidation) vs route (arbitrate).
// ───────────────────────────────────────────────────────────────────────────────────────────────────
#[test]
fn which_isolation_is_load_bearing_for_which_failure() {
    let mut rng = Rng::new(0xB0A12);

    // (TIME / DEATH) — a dormant banked skill under the alive economy. Build two identical libraries;
    // one consolidated, one left PLASTIC. Subject both to dormancy (the persistent morph "lives" while
    // OTHER modules would be active): no firing → upkeep drains energy → `cull`. A consolidated module is
    // cull-immune; a plastic one STARVES and the library forgets by DEATH.
    let (mut lib_frozen, gf, af) = build_library(true, 0xDEAD01);
    let (mut lib_plastic, gp, ap) = build_library(false, 0xDEAD01);
    for lib in [&mut lib_frozen, &mut lib_plastic] {
        for _ in 0..60 {
            lib.stimulate(&[], 10); // dormant — nothing fires
            lib.metabolize(1.0, 1.0); // not fired ⇒ −1 upkeep each cycle
            lib.cull(0.0);
        }
    }
    let frozen_task0 = mae_arbitrated(&mut lib_frozen, &gf, &af, 0, &mut rng);
    let plastic_task0 = mae_arbitrated(&mut lib_plastic, &gp, &ap, 0, &mut rng);
    let plastic_alive = lib_plastic
        .units
        .iter()
        .filter(|u| u.write == af[0])
        .count();
    let frozen_alive = lib_frozen.units.iter().filter(|u| u.write == af[0]).count();

    // (ROUTE) — the consolidated library retains every module, but WITHOUT arbitration you cannot pick
    // the competent one. Reading a FIXED module (module 0) for every context mis-routes the others.
    let route_task_last = mae_arbitrated(&mut lib_frozen, &gf, &af, N - 1, &mut rng); // arbitrated: correct
    let misroute_task_last = {
        // no routing: always read module 0's actor for task N-1 (opposite policy → wrong).
        let mut e = 0.0;
        for _ in 0..EVAL {
            let s = rng.sym();
            lib_frozen.stimulate(&inp(N - 1, s), BUD);
            e += (lib_frozen.cells()[af[0]] - target(N - 1, s)).abs();
        }
        e / EVAL as f32
    };

    // (SPACE) — recap from the shared-plastic arm: a shared readout overwrites in space.
    eprintln!("[CONTINUAL] WHICH ISOLATION IS LOAD-BEARING (each failure ↔ the primitive that prevents it):");
    eprintln!(
        "  TIME / DEATH  (dormancy under the economy): consolidated task-0 MAE {frozen_task0:.3} ({frozen_alive} unit alive)  vs  plastic {plastic_task0:.3} ({plastic_alive} alive)\n                → CONSOLIDATION (cull-immunity) is load-bearing: a plastic banked skill starves and is culled."
    );
    eprintln!(
        "  ROUTE         (pick the competent module): arbitrated task-{} MAE {route_task_last:.3}  vs  fixed-module (no routing) {misroute_task_last:.3}\n                → ARBITRATION (grown gate) is load-bearing: without routing you read the wrong module.",
        N - 1
    );
    eprintln!("  SPACE         (shared readout): see `shared_plastic_readout_catastrophically_forgets` — NESTING/placement is load-bearing: a later learn_step overwrites a shared cell.");
    eprintln!("[CONTINUAL] VERDICT: three orthogonal isolations, three orthogonal failures. NESTING stops SPACE collision, CONSOLIDATION stops TIME-overwrite AND death-by-dormancy, ARBITRATION stops MIS-ROUTE. Composed, they make the persistent morph a non-interfering continual-learning library — the Phase-B enabler.");

    // TIME/DEATH: consolidation is load-bearing.
    assert!(
        frozen_task0 < 0.15,
        "the consolidated library must survive dormancy (MAE {frozen_task0:.3})"
    );
    assert!(
        plastic_task0 > 0.5,
        "the plastic library must forget by death under dormancy (MAE {plastic_task0:.3})"
    );
    assert!(
        frozen_alive == 1 && plastic_alive == 0,
        "consolidation must grant cull-immunity: frozen {frozen_alive} alive, plastic {plastic_alive} alive"
    );
    // ROUTE: arbitration is load-bearing.
    assert!(
        route_task_last < 0.15,
        "arbitration must route the last task correctly (MAE {route_task_last:.3})"
    );
    assert!(
        misroute_task_last > 0.5,
        "without arbitration, a fixed module must mis-route (MAE {misroute_task_last:.3})"
    );
}
// ───────────────────────────────────────────────────────────────────────────────────────────────────
// 4) RECRUITMENT-ON-INTERFERENCE — a novel task grows the library instead of overwriting a banked one.
// ───────────────────────────────────────────────────────────────────────────────────────────────────
const BIRTH_GATE: f32 = 0.4; // the surprise threshold that decides a new module (module-scale birth)

/// The organism's SURPRISE on task `i` under the current library: the error EMA of the arbitrated output.
fn surprise_ema(f: &mut Field, gates: &[usize], actors: &[usize], i: usize, rng: &mut Rng) -> f32 {
    let mut ema = 0.0;
    for _ in 0..200 {
        let s = rng.sym();
        f.stimulate(&inp(i, s), BUD);
        ema = 0.9 * ema + 0.1 * (f.arbitrate(gates, actors) - target(i, s)).abs();
    }
    ema
}

#[test]
fn recruitment_grows_the_library_on_a_novel_task() {
    let mut rng = Rng::new(0x2EC0117);

    // A persistent library of 3 banked skills (value heads consolidated; router trained over the 3 known
    // contexts). This is the running organism a new task will arrive at.
    let mut f = Field::new(C + 1);
    let (mut gates, mut actors) = (Vec::new(), Vec::new());
    let mut ids: Vec<usize> = Vec::new(); // task-id of each module (order need not equal id after recruit)
    for i in 0..3 {
        phase_train(&mut f, &mut gates, &mut actors, i, true, &mut rng);
        ids.push(i);
    }
    refine_router(&mut f, &gates, &ids, &mut rng);
    let mae_task0_before = mae_arbitrated(&mut f, &gates, &actors, 0, &mut rng);

    // A FAMILIAR task (one the library already serves) leaves LOW surprise → must NOT recruit.
    let surprise_familiar = surprise_ema(&mut f, &gates, &actors, 1, &mut rng);
    // A NOVEL task (task 4: an unseen context AND an opposite policy) is mis-served by the current
    // library → HIGH surprise → the SAME surprise-birth law that grows a hidden unit grows a MODULE.
    let novel = 4usize;
    let surprise_novel = surprise_ema(&mut f, &gates, &actors, novel, &mut rng);
    let recruited = surprise_novel > BIRTH_GATE;
    if recruited {
        phase_train(&mut f, &mut gates, &mut actors, novel, true, &mut rng); // birth + bank the new module
        ids.push(novel);
        refine_router(&mut f, &gates, &ids, &mut rng); // fold the new context into the router
        f.consolidate(&gates);
    }

    // After recruiting: the novel task is served, and the banked skills are UNTOUCHED.
    let mae_novel_after = mae_arbitrated(&mut f, &gates, &actors, novel, &mut rng);
    let mae_task0_after = mae_arbitrated(&mut f, &gates, &actors, 0, &mut rng);

    eprintln!("[CONTINUAL] RECRUITMENT — a novel task arrives at a persistent 3-skill library:");
    eprintln!(
        "  surprise: familiar task {:.3} (< gate {BIRTH_GATE} ⇒ reuse)   novel task {:.3} (> gate ⇒ RECRUIT = {recruited})",
        surprise_familiar, surprise_novel
    );
    eprintln!(
        "  after recruiting: novel-task MAE {mae_novel_after:.3} (served)   banked task-0 MAE {mae_task0_before:.3} → {mae_task0_after:.3} (untouched)"
    );
    eprintln!("[CONTINUAL] VERDICT: recruitment-on-interference is the surprise-birth law at the MODULE scale — no authored trigger. High surprise on an unseen task BIRTHS a fresh module (and banks it); a familiar task reuses. The library GROWS MONOTONICALLY — the fourth mechanism of the continual composition.");

    assert!(
        surprise_familiar < BIRTH_GATE,
        "a task the library already serves must not recruit (surprise {surprise_familiar:.3})"
    );
    assert!(
        recruited,
        "high surprise on the novel task must trigger recruitment (surprise {surprise_novel:.3})"
    );
    assert!(
        mae_novel_after < 0.15,
        "the recruited module must serve the novel task (MAE {mae_novel_after:.3})"
    );
    assert!(
        (mae_task0_after - mae_task0_before).abs() < 0.05,
        "recruiting must not touch a banked skill (task 0 MAE {mae_task0_before:.3} → {mae_task0_after:.3})"
    );
}
