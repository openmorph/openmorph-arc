// SPDX-License-Identifier: MIT-0
//! GOAL-COMPLETION REWARD (#155, close the loop) — the missing emitter. A goal is a CLAMPED preferred cell (a cell
//! the organism wants activated); the organism can already SPONTANEOUSLY create one (`orient`) and REACH it
//! (`relax_values` + `alive(Act)`), but nothing rewarded the achievement — reaching the goal was a silent
//! terminus. `Config::goal_reward` closes it: when the acted option lands on a self-set goal, the organism pays
//! ITSELF the reward, folded through the SAME channel as an external one (so the reaching action is credited in its
//! learned option value, the default valence rule's `(μ, visits)` record), then CONSUMES the goal. create-goal →
//! pursue → ACHIEVE → reward, no external reward needed.
//!
//! Scored INSIDE, truth-not-trophy — three arms on the SAME chain world (a self-set goal at the far end), the
//! default acting rule:
//!   * REACH (goal_reward on, goal reachable): reaches the goal ⇒ `goals_reached()` rises AND the reaching option's
//!     valence carries the intrinsic reward: it ends `lr·goal_reward` above the OFF arm's, which walked the SAME
//!     path and folded the same visit with reward 0.
//!   * NEVER-REACH control (goal_reward on, goal UNREACHABLE — the edge into it removed): wanders, never lands on
//!     the goal ⇒ `goals_reached() == 0`, the goal option never folded (its valence stays the untried prior).
//!     "A control that never reaches it gets nothing."
//!   * OFF control (goal_reward == 0, goal reachable): reaches the goal but the emitter is gated off ⇒
//!     `goals_reached() == 0`, no intrinsic credit — the byte-identical-off guarantee.
//!
//!   cargo test -p openmorph-core --test proofs substrate_goal_reward -- --nocapture

use openmorph_core::{Event, Field, Outcome};

const N: usize = 6; // chain cells 0—1—2—3—4—5
const GOAL: usize = N - 1;
const START: usize = 0;
const MAXSTEPS: usize = 200;
const GOAL_REWARD: f32 = 1.0;

/// Chain adjacency; if `reachable` is false the edge INTO the goal is removed (the goal is isolated).
fn adjacency(reachable: bool) -> Vec<Vec<usize>> {
    let mut adj = vec![Vec::new(); N];
    for i in 0..N - 1 {
        let a = i;
        let b = i + 1;
        // skip the edge touching the goal when the goal must be unreachable
        if !reachable && (a == GOAL || b == GOAL) {
            continue;
        }
        adj[a].push(b);
        adj[b].push(a);
    }
    adj
}

/// Set up a morph that SELF-SETS its goal at the far end (via `orient` over its own drive) and plans the descent.
fn setup(goal_reward: f32, reachable: bool) -> (Field, Vec<Vec<usize>>, usize) {
    let mut f = Field::new(N);
    f.config_mut().goal_reward = goal_reward;
    // Prime a little reward-memory on the far cell so the organism's OWN `orient` selects it as the goal
    // (self-set from its grown drive — `learning_progress` is 0 here with no surprise history).
    f.reinforce(GOAL, 0.5);
    let candidates: Vec<usize> = (0..N).collect();
    let goal = f
        .orient(&candidates)
        .expect("orient selects a self-set goal");
    let adj = adjacency(reachable);
    f.relax_values(&adj, 1.0, 0); // cost-to-go field from the clamped goal
    (f, adj, goal)
}

/// Navigate from START by `alive(Act)`, descending the value field, until the `Act` that FOLDS the goal option's
/// arrival (and, when the reward is on, credits it) has run, or the step budget runs out. Returns (moves taken,
/// reached-the-goal-cell?). Both reachable arms stop at the same point, so they walk the same path.
fn navigate(f: &mut Field, adj: &[Vec<usize>]) -> (usize, bool) {
    let mut s = START;
    let mut last: Option<(usize, f32)> = None;
    for step in 0..MAXSTEPS {
        let opts = adj[s].clone();
        if opts.is_empty() {
            break;
        }
        // The emitter fires when we FEED the goal option back as `last`: stop once that fold has run.
        let folds_the_goal = matches!(last, Some((o, _)) if o == GOAL);
        let chosen = match f.alive(Event::Act {
            options: &opts,
            last,
        }) {
            Outcome::Acted { chosen } => chosen,
            _ => break,
        };
        if folds_the_goal {
            return (step, true);
        }
        last = Some((chosen, 0.0)); // external reward is 0 — the ONLY reward is the intrinsic goal-completion one
        s = chosen;
    }
    (MAXSTEPS, false)
}

#[test]
fn reaching_a_self_set_goal_rewards_the_path_a_miss_gets_nothing() {
    // REACH arm.
    let (mut f_reach, adj_r, goal) = setup(GOAL_REWARD, true);
    let (steps_r, reached_r) = navigate(&mut f_reach, &adj_r);
    let reached_count = f_reach.goals_reached();
    let val_reach = f_reach.valence_of(goal);

    // NEVER-REACH control: goal_reward on, but the goal is isolated.
    let (mut f_miss, adj_m, goal_m) = setup(GOAL_REWARD, false);
    let val_miss_before = f_miss.valence_of(goal_m);
    let (_steps_m, reached_m) = navigate(&mut f_miss, &adj_m);
    let miss_count = f_miss.goals_reached();
    let val_miss_after = f_miss.valence_of(goal_m);

    // OFF control: goal_reward == 0, goal reachable — the same path, the goal's visit folded with reward 0.
    let (mut f_off, adj_o, goal_o) = setup(0.0, true);
    let (steps_o, reached_o) = navigate(&mut f_off, &adj_o);
    let off_count = f_off.goals_reached();
    let val_off = f_off.valence_of(goal_o);
    let credit = val_reach - val_off;
    let lr = f_reach.config().free_energy_lr;

    eprintln!("[GOAL-REWARD] self-set goal at cell {GOAL}; external reward is ZERO everywhere (only the intrinsic goal-completion reward exists):");
    eprintln!(
        "[GOAL-REWARD]   REACH (on, reachable)    : reached={reached_r} in {steps_r} moves  goals_reached={reached_count}  valence(goal) {val_reach:.4}"
    );
    eprintln!(
        "[GOAL-REWARD]   OFF (goal_reward=0)      : reached={reached_o} in {steps_o} moves  goals_reached={off_count}  valence(goal) {val_off:.4}"
    );
    eprintln!(
        "[GOAL-REWARD]   NEVER-REACH (on, blocked): reached={reached_m}  goals_reached={miss_count}  valence(goal) {val_miss_before:.4}→{val_miss_after:.4}"
    );
    eprintln!(
        "[GOAL-REWARD]   credit = REACH − OFF = {credit:.4} (lr·goal_reward = {:.4})",
        lr * GOAL_REWARD
    );

    // REACH: the organism reaches its self-set goal and pays itself — the reaching option is credited.
    assert!(
        reached_r,
        "the organism must reach its reachable self-set goal"
    );
    assert!(
        reached_count >= 1,
        "reaching a self-set goal must register a completion: {reached_count}"
    );
    assert_eq!(
        (reached_o, steps_o),
        (reached_r, steps_r),
        "the OFF arm walks the same path"
    );
    assert!(
        credit > 0.9 * lr * GOAL_REWARD,
        "the reaching option's valence must carry the intrinsic reward (one fold of it at lr {lr}): REACH {val_reach:.4} vs OFF {val_off:.4}"
    );

    // NEVER-REACH: a control that never reaches gets NOTHING.
    assert!(!reached_m, "the blocked goal must be unreachable");
    assert_eq!(miss_count, 0, "a never-reached goal must yield no reward");
    assert_eq!(
        val_miss_after.to_bits(),
        val_miss_before.to_bits(),
        "a never-reached goal option is never folded: {val_miss_before:.4} → {val_miss_after:.4}"
    );

    // OFF: reaching with the emitter gated off yields nothing (byte-identical-off).
    assert_eq!(off_count, 0, "goal_reward=0 must emit no completion reward");

    eprintln!("[GOAL-REWARD] VERDICT: the loop is CLOSED — the organism creates its own goal (`orient`), pursues it (`relax_values`+`alive`), and on ACHIEVING it pays ITSELF a reward that credits the reaching option (valence {credit:+.4} over the same path with the reward off), with NO external reward anywhere. A goal it never reaches gets nothing (never folded), and with the reward gated off (goal_reward=0) reaching it is once again a silent terminus (byte-identical). create-goal → pursue → achieve → reward.");
}

/// THE FIX (2026-08-28) — the goal-completion reward must fire on the OPT-IN ACTOR PATHS too. It was silently
/// DEAD there: `free_energy_actor` and `valence_actor` both early-return from `alive(Act)` BEFORE the default goal
/// block, so `Config::goal_reward` did nothing whenever either actor was enabled — a documented config that was a
/// no-op on two of the three selection paths. Both actors now credit a reached self-set goal through their OWN
/// reward channel (the learned option value, via `goal_reward_for` + `consume_goal`), byte-identical when
/// goal_reward=0.
///
/// Drive `alive(Act)` DIRECTLY with the acted option landing ON a clamped self-set goal (no chain navigation — the
/// valence / free-energy actors score their learned values, not the relaxed value field, so a direct fire isolates
/// the emitter cleanly):
///   * ON  (actor + goal_reward): first fire credits the goal ⇒ `goals_reached` 0→1; a SECOND identical fire does
///     NOT double-credit (stays 1) — proving the goal was CONSUMED (unclamped) on the actor path.
///   * OFF (actor, goal_reward=0): the fire credits nothing ⇒ `goals_reached` stays 0 — the byte-identical-off
///     guarantee on the actor path (before the fix, ON behaved like this too — the bug).
fn actor_credits_a_reached_goal(enable: impl Fn(&mut Field)) -> (u64, u64, u64) {
    // ON: actor + goal_reward. `setup` self-sets (clamps) the goal via `orient`.
    let (mut f, _adj, goal) = setup(GOAL_REWARD, true);
    enable(&mut f);
    let opts = [goal, START];
    let fire = |f: &mut Field| {
        let _ = f.alive(Event::Act {
            options: &opts,
            last: Some((goal, 0.0)), // external reward 0 — the ONLY reward is the intrinsic goal-completion one
        });
    };
    fire(&mut f);
    let after_first = f.goals_reached();
    fire(&mut f);
    let after_second = f.goals_reached(); // must stay == after_first (goal consumed, no double-credit)

    // OFF control: same actor, goal_reward = 0 → byte-identical, no completion ever.
    let (mut g, _adj2, goal2) = setup(0.0, true);
    enable(&mut g);
    let opts2 = [goal2, START];
    let _ = g.alive(Event::Act {
        options: &opts2,
        last: Some((goal2, 0.0)),
    });
    let off = g.goals_reached();
    (after_first, after_second, off)
}

#[test]
fn goal_reward_fires_on_every_acting_rule() {
    let (fe_first, fe_second, fe_off) =
        actor_credits_a_reached_goal(|f| f.config_mut().free_energy_actor = true);
    let (va_first, va_second, va_off) =
        actor_credits_a_reached_goal(|f| f.config_mut().valence_actor = true);
    // The DEFAULT rule (valence since 1.6, no flag).
    let (df_first, df_second, df_off) = actor_credits_a_reached_goal(|_| {});
    eprintln!("[GOAL-REWARD/ACTOR] default rule     : goals_reached after 1st fire={df_first}, after 2nd={df_second}, OFF={df_off}");
    assert_eq!(
        df_first, 1,
        "the default rule must credit a reached self-set goal: {df_first}"
    );
    assert_eq!(
        df_second, 1,
        "the default rule must CONSUME the goal (no double-credit): {df_second}"
    );
    assert_eq!(
        df_off, 0,
        "the default rule + goal_reward=0 must credit nothing: {df_off}"
    );

    eprintln!("[GOAL-REWARD/ACTOR] free-energy actor: goals_reached after 1st fire={fe_first}, after 2nd={fe_second} (consumed ⇒ no double-credit), OFF={fe_off}");
    eprintln!("[GOAL-REWARD/ACTOR] valence actor    : goals_reached after 1st fire={va_first}, after 2nd={va_second} (consumed ⇒ no double-credit), OFF={va_off}");

    // Free-energy actor: the reward now fires (was silently 0 before the fix), and the goal is consumed.
    assert_eq!(
        fe_first, 1,
        "free-energy actor must credit a reached self-set goal (was dead before the fix): {fe_first}"
    );
    assert_eq!(
        fe_second, 1,
        "the goal must be CONSUMED on the free-energy path — a second fire must not double-credit: {fe_second}"
    );
    assert_eq!(
        fe_off, 0,
        "free-energy actor + goal_reward=0 must credit nothing (byte-identical-off): {fe_off}"
    );

    // Valence actor: same.
    assert_eq!(
        va_first, 1,
        "valence actor must credit a reached self-set goal (was dead before the fix): {va_first}"
    );
    assert_eq!(
        va_second, 1,
        "the goal must be CONSUMED on the valence path — a second fire must not double-credit: {va_second}"
    );
    assert_eq!(
        va_off, 0,
        "valence actor + goal_reward=0 must credit nothing (byte-identical-off): {va_off}"
    );

    eprintln!("[GOAL-REWARD/ACTOR] VERDICT: the goal-completion loop closes on every selection path — the default rule, free-energy, and the valence alias. `Config::goal_reward` is no longer a silent no-op under the opt-in actors; each credits a reached self-set goal through its own reward channel and consumes it, byte-identical when the reward is off.");
}

#[test]
fn orient_many_clamps_multiple_self_set_subgoals() {
    // A star of candidate goal cells with distinct primed drive; orient_many should clamp the top-k by the grown
    // free-energy score (drive here, since learning_progress is 0), as SIMULTANEOUS subgoals.
    let mut f = Field::new(8);
    // prime drive so the ranking is 6 > 5 > 4 > 3 > ... (cells 3..7 are candidate goals)
    for (rank, &c) in [7usize, 6, 5, 4, 3].iter().enumerate() {
        f.reinforce(c, (5 - rank) as f32); // 7←5, 6←4, 5←3, 4←2, 3←1
    }
    let candidates = vec![3usize, 4, 5, 6, 7];
    let goals = f.orient_many(&candidates, 3);

    eprintln!("[ORIENT-MANY] top-3 self-set subgoals from primed drive (7>6>5>4>3): {goals:?}");
    assert_eq!(goals.len(), 3, "orient_many must clamp exactly k=3 goals");
    assert_eq!(
        goals,
        vec![7, 6, 5],
        "orient_many must pick the top-k by the grown free-energy score, in order"
    );
    // All three are clamped simultaneously (multi-source): each is a source the planner descends from.
    for &g in &goals {
        assert_eq!(
            f.value_of(g),
            0.0,
            "each subgoal is clamped at cost-to-go 0"
        );
    }
    // A single call orients to one — the equivalence orient == orient_many(_, 1).
    let mut f1 = Field::new(8);
    f1.reinforce(7, 5.0);
    let one = f1.orient_many(&[3, 4, 5, 6, 7], 1);
    assert_eq!(one, vec![7], "orient_many(_, 1) is the single-goal orient");

    eprintln!("[ORIENT-MANY] VERDICT: multi-goal generation is now a CORE primitive — `orient_many` self-scores the candidates by the same grown free-energy quantity and clamps the top-k as SIMULTANEOUS subgoals (each a cost-to-go source `relax_values` descends from), with no authored top-k sort (the grown WTA `select` applied k times without replacement). Planning can spontaneously set multiple internal goals.");
}
