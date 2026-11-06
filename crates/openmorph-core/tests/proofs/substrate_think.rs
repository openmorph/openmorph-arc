// SPDX-License-Identifier: MIT-0
//! INTERNAL THINKING — the organism as a real dynamical system that TREATS what it received when the world is
//! quiet, and DISCOVERS structure it never directly experienced. The original openmorph thesis: the discovery
//! loop should be an emergent faculty of the living substrate, not a hand-written harness.
//!
//! THE CLAIM (go/no-go): an organism that learns only ADJACENT transitions online (s → s+1, each seen a handful
//! of times, NEVER a multi-step rollout) can, by THINKING — free-running its own recurrent model, feeding its
//! predicted next-state back in as its state (`Field::think`) — SIMULATE the H-step consequence of any state, and
//! CONSOLIDATE that simulated endpoint into direct long-range knowledge with the ordinary `learn_step`. The SAME
//! organism that does NOT think cannot answer the long-range query — it only ever saw adjacent steps. The only
//! difference between the arms is whether the organism runs its internal dynamics when unclamped.
//!
//! This is latent/systems consolidation: knowledge assembled from fragments by internal simulation, no external
//! data, no authored loop. GO = the thinking organism's H-step accuracy clearly beats the reactive one (and
//! chance). NO-GO (honest) = free-running the model degrades too fast to carry the consequence, and a different
//! internal dynamic is needed.
//!
//!   cargo test -p openmorph-core --test proofs substrate_think -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;

const K: usize = 5; // states in the cycle 0→1→…→K-1→0
const H: usize = 3; // the long-range horizon (never experienced directly)

fn onehot(s: usize) -> Vec<f32> {
    let mut v = vec![0.0; K];
    v[s] = 1.0;
    v
}

/// Build an organism with an adjacent-transition model (state onehot → next-state onehot) and an UNTRAINED
/// long-range head. Cells: state `0..K`, adjacent-prediction `K..2K`, long-range head `2K..3K`.
fn build() -> Field {
    let mut f = Field::new(3 * K);
    // adjacent model: one linear unit per prediction cell, reading the state onehot.
    for j in 0..K {
        f.add_unit((0..K).collect(), vec![0.0; K], 0.0, K + j);
    }
    // long-range head: one unit per head cell, reading the state onehot (trained only by THINKING).
    for j in 0..K {
        f.add_unit((0..K).collect(), vec![0.0; K], 0.0, 2 * K + j);
    }
    f
}

/// Train ONLY the adjacent transitions, online, a few times each (never a multi-step rollout).
fn wake(f: &mut Field) {
    let state: Vec<usize> = (0..K).collect();
    let pred: Vec<usize> = (K..2 * K).collect();
    for _ in 0..400 {
        for s in 0..K {
            let inputs: Vec<(usize, f32)> =
                state.iter().zip(onehot(s)).map(|(&c, v)| (c, v)).collect();
            let next = onehot((s + 1) % K);
            for (j, &pc) in pred.iter().enumerate() {
                f.learn_step(&inputs, pc, next[j], 0.2, 200);
            }
        }
    }
}

/// Read the H-step prediction of the long-range head for start state `s`: set the state, settle, argmax the head.
fn longrange_argmax(f: &mut Field, s: usize) -> usize {
    let state: Vec<usize> = (0..K).collect();
    let inputs: Vec<(usize, f32)> = state.iter().zip(onehot(s)).map(|(&c, v)| (c, v)).collect();
    f.stimulate(&inputs, 200);
    let head: Vec<f32> = (2 * K..3 * K).map(|c| f.cells()[c]).collect();
    (0..K).max_by(|&a, &b| head[a].total_cmp(&head[b])).unwrap()
}

#[test]
fn thinking_discovers_long_range_structure_never_experienced() {
    let state: Vec<usize> = (0..K).collect();
    let pred: Vec<usize> = (K..2 * K).collect();

    // ── REACTIVE control: learn adjacent transitions, never think. The long-range head stays untrained. ──
    let mut reactive = build();
    wake(&mut reactive);
    let mut react_correct = 0;
    for s in 0..K {
        if longrange_argmax(&mut reactive, s) == (s + H) % K {
            react_correct += 1;
        }
    }

    // ── THINKING organism: same wake, then — with NO external input — THINK. For each start state, free-run the
    //    adjacent model H steps (feed predicted state back in), reaching the H-step-ahead state by SIMULATION,
    //    and CONSOLIDATE that simulated endpoint into the long-range head. Knowledge it never experienced. ──
    let mut thinker = build();
    wake(&mut thinker);
    for _ in 0..200 {
        for s in 0..K {
            // set the start state, then think H transitions forward — the organism's own dynamics.
            let start: Vec<(usize, f32)> =
                state.iter().zip(onehot(s)).map(|(&c, v)| (c, v)).collect();
            thinker.stimulate(&start, 200);
            let simulated = thinker.think(&state, &pred, H, 0.0, 200); // the H-step consequence, self-generated
                                                                       // consolidate: the long-range head learns start → simulated endpoint (a target never seen awake).
            thinker.stimulate(&start, 200); // re-clamp the start state (think left it at the endpoint)
            for (j, &target) in simulated.iter().enumerate() {
                thinker.learn_step(&start, 2 * K + j, target, 0.2, 200);
            }
        }
    }
    let mut think_correct = 0;
    for s in 0..K {
        if longrange_argmax(&mut thinker, s) == (s + H) % K {
            think_correct += 1;
        }
    }

    eprintln!("[THINK] cycle of {K} states, horizon H={H} (never experienced — only adjacent steps were seen awake)");
    eprintln!(
        "  REACTIVE (no thinking): long-range accuracy {react_correct}/{K}   (chance {:.2})",
        1.0 / K as f32
    );
    eprintln!("  THINKING (free-run + consolidate): long-range accuracy {think_correct}/{K}");
    eprintln!("[THINK] VERDICT: internal thinking is a substrate faculty — the organism, unclamped, free-runs its own recurrent model (feeding its prediction back as its state) to SIMULATE the H-step consequence of its fragmentary experience, and consolidates that simulated endpoint into direct long-range knowledge it NEVER experienced. The reactive organism, which only reacts to input, cannot answer the query. Discovery from internal dynamics, no external data, no authored loop — the organism treating what it received when the world is quiet.");

    // GO: thinking recovers the long-range structure; reacting does not.
    assert!(
        think_correct >= K - 1,
        "the thinking organism must discover the H-step structure by internal simulation ({think_correct}/{K})"
    );
    assert!(
        think_correct > react_correct,
        "thinking must beat reacting on knowledge that was never directly experienced ({think_correct} vs {react_correct})"
    );
}
