// SPDX-License-Identifier: MIT-0
//! THE AUTONOMOUS DISCOVERY LOOP AS A SUBSTRATE FACULTY — `Field::dream`. The organism's own resting dynamics,
//! in CORE: left alone with NO external input it lives a wake-process-sleep cycle — THINK (free-run its model to
//! simulate the consequences of its fragmentary experience) → CONSOLIDATE (the simulated consequence becomes
//! direct knowledge) → SLEEP (the maintenance heartbeat consolidates the relied-upon memory and RESTS). The
//! discovery loop is not a hand-written harness; it is a faculty of the living substrate (Sébastien's original
//! openmorph thesis). Domain-blind — no perception; a modality can bank a skill vocabulary over the same dream.
//!
//!   cargo test -p openmorph-core --test proofs substrate_dream -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;

const K: usize = 5; // states in the cycle 0→1→…→K-1→0
const H: usize = 3; // horizon (never experienced directly)

fn onehot(s: usize) -> Vec<f32> {
    let mut v = vec![0.0; K];
    v[s] = 1.0;
    v
}

/// Cells: state `0..K`, adjacent-prediction `K..2K`, long-range head `2K..3K`, plus a downstream READER cell `3K`
/// over the model+heads — so the memory is RELIED UPON and the sleep heartbeat CONSOLIDATES it rather than culling
/// it as an unused leaf (the realistic case: a model is used downstream).
fn build() -> Field {
    let mut f = Field::new(3 * K + 1);
    for j in 0..K {
        f.add_unit((0..K).collect(), vec![0.0; K], 0.0, K + j); // adjacent model
        f.add_unit((0..K).collect(), vec![0.0; K], 0.0, 2 * K + j); // long-range head
    }
    let reads: Vec<usize> = (K..3 * K).collect();
    let n = reads.len();
    f.add_unit(reads, vec![1.0; n], 0.0, 3 * K); // downstream reliance
    f
}

/// Train ONLY adjacent transitions online (never a multi-step rollout).
fn wake(f: &mut Field) {
    for _ in 0..400 {
        for s in 0..K {
            let inputs: Vec<(usize, f32)> = (0..K).zip(onehot(s)).collect();
            let next = onehot((s + 1) % K);
            for (j, &target) in next.iter().enumerate() {
                f.learn_step(&inputs, K + j, target, 0.2, 200);
            }
        }
    }
}

fn longrange_accuracy(f: &mut Field) -> usize {
    (0..K)
        .filter(|&s| {
            let inputs: Vec<(usize, f32)> = (0..K).zip(onehot(s)).collect();
            f.stimulate(&inputs, 200);
            let head: Vec<f32> = (2 * K..3 * K).map(|c| f.cells()[c]).collect();
            (0..K).max_by(|&a, &b| head[a].total_cmp(&head[b])).unwrap() == (s + H) % K
        })
        .count()
}

#[test]
fn the_substrate_dreams_its_own_discovery() {
    let state: Vec<usize> = (0..K).collect();
    let pred: Vec<usize> = (K..2 * K).collect();
    let longrange: Vec<usize> = (2 * K..3 * K).collect();
    let seeds: Vec<Vec<f32>> = (0..K).map(onehot).collect();

    // REACTIVE control: same wake, then just idle (only the heartbeat, no dreaming).
    let mut reactive = build();
    wake(&mut reactive);
    for _ in 0..8 {
        reactive.live(400);
    }
    let react_acc = longrange_accuracy(&mut reactive);

    // THE DREAM FACULTY: same wake, then dream — think → consolidate → sleep, autonomously, in core.
    let mut dreamer = build();
    wake(&mut dreamer);
    let hb = dreamer.dream(&state, &pred, &longrange, &seeds, H, 0.2, 40, 200);
    let dream_acc = longrange_accuracy(&mut dreamer);

    eprintln!("[DREAM] cycle of {K} states, horizon H={H} (only adjacent steps seen awake)");
    eprintln!(
        "  REACTIVE (heartbeat only, no dreaming): long-range accuracy {react_acc}/{K}   (chance {:.2})",
        1.0 / K as f32
    );
    eprintln!(
        "  DREAM (think→consolidate→sleep, in core): long-range accuracy {dream_acc}/{K}  |  heartbeat ticks {} reactivations {} consolidated {} rested {}",
        hb.ticks, hb.reactivations, hb.consolidated, hb.slept
    );
    eprintln!("[DREAM] VERDICT: the discovery loop is a SUBSTRATE FACULTY — `Field::dream` lets the organism, with no external input, THINK its fragmentary experience forward (free-run its model), CONSOLIDATE the simulated consequence into direct long-range knowledge it NEVER experienced, and SLEEP (the maintenance heartbeat consolidates the relied-upon memory and RESTS). A reactive organism idle in the same window gains nothing. The discovery loop is the organism's own resting dynamics — in core, no perception, no authored harness.");

    assert!(
        dream_acc >= K - 1,
        "the dreaming organism must discover the H-step structure autonomously ({dream_acc}/{K})"
    );
    assert!(
        dream_acc > react_acc,
        "dreaming (think+consolidate+sleep) must beat idle-reacting ({dream_acc} vs {react_acc})"
    );
    assert!(
        hb.slept,
        "the organism must reach rest (the heartbeat's emergent stop)"
    );
    assert!(
        hb.consolidated > 0,
        "the sleep must consolidate the relied-upon memory"
    );
}
