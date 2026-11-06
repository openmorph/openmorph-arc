// SPDX-License-Identifier: MIT-0
//! WHEN TO DREAM — the discovery loop coupled to the ENERGY ECONOMY. Dreaming ([`Field::dream`], internal thinking)
//! is a LUXURY of the fed: with `Config::dream_satiation` set, an organism turns inward to process its experience
//! ONLY once its metabolism is above the floor, and each dream bout BURNS energy so dreaming spends the reserve it
//! needs and SELF-LIMITS — a homeostatic wake/forage/dream cycle, nothing scheduled. A SATED organism dreams and
//! discovers the structure it never experienced; a HUNGRY one conserves (forages) and gains nothing. The default
//! floor 0.0 leaves dreaming ungated and byte-identical (asserted by the pre-existing `substrate_dream`).
//!
//!   cargo test -p openmorph-core --test proofs substrate_dream_economy -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;

const K: usize = 5; // states in the cycle 0→1→…→K-1→0
const H: usize = 3; // horizon (never experienced directly)

fn onehot(s: usize) -> Vec<f32> {
    let mut v = vec![0.0; K];
    v[s] = 1.0;
    v
}

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
fn the_organism_dreams_only_when_sated() {
    let state: Vec<usize> = (0..K).collect();
    let pred: Vec<usize> = (K..2 * K).collect();
    let longrange: Vec<usize> = (2 * K..3 * K).collect();
    let seeds: Vec<Vec<f32>> = (0..K).map(onehot).collect();

    // Set a dream floor at half-full: the organism must be over 50% sated to turn inward.
    let floor = 0.5;

    // SATED organism: born fed (metabolism = energy_cap), well above the floor → it dreams and discovers.
    let mut fed = build();
    {
        let mut cfg = fed.config().clone();
        cfg.dream_satiation = floor;
        fed.configure(cfg);
    }
    wake(&mut fed);
    let sat_before = fed.satiation();
    assert!(
        fed.should_dream(),
        "a fed organism should be willing to dream"
    );
    let hb_fed = fed.dream(&state, &pred, &longrange, &seeds, H, 0.2, 40, 200);
    let fed_acc = longrange_accuracy(&mut fed);
    let sat_after = fed.satiation();

    // HUNGRY organism: same wake + floor, but drained below the floor first (the maintenance heartbeat burns
    // metabolic energy each tick — being awake without feeding starves it). It should CONSERVE, not dream.
    let mut hungry = build();
    {
        let mut cfg = hungry.config().clone();
        cfg.dream_satiation = floor;
        hungry.configure(cfg);
    }
    wake(&mut hungry);
    while hungry.satiation() >= floor {
        hungry.live(50); // flow time / burn the reserve until hunger crosses the floor
    }
    assert!(
        !hungry.should_dream(),
        "a starved organism must refuse to dream (satiation {:.2} < floor {floor})",
        hungry.satiation()
    );
    let _hb_hungry = hungry.dream(&state, &pred, &longrange, &seeds, H, 0.2, 40, 200);
    let hungry_acc = longrange_accuracy(&mut hungry);

    eprintln!("[DREAM-ECONOMY] dream floor = {floor} of energy_cap");
    eprintln!(
        "  SATED  : satiation {sat_before:.2}→{sat_after:.2} (dreaming burned the reserve), long-range {fed_acc}/{K}  |  rested {}",
        hb_fed.slept
    );
    eprintln!(
        "  HUNGRY : satiation {:.2} < floor → conserves (no dreaming), long-range {hungry_acc}/{K}",
        hungry.satiation()
    );
    eprintln!("[DREAM-ECONOMY] VERDICT: WHEN to dream falls out of the energy economy — the fed organism turns inward and DISCOVERS the H-step structure it never experienced; each bout BURNS the reserve so dreaming self-limits (satiation falls); the hungry organism CONSERVES and gains nothing until it forages. Nothing scheduled — a homeostatic wake/forage/dream cycle.");

    assert!(
        fed_acc >= K - 1,
        "the sated dreamer must discover the H-step structure ({fed_acc}/{K})"
    );
    assert!(
        sat_after < sat_before,
        "dreaming must BURN the metabolic reserve (self-limiting): {sat_before:.2} → {sat_after:.2}"
    );
    assert!(
        fed_acc > hungry_acc,
        "the sated organism must out-discover the hungry one that refused to dream ({fed_acc} vs {hungry_acc})"
    );
}
