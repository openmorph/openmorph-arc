// SPDX-License-Identifier: MIT-0
//! SECOND-ORDER CO-CREDIT — the pairwise sibling of the per-unit delta rule, and the map-shaper that makes a
//! downstream `cluster` recover SYNERGY modules where `co_activate` cannot.
//!
//! The kill-test (measurement before belief, per MorphBot's finding). Six input bits form two synergy blocks
//! A={0,1,2}, B={3,4,5}; the target is `AND(A) XOR AND(B)` over ±1 encodings — a task whose structure lives in
//! WITHIN-block PRODUCTS (an AND is a product), not in co-occurrence (an AND over ±1 bits is +1 for 1/8 of
//! inputs regardless of which block it is in, so block-mates do not co-fire more than cross-block pairs —
//! MorphBot measured co-firing separation ~0.004). We shape the concept-map of the six input cells two ways
//! and cluster each into two groups. `co_activate` (Hebbian co-occurrence) is the baseline that should FAIL to
//! recover the blocks; `accrue_co_credit` + `co_credit_step` (synergy `E[err·x_i·x_j]`) should recover them. If
//! co-credit does not separate the blocks, it is a clean negative and the shaper drops out (truth-not-trophy).

use openmorph_core::Field;

/// A tiny deterministic LCG so the whole probe is byte-replayable (no wall-clock, no OS RNG).
struct Lcg(u64);
impl Lcg {
    fn next_bit(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        if (self.0 >> 33) & 1 == 0 {
            -1.0
        } else {
            1.0
        }
    }
}

/// AND over ±1 bits = +1 iff all three are +1, else −1.
fn and3(a: f32, b: f32, c: f32) -> f32 {
    if a > 0.0 && b > 0.0 && c > 0.0 {
        1.0
    } else {
        -1.0
    }
}

/// XOR over the two ±1 block-ANDs (= their negated product for ±1: equal ⇒ −1, differ ⇒ +1).
fn target(bits: &[f32; 6]) -> f32 {
    let a = and3(bits[0], bits[1], bits[2]);
    let b = and3(bits[3], bits[4], bits[5]);
    -a * b
}

/// Best block-recovery score of a 2-clustering of {0..6}: fraction of the six cells placed on the correct side
/// of the A={0,1,2} / B={3,4,5} split, maximised over the two label assignments (a clustering has no inherent
/// A/B labels). 1.0 = perfect recovery, 0.5 = chance.
fn block_recovery(clusters: &[Vec<usize>]) -> f32 {
    // Map each cell to its cluster id.
    let mut cid = [usize::MAX; 6];
    for (k, cl) in clusters.iter().enumerate() {
        for &c in cl {
            if c < 6 {
                cid[c] = k;
            }
        }
    }
    // The true block of each cell.
    let truth = [0usize, 0, 0, 1, 1, 1];
    // Try both label maps (cluster0→A or cluster0→B); score the better.
    let mut best = 0.0f32;
    for flip in [false, true] {
        let mut correct = 0;
        for c in 0..6 {
            if cid[c] == usize::MAX {
                continue;
            }
            let pred = if flip {
                1 - cid[c].min(1)
            } else {
                cid[c].min(1)
            };
            if pred == truth[c] {
                correct += 1;
            }
        }
        best = best.max(correct as f32 / 6.0);
    }
    best
}

/// Train the model once, returning the mean block-recovery of the co_activate, group_by_surprise, and co_credit
/// maps, averaged over `seeds` independent runs.
fn run(seeds: u64) -> (f32, f32, f32) {
    let inputs_cells = [0usize, 1, 2, 3, 4, 5];
    let out = 6usize;
    let mut ca_sum = 0.0f32;
    let mut gs_sum = 0.0f32;
    let mut cc_sum = 0.0f32;

    for s in 0..seeds {
        // ---- co_activate arm ----
        let mut f = Field::new(0);
        let mut rng = Lcg(0x1234_5678 ^ s.wrapping_mul(0x9E37_79B9));
        for _ in 0..4000 {
            let bits: [f32; 6] = std::array::from_fn(|_| rng.next_bit());
            let t = target(&bits);
            let inp: Vec<(usize, f32)> = inputs_cells.iter().map(|&c| (c, bits[c])).collect();
            f.grow(&inp, out, t, 0.05, 64);
            // Hebbian: pull the co-active (=+1) inputs together.
            let active: Vec<usize> = inputs_cells
                .iter()
                .copied()
                .filter(|&c| bits[c] > 0.0)
                .collect();
            f.co_activate(&active, 0.05);
        }
        let ca_clusters = f.cluster(&inputs_cells, 2);
        ca_sum += block_recovery(&ca_clusters);

        // ---- group_by_surprise arm (the existing causal shaper; min(SR_i,SR_j) over per-cell importance) ----
        let mut h = Field::new(0);
        h.config_mut().surprise_currency = true; // warms cell_var + cell_sens2 during grow
        let mut rng = Lcg(0x1234_5678 ^ s.wrapping_mul(0x9E37_79B9));
        for _ in 0..4000 {
            let bits: [f32; 6] = std::array::from_fn(|_| rng.next_bit());
            let t = target(&bits);
            let inp: Vec<(usize, f32)> = inputs_cells.iter().map(|&c| (c, bits[c])).collect();
            h.grow(&inp, out, t, 0.05, 64);
            h.group_by_surprise(&inputs_cells, 0.5); // pulls pairs by min(SR_i, SR_j)
        }
        let gs_clusters = h.cluster(&inputs_cells, 2);
        gs_sum += block_recovery(&gs_clusters);

        // ---- co_credit arm (identical training seed) ----
        let mut g = Field::new(0);
        let mut rng = Lcg(0x1234_5678 ^ s.wrapping_mul(0x9E37_79B9));
        for _ in 0..4000 {
            let bits: [f32; 6] = std::array::from_fn(|_| rng.next_bit());
            let t = target(&bits);
            let inp: Vec<(usize, f32)> = inputs_cells.iter().map(|&c| (c, bits[c])).collect();
            let (err, _) = g.grow(&inp, out, t, 0.05, 64);
            // Second-order: accrue synergy err·x_i·x_j over all six inputs. A slow EMA (long window) — the
            // synergy signal averages cleanly over training; a fast decay tracks only the last ~100 noisy steps.
            g.accrue_co_credit(&inputs_cells, err, 0.9999);
        }
        // Shape the map from the accrued synergy (anneal a few passes).
        for _ in 0..200 {
            g.co_credit_step(0.1);
        }
        let cc_clusters = g.cluster(&inputs_cells, 2);
        cc_sum += block_recovery(&cc_clusters);
    }
    (
        ca_sum / seeds as f32,
        gs_sum / seeds as f32,
        cc_sum / seeds as f32,
    )
}

#[test]
fn synergy_signal_is_stronger_within_block_than_across() {
    // The SIGNAL check, upstream of the shaper/clusterer: the accrued synergy `E[err·x_i·x_j]` must be larger
    // WITHIN a block (whose bits interact via the block's AND) than ACROSS blocks (which do not), and a slower
    // EMA (longer window) sharpens the ratio. Compares err-based accrual against a raw-target reference.
    let cells = [0usize, 1, 2, 3, 4, 5];
    let mut best_err_ratio = 0.0f32;
    for decay in [0.99f32, 0.999, 0.9999] {
        let mut g = Field::new(0);
        let mut gt = Field::new(0); // target-based reference
        let mut rng = Lcg(0xABCD);
        for _ in 0..4000 {
            let bits: [f32; 6] = std::array::from_fn(|_| rng.next_bit());
            let t = target(&bits);
            let inp: Vec<(usize, f32)> = cells.iter().map(|&c| (c, bits[c])).collect();
            let (err, _) = g.grow(&inp, 6, t, 0.05, 64);
            g.accrue_co_credit(&cells, err, decay);
            gt.grow(&inp, 6, t, 0.05, 64);
            gt.accrue_co_credit(&cells, t, decay); // accrue against the raw target
        }
        let same_block = |a: usize, b: usize| (a < 3) == (b < 3);
        let mut win = (0.0, 0);
        let mut cross = (0.0, 0);
        let mut wint = (0.0, 0);
        let mut crosst = (0.0, 0);
        for a in 0..6 {
            for b in (a + 1)..6 {
                let ce = g.co_credit_of(a, b).abs();
                let ct = gt.co_credit_of(a, b).abs();
                if same_block(a, b) {
                    win.0 += ce;
                    win.1 += 1;
                    wint.0 += ct;
                    wint.1 += 1;
                } else {
                    cross.0 += ce;
                    cross.1 += 1;
                    crosst.0 += ct;
                    crosst.1 += 1;
                }
            }
        }
        let err_ratio = (win.0 / win.1 as f32) / (cross.0 / cross.1 as f32).max(1e-6);
        best_err_ratio = best_err_ratio.max(err_ratio);
        println!(
            "decay={decay}  ERR within={:.4} cross={:.4} ratio={:.2}   TARGET within={:.4} cross={:.4} ratio={:.2}",
            win.0 / win.1 as f32,
            cross.0 / cross.1 as f32,
            err_ratio,
            wint.0 / wint.1 as f32,
            crosst.0 / crosst.1 as f32,
            (wint.0 / wint.1 as f32) / (crosst.0 / crosst.1 as f32).max(1e-6),
        );
    }
    // A slow EMA must lift the within/cross synergy ratio well clear of 1 (the additive-pair baseline).
    assert!(
        best_err_ratio > 2.0,
        "within-block synergy should clearly exceed cross-block (best ratio {best_err_ratio:.2})"
    );
}

#[test]
fn co_credit_separates_synergy_modules_where_co_activation_cannot() {
    let (co_activate, group_by_surprise, co_credit) = run(8);
    println!(
        "block-recovery  co_activate={co_activate:.3}  group_by_surprise={group_by_surprise:.3}  co_credit={co_credit:.3}"
    );
    // The claim under test: co-credit (a true SECOND-ORDER product term) recovers the synergy blocks. Both
    // co_activate (co-occurrence) and group_by_surprise (per-cell min(SR), which cannot express pairwise
    // interaction when every cell is individually load-bearing) are near chance.
    assert!(
        co_credit >= 0.85,
        "co_credit should recover the synergy blocks (got {co_credit:.3})"
    );
    assert!(
        co_credit > co_activate + 0.15,
        "co_credit ({co_credit:.3}) must clearly beat co_activate ({co_activate:.3})"
    );
    assert!(
        co_credit > group_by_surprise + 0.15,
        "co_credit ({co_credit:.3}) must beat group_by_surprise ({group_by_surprise:.3}) — the second-order \
         product term expresses synergy that per-cell min(SR) cannot"
    );
}

#[test]
fn co_credit_step_is_byte_identical_when_never_accrued() {
    // A field that never accrues co-credit must be positionally unchanged by co_credit_step (byte-identical off).
    let mut f = Field::new(0);
    for _ in 0..50 {
        f.grow(&[(0, 1.0), (1, -1.0), (2, 1.0)], 3, 0.5, 0.05, 32);
    }
    f.co_activate(&[0, 1, 2], 0.1); // seed some positions
    let before: Vec<f32> = (0..4).map(|c| f.position(c)).collect::<Vec<_>>().concat();
    f.co_credit_step(0.5); // no co-credit accrued → must be a no-op
    let after: Vec<f32> = (0..4).map(|c| f.position(c)).collect::<Vec<_>>().concat();
    assert_eq!(
        before, after,
        "co_credit_step must be a no-op when nothing was accrued"
    );
}
