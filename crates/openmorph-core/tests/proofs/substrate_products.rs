// SPDX-License-Identifier: MIT-0
//! GROWN PRODUCT DISCOVERY (opt-in `Config::grow_products`, byte-identical off) — a `Field` readout is
//! `act(Σw·x + b)`; it cannot represent a PRODUCT of two live cells however wide its random-projection basis is
//! grown. That is the "this matters only when that is true" interaction (e.g. value food ONLY when hungry). The
//! `Unit::factor` mechanism (`out = cells[j]·f(x_i)`) can, but you had to AUTHOR which pair to multiply. This
//! discovers it: `grow` accrues the pairwise product-synergy `co_credit` (EMA of `err·xᵢ·xⱼ`) and, when a pair
//! dominates, births a PRODUCT feature on it instead of a random projection — the interaction is GROWN, not
//! authored into a hand-built state band.
//!
//! Scored INSIDE, truth-not-trophy: HELD-OUT MSE on `target = 0.5·x₀·x₁` (only the (0,1) pair interacts; 2,3 are
//! distractors). Same grow budget both arms — the ONLY difference is products-vs-random features — so the gap is
//! the product mechanism, not extra capacity.
//!
//!   cargo test -p openmorph-core --test proofs substrate_products -- --nocapture

use openmorph_core::{Config, Field};

const OUT: usize = 4; // cells 0..3 = inputs, 4 = output
const TRAIN: usize = 4000;
const TEST: usize = 400;
const LR: f32 = 0.1;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    /// A value in [-1, 1].
    fn sig(&mut self) -> f32 {
        ((self.next() >> 40) % 1001) as f32 / 500.0 - 1.0
    }
}

/// target = 0.5·x₀·x₁ — an interaction (x₀ matters only in proportion to x₁), kept inside the output tanh's range.
/// x₂,x₃ are distractors that do not interact.
fn target(x: [f32; 4]) -> f32 {
    0.5 * x[0] * x[1]
}

fn run(grow_products: bool) -> f32 {
    let mut f = Field::new(5);
    f.configure(Config {
        grow_products,
        lr_fanin_norm: true, // normalize the step by input energy so a wide readout is stable (both arms)
        max_hidden: 8, // a SCARCE basis — so capturing the interaction efficiently (a product) matters
        ..Config::default()
    });
    let mut rng = Lcg(0x50DA);
    for _ in 0..TRAIN {
        let x = [rng.sig(), rng.sig(), rng.sig(), rng.sig()];
        let inputs = [(0, x[0]), (1, x[1]), (2, x[2]), (3, x[3])];
        f.grow(&inputs, OUT, target(x), LR, 24);
    }
    // HELD-OUT MSE on fresh samples (learning off — just stimulate + read the prediction).
    let mut rng2 = Lcg(0xBEEF);
    let mut mse = 0.0f32;
    for _ in 0..TEST {
        let x = [rng2.sig(), rng2.sig(), rng2.sig(), rng2.sig()];
        f.stimulate(&[(0, x[0]), (1, x[1]), (2, x[2]), (3, x[3])], 24);
        let pred = f.cells()[OUT];
        let e = pred - target(x);
        mse += e * e;
    }
    mse / TEST as f32
}

#[test]
fn grown_product_discovery_beats_a_random_basis_on_an_interaction() {
    let off_mse = run(false); // random-projection features only (the status quo)
    let on_mse = run(true); // discover + birth the product feature

    eprintln!("[PRODUCTS] held-out MSE on 0.5·x₀·x₁ — random basis (off) {off_mse:.4}  vs  grown products (on) {on_mse:.4}");

    // A grown product feature captures the interaction a random affine basis cannot: much lower held-out error at
    // the SAME grow budget (the difference is products-vs-random, not capacity).
    assert!(
        on_mse < off_mse * 0.5,
        "grown product discovery must fit the interaction much better than a random basis: on {on_mse:.4} vs off {off_mse:.4}"
    );
    // And it's a real fit, not just relatively better: the variance of 0.5·x₀·x₁ over [-1,1]² is
    // 0.25·E[x₀²]E[x₁²] = 0.25/9 ≈ 0.0278 (the MSE a zero predictor would get); products must beat that comfortably.
    assert!(
        on_mse < 0.010,
        "grown products must actually fit the product (MSE ≪ the 0.028 zero-predictor variance): {on_mse:.4}"
    );

    eprintln!("[PRODUCTS] VERDICT: `grow` DISCOVERS the interacting pair from its own product-synergy (co_credit) and births a grown `Unit::factor` xⱼ·f(xᵢ) on it, fitting 0.5·x₀·x₁ (held-out MSE {on_mse:.4}) where a random-projection basis of the same size cannot ({off_mse:.4}). The 'matters only when' interaction is grown, not authored; byte-identical when grow_products=off.");
}

/// PER-TARGET, ACTIVE-PAIR discovery (opt-in `Config::products_per_target`). Several targets share one input set of
/// SPARSE one-hot cells (4 action cells, then 4 positions × 4 colours), each target the product of one action cell
/// and one colour cell: "this action AND that neighbour", the interaction a world model over one-hot cells needs.
/// Targets 0 and 1 use the SAME pair with OPPOSITE signs, so the shared synergy map (keyed by the pair alone) sums
/// their evidence to nothing.
///
/// Guards for the mode's state (the per-target co-credit, an ordered map with lazy decay): the held-out MSE is pinned
/// to its bits at `c271fded` (1.7.12, before the map became ordered), and a field snapshotted mid-stream and loaded
/// continues bit for bit.
mod per_target {
    use super::Lcg;
    use openmorph_core::{Config, Field};

    const ACTIONS: usize = 4;
    const IN: usize = ACTIONS + 16;
    /// Per target: (action cell, colour cell, sign).
    const PAIRS: [(usize, usize, f32); 4] = [(0, 4, 0.8), (0, 4, -0.8), (1, 9, 0.8), (2, 14, 0.8)];

    fn sample(rng: &mut Lcg) -> Vec<(usize, f32)> {
        let a = (rng.next() >> 33) as usize % ACTIONS;
        let mut x = vec![(a, 1.0)];
        for p in 0..4 {
            x.push((ACTIONS + p * 4 + (rng.next() >> 33) as usize % 4, 1.0));
        }
        // every input cell supplied (the rest at zero), as a world model supplies its whole patch
        let mut all: Vec<(usize, f32)> = (0..IN).map(|c| (c, 0.0)).collect();
        for (c, v) in x {
            all[c].1 = v;
        }
        all
    }

    fn target(x: &[(usize, f32)], t: usize) -> f32 {
        let (a, c, s) = PAIRS[t];
        s * x[a].1 * x[c].1
    }

    /// The trained events of the stream.
    pub(super) const TRAIN: usize = 3000;

    /// A newborn field; `mode` 0 = random features, 1 = shared products, 2 = per-target.
    pub(super) fn field(mode: u8) -> Field {
        let mut f = Field::new(IN + PAIRS.len());
        f.configure(Config {
            grow_products: mode > 0,
            products_per_target: mode == 2,
            lr_fanin_norm: true,
            max_hidden: 4,
            ..Config::default()
        });
        f
    }

    /// The training stream (one sample, then a `grow` per target), from its start.
    pub(super) fn stream() -> Lcg {
        Lcg(0x7A57)
    }

    /// `n` events of the stream: the bits of every `grow`'s error.
    pub(super) fn train(f: &mut Field, rng: &mut Lcg, n: usize) -> Vec<u32> {
        let mut out = Vec::new();
        for _ in 0..n {
            let x = sample(rng);
            for t in 0..PAIRS.len() {
                out.push(f.grow(&x, IN + t, target(&x, t), 0.3, 200).0.to_bits());
            }
        }
        out
    }

    /// Held-out MSE over the four targets after the whole stream.
    pub(super) fn run(mode: u8) -> f32 {
        let mut f = field(mode);
        train(&mut f, &mut stream(), TRAIN);
        held_out(&mut f)
    }

    /// Held-out MSE over the four targets (learning off).
    pub(super) fn held_out(f: &mut Field) -> f32 {
        let mut rng2 = Lcg(0xC0DE);
        let mut mse = 0.0f32;
        let n = 400;
        for _ in 0..n {
            let x = sample(&mut rng2);
            f.stimulate(&x, 200);
            for t in 0..PAIRS.len() {
                let e = f.cells()[IN + t] - target(&x, t);
                mse += e * e;
            }
        }
        mse / (n * PAIRS.len()) as f32
    }
}

#[test]
fn per_target_discovery_finds_each_targets_own_interaction() {
    let random = per_target::run(0);
    let shared = per_target::run(1);
    let own = per_target::run(2);
    eprintln!(
        "[PRODUCTS] four action×colour targets over sparse one-hot cells, held-out MSE: random {random:.4}, shared \
         products {shared:.4}, per-target products {own:.4}"
    );
    assert!(
        own < shared * 0.5,
        "per-target {own:.4} vs shared {shared:.4}"
    );
    assert!(
        own < random * 0.5,
        "per-target {own:.4} vs random {random:.4}"
    );
    assert!(
        own < 0.002,
        "per-target products must fit the interactions: {own:.4}"
    );
    // the same bits as at c271fded (1.7.12)
    assert_eq!(
        own.to_bits(),
        OWN_AT_1_7_12,
        "per-target held-out MSE {own:e}"
    );
}

/// The per-target arm's held-out MSE bits, recorded at `c271fded` (1.7.12, before the co-credit became an ordered map).
const OWN_AT_1_7_12: u32 = 0x36784697; // 3.6995968e-6

#[test]
fn per_target_state_continues_through_a_snapshot() {
    let mut a = per_target::field(2);
    let mut rng = per_target::stream();
    per_target::train(&mut a, &mut rng, per_target::TRAIN / 2);
    let mut b = Field::load(&a.snapshot()).expect("a per-target field loads");
    let mut rng_b = Lcg(rng.0);
    let rest = per_target::TRAIN - per_target::TRAIN / 2;
    assert_eq!(
        per_target::train(&mut a, &mut rng, rest),
        per_target::train(&mut b, &mut rng_b, rest),
        "every grow's error after the load"
    );
    assert_eq!(a.snapshot(), b.snapshot());
    let (mse_a, mse_b) = (per_target::held_out(&mut a), per_target::held_out(&mut b));
    assert_eq!(mse_a.to_bits(), mse_b.to_bits());
    assert_eq!(mse_a.to_bits(), OWN_AT_1_7_12);
}
