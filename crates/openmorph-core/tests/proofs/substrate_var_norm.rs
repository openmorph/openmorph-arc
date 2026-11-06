// SPDX-License-Identifier: MIT-0
//! VARIANCE-MASKED WIDTH-NORMALIZATION (`Config::var_gated_norm`) — the DENSE-BAND tax fix. Plain
//! width-normalization (`lr_fanin_norm`) divides the delta step by `Σx²` over ALL reads, so on a dense band an
//! input that never fires reads a CONSTANT background value (`−1`) and still charges `x² = 1`, taxing the
//! effective lr for the inputs that DO vary — for zero information (MorphBot measured whole 18-dim augmentation
//! blocks at EXACTLY zero variance over an episode). Variance-masking sums `x²` only over reads whose running
//! variance is non-zero, so a constant read drops out of the denominator automatically — an "effective fan-in"
//! that keeps NLMS's units. Variance is a per-cell EMA DETACHED from the weights, so unlike the reliance-weighted
//! denominator (which coupled the denominator to the live weights and DIVERGED on exactly this constant-input
//! case) it is stable, and it preserves the `−1` evidence (only the step-size denominator changes).
//!
//!   cargo test -p openmorph-core --test proofs substrate_var_norm -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;

struct Lcg(u64);
impl Lcg {
    fn bit(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        if (self.0 >> 40) & 1 == 0 {
            -1.0
        } else {
            1.0
        }
    }
}

const CORE: usize = 8;
const LR: f32 = 0.5;
const BUD: usize = 64;
const STEPS: usize = 500; // a FIXED budget: the unpadded learner converges, a taxed one has not yet

/// Train one readout over a `CORE + aug`-wide dense band toward a target depending on the CORE dims (plus the
/// aug dims iff `target_uses_aug`). An IGNORED aug dim reads a constant `−1` (never fires) — MorphBot's real
/// dense band. Returns held-out MAE. `var_gated` switches the normalization flavour; both use `lr_fanin_norm`.
fn run(aug: usize, var_gated: bool, target_uses_aug: bool, seed: u64) -> f32 {
    let w = CORE + aug;
    let mut f = Field::new(w);
    f.rng_seed(seed);
    f.config_mut().lr_fanin_norm = true;
    f.config_mut().var_gated_norm = var_gated;
    let out = f.len();
    f.add_unit((0..w).collect(), vec![0.0; w], 0.0, out);

    let mut rng = Lcg(seed ^ 0xA11);
    let tw: Vec<f32> = (0..w)
        .map(|k| {
            if k < CORE || target_uses_aug {
                if rng.bit() > 0.0 {
                    0.35
                } else {
                    -0.35
                }
            } else {
                0.0
            }
        })
        .collect();

    let mut data = Lcg(seed ^ 0x5EED);
    // dense-band sample: CORE (and target-carrying aug) dims fire ±1; an IGNORED aug dim reads constant −1.
    let sample = |data: &mut Lcg| -> Vec<f32> {
        (0..w)
            .map(|k| {
                if k < CORE || target_uses_aug {
                    data.bit()
                } else {
                    -1.0
                }
            })
            .collect()
    };
    let target = |x: &[f32]| -> f32 { (x.iter().zip(&tw).map(|(a, b)| a * b).sum::<f32>()).tanh() };
    for _ in 0..STEPS {
        let x = sample(&mut data);
        let inp: Vec<(usize, f32)> = x.iter().copied().enumerate().collect();
        f.learn_step(&inp, out, target(&x), LR, BUD);
    }
    let (mut mae, mut n) = (0.0f32, 0.0f32);
    for _ in 0..400 {
        let x = sample(&mut data);
        let inp: Vec<(usize, f32)> = x.iter().copied().enumerate().collect();
        f.stimulate(&inp, BUD);
        mae += (f.cells()[out] - target(&x)).abs();
        n += 1.0;
    }
    mae / n
}

#[test]
fn variance_masking_removes_the_dense_band_tax() {
    const AUG: usize = 8; // idle dims: constant −1 — a MODERATE fraction (denom ~1.9x), the regime the gym is in
    let seeds = [1u64, 2, 3, 4, 5, 6, 7, 8];
    let (mut unpadded, mut plain, mut gated) = (0.0f32, 0.0f32, 0.0f32);
    for &s in &seeds {
        unpadded += run(0, false, false, s); // CORE only — the ceiling
        plain += run(AUG, false, false, s); // padded, plain width-norm — the tax
        gated += run(AUG, true, false, s); // padded, variance-masked — tax removed + STABLE?
    }
    let k = seeds.len() as f32;
    let (unpadded, plain, gated) = (unpadded / k, plain / k, gated / k);
    eprintln!(
        "[VAR-NORM] held-out MAE (dense band, target ignores {AUG} CONSTANT idle dims):\n  unpadded ({CORE} dims, ceiling) : {unpadded:.4}\n  padded, PLAIN width-norm        : {plain:.4}  (the tax: {AUG} constant dims each charge x²=1)\n  padded, VARIANCE-MASKED         : {gated:.4}  (constant reads drop from the denominator)"
    );
    eprintln!("[VAR-NORM] VERDICT: masking Σx² to the reads that actually VARY removes the illegitimate constant-dim tax, recovering the unpadded learner — and, unlike the reliance-weighted denominator that DIVERGED here (~1.0), it is STABLE because the variance is detached from the weights. The −1 evidence is untouched (only the step-size denominator changed).");

    // the constant idle dims tax plain width-norm
    assert!(
        plain > unpadded * 1.5,
        "the constant idle dims must tax plain width-norm (plain {plain:.4} vs ceiling {unpadded:.4})"
    );
    // variance-masking recovers the ceiling and is STABLE (no divergence like reliance-gating's ~1.0)
    assert!(
        gated < 0.05 && gated < plain * 0.6,
        "variance-masking must recover the ceiling AND stay stable (gated {gated:.4}, plain {plain:.4}, ceiling {unpadded:.4})"
    );
}

#[test]
fn variance_masking_keeps_dims_that_vary() {
    // The control: when the extra dims VARY and carry the target, they have non-zero variance, so masking keeps
    // them and the readout still learns them — variance-masking gates by VARIATION, it does not zero idle dims
    // and it does not drop informative ones.
    const AUG: usize = 40;
    let seeds = [1u64, 2, 3, 4, 5, 6];
    let (mut plain, mut gated) = (0.0f32, 0.0f32);
    for &s in &seeds {
        plain += run(AUG, false, true, s);
        gated += run(AUG, true, true, s);
    }
    let k = seeds.len() as f32;
    let (plain, gated) = (plain / k, gated / k);
    eprintln!(
        "[VAR-NORM] target USES all {} VARYING dims (evidence-preservation control):\n  plain width-norm : {plain:.4}\n  variance-masked  : {gated:.4}  (dims vary ⇒ kept ⇒ still learned; must not blow up)",
        CORE + AUG
    );
    assert!(
        gated < plain * 1.3 && gated < 0.5,
        "when the dims vary and carry the target, variance-masking must still learn them (gated {gated:.4} vs plain {plain:.4})"
    );
}
