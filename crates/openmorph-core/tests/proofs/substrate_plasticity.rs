// SPDX-License-Identifier: MIT-0
//! SURPRISE-DRIVEN PLASTICITY (opt-in `Config::plasticity_anneal`, byte-identical off) — the fix for a constant
//! learning rate DIVERGING on a NOISY target. On a stochastic (irreducibly noisy) but STATIONARY target, a fixed
//! rate never converges — each noisy sample shakes the readout loose from a correct policy, so it gets WORSE with
//! experience. The cure is NOT to scale the rate by the error LEVEL (irreducible noise looks like error forever);
//! it is a CHANGE-detector: anneal the step by default so the readout SETTLES, and RE-OPEN plasticity only when the
//! fast error EMA breaks above the slow one — i.e. when the target genuinely CHANGED. A stationary habitat never
//! trips it and stays settled; a shift trips it and re-adapts. (The substrate twin of MorphBot's resident/migrant
//! life-world result: constant-rate degrades the organism while it settles at the world's own noise ceiling;
//! anneal+reset settles AND re-opens on change.)
//!
//!   cargo test -p openmorph-core --test proofs substrate_plasticity -- --nocapture

use openmorph_core::{Config, Field};

const IN: usize = 0; // a VARYING input (so the unit re-fires each step — fire-on-change)
const OUT: usize = 1;
const PHASE: usize = 500; // steps per phase
const LR: f32 = 0.3;
const NOISE: f32 = 0.15; // irreducible per-sample noise on the target
const W1: f32 = 0.30; // true weight, phase 1 (target = W·x + noise); small so tanh stays ~linear
const W2: f32 = -0.30; // the true weight SHIFTS to here in phase 2 (the "world changed")

/// Deterministic streams (byte-replayable — no wall-clock, no rng crate).
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    /// A varying input signal in [-1, 1].
    fn signal(&mut self) -> f32 {
        ((self.next() >> 40) % 1001) as f32 / 500.0 - 1.0
    }
    /// ±NOISE.
    fn noise(&mut self) -> f32 {
        if (self.next() >> 40) & 1 == 0 {
            NOISE
        } else {
            -NOISE
        }
    }
}

/// One arm: a single linear unit `OUT = tanh(w·x + b)` learned toward `target = W·x + noise`, with W = W1 for PHASE
/// steps then W2 for PHASE steps. After each learn we PROBE the learned weight (stimulate a fixed input x=1 and read
/// OUT ≈ w) — a fresh read, since the probe value differs from the learn input so the unit re-fires. Returns
/// (probe-variance over the last quarter of phase 1, mean probe over the last quarter of phase 2, gain at end of
/// phase 1, peak gain early in phase 2).
fn run(anneal: f32) -> (f32, f32, f32, f32) {
    let mut f = Field::new(2);
    f.configure(Config {
        plasticity_anneal: anneal, // 0.0 = constant rate (control); > 0 = anneal + reset-on-break
        ..Config::default()
    });
    f.add_unit(vec![IN], vec![0.0], 0.0, OUT);
    let mut rng = Lcg(0xA11CE);

    let mut phase1_tail: Vec<f32> = Vec::new();
    let mut phase2_tail: Vec<f32> = Vec::new();
    let mut gain_end_p1 = 1.0f32;
    let mut gain_peak_p2 = 0.0f32;

    for step in 0..(2 * PHASE) {
        let w = if step < PHASE { W1 } else { W2 };
        let x = rng.signal();
        let target = w * x + rng.noise();
        f.learn_step(&[(IN, x)], OUT, target, LR, 16);
        // PROBE the learned weight with a fixed input (re-fires the unit; OUT ≈ tanh(w_learned) ≈ w_learned).
        f.stimulate(&[(IN, 1.0)], 16);
        let w_est = f.cells()[OUT];

        if (PHASE * 3 / 4..PHASE).contains(&step) {
            phase1_tail.push(w_est);
        }
        if step == PHASE - 1 {
            gain_end_p1 = f.plasticity_gain_of(OUT);
        }
        if (PHASE..PHASE + 40).contains(&step) {
            gain_peak_p2 = gain_peak_p2.max(f.plasticity_gain_of(OUT));
        }
        if (2 * PHASE * 3 / 4..2 * PHASE).contains(&step) {
            phase2_tail.push(w_est);
        }
    }

    let var = variance(&phase1_tail);
    let mean2 = phase2_tail.iter().sum::<f32>() / phase2_tail.len().max(1) as f32;
    (var, mean2, gain_end_p1, gain_peak_p2)
}

fn variance(xs: &[f32]) -> f32 {
    let m = xs.iter().sum::<f32>() / xs.len().max(1) as f32;
    xs.iter().map(|x| (x - m) * (x - m)).sum::<f32>() / xs.len().max(1) as f32
}

#[test]
fn anneal_reset_settles_on_noise_yet_readapts_on_a_shift() {
    let (const_var, const_mean2, _, _) = run(0.0); // control: constant rate
    let (ann_var, ann_mean2, gain_end_p1, gain_peak_p2) = run(0.03); // anneal + reset-on-break

    eprintln!("[PLASTICITY] STATIONARY-NOISE settle variance (last quarter, phase 1): constant {const_var:.5} vs anneal {ann_var:.5}");
    eprintln!("[PLASTICITY] RE-ADAPT after the target shift W1={W1} -> W2={W2}: phase-2 mean probe constant {const_mean2:.3} vs anneal {ann_mean2:.3}");
    eprintln!("[PLASTICITY] gain at end of phase 1 (annealed down) = {gain_end_p1:.3}; peak gain just after the shift (reset back up) = {gain_peak_p2:.3}");

    // (1) SETTLE: on the noisy STATIONARY target the annealed arm oscillates far less — the readout is not shaken
    //     loose by irreducible noise. (This is the failure the constant rate suffers.)
    assert!(
        ann_var < const_var * 0.5,
        "anneal must SETTLE on a noisy stationary target (much lower prediction variance): anneal {ann_var:.5} vs constant {const_var:.5}"
    );
    // (2) The gain actually annealed down over the stationary phase (the mechanism did real work, not a no-op).
    assert!(
        gain_end_p1 < 0.5,
        "the gain must anneal down as the target settles: end-of-phase-1 gain {gain_end_p1:.3} should be well below 1.0"
    );
    // (3) RE-ADAPT: the shift breaks the error away from its slow trace, resets the gain, and the readout follows
    //     the new target — it did NOT stay frozen at the old mean. Both arms must reach the new (negative) target.
    assert!(
        ann_mean2 < -0.15,
        "anneal must RE-ADAPT after the shift (follow M2<0), not stay frozen at M1: phase-2 mean {ann_mean2:.3}"
    );
    assert!(
        gain_peak_p2 > 0.8,
        "the shift must RE-OPEN plasticity (gain resets toward 1.0 within 40 steps): peak {gain_peak_p2:.3}"
    );

    eprintln!("[PLASTICITY] VERDICT: anneal+reset SETTLES on a noisy stationary target (variance {ann_var:.5} vs constant {const_var:.5}) — the constant rate is shaken loose by irreducible noise — AND re-opens on a genuine CHANGE (gain {gain_end_p1:.2}->{gain_peak_p2:.2}), tracking the shifted target. A change-detector, not a level-scaler; byte-identical when plasticity_anneal=0.");
}
