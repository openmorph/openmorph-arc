// SPDX-License-Identifier: MIT-0
//! v2.0-dev · GATE-4 follow-up (#149): `adapt_scales` on a HIDDEN timescale — the value MorphLab couldn't see
//! when the seeded trace bank already spanned the useful range. Here the target lives at a half-life the seed
//! MISSES (`H_TRUE=40`, trace seeded far off), so `adapt_scales` has something to learn: it migrates the trace's
//! half-life toward the true timescale and beats a frozen-wrong trace, approaching a correctly-seeded control.
//! (The substrate-side companion to `substrate_growtime`'s single-metronome result and MorphLab's WM harness.)
use openmorph_core::substrate::Field;

use crate::common::Rng;

const X: usize = 0;
const H_TRUE: f32 = 40.0; // the hidden timescale the target trend lives at
const START_WRONG: f32 = 3.0; // seed the trace far too FAST

fn corr(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len() as f32;
    let (ma, mb) = (a.iter().sum::<f32>() / n, b.iter().sum::<f32>() / n);
    let (mut cov, mut va, mut vb) = (0.0f32, 0.0f32, 0.0f32);
    for (x, y) in a.iter().zip(b) {
        cov += (x - ma) * (y - mb);
        va += (x - ma).powi(2);
        vb += (y - mb).powi(2);
    }
    if va <= 1e-9 || vb <= 1e-9 {
        0.0
    } else {
        cov / (va.sqrt() * vb.sqrt())
    }
}

fn learned_half_life(f: &Field, cell: usize) -> f32 {
    let v: serde_json::Value = serde_json::to_value(f).unwrap();
    v.get("traces")
        .and_then(|t| t.as_array())
        .and_then(|arr| {
            arr.iter()
                .find(|t| t.get("cell").and_then(|c| c.as_u64()) == Some(cell as u64))
                .and_then(|t| t.get("half_life").and_then(|h| h.as_f64()))
        })
        .map(|h| h as f32)
        .unwrap_or(0.0)
}

/// One trace seeded at `start_hl`; target = the EMA of the input at the hidden `H_TRUE`. Returns
/// (held-out corr with the true trend, learned half-life).
fn run(start_hl: f32, adapt: bool, seed: u64) -> (f32, f32) {
    let mut f = Field::new(1);
    f.config_mut().max_hidden = 0; // linear readout over [x, trace] — a clean one-hop backprojection
    let tr = f.integrate(X, start_hl);
    let pred = f.len();
    f.ensure_cells(pred + 1);
    let relax_true = 1.0 - 0.5f32.powf(1.0 / H_TRUE); // per-event relax of the TRUE-timescale target EMA

    let mut rng = Rng::new(seed);
    let mut ema = 0.0f32;
    for _ in 0..40000usize {
        let x = rng.sym();
        ema += relax_true * (x - ema); // the target: the slow trend the trace must match
        f.grow(&[(X, x), (tr, f.cells()[tr])], pred, ema, 0.05, 200);
        if adapt {
            f.adapt_scales(pred, ema, 1.0, 1.0, 400.0); // learn the timescale (bounded to [1, 400])
        }
    }

    let mut rng2 = Rng::new(seed ^ 0x5AA5);
    let mut ema2 = 0.0f32;
    let (mut preds, mut truth) = (Vec::new(), Vec::new());
    for t in 0..6000usize {
        let x = rng2.sym();
        ema2 += relax_true * (x - ema2);
        f.stimulate(&[(X, x), (tr, f.cells()[tr])], 200);
        if t > 300 {
            preds.push(f.cells()[pred]);
            truth.push(ema2);
        }
    }
    (corr(&preds, &truth), learned_half_life(&f, tr))
}

#[test]
fn adapt_scales_learns_a_hidden_timescale() {
    let (frozen, hl_frozen) = run(START_WRONG, false, 0x77);
    let (adapted, hl_adapted) = run(START_WRONG, true, 0x77);
    let (control, _) = run(H_TRUE, false, 0x77); // correctly-seeded upper bound
    eprintln!("[ADAPT] hidden timescale H_true={H_TRUE}, seed start={START_WRONG}:  frozen-wrong corr={frozen:.3} (hl {hl_frozen:.1})   ADAPTED corr={adapted:.3} (hl {hl_adapted:.1})   good-seed control corr={control:.3}");

    // the trace's half-life MIGRATES from the wrong seed toward the true timescale
    assert!(
        hl_adapted > 4.0 * START_WRONG && hl_adapted > hl_frozen + 5.0,
        "adapt_scales must migrate the half-life toward H_true={H_TRUE}: {START_WRONG}→{hl_adapted:.1} (frozen stayed {hl_frozen:.1})"
    );
    // and that migration IMPROVES the fit over the frozen-wrong trace, toward the good-seed control
    assert!(
        adapted > frozen + 0.15,
        "adapting the timescale must beat the frozen-wrong trace: {frozen:.3} → {adapted:.3}"
    );

    eprintln!("[ADAPT] VERDICT: adapt_scales earns its keep on a HIDDEN timescale — the target trend lives at a half-life the seed misses ({START_WRONG} vs {H_TRUE}), so a frozen trace can't match it (corr {frozen:.2}); the half-life gradient migrates the trace toward the true scale (hl {START_WRONG}→{hl_adapted:.0}) and recovers the fit (corr {adapted:.2}, toward the good-seed control {control:.2}). MorphLab's 'no gain when the bank already spans the range' was the null case; give it a scale to find and it finds it — the temporal twin of feature-direction learning (#137).");
}
