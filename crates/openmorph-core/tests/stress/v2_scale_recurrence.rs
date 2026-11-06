// SPDX-License-Identifier: MIT-0
//! STRESS TEST — axis: RECURRENCE / MEMORY CAPACITY / STABILITY.
//!
//! The async substrate settles by a Gauss-Seidel event cascade; recurrence (a unit reading its own write
//! cell, or a cycle of units) is where settling can FAIL. `spectral_radius` is the organism's own
//! contraction self-diagnostic: ρ of the local Jacobian J = D·W at the CURRENT state. Feedforward is
//! nilpotent (ρ=0); recurrence gives ρ>0; a bistable memory latch NEEDS ρ>1 at the origin (0 unstable,
//! ±m* stable). We push it on four fronts and find the hard limit:
//!
//!   S1  N INDEPENDENT bistable latches — is ρ flat in N? capacity? cost?
//!   S2  N COUPLED cells (all-to-all reservoir) — (a) does transient ρ stay controllable as N grows, or
//!       blow past 1? RAW fixed per-edge weight vs NORMALISED g/√N.
//!   S3  Settle steps vs ρ — critical slowing near ρ→1, and settle FAILURE (never quiesces) above it.
//!   S4  N COUPLED latches — memory CAPACITY: at what coupling does cross-talk corrupt the stored bits?
//!
//! Key nuance (documented, not hidden): a WORKING bistable latch has ρ_transient > 1 by construction, yet it
//! still SETTLES because tanh saturates (D = 1-m² → 0 at the fixed point). So "ρ<1 ⇒ stable" is FALSE for
//! memory — the hard instability is a property of asymmetric COUPLING, not of ρ>1 per se.
//!
//!   cargo test --manifest-path .../Cargo.toml -p openmorph-core --test stress v2_scale_recurrence -- --ignored --nocapture --test-threads=1

use crate::common::Rng;
use openmorph_core::substrate::Field;

const RHO_ITERS: usize = 150;

// ── Independent bistable latch bank: latch i uses cue cell 2i, self cell 2i+1. ──
//    self_i = tanh(W_SELF·self_i + W_CUE·cue_i). W_SELF>1 ⇒ bistable (holds a bit); W_CUE drives it.
fn latch_bank(n: usize, w_self: f32, w_cue: f32) -> Field {
    let mut f = Field::new(2 * n);
    for i in 0..n {
        let (cue, slf) = (2 * i, 2 * i + 1);
        f.add_unit(vec![cue, slf], vec![w_cue, w_self], 0.0, slf);
    }
    f
}

/// Drive the N cues to a ±1 pattern, settle; release the cues to 0, settle; read back the self signs.
/// Returns (bit errors, store firings, hold firings).
fn store_recall(f: &mut Field, n: usize, pattern: &[f32], budget: usize) -> (usize, usize, usize) {
    let drive: Vec<(usize, f32)> = (0..n).map(|i| (2 * i, pattern[i])).collect();
    let f_store = f.stimulate(&drive, budget);
    let release: Vec<(usize, f32)> = (0..n).map(|i| (2 * i, 0.0)).collect();
    let f_hold = f.stimulate(&release, budget);
    let errs = (0..n)
        .filter(|&i| f.cells()[2 * i + 1].signum() != pattern[i].signum())
        .count();
    (errs, f_store, f_hold)
}

/// A DENSE recurrent reservoir: cell i = tanh(Σ_{j≠i} w_ij·x_j), w_ij = ±weight (asymmetric, no self term,
/// no cue — pure feedback). Fan-in = N-1, so the per-cell drive scales with N: ρ ≈ weight·√N (circular law).
fn dense_reservoir(n: usize, weight: f32, rng: &mut Rng) -> Field {
    let mut f = Field::new(n);
    for i in 0..n {
        let mut reads = Vec::with_capacity(n - 1);
        let mut ws = Vec::with_capacity(n - 1);
        for j in 0..n {
            if j != i {
                reads.push(j);
                ws.push(weight * rng.bit());
            }
        }
        f.add_unit(reads, ws, 0.0, i);
    }
    f
}

/// A SPARSE recurrent reservoir: cell i reads K random others (K auto-clamped to ≤ N-1). weight = per-edge.
fn sparse_reservoir(n: usize, k: usize, weight: f32, rng: &mut Rng) -> Field {
    let k = k.min(n.saturating_sub(1));
    let mut f = Field::new(n);
    for i in 0..n {
        let (mut reads, mut ws) = (Vec::new(), Vec::new());
        while reads.len() < k {
            let j = rng.below(n);
            if j != i && !reads.contains(&j) {
                reads.push(j);
                ws.push(weight * rng.bit());
            }
        }
        f.add_unit(reads, ws, 0.0, i);
    }
    f
}

fn mean_abs(f: &Field) -> f32 {
    let c = f.cells();
    c.iter().map(|x| x.abs()).sum::<f32>() / c.len().max(1) as f32
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────────────
#[test]
#[ignore = "stress: nightly"]
fn s1_independent_latches_flat_rho_linear_capacity() {
    eprintln!(
        "\n=== S1  N INDEPENDENT bistable latches — ρ flat in N, 1 clean bit/latch, cost O(N) ==="
    );
    eprintln!("     latch_i = tanh(2.5·self + 3.0·cue); store a random N-bit pattern, release cues, recall.");
    let (w_self, w_cue) = (2.5f32, 3.0f32);
    let (mut peak_ber, mut rho_min, mut rho_max) = (0.0f32, f32::MAX, 0.0f32);
    for &n in &[1usize, 4, 16, 64, 256] {
        let budget = 200 * n + 200;
        // transient ρ at the ORIGIN (fresh field, cells=0 ⇒ D=I): the unsaturated operator.
        let rho = latch_bank(n, w_self, w_cue).spectral_radius(RHO_ITERS);
        rho_min = rho_min.min(rho);
        rho_max = rho_max.max(rho);
        let t0 = std::time::Instant::now();
        let (mut tot_err, mut tot_store, mut tot_hold) = (0usize, 0usize, 0usize);
        let seeds = 4;
        for s in 0..seeds {
            let mut rng = Rng::new(0xB17 + s as u64 + n as u64 * 131);
            let pat: Vec<f32> = (0..n).map(|_| rng.bit()).collect();
            let mut f = latch_bank(n, w_self, w_cue);
            let (e, fs, fh) = store_recall(&mut f, n, &pat, budget);
            tot_err += e;
            tot_store += fs;
            tot_hold += fh;
        }
        let secs = t0.elapsed().as_secs_f64();
        let ber = 100.0 * tot_err as f32 / (n * seeds) as f32;
        peak_ber = peak_ber.max(ber);
        eprintln!(
            "  N={n:>4}  cells={:>4}  ρ={rho:>5.3}  bit-err={ber:>5.1}%  store≈{:>6} fir  hold≈{:>6} fir  {:>9.0} bits/s",
            2 * n,
            tot_store / seeds,
            tot_hold / seeds,
            (n * seeds) as f64 / secs.max(1e-9),
        );
    }
    eprintln!(
        "  ⇒ ρ is FLAT in N: {rho_min:.3}..{rho_max:.3} (≈W_SELF={w_self}); independent loops never couple. Capacity = N bits, 0% error to N=256."
    );
    eprintln!("     NOTE: ρ≈2.5 > 1 at EVERY N, yet every field settles & recalls perfectly — a bistable latch");
    eprintln!("     REQUIRES ρ>1 (the origin must be repelling); it settles anyway because tanh saturates (D→0).");
    assert!(
        (rho_min - w_self).abs() < 0.15 && (rho_max - w_self).abs() < 0.15,
        "independent latch ρ must stay ≈ W_SELF regardless of N ({rho_min}..{rho_max} vs {w_self})"
    );
    assert!(
        peak_ber < 1.0,
        "independent latches must recall every bit at every N (peak BER {peak_ber:.1}%)"
    );
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────────────
#[test]
#[ignore = "stress: nightly"]
fn s2_coupling_rho_controllable_or_blows_up() {
    eprintln!("\n=== S2  COUPLED all-to-all reservoir — does transient ρ stay controllable as N grows? ===");
    eprintln!("     cell_i = tanh(Σ_{{j≠i}} ±w·x_j).  RAW: fixed per-edge w=0.15 (ρ≈w√N).  NORM: w=g/√N, g=0.8 (ρ≈g).");
    let (raw_w, g) = (0.15f32, 0.8f32);
    let (mut raw_super_at, mut raw_rho_at_super) = (None, 0.0f32);
    let mut norm_max = 0.0f32;
    for &n in &[4usize, 16, 64, 256, 512] {
        let mut rng = Rng::new(0x2A + n as u64);
        let rho_raw = dense_reservoir(n, raw_w, &mut rng).spectral_radius(RHO_ITERS);
        let mut rng2 = Rng::new(0x2A + n as u64); // SAME sign pattern ⇒ ρ scales linearly with the weight
        let rho_norm =
            dense_reservoir(n, g / (n as f32).sqrt(), &mut rng2).spectral_radius(RHO_ITERS);
        norm_max = norm_max.max(rho_norm);
        // settle a small kick under the RAW operator; converge or run to budget?
        let budget = 60 * n + 500;
        let mut raw_m = dense_reservoir(n, raw_w, &mut Rng::new(0x2A + n as u64));
        let mut rng3 = Rng::new(0x99 + n as u64);
        let kick: Vec<(usize, f32)> = (0..n).map(|i| (i, 0.05 * rng3.sym())).collect();
        let fir = raw_m.stimulate(&kick, budget);
        let settled = raw_m.settled();
        if rho_raw >= 1.0 && raw_super_at.is_none() {
            raw_super_at = Some(n);
            raw_rho_at_super = rho_raw;
        }
        eprintln!(
            "  N={n:>4}  RAW ρ={rho_raw:>6.3} {}  settle {:>7} fir {:>10} |cell|={:.3}   |   NORM(g={g}) ρ={rho_norm:>5.3} {}",
            if rho_raw >= 1.0 { "SUPER-CRIT" } else { "sub-crit  " },
            fir,
            if settled { "settled" } else { "HIT-BUDGET" },
            mean_abs(&raw_m),
            if rho_norm >= 1.0 { "super" } else { "sub-crit" },
        );
    }
    eprintln!(
        "  ⇒ RAW fixed-weight all-to-all coupling: ρ ≈ w·√N GROWS with N and crosses 1 at N={:?} (ρ≈{raw_rho_at_super:.2}) →",
        raw_super_at
    );
    eprintln!("     super-critical: the reservoir SATURATES (|cell|→1) and the kick no longer decays — memory is gone.");
    eprintln!("     NORM g/√N coupling: ρ pinned ≤{norm_max:.3}<1 at EVERY N → controllable. Coupling MUST be normalised");
    eprintln!(
        "     to fan-in, else recurrence self-destabilises purely by growing the population."
    );
    assert!(
        raw_super_at.is_some() && norm_max < 1.0,
        "RAW coupling must blow ρ past 1 as N grows ({raw_super_at:?}); NORM must stay <1 (max {norm_max:.3})"
    );
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────────────
#[test]
#[ignore = "stress: nightly"]
fn s3_critical_slowing_and_settle_failure_at_rho_one() {
    eprintln!(
        "\n=== S3  Settle steps vs ρ — critical slowing near ρ→1, settle FAILURE above it ==="
    );
    eprintln!("     N=128, sparse K=8, per-edge weight=g/√K ⇒ ρ≈g. Kick all cells; sweep the gain g = target ρ.");
    let (n, k) = (128usize, 8usize);
    let budget = 400 * n; // 51_200 firings — generous headroom to detect non-convergence
    let mut rows: Vec<(f32, f32, f64, f64, f32)> = Vec::new();
    for &g in &[0.3f32, 0.6, 0.8, 0.9, 0.95, 1.0, 1.05, 1.2, 1.5, 2.0] {
        let seeds = 5;
        let (mut rho_a, mut fir_a, mut settled_a, mut sat_a) = (0.0f32, 0.0f64, 0.0f64, 0.0f32);
        for s in 0..seeds {
            let mut rng = Rng::new(0x5EED + s as u64 + (g * 1000.0) as u64);
            let mut f = sparse_reservoir(n, k, g / (k as f32).sqrt(), &mut rng);
            rho_a += f.spectral_radius(RHO_ITERS);
            let mut rng2 = Rng::new(0xC0DE + s as u64);
            let kick: Vec<(usize, f32)> = (0..n).map(|i| (i, 0.05 * rng2.sym())).collect();
            fir_a += f.stimulate(&kick, budget) as f64;
            if f.settled() {
                settled_a += 1.0;
            }
            sat_a += mean_abs(&f);
        }
        let (rho, fir, settled, sat) = (
            rho_a / seeds as f32,
            fir_a / seeds as f64,
            settled_a / seeds as f64,
            sat_a / seeds as f32,
        );
        rows.push((g, rho, fir, settled, sat));
        eprintln!(
            "  g={g:>4}  ρ={rho:>5.3}  settle={fir:>9.0} fir  settled={:>3.0}%  final |cell|={sat:.3}  {}",
            100.0 * settled,
            if settled < 1.0 { "◄ NON-CONVERGENT" } else { "" }
        );
    }
    let base = rows.iter().find(|r| (r.0 - 0.3).abs() < 1e-6).unwrap().2;
    let near_one = rows
        .iter()
        .filter(|r| r.1 >= 0.85 && r.1 <= 1.2)
        .map(|r| r.2)
        .fold(0.0, f64::max);
    let peak = rows.iter().map(|r| r.2).fold(0.0, f64::max);
    let peak_g = rows.iter().max_by(|a, b| a.2.total_cmp(&b.2)).unwrap().0;
    let worst_settled = rows
        .iter()
        .filter(|r| r.1 >= 1.05)
        .map(|r| r.3)
        .fold(1.0, f64::min);
    eprintln!(
        "  ⇒ CRITICAL SLOWING: settle cost baseline(g0.3)={base:.0} fir → peak {peak:.0} fir near ρ≈1 (peak at g={peak_g}), a {:.1}× blow-up.",
        peak / base.max(1.0)
    );
    eprintln!(
        "     SETTLE FAILURE: for ρ≳1.05 the worst seed-mean settled rate is {:.0}% — the recurrent operator no",
        100.0 * worst_settled
    );
    eprintln!("     longer reaches a fixed point within a generous budget (asymmetric feedback ⇒ limit cycles).");
    assert!(
        near_one > 3.0 * base,
        "settle cost must blow up near ρ≈1 (near-one {near_one:.0} vs base {base:.0}) — critical slowing"
    );
    assert!(
        worst_settled < 1.0,
        "the operator must FAIL to settle for some ρ≥1.05 (worst settled {:.0}%)",
        100.0 * worst_settled
    );
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────────────
#[test]
#[ignore = "stress: nightly"]
fn s4_coupled_latch_capacity_crosstalk_limit() {
    eprintln!("\n=== S4  COUPLED latches — memory CAPACITY: coupling at which cross-talk corrupts stored bits ===");
    eprintln!("     N=64: self cell i, cue cell 64+i. latch_i = tanh(2.5·self + 3.0·cue + Σ_{{K=4}} ±c·self_j).");
    eprintln!(
        "     Store random 64-bit pattern, release cues, hold via self+coupling, recall. Sweep c."
    );
    let (n, k) = (64usize, 4usize);
    let (w_self, w_cue) = (2.5f32, 3.0f32);
    let budget = 300 * n;
    let mut c_star: Option<f32> = None;
    let mut last_clean = 0.0f32;
    for &c in &[0.0f32, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0] {
        let seeds = 5;
        let (mut rho_a, mut err_a, mut settled_ct) = (0.0f32, 0usize, 0usize);
        for s in 0..seeds {
            let mut rng = Rng::new(0xCAFE + s as u64 + (c * 100.0) as u64 * 7);
            // coupled latches: self cells 0..n, cue cells n..2n; each self reads K random other selves.
            let mut f = Field::new(2 * n);
            for i in 0..n {
                let (slf, cue) = (i, n + i);
                let (mut reads, mut ws) = (vec![cue, slf], vec![w_cue, w_self]);
                let mut chosen = Vec::new();
                while chosen.len() < k.min(n - 1) {
                    let j = rng.below(n);
                    if j != i && !chosen.contains(&j) {
                        chosen.push(j);
                        reads.push(j);
                        ws.push(c * rng.bit());
                    }
                }
                f.add_unit(reads, ws, 0.0, slf);
            }
            rho_a += f.spectral_radius(RHO_ITERS);
            let pat: Vec<f32> = (0..n).map(|_| rng.bit()).collect();
            let drive: Vec<(usize, f32)> = (0..n).map(|i| (n + i, pat[i])).collect();
            f.stimulate(&drive, budget);
            let release: Vec<(usize, f32)> = (0..n).map(|i| (n + i, 0.0)).collect();
            f.stimulate(&release, budget);
            if f.settled() {
                settled_ct += 1;
            }
            err_a += (0..n)
                .filter(|&i| f.cells()[i].signum() != pat[i].signum())
                .count();
        }
        let rho = rho_a / seeds as f32;
        let ber = 100.0 * err_a as f32 / (n * seeds) as f32;
        let bits_held = (n as f32) * (1.0 - err_a as f32 / (n * seeds) as f32);
        if ber < 5.0 {
            last_clean = c;
        }
        if ber > 10.0 && c_star.is_none() {
            c_star = Some(c);
        }
        eprintln!(
            "  c={c:>4}  ρ={rho:>6.3}  bit-err={ber:>5.1}%  bits held ≈ {bits_held:>4.1}/{n}  settled {}/{seeds}  {}",
            settled_ct,
            if ber > 10.0 { "◄ CORRUPTED" } else { "" }
        );
    }
    eprintln!(
        "  ⇒ CAPACITY LIMIT: 64 independent bits hold cleanly while coupling c ≤ {last_clean}; cross-talk breaks recall at c≈{:?}.",
        c_star
    );
    eprintln!(
        "     (self-drive 2.5·m* vs coupling field ~K·c = 4c: a minority bit flips once the cross-talk overcomes"
    );
    eprintln!("     the self-hold barrier — the observed onset. Independent bits are cheap; COUPLED bits trade capacity for it.)");
    assert!(
        c_star.is_some() && c_star.unwrap() <= 1.0,
        "coupling must eventually corrupt stored bits (cross-talk capacity limit)"
    );
}
