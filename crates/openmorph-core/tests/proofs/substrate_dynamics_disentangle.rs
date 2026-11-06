// SPDX-License-Identifier: MIT-0
//! #159 (OM-1) M2 rung 3 — separation by DYNAMICS via the shipped `resonate` front-end (#145). This is the
//! MODEST, honest kernel of MorphLab's dynamics route (see docs/investigations/159, §8), after an adversarial
//! verification stripped my overclaims — read the HONEST SCOPE before the numbers.
//!
//! THE CLAIM (precise): two sources that differ in their TEMPORAL DYNAMICS — here distinct FREQUENCIES — are
//! separated from a mixture by the substrate's shipped constant-Q `resonate` filterbank (#145), and the
//! separation relies on the TIME-STRUCTURE (a time-shuffle that holds the marginal distribution EXACTLY constant
//! collapses it). Same-frequency sources are correctly un-separable (the identifiability boundary). It is the
//! substrate's grown-filterbank realization of dynamics-based (SOBI-class) separation — the read-off is a usage
//! protocol over the resonate features.
//!
//! THE WORLD (deterministic, seeded): two independent NARROWBAND GAUSSIAN sources at distinct frequencies f0, f1
//! (AR(2) driven by Gaussian noise), linearly mixed by a 45° rotation into two channels.
//!
//! HONEST SCOPE (what the adversarial verify corrected — do NOT overclaim). (1) This is FREQUENCY DEMUX of
//! sources with (near-)DISJOINT spectra, NOT disentanglement of a genuinely entangled same-pixel mixture:
//! because the bands are disjoint, a band-pass recovers q·s0 for ANY mix, so the isotropic-mix framing is
//! largely irrelevant. Genuinely OVERLAPPING spectra needing cross-channel un-mixing (a grown SOBI/Givens
//! rotation) is the harder rung-3 follow-on MorphLab measured. (2) PCA (zero-lag covariance) is degenerate on
//! the isotropic mix — a strawman; it shows only that INSTANTANEOUS statistics fail here, NOT that "statistics
//! cannot". A lagged-covariance method (SOBI) uses the SAME temporal information the resonate bank does and
//! would separate this too — resonate does not "beat statistics", it IS a grown temporal-covariance filterbank.
//! The shuffle proves reliance on TEMPORAL DYNAMICS (assumption-free), not "dynamics not statistics". (3) The
//! frequency-finding is the bank's ENERGY PROFILE (a spectral estimate), reported not asserted; the code does
//! NOT call `adapt_freq` (that grown refinement is a separate shipped capability, `substrate_resonate`).
//! Deterministic (seeded), CI-cache-safe.

use openmorph_core::substrate::Field;

use crate::common::Rng;

/// Standard normal via Box–Muller (so the sources are EXACTLY Gaussian ⇒ ICA is provably blind).
fn gaussian(rng: &mut Rng) -> f32 {
    let u1 = rng.unit().max(1e-7);
    let u2 = rng.unit();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

/// A narrowband Gaussian source: AR(2) resonator at `freq` (radius `r`) driven by Gaussian noise.
fn narrowband(t: usize, freq: f32, r: f32, rng: &mut Rng) -> Vec<f32> {
    let (a1, a2) = (2.0 * r * freq.cos(), -r * r);
    let (mut y1, mut y2) = (0.0f32, 0.0f32);
    let mut out = Vec::with_capacity(t);
    for _ in 0..t {
        let y = a1 * y1 + a2 * y2 + gaussian(rng);
        out.push(y);
        y2 = y1;
        y1 = y;
    }
    // normalize to unit variance
    let m = out.iter().sum::<f32>() / t as f32;
    let sd = (out.iter().map(|v| (v - m).powi(2)).sum::<f32>() / t as f32)
        .sqrt()
        .max(1e-6);
    out.iter().map(|v| (v - m) / sd).collect()
}

fn corr(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len() as f32;
    let (ma, mb) = (a.iter().sum::<f32>() / n, b.iter().sum::<f32>() / n);
    let mut sxy = 0.0;
    let mut sxx = 0.0;
    let mut syy = 0.0;
    for i in 0..a.len() {
        sxy += (a[i] - ma) * (b[i] - mb);
        sxx += (a[i] - ma).powi(2);
        syy += (b[i] - mb).powi(2);
    }
    (sxy / (sxx.sqrt() * syy.sqrt()).max(1e-9)).abs()
}

/// Smoothed instantaneous power (the slowly-varying envelope) of a signal — how present the source is over time.
fn envelope(sig: &[f32], alpha: f32) -> Vec<f32> {
    let mut e = 0.0f32;
    sig.iter()
        .map(|&v| {
            e = alpha * e + (1.0 - alpha) * v * v;
            e
        })
        .collect()
}

/// A 2×2 recovery matrix C[k][j] = |corr(envelope(recovered_k), envelope(source_j))|; returns (matched, crosstalk)
/// under the best of the two permutations.
fn separation(rec: &[Vec<f32>; 2], src_env: &[Vec<f32>; 2]) -> (f32, f32) {
    let re: Vec<Vec<f32>> = rec.iter().map(|r| envelope(r, 0.9)).collect();
    let c = |k: usize, j: usize| corr(&re[k], &src_env[j]);
    // permutation (0->0,1->1)
    let (m_id, x_id) = ((c(0, 0) + c(1, 1)) / 2.0, (c(0, 1) + c(1, 0)) / 2.0);
    // permutation (0->1,1->0)
    let (m_sw, x_sw) = ((c(0, 1) + c(1, 0)) / 2.0, (c(0, 0) + c(1, 1)) / 2.0);
    if m_id >= m_sw {
        (m_id, x_id)
    } else {
        (m_sw, x_sw)
    }
}

/// Recover the two sources from a mixed channel via a constant-Q `resonate` BANK (the shipped #145 spectral
/// front-end): log-spaced poles cover the band; the bank's ENERGY PROFILE peaks at each source frequency, and
/// each peak resonator's output is that source's band (the other's distinct frequency is band-pass rejected).
/// NOTE: this uses a FIXED grid + energy peak-pick (a spectral estimate); it does NOT call `adapt_freq` (the
/// grown frequency refinement, proven separately in `substrate_resonate`). Returns (recovered[2], peak-freq[2]).
fn resonate_recover(mix_ch: &[f32]) -> ([Vec<f32>; 2], [f32; 2]) {
    const B: usize = 16;
    let mut f = Field::new(1);
    // log-spaced constant-Q bank across the band (0.12 .. 2.6 rad)
    let freqs: Vec<f32> = (0..B)
        .map(|i| 0.12 * (2.6f32 / 0.12).powf(i as f32 / (B - 1) as f32))
        .collect();
    let cells: Vec<usize> = freqs.iter().map(|&w| f.resonate(0, w, 0.06)).collect();
    let n = mix_ch.len();
    let warm = n / 2;
    let mut energy = [0.0f32; B];
    let mut hist: Vec<Vec<f32>> = (0..B).map(|_| Vec::new()).collect();
    for (t, &x) in mix_ch.iter().enumerate() {
        f.stimulate(&[(0, x)], 2);
        if t >= warm {
            for b in 0..B {
                let v = f.cells()[cells[b]];
                energy[b] += v * v;
                hist[b].push(v);
            }
        }
    }
    // find the TWO dominant energy peaks (local maxima), well separated in band index
    let mut order: Vec<usize> = (0..B).collect();
    order.sort_by(|&a, &b| energy[b].partial_cmp(&energy[a]).unwrap());
    let p0 = order[0];
    let p1 = *order
        .iter()
        .find(|&&b| (b as i32 - p0 as i32).abs() >= 2)
        .unwrap_or(&order[1]);
    ([hist[p0].clone(), hist[p1].clone()], [freqs[p0], freqs[p1]])
}

/// PCA baseline: the two principal components of the 2-channel mixture (a statistical, dynamics-blind rotation).
fn pca_recover(x0: &[f32], x1: &[f32], warm: usize) -> [Vec<f32>; 2] {
    // 2×2 covariance
    let n = x0.len() as f32;
    let (m0, m1) = (x0.iter().sum::<f32>() / n, x1.iter().sum::<f32>() / n);
    let mut c00 = 0.0;
    let mut c11 = 0.0;
    let mut c01 = 0.0;
    for i in 0..x0.len() {
        c00 += (x0[i] - m0).powi(2);
        c11 += (x1[i] - m1).powi(2);
        c01 += (x0[i] - m0) * (x1[i] - m1);
    }
    c00 /= n;
    c11 /= n;
    c01 /= n;
    // principal axis angle (2×2 symmetric eig, closed form)
    let theta = 0.5 * (2.0 * c01).atan2(c00 - c11);
    let (ct, st) = (theta.cos(), theta.sin());
    let mut p0 = Vec::new();
    let mut p1 = Vec::new();
    for i in warm..x0.len() {
        p0.push(ct * (x0[i] - m0) + st * (x1[i] - m1));
        p1.push(-st * (x0[i] - m0) + ct * (x1[i] - m1));
    }
    [p0, p1]
}

/// Run one world: generate two narrowband Gaussian sources, mix, and score. `same_freq`/`shuffle` are controls.
/// Returns (resonate_gap, pca_gap, peak_freqs) — gap = matched − crosstalk of envelope correlations.
fn run(seed: u64, f0: f32, f1: f32, same_freq: bool, shuffle: bool) -> (f32, f32, [f32; 2]) {
    let t = 6000;
    let mut rng = Rng::new(seed);
    let (g0, g1) = if same_freq { (f0, f0) } else { (f0, f1) };
    let mut s0 = narrowband(t, g0, 0.94, &mut rng);
    let mut s1 = narrowband(t, g1, 0.94, &mut rng);
    if shuffle {
        // destroy the time structure (keep the marginal Gaussian statistics identical)
        for i in (1..s0.len()).rev() {
            s0.swap(i, rng.below(i + 1));
        }
        for i in (1..s1.len()).rev() {
            s1.swap(i, rng.below(i + 1));
        }
    }
    // STRONG mix (a 45° rotation) — both channels carry BOTH sources equally, so PCA/statistics cannot
    // separate a Gaussian mixture (rotation-invariant). The genuine same-pixel entanglement.
    let q = std::f32::consts::FRAC_1_SQRT_2;
    let m = [[q, q], [q, -q]];
    let x0: Vec<f32> = (0..t).map(|i| m[0][0] * s0[i] + m[0][1] * s1[i]).collect();
    let x1: Vec<f32> = (0..t).map(|i| m[1][0] * s0[i] + m[1][1] * s1[i]).collect();
    let warm = t / 2;
    let src_env = [envelope(&s0[warm..], 0.9), envelope(&s1[warm..], 0.9)];

    // GROWN resonate-bank recovery (energy profile discovers the two frequencies, peak resonators = the bands)
    let (rec, locked) = resonate_recover(&x0);
    let (r_m, r_x) = separation(&rec, &src_env);
    // FLAT (PCA) baseline over BOTH channels — a statistical rotation, blind to dynamics.
    let pca = pca_recover(&x0, &x1, warm);
    let (p_m, p_x) = separation(&pca, &src_env);

    (r_m - r_x, p_m - p_x, locked)
}

#[test]
fn resonate_separates_distinct_frequency_sources() {
    let (f0, f1) = (0.35f32, 0.95f32);
    let seeds = [0xD1u64, 0xD2, 0xD3, 0xD4, 0xD5];
    let (mut rgap, mut pgap) = (Vec::new(), Vec::new());
    let (mut peak_err, mut np) = (0.0f32, 0.0f32);
    for &s in &seeds {
        let (rg, pg, peaks) = run(s, f0, f1, false, false);
        rgap.push(rg);
        pgap.push(pg);
        let e = ((peaks[0] - f0).abs() + (peaks[1] - f1).abs())
            .min((peaks[0] - f1).abs() + (peaks[1] - f0).abs());
        peak_err += e / 2.0;
        np += 1.0;
    }
    let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
    let (rg, pg) = (mean(&rgap), mean(&pgap));
    peak_err /= np;
    eprintln!("[DYN] SEPARATION gap (matched−crosstalk, higher=better): resonate filterbank {rg:.3}  vs  zero-lag PCA {pg:.3} (PCA degenerate on the isotropic mix)");
    eprintln!("[DYN] the bank's energy profile peaks at the source frequencies (reported, not a discovery claim): mean |Δpeak| {peak_err:.3}");

    // Controls: a time-shuffle holds the marginal distribution EXACTLY constant (a permutation of the same
    // samples) and destroys ONLY the time-structure → collapse ⇒ the separation relies on TEMPORAL DYNAMICS.
    let (rg_shuf, _, _) = run(0xD1, f0, f1, false, true);
    let (rg_same, _, _) = run(0xD1, f0, f1, true, false);
    eprintln!("[DYN] CONTROL (relies on temporal dynamics): time-shuffled sources (marginals identical) → resonate gap {rg_shuf:.3} — collapses");
    eprintln!("[DYN] CONTROL (identifiability boundary): same-frequency sources → resonate gap {rg_same:.3} — correctly un-separable");

    eprintln!("[DYN] VERDICT (honest, modest): the substrate's shipped constant-Q `resonate` filterbank (#145) separates two mixed sources that differ in their TEMPORAL DYNAMICS (distinct frequencies) — and the separation RELIES on the time-structure: a shuffle that keeps the marginal distribution exactly constant collapses it, and same-frequency sources are correctly un-separable. This is a usage protocol over `resonate` — the substrate's grown-filterbank realization of dynamics-based (SOBI-class) separation. HONEST SCOPE (per adversarial verify): the sources have (near-)DISJOINT spectra, so this is frequency DEMUX, NOT disentanglement of a genuinely entangled mixture (band-pass recovers q·s0 for any mix); PCA is a zero-lag strawman (a lagged-covariance SOBI uses the SAME temporal information and would also separate — resonate does not 'beat statistics'); the frequency-finding is the energy profile (not `adapt_freq`). The genuine rung-3 disentanglement (OVERLAPPING spectra + a grown cross-channel un-mixing rotation) is the harder follow-on MorphLab measured — see docs/investigations/159 §8.");

    // ── GATES (only the claims that survive the adversarial verify) ───────────────────────────────────────
    assert!(
        rg > 0.4,
        "resonate did not separate the distinct-frequency sources (gap {rg:.3})"
    );
    assert!(rg > pg + 0.3, "resonate gap {rg:.3} not above zero-lag PCA {pg:.3} (which is degenerate on the isotropic mix)");
    assert!(rg_shuf < rg - 0.3, "CONTROL: time-shuffle did not collapse it ({rg_shuf:.3} vs {rg:.3}) — separation must rely on temporal dynamics");
    assert!(
        rg_same < rg - 0.3,
        "CONTROL: same-frequency should be un-separable ({rg_same:.3} vs {rg:.3})"
    );
}
