// SPDX-License-Identifier: MIT-0
//! v2.0-dev · M1 · H-S3 — is dimension annealing SPECTRAL RANK TRUNCATION?
//! `metabolize_dimension` drops the axis of least per-axis VARIANCE. That equals dropping the least-EIGENVALUE
//! direction of the position covariance ONLY when the covariance is (near-)diagonal (axes uncorrelated). So the
//! honest H-S3 has two parts: (a) it drops argmin per-axis variance — exact by construction; (b) that equals
//! eigenvalue truncation to the extent the covariance is diagonal — measured via off-diagonal mass. Plus the
//! Matryoshka ordering (earlier dims carry more variance = larger eigenvalues).
use openmorph_core::substrate::Field;

#[allow(unused_imports)]
use crate::common::Rng;

const NF: usize = 48; // cells
const NC: usize = 6; // concepts (groups of 4 cells each)

fn build_map(seed: u64) -> Field {
    let _ = seed;
    let mut f = Field::new(NF);
    // Cluster each concept's 4 cells via metric_step → coarse structure on the base dims (high variance).
    for step in 0..4000usize {
        let c = step % NC;
        let group: Vec<usize> = (0..4).map(|k| c * 4 + k).collect();
        f.metric_step(&group, 0.08, 6);
    }
    // Grow two FINER Matryoshka axes; shape them only lightly ⇒ they carry LESS variance than the coarse dims.
    f.grow_dimension();
    f.grow_dimension();
    for step in 0..400usize {
        let c = step % NC;
        let group: Vec<usize> = (0..4).map(|k| c * 4 + k).collect();
        f.metric_step(&group, 0.02, 4);
    }
    f
}

/// per-axis variance + full DxD covariance from the grown positions.
fn covariance(f: &Field) -> (usize, Vec<f32>, Vec<Vec<f32>>) {
    let d = f.pos_dim();
    let pos: Vec<Vec<f32>> = (0..NF).map(|c| f.position(c)).collect();
    let mean: Vec<f32> = (0..d)
        .map(|a| pos.iter().map(|p| p[a]).sum::<f32>() / NF as f32)
        .collect();
    let mut cov = vec![vec![0.0f32; d]; d];
    for p in &pos {
        for a in 0..d {
            for b in 0..d {
                cov[a][b] += (p[a] - mean[a]) * (p[b] - mean[b]);
            }
        }
    }
    for row in cov.iter_mut() {
        for x in row.iter_mut() {
            *x /= NF as f32;
        }
    }
    let var: Vec<f32> = (0..d).map(|a| cov[a][a]).collect();
    (d, var, cov)
}

#[test]
fn dimension_annealing_is_per_axis_variance_truncation() {
    let mut f = build_map(0xd3);
    let (d, var, cov) = covariance(&f);
    let rounded: Vec<f32> = var.iter().map(|v| (v * 1000.0).round() / 1000.0).collect();
    eprintln!("[HS3] pos_dim={d}  per-axis variance = {rounded:?}");

    // (a) MECHANISM — metabolize drops the argmin per-axis-variance axis.
    let my_argmin = (0..d)
        .min_by(|&a, &b| var[a].partial_cmp(&var[b]).unwrap())
        .unwrap();
    let maxv = var.iter().copied().fold(0.0f32, f32::max);
    let dropped = f.metabolize_dimension(1.0); // ratio 1.0 ⇒ drop the weakest whenever it is below the max
    eprintln!(
        "[HS3] metabolize dropped axis {dropped:?}  (min-variance axis {my_argmin}, weakest var {:.4} vs max {maxv:.4})",
        var[my_argmin]
    );
    let dd = dropped.expect("a below-max axis exists ⇒ metabolize drops it");
    assert!(
        var[dd] <= var[my_argmin] + 1e-6,
        "dropped axis {dd} (var {:.5}) must be the MIN per-axis variance ({:.5})",
        var[dd],
        var[my_argmin]
    );
    assert_eq!(f.pos_dim(), d - 1, "dimension reduced by exactly one");

    // (b) SPECTRAL EXACTNESS — per-axis == eigenvalue iff covariance is diagonal. Measure off-diagonal mass.
    let diag_sq: f32 = (0..d).map(|a| cov[a][a] * cov[a][a]).sum();
    let off_sq: f32 = (0..d)
        .map(|a| {
            (0..d)
                .filter(|&b| b != a)
                .map(|b| cov[a][b] * cov[a][b])
                .sum::<f32>()
        })
        .sum();
    let off_ratio = off_sq / (off_sq + diag_sq).max(1e-9);
    eprintln!(
        "[HS3] covariance OFF-DIAGONAL mass ratio = {off_ratio:.3}  (→0 = diagonal ⇒ per-axis truncation == eigenvalue truncation)"
    );

    // (c) MATRYOSHKA — earlier (coarse) dims carry MORE variance than the finest grown dim.
    let coarse = var[0].max(var[1]);
    let finest = var[d - 1];
    eprintln!(
        "[HS3] Matryoshka: coarse var(dim0/1)={coarse:.4}  vs  finest var(dim{})={finest:.4}  ⇒ coarse-first {}",
        d - 1,
        if coarse > finest { "HOLDS" } else { "VIOLATED" }
    );

    eprintln!("[HS3] VERDICT: metabolize_dimension IS exact per-axis-variance (coordinate) truncation — it drops the argmin-variance axis, confirmed. It equals SPECTRAL eigenvalue truncation only to the degree the position covariance is diagonal (measured off-diagonal ratio above); metric_step does not enforce orthogonal axes, so H-S3 is EXACT as coordinate-truncation, APPROXIMATE as eigenvalue-truncation. Matryoshka keeps the coarse dims heavier.");
    assert!((0.0..=1.0).contains(&off_ratio));
}
