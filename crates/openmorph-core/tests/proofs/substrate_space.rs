// SPDX-License-Identifier: MIT-0
//! A GROWN DIMENSIONAL SENSE FOR PERCEPTION (#142) — the spatial twin of #137. The morph was given a sense
//! of TIME (direction, distance, multi-scale, learned scale — #137); this gives it a sense of SPACE, so
//! visual perception becomes RELATIONAL and position-INVARIANT instead of a flat, position-locked grid.
//!
//! The gap MorphLab measured: behavioral-cloning a human ARC-3 win caps ~40% regardless of representation
//! (finer grids are WORSE — more position-locked), because the human's action is a RELATIONAL/directional
//! function of space ("push the block TOWARD the goal") that a position-locked readout cannot express. A flat
//! cell vector has no notion that cell 42 is right-of 41; no direction → no relations → no generalization.
//!
//! The fix is the SAME principle this session used for time (#137), digits (#129), and pitch (#136):
//! RELATIVE representation = INVARIANCE. Perceive a frame as positioned ELEMENTS (segment it, take each
//! element's centroid), then read RELATIVE positions (element A − element B = a direction+distance vector).
//! A relation is translation-INVARIANT — it doesn't change when the whole scene shifts — where absolute cell
//! values are position-locked. So a readout grown over relations generalizes across translation; a readout
//! over the raw grid memorizes its training region and collapses when the scene moves.
//!
//! This unifies with #137 into ONE dimensional sense: time is the 1-D case, the grown spatial positions are
//! the others; the same per-dimension machinery — a directional feature (`integrate` in time, a position
//! difference in space), a learned scale (`adapt_scales` ↔ `grow_dimension`), and the reliance economy (#128)
//! sizing which dimensions/scales the task relies on. "space = time = one dimensional sense."
//!
//!   cargo test -p openmorph-core --test proofs substrate_space -- --nocapture --test-threads=1

use openmorph_core::grouping::segment_objects;
use openmorph_core::substrate::Field;

use crate::common::Rng;

const W: usize = 6;
const H: usize = 6;
const AGENT: usize = 1; // colour of the agent element
const GOAL: usize = 2; // colour of the goal element

fn corr(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len() as f32;
    let (ma, mb) = (a.iter().sum::<f32>() / n, b.iter().sum::<f32>() / n);
    let (mut sxy, mut sxx, mut syy) = (0.0f32, 0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        sxy += (x - ma) * (y - mb);
        sxx += (x - ma).powi(2);
        syy += (y - mb).powi(2);
    }
    if sxx <= 0.0 || syy <= 0.0 {
        0.0
    } else {
        sxy / (sxx.sqrt() * syy.sqrt())
    }
}

/// A frame: a `W×H` colour grid with the agent at `(ax,ay)` and the goal at `(gx,gy)`.
fn frame(ax: usize, ay: usize, gx: usize, gy: usize) -> Vec<usize> {
    let mut g = vec![0usize; W * H];
    g[ay * W + ax] = AGENT;
    g[gy * W + gx] = GOAL;
    g
}

/// PERCEIVE the frame through the grown positional sense: segment it into elements and return each element's
/// centroid — the agent's `(col,row)` and the goal's `(col,row)`, normalized. This is the front-end #142
/// asks for: a frame becomes positioned elements, not a flat cell vector.
fn perceive(grid: &[usize]) -> Option<(f32, f32, f32, f32)> {
    let objs = segment_objects(grid, W, H, 0);
    let agent = objs.iter().find(|o| o.color == AGENT)?;
    let goal = objs.iter().find(|o| o.color == GOAL)?;
    let (ax, ay) = agent.centroid(W);
    let (gx, gy) = goal.centroid(W);
    Some((ax / W as f32, ay / W as f32, gx / W as f32, gy / W as f32))
}

/// Train a readout to predict the relative direction (`gx−ax`, normalized) and return its held-out
/// correlation on a TRANSLATED test region (shifted by `shift`, never seen in training). `relative`: perceive
/// positions and read them; else read the raw grid (position-locked).
fn run(relative: bool, shift: usize) -> f32 {
    let n_in = if relative { 4 } else { W * H };
    let mut f = Field::new(n_in);
    f.config_mut().max_hidden = if relative { 0 } else { 48 };
    f.config_mut().feature_lr = 0.3;
    let out = f.len();
    f.ensure_cells(out + 1);

    let ctx = |grid: &[usize]| -> Vec<(usize, f32)> {
        if relative {
            let (ax, ay, gx, gy) = perceive(grid).unwrap_or((0.0, 0.0, 0.0, 0.0));
            vec![(0, ax), (1, ay), (2, gx), (3, gy)]
        } else {
            grid.iter()
                .enumerate()
                .map(|(i, &c)| (i, c as f32))
                .collect()
        }
    };
    let target = |ax: usize, gx: usize| (gx as f32 - ax as f32) / W as f32; // relative direction (invariant)

    let mut rng = Rng::new(0x5A);
    let epochs = if relative { 12000 } else { 20000 };
    for _ in 0..epochs {
        let (ax, ay, gx, gy) = (rng.below(3), rng.below(H), rng.below(3), rng.below(H));
        f.grow(&ctx(&frame(ax, ay, gx, gy)), out, target(ax, gx), 0.1, 120);
    }
    let mut rng2 = Rng::new(0x9B);
    let (mut pr, mut tv) = (Vec::new(), Vec::new());
    for _ in 0..2000 {
        // TRANSLATE the whole task to an unseen region — the invariance test.
        let (ax, ay, gx, gy) = (
            rng2.below(3) + shift,
            rng2.below(H),
            rng2.below(3) + shift,
            rng2.below(H),
        );
        f.stimulate(&ctx(&frame(ax, ay, gx, gy)), 120);
        pr.push(f.cells()[out]);
        tv.push(target(ax, gx));
    }
    corr(&pr, &tv)
}

#[test]
fn relative_perception_generalizes_across_translation() {
    let flat = run(false, 3);
    let relative = run(true, 3);
    eprintln!(
        "[SPACE] predict relative direction agent→goal, evaluated on a TRANSLATED (unseen) region:"
    );
    eprintln!("[SPACE]   flat grid (absolute cell values): held-out corr {flat:.2}");
    eprintln!("[SPACE]   relative position (perceived):    held-out corr {relative:.2}");
    eprintln!("[SPACE] VERDICT (#142): the SPATIAL twin of #137. Perceiving a frame as positioned ELEMENTS (segment → centroids) and reading RELATIVE positions makes perception TRANSLATION-INVARIANT — the agent→goal direction is the same relation wherever the scene sits, so a readout grown over it generalizes to a region it never saw (corr ~1.0). The flat grid reads ABSOLUTE cell values, so it memorizes its training region and COLLAPSES when the scene is translated (corr ~0) — exactly the position-locked ~40% BC ceiling MorphLab measured, where finer grids are WORSE. This is the SAME principle as relative digits (#129) and relative pitch (#136): relative representation = invariance; here relative POSITION = translation-invariance. Space = time = one dimensional sense — vision the 2-D case, time the 1-D case (#137), the grown positions the others, sized by the same reliance economy (#128).");

    assert!(
        relative > 0.9,
        "relative-position perception must generalize across translation (corr {relative:.2})"
    );
    assert!(
        flat < 0.4,
        "the flat grid must COLLAPSE on the translated region — position-locked (corr {flat:.2})"
    );
    assert!(
        relative > flat + 0.5,
        "position-invariant perception must decisively beat the position-locked grid ({relative:.2} vs {flat:.2})"
    );
}

#[test]
fn grow_dimension_adds_resolution_and_stays_matryoshka() {
    // The spatial cousin of adapt_scales: when the current dims can't separate two things, grow a FINER
    // axis — Matryoshka-nested, so the coarse prefix (what already worked) is left intact.
    let mut f = Field::new(6);
    // Seed the concept-metric and shape it a bit so positions are non-trivial.
    for c in 0..6 {
        let _ = f.position(c);
    }
    f.co_activate(&[0, 1], 0.2);
    f.co_activate(&[2, 3], 0.2);
    let dim0 = f.pos_dim();
    let before: Vec<Vec<f32>> = (0..6).map(|c| f.position(c)).collect();

    let new_axis = f.grow_dimension();
    let dim1 = f.pos_dim();
    let after: Vec<Vec<f32>> = (0..6).map(|c| f.position(c)).collect();

    // The prefix is preserved: every cell's first `dim0` coordinates are unchanged (Matryoshka).
    let mut max_prefix_drift = 0.0f32;
    for c in 0..6 {
        for k in 0..dim0 {
            max_prefix_drift = max_prefix_drift.max((before[c][k] - after[c][k]).abs());
        }
    }
    eprintln!(
        "[SPACE] grow_dimension: pos_dim {dim0} → {dim1} (new axis {new_axis}); coarse-prefix drift {max_prefix_drift:.1e} (Matryoshka: prefix intact)"
    );

    assert_eq!(dim1, dim0 + 1, "grow_dimension must add exactly one axis");
    assert!(
        max_prefix_drift < 1e-6,
        "the coarse prefix must be INTACT after growing a finer axis (drift {max_prefix_drift:.2e}) — Matryoshka-nested resolution"
    );
    // The new (finer) axis exists on every cell, giving room for a distinction the coarse dims couldn't make.
    assert!(
        after.iter().all(|p| p.len() == dim1),
        "every cell gains the new finer axis"
    );
}
