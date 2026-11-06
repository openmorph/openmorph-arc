// SPDX-License-Identifier: MIT-0
//! WORKED PATH — the perception + world-model FRONT-END for a 2-D colour stream (for MorphLab's ARC-3).
//!
//! MorphLab's finding: a morph that reads an 8×8 AVERAGE of the frame can only wander — the blur throws
//! away *which object moved*. The fix is the substrate's own perception primitives:
//!   1. `grouping::segment_objects` turns the raw colour frame into OBJECTS (position, colour, size) —
//!      structure, not a blur.
//!   2. `observe` runs on that OBJECT-CENTRIC representation to grow a WORLD MODEL; the per-cell prediction
//!      error is the dense intrinsic reward.
//!   3. WIRING (MorphLab's grow_backbone finding): LINEAR by default; grow features (`grow_deep_step`,
//!      residual-birth) only where linear fails. Object motion on a one-hot position is near-linear.
//!
//! ── SCOPE (truth-not-trophy — MorphLab #114). This demonstrates the MOVING-OBJECT case: a rigid object
//! whose CENTROID shifts. Two honest cautions, because both are things the vision forbids authoring:
//!   • CENTROID is not THE state variable — it is a hand-picked shortcut that only fits games with a moving
//!     object. Most ARC-3 games are REGION-TRANSFORMS (recolor/toggle/reshape ~74 cells), where the centroid
//!     is quasi-static and `sign(Δcentroid)` amplifies segmentation JITTER into a spurious constant — a
//!     constant predictor then scores a FAKE 100%. The non-authored target is `observe`'s general one:
//!     predict the RAW next cells (object-centric), and let the transformation EMERGE. Do not author a
//!     "predict-the-centroid" or "predict-the-mask" driver — grow the transduction that predicts the cells.
//!   • CORRESPONDENCE across frames (which object is which next frame) is NOT a hand-coded matcher
//!     (nearest-centroid / Hungarian). The correspondence is the mapping that MINIMIZES prediction error —
//!     it EMERGES from the world model. `relate` exposes the cross-frame structure; the organism grows which
//!     edge is real. (`membrane_field.rs`'s `QueryableImage` is an EXAMPLE Interface, not a core primitive:
//!     active sensing is the organism WRITING a fovea position — grown — into a `Both` port and reading the
//!     value; a fixed coarse downsample is itself the authored blur this front-end warns against.)
//!   • TEST THE MECHANISM INSIDE, never the outside number: check that the prediction is ACTION-CONDITIONAL
//!     (Δ changes with the action). A constant predictor hits 100% on a jittery target — score the inside.
//!
//! This proves the moving-object front-end: segment → object-centric `observe` predicts the object's exact
//! next position; the same `observe` on an averaged blur cannot resolve it.
//!
//!   cargo test -p openmorph-core --test proofs arc3_frontend -- --nocapture --test-threads=1

use openmorph_core::grouping::segment_objects;
use openmorph_core::substrate::Field;

const W: usize = 12; // frame width
const H: usize = 5; // frame height
const Y0: usize = 2; // the object's row
const COLOR: usize = 4; // the object's colour (background = 0)
const WC: usize = 3; // blur columns (each averages W/WC = 4 fine columns → loses exact position)
const BUD: usize = 100;

/// The world: one coloured object at column `x` (row Y0). It moves +1 each step (wrapping).
fn frame(x: usize) -> Vec<usize> {
    let mut g = vec![0usize; W * H];
    g[Y0 * W + x] = COLOR;
    g
}

/// OBJECT-CENTRIC front-end: segment the frame, take the object's centroid column. This is the
/// structure the blur destroys — "there is an object, it is HERE".
fn centroid(x: usize) -> usize {
    let objs = segment_objects(&frame(x), W, H, 0);
    let o = &objs[0];
    o.cells.iter().map(|&c| c % W).sum::<usize>() / o.cells.len()
}

/// BLUR front-end: average presence over each coarse column-group — the exact column is gone.
fn blur(x: usize) -> Vec<f32> {
    let g = frame(x);
    let mut b = vec![-1.0f32; WC];
    for col in 0..W {
        if (0..H).any(|row| g[row * W + col] != 0) {
            b[col / (W / WC)] = 1.0;
        }
    }
    b
}

fn onehot(pos: usize) -> Vec<f32> {
    (0..W).map(|i| if i == pos { 1.0 } else { -1.0 }).collect()
}
fn argmax(cells: &[f32], base: usize) -> usize {
    (0..W)
        .max_by(|&i, &j| cells[base + i].partial_cmp(&cells[base + j]).unwrap())
        .unwrap()
}

/// A linear predictor (no backbone — the robust default): W units, each reading `ctx`, writing a
/// prediction cell. Returns (field, pred_base).
fn predictor(n_cells: usize, ctx: &[usize]) -> (Field, usize) {
    let mut f = Field::new(n_cells);
    let pred = f.len();
    for p in 0..W {
        let n = ctx.len();
        f.add_unit(ctx.to_vec(), vec![0.0; n], 0.0, pred + p);
    }
    (f, pred)
}

/// Object-centric world model: `observe` predicts the next centroid one-hot from the current one
/// (target-free). Returns exact next-position accuracy.
fn object_centric() -> f64 {
    let ctx: Vec<usize> = (0..W).collect(); // the centroid one-hot band IS the context (autonomous)
    let (mut f, pred) = predictor(W, &ctx);
    let mut x = 0usize;
    f.stimulate(
        &onehot(centroid(x))
            .into_iter()
            .enumerate()
            .collect::<Vec<_>>(),
        BUD,
    );
    for _ in 0..6000 {
        let nx = (x + 1) % W;
        f.observe(0, pred, W, &onehot(centroid(nx)), 0.1, BUD); // learn shift, take next in
        x = nx;
    }
    let (mut ok, mut tot) = (0, 0);
    for x in 0..W {
        f.stimulate(
            &onehot(centroid(x))
                .into_iter()
                .enumerate()
                .collect::<Vec<_>>(),
            BUD,
        );
        if argmax(f.cells(), pred) == centroid((x + 1) % W) {
            ok += 1;
        }
        tot += 1;
    }
    ok as f64 / tot as f64
}

/// Blur world model: predict the exact next centroid from the averaged blur (self-supervised — the
/// actual next position is observed after the fact, but must be predicted FROM the blur).
fn blurred() -> f64 {
    let blur_cells: Vec<usize> = (W..W + WC).collect(); // context = the coarse blur
    let (mut f, pred) = predictor(W + WC, &blur_cells);
    let mut x = 0usize;
    for _ in 0..6000 {
        let nx = (x + 1) % W;
        let bp: Vec<(usize, f32)> = blur(x)
            .into_iter()
            .enumerate()
            .map(|(i, v)| (W + i, v))
            .collect();
        f.stimulate(&bp, BUD); // present the current blur
        f.observe_from(&blur_cells, 0, pred, W, &onehot(centroid(nx)), 0.1, BUD);
        x = nx;
    }
    let (mut ok, mut tot) = (0, 0);
    for x in 0..W {
        let bp: Vec<(usize, f32)> = blur(x)
            .into_iter()
            .enumerate()
            .map(|(i, v)| (W + i, v))
            .collect();
        f.stimulate(&bp, BUD);
        if argmax(f.cells(), pred) == centroid((x + 1) % W) {
            ok += 1;
        }
        tot += 1;
    }
    ok as f64 / tot as f64
}

#[test]
fn segment_then_observe_beats_the_blur_front_end() {
    let oc = object_centric();
    let bl = blurred();

    eprintln!(
        "[FRONTEND] predict the object's exact next position (a {W}-wide frame, one moving object):\n  OBJECT-CENTRIC (segment_objects → observe): {:.0}%\n  BLUR ({WC}-column average → observe)        : {:.0}%  (can't resolve within a group)",
        oc * 100.0,
        bl * 100.0,
    );
    eprintln!("[FRONTEND] VERDICT (for ARC-3): the front-end is the gate. `segment_objects` gives the morph OBJECTS (this one, here, moved), so `observe` grows a world model that predicts motion — the dense signal an acting morph plugs into. An averaged frame throws the structure away and can only wander. Wiring: linear-default `observe` (grow features only where linear fails), segment for structure, and the foveated QueryableImage (membrane_field) to attend instead of blur. This is the perception path to beat random on the real 64×64 stream.");

    assert!(
        oc >= 0.95,
        "object-centric observe should predict the motion (got {oc:.2})"
    );
    assert!(
        oc > bl + 0.3,
        "the object-centric front-end must beat the blur substantially (oc {oc:.2} vs blur {bl:.2})"
    );
}
