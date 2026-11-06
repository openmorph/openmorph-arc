// SPDX-License-Identifier: MIT-0
//! GROUPING — structural perception, all domain-blind (a general primitive, never dataset-specific).
//! Three universal operations turn a raw channel into structure the substrate can grow over:
//!
//! - **group** ([`segment_objects`]) — 4-connected same-colour components (union-find) bind cells into `Object`s.
//! - **reduce** ([`aggregate`]) — fold a variable set of per-element scalars into FIXED stats (count / sum /
//!   min / max / mean / spread), so a variable-cardinality scene meets the fixed-width substrate. (`spread` =
//!   dispersion; over an ensemble of grown estimators it is the value-of-information uncertainty.)
//! - **relate** ([`relate`]) — the raw pairwise structure among objects (hypergraph edges: relative
//!   position, size ratio, colour equality, adjacency).
//!
//! None of these encodes a task — they expose STRUCTURE, like a raw channel exposes values; the organism
//! grows the meaning. Group → reduce / relate is the general pipeline for reasoning over sets and relations.

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use crate::compat::Float;
#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use alloc::{
    boxed::Box,
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};
use serde::{Deserialize, Serialize};

/// A segmented OBJECT — a 4-connected region of same-colour non-background cells, plus the derived
/// properties an object-level rule keys on.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Object {
    /// The grid cell indices this object occupies.
    pub cells: Vec<usize>,
    /// The object's (uniform) colour.
    pub color: usize,
    /// Bounding-box width (columns spanned).
    pub bbox_w: usize,
    /// Bounding-box height (rows spanned).
    pub bbox_h: usize,
}

impl Object {
    /// Cell count (area).
    pub fn size(&self) -> usize {
        self.cells.len()
    }
    /// Centre of the object's cells as `(col, row)` in ABSOLUTE grid coordinates (mean column, mean row —
    /// NOT normalized to `[0,1]`; divide by `width`/`height` yourself if you need that) on a grid of
    /// `width` — for spatial relations. `width` is only used to recover `(col, row)` from cell indices; a
    /// `width` of `0` is read as `1` (a single column) rather than dividing by zero.
    pub fn centroid(&self, width: usize) -> (f32, f32) {
        if self.cells.is_empty() {
            return (0.0, 0.0);
        }
        let width = width.max(1);
        let n = self.cells.len() as f32;
        let (mut sc, mut sr) = (0.0f32, 0.0f32);
        for &c in &self.cells {
            sc += (c % width) as f32;
            sr += (c / width) as f32;
        }
        (sc / n, sr / n)
    }
    /// Bounding box as `(min_col, min_row, max_col, max_row)` on a grid of `width` (`0` read as `1`).
    fn bbox(&self, width: usize) -> (usize, usize, usize, usize) {
        let width = width.max(1);
        let mut b = (usize::MAX, usize::MAX, 0usize, 0usize);
        for &c in &self.cells {
            let (col, row) = (c % width, c / width);
            b.0 = b.0.min(col);
            b.1 = b.1.min(row);
            b.2 = b.2.max(col);
            b.3 = b.3.max(row);
        }
        b
    }
}

/// Fixed statistics of a variable-size set of per-element scalars — the general REDUCE (fold) that maps a
/// set to a fixed representation the substrate can grow over. `segment_objects` GROUPS a channel into a
/// variable set; `aggregate` REDUCES it — the two universal structural operations. Pure statistics, no
/// task meaning. Empty set ⇒ all zero (with `min`/`max` 0.0).
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Aggregate {
    /// Number of elements.
    pub count: f32,
    /// Sum of the elements.
    pub sum: f32,
    /// Smallest element (0.0 if empty).
    pub min: f32,
    /// Largest element (0.0 if empty).
    pub max: f32,
    /// Mean of the elements (0.0 if empty).
    pub mean: f32,
    /// SPREAD — the population standard deviation (0.0 if empty or all-equal). The second moment, completing
    /// count/sum/min/max/mean. As a general reduction it is the set's DISPERSION; over an ensemble of grown
    /// value-estimators with varied priors it is exactly the epistemic UNCERTAINTY (disagreement) that drives
    /// value-of-information exploration — so acting on `mean + spread` explores where the estimators disagree
    /// and commits as they converge, no authored exploration constant. Nothing here is task-specific; it is a
    /// statistic, and the organism grows the meaning.
    pub spread: f32,
}

impl Aggregate {
    /// As a fixed `[count, sum, min, max, mean]` vector — feed it straight into cells for the organism to
    /// grow over (a variable-size scene reduced to a fixed input). `spread` is intentionally NOT in this
    /// vector (kept 5-wide for consumers that pin the width); read it as the field.
    pub fn to_vec(&self) -> [f32; 5] {
        [self.count, self.sum, self.min, self.max, self.mean]
    }
}

/// REDUCE a variable-size set of scalars to fixed statistics — the general fold.
pub fn aggregate(values: &[f32]) -> Aggregate {
    if values.is_empty() {
        return Aggregate::default();
    }
    let count = values.len() as f32;
    let sum: f32 = values.iter().sum();
    let min = values.iter().copied().fold(f32::INFINITY, f32::min);
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mean = sum / count;
    let variance = values.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / count;
    Aggregate {
        count,
        sum,
        min,
        max,
        mean,
        spread: variance.sqrt(),
    }
}

/// REDUCE a per-object feature over a set of objects (group → reduce): e.g.
/// `aggregate_objects(&objs, |o| o.size() as f32)` gives count/sum/min/max/mean of object sizes.
pub fn aggregate_objects(objects: &[Object], feat: impl Fn(&Object) -> f32) -> Aggregate {
    aggregate(&objects.iter().map(feat).collect::<Vec<_>>())
}

/// A pairwise RELATION between two objects — the raw structural facts of an edge in the object hypergraph.
/// No task meaning: relative position, size ratio, colour equality, adjacency — the organism grows which
/// relations matter, exactly as it grows over `segment_objects`' regions.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Relation {
    /// Index of the first object (into the `objects` slice).
    pub a: usize,
    /// Index of the second object.
    pub b: usize,
    /// Column offset centre(b) − centre(a), normalized by the grid width (relative position).
    pub dx: f32,
    /// Row offset centre(b) − centre(a), normalized by the grid height.
    pub dy: f32,
    /// `b.size / a.size` (relative scale; 0.0 if `a` is empty).
    pub size_ratio: f32,
    /// Whether the two objects share a colour.
    pub same_color: bool,
    /// Whether their bounding boxes touch or overlap (proximity, within one cell in both axes).
    pub adjacent: bool,
}

/// RELATE — all ordered pairwise relations among `objects` (the hypergraph edges over the set). Raw
/// structural facts only; the substrate grows over them (e.g. "is there an adjacent same-colour pair").
/// A zero `width`/`height` is read as `1` (no division by zero).
pub fn relate(objects: &[Object], width: usize, height: usize) -> Vec<Relation> {
    let w = width.max(1) as f32;
    let h = height.max(1) as f32;
    let n = objects.len();
    // Precompute each object's centroid / bbox / size ONCE (O(n)); the old code recomputed the inner
    // object's centroid+bbox on every outer iteration, making it O(n²) full-cell passes. Pure functions of
    // the object, so caching them is byte-identical — same f32s, same emitted `Relation` order.
    let centroids: Vec<(f32, f32)> = objects.iter().map(|o| o.centroid(width)).collect();
    let bboxes: Vec<(usize, usize, usize, usize)> = objects.iter().map(|o| o.bbox(width)).collect();
    let sizes: Vec<f32> = objects.iter().map(|o| o.size() as f32).collect();
    let mut rels = Vec::with_capacity(n.saturating_mul(n.saturating_sub(1)));
    for a in 0..n {
        let (ca, ra) = centroids[a];
        let ba = bboxes[a];
        let sa = sizes[a];
        for b in 0..n {
            if a == b {
                continue;
            }
            let (cb, rb) = centroids[b];
            let bb = bboxes[b];
            let gap_x = ba.0.max(bb.0).saturating_sub(ba.2.min(bb.2));
            let gap_y = ba.1.max(bb.1).saturating_sub(ba.3.min(bb.3));
            rels.push(Relation {
                a,
                b,
                dx: (cb - ca) / w,
                dy: (rb - ra) / h,
                size_ratio: if sa > 0.0 { sizes[b] / sa } else { 0.0 },
                same_color: objects[a].color == objects[b].color,
                adjacent: gap_x <= 1 && gap_y <= 1,
            });
        }
    }
    rels
}

/// Segment a grid into OBJECTS — the 4-connected components of same-colour, non-`background` cells, in
/// first-seen (row-major) order. Computed by a single UNION-FIND pass (near-linear, `O(n·α)`), which
/// replaced the earlier iterated label-propagation relaxation (each cell adopting the minimum label among
/// itself and its same-colour 4-neighbours until nothing changed, `O(n·diameter)`): connected components
/// are unique, so the partition, object order, cells and bounding boxes are identical to that fixed point.
/// A GENERAL grouping / binding primitive (not ARC-specific). A `grid` shorter than `width × height`
/// yields no objects.
pub fn segment_objects(
    grid: &[usize],
    width: usize,
    height: usize,
    background: usize,
) -> Vec<Object> {
    let n = width * height;
    // A malformed frame whose length does not cover width×height would index out of bounds in the scan below;
    // return no objects rather than panic (defensive, matching core's guard style — a well-formed frame with
    // `grid.len() >= n` is unaffected, so this is byte-identical on every valid call).
    if grid.len() < n {
        return Vec::new();
    }
    // UNION-FIND connected components — a near-linear (O(n·α)) single pass that replaces the old iterative
    // label-propagation fixed point (O(n·diameter): each sweep moved a label ONE cell, so a snake of diameter
    // ~w+h cost ~w+h full sweeps — the 57 ms/frame the ARC profile pinned as 71% of an action). The result is
    // IDENTICAL: connected components are unique, and the gather below keys on shared roots (not the label VALUE)
    // and orders objects by the same first-seen `0..n` scan, so `label[cell]` = `find(cell)` here yields the same
    // partition, same object order, same cells, same bbox — byte-identical output, just without the iteration.
    let mut label: Vec<usize> = (0..n).collect(); // parent array; a root points to itself
                                                  // Iterative find with full path compression (no recursion — the grid can be 64×64+ and deep chains form).
    fn find(label: &mut [usize], mut x: usize) -> usize {
        let mut root = x;
        while label[root] != root {
            root = label[root];
        }
        while label[x] != root {
            let next = label[x];
            label[x] = root;
            x = next;
        }
        root
    }
    // One pass: union each non-background cell with its RIGHT and DOWN same-colour neighbour (4-connectivity —
    // right+down cover every edge exactly once since union is symmetric). Attach the larger root index under the
    // smaller so roots trend toward the min cell index (matches the old min-label basins; the gather is
    // value-agnostic regardless, but this keeps roots stable/deterministic).
    for r in 0..height {
        for c in 0..width {
            let cell = r * width + c;
            let col = grid[cell];
            if col == background {
                continue;
            }
            if c + 1 < width {
                let right = cell + 1;
                if grid[right] == col {
                    let (a, b) = (find(&mut label, cell), find(&mut label, right));
                    if a != b {
                        let (hi, lo) = if a > b { (a, b) } else { (b, a) };
                        label[hi] = lo;
                    }
                }
            }
            if r + 1 < height {
                let down = cell + width;
                if grid[down] == col {
                    let (a, b) = (find(&mut label, cell), find(&mut label, down));
                    if a != b {
                        let (hi, lo) = if a > b { (a, b) } else { (b, a) };
                        label[hi] = lo;
                    }
                }
            }
        }
    }
    // Resolve every non-background cell to its component root (final label) before the gather.
    for cell in 0..n {
        if grid[cell] != background {
            label[cell] = find(&mut label, cell);
        }
    }
    // Gather each settled basin (shared final label) into an object, preserving first-seen order.
    // Labels are cell indices in `0..n`, so a dense `Vec` sentinel maps label → object index in O(1) with
    // no hashing or map growth (the old `HashMap<usize,usize>` hashed a value that was already a dense index).
    // First-seen order still comes from the `for cell in 0..n` scan, so object order is byte-identical.
    let mut idx: Vec<usize> = vec![usize::MAX; n];
    let mut objects: Vec<Object> = Vec::new();
    let mut bbox: Vec<(usize, usize, usize, usize)> = Vec::new(); // min_c, min_r, max_c, max_r
    for cell in 0..n {
        if grid[cell] == background {
            continue;
        }
        let lab = label[cell];
        let (r, c) = (cell / width, cell % width);
        let oi = if idx[lab] == usize::MAX {
            idx[lab] = objects.len();
            objects.push(Object {
                cells: Vec::new(),
                color: grid[cell],
                bbox_w: 0,
                bbox_h: 0,
            });
            bbox.push((c, r, c, r));
            objects.len() - 1
        } else {
            idx[lab]
        };
        objects[oi].cells.push(cell);
        let b = &mut bbox[oi];
        b.0 = b.0.min(c);
        b.1 = b.1.min(r);
        b.2 = b.2.max(c);
        b.3 = b.3.max(r);
    }
    for (o, b) in objects.iter_mut().zip(bbox.iter()) {
        o.bbox_w = b.2 - b.0 + 1;
        o.bbox_h = b.3 - b.1 + 1;
    }
    objects
}
