// SPDX-License-Identifier: MIT-0
//! The grown CONCEPT MAP — per-cell positions (`pos_dim`, `place_near`, `seed_coord`), co-activation and
//! co-credit, the metric step, and dimension growth/annealing.

use super::*;

/// A deterministic ±1 entry of the fixed random-PROJECTION matrix — element `(axis, i)` of a Rademacher
/// (Johnson–Lindenstrauss) projection from percept dimension `i` onto concept-map axis `axis`. A pure hash of
/// `(i, axis)`, so it is stable and reproducible and INDEPENDENT of the growth PRNG. Used by
/// [`Field::seed_coord`] to turn a high-dimensional percept into a distance-preserving map coordinate.
fn proj_sign(i: usize, axis: usize) -> f32 {
    // splitmix64 over (i, axis) → a single sign bit
    let mut z = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (axis as u64)
            .wrapping_add(1)
            .wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    if z & 1 == 0 {
        1.0
    } else {
        -1.0
    }
}

impl Field {
    // ── concept map ──
    /// The current dimensionality of the concept-metric (grows on demand; `config.pos_dim_init` — default 2 —
    /// once the map is in use). The `.max(2)` keeps the ≥2 invariant the displacement/metric machinery relies on
    /// even if a tuner sets the floor lower.
    pub fn pos_dim(&self) -> usize {
        if self.pos_dim == 0 {
            self.config.pos_dim_init.max(2)
        } else {
            self.pos_dim
        }
    }

    /// Ensure the concept-metric has a `pos_dim`-vector for every cell (lazy seed with [`scatter`]). `pos_dim`
    /// starts at `config.pos_dim_init` (default 2, floored at 2) the first time the map is touched.
    fn ensure_positions(&mut self) {
        if self.pos_dim == 0 {
            self.pos_dim = self.config.pos_dim_init.max(2);
        }
        let d = self.pos_dim;
        let want = self.mem.medium.len() * d;
        while self.pos.len() < want {
            let flat = self.pos.len();
            let (cell, axis) = (flat / d, flat % d);
            self.pos.push(scatter(cell, axis));
        }
    }

    /// The GROWN position of `cell` — a `pos_dim`-vector in the organism's concept-metric (seed scatter if the
    /// map is untouched). Distance between two cells' positions ≈ how related they are; the FIRST coordinates
    /// are the most general (Matryoshka), so a prefix is a valid coarse position.
    pub fn position(&self, cell: usize) -> Vec<f32> {
        let d = self.pos_dim();
        if (cell + 1) * d <= self.pos.len() {
            self.pos[cell * d..(cell + 1) * d].to_vec()
        } else {
            (0..d).map(|a| scatter(cell, a)).collect()
        }
    }

    /// One coordinate of a cell's position, read straight from the flat buffer (seed scatter if ungrown) — no
    /// allocation. The building block for the hot map loops.
    #[inline]
    pub(super) fn coord(&self, cell: usize, axis: usize) -> f32 {
        let d = self.pos_dim();
        if (cell + 1) * d <= self.pos.len() {
            self.pos[cell * d + axis]
        } else {
            scatter(cell, axis)
        }
    }

    /// Distance between two cells in the concept-metric, using the first `dims` coordinates (`0`/oversized ⇒
    /// all of them). The Matryoshka selector: a small `dims` is a CHEAP COARSE distance (broad relatedness);
    /// the full `pos_dim` is the fine distance. Reads the flat position buffer directly — NO allocation (was
    /// two `Vec`s per call), the hot-loop win for navigation / retrieval / foraging.
    pub fn distance(&self, a: usize, b: usize, dims: usize) -> f32 {
        let d = self.pos_dim();
        let k = if dims == 0 { d } else { dims.min(d) };
        (0..k)
            .map(|axis| {
                let df = self.coord(a, axis) - self.coord(b, axis);
                df * df
            })
            .sum::<f32>()
            .sqrt()
    }

    /// BORN-NEAR-CREATOR — place `cell` at the centroid of its `anchors` (the cells it was grown to serve),
    /// plus a small deterministic jitter so siblings don't collapse. Call it right after growing a feature so
    /// lineage seeds the map: a concept starts life next to what it came from.
    pub fn place_near(&mut self, cell: usize, anchors: &[usize], jitter: f32) {
        self.ensure_positions();
        let d = self.pos_dim;
        if (cell + 1) * d > self.pos.len() || anchors.is_empty() {
            return;
        }
        let n = anchors.len() as f32;
        for axis in 0..d {
            let mut c = 0.0f32;
            for &a in anchors {
                c += self.position(a)[axis];
            }
            self.pos[cell * d + axis] = c / n + jitter * scatter(cell, axis);
        }
    }

    /// PLACE AT — stamp `cell`'s concept-map position directly to `pos` (extra axes zeroed, missing ones
    /// padded 0). Where [`Self::place_near`] seeds by lineage and `metric_step` shapes by usage, this sets an
    /// EXACT key — indexing a skill by an explicit signature (e.g. its [`Self::fingerprint`]) or a supplied
    /// coordinate, so [`Self::route`] can find it. Grows the map to 2-D on first use like the other map ops.
    pub fn place_at(&mut self, cell: usize, pos: &[f32]) {
        self.ensure_positions();
        let d = self.pos_dim;
        if (cell + 1) * d > self.pos.len() {
            return;
        }
        for axis in 0..d {
            self.pos[cell * d + axis] = pos.get(axis).copied().unwrap_or(0.0);
        }
    }

    /// SEED a cell's concept-map coordinate FROM a percept / grown-latent vector — the percept→coordinate seam
    /// (the OM-5 archive key and OM-9 egocentric key). Where [`Self::place_at`] takes an EXPLICIT coordinate,
    /// this DERIVES one from a higher-dimensional percept by a fixed random PROJECTION into the map's `pos_dim`
    /// (Johnson–Lindenstrauss: a random projection approximately preserves relative distances). So distinct
    /// percepts land at distinct coordinates, SIMILAR percepts land NEAR each other, and the mapping is STABLE
    /// and reproducible — the same percept always yields the same coordinate, because the projection is a pure
    /// hash (`proj_sign`), independent of the growth PRNG. This is what turns a repeating world-state (or a
    /// grown latent) into a REPEATING map key: an archive can dedup by [`Self::distance`] instead of minting one
    /// key per raw frame, and `co_activate` / `metric_step` then refine the seeded coordinates by usage.
    ///
    /// It does NOT interpret the percept — feed it a raw frame and you get a distance-preserving hash of the
    /// pixels; feed it a GROWN predictive latent and you get a hash of the abstraction. The quality of the key
    /// is the quality of the percept, which is exactly why the M2 latent (grown, predictive) is the upstream
    /// lever — `seed_coord` is the seam that carries whatever representation the organism grows onto its map.
    pub fn seed_coord(&mut self, cell: usize, percept: &[f32]) {
        self.ensure_positions();
        let d = self.pos_dim;
        if (cell + 1) * d > self.pos.len() {
            return;
        }
        // 1/√D scaling keeps coordinates bounded regardless of percept dimensionality (JL normalization).
        let norm = (percept.len().max(1) as f32).sqrt();
        for axis in 0..d {
            let mut acc = 0.0f32;
            for (i, &p) in percept.iter().enumerate() {
                acc += p * proj_sign(i, axis);
            }
            self.pos[cell * d + axis] = acc / norm;
        }
    }

    /// The cells with a live position among `cells` (those inside the current `pos` buffer).
    fn live_positions(&self, cells: &[usize]) -> Vec<usize> {
        let d = self.pos_dim;
        cells
            .iter()
            .copied()
            .filter(|&c| (c + 1) * d <= self.pos.len())
            .collect()
    }

    /// Pull a co-active set a step `rate` toward its per-axis centroid — the ATTRACTION half shared by
    /// `co_activate` and `metric_step` (no-op for fewer than two cells).
    fn attract(&mut self, live: &[usize], rate: f32) {
        if live.len() < 2 {
            return;
        }
        let d = self.pos_dim;
        let n = live.len() as f32;
        for axis in 0..d {
            let ctr: f32 = live.iter().map(|&c| self.pos[c * d + axis]).sum::<f32>() / n;
            for &c in live {
                let p = &mut self.pos[c * d + axis];
                *p += rate * (ctr - *p);
            }
        }
    }

    /// CO-ACTIVATION ATTRACTION — the cells in `cells` fired together, so pull each a step `rate ∈ [0,1]`
    /// toward their shared centroid, in every current dimension (Hebbian × geometry: co-active ⇒ closer). Over
    /// experience this clusters concepts that are useful together. Light, online, O(cells·dim) — no force field.
    pub fn co_activate(&mut self, cells: &[usize], rate: f32) {
        // co_activate IS attraction with no repulsion — the `n_neg = 0` case of `metric_step`. One
        // implementation: topology-only (attract) vs metric-faithful (attract + repel) is now a single dial.
        self.metric_step(cells, rate, 0);
    }

    /// ACCRUE SECOND-ORDER CO-CREDIT — the pairwise sibling of the per-unit delta rule, and the fix for the
    /// blind spot in [`Self::co_activate`]. Co-activation pulls cells together by CO-OCCURRENCE (Hebbian), but
    /// composition modules are SYNERGIES: two cells that INTERACT to produce the output need not co-fire more
    /// than independent cells (an AND over ±1 bits fires 1/8 of the time whatever module it belongs to, so
    /// co-firing carries almost no module signal — measured separation ~0.004). This accrues, for every pair
    /// `(i, j)` in `cells` (`i < j`), an EMA of `err · x_i · x_j` — precisely the gradient a virtual product
    /// weight `w_ij·x_i·x_j` would receive under the same residual `err` the per-unit rule already uses. A pair
    /// whose PRODUCT consistently explains the residual accumulates large `|co_credit|`; an additive pair
    /// (whose product adds nothing beyond the two individual terms) averages to ~0. Call it each event during
    /// training with the readout's residual, then [`Self::co_credit_step`] to shape the map. Local, online,
    /// O(cells²); deterministic; calling this is the ONLY thing that touches the accumulator (it round-trips
    /// through a snapshot), so every other path stays byte-identical.
    pub fn accrue_co_credit(&mut self, cells: &[usize], err: f32, decay: f32) {
        let n = cells.len();
        for a in 0..n {
            let ci = cells[a];
            let xi = self.mem.medium.get(ci).copied().unwrap_or(0.0);
            for &cj in cells.iter().take(n).skip(a + 1) {
                let xj = self.mem.medium.get(cj).copied().unwrap_or(0.0);
                let key = if ci < cj { (ci, cj) } else { (cj, ci) };
                let sample = err * xi * xj;
                let e = self.co_credit.entry(key).or_insert(0.0);
                *e = decay * *e + (1.0 - decay) * sample;
            }
        }
    }

    /// The accrued second-order co-credit of a cell pair (order-insensitive) — the EMA of `err·x_i·x_j` laid
    /// down by [`Self::accrue_co_credit`]. `0.0` for an unseen pair. Introspection for the synergy map.
    pub fn co_credit_of(&self, a: usize, b: usize) -> f32 {
        let key = if a < b { (a, b) } else { (b, a) };
        self.co_credit.get(&key).copied().unwrap_or(0.0)
    }

    /// SHAPE THE CONCEPT-MAP BY SYNERGY — carve a metric where distance means "how much these cells INTERACT."
    /// Where [`Self::co_activate`] moves a whole co-active SET toward one centroid (co-occurrence), this applies
    /// a SIGNED per-pair force around the mean accrued [`Self::accrue_co_credit`] magnitude: a pair whose
    /// product-synergy is ABOVE the mean is pulled toward its midpoint, one BELOW the mean is pushed apart
    /// (`force = rate · (|co_credit|/max − mean)`). The mean-centred sign is essential — attraction alone
    /// collapses the whole map (the same reason `co_activate` needs `spread_positions` and `metric_step` needs
    /// negative sampling); the repulsion on additive pairs is what SEPARATES synergy modules. Over experience a
    /// downstream [`Self::cluster`] then recovers SYNERGY modules that a co-activation map cannot (an AND-block
    /// pair interacts but does not co-fire more than a cross-block pair). No-op when no co-credit has been
    /// accrued (byte-identical off). Deterministic (iteration is over a sorted key order, not `HashMap` order).
    pub fn co_credit_step(&mut self, rate: f32) {
        if self.frozen || self.co_credit.is_empty() {
            return;
        }
        self.ensure_positions();
        let d = self.pos_dim;
        // Deterministic order (HashMap iteration order is not replay-stable).
        let mut pairs: Vec<((usize, usize), f32)> =
            self.co_credit.iter().map(|(&k, &v)| (k, v)).collect();
        pairs.sort_by_key(|(k, _)| *k);
        let max = pairs.iter().map(|(_, v)| v.abs()).fold(0.0f32, f32::max);
        if max <= 0.0 {
            return;
        }
        // Mean synergy ratio — the pivot: above ⇒ attract (interacting), below ⇒ repel (additive/independent).
        let mean = pairs.iter().map(|(_, v)| v.abs() / max).sum::<f32>() / pairs.len() as f32;
        for ((i, j), v) in pairs {
            if (i + 1) * d > self.pos.len() || (j + 1) * d > self.pos.len() {
                continue;
            }
            let w = rate * (v.abs() / max - mean); // +: pull interacting pair together; −: push additive pair apart
            for axis in 0..d {
                let mid = 0.5 * (self.pos[i * d + axis] + self.pos[j * d + axis]);
                self.pos[i * d + axis] += w * (mid - self.pos[i * d + axis]);
                self.pos[j * d + axis] += w * (mid - self.pos[j * d + axis]);
            }
        }
    }

    /// METRIC-PRESERVING map update — attraction WITH NEGATIVE-SAMPLING REPULSION. [`Self::co_activate`] only
    /// attracts, so it preserves TOPOLOGY (neighbours cluster) but not the METRIC (the map folds; displacements
    /// are inconsistent). This adds the missing force: after pulling the co-active `positive` cells toward their
    /// centroid, push each APART from `n_neg` RANDOM cells with a force strong when close and vanishing when far
    /// (`1/(1+d²)`). That is exactly the word2vec / UMAP rule — and it is WHY those embeddings make distance
    /// faithful and relations CONSISTENT displacements (king−man+woman=queen). Use it in place of
    /// `co_activate` (+`spread_positions`) when the map must be metric-faithful (sharp analogy, planning,
    /// skill-space frontier). Deterministic (the field's own PRNG), so replay is intact.
    pub fn metric_step(&mut self, positive: &[usize], rate: f32, n_neg: usize) {
        if self.frozen {
            return;
        }
        self.ensure_positions();
        let d = self.pos_dim;
        let soft = self.config.metric_repel_softening; // `s` in rate/(s+dist²) — repulsion length-scale
                                                       // ── attraction: pull the co-active set toward its centroid (per axis) — shared with `co_activate`. ──
        let live = self.live_positions(positive);
        self.attract(&live, rate);
        // ── repulsion: push each co-active cell apart from n_neg random NEGATIVES (the metric-carving force). ──
        for &c in &live {
            for _ in 0..n_neg {
                let k = self.rand_cell();
                if k == c || (k + 1) * d > self.pos.len() {
                    continue;
                }
                let mut dist2 = 0.0f32;
                for axis in 0..d {
                    let df = self.pos[c * d + axis] - self.pos[k * d + axis];
                    dist2 += df * df;
                }
                let f = rate / (soft + dist2); // strong when close, ~0 when far
                for axis in 0..d {
                    let df = self.pos[c * d + axis] - self.pos[k * d + axis];
                    self.pos[c * d + axis] += f * df;
                    self.pos[k * d + axis] -= f * df;
                }
            }
        }
    }

    /// GROUP BY SURPRISE-REDUCTION (Gap 7 spike — the grouping face of the ONE currency, [`Config::surprise_currency`]).
    /// The CAUSAL analog of [`Self::co_activate`]. Where `co_activate` pulls a co-active set together UNIFORMLY (co-firing
    /// = correlation, which a COMMON CAUSE inflates — two effects of one cause correlate MORE than a real driver
    /// correlates with its effect), this pulls each PAIR together in proportion to how much BOTH cells reduce the
    /// output's surprise: the force on a pair is `rate · min(SR_i, SR_j)` over the raw per-cell
    /// [`Self::surprise_reduction_of`] (the END-TO-END ablation importance `(∂output/∂c)²·Var(c)`). A pair coheres
    /// only when BOTH members are causally load-bearing for the warmed target, so a noise / redundant member (SR≈0)
    /// cannot drag a group and a SPURIOUS common-cause pair (one member redundant GIVEN the shared cause, so its
    /// conditional sensitivity ≈ 0) does NOT group — exactly where co-firing mis-ranks it (`substrate_causal_grouping`,
    /// `spike_group_by_surprise`). The per-cell SR is relative to the CURRENTLY-warmed output, so warm a target's
    /// predictor (a learn cascade with the flag on) before grouping its pair, and the currency reads as "sensitivity
    /// to THIS target". The gate is the RAW `surprise_reduction_of` (NOT normalized by the group's max) — a redundant
    /// member reads a near-zero absolute SR (`substrate_causal_grouping`: the conditional contribution collapses),
    /// which no in-group rescaling should inflate; `rate` is the step size as for [`Self::co_activate`], the pair's
    /// force is clamped to `< 1` (no overshoot). Gated: a NO-OP touching no `pos` unless `surprise_currency` is on
    /// (else every SR reads 0 ⇒ no force) ⇒ byte-identical when off. O(k²·dim) over the grouped set — for the small
    /// sets grouping runs on.
    pub fn group_by_surprise(&mut self, cells: &[usize], rate: f32) {
        if self.frozen || !self.config.surprise_currency {
            return;
        }
        self.ensure_positions();
        let live = self.live_positions(cells);
        if live.len() < 2 {
            return;
        }
        let d = self.pos_dim;
        // Per-cell surprise-reduction (RAW — a redundant member's absolute SR is ~0 and must stay that way).
        let sr: Vec<f32> = live
            .iter()
            .map(|&c| self.surprise_reduction_of(c))
            .collect();
        if sr.iter().all(|&s| s <= 0.0) {
            return; // nothing has warmed a surprise signal yet — no force (still byte-safe: pos untouched).
        }
        // Pairwise attraction GATED by min(SR_i, SR_j): both members must reduce surprise for the pair to cohere.
        for a in 0..live.len() {
            for b in (a + 1)..live.len() {
                let g = (rate * sr[a].min(sr[b])).clamp(0.0, 0.9);
                if g <= 0.0 {
                    continue;
                }
                let (ci, cj) = (live[a] * d, live[b] * d);
                for axis in 0..d {
                    let mid = 0.5 * (self.pos[ci + axis] + self.pos[cj + axis]);
                    self.pos[ci + axis] += g * (mid - self.pos[ci + axis]);
                    self.pos[cj + axis] += g * (mid - self.pos[cj + axis]);
                }
            }
        }
    }

    /// GROW A DIMENSION — widen the concept-metric by one axis when the current dimensions can't separate
    /// concepts that should be distinct (the map "gets stuck"). Because earlier axes were shaped first (by the
    /// general structure) and the new axis starts as a small distinct jitter that co-activation then shapes to
    /// the residual, the embedding stays NESTED: the new dimension adds a FINER distinction without disturbing
    /// the coarse prefix. Returns the new `pos_dim`.
    pub fn grow_dimension(&mut self) -> usize {
        self.ensure_positions();
        let (d, cells) = (self.pos_dim, self.mem.medium.len());
        let mut wider = Vec::with_capacity(cells * (d + 1));
        for c in 0..cells {
            wider.extend_from_slice(&self.pos[c * d..(c + 1) * d]);
            wider.push(scatter(c, d)); // the new (fine) axis: a small distinct seed
        }
        self.pos = wider;
        self.pos_dim = d + 1;
        d + 1
    }

    /// METABOLIZE A DIMENSION — the dual of [`Self::grow_dimension`]: drop the concept-metric axis that carries
    /// the LEAST separation (lowest variance across cells), but only if that weakest axis is below
    /// `min_var_ratio` of the strongest (so a genuinely-used axis is never lost). Together they are DIMENSION
    /// ANNEALING: the map EXPANDS an axis when concepts are confused ([`Self::grow_dimension`] on ambiguity) to
    /// unfold and separate them, then CONTRACTS the now-redundant axes here — the separation it grew survives the
    /// projection, so the representation relaxes back to a compact one. Never drops below `pos_dim` 2. Returns the
    /// axis dropped, or `None` if all axes carry enough (nothing metabolized). O(cells·dim).
    #[doc(hidden)]
    pub fn metabolize_dimension(&mut self, min_var_ratio: f32) -> Option<usize> {
        if self.frozen {
            return None; // structural change — only while ALIVE (a crystallised map is fixed)
        }
        self.ensure_positions();
        let (d, cells) = (self.pos_dim, self.mem.medium.len());
        if d <= 2 || cells == 0 {
            return None;
        }
        let n = cells as f32;
        let var: Vec<f32> = (0..d)
            .map(|axis| {
                let mean: f32 = (0..cells).map(|c| self.pos[c * d + axis]).sum::<f32>() / n;
                (0..cells)
                    .map(|c| (self.pos[c * d + axis] - mean).powi(2))
                    .sum::<f32>()
                    / n
            })
            .collect();
        let maxv = var.iter().copied().fold(0.0f32, f32::max);
        if maxv <= 0.0 {
            return None;
        }
        let (weak, weakv) = var
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(core::cmp::Ordering::Equal))
            .map(|(i, &v)| (i, v))
            .unwrap();
        if weakv >= min_var_ratio * maxv {
            return None; // even the weakest axis carries enough separation — keep them all
        }
        // rebuild the position buffer without the weak axis
        let mut narrower = Vec::with_capacity(cells * (d - 1));
        for c in 0..cells {
            for axis in 0..d {
                if axis != weak {
                    narrower.push(self.pos[c * d + axis]);
                }
            }
        }
        self.pos = narrower;
        self.pos_dim = d - 1;
        Some(weak)
    }

    /// Re-standardize the concept-metric to zero mean and unit spread PER AXIS. Co-activation attraction alone
    /// would shrink a connected map toward a point; this restores scale while PRESERVING the relative structure
    /// it grew (the cheap O(cells·dim) alternative to an O(cells²) repulsive force). Call periodically.
    #[doc(hidden)]
    pub fn spread_positions(&mut self) {
        self.ensure_positions();
        let (d, cells) = (self.pos_dim, self.mem.medium.len());
        if cells == 0 {
            return;
        }
        let n = cells as f32;
        for axis in 0..d {
            let mean: f32 = (0..cells).map(|c| self.pos[c * d + axis]).sum::<f32>() / n;
            let var: f32 = (0..cells)
                .map(|c| (self.pos[c * d + axis] - mean).powi(2))
                .sum::<f32>()
                / n;
            let sd = var.sqrt().max(1e-6);
            for c in 0..cells {
                let p = &mut self.pos[c * d + axis];
                *p = (*p - mean) / sd;
            }
        }
    }
}
