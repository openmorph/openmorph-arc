// SPDX-License-Identifier: MIT-0
//! The FRACTAL INDEX over the concept map — `cluster`, `index_skills`, `route`, `cophenetic`.

use super::*;

/// Whether a [`SkillIndex`] subtree contains `cell` (walks to the leaves).
fn skill_subtree_contains(idx: &SkillIndex, cell: usize) -> bool {
    match idx {
        SkillIndex::Leaf(cells) => cells.contains(&cell),
        SkillIndex::Branch { kids, .. } => kids.iter().any(|k| skill_subtree_contains(k, cell)),
    }
}
/// A [`SkillIndex`] subtree's height in levels-from-leaf (a leaf is 0).
fn skill_subtree_height(idx: &SkillIndex) -> usize {
    match idx {
        SkillIndex::Leaf(_) => 0,
        SkillIndex::Branch { kids, .. } => {
            1 + kids.iter().map(skill_subtree_height).max().unwrap_or(0)
        }
    }
}
/// The cophenetic height of `a`,`b` on a region tree: descend while BOTH live in the same child (they merge
/// deeper), else this region is their lowest common ancestor and its height is the distance. See
/// [`Field::cophenetic`].
fn cophenetic_walk(idx: &SkillIndex, a: usize, b: usize) -> usize {
    match idx {
        SkillIndex::Leaf(_) => 0,
        SkillIndex::Branch { kids, .. } => {
            for k in kids {
                if skill_subtree_contains(k, a) && skill_subtree_contains(k, b) {
                    return cophenetic_walk(k, a, b); // both in one child ⇒ they merge deeper
                }
            }
            skill_subtree_height(idx) // this region is their LCA
        }
    }
}

impl Field {
    /// FARTHEST-POINT clustering over the grown concept-map — the ONE self-similar operator behind
    /// both the fractal router index AND self-decomposition (previously copy-pasted across ~6 test harnesses).
    /// Picks `k` farthest-apart members of `ids` (seed 0 = `ids[0]`, then repeatedly the member whose *minimum*
    /// distance to the current seeds is *maximal*), assigns each id to its nearest seed, and drops empty groups.
    /// Distances are the full concept-metric ([`Self::distance`]); deterministic (`total_cmp` tie-breaks).
    pub fn cluster(&self, ids: &[usize], k: usize) -> Vec<Vec<usize>> {
        if ids.is_empty() {
            return Vec::new();
        }
        // Dedup (first-seen order) so the farthest-point seed loop can never exhaust the distinct pool and
        // `max_by(..).unwrap()` on an empty filter panic. A no-op for unique inputs — byte-identical there.
        let deduped: Vec<usize> = {
            let mut seen: Vec<usize> = Vec::with_capacity(ids.len());
            for &i in ids {
                if !seen.contains(&i) {
                    seen.push(i);
                }
            }
            seen
        };
        let ids: &[usize] = &deduped;
        let mut seeds = vec![ids[0]];
        while seeds.len() < k.min(ids.len()) {
            let next = *ids
                .iter()
                .filter(|i| !seeds.contains(i))
                .max_by(|&&a, &&b| {
                    let da = seeds
                        .iter()
                        .map(|&s| self.distance(a, s, 0))
                        .fold(f32::MAX, f32::min);
                    let db = seeds
                        .iter()
                        .map(|&s| self.distance(b, s, 0))
                        .fold(f32::MAX, f32::min);
                    da.total_cmp(&db)
                })
                .unwrap();
            seeds.push(next);
        }
        let mut groups = vec![Vec::new(); seeds.len()];
        for &i in ids {
            let b = (0..seeds.len())
                .min_by(|&a, &b| {
                    self.distance(i, seeds[a], 0)
                        .total_cmp(&self.distance(i, seeds[b], 0))
                })
                .unwrap();
            groups[b].push(i);
        }
        groups.into_iter().filter(|g| !g.is_empty()).collect()
    }

    /// Build the fractal [`SkillIndex`] over `skills` (their concept-map positions): recursively [`Self::cluster`]
    /// into ≤`branching` groups until a group has ≤`leaf` members. `O(M log M)`. A degenerate cluster (≤1 group)
    /// stops as a leaf. Rebuild after adding skills to a live library.
    pub fn index_skills(&self, skills: &[usize], leaf: usize, branching: usize) -> SkillIndex {
        if skills.len() <= leaf.max(1) {
            return SkillIndex::Leaf(skills.to_vec());
        }
        let groups = self.cluster(skills, branching);
        if groups.len() <= 1 {
            return SkillIndex::Leaf(skills.to_vec());
        }
        SkillIndex::Branch {
            centroids: groups
                .iter()
                .map(|g| {
                    let d = self.pos_dim();
                    let mut c = vec![0.0f32; d];
                    for &i in g {
                        let p = self.position(i);
                        for (ck, pk) in c.iter_mut().zip(p.iter()) {
                            *ck += *pk;
                        }
                    }
                    for x in &mut c {
                        *x /= g.len() as f32;
                    }
                    c
                })
                .collect(),
            kids: groups
                .iter()
                .map(|g| self.index_skills(g, leaf, branching))
                .collect(),
        }
    }

    /// ROUTE a query POSITION to the nearest skill cell by coarse-to-fine descent of `index` — at each `Branch`
    /// recurse into the `beam` nearest centroids (L2 over the Matryoshka `dims` prefix; `0` ⇒ full `pos_dim`),
    /// keeping the globally nearest returned skill; at a `Leaf`, argmin over the members. Returns
    /// `(nearest_skill_cell, comparisons)` — the comparison count is ~O(√M), not the flat O(M). `beam=1` is the
    /// cheapest single-region descent (lossy at region borders); `beam=2` recovers nearly all the flat-nearest
    /// for a small extra cost. Decide the winner with this (no settle), then drive the query only into the
    /// winner's read band so fire-on-change settles ONLY that sub-morph — the library settles the active path.
    /// An index over no skills returns cell `0` (debug-asserted — `0` is a real cell).
    pub fn route(
        &self,
        index: &SkillIndex,
        query: &[f32],
        beam: usize,
        dims: usize,
    ) -> (usize, usize) {
        // L2 over a Matryoshka prefix of a query vector against a cell's grown position (no allocation).
        let dq = |cell: usize| -> f32 {
            let k = if dims == 0 {
                self.pos_dim()
            } else {
                dims.min(self.pos_dim())
            };
            (0..k.min(query.len()))
                .map(|a| {
                    let df = self.coord(cell, a) - query[a];
                    df * df
                })
                .sum::<f32>()
                .sqrt()
        };
        // L2 over a Matryoshka prefix of a query vector against a centroid vector.
        let dc = |c: &[f32]| -> f32 {
            let k = if dims == 0 {
                c.len()
            } else {
                dims.min(c.len())
            };
            (0..k.min(query.len()))
                .map(|a| {
                    let df = c[a] - query[a];
                    df * df
                })
                .sum::<f32>()
                .sqrt()
        };
        match index {
            SkillIndex::Leaf(ids) => {
                debug_assert!(
                    !ids.is_empty(),
                    "route: empty leaf — falls back to cell 0 (index built over no skills?)"
                );
                let best = ids
                    .iter()
                    .copied()
                    .min_by(|&a, &b| dq(a).total_cmp(&dq(b)))
                    .unwrap_or(0);
                (best, ids.len())
            }
            SkillIndex::Branch { centroids, kids } => {
                let mut order: Vec<usize> = (0..centroids.len()).collect();
                order.sort_by(|&a, &b| dc(&centroids[a]).total_cmp(&dc(&centroids[b])));
                let mut cost = centroids.len();
                let mut best: Option<(usize, f32)> = None;
                for &g in order.iter().take(beam.max(1)) {
                    let (cell, c) = self.route(&kids[g], query, beam, dims);
                    cost += c;
                    let d = dq(cell);
                    if best.is_none_or(|(_, bd)| d < bd) {
                        best = Some((cell, d));
                    }
                }
                debug_assert!(best.is_some(), "route: empty branch — falls back to cell 0");
                (best.map(|(c, _)| c).unwrap_or(0), cost)
            }
        }
    }

    /// COPHENETIC distance on the grown region tree — the height (levels-from-leaf) of the SHALLOWEST region of a
    /// [`Self::index_skills`] tree that contains BOTH cells. An ULTRAMETRIC (it uses MAX, not `+`, so every
    /// triangle is isosceles — the strong triangle inequality) READ-DISTANCE over the emergent cluster hierarchy:
    /// two cells in the same leaf are `0` apart, cells that meet only at the root are maximally apart, and all
    /// WITHIN-region variation COLLAPSES to the region's height. This is the discrete, exact, gradient-free
    /// RETRIEVAL side of the "both worlds" design — the safe cousin of a hyperbolic embedding — and it NEVER
    /// touches the Euclidean LEARNING metric (`&self` read: no `pos`/`distance`/`metric_step`, so the displacement
    /// faculties are untouched). It is DISTINCT from [`Self::valuation`]: cophenetic is over the CLUSTER/region
    /// tree (emergent from positions), valuation over the COMPOSITION/`nested` tree (grown by composition) — keep
    /// both. A cell absent from `index` reads as maximally distant (the tree height). Stale between
    /// [`Self::index_skills`] rebuilds, exactly like [`Self::route`].
    #[doc(hidden)]
    pub fn cophenetic(&self, index: &SkillIndex, a: usize, b: usize) -> usize {
        cophenetic_walk(index, a, b)
    }
}
