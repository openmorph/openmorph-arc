// SPDX-License-Identifier: MIT-0
//! The prime/tree STRUCTURE read-algebra over the nested composition graph (`structure_key`, `gcs`,
//! `valuation`).

use super::*;

// FNV-1a 64-bit — the hand-rolled, zero-dep hash behind the STRUCTURE KEY (the prime/tree read-algebra). Folds a
// value byte-by-byte, deterministic and machine-independent, so a structural fingerprint replays identically.
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
/// A distinct marker for a NESTED (sub-morph) unit, kept far from any small fan-in so a leaf never collides with it.
const STRUCT_NESTED_TAG: u64 = 0x4E45_5354_4544_0001;
#[inline]
fn fnv_fold(mut h: u64, x: u64) -> u64 {
    for b in x.to_le_bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}
/// The sorted-multiset INTERSECTION of two SORTED slices (shared elements WITH multiplicity) — the kernel of
/// `gcs`/`coprime`. O(n+m), no allocation beyond the result.
fn multiset_intersect(a: &[u64], b: &[u64]) -> Vec<u64> {
    let (mut i, mut j, mut out) = (0usize, 0usize, Vec::new());
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            core::cmp::Ordering::Less => i += 1,
            core::cmp::Ordering::Greater => j += 1,
            core::cmp::Ordering::Equal => {
                out.push(a[i]);
                i += 1;
                j += 1;
            }
        }
    }
    out
}

impl Field {
    // ── The PRIME/TREE STRUCTURE READ-ALGEBRA (#172) ─────────────────────────────────────────────────────────
    // The composition graph — a morph's `nested` sub-morphs, recursively — already IS a prime factorization:
    // `encapsulate`/`nest` extract/multiply a factor, a flat morph is an irreducible atom. These read-queries
    // expose the "prime world" (factor / greatest-common-subtree = the structural gcd / coprimality = independence
    // / valuation = nesting depth) as GROWN facts about that graph — the discrete, exact, ADDITIVE companion to the
    // continuous Euclidean behaviour metric (`embed`). Nothing authored; nothing materialized as a bignum (the
    // ed25519-safe blake3 content-address is the cross-morph key — see mycelium); the substrate learning metric is
    // untouched (these are `&self` reads). Behaviour is `embed`'s job; this is SHAPE.

    /// STRUCTURE KEY — a recursive, order-invariant hash of this morph's COMPOSITION-TREE SHAPE: the cell count,
    /// and per unit (leaf or nested alike) its host fan-in and which second-order capabilities it uses
    /// (gate/factor/activation), plus, for a nested unit, its sub-morph's OWN structure key. NOT its weights — two
    /// structurally-identical morphs share a key regardless of unit order or what they weigh (behaviour is
    /// [`Self::embed`]'s job). The in-core, hand-rolled (FNV-1a, zero-dep, deterministic) stand-in for the mycelium
    /// blake3 content-address: the "prime" identity of a structure.
    ///
    /// ORDER SEMANTICS (deliberate): the tokens are sorted PER LEVEL inside the recursion, so this is the canonical
    /// key of the UNORDERED rooted composition tree (an AHU-style tree hash). A morph's units are a set — async,
    /// fire-on-change, insertion order carries no meaning — so there is no planar order for the key to see.
    /// `(AB)C` and `A(BC)` collide exactly when they are isomorphic as unordered trees (e.g. every leaf has one
    /// shape) and are separated when they are not (`{{A,B},C}` vs `{A,{B,C}}` for distinct shapes). Up to 64-bit
    /// hash collisions and the `struct_walk_max_depth` truncation, trees that differ in these labels (cell count,
    /// per-unit fan-in and capability flags, nesting) get distinct keys. Which cell feeds which (wiring, and so
    /// sequence order) is NOT in the key: see [`Self::structural_embed`] and [`Self::embed`]. On a NESTED unit a
    /// forward `factor` and the number of extra host cells it writes (`nest_multi`) change what the host sees; its
    /// host fan-in is a shape fact that changes the value only when it differs from the sub-morph's source count
    /// (reads beyond the sources change what re-fires it, not what it computes), and its activation and gate flags
    /// are inert there (a nested cell fires its sub-morph and is skipped by learning). They are hashed so leaf and
    /// nested tokens carry the same shape facts.
    pub fn structure_key(&self) -> u64 {
        self.structure_key_depth(self.config.struct_walk_max_depth)
    }
    fn structure_key_depth(&self, depth: usize) -> u64 {
        // SORTED per-unit tokens ⇒ invariant to unit insertion order (the UNORDERED prime factorization).
        let mut tokens: Vec<u64> = Vec::with_capacity(self.units.len());
        for u in &self.units {
            // Every unit, leaf or nested: its host fan-in + which second-order capabilities it uses (shape, not
            // weights).
            let flags = (u.gate.is_some() as u64)
                | ((u.factor.is_some() as u64) << 1)
                | ((u.activation.is_some() as u64) << 2);
            let shape = fnv_fold(fnv_fold(FNV_OFFSET, u.reads.len() as u64), flags);
            let tok = if let Some(nested) = &u.nested {
                let sub = if depth > 0 {
                    nested.sub.structure_key_depth(depth - 1)
                } else {
                    0 // recursion guard: a pathologically deep tower is truncated, not walked forever
                };
                // nested: the tag, the same shape facts a leaf carries, how many EXTRA host cells it writes
                // (`nest_multi`: host-visible outputs), then the sub-morph's own key.
                fnv_fold(
                    fnv_fold(
                        fnv_fold(fnv_fold(FNV_OFFSET, STRUCT_NESTED_TAG), shape),
                        nested.extra.len() as u64,
                    ),
                    sub,
                )
            } else {
                shape
            };
            tokens.push(tok);
        }
        tokens.sort_unstable();
        let mut h = fnv_fold(FNV_OFFSET, self.mem.medium.len() as u64); // the frame (cell count)
        for t in &tokens {
            h = fnv_fold(h, *t);
        }
        h
    }

    /// FACTORIZATION — the sorted multiset of this morph's IMMEDIATE nested sub-morph [`Self::structure_key`]s:
    /// its direct PRIME FACTORS (the sub-morphs it is composed of). Empty for a flat (un-nested) morph — an
    /// irreducible atom. The composition graph IS the factorization; this reads it off, nothing authored.
    #[doc(hidden)]
    pub fn factorization(&self) -> Vec<u64> {
        let mut factors: Vec<u64> = self
            .units
            .iter()
            .filter_map(|u| u.nested.as_ref())
            .map(|n| n.sub.structure_key())
            .collect();
        factors.sort_unstable();
        factors
    }

    /// GREATEST COMMON SUBTREE — the sorted multiset of sub-morph [`Self::structure_key`]s shared by two morphs'
    /// IMMEDIATE [`Self::factorization`]s: the substructure they have in COMMON (the composability signal —
    /// shared reusable machinery). The structural "gcd", by multiset intersection. Empty ⇒ [`Self::coprime`].
    #[doc(hidden)]
    pub fn gcs(&self, other: &Field) -> Vec<u64> {
        multiset_intersect(&self.factorization(), &other.factorization())
    }

    /// COPRIME — two morphs share NO immediate sub-morph ([`Self::gcs`] empty): structurally INDEPENDENT (the
    /// speciation signal). NOTE: this is SYNTACTIC independence — two behaviourally-identical sub-morphs grown
    /// from different seeds have different structure keys and read coprime though they compute the same function.
    /// Gate through behavioural dedup ([`Self::embed_distance`]) first where SEMANTIC identity is what matters.
    #[doc(hidden)]
    pub fn coprime(&self, other: &Field) -> bool {
        self.gcs(other).is_empty()
    }

    /// VALUATION — the p-adic `v_p` analog: how deeply (max nesting depth; an immediate child is depth 1) and how
    /// many times (multiplicity) a sub-morph with [`Self::structure_key`] `subkey` occurs ANYWHERE in this morph's
    /// composition tree. `(0, 0)` if absent. A recursive read over the `nested` tree, nothing authored.
    #[doc(hidden)]
    pub fn valuation(&self, subkey: u64) -> (usize, usize) {
        let (mut max_depth, mut count) = (0usize, 0usize);
        self.valuation_walk(
            subkey,
            1,
            self.config.struct_walk_max_depth,
            &mut max_depth,
            &mut count,
        );
        (max_depth, count)
    }
    fn valuation_walk(
        &self,
        subkey: u64,
        depth: usize,
        budget: usize,
        max_depth: &mut usize,
        count: &mut usize,
    ) {
        if budget == 0 {
            return;
        }
        for u in &self.units {
            if let Some(n) = &u.nested {
                if n.sub.structure_key() == subkey {
                    *count += 1;
                    *max_depth = (*max_depth).max(depth);
                }
                n.sub
                    .valuation_walk(subkey, depth + 1, budget - 1, max_depth, count);
            }
        }
    }
}
