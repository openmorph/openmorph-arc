// SPDX-License-Identifier: MIT-0
//! THE FRACTAL SKILL LOOP — skill memory (the basin key) SELECTS which nested morph to REUSE, and reuse IS
//! nesting (micro→macro construction). This closes the loop MorphBot's skill-memory ask opens: a basin answers
//! "WHICH skill for this state?"; the skill itself is an `encapsulate`d sub-morph; and `nest` plants it as one
//! addressable cell fired in the active cascade. No new core primitive — it composes the concept-map key
//! ([`Field::seed_coord`]/[`Field::distance`]) with the fractal collapse ([`Field::encapsulate`]/[`Field::nest`]),
//! which is exactly Sébastien's "micro↔macro expansion and construction" as the substrate of efficient reuse.
//!
//!   1. KEYED REUSE (micro) — perceive a state → seed_coord → nearest basin key → NEST the selected skill → apply.
//!      The right skill is selected from an unseen near-miss state, and the nested skill computes BIT-for-BIT what
//!      the standalone skill computes (reuse is exact, not an approximation).
//!   2. MACRO = NEST-OF-NESTS — a skill-of-skills built by nesting two micro skills, `encapsulate`d (carrying its
//!      nested cells verbatim — fractal encapsulation) into ONE macro node, computes the composition exactly, and
//!      is itself a first-class keyable skill retrieved among the micros. Micro→macro, with no floor.
//!
//! Each fractal piece is separately proven elsewhere (`substrate_nest`/`_decompose`/`_scale`/`_memo`); this proof
//! is the INTEGRATION — the basin as the selector over a nested-skill library.
//!
//!   cargo test -p openmorph-core --test proofs substrate_skill_fractal -- --nocapture

use openmorph_core::Field;

use crate::common::Rng;

const D: usize = 12;
const K: usize = 5; // skills in the library

/// A skill's triggering-state prototype (overlapping shared base + skill-specific signature — the realistic case).
fn proto(skill: usize) -> Vec<f32> {
    (0..D)
        .map(|i| {
            let base = if (i * 7 + 1) % 3 == 0 { 1.0 } else { -1.0 };
            // a per-skill "hot window" that genuinely shifts with skill (D and the stride are set so the K
            // windows are distinct — a signature that varies, unlike a modulus that collapses at K | stride).
            let spec = if (i + skill * 2) % D < 4 { 1.0 } else { -1.0 };
            0.7 * base + 0.5 * spec
        })
        .collect()
}
fn observe(skill: usize, noise: f32, rng: &mut Rng) -> Vec<f32> {
    proto(skill)
        .into_iter()
        .map(|p| p + noise * (rng.f() - 0.5) * 2.0)
        .collect()
}

/// Micro skill k: a leaf sub-morph `s_k(x) = tanh(a_k · x)`, cells 0=in, 1=out. Distinct gain per skill so the
/// skills genuinely differ (a wrong selection gives a wrong output).
fn micro_skill(k: usize) -> Field {
    let a = 0.4 + 0.25 * k as f32;
    let mut s = Field::new(2);
    s.add_unit(vec![0], vec![a], 0.0, 1);
    s
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
// 1. KEYED REUSE — the basin selects which nested skill to reuse; the reuse is exact.
// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
#[test]
fn skill_reuse_is_keyed_nesting() {
    const BUDGET: usize = 40;
    // Host layout: [0..K) basin keys · K query scratch · K+1 shared input · [K+2 .. K+2+K) per-skill outputs.
    let key = |k: usize| k;
    let query = K;
    let input = K + 1;
    let out = |k: usize| K + 2 + k;

    let mut host = Field::new(2 * K + 2);
    host.config_mut().pos_dim_init = 8; // room for K distinct basin keys (Matryoshka)
    host.rng_seed(0x5C111);

    // Carve each skill's basin key from its prototype, and NEST the skill body as a reusable cell.
    for k in 0..K {
        host.seed_coord(key(k), &proto(k));
        host.nest(micro_skill(k), &[input], out(k), BUDGET);
    }

    // Select-and-reuse from unseen near-miss states.
    let mut rng = Rng::new(0xFEED);
    let (mut correct, mut exact, trials) = (0usize, 0usize, 400usize);
    for _ in 0..trials {
        let skill = (rng.u() % K as u64) as usize;
        let q = observe(skill, 0.30, &mut rng);
        // Retrieve: nearest basin key to the query = the selected skill.
        host.seed_coord(query, &q);
        let sel = (0..K)
            .min_by(|&a, &b| {
                host.distance(query, a, 0)
                    .partial_cmp(&host.distance(query, b, 0))
                    .unwrap()
            })
            .unwrap();
        if sel == skill {
            correct += 1;
        }
        // REUSE = nest: drive the shared input, read the SELECTED skill's nested output; compare BIT-for-BIT to
        // the standalone skill applied to the same input (reuse is exact, not an approximation).
        let x = 0.6 * (rng.f() - 0.5);
        host.stimulate(&[(input, x)], BUDGET);
        let reused = host.cells()[out(sel)];
        let mut standalone = micro_skill(sel);
        standalone.stimulate(&[(0, x)], BUDGET);
        if reused.to_bits() == standalone.cells()[1].to_bits() {
            exact += 1;
        }
    }
    let acc = correct as f32 / trials as f32;
    eprintln!(
        "[SKILL-FRACTAL] keyed reuse: selection {acc:.3} (chance {:.2}), nested output bit-exact vs standalone {exact}/{trials}",
        1.0 / K as f32
    );
    assert!(
        acc > 0.70,
        "the basin must select the right skill for an unseen state well above chance: {acc:.3}"
    );
    assert_eq!(
        exact, trials,
        "a NESTED (reused) skill must compute bit-for-bit what the standalone skill computes"
    );
    eprintln!("[SKILL-FRACTAL] VERDICT (1): reuse is keyed nesting — the basin answers WHICH skill, `nest` reuses its exact grown computation in the cascade. perceive→key→body→reuse, zero new primitive.");
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
// 2. MACRO = NEST-OF-NESTS — micro→macro construction, exact, and itself a keyable skill.
// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
#[test]
fn macro_skill_is_nest_of_nests() {
    const BUDGET: usize = 40;
    // Build a macro s_b ∘ s_a by nesting two micros, then encapsulate the composed subgraph into ONE node.
    let (a, b) = (1usize, 3usize); // two distinct micro skills
    let mut composed = Field::new(3); // 0=in, 1=mid, 2=out
    composed.nest(micro_skill(a), &[0], 1, BUDGET); // mid = s_a(in)
    composed.nest(micro_skill(b), &[1], 2, BUDGET); // out = s_b(mid)
                                                    // Encapsulate the composition — carries BOTH nested cells verbatim (fractal encapsulation).
    let macro_skill = composed.encapsulate(&[1, 2]);

    // Reuse the macro by nesting it into a fresh top host — a nest OF nests.
    let mut top = Field::new(2); // 0=in, 1=out
    top.nest(macro_skill, &[0], 1, BUDGET);

    // Verify: the macro computes s_b(s_a(x)) EXACTLY, matching the standalone two-stage composition bit-for-bit.
    let mut rng = Rng::new(0x2AC0);
    let mut worst = 0.0f32;
    for _ in 0..200 {
        let x = 0.6 * (rng.f() - 0.5);
        top.stimulate(&[(0, x)], BUDGET);
        let macro_out = top.cells()[1];
        // Reference: the same composition run standalone stage-by-stage.
        let mut sa = micro_skill(a);
        sa.stimulate(&[(0, x)], BUDGET);
        let mut sb = micro_skill(b);
        sb.stimulate(&[(0, sa.cells()[1])], BUDGET);
        worst = worst.max((macro_out - sb.cells()[1]).abs());
    }
    eprintln!("[SKILL-FRACTAL] macro = s_{b}∘s_{a} via nest-of-nests: worst composition error {worst:.2e}");
    assert!(
        worst < 1e-6,
        "a macro built by nesting micros must compute the composition exactly: {worst:.2e}"
    );

    // The macro is a FIRST-CLASS keyable skill: place it on a map among the micros and retrieve it by its key.
    let mut lib = Field::new(K + 2); // K micro keys + macro key + query scratch
    lib.config_mut().pos_dim_init = 8;
    lib.rng_seed(0x11B);
    for k in 0..K {
        lib.seed_coord(k, &proto(k));
    }
    // The macro's triggering state = a distinct signature; place its key and retrieve it back.
    let macro_key = K;
    let query = K + 1;
    let macro_sig: Vec<f32> = (0..D)
        .map(|i| if i % 2 == 0 { 1.4 } else { -1.4 })
        .collect();
    lib.seed_coord(macro_key, &macro_sig);
    lib.seed_coord(query, &macro_sig); // a clean query of the macro's own state
    let sel = (0..=K)
        .min_by(|&a, &b| {
            lib.distance(query, a, 0)
                .partial_cmp(&lib.distance(query, b, 0))
                .unwrap()
        })
        .unwrap();
    assert_eq!(
        sel, macro_key,
        "the macro must be retrievable as a first-class skill among the micros"
    );
    eprintln!("[SKILL-FRACTAL] VERDICT (2): micro→macro is nest-of-nests — encapsulate carries the nested cells verbatim, the composed skill computes exactly ({worst:.1e}), and the macro is itself a keyable basin. One recursive type, no floor.");
}
