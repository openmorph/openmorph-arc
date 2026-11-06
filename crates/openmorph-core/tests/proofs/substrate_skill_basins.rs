// SPDX-License-Identifier: MIT-0
//! SKILL MEMORY AS REWARD-CARVED ATTRACTOR-BASINS (MorphBot ask, 2026-08-16) — and the honest finding that it
//! needs NO new core primitive: it FALLS OUT of the existing concept-map + value machinery.
//!
//! MorphBot asked for skills held as continuous attractor-basins on the grown concept-map, naming two "missing"
//! field ops: (1) reward-weighted basin CARVING (deepen the landscape around a coordinate ∝ reward·recency) and
//! (2) LATERAL INHIBITION / competitive normalisation so distinct skill basins stay separable — plus a
//! `skill = basin + effect-policy` binding that snapshots with the morph. The truth-not-trophy answer is that
//! openmorph ALREADY has all four pieces, so we PROVE the composition rather than add machinery (simplest is
//! most beautiful — a new special-case method would be a smell):
//!
//! * carve (reward-weighted deposit) = [`Field::reinforce`] — `drive[cell] += reward`, persistent (serde).
//! * basin retrieval / generalisation = [`Field::seed_coord`] (percept→coordinate, similar→near, a JL hash) then
//!   nearest-by-[`Field::distance`] — a near-miss state descends to the nearest carved basin (exact-match hashes miss).
//! * lateral inhibition (separability) = [`Field::metric_step`]'s NEGATIVE-SAMPLING REPULSION (the word2vec / UMAP
//!   force) — but see (B) below: it is NOT needed here and HURTS.
//! * skill = basin + policy, persistent = the `(coordinate, effect-policy)` pair; `pos` / `drive` are serde, so
//!   the skill memory snapshots WITH the morph.
//!
//! Four measurements, each scored INSIDE (truth-not-trophy):
//! A. GENERALISATION — held-out noisy near-miss states retrieve the right skill far above chance; a SHUFFLED-
//!      coordinate control (the map's percept→coordinate binding destroyed) collapses to chance ⇒ the grown
//!      geometry is load-bearing (basins generalise; the binding is what carries it).
//! B. LATERAL INHIBITION IS UNNECESSARY — AND HARMFUL BOLTED ON (the honest-negative answer to ask (2)).
//!      Because seed_coord projects the QUERY and the KEYS through the SAME JL projection, nearest-key retrieval
//!      is ALREADY as separable as the percepts allow — the separability is intrinsic to the shared projection,
//!      not something a map force adds. A round of metric_step negative-sampling repulsion (the natural
//!      "lateral inhibition") DISCARDS that JL structure and collapses retrieval to chance. So ask (2) is not a
//!      missing primitive: the separability CEILING is percept distinctness (the M2 latent feeding seed_coord),
//!      never a repulsion force on the map.
//! C. REWARD-WEIGHTING (carve = reinforce) — `drive_of` deepens monotonically with reward, and a value-weighted
//!      descent (`−distance + λ·drive`) resolves a tie between two equidistant basins toward the MORE-REWARDED
//!      one where pure distance is a coin-flip.
//! D. PERSISTENCE — a serde round-trip leaves positions, drive, and retrieval bit-unchanged: the skill memory
//!      lives IN the morph.
//!
//!   cargo test -p openmorph-core --test proofs substrate_skill_basins -- --nocapture

use openmorph_core::Field;

use crate::common::Rng;

const D: usize = 12; // percept dimensionality (the raw state vector)
const K: usize = 6; // number of distinct skills

/// A skill's prototype percept. To make separability (B) load-bearing, the prototypes OVERLAP: a shared base
/// pattern plus a small skill-specific signature — so seed_coord places them CLOSE (the hard case where naive
/// basins merge). `sig` scales the skill-specific part; small `sig` ⇒ heavy overlap.
fn proto(skill: usize, sig: f32) -> Vec<f32> {
    (0..D)
        .map(|i| {
            let base = if (i * 7 + 1) % 3 == 0 { 1.0 } else { -1.0 }; // shared across all skills
            let spec = if (i + skill * 5) % K < 2 { 1.0 } else { -1.0 }; // skill-specific
            0.7 * base + sig * spec
        })
        .collect()
}

/// A noisy observation of `skill` — the prototype plus fresh Gaussian-ish noise (a near-miss state).
fn observe(skill: usize, sig: f32, noise: f32, rng: &mut Rng) -> Vec<f32> {
    proto(skill, sig)
        .into_iter()
        .map(|p| p + noise * (rng.f() - 0.5) * 2.0)
        .collect()
}

/// Build a skill-memory morph: K skill-key cells (0..K) carved by reward, plus one scratch query cell (K).
/// `repel` toggles the lateral-inhibition step (metric_step repulsion). `reward` per skill drives the carve depth.
fn build(sig: f32, repel: bool, reward: &[f32], seed: u64) -> Field {
    let mut f = Field::new(K + 1);
    // Give the concept-map room for K distinct skills: a wider JL projection separates them better (Matryoshka —
    // the map grows finer axes when it needs resolution). seed_coord projects into every `pos_dim` axis.
    f.config_mut().pos_dim_init = 8;
    f.rng_seed(seed);
    // Carve each skill's basin: seed its coordinate from the (clean) prototype, deposit reward-weighted drive.
    for k in 0..K {
        f.seed_coord(k, &proto(k, sig));
    }
    // Reward-weighted carving (reinforce) over several rewarded visits — recency/repetition deepen the basin.
    for _ in 0..8 {
        for (k, &r) in reward.iter().enumerate() {
            f.reinforce(k, r);
        }
    }
    // Lateral inhibition: push each basin apart from random others (pure repulsion — a single positive cell makes
    // the attraction half a no-op, so metric_step([k], rate, n_neg) is exactly competitive separation).
    if repel {
        for _ in 0..40 {
            for k in 0..K {
                f.metric_step(&[k], 0.25, 4);
            }
        }
    }
    f
}

/// Retrieve the skill whose carved basin the query descends into: seed the query coordinate onto the scratch
/// cell and take the nearest key by concept-metric distance. `lambda>0` adds a value-weighted pull toward
/// DEEPER (more-rewarded) basins — the descent reads BOTH geometry and carved depth.
fn retrieve(f: &mut Field, query: &[f32], lambda: f32) -> usize {
    let q = K; // the scratch query cell
    f.seed_coord(q, query);
    (0..K)
        .min_by(|&a, &b| {
            let sa = f.distance(q, a, 0) - lambda * f.drive_of(a);
            let sb = f.distance(q, b, 0) - lambda * f.drive_of(b);
            sa.partial_cmp(&sb).unwrap()
        })
        .unwrap()
}

/// Retrieval accuracy over `trials` held-out noisy queries of random skills.
fn accuracy(f: &mut Field, sig: f32, noise: f32, lambda: f32, trials: usize, rng: &mut Rng) -> f32 {
    let mut hits = 0usize;
    for _ in 0..trials {
        let skill = (rng.u() % K as u64) as usize;
        let q = observe(skill, sig, noise, rng);
        if retrieve(f, &q, lambda) == skill {
            hits += 1;
        }
    }
    hits as f32 / trials as f32
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
// A + B: generalisation and the lateral-inhibition separability lever.
// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
#[test]
fn skill_basins_generalise_and_stay_separable() {
    let reward = [1.0f32; K]; // uniform reward here — isolate geometry (C tests reward-weighting)
    let sig = 0.16; // OVERLAPPING prototypes (heavy shared base) — the HARD case
    let noise = 0.35;
    let trials = 600;
    let chance = 1.0 / K as f32;

    // ── A: seed_coord + reinforce basins retrieve unseen near-misses far above chance; a SHUFFLED-coordinate
    // control (percept→coordinate binding destroyed) collapses to chance ⇒ the grown geometry carries it. ──
    let mut rng = Rng::new(0xB0A7);
    let mut basins = build(sig, false, &reward, 0x5EED);
    let acc_basin = accuracy(&mut basins, sig, noise, 0.0, trials, &mut rng);

    // Shuffle control: rebuild, then OVERWRITE each key's coordinate with a DIFFERENT skill's projection (rotate
    // by 1) — the reinforce depth and cell count are identical, only the percept→coordinate binding is broken.
    let mut shuffled = build(sig, false, &reward, 0x5EED);
    for k in 0..K {
        let mislabel = proto((k + 1) % K, sig);
        shuffled.seed_coord(k, &mislabel); // key k now sits where skill (k+1) belongs
    }
    let acc_shuf = accuracy(&mut shuffled, sig, noise, 0.0, trials, &mut rng);

    // ── B: the honest negative on ask (2). metric_step negative-sampling repulsion (the natural "lateral
    // inhibition") DISCARDS the shared JL projection and collapses retrieval — separability is already intrinsic
    // to seed_coord co-projecting query and keys, so there is nothing for a repulsion force to add. ──
    let mut repelled = build(sig, true, &reward, 0x5EED);
    let acc_repel = accuracy(&mut repelled, sig, noise, 0.0, trials, &mut rng);

    eprintln!("[SKILL-MEM] overlapping prototypes (sig={sig}), near-miss noise={noise}, K={K} (chance {chance:.2}):");
    eprintln!("[SKILL-MEM]   A generalisation:  basin {acc_basin:.3}   shuffled-coordinate control {acc_shuf:.3}");
    eprintln!("[SKILL-MEM]   B lateral inhib:   basin {acc_basin:.3}  ->  + metric_step repel {acc_repel:.3}");

    // A — carved basins generalise to unseen near-misses well above chance; the shuffle control collapses.
    assert!(
        acc_basin > 0.55,
        "carved basins must retrieve the right skill from unseen near-misses well above chance {chance:.2}: {acc_basin:.3}"
    );
    assert!(
        acc_basin > acc_shuf + 0.30,
        "the percept->coordinate binding must be load-bearing (basin {acc_basin:.3} vs shuffled {acc_shuf:.3})"
    );
    // B — lateral inhibition HURTS: the honest negative. Repulsion throws away the JL structure retrieval needs.
    assert!(
        acc_repel < acc_basin - 0.20,
        "metric_step repulsion must HURT (it discards the shared JL projection): {acc_basin:.3} -> {acc_repel:.3}"
    );

    eprintln!("[SKILL-MEM] VERDICT (A+B): skill-memory-as-basins needs NO new primitive. seed_coord carves the key + nearest-distance retrieval GENERALISES to near-misses ({acc_basin:.3}, shuffle {acc_shuf:.3}). Ask (2) 'lateral inhibition' is an HONEST NEGATIVE: because query and keys share ONE JL projection, separability is intrinsic; a metric_step repulsion force DISCARDS it and collapses retrieval ({acc_basin:.3} -> {acc_repel:.3}). The separability ceiling is PERCEPT distinctness (the M2 latent feeding seed_coord), not a map force.");
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
// C: reward-weighted carving = reinforce (the value face of the landscape).
// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
#[test]
fn carving_is_reward_weighted_reinforce() {
    // Carve depth must grow monotonically with reward (reinforce accumulates).
    let reward: Vec<f32> = (0..K).map(|k| 0.25 * (k as f32 + 1.0)).collect();
    let f = build(0.5, false, &reward, 0xCA47);
    for k in 1..K {
        assert!(
            f.drive_of(k) > f.drive_of(k - 1),
            "a more-rewarded skill must carve a DEEPER basin: drive_of({k})={:.3} !> drive_of({})={:.3}",
            f.drive_of(k),
            k - 1,
            f.drive_of(k - 1)
        );
    }
    eprintln!(
        "[SKILL-MEM] C carve depth vs reward: {:?}",
        (0..K).map(|k| f.drive_of(k)).collect::<Vec<_>>()
    );

    // A value-weighted descent breaks an equidistant tie toward the deeper (more-rewarded) basin, where pure
    // distance is a coin-flip. Place two skill keys EQUIDISTANT from a query, one carved deeper than the other.
    let mut g = Field::new(3);
    g.place_at(0, &[1.0, 0.0]); // shallow basin, distance 1 from query
    g.place_at(1, &[-1.0, 0.0]); // deep basin,   distance 1 from query
    g.place_at(2, &[0.0, 0.0]); // the query cell
    g.reinforce(0, 0.2);
    g.reinforce(1, 2.0); // basin 1 is far more rewarded
    let pure = retrieve(&mut g, &[0.0, 0.0], 0.0); // distance-only: a tie (deterministic min picks 0)
    let weighted = retrieve(&mut g, &[0.0, 0.0], 0.5); // value-weighted: prefers the deeper basin 1
    eprintln!("[SKILL-MEM] C tie-break: pure-distance -> skill {pure}   value-weighted -> skill {weighted}");
    assert_eq!(
        weighted, 1,
        "value-weighted descent must resolve an equidistant tie toward the more-rewarded basin"
    );

    eprintln!("[SKILL-MEM] VERDICT (C): reward-weighted carving IS reinforce — drive_of deepens with reward and a `−distance + λ·drive` descent reads both geometry and carved depth. No `carve` op needed.");
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
// D: the skill memory (basin + carve) snapshots WITH the morph.
// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
#[test]
fn skill_memory_snapshots_with_the_morph() {
    let reward: Vec<f32> = (0..K).map(|k| 0.3 * (k as f32 + 1.0)).collect();
    let mut before = build(0.16, true, &reward, 0x5A4E);

    let json = serde_json::to_string(&before).unwrap();
    let mut after: Field = serde_json::from_str(&json).unwrap();

    // Positions (the carved basin geometry) and drive (the carved depth) are bit-identical across the round-trip.
    for k in 0..K {
        assert_eq!(
            before.drive_of(k).to_bits(),
            after.drive_of(k).to_bits(),
            "carve depth (drive) must survive the snapshot for skill {k}"
        );
        let (pb, pa) = (before.position(k), after.position(k));
        for a in 0..pb.len() {
            assert_eq!(
                pb[a].to_bits(),
                pa[a].to_bits(),
                "basin coordinate must survive the snapshot (skill {k}, axis {a})"
            );
        }
    }

    // And retrieval is identical after reload — the skill memory is fully inside the morph.
    let mut rng = Rng::new(0x9017);
    let mut agree = 0usize;
    let trials = 200;
    for _ in 0..trials {
        let skill = (rng.u() % K as u64) as usize;
        let q = observe(skill, 0.16, 0.35, &mut rng);
        if retrieve(&mut before, &q, 0.2) == retrieve(&mut after, &q, 0.2) {
            agree += 1;
        }
    }
    assert_eq!(
        agree, trials,
        "retrieval must be identical after a snapshot round-trip ({agree}/{trials})"
    );
    eprintln!("[SKILL-MEM] VERDICT (D): the skill memory (basin coordinates + carved drive) round-trips through serde bit-for-bit and retrieval is unchanged — skills live IN the morph, snapshotting with it.");
}
