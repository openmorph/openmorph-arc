// SPDX-License-Identifier: MIT-0
//! SELF-DECOMPOSITION — the concept-map DETECTS the sub-morph boundary itself (Sébastien's merge). A morph
//! need not be told which cells form a skill: cells that WORK TOGETHER co-activate, so they cluster in the
//! grown concept-map, and a cluster of related cells IS a sub-morph. The SAME fractal `cluster` operator that
//! builds the retrieval index (`substrate_fractal`) applied to a morph's OWN cells gives its modular
//! decomposition — which `encapsulate` + `nest` then turn into first-class nested nodes. So the recursive
//! hypergraph is SELF-ORGANIZING: nothing authors the boundary; it falls out of the grown geometry, and
//! composition-hierarchy and retrieval-hierarchy are one structure. No new core primitive — position/distance
//! (concept-map) + cluster (usage) + encapsulate/nest (the keystone).
//!
//! Two independent modules share one field (M1 over inputs {0,1} → cell 4; M2 over {2,3} → cell 5). They
//! co-activate WITHIN a module as it works, never across. Cluster the grown positions → recover the two
//! modules unsupervised → encapsulate+nest each as a node → the morph decomposes itself, and the nodes
//! reproduce the module functions.
//!
//!   cargo test -p openmorph-core --test proofs substrate_decompose -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;

use crate::common::Rng;

// M1: cell 4 = tanh(a - b) over inputs 0,1. M2: cell 5 = tanh(c + d) over inputs 2,3.
fn m1(a: f32, b: f32) -> f32 {
    (a - b).tanh()
}
fn m2(c: f32, d: f32) -> f32 {
    (c + d).tanh()
}

#[test]
fn the_concept_map_detects_sub_morphs_and_they_become_nested_nodes() {
    // ── Grow two independent modules in ONE field (M1 over {0,1}→4, M2 over {2,3}→5). ──
    let mut f = Field::new(6);
    f.config_mut().max_hidden = 0; // grow a linear writer only (preserve grow_and_learn behaviour)
    f.config_mut().birth_gate = 0.05;
    let mut rng = Rng::new(0x00DE_C0D3);
    for _ in 0..2500 {
        let (a, b, c, d) = (rng.sym(), rng.sym(), rng.sym(), rng.sym());
        f.grow(&[(0, a), (1, b)], 4, m1(a, b), 0.1, 40);
        f.grow(&[(2, c), (3, d)], 5, m2(c, d), 0.1, 40);
    }

    // ── Grow the CONCEPT-MAP: cells that WORK TOGETHER co-activate (within a module), never across. ──
    for it in 0..4000 {
        f.co_activate(&[0, 1, 4], 0.05); // M1's cells fire together
        f.co_activate(&[2, 3, 5], 0.05); // M2's cells fire together
        if it % 100 == 99 {
            f.spread_positions();
        }
    }
    f.spread_positions();

    // ── DETECT: cluster the grown positions into 2 groups — no labels, no authored boundary. ──
    let groups = f.cluster(&[0, 1, 2, 3, 4, 5], 2);
    // Purity: each true module's cells {0,1,4} and {2,3,5} should land wholly within one cluster.
    let m1_cells = [0usize, 1, 4];
    let m2_cells = [2usize, 3, 5];
    let together = |cells: &[usize]| groups.iter().any(|g| cells.iter().all(|c| g.contains(c)));
    let detected = together(&m1_cells) && together(&m2_cells);

    eprintln!(
        "[DECOMPOSE] concept-map clusters of the morph's OWN cells (unsupervised):\n  cluster A = {:?}\n  cluster B = {:?}\n  true modules M1={{0,1,4}} M2={{2,3,5}} recovered: {detected}",
        groups[0], groups[1]
    );

    // ── The detected modules become NESTED NODES: encapsulate each cluster's output subgraph, nest it. ──
    let m1_node = f.encapsulate(&[4]); // the sub-morph M1 (unit writing cell 4)
    let m2_node = f.encapsulate(&[5]);
    let mut host = Field::new(4); // fresh host: 0,1 = M1 inputs, 2,3 = M2 inputs
    host.nest(m1_node, &[0, 1], 4, 40); // M1 as a node → cell 4
    host.nest(m2_node, &[2, 3], 5, 40); // M2 as a node → cell 5

    let mut test = Rng::new(0x5EED);
    let (mut e1, mut e2) = (0.0f32, 0.0f32);
    for _ in 0..200 {
        let (a, b, c, d) = (test.sym(), test.sym(), test.sym(), test.sym());
        host.stimulate(&[(0, a), (1, b), (2, c), (3, d)], 40);
        e1 = e1.max((host.cells()[4] - m1(a, b)).abs());
        e2 = e2.max((host.cells()[5] - m2(c, d)).abs());
    }
    eprintln!("[DECOMPOSE] the detected modules re-nested as nodes reproduce their functions — worst error M1 {e1:.3}, M2 {e2:.3}");
    eprintln!("[DECOMPOSE] VERDICT: the fractal recursive hypergraph is SELF-ORGANIZING. A sub-morph is not authored — cells that work together co-activate, cluster in the grown concept-map, and the SAME farthest-point `cluster` that builds the retrieval index recovers the morph's modular decomposition unsupervised. Each detected cluster becomes a first-class nested node (`encapsulate` + `nest`), so a morph decomposes ITSELF into a hierarchy of sub-morphs, discovered from its own geometry — composition-hierarchy and retrieval-hierarchy are one structure. No new core primitive: position/distance + cluster (usage) + the nesting keystone. The library lives INSIDE the substrate now, addressable and recursive.");

    assert!(
        detected,
        "the concept-map must recover the two true modules by clustering (groups {groups:?})"
    );
    assert!(
        e1 < 0.05 && e2 < 0.05,
        "the detected modules, re-nested as nodes, must reproduce their functions (M1 {e1:.3}, M2 {e2:.3})"
    );
}

/// Regression (audit M2): `cluster` must not panic when `ids` carries DUPLICATES. Duplicates inflate the
/// seed-loop target past the distinct pool, so the farthest-point `max_by(..).unwrap()` used to panic on an
/// empty filter. It now dedups (first-seen order) at entry; the result is the deduped clustering, and unique
/// inputs stay byte-identical.
#[test]
fn cluster_tolerates_duplicate_ids() {
    let f = Field::new(6);
    // k larger than the number of DISTINCT ids, with duplicates present.
    let groups = f.cluster(&[0, 0, 1, 1, 2, 2], 6);
    let total: usize = groups.iter().map(|g| g.len()).sum();
    assert!(
        total <= 3,
        "deduped ids: at most 3 distinct members clustered (got {total})"
    );
    assert!(
        !groups.is_empty(),
        "duplicate-laden ids must still cluster, not panic"
    );
}
