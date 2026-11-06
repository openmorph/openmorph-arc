// SPDX-License-Identifier: MIT-0
//! GROUPING TYPES ARE SERDE DATA — `Object`, `Aggregate` and `Relation` derive `Serialize`/`Deserialize` like every
//! other public core type (the `Debug, Clone, Serialize, Deserialize` convention), so a segmented scene, its reduced
//! statistics and its relation hypergraph can be recorded, traced, or sent across a binding and read back exactly.
//!
//!   cargo test -p openmorph-core --release --test proofs grouping_serde -- --nocapture

use openmorph_core::grouping::{aggregate, relate, segment_objects, Aggregate, Object, Relation};

#[test]
fn objects_aggregates_and_relations_round_trip_through_json() {
    let grid = [0, 1, 1, 0, 0, 2, 3, 0, 2];
    let objs = segment_objects(&grid, 3, 3, 0);
    let json = serde_json::to_string(&objs).unwrap();
    let back: Vec<Object> = serde_json::from_str(&json).unwrap();
    assert_eq!(back.len(), objs.len());
    for (a, b) in objs.iter().zip(&back) {
        assert_eq!(
            (&a.cells, a.color, a.bbox_w, a.bbox_h),
            (&b.cells, b.color, b.bbox_w, b.bbox_h)
        );
    }

    let agg = aggregate(&[1.0, 2.0, 4.0]);
    let back: Aggregate = serde_json::from_str(&serde_json::to_string(&agg).unwrap()).unwrap();
    assert_eq!(back, agg);

    let rels = relate(&objs, 3, 3);
    let back: Vec<Relation> = serde_json::from_str(&serde_json::to_string(&rels).unwrap()).unwrap();
    assert_eq!(back.len(), rels.len());
    for (a, b) in rels.iter().zip(&back) {
        assert_eq!(
            (a.a, a.b, a.same_color, a.adjacent),
            (b.a, b.b, b.same_color, b.adjacent)
        );
        assert_eq!(
            (a.dx.to_bits(), a.dy.to_bits(), a.size_ratio.to_bits()),
            (b.dx.to_bits(), b.dy.to_bits(), b.size_ratio.to_bits())
        );
    }
    eprintln!(
        "[SERDE] {} objects, {} relations and an Aggregate round-trip bit-exactly",
        objs.len(),
        rels.len()
    );
}
