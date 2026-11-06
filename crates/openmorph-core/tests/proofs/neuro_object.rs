// SPDX-License-Identifier: MIT-0
//! Nature-based SEGMENTATION as a label-propagation field (perceptual grouping / binding). One local
//! rule (take the min label among same-colour 4-neighbours), constant in space, settling in time to
//! the connected-region basins — the same substrate as the attractor dynamics. A GENERAL primitive
//! the grown 2-D work builds on; NOT a hand-coded flood fill, and NOT an authored transform.
//!
//!   cargo test -p openmorph-core --test proofs neuro_object -- --nocapture --test-threads=1

use openmorph_core::grouping::segment_objects;

const W: usize = 9;

fn bars_grid(bars: &[(usize, usize, usize)], in_color: usize, h: usize) -> Vec<usize> {
    let mut g = vec![0usize; W * h];
    for &(r, c0, len) in bars {
        for c in c0..c0 + len {
            g[r * W + c] = in_color;
        }
    }
    g
}

#[test]
fn segmentation_field_settles_to_the_connected_regions() {
    // Two 4-disconnected bars (lengths 2 and 3) on separate rows.
    let g = bars_grid(&[(0, 1, 2), (2, 1, 3)], 7, 5);
    let mut objs = segment_objects(&g, W, 5, 0);
    objs.sort_by_key(|o| o.size());
    assert_eq!(objs.len(), 2, "two disconnected bars = two settled basins");
    assert_eq!(objs[0].size(), 2);
    assert_eq!(objs[1].size(), 3);
    assert_eq!(objs[1].bbox_w, 3);
    assert_eq!(objs[1].bbox_h, 1);

    // A single connected L is ONE object regardless of shape.
    let l = {
        let mut g = vec![0usize; W * 5];
        g[W + 1] = 4;
        g[2 * W + 1] = 4;
        g[2 * W + 2] = 4;
        g
    };
    let objs = segment_objects(&l, W, 5, 0);
    assert_eq!(objs.len(), 1, "a connected L is one basin");
    assert_eq!(objs[0].size(), 3);
    eprintln!("[SEGMENT] label-propagation field settles to connected regions: 2 bars → 2 basins, L → 1 basin. Nature-based grouping (local rule, constant in space, settling in time), not a flood fill.");
}
