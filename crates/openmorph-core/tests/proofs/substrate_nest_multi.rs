// SPDX-License-Identifier: MIT-0
//! MULTI-OUTPUT NEST — a nested cell exposing a sub-morph's FULL VECTOR output, not just one scalar. The
//! fractal-collapse thesis is "a cell can be a MORPH"; but until now `nest` collapsed a sub-morph to its FIRST
//! sink (`sink_cells_of(...).first()`), so a morph whose output is `f: D → V` with `|V|>1` could only be planted
//! as a scalar FEATURE, never a real sub-morph. `nest_multi(sub, reads, writes, budget)` maps the sub's sink
//! cells positionally to `writes` and writes them ALL on ONE sub-settle — the missing primitive (MorphBot F5).
//!
//!   cargo test -p openmorph-core --test proofs substrate_nest_multi -- --nocapture

use openmorph_core::substrate::Field;

/// A sub-morph over inputs (0, 1) with a TWO-vector output: cell 2 = tanh(x0 + x1), cell 3 = tanh(x0 − x1).
fn vector_sub() -> Field {
    let mut s = Field::new(4); // cells 0,1 = inputs; 2,3 = the two outputs (sinks)
    s.add_unit(vec![0, 1], vec![1.0, 1.0], 0.0, 2); // sum
    s.add_unit(vec![0, 1], vec![1.0, -1.0], 0.0, 3); // difference
    s
}

fn approx(a: f32, b: f32) -> bool {
    (a - b).abs() < 1e-5
}

#[test]
fn nest_multi_exposes_the_full_vector_output() {
    let sub = vector_sub();
    // Host: cells 0,1 = inputs; plant the sub exposing BOTH outputs at host cells 2 and 3.
    let mut host = Field::new(4);
    host.nest_multi(sub, &[0, 1], &[2, 3], 40);

    for &(x0, x1) in &[(0.5f32, -0.3), (0.9, 0.2), (-0.4, -0.6), (0.1, 0.8)] {
        host.stimulate(&[(0, x0), (1, x1)], 40);
        let sum = host.cells()[2];
        let dif = host.cells()[3];
        assert!(
            approx(sum, (x0 + x1).tanh()),
            "primary output (host cell 2) must be tanh(x0+x1): got {sum} want {}",
            (x0 + x1).tanh()
        );
        assert!(
            approx(dif, (x0 - x1).tanh()),
            "SECOND output (host cell 3) must be tanh(x0−x1): got {dif} want {} — the vector output is exposed",
            (x0 - x1).tanh()
        );
    }
    eprintln!("[NEST-MULTI] a 2-vector sub-morph nests as ONE node exposing BOTH sinks (sum + difference) on one settle — a cell is a real SUB-MORPH, not a scalar feature.");
}

#[test]
fn the_second_output_fires_its_readers_in_the_host_cascade() {
    // A downstream host unit reads the SECOND nested output (cell 3) — it must fire when the sub-settle writes it,
    // proving the extra outputs enqueue their readers in the host cascade (not just land as dead values).
    let sub = vector_sub();
    let mut host = Field::new(5);
    host.nest_multi(sub, &[0, 1], &[2, 3], 40);
    // cell 4 = 2·(the difference output at cell 3) — a reader of the SECOND nested output.
    host.add_unit(vec![3], vec![2.0], 0.0, 4);
    host.stimulate(&[(0, 0.6), (1, -0.2)], 40);
    let diff = (0.6f32 - (-0.2)).tanh(); // the second nested output (cell 3)
    let want = (2.0 * diff).tanh(); // the reader unit computes tanh(2·cell3)
    assert!(
        approx(host.cells()[4], want),
        "a downstream reader of the SECOND output must fire in the host cascade: got {} want {want}",
        host.cells()[4]
    );
    eprintln!("[NEST-MULTI] the extra vector outputs reach their readers — a host unit reading the second sink fires in the cascade.");
}

#[test]
fn single_output_nest_is_byte_identical() {
    // `nest` (single write) == `nest_multi` with one write == the historical scalar node: the primary sink flows
    // through the unit's cell exactly as before, and no extra cells are touched.
    let sub_a = vector_sub();
    let mut h1 = Field::new(4);
    h1.nest(sub_a, &[0, 1], 2, 40); // via the wrapper
    let sub_b = vector_sub();
    let mut h2 = Field::new(4);
    h2.nest_multi(sub_b, &[0, 1], &[2], 40); // explicit single-write
    for &(x0, x1) in &[(0.5f32, -0.3), (0.9, 0.2)] {
        h1.stimulate(&[(0, x0), (1, x1)], 40);
        h2.stimulate(&[(0, x0), (1, x1)], 40);
        assert_eq!(
            h1.cells()[2].to_bits(),
            h2.cells()[2].to_bits(),
            "nest and single-write nest_multi must be byte-identical"
        );
        // cell 3 was NOT a write target here, so it stays 0 (the second sink is not exposed).
        assert_eq!(h1.cells()[3], 0.0, "an un-mapped sink must not be written");
    }
    eprintln!("[NEST-MULTI] single-write nest is byte-identical to the historical scalar node (the second sink stays unexposed).");
}
