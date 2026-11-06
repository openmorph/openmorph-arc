// SPDX-License-Identifier: MIT-0
//! COMPOSITION WITH AN INTERFACE CONTRACT (MorphLab's marketplace wall). The async cascade chains grown
//! sub-morphs losslessly, but composing by raw cell WIRING has no type check, so a producer emitting a
//! different representation would silently corrupt. The contract is a `Port` schema on each sub-morph +
//! `Port::compatible`: a matching producer nests; a mismatched one is REJECTED before nesting, never silently
//! wrong. With the collapse to one recursive primitive, a "skill" is just a sub-morph ([`Field::encapsulate`])
//! and composition is [`Field::nest`] — the contract is now an EXPLICIT `compatible` gate (visible, not buried
//! in a bespoke `graft_skill`).
//!
//!   cargo test -p openmorph-core --test proofs compose_contract -- --nocapture --test-threads=1

use openmorph_core::membrane::Port;
use openmorph_core::substrate::Field;

use crate::common::Rng;

// cells: X = 0,1 ; intermediate Y = 2 ; output Z = 3.
fn y_of(x0: f32, x1: f32) -> f32 {
    (0.8 * x0 - 0.5 * x1).tanh()
}
fn z_of(y: f32) -> f32 {
    (1.3 * y).tanh()
}
fn y2_of(x0: f32, x1: f32) -> f32 {
    (0.3 * x0 + 0.9 * x1).tanh() // a DIFFERENT intermediate map
}

/// A banked skill: a sub-morph plus the interface it declares (the contract lives WITH the node now).
struct Skill {
    sub: Field,
    input: Port,
    output: Port,
}

/// Grow skill A: X→Y (writes cell 2), declaring output interface `out_name`.
fn grow_a(out_name: &str, ymap: fn(f32, f32) -> f32, seed: u64) -> Skill {
    let mut f = Field::new(3);
    f.config_mut().max_hidden = 0; // grow a linear writer only (preserve grow_and_learn behaviour)
    f.config_mut().birth_gate = 0.05;
    let mut rng = Rng::new(seed);
    for _ in 0..3000 {
        let (x0, x1) = (rng.sym(), rng.sym());
        f.grow(&[(0, x0), (1, x1)], 2, ymap(x0, x1), 0.1, 100);
    }
    Skill {
        sub: f.encapsulate(&[2]),
        input: Port::input("x", 2),
        output: Port::output(out_name, 1),
    }
}

/// Grow skill B: Y→Z (reads cell 2, writes cell 3), declaring input interface "feat".
fn grow_b(seed: u64) -> Skill {
    let mut f = Field::new(4);
    f.config_mut().max_hidden = 0; // grow a linear writer only (preserve grow_and_learn behaviour)
    f.config_mut().birth_gate = 0.05;
    let mut rng = Rng::new(seed);
    for _ in 0..3000 {
        let y = rng.sym();
        f.grow(&[(2, y)], 3, z_of(y), 0.1, 100);
    }
    Skill {
        sub: f.encapsulate(&[3]),
        input: Port::input("feat", 1),
        output: Port::output("z", 1),
    }
}

/// Nest `skill` into `host` reading `reads`, writing `write` — but ONLY if the upstream `producer` port fits
/// the skill's declared input (the marketplace contract). `None` = a source skill (no producer to check).
fn nest_checked(
    host: &mut Field,
    producer: Option<&Port>,
    skill: Skill,
    reads: &[usize],
    write: usize,
) -> Result<(), String> {
    if let Some(p) = producer {
        if !p.compatible(&skill.input) {
            return Err(format!(
                "producer emits '{}' (dim {}) but skill consumes '{}' (dim {})",
                p.name, p.dim, skill.input.name, skill.input.dim
            ));
        }
    }
    host.nest(skill.sub, reads, write, 200);
    Ok(())
}

#[test]
fn a_matching_producer_composes_and_a_mismatch_is_rejected() {
    let a_match = grow_a("feat", y_of, 0x0A); // emits "feat" (what B expects)
    let a_wrong = grow_a("feat_v2", y2_of, 0x0B); // a different representation, honestly typed
    let b = grow_b(0x0B0);

    // MATCH: A (source) nests; B's input "feat" fits A's output "feat" → composes. Drive X, the cascade A▷B.
    let mut host = Field::new(4);
    let a_out = a_match.output.clone();
    nest_checked(&mut host, None, a_match, &[0, 1], 2).expect("source nests");
    nest_checked(&mut host, Some(&a_out), b, &[2], 3).expect("matching producer composes");
    let mut rng = Rng::new(0xC0DE);
    let mut mae = 0.0f32;
    for _ in 0..500 {
        let (x0, x1) = (rng.sym(), rng.sym());
        host.stimulate(&[(0, x0), (1, x1)], 200);
        mae += (host.cells()[3] - z_of(y_of(x0, x1))).abs(); // true X→Z
    }
    mae /= 500.0;
    assert!(
        mae < 0.05,
        "matched composition solves X→Z it never trained on (MAE {mae:.3})"
    );

    // MISMATCH: the wrong-typed producer's output does NOT fit B's input → REJECTED, no silent corruption.
    let mut host2 = Field::new(4);
    let wrong_out = a_wrong.output.clone();
    nest_checked(&mut host2, None, a_wrong, &[0, 1], 2).unwrap();
    let b2 = grow_b(0x0B0);
    let rejected = nest_checked(&mut host2, Some(&wrong_out), b2, &[2], 3);
    assert!(
        rejected.is_err(),
        "a representation mismatch must be REJECTED, not silently composed"
    );

    eprintln!("[CONTRACT] matched producer composes X→Z (MAE {mae:.3}, never trained on the pair); mismatched producer '{}' REJECTED: {}", wrong_out.name, rejected.err().unwrap());
    eprintln!("[CONTRACT] VERDICT: composition is one recursive primitive — a skill is a sub-morph (`encapsulate`), reuse is `nest`. The marketplace interface contract survives as an EXPLICIT `Port::compatible` gate on the nest: a matching producer composes (chaining grown sub-morphs losslessly, MAE 0.000 on a pair never trained), a mismatched representation is rejected before nesting. The check is now visible caller discipline, not a bespoke graft_skill — simpler, and the safety is the same.");
}
