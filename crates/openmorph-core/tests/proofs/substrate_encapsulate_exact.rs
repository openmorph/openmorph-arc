// SPDX-License-Identifier: MIT-0
//! ENCAPSULATE IS EXACT — a banked skill computes the parent's function BIT-for-bit, whatever it is built of.
//! "Reuse IS nesting" only holds if `encapsulate` lifts the function the parent computes, not a look-alike.
//! `substrate_encapsulate_backbone` pins the transitive backbone and `substrate_transcribe` (D) the per-unit
//! law; this pins everything else a unit carries into the settle:
//!
//! - its heritable projection gain (`set_scale`, or a genome `init_scale ≠ 1`) — the lifted unit used to be
//!   re-seeded from the sub's DEFAULT genome, so an evolved/scaled skill computed another function;
//! - its forward FACTOR (the second-order term) from a RAW input — the factor cell must be a source of the
//!   sub, fed by the host, and a registered reader edge (a factor-only change must re-fire the unit);
//! - its factor from a GROWN cell — the closure must follow the factor edge and pull in the factor's writer;
//! - a `grow_products` product feature (`x_j·f(x_i)`) grown by the organism itself;
//! - transcribed `Relu` / `Identity` units;
//! - its other heritable traits (`lr`, `mut_rate`, `momentum`, `plasticity`) and its write-`gate` — carried only
//!   when the gate cell lives INSIDE the skill (a lifted unit writes it, or it is one of the skill's inputs). A gate
//!   on an outside cell is invisible to the sub, so it is dropped and the banked unit learns ungated, as before —
//!   pinned under `nest_credit` (the one path where a banked unit's gate acts) at the end of this file.
//!
//! Each lifted copy — the sub nested into a blank host, the sub settled on its own, and the skill banked in
//! place (`bank_skill`) — is driven through the SAME input sequence as the parent (including factor-only
//! changes) and must answer `.to_bits()`-identically at every step.
//!
//!   cargo test -p openmorph-core --release --test proofs substrate_encapsulate_exact -- --nocapture

use openmorph_core::{Activation, Config, Field};

const BUDGET: usize = 512;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    /// A value in [-1, 1].
    fn sig(&mut self) -> f32 {
        ((self.next() >> 40) % 2001) as f32 / 1000.0 - 1.0
    }
}

/// `n` random input vectors of width `dim`; when `factor_only` names an input position, every random vector is
/// followed by a copy that changes ONLY that input (a factor-only change must re-fire the product unit).
fn sequence(seed: u64, dim: usize, n: usize, factor_only: Option<usize>) -> Vec<Vec<f32>> {
    let mut rng = Lcg(seed);
    let mut out = Vec::new();
    for _ in 0..n {
        let v: Vec<f32> = (0..dim).map(|_| rng.sig()).collect();
        out.push(v.clone());
        if let Some(k) = factor_only {
            let mut w = v;
            w[k] = -0.5 * w[k] + 0.25;
            out.push(w);
        }
    }
    out
}

/// Drive the parent and its three lifted copies through `steps` and return, per copy, how many steps answered
/// NOT bit-identically to the parent plus the worst |Δ|: `[(nested, …), (sub-direct, …), (banked, …)]`.
fn lift_mismatches(
    parent: &Field,
    inputs: &[usize],
    out: usize,
    steps: &[Vec<f32>],
) -> [(usize, f32); 3] {
    let sub = parent.encapsulate(&[out]);
    // (1) nested into a blank host at a fresh cell.
    let w = inputs.iter().copied().max().unwrap_or(0) + 1;
    let mut host = Field::new(w);
    host.nest(sub.clone(), inputs, w, BUDGET);
    // (2) the sub settled on its own (it keeps the parent's cell ids).
    let mut direct = sub;
    // (3) banked in place, next to the reference computation.
    let mut banked = parent.clone();
    let b = banked.bank_skill(out, BUDGET);
    let mut reference = parent.clone();

    let mut res = [(0usize, 0.0f32); 3];
    for v in steps {
        let stim: Vec<(usize, f32)> = inputs.iter().copied().zip(v.iter().copied()).collect();
        reference.stimulate(&stim, BUDGET);
        host.stimulate(&stim, BUDGET);
        direct.stimulate(&stim, BUDGET);
        banked.stimulate(&stim, BUDGET);
        let want = reference.cells()[out];
        let got = [
            host.cells()[w],
            direct.cells().get(out).copied().unwrap_or(0.0),
            banked.cells()[b],
        ];
        for (r, g) in res.iter_mut().zip(got) {
            if g.to_bits() != want.to_bits() {
                r.0 += 1;
            }
            r.1 = r.1.max((g - want).abs());
        }
        assert_eq!(
            banked.cells()[out].to_bits(),
            want.to_bits(),
            "banking a skill must not disturb the parent's own computation"
        );
    }
    res
}

fn report_and_assert(tag: &str, res: [(usize, f32); 3], n_steps: usize) {
    for (name, (bad, worst)) in ["nested", "sub-direct", "bank_skill"].iter().zip(res) {
        eprintln!("[ENCAPSULATE-EXACT] {tag:<11} {name:<10}: {bad:>3}/{n_steps} steps differ, worst |Δ| {worst:.3e}");
    }
    for (name, (bad, worst)) in ["nested", "sub-direct", "bank_skill"].iter().zip(res) {
        assert_eq!(
            bad, 0,
            "{tag}: the {name} lift must answer bit-identically to the parent at every step ({bad}/{n_steps} differ, worst |Δ| {worst:e})"
        );
    }
}

/// A hand-built second-order skill: every kind of edge and trait a unit carries into the settle.
/// Cells 0..=4 are raw inputs; 10 is the skill's output; 11 is a SIBLING skill outside the closure.
fn second_order_parent() -> Field {
    let mut f = Field::new(12);
    f.configure(Config {
        init_momentum: 0.25, // a non-default genome: every unit is born with these traits
        init_mut_rate: 0.07,
        ..Config::default()
    });
    // c5 = tanh(2.5·(0.7x0 − 0.4x1 + 0.1)) — an evolved/scaled gain.
    f.add_unit(vec![0, 1], vec![0.7, -0.4], 0.1, 5);
    f.set_scale(5, 2.5);
    // c6 = x3 · tanh(1.1x2 − 0.2) — factor from a RAW input (x3 is read ONLY as a factor).
    f.add_unit(vec![2], vec![1.1], -0.2, 6);
    f.set_factor(6, Some(3));
    // c7 = tanh(0.6·(0.8x4 + 0.3x0 − 0.1)) — a computed cell read ONLY as c8's factor (x4 feeds only it).
    f.add_unit(vec![4, 0], vec![0.8, 0.3], -0.1, 7);
    f.set_scale(7, 0.6);
    // c8 = c7 · tanh(0.9c5 + 0.05) — factor from a GROWN (unit-written) cell.
    f.add_unit(vec![5], vec![0.9], 0.05, 8);
    f.set_factor(8, Some(7));
    f.set_lr(8, 0.37);
    // c9 = relu(1.3c6 − 0.5c8 + 0.2) — a transcribed Relu unit, consolidated.
    f.add_unit(vec![6, 8], vec![1.3, -0.5], 0.2, 9);
    f.set_activation(9, Some(Activation::Relu));
    f.set_plasticity(&[9], 0.0);
    // c10 = 0.6c9 + 0.4c5 − 0.3c6 — a transcribed Identity readout (the skill output), write-gated by c9.
    f.add_unit(vec![9, 5, 6], vec![0.6, 0.4, -0.3], 0.0, 10);
    f.set_activation(10, Some(Activation::Identity));
    f.set_gate(&[10], 9);
    // c11 = tanh(x1) — a sibling skill NOT feeding c10; it gates c5 (a gate is not a dependency: c11 stays out).
    f.add_unit(vec![1], vec![1.0], 0.0, 11);
    f.set_gate(&[5], 11);
    f
}

#[test]
fn a_second_order_scaled_transcribed_skill_lifts_bit_exactly() {
    let parent = second_order_parent();
    let steps = sequence(0x5CA1E, 5, 48, Some(3)); // every other step changes ONLY the raw factor x3
    let res = lift_mismatches(&parent, &[0, 1, 2, 3, 4], 10, &steps);
    report_and_assert("second-order", res, steps.len());
}

#[test]
fn lifted_units_carry_their_traits_and_edges() {
    let parent = second_order_parent();
    let sub = parent.encapsulate(&[10]);
    sub.validate()
        .expect("every cell a lifted unit references (factor, gate) must exist in the sub");
    let written: Vec<usize> = sub.units.iter().map(|u| u.write).collect();
    eprintln!("[ENCAPSULATE-EXACT] lifted cells {written:?} (the factor's writer c7 must be pulled in, sibling c11 not)");
    assert_eq!(
        written,
        vec![5, 6, 7, 8, 9, 10],
        "the closure follows reads AND factors, in parent unit order, and stops at the skill's boundary"
    );
    for &c in &written {
        assert_eq!(sub.scale_of(c), parent.scale_of(c), "scale of c{c}");
        assert_eq!(sub.lr_of(c), parent.lr_of(c), "lr of c{c}");
        assert_eq!(
            sub.mut_rate_of(c),
            parent.mut_rate_of(c),
            "mut_rate of c{c}"
        );
        assert_eq!(
            sub.momentum_of(c),
            parent.momentum_of(c),
            "momentum of c{c}"
        );
        assert_eq!(
            sub.plasticity_of(c),
            parent.plasticity_of(c),
            "plasticity of c{c}"
        );
    }
    for (su, pu) in sub.units.iter().zip(parent.units.iter()) {
        assert_eq!(su.write, pu.write);
        assert_eq!(su.factor, pu.factor, "factor of c{}", su.write);
        assert_eq!(su.activation, pu.activation, "activation of c{}", su.write);
    }
    // c10's gate c9 is written INSIDE the skill, so it travels; c5's gate c11 is an outside sibling, so it is
    // dropped (the sub could never see c11's value — a gate stuck at 0 would freeze c5 for good).
    let gate_of = |f: &Field, c: usize| f.units.iter().find(|u| u.write == c).and_then(|u| u.gate);
    assert_eq!(gate_of(&parent, 10), Some(9));
    assert_eq!(gate_of(&sub, 10), Some(9), "an inside gate travels");
    assert_eq!(gate_of(&parent, 5), Some(11));
    assert_eq!(gate_of(&sub, 5), None, "an outside gate is dropped");
}

#[test]
fn a_grown_product_skill_lifts_bit_exactly() {
    // Grow a readout for 0.5·x0·x1 with product discovery on, under a non-default genome gain (init_scale 0.8):
    // the organism births its own `x_j·f(x_i)` product feature(s) next to random-projection features.
    let mut f = Field::new(5);
    f.configure(Config {
        grow_products: true,
        lr_fanin_norm: true,
        max_hidden: 8,
        init_scale: 0.8,
        ..Config::default()
    });
    let mut rng = Lcg(0x9D0C);
    for _ in 0..1500 {
        let x = [rng.sig(), rng.sig(), rng.sig(), rng.sig()];
        f.grow(
            &[(0, x[0]), (1, x[1]), (2, x[2]), (3, x[3])],
            4,
            0.5 * x[0] * x[1],
            0.1,
            64,
        );
    }
    let products = f.units.iter().filter(|u| u.factor.is_some()).count();
    eprintln!(
        "[ENCAPSULATE-EXACT] grown organism: {} units, {products} grown product feature(s)",
        f.n_units()
    );
    assert!(
        products >= 1,
        "the proof needs a GROWN product feature to lift"
    );
    // A common starting state for the parent and its lifted copies (a fresh sub starts at rest).
    f.mem.medium.iter_mut().for_each(|v| *v = 0.0);
    let steps = sequence(0xFEED, 4, 64, None);
    let res = lift_mismatches(&f, &[0, 1, 2, 3], 4, &steps);
    report_and_assert("grown-prod", res, steps.len());
}

#[test]
fn a_default_skill_lifts_byte_identically_to_a_plain_rebuild() {
    // The default path must not move: a factor-free skill with default traits lifts to EXACTLY the field one
    // would get by re-adding its units (in parent order) to a fresh field under the parent's genome.
    let mut m = Field::new(7);
    m.add_unit(vec![0], vec![3.0], -2.0, 1);
    m.add_unit(vec![1, 4], vec![2.0, 0.5], 1.0, 2);
    m.add_unit(vec![5], vec![1.0], 0.0, 4); // backbone written AFTER its reader was born
    m.add_unit(vec![0], vec![0.5], 0.0, 6); // a sibling NOT feeding the output
    m.add_unit(vec![2, 0], vec![1.5, -0.7], -0.5, 3);
    let sub = m.encapsulate(&[3]);
    let mut plain = Field::new(6);
    plain.add_unit(vec![0], vec![3.0], -2.0, 1);
    plain.add_unit(vec![1, 4], vec![2.0, 0.5], 1.0, 2);
    plain.add_unit(vec![5], vec![1.0], 0.0, 4);
    plain.add_unit(vec![2, 0], vec![1.5, -0.7], -0.5, 3);
    plain.configure(m.config().clone());
    assert_eq!(
        serde_json::to_string(&sub).unwrap(),
        serde_json::to_string(&plain).unwrap(),
        "a default, factor-free skill must lift byte-identically"
    );
}

/// A banked skill under `nest_credit` — the host's composed error tunes the sub's readout, and that readout's
/// write-`gate` is the one thing a banked unit's gate acts on. Parent: hidden `c2 = tanh(1.3·x0)`, readout
/// `c3 = tanh(0.2·x0 + 0.3·c2 + 0.25·x1)`; `gate` (if any) is set on `c3`. The skill is banked in place and a host
/// readout over the banked cell is trained for 50 events, driving input `x1` and the outside cell `c4` to the given
/// values. Returns the sub readout's carried gate and its weights before and after.
fn nest_credit_readout(
    gate: Option<usize>,
    x1: f32,
    c4: f32,
) -> (Option<usize>, Vec<f32>, Vec<f32>) {
    let mut f = Field::new(5);
    f.configure(Config {
        nest_credit: true,
        nest_seeds_position: false,
        ..Config::default()
    });
    f.add_unit(vec![0], vec![1.3], 0.0, 2);
    f.add_unit(vec![0, 2, 1], vec![0.2, 0.3, 0.25], 0.0, 3);
    if let Some(g) = gate {
        f.set_gate(&[3], g);
    }
    let b = f.bank_skill(3, BUDGET);
    let hr = b + 1;
    f.add_unit(vec![b], vec![1.0], 0.0, hr);
    let readout = |f: &Field| {
        let nst = f
            .units
            .iter()
            .find_map(|u| u.nested.as_ref())
            .expect("the banked skill is a nested cell");
        let u = nst
            .sub
            .units
            .iter()
            .find(|u| u.write == 3)
            .expect("the sub's readout");
        (u.gate, u.weights.clone())
    };
    let (carried, before) = readout(&f);
    for k in 0..50 {
        let x = (k as f32 * 0.37).sin();
        f.learn_step(&[(0, x), (1, x1), (4, c4)], hr, 0.9 * x, 0.1, BUDGET);
    }
    (carried, before, readout(&f).1)
}

fn bits(w: &[f32]) -> Vec<u32> {
    w.iter().map(|w| w.to_bits()).collect()
}

#[test]
fn an_outside_gate_is_dropped_so_nest_credit_still_tunes_the_banked_skill() {
    // The readout is gated by c4, a cell the skill neither reads nor writes: the sub can never see it. Carrying the
    // gate would leave the sub's c4 at 0 — a CLOSED gate — and `nest_credit` would silently freeze the skill even
    // though the parent's gate is open. Dropping it keeps the pre-existing contract: learn exactly as ungated.
    let (carried, before, gated) = nest_credit_readout(Some(4), 0.5, 1.0);
    let (_, _, ungated) = nest_credit_readout(None, 0.5, 1.0);
    eprintln!("[ENCAPSULATE-EXACT] nest_credit, outside gate c4 (open): carried {carried:?}, sub readout {before:?} -> {gated:?} (ungated twin {ungated:?})");
    assert_ne!(gated, before, "the banked readout must still learn");
    assert_eq!(
        bits(&gated),
        bits(&ungated),
        "an outside gate is dropped: the banked skill learns bit-identically to its ungated twin"
    );
    assert_eq!(carried, None, "a gate on an outside cell does not travel");
}

#[test]
fn an_inside_gate_travels_and_gates_nest_credit() {
    // The readout is gated by its own input x1 — a source of the skill, fed by the host, so the sub sees the live
    // gate value. Closed (x1 = 0) the banked readout is PROTECTED, exactly as the parent's own unit would be; open
    // (x1 = 1) it learns bit-identically to an ungated twin.
    let (carried, before, closed) = nest_credit_readout(Some(1), 0.0, 0.0);
    let (_, _, open) = nest_credit_readout(Some(1), 1.0, 0.0);
    let (_, _, ungated) = nest_credit_readout(None, 1.0, 0.0);
    eprintln!("[ENCAPSULATE-EXACT] nest_credit, inside gate x1: carried {carried:?}, closed {before:?} -> {closed:?}; open -> {open:?} (ungated twin {ungated:?})");
    assert_eq!(
        carried,
        Some(1),
        "a gate on one of the skill's own inputs travels"
    );
    assert_eq!(
        bits(&closed),
        bits(&before),
        "a closed inside gate protects the banked readout"
    );
    assert_ne!(open, before, "an open inside gate lets it learn");
    assert_eq!(
        bits(&open),
        bits(&ungated),
        "a fully open gate is a full step"
    );
}
