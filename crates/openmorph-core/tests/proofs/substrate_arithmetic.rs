// SPDX-License-Identifier: MIT-0
//! EXACT PLACE-VALUE ARITHMETIC (#129) — exact multi-digit arithmetic FALLS OUT of `grow` + a carry scan, with
//! NO new primitive. This is the growable place-value skill the symbol proof (#127) named as the open piece
//! ("exact multi-digit arithmetic needs a growable PLACE-VALUE skill: decompose → per-place op + carry →
//! recombine").
//!
//! The mechanism, all grown, nothing authored:
//!   - CARRY is a THRESHOLD (`a+b+c_in ≥ B`). `grow` learns thresholds exactly (like `substrate_operator`'s
//!     AND/OR/XOR) — 200/200.
//!   - DIGIT is `grow`n over `[a, b, c_in, carry]`. Given the grown CARRY concept, `digit = sum − B·carry`
//!     is LINEAR, so the wraparound (mod B) EMERGES — it is never fitted as a discontinuous sawtooth (which
//!     smooth features cannot do). The digit output uses a sub-saturation amplitude so no digit sits at the
//!     tanh extreme (an encoding detail, not an algorithm).
//!   - MULTI-DIGIT is the SAME one grown per-place skill REUSED across places with the carry threaded — the
//!     fractal reuse (#64). Because it is one small skill scanned over the sequence, it is EXACT at ANY length
//!     (trained only on single-place facts), where a MONOLITH memorizes its training length and cannot even
//!     represent a longer number. This is the same shape as #137's temporal scan: a grown per-step op + a
//!     carried state over a sequence.
//!
//!   cargo test -p openmorph-core --test proofs substrate_arithmetic -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;

const B: usize = 10; // base (decimal)
const A: usize = 0;
const BB: usize = 1;
const CIN: usize = 2;
const COUT: usize = 3; // carry-out (grown threshold)
const DIG: usize = 4; // digit (grown over [a,b,c,carry])
const AMP: f32 = 0.85; // digit output amplitude — keep every digit off the tanh saturation extremes

fn ins(a: usize, b: usize, c: usize) -> Vec<(usize, f32)> {
    vec![
        (A, a as f32 / B as f32),
        (BB, b as f32 / B as f32),
        (CIN, c as f32 / 2.0),
    ]
}
fn digit_target(d: usize) -> f32 {
    ((d as f32 / (B - 1) as f32) * 2.0 - 1.0) * AMP
}
fn digit_decode(v: f32) -> usize {
    (((v / AMP + 1.0) / 2.0 * (B - 1) as f32).round() as i64).clamp(0, (B - 1) as i64) as usize
}

/// GROW the per-place adder — a self-contained 5-cell morph. Trained on all B·B·2 single-place facts (the
/// addition table a child memorizes). Carry and digit are both GROWN; the digit's receptive field includes
/// the grown carry concept, so `digit = sum − B·carry` is learned as a linear relation. Nothing authored.
fn grow_place_adder() -> Field {
    let mut f = Field::new(DIG + 1);
    f.rng_seed(0x0ADD);
    f.config_mut().birth_gate = 0.04;
    f.config_mut().max_hidden = 20;
    f.config_mut().feature_lr = 0.3;
    for _ in 0..900 {
        for a in 0..B {
            for b in 0..B {
                for c in 0..2 {
                    let s = a + b + c;
                    // grow the carry threshold from (a,b,c_in)
                    f.grow(
                        &ins(a, b, c),
                        COUT,
                        if s >= B { 1.0 } else { -1.0 },
                        0.1,
                        200,
                    );
                    // grow the digit over (a,b,c_in, the committed carry decision)
                    let carry = if f.cells()[COUT] > 0.0 { 1.0 } else { -1.0 };
                    let mut d_in = ins(a, b, c);
                    d_in.push((COUT, carry));
                    f.grow(&d_in, DIG, digit_target(s % B), 0.1, 200);
                }
            }
        }
    }
    f
}

/// One place of the scan: settle to the carry DECISION, then settle the digit given that committed carry.
/// (The carry is a bit the morph decides and then uses — for this digit and for the next place's carry-in.)
fn place(f: &mut Field, a: usize, b: usize, c: usize) -> (usize, usize) {
    f.stimulate(&ins(a, b, c), 200);
    let carry = usize::from(f.cells()[COUT] > 0.0);
    let mut d_in = ins(a, b, c);
    d_in.push((COUT, if carry == 1 { 1.0 } else { -1.0 }));
    f.stimulate(&d_in, 200);
    (digit_decode(f.cells()[DIG]), carry)
}

/// ADD two numbers (little-endian digit vectors) by REUSING the one grown per-place skill across every place,
/// threading the carry. The scan is length-free — the same 5-cell morph handles any number of places.
fn add(f: &mut Field, x: &[usize], y: &[usize]) -> Vec<usize> {
    let p = x.len().max(y.len());
    let mut out = Vec::with_capacity(p + 1);
    let mut carry = 0;
    for k in 0..p {
        let (a, b) = (
            x.get(k).copied().unwrap_or(0),
            y.get(k).copied().unwrap_or(0),
        );
        let (d, c) = place(f, a, b, carry);
        out.push(d);
        carry = c;
    }
    out.push(carry);
    out
}

fn to_digits(mut n: u128, p: usize) -> Vec<usize> {
    (0..p)
        .map(|_| {
            let d = (n % B as u128) as usize;
            n /= B as u128;
            d
        })
        .collect()
}

/// A MONOLITH that grows to add whole 3-digit numbers in one shot (6 input digits → 4 output digits), the
/// "learn arithmetic as one big function" baseline. Returns its held-out exact-match rate. It memorizes its
/// training length and — having a FIXED input width — cannot even represent a longer number.
fn grow_monolith() -> f32 {
    use crate::common::Rng;
    let n_in = 6; // x: 3 digits, y: 3 digits
    let n_out = 4; // sum: up to 4 digits
    let mut f = Field::new(n_in + n_out);
    f.rng_seed(0x3117);
    f.config_mut().birth_gate = 0.03;
    f.config_mut().max_hidden = 64;
    f.config_mut().feature_lr = 0.3;
    let enc = |x: u128, y: u128| -> Vec<(usize, f32)> {
        let xd = to_digits(x, 3);
        let yd = to_digits(y, 3);
        (0..3)
            .map(|i| (i, xd[i] as f32 / B as f32))
            .chain((0..3).map(|i| (3 + i, yd[i] as f32 / B as f32)))
            .collect()
    };
    let mut rng = Rng::new(0x9);
    // Train on random 3-digit pairs (held-out mask by parity of a hash).
    for _ in 0..90_000 {
        let x = rng.below(1000) as u128;
        let y = rng.below(1000) as u128;
        if (x * 1000 + y).is_multiple_of(5) {
            continue; // hold out ~20%
        }
        let sd = to_digits(x + y, n_out);
        for (i, &d) in sd.iter().enumerate() {
            f.grow(&enc(x, y), n_in + i, digit_target(d), 0.05, 200);
        }
    }
    // Held-out 3-digit exact-match.
    let (mut ok, mut tot) = (0, 0);
    let mut rt = Rng::new(0x1234);
    for _ in 0..600 {
        let x = rt.below(1000) as u128;
        let y = rt.below(1000) as u128;
        if !(x * 1000 + y).is_multiple_of(5) {
            continue;
        }
        f.stimulate(&enc(x, y), 300);
        let got: Vec<usize> = (0..n_out)
            .map(|i| digit_decode(f.cells()[n_in + i]))
            .collect();
        if got == to_digits(x + y, n_out) {
            ok += 1;
        }
        tot += 1;
    }
    ok as f32 / tot.max(1) as f32
}

#[test]
fn exact_place_value_arithmetic() {
    let mut f = grow_place_adder();

    // (1) The grown per-place transducer is EXACT on every single-digit fact.
    let mut exact = 0;
    for a in 0..B {
        for b in 0..B {
            for c in 0..2 {
                let (d, cr) = place(&mut f, a, b, c);
                if d == (a + b + c) % B && cr == usize::from(a + b + c >= B) {
                    exact += 1;
                }
            }
        }
    }

    // (2) Multi-digit addition is EXACT across a battery of LENGTHS (2..18 digits) — one skill, any length.
    let cases: [(u128, u128); 9] = [
        (7, 8),
        (99, 1),
        (12_345, 67_890),
        (99_999, 99_999),
        (500, 500),
        (123_456, 654_321),
        (999_999_999_999, 1),
        (135_792_468_013_579, 864_207_531_986_421),
        (111_111_111_111_111_111, 888_888_888_888_888_889),
    ];
    let mut multi_ok = 0;
    for (x, y) in cases {
        let p = 18;
        if add(&mut f, &to_digits(x, p), &to_digits(y, p)) == to_digits(x + y, p + 1) {
            multi_ok += 1;
        }
    }

    // (3) The MONOLITH baseline: learn addition as one big function. It cannot generalize.
    let mono = grow_monolith();

    eprintln!("[ARITH] grown per-place transducer EXACT on {exact}/200 single-digit facts");
    eprintln!("[ARITH] multi-digit addition EXACT on {multi_ok}/{} cases (lengths up to 18 digits, one grown skill scanned)", cases.len());
    eprintln!("[ARITH] MONOLITH (learn 3-digit addition as one function): held-out exact-match {:.0}% — and STRUCTURALLY 0% at any longer length (fixed input width)", mono * 100.0);
    eprintln!("[ARITH] VERDICT (#129): EXACT multi-digit arithmetic FALLS OUT of `grow` + a carry scan — no new primitive. Carry is a grown threshold (exact, like substrate_operator); the digit is grown over the grown carry concept, so the mod-B wraparound EMERGES as `sum − B·carry` (linear) instead of being fitted as a discontinuous sawtooth; and the ONE grown per-place skill, reused across places with the carry threaded, computes ANY-length sum exactly (the fractal reuse #64) — from only the 200 single-place facts. A monolith that learns arithmetic as one big function memorizes its training length and cannot even represent a longer number. This is the growable PLACE-VALUE skill #127 named, and the same shape as #137's temporal scan (a grown per-step op + a carried state over a sequence).");

    assert_eq!(
        exact, 200,
        "the grown per-place adder must be EXACT on every single-digit fact"
    );
    assert_eq!(
        multi_ok,
        cases.len(),
        "multi-digit addition must be exact at every length"
    );
    assert!(
        mono < 0.5,
        "the monolith must FAIL to generalize (held-out {:.2}) — else there is no place-value advantage to show",
        mono
    );
}
