// SPDX-License-Identifier: MIT-0
//! EXACT MULTIPLICATION = repeated addition, a scan over #129's exact adder (closes MorphLab's mul wall).
//!
//! The times-table walls a smooth-tanh basis (`mul 0%`, `a·b` 28% held-out) because a raw product is a
//! bilinear function it can't decode-then-multiply. But multiplication needs no new mechanism: `a·b` is `b`
//! added to itself `a` times — a SCAN over the exact per-place adder #129 already grows. So exact
//! multiplication FALLS OUT of exact addition, at ANY length, and generalizes (the adder was trained only on
//! the 200 single-place facts), where a monolith that learns the product directly memorizes and collapses.
//! Same "one scan" shape as #129/#137/#139 — a grown per-step op scanned with a carried state.
//!
//!   cargo test -p openmorph-core --test proofs substrate_multiply -- --nocapture --test-threads=1
use openmorph_core::substrate::Field;

const B: usize = 10;
const A: usize = 0;
const BB: usize = 1;
const CIN: usize = 2;
const COUT: usize = 3;
const DIG: usize = 4;
const AMP: f32 = 0.85;

fn ins(a: usize, b: usize, c: usize) -> Vec<(usize, f32)> {
    vec![
        (A, a as f32 / B as f32),
        (BB, b as f32 / B as f32),
        (CIN, c as f32 / 2.0),
    ]
}
fn dtgt(d: usize) -> f32 {
    ((d as f32 / (B - 1) as f32) * 2.0 - 1.0) * AMP
}
fn ddec(v: f32) -> usize {
    (((v / AMP + 1.0) / 2.0 * (B - 1) as f32).round() as i64).clamp(0, (B - 1) as i64) as usize
}

/// Grow the exact per-place adder (the #129 mechanism): carry = a grown threshold, digit grown over the carry.
fn grow_adder() -> Field {
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
                    f.grow(
                        &ins(a, b, c),
                        COUT,
                        if s >= B { 1.0 } else { -1.0 },
                        0.1,
                        200,
                    );
                    let carry = if f.cells()[COUT] > 0.0 { 1.0 } else { -1.0 };
                    let mut d = ins(a, b, c);
                    d.push((COUT, carry));
                    f.grow(&d, DIG, dtgt(s % B), 0.1, 200);
                }
            }
        }
    }
    f
}
fn place(f: &mut Field, a: usize, b: usize, c: usize) -> (usize, usize) {
    f.stimulate(&ins(a, b, c), 200);
    let carry = usize::from(f.cells()[COUT] > 0.0);
    let mut d = ins(a, b, c);
    d.push((COUT, if carry == 1 { 1.0 } else { -1.0 }));
    f.stimulate(&d, 200);
    (ddec(f.cells()[DIG]), carry)
}
/// x + y (little-endian digit vecs) by scanning the grown adder across places (the #129 carry scan).
fn add(f: &mut Field, x: &[usize], y: &[usize]) -> Vec<usize> {
    let p = x.len().max(y.len());
    let mut out = Vec::with_capacity(p + 1);
    let mut carry = 0;
    for k in 0..p {
        let (a, b) = (
            x.get(k).copied().unwrap_or(0),
            y.get(k).copied().unwrap_or(0),
        );
        let (dd, c) = place(f, a, b, carry);
        out.push(dd);
        carry = c;
    }
    out.push(carry);
    out
}
/// a · y = y added to itself `a` times — the OUTER scan over the exact adder. Exact for any length.
fn mul_by_digit(f: &mut Field, a: usize, y: &[usize]) -> Vec<usize> {
    let mut acc = vec![0usize; 1];
    for _ in 0..a {
        acc = add(f, &acc, y);
        while acc.len() > 1 && *acc.last().unwrap() == 0 {
            acc.pop();
        } // trim leading zeros
    }
    acc
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
fn value(d: &[usize]) -> u128 {
    d.iter()
        .rev()
        .fold(0u128, |acc, &x| acc * B as u128 + x as u128)
}

#[test]
fn exact_multiplication_is_repeated_addition() {
    let mut f = grow_adder();

    // Single-digit × multi-digit, exact, across LENGTHS the adder never saw as a whole (it saw single places).
    let cases: [(usize, u128); 8] = [
        (7, 8),
        (9, 99),
        (3, 12345),
        (8, 99999),
        (6, 123456),
        (9, 987654321),
        (7, 111111),
        (4, 250),
    ];
    let mut ok = 0;
    for (a, y) in cases {
        let got = value(&mul_by_digit(&mut f, a, &to_digits(y, 12)));
        if got == a as u128 * y {
            ok += 1;
        } else {
            eprintln!("[MUL]   {a}·{y}: got {got} want {}", a as u128 * y);
        }
    }
    // The single-digit TIMES-TABLE (a·b, both digits) — exact, and it GENERALIZES (never trained on products).
    let mut table = 0;
    for a in 0..B {
        for b in 0..B {
            if value(&mul_by_digit(&mut f, a, &to_digits(b as u128, 2))) == (a * b) as u128 {
                table += 1;
            }
        }
    }

    eprintln!(
        "[MUL] single-digit × multi-digit exact on {ok}/{} cases (up to 10 digits)",
        cases.len()
    );
    eprintln!("[MUL] the full 10×10 TIMES-TABLE exact on {table}/100 (generalizes — trained ONLY on 200 single-place ADD facts, never a product)");
    eprintln!("[MUL] VERDICT: the multiplication wall (a smooth basis can't decode-then-multiply — MorphLab mul 0%) needs NO new mechanism. a·b is b added to itself a times: an OUTER scan over the EXACT per-place adder #129 already grows. So exact multiplication FALLS OUT of exact addition — the whole times-table correct, at any length, from ONLY the single-place add facts (no product ever trained), where a monolith over the raw product memorizes and collapses on held-out combos. Same one-scan shape as #129/#137/#139. The residual (multi-DIGIT × multi-digit, and the smooth-basis kink/high-freq walls) is the honest next rung.");

    assert_eq!(
        ok,
        cases.len(),
        "single-digit × multi-digit must be exact at every length"
    );
    assert_eq!(
        table, 100,
        "the full times-table must be exact and generalize (no product ever trained)"
    );
}
