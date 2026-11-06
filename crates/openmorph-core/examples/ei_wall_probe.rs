// SPDX-License-Identifier: MIT-0
//! rc2 — does INHIBITION break the parity wall? A self-contained mechanism test (no ecology).
//!
//! Hypothesis (Sébastien): the substrate can't compute XOR/parity because it lacks genuine
//! excitatory/inhibitory cells (signed weights). A single linear-threshold unit cannot express
//! parity (not linearly separable). The classic fix is a HIDDEN LAYER — and biology's hidden layer
//! is a mix of excitatory + inhibitory cells with SIGNED connections, learned locally (no backprop).
//!
//! We test three feature spaces feeding one locally-learned (perceptron) linear readout:
//!   RAW      — the input bits themselves (a single linear unit; the current substrate's reach).
//!   POS-EXP  — H random hidden threshold units with NON-NEGATIVE (excitatory-only) weights.
//!   E/I-EXP  — H random hidden threshold units with SIGNED (excitatory + inhibitory) weights.
//! If E/I-EXP breaks held-out parity where RAW and POS-EXP don't, inhibition is the missing piece.
//!
//! Nature-based: random fixed hidden connectivity + a local (perceptron) output rule is reservoir /
//! extreme-learning-machine computing — biologically plausible (random projections + local learning),
//! NOT backprop. The hidden layer is not even trained here.
//!
//! Run:  cargo run --release -p openmorph-core --example ei_wall_probe

// splitmix64 — deterministic (no Math::random), so a run is reproducible.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 11) as f32 / (1u64 << 53) as f32 // [0,1)
    }
    // a weight in {-1,0,+1} (signed) or {0,+1} (positive-only).
    fn weight(&mut self, signed: bool) -> f32 {
        let u = self.unit();
        if signed {
            if u < 0.34 {
                -1.0
            } else if u < 0.67 {
                0.0
            } else {
                1.0
            }
        } else if u < 0.5 {
            0.0
        } else {
            1.0
        }
    }
}

fn bits(c: usize, n: usize) -> Vec<f32> {
    (0..n).map(|i| ((c >> i) & 1) as f32).collect()
}

// A random hidden threshold layer: h_j = step(v_j · x + bias_j > 0). Fixed (a reservoir).
struct Hidden {
    w: Vec<Vec<f32>>, // [H][n]
    b: Vec<f32>,      // [H]
}
impl Hidden {
    fn random(h: usize, n: usize, signed: bool, rng: &mut Rng) -> Self {
        let w = (0..h)
            .map(|_| (0..n).map(|_| rng.weight(signed)).collect())
            .collect();
        // biases spread across the possible pre-activation range so units tile different thresholds.
        let b = (0..h)
            .map(|_| (rng.unit() * (n as f32 + 1.0)).floor() - n as f32 * 0.5)
            .collect();
        Hidden { w, b }
    }
    fn feats(&self, x: &[f32]) -> Vec<f32> {
        self.w
            .iter()
            .zip(&self.b)
            .map(|(wj, &bj)| {
                let s: f32 = wj.iter().zip(x).map(|(a, b)| a * b).sum::<f32>() + bj;
                if s > 0.0 {
                    1.0
                } else {
                    0.0
                }
            })
            .collect()
    }
}

// perceptron: learn a signed weight + bias over features to reproduce a binary label. Local rule.
fn perceptron_fit(feats: &[Vec<f32>], y: &[usize]) -> (Vec<f32>, f32) {
    let d = feats[0].len();
    let mut w = vec![0.0f32; d];
    let mut b = 0.0f32;
    let lr = 0.1f32;
    for _ in 0..2000 {
        let mut errs = 0;
        for (f, &yi) in feats.iter().zip(y) {
            let pred = if w.iter().zip(f).map(|(a, x)| a * x).sum::<f32>() + b > 0.0 {
                1
            } else {
                0
            };
            let e = yi as f32 - pred as f32;
            if e != 0.0 {
                errs += 1;
                for (wi, &xi) in w.iter_mut().zip(f) {
                    *wi += lr * e * xi;
                }
                b += lr * e;
            }
        }
        if errs == 0 {
            break;
        }
    }
    (w, b)
}
fn eval(w: &[f32], b: f32, f: &[f32]) -> usize {
    if w.iter().zip(f).map(|(a, x)| a * x).sum::<f32>() + b > 0.0 {
        1
    } else {
        0
    }
}

// (train-fit, held-out) for a feature space, averaged over `seeds` random reservoirs.
fn run(
    n: usize,
    label: &dyn Fn(usize) -> usize,
    held: &[usize],
    mode: &str,
    h: usize,
    seeds: u64,
) -> (f32, f32) {
    let train: Vec<usize> = (0..(1 << n)).filter(|c| !held.contains(c)).collect();
    let (mut tr, mut ho) = (0.0f32, 0.0f32);
    for s in 0..seeds {
        let mut r2 = Rng(0xEED ^ (s.wrapping_mul(0x9E37)));
        let hid = match mode {
            "raw" => None,
            "pos" => Some(Hidden::random(h, n, false, &mut r2)),
            _ => Some(Hidden::random(h, n, true, &mut r2)),
        };
        let feat = |c: usize| -> Vec<f32> {
            let x = bits(c, n);
            match &hid {
                None => x,
                Some(hh) => hh.feats(&x),
            }
        };
        let tf: Vec<Vec<f32>> = train.iter().map(|&c| feat(c)).collect();
        let ty: Vec<usize> = train.iter().map(|&c| label(c)).collect();
        let (w, b) = perceptron_fit(&tf, &ty);
        tr += train
            .iter()
            .filter(|&&c| eval(&w, b, &feat(c)) == label(c))
            .count() as f32
            / train.len() as f32;
        ho += held
            .iter()
            .filter(|&&c| eval(&w, b, &feat(c)) == label(c))
            .count() as f32
            / held.len() as f32;
    }
    (tr / seeds as f32, ho / seeds as f32)
}

fn sweep(name: &str, n: usize, label: &dyn Fn(usize) -> usize, held: &[usize], seeds: u64) {
    println!("── {name} (n={n}, held {held:?}, chance 0.5) ──  [train-fit / held-out]");
    let (t, h0) = run(n, label, held, "raw", 0, seeds);
    println!("  RAW (1 unit)   {t:.2} / {h0:.2}");
    for hh in [8usize, 32] {
        let (tp, hp) = run(n, label, held, "pos", hh, seeds);
        let (te, he) = run(n, label, held, "ei", hh, seeds);
        println!("  POS  H={hh:<3}     {tp:.2} / {hp:.2}        E/I  H={hh:<3}  {te:.2} / {he:.2}");
    }
}

fn popcount(c: usize, n: usize) -> usize {
    (0..n).map(|i| (c >> i) & 1).sum()
}

fn main() {
    let seeds = 40;
    println!("Does inhibition break the wall? random reservoir + one local perceptron readout.\n");
    // 1) PARITY — needs hidden inhibition (XOR), but pathological to generalize from partial data.
    sweep(
        "parity3",
        3,
        &|c| popcount(c, 3) % 2,
        &[0b011, 0b100],
        seeds,
    );
    // 2) PARITY, FULL data (train==test) — isolates EXPRESSIVENESS from generalization.
    sweep("parity3-FULLFIT", 3, &|c| popcount(c, 3) % 2, &[], seeds);
    // 3) BAND: exactly-2-of-4 — non-monotone (needs inhibition to cap the high end) but GENERALIZABLE.
    sweep(
        "exactly2of4",
        4,
        &|c| usize::from(popcount(c, 4) == 2),
        &[0b0011, 0b1100],
        seeds,
    );
    println!("\nread-off:");
    println!("  parity3: if held-out stays ≈ chance for ALL, parity-from-partial is a STRUCTURE-");
    println!(
        "    DISCOVERY wall, not an inhibition one (the authored Xor is just a hardcoded prior)."
    );
    println!(
        "  parity3-FULLFIT: if E/I train-fit ≈ 1.0, inhibition DOES let the substrate EXPRESS"
    );
    println!("    parity — expressiveness is not the gap; generalization/discovery is.");
    println!(
        "  exactly2of4: a task that needs inhibition AND generalizes — if E/I held-out > POS, that"
    );
    println!("    is inhibition genuinely helping where it can.");
}
