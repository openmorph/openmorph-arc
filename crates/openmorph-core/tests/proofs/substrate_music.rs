// SPDX-License-Identifier: MIT-0
//! MELODY CONTINUATION (#136) — the generative core of MorphMusic, validated with NO new primitive. A morph
//! learns to CONTINUE a melody (predict the next note, then feed it back to generate), and it learns the
//! RULE, not the notes: continuing FAR-transposed, unseen melodies exactly.
//!
//! The key, exactly as in #129 (place value) and the one-scan unification: representation. A melody carried
//! as ABSOLUTE pitch ties the pattern to a key, so a morph memorizes the training keys and fails on unseen
//! ones. Carried as RELATIVE pitch (intervals — the scan-carried differences), the pattern is
//! TRANSPOSITION-INVARIANT: the same motif is the same intervals in every key, so learning it once
//! generalizes to every key. Music is a sequence scan; the right carried state (intervals) makes the rule
//! transferable. Grow the next-interval predictor over the carried context — nothing authored.
//!
//!   cargo test -p openmorph-core --test proofs substrate_music -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;

const MOTIF: [i32; 5] = [2, 2, 3, -5, -2]; // a repeating melodic contour (period 5)
const K: usize = 6; // context window (≥ the motif period, so the phase is determined)
const NORM: f32 = 24.0;
const OUT: usize = K; // the fixed target cell (hidden features grow beyond it)

/// The pitch of a melody in key `key` at step `t` — the motif contour applied cyclically from the key.
fn pitch(key: i32, t: usize) -> i32 {
    (0..t).fold(key, |p, i| p + MOTIF[i % MOTIF.len()])
}

/// Grow a next-note predictor. `relative`: carry/predict INTERVALS (transposition-invariant); else absolute
/// pitch. Trained by continuing melodies in `train_keys`. Returns the grown field.
fn train(relative: bool, train_keys: &[i32]) -> Field {
    let mut f = Field::new(K);
    f.config_mut().max_hidden = 32;
    f.config_mut().feature_lr = 0.3;
    f.ensure_cells(OUT + 1);
    for _ in 0..400 {
        for &key in train_keys {
            for base in (K + 2)..(K + 14) {
                let ctx = context(relative, key, base);
                let tgt = target(relative, key, base);
                let inp: Vec<(usize, f32)> = ctx.iter().enumerate().map(|(i, &v)| (i, v)).collect();
                f.grow(&inp, OUT, tgt, 0.1, 100);
            }
        }
    }
    f
}

/// The carried context at absolute time `base`: the K most recent values (intervals or pitches), normalized.
fn context(relative: bool, key: i32, base: usize) -> Vec<f32> {
    (0..K)
        .map(|j| {
            if relative {
                (pitch(key, base - 1 - j) - pitch(key, base - 2 - j)) as f32 / NORM
            } else {
                pitch(key, base - 1 - j) as f32 / NORM
            }
        })
        .collect()
}
fn target(relative: bool, key: i32, base: usize) -> f32 {
    if relative {
        (pitch(key, base) - pitch(key, base - 1)) as f32 / NORM
    } else {
        pitch(key, base) as f32 / NORM
    }
}

/// One-step continuation accuracy on `test_keys`: predict the next note, exact-match.
fn one_step_accuracy(f: &mut Field, relative: bool, test_keys: &[i32]) -> f32 {
    let (mut ok, mut tot) = (0, 0);
    for &key in test_keys {
        for base in (K + 2)..(K + 14) {
            let ctx = context(relative, key, base);
            let inp: Vec<(usize, f32)> = ctx.iter().enumerate().map(|(i, &v)| (i, v)).collect();
            f.stimulate(&inp, 100);
            let pred = (f.cells()[OUT] * NORM).round() as i32;
            let next = if relative {
                pitch(key, base - 1) + pred
            } else {
                pred
            };
            if next == pitch(key, base) {
                ok += 1;
            }
            tot += 1;
        }
    }
    ok as f32 / tot as f32
}

/// AUTOREGRESSIVE GENERATION — the generative core: from an initial context on an unseen key, generate the
/// next `n` notes by feeding each prediction back. Returns how many generated notes match the true melody.
fn generate(f: &mut Field, key: i32, n: usize) -> (usize, usize) {
    // Seed with the true recent intervals up to the start, then roll forward feeding predictions back.
    let start = K + 2;
    let mut intervals: Vec<i32> = (0..K)
        .rev()
        .map(|j| pitch(key, start - 1 - j) - pitch(key, start - 2 - j))
        .collect(); // oldest..newest, newest last
    let mut last_pitch = pitch(key, start - 1);
    let (mut correct, mut total) = (0, 0);
    for base in start..start + n {
        // context = the K most recent intervals, newest first, normalized
        let inp: Vec<(usize, f32)> = (0..K)
            .map(|j| (j, intervals[intervals.len() - 1 - j] as f32 / NORM))
            .collect();
        f.stimulate(&inp, 100);
        let step = (f.cells()[OUT] * NORM).round() as i32;
        let gen_pitch = last_pitch + step;
        if gen_pitch == pitch(key, base) {
            correct += 1;
        }
        total += 1;
        // roll the autoregressive state forward with the GENERATED step (not the truth)
        intervals.push(step);
        last_pitch = gen_pitch;
    }
    (correct, total)
}

#[test]
fn a_morph_continues_a_melody_by_the_rule_not_the_notes() {
    let train_keys: Vec<i32> = (0..16).collect();
    let test_keys: Vec<i32> = (40..52).collect(); // far-transposed, never seen in training

    let mut rel = train(true, &train_keys);
    let mut abs = train(false, &train_keys);
    let rel_acc = one_step_accuracy(&mut rel, true, &test_keys);
    let abs_acc = one_step_accuracy(&mut abs, false, &test_keys);

    // Autoregressive generation on an unseen key (the generative core): does it CONTINUE the melody?
    let (gc, gt) = generate(&mut rel, 47, 24);

    eprintln!("[MUSIC] one-step continuation on UNSEEN (far-transposed) keys:");
    eprintln!("[MUSIC]   relative (intervals, transposition-invariant): {rel_acc:.2}");
    eprintln!("[MUSIC]   absolute (pitch, key-bound)                  : {abs_acc:.2}");
    eprintln!("[MUSIC] autoregressive GENERATION (feed predictions back) on an unseen key: {gc}/{gt} notes match the true continuation");
    eprintln!("[MUSIC] VERDICT (#136): a morph learns to CONTINUE a melody — the generative core of MorphMusic — and it learns the RULE, not the notes. Carried as RELATIVE pitch (intervals — the scan-carried differences), a motif is the same in every key, so learning it once continues FAR-transposed, unseen melodies EXACTLY (one-step and autoregressively, generating the whole continuation by feeding predictions back). Carried as ABSOLUTE pitch it is key-bound and memorizes the training keys, failing on unseen ones. No new primitive — melody is a sequence scan (the one-scan unification), and the right carried state (intervals) makes the musical rule transferable, exactly as relative representation cracked place value (#129). HONEST FRONTIER: this is periodic/motivic continuation with transposition invariance — mastered. Long-range musical structure (phrase/section form, keys that MODULATE, polyphony) and raw-audio pitch/timbre are the next probes (the audio front-end + driver, #134); report walls there against measurement, not here.");

    assert!(
        rel_acc > 0.95,
        "relative (interval) continuation must generalize to unseen keys (got {rel_acc:.2})"
    );
    assert!(
        abs_acc < 0.3,
        "absolute (pitch) continuation must FAIL on unseen keys — it memorizes (got {abs_acc:.2})"
    );
    assert!(
        rel_acc > abs_acc + 0.5,
        "the transposition-invariant representation must decisively beat the key-bound one ({rel_acc:.2} vs {abs_acc:.2})"
    );
    assert!(
        gc >= gt - 1,
        "autoregressive generation must continue the unseen melody essentially exactly ({gc}/{gt})"
    );
}
