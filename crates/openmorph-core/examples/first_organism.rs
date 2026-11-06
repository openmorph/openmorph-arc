// SPDX-License-Identifier: MIT-0
//! YOUR FIRST ORGANISM — teach a Morph the relation `y = tanh(2x)` from a stream of examples, learning AT
//! INFERENCE (no epochs, no backprop, no argmax). This is the 90% path: create a `Field`, drive it with
//! `alive(Event::Perceive { .. })`, then `stimulate` + read a cell.
//!
//! Run it:  `cargo run --release --example first_organism`
//!
//! (`alive` is the one entry point you drive the organism through; `grow`/`learn_step` are the primitives it
//! wraps. Prefer `alive`.)

use openmorph_core::{Event, Field};

fn main() {
    // A Morph is an ecology of UNITS on a membrane of CELLS. Here: 2 cells — cell 0 the world writes (the input),
    // cell 1 the world reads (the output). Units are GROWN from surprise as we teach it; we author none.
    let mut morph = Field::new(2);

    // TEACH at inference: a stream of (input → target) events. Each event grows units on surprise, learns per
    // event with a local delta rule, and self-sizes the grown basis via the energy economy. No epoch loop.
    for step in 0..4000 {
        let x = -1.0 + 2.0 * ((step % 100) as f32) / 99.0; // a deterministic sweep of [-1, 1] (reproducible)
        morph.alive(Event::Perceive {
            inputs: &[(0, x)],        // the world writes the input onto cell 0
            target_cell: 1,           // learn to predict cell 1 …
            target: (2.0 * x).tanh(), // … toward the relation y = tanh(2x)
            lr: 0.1,
            // An anytime settle budget (max firings per settle). A settle of a grown target costs its hidden
            // features + 2, and `grow` may birth up to `max_hidden` (64) of them, so give it at least 66.
            budget: 100,
        });
    }

    // ASK: write an input, let the cascade settle, read the output cell. On held-out points the grown organism
    // reproduces the relation it was never given a formula for.
    println!(" input   predicted     truth");
    let mut worst = 0.0f32;
    for i in 0..=8 {
        let x = -1.0 + 2.0 * i as f32 / 8.0;
        morph.stimulate(&[(0, x)], 100); // perturb + settle
        let y = morph.cells()[1]; // read the settled output
        let truth = (2.0 * x).tanh();
        worst = worst.max((y - truth).abs());
        println!("{x:+.3}    {y:+.4}     {truth:+.4}");
    }
    println!(
        "\n{} units grown; worst held-out error {worst:.4}.",
        morph.n_units()
    );
    assert!(
        worst < 0.1,
        "the grown organism should track y=tanh(2x) closely"
    );
    // A settle that ran out of budget returns a not-yet-settled value without an error; this counts them.
    assert_eq!(
        morph.truncated_settles(),
        0,
        "every settle completed within its budget"
    );
}
