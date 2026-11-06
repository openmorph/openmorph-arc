# openmorph-core

The digital organism — and the whole project. A **Morph** is an async ecology of cells on one shared
**membrane**: it **fires on change**, **settles** to a fixed point with no global tick, **grows** its own
transducers from surprise, **learns** per event, **acts** on reward, and **lives** on an **energy** economy —
with **zero ML dependencies**, on a CPU down to a microcontroller. The organism is `Field` + `Membrane`.

Two rules govern every line: **nothing is authored** (every faculty is grown/learned — no hand-authored chooser,
no task DSL, no per-modality driver, no hand-set weights) and **simplest is most beautiful** (a real fix deletes machinery).

## The public modules (`src/lib.rs`)

- **`substrate`** — the async **`Field`**: it owns a `Membrane` (its cells *are* the medium) plus `Unit`s (organs:
  read some cells, write one, as `tanh(gain·Σw·x+b)`). A unit fires only when a cell it reads *changed* (or once,
  as a refresh, after its own weights changed outside of firing); firing enqueues its readers — an event cascade
  that **settles** to a fixed point or stops at an anytime **budget**.
  Everything is grown here: `grow` (the one growth verb — surprise-birth of a writer + hidden features), `learn_step`
  (a per-event local delta rule), the alive economy (`metabolize`/`cull`/`reproduce`), `encapsulate`/`nest` (a cell
  can be a Morph — recursion), the concept-map, temporal memory (`integrate`/`delay`/`resonate`), `merge`, and the
  `alive(Event)` entry point (`Act` selects under `Config::act_rule`: the `Valence` rule by default since 1.6).
- **`membrane`** — one growable medium + the universal, self-describing **`Port`**/`Interface` any world plugs into
  (`Flow::{In, Out, Both}`); no privileged perception-vs-action split.
- **`grouping`** — domain-blind structural perception: `segment_objects`, `aggregate`, `relate`.
- **`config`** — the `Config`: every substrate meta-constant as a tunable, serde field, with `Config::schema()`
  (authored vs heritable levers). `Default` is the historical values except `width_norm_min = 128` (#468),
  `act_rule = Valence` and the five 1.6 learning levers: `stability_cap = 0.5` (a cap on the delta rule's base
  rate), `feature_lr = 0.3`, `fanin_init_gain = 3`, `reliance_norm = Mean` and `grow_within_budget = true`.
  `Config::legacy()` restores every 1.6 learning change bit for bit, and acts with Valence (the pre-1.6 `ActScore` rule is removed). See `docs/core/config-reference.md`.
- **`error`** — `OpenMorphError`.

The physics research layer (the substrate's equations as an editable, searchable spec, its engine and the
`EngineField` façade) is not in this crate: it is `openmorph-physics`, which depends on core. Core has one feature,
`std` (default); `--no-default-features` builds it `no_std` + `alloc`.

## Usage

`alive(Event::…)` is the one entry point you drive an organism through; `grow`/`learn_step`
are the low-level primitives it wraps. Prefer `alive`.

```rust
use openmorph_core::{Event, Field};

// Two cells: cell 0 the world writes (input), cell 1 the world reads (output). Units are grown, not authored.
let mut morph = Field::new(2);

// Live loop: each `alive` event settles the cascade, grows units on surprise, and learns per event — no epochs.
for step in 0..4000 {
    let x = -1.0 + 2.0 * ((step % 100) as f32) / 99.0;
    morph.alive(Event::Perceive {
        inputs: &[(0, x)],        // the world writes the input onto cell 0
        target_cell: 1,           // learn to predict cell 1 …
        target: (2.0 * x).tanh(), // … toward the relation y = tanh(2x)
        lr: 0.1,
        budget: 50,               // settle budget; below max_hidden + 2 (66) it also caps how many features grow
    });
}

// Think: perturb and read the settled answer.
morph.stimulate(&[(0, 0.5)], 50);
let answer = morph.cells()[1]; // ≈ tanh(1.0)
assert_eq!(morph.truncated_settles(), 0); // the budget bounded the pool, so no settle was cut short
```

(Measured on the 1.6 default: 28 units, `|answer − tanh(1)|` = 0.018 and no truncated settle. Under
`Config::legacy()` the same loop grows 6 units and answers within 0.0006, but a budget below its pool's settle cost
would truncate silently.)

(An interface can also be a set of self-describing **ports** you `attach` to the membrane; the
driving verb stays `alive`. See `examples/first_organism.rs` for the complete runnable version.)

`f32` throughout, `thiserror` for errors, deterministic (a single FIFO event queue; the only randomness is a
seeded PRNG for grown random-projection features) — research stays byte-replayable. Snapshots round-trip via serde,
and a loaded organism continues bit for bit (`tests/substrate_snapshot_continue.rs`).

## License

[MIT-0](../../LICENSE-MIT0) (MIT No Attribution). Copyright (c) 2026 Petitgen Limited.
