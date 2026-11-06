// SPDX-License-Identifier: MIT-0
//! STRESS — the v2 scalability sweeps: instruments that report WHERE scaling becomes impractical (width, depth,
//! recurrence, library routing, the economy, the delay bank), not per-push regression gates. Every test is
//! `#[ignore = "stress: nightly"]`; CI runs this tier nightly (`.github/workflows/ci.yml`, job `nightly`).
//! No assertion reads the wall clock: costs are asserted in the organism's own events (`Field::clock`,
//! `nested_settles`, emitted updates) and timings are printed for information only. (Three sweeps — width, depth,
//! library — still STOP early on a wall-time budget, so how far their printed tables reach depends on the machine.)
//!
//!     cargo test -p openmorph-core --test stress -- --ignored --nocapture
//!     cargo test -p openmorph-core --test stress v2_scale_width -- --ignored

#[path = "../common/mod.rs"]
mod common;

mod v2_scale_delaybank;
mod v2_scale_depth;
mod v2_scale_economy;
mod v2_scale_library;
mod v2_scale_recurrence;
mod v2_scale_width;
