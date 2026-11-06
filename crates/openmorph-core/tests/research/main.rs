// SPDX-License-Identifier: MIT-0
//! RESEARCH — spikes, repro probes, calibrations and print-only measurements: the instruments behind a design
//! decision (the surprise-currency increments, the Field-Theory branches, the falsified state-key probe, …), not
//! capability gates. Every test is `#[ignore]`d — `"research: on demand"`, or an older, more specific reason — and
//! CI runs the tier nightly as a non-blocking report (`.github/workflows/ci.yml`, job `nightly`). A spike whose
//! claim is later shipped as a capability should move to `tests/proofs` (a `git mv` + one `mod` line).
//!
//!     cargo test -p openmorph-core --test research -- --ignored --nocapture
//!     cargo test -p openmorph-core --test research spike_ -- --ignored

#[path = "../common/mod.rs"]
mod common;

mod behavioural_boundaries;
mod law_bootstrap;
mod meta_discovery_regimes;
mod repro_compose_probe;
mod repro_depth_probe;
mod spike_ablation_sensitivity;
mod spike_currency_stability;
mod spike_group_by_surprise;
mod spike_metabolize_surprise;
mod spike_surprise_currency;
mod substrate_conservation;
mod substrate_curvature_time;
mod substrate_meta_evolve;
mod substrate_novelty_field;
mod substrate_operator_minimality;
mod substrate_predictive_latent;
mod substrate_relational_koopman;
mod substrate_state_key;
mod substrate_temperature;
mod tropical_fan;
