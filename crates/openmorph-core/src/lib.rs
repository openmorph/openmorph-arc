// SPDX-License-Identifier: MIT-0
#![cfg_attr(not(feature = "std"), no_std)]
//! # openmorph-core
//!
//! The digital organism — now the **async membrane substrate**: a growable medium of cells that any
//! interface plugs into, and a set of units that fire on change, settle without a global tick, grow
//! their own transducers from surprise, learn per-event, act on reward, and live on an energy
//! economy. Zero ML dependencies; runs on a CPU down to a microcontroller.
//!
//! The tick-based organism (`organ_birth`/`organ_compose`/`motor_organ`/`population`/`laboratory`)
//! was retired once the async substrate reached parity (git remembers it). What remains is the
//! substrate itself — see [`membrane`] and [`substrate`] — plus [`grouping`], the nature-based
//! label-propagation field for segmenting a raw channel into objects.
//!
//! # Example — grow a transducer from a stream (nothing authored)
//!
//! A blank organism grows its OWN mapping for a channel it has never seen: it births a unit on
//! surprise and tunes it per event — no epoch, no authored encoder.
//!
//! ```
//! use openmorph_core::Field;
//!
//! let mut morph = Field::new(2); // cell 0 = input channel, cell 1 = target channel
//! morph.config_mut().max_hidden = 0; // linear-in-tanh — grow the writer, no hidden depth
//! // Feed a stream; the morph grows + learns the relation target = tanh(1.5 · x).
//! for _ in 0..2_000 {
//!     for &x in &[-1.0_f32, 1.0] {
//!         morph.grow(&[(0, x)], 1, (1.5 * x).tanh(), 0.1, 50);
//!     }
//! }
//! morph.stimulate(&[(0, 1.0)], 50); // perturb with a fresh input and let it settle
//! assert!((morph.cells()[1] - (1.5_f32).tanh()).abs() < 0.05); // it grew the mapping
//! ```

#![allow(clippy::needless_range_loop)]
#![allow(clippy::type_complexity)]

extern crate alloc;

mod compat;
pub mod config;
pub mod error;
pub mod grouping;
pub mod membrane;
pub mod snapshot;
pub mod substrate;

// Re-export the core surface at crate root.
pub use config::{ActRule, Activation, Config, LeverInfo, LeverKind, RelianceNorm, RuleGenome};
pub use error::{OpenMorphError, Result};
pub use membrane::{Binding, Flow, Interface, Membrane, Port};
pub use substrate::{
    cosine_distance, Episode, Event, Field, Heartbeat, Nested, Outcome, SkillIndex, Unit,
};
