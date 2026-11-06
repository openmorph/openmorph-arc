// SPDX-License-Identifier: MIT-0
//! `no_std` compatibility shims. Under the default `std` feature these ARE the standard-library types and
//! methods (byte-identical to before); under `--no-default-features` (bare metal, `#![no_std] + alloc`) floats
//! come from `num-traits`/`libm` and the hash maps become deterministic `alloc::collections` B-trees. This is
//! the single home the substrate reaches through for anything the `core`/`alloc` prelude lacks — and the natural
//! home for the (future) CANONICAL cross-host float implementation (D2 hash/replay determinism).

// ── collections: std hash maps by default; ordered B-tree maps on bare metal (alloc-only, deterministic). The
//    substrate's maps are transient caches keyed by `Ord` types (Vec<u32>, (usize,usize), usize), so a B-tree is
//    a drop-in with an added bonus: deterministic iteration. ──
#[cfg(not(feature = "std"))]
pub use alloc::collections::{BTreeMap as HashMap, BTreeSet as HashSet};
#[cfg(feature = "std")]
pub use std::collections::{HashMap, HashSet};

// ── floats: on bare metal, `num-traits::Float` re-supplies the f32 math methods (`tanh`/`abs`/`sqrt`/…) via
//    libm. (`clamp`/`min`/`max` are comparison-based and already in `core`, so no shim is needed for them.)
//    Under std the inherent f32 methods shadow these, so nothing changes. ──
#[cfg(not(feature = "std"))]
pub use num_traits::Float;
