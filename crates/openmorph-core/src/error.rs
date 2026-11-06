// SPDX-License-Identifier: MIT-0
//! Error types for openmorph-core.

#[cfg(not(feature = "std"))]
use alloc::string::String;

/// Errors that can occur in openmorph-core operations.
#[derive(Debug, thiserror::Error)]
pub enum OpenMorphError {
    /// Serialization or deserialization failure (a `Field` round-trip).
    #[error("serialization error: {0}")]
    Serialization(String),
    /// A morph could not be transferred into a physics because a required variable is missing or the
    /// matter layout does not match — the `Port::compatible`-style gate refusing a lossy transfer (§6). Raised by
    /// the `openmorph-physics` crate (its `transfer` / `Library`).
    #[error("incompatible physics transfer: {0}")]
    Incompatible(String),
    /// A named physics or morph was not present in an `openmorph-physics` `Library` (Phase 4) — a resolution
    /// miss (a typo / an un-added entry), distinct from an [`Self::Incompatible`] transfer refusal.
    #[error("not found in library: {0}")]
    NotFound(String),
    /// A deserialized [`crate::Field`] violated a structural invariant the settle/learn paths assume — a
    /// unit reading a cell that does not exist, a length-mismatched parallel array, an out-of-range
    /// `gate`/`factor`/adjacency index, etc. Reported by [`crate::Field::validate`], the OPT-IN check a
    /// careful caller runs after loading an UNTRUSTED snapshot (serde itself does not enforce these, so a
    /// crafted snapshot would otherwise panic-on-use). Memory-safe either way (a Rust panic, not an OOB
    /// access), but this turns a later panic into an up-front, descriptive rejection.
    #[error("invalid snapshot: {0}")]
    InvalidSnapshot(String),
}

impl OpenMorphError {
    /// True when this error is the physics-transfer compatibility refusal ([`Self::Incompatible`]) — a small
    /// ergonomic for `openmorph-physics`'s portability proof.
    pub fn is_incompatible(&self) -> bool {
        matches!(self, OpenMorphError::Incompatible(_))
    }
}

/// Convenience alias for `Result<T, OpenMorphError>`.
pub type Result<T> = core::result::Result<T, OpenMorphError>;
