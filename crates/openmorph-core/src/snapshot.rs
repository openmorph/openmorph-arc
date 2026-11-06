// SPDX-License-Identifier: MIT-0
//! The versioned snapshot envelope — how an organism (or anything that carries one) persists across releases.
//!
//! A snapshot is `MAGIC` (`b"OMSN"`), the envelope [`FORMAT`] as a little-endian `u16`, then a JSON payload.
//! JSON is self-describing, so a field added under `#[serde(default)]` loads from an older snapshot and an old
//! genome keeps its legacy levers ([`crate::Config`]'s `snapshot_config`). A positional format (bincode) breaks on
//! every field change, which is why the envelope does not carry one. The version is what a *breaking* change bumps:
//! [`decode`] reads every format up to [`FORMAT`] (migrating an older one in place) and refuses a newer one with a
//! clear error instead of a garbled organism.
//!
//! Bytes without the magic that start with `{` are read as the plain-JSON snapshots written before the envelope
//! (format 0), so every existing Python/WASM/perception snapshot still loads.

use alloc::vec::Vec;
use alloc::{format, string::ToString};
use serde::{de::DeserializeOwned, Serialize};

use crate::OpenMorphError;

/// The four bytes every enveloped snapshot starts with.
pub const MAGIC: [u8; 4] = *b"OMSN";

/// The envelope format this build writes. Format 0 is the bare JSON written before the envelope existed.
pub const FORMAT: u16 = 1;

/// Serialize `value` into an enveloped snapshot (magic, [`FORMAT`], JSON payload).
pub fn encode<T: Serialize + ?Sized>(value: &T) -> Result<Vec<u8>, OpenMorphError> {
    let payload =
        serde_json::to_vec(value).map_err(|e| OpenMorphError::Serialization(e.to_string()))?;
    let mut out = Vec::with_capacity(MAGIC.len() + 2 + payload.len());
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&FORMAT.to_le_bytes());
    out.extend_from_slice(&payload);
    Ok(out)
}

/// The envelope format of `bytes`: `Some(0)` for a pre-envelope JSON snapshot, `Some(v)` for an envelope, `None`
/// when the bytes are neither (a raw bincode blob, a truncated header, another file).
pub fn format_of(bytes: &[u8]) -> Option<u16> {
    match bytes {
        [m0, m1, m2, m3, v0, v1, ..] if [*m0, *m1, *m2, *m3] == MAGIC => {
            Some(u16::from_le_bytes([*v0, *v1]))
        }
        _ if bytes.iter().find(|b| !b.is_ascii_whitespace()) == Some(&b'{') => Some(0),
        _ => None,
    }
}

/// Deserialize a snapshot written by [`encode`] — or by any earlier release (format 0, bare JSON). A snapshot from
/// a NEWER format, bytes that are not a snapshot, and a payload that does not decode as `T` are errors, never panics.
pub fn decode<T: DeserializeOwned>(bytes: &[u8]) -> Result<T, OpenMorphError> {
    let payload = match format_of(bytes) {
        Some(0) => bytes,
        // A breaking change bumps FORMAT and migrates the older payload here (e.g. over a `serde_json::Value`).
        Some(FORMAT) => &bytes[MAGIC.len() + 2..],
        Some(v) => {
            return Err(OpenMorphError::InvalidSnapshot(format!(
                "snapshot format {v} is newer than this build reads ({FORMAT}); upgrade openmorph"
            )))
        }
        None => {
            return Err(OpenMorphError::InvalidSnapshot(
                "not an openmorph snapshot".to_string(),
            ))
        }
    };
    serde_json::from_slice(payload).map_err(|e| OpenMorphError::Serialization(e.to_string()))
}

impl crate::Field {
    /// This organism as an enveloped snapshot ([`encode`]): loads back with [`Self::load`] into a byte-identical
    /// continuation, and keeps loading after a release adds fields.
    pub fn snapshot(&self) -> Vec<u8> {
        encode(self).expect("Field serializes")
    }

    /// Load an organism from [`Self::snapshot`] bytes, or from a bare-JSON snapshot of an earlier release. The
    /// loaded organism is [`Self::validate`]d, so a corrupt or crafted snapshot is rejected here rather than
    /// panicking on a later settle.
    pub fn load(bytes: &[u8]) -> Result<Self, OpenMorphError> {
        let field: Self = decode(bytes)?;
        field.validate()?;
        Ok(field)
    }
}
