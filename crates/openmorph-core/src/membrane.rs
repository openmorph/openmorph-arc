// SPDX-License-Identifier: MIT-0
//! The MEMBRANE — one growable medium the organism lives against, and the universal port any
//! interface plugs into. This is rc6's foundation: there is no privileged "perception" vs "action",
//! only **channels** on a shared medium. A channel the *world* writes, the morph senses; one the
//! *world* reads, the morph acts through; one it does both, is both — like a real membrane.
//!
//! An interface is **self-describing** (its [`Port`]s — the USB descriptor / protobuf schema), and
//! it **exchanges** raw values with the medium at its own pace. The membrane knows nothing about any
//! modality: it only allocates medium slices and moves raw floats. All meaning (turning a raw
//! channel into useful cells, and cell patterns into actions) is **grown by the organism**, never
//! authored here — so ANY interface plugs in with no per-modality driver, and new ones attach on the
//! fly. ARC, a game, a webcam, a program: each is just a set of ports on the medium.
//!
//! Step 1 (this module): the medium + dynamic attach/detach + heterogeneous plug-and-play. Grown
//! transduction, dissolving the perceive/act split, answering-as-act, and the async scheduler come
//! next — see `docs/rc6-membrane-async.md`.

use serde::{Deserialize, Serialize};

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use crate::compat::Float;
#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use alloc::{
    boxed::Box,
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};

/// Direction of a port, from the WORLD's point of view. The morph's "sensor vs motor" role is
/// EMERGENT (from which cells its organs read vs write) — the flow only says who the *world* is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Flow {
    /// The world WRITES this channel each exchange (the morph senses it).
    In,
    /// The world READS this channel each exchange (the morph acts through it).
    Out,
    /// The world writes AND reads it — a shared channel (the morph acts, then senses its own act;
    /// e.g. a game-select / self-state / mouse-position channel). First-class, not a hack.
    Both,
}

/// One raw port on an interface — the self-description (a protobuf schema field): a name, a flow, a
/// width, and an optional raw shape. RAW structure only; never derived meaning (that would re-author
/// what the morph should grow).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Port {
    /// Human/harness label (the morph ignores it — meaning is grown, not named).
    pub name: String,
    /// Who the world is on this channel.
    pub flow: Flow,
    /// Number of medium cells this port occupies.
    pub dim: usize,
    /// Optional raw STRUCTURE hint (e.g. a 2-D grid `(width, height)`), not meaning.
    pub shape: Option<(usize, usize)>,
}

impl Port {
    fn bare(name: &str, flow: Flow, dim: usize) -> Self {
        Self {
            name: name.into(),
            flow,
            dim,
            shape: None,
        }
    }
    /// A world-writes (sensory) port.
    pub fn input(name: &str, dim: usize) -> Self {
        Self::bare(name, Flow::In, dim)
    }
    /// A world-reads (motor) port.
    pub fn output(name: &str, dim: usize) -> Self {
        Self::bare(name, Flow::Out, dim)
    }
    /// A world-writes-and-reads (shared) port.
    pub fn both(name: &str, dim: usize) -> Self {
        Self::bare(name, Flow::Both, dim)
    }
    /// Attach a raw 2-D structure hint.
    pub fn with_shape(mut self, w: usize, h: usize) -> Self {
        self.shape = Some((w, h));
        self
    }

    /// Whether a value on THIS port fits a consumer expecting `other` — the interface contract: same
    /// name (the declared representation TYPE) and same width. A producer of a different interface
    /// declares a different name, so a mismatch is caught explicitly instead of silently composing
    /// garbage. (Semantic honesty beyond the declared type is a reputation matter — see `mycelium`.)
    pub fn compatible(&self, other: &Port) -> bool {
        self.name == other.name && self.dim == other.dim
    }
}

/// Where an attached interface's ports live in the medium — returned by [`Membrane::attach`] and
/// passed back to the interface each exchange so it can address its own slices. Also carries the
/// `Port` descriptors so a live binding SELF-DESCRIBES (a host/UI can enumerate what is plugged in and
/// render each channel with no per-driver code).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Binding {
    /// Per-port `(offset, dim, flow)` into the medium, in the order the interface declared them.
    pub slots: Vec<(usize, usize, Flow)>,
    /// The port descriptors, in the same order as `slots` (introspection). `#[serde(default)]` so a
    /// snapshot predating this field loads with an empty descriptor list.
    #[serde(default)]
    pub ports: Vec<Port>,
}

impl Binding {
    /// The medium offset + dim of port `i` (the order it was declared). An out-of-range `i` returns
    /// `(0, 0)` (a zero-width slot) rather than panicking; byte-identical to the direct index for every
    /// valid `i`.
    pub fn slot(&self, i: usize) -> (usize, usize) {
        match self.slots.get(i) {
            Some(&(o, d, _)) => (o, d),
            None => (0, 0),
        }
    }
}

/// A self-describing thing the morph can plug into the membrane. The world implements this — a game,
/// an ARC task, a webcam, a program. `ports` is the descriptor; `exchange` moves raw values in/out at
/// the interface's OWN pace (nothing forces a global tick — a fast mouse and a slow API both fit).
pub trait Interface {
    /// The channels this interface exposes (its schema).
    fn ports(&self) -> Vec<Port>;
    /// Move raw values: WRITE the medium for `In`/`Both` ports, READ it for `Out`/`Both` ports.
    fn exchange(&mut self, m: &mut Membrane, at: &Binding);
}

/// THE MEMBRANE — a growable medium (the shared cell values) plus the live bindings. Attaching an
/// interface grows the medium by its ports' total width and returns a [`Binding`]; detaching zeroes
/// its region. The membrane never interprets a value — it is pure substrate.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Membrane {
    /// The shared medium — the cells the organism reads and writes.
    pub medium: Vec<f32>,
    /// Live bindings (one per attached interface), for detach + introspection.
    bindings: Vec<Binding>,
}

impl Membrane {
    /// An empty membrane.
    pub fn new() -> Self {
        Self::default()
    }

    /// Total medium width (number of cells).
    pub fn len(&self) -> usize {
        self.medium.len()
    }
    /// Whether the medium has no cells.
    pub fn is_empty(&self) -> bool {
        self.medium.is_empty()
    }

    /// ATTACH an interface's ports: grow the medium by their total width and return a binding placing
    /// each port at a fresh contiguous slice. Dynamic — call any time; the medium simply grows, so a
    /// new interface (a new game, a webcam) plugs into a LIVING morph on the fly.
    pub fn attach(&mut self, ports: &[Port]) -> Binding {
        let mut slots = Vec::with_capacity(ports.len());
        for p in ports {
            let base = self.medium.len();
            self.medium.resize(base + p.dim, 0.0);
            slots.push((base, p.dim, p.flow));
        }
        let b = Binding {
            slots,
            ports: ports.to_vec(),
        };
        self.bindings.push(b.clone());
        b
    }

    /// DETACH: quiet an interface's channels (zero its medium region). The cells stay (the morph's
    /// organs that grew into them simply stop hearing anything) — detaching a webcam shouldn't
    /// renumber a game's channels. `attach` a new interface to reuse fresh cells.
    pub fn detach(&mut self, b: &Binding) {
        for &(off, dim, _) in &b.slots {
            for c in off..(off + dim).min(self.medium.len()) {
                self.medium[c] = 0.0;
            }
        }
        self.bindings.retain(|x| x.slots != b.slots);
    }

    /// Write raw values into a port's slice (used by an interface's `exchange` for `In`/`Both`).
    pub fn write(&mut self, b: &Binding, port: usize, values: &[f32]) {
        let (off, dim) = b.slot(port);
        let n = self.medium.len();
        for (k, &v) in values.iter().take(dim).enumerate() {
            // Out-of-range slot ⇒ no-op; every valid index writes as before.
            if off + k < n {
                self.medium[off + k] = v;
            }
        }
    }
    /// Read raw values from a port's slice (used by an interface's `exchange` for `Out`/`Both`).
    pub fn read(&self, b: &Binding, port: usize) -> &[f32] {
        let (off, dim) = b.slot(port);
        // Out-of-range slot ⇒ empty slice; every valid index reads as before.
        if off + dim <= self.medium.len() {
            &self.medium[off..off + dim]
        } else {
            &[]
        }
    }
    /// Exchange one attached interface (write its `In`/`Both`, read its `Out`/`Both`).
    pub fn exchange<I: Interface + ?Sized>(&mut self, iface: &mut I, at: &Binding) {
        iface.exchange(self, at);
    }

    /// INTROSPECTION — the live bindings, each carrying its `Port` descriptors. A host/UI enumerates
    /// what is plugged in and renders each channel with no per-driver code.
    pub fn bindings(&self) -> &[Binding] {
        &self.bindings
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bare_port_describes_name_flow_dim_shape() {
        let p = Port::input("x", 8).with_shape(4, 2);
        assert_eq!(p.dim, 8);
        assert_eq!(p.flow, Flow::In);
        assert_eq!(p.shape, Some((4, 2)));
        let json = serde_json::to_string(&p).unwrap();
        let back: Port = serde_json::from_str(&json).unwrap();
        assert_eq!(
            (back.name.as_str(), back.dim, back.shape),
            ("x", 8, Some((4, 2)))
        );
    }

    #[test]
    fn port_serialized_with_retired_host_metadata_still_loads() {
        // A Port serialized while it still carried the retired host/UI metadata (`kind`/`unit`/`range`/`rate`/
        // `desc`/`viz`) must still deserialize: serde ignores the unknown keys, the raw structure survives.
        let old = r#"{"name":"mic","flow":"In","dim":1,"shape":null,"kind":"Audio","unit":"norm",
                      "range":[-1.0,1.0],"rate":48000.0,"desc":"mono microphone","viz":"Waveform"}"#;
        let p: Port = serde_json::from_str(old).unwrap();
        assert_eq!(
            (p.name.as_str(), p.flow, p.dim, p.shape),
            ("mic", Flow::In, 1, None)
        );
    }

    #[test]
    fn membrane_bindings_enumerate_live_ports_and_values() {
        let mut m = Membrane::new();
        let b = m.attach(&[
            Port::input("cam", 4).with_shape(2, 2),
            Port::output("act", 2),
        ]);
        // the live binding carries its port descriptors, shape preserved.
        assert_eq!(m.bindings().len(), 1);
        let ports = &m.bindings()[0].ports;
        assert_eq!(ports.len(), 2);
        assert_eq!(ports[0].shape, Some((2, 2)));
        assert_eq!(ports[1].flow, Flow::Out);
        // live values are readable per port.
        m.write(&b, 0, &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(m.read(&b, 0), &[1.0, 2.0, 3.0, 4.0]);
    }
}
