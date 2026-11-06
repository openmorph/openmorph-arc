// SPDX-License-Identifier: MIT-0
//! rc6 step 1 — the MEMBRANE is universal: heterogeneous, self-describing interfaces plug into ONE
//! growable medium on the fly, and the membrane moves raw values with NO per-modality code. A grid
//! sensor (In), buttons (Out), and a mouse (Both) share the same medium; attach/detach is dynamic.
//!
//!   cargo test -p openmorph-core --test proofs membrane_plug -- --nocapture --test-threads=1

use openmorph_core::membrane::{Binding, Flow, Interface, Membrane, Port};

/// A camera-like sensor: the world WRITES a frame into the medium each exchange.
struct GridSensor {
    w: usize,
    h: usize,
    frame: Vec<f32>,
}
impl Interface for GridSensor {
    fn ports(&self) -> Vec<Port> {
        vec![Port::input("frame", self.w * self.h).with_shape(self.w, self.h)]
    }
    fn exchange(&mut self, m: &mut Membrane, at: &Binding) {
        m.write(at, 0, &self.frame); // world → medium (the morph will sense it)
    }
}

/// Buttons: the world READS the morph's action cells each exchange (motor).
struct Buttons {
    pressed: Vec<f32>,
}
impl Interface for Buttons {
    fn ports(&self) -> Vec<Port> {
        vec![Port::output("buttons", 3)]
    }
    fn exchange(&mut self, m: &mut Membrane, at: &Binding) {
        self.pressed = m.read(at, 0).to_vec(); // medium → world (whatever the morph wrote)
    }
}

/// A mouse: BOTH — the world writes the current position, and reads the morph's requested move.
struct Mouse {
    pos: [f32; 2],
    requested: [f32; 2],
}
impl Interface for Mouse {
    fn ports(&self) -> Vec<Port> {
        vec![Port::both("mouse", 2)]
    }
    fn exchange(&mut self, m: &mut Membrane, at: &Binding) {
        self.requested.copy_from_slice(&m.read(at, 0)[..2]); // read the morph's request
        m.write(at, 0, &self.pos); // then write the true position (shared channel)
    }
}

#[test]
fn heterogeneous_interfaces_share_one_growable_medium() {
    let mut m = Membrane::new();
    assert!(m.is_empty());

    // Plug in a 3×2 camera and a 3-button pad — the medium grows to hold both.
    let mut cam = GridSensor {
        w: 3,
        h: 2,
        frame: vec![0.0; 6],
    };
    let mut pad = Buttons { pressed: vec![] };
    let b_cam = m.attach(&cam.ports());
    let b_pad = m.attach(&pad.ports());
    assert_eq!(m.len(), 6 + 3, "medium = frame(6) + buttons(3)");

    // The world writes a frame; the membrane just moves raw floats (no modality knowledge).
    cam.frame = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    m.exchange(&mut cam, &b_cam);
    assert_eq!(m.read(&b_cam, 0), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    // The morph "acts": it writes the button cells directly (same medium, no special path).
    let (off, dim) = b_pad.slot(0);
    for k in 0..dim {
        m.medium[off + k] = if k == 1 { 0.9 } else { 0.0 }; // press button 1
    }
    m.exchange(&mut pad, &b_pad);
    assert_eq!(
        pad.pressed,
        vec![0.0, 0.9, 0.0],
        "the pad read the morph's action off the medium"
    );

    // Dynamic: unplug the camera, plug in a mouse (Both) — on the fly, medium grows again.
    m.detach(&b_cam);
    assert_eq!(m.read(&b_cam, 0), &[0.0; 6], "detached channel goes quiet");
    let mut mouse = Mouse {
        pos: [7.0, 8.0],
        requested: [0.0, 0.0],
    };
    let b_mouse = m.attach(&mouse.ports());
    // the morph requests a move by writing the mouse cells…
    let (mo, _) = b_mouse.slot(0);
    m.medium[mo] = 0.5;
    m.medium[mo + 1] = -0.5;
    m.exchange(&mut mouse, &b_mouse);
    assert_eq!(
        mouse.requested,
        [0.5, -0.5],
        "mouse read the morph's requested move (Both: read side)"
    );
    assert_eq!(
        m.read(&b_mouse, 0),
        &[7.0, 8.0],
        "mouse then wrote the true position (Both: write side)"
    );

    // Verify the ports self-describe (the schema the morph grows against — structure, not meaning).
    assert_eq!(cam.ports()[0].flow, Flow::In);
    assert_eq!(cam.ports()[0].shape, Some((3, 2)));
    assert_eq!(pad.ports()[0].flow, Flow::Out);
    assert_eq!(mouse.ports()[0].flow, Flow::Both);

    eprintln!("[MEMBRANE] one growable medium; camera(In 3×2) + buttons(Out 3) + mouse(Both 2) plugged in and swapped on the fly — the membrane moved raw values with NO per-modality code. Meaning is grown by the organism, not authored here. rc6 step 1 ✓.");
}
