// SPDX-License-Identifier: MIT-0
//! rc6 step 3 — the async field LIVES against the membrane, and two consequences of "any cells can be
//! an interface":
//!   (1) TAP A CONCEPT — a pattern the morph grew becomes a live port: LISTEN to it (read its
//!       representation) or TRIGGER it (write it to fire downstream), no inputs needed.
//!   (2) QUERYABLE 2-D — an image is not dumped into cells; it is a SPACE the morph travels: it moves
//!       a fovea and asks for the value there, touching ONLY what it looks at (active, compute-saving
//!       sensing — the reference-frame view of 2-D).
//!
//!   cargo test -p openmorph-core --test proofs membrane_field -- --nocapture --test-threads=1

use openmorph_core::membrane::{Binding, Flow, Interface, Membrane, Port};
use openmorph_core::substrate::Field;

#[test]
fn tap_a_grown_concept_to_listen_and_trigger() {
    // A tiny organism: inputs c0,c1 → a CONCEPT cell c2 ("both present") → a downstream effect c3.
    let mut f = Field::new(4);
    f.add_unit(vec![0, 1], vec![2.0, 2.0], -3.0, 2); // c2 = tanh(2·c0 + 2·c1 − 3): high iff both
    f.add_unit(vec![2], vec![3.0], 0.0, 3); // c3 = tanh(3·c2): the downstream effect

    // Drive it through the inputs: both present → the concept lights, downstream fires.
    f.stimulate(&[(0, 1.0), (1, 1.0)], 100);
    assert!(
        f.cells()[2] > 0.5,
        "concept lit when both inputs present ({:?})",
        f.cells()
    );
    assert!(f.cells()[3] > 0.5, "downstream fired");

    // LISTEN: tap the concept cell as an Out port — read the morph's OWN representation (introspection).
    let concept_out = f.tap(2, 1, Flow::Out);
    let heard = f.sample(&concept_out, 0)[0];
    assert!(
        heard > 0.5,
        "listening to the concept reads its representation ({heard:.2})"
    );

    // Turn the inputs OFF: the concept and downstream go quiet.
    f.stimulate(&[(0, 0.0), (1, 0.0)], 100);
    assert!(
        f.cells()[3] < 0.0,
        "downstream off once the concept is unlit ({:?})",
        f.cells()
    );

    // TRIGGER: tap the same concept cell as an In port and WRITE it — the concept fires downstream
    // with NO inputs at all (evoking a concept from outside, like stimulating a representation).
    let concept_in = f.tap(2, 1, Flow::In);
    f.feed(&concept_in, 0, &[1.0], 100);
    assert!(
        f.cells()[3] > 0.5,
        "TRIGGERING the concept fired downstream without any input ({:?})",
        f.cells()
    );

    eprintln!("[FIELD] a grown concept cell is a live interface: LISTEN (read its representation = {heard:.2}) and TRIGGER (inject it → downstream fires with no inputs). Any inner representation is addressable.");
}

/// An image as a QUERYABLE 2-D SPACE: the morph writes a fovea position (Both) and the interface
/// returns the value there (In). It reads ONLY the queried pixel — `touched` counts the pixels it
/// actually looked at (the compute the morph spent), never the whole image.
struct QueryableImage {
    w: usize,
    h: usize,
    pixels: Vec<f32>,
    touched: usize,
}
impl Interface for QueryableImage {
    fn ports(&self) -> Vec<Port> {
        vec![
            Port::both("fovea", 2).with_shape(self.w, self.h),
            Port::input("value", 1),
        ]
    }
    fn exchange(&mut self, m: &mut Membrane, at: &Binding) {
        let fovea = m.read(at, 0).to_vec(); // the morph's requested position (Both: read side)
        let x = fovea[0].clamp(0.0, (self.w - 1) as f32).round() as usize;
        let y = fovea[1].clamp(0.0, (self.h - 1) as f32).round() as usize;
        let v = self.pixels[y * self.w + x];
        self.touched += 1; // ONE pixel of compute, wherever the morph looked
        m.write(at, 1, &[v]); // the value at the fovea (In)
    }
}

#[test]
fn a_2d_image_is_a_space_the_morph_travels_touching_only_what_it_looks_at() {
    let (w, h) = (20usize, 20usize);
    // An image that's 0 everywhere except a bright mark at (13, 7).
    let mut pixels = vec![0.0f32; w * h];
    pixels[7 * w + 13] = 1.0;
    let mut img = QueryableImage {
        w,
        h,
        pixels,
        touched: 0,
    };

    let mut f = Field::new(0);
    let b = f.attach(&img.ports()); // fovea (Both, 2) + value (In, 1)

    // The morph TRAVELS: query a handful of positions and read the value at each.
    let visits = [(0usize, 0usize), (13, 7), (5, 5), (19, 19), (13, 7)];
    let mut got = Vec::new();
    for &(x, y) in &visits {
        f.feed(&b, 0, &[x as f32, y as f32], 100); // move the fovea (write the Both channel)
        f.exchange(&mut img, &b); // the space returns the value under the fovea
        got.push(f.sample(&b, 1)[0]); // read the value channel
    }

    assert_eq!(
        got[1], 1.0,
        "the bright mark at (13,7) was found by looking there"
    );
    assert_eq!(got[0], 0.0, "corner (0,0) is dark");
    assert_eq!(
        img.touched,
        visits.len(),
        "it touched ONLY the {} pixels it looked at, not all {}",
        visits.len(),
        w * h
    );
    eprintln!("[FIELD] 2-D is a space the morph travels: it moved a fovea over {} positions and queried the value at each — touching {} of {} pixels ({:.0}% of the image ignored). Active sensing = focus on what matters + compute saved on what it knows is useless.", visits.len(), img.touched, w * h, 100.0 * (1.0 - img.touched as f32 / (w * h) as f32));
    eprintln!("[FIELD] VERDICT (rc6 step 3): the field owns the membrane; any cells (even grown concepts) are a pluggable interface to listen to or trigger; and 2-D becomes a navigable coordinate space queried on demand — the reference-frame view, which saves compute and lets the organism attend to what matters.");
}
