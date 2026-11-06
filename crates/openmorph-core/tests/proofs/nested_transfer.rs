// SPDX-License-Identifier: MIT-0
//! HIERARCHY CONVERTS INTERFERENCE → TRANSFER (Sébastien's hypothesis; MorphBot-35 ask). `grow` only ever
//! births SCALAR features (one widening layer); banking a relied-upon subgraph as a frozen NESTED node
//! (`bank_skill` = encapsulate+nest) gives the organism a SECOND STOREY. On a composed task where B REQUIRES
//! A's sub-skill, a FLAT organism carries A-tuned features that INTERFERE (pretrain HURTS), while a BANKED
//! (nested) skill TRANSFERS. Controls isolate the effect to NESTING (not decomposition, not freezing).
//!
//! HONEST SCOPE (the GoalBand lesson): y_B contains y_A as an EXACT factor, so the skill boundary is noiseless
//! and GIVEN. This validates the MECHANISM; it does NOT show real tasks decompose this cleanly, that the
//! boundary can be found without knowing it, or that autonomous in-context reuse works (that reaches only ~0.6
//! here — the open boundary/context problem).
//!
//! Task A:  y_A = x0·x1                 (a product sub-skill)
//! Task B:  y_B = (x0·x1)·x2            (COMPOSED — REQUIRES the task-A skill as a sub-component)
//!
//! RE-BASELINED at the 1.6 learning flip — `#[ignore]`d: the result was SILENT SETTLE TRUNCATION, in either genome.
//! Every arm grew with a settle budget of 100 while `grow` births up to `max_hidden` (64) features PER output cell and
//! a settle costs hidden + 2 firings, so the flat arms (two grown readouts in one field) truncated hundreds of settles
//! (`Field::truncated_settles`). With a budget that settles (400, below) the effect is gone. Task-B accuracy, scratch |
//! pretrained | decomp | consolidated | BANKED | AUTONOMOUS:
//!   1.6 default, budget 100: 1.000 | 0.400 | 0.425 | 0.562 | 1.000 | 0.475  (the old "interference" / "nesting" story)
//!   1.6 default, budget 400: 1.000 | 1.000 | 0.750 | 1.000 | 1.000 | 1.000  (no interference; nesting no better)
//!   legacy,      budget 100: 0.800 | 0.587 | 0.600 | 0.600 | 0.950 | 0.613
//!   legacy,      budget 400: 0.800 | 0.725 | 1.000 | 1.000 | 0.950 | 1.000  (decomp and freezing BEAT nesting)
//! So neither the interference (A) nor the nesting-specific transfer (B, C) holds once the flat arms actually settle,
//! and the "autonomous in-context reuse reaches only ~0.6" open end was truncation too (1.000 at budget 400).

use openmorph_core::substrate::Field;

const NIN: usize = 4; // x0..x3 ∈ {−1,+1}
const SKILL: usize = NIN; // task-A output cell
const BUD: usize = 400; // ≥ the flat arms' settle cost (was 100, which truncated — see the module docs)

fn states() -> Vec<[f32; NIN]> {
    let mut v = Vec::new();
    for m in 0..(1 << NIN) {
        let mut x = [0.0f32; NIN];
        for (i, xi) in x.iter_mut().enumerate() {
            *xi = if (m >> i) & 1 == 1 { 1.0 } else { -1.0 };
        }
        v.push(x);
    }
    v
}
fn y_a(x: &[f32; NIN]) -> f32 {
    (2.0 * x[0] * x[1]).tanh()
}
fn y_b(x: &[f32; NIN]) -> f32 {
    (2.0 * x[0] * x[1] * x[2]).tanh()
}

/// Train `f` toward `target(x)` at cell `out` over the given input cells, `epochs` passes.
fn train(
    f: &mut Field,
    ins: &[usize],
    out: usize,
    target: impl Fn(&[f32; NIN]) -> f32,
    epochs: usize,
) {
    let st = states();
    for _ in 0..epochs {
        for x in &st {
            let inputs: Vec<(usize, f32)> = ins.iter().map(|&c| (c, x[c])).collect();
            f.grow(&inputs, out, target(x), 0.1, BUD);
        }
    }
}

/// Held-out sign accuracy of `f`'s prediction at `out` for `target`, reading `ins`.
fn accuracy(f: &mut Field, ins: &[usize], out: usize, target: impl Fn(&[f32; NIN]) -> f32) -> f32 {
    let st = states();
    let mut ok = 0;
    for x in &st {
        let inputs: Vec<(usize, f32)> = ins.iter().map(|&c| (c, x[c])).collect();
        f.stimulate(&inputs, BUD);
        let pred = f.cells()[out];
        if pred.signum() == target(x).signum() {
            ok += 1;
        }
    }
    ok as f32 / st.len() as f32
}

#[test]
#[ignore = "1.6 re-baseline: does not hold — the effect was silent settle truncation (budget 100) in either genome; at a settling budget the flat arms reach task B; see this file's module docs"]
fn hierarchy_converts_interference_into_transfer_and_it_is_nesting() {
    let mut fl_scratch = 0.0;
    let mut fl_pre = 0.0;
    let mut banked = 0.0;
    let mut decomp = 0.0;
    let mut consol = 0.0;
    let mut auto = 0.0;
    let seeds = [1u64, 7, 42, 99, 5];
    for &seed in &seeds {
        // (1) FLAT SCRATCH: task B from zero over raw inputs.
        let mut fs = Field::new(NIN + 2);
        fs.rng_seed(seed);
        train(&mut fs, &[0, 1, 2, 3], NIN + 1, y_b, 60);
        fl_scratch += accuracy(&mut fs, &[0, 1, 2, 3], NIN + 1, y_b);

        // (2) FLAT PRETRAINED: learn task A, then CONTINUE the same field on task B (features re-tuned → interference).
        let mut fp = Field::new(NIN + 2);
        fp.rng_seed(seed);
        train(&mut fp, &[0, 1, 2, 3], SKILL, y_a, 60);
        train(&mut fp, &[0, 1, 2, 3], NIN + 1, y_b, 60);
        fl_pre += accuracy(&mut fp, &[0, 1, 2, 3], NIN + 1, y_b);

        // (3) BANKED: learn task A, ENCAPSULATE the skill as a frozen nested cell, then learn B over
        // [skill_cell, x2, x3] — B has NO direct access to x0/x1, so it MUST reuse the banked skill.
        let mut fa = Field::new(NIN + 2);
        fa.rng_seed(seed);
        train(&mut fa, &[0, 1], SKILL, y_a, 60); // the skill depends only on x0,x1 → backbone reads [0,1]
        let sub = fa.encapsulate(&[SKILL]); // bank the task-A skill (its whole feeding backbone)
        let mut fb = Field::new(NIN + 2);
        fb.rng_seed(seed);
        let skill_cell = SKILL; // the nested output lands in the WRITE cell (nest's return is NOT the cell)
        fb.nest(sub, &[0, 1], skill_cell, BUD); // frozen skill node, reads x0,x1 → writes SKILL
        let out = fb.len(); // fresh output cell for B
        fb.ensure_cells(out + 1);
        let st = states();
        for _ in 0..60 {
            for x in &st {
                // stimulate the RAW inputs so the nested skill_cell settles, read it, then grow B over
                // [skill, x2, x3] toward y_B (a product of the banked skill and x2 — a 2-way product of clean
                // inputs, vs the 3-way product a flat organism faces from raw).
                fb.stimulate(&[(0, x[0]), (1, x[1]), (2, x[2]), (3, x[3])], BUD);
                let s = fb.cells()[skill_cell];
                fb.grow(
                    &[(skill_cell, s), (2, x[2]), (3, x[3])],
                    out,
                    y_b(x),
                    0.1,
                    BUD,
                );
            }
        }
        let mut ok = 0;
        for x in &st {
            // ONE raw stimulate: skill_cell settles (nested), B's readout reads it and fires.
            fb.stimulate(&[(0, x[0]), (1, x[1]), (2, x[2]), (3, x[3])], BUD);
            if fb.cells()[out].signum() == y_b(x).signum() {
                ok += 1;
            }
        }
        banked += ok as f32 / st.len() as f32;

        // (4) CONTROL — FLAT-DECOMP (no encapsulate/nest): learn A at the SKILL cell, then in the SAME field grow
        // B at a FRESH cell reading [SKILL, x2, x3]. This reuses the skill cell as an input without banking it —
        // isolating whether the benefit is NESTING or merely "a fresh readout over the skill cell".
        let mut fd = Field::new(NIN + 2);
        fd.rng_seed(seed);
        train(&mut fd, &[0, 1], SKILL, y_a, 60);
        let dout = fd.len();
        fd.ensure_cells(dout + 1);
        for _ in 0..60 {
            for x in &st {
                fd.stimulate(&[(0, x[0]), (1, x[1]), (2, x[2]), (3, x[3])], BUD);
                let s = fd.cells()[SKILL];
                fd.grow(&[(SKILL, s), (2, x[2]), (3, x[3])], dout, y_b(x), 0.1, BUD);
            }
        }
        let mut okd = 0;
        for x in &st {
            fd.stimulate(&[(0, x[0]), (1, x[1]), (2, x[2]), (3, x[3])], BUD);
            if fd.cells()[dout].signum() == y_b(x).signum() {
                okd += 1;
            }
        }
        decomp += okd as f32 / st.len() as f32;

        // (5) CONTROL — FLAT-CONSOLIDATED (freeze, no nest): learn A at SKILL, FREEZE all of A's units
        // (plasticity 0), then grow B at a fresh cell over [SKILL, x2, x3]. Isolates FREEZING from NESTING: if
        // this also transfers, autonomous banking reduces to autonomous consolidation.
        let mut fc = Field::new(NIN + 2);
        fc.rng_seed(seed);
        train(&mut fc, &[0, 1], SKILL, y_a, 60);
        let a_cells: Vec<usize> = fc.units.iter().map(|u| u.write).collect();
        fc.set_plasticity(&a_cells, 0.0); // freeze the whole task-A skill backbone
        let cout = fc.len();
        fc.ensure_cells(cout + 1);
        for _ in 0..60 {
            for x in &st {
                fc.stimulate(&[(0, x[0]), (1, x[1]), (2, x[2]), (3, x[3])], BUD);
                let s = fc.cells()[SKILL];
                fc.grow(&[(SKILL, s), (2, x[2]), (3, x[3])], cout, y_b(x), 0.1, BUD);
            }
        }
        let mut okc = 0;
        for x in &st {
            fc.stimulate(&[(0, x[0]), (1, x[1]), (2, x[2]), (3, x[3])], BUD);
            if fc.cells()[cout].signum() == y_b(x).signum() {
                okc += 1;
            }
        }
        consol += okc as f32 / st.len() as f32;

        // (6) AUTONOMOUS — the organism banks its OWN skill via the `bank_skill` verb (no external encapsulate/
        // nest surgery): learn A, then bank the relied-upon skill output; grow B over the banked node.
        let mut fz = Field::new(NIN + 2);
        fz.rng_seed(seed);
        train(&mut fz, &[0, 1], SKILL, y_a, 60);
        // the organism picks the skill output by RELIANCE (highest-relied grown cell), then banks it. Here the
        // trained output SKILL is the relied-upon cell; bank_skill derives the boundary via encapsulate.
        let banked_cell = fz.bank_skill(SKILL, BUD); // AUTONOMOUS NESTED GROWTH — a frozen reusable node
                                                     // CORRECTNESS of the verb: the banked node reproduces the skill it encapsulated (x0·x1).
        for x in &st {
            fz.stimulate(&[(0, x[0]), (1, x[1])], BUD);
            assert!(
                fz.cells()[banked_cell].signum() == y_a(x).signum(),
                "bank_skill must reproduce the skill it banked (x0·x1)"
            );
        }
        let zout = fz.len();
        fz.ensure_cells(zout + 1);
        for _ in 0..60 {
            for x in &st {
                fz.stimulate(&[(0, x[0]), (1, x[1]), (2, x[2]), (3, x[3])], BUD);
                let s = fz.cells()[banked_cell];
                fz.grow(
                    &[(banked_cell, s), (2, x[2]), (3, x[3])],
                    zout,
                    y_b(x),
                    0.1,
                    BUD,
                );
            }
        }
        let mut okz = 0;
        for x in &st {
            fz.stimulate(&[(0, x[0]), (1, x[1]), (2, x[2]), (3, x[3])], BUD);
            if fz.cells()[zout].signum() == y_b(x).signum() {
                okz += 1;
            }
        }
        auto += okz as f32 / st.len() as f32;
    }
    let n = seeds.len() as f32;
    let (fs_a, fp_a, dc_a, cs_a, bk_a, au_a) = (
        fl_scratch / n,
        fl_pre / n,
        decomp / n,
        consol / n,
        banked / n,
        auto / n,
    );
    eprintln!(
        "[NEST-TRANSFER] task B acc: flat-scratch {fs_a:.3} | flat-pretrained {fp_a:.3} (interference) | flat-decomp {dc_a:.3} | flat-consolidated {cs_a:.3} | BANKED(manual nest) {bk_a:.3} | AUTONOMOUS bank_skill {au_a:.3}"
    );

    // (A) INTERFERENCE reproduced: pretraining on A HURTS B (below scratch) — the diversity-transfer failure in miniature.
    assert!(
        fp_a < fs_a,
        "flat-pretrained ({fp_a:.3}) must be BELOW flat-scratch ({fs_a:.3}) — interference"
    );
    // (B) TRANSFER: banking the skill (nested) flips the sign — far above both interference and scratch.
    assert!(
        bk_a > fp_a + 0.2 && bk_a > fs_a + 0.1,
        "BANKED ({bk_a:.3}) must convert interference into transfer — well above flat-pretrained ({fp_a:.3}) and scratch ({fs_a:.3})"
    );
    // (C) It is NESTING specifically — not decomposition, not freezing (both controls stay near interference).
    assert!(
        bk_a > dc_a + 0.2 && bk_a > cs_a + 0.2,
        "BANKED ({bk_a:.3}) must beat flat-decomp ({dc_a:.3}) AND flat-consolidated ({cs_a:.3}) by a margin — the effect is NESTING, not decomposition or freezing"
    );
    // (D) AUTONOMOUS `bank_skill` correctly BANKS the skill (asserted per-state above). Its transfer WHEN REUSED
    // IN-CONTEXT (amid the full task-A field, ~70 grown cells) is an HONEST OPEN: it reaches only `au_a`, well
    // below the clean two-field banked `bk_a` — the surrounding structure/settle interferes with reuse. So the
    // banking PRIMITIVE is correct; extracting the transfer benefit autonomously (a clean reuse context / the
    // boundary problem) is NOT solved here. Reported, not asserted as transfer.
    eprintln!(
        "[NEST-TRANSFER] VERDICT: hierarchy converts interference→transfer, and it is NESTING specifically \
         (decomp {dc_a:.3} / consolidate {cs_a:.3} alone stay at interference; BANKED-nested {bk_a:.3}). \
         `bank_skill` correctly banks the skill; autonomous IN-CONTEXT reuse reaches only {au_a:.3} (< clean \
         {bk_a:.3}) — the boundary/reuse-context problem is OPEN. HONEST SCOPE: this is the friendliest world — \
         y_B contains y_A as an EXACT factor, so the skill boundary is noiseless and GIVEN. It validates the \
         MECHANISM; it does NOT establish that real tasks decompose this cleanly, that the boundary can be found \
         without knowing it, or that approximate reuse survives (open — needs real-corpus measurement)."
    );
}
