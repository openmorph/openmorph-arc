// SPDX-License-Identifier: MIT-0
//! v2.0-dev STRESS TEST — SUB-MORPH LIBRARY / ROUTING scalability.
//!
//! AXIS: build M distinct nested skills (each a sub-morph) and route a query to the ONE relevant skill.
//! We sweep M and answer three questions HONESTLY, reporting only observed numbers:
//!   (a) SETTLE COST — with no router (all skills on a shared input bus / broadcast), does the whole
//!       library settle (O(M) sub-settles) or only the active skill? Measured via `nested_settles()`.
//!   (b) ROUTING — if we key skills by a position and route with the fractal map-of-maps index
//!       (`substrate_fractal`, extended far past M=256), do routing comparisons scale ~O(√M), and does
//!       the routed path settle only 1 skill? Also: does routing ACCURACY hold at large M?
//!   (c) MEMORY — how do #units, #cells, and process RSS grow with M, and where is it impractical?
//!
//! The thesis it tests: the substrate gives you RECURSION (`nest`) AND — now — content-addressed routing,
//! the CORE ROUTER (#64: `cluster`/`index_skills`/`route`, the fractal map-of-maps over the grown
//! `pos`/`distance`). The default bus/broadcast is strictly O(M); the router settles only the ONE active
//! skill in ~O(√M) comparisons. This stresses that win far past M=256.
//!
//!   cargo test -p openmorph-core --test stress v2_scale_library -- --ignored --nocapture --test-threads=1

use openmorph_core::substrate::Field;
use std::time::Instant;

#[allow(unused_imports)]
use crate::common::Rng;

// ── best-effort peak RSS via getrusage (macOS: ru_maxrss is BYTES; Linux: KiB). Best-effort only. ──
#[repr(C)]
struct Rusage {
    words: [i64; 36],
}
extern "C" {
    fn getrusage(who: i32, usage: *mut Rusage) -> i32;
}
fn peak_rss_mb() -> f64 {
    let mut u = Rusage { words: [0; 36] };
    let r = unsafe {
        getrusage(0 /*RUSAGE_SELF*/, &mut u)
    };
    if r != 0 {
        return -1.0;
    }
    // layout: ru_utime(timeval,16B)=words[0..2], ru_stime(16B)=words[2..4], ru_maxrss=words[4].
    let maxrss = u.words[4];
    if cfg!(target_os = "macos") {
        maxrss as f64 / (1024.0 * 1024.0) // bytes → MiB
    } else {
        maxrss as f64 / 1024.0 // KiB → MiB
    }
}

fn dist(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

/// A distinct leaf skill k: cell 0 = input, cell 1 = tanh(w_k · x). (Function differs per k; only its
/// POSITION on the concept-map matters for routing — the point is settle/route/memory cost, not accuracy
/// of the function.)
fn skill(k: usize) -> Field {
    let mut s = Field::new(2);
    let w = 0.5 + (k % 13) as f32 * 0.1;
    s.add_unit(vec![0], vec![w], 0.0, 1);
    s
}

#[test]
#[ignore = "stress: nightly"]
fn sub_morph_library_routing_scaling() {
    // scales: perfect squares so keys tile a √M×√M grid concept-map. We push PAST M=4096 to locate the wall.
    let scales: [usize; 8] = [16, 64, 256, 1024, 4096, 16384, 65536, 262144];
    let time_budget_s = 150.0; // hard wall-time guard: stop the sweep before we blow the 4-min cap.
    let sweep_start = Instant::now();

    eprintln!("[LIB] ============================================================================");
    eprintln!("[LIB] SUB-MORPH LIBRARY / ROUTING — sweep M, each row is OBSERVED (getrusage RSS best-effort)");
    eprintln!("[LIB] ============================================================================");

    let mut last_broadcast_us_per_skill = 0.0f64; // for the extrapolated impractical-M estimate
                                                  // (M, build_ms, bc_settles, bc_ms, us_per_skill, flat_cmp, fractal_cmp, route_acc, units, cells, rss_mib)
    type Row = (usize, f64, u64, f64, f64, f64, f64, f64, usize, usize, f64);
    let mut rows: Vec<Row> = Vec::new();
    let mut reached = 0usize;

    for &m in &scales {
        if sweep_start.elapsed().as_secs_f64() > time_budget_s {
            eprintln!("[LIB] STOP: wall-time budget hit before M={m} — that scale is impractical in this test's time bound.");
            break;
        }
        let side = (m as f64).sqrt() as usize;
        debug_assert_eq!(side * side, m);

        // ── build the routed host: skill k reads its OWN input cell 2k, writes 2k+1 (no shared bus). ──
        let t_build = Instant::now();
        let mut host = Field::new(0);
        for k in 0..m {
            host.nest(skill(k), &[2 * k], 2 * k + 1, 4);
        }
        // key positions on a √M×√M grid, stamped into a key-Field; the CORE router indexes them (#64).
        let mut keys = Field::new(m);
        for i in 0..m {
            keys.place_at(i, &[(i % side) as f32, (i / side) as f32]);
        }
        let index = keys.index_skills(&(0..m).collect::<Vec<_>>(), 4, 4);
        let build_ms = t_build.elapsed().as_secs_f64() * 1e3;
        let rss_mb = peak_rss_mb();
        let (units, cells) = (host.units.len(), host.len());

        // ── (a) NO ROUTER / BROADCAST: feed ALL M inputs (changing values) → every skill fires once. ──
        // This is the cost when you do NOT know which skill is relevant and settle the library.
        let ns0 = host.nested_settles();
        let broadcast_inputs: Vec<(usize, f32)> = (0..m)
            .map(|k| (2 * k, 0.11 + (k as f32) * 1e-4)) // distinct, non-zero, fresh (memo miss)
            .collect();
        let t_bc = Instant::now();
        host.stimulate(&broadcast_inputs, m + 8);
        let bc_ms = t_bc.elapsed().as_secs_f64() * 1e3;
        let bc_settles = host.nested_settles() - ns0; // EXPECT == M (the whole library settled)
        last_broadcast_us_per_skill = (bc_ms * 1e3) / (m as f64);

        // ── (b) FRACTAL-INDEX ROUTING + routed settle: route to ONE skill, settle only it. ──
        let n_route = if m <= 4096 {
            300
        } else if m <= 65536 {
            120
        } else {
            40
        };
        let (mut flat_cmp, mut frac_cmp, mut ok) = (0.0f64, 0.0f64, 0.0f64);
        let mut routed_settles = 0u64;
        let mut seed = 0x9E3779B97F4A7C15u64;
        let t_rt = Instant::now();
        for q in 0..n_route {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let target = (seed >> 33) as usize % m;
            let jit = (((seed >> 20) & 0xff) as f32 / 255.0 - 0.5) * 0.3; // ±0.15 within the unit grid cell
            let qy = vec![(target % side) as f32 + jit, (target / side) as f32 - jit];
            // FLAT baseline: scan all M keys (a bare `distance()` sweep — O(M) per lookup).
            let flat = (0..m)
                .min_by(|&a, &b| {
                    dist(&qy, &keys.position(a)).total_cmp(&dist(&qy, &keys.position(b)))
                })
                .unwrap();
            flat_cmp += m as f64;
            // CORE router: coarse-to-fine descent of the fractal index (#64), ~O(√M).
            let (frac, c) = keys.route(&index, &qy, 1, 0);
            frac_cmp += c as f64;
            if frac == flat {
                ok += 1.0;
            }
            // settle ONLY the routed skill: feed its dedicated input with a fresh value → 1 sub-settle.
            // step 5e-3 ≫ settle_eps(1e-4): a re-route to the SAME skill still changes its input past the
            // fire threshold, and the value is always fresh (memo miss), so exactly ONE sub-settle per query.
            let nprev = host.nested_settles();
            host.stimulate(&[(2 * frac, 0.37 + (q as f32) * 5e-3)], 8);
            routed_settles += host.nested_settles() - nprev;
        }
        let rt_ms = t_rt.elapsed().as_secs_f64() * 1e3;
        let n = n_route as f64;
        let (fl, fr) = (flat_cmp / n, frac_cmp / n);
        let acc = 100.0 * ok / n;
        let routed_per_q = routed_settles as f64 / n;
        let sqrt_ratio = fr / (m as f64).sqrt();

        eprintln!("[LIB] ------------------------------------------------------------");
        eprintln!("[LIB] M={m:<6} side={side:<3}  units={units:<6} cells={cells:<6}  build={build_ms:>8.1}ms  RSS={rss_mb:>7.1}MiB");
        eprintln!(
            "[LIB]   (a) SETTLE  broadcast(no router): {bc_settles:>6} sub-settles/query (== M? {}) in {bc_ms:>8.2}ms  → {:.3} µs/skill",
            bc_settles == m as u64,
            last_broadcast_us_per_skill
        );
        eprintln!(
            "[LIB]   (b) ROUTE   flat={fl:>7.0} cmp | fractal={fr:>6.1} cmp  ({:.1}× fewer, {:.2}×√M)  acc={acc:>5.1}%  → routed settles {routed_per_q:.3}/query",
            fl / fr,
            sqrt_ratio
        );
        eprintln!(
            "[LIB]   (c) MEMORY  {units} host units + {units} nested sub-Fields, {cells} host cells  (both O(M))  | route phase {rt_ms:.1}ms/{n_route}q"
        );

        rows.push((
            m, build_ms, bc_settles, bc_ms, fl, fr, sqrt_ratio, acc, units, cells, rss_mb,
        ));
        reached = m;

        // per-scale correctness of the MEASUREMENT (not hiding the limit): broadcast really settled all M,
        // and the routed path really settled ~1 skill.
        assert_eq!(
            bc_settles, m as u64,
            "M={m}: broadcasting the library must settle ALL M sub-morphs (O(M)) — got {bc_settles}"
        );
        assert!(
            (0.5..=1.05).contains(&routed_per_q),
            "M={m}: the routed path must settle ~1 skill (O(1), not O(M)), got {routed_per_q:.3}"
        );
    }

    // ── SUMMARY + the honest limit ──────────────────────────────────────────────────────────────
    eprintln!("[LIB] ============================================================================");
    eprintln!("[LIB] SUMMARY (observed):");
    eprintln!("[LIB]  M      build_ms   bc_settles   bc_ms   flat_cmp  frac_cmp  frac/√M  acc%   units   cells    RSS_MiB");
    for (m, b, bs, bcm, fl, fr, sr, acc, u, c, rss) in &rows {
        eprintln!(
            "[LIB]  {m:<6} {b:>8.1}   {bs:>9}   {bcm:>6.1}  {fl:>8.0}  {fr:>7.1}   {sr:>5.2}   {acc:>4.0}   {u:>6}  {c:>6}   {rss:>7.1}"
        );
    }

    // Verify the two structural growth laws across the sweep from the observed rows.
    // (b) fractal comparisons stay a small constant × √M (sub-linear) while flat is exactly M.
    let worst_sqrt_ratio = rows.iter().map(|r| r.6).fold(0.0f32 as f64, f64::max);
    let worst_acc = rows.iter().map(|r| r.7).fold(100.0f64, f64::min);
    // (a)/(c) the linear laws: check bc_settles == M held every row and units/cells scale O(M).
    let settle_is_linear = rows.iter().all(|r| r.2 == r.0 as u64);
    let mem_is_linear = rows.iter().all(|r| r.8 == r.0 && r.9 == 2 * r.0);

    // Extrapolate the broadcast (no-router) impractical point from the last observed µs/skill.
    let impractical_m_100ms = if last_broadcast_us_per_skill > 0.0 {
        (100_000.0 / last_broadcast_us_per_skill) as u64 // M at which one bus query ≈ 100 ms
    } else {
        0
    };

    eprintln!("[LIB] ----------------------------------------------------------------------------");
    eprintln!("[LIB] VERDICT (truth-not-trophy):");
    eprintln!("[LIB]  • (a) WITHOUT ROUTING, O(M). `nest` gives recursion; if you do NOT route (shared input / a");
    eprintln!("[LIB]        broadcast), settle cost is STRICTLY O(M): bc_settles == M at every");
    eprintln!("[LIB]        scale (reached M={reached}). One un-routed query at M={reached} did {} sub-settles.",
        rows.last().map(|r| r.2).unwrap_or(0));
    eprintln!("[LIB]        Extrapolating {last_broadcast_us_per_skill:.3} µs/skill, a bus query hits ~100ms at M≈{impractical_m_100ms}.");
    eprintln!("[LIB]  • (b) THE √M WIN IS NOW IN CORE (#64). Routed to ONE skill via the core `route`, only it settles");
    eprintln!("[LIB]        (~1.0/query). The fractal map-of-maps (`cluster`/`index_skills`/`route`) routes in");
    eprintln!("[LIB]        ~{worst_sqrt_ratio:.2}×√M comparisons (vs a flat `distance()` sweep = M) — a substrate primitive over");
    eprintln!("[LIB]        the grown `pos`/`distance`. Single-path routing accuracy worst = {worst_acc:.1}%.");
    eprintln!("[LIB]  • (c) MEMORY / STRUCTURE grows O(M): units==M and cells==2M every row; each skill carries a");
    eprintln!("[LIB]        whole sub-Field (~20 parallel arrays). RSS climbs linearly (see table). The build itself");
    eprintln!(
        "[LIB]        is O(M). This linear footprint — not runtime — is the practical ceiling."
    );
    eprintln!("[LIB] ============================================================================");

    // Assertions DOCUMENT the limits (they pass by describing reality, they do not hide it):
    assert!(
        reached >= 4096,
        "the sweep must reach at least the requested M=4096 (reached {reached})"
    );
    assert!(
        settle_is_linear,
        "the un-routed/broadcast settle cost is O(M) — bc_settles must equal M at every scale"
    );
    assert!(
        mem_is_linear,
        "library memory is O(M): units==M and cells==2M at every scale"
    );
    assert!(
        worst_sqrt_ratio < 3.0,
        "the fractal index must route in < 3×√M comparisons at every scale (worst {worst_sqrt_ratio:.2})"
    );
    // accuracy is REPORTED, not gated high — but a total collapse would be a different (routing) failure.
    assert!(
        worst_acc > 50.0,
        "single-path fractal routing degraded below 50% (worst {worst_acc:.1}%) — that is the routing-accuracy wall"
    );
}
