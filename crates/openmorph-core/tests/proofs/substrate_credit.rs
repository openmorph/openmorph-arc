// SPDX-License-Identifier: MIT-0
//! CREDIT-THROUGH-TIME from a TERMINAL reward — the "dense from sparse" theme, credit-ASSIGNMENT half.
//!
//! MorphLab's sweep: a latch is 100% under dense supervision but only ~46% when the reward is terminal-only.
//! The diagnosis is the same one that runs through #137/#139/#142: the substrate grows beautifully when the
//! signal is DENSE, so the work is manufacturing a dense signal from the sparse/delayed one. It needs no new
//! mechanism. Bootstrapping the morph's OWN estimate of the next state one step (TD) turns a single terminal
//! reward into a dense per-step target — `V(s) ← γ·V(next)` over the transitions it EXPERIENCED — and the
//! ordinary local delta rule learns it. A Monte-Carlo target (every state ← the final reward, no bootstrap)
//! is the wall: credit never differentiates the steps.
//!
//! TRUTH-NOT-TROPHY: this is the credit-ASSIGNMENT half only — the agent still has to REACH the reward. The
//! residual frontier is EXPLORATION under sparse reward + planning over a learned model, which is a harness
//! protocol (go-explore / behaviour-frontier #118), not a substrate primitive. This proof asserts only what
//! the substrate owns: given experienced transitions, TD manufactures the dense signal and credit travels.
//!
//!   cargo test -p openmorph-core --test proofs substrate_credit -- --nocapture --test-threads=1
use openmorph_core::substrate::Field;

const N: usize = 8; // chain length — credit must travel N steps back from the terminal reward
const GAMMA: f32 = 0.9;

fn onehot(s: usize) -> Vec<(usize, f32)> {
    (0..N).map(|i| (i, f32::from(i == s))).collect()
}
fn truth(s: usize) -> f32 {
    GAMMA.powi((N - 1 - s) as i32)
} // true discounted value, R=1 at terminal

/// TD: target bootstraps the morph's own estimate of the EXPERIENCED next state (dense every step).
fn run_td(seed: u64) -> f32 {
    let mut f = Field::new(N);
    let out = f.len();
    f.ensure_cells(out + 1);
    f.rng_seed(seed);
    f.config_mut().feature_lr = 0.2;
    f.config_mut().max_hidden = 24;
    for _ in 0..4000 {
        for s in 0..N {
            let target = if s == N - 1 {
                1.0
            } else {
                f.stimulate(&onehot(s + 1), 40);
                GAMMA * f.cells()[out]
            };
            f.grow(&onehot(s), out, target, 0.08, 40);
        }
    }
    let mut e = 0.0;
    for s in 0..N {
        f.stimulate(&onehot(s), 40);
        e += (f.cells()[out] - truth(s)).abs();
    }
    e / N as f32
}
/// Monte-Carlo: every state's target is the terminal reward, no bootstrap — the wall.
fn run_mc(seed: u64) -> f32 {
    let mut f = Field::new(N);
    let out = f.len();
    f.ensure_cells(out + 1);
    f.rng_seed(seed);
    f.config_mut().feature_lr = 0.2;
    f.config_mut().max_hidden = 24;
    for _ in 0..4000 {
        for s in 0..N {
            f.grow(&onehot(s), out, 1.0, 0.08, 40);
        }
    }
    let mut e = 0.0;
    for s in 0..N {
        f.stimulate(&onehot(s), 40);
        e += (f.cells()[out] - truth(s)).abs();
    }
    e / N as f32
}

#[test]
fn credit_travels_from_terminal_reward() {
    let (mut td, mut mc) = (0.0, 0.0);
    let seeds = 4;
    for s in 0..seeds {
        td += run_td(0x100 + s);
        mc += run_mc(0x100 + s);
    }
    let (td, mc) = (td / seeds as f32, mc / seeds as f32);
    eprintln!("[CREDIT] terminal reward over an {N}-step chain, MAE vs the true discounted value:");
    eprintln!("[CREDIT]   Monte-Carlo (terminal target, no bootstrap): {mc:.3}  (credit can't differentiate the steps — the wall)");
    eprintln!("[CREDIT]   TD bootstrap (dense from own next-estimate):  {td:.3}  (credit travels the full chain)");
    eprintln!("[CREDIT] VERDICT: the credit-ASSIGNMENT half of MorphLab's terminal-reward wall (~46%) needs NO new mechanism. Bootstrapping the morph's OWN estimate of the experienced next state turns one terminal reward into a DENSE per-step target (V(s)←γ·V(next)), and the same local delta rule learns it — MAE {mc:.3}→{td:.3}. Same 'dense from sparse' theme as #137/#139/#142. HONEST RESIDUAL: reaching the reward in the first place (EXPLORATION under sparse reward + planning over a learned model) is the real open frontier — a harness protocol (go-explore / behaviour-frontier #118), not a substrate gap. This asserts only what the substrate owns.");
    assert!(
        td < mc * 0.2,
        "TD bootstrap must let credit travel where Monte-Carlo walls (td {td:.3} vs mc {mc:.3})"
    );
}
