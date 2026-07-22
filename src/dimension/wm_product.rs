//! Phase 5b — External act-loop product metric (return, not chat).
//!
//! Runs pinned acting agents through multi-episode horizons where the **product
//! gate** is task return vs random/VG + pin stability + regime agreement.
//! Chat / Luna metrics are explicitly excluded.

use std::path::Path;

use super::wm_act::{
    run_phase3t_disk_act_seed, run_phase3t_host_act_seed, run_phase3t_visuomotor_act_seed,
    ActSeedResult,
};
use super::wm_open::run_phase3s_spacekit_host_seed;

#[derive(Clone, Debug)]
pub struct ProductActLoopResult {
    pub disk: ActSeedResult,
    pub visuomotor: ActSeedResult,
    pub host_pin_ok: bool,
    pub spacekit_host_ok: bool,
    pub return_beats_random: bool,
    pub return_beats_vg: bool,
    pub pin_stable_all: bool,
    pub chat_metric_used: bool,
    pub product_gate_pass: bool,
    pub note: String,
}

/// Phase 5b: product-shaped act loop — return is the ship metric.
pub fn run_phase5b_product_act_loop(seed: u64, work_dir: &Path) -> ProductActLoopResult {
    let _ = std::fs::create_dir_all(work_dir);
    let disk = run_phase3t_disk_act_seed(seed, work_dir);
    let visuomotor = run_phase3t_visuomotor_act_seed(seed.wrapping_add(5), work_dir);
    let host_pin_ok = run_phase3t_host_act_seed(seed.wrapping_add(9), work_dir);
    let spacekit = run_phase3s_spacekit_host_seed(seed.wrapping_add(11), work_dir);
    let spacekit_host_ok = spacekit.load_ok && spacekit.step_ok && spacekit.pin_stable_reload;

    // Product ship metric = **disk** return (visuomotor remains diagnostic transfer).
    let return_beats_random = disk.return_wm > disk.return_random + 0.02;
    let return_beats_vg = disk.return_wm > disk.return_vg + 0.01;
    let pin_stable_all = disk.pin_stable && visuomotor.pin_stable && host_pin_ok;
    let chat_metric_used = disk.chat_metric_used || visuomotor.chat_metric_used;
    let product_gate_pass = !chat_metric_used
        && pin_stable_all
        && return_beats_random
        && return_beats_vg
        && !disk.degenerate
        && disk.regime_agreement >= 0.70
        && spacekit_host_ok;

    ProductActLoopResult {
        disk,
        visuomotor,
        host_pin_ok,
        spacekit_host_ok,
        return_beats_random,
        return_beats_vg,
        pin_stable_all,
        chat_metric_used,
        product_gate_pass,
        note: "5b: product act-loop — disk return vs random/VG is ship metric; visuomotor diagnostic; chat non-certifier"
            .into(),
    }
}
