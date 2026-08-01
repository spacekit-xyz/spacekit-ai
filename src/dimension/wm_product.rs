//! Phase 5b/5c/5f — Product act-loop metrics (return, not chat).
//!
//! - **5b:** disk return is the ship metric; visuomotor diagnostic; SpaceKit pin.
//! - **5c:** external product loop — DM citizen + disk return + SpaceKit host + pin.
//! - **5f:** live SpaceKit acting-host episode — multi-step return + pin reload.

use std::path::Path;

use rand::{rngs::StdRng, Rng, SeedableRng};

use super::wm_act::{
    act_step_disk, run_phase3t_disk_act_seed, run_phase3t_host_act_seed,
    run_phase3t_visuomotor_act_seed, train_disk_acting_bundle, ActSeedResult, ActingHostRequest,
    ActingHostSession,
};
use super::wm_dm::run_phase5a_wm_dm_spike;
use super::wm_open::run_phase3s_spacekit_host_seed;
use super::wm_transfer::{step_dynamics_action, WmAction};

#[derive(Clone, Debug)]
pub struct ProductActLoopResult {
    pub disk: ActSeedResult,
    pub visuomotor: ActSeedResult,
    pub host_pin_ok: bool,
    pub spacekit_host_ok: bool,
    pub return_beats_random: bool,
    pub return_beats_vg: bool,
    pub visuomotor_beats_random: bool,
    pub visuomotor_beats_vg: bool,
    pub pin_stable_all: bool,
    pub chat_metric_used: bool,
    pub product_gate_pass: bool,
    pub note: String,
}

/// Phase 5b: product-shaped act loop — disk return is the ship metric.
pub fn run_phase5b_product_act_loop(seed: u64, work_dir: &Path) -> ProductActLoopResult {
    let _ = std::fs::create_dir_all(work_dir);
    let disk = run_phase3t_disk_act_seed(seed, work_dir);
    let visuomotor = run_phase3t_visuomotor_act_seed(seed.wrapping_add(5), work_dir);
    let host_pin_ok = run_phase3t_host_act_seed(seed.wrapping_add(9), work_dir);
    let spacekit = run_phase3s_spacekit_host_seed(seed.wrapping_add(11), work_dir);
    let spacekit_host_ok = spacekit.load_ok && spacekit.step_ok && spacekit.pin_stable_reload;

    let return_beats_random = disk.return_wm > disk.return_random + 0.02;
    let return_beats_vg = disk.return_wm > disk.return_vg + 0.01;
    let visuomotor_beats_random = visuomotor.return_wm > visuomotor.return_random + 0.01;
    let visuomotor_beats_vg = visuomotor.return_wm > visuomotor.return_vg + 0.005;
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
        visuomotor_beats_random,
        visuomotor_beats_vg,
        pin_stable_all,
        chat_metric_used,
        product_gate_pass,
        note: "5b: product act-loop — disk return ship metric; visuomotor diagnostic; chat non-certifier"
            .into(),
    }
}

#[derive(Clone, Debug)]
pub struct ExternalProductLoopResult {
    pub disk_return_ok: bool,
    pub visuomotor_return_ok: bool,
    pub dm_citizen_ok: bool,
    pub spacekit_host_ok: bool,
    pub host_pin_ok: bool,
    pub pin_stable_all: bool,
    pub chat_metric_used: bool,
    pub product_gate_pass: bool,
    pub disk_return_wm: f32,
    pub disk_return_random: f32,
    pub disk_return_vg: f32,
    pub note: String,
}

/// Phase 5c — external product loop: DM citizens + return + SpaceKit pin (not chat).
pub fn run_phase5c_external_product_loop(seed: u64, work_dir: &Path) -> ExternalProductLoopResult {
    let _ = std::fs::create_dir_all(work_dir);
    let dm = run_phase5a_wm_dm_spike(seed, &work_dir.join("dm"));
    let act = run_phase5b_product_act_loop(seed.wrapping_add(2), &work_dir.join("act"));

    let disk_return_ok = act.return_beats_random && act.return_beats_vg;
    let visuomotor_return_ok = act.visuomotor_beats_random && act.visuomotor_beats_vg;
    let dm_citizen_ok = dm.checkpoint_roundtrip_ok
        && dm.act_ok_after_load
        && dm.deploy_ok_after_load
        && dm.pin_stable
        && dm.paths_portable;
    let product_gate_pass = !act.chat_metric_used
        && disk_return_ok
        && dm_citizen_ok
        && act.spacekit_host_ok
        && act.host_pin_ok
        && act.pin_stable_all;

    ExternalProductLoopResult {
        disk_return_ok,
        visuomotor_return_ok,
        dm_citizen_ok,
        spacekit_host_ok: act.spacekit_host_ok,
        host_pin_ok: act.host_pin_ok,
        pin_stable_all: act.pin_stable_all,
        chat_metric_used: act.chat_metric_used,
        product_gate_pass,
        disk_return_wm: act.disk.return_wm,
        disk_return_random: act.disk.return_random,
        disk_return_vg: act.disk.return_vg,
        note: "5c: external product loop — DM citizens + disk return + SpaceKit pin; Luna not certifier"
            .into(),
    }
}

#[derive(Clone, Debug)]
pub struct LiveSpacekitEpisodeResult {
    pub return_wm: f32,
    pub return_random: f32,
    pub return_beats_random: bool,
    pub host_steps_ok: bool,
    pub pin_stable_reload: bool,
    pub encoder_fingerprint: u64,
    pub n_steps: usize,
    pub chat_metric_used: bool,
    pub product_gate_pass: bool,
    pub note: String,
}

/// Phase 5f — live SpaceKit acting-host episode.
///
/// Multi-step disk episode through `ActingHostSession` JSON ops; ship metric is
/// task return + pin across host reload. Luna/chat is not a certifier.
pub fn run_phase5f_live_spacekit_episode(seed: u64, work_dir: &Path) -> LiveSpacekitEpisodeResult {
    use super::wm_act::{goal_disk, sample_disk};

    let _ = std::fs::create_dir_all(work_dir);
    let bundle = train_disk_acting_bundle(seed);
    let path = work_dir.join(format!("live_act_{seed}.json"));
    bundle.save(&path).expect("save acting");
    let pin = bundle.encoder_fingerprint;

    let mut host = ActingHostSession::new();
    let load = host.handle(ActingHostRequest::LoadActing {
        path: path.display().to_string(),
    });

    let mut rng = StdRng::seed_from_u64(seed.wrapping_add(77));
    let horizon = 20usize;
    let episodes = 24usize;
    let mut ret_wm = 0.0f32;
    let mut ret_rand = 0.0f32;
    let mut steps_ok = 0usize;
    let mut n_steps = 0usize;

    for _ in 0..episodes {
        let mut obs = sample_disk(rng.gen_bool(0.5), &mut rng);
        let mut obs_r = obs.clone();
        for _ in 0..horizon {
            let resp = host.handle(ActingHostRequest::Act { obs: obs.clone() });
            n_steps += 1;
            if resp.ok {
                steps_ok += 1;
                if let Some(d) = resp.decision {
                    let act = WmAction::from_u8(d.action);
                    let next = step_dynamics_action(&obs, act, 1.0);
                    ret_wm += goal_disk(&obs, &next);
                    obs = next;
                }
            } else if let Ok(d) = act_step_disk(&bundle, &obs) {
                // Fallback should not be needed; count as fail for host_steps_ok.
                let act = WmAction::from_u8(d.action);
                let next = step_dynamics_action(&obs, act, 1.0);
                ret_wm += goal_disk(&obs, &next);
                obs = next;
            }

            let ar = WmAction::from_u8(rng.gen_range(0..4));
            let next_r = step_dynamics_action(&obs_r, ar, 1.0);
            ret_rand += goal_disk(&obs_r, &next_r);
            obs_r = next_r;
        }
    }

    let mut host2 = ActingHostSession::new();
    let reload = host2.handle(ActingHostRequest::LoadActing {
        path: path.display().to_string(),
    });
    let fp = host2.handle(ActingHostRequest::Fingerprint);
    let pin_stable_reload = load.ok
        && reload.ok
        && load.fingerprint == reload.fingerprint
        && reload.fingerprint == Some(pin)
        && fp.fingerprint == Some(pin);

    let return_wm = ret_wm / episodes as f32;
    let return_random = ret_rand / episodes as f32;
    let return_beats_random = return_wm > return_random + 0.02;
    let host_steps_ok = load.ok && steps_ok == n_steps && n_steps > 0;
    let chat_metric_used = false;
    let product_gate_pass =
        !chat_metric_used && host_steps_ok && pin_stable_reload && return_beats_random;

    LiveSpacekitEpisodeResult {
        return_wm,
        return_random,
        return_beats_random,
        host_steps_ok,
        pin_stable_reload,
        encoder_fingerprint: pin,
        n_steps,
        chat_metric_used,
        product_gate_pass,
        note: "5f: live ActingHostSession episode — return + pin; chat non-certifier".into(),
    }
}
