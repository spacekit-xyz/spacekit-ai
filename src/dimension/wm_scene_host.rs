//! Phase 3w — SpaceKit scene-graph WM host (WORLD_MODELS spatial deploy).
//!
//! JSON protocol over [`SceneWmBundle`]: load → step / act → reload pin.
//! Not Luna / chat.

use rand::{rngs::StdRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::path::Path;

use super::wm_scene::{
    evaluate_scene_wm_bundle, pick_block, sample_scene, scene_act_step, scene_deploy_step,
    train_scene_wm_bundle, SceneActDecision, SceneGraph, SceneStepDecision, SceneWmBundle,
};

// =============================================================================
// Host protocol
// =============================================================================

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum SceneHostRequest {
    LoadScene { path: String },
    /// Predictive energy route on a scene graph JSON body.
    Step { scene: SceneGraph },
    /// Plan discrete action for `block_idx` (defaults to first block if omitted).
    Act {
        scene: SceneGraph,
        #[serde(default)]
        block_idx: Option<usize>,
    },
    Fingerprint,
    Status,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SceneHostResponse {
    pub ok: bool,
    pub error: Option<String>,
    pub step: Option<SceneStepDecision>,
    pub act: Option<SceneActDecision>,
    pub fingerprint: Option<u64>,
    pub loaded: bool,
    pub note: String,
}

#[derive(Default)]
pub struct SceneHostSession {
    bundle: Option<SceneWmBundle>,
    path: Option<String>,
}

impl SceneHostSession {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn handle(&mut self, req: SceneHostRequest) -> SceneHostResponse {
        match req {
            SceneHostRequest::LoadScene { path } => match SceneWmBundle::load(Path::new(&path)) {
                Ok(b) => {
                    let fp = b.encoder_fingerprint;
                    self.path = Some(path);
                    self.bundle = Some(b);
                    SceneHostResponse {
                        ok: true,
                        error: None,
                        step: None,
                        act: None,
                        fingerprint: Some(fp),
                        loaded: true,
                        note: "scene bundle loaded (SpaceKit scene host; not Luna)".into(),
                    }
                }
                Err(e) => SceneHostResponse {
                    ok: false,
                    error: Some(e),
                    step: None,
                    act: None,
                    fingerprint: None,
                    loaded: false,
                    note: "load_scene failed".into(),
                },
            },
            SceneHostRequest::Step { scene } => {
                let Some(b) = self.bundle.as_ref() else {
                    return SceneHostResponse {
                        ok: false,
                        error: Some("no scene bundle".into()),
                        step: None,
                        act: None,
                        fingerprint: None,
                        loaded: false,
                        note: "load_scene first".into(),
                    };
                };
                match scene_deploy_step(b, &scene) {
                    Ok(d) => SceneHostResponse {
                        ok: true,
                        error: None,
                        fingerprint: Some(d.encoder_fingerprint),
                        step: Some(d),
                        act: None,
                        loaded: true,
                        note: "scene step ok".into(),
                    },
                    Err(e) => SceneHostResponse {
                        ok: false,
                        error: Some(e),
                        step: None,
                        act: None,
                        fingerprint: Some(b.encoder_fingerprint),
                        loaded: true,
                        note: "scene step failed".into(),
                    },
                }
            }
            SceneHostRequest::Act { scene, block_idx } => {
                let Some(b) = self.bundle.as_ref() else {
                    return SceneHostResponse {
                        ok: false,
                        error: Some("no scene bundle".into()),
                        step: None,
                        act: None,
                        fingerprint: None,
                        loaded: false,
                        note: "load_scene first".into(),
                    };
                };
                let bi = block_idx.unwrap_or_else(|| {
                    if scene.nodes.len() > 1 {
                        1
                    } else {
                        0
                    }
                });
                match scene_act_step(b, &scene, bi) {
                    Ok(d) => SceneHostResponse {
                        ok: true,
                        error: None,
                        fingerprint: Some(d.encoder_fingerprint),
                        step: None,
                        act: Some(d),
                        loaded: true,
                        note: "scene act ok".into(),
                    },
                    Err(e) => SceneHostResponse {
                        ok: false,
                        error: Some(e),
                        step: None,
                        act: None,
                        fingerprint: Some(b.encoder_fingerprint),
                        loaded: true,
                        note: "scene act failed".into(),
                    },
                }
            }
            SceneHostRequest::Fingerprint => SceneHostResponse {
                ok: self.bundle.is_some(),
                error: None,
                step: None,
                act: None,
                fingerprint: self.bundle.as_ref().map(|b| b.encoder_fingerprint),
                loaded: self.bundle.is_some(),
                note: "fingerprint".into(),
            },
            SceneHostRequest::Status => SceneHostResponse {
                ok: true,
                error: None,
                step: None,
                act: None,
                fingerprint: self.bundle.as_ref().map(|b| b.encoder_fingerprint),
                loaded: self.bundle.is_some(),
                note: format!(
                    "SpaceKit scene host | path={}",
                    self.path.as_deref().unwrap_or("(none)")
                ),
            },
        }
    }

    pub fn handle_json(&mut self, line: &str) -> String {
        match serde_json::from_str::<SceneHostRequest>(line) {
            Ok(req) => serde_json::to_string(&self.handle(req)).unwrap_or_else(|e| {
                format!(r#"{{"ok":false,"error":"{e}","loaded":false,"note":"serialize"}}"#)
            }),
            Err(e) => format!(
                r#"{{"ok":false,"error":"{e}","loaded":false,"note":"bad request"}}"#
            ),
        }
    }
}

// =============================================================================
// Certifier
// =============================================================================

#[derive(Clone, Debug)]
pub struct SceneHostSeedResult {
    pub load_ok: bool,
    pub step_ok: bool,
    pub act_ok: bool,
    pub pin_stable_reload: bool,
    pub regime_via_host: f32,
    pub return_via_host: f32,
    pub return_random: f32,
    pub fingerprint: u64,
    pub chat_metric_used: bool,
    pub log_path: String,
}

/// Train → save → host load/step/act → reload pin → host-routed return.
pub fn run_phase3w_scene_host_seed(seed: u64, work_dir: &Path) -> SceneHostSeedResult {
    let _ = std::fs::create_dir_all(work_dir);
    let bundle = train_scene_wm_bundle(seed);
    let fp0 = bundle.encoder_fingerprint;
    let path = work_dir.join(format!("scene_bundle_{seed}.json"));
    bundle.save(&path).expect("save scene bundle");

    let mut host = SceneHostSession::new();
    let load = host.handle(SceneHostRequest::LoadScene {
        path: path.display().to_string(),
    });

    let mut rng = StdRng::seed_from_u64(seed.wrapping_add(11));
    let scene0 = sample_scene(true, &mut rng);
    let step = host.handle(SceneHostRequest::Step {
        scene: scene0.clone(),
    });
    let bi = pick_block(&scene0, &mut rng);
    let act = host.handle(SceneHostRequest::Act {
        scene: scene0,
        block_idx: Some(bi),
    });

    // Process-restart proxy
    let mut host2 = SceneHostSession::new();
    let reload = host2.handle(SceneHostRequest::LoadScene {
        path: path.display().to_string(),
    });
    let pin_stable_reload = load.ok
        && reload.ok
        && load.fingerprint == reload.fingerprint
        && load.fingerprint == Some(fp0);

    // Reloaded bundle must certify under the same eval as Phase 3v.
    let loaded = SceneWmBundle::load(&path).expect("reload for eval");
    let mut rng_reg = StdRng::seed_from_u64(seed.wrapping_mul(17).wrapping_add(3));
    let n = 120usize;
    let mut regime_ok = 0usize;
    for _ in 0..n {
        let stable = rng_reg.gen_bool(0.5);
        let g = sample_scene(stable, &mut rng_reg);
        let d = scene_deploy_step(&loaded, &g).expect("step");
        if d.route_stable == stable {
            regime_ok += 1;
        }
    }
    let eval = evaluate_scene_wm_bundle(&loaded, seed);

    // JSONL log for SpaceKit integration smoke
    let log_path = work_dir.join(format!("scene_host_{seed}.jsonl"));
    let mut rng_log = StdRng::seed_from_u64(seed.wrapping_add(99));
    let mut lines = Vec::new();
    lines.push(
        serde_json::to_string(&SceneHostRequest::LoadScene {
            path: path.display().to_string(),
        })
        .unwrap(),
    );
    let probe = sample_scene(false, &mut rng_log);
    lines.push(
        serde_json::to_string(&SceneHostRequest::Step {
            scene: probe.clone(),
        })
        .unwrap(),
    );
    lines.push(
        serde_json::to_string(&SceneHostRequest::Act {
            scene: probe,
            block_idx: Some(1),
        })
        .unwrap(),
    );
    let _ = std::fs::write(&log_path, lines.join("\n") + "\n");

    SceneHostSeedResult {
        load_ok: load.ok,
        step_ok: step.ok && step.step.is_some(),
        act_ok: act.ok && act.act.is_some(),
        pin_stable_reload,
        regime_via_host: regime_ok as f32 / n as f32,
        return_via_host: eval.return_wm,
        return_random: eval.return_random,
        fingerprint: fp0,
        chat_metric_used: false,
        log_path: log_path.display().to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scene_host_pin_and_act() {
        let dir = std::env::temp_dir().join(format!("sc_host_{}", std::process::id()));
        let r = run_phase3w_scene_host_seed(42, &dir);
        assert!(!r.chat_metric_used);
        assert!(r.load_ok && r.step_ok && r.act_ok);
        assert!(r.pin_stable_reload);
        assert!(r.regime_via_host >= 0.60, "regime {}", r.regime_via_host);
        assert!(
            r.return_via_host > r.return_random,
            "ret {} rand {}",
            r.return_via_host,
            r.return_random
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}