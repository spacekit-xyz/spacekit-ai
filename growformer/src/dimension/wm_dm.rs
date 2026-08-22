//! Phase 5a — World-model citizens inside DimensionManager (spike).
//!
//! Parallel WM bundles become named Main citizens: GroupId + embedding stub +
//! encoder fingerprint + serialized acting/composed bundle. Inference delegates
//! to existing `act_step_disk` / `deploy_step`.
//!
//! Does **not** claim full AMI or shared-backbone CL. Chat is not a certifier.

use std::path::Path;

use crate::environment::NeuralEnvironment;
use crate::types::{EnvironmentConfig, GroupId};
use rand::rngs::StdRng;
use rand::SeedableRng;

use super::embedding::{build_tag_vector, GroupEmbedding, TAG_VECTOR_DIM};
use super::manager::{DimensionManager, DimensionManagerConfig};
use super::wm_act::{act_step_disk, train_disk_acting_bundle, ActDecision, ActingWmBundle};
use super::wm_citizen::{WmCitizenKind, WmCitizenRecord};
use super::wm_transfer::{
    deploy_step, load_composed_bundle, save_composed_bundle, train_composed_bundle,
    ComposedWmBundle, DeployDecision,
};

#[derive(Clone, Debug)]
pub struct WmDmSpikeResult {
    pub group_id: GroupId,
    pub encoder_fingerprint: u64,
    pub pin_stable: bool,
    pub reload_fingerprint_match: bool,
    pub act_ok: bool,
    pub deploy_ok: bool,
    pub checkpoint_roundtrip_ok: bool,
    pub act_ok_after_load: bool,
    pub deploy_ok_after_load: bool,
    pub paths_portable: bool,
    pub chat_metric_used: bool,
    pub n_citizens: usize,
    pub note: String,
}

fn stub_frozen_env(obs_dim: usize, seed: u64) -> NeuralEnvironment {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut env = NeuralEnvironment::new(EnvironmentConfig::default());
    env.build_layers(&[obs_dim.max(4), 8, 1], &mut rng);
    env.freeze_all();
    env
}

impl DimensionManager {
    /// Promote an acting WM bundle as a Main citizen (GroupId + pin + path).
    pub fn promote_wm_acting_citizen(
        &mut self,
        task_name: &str,
        bundle: &ActingWmBundle,
        bundle_path: &Path,
        prototype: &[f32],
    ) -> Result<GroupId, String> {
        bundle.verify()?;
        let path_str = bundle_path
            .to_str()
            .ok_or_else(|| "bundle path not utf8".to_string())?
            .to_string();
        let gid = self.next_group_id;
        let env = stub_frozen_env(prototype.len().max(4), gid as u64 + 7);
        let metatags = vec![
            "wm".into(),
            "acting".into(),
            "disk".into(),
            format!("pin_{:016x}", bundle.encoder_fingerprint),
        ];
        let emb = GroupEmbedding {
            group_id: gid,
            vector: prototype.to_vec(),
            task_name: task_name.to_string(),
            accuracy: 1.0,
            intrinsic_dim: None,
            description: Some("WM acting citizen (spike)".into()),
            tag_vector: build_tag_vector(&metatags, TAG_VECTOR_DIM),
            metatags,
            language_vector: Vec::new(),
        };
        self.main
            .register_group(gid, task_name.to_string(), env, emb, 1.0, 0);
        self.observer.embedding_library = self.main.embedding_library.clone();
        self.wm_citizens.insert(
            gid,
            WmCitizenRecord {
                group_id: gid,
                task_name: task_name.to_string(),
                kind: WmCitizenKind::ActingDisk,
                encoder_fingerprint: bundle.encoder_fingerprint,
                bundle_path: path_str,
            },
        );
        self.next_group_id = self.next_group_id.saturating_add(1);
        Ok(gid)
    }

    /// Promote a composed energy WM bundle as a Main citizen.
    pub fn promote_wm_composed_citizen(
        &mut self,
        task_name: &str,
        bundle: &ComposedWmBundle,
        bundle_path: &Path,
        prototype: &[f32],
    ) -> Result<GroupId, String> {
        bundle.verify()?;
        let path_str = bundle_path
            .to_str()
            .ok_or_else(|| "bundle path not utf8".to_string())?
            .to_string();
        let gid = self.next_group_id;
        let env = stub_frozen_env(prototype.len().max(4), gid as u64 + 11);
        let metatags = vec![
            "wm".into(),
            "composed".into(),
            format!("pin_{:016x}", bundle.encoder_fingerprint),
        ];
        let emb = GroupEmbedding {
            group_id: gid,
            vector: prototype.to_vec(),
            task_name: task_name.to_string(),
            accuracy: 1.0,
            intrinsic_dim: None,
            description: Some("WM composed citizen (spike)".into()),
            tag_vector: build_tag_vector(&metatags, TAG_VECTOR_DIM),
            metatags,
            language_vector: Vec::new(),
        };
        self.main
            .register_group(gid, task_name.to_string(), env, emb, 1.0, 0);
        self.observer.embedding_library = self.main.embedding_library.clone();
        self.wm_citizens.insert(
            gid,
            WmCitizenRecord {
                group_id: gid,
                task_name: task_name.to_string(),
                kind: WmCitizenKind::Composed,
                encoder_fingerprint: bundle.encoder_fingerprint,
                bundle_path: path_str,
            },
        );
        self.next_group_id = self.next_group_id.saturating_add(1);
        Ok(gid)
    }

    pub fn wm_citizen(&self, gid: GroupId) -> Option<&WmCitizenRecord> {
        self.wm_citizens.get(&gid)
    }

    /// Route to a WM citizen by metatag (e.g. `"wm"`, `"acting"`, `"composed"`).
    pub fn route_wm_citizen_by_tag(&self, tag: &str) -> Option<GroupId> {
        self.main
            .embedding_library
            .iter()
            .find(|e| {
                e.metatags.iter().any(|m| m == tag) && self.wm_citizens.contains_key(&e.group_id)
            })
            .map(|e| e.group_id)
    }

    /// Act via citizen's pinned acting bundle (disk).
    pub fn wm_act_disk(&self, gid: GroupId, obs: &[f32]) -> Result<ActDecision, String> {
        let c = self
            .wm_citizens
            .get(&gid)
            .ok_or_else(|| format!("no wm citizen {gid}"))?;
        if c.kind != WmCitizenKind::ActingDisk {
            return Err("citizen is not ActingDisk".into());
        }
        let bundle = ActingWmBundle::load(Path::new(&c.bundle_path))?;
        if bundle.encoder_fingerprint != c.encoder_fingerprint {
            return Err("citizen pin drift vs bundle".into());
        }
        act_step_disk(&bundle, obs)
    }

    /// Deploy-step via citizen's pinned composed bundle.
    pub fn wm_deploy_step(&self, gid: GroupId, obs: &[f32]) -> Result<DeployDecision, String> {
        let c = self
            .wm_citizens
            .get(&gid)
            .ok_or_else(|| format!("no wm citizen {gid}"))?;
        if c.kind != WmCitizenKind::Composed {
            return Err("citizen is not Composed".into());
        }
        let bundle = load_composed_bundle(Path::new(&c.bundle_path))?;
        if bundle.encoder_fingerprint != c.encoder_fingerprint {
            return Err("citizen pin drift vs bundle".into());
        }
        deploy_step(&bundle, obs)
    }
}

/// Rewrite citizen bundle paths to absolute (portable across CWD for reload).
pub fn canonicalize_wm_citizen_paths(dm: &mut DimensionManager) {
    for c in dm.wm_citizens.values_mut() {
        if let Ok(p) = std::fs::canonicalize(&c.bundle_path) {
            if let Some(s) = p.to_str() {
                c.bundle_path = s.to_string();
            }
        }
    }
}

/// Install acting + composed WM citizens into an existing DimensionManager (sidecar bundles).
/// Used by 5a spike and 5e brain.bin graduation.
pub fn install_wm_citizens(
    dm: &mut DimensionManager,
    seed: u64,
    work_dir: &Path,
) -> Result<(GroupId, GroupId), String> {
    let sidecar = work_dir.join("wm_sidecar");
    std::fs::create_dir_all(&sidecar).map_err(|e| e.to_string())?;

    let acting = train_disk_acting_bundle(seed);
    let act_path = sidecar.join(format!("dm_acting_{seed}.json"));
    acting.save(&act_path)?;
    let proto_act = acting
        .geo_encoder
        .as_ref()
        .map(|e| e.encode(&[0.2, 0.1, 0.0, 0.0]))
        .unwrap_or_else(|| vec![0.0; 16]);
    let act_gid = dm.promote_wm_acting_citizen("wm_disk_act", &acting, &act_path, &proto_act)?;

    let composed = train_composed_bundle(seed.wrapping_add(3));
    let comp_path = sidecar.join(format!("dm_composed_{seed}.json"));
    save_composed_bundle(&comp_path, &composed)?;
    let proto_c = composed.encoder.encode(&[0.15, -0.1, 0.05, 0.0]);
    let dep_gid = dm.promote_wm_composed_citizen("wm_composed", &composed, &comp_path, &proto_c)?;

    canonicalize_wm_citizen_paths(dm);
    Ok((act_gid, dep_gid))
}

/// Phase 5a: promote acting+composed, pin, act/deploy, **DM checkpoint roundtrip**.
pub fn run_phase5a_wm_dm_spike(seed: u64, work_dir: &Path) -> WmDmSpikeResult {
    use crate::systems::checkpoint::{
        deserialize_checkpoint_from_bytes, serialize_checkpoint_to_bytes,
    };

    let _ = std::fs::create_dir_all(work_dir);
    let config = DimensionManagerConfig {
        mirror_config: EnvironmentConfig::default(),
        mirror_layer_sizes: vec![4, 8, 1],
        promotion_check_interval: 999_999,
        max_concurrent_mirrors: 1,
        calibration_samples: 8,
        reserve_pool_size: 0,
    };
    let mut dm = DimensionManager::new(config);

    let (act_gid, dep_gid) =
        install_wm_citizens(&mut dm, seed, work_dir).expect("install wm citizens");
    let acting_fp = dm
        .wm_citizen(act_gid)
        .map(|c| c.encoder_fingerprint)
        .unwrap_or(0);
    let act_path = dm
        .wm_citizen(act_gid)
        .map(|c| c.bundle_path.clone())
        .unwrap_or_default();

    let paths_portable = dm
        .wm_citizens
        .values()
        .all(|c| Path::new(&c.bundle_path).is_absolute());

    let obs = [0.25f32, 0.05, 0.0, 0.0];
    let act_ok = dm.wm_act_disk(act_gid, &obs).is_ok();
    let deploy_ok = dm.wm_deploy_step(dep_gid, &obs).is_ok();

    let reloaded = ActingWmBundle::load(Path::new(&act_path)).expect("reload");
    let pin_stable = reloaded.verify().is_ok();
    let citizen = dm.wm_citizen(act_gid).expect("citizen");
    let reload_fingerprint_match = reloaded.encoder_fingerprint == citizen.encoder_fingerprint
        && citizen.encoder_fingerprint == acting_fp;

    // Persist DM (wm_citizens included) and restore — Preview+ graduation gate.
    let bytes = serialize_checkpoint_to_bytes(&dm).expect("serialize dm");
    let dm2: DimensionManager = deserialize_checkpoint_from_bytes(&bytes).expect("deserialize dm");
    let checkpoint_roundtrip_ok = dm2.wm_citizens.len() == dm.wm_citizens.len()
        && dm2.wm_citizen(act_gid).map(|c| c.encoder_fingerprint) == Some(acting_fp);
    let act_ok_after_load = dm2.wm_act_disk(act_gid, &obs).is_ok();
    let deploy_ok_after_load = dm2.wm_deploy_step(dep_gid, &obs).is_ok();

    WmDmSpikeResult {
        group_id: act_gid,
        encoder_fingerprint: acting_fp,
        pin_stable,
        reload_fingerprint_match,
        act_ok,
        deploy_ok,
        checkpoint_roundtrip_ok,
        act_ok_after_load,
        deploy_ok_after_load,
        paths_portable,
        chat_metric_used: false,
        n_citizens: dm.wm_citizens.len(),
        note: "5a: WM citizens in DM + sidecar bundles + checkpoint roundtrip (Preview+)".into(),
    }
}
