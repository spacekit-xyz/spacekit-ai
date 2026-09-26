//! Checkpoint save/load for row-2 vanilla transformer (schema 1).

use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Write;
use std::path::Path;

use crate::lm_config::TrainConfigV2;
use crate::real_linear::LinearReal;
use crate::vanilla_llm::{VanillaBlock, VanillaLLM};

use super::vanilla_train::{VanillaModelState, VanillaOptimState};

const VANILLA_CHECKPOINT_SCHEMA: u32 = 1;

#[derive(Debug, Serialize, Deserialize)]
pub struct RealLinearDto {
    pub weights: Vec<Vec<f32>>,
    pub bias: Vec<f32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct VanillaBlockDto {
    pub norm1_gamma: Vec<f32>,
    pub norm1_beta: Vec<f32>,
    pub wq: RealLinearDto,
    pub wk: RealLinearDto,
    pub wv: RealLinearDto,
    pub wo: RealLinearDto,
    pub norm2_gamma: Vec<f32>,
    pub norm2_beta: Vec<f32>,
    pub fc1: RealLinearDto,
    pub fc2: RealLinearDto,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct VanillaCheckpoint {
    pub schema: u32,
    pub step: u64,
    pub cfg: TrainConfigV2,
    pub embedding: Vec<Vec<f32>>,
    pub blocks: Vec<VanillaBlockDto>,
    pub final_norm_gamma: Vec<f32>,
    pub final_norm_beta: Vec<f32>,
    pub head: RealLinearDto,
}

fn snap_linear(l: &LinearReal) -> RealLinearDto {
    RealLinearDto {
        weights: l.weights.clone(),
        bias: l.bias.clone(),
    }
}

fn snap_block(b: &VanillaBlock) -> VanillaBlockDto {
    VanillaBlockDto {
        norm1_gamma: b.norm1.gamma.clone(),
        norm1_beta: b.norm1.beta.clone(),
        wq: snap_linear(&b.attn.w_q),
        wk: snap_linear(&b.attn.w_k),
        wv: snap_linear(&b.attn.w_v),
        wo: snap_linear(&b.attn.w_o),
        norm2_gamma: b.norm2.gamma.clone(),
        norm2_beta: b.norm2.beta.clone(),
        fc1: snap_linear(&b.ffn.fc1),
        fc2: snap_linear(&b.ffn.fc2),
    }
}

fn apply_linear(l: &mut LinearReal, d: &RealLinearDto) -> Result<(), String> {
    if l.weights.len() != d.weights.len() || l.bias.len() != d.bias.len() {
        return Err("linear shape mismatch".into());
    }
    l.weights = d.weights.clone();
    l.bias = d.bias.clone();
    Ok(())
}

fn apply_block(b: &mut VanillaBlock, d: &VanillaBlockDto) -> Result<(), String> {
    b.norm1.gamma = d.norm1_gamma.clone();
    b.norm1.beta = d.norm1_beta.clone();
    apply_linear(&mut b.attn.w_q, &d.wq)?;
    apply_linear(&mut b.attn.w_k, &d.wk)?;
    apply_linear(&mut b.attn.w_v, &d.wv)?;
    apply_linear(&mut b.attn.w_o, &d.wo)?;
    b.norm2.gamma = d.norm2_gamma.clone();
    b.norm2.beta = d.norm2_beta.clone();
    apply_linear(&mut b.ffn.fc1, &d.fc1)?;
    apply_linear(&mut b.ffn.fc2, &d.fc2)?;
    Ok(())
}

fn write_ckpt(path: &Path, ckpt: &VanillaCheckpoint) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let json = serde_json::to_string_pretty(ckpt).map_err(|e| e.to_string())?;
    let mut f = File::create(path).map_err(|e| e.to_string())?;
    f.write_all(json.as_bytes()).map_err(|e| e.to_string())
}

pub fn save_vanilla_state(path: &Path, state: &VanillaModelState) -> Result<(), String> {
    let ckpt = VanillaCheckpoint {
        schema: VANILLA_CHECKPOINT_SCHEMA,
        step: state.step,
        cfg: state.cfg.clone(),
        embedding: state.model.embedding.clone(),
        blocks: state.model.blocks.iter().map(snap_block).collect(),
        final_norm_gamma: state.model.final_norm.gamma.clone(),
        final_norm_beta: state.model.final_norm.beta.clone(),
        head: snap_linear(&state.model.head),
    };
    write_ckpt(path, &ckpt)
}

pub fn load_vanilla_state(path: &Path) -> Result<VanillaModelState, String> {
    let raw = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let ckpt: VanillaCheckpoint = serde_json::from_str(&raw).map_err(|e| e.to_string())?;
    if ckpt.schema != VANILLA_CHECKPOINT_SCHEMA {
        return Err(format!(
            "unsupported vanilla checkpoint schema {}",
            ckpt.schema
        ));
    }
    if !ckpt.cfg.vanilla {
        return Err("checkpoint cfg.vanilla is false — not a row-2 vanilla checkpoint".into());
    }
    let mut model = VanillaLLM::new(
        ckpt.cfg.vocab_size,
        ckpt.cfg.d_model,
        ckpt.cfg.n_heads,
        ckpt.cfg.d_ff,
        ckpt.cfg.n_blocks,
        ckpt.cfg.init_seed,
    );
    if ckpt.embedding.len() != model.embedding.len() {
        return Err("embedding vocab mismatch".into());
    }
    model.embedding = ckpt.embedding;
    if ckpt.blocks.len() != model.blocks.len() {
        return Err("block count mismatch".into());
    }
    for (b, dto) in model.blocks.iter_mut().zip(&ckpt.blocks) {
        apply_block(b, dto)?;
    }
    model.final_norm.gamma = ckpt.final_norm_gamma;
    model.final_norm.beta = ckpt.final_norm_beta;
    apply_linear(&mut model.head, &ckpt.head)?;
    if ckpt.cfg.tie_embeddings {
        model.sync_tied_head();
    }
    Ok(VanillaModelState::from_loaded(ckpt.cfg, model, ckpt.step))
}

/// Sidecar path holding optimiser moments for a checkpoint:
/// `foo.json` → `foo.optim.json`.
pub fn optim_sidecar_path(checkpoint: &Path) -> std::path::PathBuf {
    checkpoint.with_extension("optim.json")
}

/// Write Adam moments + step so a run can be resumed exactly (`--resume`).
/// Kept out of the main checkpoint so inference files stay small.
pub fn save_vanilla_optim(checkpoint: &Path, state: &VanillaModelState) -> Result<(), String> {
    let path = optim_sidecar_path(checkpoint);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let json = serde_json::to_string(&state.optim_snapshot()).map_err(|e| e.to_string())?;
    std::fs::write(&path, json).map_err(|e| format!("write {}: {e}", path.display()))
}

/// Load a checkpoint *and* its optimiser sidecar (errors if the sidecar is
/// missing or does not match the checkpoint's step/shape).
pub fn load_vanilla_state_for_resume(checkpoint: &Path) -> Result<VanillaModelState, String> {
    let mut state = load_vanilla_state(checkpoint)?;
    let path = optim_sidecar_path(checkpoint);
    let raw = std::fs::read_to_string(&path).map_err(|e| {
        format!(
            "resume needs optimizer state {} ({e}); use --init-from for a weights-only start",
            path.display()
        )
    })?;
    let o: VanillaOptimState = serde_json::from_str(&raw).map_err(|e| e.to_string())?;
    state.restore_optim(o)?;
    Ok(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lm_config::TrainConfigV2;
    use crate::v2::data::TrainExample;
    use crate::v2::vanilla_train::train_step_vanilla_accum;

    fn tiny_cfg() -> TrainConfigV2 {
        let mut c = TrainConfigV2::small(24);
        c.d_model = 8;
        c.n_heads = 2;
        c.d_ff = 16;
        c.n_blocks = 1;
        c.warmup_steps = 2;
        c.total_steps = 20;
        c.lr_max = 1e-2;
        c
    }

    fn ex() -> TrainExample {
        TrainExample::lm_sequence(vec![5, 6, 7, 8, 9, 10, 11, 12])
    }

    #[test]
    fn resume_matches_uninterrupted_run() {
        let dir = std::env::temp_dir().join(format!("gf-resume-{}", std::process::id()));
        let ck = dir.join("ck.json");

        let mut a = VanillaModelState::new(tiny_cfg());
        for _ in 0..3 {
            train_step_vanilla_accum(&mut a, &[ex()]);
        }
        save_vanilla_state(&ck, &a).unwrap();
        save_vanilla_optim(&ck, &a).unwrap();
        for _ in 0..3 {
            train_step_vanilla_accum(&mut a, &[ex()]);
        }

        let mut b = load_vanilla_state_for_resume(&ck).unwrap();
        for _ in 0..3 {
            train_step_vanilla_accum(&mut b, &[ex()]);
        }
        assert_eq!(a.step, b.step);
        for (ra, rb) in a.model.embedding.iter().zip(&b.model.embedding) {
            for (x, y) in ra.iter().zip(rb) {
                assert!((x - y).abs() < 1e-5, "resumed weights diverged: {x} vs {y}");
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn legacy_checkpoint_keeps_unscaled_embeddings() {
        let dir = std::env::temp_dir().join(format!("gf-legacy-{}", std::process::id()));
        let ck = dir.join("old.json");
        let st = VanillaModelState::new(tiny_cfg());
        save_vanilla_state(&ck, &st).unwrap();
        // Simulate a checkpoint written before `embed_scale` existed.
        let mut v: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&ck).unwrap()).unwrap();
        v["cfg"].as_object_mut().unwrap().remove("embed_scale");
        std::fs::write(&ck, v.to_string()).unwrap();
        let loaded = load_vanilla_state(&ck).unwrap();
        assert_eq!(loaded.model.embed_scale, 1.0);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
