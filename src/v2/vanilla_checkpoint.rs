//! Checkpoint save/load for row-2 vanilla transformer (schema 1).

use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Write;
use std::path::Path;

use crate::vanilla_llm::{VanillaBlock, VanillaLLM};
use crate::LinearReal;

use super::train_v2::TrainConfigV2;
use super::vanilla_train::VanillaModelState;

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
        return Err(format!("unsupported vanilla checkpoint schema {}", ckpt.schema));
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
