//! Serialize / deserialize full LM weights + tokenizer + [`TrainConfigV2`] for CLI and resumes.

use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Write;
use std::path::Path;

use crate::{CliffordBlock, CliffordLinear, FfnVariant, LinearReal};

use super::data::Tokenizer;
use super::train_v2::{ModelStateV2, TrainConfigV2};

// Schema 2: output head is a real-valued projection (LinearReal) over the
// flattened 16·d_model residual stream rather than a grade-0 CliffordLinear.
const LM_CHECKPOINT_SCHEMA: u32 = 3;

#[derive(Debug, Serialize, Deserialize)]
pub struct LinearDto {
    pub weights: Vec<Vec<[f32; 16]>>,
    pub bias: Vec<[f32; 16]>,
}

/// Real-valued output head: `weights[vocab][16·d_model]`, `bias[vocab]`.
#[derive(Debug, Serialize, Deserialize)]
pub struct RealHeadDto {
    pub weights: Vec<Vec<f32>>,
    pub bias: Vec<f32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BlockDto {
    pub norm1_gamma: Vec<f32>,
    pub norm1_beta: Vec<f32>,
    pub wq: LinearDto,
    pub wk: LinearDto,
    pub wv: LinearDto,
    pub wo: LinearDto,
    pub norm2_gamma: Vec<f32>,
    pub norm2_beta: Vec<f32>,
    pub fc1: LinearDto,
    pub fc2: LinearDto,
    /// Dense FFN ablation weights (schema ≥ 3).  When present, `cfg.dense_ffn` must be true.
    #[serde(default)]
    pub dense_fc1: Option<RealHeadDto>,
    #[serde(default)]
    pub dense_fc2: Option<RealHeadDto>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LmCheckpoint {
    pub schema: u32,
    pub step: u64,
    pub tokenizer: Vec<String>,
    pub cfg: TrainConfigV2,
    pub embedding: Vec<Vec<[f32; 16]>>,
    pub blocks: Vec<BlockDto>,
    #[serde(default)]
    pub final_norm_gamma: Vec<f32>,
    #[serde(default)]
    pub final_norm_beta: Vec<f32>,
    pub head: RealHeadDto,
}

fn snap_linear(layer: &CliffordLinear) -> LinearDto {
    LinearDto {
        weights: layer
            .weights
            .iter()
            .map(|row| row.iter().map(|m| m.c).collect())
            .collect(),
        bias: layer.bias.iter().map(|m| m.c).collect(),
    }
}

fn apply_linear(layer: &mut CliffordLinear, d: &LinearDto) -> Result<(), String> {
    if layer.weights.len() != d.weights.len() || layer.bias.len() != d.bias.len() {
        return Err("linear shape mismatch".into());
    }
    for i in 0..layer.out_dim {
        if layer.weights[i].len() != d.weights[i].len() {
            return Err(format!("linear weight row {i} width mismatch"));
        }
        for j in 0..layer.in_dim {
            layer.weights[i][j].c = d.weights[i][j];
        }
        layer.bias[i].c = d.bias[i];
    }
    Ok(())
}

fn snap_real_head(h: &LinearReal) -> RealHeadDto {
    RealHeadDto {
        weights: h.weights.clone(),
        bias: h.bias.clone(),
    }
}

fn apply_real_head(h: &mut LinearReal, d: &RealHeadDto) -> Result<(), String> {
    if h.weights.len() != d.weights.len() || h.bias.len() != d.bias.len() {
        return Err("head shape mismatch".into());
    }
    for o in 0..h.out_dim {
        if h.weights[o].len() != d.weights[o].len() {
            return Err(format!("head weight row {o} width mismatch"));
        }
        h.weights[o].clone_from(&d.weights[o]);
    }
    h.bias.clone_from(&d.bias);
    Ok(())
}

fn snap_block(block: &CliffordBlock) -> BlockDto {
    let (fc1, fc2, dense_fc1, dense_fc2) = match &block.ffn {
        FfnVariant::Clifford(f) => (snap_linear(&f.fc1), snap_linear(&f.fc2), None, None),
        FfnVariant::Dense(f) => (
            LinearDto {
                weights: vec![],
                bias: vec![],
            },
            LinearDto {
                weights: vec![],
                bias: vec![],
            },
            Some(snap_real_head(&f.fc1)),
            Some(snap_real_head(&f.fc2)),
        ),
    };
    BlockDto {
        norm1_gamma: block.norm1.gamma.clone(),
        norm1_beta: block.norm1.beta.clone(),
        wq: snap_linear(&block.attn.w_q),
        wk: snap_linear(&block.attn.w_k),
        wv: snap_linear(&block.attn.w_v),
        wo: snap_linear(&block.attn.w_o),
        norm2_gamma: block.norm2.gamma.clone(),
        norm2_beta: block.norm2.beta.clone(),
        fc1,
        fc2,
        dense_fc1,
        dense_fc2,
    }
}

fn apply_block(block: &mut CliffordBlock, d: &BlockDto, dense_ffn: bool) -> Result<(), String> {
    if block.norm1.gamma.len() != d.norm1_gamma.len() {
        return Err("norm1_gamma len mismatch".into());
    }
    block.norm1.gamma.clone_from(&d.norm1_gamma);
    block.norm1.beta.clone_from(&d.norm1_beta);
    apply_linear(&mut block.attn.w_q, &d.wq)?;
    apply_linear(&mut block.attn.w_k, &d.wk)?;
    apply_linear(&mut block.attn.w_v, &d.wv)?;
    apply_linear(&mut block.attn.w_o, &d.wo)?;
    block.norm2.gamma.clone_from(&d.norm2_gamma);
    block.norm2.beta.clone_from(&d.norm2_beta);

    if dense_ffn {
        let d1 = d
            .dense_fc1
            .as_ref()
            .ok_or("dense_ffn checkpoint missing dense_fc1")?;
        let d2 = d
            .dense_fc2
            .as_ref()
            .ok_or("dense_ffn checkpoint missing dense_fc2")?;
        match &mut block.ffn {
            FfnVariant::Dense(f) => {
                apply_real_head(&mut f.fc1, d1)?;
                apply_real_head(&mut f.fc2, d2)?;
            }
            FfnVariant::Clifford(_) => {
                return Err("cfg.dense_ffn but block has Clifford FFN".into());
            }
        }
    } else {
        match &mut block.ffn {
            FfnVariant::Clifford(f) => {
                apply_linear(&mut f.fc1, &d.fc1)?;
                apply_linear(&mut f.fc2, &d.fc2)?;
            }
            FfnVariant::Dense(_) => {
                return Err("cfg expects Clifford FFN but block is dense".into());
            }
        }
    }
    Ok(())
}

fn write_ckpt_json(path: &Path, ckpt: &LmCheckpoint) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let json = serde_json::to_string_pretty(ckpt).map_err(|e| e.to_string())?;
    let mut f = File::create(path).map_err(|e| e.to_string())?;
    f.write_all(json.as_bytes()).map_err(|e| e.to_string())?;
    Ok(())
}

fn build_state_from_ckpt(ckpt: LmCheckpoint) -> Result<ModelStateV2, String> {
    let LmCheckpoint {
        schema,
        step,
        tokenizer: _,
        cfg,
        embedding,
        blocks,
        final_norm_gamma,
        final_norm_beta,
        head,
    } = ckpt;

    if schema != LM_CHECKPOINT_SCHEMA && schema != 2 {
        return Err(format!("unsupported LM checkpoint schema {}", schema));
    }

    let dense_ffn = cfg.dense_ffn;
    let mut state = ModelStateV2::new(cfg);
    state.step = step;

    if embedding.len() != state.model.embedding.len() {
        return Err("embedding vocab mismatch".into());
    }
    for (i, row) in embedding.into_iter().enumerate() {
        if row.len() != state.model.embedding[i].len() {
            return Err(format!("embedding row {i} d_model mismatch"));
        }
        for (d, arr) in row.into_iter().enumerate() {
            state.model.embedding[i][d].c = arr;
        }
    }

    if blocks.len() != state.model.blocks.len() {
        return Err("n_blocks mismatch".into());
    }
    for (b, dto) in blocks.into_iter().enumerate() {
        apply_block(&mut state.model.blocks[b], &dto, dense_ffn)?;
    }

    if !final_norm_gamma.is_empty() {
        if final_norm_gamma.len() != state.model.final_norm.gamma.len() {
            return Err("final_norm_gamma len mismatch".into());
        }
        state.model.final_norm.gamma = final_norm_gamma;
        state.model.final_norm.beta = final_norm_beta;
    }

    apply_real_head(&mut state.model.head, &head)?;
    // Weight tying: reconstruct the head mirror from the (authoritative) embedding
    // so inference reads a consistent matrix regardless of what was serialized.
    if state.cfg.tie_embeddings {
        state.model.sync_tied_head();
    }
    Ok(state)
}

/// Write weights, config, optimizer step counter, and tokenizer vocabulary.
pub fn save_lm_checkpoint(
    path: &Path,
    state: &ModelStateV2,
    tokenizer: &Tokenizer,
) -> Result<(), String> {
    if tokenizer.vocab_size() != state.cfg.vocab_size {
        return Err(format!(
            "tokenizer vocab {} != cfg.vocab_size {}",
            tokenizer.vocab_size(),
            state.cfg.vocab_size
        ));
    }

    let ckpt = LmCheckpoint {
        schema: LM_CHECKPOINT_SCHEMA,
        step: state.step,
        tokenizer: tokenizer.id_to_word.clone(),
        cfg: state.cfg.clone(),
        embedding: state
            .model
            .embedding
            .iter()
            .map(|row| row.iter().map(|m| m.c).collect())
            .collect(),
        blocks: state.model.blocks.iter().map(snap_block).collect(),
        final_norm_gamma: state.model.final_norm.gamma.clone(),
        final_norm_beta: state.model.final_norm.beta.clone(),
        head: snap_real_head(&state.model.head),
    };

    write_ckpt_json(path, &ckpt)
}

/// TinyStories / BPE: weights only — keep the `.tok` file next to this JSON.
pub fn save_lm_state(path: &Path, state: &ModelStateV2) -> Result<(), String> {
    let ckpt = LmCheckpoint {
        schema: LM_CHECKPOINT_SCHEMA,
        step: state.step,
        tokenizer: Vec::new(),
        cfg: state.cfg.clone(),
        embedding: state
            .model
            .embedding
            .iter()
            .map(|row| row.iter().map(|m| m.c).collect())
            .collect(),
        blocks: state.model.blocks.iter().map(snap_block).collect(),
        final_norm_gamma: state.model.final_norm.gamma.clone(),
        final_norm_beta: state.model.final_norm.beta.clone(),
        head: snap_real_head(&state.model.head),
    };
    write_ckpt_json(path, &ckpt)
}

/// Fresh Adam state; loads weights and `cfg` from disk.
pub fn load_lm_checkpoint(path: &Path) -> Result<(ModelStateV2, Tokenizer), String> {
    let raw = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let mut ckpt: LmCheckpoint = serde_json::from_str(&raw).map_err(|e| e.to_string())?;
    if ckpt.schema != LM_CHECKPOINT_SCHEMA && ckpt.schema != 2 {
        return Err(format!("unsupported LM checkpoint schema {}", ckpt.schema));
    }

    if ckpt.tokenizer.is_empty() {
        return Err(
            "checkpoint has no embedded tokenizer (TinyStories/BPE: use load_lm_state + .tok file)"
                .into(),
        );
    }

    let tokenizer = Tokenizer::from_vocab_list(ckpt.tokenizer.clone());
    if tokenizer.vocab_size() != ckpt.cfg.vocab_size {
        return Err(format!(
            "tokenizer len {} != cfg.vocab_size {}",
            tokenizer.vocab_size(),
            ckpt.cfg.vocab_size
        ));
    }

    ckpt.cfg.vocab_size = tokenizer.vocab_size();
    let state = build_state_from_ckpt(ckpt)?;
    Ok((state, tokenizer))
}

/// Load weights without word tokenizer (pair with `BpeTokenizer::load` for decode).
pub fn load_lm_state(path: &Path) -> Result<ModelStateV2, String> {
    let raw = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let ckpt: LmCheckpoint = serde_json::from_str(&raw).map_err(|e| e.to_string())?;
    build_state_from_ckpt(ckpt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v2::train_v2::TrainConfigV2;

    #[test]
    fn roundtrip_lm_checkpoint() {
        let tok = Tokenizer::new();
        let cfg = TrainConfigV2::small(tok.vocab_size());
        let state = ModelStateV2::new(cfg);
        let path = std::env::temp_dir().join("gfllm_lm_ckpt_test.json");
        save_lm_checkpoint(&path, &state, &tok).unwrap();
        let (state2, tok2) = load_lm_checkpoint(&path).unwrap();
        assert_eq!(state2.cfg.vocab_size, state.cfg.vocab_size);
        assert_eq!(state2.model.embedding.len(), state.model.embedding.len());
        assert_eq!(tok2.vocab_size(), tok.vocab_size());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn roundtrip_lm_state_no_tokenizer() {
        let tok = Tokenizer::new();
        let cfg = TrainConfigV2::small(tok.vocab_size());
        let mut state = ModelStateV2::new(cfg);
        state.step = 42;
        let path = std::env::temp_dir().join("gfllm_lm_state_test.json");
        save_lm_state(&path, &state).unwrap();
        let state2 = load_lm_state(&path).unwrap();
        assert_eq!(state2.step, 42);
        assert_eq!(state2.cfg.vocab_size, state.cfg.vocab_size);
        match load_lm_checkpoint(&path) {
            Err(e) => assert!(e.contains("no embedded tokenizer"), "unexpected err: {e}"),
            Ok(_) => panic!("expected load_lm_checkpoint to reject empty tokenizer"),
        }
        let _ = std::fs::remove_file(&path);
    }
}
