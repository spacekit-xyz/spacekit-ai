//! Train / infer pooled Clifford classifier on JSONL + optional world grounding.

use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::domain_data::load_jsonl_dir;
use crate::optim::{clip_grad_norm, AdamConfig, LayerOptimizer, MvAdamState, adam_step};
use crate::pooled_classifier::PooledClassifier;
use crate::world_grounding::{GroundingExtractor, GROUND_FEATURE_DIM};
use crate::{CliffordAlgebra, Multivector};

const BYTE_VOCAB: usize = 256;
const CHECKPOINT_SCHEMA: u32 = 1;

#[derive(Debug, Clone)]
pub struct TrainConfig {
    pub data_dir: PathBuf,
    pub task_name: String,
    pub epochs: usize,
    pub max_seq_len: usize,
    pub d_model: usize,
    pub checkpoint_path: PathBuf,
    pub grounding_paths: Vec<PathBuf>,
    pub lr: f32,
}

#[derive(Serialize, Deserialize)]
struct Checkpoint {
    schema: u32,
    task: String,
    labels: Vec<String>,
    d_model: usize,
    ground_dim: usize,
    embedding: Vec<Vec<[f32; 16]>>,
    head_weights: Vec<Vec<[f32; 16]>>,
    head_bias: Vec<[f32; 16]>,
    ground_w: Vec<Vec<f32>>,
    ground_b: Vec<f32>,
}

fn mv_to_arr(m: &Multivector) -> [f32; 16] {
    m.c
}

fn arr_to_mv(a: &[f32; 16]) -> Multivector {
    Multivector { c: *a }
}

fn bytes_to_ids(text: &str, max_len: usize) -> Vec<usize> {
    let bs: Vec<u8> = text.as_bytes().iter().copied().take(max_len).collect();
    if bs.is_empty() {
        vec![b' ' as usize]
    } else {
        bs.into_iter().map(|b| b as usize).collect()
    }
}

fn label_index(labels: &[String], label: &str) -> Option<usize> {
    labels.iter().position(|s| s == label)
}

fn adam_f32(
    param: f32,
    grad: f32,
    m: &mut f32,
    v: &mut f32,
    step: u64,
    cfg: &AdamConfig,
) -> f32 {
    let t = step as f32;
    let b1 = cfg.beta1;
    let b2 = cfg.beta2;
    let bc1 = 1.0 - b1.powf(t);
    let bc2 = 1.0 - b2.powf(t);
    *m = b1 * *m + (1.0 - b1) * grad;
    *v = b2 * *v + (1.0 - b2) * grad * grad;
    let m_hat = *m / bc1;
    let v_hat = *v / bc2;
    param - cfg.lr * m_hat / (v_hat.sqrt() + cfg.eps)
}

pub fn train_classifier(cfg: TrainConfig) -> Result<(), String> {
    let (examples, labels) = load_jsonl_dir(&cfg.data_dir)?;
    let n_classes = labels.len();
    if n_classes < 2 {
        return Err("need at least two distinct labels".into());
    }

    let alg = Arc::new(CliffordAlgebra::sta());
    let grounding = if cfg.grounding_paths.is_empty() {
        None
    } else {
        Some(GroundingExtractor::from_toml_files(&cfg.grounding_paths)?)
    };
    let ground_dim = grounding.as_ref().map(|x| x.dim).unwrap_or(GROUND_FEATURE_DIM);
    let zero_g = GroundingExtractor::zero_features(ground_dim);

    let mut model = PooledClassifier::new(
        alg.clone(),
        BYTE_VOCAB,
        cfg.d_model,
        n_classes,
        ground_dim,
    );

    let adam_cfg = AdamConfig {
        lr: cfg.lr,
        ..Default::default()
    };
    let mut head_opt = LayerOptimizer::new(n_classes, cfg.d_model, adam_cfg.clone());
    let mut emb_state: Vec<Vec<MvAdamState>> =
        vec![vec![MvAdamState::zero(); cfg.d_model]; BYTE_VOCAB];
    let mut gw_state: Vec<Vec<(f32, f32, u64)>> = (0..cfg.d_model)
        .map(|_| vec![(0.0f32, 0.0f32, 0u64); ground_dim])
        .collect();
    let mut gb_state: Vec<(f32, f32, u64)> = vec![(0.0f32, 0.0f32, 0u64); cfg.d_model];

    let mut rng_state = 0u64;
    for ep in 0..cfg.epochs {
        let mut order: Vec<usize> = (0..examples.len()).collect();
        // deterministic shuffle
        for i in (1..order.len()).rev() {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let j = (rng_state % (i as u64 + 1)) as usize;
            order.swap(i, j);
        }

        let mut loss_acc = 0.0f32;
        for &ix in &order {
            let ex = &examples[ix];
            let Some(y) = label_index(&labels, &ex.label) else {
                continue;
            };
            let ids = bytes_to_ids(&ex.text, cfg.max_seq_len);
            let g_feat = grounding
                .as_ref()
                .map(|gg| gg.features(&ex.text))
                .unwrap_or_else(|| zero_g.clone());

            let (loss, grad_head, grad_emb, grad_gw, grad_gb) =
                model.backward_one(&ids, &g_feat, y);
            loss_acc += loss;

            let mut gh = grad_head;
            clip_grad_norm(&mut gh, 2.0);
            head_opt.step(&mut model.head.weights, &mut model.head.bias, &gh);

            for d in 0..cfg.d_model {
                let (ref mut m, ref mut v, ref mut st) = gb_state[d];
                *st += 1;
                model.ground_b[d] = adam_f32(model.ground_b[d], grad_gb[d], m, v, *st, &adam_cfg);
            }
            for d in 0..cfg.d_model {
                for j in 0..ground_dim {
                    let (ref mut m, ref mut v, ref mut st) = gw_state[d][j];
                    *st += 1;
                    model.ground_w[d][j] =
                        adam_f32(model.ground_w[d][j], grad_gw[d][j], m, v, *st, &adam_cfg);
                }
            }

            for &tid in &ids {
                for d in 0..cfg.d_model {
                    model.embedding[tid][d] = adam_step(
                        &model.embedding[tid][d],
                        &grad_emb[tid][d],
                        &mut emb_state[tid][d],
                        &adam_cfg,
                    );
                }
            }
        }

        let n = examples.len().max(1) as f32;
        println!(
            "epoch {:>4}  loss {:.4}",
            ep + 1,
            loss_acc / n
        );
    }

    save_checkpoint(
        &cfg.checkpoint_path,
        &Checkpoint {
            schema: CHECKPOINT_SCHEMA,
            task: cfg.task_name.clone(),
            labels: labels.clone(),
            d_model: cfg.d_model,
            ground_dim,
            embedding: model
                .embedding
                .iter()
                .map(|row| row.iter().map(mv_to_arr).collect())
                .collect(),
            head_weights: model
                .head
                .weights
                .iter()
                .map(|row| row.iter().map(mv_to_arr).collect())
                .collect(),
            head_bias: model.head.bias.iter().map(mv_to_arr).collect(),
            ground_w: model.ground_w.clone(),
            ground_b: model.ground_b.clone(),
        },
    )?;
    println!("wrote {}", cfg.checkpoint_path.display());
    Ok(())
}

fn save_checkpoint(path: &Path, ckpt: &Checkpoint) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("create_dir_all: {}", e))?;
    }
    let json = serde_json::to_string_pretty(ckpt).map_err(|e| e.to_string())?;
    let mut f = File::create(path).map_err(|e| e.to_string())?;
    f.write_all(json.as_bytes()).map_err(|e| e.to_string())?;
    Ok(())
}

pub fn load_checkpoint(path: &Path) -> Result<(PooledClassifier, Vec<String>), String> {
    let raw = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    load_checkpoint_bytes(&raw)
}

fn load_checkpoint_bytes(raw: &str) -> Result<(PooledClassifier, Vec<String>), String> {
    let ckpt: Checkpoint = serde_json::from_str(raw).map_err(|e| e.to_string())?;
    if ckpt.schema != CHECKPOINT_SCHEMA {
        return Err(format!("unsupported checkpoint schema {}", ckpt.schema));
    }
    let labels = ckpt.labels.clone();
    let alg = Arc::new(CliffordAlgebra::sta());
    let n_classes = ckpt.labels.len();
    let mut model = PooledClassifier::new(
        alg,
        BYTE_VOCAB,
        ckpt.d_model,
        n_classes,
        ckpt.ground_dim,
    );

    for (i, row) in ckpt.embedding.into_iter().enumerate() {
        for (d, arr) in row.into_iter().enumerate() {
            model.embedding[i][d] = arr_to_mv(&arr);
        }
    }
    for (d, row) in ckpt.head_weights.into_iter().enumerate() {
        for (i, arr) in row.into_iter().enumerate() {
            model.head.weights[d][i] = arr_to_mv(&arr);
        }
    }
    for (d, arr) in ckpt.head_bias.into_iter().enumerate() {
        model.head.bias[d] = arr_to_mv(&arr);
    }
    model.ground_w = ckpt.ground_w;
    model.ground_b = ckpt.ground_b;

    Ok((model, labels))
}

pub struct InferPack {
    pub model: PooledClassifier,
    pub labels: Vec<String>,
    pub grounding: Option<GroundingExtractor>,
}

pub fn load_infer_pack(path: &Path, grounding_paths: &[PathBuf]) -> Result<InferPack, String> {
    let (model, labels) = load_checkpoint(path)?;
    let grounding = if grounding_paths.is_empty() {
        None
    } else {
        Some(GroundingExtractor::from_toml_files(grounding_paths)?)
    };
    Ok(InferPack {
        model,
        labels,
        grounding,
    })
}

pub fn infer_one(pack: &InferPack, text: &str, max_seq_len: usize) -> Result<String, String> {
    let ids = bytes_to_ids(text, max_seq_len);
    let ground_dim = pack.model.ground_dim;
    let z = GroundingExtractor::zero_features(ground_dim);
    let g = pack
        .grounding
        .as_ref()
        .map(|gg| gg.features(text))
        .unwrap_or(z);
    let (logits, _) = pack.model.forward_logits(&ids, &g);
    let (best, _) = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .ok_or_else(|| "empty logits".to_string())?;
    pack.labels
        .get(best)
        .cloned()
        .ok_or_else(|| "label index out of range".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overfit_three_examples() {
        let alg = Arc::new(CliffordAlgebra::sta());
        let mut m = PooledClassifier::new(alg, BYTE_VOCAB, 8, 2, GROUND_FEATURE_DIM);
        let g = GroundingExtractor::zero_features(GROUND_FEATURE_DIM);
        let ex = [
            ("hello pos", 0usize),
            ("good vibes", 0usize),
            ("terrible bad", 1usize),
        ];
        let adam_cfg = AdamConfig { lr: 0.05, ..Default::default() };
        let mut head_opt = LayerOptimizer::new(2, 8, adam_cfg.clone());
        let mut emb_state = vec![vec![MvAdamState::zero(); 8]; BYTE_VOCAB];
        let mut gw_state: Vec<Vec<(f32, f32, u64)>> = vec![vec![(0.0, 0.0, 0); GROUND_FEATURE_DIM]; 8];
        let mut gb_state: Vec<(f32, f32, u64)> = vec![(0.0, 0.0, 0); 8];

        for _ in 0..400 {
            for &(t, y) in &ex {
                let ids = bytes_to_ids(t, 64);
                let (loss, gh, ge, gg_w, gg_b) = m.backward_one(&ids, &g, y);
                assert!(loss.is_finite(), "loss={}", loss);
                let mut gh2 = gh;
                clip_grad_norm(&mut gh2, 5.0);
                head_opt.step(&mut m.head.weights, &mut m.head.bias, &gh2);
                for d in 0..8 {
                    let (ref mut mm, ref mut vv, ref mut st) = gb_state[d];
                    *st += 1;
                    m.ground_b[d] = adam_f32(m.ground_b[d], gg_b[d], mm, vv, *st, &adam_cfg);
                }
                for d in 0..8 {
                    for j in 0..GROUND_FEATURE_DIM {
                        let (ref mut mm, ref mut vv, ref mut st) = gw_state[d][j];
                        *st += 1;
                        m.ground_w[d][j] =
                            adam_f32(m.ground_w[d][j], gg_w[d][j], mm, vv, *st, &adam_cfg);
                    }
                }
                for &tid in &ids {
                    for d in 0..8 {
                        m.embedding[tid][d] = adam_step(
                            &m.embedding[tid][d],
                            &ge[tid][d],
                            &mut emb_state[tid][d],
                            &adam_cfg,
                        );
                    }
                }
            }
        }

        let (l1, _) = m.forward_logits(&bytes_to_ids(ex[0].0, 64), &g);
        let (l2, _) = m.forward_logits(&bytes_to_ids(ex[2].0, 64), &g);
        assert!(l1[0] > l1[1]);
        assert!(l2[1] > l2[0]);
    }
}
