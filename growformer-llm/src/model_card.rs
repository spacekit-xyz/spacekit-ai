//! Portable identity + capability descriptor for a single **domain specialist**
//! ("brain micro-model"), and a sidecar manifest so a host program can index and
//! load *many* specialists from disk.
//!
//! This layer is deliberately **arch-agnostic and always compiled** — it depends
//! only on [`TrainConfigV2`] and `serde`, never on the model weight types — so a
//! program can read cards / route over a fleet without pulling in the Clifford or
//! vanilla model code. Loading and running a specialist lives in [`crate::fleet`]
//! (feature `clifford-lm`).
//!
//! A specialist on disk is two files that live next to each other:
//!   * `something.gfcard.json`  — this manifest (identity + caps + weights ref)
//!   * the weights checkpoint    — the existing `LmCheckpoint` / `VanillaCheckpoint` JSON
//!
//! The manifest never duplicates weights; it *points* at the checkpoint. That
//! keeps existing checkpoints untouched and makes a "fleet" just a directory of
//! `*.gfcard.json` files.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::lm_config::TrainConfigV2;

/// Manifest file extension (a specialist is discovered by this suffix).
pub const CARD_EXT: &str = "gfcard.json";

/// Current manifest envelope version.
pub const CARD_FORMAT_VERSION: u32 = 1;

/// Which model stack the weights belong to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Arch {
    /// Product default: real-valued transformer (`VanillaLLM`).
    Vanilla,
    /// Research/historical Clifford Cl(1,3) LM (`CliffordLLM`).
    Clifford,
}

/// FFN shape — extends as sparse-activation lands (see `SPARSE_ACTIVATION_PLAN.md`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FfnKind {
    Dense,
    Clifford,
    /// Top-k mixture-of-experts (subtopic experts). `n_experts == 1` behaves as dense.
    Moe {
        n_experts: usize,
        top_k: usize,
        skip_expert: bool,
    },
}

/// Weight quantization of the referenced checkpoint payload.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuantKind {
    /// Full-precision f32 (today's JSON checkpoints).
    None,
    /// Per-channel int8 (reserved for Stage 4).
    Int8,
}

/// Machine-readable feature flags a host uses to decide how to load/serve.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Capabilities {
    pub tied_embeddings: bool,
    pub ffn: FfnKind,
    pub quant: QuantKind,
    /// Grounding TOML ids this specialist expects at inference (may be empty).
    #[serde(default)]
    pub grounding: Vec<String>,
    /// Per-token conditional compute / early-exit enabled (Stage 3a).
    #[serde(default)]
    pub early_exit: bool,
}

impl Capabilities {
    /// Derive capabilities from a training config (best-effort; grounding/quant
    /// default to empty/None and can be overridden by the caller).
    pub fn from_cfg(cfg: &TrainConfigV2) -> Self {
        let ffn = if cfg.vanilla {
            FfnKind::Dense
        } else if cfg.dense_ffn {
            FfnKind::Dense
        } else {
            FfnKind::Clifford
        };
        Self {
            tied_embeddings: cfg.tie_embeddings,
            ffn,
            quant: QuantKind::None,
            grounding: Vec::new(),
            early_exit: false,
        }
    }
}

/// Human/discovery metadata — what makes a specialist findable and trustworthy.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelCard {
    /// The ONE subject this micro-model understands, e.g. `"fintech-sentiment"`.
    /// Used as the fleet key and by the default router.
    pub subject: String,
    /// Free-text trigger words the default router matches against a prompt.
    #[serde(default)]
    pub keywords: Vec<String>,
    #[serde(default)]
    pub created_utc: String,
    #[serde(default)]
    pub train_steps: u64,
    /// Domain-corpus eval (bits/byte = compression rate; the repo's north-star).
    #[serde(default)]
    pub eval_bits_per_byte: Option<f32>,
    /// Base model for a fine-tuned / distilled specialist, if any.
    #[serde(default)]
    pub base_model: Option<String>,
    #[serde(default)]
    pub notes: String,
}

impl ModelCard {
    pub fn new(subject: impl Into<String>) -> Self {
        Self {
            subject: subject.into(),
            keywords: Vec::new(),
            created_utc: String::new(),
            train_steps: 0,
            eval_bits_per_byte: None,
            base_model: None,
            notes: String::new(),
        }
    }
}

/// Sidecar manifest: identity + capabilities + a relative pointer to the weights.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SpecialistManifest {
    pub format_version: u32,
    pub arch: Arch,
    pub caps: Capabilities,
    pub card: ModelCard,
    /// Weights checkpoint path, relative to the manifest's own directory.
    pub weights_path: String,
    /// Tokenizer sidecar (`.tok`) for BPE/vanilla checkpoints that don't embed
    /// a vocabulary. `None` when the checkpoint embeds its tokenizer
    /// (e.g. `LmCheckpoint` written by `save_lm_checkpoint`).
    #[serde(default)]
    pub tokenizer_path: Option<String>,
}

impl SpecialistManifest {
    /// Build a manifest for an existing checkpoint. `weights_path` is stored as
    /// given (keep it relative to where the manifest will be written).
    pub fn for_checkpoint(
        arch: Arch,
        cfg: &TrainConfigV2,
        card: ModelCard,
        weights_path: impl Into<String>,
    ) -> Self {
        Self {
            format_version: CARD_FORMAT_VERSION,
            arch,
            caps: Capabilities::from_cfg(cfg),
            card,
            weights_path: weights_path.into(),
            tokenizer_path: None,
        }
    }

    pub fn load(path: &Path) -> Result<Self, String> {
        let raw = std::fs::read_to_string(path)
            .map_err(|e| format!("read manifest {}: {}", path.display(), e))?;
        let m: SpecialistManifest = serde_json::from_str(&raw)
            .map_err(|e| format!("parse manifest {}: {}", path.display(), e))?;
        if m.format_version != CARD_FORMAT_VERSION {
            return Err(format!(
                "unsupported card format_version {} (expected {})",
                m.format_version, CARD_FORMAT_VERSION
            ));
        }
        Ok(m)
    }

    pub fn save(&self, path: &Path) -> Result<(), String> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        let json = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        std::fs::write(path, json).map_err(|e| format!("write {}: {}", path.display(), e))
    }

    /// Absolute weights path given the directory the manifest lives in.
    pub fn weights_abs(&self, manifest_dir: &Path) -> PathBuf {
        manifest_dir.join(&self.weights_path)
    }

    /// Absolute tokenizer sidecar path, if any.
    pub fn tokenizer_abs(&self, manifest_dir: &Path) -> Option<PathBuf> {
        self.tokenizer_path.as_ref().map(|p| manifest_dir.join(p))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lm_config::TrainConfigV2;

    #[test]
    fn manifest_roundtrip() {
        let cfg = TrainConfigV2::small_clifford(64);
        let mut card = ModelCard::new("fintech-sentiment");
        card.keywords = vec!["stock".into(), "earnings".into(), "market".into()];
        let m = SpecialistManifest::for_checkpoint(Arch::Clifford, &cfg, card, "weights.json");

        let path = std::env::temp_dir().join("gfllm_card_test.gfcard.json");
        m.save(&path).unwrap();
        let back = SpecialistManifest::load(&path).unwrap();
        assert_eq!(back.arch, Arch::Clifford);
        assert_eq!(back.card.subject, "fintech-sentiment");
        assert_eq!(back.weights_path, "weights.json");
        assert_eq!(back.caps.ffn, FfnKind::Clifford);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn caps_from_vanilla_cfg_is_dense() {
        let cfg = TrainConfigV2::small(64); // vanilla = true
        let caps = Capabilities::from_cfg(&cfg);
        assert_eq!(caps.ffn, FfnKind::Dense);
    }
}
