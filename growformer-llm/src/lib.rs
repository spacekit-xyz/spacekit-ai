//! Growformer LLM — vanilla-first language model core (Bet B).
//!
//! **Product default:** param-matched vanilla transformer.
//! Clifford Cl(1,3) LM is behind `feature = "clifford-lm"` (on by default for
//! historical checkpoints; omit for a slim product build once callers are
//! migrated).
//!
//! **Brain memory** (`feature = "brain-memory"`) is independent of LM algebra.

// ─── Vanilla-first core (always on) ───────────────────────────────────────────

pub mod bpe;
pub mod lm_config;
pub mod param_budget;
pub mod real_linear;
pub mod real_ops;
pub mod standard_layer_norm;
pub mod tinystories;
pub mod vanilla_llm;

pub use lm_config::TrainConfigV2;
pub use real_linear::LinearReal;
pub use real_ops::{
    cosine_lr_with_warmup, cross_entropy, real_linear_backward, AdamConfig, RealHeadGrad,
    RealHeadOptimizer,
};
pub use vanilla_llm::{
    add_sinusoidal_pe, vanilla_forward_logits, VanillaAttention, VanillaBlock, VanillaFFN,
    VanillaLLM,
};

pub mod v2;

#[cfg(feature = "brain-memory")]
pub mod brain_infer_config;
#[cfg(feature = "brain-memory")]
pub mod brain_memory;

// ─── Clifford research stack (`clifford-lm`) ──────────────────────────────────

#[cfg(feature = "clifford-lm")]
pub mod attention_score;
#[cfg(feature = "clifford-lm")]
pub mod backprop;
#[cfg(feature = "clifford-lm")]
pub mod blade;
#[cfg(feature = "clifford-lm")]
pub mod cayley_const;
#[cfg(feature = "clifford-lm")]
pub mod cl1;
#[cfg(feature = "clifford-lm")]
pub mod clifford_layer_norm;
#[cfg(feature = "clifford-lm")]
mod clifford_llm;
#[cfg(feature = "clifford-lm")]
pub mod ffn;
#[cfg(feature = "clifford-lm")]
pub mod kv_cache;
#[cfg(feature = "clifford-lm")]
pub mod lm_cone_router;
#[cfg(feature = "clifford-lm")]
pub mod mask;
#[cfg(feature = "clifford-lm")]
pub mod optim;
#[cfg(feature = "clifford-lm")]
pub mod positional;

#[cfg(feature = "clifford-lm")]
pub use attention_score::{attention_pair_score, AttentionScoreMode};
#[cfg(feature = "clifford-lm")]
pub use blade::{
    bivector, bivector_part, blades_of_grade, display, grade_of, project_grade, scalar_part,
    vector, vector_part, BLADE_GRADES, BLADE_METRIC_WEIGHT, BLADE_NAMES, E0, E01, E012, E0123,
    E013, E02, E023, E03, E1, E12, E123, E13, E2, E23, E3, REVERSE_SIGNS, SCALAR,
};
#[cfg(feature = "clifford-lm")]
pub use cayley_const::{CayleyCell, CliffordAlgebraConst, CAYLEY_STA};
#[cfg(feature = "clifford-lm")]
pub use clifford_llm::{
    CayleyEntry, CliffordAlgebra, CliffordAttention, CliffordBlock, CliffordFFN, CliffordLLM,
    CliffordLayerNorm, CliffordLinear, Multivector,
};
#[cfg(feature = "clifford-lm")]
pub use ffn::{
    clifford_ffn_scalars, dense_ffn_scalars, flatten_mvs, matched_dense_ffn_hidden, unflatten_mvs,
    DenseFFN, FfnVariant,
};
#[cfg(feature = "clifford-lm")]
pub use kv_cache::{cached_attention_step, KVCache, LayerKVCache};
#[cfg(feature = "clifford-lm")]
pub use mask::{
    apply_causal_mask, apply_padding_mask, causal_masked, mask_scores, padding_mask_from_ids,
    trim_scores, CausalMask,
};
#[cfg(feature = "clifford-lm")]
pub use positional::{
    apply_rotor, make_rotor, BivectorPlane, PlaneKind, RotorPositionalEncoding, ALL_PLANES,
};

pub mod chat;
pub mod domain_data;

pub use chat::{
    default_chatbot_system, role_marker_cut, ChatMessage, ChatRole, ChatTranscript, MARK_ASSISTANT,
    MARK_SYSTEM, MARK_USER,
};

// Legacy classifier (Clifford-dependent).
#[cfg(feature = "clifford-lm")]
pub mod pooled_classifier;
#[cfg(feature = "clifford-lm")]
pub mod train;
#[cfg(feature = "clifford-lm")]
pub mod world_grounding;
