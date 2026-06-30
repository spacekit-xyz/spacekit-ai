// lib.rs — Clifford LLM crate root
//
// Re-exports all public types and functions from each module so users can
// write `use clifford_llm::*` and get everything.
//
// Module layout:
//   blade        — blade index constants, grade utilities, display
//   cayley_const — compile-time Cayley table and CliffordAlgebraConst
//   backprop     — gradient types and backward pass
//   optim        — Adam optimiser and LR schedule
//   positional   — rotor-based positional encoding
//   mask         — causal and padding masks
//   kv_cache     — KV cache for autoregressive inference

// ─── Core types (from clifford_llm.rs) ───────────────────────────────────────
//
// Include the original clifford_llm.rs as a module.  If you have split it into
// a separate algebra.rs + layers.rs, adjust these lines accordingly.

mod clifford_llm;

pub mod domain_data;
pub mod pooled_classifier;
pub mod train;
pub mod world_grounding;

pub mod bpe;
pub mod tinystories;

/// Taped forward, full backward through Clifford blocks, LM training (`train_v2`), sampling.
pub mod v2;
pub use clifford_llm::{
    // Core algebra
    Multivector,
    CliffordAlgebra,
    CayleyEntry,
    // Layers
    CliffordLinear,
    CliffordLayerNorm,
    CliffordAttention,
    CliffordFFN,
    CliffordBlock,
    CliffordLLM,
    LinearReal,
};

// ─── Modules ──────────────────────────────────────────────────────────────────

pub mod blade;
pub mod cayley_const;
pub mod backprop;
pub mod optim;
pub mod positional;
pub mod mask;
pub mod kv_cache;

// ─── Convenience re-exports ───────────────────────────────────────────────────

// blade
pub use blade::{
    SCALAR, E0, E1, E01, E2, E02, E12, E012, E3, E03, E13, E013, E23, E023, E123, E0123,
    BLADE_NAMES, BLADE_GRADES, REVERSE_SIGNS,
    grade_of, blades_of_grade, project_grade, scalar_part, vector_part, bivector_part,
    vector, bivector, display,
};

// cayley_const
pub use cayley_const::{CayleyCell, CAYLEY_STA, CliffordAlgebraConst};

// backprop
pub use backprop::{
    GradLinear, RealHeadGrad,
    geo_product_backward, linear_backward,
    cross_entropy, scalar_head_backward, real_head_backward, layer_norm_backward,
};

// optim
pub use optim::{
    AdamConfig, MvAdamState, LayerOptimizer, RealHeadOptimizer,
    adam_step, cosine_lr_with_warmup, grad_norm, clip_grad_norm,
};

// positional
pub use positional::{
    PlaneKind, BivectorPlane, ALL_PLANES,
    make_rotor, apply_rotor, RotorPositionalEncoding,
};

// mask
pub use mask::{
    CausalMask,
    apply_causal_mask, causal_masked, apply_padding_mask,
    mask_scores, padding_mask_from_ids, trim_scores,
};

// kv_cache
pub use kv_cache::{LayerKVCache, KVCache, cached_attention_step};
