//! **v2** — LM train / sample / checkpoint pipeline.
//!
//! Vanilla path is always available. Clifford tape/train modules require
//! `feature = "clifford-lm"`.

pub mod arithmetic;
pub mod data;
pub mod sample;
pub mod vanilla_checkpoint;
pub mod vanilla_train;

#[cfg(feature = "clifford-lm")]
pub mod attention_backward;
#[cfg(feature = "clifford-lm")]
pub mod block_backward;
#[cfg(feature = "clifford-lm")]
pub mod checkpoint;
#[cfg(feature = "clifford-lm")]
pub mod embedding;
#[cfg(feature = "clifford-lm")]
pub mod inference;
#[cfg(feature = "clifford-lm")]
pub mod tape;
#[cfg(feature = "clifford-lm")]
pub mod train_v2;

pub use crate::lm_config::TrainConfigV2;
pub use crate::vanilla_llm::vanilla_forward_logits;
pub use arithmetic::{
    cumulative, find_symbol, quantize, ArithmeticDecoder, ArithmeticEncoder, FREQ_BITS, FREQ_TOTAL,
};
pub use data::{encode_record, Dataset, RawRecord, Tokenizer, TrainExample};
pub use sample::{
    apply_repetition_penalty, apply_temperature, apply_top_k, apply_top_p, argmax, generate_stream,
    multinomial, sample_next, softmax as sample_softmax, SampleConfig, SimpleRng, TokenCallback,
};
pub use vanilla_checkpoint::{load_vanilla_state, save_vanilla_state};
pub use vanilla_train::{
    corpus_semantic_init_vanilla, eval_vanilla_lm_loss, randomize_vanilla_model,
    train_step_vanilla_accum, VanillaModelState,
};

#[cfg(feature = "clifford-lm")]
pub use attention_backward::{attention_backward, AttentionGrads};
#[cfg(feature = "clifford-lm")]
pub use block_backward::{block_backward, BlockGrads, FfnGrad};
#[cfg(feature = "clifford-lm")]
pub use checkpoint::{load_lm_checkpoint, load_lm_state, save_lm_checkpoint, save_lm_state};
#[cfg(feature = "clifford-lm")]
pub use embedding::{EmbeddingGrad, EmbeddingOptimizer};
#[cfg(feature = "clifford-lm")]
pub use inference::InferenceCache;
#[cfg(feature = "clifford-lm")]
pub use tape::{
    attention_forward_taped, block_forward_taped, ffn_forward_taped, model_forward_logits,
    model_forward_taped, norm_forward_taped, AttentionTape, BlockTape, FfnTape, LayerNormStats,
    Tape,
};
#[cfg(feature = "clifford-lm")]
pub use train_v2::{
    corpus_semantic_init, structured_embedding_init, train_step_v2, train_step_v2_accum,
    train_step_v2_head_only, train_v2, BlockOptimizer, ModelStateV2,
};
