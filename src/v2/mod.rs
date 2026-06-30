//! **v2** — taped forward, full Clifford-attention backward, LM `train_v2`, sparse embeddings, sampling.
//!
//! Read order: [`tape`] → [`attention_backward`] → [`block_backward`] → [`embedding`] → [`sample`] → [`train_v2`].

pub mod arithmetic;
pub mod checkpoint;
pub mod data;
pub mod tape;
pub mod attention_backward;
pub mod block_backward;
pub mod embedding;
pub mod inference;
pub mod sample;
pub mod train_v2;

pub use arithmetic::{
    cumulative, find_symbol, quantize, ArithmeticDecoder, ArithmeticEncoder, FREQ_BITS, FREQ_TOTAL,
};

pub use attention_backward::{attention_backward, AttentionGrads};
pub use block_backward::{block_backward, BlockGrads};
pub use checkpoint::{load_lm_checkpoint, save_lm_checkpoint};
pub use data::{encode_record, Dataset, RawRecord, Tokenizer, TrainExample};
pub use embedding::{EmbeddingGrad, EmbeddingOptimizer};
pub use sample::{
    apply_repetition_penalty, apply_temperature, apply_top_k, apply_top_p, argmax, generate_stream,
    multinomial, sample_next, softmax as sample_softmax, SampleConfig, SimpleRng, TokenCallback,
};
pub use inference::InferenceCache;
pub use tape::{
    attention_forward_taped, block_forward_taped, ffn_forward_taped, model_forward_logits,
    model_forward_taped, norm_forward_taped, AttentionTape, BlockTape, FfnTape, LayerNormStats,
    Tape,
};
pub use train_v2::{
    corpus_semantic_init, structured_embedding_init, train_step_v2, train_step_v2_accum,
    train_step_v2_head_only, train_v2, BlockOptimizer, ModelStateV2, TrainConfigV2,
};
