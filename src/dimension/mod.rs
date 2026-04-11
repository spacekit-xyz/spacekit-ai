//! Fractal Topology — Main Dimension, Mirror Dimension, Promotion Gate, GlobalObserver.
//! Phase 3: isolated env per task; no shared substrate.

pub mod composition;
pub mod action;
pub mod action_classifier;
pub mod codegen;
pub mod generation;
pub mod generation_head;
pub mod group_gen;
pub mod embedding;
pub mod language;
pub mod main_dim;
pub mod mirror_dim;
pub mod policy;
pub mod promotion;
pub mod router;
pub mod observer;
pub mod manager;
pub mod tool;
pub mod paramecium;

pub use composition::{EpisodicMemory, Episode, VirtualGroup};
pub use action::{ActionJson, ActionType, action_from_routing};
pub use codegen::{CodeGeneration, generate_code_from_action};
pub use generation::{GeneratedResponse, render_action_template};
pub use embedding::{GroupEmbedding, compute_group_embedding, cosine_similarity, retrieve_relevant_groups};
pub use language::{
    causal_index_token, CalibrationCoverage, CalibrationDataset, CalibrationRequirements, CalibrationReport,
    CausalAnnotation, EncoderPreset, EmaSmoother, GroupAdapter, HashingLanguageEncoder, LanguageBridge,
    LanguageConfig, LanguageEncoder, LanguageRoutingDecision, LanguageRuntime, LanguageSample,
    route_language_embedding, sentiment_lattice_index_body_with_causal, SENTIMENT_CAUSAL_INDEX_CORE,
};
pub use crate::clifford::GroupRotor;
pub use main_dim::{MainDimension, FrozenGroupEnv};
pub use mirror_dim::{MirrorDimension, EpochResult};
pub use promotion::{PromotionGateConfig, PromotionDecision, evaluate_promotion, promote};
pub use policy::{ContinuousPolicy, Policy};
pub use router::{LearnedRouter, attend_by_query};
pub use observer::{GlobalObserver, RoutingConfig};
pub use action_classifier::{ActionClassifier, action_target_to_type, action_type_one_hot, group_id_one_hot};
pub use generation_head::GenerationHead;
pub use group_gen::GroupGenEnv;
pub use group_gen::IndexedGenEnv;
pub use manager::{DimensionManager, DimensionManagerConfig, GroupSummary, MirrorSummary, EpisodicSummary, CheckpointSizeSummary};
pub use tool::{ToolSchema, ToolParam, ParamType, ToolRegistry, ToolCallInfo, ToolResult};
