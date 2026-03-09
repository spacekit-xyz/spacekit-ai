//! Fractal Topology — Main Dimension, Mirror Dimension, Promotion Gate, GlobalObserver.
//! Phase 3: isolated env per task; no shared substrate.

pub mod composition;
pub mod embedding;
pub mod language;
pub mod main_dim;
pub mod mirror_dim;
pub mod policy;
pub mod promotion;
pub mod router;
pub mod observer;
pub mod manager;

pub use composition::{EpisodicMemory, Episode, VirtualGroup};
pub use embedding::{GroupEmbedding, compute_group_embedding, cosine_similarity, retrieve_relevant_groups};
pub use language::{
    CalibrationCoverage, CalibrationDataset, CalibrationRequirements, CalibrationReport,
    EncoderPreset, EmaSmoother, HashingLanguageEncoder, LanguageBridge, LanguageConfig,
    LanguageRoutingDecision, LanguageRuntime, LanguageSample, route_language_embedding,
};
pub use main_dim::{MainDimension, FrozenGroupEnv};
pub use mirror_dim::{MirrorDimension, EpochResult};
pub use promotion::{PromotionGateConfig, PromotionDecision, evaluate_promotion, promote};
pub use policy::{ContinuousPolicy, Policy};
pub use router::{LearnedRouter, attend_by_query};
pub use observer::{GlobalObserver, RoutingConfig};
pub use manager::{DimensionManager, DimensionManagerConfig, GroupSummary, MirrorSummary};
