pub mod category;
pub mod compose;
pub mod curriculum;
pub mod disentanglement;
pub mod embedding;
pub mod forward;
pub mod growformer;
pub mod inference;
pub mod linear_head;
pub mod node;
pub mod pythagoras;
pub mod sentiment;
pub mod training;

pub use category::{
    CategoricalDAG, Composed, Layer, MorphismKind, NaturalTransform, Network, NodeId,
};
pub use compose::{CategoricalComposer, CategoricalDecomposition, ComposedOutput, ProgramTemplate};
pub use curriculum::{CurriculumScheduler, Stage};
pub use disentanglement::{DisentanglementLoss, DisentanglementWeights, LossBreakdown, SimpleRng};
pub use embedding::{CharHashEmbedder, SentenceEmbedder, TokenHashEmbedder};
pub use forward::{bifunctor_branch_vectors, char_hash_embed, record_embedding};
pub use growformer::{GrowformerNode, GrowformerTrainer, TrainerConfig};
pub use inference::{infer_from_embedding, InferenceDetail, InferenceResult};
pub use linear_head::LinearHead;
pub use node::CategoricalNode;
pub use pythagoras::PythagorasNode;
pub use sentiment::{ParsedInput, SentimentFunctor};
pub use training::{
    infer_plural_from_text, semantic_intent_to_label, AuxCategory, SentimentJsonlSelection,
    SentimentLabel, TrainingBatch, TrainingRecord,
};
