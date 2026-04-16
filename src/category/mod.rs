pub mod category;
pub mod pythagoras;
pub mod node;
pub mod forward;
pub mod embedding;
pub mod linear_head;
pub mod disentanglement;
pub mod curriculum;
pub mod training;
pub mod sentiment;
pub mod inference;
pub mod growformer;
pub mod compose;

pub use category::{
    CategoricalDAG, Composed, Layer, MorphismKind, NaturalTransform, Network, NodeId,
};
pub use pythagoras::PythagorasNode;
pub use node::CategoricalNode;
pub use disentanglement::{DisentanglementLoss, DisentanglementWeights, LossBreakdown, SimpleRng};
pub use curriculum::{CurriculumScheduler, Stage};
pub use training::{
    infer_plural_from_text, semantic_intent_to_label, SentimentJsonlSelection, AuxCategory,
    SentimentLabel, TrainingBatch, TrainingRecord,
};
pub use embedding::{CharHashEmbedder, SentenceEmbedder, TokenHashEmbedder};
pub use forward::{bifunctor_branch_vectors, char_hash_embed, record_embedding};
pub use growformer::{GrowformerNode, GrowformerTrainer, TrainerConfig};
pub use inference::{infer_from_embedding, InferenceDetail, InferenceResult};
pub use linear_head::LinearHead;
pub use sentiment::{ParsedInput, SentimentFunctor};
pub use compose::{CategoricalComposer, CategoricalDecomposition, ComposedOutput, ProgramTemplate};
