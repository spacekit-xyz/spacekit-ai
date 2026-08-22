//! Built-in [`super::harness::BrainInferencePlugin`] implementations.

pub mod chat_policy;
pub mod lattice_shortcuts;

pub use chat_policy::ChatPolicyPlugin;
pub use lattice_shortcuts::{
    format_retrieved_sentiment_line, LatticeShortcutsPlugin, TEMPLATE_ID_USER_ANCHORED, TOPIC_KEYS,
};

use super::harness::InferenceHarness;

/// Default registry; append more `Box<dyn BrainInferencePlugin>` as needed.
pub fn default_inference_harness() -> InferenceHarness {
    InferenceHarness::new(vec![
        Box::new(LatticeShortcutsPlugin),
        Box::new(ChatPolicyPlugin),
    ])
}
