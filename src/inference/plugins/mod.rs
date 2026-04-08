//! Built-in [`super::harness::BrainInferencePlugin`] implementations.

mod sentiment_lattice;

pub use sentiment_lattice::{SentimentLatticePlugin, TEMPLATE_ID_USER_ANCHORED, TOPIC_KEYS};

use super::harness::InferenceHarness;

/// Default registry (sentiment lattice; append more `Box<dyn BrainInferencePlugin>` as needed).
pub fn default_inference_harness() -> InferenceHarness {
    InferenceHarness::new(vec![Box::new(SentimentLatticePlugin)])
}
