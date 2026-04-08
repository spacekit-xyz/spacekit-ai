//! Optional inference-time augmentations (data-driven heuristics, brain-profile hooks).
//! Keeps domain policy out of the core generation loop in `service.rs`.
//!
//! Runtime packaging: [`crate::brain::BrainPackage`] may carry a UTF-8 TOML
//! [`manifest::BrainPluginsManifest`] in `plugins_blob`, parsed on [`crate::service::LanguageService::load_brain`].
//!
//! Plugin logic lives under [`plugins`] and is driven by [`harness::InferenceHarness`].

pub mod harness;
pub mod manifest;
pub mod plugins;

pub use harness::{
    BrainInferencePlugin, GenerationPreemptOutcome, InferenceHarness, TemplatePostprocessFlags,
};
pub use manifest::{
    parse_plugins_manifest_bytes, serialize_plugins_manifest, sentiment_inference_config_disk,
    BrainPluginsManifest, SentimentInferenceConfig,
};
pub use plugins::{
    default_inference_harness, SentimentLatticePlugin, TEMPLATE_ID_USER_ANCHORED, TOPIC_KEYS,
};
