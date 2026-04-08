//! Optional inference-time augmentations (data-driven heuristics, brain-profile hooks).
//! Keeps domain policy out of the core generation loop in `service.rs`.
//!
//! Runtime packaging: [`crate::brain::BrainPackage`] may carry a UTF-8 TOML
//! [`manifest::BrainPluginsManifest`] in `plugins_blob`, parsed on [`crate::service::LanguageService::load_brain`].
//!
//! Plugin logic lives under [`plugins`] and is driven by [`harness::InferenceHarness`].

pub mod harness;
pub mod inference_toml;
pub mod manifest;
pub mod plugins;

pub use harness::{
    BrainInferencePlugin, GenerationPreemptOutcome, InferenceHarness, TemplatePostprocessFlags,
};
pub use inference_toml::{
    inference_rules_runtime, inference_toml_loaded, set_inference_toml_cli_paths, LoadedInferenceToml,
};
pub use manifest::{
    inference_thresholds_disk, parse_plugins_manifest_bytes, serialize_plugins_manifest,
    BrainPluginsManifest, InferenceThresholds,
};
pub use plugins::{
    default_inference_harness, LatticeShortcutsPlugin, TEMPLATE_ID_USER_ANCHORED, TOPIC_KEYS,
};
