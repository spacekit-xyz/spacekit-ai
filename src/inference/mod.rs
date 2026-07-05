//! Optional inference-time augmentations (data-driven heuristics, brain-profile hooks).
//! Keeps domain policy out of the core generation loop in `service.rs`.
//!
//! Runtime packaging: [`crate::brain::BrainPackage`] may carry a UTF-8 TOML
//! [`manifest::BrainPluginsManifest`] in `plugins_blob`, parsed on [`crate::service::LanguageService::load_brain`].
//!
//! Plugin logic lives under [`plugins`] and is driven by [`harness::InferenceHarness`].

pub mod causal_hints;
pub mod causal_relation;
pub mod grounding_expand;
pub mod grounding_loop;
pub mod world_grounding;
pub mod harness;
pub mod inference_guardrails;
pub mod inference_toml;
pub mod manifest;
pub mod plugins;
pub mod retrieval_rescore;
pub mod retrieval_lexicon;
pub mod sentiment_generation_lexicon;
pub mod frame_lexicon;

pub use harness::{
    BrainInferencePlugin, GenerationPreemptOutcome, InferenceHarness, TemplatePostprocessFlags,
};
pub use inference_guardrails::{set_inference_guardrails_jsonl_path, GuardrailsDiskSummary};
pub use inference_toml::{
    inference_rules_runtime, inference_toml_directory, inference_toml_loaded,
    print_train_inference_disk_summary, set_inference_toml_cli_paths, FragmentComposeConfig,
    FragmentDecomposeConfig, FragmentIntentHint, LoadedInferenceToml,
};
pub use manifest::{
    inference_thresholds_disk, parse_plugins_manifest_bytes, serialize_plugins_manifest,
    BrainPluginsManifest, InferenceThresholds,
};
pub use plugins::{
    default_inference_harness, format_retrieved_sentiment_line, LatticeShortcutsPlugin,
    TEMPLATE_ID_USER_ANCHORED, TOPIC_KEYS,
};
