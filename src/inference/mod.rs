//! Optional inference-time augmentations (data-driven heuristics, brain-profile hooks).
//! Keeps domain policy out of the core generation loop in `service.rs`.
//!
//! Runtime packaging: [`crate::brain::BrainPackage`] may carry a UTF-8 TOML
//! [`manifest::BrainPluginsManifest`] in `plugins_blob`, parsed on [`crate::service::LanguageService::load_brain`].
//!
//! Plugin logic lives under [`plugins`] and is driven by [`harness::InferenceHarness`].

pub mod causal_hints;
pub mod causal_relation;
pub mod chat_policy;
pub mod chat_structure_metacog;
pub mod context_frame;
pub mod frame_lexicon;
pub mod grounding_expand;
pub mod grounding_loop;
pub mod harness;
pub mod inference_guardrails;
pub mod inference_toml;
pub mod lookup_graph;
pub mod manifest;
pub mod plugins;
pub mod retrieval_lexicon;
pub mod retrieval_rescore;
pub mod sentiment_generation_lexicon;
pub mod world_grounding;

pub use chat_policy::{BleedHit, ChatPolicyLocale, ChatPolicySection};
pub use chat_structure_metacog::{evaluate as evaluate_chat_structure, ChatStructureOutcome};
pub use context_frame::{ContextFrame, ContextFrameConfig, MoodGradient, SpeechAct};
pub use harness::{
    BrainInferencePlugin, GenerationPreemptOutcome, InferenceHarness, TemplatePostprocessFlags,
};
pub use inference_guardrails::{set_inference_guardrails_jsonl_path, GuardrailsDiskSummary};
pub use inference_toml::{
    inference_rules_runtime, inference_toml_directory, inference_toml_loaded,
    print_train_inference_disk_summary, set_inference_toml_cli_paths, FragmentComposeConfig,
    FragmentDecomposeConfig, FragmentIntentHint, FragmentReasoningPassConfig, LoadedInferenceToml,
};
pub use manifest::{
    inference_thresholds_disk, parse_plugins_manifest_bytes, serialize_plugins_manifest,
    BrainPluginsManifest, InferenceThresholds,
};
pub use plugins::{
    default_inference_harness, format_retrieved_sentiment_line, ChatPolicyPlugin,
    LatticeShortcutsPlugin, TEMPLATE_ID_USER_ANCHORED, TOPIC_KEYS,
};
