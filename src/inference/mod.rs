//! Optional inference-time augmentations (data-driven heuristics, brain-profile hooks).
//! Keeps domain policy out of the core generation loop in `service.rs`.
//!
//! Runtime packaging: [`crate::brain::BrainPackage`] may carry a UTF-8 TOML
//! [`manifest::BrainPluginsManifest`] in `plugins_blob`, parsed on [`crate::service::LanguageService::load_brain`].

pub mod manifest;
pub mod sentiment;

pub use manifest::{parse_plugins_manifest_bytes, serialize_plugins_manifest, BrainPluginsManifest};
