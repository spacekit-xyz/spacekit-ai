//! TOML manifest embedded in [`crate::brain::BrainPackage::plugins_blob`] for runtime inference plugins.

use std::borrow::Cow;
use std::sync::{Arc, OnceLock};

use serde::{Deserialize, Serialize};
use toml::Table;

/// Numeric gates for lattice-style shortcut inference; serialized under **`[sentiment]`** in brain
/// packages for backward compatibility with existing `.bin` exports.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct InferenceThresholds {
    pub meta_gk_margin: f32,
    pub meta_gk_confidence: f32,
    pub min_meta_confidence_user_anchored: f32,
    pub mixed_override_confidence: f32,
    pub default_line_confidence: f32,
    /// Confidence for hedged / ambiguous lines (not a clear pole).
    pub ambiguous_line_confidence: f32,
}

impl Default for InferenceThresholds {
    fn default() -> Self {
        Self {
            meta_gk_margin: 0.05,
            meta_gk_confidence: 0.55,
            min_meta_confidence_user_anchored: 0.28,
            mixed_override_confidence: 0.88,
            default_line_confidence: 0.92,
            ambiguous_line_confidence: 0.58,
        }
    }
}

/// Defaults from the embedded inference TOML when the brain manifest has no override table.
pub fn inference_thresholds_disk() -> Arc<InferenceThresholds> {
    static LOCK: OnceLock<Arc<InferenceThresholds>> = OnceLock::new();
    LOCK.get_or_init(|| {
        Arc::new(
            crate::inference::inference_toml::inference_toml_loaded()
                .thresholds
                .clone(),
        )
    })
    .clone()
}

/// Brain-bundled `[sentiment]` TOML overrides disk/env inference TOML when present.
pub(crate) fn resolved_inference_thresholds<'a>(
    brain_override: Option<&'a InferenceThresholds>,
) -> Cow<'a, InferenceThresholds> {
    match brain_override {
        Some(c) => Cow::Borrowed(c),
        None => Cow::Owned((*inference_thresholds_disk()).clone()),
    }
}

/// Bundled plugin settings shipped inside a brain `.bin` (UTF-8 TOML).
///
/// Known top-level tables:
/// - `[sentiment]` — numeric gates (see [`InferenceThresholds`]); serde key kept for existing brains.
/// - `[language_detection]` — locale / detector hints (opaque table until a plugin consumes it).
/// - `[badwords]` — list paths, severity, locales, etc. (opaque table).
///
/// Add more `Option<Table>` fields here as plugins gain first-class support.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct BrainPluginsManifest {
    #[serde(default, rename = "sentiment")]
    pub inference_thresholds: Option<InferenceThresholds>,
    #[serde(default)]
    pub language_detection: Option<Table>,
    #[serde(default)]
    pub badwords: Option<Table>,
}

impl BrainPluginsManifest {
    /// True if anything would be written to the brain package plugins blob.
    pub fn has_embeddable_content(&self) -> bool {
        self.inference_thresholds.is_some()
            || self
                .language_detection
                .as_ref()
                .map_or(false, |t| !t.is_empty())
            || self.badwords.as_ref().map_or(false, |t| !t.is_empty())
    }
}

/// Parse the plugins section from brain package bytes (must be valid UTF-8 TOML).
pub fn parse_plugins_manifest_bytes(blob: &[u8]) -> Result<BrainPluginsManifest, String> {
    let s = std::str::from_utf8(blob).map_err(|e| format!("plugins blob: invalid UTF-8: {}", e))?;
    toml::from_str(s).map_err(|e| format!("plugins manifest TOML: {}", e))
}

pub fn serialize_plugins_manifest(manifest: &BrainPluginsManifest) -> Result<Vec<u8>, String> {
    if !manifest.has_embeddable_content() {
        return Ok(Vec::new());
    }
    toml::to_string_pretty(manifest)
        .map_err(|e| format!("plugins manifest serialize: {}", e))
        .map(|s| s.into_bytes())
}
