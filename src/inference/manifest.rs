//! TOML manifest embedded in [`crate::brain::BrainPackage::plugins_blob`] for runtime inference plugins.

use serde::{Deserialize, Serialize};
use toml::Table;

use super::sentiment::SentimentInferenceConfig;

/// Bundled plugin settings shipped inside a brain `.bin` (UTF-8 TOML).
///
/// Known top-level tables:
/// - `[sentiment]` — typed thresholds (see [`SentimentInferenceConfig`]).
/// - `[language_detection]` — locale / detector hints (opaque table until a plugin consumes it).
/// - `[badwords]` — list paths, severity, locales, etc. (opaque table).
///
/// Add more `Option<Table>` fields here as plugins gain first-class support.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct BrainPluginsManifest {
    #[serde(default)]
    pub sentiment: Option<SentimentInferenceConfig>,
    #[serde(default)]
    pub language_detection: Option<Table>,
    #[serde(default)]
    pub badwords: Option<Table>,
}

impl BrainPluginsManifest {
    /// True if anything would be written to the brain package plugins blob.
    pub fn has_embeddable_content(&self) -> bool {
        self.sentiment.is_some()
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
