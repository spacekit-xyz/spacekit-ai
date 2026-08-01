//! Growformer project manifest (`*.gf.toml`): paths and metadata for train / infer / inference rules.
//! Paths in the file are resolved relative to the manifest's directory unless absolute.

use serde::Deserialize;
use std::path::{Path, PathBuf};

#[derive(Debug, Deserialize)]
pub struct GrowformerProjectFile {
    #[serde(default = "schema_version_default")]
    pub schema_version: u32,
    #[serde(default)]
    pub project: Option<ProjectSection>,
    #[serde(default)]
    pub train: Option<TrainSection>,
    #[serde(default)]
    pub inference: Option<InferenceSection>,
    #[serde(default)]
    pub infer: Option<InferSection>,
    #[serde(default)]
    pub certified_routing: Option<CertifiedRoutingSection>,
}

fn schema_version_default() -> u32 {
    1
}

#[derive(Debug, Deserialize)]
pub struct ProjectSection {
    pub name: Option<String>,
    pub author: Option<String>,
    pub description: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct TrainSection {
    pub auto: Option<bool>,
    /// When true: run GrowformerLang MetaCodebook (Stage 2b) and train per-group code lattices from `expected_code` in JSONL. Use only for a dedicated code-generation brain.
    pub code_brain: Option<bool>,
    pub data_dir: Option<String>,
    pub brain_output: Option<String>,
    pub brain_plugins_toml: Option<String>,
    pub brain_epochs: Option<u32>,
    pub brain_gen_epochs: Option<u32>,
    pub brain_gen_replicas: Option<u32>,
    /// Path to a GLE student checkpoint JSON (relative to manifest).
    /// Enables the neural encoder instead of the hash-based encoder.
    pub gle_checkpoint: Option<String>,
    /// Encoder preset override: "clifford_e8" for MLP-free Clifford encoder.
    /// When set, takes precedence over gle_checkpoint.
    pub encoder: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct InferenceSection {
    pub toml: Option<String>,
    pub defaults_toml: Option<String>,
    /// Topic routing graph (e.g. `data/knowledge_graph_pet_overlay.toml`). Loaded alone when no base `knowledge_graph.toml` exists.
    pub topic_graph: Option<String>,
    /// World grounding graph (e.g. `data/pet_world_grounding.toml`) for BM25 concept expansion.
    pub grounding_toml: Option<String>,
    /// Compact lookup graph JSON (e.g. `data/wordnet_graph.json`) for exact lemma ego-network responses.
    pub lookup_graph_json: Option<String>,
    /// Fragment library JSONL (e.g. `data/kitsu_fragments_v2.jsonl`). Overrides `[fragment_compose].library` when set.
    pub fragments_jsonl: Option<String>,
    /// Optional JSONL of extra `lexical_topic` / `lattice_misfire` guardrails (merged after TOML).
    pub guardrails_jsonl: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct InferSection {
    pub brain: Option<String>,
}

/// Certified semantic-primary routing (Python coder ship path).
/// See spacekit-projects/coding/python/docs/GROWFORMER_SEMANTIC_ROUTER.md.
#[derive(Debug, Deserialize, Clone)]
pub struct CertifiedRoutingSection {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_semantic_graph")]
    pub semantic_graph: String,
    #[serde(default = "default_thresholds")]
    pub thresholds: String,
    pub snippet_catalog: Option<String>,
    pub verdict: Option<String>,
    /// When false, serve curated snippets (no open codegen lattices).
    #[serde(default)]
    pub code_generation: bool,
    #[serde(default)]
    pub require_runtime_parity: bool,
    #[serde(default)]
    pub require_abstain_parity: bool,
}

fn default_semantic_graph() -> String {
    "data/knowledge_graph_semantic.toml".into()
}

fn default_thresholds() -> String {
    "agent/routing_thresholds.json".into()
}

pub fn read_project_file(path: &Path) -> Result<GrowformerProjectFile, String> {
    let s = std::fs::read_to_string(path)
        .map_err(|e| format!("--project {}: {}", path.display(), e))?;
    toml::from_str(&s).map_err(|e| format!("--project {}: TOML: {}", path.display(), e))
}

pub fn manifest_base_dir(manifest_path: &Path) -> PathBuf {
    manifest_path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf()
}

pub fn resolve_against(base: &Path, rel: &str) -> PathBuf {
    let p = Path::new(rel);
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        base.join(p)
    }
}

/// Content for `growformer init`.
pub fn init_template(default_name: &str) -> String {
    format!(
        r#"# Growformer project manifest (schema version 1). Paths are relative to this file's directory.
schema_version = 1

[project]
name = "{name}"
author = "Your Name"
description = "Short description for the exported brain package."

[train]
auto = true
# code_brain = true   # only for a standalone code brain: MetaCodebook + expected_code lattices
data_dir = "data/sentiment"
brain_output = "agent-data/example/brain.bin"
brain_plugins_toml = "data/plugins/example-brain-plugins.toml"
# brain_epochs = 30
# brain_gen_epochs = 0
# brain_gen_replicas = 1

[inference]
# Shortcut rules + numeric gates (equivalent to --inference-toml).
toml = "data/sentiment/inference_sentiment_core.toml"
# Topic routing (pet companions: usually knowledge_graph_pet_overlay.toml).
# topic_graph = "data/knowledge_graph_pet_overlay.toml"
# Concept grounding graph (pet companions: pet_world_grounding.toml).
# grounding_toml = "data/pet_world_grounding.toml"
# Fragment library JSONL (when [fragment_compose] is enabled in inference TOML).
# fragments_jsonl = "data/my_fragments_v1.jsonl"
# Optional baseline for merging empty [rules] arrays (equivalent to --inference-defaults-toml).
# defaults_toml = "data/sentiment/inference_sentiment_core.toml"
# Optional guardrails JSONL (equivalent to --inference-guardrails-jsonl); merged after TOML rules.
# guardrails_jsonl = "data/sentiment/inference_guardrails.jsonl"

[infer]
# Default brain for `growformer --infer --project this_file.gf.toml` when --brain is omitted.
brain = "agent-data/example/brain.bin"
"#,
        name = default_name
    )
}
