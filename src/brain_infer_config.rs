//! Bootstrap growformer inference config for `brain-infer` / `brain-raw-diag`.
//!
//! Mirrors the inference-related startup in growformer CLI (`--project`, inference TOML,
//! guardrails JSONL, topic graph, optional runtime grounding overlay) so brain-memory
//! paths match native `--infer --project *.gf.toml`.

use std::path::{Path, PathBuf};

use growformer::inference::{
    inference_toml_loaded, set_inference_guardrails_jsonl_path, set_inference_toml_cli_paths,
};
use growformer::inference::world_grounding::load_grounding_graph_from_str;
use serde::Deserialize;

/// Minimal `*.gf.toml` parse (mirrors `growformer::project_gf` without the CLI feature gate).
#[derive(Debug, Deserialize)]
struct GrowformerProjectFile {
    #[serde(default = "schema_version_default")]
    schema_version: u32,
    #[serde(default)]
    inference: Option<InferenceSection>,
}

fn schema_version_default() -> u32 {
    1
}

#[derive(Debug, Deserialize)]
struct InferenceSection {
    toml: Option<String>,
    defaults_toml: Option<String>,
    topic_graph: Option<String>,
    grounding_toml: Option<String>,
    guardrails_jsonl: Option<String>,
}

fn read_project_file(path: &Path) -> Result<GrowformerProjectFile, String> {
    let s = std::fs::read_to_string(path)
        .map_err(|e| format!("--project {}: {}", path.display(), e))?;
    toml::from_str(&s).map_err(|e| format!("--project {}: TOML: {}", path.display(), e))
}

fn manifest_base_dir(manifest_path: &Path) -> PathBuf {
    manifest_path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf()
}

fn resolve_against(base: &Path, rel: &str) -> PathBuf {
    let p = Path::new(rel);
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        base.join(p)
    }
}

/// CLI / battery options for growformer inference bootstrap.
#[derive(Debug, Clone, Default)]
pub struct BrainInferConfig {
    /// `*.gf.toml` project manifest (sets inference TOML, guardrails, topic graph, grounding).
    pub project: Option<PathBuf>,
    /// Override `[inference].toml` / primary inference pack.
    pub inference_toml: Option<PathBuf>,
    /// Override `[inference].defaults_toml`.
    pub inference_defaults_toml: Option<PathBuf>,
    /// Override `[inference].guardrails_jsonl`.
    pub guardrails_jsonl: Option<PathBuf>,
    /// Print resolved paths and TOML row counts.
    pub verbose: bool,
}

#[derive(Debug, Clone)]
pub struct AppliedBrainInferConfig {
    pub project: Option<PathBuf>,
    pub inference_toml: Option<PathBuf>,
    pub inference_defaults_toml: Option<PathBuf>,
    pub guardrails_jsonl: Option<PathBuf>,
    pub topic_graph: PathBuf,
    pub grounding_toml: Option<PathBuf>,
    pub topic_graph_loaded: bool,
}

impl BrainInferConfig {
    fn merge_from_project(&self, gf: &GrowformerProjectFile, base: &Path) -> MergedPaths {
        let mut m = MergedPaths::default();
        if let Some(inf) = &gf.inference {
            if self.inference_toml.is_none() {
                if let Some(t) = inf.toml.as_deref() {
                    m.inference_toml = Some(resolve_against(base, t));
                }
            }
            if self.inference_defaults_toml.is_none() {
                if let Some(t) = inf.defaults_toml.as_deref() {
                    m.inference_defaults_toml = Some(resolve_against(base, t));
                }
            }
            if self.guardrails_jsonl.is_none() {
                if let Some(p) = inf.guardrails_jsonl.as_deref() {
                    m.guardrails_jsonl = Some(resolve_against(base, p));
                }
            }
            if let Some(t) = inf.topic_graph.as_deref() {
                m.topic_graph = Some(resolve_against(base, t));
            } else {
                let pet = base.join("data/knowledge_graph_pet_overlay.toml");
                if pet.is_file() {
                    m.topic_graph = Some(pet);
                }
            }
            if let Some(t) = inf.grounding_toml.as_deref() {
                m.grounding_toml = Some(resolve_against(base, t));
            } else {
                let pet = base.join("data/pet_world_grounding.toml");
                if pet.is_file() {
                    m.grounding_toml = Some(pet);
                }
            }
        }
        m
    }

    /// Apply inference config globally (call before `BrainMemoryRuntime::from_bytes`).
    pub fn apply(&self) -> Result<AppliedBrainInferConfig, String> {
        let project = self.project.clone();
        let mut inference_toml = self.inference_toml.clone();
        let mut inference_defaults_toml = self.inference_defaults_toml.clone();
        let mut guardrails_jsonl = self.guardrails_jsonl.clone();
        let mut topic_graph_overlay: Option<PathBuf> = None;
        let mut grounding_toml: Option<PathBuf> = None;

        if let Some(ref proj) = project {
            let gf = read_project_file(proj)?;
            if gf.schema_version != 1 {
                return Err(format!(
                    "unsupported schema_version {} in {} (expected 1)",
                    gf.schema_version,
                    proj.display()
                ));
            }
            let base = manifest_base_dir(proj);
            let merged = self.merge_from_project(&gf, &base);
            if inference_toml.is_none() {
                inference_toml = merged.inference_toml;
            }
            if inference_defaults_toml.is_none() {
                inference_defaults_toml = merged.inference_defaults_toml;
            }
            if guardrails_jsonl.is_none() {
                guardrails_jsonl = merged.guardrails_jsonl;
            }
            topic_graph_overlay = merged.topic_graph;
            if grounding_toml.is_none() {
                grounding_toml = merged.grounding_toml;
            }
        }

        set_inference_toml_cli_paths(inference_toml.clone(), inference_defaults_toml.clone());
        set_inference_guardrails_jsonl_path(guardrails_jsonl.clone());

        let topic_graph = resolve_knowledge_graph_path(project.as_deref());
        let pet_overlays: Vec<PathBuf> = topic_graph_overlay.into_iter().collect();
        let topic_graph_loaded = match growformer::growformer_lang::try_init_topic_graph_bundle_with_extras(
            &topic_graph.to_string_lossy(),
            &pet_overlays,
        ) {
            Ok(()) => growformer::growformer_lang::topic_graph_loaded(),
            Err(e) => {
                if self.verbose {
                    eprintln!("Warning: failed to load topic graph: {e}");
                }
                false
            }
        };

        if let Some(ref gpath) = grounding_toml {
            if gpath.is_file() {
                match std::fs::read_to_string(gpath) {
                    Ok(s) => {
                        if let Err(e) = load_grounding_graph_from_str(&s) {
                            eprintln!(
                                "Warning: failed to load grounding graph {}: {}",
                                gpath.display(),
                                e
                            );
                        } else if self.verbose {
                            println!("Grounding graph: loaded runtime overlay from {}", gpath.display());
                        }
                    }
                    Err(e) => eprintln!(
                        "Warning: failed to read grounding graph {}: {}",
                        gpath.display(),
                        e
                    ),
                }
            }
        }

        if self.verbose {
            let loaded = inference_toml_loaded();
            println!(
                "Brain infer config: project={}",
                project
                    .as_ref()
                    .map(|p| p.display().to_string())
                    .unwrap_or_else(|| "(none)".into())
            );
            println!(
                "  inference_toml={}",
                inference_toml
                    .as_ref()
                    .map(|p| p.display().to_string())
                    .unwrap_or_else(|| "(auto-discover)".into())
            );
            if let Some(ref p) = guardrails_jsonl {
                println!("  guardrails_jsonl={}", p.display());
            }
            println!(
                "  topic_graph={} loaded={}",
                topic_graph.display(),
                topic_graph_loaded
            );
            println!(
                "  TOML rows: {} headline_lexical_topic, +{} guardrail misfire rows",
                loaded.rules_section.headline_lexical_topic.len(),
                loaded.guardrails.misfire_rows_appended,
            );
        }

        Ok(AppliedBrainInferConfig {
            project,
            inference_toml,
            inference_defaults_toml,
            guardrails_jsonl,
            topic_graph,
            grounding_toml,
            topic_graph_loaded,
        })
    }
}

#[derive(Default)]
struct MergedPaths {
    inference_toml: Option<PathBuf>,
    inference_defaults_toml: Option<PathBuf>,
    guardrails_jsonl: Option<PathBuf>,
    topic_graph: Option<PathBuf>,
    grounding_toml: Option<PathBuf>,
}

/// Resolve `data/knowledge_graph.toml` the same way as growformer CLI, with a repo fallback
/// when manifests live under `scripts/` (paths in gf.toml use `../data/...`).
fn resolve_knowledge_graph_path(project: Option<&Path>) -> PathBuf {
    if let Some(proj) = project {
        let base = manifest_base_dir(proj);
        let direct = base.join("data/knowledge_graph.toml");
        if direct.is_file() {
            return direct;
        }
        let sibling = base.join("../data/knowledge_graph.toml");
        if sibling.is_file() {
            return sibling;
        }
    }
    PathBuf::from("data/knowledge_graph.toml")
}

/// Canonical SpaceKit sentiment project root (crypto + fintech micro-brains).
/// Override with `SPACEKIT_SENTIMENT_ROOT` when layouts differ.
pub fn spacekit_sentiment_root() -> PathBuf {
    std::env::var("SPACEKIT_SENTIMENT_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("../../spacekit/spacekit-projects/sentiment"))
}

/// Pre-registered 4-prompt battery with per-brain project manifests (config parity).
#[derive(Clone)]
pub struct BatteryCase {
    pub label: &'static str,
    pub project: PathBuf,
    pub brain: PathBuf,
    pub prompt: &'static str,
    /// When set, default `--battery` skips this case (diagnostic-only / not scored).
    pub skip_reason: Option<&'static str>,
}

/// Cases included in default `--battery` runs (trained SpaceKit brains only).
pub fn scored_battery_cases(battery_brains: bool) -> impl Iterator<Item = BatteryCase> {
    battery_cases(battery_brains).into_iter().filter(|c| c.skip_reason.is_none())
}

pub fn battery_cases(battery_brains: bool) -> [BatteryCase; 4] {
    let sk = spacekit_sentiment_root();
    let neurokit = PathBuf::from("../growformer");

    let (sentiment_brain, crypto_brain, fintech_brain) = if battery_brains {
        (
            neurokit.join("agent-data/sentiment-analysis/sentiment-brain-v3-battery.bin"),
            neurokit.join("agent-data/crypto-analysis/crypto-brain-battery.bin"),
            neurokit.join("agent-data/fintech-analysis/fintech-brain-battery.bin"),
        )
    } else {
        (
            neurokit.join("agent-data/sentiment-analysis/sentiment-brain-v3.bin"),
            sk.join("crypto/agent/crypto-brain.bin"),
            sk.join("fintech/agent/fintech-brain.bin"),
        )
    };

    const UNTRAINED_SENTIMENT: &str = "sentiment-brain-v3.bin is not trained — skip until a SpaceKit general-sentiment brain exists";

    [
        BatteryCase {
            label: "case1_sentiment_bitcoin",
            project: neurokit.join("scripts/sentiment-analysis.gf.toml"),
            brain: sentiment_brain.clone(),
            prompt: "Bitcoin crashed after the ETF delay",
            skip_reason: Some(UNTRAINED_SENTIMENT),
        },
        BatteryCase {
            label: "case2_crypto_bitcoin",
            project: sk.join("crypto/crypto-sentiment-analysis.gf.toml"),
            brain: crypto_brain,
            prompt: "Bitcoin crashed after the ETF delay",
            skip_reason: None,
        },
        BatteryCase {
            label: "case3_fintech_chase",
            project: sk.join("fintech/fintech-sentiment-analysis.gf.toml"),
            brain: fintech_brain,
            prompt: "Chase raised my mortgage rate without notice",
            skip_reason: None,
        },
        BatteryCase {
            label: "case4_sentiment_chase_wrong_brain",
            project: neurokit.join("scripts/sentiment-analysis.gf.toml"),
            brain: sentiment_brain.clone(),
            prompt: "Chase raised my mortgage rate without notice",
            skip_reason: Some(UNTRAINED_SENTIMENT),
        },
    ]
}

/// Held-out paraphrase eval (pre-registered; prompts not in train globs).
#[derive(Clone)]
pub struct HeldoutCase {
    pub label: String,
    pub project: PathBuf,
    pub brain: PathBuf,
    pub prompt: String,
    pub expected_topic: Option<String>,
}

pub fn heldout_battery_cases() -> Result<Vec<HeldoutCase>, String> {
    let path = PathBuf::from("../growformer/data/sentiment/eval_battery_heldout_prompts.jsonl");
    let raw = std::fs::read_to_string(&path)
        .map_err(|e| format!("held-out prompts {}: {e}", path.display()))?;
    let sk = spacekit_sentiment_root();
    let mut out = Vec::new();
    for line in raw.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let obj: serde_json::Value =
            serde_json::from_str(line).map_err(|e| format!("held-out JSONL: {e}"))?;
        let corpus = obj
            .get("corpus")
            .and_then(|v| v.as_str())
            .ok_or("held-out row missing corpus")?;
        let (project, brain) = match corpus {
            "crypto" => (
                sk.join("crypto/crypto-sentiment-analysis.gf.toml"),
                sk.join("crypto/agent/crypto-brain.bin"),
            ),
            "fintech" => (
                sk.join("fintech/fintech-sentiment-analysis.gf.toml"),
                sk.join("fintech/agent/fintech-brain.bin"),
            ),
            other => return Err(format!("held-out corpus `{other}` unsupported")),
        };
        out.push(HeldoutCase {
            label: obj
                .get("case_id")
                .and_then(|v| v.as_str())
                .unwrap_or("heldout")
                .to_string(),
            project,
            brain,
            prompt: obj
                .get("prompt")
                .and_then(|v| v.as_str())
                .ok_or("held-out row missing prompt")?
                .to_string(),
            expected_topic: obj
                .get("expected_topic")
                .and_then(|v| v.as_str())
                .map(str::to_string),
        });
    }
    Ok(out)
}
