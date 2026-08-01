//! Certified semantic-primary router (mpnet centroids + abstain).
//!
//! Parity target: `routing_lib.route_eval_rows_certified` in the Python coder project.
//! Encoding uses a Python sentence-transformers bridge (`scripts/mpnet_encode_stdin.py`)
//! so decisions match the offline certifier bit-for-bit (including abstain).

#![cfg(all(not(target_arch = "wasm32"), feature = "cli"))]

use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::RwLock;

#[derive(Debug, Clone)]
pub struct RouteDecision {
    pub topic: Option<String>,
    pub top1: f32,
    pub margin: f32,
}

#[derive(Debug, Clone)]
pub struct Snippet {
    pub intent: String,
    pub code_language: String,
    pub expected_response: String,
    pub expected_code: String,
}

#[derive(Debug, Clone)]
struct SemanticNode {
    topic: String,
    centroid: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct SemanticRouter {
    model: String,
    nodes: Vec<SemanticNode>,
    match_threshold: f32,
    margin_threshold: f32,
    snippets: Vec<Snippet>,
    code_generation: bool,
    encode_script: PathBuf,
}

#[derive(Debug, Clone)]
pub struct CertifiedRoutingPaths {
    pub enabled: bool,
    pub semantic_graph: PathBuf,
    pub thresholds: PathBuf,
    pub snippet_catalog: Option<PathBuf>,
    pub code_generation: bool,
}

static ROUTER: RwLock<Option<SemanticRouter>> = RwLock::new(None);

#[derive(Deserialize)]
struct SemanticToml {
    #[serde(default)]
    semantic_routing: SemanticRoutingSection,
    #[serde(default)]
    nodes: Vec<NodeToml>,
}

#[derive(Deserialize, Default)]
struct SemanticRoutingSection {
    #[serde(default = "default_model")]
    model: String,
    #[serde(default = "default_match")]
    match_threshold: f32,
    #[serde(default = "default_margin")]
    margin_threshold: f32,
}

fn default_model() -> String {
    "sentence-transformers/all-mpnet-base-v2".into()
}
fn default_match() -> f32 {
    0.42
}
fn default_margin() -> f32 {
    0.06
}

#[derive(Deserialize)]
struct NodeToml {
    #[serde(default)]
    topic: String,
    #[serde(default)]
    centroid: Vec<f32>,
}

#[derive(Deserialize)]
struct ThresholdsJson {
    match_threshold: Option<f32>,
    margin_threshold: Option<f32>,
}

#[derive(Deserialize)]
struct SnippetCatalogToml {
    #[serde(default)]
    snippet: Vec<SnippetToml>,
}

#[derive(Deserialize)]
struct SnippetToml {
    intent: String,
    #[serde(default)]
    code_language: String,
    #[serde(default)]
    expected_response: String,
    #[serde(default)]
    expected_code: String,
}

fn load_snippet_catalog(path: &Path) -> Result<Vec<Snippet>, String> {
    let cat_raw = std::fs::read_to_string(path)
        .map_err(|e| format!("snippet catalog {}: {e}", path.display()))?;
    let cat_doc: SnippetCatalogToml =
        toml::from_str(&cat_raw).map_err(|e| format!("snippet catalog {}: {e}", path.display()))?;
    Ok(cat_doc
        .snippet
        .into_iter()
        .map(|s| Snippet {
            intent: s.intent,
            code_language: if s.code_language.is_empty() {
                "python".into()
            } else {
                s.code_language
            },
            expected_response: s.expected_response,
            expected_code: s.expected_code.trim().to_string(),
        })
        .collect())
}

fn find_encode_script() -> PathBuf {
    if let Ok(p) = std::env::var("GROWFORMER_MPNET_ENCODE_SCRIPT") {
        let pb = PathBuf::from(p);
        if pb.is_file() {
            return pb;
        }
    }
    // growformer/scripts/mpnet_encode_stdin.py relative to CARGO_MANIFEST_DIR at compile time
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let cand = manifest.join("scripts/mpnet_encode_stdin.py");
    if cand.is_file() {
        return cand;
    }
    PathBuf::from("scripts/mpnet_encode_stdin.py")
}

fn python_bin() -> String {
    std::env::var("GROWFORMER_PYTHON").unwrap_or_else(|_| "python3".into())
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..n {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let na = na.sqrt();
    let nb = nb.sqrt();
    if na < 1e-12 || nb < 1e-12 {
        return 0.0;
    }
    dot / (na * nb)
}

impl SemanticRouter {
    pub fn load(paths: &CertifiedRoutingPaths) -> Result<Self, String> {
        let raw = std::fs::read_to_string(&paths.semantic_graph).map_err(|e| {
            format!(
                "certified semantic graph {}: {e}",
                paths.semantic_graph.display()
            )
        })?;
        let doc: SemanticToml = toml::from_str(&raw).map_err(|e| {
            format!(
                "certified semantic graph {}: {e}",
                paths.semantic_graph.display()
            )
        })?;

        let mut match_threshold = doc.semantic_routing.match_threshold;
        let mut margin_threshold = doc.semantic_routing.margin_threshold;
        if paths.thresholds.is_file() {
            let thr_raw = std::fs::read_to_string(&paths.thresholds)
                .map_err(|e| format!("routing thresholds {}: {e}", paths.thresholds.display()))?;
            let thr: ThresholdsJson = serde_json::from_str(&thr_raw)
                .map_err(|e| format!("routing thresholds {}: {e}", paths.thresholds.display()))?;
            if let Some(m) = thr.match_threshold {
                match_threshold = m;
            }
            if let Some(g) = thr.margin_threshold {
                margin_threshold = g;
            }
        }

        let nodes: Vec<SemanticNode> = doc
            .nodes
            .into_iter()
            .filter(|n| !n.topic.is_empty() && !n.centroid.is_empty())
            .map(|n| SemanticNode {
                topic: n.topic,
                centroid: n.centroid,
            })
            .collect();
        if nodes.is_empty() {
            return Err("certified semantic graph has no centroid nodes".into());
        }

        let mut snippets = Vec::new();
        if let Some(ref cat) = paths.snippet_catalog {
            if cat.is_file() {
                match load_snippet_catalog(cat) {
                    Ok(loaded) => snippets = loaded,
                    Err(e) => {
                        // Routing still works without snippets; codegen short-circuit needs them.
                        if paths.code_generation {
                            return Err(e);
                        }
                        eprintln!(
                            "[growformer] warning: snippet catalog not loaded ({e}); \
                             certified routing continues without snippet short-circuit"
                        );
                    }
                }
            }
        }

        Ok(Self {
            model: doc.semantic_routing.model,
            nodes,
            match_threshold,
            margin_threshold,
            snippets,
            code_generation: paths.code_generation,
            encode_script: find_encode_script(),
        })
    }

    pub fn code_generation_enabled(&self) -> bool {
        self.code_generation
    }

    pub fn lookup_snippet(&self, intent: &str) -> Option<&Snippet> {
        self.snippets
            .iter()
            .find(|s| s.intent.eq_ignore_ascii_case(intent))
    }

    pub fn encode_texts(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        if !self.encode_script.is_file() {
            return Err(format!(
                "mpnet encode script missing: {} (set GROWFORMER_MPNET_ENCODE_SCRIPT)",
                self.encode_script.display()
            ));
        }
        let req = serde_json::json!({
            "model": self.model,
            "texts": texts,
        });
        let mut child = Command::new(python_bin())
            .arg(&self.encode_script)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("spawn mpnet encoder: {e}"))?;
        {
            use std::io::Write;
            let stdin = child.stdin.as_mut().ok_or("mpnet encoder stdin")?;
            stdin
                .write_all(req.to_string().as_bytes())
                .map_err(|e| format!("mpnet encoder stdin write: {e}"))?;
        }
        let out = child
            .wait_with_output()
            .map_err(|e| format!("mpnet encoder wait: {e}"))?;
        if !out.status.success() {
            let err = String::from_utf8_lossy(&out.stderr);
            return Err(format!("mpnet encoder failed: {err}"));
        }
        #[derive(Deserialize)]
        struct Resp {
            embeddings: Vec<Vec<f32>>,
        }
        let resp: Resp =
            serde_json::from_slice(&out.stdout).map_err(|e| format!("mpnet encoder JSON: {e}"))?;
        if resp.embeddings.len() != texts.len() {
            return Err(format!(
                "mpnet encoder returned {} vectors for {} texts",
                resp.embeddings.len(),
                texts.len()
            ));
        }
        Ok(resp.embeddings)
    }

    pub fn infer_embedding(&self, emb: &[f32]) -> RouteDecision {
        let mut scored: Vec<(String, f32)> = self
            .nodes
            .iter()
            .map(|n| (n.topic.clone(), cosine_similarity(emb, &n.centroid)))
            .collect();
        if scored.is_empty() {
            return RouteDecision {
                topic: None,
                top1: 0.0,
                margin: 0.0,
            };
        }
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top1_topic = scored[0].0.clone();
        let top1 = scored[0].1;
        let top2 = scored.get(1).map(|x| x.1).unwrap_or(0.0);
        let margin = top1 - top2;
        if top1 >= self.match_threshold && margin >= self.margin_threshold {
            RouteDecision {
                topic: Some(top1_topic),
                top1,
                margin,
            }
        } else {
            RouteDecision {
                topic: None,
                top1,
                margin,
            }
        }
    }

    pub fn infer(&self, text: &str) -> Result<RouteDecision, String> {
        let emb = self.encode_texts(&[text.trim().to_string()])?;
        Ok(self.infer_embedding(&emb[0]))
    }

    pub fn infer_batch(&self, texts: &[String]) -> Result<Vec<RouteDecision>, String> {
        let embs = self.encode_texts(texts)?;
        Ok(embs.iter().map(|e| self.infer_embedding(e)).collect())
    }
}

pub fn install(router: SemanticRouter) -> Result<(), String> {
    let mut guard = ROUTER
        .write()
        .map_err(|_| "semantic router lock poisoned".to_string())?;
    *guard = Some(router);
    Ok(())
}

pub fn clear() {
    if let Ok(mut guard) = ROUTER.write() {
        *guard = None;
    }
}

pub fn is_loaded() -> bool {
    ROUTER.read().map(|g| g.is_some()).unwrap_or(false)
}

/// When loaded: returns Some(decision). Caller must treat topic=None as abstain
/// (no keyword fallback). When not loaded: returns None (use keyword TopicGraph).
pub fn infer(text: &str) -> Option<Result<RouteDecision, String>> {
    let guard = ROUTER.read().ok()?;
    let router = guard.as_ref()?;
    Some(router.infer(text))
}

pub fn with_router<R>(f: impl FnOnce(&SemanticRouter) -> R) -> Option<R> {
    let guard = ROUTER.read().ok()?;
    guard.as_ref().map(f)
}

pub fn try_init_from_paths(paths: &CertifiedRoutingPaths) -> Result<(), String> {
    if !paths.enabled {
        clear();
        return Ok(());
    }
    let router = SemanticRouter::load(paths)?;
    install(router)
}

/// Resolve certified_routing paths from a *.gf.toml + section values.
pub fn paths_from_project(
    manifest_base: &Path,
    enabled: bool,
    semantic_graph: &str,
    thresholds: &str,
    snippet_catalog: Option<&str>,
    code_generation: bool,
) -> CertifiedRoutingPaths {
    use crate::project_gf::resolve_against;
    CertifiedRoutingPaths {
        enabled,
        semantic_graph: resolve_against(manifest_base, semantic_graph),
        thresholds: resolve_against(manifest_base, thresholds),
        snippet_catalog: snippet_catalog.map(|p| resolve_against(manifest_base, p)),
        code_generation,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_identical_is_one() {
        let v = vec![0.0f32, 3.0, 4.0];
        let s = cosine_similarity(&v, &v);
        assert!((s - 1.0).abs() < 1e-5);
    }

    #[test]
    fn abstain_when_margin_low() {
        let router = SemanticRouter {
            model: "test".into(),
            nodes: vec![
                SemanticNode {
                    topic: "a".into(),
                    centroid: vec![1.0, 0.0],
                },
                SemanticNode {
                    topic: "b".into(),
                    centroid: vec![0.99, 0.141067],
                },
            ],
            match_threshold: 0.42,
            margin_threshold: 0.06,
            snippets: vec![],
            code_generation: false,
            encode_script: PathBuf::from("unused"),
        };
        // nearly equal scores → small margin → abstain
        let emb = [1.0f32, 0.0];
        let d = router.infer_embedding(&emb);
        assert!(d.topic.is_none(), "expected abstain, got {:?}", d.topic);
    }
}
