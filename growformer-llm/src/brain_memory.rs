//! Load growformer `brain.bin` packages as LM memory units (routing + lattice retrieval).
//!
//! Requires `--features brain-memory` (enabled by default). The growformer crate supplies
//! encode → bridge → route → lattice retrieval; this module exposes that as a prefix the
//! LM can condition on.

use std::path::Path;

use growformer::dimension::group_gen::RawLatticeDiagnosticReport;
use growformer::runtime::{BrainInfo, Runtime};

use crate::brain_infer_config::BrainInferConfig;

/// How lattice text was chosen for LM prefixing (HYBRID_DOMAIN_BRAIN product path).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemorySource {
    /// Pre-gate raw top-1 (scenario topic + witness or forced-topic match).
    RawLattice,
    /// Full generation path (metacog + grounding gate + user-anchored).
    FullGeneration,
    /// Label from the supervised classifier; explanation retrieved from that
    /// label's sub-lattice (see [`BrainMemoryRuntime::query_labeled`]).
    LabelForced,
}

impl MemorySource {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::RawLattice => "raw_lattice",
            Self::FullGeneration => "full_generation",
            Self::LabelForced => "label_forced",
        }
    }
}

/// Scenario retrieval buckets (not generic polarity-only topics).
fn is_scenario_lattice_topic(topic: &str) -> bool {
    matches!(
        topic,
        "etf_delay_bearish" | "mortgage_rate_complaint" | "fee_complaint"
    )
}

/// The stored answer for a raw-lattice hit.
///
/// Sentiment lattice programs are stored as
/// `"<training prompt> __GROWFORMER_SENT_WITNESS__ <expected response>"`.
/// The answer is the part **after** the witness marker. (This used to return
/// the part before it — i.e. the nearest training *prompt* — so the chat
/// echoed a paraphrase of the user's question instead of the sentiment reply.)
/// Uses the untruncated program text when available; the 160-char preview can
/// cut the answer mid-word.
fn lattice_answer_from_candidate(c: &growformer::dimension::group_gen::RawLatticeCandidate) -> String {
    use growformer::dimension::language::SENTIMENT_LATTICE_WITNESS_CORE as WITNESS;
    let full = if c.full_text.is_empty() {
        c.text_preview.as_str()
    } else {
        c.full_text.as_str()
    };
    if let Some(idx) = full.find(WITNESS) {
        let answer = full[idx + WITNESS.len()..].trim();
        if !answer.is_empty() {
            return answer.to_string();
        }
        let body = full[..idx].trim();
        if !body.is_empty() {
            return body.to_string();
        }
    }
    growformer::dimension::language::strip_sentiment_lattice_witness_for_display(full)
}

/// The training prompt a lattice program was built from (text before the witness marker).
fn stored_prompt_of_candidate(c: &growformer::dimension::group_gen::RawLatticeCandidate) -> String {
    use growformer::dimension::language::SENTIMENT_LATTICE_WITNESS_CORE as WITNESS;
    let full = if c.full_text.is_empty() {
        c.text_preview.as_str()
    } else {
        c.full_text.as_str()
    };
    match full.find(WITNESS) {
        Some(i) => full[..i].trim().to_string(),
        None => full.split("\n---\n").next().unwrap_or(full).trim().to_string(),
    }
}

fn raw_candidate_usable(
    c: &growformer::dimension::group_gen::RawLatticeCandidate,
    report: &RawLatticeDiagnosticReport,
    prompt: &str,
) -> bool {
    if growformer::inference::frame_lexicon::matches_reject_frame(prompt) {
        return false;
    }
    if c.hard_reject {
        return false;
    }
    if c.witness_ok {
        return true;
    }
    if let Some(ref forced) = report.forced_topic {
        if c.topic == *forced
            && is_scenario_lattice_topic(forced)
            && c.above_score_floor
            && !c.soft_reject
        {
            return true;
        }
    }
    false
}

/// Compact oracle-free features derived from a brain query (for routers / diagnostics).
pub const BRAIN_FEATURE_DIM: usize = 8;

#[derive(Debug, Clone)]
pub struct BrainMemoryQuery {
    pub bridge_vector: Vec<f32>,
    pub bridge_confidence: f32,
    pub group_id: Option<u32>,
    pub route_margin: f32,
    pub route_confidence: f32,
    pub route_rejected_ood: bool,
    pub memory_text: String,
    pub memory_template_id: String,
    pub memory_confidence: f32,
    pub action_type: String,
    pub action_confidence: f32,
}

/// Loaded growformer brain checkpoint (`GWFBRPKG` or legacy JSON).
pub struct BrainMemoryRuntime {
    runtime: Runtime,
}

impl BrainMemoryRuntime {
    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        Ok(Self {
            runtime: Runtime::from_brain_bytes(data)?,
        })
    }

    /// Load brain bytes after applying project / inference TOML bootstrap (`BrainInferConfig::apply`).
    pub fn from_bytes_with_config(data: &[u8], cfg: &BrainInferConfig) -> Result<Self, String> {
        cfg.apply()?;
        let rt = Self::from_bytes(data)?;
        // `load_brain` replaces disk inference rules with brain-embedded `[rules]`; restore
        // project / `--inference-toml` paths (same as growformer CLI after `--train-brain`).
        growformer::inference::inference_toml::force_native_inference_rebuild_from_disk();
        Ok(rt)
    }

    pub fn from_path(path: &Path) -> Result<Self, String> {
        let data = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
        Self::from_bytes(&data)
    }

    pub fn from_path_with_config(path: &Path, cfg: &BrainInferConfig) -> Result<Self, String> {
        let data = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
        Self::from_bytes_with_config(&data, cfg)
    }

    pub fn brain_info(&self) -> BrainInfo {
        self.runtime.brain_info()
    }

    /// HYBRID retrieval: prefer raw lattice top-1 when rubric passes; else full generation path.
    ///
    /// The raw-lattice probe is best-effort: prompts that don't route into the
    /// brain's lattice (off-domain, e.g. "Hey Luna" to the crypto brain) make it
    /// return "no encoded routing path". That used to abort the whole query;
    /// now it falls back to the full generation path (which has the OOD /
    /// guardrail handling) instead.
    pub fn query_hybrid(&mut self, text: &str) -> Result<(BrainMemoryQuery, MemorySource), String> {
        let raw = match self.raw_lattice_diagnostic(text, 1) {
            Ok(r) => Some(r),
            Err(e) => {
                eprintln!("[brain] raw lattice skipped ({e}); using full generation path");
                None
            }
        };
        let mut q = self.query(text)?;
        let source = if let Some((c, raw)) = raw.as_ref().and_then(|r| r.candidates.first().map(|c| (c, r))) {
            if raw_candidate_usable(c, raw, text) {
                q.memory_text = lattice_answer_from_candidate(c);
                q.memory_template_id = format!("raw_lattice_prog_{}", c.prog_idx);
                q.memory_confidence = c.score;
                MemorySource::RawLattice
            } else {
                MemorySource::FullGeneration
            }
        } else {
            MemorySource::FullGeneration
        };
        Ok((q, source))
    }

    /// Classifier-routed query: the supervised label classifier chooses the topic,
    /// the brain retrieves an explanation from that topic's sub-lattice.
    ///
    /// The brain's own answer is kept when it is a non-domain reply the classifier
    /// can't know about (identity, greeting, tool calls, companion fallbacks), or
    /// when the classifier is less confident than `min_conf`.
    pub fn query_labeled(
        &mut self,
        text: &str,
        labeler: &crate::label_classifier::LabelClassifier,
        min_conf: f32,
    ) -> Result<(BrainMemoryQuery, MemorySource, crate::label_classifier::LabelScore), String> {
        use crate::label_classifier::{display_label, reply_has_label};
        let top = labeler.predict_top(text);
        let mut q = self.query(text)?;
        let tid = q.memory_template_id.to_ascii_lowercase();
        let brain_owns = ["greeting", "identity", "tool_call", "pet_", "lookup_graph"]
            .iter()
            .any(|p| tid.starts_with(p));
        if brain_owns || top.prob < min_conf {
            return Ok((q, MemorySource::FullGeneration, top));
        }
        // Pull the label's stored examples and keep the one whose *stored prompt*
        // is lexically closest to this prompt, so the explanation fits the input
        // (the brain's own embedding ranks nearly every example alike within a topic).
        let forced = self
            .raw_lattice_diagnostic_with_force_topic(text, 256, Some(&top.label))
            .ok()
            .and_then(|r| {
                r.candidates
                    .into_iter()
                    .filter(|c| !c.hard_reject)
                    .map(|c| {
                        let sim = labeler.similarity(text, &stored_prompt_of_candidate(&c));
                        (sim, c)
                    })
                    .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(_, c)| c)
            });
        let body = match &forced {
            Some(c) => lattice_answer_from_candidate(c),
            None => String::new(),
        };
        let sentiment = labeler.is_sentiment_scheme();
        q.memory_text = if body.trim().is_empty() {
            if sentiment {
                format!(
                    "{} — classified from the wording; no stored explanation for this label yet.",
                    display_label(&top.label)
                )
            } else {
                // Non-sentiment scheme with nothing stored: keep the brain's own reply.
                return Ok((q, MemorySource::FullGeneration, top));
            }
        } else if sentiment && !reply_has_label(&body) {
            format!("{} — {}", display_label(&top.label), body.trim())
        } else {
            body.trim().to_string()
        };
        q.memory_template_id = match &forced {
            Some(c) => format!("label_forced:{}:prog_{}", top.label, c.prog_idx),
            None => format!("label_forced:{}:none", top.label),
        };
        q.memory_confidence = top.prob;
        Ok((q, MemorySource::LabelForced, top))
    }

    /// Route + retrieve a lattice memory unit for `text` (full generation path: metacog + gates).
    pub fn query(&mut self, text: &str) -> Result<BrainMemoryQuery, String> {
        let bridged = self
            .runtime
            .svc
            .active_dm()
            .language_runtime
            .bridge_text_stateless(text)?;
        let route = self
            .runtime
            .svc
            .active_dm_mut()
            .route_text_stateless(text)?;
        let (action, resp) = self.runtime.svc.generation(text)?;
        Ok(BrainMemoryQuery {
            bridge_vector: bridged.routed_vector,
            bridge_confidence: bridged.confidence,
            group_id: route.chosen_group_id,
            route_margin: route.margin,
            route_confidence: route.confidence,
            route_rejected_ood: route.rejected_as_ood,
            memory_text: resp.text,
            memory_template_id: resp.template_id,
            memory_confidence: resp.confidence,
            action_type: format!("{:?}", action.action_type),
            action_confidence: action.confidence,
        })
    }

    /// Pre-gate lattice top-K: same routing/topic/subject setup as generation, but no metacog
    /// or grounding gate. Use this to decide gate bug vs core retrieval bug.
    pub fn raw_lattice_diagnostic(
        &mut self,
        text: &str,
        top_k: usize,
    ) -> Result<RawLatticeDiagnosticReport, String> {
        self.runtime.raw_lattice_diagnostic(text, top_k)
    }

    pub fn raw_lattice_diagnostic_with_force_topic(
        &mut self,
        text: &str,
        top_k: usize,
        force_topic: Option<&str>,
    ) -> Result<RawLatticeDiagnosticReport, String> {
        self.runtime
            .raw_lattice_diagnostic_with_force_topic(text, top_k, force_topic)
    }
}

/// JSON-serializable mirror of [`RawLatticeDiagnosticReport`] (avoids cross-crate serde mismatch).
#[derive(serde::Serialize)]
pub struct RawLatticeDiagnosticJson {
    pub prompt: String,
    pub group_idx: Option<usize>,
    pub topic_hint: Option<String>,
    pub subject_keywords: Vec<String>,
    pub forced_topic: Option<String>,
    pub retrieval_path: String,
    pub candidates: Vec<RawLatticeCandidateJson>,
}

#[derive(serde::Serialize)]
pub struct RawLatticeCandidateJson {
    pub rank: usize,
    pub prog_idx: usize,
    pub score: f32,
    pub topic: String,
    pub text_preview: String,
    pub witness_ok: bool,
    pub hard_reject: bool,
    pub soft_reject: bool,
    pub graph_confident: bool,
    pub above_score_floor: bool,
}

impl From<&RawLatticeDiagnosticReport> for RawLatticeDiagnosticJson {
    fn from(r: &RawLatticeDiagnosticReport) -> Self {
        Self {
            prompt: r.prompt.clone(),
            group_idx: r.group_idx,
            topic_hint: r.topic_hint.clone(),
            subject_keywords: r.subject_keywords.clone(),
            forced_topic: r.forced_topic.clone(),
            retrieval_path: r.retrieval_path.clone(),
            candidates: r
                .candidates
                .iter()
                .map(|c| RawLatticeCandidateJson {
                    rank: c.rank,
                    prog_idx: c.prog_idx,
                    score: c.score,
                    topic: c.topic.clone(),
                    text_preview: c.text_preview.clone(),
                    witness_ok: c.witness_ok,
                    hard_reject: c.hard_reject,
                    soft_reject: c.soft_reject,
                    graph_confident: c.graph_confident,
                    above_score_floor: c.above_score_floor,
                })
                .collect(),
        }
    }
}

pub fn raw_lattice_report_json(report: &RawLatticeDiagnosticReport) -> Result<String, String> {
    let j = RawLatticeDiagnosticJson::from(report);
    serde_json::to_string_pretty(&j).map_err(|e| e.to_string())
}

/// Prefix string for LM conditioning: brain routing metadata + retrieved lattice text.
pub fn format_lm_memory_prefix(q: &BrainMemoryQuery) -> String {
    format_lm_memory_prefix_with_source(q, MemorySource::FullGeneration)
}

pub fn format_lm_memory_prefix_with_source(q: &BrainMemoryQuery, source: MemorySource) -> String {
    format!(
        "[brain memory={} route group={} margin={:.3} action={} mem_conf={:.2}]\n{}\n\n",
        source.as_str(),
        q.group_id
            .map(|g| g.to_string())
            .unwrap_or_else(|| "none".into()),
        q.route_margin,
        q.action_type,
        q.memory_confidence,
        q.memory_text.trim()
    )
}

/// Eight scalars: bridge conf, route margin/conf, mem conf, action conf, 3 bridge slices.
pub fn brain_router_features(q: &BrainMemoryQuery) -> [f32; BRAIN_FEATURE_DIM] {
    let v = &q.bridge_vector;
    let s = |i: usize| v.get(i).copied().unwrap_or(0.0);
    let mid = v.len() / 2;
    [
        q.bridge_confidence,
        q.route_margin,
        q.route_confidence,
        q.memory_confidence,
        q.action_confidence,
        s(0),
        s(mid),
        s(v.len().saturating_sub(1)),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn brain_features_len() {
        let q = BrainMemoryQuery {
            bridge_vector: vec![0.1; growformer::dimension::language::DEFAULT_BRIDGE_DIM],
            bridge_confidence: 0.5,
            group_id: Some(1),
            route_margin: 0.2,
            route_confidence: 0.8,
            memory_text: "hello".into(),
            memory_template_id: "t".into(),
            memory_confidence: 0.9,
            action_type: "GeneralAssist".into(),
            action_confidence: 0.7,
            route_rejected_ood: false,
        };
        assert_eq!(brain_router_features(&q).len(), BRAIN_FEATURE_DIM);
        assert!(format_lm_memory_prefix(&q).contains("hello"));
        assert!(
            format_lm_memory_prefix_with_source(&q, MemorySource::RawLattice)
                .contains("raw_lattice")
        );
    }
}
