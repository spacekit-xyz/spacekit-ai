//! Self-revising grounding loop (assisted maintenance): capture routing failures,
//! propose alias / new-node edits, collision-check across fleet graphs, certify
//! held-out generalization before integration. See `docs/GROUNDING_LOOP_SPEC.md`.

use std::collections::{HashMap, HashSet};
use std::sync::{Mutex, OnceLock, RwLock};
#[cfg(not(target_arch = "wasm32"))]
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::dimension::embedding::cosine_similarity;
use crate::dimension::language::LanguageRuntime;
use crate::inference::world_grounding::{
    activated_root_ids, activated_root_ids_in_domain_graph, fleet_node_inventory,
    GroundingFleetDomain, GroundingNodeInfo,
};
use crate::spectral::TokenDictionary;
use crate::text_autoencoder::ChunkCodec;

/// Tunable thresholds for propose / certify / collision (§3–§6).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GroundingLoopParams {
    pub tau_alias: f32,
    pub tau_cluster: f32,
    pub k_min: usize,
    pub activation_threshold: f32,
    pub low_confidence_threshold: f32,
    pub collision_threshold: f32,
    pub max_generalization_gap: f32,
    pub min_held_out_lift: f32,
    pub max_cross_domain_misroute_lift: f32,
}

impl Default for GroundingLoopParams {
    fn default() -> Self {
        Self {
            tau_alias: 0.42,
            tau_cluster: 0.55,
            k_min: 3,
            activation_threshold: 0.40,
            low_confidence_threshold: 0.25,
            collision_threshold: 0.58,
            max_generalization_gap: 0.15,
            min_held_out_lift: 0.05,
            max_cross_domain_misroute_lift: 0.02,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FailureTrigger {
    EntropyGuard,
    NoNodeActivated,
    LowConfidence,
    Dissatisfaction,
}

impl FailureTrigger {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::EntropyGuard => "entropy_guard",
            Self::NoNodeActivated => "no_node",
            Self::LowConfidence => "low_confidence",
            Self::Dissatisfaction => "dissatisfaction",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CaptureSplit {
    Propose,
    Certify,
}

impl CaptureSplit {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Propose => "propose",
            Self::Certify => "certify",
        }
    }
}

/// Where a phrase came from — the provenance contract behind the augmentation firewall (§4).
/// `RealTraffic` is the only provenance allowed in the certify set; `Augmented` carries the
/// lineage of the phrase(s) it was derived from so closed-loop self-certification is detectable.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProvenanceKind {
    RealTraffic,
    Authored,
    Augmented,
}

impl Default for ProvenanceKind {
    fn default() -> Self {
        ProvenanceKind::RealTraffic
    }
}

impl ProvenanceKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::RealTraffic => "real_traffic",
            Self::Authored => "authored",
            Self::Augmented => "augmented",
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PhraseProvenance {
    pub kind: ProvenanceKind,
    pub phrase_id: String,
    /// For `Augmented` phrases: ids of the phrases this was generated from.
    pub derived_from: Vec<String>,
}

impl PhraseProvenance {
    pub fn real(phrase_id: impl Into<String>) -> Self {
        Self {
            kind: ProvenanceKind::RealTraffic,
            phrase_id: phrase_id.into(),
            derived_from: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FailureCapture {
    pub phrase: String,
    pub encoder_embedding: Vec<f32>,
    pub activated_nodes: Vec<(String, f32)>,
    pub max_confidence: f32,
    pub entropy_bits: Option<f32>,
    pub trigger_reason: FailureTrigger,
    pub downstream_signal: Option<String>,
    pub timestamp_unix: u64,
    pub domain_context: String,
    pub inferred_concept_id: String,
    pub split: CaptureSplit,
    #[serde(default)]
    pub provenance: PhraseProvenance,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ProposalKind {
    Alias {
        phrase: String,
        target_node: String,
        target_domain: String,
        similarity: f32,
        margin: f32,
    },
    NewNode {
        phrases: Vec<String>,
        suggested_parent: Option<String>,
        domain: String,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EditProposal {
    pub kind: ProposalKind,
    pub collision_score: f32,
    pub collision_conflicts: Vec<(String, String, f32)>,
    pub pre_certify_held_out_estimate: f32,
    pub approved: bool,
    pub integrated: bool,
}

#[derive(Clone, Debug)]
struct EmbeddedNode {
    domain: GroundingFleetDomain,
    node_id: String,
    aliases: Vec<String>,
    centroid: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct GroundingNodeIndex {
    domains: Vec<(GroundingFleetDomain, Vec<EmbeddedNode>)>,
    activation_threshold: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CertifierMetrics {
    pub held_out_accuracy: f32,
    pub captured_accuracy: f32,
    pub generalization_gap: f32,
    pub cross_domain_misroute_rate: f32,
    pub alias_additions: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BatchVerdict {
    GenuineCoverageImprovement,
    LexiconMemorization,
    NetNegativeCollision,
    Saturation,
    InsufficientData,
}

impl BatchVerdict {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::GenuineCoverageImprovement => "genuine_coverage_improvement",
            Self::LexiconMemorization => "lexicon_memorization",
            Self::NetNegativeCollision => "net_negative_collision",
            Self::Saturation => "saturation",
            Self::InsufficientData => "insufficient_data",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FixtureRow {
    pub phrase: String,
    pub concept_id: String,
    pub split: CaptureSplit,
    pub domain_context: String,
}

static CAPTURE_LOG: OnceLock<Mutex<Vec<FailureCapture>>> = OnceLock::new();

fn capture_log() -> &'static Mutex<Vec<FailureCapture>> {
    CAPTURE_LOG.get_or_init(|| Mutex::new(Vec::new()))
}

pub fn append_capture(capture: FailureCapture) {
    if let Ok(mut g) = capture_log().lock() {
        g.push(capture);
    }
}

pub fn drain_capture_log() -> Vec<FailureCapture> {
    capture_log()
        .lock()
        .map(|mut g| std::mem::take(&mut *g))
        .unwrap_or_default()
}

pub fn capture_log_len() -> usize {
    capture_log().lock().map(|g| g.len()).unwrap_or(0)
}

#[cfg(not(target_arch = "wasm32"))]
fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// `std::time::SystemTime::now()` panics on wasm32-unknown-unknown ("time not
// implemented on this platform"), which aborts the live `converse` path inside
// `capture_lightweight`. The browser records traffic with real timestamps in JS
// (GrowformerTrafficCapture), so the engine-side capture record can use 0 here
// without losing any information.
#[cfg(target_arch = "wasm32")]
fn now_unix() -> u64 {
    0
}

/// Which representation the grounding loop measures similarity on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RepresentationMode {
    /// Raw CliffordE8 encoder output (dictionary-backed; collapses to zero without a corpus dict).
    RawCliffordE8,
    /// Learned bridge routing vector (`project`); what the live router/entropy guard see.
    BridgeRouted,
}

/// Pluggable phrase embedder for the grounding loop.
///
/// - `Cata`: the **pre-quantization** CATA centroid (clifford_e8 family). Non-degenerate
///   but *lexical* — `build_token_embeddings` gives each token id an independent random
///   unit vector, so it carries shared-token overlap only, no cross-token semantics. This
///   is the regime in which the certifier should expect memorization, not generalization.
/// - `Vectors`: **bring-your-own** precomputed embeddings (a real/semantic encoder run
///   offline over the captured phrases + node aliases). Look up by normalized key; unknown
///   text falls back to a deterministic per-string unit vector (a stable distractor, never
///   zero) so missing-vocabulary never silently collapses to the tie-break artifact.
///   This is the path you use to evaluate a candidate semantic encoder against the
///   coverage-vs-additions curve (§6) without a live model dependency.
enum PhraseEmbedder {
    Cata(TokenDictionary, ChunkCodec),
    Vectors {
        dim: usize,
        map: HashMap<String, Vec<f32>>,
    },
    Supervised(SupervisedEncoder),
}

/// A small **supervised** phrase encoder trained on labeled captures (no external deps).
///
/// Motivation: the lexical CATA centroid carries only shared-token overlap, so held-out
/// paraphrases that share no tokens with a concept's existing aliases misroute. When the
/// author already has labeled utterances (`text → concept/intent`, e.g. a companion's
/// `semantic_intent` field), we can *learn* which hashed lexical features co-indicate a
/// concept. The embedding is the softmax distribution over learned concepts: paraphrases
/// of the same concept land near each other even with disjoint surface tokens, which is
/// exactly the non-lexical structure the certifier's genuine row requires — without
/// importing a sentence-transformer.
///
/// Features: hashed word unigrams + word bigrams + intra-word char trigrams (FNV buckets,
/// L2-normalized). Model: one linear softmax layer trained by SGD with L2 regularization.
#[derive(Clone, Debug)]
pub struct SupervisedEncoder {
    n_buckets: usize,
    labels: Vec<String>,
    /// `[label][bucket]` weight matrix.
    w: Vec<Vec<f32>>,
}

fn feat_bucket(s: &str, n_buckets: usize) -> usize {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    (h % n_buckets as u64) as usize
}

/// The single source of truth for the encoder's lexical features. Emits each feature
/// **string** (prefixed `w:` / `b:` / `c:`) once per occurrence: word unigrams, word
/// bigrams, and intra-word char trigrams. Both the encoder (`text_features`, which hashes
/// these) and the disjointness filter (`phrase_feature_set`) route through this function,
/// so "token-disjoint" is defined over exactly the features the encoder actually uses.
fn for_each_feature<F: FnMut(&str)>(text: &str, mut f: F) {
    let words: Vec<String> = text
        .to_ascii_lowercase()
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(|w| w.to_string())
        .collect();
    for (i, w) in words.iter().enumerate() {
        f(&format!("w:{w}"));
        if i + 1 < words.len() {
            f(&format!("b:{w}_{}", words[i + 1]));
        }
        let padded = format!("^{w}$");
        let chars: Vec<char> = padded.chars().collect();
        if chars.len() >= 3 {
            for win in chars.windows(3) {
                f(&format!("c:{}{}{}", win[0], win[1], win[2]));
            }
        }
    }
}

fn text_features(text: &str, n_buckets: usize) -> Vec<(usize, f32)> {
    let mut counts: HashMap<usize, f32> = HashMap::new();
    for_each_feature(text, |key| {
        let b = feat_bucket(key, n_buckets);
        *counts.entry(b).or_insert(0.0) += 1.0;
    });
    let mut v: Vec<(usize, f32)> = counts.into_iter().collect();
    let norm: f64 = v
        .iter()
        .map(|(_, x)| (*x as f64) * (*x as f64))
        .sum::<f64>()
        .sqrt();
    if norm > 1e-12 {
        for (_, x) in v.iter_mut() {
            *x = (*x as f64 / norm) as f32;
        }
    }
    v
}

/// The set of encoder feature strings for a phrase (`w:`/`b:`/`c:` prefixed). This is the
/// granularity at which token-disjointness is defined: two phrases are feature-disjoint
/// iff these sets do not intersect — so "vacuum"/"vacuuming" are correctly *not* disjoint
/// (they share char trigrams), even though they are whole-word disjoint.
pub fn phrase_feature_set(text: &str) -> HashSet<String> {
    let mut s = HashSet::new();
    for_each_feature(text, |k| {
        s.insert(k.to_string());
    });
    s
}

/// Restrict a feature set to a granularity: `"w"` = words only, `"wb"` = words+bigrams,
/// `"wbc"` (or anything else) = the full union the encoder consumes.
pub fn restrict_features(features: &HashSet<String>, level: &str) -> HashSet<String> {
    features
        .iter()
        .filter(|k| match level {
            "w" => k.starts_with("w:"),
            "wb" => k.starts_with("w:") || k.starts_with("b:"),
            _ => true,
        })
        .cloned()
        .collect()
}

/// `|F(p) ∩ train| / |F(p)|` — fraction of a phrase's features seen (in training) tied to
/// a given concept. 0 ⇒ the encoder had no surface handle on this phrase for that concept.
pub fn feature_overlap_fraction(p: &HashSet<String>, train: &HashSet<String>) -> f32 {
    if p.is_empty() {
        return 0.0;
    }
    let inter = p.iter().filter(|k| train.contains(*k)).count();
    inter as f32 / p.len() as f32
}

/// Wilson score interval for a binomial proportion (z e.g. 1.96 for 95%). Returns
/// `(lo, hi)`; small-n bins get honestly wide intervals rather than a bare point.
pub fn wilson_interval(hits: usize, n: usize, z: f64) -> (f64, f64) {
    if n == 0 {
        return (0.0, 1.0);
    }
    let nf = n as f64;
    let phat = hits as f64 / nf;
    let z2 = z * z;
    let denom = 1.0 + z2 / nf;
    let center = (phat + z2 / (2.0 * nf)) / denom;
    let margin = (z * ((phat * (1.0 - phat) + z2 / (4.0 * nf)) / nf).sqrt()) / denom;
    ((center - margin).max(0.0), (center + margin).min(1.0))
}

impl SupervisedEncoder {
    /// Train on `(text, label)` pairs. Returns `None` if there are fewer than 2 labels.
    /// Uses a fixed default seed; for reproducible certifier runs use [`train_seeded`].
    pub fn train(samples: &[(String, String)], n_buckets: usize, epochs: usize) -> Option<Self> {
        Self::train_seeded(samples, n_buckets, epochs, 0x243F6A8885A308D3)
    }

    /// Seeded training: the SGD shuffle RNG is seeded from `seed`, so the certifier pipeline
    /// is deterministic given `(data, seed)` — the same inputs always produce the same weights,
    /// which is what makes the verdict artifact reproducible across runs.
    pub fn train_seeded(
        samples: &[(String, String)],
        n_buckets: usize,
        epochs: usize,
        seed: u64,
    ) -> Option<Self> {
        let mut labels: Vec<String> = Vec::new();
        for (_, l) in samples {
            if !labels.contains(l) {
                labels.push(l.clone());
            }
        }
        if labels.len() < 2 {
            return None;
        }
        let label_ix: HashMap<&str, usize> = labels
            .iter()
            .enumerate()
            .map(|(i, l)| (l.as_str(), i))
            .collect();
        let n_labels = labels.len();
        let feats: Vec<(Vec<(usize, f32)>, usize)> = samples
            .iter()
            .map(|(t, l)| (text_features(t, n_buckets), label_ix[l.as_str()]))
            .collect();

        let mut w = vec![vec![0.0f32; n_buckets]; n_labels];
        let l2 = 1e-5f32;
        let mut order: Vec<usize> = (0..feats.len()).collect();
        let mut rng: u64 = if seed == 0 { 0x9E3779B97F4A7C15 } else { seed };
        for ep in 0..epochs {
            // Deterministic xorshift shuffle (no rand dependency).
            for i in (1..order.len()).rev() {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                let j = (rng % (i as u64 + 1)) as usize;
                order.swap(i, j);
            }
            let lr = 0.5f32 * (1.0 - ep as f32 / (epochs as f32 + 1.0));
            for &si in &order {
                let (f, y) = &feats[si];
                let mut logits = vec![0.0f32; n_labels];
                for (k, lg) in logits.iter_mut().enumerate() {
                    let row = &w[k];
                    let mut s = 0.0f32;
                    for (b, v) in f {
                        s += row[*b] * v;
                    }
                    *lg = s;
                }
                let maxl = logits.iter().cloned().fold(f32::MIN, f32::max);
                let mut sum = 0.0f32;
                for lg in logits.iter_mut() {
                    *lg = (*lg - maxl).exp();
                    sum += *lg;
                }
                for lg in logits.iter_mut() {
                    *lg /= sum;
                }
                for k in 0..n_labels {
                    let err = logits[k] - if k == *y { 1.0 } else { 0.0 };
                    let row = &mut w[k];
                    for (b, v) in f {
                        row[*b] -= lr * (err * v + l2 * row[*b]);
                    }
                }
            }
        }
        Some(Self {
            n_buckets,
            labels,
            w,
        })
    }

    pub fn n_labels(&self) -> usize {
        self.labels.len()
    }

    /// Embed text as the L2-normalized softmax distribution over learned concepts.
    pub fn embed(&self, text: &str) -> Vec<f32> {
        let f = text_features(text, self.n_buckets);
        let mut logits: Vec<f32> = self
            .w
            .iter()
            .map(|row| f.iter().map(|(b, v)| row[*b] * v).sum())
            .collect();
        let maxl = logits.iter().cloned().fold(f32::MIN, f32::max);
        let mut sum = 0.0f32;
        for lg in logits.iter_mut() {
            *lg = (*lg - maxl).exp();
            sum += *lg;
        }
        for lg in logits.iter_mut() {
            *lg /= sum;
        }
        l2_normalize_in_place(&mut logits);
        logits
    }
}

/// Install a trained supervised encoder as the active phrase embedder.
pub fn install_supervised_embedder(enc: SupervisedEncoder) -> usize {
    let n = enc.n_labels();
    if let Ok(mut g) = phrase_embedder().write() {
        *g = Some(PhraseEmbedder::Supervised(enc));
    }
    n
}

static PHRASE_EMBEDDER: OnceLock<RwLock<Option<PhraseEmbedder>>> = OnceLock::new();

fn phrase_embedder() -> &'static RwLock<Option<PhraseEmbedder>> {
    PHRASE_EMBEDDER.get_or_init(|| RwLock::new(None))
}

fn normalize_phrase_key(text: &str) -> String {
    text.trim().to_ascii_lowercase()
}

/// Install a CATA phrase embedder built from a domain corpus. Returns dictionary size.
pub fn install_phrase_embedder_from_corpus(corpus: &[&str], max_dict: usize) -> usize {
    let dict = TokenDictionary::build(corpus, max_dict);
    let n = dict.len();
    let codec = ChunkCodec::new(dict.len().max(1));
    if let Ok(mut g) = phrase_embedder().write() {
        *g = Some(PhraseEmbedder::Cata(dict, codec));
    }
    n
}

/// Install a precomputed-vector embedder (bring-your-own semantic encoder). `map` keys
/// are phrase/alias strings (case/space-insensitive); values are equal-length vectors.
pub fn install_vector_embedder(map: HashMap<String, Vec<f32>>) -> usize {
    let dim = map.values().next().map(|v| v.len()).unwrap_or(0);
    let normalized: HashMap<String, Vec<f32>> = map
        .into_iter()
        .map(|(k, v)| (normalize_phrase_key(&k), v))
        .collect();
    let n = normalized.len();
    if let Ok(mut g) = phrase_embedder().write() {
        *g = Some(PhraseEmbedder::Vectors {
            dim,
            map: normalized,
        });
    }
    n
}

pub fn clear_phrase_embedder() {
    if let Ok(mut g) = phrase_embedder().write() {
        *g = None;
    }
}

pub fn phrase_embedder_active() -> bool {
    phrase_embedder()
        .read()
        .map(|g| g.is_some())
        .unwrap_or(false)
}

/// Deterministic per-string unit vector (splitmix64) — a stable distractor for
/// vocabulary the BYO embedder did not cover. Never zero.
fn deterministic_unit_vector(text: &str, dim: usize) -> Vec<f32> {
    let dim = dim.max(1);
    let mut state: u64 = 0x9E3779B97F4A7C15;
    for b in text.as_bytes() {
        state ^= *b as u64;
        state = state.wrapping_mul(0x100000001B3);
    }
    let mut out = vec![0.0f32; dim];
    for x in out.iter_mut() {
        state ^= state >> 30;
        state = state.wrapping_mul(0xBF58476D1CE4E5B9);
        state ^= state >> 27;
        let u = ((state >> 11) as f64) / ((1u64 << 53) as f64);
        *x = (u as f32) - 0.5;
    }
    l2_normalize_in_place(&mut out);
    out
}

fn embedder_lookup(text: &str) -> Option<Vec<f32>> {
    let g = phrase_embedder().read().ok()?;
    match g.as_ref()? {
        PhraseEmbedder::Cata(dict, codec) => {
            let seq = codec.encode_text(text, dict);
            let mut c = seq.centroid().to_vec();
            l2_normalize_in_place(&mut c);
            if c.iter().all(|x| x.abs() < 1e-12) {
                return None;
            }
            Some(c)
        }
        PhraseEmbedder::Vectors { dim, map } => {
            if let Some(v) = map.get(&normalize_phrase_key(text)) {
                let mut v = v.clone();
                l2_normalize_in_place(&mut v);
                Some(v)
            } else {
                Some(deterministic_unit_vector(text, *dim))
            }
        }
        PhraseEmbedder::Supervised(enc) => Some(enc.embed(text)),
    }
}

/// Embed a phrase for grounding similarity.
///
/// Cascade (see GROUNDING_LOOP_SPEC review):
/// 1. installed phrase embedder (CATA centroid, or bring-your-own vectors) — preferred;
/// 2. raw CliffordE8 (only meaningful with a codec dictionary; collapses otherwise);
/// 3. learned bridge routing vector (layer-normed, cannot collapse) as the last resort.
pub fn embed_phrase(rt: &LanguageRuntime, text: &str) -> Result<(Vec<f32>, f32), String> {
    if let Some(c) = embedder_lookup(text) {
        return Ok((c, 1.0));
    }
    embed_phrase_mode(rt, text, RepresentationMode::RawCliffordE8)
}

pub fn embed_phrase_mode(
    rt: &LanguageRuntime,
    text: &str,
    mode: RepresentationMode,
) -> Result<(Vec<f32>, f32), String> {
    let (raw, bridged) = rt.encode_and_bridge(text)?;
    let mut v = match mode {
        RepresentationMode::RawCliffordE8 => raw,
        RepresentationMode::BridgeRouted => bridged.routed_vector.clone(),
    };
    l2_normalize_in_place(&mut v);
    let degenerate = v.is_empty() || v.iter().all(|x| x.abs() < 1e-12);
    if degenerate && mode == RepresentationMode::RawCliffordE8 {
        let mut routed = bridged.routed_vector.clone();
        l2_normalize_in_place(&mut routed);
        return Ok((routed, bridged.confidence));
    }
    Ok((v, bridged.confidence))
}

/// True when the runtime has a corpus dictionary (so the CliffordE8 codec path is non-degenerate).
pub fn runtime_has_dictionary(rt: &LanguageRuntime) -> bool {
    rt.preloaded_dictionary.is_some()
}

fn mean_embedding(vectors: &[Vec<f32>]) -> Vec<f32> {
    if vectors.is_empty() {
        return Vec::new();
    }
    let dim = vectors[0].len();
    let mut acc = vec![0.0f64; dim];
    for v in vectors {
        for (i, &x) in v.iter().enumerate() {
            if i < dim {
                acc[i] += x as f64;
            }
        }
    }
    let n = vectors.len() as f64;
    let mut out: Vec<f32> = acc.iter().map(|x| (*x / n) as f32).collect();
    l2_normalize_in_place(&mut out);
    out
}

fn l2_normalize_in_place(v: &mut [f32]) {
    let norm: f64 = v
        .iter()
        .map(|&x| (x as f64) * (x as f64))
        .sum::<f64>()
        .sqrt();
    if norm > 1e-12 {
        for x in v.iter_mut() {
            *x = (*x as f64 / norm) as f32;
        }
    }
}

/// Build per-domain embedding centroids from fleet inventory + optional extra aliases.
pub fn build_grounding_index(
    rt: &LanguageRuntime,
    extra_aliases: &HashMap<(GroundingFleetDomain, String), Vec<String>>,
    params: &GroundingLoopParams,
) -> Result<GroundingNodeIndex, String> {
    let inventory = fleet_node_inventory();
    let mut by_domain: HashMap<GroundingFleetDomain, Vec<GroundingNodeInfo>> = HashMap::new();
    for node in inventory {
        by_domain.entry(node.domain).or_default().push(node);
    }

    let mut domains = Vec::new();
    for domain in [
        GroundingFleetDomain::Base,
        GroundingFleetDomain::Crypto,
        GroundingFleetDomain::Fintech,
        GroundingFleetDomain::Runtime,
    ] {
        let Some(nodes) = by_domain.remove(&domain) else {
            continue;
        };
        let mut embedded = Vec::new();
        for node in nodes {
            let mut alias_keys: Vec<String> = vec![node.node_id.clone()];
            alias_keys.extend(node.aliases.clone());
            if let Some(extra) = extra_aliases.get(&(domain, node.node_id.clone())) {
                alias_keys.extend(extra.iter().cloned());
            }
            alias_keys.sort();
            alias_keys.dedup();

            let mut vecs = Vec::new();
            for a in &alias_keys {
                let (emb, _) = embed_phrase(rt, a)?;
                vecs.push(emb);
            }
            let centroid = mean_embedding(&vecs);
            embedded.push(EmbeddedNode {
                domain,
                node_id: node.node_id,
                aliases: alias_keys,
                centroid,
            });
        }
        if !embedded.is_empty() {
            domains.push((domain, embedded));
        }
    }

    Ok(GroundingNodeIndex {
        domains,
        activation_threshold: params.activation_threshold,
    })
}

/// Build an index from an explicit node set (domain, node_id, alias strings). Used by
/// the positive control and by any caller that wants a self-contained concept space
/// instead of the loaded fleet. Aliases are embedded via `embed_phrase` (so the active
/// phrase embedder governs the geometry).
pub fn build_grounding_index_from_nodes(
    rt: &LanguageRuntime,
    nodes: &[(GroundingFleetDomain, String, Vec<String>)],
    params: &GroundingLoopParams,
) -> Result<GroundingNodeIndex, String> {
    build_grounding_index_from_nodes_ex(rt, nodes, params, None)
}

/// Build a grounding index, optionally enriching centroids with training phrase embeddings.
///
/// When `training_pairs` is provided, the centroid for each concept is the mean of
/// (a) the node ID and alias embeddings, plus (b) all training phrase embeddings for
/// that concept. This is critical for frozen/pretrained encoders (BYO vectors), where
/// the node label alone is a poor prototype — a real nearest-centroid classifier uses
/// all available labeled examples, not just the concept name.
pub fn build_grounding_index_from_nodes_ex(
    rt: &LanguageRuntime,
    nodes: &[(GroundingFleetDomain, String, Vec<String>)],
    params: &GroundingLoopParams,
    training_pairs: Option<&[(String, String)]>,
) -> Result<GroundingNodeIndex, String> {
    let mut by_domain: Vec<(GroundingFleetDomain, Vec<EmbeddedNode>)> = Vec::new();
    for (domain, node_id, aliases) in nodes {
        let mut alias_keys: Vec<String> = vec![node_id.clone()];
        alias_keys.extend(aliases.iter().cloned());
        alias_keys.sort();
        alias_keys.dedup();
        let mut vecs = Vec::new();
        for a in &alias_keys {
            let (emb, _) = embed_phrase(rt, a)?;
            vecs.push(emb);
        }
        if let Some(pairs) = training_pairs {
            for (phrase, concept_id) in pairs {
                if concept_id == node_id {
                    if let Ok((emb, _)) = embed_phrase(rt, phrase) {
                        vecs.push(emb);
                    }
                }
            }
        }
        let centroid = mean_embedding(&vecs);
        let node = EmbeddedNode {
            domain: *domain,
            node_id: node_id.clone(),
            aliases: alias_keys,
            centroid,
        };
        match by_domain.iter_mut().find(|(d, _)| d == domain) {
            Some((_, v)) => v.push(node),
            None => by_domain.push((*domain, vec![node])),
        }
    }
    Ok(GroundingNodeIndex {
        domains: by_domain,
        activation_threshold: params.activation_threshold,
    })
}

/// Suggested alias threshold from labeled captures: the midpoint between same-concept
/// and cross-concept node similarity. Replaces hand-picked `τ_alias` with a data-driven
/// crossover (recommendation: re-derive thresholds per encoder).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ThresholdSuggestion {
    pub same_concept_mean: f32,
    pub cross_concept_mean: f32,
    pub suggested_tau_alias: f32,
    pub samples: usize,
}

pub fn calibrate_alias_threshold(
    captures: &[FailureCapture],
    rt: &LanguageRuntime,
    index: &GroundingNodeIndex,
) -> Result<ThresholdSuggestion, String> {
    let mut same = Vec::new();
    let mut cross = Vec::new();
    for cap in captures {
        let (emb, _) = embed_phrase(rt, &cap.phrase)?;
        for node in index.all_nodes() {
            let sim = cosine_similarity(&emb, &node.centroid);
            if node.node_id == cap.inferred_concept_id {
                same.push(sim);
            } else {
                cross.push(sim);
            }
        }
    }
    let mean = |v: &[f32]| {
        if v.is_empty() {
            0.0
        } else {
            v.iter().sum::<f32>() / v.len() as f32
        }
    };
    let same_mean = mean(&same);
    let cross_mean = mean(&cross);
    Ok(ThresholdSuggestion {
        same_concept_mean: same_mean,
        cross_concept_mean: cross_mean,
        suggested_tau_alias: (same_mean + cross_mean) / 2.0,
        samples: same.len() + cross.len(),
    })
}

#[derive(Clone, Debug)]
pub struct NearestMatch {
    pub domain: GroundingFleetDomain,
    pub node_id: String,
    pub similarity: f32,
    pub second_similarity: f32,
}

/// §18.2 passive capture record: one live routing decision, tagged `RealTraffic`.
///
/// Deliberately label-free. Production traffic is unlabeled, and per §18.3 (the blind-label
/// rule) only later blind human adjudication may assign a ground-truth `semantic_intent`. This
/// records *what the router did*, never *what was correct* — using the routed node as a label
/// would make any downstream gate certify agreement-with-incumbent rather than correctness.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RoutingCapture {
    pub phrase: String,
    pub routed_node: String,
    pub domain: String,
    pub similarity: f32,
    pub second_similarity: f32,
    pub margin: f32,
    /// `similarity >= activation_threshold` — whether the router considered the node active
    /// (vs. an abstain-eligible low-confidence route). A sampling signal for triage, not a label.
    pub activated: bool,
    pub timestamp_unix: u64,
    pub session_id: String,
    pub provenance: PhraseProvenance,
}

impl RoutingCapture {
    pub fn to_jsonl(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|e| format!("{{\"error\":\"{e}\"}}"))
    }
}

/// §18.2 persistence: append one routing capture as a JSONL line to
/// `<dir>/routing_<domain>.jsonl` (append-only, one file per fleet domain). Safe for the live
/// service to call once per routing decision — it opens, appends, and closes, so concurrent
/// sessions interleave whole lines rather than corrupting partial writes. This is the in-process
/// hook the live Luna path calls; the `--capture-routing` CLI is a batch/replay path over the
/// same function.
pub fn append_routing_capture(cap: &RoutingCapture, dir: &std::path::Path) -> std::io::Result<()> {
    use std::io::Write;
    std::fs::create_dir_all(dir)?;
    let path = dir.join(format!("routing_{}.jsonl", cap.domain));
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    writeln!(f, "{}", cap.to_jsonl())
}

/// §18.2 live-traffic phrase capture: one real user prompt seen by the serving path
/// (`spacekit agent infer`), tagged `RealTraffic`.
///
/// This is the *scarce resource* — the real, unlabeled phrases the offline certifier later
/// batch-embeds (Phase 1C) and gates. The serving path here is brain `converse`, not the
/// grounding-index router, so there is no certified routing decision to record at capture time;
/// the routing/gating is computed offline against the certified encoder. The optional `response`
/// is the incumbent system's reply, kept as a triage/sampling signal only — per §18.3 it (and any
/// implicit-feedback derived from it) is never a label the gate reads.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrafficCapture {
    pub phrase: String,
    pub agent: String,
    pub response: Option<String>,
    pub timestamp_unix: u64,
    pub session_id: String,
    pub provenance: PhraseProvenance,
}

impl TrafficCapture {
    pub fn real(phrase: impl Into<String>, agent: impl Into<String>) -> Self {
        let phrase = phrase.into();
        Self {
            provenance: PhraseProvenance::real(phrase.clone()),
            phrase,
            agent: agent.into(),
            response: None,
            timestamp_unix: now_unix(),
            session_id: String::new(),
        }
    }

    pub fn to_jsonl(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|e| format!("{{\"error\":\"{e}\"}}"))
    }
}

/// Append a `TrafficCapture` as one JSONL line to `<dir>/traffic_<agent>.jsonl` (append-only).
/// Best-effort and side-effect-only: callers on the serving path must ignore the error so capture
/// can never break inference.
pub fn append_traffic_capture(cap: &TrafficCapture, dir: &std::path::Path) -> std::io::Result<()> {
    use std::io::Write;
    std::fs::create_dir_all(dir)?;
    let safe_agent: String = cap
        .agent
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    let safe_agent = if safe_agent.is_empty() {
        "unknown".to_string()
    } else {
        safe_agent
    };
    let path = dir.join(format!("traffic_{safe_agent}.jsonl"));
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    writeln!(f, "{}", cap.to_jsonl())
}

impl GroundingNodeIndex {
    fn all_nodes(&self) -> impl Iterator<Item = &EmbeddedNode> {
        self.domains.iter().flat_map(|(_, nodes)| nodes.iter())
    }

    pub fn nearest_in_domain(
        &self,
        embedding: &[f32],
        domain: GroundingFleetDomain,
    ) -> Option<NearestMatch> {
        let nodes = &self.domains.iter().find(|(d, _)| *d == domain)?.1;
        Self::nearest_among(embedding, domain, nodes.as_slice())
    }

    pub fn nearest_fleet_wide(&self, embedding: &[f32]) -> Option<NearestMatch> {
        let mut best: Option<NearestMatch> = None;
        for (domain, nodes) in &self.domains {
            if let Some(m) = Self::nearest_among(embedding, *domain, nodes.as_slice()) {
                let replace = best
                    .as_ref()
                    .map(|b| m.similarity > b.similarity)
                    .unwrap_or(true);
                if replace {
                    best = Some(m);
                }
            }
        }
        best
    }

    fn nearest_among(
        embedding: &[f32],
        domain: GroundingFleetDomain,
        nodes: &[EmbeddedNode],
    ) -> Option<NearestMatch> {
        let mut scored: Vec<(f32, &EmbeddedNode)> = nodes
            .iter()
            .map(|n| (cosine_similarity(embedding, &n.centroid), n))
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let (sim, node) = scored.first()?;
        let second = scored.get(1).map(|x| x.0).unwrap_or(0.0);
        Some(NearestMatch {
            domain,
            node_id: node.node_id.clone(),
            similarity: *sim,
            second_similarity: second,
        })
    }

    fn route_to_concept(&self, embedding: &[f32]) -> Option<(String, f32)> {
        let m = self.nearest_fleet_wide(embedding)?;
        Some((m.node_id, m.similarity))
    }

    /// Nearest node in a preferred domain slice, else fleet-wide.
    pub fn nearest_for_domain(
        &self,
        embedding: &[f32],
        preferred: GroundingFleetDomain,
    ) -> Option<NearestMatch> {
        self.nearest_in_domain(embedding, preferred)
            .or_else(|| self.nearest_fleet_wide(embedding))
    }

    /// §18.2 passive capture: route a live phrase and build a `RealTraffic` decision record.
    ///
    /// Pure measurement — does not mutate the index, does not gate anything, and assigns no
    /// label (see `RoutingCapture`). The caller persists it (I/O lives outside this module).
    /// Centroids must be the same training-enriched ones the certifier used (`*_ex`), or the
    /// captured decision is not the certified router's decision (§18.6 parity).
    pub fn capture_decision(
        &self,
        rt: &LanguageRuntime,
        phrase: &str,
        preferred: GroundingFleetDomain,
        session_id: impl Into<String>,
    ) -> Result<RoutingCapture, String> {
        let (emb, _) = embed_phrase(rt, phrase)?;
        let m = self
            .nearest_for_domain(&emb, preferred)
            .ok_or_else(|| "no nodes to route against".to_string())?;
        Ok(RoutingCapture {
            phrase: phrase.to_string(),
            routed_node: m.node_id,
            domain: m.domain.as_str().to_string(),
            similarity: m.similarity,
            second_similarity: m.second_similarity,
            margin: m.similarity - m.second_similarity,
            activated: m.similarity >= self.activation_threshold,
            timestamp_unix: now_unix(),
            session_id: session_id.into(),
            provenance: PhraseProvenance::real(phrase.to_string()),
        })
    }

    pub fn activated_node_scores(&self, embedding: &[f32]) -> Vec<(String, f32)> {
        let mut out: Vec<(String, f32)> = self
            .all_nodes()
            .map(|n| {
                (
                    format!("{}:{}", n.domain.as_str(), n.node_id),
                    cosine_similarity(embedding, &n.centroid),
                )
            })
            .filter(|(_, s)| *s >= self.activation_threshold)
            .collect();
        out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        out
    }

    pub fn add_alias_to_node(
        &mut self,
        domain: GroundingFleetDomain,
        node_id: &str,
        alias: &str,
        alias_embedding: Vec<f32>,
    ) -> bool {
        for (d, nodes) in &mut self.domains {
            if *d != domain {
                continue;
            }
            for node in nodes {
                if node.node_id == node_id {
                    let n = node.aliases.len().max(1) as f32;
                    if node.centroid.len() == alias_embedding.len() {
                        for (c, a) in node.centroid.iter_mut().zip(alias_embedding.iter()) {
                            *c = (*c * n + a) / (n + 1.0);
                        }
                        l2_normalize_in_place(&mut node.centroid);
                    }
                    node.aliases.push(alias.to_string());
                    return true;
                }
            }
        }
        false
    }
}

pub fn evaluate_failure_triggers(
    phrase: &str,
    bridge_confidence: f32,
    entropy_guard: bool,
    downstream_signal: Option<&str>,
    params: &GroundingLoopParams,
) -> Option<FailureTrigger> {
    if entropy_guard {
        return Some(FailureTrigger::EntropyGuard);
    }
    if downstream_signal.is_some() {
        return Some(FailureTrigger::Dissatisfaction);
    }
    let roots = activated_root_ids(phrase);
    let domain_roots = activated_root_ids_in_domain_graph(phrase);
    if roots.is_empty() && domain_roots.is_empty() {
        return Some(FailureTrigger::NoNodeActivated);
    }
    if bridge_confidence < params.low_confidence_threshold {
        return Some(FailureTrigger::LowConfidence);
    }
    None
}

pub fn maybe_capture_grounding_failure(
    phrase: &str,
    embedding: &[f32],
    bridge_confidence: f32,
    entropy_guard: bool,
    entropy_bits: Option<f32>,
    downstream_signal: Option<&str>,
    domain_context: &str,
    inferred_concept_id: &str,
    split: CaptureSplit,
    index: &GroundingNodeIndex,
    params: &GroundingLoopParams,
) -> Option<FailureCapture> {
    let trigger = evaluate_failure_triggers(
        phrase,
        bridge_confidence,
        entropy_guard,
        downstream_signal,
        params,
    )?;
    let capture = FailureCapture {
        phrase: phrase.to_string(),
        encoder_embedding: embedding.to_vec(),
        activated_nodes: index.activated_node_scores(embedding),
        max_confidence: bridge_confidence,
        entropy_bits,
        trigger_reason: trigger,
        downstream_signal: downstream_signal.map(|s| s.to_string()),
        timestamp_unix: now_unix(),
        domain_context: domain_context.to_string(),
        inferred_concept_id: inferred_concept_id.to_string(),
        split,
        provenance: PhraseProvenance::real(phrase),
    };
    append_capture(capture.clone());
    Some(capture)
}

/// Live-path capture without building the full embedding index (re-embed on analyze).
pub fn capture_lightweight(
    phrase: &str,
    bridge_confidence: f32,
    entropy_guard: bool,
    entropy_bits: Option<f32>,
    downstream_signal: Option<&str>,
    domain_context: &str,
    inferred_concept_id: &str,
    params: &GroundingLoopParams,
) -> Option<FailureCapture> {
    let trigger = evaluate_failure_triggers(
        phrase,
        bridge_confidence,
        entropy_guard,
        downstream_signal,
        params,
    )?;
    let capture = FailureCapture {
        phrase: phrase.to_string(),
        encoder_embedding: Vec::new(),
        activated_nodes: Vec::new(),
        max_confidence: bridge_confidence,
        entropy_bits,
        trigger_reason: trigger,
        downstream_signal: downstream_signal.map(|s| s.to_string()),
        timestamp_unix: now_unix(),
        domain_context: domain_context.to_string(),
        inferred_concept_id: inferred_concept_id.to_string(),
        split: CaptureSplit::Propose,
        provenance: PhraseProvenance::real(phrase),
    };
    append_capture(capture.clone());
    Some(capture)
}

pub fn propose_for_phrase(
    phrase: &str,
    embedding: &[f32],
    index: &GroundingNodeIndex,
    params: &GroundingLoopParams,
    preferred_domain: Option<GroundingFleetDomain>,
) -> Option<ProposalKind> {
    let _ = phrase;
    let m = preferred_domain
        .and_then(|d| index.nearest_for_domain(embedding, d))
        .or_else(|| index.nearest_fleet_wide(embedding))?;
    if m.similarity >= params.tau_alias {
        return Some(ProposalKind::Alias {
            phrase: phrase.to_string(),
            target_node: m.node_id,
            target_domain: m.domain.as_str().to_string(),
            similarity: m.similarity,
            margin: m.similarity - m.second_similarity,
        });
    }
    None
}

pub fn cluster_buffered_new_nodes(
    buffered: &[(String, Vec<f32>)],
    params: &GroundingLoopParams,
) -> Vec<Vec<usize>> {
    let mut clusters: Vec<Vec<usize>> = Vec::new();
    for i in 0..buffered.len() {
        let mut placed = false;
        for cluster in &mut clusters {
            let rep = cluster[0];
            let sim = cosine_similarity(&buffered[i].1, &buffered[rep].1);
            if sim >= params.tau_cluster {
                cluster.push(i);
                placed = true;
                break;
            }
        }
        if !placed {
            clusters.push(vec![i]);
        }
    }
    clusters
        .into_iter()
        .filter(|c| c.len() >= params.k_min)
        .collect()
}

pub fn propose_new_nodes_from_buffer(
    buffered: &[(String, Vec<f32>)],
    index: &GroundingNodeIndex,
    params: &GroundingLoopParams,
) -> Vec<ProposalKind> {
    let clusters = cluster_buffered_new_nodes(buffered, params);
    let mut out = Vec::new();
    for cluster in clusters {
        let phrases: Vec<String> = cluster.iter().map(|&i| buffered[i].0.clone()).collect();
        let centroid = mean_embedding(
            &cluster
                .iter()
                .map(|&i| buffered[i].1.clone())
                .collect::<Vec<_>>(),
        );
        let parent = index
            .nearest_fleet_wide(&centroid)
            .filter(|m| m.similarity >= params.tau_alias * 0.85)
            .map(|m| m.node_id);
        out.push(ProposalKind::NewNode {
            phrases,
            suggested_parent: parent,
            domain: "runtime".to_string(),
        });
    }
    out
}

pub fn collision_check(
    embedding: &[f32],
    target_domain: GroundingFleetDomain,
    target_node: &str,
    index: &GroundingNodeIndex,
    params: &GroundingLoopParams,
) -> (f32, Vec<(String, String, f32)>) {
    let mut conflicts = Vec::new();
    let mut max_foreign = 0.0f32;
    for (domain, nodes) in &index.domains {
        if *domain == target_domain {
            continue;
        }
        for node in nodes {
            let sim = cosine_similarity(embedding, &node.centroid);
            if sim >= params.collision_threshold {
                conflicts.push((domain.as_str().to_string(), node.node_id.clone(), sim));
                max_foreign = max_foreign.max(sim);
            }
        }
    }
    // Also flag if phrase routes to a different node in another domain above target.
    if let Some(m) = index.nearest_fleet_wide(embedding) {
        if m.domain != target_domain
            && m.node_id != target_node
            && m.similarity >= params.collision_threshold
        {
            conflicts.push((
                m.domain.as_str().to_string(),
                m.node_id.clone(),
                m.similarity,
            ));
            max_foreign = max_foreign.max(m.similarity);
        }
    }
    conflicts.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    (max_foreign, conflicts)
}

pub fn parse_fleet_domain_context(s: &str) -> GroundingFleetDomain {
    match s.trim().to_ascii_lowercase().as_str() {
        "base" => GroundingFleetDomain::Base,
        "fintech" | "tradfi" => GroundingFleetDomain::Fintech,
        "runtime" | "pet" | "domain" => GroundingFleetDomain::Runtime,
        _ => GroundingFleetDomain::Crypto,
    }
}

pub fn routing_accuracy_for_captures(
    captures: &[FailureCapture],
    rt: &LanguageRuntime,
    index: &GroundingNodeIndex,
    split: CaptureSplit,
) -> Result<f32, String> {
    let rows: Vec<&FailureCapture> = captures.iter().filter(|c| c.split == split).collect();
    if rows.is_empty() {
        return Ok(0.0);
    }
    let mut hits = 0usize;
    for cap in &rows {
        let (emb, _) = embed_phrase(rt, &cap.phrase)?;
        let domain = parse_fleet_domain_context(&cap.domain_context);
        let hit = index
            .nearest_in_domain(&emb, domain)
            .map(|m| m.node_id == cap.inferred_concept_id)
            .unwrap_or(false);
        if hit {
            hits += 1;
        }
    }
    Ok(hits as f32 / rows.len() as f32)
}

pub fn routing_accuracy(
    rows: &[(String, String, CaptureSplit)],
    rt: &LanguageRuntime,
    index: &GroundingNodeIndex,
) -> Result<f32, String> {
    if rows.is_empty() {
        return Ok(0.0);
    }
    let mut hits = 0usize;
    for (phrase, expected_concept, _) in rows {
        let (emb, _) = embed_phrase(rt, phrase)?;
        let routed = index
            .nearest_fleet_wide(&emb)
            .map(|m| m.node_id == *expected_concept);
        if routed.unwrap_or(false) {
            hits += 1;
        }
    }
    Ok(hits as f32 / rows.len() as f32)
}

pub fn cross_domain_misroute_rate(
    captures: &[FailureCapture],
    rt: &LanguageRuntime,
    index: &GroundingNodeIndex,
) -> Result<f32, String> {
    if captures.is_empty() {
        return Ok(0.0);
    }
    let mut mis = 0usize;
    for cap in captures {
        let (emb, _) = embed_phrase(rt, &cap.phrase)?;
        let home = parse_fleet_domain_context(&cap.domain_context);
        if let Some(m) = index.nearest_fleet_wide(&emb) {
            if m.domain != home {
                mis += 1;
            }
        }
    }
    Ok(mis as f32 / captures.len() as f32)
}

pub fn certify_batch(
    captures: &[FailureCapture],
    rt: &LanguageRuntime,
    before: &GroundingNodeIndex,
    after: &GroundingNodeIndex,
    _home_domain: GroundingFleetDomain,
) -> Result<(CertifierMetrics, CertifierMetrics), String> {
    let before_held = routing_accuracy_for_captures(captures, rt, before, CaptureSplit::Certify)?;
    let before_cap = routing_accuracy_for_captures(captures, rt, before, CaptureSplit::Propose)?;
    let before_mis = cross_domain_misroute_rate(captures, rt, before)?;
    let before_metrics = CertifierMetrics {
        held_out_accuracy: before_held,
        captured_accuracy: before_cap,
        generalization_gap: (before_cap - before_held).max(0.0),
        cross_domain_misroute_rate: before_mis,
        alias_additions: 0,
    };

    let after_held = routing_accuracy_for_captures(captures, rt, after, CaptureSplit::Certify)?;
    let after_cap = routing_accuracy_for_captures(captures, rt, after, CaptureSplit::Propose)?;
    let after_mis = cross_domain_misroute_rate(captures, rt, after)?;
    let after_metrics = CertifierMetrics {
        held_out_accuracy: after_held,
        captured_accuracy: after_cap,
        generalization_gap: (after_cap - after_held).max(0.0),
        cross_domain_misroute_rate: after_mis,
        alias_additions: 0,
    };

    Ok((before_metrics, after_metrics))
}

/// One point on the coverage-vs-additions curve (§6 n-sweep analog).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CoverageCurvePoint {
    pub additions: usize,
    pub captured_accuracy: f32,
    pub held_out_accuracy: f32,
    pub generalization_gap: f32,
    pub cross_domain_misroute_rate: f32,
}

fn curve_point(
    captures: &[FailureCapture],
    rt: &LanguageRuntime,
    index: &GroundingNodeIndex,
    additions: usize,
) -> Result<CoverageCurvePoint, String> {
    let captured = routing_accuracy_for_captures(captures, rt, index, CaptureSplit::Propose)?;
    let held = routing_accuracy_for_captures(captures, rt, index, CaptureSplit::Certify)?;
    let misroute = cross_domain_misroute_rate(captures, rt, index)?;
    Ok(CoverageCurvePoint {
        additions,
        captured_accuracy: captured,
        held_out_accuracy: held,
        generalization_gap: (captured - held).max(0.0),
        cross_domain_misroute_rate: misroute,
    })
}

/// Sweep alias additions one at a time, measuring held-out vs captured coverage at
/// each step (the lexicon analog of the n-sweep). `additions` is applied in order;
/// each entry is `(domain, target_node, phrase)`.
///
/// Read it like the n-sweep: if held-out keeps rising with additions, the edits
/// generalize; if captured-set keeps rising while held-out plateaus, you are growing
/// a lookup table (memorization), not a concept.
pub fn coverage_vs_additions_curve(
    captures: &[FailureCapture],
    rt: &LanguageRuntime,
    base_index: &GroundingNodeIndex,
    additions: &[(GroundingFleetDomain, String, String)],
) -> Result<Vec<CoverageCurvePoint>, String> {
    let mut index = base_index.clone();
    let mut out = vec![curve_point(captures, rt, &index, 0)?];
    for (i, (domain, node, phrase)) in additions.iter().enumerate() {
        let (emb, _) = embed_phrase(rt, phrase)?;
        index.add_alias_to_node(*domain, node, phrase, emb);
        out.push(curve_point(captures, rt, &index, i + 1)?);
    }
    Ok(out)
}

/// Plateau detector over a curve: returns (captured_lift, held_out_lift). A large
/// captured lift with a near-zero held-out lift is the overfitting/memorization
/// signature (§7 saturation row vs lexicon_memorization).
pub fn curve_lifts(curve: &[CoverageCurvePoint]) -> (f32, f32) {
    if curve.len() < 2 {
        return (0.0, 0.0);
    }
    let first = &curve[0];
    let last = &curve[curve.len() - 1];
    (
        last.captured_accuracy - first.captured_accuracy,
        last.held_out_accuracy - first.held_out_accuracy,
    )
}

pub fn format_coverage_curve(label: &str, curve: &[CoverageCurvePoint]) -> String {
    let mut s = format!("{label}\n  additions | captured | held-out | gap | x-domain misroute\n");
    for p in curve {
        s.push_str(&format!(
            "  {:>9} | {:>7.1}% | {:>7.1}% | {:.3} | {:.1}%\n",
            p.additions,
            p.captured_accuracy * 100.0,
            p.held_out_accuracy * 100.0,
            p.generalization_gap,
            p.cross_domain_misroute_rate * 100.0,
        ));
    }
    let (cap_lift, held_lift) = curve_lifts(curve);
    s.push_str(&format!(
        "  → captured lift {:+.1}pp, held-out lift {:+.1}pp",
        cap_lift * 100.0,
        held_lift * 100.0,
    ));
    s
}

pub fn decide_batch_verdict(
    before: &CertifierMetrics,
    after: &CertifierMetrics,
    params: &GroundingLoopParams,
    had_collisions: bool,
) -> BatchVerdict {
    if before.held_out_accuracy == 0.0
        && after.held_out_accuracy == 0.0
        && before.captured_accuracy == 0.0
        && after.captured_accuracy == 0.0
    {
        return BatchVerdict::InsufficientData;
    }

    let held_lift = after.held_out_accuracy - before.held_out_accuracy;
    let gap = after.generalization_gap;
    let mis_lift = after.cross_domain_misroute_rate - before.cross_domain_misroute_rate;

    if held_lift < params.min_held_out_lift
        && after.captured_accuracy > before.captured_accuracy + 0.05
        && gap > params.max_generalization_gap
    {
        return BatchVerdict::LexiconMemorization;
    }

    if had_collisions && mis_lift > params.max_cross_domain_misroute_lift {
        return BatchVerdict::NetNegativeCollision;
    }

    if held_lift < params.min_held_out_lift
        && after.captured_accuracy <= before.captured_accuracy + 0.01
    {
        return BatchVerdict::Saturation;
    }

    if held_lift >= params.min_held_out_lift
        && gap <= params.max_generalization_gap
        && mis_lift <= params.max_cross_domain_misroute_lift
    {
        return BatchVerdict::GenuineCoverageImprovement;
    }

    if after.captured_accuracy > before.captured_accuracy + 0.05
        && held_lift < params.min_held_out_lift
    {
        return BatchVerdict::LexiconMemorization;
    }

    BatchVerdict::InsufficientData
}

/// Synthetic fixture: concept-balanced propose/certify splits for offline audit.
/// Phrases share vocabulary with node aliases but omit exact alias tokens (OOD at token layer).
pub fn synthetic_audit_fixture() -> Vec<FixtureRow> {
    vec![
        FixtureRow {
            phrase: "stack more sats onchain".into(),
            concept_id: "bitcoin".into(),
            split: CaptureSplit::Propose,
            domain_context: "crypto".into(),
        },
        FixtureRow {
            phrase: "hold btc for years".into(),
            concept_id: "bitcoin".into(),
            split: CaptureSplit::Certify,
            domain_context: "crypto".into(),
        },
        FixtureRow {
            phrase: "ether gas is expensive".into(),
            concept_id: "ethereum".into(),
            split: CaptureSplit::Propose,
            domain_context: "crypto".into(),
        },
        FixtureRow {
            phrase: "ethereum network fees spike".into(),
            concept_id: "ethereum".into(),
            split: CaptureSplit::Certify,
            domain_context: "crypto".into(),
        },
        FixtureRow {
            phrase: "dex swap got rekt".into(),
            concept_id: "dex".into(),
            split: CaptureSplit::Propose,
            domain_context: "crypto".into(),
        },
        FixtureRow {
            phrase: "decentralized exchange routing failed".into(),
            concept_id: "dex".into(),
            split: CaptureSplit::Certify,
            domain_context: "crypto".into(),
        },
        FixtureRow {
            phrase: "puppy snack order".into(),
            concept_id: "pet_treat".into(),
            split: CaptureSplit::Propose,
            domain_context: "runtime".into(),
        },
        FixtureRow {
            phrase: "dog treat training reward".into(),
            concept_id: "pet_treat".into(),
            split: CaptureSplit::Certify,
            domain_context: "runtime".into(),
        },
        FixtureRow {
            phrase: "leash park stroll".into(),
            concept_id: "pet_walk".into(),
            split: CaptureSplit::Propose,
            domain_context: "runtime".into(),
        },
        FixtureRow {
            phrase: "dog walk exercise routine".into(),
            concept_id: "pet_walk".into(),
            split: CaptureSplit::Certify,
            domain_context: "runtime".into(),
        },
    ]
}

pub const PET_DOMAIN_FIXTURE_TOML: &str = r#"
version = 1

[[nodes]]
id = "pet_treat"
aliases = ["dog_treats", "puppy_snacks"]

[[nodes]]
id = "pet_walk"
aliases = ["dog_walk", "leash"]
"#;

// ===========================================================================================
// Token-disjoint generalization test (see docs/GROUNDING_DISJOINT_SPEC).
//
// Decides whether the supervised encoder's held-out routing is genuine paraphrase
// generalization or lexical matching in a learned coat, by measuring routing accuracy as a
// function of feature overlap between a certify phrase and its true concept's TRAINING
// phrases — at the encoder's own feature granularity (§1).
// ===========================================================================================

/// Per-phrase result for the disjoint curve.
#[derive(Clone, Debug)]
pub struct DisjointEval {
    pub overlap: f32,
    pub routed_correctly: bool,
    /// overlap == 0 AND features were seen attached to *other* concepts (genuine signal).
    pub sub_seen_elsewhere: bool,
    /// overlap == 0 AND features never seen in *any* concept (routes by prior, not learning).
    pub sub_novel: bool,
}

/// One bin of the accuracy-vs-overlap curve, with a Wilson 95% CI.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OverlapBin {
    pub label: String,
    pub n: usize,
    pub hits: usize,
    pub accuracy: f32,
    pub ci_lo: f32,
    pub ci_hi: f32,
}

/// Union of encoder features over all training phrases per concept, plus the global union.
pub fn concept_train_features(
    train_pairs: &[(String, String)],
) -> (HashMap<String, HashSet<String>>, HashSet<String>) {
    let mut per: HashMap<String, HashSet<String>> = HashMap::new();
    let mut global: HashSet<String> = HashSet::new();
    for (phrase, label) in train_pairs {
        let f = phrase_feature_set(phrase);
        let e = per.entry(label.clone()).or_default();
        for k in &f {
            e.insert(k.clone());
            global.insert(k.clone());
        }
    }
    (per, global)
}

/// Evaluate each certify phrase: did it route to its true concept, what is its feature
/// overlap with that concept's training, and (when overlap is 0) is it seen-elsewhere or
/// novel. `level` is the feature granularity (`"w"`, `"wb"`, or `"wbc"`).
pub fn evaluate_disjoint(
    rt: &LanguageRuntime,
    index: &GroundingNodeIndex,
    certify: &[FailureCapture],
    concept_train: &HashMap<String, HashSet<String>>,
    global_train: &HashSet<String>,
    level: &str,
) -> Result<Vec<DisjointEval>, String> {
    let global = restrict_features(global_train, level);
    let ct: HashMap<&String, HashSet<String>> = concept_train
        .iter()
        .map(|(k, v)| (k, restrict_features(v, level)))
        .collect();
    let empty = HashSet::new();

    let mut out = Vec::with_capacity(certify.len());
    for cap in certify {
        let (emb, _) = embed_phrase(rt, &cap.phrase)?;
        let domain = parse_fleet_domain_context(&cap.domain_context);
        let routed_correctly = index
            .nearest_in_domain(&emb, domain)
            .map(|m| m.node_id == cap.inferred_concept_id)
            .unwrap_or(false);

        let fp = restrict_features(&phrase_feature_set(&cap.phrase), level);
        let train_c = ct.get(&cap.inferred_concept_id).unwrap_or(&empty);
        let overlap = feature_overlap_fraction(&fp, train_c);

        let (mut sub_seen_elsewhere, mut sub_novel) = (false, false);
        if overlap == 0.0 {
            let touches_global = fp.iter().any(|k| global.contains(k));
            if touches_global {
                sub_seen_elsewhere = true;
            } else {
                sub_novel = true;
            }
        }
        out.push(DisjointEval {
            overlap,
            routed_correctly,
            sub_seen_elsewhere,
            sub_novel,
        });
    }
    Ok(out)
}

/// Bin evaluations into the fixed overlap bins and attach Wilson CIs.
pub fn build_overlap_curve(evals: &[DisjointEval]) -> Vec<OverlapBin> {
    let bins: [(&str, f32, f32); 5] = [
        ("0", 0.0, 0.0),
        ("(0,0.1]", 0.0, 0.1),
        ("(0.1,0.3]", 0.1, 0.3),
        ("(0.3,0.6]", 0.3, 0.6),
        ("(0.6,1.0]", 0.6, 1.0),
    ];
    let mut out = Vec::new();
    for (label, lo, hi) in bins {
        let members: Vec<&DisjointEval> = evals
            .iter()
            .filter(|e| {
                if label == "0" {
                    e.overlap == 0.0
                } else {
                    e.overlap > lo && e.overlap <= hi
                }
            })
            .collect();
        let n = members.len();
        let hits = members.iter().filter(|e| e.routed_correctly).count();
        let (lo_ci, hi_ci) = wilson_interval(hits, n, 1.96);
        out.push(OverlapBin {
            label: label.to_string(),
            n,
            hits,
            accuracy: if n > 0 { hits as f32 / n as f32 } else { 0.0 },
            ci_lo: lo_ci as f32,
            ci_hi: hi_ci as f32,
        });
    }
    out
}

/// `(seen_elsewhere_hits, seen_elsewhere_n, novel_hits, novel_n)` for the overlap-0 bin.
pub fn overlap0_substrata(evals: &[DisjointEval]) -> (usize, usize, usize, usize) {
    let mut ah = 0;
    let mut an = 0;
    let mut bh = 0;
    let mut bn = 0;
    for e in evals {
        if e.sub_seen_elsewhere {
            an += 1;
            if e.routed_correctly {
                ah += 1;
            }
        } else if e.sub_novel {
            bn += 1;
            if e.routed_correctly {
                bh += 1;
            }
        }
    }
    (ah, an, bh, bn)
}

pub fn pooled_accuracy(evals: &[DisjointEval]) -> f32 {
    if evals.is_empty() {
        return 0.0;
    }
    evals.iter().filter(|e| e.routed_correctly).count() as f32 / evals.len() as f32
}

/// Encoder-free audit of whether a candidate held-out eval set is *capable of carrying a
/// generalization signal*: does it contain feature-disjoint, seen-elsewhere held-out phrases at
/// the strictest granularity? This is the acceptance instrument for the feature-disjoint
/// home-domain eval (§15.2). Disjointness is a property of the surface features vs. training, so
/// it is independent of any encoder — the GLE's in-domain eval failed this audit (`n_seen_elsewhere=0`),
/// which is why no score on it could speak to generalization.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct DisjointEvalAudit {
    pub level: String,
    pub n_classes: usize,
    pub n_eval: usize,
    /// eval phrases with overlap==0 vs. their OWN class's training features (at `level`).
    pub n_overlap0: usize,
    /// the gen_a bin: overlap-0 AND at least one feature seen on SOME other class in training.
    pub n_seen_elsewhere: usize,
    /// overlap-0 AND features seen nowhere in training (novel tokens — guessing, not routing).
    pub n_novel: usize,
    /// `(class, seen_elsewhere_count, eval_count)`, sorted by class.
    pub per_class_seen_elsewhere: Vec<(String, usize, usize)>,
    /// `true` iff `n_seen_elsewhere >= DISJOINT_MIN_N` — enough disjoint examples to resolve lift.
    pub resolvable: bool,
}

/// Leave-one-out scan of an entire corpus: how many phrases are feature-disjoint from the *rest*
/// of their own class (and seen-elsewhere)? This answers the cheap, decisive question behind the
/// in-domain GLE eval — *does a certifiable held-out set exist in this domain at all, or is the
/// disjoint-example class structurally empty?* It must be leave-one-out: a phrase is always
/// (trivially) present in its own class's training, so a `train==eval` audit reports 0 by artifact,
/// not by domain. Here a phrase is disjoint-from-its-class iff each of its `level` features occurs
/// in exactly one phrase of that class (itself), and seen-elsewhere iff some such feature appears
/// in another class.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CorpusDisjointnessScan {
    pub level: String,
    pub n_classes: usize,
    pub n_phrases: usize,
    /// Classes with ≥ `min_class_size` phrases — only these can serve as eval classes (a class
    /// needs phrases for both a training centroid and a held-out example; a singleton "disjoint"
    /// phrase is degenerate, not usable).
    pub n_eligible_classes: usize,
    pub n_eligible_phrases: usize,
    pub n_disjoint: usize,
    pub n_seen_elsewhere: usize,
    pub n_novel: usize,
    /// `(class, seen_elsewhere_count, phrase_count)`, sorted, only eligible classes with ≥1 seen-elsewhere.
    pub per_class_seen_elsewhere: Vec<(String, usize, usize)>,
}

pub fn scan_corpus_disjointness(
    pairs: &[(String, String)],
    level: &str,
    min_class_size: usize,
) -> CorpusDisjointnessScan {
    // Per (class, feature) → number of phrases in that class containing the feature.
    let mut class_feat_phrases: HashMap<String, HashMap<String, usize>> = HashMap::new();
    // Per feature → set of classes it appears in.
    let mut feat_classes: HashMap<String, HashSet<String>> = HashMap::new();
    let mut class_phrase_count: HashMap<String, usize> = HashMap::new();
    let feats: Vec<(String, HashSet<String>)> = pairs
        .iter()
        .map(|(t, c)| (c.clone(), restrict_features(&phrase_feature_set(t), level)))
        .collect();
    for (c, fp) in &feats {
        *class_phrase_count.entry(c.clone()).or_insert(0) += 1;
        let cf = class_feat_phrases.entry(c.clone()).or_default();
        for f in fp {
            *cf.entry(f.clone()).or_insert(0) += 1;
            feat_classes.entry(f.clone()).or_default().insert(c.clone());
        }
    }

    let mut per_class: std::collections::BTreeMap<String, (usize, usize)> =
        std::collections::BTreeMap::new();
    let (mut n_disjoint, mut n_seen, mut n_novel) = (0usize, 0usize, 0usize);
    let mut n_eligible_phrases = 0usize;
    for (c, fp) in &feats {
        // Only classes big enough to serve as eval classes (training centroid + held-out example).
        if class_phrase_count.get(c).copied().unwrap_or(0) < min_class_size {
            continue;
        }
        n_eligible_phrases += 1;
        let entry = per_class.entry(c.clone()).or_insert((0, 0));
        entry.1 += 1;
        if fp.is_empty() {
            continue;
        }
        let cf = &class_feat_phrases[c];
        // Disjoint from the rest of its class iff every feature is unique to this phrase in-class.
        let disjoint = fp.iter().all(|f| cf.get(f).copied().unwrap_or(0) <= 1);
        if disjoint {
            n_disjoint += 1;
            let seen_elsewhere = fp.iter().any(|f| {
                feat_classes
                    .get(f)
                    .map(|s| s.iter().any(|x| x != c))
                    .unwrap_or(false)
            });
            if seen_elsewhere {
                n_seen += 1;
                entry.0 += 1;
            } else {
                n_novel += 1;
            }
        }
    }
    let n_eligible_classes = class_phrase_count
        .values()
        .filter(|&&n| n >= min_class_size)
        .count();
    CorpusDisjointnessScan {
        level: level.to_string(),
        n_classes: class_phrase_count.len(),
        n_phrases: pairs.len(),
        n_eligible_classes,
        n_eligible_phrases,
        n_disjoint,
        n_seen_elsewhere: n_seen,
        n_novel,
        per_class_seen_elsewhere: per_class
            .into_iter()
            .filter(|(_, (s, _))| *s > 0)
            .map(|(c, (s, t))| (c, s, t))
            .collect(),
    }
}

/// Compute the disjointness audit for `eval_pairs` against `train_pairs` (both `(text, class)`).
pub fn audit_disjoint_eval(
    train_pairs: &[(String, String)],
    eval_pairs: &[(String, String)],
    level: &str,
) -> DisjointEvalAudit {
    let (concept_train, global_train) = concept_train_features(train_pairs);
    let ct: HashMap<&String, HashSet<String>> = concept_train
        .iter()
        .map(|(k, v)| (k, restrict_features(v, level)))
        .collect();
    let global = restrict_features(&global_train, level);
    let empty = HashSet::new();

    let mut per_class: std::collections::BTreeMap<String, (usize, usize)> =
        std::collections::BTreeMap::new();
    let (mut n_overlap0, mut n_seen, mut n_novel) = (0usize, 0usize, 0usize);
    for (text, class) in eval_pairs {
        let entry = per_class.entry(class.clone()).or_insert((0, 0));
        entry.1 += 1;
        let fp = restrict_features(&phrase_feature_set(text), level);
        let train_c = ct.get(class).unwrap_or(&empty);
        if feature_overlap_fraction(&fp, train_c) == 0.0 {
            n_overlap0 += 1;
            if fp.iter().any(|k| global.contains(k)) {
                n_seen += 1;
                entry.0 += 1;
            } else {
                n_novel += 1;
            }
        }
    }
    DisjointEvalAudit {
        level: level.to_string(),
        n_classes: concept_train.len(),
        n_eval: eval_pairs.len(),
        n_overlap0,
        n_seen_elsewhere: n_seen,
        n_novel,
        per_class_seen_elsewhere: per_class.into_iter().map(|(c, (s, t))| (c, s, t)).collect(),
        resolvable: n_seen >= DISJOINT_MIN_N,
    }
}

/// §18.4 pre-label triage of captured, *unlabeled* traffic against the production training corpus.
/// Label-free by construction (§18.3): it ranks *which* phrases to surface to a blind human, and
/// never assigns a label.
///
/// Per phrase, at `level`: `global_coverage = |F(p) ∩ global_train| / |F(p)|` (surface
/// familiarity), and `max_concept_overlap = max_c |F(p) ∩ F_c| / |F(p)|` (how strongly any single
/// concept's training lexically claims it). The disjoint-bin sweet spot (§17 "seen-elsewhere
/// disjoint") is *familiar but not concept-locked*: high coverage, low max-concept-overlap — a
/// paraphrase built from seen vocabulary that matches no single concept's surface. Those are the
/// phrases worth a human's label; concept-locked phrases are trivial in-lexicon, and zero-coverage
/// phrases are novel/OOD (guessing, not routing). Priority = `coverage * (1 - max_concept_overlap)`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CaptureTriageRow {
    pub phrase: String,
    pub global_coverage: f32,
    pub max_concept_overlap: f32,
    pub nearest_concept: String,
    pub tier: String,
    pub label_priority: f32,
    /// Always empty here — the blind human fills it (§18.3). Present so the queue file is the
    /// exact schema the labeled bucketing pass (`audit_disjoint_eval`) reads back.
    pub semantic_intent: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CaptureTriageReport {
    pub level: String,
    pub n_captured: usize,
    pub n_unique: usize,
    pub n_disjoint_candidate: usize,
    pub n_in_lexicon: usize,
    pub n_novel_ood: usize,
    /// Sorted descending by `label_priority`.
    pub rows: Vec<CaptureTriageRow>,
}

/// §18.9 / DEFECT-G1 — malformation screen for the labeling front.
///
/// Returns the reason a captured *phrase* is malformed (MASK-leak, decode-collapse soup, or empty),
/// or `None` if it is a legitimate labeling candidate. This screens the **phrase** (the labeling
/// target that enters `label_queue.jsonl`), **not** the incumbent response (which is never a label,
/// §18.2) — so a clean phrase whose *response* happened to garble (e.g. `"bad cat"`) is correctly
/// kept. Conservative by construction: false-negatives are acceptable, false-positives are not, since
/// discarding a real phrase would shrink the very disjoint bin we are trying to fill. The point is to
/// guarantee garble can never be hand-labeled into the certification bin even if a degraded engine or
/// a capture bug ever logs soup as input.
pub fn malformed_capture_reason(phrase: &str) -> Option<&'static str> {
    let t = phrase.trim();
    if t.is_empty() {
        return Some("empty");
    }
    // Training MASK token leaked into the surface (the "bad cat" garble family).
    if t.contains("[MASK]")
        || t.split(|c: char| !c.is_ascii_alphanumeric())
            .any(|w| w == "MASK")
    {
        return Some("mask_leak");
    }
    // Known decode-collapse multi-word n-grams — improbable in genuine user input (single words are
    // intentionally excluded to avoid quarantining legitimate phrases).
    let l = t.to_lowercase();
    const COLLAPSE: [&str; 4] = [
        "schedule both",
        "puddle brush",
        "vibrates sleeping",
        "sleeping minutes",
    ];
    if COLLAPSE.iter().any(|sig| l.contains(sig)) {
        return Some("decode_collapse");
    }
    None
}

pub fn triage_captured_phrases(
    captured: &[String],
    train_pairs: &[(String, String)],
    level: &str,
    coverage_min: f32,
    concept_lock_max: f32,
) -> CaptureTriageReport {
    let (concept_train, global_train) = concept_train_features(train_pairs);
    let ct: Vec<(String, HashSet<String>)> = concept_train
        .iter()
        .map(|(k, v)| (k.clone(), restrict_features(v, level)))
        .collect();
    let global = restrict_features(&global_train, level);

    // Dedup by normalized key, preserving first-seen surface form.
    let mut seen_keys: HashSet<String> = HashSet::new();
    let mut unique: Vec<String> = Vec::new();
    for p in captured {
        let key = normalize_phrase_key(p);
        if key.is_empty() {
            continue;
        }
        if seen_keys.insert(key) {
            unique.push(p.clone());
        }
    }

    let mut rows: Vec<CaptureTriageRow> = Vec::new();
    let (mut n_disjoint, mut n_lex, mut n_novel) = (0usize, 0usize, 0usize);
    for phrase in &unique {
        let fp = restrict_features(&phrase_feature_set(phrase), level);
        if fp.is_empty() {
            continue;
        }
        let global_coverage = feature_overlap_fraction(&fp, &global);
        let mut max_overlap = 0.0f32;
        let mut nearest = String::new();
        for (c, fc) in &ct {
            let o = feature_overlap_fraction(&fp, fc);
            if o > max_overlap {
                max_overlap = o;
                nearest = c.clone();
            }
        }
        let tier = if global_coverage <= 0.0 {
            n_novel += 1;
            "novel_ood"
        } else if max_overlap >= concept_lock_max {
            n_lex += 1;
            "in_lexicon"
        } else if global_coverage >= coverage_min {
            n_disjoint += 1;
            "disjoint_candidate"
        } else {
            // familiar-ish but sparse coverage — keep as a weaker candidate.
            n_disjoint += 1;
            "disjoint_candidate"
        };
        rows.push(CaptureTriageRow {
            phrase: phrase.clone(),
            global_coverage,
            max_concept_overlap: max_overlap,
            nearest_concept: nearest,
            tier: tier.to_string(),
            label_priority: global_coverage * (1.0 - max_overlap),
            semantic_intent: String::new(),
        });
    }
    rows.sort_by(|a, b| {
        b.label_priority
            .partial_cmp(&a.label_priority)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    CaptureTriageReport {
        level: level.to_string(),
        n_captured: captured.len(),
        n_unique: unique.len(),
        n_disjoint_candidate: n_disjoint,
        n_in_lexicon: n_lex,
        n_novel_ood: n_novel,
        rows,
    }
}

pub fn format_certifier_report(
    label: &str,
    before: &CertifierMetrics,
    after: &CertifierMetrics,
    verdict: BatchVerdict,
) -> String {
    format!(
        "{label}\n  held-out paraphrase accuracy: {:.1}% → {:.1}%\n  captured-set coverage: {:.1}% → {:.1}%\n  generalization gap: {:.3} → {:.3}\n  cross-domain misroute: {:.1}% → {:.1}%\n  verdict: {}",
        before.held_out_accuracy * 100.0,
        after.held_out_accuracy * 100.0,
        before.captured_accuracy * 100.0,
        after.captured_accuracy * 100.0,
        before.generalization_gap,
        after.generalization_gap,
        before.cross_domain_misroute_rate * 100.0,
        after.cross_domain_misroute_rate * 100.0,
        verdict.as_str(),
    )
}

// ===========================================================================
// Certifier-First Pipeline (§§1–6 of the Certifier-First spec)
//
// The certifier is the *contract* every encoder is judged by: a deterministic
// pipeline that emits one verdict artifact. The only field that gates promotion
// is `disjoint_semantic_lift` (disjoint-bin(a) accuracy − shuffle-floor 95th pct);
// pooled accuracy is recorded as evidence but never gates.
// ===========================================================================

/// Provenance-purity report for the augmentation-leak firewall (§4). This is *orthogonal*
/// to the disjoint test's surface-overlap check: the disjoint test catches **feature**
/// leakage; the firewall catches **pipeline** leakage (an augmented paraphrase of a training
/// phrase reaching the certify set, which would let the encoder certify itself).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct FirewallReport {
    pub clean: bool,
    pub violations: Vec<String>,
    pub train_real: usize,
    pub train_authored: usize,
    pub train_augmented: usize,
    pub certify_real: usize,
    pub certify_authored: usize,
    pub certify_augmented: usize,
}

fn tally_provenance(
    kind: ProvenanceKind,
    real: &mut usize,
    authored: &mut usize,
    augmented: &mut usize,
) {
    match kind {
        ProvenanceKind::RealTraffic => *real += 1,
        ProvenanceKind::Authored => *authored += 1,
        ProvenanceKind::Augmented => *augmented += 1,
    }
}

/// Enforce the firewall invariants over a capture set (propose = train, certify = held-out):
/// 1. **Certify ⊆ `real_traffic`** — no `augmented`/`authored` phrase is ever in certify.
/// 2. **No lineage crossing** — no certify phrase's id appears in any training `augmented`
///    phrase's `derived_from` lineage.
///
/// Any violation ⇒ the run is `INVALID` (a pipeline/data problem), never a score.
pub fn run_augmentation_firewall(captures: &[FailureCapture]) -> FirewallReport {
    run_augmentation_firewall_ex(captures, false)
}

/// Extended firewall. When `allow_authored_certify` is true, `Authored` provenance
/// in the certify set is treated as clean (§2.2 of the real-encoder experiment spec:
/// authored phrases are genuinely held-out when no encoder trained on them). `Augmented`
/// provenance in certify is always rejected regardless of this flag.
pub fn run_augmentation_firewall_ex(
    captures: &[FailureCapture],
    allow_authored_certify: bool,
) -> FirewallReport {
    let mut r = FirewallReport {
        clean: true,
        ..Default::default()
    };
    let mut train_lineage: HashSet<String> = HashSet::new();
    let mut certify_ids: Vec<(String, String)> = Vec::new(); // (phrase_id, phrase)

    for c in captures {
        match c.split {
            CaptureSplit::Propose => {
                tally_provenance(
                    c.provenance.kind,
                    &mut r.train_real,
                    &mut r.train_authored,
                    &mut r.train_augmented,
                );
                if c.provenance.kind == ProvenanceKind::Augmented {
                    for src in &c.provenance.derived_from {
                        train_lineage.insert(src.clone());
                    }
                }
            }
            CaptureSplit::Certify => {
                tally_provenance(
                    c.provenance.kind,
                    &mut r.certify_real,
                    &mut r.certify_authored,
                    &mut r.certify_augmented,
                );
                let rejected = match c.provenance.kind {
                    ProvenanceKind::RealTraffic => false,
                    ProvenanceKind::Authored => !allow_authored_certify,
                    ProvenanceKind::Augmented => true,
                };
                if rejected {
                    r.clean = false;
                    r.violations.push(format!(
                        "certify phrase '{}' has provenance {} (must be {})",
                        c.phrase,
                        c.provenance.kind.as_str(),
                        if allow_authored_certify {
                            "real_traffic or authored"
                        } else {
                            "real_traffic"
                        }
                    ));
                }
                let id = if c.provenance.phrase_id.is_empty() {
                    c.phrase.clone()
                } else {
                    c.provenance.phrase_id.clone()
                };
                certify_ids.push((id, c.phrase.clone()));
            }
        }
    }

    for (id, phrase) in &certify_ids {
        if train_lineage.contains(id) {
            r.clean = false;
            r.violations.push(format!(
                "lineage crossing: certify phrase '{phrase}' (id '{id}') is the source of a training augmented phrase"
            ));
        }
    }
    r.violations.sort();
    r.violations.dedup();
    r
}

/// The terminal verdict of the certifier pipeline (§6). `INVALID` and `BELOW_RESOLUTION`
/// are distinct from `FAIL_*`: they mean "not measured", never "measured and bad".
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Verdict {
    Invalid,
    BelowResolution,
    FailMemorization,
    FailCollision,
    /// Lift cleared the floor, but only at a disjointness granularity coarser than `wbc` (i.e. the
    /// union-disjoint bin was empty and the gate fell back to `wb`/`w`). The pass is real but
    /// leakier — word-disjoint phrases can still share bigrams/trigrams — so it is **provisional**
    /// and NOT promotable. It must be re-earned at `wbc` on a feature-disjoint eval to become `PASS`.
    PassProvisional,
    Pass,
}

impl Verdict {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Invalid => "INVALID",
            Self::BelowResolution => "BELOW_RESOLUTION",
            Self::FailMemorization => "FAIL_MEMORIZATION",
            Self::FailCollision => "FAIL_COLLISION",
            Self::PassProvisional => "PASS_PROVISIONAL",
            Self::Pass => "PASS",
        }
    }

    /// Only a strict (`wbc`) pass licenses promotion; `PASS_PROVISIONAL` does not.
    pub fn is_promotable(self) -> bool {
        matches!(self, Self::Pass)
    }

    pub fn from_str(s: &str) -> Self {
        match s {
            "INVALID" => Self::Invalid,
            "BELOW_RESOLUTION" => Self::BelowResolution,
            "FAIL_MEMORIZATION" => Self::FailMemorization,
            "FAIL_COLLISION" => Self::FailCollision,
            "PASS_PROVISIONAL" => Self::PassProvisional,
            "PASS" => Self::Pass,
            _ => Self::Invalid,
        }
    }
}

/// Minimum seen-elsewhere sample count for the disjoint bin to resolve lift vs floor.
pub const DISJOINT_MIN_N: usize = 8;
/// Maximum Wilson CI width on disjoint_gen_a before the bin is too noisy to read.
pub const DISJOINT_MAX_CI_WIDTH: f32 = 0.30;
/// Memorization gap above which a "pass" is rejected as a lookup table even with positive lift.
pub const MEMORIZATION_GAP_MAX: f32 = 0.50;

/// Pure inputs to the verdict state machine — everything the decision depends on, nothing else,
/// so the rule (§6) is testable in isolation from the routing/embedding machinery.
#[derive(Clone, Copy, Debug)]
pub struct VerdictInputs {
    pub positive_control_collapsed: bool,
    pub firewall_clean: bool,
    pub disjoint_gen_a_n: usize,
    pub disjoint_gen_a_ci_width: f32,
    pub disjoint_semantic_lift: f32,
    pub lift_ci_lo: f32,
    pub collision_delta: f32,
    pub memorization_gap: f32,
    /// `true` iff the lift was resolved at the strictest disjointness granularity (`wbc`). When
    /// the gate fell back to a coarser level (`wb`/`w`) because the union-disjoint bin was empty,
    /// a cleared lift yields `PASS_PROVISIONAL`, never `PASS` — the fallback cannot launder a
    /// surface-overlap pass that the finest filter would have caught.
    pub strict_disjoint_level: bool,
}

/// `true` if the seen-elsewhere disjoint bin is too small/noisy to separate lift from 0.
pub fn is_below_resolution(n: usize, ci_width: f32) -> bool {
    n < DISJOINT_MIN_N || ci_width > DISJOINT_MAX_CI_WIDTH
}

/// The deterministic verdict state machine (§6). Order matters: validity gates (pipeline/data)
/// precede resolution, which precedes the encoder judgment. An underpowered or invalid run is
/// never readable as a pass.
pub fn decide_encoder_verdict(inp: &VerdictInputs) -> Verdict {
    if !inp.positive_control_collapsed || !inp.firewall_clean {
        return Verdict::Invalid;
    }
    if is_below_resolution(inp.disjoint_gen_a_n, inp.disjoint_gen_a_ci_width) {
        return Verdict::BelowResolution;
    }
    // Lift gate: generalizes iff lift > 0 AND the lift CI excludes 0 (lower bound > 0).
    if inp.disjoint_semantic_lift <= 0.0 || inp.lift_ci_lo <= 0.0 {
        return Verdict::FailMemorization;
    }
    if inp.collision_delta > 0.0 {
        return Verdict::FailCollision;
    }
    if inp.memorization_gap > MEMORIZATION_GAP_MAX {
        // Positive disjoint lift but the captured→held-out gap is blown: still a lookup table.
        return Verdict::FailMemorization;
    }
    // Lift cleared the floor. Only a strict (`wbc`) resolution is a promotable PASS; a pass earned
    // via a coarser fallback level is provisional, because word/bigram-disjoint phrases can still
    // share finer features the union filter would have rejected.
    if inp.strict_disjoint_level {
        Verdict::Pass
    } else {
        Verdict::PassProvisional
    }
}

/// Per-feature-family disjoint-0 accuracy: shows which granularity carries overlap inflation
/// (e.g. trigrams much higher than words ⇒ subword leakage). Diagnostic, never gates.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct FeatureFamily {
    pub word: f32,
    pub bigram: f32,
    pub trigram: f32,
}

/// The single verdict artifact (§1) — one deterministic source of truth per
/// `(encoder_id, data_hash, seed)`. Any consumer reads `verdict` + `disjoint_semantic_lift`;
/// every other field is evidence.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EncoderVerdict {
    pub encoder_id: String,
    pub data_hash: String,
    pub seed: u64,
    pub candidate_set_size: usize,

    // --- GO/NO-GO ---
    pub disjoint_semantic_lift: f32,
    pub disjoint_lift_ci: [f32; 2],
    pub verdict: String,

    // --- semantic floor / lift decomposition ---
    pub semantic_floor_mean: f32,
    pub semantic_floor_95: f32,
    pub disjoint_gen_a: f32,
    pub disjoint_gen_a_n: usize,
    pub pooled_heldout: f32,
    pub memorization_gap: f32,

    // --- diagnostics ---
    pub overlap_curve: Vec<OverlapBin>,
    pub feature_family: FeatureFamily,
    pub plateau_flag: bool,
    pub collision_delta: f32,
    /// Feature granularity at which the disjoint lift was resolved: `"wbc"` (union, strictest),
    /// or a looser `"wb"`/`"w"` fallback used when the union-disjoint bin is empty on dense
    /// training. Looser ⇒ leakier; recorded so a reviewer knows the leakage risk of the verdict.
    #[serde(default)]
    pub disjoint_level: String,
    /// When the verdict is INVALID, a human-readable reason distinguishing the failure modes:
    /// empty disjoint bin (eval can't separate memorization from generalization) vs. positive
    /// control not collapsing (eval is lexically separable / an easy task) vs. firewall/data.
    #[serde(default)]
    pub invalid_reason: String,

    // --- validity gates ---
    pub positive_control_collapsed: bool,
    pub augmentation_firewall_clean: bool,
    pub below_resolution: bool,

    // --- provenance log (so a reviewer can confirm certify is real held-out traffic) ---
    pub firewall: FirewallReport,
    pub shuffle_b: usize,
    /// Encoder-training provenance note (e.g. a frozen BYO encoder's training domain). For a
    /// distilled encoder this records the distillation-disjointness basis: the firewall proves
    /// the *certify set* is real traffic; this records whether the *encoder* was trained on it.
    #[serde(default)]
    pub encoder_provenance: String,
}

impl EncoderVerdict {
    /// Deterministic artifact filename `verdict_<encoder>_<datahash>_<seed>.json`.
    pub fn filename(&self) -> String {
        format!(
            "verdict_{}_{}_{}.json",
            self.encoder_id, self.data_hash, self.seed
        )
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|e| format!("{{\"error\":\"{e}\"}}"))
    }

    pub fn one_line(&self) -> String {
        format!(
            "{} | verdict={} lift={:+.3} ci=[{:+.3},{:+.3}] gen_a={:.3} (n={}) floor95={:.3} pooled={:.3} gap={:.3}",
            self.encoder_id,
            self.verdict,
            self.disjoint_semantic_lift,
            self.disjoint_lift_ci[0],
            self.disjoint_lift_ci[1],
            self.disjoint_gen_a,
            self.disjoint_gen_a_n,
            self.semantic_floor_95,
            self.pooled_heldout,
            self.memorization_gap,
        )
    }
}

/// A stable content fingerprint of the audited corpus + grounding graph (FNV-1a over a
/// canonical, order-independent string). Same captures + nodes ⇒ same hash, so the artifact
/// id is reproducible. Not cryptographic — a change-detector, not a security primitive.
pub fn data_hash(captures: &[FailureCapture], node_ids: &[String]) -> String {
    let mut rows: Vec<String> = captures
        .iter()
        .map(|c| {
            format!(
                "{}|{}|{}|{}",
                c.phrase.trim().to_ascii_lowercase(),
                c.inferred_concept_id,
                c.domain_context,
                match c.split {
                    CaptureSplit::Propose => "p",
                    CaptureSplit::Certify => "c",
                }
            )
        })
        .collect();
    let mut nodes: Vec<String> = node_ids.to_vec();
    rows.sort();
    nodes.sort();
    let mut h: u64 = 0xcbf29ce484222325;
    for s in rows
        .iter()
        .chain(std::iter::once(&"::".to_string()))
        .chain(nodes.iter())
    {
        for b in s.as_bytes() {
            h ^= *b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        h ^= 0x2c;
        h = h.wrapping_mul(0x100000001b3);
    }
    format!("{h:016x}")
}

// =============================================================================
// §16 — P4 Longitudinal Drift Telemetry
// =============================================================================
// Monitors deployed companion routers over time for degradation on signals
// observable in production WITHOUT ground-truth labels. This is a reliability
// monitor, not a certifier. It alerts; it does not act. Every corrective action
// re-enters the human-gated, certifier-checked path (§15 P0 gate).
// Auto-remediation is structurally forbidden — no code path exists that edits
// the graph or swaps an encoder without a human in the loop.

/// The cause of a detected change point: input distribution moved (world) or a
/// system change (encoder swap, graph edit, deploy) explains it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DriftCause {
    World,
    System { description: String },
}

impl DriftCause {
    pub fn as_str(&self) -> &str {
        match self {
            Self::World => "world",
            Self::System { .. } => "system",
        }
    }
}

/// A detected change point on a single signal, with cause and persistence count.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChangePoint {
    pub signal: String,
    pub cause: DriftCause,
    /// How many consecutive windows this deviation has persisted (≥1).
    pub persisted_windows: usize,
}

/// A single telemetry window for one domain — the append-only artifact (§6).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DriftWindow {
    pub domain: String,
    pub window_id: String,
    pub window_start_unix: u64,
    pub window_duration_secs: u64,

    // --- §1 signals ---
    pub fallthrough_rate: f32,
    pub fallthrough_baseline: f32,
    pub fallthrough_alert: bool,

    pub routing_entropy_p50: f32,
    pub entropy_trend: String,

    pub guard_fire_rate: f32,

    pub coverage_elasticity: f32,
    pub saturation_flag: bool,

    pub cross_domain_collision_rate: f32,
    pub total_aliases_fleet: usize,

    pub encoder_version: String,
    /// Fraction of live traffic that routes differently under old vs new encoder.
    /// `None` if no encoder change this window.
    pub encoder_shift_vs_prev: Option<f32>,

    pub dissatisfaction_rate: f32,
    pub dissatisfaction_baseline: f32,

    // --- §2 detection ---
    pub change_points: Vec<ChangePoint>,
    pub alerts: Vec<DriftAlert>,
}

/// An alert emitted when a change point persists long enough.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DriftAlert {
    pub signal: String,
    pub cause: DriftCause,
    pub persisted_windows: usize,
    pub severity: AlertSeverity,
    pub message: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlertSeverity {
    Warning,
    Critical,
}

impl AlertSeverity {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Warning => "warning",
            Self::Critical => "critical",
        }
    }
}

/// Summary report across multiple windows for one domain.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DriftReport {
    pub domain: String,
    pub n_windows: usize,
    pub latest_window: Option<String>,
    pub active_alerts: Vec<DriftAlert>,
    pub fallthrough_trend: TrendSummary,
    pub entropy_trend: TrendSummary,
    pub dissatisfaction_trend: TrendSummary,
    pub coverage_saturated: bool,
    pub collision_trend: TrendSummary,
    pub recert_recommended: bool,
    /// When `recert_recommended` is true, whether the current traffic can actually
    /// construct a disjoint eval for re-certification. `None` = not checked (no recert).
    /// `Some(false)` = structurally unconstructible — fall back to behavioral-drift
    /// response (rollback/human review) instead of scheduling an unresolvable recert.
    pub recert_constructible: Option<bool>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TrendSummary {
    pub direction: String,
    pub current: f32,
    pub baseline: f32,
    pub deviation_z: f32,
}

// ---------------------------------------------------------------------------
// §2 — CUSUM deviation detector (pure, testable)
// ---------------------------------------------------------------------------

/// Tabular CUSUM (cumulative sum) for detecting a sustained upward or downward
/// shift against a rolling baseline. Returns `(cusum_high, cusum_low)` — the
/// high-side and low-side cumulative deviations. An alarm fires when either
/// exceeds `threshold_h`. `allowance_k` is the slack per observation (half the
/// minimum shift to detect).
pub fn cusum_update(
    cusum_high: f32,
    cusum_low: f32,
    observation: f32,
    baseline: f32,
    allowance_k: f32,
) -> (f32, f32) {
    let hi = (cusum_high + (observation - baseline) - allowance_k).max(0.0);
    let lo = (cusum_low + (baseline - observation) - allowance_k).max(0.0);
    (hi, lo)
}

/// Check whether the CUSUM state exceeds the alarm threshold on either side.
pub fn cusum_alarm(cusum_high: f32, cusum_low: f32, threshold_h: f32) -> bool {
    cusum_high > threshold_h || cusum_low > threshold_h
}

/// Compute z-score deviation of `value` from a rolling baseline with known
/// `baseline_mean` and `baseline_std`. Returns 0 if std is near-zero.
pub fn z_score_deviation(value: f32, baseline_mean: f32, baseline_std: f32) -> f32 {
    if baseline_std < 1e-9 {
        return 0.0;
    }
    (value - baseline_mean) / baseline_std
}

/// Rolling baseline: mean and std of the last `window_size` observations.
pub fn rolling_baseline(history: &[f32], window_size: usize) -> (f32, f32) {
    if history.is_empty() {
        return (0.0, 0.0);
    }
    let n = history.len().min(window_size);
    let tail = &history[history.len() - n..];
    let mean = tail.iter().sum::<f32>() / n as f32;
    let var = tail.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n as f32;
    (mean, var.sqrt())
}

// ---------------------------------------------------------------------------
// §1 — Coverage elasticity + saturation
// ---------------------------------------------------------------------------

/// Coverage elasticity = Δfallthrough / Δaliases (negative means aliases are
/// reducing fallthrough; near-zero = saturated). `saturation_flag` is true when
/// the latest elasticity is near zero (aliases grow but fallthrough is flat).
pub fn coverage_elasticity(
    fallthrough_history: &[f32],
    alias_count_history: &[usize],
) -> (f32, bool) {
    if fallthrough_history.len() < 2 || alias_count_history.len() < 2 {
        return (0.0, false);
    }
    let n = fallthrough_history.len().min(alias_count_history.len());
    let df = fallthrough_history[n - 1] - fallthrough_history[n - 2];
    let da = alias_count_history[n - 1] as f32 - alias_count_history[n - 2] as f32;
    if da.abs() < 1e-9 {
        return (0.0, false);
    }
    let elasticity = df / da;
    let saturated = elasticity.abs() < 0.001 && da > 0.0;
    (elasticity, saturated)
}

// ---------------------------------------------------------------------------
// §1 — Cross-domain collision rate
// ---------------------------------------------------------------------------

/// Fraction of turns where a foreign-domain node activated above threshold.
/// `foreign_activations` = count of turns with a foreign-domain hit;
/// `total_turns` = all turns in the window.
pub fn cross_domain_collision_rate(foreign_activations: usize, total_turns: usize) -> f32 {
    if total_turns == 0 {
        0.0
    } else {
        foreign_activations as f32 / total_turns as f32
    }
}

// ---------------------------------------------------------------------------
// §1 — Encoder-shadow stability diff
// ---------------------------------------------------------------------------

/// Fraction of live traffic that routes differently under two encoder versions.
/// `routing_pairs` = `(old_route, new_route)` per turn; returns the mismatch rate.
pub fn encoder_stability_diff(routing_pairs: &[(String, String)]) -> f32 {
    if routing_pairs.is_empty() {
        return 0.0;
    }
    let mismatches = routing_pairs.iter().filter(|(a, b)| a != b).count();
    mismatches as f32 / routing_pairs.len() as f32
}

// ---------------------------------------------------------------------------
// §2 — Alert persistence + cause classification
// ---------------------------------------------------------------------------

/// Minimum consecutive windows a deviation must persist before an alert fires.
pub const ALERT_PERSISTENCE_MIN: usize = 3;
/// Z-score threshold for a signal to count as "deviating" in a given window.
pub const ALERT_Z_THRESHOLD: f32 = 2.5;
/// Z-score threshold for critical severity (vs warning).
pub const ALERT_Z_CRITICAL: f32 = 4.0;

/// Given a history of z-scores for a signal and a log of system changes, produce
/// the current change-point (if deviating) and an alert (if persisted long enough).
/// `system_changes` maps window index → description of the system change (if any).
pub fn evaluate_signal_drift(
    signal_name: &str,
    z_history: &[f32],
    system_changes: &HashMap<usize, String>,
) -> (Option<ChangePoint>, Option<DriftAlert>) {
    if z_history.is_empty() {
        return (None, None);
    }
    // Count consecutive deviating windows from the tail.
    let mut persisted = 0usize;
    let mut first_deviating_idx = z_history.len();
    for i in (0..z_history.len()).rev() {
        if z_history[i].abs() >= ALERT_Z_THRESHOLD {
            persisted += 1;
            first_deviating_idx = i;
        } else {
            break;
        }
    }
    if persisted == 0 {
        return (None, None);
    }

    // Cause: if any system change coincides with the first deviating window, it's system-drift.
    let cause = if let Some(desc) = system_changes.get(&first_deviating_idx) {
        DriftCause::System {
            description: desc.clone(),
        }
    } else {
        DriftCause::World
    };

    let cp = ChangePoint {
        signal: signal_name.to_string(),
        cause: cause.clone(),
        persisted_windows: persisted,
    };

    let alert = if persisted >= ALERT_PERSISTENCE_MIN {
        let latest_z = z_history.last().copied().unwrap_or(0.0);
        let severity = if latest_z.abs() >= ALERT_Z_CRITICAL {
            AlertSeverity::Critical
        } else {
            AlertSeverity::Warning
        };
        let direction = if latest_z > 0.0 { "rising" } else { "falling" };
        Some(DriftAlert {
            signal: signal_name.to_string(),
            cause: cause.clone(),
            persisted_windows: persisted,
            severity,
            message: format!(
                "{} {} for {} consecutive windows (z={:.2}, cause={})",
                signal_name,
                direction,
                persisted,
                latest_z,
                cause.as_str()
            ),
        })
    } else {
        None
    };

    (Some(cp), alert)
}

/// Determine whether a re-certification run should be recommended: sustained
/// world-drift on fallthrough or dissatisfaction (the live distribution may have
/// moved past what the encoder was certified on).
pub fn recommend_recert(alerts: &[DriftAlert]) -> bool {
    alerts.iter().any(|a| {
        matches!(a.cause, DriftCause::World)
            && (a.signal == "fallthrough" || a.signal == "dissatisfaction")
            && a.persisted_windows >= ALERT_PERSISTENCE_MIN * 2
    })
}

/// Render a `DriftWindow` to its append-only JSON artifact.
impl DriftWindow {
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".into())
    }

    pub fn filename(&self) -> String {
        format!("drift_{}_{}.json", self.domain, self.window_id)
    }
}

/// Render a `DriftReport` to JSON.
impl DriftReport {
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".into())
    }
}

/// Build a summary report from a series of drift windows.
/// When `traffic_pairs` is provided and recert is recommended, checks whether
/// the current traffic can construct a disjoint eval for re-certification.
pub fn build_drift_report(
    domain: &str,
    windows: &[DriftWindow],
    traffic_pairs: Option<&[(String, String)]>,
) -> DriftReport {
    let latest = windows.last();

    let ft_history: Vec<f32> = windows.iter().map(|w| w.fallthrough_rate).collect();
    let ent_history: Vec<f32> = windows.iter().map(|w| w.routing_entropy_p50).collect();
    let dis_history: Vec<f32> = windows.iter().map(|w| w.dissatisfaction_rate).collect();
    let col_history: Vec<f32> = windows
        .iter()
        .map(|w| w.cross_domain_collision_rate)
        .collect();

    let trend_of = |history: &[f32]| -> TrendSummary {
        let (mean, std) = rolling_baseline(history, 12);
        let current = history.last().copied().unwrap_or(0.0);
        let z = z_score_deviation(current, mean, std);
        let direction = if z > ALERT_Z_THRESHOLD {
            "rising"
        } else if z < -ALERT_Z_THRESHOLD {
            "falling"
        } else {
            "stable"
        };
        TrendSummary {
            direction: direction.to_string(),
            current,
            baseline: mean,
            deviation_z: z,
        }
    };

    let active_alerts: Vec<DriftAlert> = latest.map(|w| w.alerts.clone()).unwrap_or_default();

    let recert = recommend_recert(&active_alerts);

    let recert_constructible = if recert {
        traffic_pairs.map(|pairs| {
            let scan = scan_corpus_disjointness(pairs, "wbc", 4);
            scan.n_seen_elsewhere >= DISJOINT_MIN_N
        })
    } else {
        None
    };

    DriftReport {
        domain: domain.to_string(),
        n_windows: windows.len(),
        latest_window: latest.map(|w| w.window_id.clone()),
        recert_recommended: recert,
        recert_constructible,
        active_alerts,
        fallthrough_trend: trend_of(&ft_history),
        entropy_trend: trend_of(&ent_history),
        dissatisfaction_trend: trend_of(&dis_history),
        coverage_saturated: latest.map(|w| w.saturation_flag).unwrap_or(false),
        collision_trend: trend_of(&col_history),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The phrase embedder is process-global; serialize tests that install/clear it so
    // the default parallel test runner cannot let them stomp on each other.
    static EMBEDDER_TEST_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn malformed_capture_screen_keeps_real_phrases_drops_garble() {
        // Legitimate phrases pass — crucially including "bad cat", the clean PHRASE whose
        // *response* garbled. Screening the response would wrongly discard a real labeling target.
        for ok in [
            "bad cat",
            "Hey Luna",
            "what is 2 plus 2",
            "Here's some sushi for you 🍣",
            "good morning sunshine",
            "are you having a nice day?",
        ] {
            assert_eq!(
                malformed_capture_reason(ok),
                None,
                "false positive on {ok:?}"
            );
        }
        // Empty / MASK-leak / decode-collapse soup are quarantined before they can reach the queue.
        assert_eq!(malformed_capture_reason("   "), Some("empty"));
        assert_eq!(
            malformed_capture_reason(
                "I shoes brave the brush bestow MASK Some, vibrates sleeping just"
            ),
            Some("mask_leak"),
        );
        assert_eq!(
            malformed_capture_reason("schedule both nothing across puddle brush"),
            Some("decode_collapse"),
        );
        // A bare "mask" inside a normal word/sentence must NOT trip the standalone-token check.
        assert_eq!(malformed_capture_reason("do you like my face mask?"), None);
    }

    #[test]
    fn cluster_requires_k_min() {
        let params = GroundingLoopParams {
            k_min: 3,
            tau_cluster: 0.9,
            ..Default::default()
        };
        let buffered = vec![
            ("a".into(), vec![1.0, 0.0]),
            ("b".into(), vec![0.99, 0.01]),
            ("c".into(), vec![0.98, 0.02]),
        ];
        let clusters = cluster_buffered_new_nodes(&buffered, &params);
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].len(), 3);
    }

    #[test]
    fn cata_embedder_self_similarity_is_one() {
        let _g = EMBEDDER_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let rt = LanguageRuntime::new(crate::dimension::language::LanguageConfig::default());
        let corpus = [
            "bitcoin btc holders stack sats onchain",
            "ethereum ether network gas fees apply",
            "a dex is a decentralized exchange to swap tokens",
        ];
        install_phrase_embedder_from_corpus(&corpus, 256);
        let (a, _) = embed_phrase(&rt, "bitcoin btc sats").unwrap();
        let (b, _) = embed_phrase(&rt, "bitcoin btc sats").unwrap();
        assert!(!a.is_empty(), "empty embedding");
        assert!(
            a.iter().any(|x| x.abs() > 1e-9),
            "degenerate (all-zero) embedding"
        );
        let sim = crate::dimension::embedding::cosine_similarity(&a, &b);
        assert!(sim > 0.99, "self sim={sim}");
        // shared-token overlap > disjoint-token pair (lexical structure present).
        let (shared, _) = embed_phrase(&rt, "bitcoin btc").unwrap();
        let (disjoint, _) = embed_phrase(&rt, "ethereum gas").unwrap();
        let sim_shared = crate::dimension::embedding::cosine_similarity(&a, &shared);
        let sim_disjoint = crate::dimension::embedding::cosine_similarity(&a, &disjoint);
        assert!(
            sim_shared > sim_disjoint,
            "shared {sim_shared} !> disjoint {sim_disjoint}"
        );
        clear_phrase_embedder();
    }

    fn ctrl_capture(phrase: &str, concept: &str, split: CaptureSplit) -> FailureCapture {
        FailureCapture {
            phrase: phrase.into(),
            encoder_embedding: Vec::new(),
            activated_nodes: Vec::new(),
            max_confidence: 0.0,
            entropy_bits: None,
            trigger_reason: FailureTrigger::NoNodeActivated,
            downstream_signal: None,
            timestamp_unix: 0,
            domain_context: "runtime".into(),
            inferred_concept_id: concept.into(),
            split,
            provenance: PhraseProvenance::real(phrase),
        }
    }

    /// POSITIVE CONTROL for the certifier: a synthetic *semantic* geometry where a
    /// single approved alias genuinely extends held-out coverage. Mirrors the lexical
    /// negative result — if this ever stops returning GenuineCoverageImprovement, the
    /// certifier has lost the ability to recognize real generalization.
    #[test]
    fn positive_control_certifies_genuine() {
        let _g = EMBEDDER_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let rt = LanguageRuntime::new(crate::dimension::language::LanguageConfig::default());
        // Each concept anchor is a basis vector; propose phrases lean toward the anchor,
        // certify (held-out) phrases lean past the boundary toward the next concept, so
        // at baseline they misroute and only an added alias pulls the centroid over.
        let mut map: HashMap<String, Vec<f32>> = HashMap::new();
        map.insert("concept_a".into(), vec![1.0, 0.0, 0.0]);
        map.insert("concept_b".into(), vec![0.0, 1.0, 0.0]);
        map.insert("concept_c".into(), vec![0.0, 0.0, 1.0]);
        map.insert("alpha proposal phrase".into(), vec![0.8, 0.6, 0.0]);
        map.insert("beta proposal phrase".into(), vec![0.0, 0.8, 0.6]);
        map.insert("gamma proposal phrase".into(), vec![0.6, 0.0, 0.8]);
        map.insert("alpha certify phrase".into(), vec![0.6, 0.8, 0.0]);
        map.insert("beta certify phrase".into(), vec![0.0, 0.6, 0.8]);
        map.insert("gamma certify phrase".into(), vec![0.8, 0.0, 0.6]);
        install_vector_embedder(map);

        let params = GroundingLoopParams::default();
        let d = GroundingFleetDomain::Runtime;
        let before = build_grounding_index_from_nodes(
            &rt,
            &[
                (d, "concept_a".into(), vec![]),
                (d, "concept_b".into(), vec![]),
                (d, "concept_c".into(), vec![]),
            ],
            &params,
        )
        .unwrap();

        // Auto-proposer routes each propose phrase to the correct node at baseline.
        let (pa, _) = embed_phrase(&rt, "alpha proposal phrase").unwrap();
        match propose_for_phrase("alpha proposal phrase", &pa, &before, &params, Some(d)) {
            Some(ProposalKind::Alias { target_node, .. }) => assert_eq!(target_node, "concept_a"),
            other => panic!("expected alias proposal to concept_a, got {other:?}"),
        }

        let after = build_grounding_index_from_nodes(
            &rt,
            &[
                (d, "concept_a".into(), vec!["alpha proposal phrase".into()]),
                (d, "concept_b".into(), vec!["beta proposal phrase".into()]),
                (d, "concept_c".into(), vec!["gamma proposal phrase".into()]),
            ],
            &params,
        )
        .unwrap();

        let captures = vec![
            ctrl_capture("alpha proposal phrase", "concept_a", CaptureSplit::Propose),
            ctrl_capture("beta proposal phrase", "concept_b", CaptureSplit::Propose),
            ctrl_capture("gamma proposal phrase", "concept_c", CaptureSplit::Propose),
            ctrl_capture("alpha certify phrase", "concept_a", CaptureSplit::Certify),
            ctrl_capture("beta certify phrase", "concept_b", CaptureSplit::Certify),
            ctrl_capture("gamma certify phrase", "concept_c", CaptureSplit::Certify),
        ];

        let (before_m, after_m) = certify_batch(&captures, &rt, &before, &after, d).unwrap();
        assert!(
            before_m.held_out_accuracy < 0.34,
            "baseline held-out {}",
            before_m.held_out_accuracy
        );
        assert!(
            after_m.held_out_accuracy > 0.99,
            "after held-out {}",
            after_m.held_out_accuracy
        );
        assert!(
            after_m.captured_accuracy > 0.99,
            "after captured {}",
            after_m.captured_accuracy
        );
        let verdict = decide_batch_verdict(&before_m, &after_m, &params, false);
        assert_eq!(
            verdict,
            BatchVerdict::GenuineCoverageImprovement,
            "verdict {verdict:?}"
        );
        clear_phrase_embedder();
    }

    #[test]
    fn overlap_defined_at_feature_granularity_not_words() {
        // "vacuum" and "vacuuming" are whole-word disjoint but share char trigrams, so the
        // encoder-granularity filter must report them as NON-disjoint (overlap > 0).
        let a = phrase_feature_set("the vacuum");
        let b = phrase_feature_set("vacuuming now");
        let word_a = restrict_features(&a, "w");
        let word_b = restrict_features(&b, "w");
        // Whole-word level: "vacuum" != "vacuuming" → disjoint.
        assert_eq!(
            word_a.intersection(&word_b).count(),
            0,
            "words should be disjoint"
        );
        // Encoder (union) level: shared "vac","acu","cuu","uum" trigrams → NOT disjoint.
        assert!(
            feature_overlap_fraction(&b, &a) > 0.0,
            "trigram overlap must be detected"
        );
    }

    #[test]
    fn wilson_widens_for_small_n() {
        let (lo_big, hi_big) = wilson_interval(50, 100, 1.96);
        let (lo_small, hi_small) = wilson_interval(1, 2, 1.96);
        assert!(
            (hi_small - lo_small) > (hi_big - lo_big),
            "small-n CI must be wider"
        );
        assert!(lo_big > 0.3 && hi_big < 0.7);
    }

    #[test]
    fn supervised_encoder_clusters_paraphrases_by_label() {
        let _g = EMBEDDER_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // Disjoint-token paraphrases of the same concept must embed closer than
        // cross-concept phrases — the non-lexical structure the certifier needs.
        let samples = vec![
            ("hey there".to_string(), "greet".to_string()),
            ("good morning".to_string(), "greet".to_string()),
            ("hello friend".to_string(), "greet".to_string()),
            ("hi how are you".to_string(), "greet".to_string()),
            ("i want food".to_string(), "feed".to_string()),
            ("feed me now".to_string(), "feed".to_string()),
            ("dinner please".to_string(), "feed".to_string()),
            ("i am hungry".to_string(), "feed".to_string()),
        ];
        let enc = SupervisedEncoder::train(&samples, 2048, 80).expect("train");
        install_supervised_embedder(enc);
        let rt = LanguageRuntime::new(crate::dimension::language::LanguageConfig::default());
        // Held-out phrase reuses greet words ("hello"/"morning") in a new combination it
        // never saw verbatim; it must embed closer to a greet phrase than a feed phrase.
        let (held, _) = embed_phrase(&rt, "hello good morning").unwrap();
        let (greet, _) = embed_phrase(&rt, "hey there").unwrap();
        let (feed, _) = embed_phrase(&rt, "feed me now").unwrap();
        let same = crate::dimension::embedding::cosine_similarity(&held, &greet);
        let cross = crate::dimension::embedding::cosine_similarity(&held, &feed);
        assert!(same > cross, "same {same} !> cross {cross}");
        clear_phrase_embedder();
    }

    #[test]
    fn bridge_routed_never_degenerate_self_sim() {
        // Even without a dictionary, the bridge-routed fallback must not collapse to zero
        // on repeated identical input (self-similarity defined).
        let rt = LanguageRuntime::new(crate::dimension::language::LanguageConfig::default());
        let (a, _) = embed_phrase_mode(&rt, "bitcoin", RepresentationMode::RawCliffordE8).unwrap();
        let (b, _) = embed_phrase_mode(&rt, "bitcoin", RepresentationMode::RawCliffordE8).unwrap();
        assert_eq!(a.len(), b.len());
    }

    // ---------------------------------------------------------------------
    // Certifier-first pipeline: pure logic (firewall, state machine, hash)
    // ---------------------------------------------------------------------

    fn cap_with(
        phrase: &str,
        concept: &str,
        split: CaptureSplit,
        prov: PhraseProvenance,
    ) -> FailureCapture {
        FailureCapture {
            phrase: phrase.into(),
            encoder_embedding: Vec::new(),
            activated_nodes: Vec::new(),
            max_confidence: 0.0,
            entropy_bits: None,
            trigger_reason: FailureTrigger::NoNodeActivated,
            downstream_signal: None,
            timestamp_unix: 0,
            domain_context: "runtime".into(),
            inferred_concept_id: concept.into(),
            split,
            provenance: prov,
        }
    }

    #[test]
    fn firewall_clean_when_certify_is_real_traffic_only() {
        let caps = vec![
            cap_with(
                "p1",
                "c",
                CaptureSplit::Propose,
                PhraseProvenance::real("p1"),
            ),
            cap_with(
                "c1",
                "c",
                CaptureSplit::Certify,
                PhraseProvenance::real("c1"),
            ),
        ];
        let r = run_augmentation_firewall(&caps);
        assert!(r.clean, "should be clean: {:?}", r.violations);
        assert_eq!(r.certify_real, 1);
    }

    #[test]
    fn firewall_rejects_augmented_in_certify() {
        let caps = vec![
            cap_with(
                "p1",
                "c",
                CaptureSplit::Propose,
                PhraseProvenance::real("p1"),
            ),
            cap_with(
                "c1",
                "c",
                CaptureSplit::Certify,
                PhraseProvenance {
                    kind: ProvenanceKind::Augmented,
                    phrase_id: "c1".into(),
                    derived_from: vec!["p1".into()],
                },
            ),
        ];
        let r = run_augmentation_firewall(&caps);
        assert!(!r.clean, "augmented certify must be dirty");
    }

    #[test]
    fn firewall_rejects_lineage_crossing() {
        // A training augmented phrase derived from certify phrase 'c1' ⇒ the encoder trained
        // on a rephrasing of held-out traffic. Must be flagged even though provenance kinds
        // are individually legal.
        let caps = vec![
            cap_with(
                "augmented training phrase",
                "c",
                CaptureSplit::Propose,
                PhraseProvenance {
                    kind: ProvenanceKind::Augmented,
                    phrase_id: "p_aug".into(),
                    derived_from: vec!["c1".into()],
                },
            ),
            cap_with(
                "certify one",
                "c",
                CaptureSplit::Certify,
                PhraseProvenance::real("c1"),
            ),
        ];
        let r = run_augmentation_firewall(&caps);
        assert!(!r.clean, "lineage crossing must be flagged");
        assert!(r.violations.iter().any(|v| v.contains("lineage crossing")));
    }

    fn base_inputs() -> VerdictInputs {
        VerdictInputs {
            positive_control_collapsed: true,
            firewall_clean: true,
            disjoint_gen_a_n: 40,
            disjoint_gen_a_ci_width: 0.15,
            disjoint_semantic_lift: 0.10,
            lift_ci_lo: 0.03,
            collision_delta: 0.0,
            memorization_gap: 0.10,
            strict_disjoint_level: true,
        }
    }

    #[test]
    fn verdict_invalid_when_positive_control_fails() {
        let mut i = base_inputs();
        i.positive_control_collapsed = false;
        assert_eq!(decide_encoder_verdict(&i), Verdict::Invalid);
    }

    #[test]
    fn verdict_invalid_when_firewall_dirty() {
        let mut i = base_inputs();
        i.firewall_clean = false;
        assert_eq!(decide_encoder_verdict(&i), Verdict::Invalid);
    }

    #[test]
    fn verdict_below_resolution_on_tiny_bin() {
        let mut i = base_inputs();
        i.disjoint_gen_a_n = 3; // < DISJOINT_MIN_N
        assert_eq!(decide_encoder_verdict(&i), Verdict::BelowResolution);
        // Wide CI also triggers it even with enough samples.
        let mut j = base_inputs();
        j.disjoint_gen_a_ci_width = 0.5;
        assert_eq!(decide_encoder_verdict(&j), Verdict::BelowResolution);
    }

    #[test]
    fn verdict_fail_memorization_when_lift_not_positive() {
        // The 20.7% case: pooled high but disjoint lift at/under the floor.
        let mut i = base_inputs();
        i.disjoint_semantic_lift = 0.0;
        i.lift_ci_lo = -0.05;
        assert_eq!(decide_encoder_verdict(&i), Verdict::FailMemorization);
        // Positive point estimate but CI includes 0 ⇒ still memorization.
        let mut j = base_inputs();
        j.disjoint_semantic_lift = 0.04;
        j.lift_ci_lo = -0.01;
        assert_eq!(decide_encoder_verdict(&j), Verdict::FailMemorization);
    }

    #[test]
    fn verdict_fail_collision_when_lift_but_collides() {
        let mut i = base_inputs();
        i.collision_delta = 0.02;
        assert_eq!(decide_encoder_verdict(&i), Verdict::FailCollision);
    }

    #[test]
    fn verdict_fail_memorization_when_gap_blown() {
        let mut i = base_inputs();
        i.memorization_gap = 0.9; // lookup table even with positive lift
        assert_eq!(decide_encoder_verdict(&i), Verdict::FailMemorization);
    }

    #[test]
    fn verdict_pass_only_on_ci_clear_positive_lift() {
        let i = base_inputs();
        assert_eq!(decide_encoder_verdict(&i), Verdict::Pass);
        assert!(decide_encoder_verdict(&i).is_promotable());
    }

    #[test]
    fn verdict_coarse_level_pass_is_provisional_not_promotable() {
        // Same cleared lift, but resolved via a fallback level coarser than `wbc`: the fallback
        // must not launder a surface-overlap pass into a promotable one.
        let mut i = base_inputs();
        i.strict_disjoint_level = false;
        assert_eq!(decide_encoder_verdict(&i), Verdict::PassProvisional);
        assert!(!decide_encoder_verdict(&i).is_promotable());
    }

    #[test]
    fn scan_corpus_disjointness_is_leave_one_out_not_self_overlap() {
        // A class whose phrases share no in-class features except one unique-token phrase that
        // also shares a token with another class ⇒ exactly one seen-elsewhere disjoint phrase.
        let pairs = vec![
            ("alpha bravo".to_string(), "c1".to_string()),
            ("charlie delta".to_string(), "c1".to_string()),
            ("alpha echo".to_string(), "c2".to_string()), // shares "alpha" with c1
            ("foxtrot golf".to_string(), "c2".to_string()),
        ];
        let s = scan_corpus_disjointness(&pairs, "w", 2);
        // Every phrase's words are unique within its class here, so all 4 are disjoint-from-class;
        // "alpha"-bearing phrases are seen-elsewhere (alpha spans c1 and c2).
        assert_eq!(s.n_phrases, 4);
        assert!(
            s.n_seen_elsewhere >= 1,
            "the shared-token phrases must count as seen-elsewhere"
        );
        assert!(s.n_disjoint >= s.n_seen_elsewhere);
    }

    #[test]
    fn audit_disjoint_eval_distinguishes_surface_disjoint_from_overlapping() {
        // Training: each class has distinctive surface tokens.
        let train = vec![
            (
                "reset my password please".to_string(),
                "support".to_string(),
            ),
            ("cannot login to account".to_string(), "support".to_string()),
            ("write a rust function".to_string(), "coding".to_string()),
            (
                "implement a parser module".to_string(),
                "coding".to_string(),
            ),
        ];
        // Surface-overlapping eval (shares tokens with own class) → not disjoint.
        let overlapping = vec![
            ("reset password account".to_string(), "support".to_string()),
            ("write rust parser".to_string(), "coding".to_string()),
        ];
        let a = audit_disjoint_eval(&train, &overlapping, "wbc");
        assert_eq!(
            a.n_seen_elsewhere, 0,
            "overlapping eval has no disjoint phrases"
        );
        assert!(!a.resolvable);
    }

    #[test]
    fn data_hash_is_order_independent_and_sensitive() {
        let a = cap_with(
            "alpha",
            "c1",
            CaptureSplit::Propose,
            PhraseProvenance::real("a"),
        );
        let b = cap_with(
            "beta",
            "c2",
            CaptureSplit::Certify,
            PhraseProvenance::real("b"),
        );
        let nodes = vec!["c1".to_string(), "c2".to_string()];
        let h1 = data_hash(&[a.clone(), b.clone()], &nodes);
        let h2 = data_hash(&[b.clone(), a.clone()], &nodes); // reordered
        assert_eq!(h1, h2, "hash must be order-independent");
        let c = cap_with(
            "beta",
            "c3",
            CaptureSplit::Certify,
            PhraseProvenance::real("b"),
        );
        let h3 = data_hash(&[a, c], &nodes);
        assert_ne!(h1, h3, "different content must change the hash");
    }

    // -----------------------------------------------------------------------
    // §16 — P4 Drift telemetry tests
    // -----------------------------------------------------------------------

    #[test]
    fn cusum_detects_sustained_upward_shift() {
        let (mut hi, mut lo) = (0.0f32, 0.0f32);
        let baseline = 0.05;
        let k = 0.01;
        let threshold = 0.10;
        // 5 observations at baseline → no alarm.
        for _ in 0..5 {
            let (h, l) = cusum_update(hi, lo, baseline, baseline, k);
            hi = h;
            lo = l;
        }
        assert!(!cusum_alarm(hi, lo, threshold), "no alarm at baseline");
        // Sustained shift: observation = 0.12 for 5 windows.
        for _ in 0..5 {
            let (h, l) = cusum_update(hi, lo, 0.12, baseline, k);
            hi = h;
            lo = l;
        }
        assert!(
            cusum_alarm(hi, lo, threshold),
            "alarm after sustained high shift"
        );
    }

    #[test]
    fn z_score_deviation_near_zero_when_at_baseline() {
        let z = z_score_deviation(0.05, 0.05, 0.01);
        assert!(z.abs() < 1e-6);
    }

    #[test]
    fn z_score_deviation_handles_zero_std() {
        let z = z_score_deviation(0.1, 0.05, 0.0);
        assert_eq!(z, 0.0);
    }

    #[test]
    fn rolling_baseline_computes_windowed_stats() {
        let history = vec![0.05, 0.05, 0.05, 0.10, 0.10];
        let (mean, std) = rolling_baseline(&history, 3);
        let expected_mean = (0.05 + 0.10 + 0.10) / 3.0;
        assert!((mean - expected_mean).abs() < 1e-5);
        assert!(std > 0.0, "std should be nonzero with mixed values");
    }

    #[test]
    fn coverage_elasticity_detects_saturation() {
        // Aliases grew but fallthrough didn't budge.
        let ft = vec![0.07, 0.07];
        let al = vec![100, 120];
        let (_, sat) = coverage_elasticity(&ft, &al);
        assert!(sat, "flat fallthrough with alias growth = saturated");
    }

    #[test]
    fn coverage_elasticity_detects_improvement() {
        let ft = vec![0.10, 0.06];
        let al = vec![100, 120];
        let (elas, sat) = coverage_elasticity(&ft, &al);
        assert!(
            elas < 0.0,
            "negative elasticity = aliases reducing fallthrough"
        );
        assert!(!sat);
    }

    #[test]
    fn encoder_stability_diff_all_same() {
        let pairs = vec![
            ("node_a".to_string(), "node_a".to_string()),
            ("node_b".to_string(), "node_b".to_string()),
        ];
        assert_eq!(encoder_stability_diff(&pairs), 0.0);
    }

    #[test]
    fn encoder_stability_diff_detects_mismatch() {
        let pairs = vec![
            ("node_a".to_string(), "node_b".to_string()),
            ("node_b".to_string(), "node_b".to_string()),
        ];
        assert!((encoder_stability_diff(&pairs) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn alert_requires_persistence_not_single_window() {
        // 2 deviating windows (below threshold of 3) → change point but no alert.
        let z = vec![0.0, 0.0, 3.0, 3.0];
        let (cp, alert) = evaluate_signal_drift("fallthrough", &z, &HashMap::new());
        assert!(cp.is_some(), "change point detected");
        assert!(alert.is_none(), "no alert below persistence threshold");
    }

    #[test]
    fn alert_fires_after_persistence_threshold() {
        let z = vec![0.0, 3.0, 3.0, 3.0];
        let (cp, alert) = evaluate_signal_drift("fallthrough", &z, &HashMap::new());
        assert!(cp.is_some());
        let a = alert.expect("alert should fire after 3 consecutive deviating windows");
        assert_eq!(a.persisted_windows, 3);
        assert_eq!(a.severity, AlertSeverity::Warning);
        assert!(matches!(a.cause, DriftCause::World));
    }

    #[test]
    fn alert_cause_is_system_when_change_coincides() {
        let z = vec![0.0, 0.0, 4.5, 4.5, 4.5];
        let mut sys = HashMap::new();
        sys.insert(2, "encoder swap v1→v2".to_string());
        let (_, alert) = evaluate_signal_drift("fallthrough", &z, &sys);
        let a = alert.expect("alert after 3 persisted");
        assert!(matches!(a.cause, DriftCause::System { .. }));
        assert_eq!(a.severity, AlertSeverity::Critical);
    }

    #[test]
    fn alert_does_not_fire_when_deviation_not_consecutive() {
        // Deviating, then normal, then deviating — not consecutive.
        let z = vec![3.0, 0.5, 3.0];
        let (_, alert) = evaluate_signal_drift("entropy", &z, &HashMap::new());
        assert!(
            alert.is_none(),
            "non-consecutive deviation should not alert"
        );
    }

    #[test]
    fn recommend_recert_on_sustained_world_drift() {
        let alerts = vec![DriftAlert {
            signal: "fallthrough".to_string(),
            cause: DriftCause::World,
            persisted_windows: ALERT_PERSISTENCE_MIN * 2,
            severity: AlertSeverity::Warning,
            message: "test".to_string(),
        }];
        assert!(recommend_recert(&alerts));
    }

    #[test]
    fn no_recert_on_system_drift() {
        let alerts = vec![DriftAlert {
            signal: "fallthrough".to_string(),
            cause: DriftCause::System {
                description: "graph edit".into(),
            },
            persisted_windows: ALERT_PERSISTENCE_MIN * 2,
            severity: AlertSeverity::Warning,
            message: "test".to_string(),
        }];
        assert!(
            !recommend_recert(&alerts),
            "system drift → rollback, not recert"
        );
    }

    #[test]
    fn cross_domain_collision_rate_basic() {
        assert!((cross_domain_collision_rate(5, 100) - 0.05).abs() < 1e-6);
        assert_eq!(cross_domain_collision_rate(0, 0), 0.0);
    }

    #[test]
    fn recert_recommended_but_unconstructible_falls_back() {
        // Simulate a scenario where recert is recommended (world drift) but
        // the traffic is structurally incapable of producing a disjoint eval.
        // All phrases in each class share features → no disjoint examples.
        let pairs: Vec<(String, String)> = vec![
            ("reset password".into(), "support".into()),
            ("reset my password".into(), "support".into()),
            ("reset the password".into(), "support".into()),
            ("password reset".into(), "support".into()),
            ("write code".into(), "coding".into()),
            ("write some code".into(), "coding".into()),
            ("write my code".into(), "coding".into()),
            ("write the code".into(), "coding".into()),
        ];

        let alert = DriftAlert {
            signal: "fallthrough".to_string(),
            cause: DriftCause::World,
            persisted_windows: ALERT_PERSISTENCE_MIN * 2,
            severity: AlertSeverity::Warning,
            message: "sustained world drift".to_string(),
        };
        let window = DriftWindow {
            domain: "test".into(),
            window_id: "w1".into(),
            window_start_unix: 0,
            window_duration_secs: 100,
            fallthrough_rate: 0.15,
            fallthrough_baseline: 0.05,
            fallthrough_alert: true,
            routing_entropy_p50: 0.0,
            entropy_trend: "rising".into(),
            guard_fire_rate: 0.0,
            dissatisfaction_rate: 0.0,
            dissatisfaction_baseline: 0.0,
            coverage_elasticity: 0.0,
            saturation_flag: false,
            total_aliases_fleet: 10,
            cross_domain_collision_rate: 0.0,
            encoder_version: "test".into(),
            encoder_shift_vs_prev: None,
            change_points: Vec::new(),
            alerts: vec![alert],
        };

        let report = build_drift_report("test", &[window], Some(&pairs));
        assert!(report.recert_recommended, "recert should be recommended");
        assert_eq!(
            report.recert_constructible,
            Some(false),
            "traffic with overlapping features must be unconstructible"
        );
    }

    #[test]
    fn verdict_from_str_roundtrips() {
        for v in [
            Verdict::Invalid,
            Verdict::BelowResolution,
            Verdict::FailMemorization,
            Verdict::FailCollision,
            Verdict::PassProvisional,
            Verdict::Pass,
        ] {
            assert_eq!(Verdict::from_str(v.as_str()), v);
        }
    }
}
