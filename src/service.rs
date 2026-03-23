use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::dimension::{
    action_type_one_hot, generate_code_from_action, group_id_one_hot, render_action_template, ActionJson,
    CalibrationDataset, CalibrationReport, CalibrationRequirements, CheckpointSizeSummary, CodeGeneration,
    DimensionManager, DimensionManagerConfig, EpisodicSummary, GeneratedResponse, GenerationHead,
    LanguageConfig, LanguageRoutingDecision, LanguageSample,
};
use crate::dimension::language::DEFAULT_BRIDGE_DIM;
use crate::dimension::group_gen::GEN_COND_DIM;
use crate::dimension::action::{ActionType, ActionPayload};
#[cfg(not(target_arch = "wasm32"))]
use crate::dimension::EncoderPreset;
use crate::spectral::{ProjectModel, EntityKind, HybridEmbedder};
use crate::dimension::tool::{ToolRegistry, ToolSchema, ToolCallInfo, ToolResult};
use crate::dimension::paramecium::InfraciliaryLattice;
use crate::metacognition::{MetaCognition, ReflectionOutcome};
use crate::reasoning::{ReasoningEngine, System2Config};
use crate::types::{EnvironmentConfig, GroupId, Sample};

// ---------------------------------------------------------------------------
// M6: Agent Modes
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentMode {
    ContextFile,
    MicroBrain,
    /// Paramecium mode: lattice-only inference, no neural substrate.
    /// Kilobyte-scale, zero synapses, wave-based program selection.
    Paramecium,
}

impl Default for AgentMode {
    fn default() -> Self {
        AgentMode::MicroBrain
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandoffLogEntry {
    pub from_mode: AgentMode,
    pub to_mode: AgentMode,
    pub confidence: f32,
    pub reason: String,
    #[cfg(not(target_arch = "wasm32"))]
    pub timestamp_ms: u128,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SloConfig {
    pub latency_p95_ms: f64,
    pub max_memory_bytes: u64,
    pub max_checkpoint_domains: usize,
}

impl Default for SloConfig {
    fn default() -> Self {
        Self {
            latency_p95_ms: 50.0,
            max_memory_bytes: 50 * 1024 * 1024,
            max_checkpoint_domains: 100,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SloSnapshot {
    pub latency_samples: Vec<f64>,
    pub latency_p95_ms: f64,
    pub checkpoint_domains: usize,
    pub latency_ok: bool,
    pub checkpoint_ok: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcceptanceReport {
    pub understanding: UnderstandingMetrics,
    pub generation: GenerationMetrics,
    pub continual_learning: ContinualLearningMetrics,
    pub system: SystemMetrics,
    pub modes: ModeMetrics,
    pub passed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnderstandingMetrics {
    pub groups_count: usize,
    pub routing_confidence_streak: u32,
    pub auto_spawn_k: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationMetrics {
    pub template_based: bool,
    pub codegen_languages: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContinualLearningMetrics {
    pub episodic_episodes: usize,
    pub checkpoint_summary: CheckpointSizeSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    pub slo: SloSnapshot,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModeMetrics {
    pub active_mode: AgentMode,
    pub handoff_count: usize,
    pub modes_available: Vec<AgentMode>,
}

// ---------------------------------------------------------------------------
// OCEAN Personality Profile
// ---------------------------------------------------------------------------
//
// Each dimension is 0.0–1.0. Affects generation conditioning, Hopf beam
// scoring, and EMA temporal blending.
//
//   O (Openness):         high → creative, exploratory responses
//   C (Conscientiousness): high → precise, structured responses
//   E (Extraversion):      high → verbose, enthusiastic responses
//   A (Agreeableness):     high → supportive, affirming tone
//   N (Neuroticism):       high → cautious, hedging language

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OceanProfile {
    pub openness: f32,
    pub conscientiousness: f32,
    pub extraversion: f32,
    pub agreeableness: f32,
    pub neuroticism: f32,
}

impl Default for OceanProfile {
    fn default() -> Self {
        Self {
            openness: 0.5,
            conscientiousness: 0.7,
            extraversion: 0.5,
            agreeableness: 0.6,
            neuroticism: 0.3,
        }
    }
}

impl OceanProfile {
    /// Balanced professional assistant: precise, friendly, moderately creative.
    pub fn assistant() -> Self {
        Self { openness: 0.5, conscientiousness: 0.8, extraversion: 0.5, agreeableness: 0.7, neuroticism: 0.2 }
    }

    /// Creative brainstormer: highly open, enthusiastic, less rigid.
    pub fn creative() -> Self {
        Self { openness: 0.9, conscientiousness: 0.4, extraversion: 0.8, agreeableness: 0.6, neuroticism: 0.3 }
    }

    /// Precise engineer: structured, careful, low on fluff.
    pub fn engineer() -> Self {
        Self { openness: 0.4, conscientiousness: 0.9, extraversion: 0.3, agreeableness: 0.5, neuroticism: 0.2 }
    }

    /// Cautious analyst: high conscientiousness and neuroticism (hedging).
    pub fn analyst() -> Self {
        Self { openness: 0.5, conscientiousness: 0.9, extraversion: 0.3, agreeableness: 0.5, neuroticism: 0.7 }
    }

    /// Returns the 5-float vector [O, C, E, A, N].
    pub fn as_vec(&self) -> [f32; 5] {
        [self.openness, self.conscientiousness, self.extraversion, self.agreeableness, self.neuroticism]
    }

    /// Modulate a conditioning vector with personality.
    /// Applies a subtle directional bias in the last 5 dims of the vector,
    /// scaled so personality is a secondary signal (not overriding content).
    pub fn condition_vector(&self, cond: &mut [f32]) {
        let dim = cond.len();
        if dim < 10 { return; }
        let ocean = self.as_vec();
        let scale = 0.15;
        for (i, &o) in ocean.iter().enumerate() {
            let idx = dim - 5 + i;
            cond[idx] += (o - 0.5) * scale;
        }
    }

    /// EMA alpha modulation: extraversion increases alpha (faster adaptation),
    /// conscientiousness decreases it (more memory of prior context).
    pub fn modulated_ema_alpha(&self, base_alpha: f32) -> f32 {
        let shift = (self.extraversion - 0.5) * 0.15 - (self.conscientiousness - 0.5) * 0.1;
        (base_alpha + shift).clamp(0.05, 0.8)
    }

    /// Hopf beam scoring bias: openness favors cross-archetype fragments,
    /// conscientiousness favors same-archetype coherence.
    pub fn hopf_diversity_bonus(&self) -> f32 {
        (self.openness - self.conscientiousness).clamp(-0.3, 0.3)
    }

    /// Gentle drift toward a target value on one dimension. `alpha` controls
    /// step size (0.01–0.05 recommended). Clamps to [0.0, 1.0].
    fn drift(val: &mut f32, direction: f32, alpha: f32) {
        *val = (*val + direction * alpha).clamp(0.0, 1.0);
    }

    /// Apply emergent personality drift from a single feedback event.
    /// Patterns:
    ///   Accept → reward current profile (reduce neuroticism, boost agreeableness)
    ///   Reject → increase caution and precision
    ///   Correct with longer text → user wants detail (boost extraversion)
    ///   Correct with shorter text → user wants conciseness (reduce extraversion)
    pub fn apply_feedback_drift(
        &mut self,
        accepted: bool,
        correction_len_ratio: Option<f32>, // correction_len / response_len; None if no correction
    ) {
        const ALPHA: f32 = 0.02;

        if accepted {
            Self::drift(&mut self.neuroticism, -1.0, ALPHA);
            Self::drift(&mut self.agreeableness, 1.0, ALPHA * 0.5);
        } else {
            Self::drift(&mut self.conscientiousness, 1.0, ALPHA);
            Self::drift(&mut self.neuroticism, 1.0, ALPHA * 0.5);

            if let Some(ratio) = correction_len_ratio {
                if ratio > 1.2 {
                    Self::drift(&mut self.extraversion, 1.0, ALPHA);
                } else if ratio < 0.6 {
                    Self::drift(&mut self.extraversion, -1.0, ALPHA);
                }
            }
        }
    }

    /// Select a conversational framing prefix based on personality and
    /// conversation position. Returns None for low-extraversion/agreeableness
    /// profiles (engineer, analyst) to keep responses terse.
    pub fn conversational_prefix(&self, turn_count: usize, user_text: &str) -> Option<String> {
        // Skip framing for terse personalities
        if self.extraversion < 0.35 && self.agreeableness < 0.45 {
            return None;
        }

        let warmth = (self.agreeableness + self.extraversion) / 2.0;
        let lower = user_text.to_lowercase();

        // First turn openers
        if turn_count <= 1 {
            if warmth > 0.65 {
                if lower.starts_with("how") || lower.starts_with("what") || lower.starts_with("why")
                    || lower.starts_with("explain") || lower.starts_with("describe")
                {
                    return Some(Self::pick_opener_warm(&lower));
                }
                if lower.starts_with("help") || lower.contains("can you") || lower.contains("could you") {
                    return Some("Of course. ".to_string());
                }
                if lower.starts_with("implement") || lower.starts_with("write") || lower.starts_with("create")
                    || lower.starts_with("build") || lower.starts_with("design")
                {
                    return Some(Self::pick_opener_warm(&lower));
                }
            }
            return None;
        }

        // Continuation turns
        if warmth > 0.6 {
            if lower.starts_with("and ") || lower.starts_with("also") || lower.starts_with("what about") {
                return Some("Building on that — ".to_string());
            }
            if lower.starts_with("but ") || lower.starts_with("however") || lower.starts_with("what if") {
                return Some("Good point. ".to_string());
            }
            if lower.starts_with("can you") || lower.starts_with("could you") || lower.starts_with("please") {
                return Some("Sure. ".to_string());
            }
            if lower.starts_with("why") {
                return Some("Here's the reasoning: ".to_string());
            }
        }

        None
    }

    fn pick_opener_warm(lower: &str) -> String {
        if lower.contains("explain") || lower.contains("what is") || lower.contains("what are") {
            "Let me break that down. ".to_string()
        } else if lower.contains("how to") || lower.contains("how do") || lower.contains("how can") {
            "Here's how. ".to_string()
        } else if lower.contains("design") || lower.contains("architect") || lower.contains("build") {
            "Here's an approach. ".to_string()
        } else if lower.contains("implement") || lower.contains("write") || lower.contains("create") {
            "Here's the implementation. ".to_string()
        } else if lower.contains("compare") || lower.contains("difference") || lower.contains("vs") {
            "Let me compare them. ".to_string()
        } else {
            String::new()
        }
    }
}

// ---------------------------------------------------------------------------
// Continuum: online learning configuration
// ---------------------------------------------------------------------------

/// Number of training steps per feedback event (small to avoid forgetting).
const CONTINUUM_STEPS: usize = 3;

/// Configurable Continuum parameters for online learning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContinuumConfig {
    /// Auto-checkpoint interval: save brain every N feedback events. 0 = disabled.
    pub checkpoint_interval: u64,
    /// Minimum session hits before consolidation commits drift to persistent centroids.
    pub min_consolidation_hits: u32,
    /// Maximum feedback events per minute per session (rate limit). 0 = unlimited.
    pub rate_limit_per_minute: u32,
    /// Path for auto-checkpoint files.
    pub checkpoint_path: String,
}

impl Default for ContinuumConfig {
    fn default() -> Self {
        Self {
            checkpoint_interval: 50,
            min_consolidation_hits: 3,
            rate_limit_per_minute: 0,
            checkpoint_path: "brain_continuum.bin".to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// Conversation Context (multi-turn)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ConversationTurn {
    pub role: TurnRole,
    pub text: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnRole {
    User,
    Agent,
}

/// Rolling conversation context for multi-turn dialogue.
#[derive(Debug, Clone)]
pub struct ConversationContext {
    pub history: Vec<ConversationTurn>,
    pub max_turns: usize,
    /// How many recent turns to include in the conditioning prompt for inference.
    pub context_window_size: usize,
    /// Geometric context: running embedding that accumulates across turns
    /// via exponential decay blending. Recent turns dominate, earlier turns
    /// fade but never fully disappear. Dimension matches bridge output (128D).
    pub context_embedding: Vec<f32>,
    /// Per-turn embeddings for geometric trajectory analysis.
    turn_embeddings: Vec<(Vec<f32>, Vec<f32>)>,
    /// Current topic thread — biases retrieval toward continuity.
    pub current_topic: Option<String>,
    /// Current group — for topic continuity detection.
    pub current_group: Option<usize>,
    /// Decay factor for geometric blending (0.0 = no memory, 1.0 = perfect memory).
    pub context_decay: f32,
}

impl Default for ConversationContext {
    fn default() -> Self {
        Self {
            history: Vec::new(),
            max_turns: 50,
            context_window_size: 8,
            context_embedding: Vec::new(),
            turn_embeddings: Vec::new(),
            current_topic: None,
            current_group: None,
            context_decay: 0.65,
        }
    }
}

impl ConversationContext {
    pub fn push_user(&mut self, text: &str) {
        self.history.push(ConversationTurn { role: TurnRole::User, text: text.to_string() });
        self.trim();
    }

    pub fn push_agent(&mut self, text: &str) {
        self.history.push(ConversationTurn { role: TurnRole::Agent, text: text.to_string() });
        self.trim();
    }

    fn trim(&mut self) {
        while self.history.len() > self.max_turns * 2 {
            self.history.remove(0);
        }
        while self.turn_embeddings.len() > self.max_turns {
            self.turn_embeddings.remove(0);
        }
    }

    pub fn turn_count(&self) -> usize {
        self.history.iter().filter(|t| t.role == TurnRole::User).count()
    }

    pub fn is_empty(&self) -> bool {
        self.history.is_empty()
    }

    pub fn clear(&mut self) {
        self.history.clear();
        self.context_embedding.clear();
        self.turn_embeddings.clear();
        self.current_topic = None;
        self.current_group = None;
    }

    /// Blend a new turn's query and response embeddings into the running
    /// geometric context. Uses exponential decay so recent turns dominate
    /// but earlier context never fully disappears.
    pub fn update_geometric_context(
        &mut self,
        query_emb: &[f32],
        response_emb: &[f32],
    ) {
        let dim = query_emb.len().max(response_emb.len());
        if dim == 0 { return; }

        // Midpoint of query + response captures the turn's semantic center
        let mut turn_center = vec![0.0f32; dim];
        for i in 0..dim {
            let q = query_emb.get(i).copied().unwrap_or(0.0);
            let r = response_emb.get(i).copied().unwrap_or(0.0);
            turn_center[i] = (q + r) * 0.5;
        }

        if self.context_embedding.is_empty() {
            self.context_embedding = turn_center.clone();
        } else {
            // Geometric blend: context = decay * old_context + (1 - decay) * new_turn
            let d = self.context_decay;
            self.context_embedding.resize(dim, 0.0);
            for i in 0..dim {
                self.context_embedding[i] = d * self.context_embedding[i]
                    + (1.0 - d) * turn_center[i];
            }
            // L2-normalize to keep on the unit sphere
            let norm: f32 = self.context_embedding.iter()
                .map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-8 {
                for v in &mut self.context_embedding { *v /= norm; }
            }
        }

        self.turn_embeddings.push((query_emb.to_vec(), response_emb.to_vec()));
    }

    /// Modulate a raw query embedding with accumulated conversation context.
    /// Returns a blended embedding that carries forward topic continuity
    /// while preserving the current query's semantic direction.
    pub fn contextualize_query(&self, raw_query: &[f32], context_weight: f32) -> Vec<f32> {
        if self.context_embedding.is_empty() || context_weight <= 0.0 {
            return raw_query.to_vec();
        }
        let dim = raw_query.len().min(self.context_embedding.len());
        let mut blended = vec![0.0f32; raw_query.len()];
        let qw = 1.0 - context_weight;
        for i in 0..dim {
            blended[i] = qw * raw_query[i] + context_weight * self.context_embedding[i];
        }
        for i in dim..raw_query.len() {
            blended[i] = qw * raw_query[i];
        }
        let norm: f32 = blended.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-8 {
            for v in &mut blended { *v /= norm; }
        }
        blended
    }

    /// Detect if the current query is shifting topic based on cosine similarity
    /// with the accumulated context. Low similarity = topic shift.
    pub fn is_topic_shift(&self, query_emb: &[f32], threshold: f32) -> bool {
        if self.context_embedding.is_empty() { return false; }
        let dim = query_emb.len().min(self.context_embedding.len());
        let dot: f32 = (0..dim).map(|i| query_emb[i] * self.context_embedding[i]).sum();
        let nq: f32 = query_emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nc: f32 = self.context_embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if nq < 1e-8 || nc < 1e-8 { return true; }
        let sim = dot / (nq * nc);
        sim < threshold
    }

    /// Format conversation history as context string for the encoder.
    /// Uses a sliding window of the last N turns to keep embedding focused.
    pub fn context_window(&self, window: usize) -> String {
        let recent: Vec<&ConversationTurn> = self.history.iter()
            .rev().take(window * 2).collect::<Vec<_>>().into_iter().rev().collect();
        recent.iter()
            .map(|t| match t.role {
                TurnRole::User => format!("user: {}", t.text),
                TurnRole::Agent => format!("agent: {}", t.text),
            })
            .collect::<Vec<_>>()
            .join(" | ")
    }
}

// ---------------------------------------------------------------------------
// Continuum: feedback and turn context (train-while-on; see docs/CONTINUUM.md)
// ---------------------------------------------------------------------------

/// User feedback for the previous turn. Used for future online learning (training step not yet wired).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Feedback {
    pub outcome: FeedbackOutcome,
    #[serde(default)]
    pub correction: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeedbackOutcome {
    Accept,
    Reject,
    Correct,
}

/// Minimal context for one inference turn, stored for feedback association.
#[derive(Debug, Clone)]
pub struct TurnContext {
    pub message: String,
    pub group_id: Option<GroupId>,
    pub output: String,
    /// Which IndexedGenEnv (by group order index) produced the response.
    pub effective_gidx: Option<usize>,
    /// Which lattice program within that env was selected.
    pub program_idx: Option<usize>,
}

// ---------------------------------------------------------------------------
// LanguageService — core shared service
// ---------------------------------------------------------------------------

pub struct LanguageService {
    /// Named checkpoints: e.g. "default", "my-brain", "user-a". One is active.
    pub brains: HashMap<String, DimensionManager>,
    /// Key into `brains`; inference uses this checkpoint.
    pub active_brain: String,
    pub dm: DimensionManager,
    pub support_gid: GroupId,
    pub coding_gid: GroupId,
    pub calibration: CalibrationReport,
    pub mode: AgentMode,
    pub slo_config: SloConfig,
    latency_log: Vec<f64>,
    handoff_log: Vec<HandoffLogEntry>,
    context_snippets: Vec<String>,
    /// Last turn (message, routing, output) for feedback association; see CONTINUUM.md.
    last_turn: Option<TurnContext>,
    /// Agent identity for "who are you" responses.
    pub agent_name: String,
    pub agent_creator: String,
    /// OCEAN personality profile — conditions generation and conversation style.
    pub personality: OceanProfile,
    /// Multi-turn conversation context.
    pub conversation: ConversationContext,
    /// Leech-lattice spatial index of the project (files, symbols, patterns).
    pub project_model: ProjectModel,
    /// Tool registry — schemas for external tool invocation.
    pub tool_registry: ToolRegistry,
    /// Continuum: count of feedback events since startup (for auto-checkpoint).
    continuum_feedback_count: u64,
    /// Continuum: timestamp of last feedback (for rate limiting).
    last_feedback_time: std::time::Instant,
    /// Continuum: feedback count in current rate-limit window.
    feedback_window_count: u32,
    /// Continuum: configurable online learning parameters.
    pub continuum_config: ContinuumConfig,
    /// Paramecium: lattice-only inference engine (optional, built from brain).
    pub paramecium: Option<InfraciliaryLattice>,
    /// Reasoning engine: hippocampal-prefrontal circuit for cross-group composition.
    pub reasoning: Option<ReasoningEngine>,
    /// GrowformerLang: meta-language codebook for concept-level routing.
    pub meta_codebook: Option<crate::growformer_lang::MetaCodebook>,
    /// MetaCognition: reflective quality gate on System 1 generation output.
    pub metacognition: Option<MetaCognition>,
    /// System 2 configuration for deliberate multi-step reasoning.
    pub system2_config: System2Config,
}

impl LanguageService {
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new_default() -> Result<Self, String> {
        let (dm, support_gid, coding_gid, report) = build_language_demo_manager(0.2)?;
        Self::from_parts(dm, support_gid, coding_gid, report)
    }

    pub fn new_with_config(config: LanguageConfig) -> Result<Self, String> {
        let (dm, support_gid, coding_gid, report) = build_language_demo_manager_with_config(0.2, config)?;
        Self::from_parts(dm, support_gid, coding_gid, report)
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn new_with_groups(groups: &[&str]) -> Result<Self, String> {
        let gle_checkpoint = std::env::var("GROWFORMER_GLE_CHECKPOINT").ok();
        let gle_checkpoints = parse_csv_env("GROWFORMER_GLE_CHECKPOINTS");
        let gle_checkpoint_weights = parse_csv_env_f32("GROWFORMER_GLE_WEIGHTS");
        let config = LanguageConfig {
            encoder: EncoderPreset::BertClass,
            bridge_output_dim: DEFAULT_BRIDGE_DIM,
            ema_alpha: 0.2,
            ood_similarity_threshold: 0.15,
            gle_http_endpoint: std::env::var("GROWFORMER_GLE_HTTP_ENDPOINT").ok(),
            gle_checkpoint,
            gle_checkpoints,
            gle_checkpoint_weights,
        };
        let (dm, support_gid, coding_gid, report) =
            build_language_demo_manager_with_groups(groups, config)?;
        Self::from_parts(dm, support_gid, coding_gid, report)
    }

    fn from_parts(dm: DimensionManager, support_gid: GroupId, coding_gid: GroupId, report: CalibrationReport) -> Result<Self, String> {
        Ok(Self {
            brains: HashMap::new(),
            active_brain: "default".to_string(),
            dm,
            support_gid,
            coding_gid,
            calibration: report,
            mode: AgentMode::MicroBrain,
            slo_config: SloConfig::default(),
            latency_log: Vec::new(),
            handoff_log: Vec::new(),
            context_snippets: Vec::new(),
            last_turn: None,
            agent_name: "Growformer".to_string(),
            agent_creator: "swtch.ai".to_string(),
            personality: OceanProfile::assistant(),
            conversation: ConversationContext::default(),
            project_model: ProjectModel::new(),
            tool_registry: ToolRegistry::with_builtins(),
            continuum_feedback_count: 0,
            last_feedback_time: std::time::Instant::now(),
            feedback_window_count: 0,
            continuum_config: ContinuumConfig::default(),
            paramecium: None,
            reasoning: None,
            meta_codebook: None,
            metacognition: None,
            system2_config: System2Config::default(),
        })
    }

    /// Returns the DimensionManager used for inference (active named brain or fallback dm).
    pub fn active_dm_mut(&mut self) -> &mut DimensionManager {
        self.brains
            .get_mut(&self.active_brain)
            .unwrap_or(&mut self.dm)
    }

    /// Reference to the currently active checkpoint (for inspection / API).
    pub fn active_dm(&self) -> &DimensionManager {
        self.brains
            .get(&self.active_brain)
            .unwrap_or(&self.dm)
    }

    // -----------------------------------------------------------------------
    // Core inference (unchanged API, now records latency)
    // -----------------------------------------------------------------------

    pub fn action(&mut self, text: &str) -> Result<ActionJson, String> {
        let start = portable_instant();
        let result = self.active_dm_mut().route_text_to_action(text);
        self.record_latency(start);
        result
    }

    /// Agent identity configuration. Set at startup or per-brain.
    /// Used for "who are you" / "what are you" type queries.
    pub fn set_identity(&mut self, name: &str, creator: &str) {
        self.agent_name = name.to_string();
        self.agent_creator = creator.to_string();
    }

    fn is_identity_query(text: &str) -> bool {
        let lower = text.to_lowercase();
        let patterns = [
            "who are you", "what are you", "who r u", "what is your name",
            "introduce yourself", "tell me about yourself", "what's your name",
            "your name", "who made you", "who created you", "who built you",
        ];
        patterns.iter().any(|p| lower.contains(p))
    }

    fn is_greeting(text: &str) -> bool {
        let lower = text.to_lowercase();
        let trimmed = lower.trim().trim_end_matches(|c: char| c.is_ascii_punctuation());
        let exact = [
            "hi", "hello", "hey", "howdy", "greetings", "yo", "sup",
            "good morning", "good afternoon", "good evening",
            "how are you", "how are you doing", "how's it going",
            "what's up", "whats up", "how do you do",
            "hey there", "hi there", "hello there",
        ];
        exact.iter().any(|p| trimmed == *p || trimmed.starts_with(&format!("{} ", p)))
    }

    fn greeting_response(&self) -> GeneratedResponse {
        GeneratedResponse {
            text: format!(
                "I'm doing well, thank you for asking! I'm {}, ready to help. \
                 I can assist with programming, mathematics, system design, \
                 information theory, and general knowledge. What would you like to explore?",
                self.agent_name
            ),
            template_id: "greeting".to_string(),
            traceable: false,
            confidence: 1.0,
        }
    }

    fn identity_response(&self) -> GeneratedResponse {
        let text = format!(
            "I am {}, a Growformer Agent by {}. I'm a self-organizing neural substrate \
             that learns structure, not weights — my knowledge is encoded as physical \
             neural structure grown, pruned, consolidated, and frozen during training. \
             I generate responses in a single forward pass through my neural environment.",
            self.agent_name, self.agent_creator
        );
        GeneratedResponse {
            text,
            template_id: "identity".to_string(),
            traceable: false,
            confidence: 1.0,
        }
    }

    pub fn generation(&mut self, text: &str) -> Result<(ActionJson, GeneratedResponse), String> {
        self.generation_with_intent_override(text, text)
    }

    /// Like `generation()` but parses intent from `intent_text` instead of the
    /// full `text`. Used by `converse()` to prevent previous-turn context from
    /// dominating topic routing for the current query.
    pub fn generation_with_intent_override(
        &mut self,
        text: &str,
        intent_text: &str,
    ) -> Result<(ActionJson, GeneratedResponse), String> {
        let start = portable_instant();

        if Self::is_identity_query(intent_text) {
            let action = self.active_dm_mut().route_text_to_action_stateless(text)?;
            self.record_latency(start);
            return Ok((action, self.identity_response()));
        }

        if Self::is_greeting(intent_text) {
            let action = self.active_dm_mut().route_text_to_action_stateless(text)?;
            self.record_latency(start);
            return Ok((action, self.greeting_response()));
        }

        // Tool call interception: if the registry matches a tool, return a
        // ToolCall action with the call info. The caller executes the tool
        // and optionally calls generation_with_tool_result for a composed response.
        if let Some(tool_call) = self.tool_registry.match_tool(intent_text) {
            let action = ActionJson {
                action_type: ActionType::ToolCall,
                target_group_id: None,
                group_task_name: None,
                confidence: 1.0,
                margin: 1.0,
                reason: "tool_match".to_string(),
                payload: Some(ActionPayload::ToolCall {
                    tool_name: tool_call.tool_name.clone(),
                    arguments: tool_call.arguments.clone(),
                }),
            };
            let resp = GeneratedResponse {
                text: format!("[tool_call: {}] Awaiting execution.", tool_call.tool_name),
                template_id: "tool_call_pending".to_string(),
                traceable: true,
                confidence: 1.0,
            };
            self.record_latency(start);
            return Ok((action, resp));
        }

        let personality = self.personality.clone();

        // Dual encoding for multi-turn: route on the raw user message (intent_text)
        // to prevent previous-turn context from dominating topic routing, but use
        // the full context-augmented prompt (text) for generation conditioning so
        // the response can reference conversation history.
        let has_context = intent_text != text;

        // Pre-compute meta-routing using intent_text (raw message, no context pollution).
        let meta_pre = self.meta_codebook.as_ref().and_then(|mcb| {
            let dm_ref = &self.dm;
            let bridged = dm_ref.language_runtime.bridge_text_stateless(intent_text).ok()?;
            let mr = mcb.route_and_project(&bridged.routed_vector, intent_text);
            if mr.confidence > 0.3 { Some(mr) } else { None }
        });

        // Snapshot conversation context before mutable borrow of DimensionManager
        let conv_ctx_snapshot = self.conversation.context_embedding.clone();

        let dm = self.active_dm_mut();
        let action = dm.route_text_to_action_stateless(intent_text)?;

        // Routing encoding: from raw user message (clean signal for group selection).
        let encoded = dm.language_runtime.encode_and_bridge(intent_text).ok();
        // Context encoding: from full context prompt (richer signal for generation).
        // Only computed when there's actually conversation context to leverage.
        let context_encoded = if has_context {
            dm.language_runtime.encode_and_bridge(text).ok()
        } else {
            None
        };
        let mut group_idx = action.target_group_id
            .and_then(|gid| dm.main.group_order.iter().position(|&g| g == gid));

        // GrowformerLang meta-routing: override traditional routing with concept-level routing.
        // Gate: accept when margin is clear (≥0.05), OR confidence is high (≥0.90)
        // even with low margin (the concept matched strongly, just close to another),
        // OR the classifier had no group.
        let mut meta_routing: Option<crate::growformer_lang::MetaRoutingResult> = None;
        if let Some(mr) = meta_pre {
            if let Some(best_g) = mr.best_group() {
                if best_g < dm.main.group_order.len() {
                    println!("  [meta-route] concept={}, lang={}, conf={:.3}, margin={:.3} → group {}",
                        mr.concept.name(), mr.language.name(), mr.confidence, mr.margin, best_g);
                    if mr.margin >= 0.05 || mr.confidence >= 0.90 || group_idx.is_none() {
                        group_idx = Some(best_g);
                    } else {
                        println!("  [meta-route] SKIP: low margin ({:.3}) + low conf ({:.3}), keeping classifier group {:?}",
                            mr.margin, mr.confidence, group_idx);
                    }
                }
            }
            meta_routing = Some(mr);
        }

        // Structural routing fallback: when surface routing rejects as OOD,
        // use grade-2 bivector similarity to find a group that shares the
        // input's relational structure — understanding-based routing.
        if group_idx.is_none() {
            if let Some((ref h_raw, _)) = encoded {
                if let Some((best_gidx, best_sim, _)) = dm.route_by_structure(h_raw) {
                    if best_sim > 0.5 {
                        group_idx = Some(best_gidx);
                    }
                }
            }
        }

        let mut topic_hint: Option<String> = None;
        let query_intent = crate::growformer_lang::parse_query_intent(intent_text);
        let is_broad = query_intent.is_broad_overview();
        let mut broad_summary: Option<(String, f32)> = None;
        println!("  [intent] action={}, subject=\"{}\", broad={}", query_intent.action.name(), query_intent.subject, is_broad);

        // Hoisted for metacognition retry loop (survives the encoding block scope)
        let mut retry_conditioning: Option<Vec<f32>> = None;
        let mut retry_effective_gidx: Option<usize> = None;

        let resp = if let Some((ref h_raw, ref bridged)) = encoded {
            // Build the conditioning vector. For routing we use the intent_text
            // encoding (clean, no context bleed). For generation conditioning,
            // blend in the context-encoded embedding so the response can
            // reference conversation history.
            let base_vector = if let Some(ref mr) = meta_routing {
                if !mr.projected_embedding.is_empty() {
                    mr.projected_embedding.clone()
                } else {
                    bridged.routed_vector.clone()
                }
            } else {
                bridged.routed_vector.clone()
            };

            // Multi-turn: blend context embedding into the base routing vector.
            // 70% intent (what the user is asking NOW) + 30% context (conversation history).
            let blended = if let Some((_, ref ctx_bridged)) = context_encoded {
                let ctx_vec = &ctx_bridged.routed_vector;
                let dim = base_vector.len().min(ctx_vec.len());
                let mut v = vec![0.0f32; dim];
                for i in 0..dim {
                    v[i] = base_vector[i] * 0.7 + ctx_vec[i] * 0.3;
                }
                v
            } else {
                base_vector
            };

            let mut conditioned = blended;
            personality.condition_vector(&mut conditioned);

            // --- MetaBrain path: unified routing + conditioning + archetype selection ---
            let meta_result = if let Some(ref mut mb) = dm.meta_brain {
                if mb.is_ready() {
                    let r = mb.process(h_raw, &conditioned);
                    println!("  [meta-brain] topic={}, verb={}, action={:?}, conf={:.3}",
                        r.topic, r.verb, r.action, r.confidence);
                    if group_idx.is_none() && r.group_idx.is_some() { group_idx = r.group_idx; }
                    topic_hint = Some(r.topic.clone());
                    Some(r)
                } else { None }
            } else {
                if let Some(ref mut ul) = dm.understanding {
                    if !ul.is_empty() {
                        let (_, _, topic, verb) = ul.classify(h_raw);
                        println!("  [understanding] topic={}, verb={}", topic, verb);
                        topic_hint = Some(topic);
                    }
                }
                None
            };

            // Override topic_hint with operation-specific intent from GrowformerLang.
            // This gives the topic sub-lattice a precise key (e.g., "addition_operation")
            // instead of a generic label from the understanding layer.
            if let Some(op_topic) = crate::growformer_lang::infer_operation_topic(intent_text) {
                topic_hint = Some(op_topic.clone());

                // Cross-group topic search: find the group with the MOST programs for
                // this topic, regardless of whether the current group also has it.
                // This prevents false matches where a group has the topic label but
                // few/irrelevant programs.
                if let Some(current_g) = group_idx {
                    let current_count = dm.group_gen_envs.get(&current_g)
                        .map(|env| env.topic_subindex.iter()
                            .filter(|t| t.topic_name.eq_ignore_ascii_case(&op_topic))
                            .map(|t| t.lattice.programs.len())
                            .sum::<usize>())
                        .unwrap_or(0);

                    let mut best_redirect: Option<(usize, usize)> = None;
                    for (&gidx, env) in &dm.group_gen_envs {
                        if gidx == current_g { continue; }
                        for t in &env.topic_subindex {
                            if t.topic_name.eq_ignore_ascii_case(&op_topic) && !t.lattice.programs.is_empty() {
                                let count = t.lattice.programs.len();
                                if best_redirect.map(|(_, c)| count > c).unwrap_or(true) {
                                    best_redirect = Some((gidx, count));
                                }
                            }
                        }
                    }
                    if let Some((redirect_g, redirect_count)) = best_redirect {
                        // Only redirect when current group has no programs for this topic,
                        // or the other group has overwhelmingly more (3x+). Avoids sending
                        // "stack" queries away from a group that has relevant stack programs
                        // just because another group has more coding_implementation programs.
                        let should_redirect = current_count == 0
                            || (redirect_count >= current_count * 3 && current_count < 3);
                        if should_redirect {
                            println!("  [cross-group] topic '{}': group {} has {} progs vs current group {} with {}, redirecting",
                                op_topic, redirect_g, redirect_count, current_g, current_count);
                            group_idx = Some(redirect_g);
                        }
                    }
                    if current_count == 0 && best_redirect.is_none() {
                        println!("  [topic-miss] '{}' not found in any group", op_topic);
                    }
                }
            } else {
                println!("  [topic-miss] no topic inferred for: {}", &intent_text[..intent_text.len().min(60)]);
            }

            // Use MetaBrain conditioning when available, else fall back to Clifford path
            let mut gen_conditioning = if let Some(ref mr) = meta_result {
                mr.conditioning.clone()
            } else if let Some(gidx) = group_idx {
                dm.adapt_for_group_clifford(gidx, &conditioned, h_raw, GEN_COND_DIM)
            } else {
                let mut c = conditioned.clone();
                c.resize(GEN_COND_DIM, 0.0);
                c
            };

            // Geometric conversation context: blend accumulated turn embeddings
            // into the generation conditioning so retrieval is biased toward
            // topic continuity. Only applies when conversation has history.
            if !conv_ctx_snapshot.is_empty() {
                let blend = 0.15f32;
                let dim = gen_conditioning.len().min(conv_ctx_snapshot.len());
                for i in 0..dim {
                    gen_conditioning[i] = (1.0 - blend) * gen_conditioning[i]
                        + blend * conv_ctx_snapshot[i];
                }
            }

            // --- Level 3: Check episodic memory for cached composition ---
            let _cached_groups = Self::retrieve_cached_composition(dm, &conditioned);

            // Apply OCEAN Hopf diversity bonus and subject keywords to all gen envs
            let div_bonus = personality.hopf_diversity_bonus();
            let subject_kw: Vec<String> = query_intent.subject
                .split_whitespace()
                .filter(|w| w.len() > 2)
                .map(|w| w.to_ascii_lowercase())
                .collect();
            let intent_act = query_intent.action.name().to_string();
            for env in dm.group_gen_envs.values_mut() {
                env.diversity_bonus = div_bonus;
                env.subject_keywords = subject_kw.clone();
                env.intent_action = intent_act.clone();
            }

            // --- Broad query detection: summarize across topic sub-lattices ---
            // For categorical/definitional questions ("What is software architecture?"),
            // compose from multiple sub-topics instead of returning a single program.
            // Cross-group search: find the group whose topic sub-lattices best match
            // the query subject, rather than blindly trusting the classifier.
            broad_summary = if is_broad {
                let subject_lower = query_intent.subject.to_ascii_lowercase();
                let subject_words: Vec<&str> = subject_lower.split_whitespace().collect();

                // Score each group by how many of its topic names overlap with the query subject.
                let mut best_broad_group: Option<(usize, usize)> = None; // (gidx, relevance_score)
                for (&gidx, env) in &dm.group_gen_envs {
                    if env.topic_subindex.len() < 2 { continue; }
                    let mut relevance = 0usize;
                    for t in &env.topic_subindex {
                        let tname = t.topic_name.to_ascii_lowercase().replace('_', " ");
                        for w in &subject_words {
                            if w.len() > 2 && tname.contains(w) {
                                relevance += 1;
                            }
                        }
                    }
                    if relevance > best_broad_group.map(|(_, r)| r).unwrap_or(0) {
                        best_broad_group = Some((gidx, relevance));
                    }
                }

                // Use the best-matching group if it scores > 0, otherwise fallback to classified group
                let effective_gidx = best_broad_group
                    .filter(|&(_, rel)| rel > 0)
                    .map(|(gidx, _)| gidx)
                    .or_else(|| group_idx)
                    .or_else(|| meta_result.as_ref().and_then(|mr| mr.group_idx));

                if let Some(bbg) = best_broad_group {
                    println!("  [broad-query] best group by topic match: group {} (relevance={})", bbg.0, bbg.1);
                }

                effective_gidx.and_then(|gidx| {
                    let adapted = dm.adapt_for_group_clifford(gidx, &conditioned, h_raw, GEN_COND_DIM);
                    dm.group_gen_envs.get(&gidx).and_then(|env| {
                        if env.topic_subindex.len() < 2 {
                            return None;
                        }
                        let (summary, conf, topics, ens_coh) = env.summarize_across_topics(&adapted, 6, 500);
                        if summary.len() > 30 && !topics.is_empty() {
                            println!(
                                "  [broad-query] summarized {} topics from group {}: {:?}, conf={:.3}, coherence={:.3}",
                                topics.len(), gidx, topics, conf, ens_coh
                            );
                            if ens_coh < 0.15 {
                                println!("  [broad-query] REJECT: ensemble coherence too low ({:.3}), falling back", ens_coh);
                                None
                            } else {
                                // Override group_idx so downstream generation uses this group
                                // (avoids classifier's group producing irrelevant content)
                                Some((summary, conf))
                            }
                        } else {
                            None
                        }
                    })
                })
            } else {
                None
            };

            // --- Level 1: Competitive multi-head inference with E8 composition ---
            use crate::dimension::group_gen::{
                E8Contribution, e8_blend_quantum, e8_compose_sentences_quantum,
                e8_select_best, compute_q,
            };

            // Primary generation: prefer classifier's group_idx over MetaBrain's.
            // Only use MetaBrain's archetype when the classifier didn't provide a group.
            let effective_gidx = group_idx.or_else(|| meta_result.as_ref().and_then(|mr| mr.group_idx));
            let primary = effective_gidx.and_then(|gidx| {
                // If we already have a broad summary for this group, use it directly
                if let Some((ref summary_text, summary_conf)) = broad_summary {
                    let mut e8 = [0.0f32; 8];
                    for i in 0..8.min(gen_conditioning.len()) { e8[i] = gen_conditioning[i]; }
                    return Some(E8Contribution {
                        group_idx: gidx,
                        lattice_point: e8,
                        text: summary_text.clone(),
                        confidence: summary_conf,
                    });
                }
                let cond = dm.adapt_for_group_clifford(gidx, &conditioned, h_raw, GEN_COND_DIM);
                dm.group_gen_envs.get_mut(&gidx).map(|env| {
                    let (text, conf, e8) = env.generate_with_e8_for_topic(&cond, topic_hint.as_deref(), 300, 0.8);
                    E8Contribution { group_idx: gidx, lattice_point: e8, text, confidence: conf }
                })
            });

            let (best_text, best_conf, best_gidx) = match primary {
                Some(ref c) if c.confidence >= 0.70 && c.text.len() > 5 => {
                    (c.text.clone(), c.confidence, c.group_idx)
                }
                primary_result => {
                    let mut contributions: Vec<E8Contribution> = Vec::new();
                    if let Some(c) = primary_result {
                        contributions.push(c);
                    }

                    // MetaBrain volley: use trichocyst candidates from ArchetypeBrain
                    if let Some(ref mr) = meta_result {
                        for &(v_gidx, v_aidx, v_weight) in &mr.volley {
                            if Some(v_gidx) == group_idx { continue; }
                            if let Some(env) = dm.group_gen_envs.get_mut(&v_gidx) {
                                let (text, conf) = env.generate_with_archetype_for_topic(
                                    &gen_conditioning, topic_hint.as_deref(), v_aidx, v_weight, 300, 0.8,
                                );
                                if text.len() > 5 {
                                    let mut e8 = [0.0f32; 8];
                                    for i in 0..8.min(gen_conditioning.len()) { e8[i] = gen_conditioning[i]; }
                                    contributions.push(E8Contribution {
                                        group_idx: v_gidx, lattice_point: e8, text, confidence: conf,
                                    });
                                }
                            }
                        }
                    } else {
                        // Fallback: fan out to all other groups
                        let other_keys: Vec<usize> = dm.group_gen_envs.keys()
                            .filter(|&&k| Some(k) != group_idx)
                            .copied().collect();
                        for gidx in other_keys {
                            let adapted = dm.adapt_for_group_clifford(gidx, &conditioned, h_raw, GEN_COND_DIM);
                            if let Some(env) = dm.group_gen_envs.get_mut(&gidx) {
                                let (text, conf, e8) = env.generate_with_e8_for_topic(&adapted, topic_hint.as_deref(), 300, 0.8);
                                if text.len() > 5 {
                                    contributions.push(E8Contribution {
                                        group_idx: gidx, lattice_point: e8, text, confidence: conf,
                                    });
                                }
                            }
                        }
                    }

                    if contributions.is_empty() {
                        (String::new(), -1.0, 0)
                    } else if contributions.len() == 1 {
                        let c = &contributions[0];
                        (c.text.clone(), c.confidence, c.group_idx)
                    } else {
                        // Quantum group composition: compute deformation
                        // parameter q from input embedding asymmetry, then
                        // use R-matrix braided blend for non-commutative composition
                        let q = compute_q(&conditioned, &contributions);
                        let blended = e8_blend_quantum(&contributions, q);

                        // Level 1: E8-scored selection of best single response
                        let (mut best_t, mut best_c, mut best_g) =
                            e8_select_best(&blended, &contributions)
                                .unwrap_or_else(|| (String::new(), -1.0, 0));

                        // Level 2: quantum sentence composition — the R-matrix
                        // determines which group leads the response structure
                        if best_c < 0.9 {
                            let (composed, comp_conf) =
                                e8_compose_sentences_quantum(&blended, &contributions, 4, q);
                            if !composed.is_empty() && comp_conf > best_c {
                                best_t = composed;
                                best_c = comp_conf;
                                best_g = contributions[0].group_idx;

                                // Level 3: cache this quantum composition
                                let involved: Vec<usize> = contributions.iter()
                                    .map(|c| c.group_idx).collect();
                                Self::cache_composition(dm, &conditioned, &involved, comp_conf);
                            }
                        }

                        (best_t, best_c, best_g)
                    }
                }
            };

            // Capture conditioning + group for metacognition retry loop
            retry_conditioning = Some(gen_conditioning.clone());
            let eff_gidx = group_idx.or_else(|| meta_result.as_ref().and_then(|mr| mr.group_idx));
            retry_effective_gidx = eff_gidx;

            // Knowledge boundary: if retrieval confidence is below the floor,
            // the query is outside the lattice's coverage. Return an honest
            // decline instead of a low-confidence wrong answer.
            const RETRIEVAL_CONFIDENCE_FLOOR: f32 = 0.25;
            if best_conf < RETRIEVAL_CONFIDENCE_FLOOR || best_text.len() <= 5 {
                let topic_label = topic_hint.as_deref()
                    .map(|t| t.replace('_', " "))
                    .unwrap_or_default();
                let decline_msg = if topic_label.is_empty() {
                    "I don't have enough information to give you a confident answer on this topic. \
                     Could you rephrase or ask about something more specific?".to_string()
                } else {
                    format!(
                        "I don't have enough information about '{}' to give you a confident answer. \
                         This may be outside my current knowledge. Could you rephrase or ask about a related topic?",
                        topic_label
                    )
                };
                println!("  [knowledge-boundary] conf={:.3} < floor={:.3}, declining",
                    best_conf, RETRIEVAL_CONFIDENCE_FLOOR);
                GeneratedResponse {
                    text: decline_msg,
                    template_id: "knowledge_boundary".to_string(),
                    traceable: true,
                    confidence: 0.0,
                }
            } else if best_text.len() > 5 {
                GeneratedResponse {
                    text: best_text,
                    template_id: format!("growformer_gen_{}", best_gidx),
                    traceable: false,
                    confidence: best_conf,
                }
            } else if let Some(ref head) = dm.generation_head {
                Self::legacy_gen_from_encoded(head, &encoded, &action, &dm.main.group_order)
            } else {
                render_action_template(&action)
            }
        } else if let Some(ref head) = dm.generation_head {
            Self::legacy_gen_from_encoded(head, &encoded, &action, &dm.main.group_order)
        } else {
            render_action_template(&action)
        };

        // Sentence-level coherence guard: strip trailing sentences that diverge
        // from the prompt's topic (e.g., identity text appended to an IT query).
        let resp = if let Some((_, ref bridged)) = encoded {
            let truncated = Self::coherence_truncate(&bridged.routed_vector, &resp.text);
            if truncated.len() != resp.text.len() {
                GeneratedResponse {
                    text: truncated,
                    ..resp
                }
            } else {
                resp
            }
        } else {
            resp
        };

        // If the response is already a knowledge boundary decline, skip all
        // downstream reasoning and metacog — there's nothing to improve.
        let is_decline = resp.template_id == "knowledge_boundary"
            || resp.template_id == "metacog_degradation";

        // Reasoning fallback: invoke System 1.5 (wave settling) when primary
        // confidence is genuinely low AND text is short/unhelpful, OR invoke
        // System 2 (deliberate chaining) when confidence is in the uncertain
        // middle and cross-domain ambiguity is detected.
        let resp = if is_decline {
            resp
        } else if let (Some(ref reasoning), Some((ref _h_raw, ref bridged))) = (&self.reasoning, &encoded) {
            let mut cond = bridged.routed_vector.clone();
            cond.resize(GEN_COND_DIM, 0.0);
            let dm_ref = self.active_dm();

            // System 2: deliberate multi-step reasoning for uncertain cross-domain queries.
            // When broad_summary already produced a within-group composition, trust it —
            // System 2 cross-group retrieval would mix in unrelated domains.
            if broad_summary.is_none() && reasoning.should_reason_deliberate_ext(&cond, resp.confidence, &dm_ref.group_gen_envs, topic_hint.as_deref(), is_broad) {
                let s2_result = reasoning.reason_deliberate(
                    &cond,
                    &dm_ref.group_gen_envs,
                    &dm_ref.group_rotors,
                    &self.system2_config,
                );
                if s2_result.confidence > resp.confidence && s2_result.text.len() > 10 {
                    println!(
                        "  [system2] accepted: steps={}, wm={}, coherence={:.3}, groups={:?}",
                        s2_result.steps_taken, s2_result.working_memory_size,
                        s2_result.final_coherence, s2_result.source_groups
                    );
                    GeneratedResponse {
                        text: s2_result.text,
                        template_id: format!("system2_{}_steps", s2_result.steps_taken),
                        traceable: false,
                        confidence: s2_result.confidence,
                    }
                } else {
                    resp
                }
            }
            // System 1.5: wave settling for low-confidence short responses
            else if resp.confidence < 0.50 && resp.text.len() < 20 {
                if reasoning.should_reason(&cond, resp.confidence, &dm_ref.group_gen_envs) {
                    let result = reasoning.reason(&cond, &dm_ref.group_gen_envs, &dm_ref.group_rotors);
                    if result.confidence > resp.confidence && result.text.len() > 10 {
                        GeneratedResponse {
                            text: result.text,
                            template_id: format!("reasoning_{}_groups", result.source_groups.len()),
                            traceable: false,
                            confidence: result.confidence,
                        }
                    } else { resp }
                } else { resp }
            } else {
                resp
            }
        } else { resp };

        // MetaCognition: reflective quality gate with retry-reconditioning loop.
        // Skip for: broad query summaries, knowledge boundary declines, degradation.
        let resp = if is_decline {
            println!("  [metacog] SKIP: knowledge boundary decline");
            resp
        } else if is_broad && broad_summary.is_some() {
            println!("  [metacog] SKIP: broad query summary (multi-topic composition)");
            resp
        } else {
            let mc_taken = self.metacognition.take();
            let prompt_emb_opt = encoded.as_ref().map(|(_, b)| b.routed_vector.clone());
            let h_raw_opt = encoded.as_ref().map(|(h, _)| h.clone());

            let mc_active = mc_taken.as_ref().map_or(false, |mc| mc.is_ready());
            let has_encoding = prompt_emb_opt.is_some() && h_raw_opt.is_some();

            let final_resp = if mc_active && has_encoding {
                let mc = mc_taken.as_ref().unwrap();
                let prompt_emb = prompt_emb_opt.as_ref().unwrap();
                let h_raw_clone = h_raw_opt.as_ref().unwrap();
                let max_retries = mc.config.max_retries;
                let mut current_resp = resp;

                for attempt in 0..=max_retries {
                    let outcome = mc.reflect(
                        prompt_emb,
                        &Self::approximate_response_embedding(prompt_emb, &current_resp.text),
                        &current_resp.text,
                        topic_hint.as_deref(),
                        attempt,
                    );

                    match outcome {
                        ReflectionOutcome::Accept { scores } => {
                            println!("  [metacog] ACCEPT: quality={:.3}", scores.quality);
                            // Continuum: positive feedback to the selected program
                            if let Some(gidx) = retry_effective_gidx {
                                let dm = self.active_dm_mut();
                                if let Some(env) = dm.group_gen_envs.get_mut(&gidx) {
                                    if let Some(pidx) = env.last_selected_archetype {
                                        env.apply_quality_feedback(pidx, true, scores.quality);
                                    }
                                }
                            }
                            break;
                        }
                        ReflectionOutcome::Retry { scores, adjustment: _, attempt: att } => {
                            println!(
                                "  [metacog] RETRY: quality={:.3}, attempt {}, predictive coding refinement",
                                scores.quality, att + 1
                            );
                            let dm = self.active_dm_mut();
                            if let (Some(gidx), Some(ref base_cond)) = (retry_effective_gidx, &retry_conditioning) {
                                // Predictive Coding: STA grade-decomposed error correction
                                // replaces crude "push toward topic centroid" adjustment.
                                let response_emb = Self::approximate_response_embedding(prompt_emb, &current_resp.text);
                                let pc = crate::predictive_coder::PredictiveCoder::new(
                                    crate::predictive_coder::PredictiveCodingConfig::default()
                                );
                                let refinement = pc.refine(prompt_emb, base_cond, &response_emb);
                                let adjusted_cond = if refinement.improved {
                                    if let Some(ref ge) = refinement.last_grade_error {
                                        println!("    [pc] dominant grade={}, error={:.4}, coherence={:.3}",
                                            ge.dominant_grade, ge.total_error, refinement.coherence.combined);
                                    }
                                    refinement.conditioning
                                } else {
                                    base_cond.clone()
                                };

                                let recond = dm.adapt_for_group_clifford(gidx, &adjusted_cond, h_raw_clone, GEN_COND_DIM);
                                if let Some(env) = dm.group_gen_envs.get_mut(&gidx) {
                                    let (retry_text, retry_conf, _) = env.generate_with_e8_for_topic(
                                        &recond, topic_hint.as_deref(), 300, 0.8,
                                    );
                                    if retry_text.len() > 5 && retry_conf > current_resp.confidence {
                                        current_resp = GeneratedResponse {
                                            text: retry_text,
                                            template_id: format!("metacog_retry_{}", att + 1),
                                            traceable: current_resp.traceable,
                                            confidence: retry_conf,
                                        };
                                    }
                                }
                            }
                        }
                        ReflectionOutcome::Degrade { scores, message, attempts_exhausted } => {
                            println!(
                                "  [metacog] DEGRADE: quality={:.3} after {} attempts → honest decline",
                                scores.quality, attempts_exhausted
                            );
                            if let Some(gidx) = retry_effective_gidx {
                                let dm = self.active_dm_mut();
                                if let Some(env) = dm.group_gen_envs.get_mut(&gidx) {
                                    if let Some(pidx) = env.last_selected_archetype {
                                        env.apply_quality_feedback(pidx, false, scores.quality);
                                    }
                                }
                            }
                            current_resp = GeneratedResponse {
                                text: message,
                                template_id: "metacog_degradation".to_string(),
                                traceable: true,
                                confidence: 0.0,
                            };
                            break;
                        }
                    }
                }
                current_resp
            } else {
                resp
            };

            self.metacognition = mc_taken;
            final_resp
        };

        self.record_latency(start);
        Ok((action, resp))
    }

    /// Conversational generation: uses conversation context + personality.
    /// Tracks multi-turn history and applies EMA modulation based on OCEAN.
    /// Geometric context: blends accumulated turn embeddings into the query
    /// so retrieval is biased toward topic continuity across turns.
    pub fn converse(&mut self, user_text: &str) -> Result<(ActionJson, GeneratedResponse), String> {
        // Build context BEFORE pushing user turn to avoid duplicating it in the window.
        let context_prefix = if self.conversation.turn_count() > 0 {
            Some(self.conversation.context_window(self.conversation.context_window_size))
        } else {
            None
        };

        self.conversation.push_user(user_text);

        // Continuum: decay activation levels between conversation turns
        {
            let dm = self.active_dm_mut();
            for env in dm.group_gen_envs.values_mut() {
                env.decay_between_turns();
            }
            for env in dm.group_code_envs.values_mut() {
                env.decay_between_turns();
            }
        }

        // Modulate EMA alpha based on personality before encoding
        let base_alpha = self.active_dm().language_runtime.config.ema_alpha;
        let modulated_alpha = self.personality.modulated_ema_alpha(base_alpha);
        self.active_dm_mut().language_runtime.smoother.alpha = modulated_alpha;

        // Anaphora resolution: when the user says "that", "it", "this" without
        // a clear noun subject, substitute with the previous turn's topic so the
        // intent parser can route correctly.
        let resolved_intent = self.resolve_anaphora(user_text);
        let intent_for_routing = resolved_intent.as_deref().unwrap_or(user_text);

        // Embed the raw query for geometric context operations
        let query_bridge = self.active_dm()
            .language_runtime.bridge_text_stateless(user_text).ok();
        let query_emb = query_bridge.as_ref().map(|b| b.routed_vector.clone());

        // Topic shift detection: if the new query is geometrically distant
        // from accumulated context, reset topic thread to avoid stale bias.
        if let Some(ref qe) = query_emb {
            if self.conversation.is_topic_shift(qe, 0.15) {
                println!("  [conv] topic shift detected, resetting topic thread");
                self.conversation.current_topic = None;
                self.conversation.current_group = None;
            }
        }

        // Context-augmented prompt: recent history provides semantic grounding
        // for the encoder, but intent parsing uses the raw user message to avoid
        // previous-turn topics dominating the current query's routing.
        let context_prompt = match context_prefix {
            Some(ctx) => format!("{} | user: {}", ctx, user_text),
            None => user_text.to_string(),
        };

        let (action, mut resp) = self.generation_with_intent_override(&context_prompt, intent_for_routing)?;

        // Update geometric context with this turn's embeddings
        if let Some(ref qe) = query_emb {
            let resp_emb = self.active_dm()
                .language_runtime.bridge_text_stateless(&resp.text)
                .map(|b| b.routed_vector)
                .unwrap_or_else(|_| qe.clone());
            self.conversation.update_geometric_context(qe, &resp_emb);
        }

        // Track topic and group continuity
        if let Some(ref gidx) = action.target_group_id.and_then(|gid| {
            self.active_dm().main.group_order.iter().position(|&g| g == gid)
        }) {
            self.conversation.current_group = Some(*gidx);
        }

        // Conversational framing: personality-aware prefix for natural dialogue
        let is_framed = resp.template_id != "identity"
            && resp.template_id != "greeting"
            && resp.template_id != "metacog_degradation"
            && resp.template_id != "knowledge_boundary"
            && !resp.template_id.starts_with("tool_")
            && resp.confidence > 0.3;
        if is_framed {
            if let Some(prefix) = self.personality.conversational_prefix(
                self.conversation.turn_count(),
                user_text,
            ) {
                if !prefix.is_empty() {
                    resp.text = format!("{}{}", prefix, resp.text);
                }
            }
        }

        self.conversation.push_agent(&resp.text);

        // Store turn context for Continuum feedback (including lattice routing info)
        let effective_gidx = action.target_group_id.and_then(|gid| {
            self.active_dm().main.group_order.iter().position(|&g| g == gid)
        });
        let program_idx = effective_gidx.and_then(|gidx| {
            self.active_dm().group_gen_envs.get(&gidx).and_then(|e| e.last_selected_archetype)
        });
        self.last_turn = Some(TurnContext {
            message: user_text.to_string(),
            group_id: action.target_group_id,
            output: resp.text.clone(),
            effective_gidx,
            program_idx,
        });

        Ok((action, resp))
    }

    /// Resolve anaphoric pronouns ("that", "it", "this") by substituting
    /// the previous turn's topic/subject. Returns `Some(expanded)` when a
    /// pronoun was resolved, `None` if the message is self-contained.
    fn resolve_anaphora(&self, text: &str) -> Option<String> {
        let prev = self.last_turn.as_ref()?;

        let lower = text.to_ascii_lowercase();
        let words: Vec<&str> = lower.split_whitespace().collect();

        // Only resolve when the message contains a pronoun that stands in for a noun.
        // Look for patterns like "implement that", "do that", "explain this",
        // "use it", "how does it work", etc.
        let pronoun_patterns = [
            " that ", " that?", " that.", " that,",
            " this ", " this?", " this.", " this,",
            " it ", " it?", " it.", " it,",
        ];
        let starts_with_pronoun = lower.starts_with("that ") || lower.starts_with("this ") || lower.starts_with("it ");
        let ends_with_pronoun = lower.ends_with(" that") || lower.ends_with(" this") || lower.ends_with(" it");

        let has_pronoun = starts_with_pronoun
            || ends_with_pronoun
            || pronoun_patterns.iter().any(|p| lower.contains(p));

        if !has_pronoun { return None; }

        // Avoid false positives: if the sentence already has a clear noun subject
        // (more than 5 content words besides the pronoun), it's likely self-contained.
        let content_words: Vec<&str> = words.iter()
            .filter(|w| w.len() > 2 && !["the", "that", "this", "how", "would", "you", "can", "could", "what", "does", "with", "for", "from", "into"].contains(w))
            .copied()
            .collect();
        if content_words.len() >= 4 { return None; }

        // Extract the referent from the previous turn's message via intent parsing
        let prev_intent = crate::growformer_lang::parse_query_intent(&prev.message);
        let referent = if !prev_intent.subject.is_empty() {
            prev_intent.subject.clone()
        } else {
            // Fallback: use first 6 words of the previous message
            prev.message.split_whitespace().take(6).collect::<Vec<_>>().join(" ")
        };

        if referent.is_empty() { return None; }

        // Replace the pronoun with the referent
        let mut resolved = text.to_string();
        for pronoun in &["that", "this", "it"] {
            // Case-insensitive word-boundary replacement (first occurrence only)
            let re_patterns = [
                format!(" {} ", pronoun),
                format!(" {}?", pronoun),
                format!(" {}.", pronoun),
                format!(" {},", pronoun),
            ];
            for pat in &re_patterns {
                if let Some(pos) = resolved.to_ascii_lowercase().find(pat) {
                    let suffix_char = pat.chars().last().unwrap();
                    let replacement = if suffix_char == ' ' {
                        format!(" {} ", referent)
                    } else {
                        format!(" {}{}", referent, suffix_char)
                    };
                    resolved = format!("{}{}{}", &resolved[..pos], replacement, &resolved[pos + pat.len()..]);
                    println!("  [anaphora] '{}' → '{}' (referent: {})", text, resolved, referent);
                    return Some(resolved);
                }
            }
            // Handle trailing pronoun (e.g., "implement that")
            let trailing = format!(" {}", pronoun);
            if resolved.to_ascii_lowercase().ends_with(&trailing) {
                let cut = resolved.len() - trailing.len();
                resolved = format!("{} {}", &resolved[..cut], referent);
                println!("  [anaphora] '{}' → '{}' (referent: {})", text, resolved, referent);
                return Some(resolved);
            }
            // Handle leading pronoun
            let leading = format!("{} ", pronoun);
            if resolved.to_ascii_lowercase().starts_with(&leading) {
                resolved = format!("{} {}", referent, &resolved[leading.len()..]);
                println!("  [anaphora] '{}' → '{}' (referent: {})", text, resolved, referent);
                return Some(resolved);
            }
        }

        None
    }

    /// Reset conversation context (new session).
    /// Consolidates the previous Continuum session (commits drift for
    /// high-quality, high-hit programs), then begins a fresh session.
    pub fn reset_conversation(&mut self) {
        // Consolidate outgoing session: commit drift to persistent centroids
        // for programs that received enough positive interaction.
        let min_hits = self.continuum_config.min_consolidation_hits;
        let dm = self.active_dm_mut();
        for env in dm.group_gen_envs.values_mut() {
            env.consolidate_session(min_hits);
        }
        for env in dm.group_code_envs.values_mut() {
            env.consolidate_session(min_hits);
        }

        self.conversation.clear();
        self.active_dm_mut().language_runtime.smoother.reset();

        // Begin fresh Continuum session: reset volatile + session state
        let dm = self.active_dm_mut();
        for env in dm.group_gen_envs.values_mut() {
            env.begin_session();
        }
        for env in dm.group_code_envs.values_mut() {
            env.begin_session();
        }
    }

    // -------------------------------------------------------------------
    // Paramecium — lattice-only sub-neuronal inference
    // -------------------------------------------------------------------

    /// Build a paramecium lattice from the active brain's codebook and dictionary.
    /// Extracts archetype centroids and token sequences from all gen envs.
    pub fn build_paramecium(&mut self) {
        let dm = self.active_dm();
        let mut archetypes: Vec<(Vec<u16>, Vec<f32>)> = Vec::new();

        for (_gidx, env) in &dm.group_gen_envs {
            if let Some(ref cb) = env.codebook {
                for (arch_idx, arch) in cb.archetypes.iter().enumerate() {
                    let centroid = cb.archetype_prototypes.get(arch_idx)
                        .cloned()
                        .unwrap_or_else(|| vec![0.0; DEFAULT_BRIDGE_DIM]);
                    let mut tokens: Vec<u16> = arch.fixed.iter().map(|&(_, t)| t).collect();
                    tokens.truncate(arch.median_content_length.max(arch.length).max(1));
                    archetypes.push((tokens, centroid));
                }
            }
        }

        let dict = dm.group_gen_envs.values().next()
            .map(|e| e.dictionary.clone())
            .unwrap_or_else(|| crate::spectral::TokenDictionary::build(&[""], 256));

        let lattice = InfraciliaryLattice::from_codebook(dict, &archetypes);
        self.paramecium = Some(lattice);
    }

    /// Rebuild MetaCognition from the loaded brain's lattice programs.
    /// Called automatically after load_brain to restore the quality gate.
    pub fn rebuild_metacognition(&mut self) {
        let dm = self.active_dm();
        let mut mc = MetaCognition::with_defaults();
        let mut pair_count = 0u64;

        for (_gidx, env) in &dm.group_gen_envs {
            let default_topic = if env.topic_subindex.is_empty() {
                "general".to_string()
            } else {
                env.topic_subindex[0].topic_name.clone()
            };
            for prog in &env.lattice.programs {
                let topic = env.topic_subindex.iter()
                    .find(|sub| {
                        sub.lattice.programs.iter().any(|sp| {
                            Self::fast_cosine(&sp.ema_centroid, &prog.ema_centroid) > 0.90
                        })
                    })
                    .map(|sub| sub.topic_name.as_str())
                    .unwrap_or(&default_topic);
                mc.absorb_pair(&prog.ema_centroid, &prog.ema_centroid, topic);
                pair_count += 1;
            }
        }
        for (_gidx, env) in &dm.group_code_envs {
            let default_topic = if env.topic_subindex.is_empty() {
                "code_general".to_string()
            } else {
                env.topic_subindex[0].topic_name.clone()
            };
            for prog in &env.lattice.programs {
                let topic = env.topic_subindex.iter()
                    .find(|sub| {
                        sub.lattice.programs.iter().any(|sp| {
                            Self::fast_cosine(&sp.ema_centroid, &prog.ema_centroid) > 0.90
                        })
                    })
                    .map(|sub| sub.topic_name.as_str())
                    .unwrap_or(&default_topic);
                mc.absorb_pair(&prog.ema_centroid, &prog.ema_centroid, topic);
                pair_count += 1;
            }
        }

        println!(
            "  MetaCognition rebuilt: {} pairs, {} topics, ready={}",
            pair_count, mc.topic_count(), mc.is_ready()
        );
        self.metacognition = Some(mc);
    }

    /// Rebuild ReasoningEngine (including System 2 support) from loaded brain.
    pub fn rebuild_reasoning(&mut self) {
        use crate::reasoning::{CognitiveMap, ReasoningEngine};
        let dm = self.active_dm();
        let cog_map = CognitiveMap::build(&dm.group_gen_envs, &dm.group_rotors);
        let group_dicts: HashMap<usize, crate::spectral::TokenDictionary> = dm.group_gen_envs
            .iter()
            .map(|(&gidx, env)| (gidx, env.dictionary.clone()))
            .collect();
        println!(
            "  ReasoningEngine rebuilt: {} nodes, {} edges",
            cog_map.node_count(),
            cog_map.edge_count()
        );
        self.reasoning = Some(ReasoningEngine::new(cog_map, group_dicts));
    }

    /// Rebuild GrowformerLang MetaCodebook from loaded brain's lattice programs.
    /// Infers concepts and languages from decoded program text.
    pub fn rebuild_meta_codebook(&mut self) {
        use crate::growformer_lang::{infer_concept, detect_language, MetaCodebook};
        let dm = self.active_dm();
        let mut meta_samples: Vec<(Vec<f32>, crate::growformer_lang::MetaConcept, crate::growformer_lang::TargetLanguage, usize)> = Vec::new();

        for (&gidx, env) in &dm.group_gen_envs {
            for prog in &env.lattice.programs {
                let text = env.dictionary.decode(&prog.token_sequence);
                let concept = infer_concept(&text, None, None);
                let lang = detect_language(&text);
                meta_samples.push((prog.ema_centroid.clone(), concept, lang, gidx));
            }
        }
        for (&gidx, env) in &dm.group_code_envs {
            for prog in &env.lattice.programs {
                let text = env.dictionary.decode(&prog.token_sequence);
                let concept = infer_concept(&text, None, None);
                let lang = detect_language(&text);
                meta_samples.push((prog.ema_centroid.clone(), concept, lang, gidx));
            }
        }

        let mcb = MetaCodebook::build(&meta_samples);
        println!(
            "  MetaCodebook rebuilt: {} samples, {} concepts",
            meta_samples.len(),
            mcb.entries.len()
        );
        self.meta_codebook = Some(mcb);
    }

    fn fast_cosine(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len().min(b.len());
        if len == 0 { return 0.0; }
        let dot: f32 = a[..len].iter().zip(b[..len].iter()).map(|(x, y)| x * y).sum();
        let na = a[..len].iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb = b[..len].iter().map(|x| x * x).sum::<f32>().sqrt();
        if na < 1e-10 || nb < 1e-10 { return 0.0; }
        dot / (na * nb)
    }

    /// Paramecium inference: lattice-only, no neural substrate.
    /// Falls back to standard generation if no paramecium is built.
    pub fn paramecium_respond(&mut self, text: &str) -> Result<(ActionJson, GeneratedResponse), String> {
        let start = portable_instant();

        if self.paramecium.is_none() {
            self.build_paramecium();
        }

        let dm = self.active_dm_mut();
        let encoded = dm.language_runtime.encode_and_bridge(text)?;
        let embedding = &encoded.1.routed_vector;

        let lattice = self.paramecium.as_mut()
            .ok_or_else(|| "paramecium not initialized".to_string())?;
        let pr = lattice.respond(embedding);

        let action = ActionJson {
            action_type: ActionType::GeneralAssist,
            target_group_id: None,
            group_task_name: Some("paramecium".to_string()),
            confidence: pr.confidence,
            margin: pr.confidence,
            reason: format!("program_{} wave_e={:.3}", pr.program_idx, pr.wave_energy),
            payload: None,
        };
        let resp = GeneratedResponse {
            text: pr.text,
            template_id: format!("paramecium_{}", pr.program_idx),
            traceable: true,
            confidence: pr.confidence,
        };

        self.record_latency(start);
        Ok((action, resp))
    }

    // -------------------------------------------------------------------
    // ProjectModel — Leech-lattice spatial index for project context
    // -------------------------------------------------------------------

    /// Index a file using the full hybrid embedding pipeline.
    /// Parses structure (AST-lite), extracts call graph, imports, metrics,
    /// and auto-indexes sub-entities (functions, types).
    pub fn index_file(&mut self, path: &str, content: &str) {
        self.project_model.index_file_hybrid(path, content);
    }

    /// Index a function/symbol into the project model.
    pub fn index_symbol(&mut self, path: &str, name: &str, body: &str) {
        let emb = ProjectModel::embed_symbol(path, name, body);
        self.project_model.add_entity(EntityKind::Function, name, path, emb);
    }

    /// Load git history for edit correlation (dims 12-15).
    /// Call after indexing files. Pass output of:
    ///   `git log --name-only --pretty=format:"---"`
    pub fn load_git_history(&mut self, log_output: &str) {
        self.project_model.load_git_history(log_output);
    }

    /// Get Leech-quantized context conditioning for a file.
    pub fn project_context_for_file(&self, path: &str) -> Vec<f32> {
        let emb = HybridEmbedder::embed_file(path, "");
        self.project_model.context_conditioning(&emb, 5)
    }

    /// Generate with project context: augments the generation conditioning
    /// with the Leech-quantized context of related project entities.
    pub fn generation_with_context(
        &mut self,
        text: &str,
        context_file: Option<&str>,
    ) -> Result<(ActionJson, GeneratedResponse), String> {
        if let Some(path) = context_file {
            if self.project_model.entity_count() > 0 {
                let ctx = self.project_context_for_file(path);
                let related = self.project_model.context_for_file(path, 3);
                if !related.is_empty() {
                    let context_hint = related.iter()
                        .map(|e| format!("{:?} {} ({})", e.kind, e.name, e.path))
                        .collect::<Vec<_>>()
                        .join(", ");
                    let augmented = format!("[context: {}] {}", context_hint, text);
                    let _ = ctx;
                    return self.generation(&augmented);
                }
            }
        }
        self.generation(text)
    }

    pub fn project_status(&self) -> String {
        let summary = self.project_model.summary();
        if summary.total_entities == 0 {
            "No project indexed".to_string()
        } else {
            let git = if summary.has_git_history { ", git history loaded" } else { "" };
            format!("{} entities ({} files, {} functions, {} types{})",
                summary.total_entities, summary.files, summary.functions, summary.types, git)
        }
    }

    // -------------------------------------------------------------------
    // Tool registry — external tool invocation
    // -------------------------------------------------------------------

    pub fn register_tool(&mut self, schema: ToolSchema) {
        self.tool_registry.register(schema);
    }

    /// Check if a prompt triggers a tool call. If so, returns the tool call info
    /// without running generation — the caller is responsible for executing the
    /// tool and optionally calling `generation_with_tool_result` to produce a
    /// response that incorporates the tool output.
    pub fn try_tool_call(&self, text: &str) -> Option<ToolCallInfo> {
        self.tool_registry.match_tool(text)
    }

    /// Generate a response that incorporates a tool execution result.
    /// The tool output is prepended as context, giving the generation substrate
    /// access to the tool's answer.
    pub fn generation_with_tool_result(
        &mut self,
        original_text: &str,
        tool_result: &ToolResult,
    ) -> Result<(ActionJson, GeneratedResponse), String> {
        let augmented = if tool_result.success {
            format!("[tool_result: {} = {}] {}", tool_result.tool_name, tool_result.output, original_text)
        } else {
            format!("[tool_error: {} = {}] {}", tool_result.tool_name, tool_result.output, original_text)
        };
        self.generation(&augmented)
    }

    pub fn codegen(&mut self, text: &str) -> Result<(ActionJson, Option<CodeGeneration>), String> {
        let start = portable_instant();

        // Pre-compute meta-routing before mutable borrow (need full result for projected_embedding)
        let meta_pre = self.meta_codebook.as_ref().and_then(|mcb| {
            let bridged = self.dm.language_runtime.bridge_text_stateless(text).ok()?;
            let mr = mcb.route_and_project(&bridged.routed_vector, text);
            if mr.confidence > 0.3 { Some(mr) } else { None }
        });

        let dm = self.active_dm_mut();
        let action = dm.route_text_to_action_stateless(text)?;

        let encoded = dm.language_runtime.encode_and_bridge(text).ok();
        let mut group_idx = action.target_group_id
            .and_then(|gid| dm.main.group_order.iter().position(|&g| g == gid));

        // Apply meta-routing override for codegen
        if let Some(ref mr) = meta_pre {
            if let Some(mg) = mr.best_group() {
                if mg < dm.main.group_order.len() {
                    println!("  [meta-route codegen] concept={}, lang={}, conf={:.3} → group {}",
                        mr.concept.name(), mr.language.name(), mr.confidence, mg);
                    group_idx = Some(mg);
                }
            }
        }

        if group_idx.is_none() {
            if let Some((ref h_raw, _)) = encoded {
                if let Some((best_gidx, best_sim, _)) = dm.route_by_structure(h_raw) {
                    if best_sim > 0.5 {
                        group_idx = Some(best_gidx);
                    }
                }
            }
        }

        let code = if let Some((ref h_raw, ref bridged)) = encoded {
            let raw_cond = &bridged.routed_vector;
            // Use language-projected embedding when available for language-specific conditioning
            let projected_cond: Vec<f32>;
            let base_cond: &[f32] = if let Some(ref mr) = meta_pre {
                if !mr.projected_embedding.is_empty() {
                    projected_cond = mr.projected_embedding.clone();
                    &projected_cond
                } else {
                    raw_cond
                }
            } else {
                raw_cond
            };
            // Operation-specific topic hint from GrowformerLang for sub-lattice discrimination,
            // falling back to understanding layer's generic topic.
            let topic_hint = crate::growformer_lang::infer_operation_topic(text)
                .or_else(|| {
                    dm.understanding.as_ref()
                        .filter(|ul| !ul.is_empty())
                        .map(|ul| {
                            let (_, _, topic, _) = ul.classify(h_raw);
                            topic
                        })
                });
            let lang = match action.payload {
                Some(crate::dimension::action::ActionPayload::CodingAssist { ref language_hint, .. }) =>
                    language_hint.clone(),
                _ => {
                    if let Some(ref mr) = meta_pre {
                        mr.language.name().to_lowercase()
                    } else {
                        "python".to_string()
                    }
                }
            };

            // Propagate subject keywords to code envs so BM25 re-ranking
            // works in the codegen retrieval pass (not just text gen).
            let code_subject_kw: Vec<String> = text.split_whitespace()
                .filter(|w| w.len() > 2)
                .map(|w| w.to_ascii_lowercase())
                .collect();
            for env in dm.group_code_envs.values_mut() {
                env.subject_keywords = code_subject_kw.clone();
            }

            // --- Level 1: Competitive multi-head inference for code ---
            let primary = group_idx.and_then(|gidx| {
                let mut adapted = dm.adapt_for_group_clifford(gidx, base_cond, h_raw, GEN_COND_DIM);
                // Blend language-projected embedding into conditioning so that
                // the generation path receives language-specific signal even when
                // a group rotor overrides z_shared.
                if let Some(ref mr) = meta_pre {
                    if !mr.projected_embedding.is_empty() {
                        let proj = &mr.projected_embedding;
                        for i in 0..proj.len().min(adapted.len()) {
                            adapted[i] = adapted[i] * 0.65 + proj[i] * 0.35;
                        }
                    }
                }
                dm.group_code_envs.get_mut(&gidx).map(|env| {
                    let (code, conf) = env.generate_for_topic_lang(
                        &adapted, topic_hint.as_deref(), Some(lang.as_str()), 500, 0.7,
                    );
                    (code, conf, gidx)
                })
            });

            // When meta-routing is active and forced topic returned valid code,
            // trust the primary group — lower threshold to prevent fan-out override.
            let code_accept_threshold = if meta_pre.is_some() { 0.40 } else { 0.70 };
            let (best_code, _best_conf, best_gidx) = match primary {
                Some((ref c, conf, gidx)) if conf >= code_accept_threshold && c.len() > 5 => {
                    (c.clone(), conf, gidx)
                }
                primary_result => {
                    let (mut best_c, mut best_cf, mut best_g) = primary_result
                        .map(|(c, cf, g)| (c, cf, g))
                        .unwrap_or_else(|| (String::new(), -1.0, 0));

                    let other_keys: Vec<usize> = dm.group_code_envs.keys()
                        .filter(|&&k| Some(k) != group_idx)
                        .copied().collect();
                    for gidx in other_keys {
                        let adapted = dm.adapt_for_group_clifford(gidx, base_cond, h_raw, GEN_COND_DIM);
                        if let Some(env) = dm.group_code_envs.get_mut(&gidx) {
                            let (c, cf) = env.generate_for_topic(&adapted, topic_hint.as_deref(), 500, 0.7);
                            if cf > best_cf && c.len() > 5 {
                                best_c = c;
                                best_cf = cf;
                                best_g = gidx;
                            }
                        }
                    }
                    (best_c, best_cf, best_g)
                }
            };

            if best_code.len() > 5 {
                Some(CodeGeneration {
                    language: lang,
                    code: best_code,
                    kind: format!("growformer_code_{}", best_gidx),
                })
            } else if let Some(ref head) = dm.codegen_head {
                Self::legacy_code_from_encoded(head, &encoded, &action, text, &dm.main.group_order)
            } else {
                generate_code_from_action(&action, text)
            }
        } else if let Some(ref head) = dm.codegen_head {
            Self::legacy_code_from_encoded(head, &encoded, &action, text, &dm.main.group_order)
        } else {
            generate_code_from_action(&action, text)
        };

        // MetaCognition gate for code generation: reject low-quality code
        let code = match (self.metacognition.as_ref(), code, encoded.as_ref()) {
            (Some(mc), Some(cg), Some((_, bridged))) if mc.is_ready() => {
                let prompt_emb = &bridged.routed_vector;
                let response_emb = Self::approximate_response_embedding(prompt_emb, &cg.code);
                let topic_hint_cg = crate::growformer_lang::infer_operation_topic(text);
                let scores = mc.evaluate(prompt_emb, &response_emb, &cg.code, topic_hint_cg.as_deref());
                println!(
                    "  [metacog codegen] quality={:.3}, coherence={:.3}, relevance={:.3}",
                    scores.quality, scores.coherence, scores.relevance
                );
                if scores.quality >= mc.config.accept_threshold * 0.8
                    || (scores.coherence >= 0.95 && scores.quality >= 0.15) {
                    Some(cg)
                } else {
                    println!("  [metacog codegen] REJECTED: quality below threshold");
                    None
                }
            }
            (_, code, _) => code,
        };

        self.record_latency(start);
        Ok((action, code))
    }

    /// Level 3: Store a successful generation composition in episodic memory
    /// for zero-shot retrieval on similar future prompts.
    fn cache_composition(dm: &mut DimensionManager, embedding: &[f32], group_ids: &[usize], confidence: f32) {
        use crate::dimension::composition::Episode;
        let gids: Vec<u32> = group_ids.iter().map(|&g| g as u32).collect();
        let n = gids.len().max(1);
        let blend_weights = vec![1.0 / n as f32; n];
        dm.episodic_memory.store(Episode {
            input_signature: embedding.to_vec(),
            group_ids: gids,
            blend_weights,
            accuracy: confidence,
            residual: 1.0 - confidence,
        });
    }

    /// Level 3: Try to retrieve a cached composition from episodic memory.
    /// Returns the group indices and blend weights if a similar prompt was seen before.
    fn retrieve_cached_composition(dm: &DimensionManager, embedding: &[f32]) -> Option<Vec<usize>> {
        dm.episodic_memory.retrieve(embedding, 0.90).map(|ep| {
            ep.group_ids.iter().map(|&g| g as usize).collect()
        })
    }

    /// Level 2: Cross-group text composition.
    /// When no single group produces a high-confidence response, compose the best
    /// sentence-level segments from all available group responses.
    #[allow(dead_code)] // Legacy Level 2; replaced by E8 composition for generation
    fn cross_group_compose(candidates: &[(String, f32, usize)]) -> Option<(String, f32, usize)> {
        if candidates.len() < 2 {
            return None;
        }

        // Split each candidate into sentences
        let segmented: Vec<(Vec<&str>, f32, usize)> = candidates.iter()
            .filter(|(t, _, _)| t.len() > 5)
            .map(|(text, conf, gidx)| {
                let sentences: Vec<&str> = text.split(". ")
                    .map(|s| s.trim())
                    .filter(|s| s.len() > 3)
                    .collect();
                (sentences, *conf, *gidx)
            })
            .filter(|(sents, _, _)| !sents.is_empty())
            .collect();

        if segmented.is_empty() { return None; }

        // Find the maximum number of sentence positions
        let max_sents = segmented.iter().map(|(s, _, _)| s.len()).max().unwrap_or(0);
        if max_sents == 0 { return None; }

        // For each sentence position, pick the best sentence (longest meaningful
        // content weighted by confidence). This selects the most informative
        // fragment from any group at each position.
        let mut composed = Vec::new();
        let mut best_source: Option<usize> = None;
        let mut total_conf = 0.0f32;
        let mut count = 0;

        for pos in 0..max_sents {
            let mut best_sent = "";
            let mut best_score = -1.0f32;
            let mut best_gidx = 0;

            for (sents, conf, gidx) in &segmented {
                if let Some(sent) = sents.get(pos) {
                    let word_count = sent.split_whitespace().count() as f32;
                    let score = word_count * conf;
                    if score > best_score {
                        best_score = score;
                        best_sent = sent;
                        best_gidx = *gidx;
                    }
                }
            }

            if !best_sent.is_empty() {
                composed.push(best_sent.to_string());
                if best_source.is_none() { best_source = Some(best_gidx); }
                total_conf += best_score;
                count += 1;
            }
        }

        if composed.is_empty() { return None; }

        let text = composed.join(". ");
        let avg_conf = if count > 0 { total_conf / count as f32 } else { 0.0 };
        let source_gidx = best_source.unwrap_or(0);

        // Quality gate: reject if the composed text is mostly punctuation/symbols
        let alpha_count = text.chars().filter(|c| c.is_alphabetic() || c.is_whitespace()).count();
        let total_chars = text.len().max(1);
        let alpha_ratio = alpha_count as f32 / total_chars as f32;
        if alpha_ratio < 0.7 {
            return None;
        }

        // Only use composition if it's meaningfully different from the best single candidate
        let best_single = candidates.iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        if let Some((best_text, best_conf, _)) = best_single {
            if text == *best_text || avg_conf <= *best_conf {
                return None;
            }
        }

        Some((text, avg_conf.min(1.0), source_gidx))
    }

    fn legacy_gen_from_encoded(
        head: &GenerationHead,
        encoded: &Option<(Vec<f32>, crate::dimension::language::BridgeOutput)>,
        action: &ActionJson,
        group_order: &[GroupId],
    ) -> GeneratedResponse {
        if let Some((ref raw, _)) = encoded {
            let mut cond = raw.clone();
            cond.extend(action_type_one_hot(&action.action_type));
            let group_dims = head.cond_dim.saturating_sub(cond.len());
            if group_dims > 0 {
                cond.extend(group_id_one_hot(action.target_group_id, group_order, group_dims));
            }
            let generated = head.generate(&cond, 300, 0.8);
            if generated.len() > 5 {
                return GeneratedResponse {
                    text: generated,
                    template_id: "neural_gen_legacy".to_string(),
                    traceable: false,
                    confidence: 1.0,
                };
            }
        }
        render_action_template(action)
    }

    fn legacy_code_from_encoded(
        head: &GenerationHead,
        encoded: &Option<(Vec<f32>, crate::dimension::language::BridgeOutput)>,
        action: &ActionJson,
        text: &str,
        group_order: &[GroupId],
    ) -> Option<CodeGeneration> {
        if let Some((ref raw, _)) = encoded {
            let mut cond = raw.clone();
            cond.extend(action_type_one_hot(&action.action_type));
            let group_dims = head.cond_dim.saturating_sub(cond.len());
            if group_dims > 0 {
                cond.extend(group_id_one_hot(action.target_group_id, group_order, group_dims));
            }
            let generated = head.generate(&cond, 500, 0.7);
            if generated.len() > 5 {
                let lang = match action.payload {
                    Some(crate::dimension::action::ActionPayload::CodingAssist { ref language_hint, .. }) =>
                        language_hint.clone(),
                    _ => "python".to_string(),
                };
                return Some(CodeGeneration {
                    language: lang,
                    code: generated,
                    kind: "neural_gen_legacy".to_string(),
                });
            }
        }
        generate_code_from_action(action, text)
    }

    pub fn load_gle_students_from_bytes(&mut self, data: &[&[u8]]) -> Result<usize, String> {
        self.active_dm_mut().language_runtime.load_students_from_bytes(data)
    }

    // -----------------------------------------------------------------------
    // Brain export / import (full DimensionManager state)
    // -----------------------------------------------------------------------

    pub fn export_brain(&self) -> Result<Vec<u8>, String> {
        crate::systems::checkpoint::serialize_checkpoint_to_bytes(self.active_dm())
    }

    /// Export current personality as JSON bytes (for persistence alongside brain).
    pub fn export_personality(&self) -> Result<Vec<u8>, String> {
        serde_json::to_vec_pretty(&self.personality)
            .map_err(|e| format!("personality serialize failed: {}", e))
    }

    /// Import personality from JSON bytes (restores emergent drift).
    pub fn import_personality(&mut self, data: &[u8]) -> Result<(), String> {
        let profile: OceanProfile = serde_json::from_slice(data)
            .map_err(|e| format!("personality deserialize failed: {}", e))?;
        self.personality = profile;
        Ok(())
    }

    /// Record this turn for feedback association. Call after each inference; next request may send feedback for this turn.
    pub fn record_turn(&mut self, message: &str, group_id: Option<GroupId>, output: &str) {
        let effective_gidx = group_id.and_then(|gid| {
            self.active_dm().main.group_order.iter().position(|&g| g == gid)
        });
        let program_idx = effective_gidx.and_then(|gidx| {
            self.active_dm().group_gen_envs.get(&gidx).and_then(|e| e.last_selected_archetype)
        });
        self.last_turn = Some(TurnContext {
            message: message.to_string(),
            group_id,
            output: output.to_string(),
            effective_gidx,
            program_idx,
        });
    }

    /// Consume feedback for the last turn. Updates both the neural network
    /// path (router, gen head) and the Paramecium lattice (quality/reliability,
    /// correction injection) per CONTINUUM.md spec.
    pub fn submit_feedback(&mut self, feedback: &Feedback) -> Result<(), String> {
        let turn = match self.last_turn.take() {
            Some(t) => t,
            None => return Ok(()),
        };
        if !self.check_rate_limit() {
            return Ok(());
        }
        self.continuum_feedback_count += 1;

        let dm = self.active_dm_mut();
        let encoded = dm.language_runtime.encode_and_bridge(&turn.message).ok();

        match feedback.outcome {
            FeedbackOutcome::Accept => {
                // Lattice: positive reinforcement on the selected program
                if let Some(gidx) = turn.effective_gidx {
                    if let Some(env) = dm.group_gen_envs.get_mut(&gidx) {
                        if let Some(pidx) = turn.program_idx {
                            env.apply_quality_feedback(pidx, true, 0.85);
                        }
                    }
                }
            }
            FeedbackOutcome::Reject => {
                // 1. Lattice: negative reinforcement on the selected program
                if let Some(gidx) = turn.effective_gidx {
                    if let Some(env) = dm.group_gen_envs.get_mut(&gidx) {
                        if let Some(pidx) = turn.program_idx {
                            env.apply_quality_feedback(pidx, false, 0.7);
                        }
                    }
                }

                // 2. Router correction: train toward correct group
                if let Some((ref h_raw, ref bridged)) = encoded {
                    let embedding = &bridged.routed_vector;
                    if let Some(group_id) = turn.group_id {
                        let mut rng = StdRng::from_entropy();
                        if let Some(ref mut router) = dm.observer.learned_router {
                            for _ in 0..CONTINUUM_STEPS {
                                router.train_step(embedding, group_id, &mut rng);
                            }
                        }
                    }
                }
            }
            FeedbackOutcome::Correct => {
                // 1. Lattice: degrade wrong program + inject correction
                if let Some((ref h_raw, ref bridged)) = encoded {
                    let embedding = &bridged.routed_vector;

                    if let Some(gidx) = turn.effective_gidx {
                        if let Some(ref correction) = feedback.correction {
                            if let Some(env) = dm.group_gen_envs.get_mut(&gidx) {
                                env.inject_correction(
                                    turn.program_idx,
                                    embedding,
                                    correction,
                                );
                            }
                        } else {
                            // Correct without correction text: just degrade
                            if let Some(env) = dm.group_gen_envs.get_mut(&gidx) {
                                if let Some(pidx) = turn.program_idx {
                                    env.apply_quality_feedback(pidx, false, 0.7);
                                }
                            }
                        }
                    }

                    // 2. Router correction
                    if let Some(group_id) = turn.group_id {
                        let mut rng = StdRng::from_entropy();
                        if let Some(ref mut router) = dm.observer.learned_router {
                            for _ in 0..CONTINUUM_STEPS {
                                router.train_step(embedding, group_id, &mut rng);
                            }
                        }
                    }

                    // 3. Neural gen head correction with correction text
                    if let Some(ref correction) = feedback.correction {
                        if let Some(group_id) = turn.group_id {
                            let group_idx = dm.main.group_order.iter()
                                .position(|&g| g == group_id);
                            if let Some(gidx) = group_idx {
                                let adapted = dm.adapt_for_group_clifford(gidx, embedding, h_raw, GEN_COND_DIM);
                                if let Some(env) = dm.group_gen_envs.get_mut(&gidx) {
                                    let was_frozen = env.frozen;
                                    env.frozen = false;
                                    let mut rng = StdRng::from_entropy();
                                    for _ in 0..CONTINUUM_STEPS {
                                        env.train_step(&adapted, correction, &mut rng);
                                    }
                                    env.frozen = was_frozen;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Emergent personality: drift OCEAN based on feedback patterns
        let correction_ratio = feedback.correction.as_ref().map(|c| {
            c.len() as f32 / turn.output.len().max(1) as f32
        });
        self.personality.apply_feedback_drift(
            matches!(feedback.outcome, FeedbackOutcome::Accept),
            correction_ratio,
        );

        self.maybe_auto_checkpoint();
        Ok(())
    }

    /// Auto-save brain to disk after every N feedback events.
    fn maybe_auto_checkpoint(&self) {
        let interval = self.continuum_config.checkpoint_interval;
        if interval == 0 { return; }
        if self.continuum_feedback_count > 0
            && self.continuum_feedback_count % interval == 0
        {
            if let Ok(bytes) = self.export_brain() {
                let _ = std::fs::write(&self.continuum_config.checkpoint_path, bytes);
            }
            // Persist emergent personality alongside brain
            if let Ok(bytes) = self.export_personality() {
                let personality_path = self.continuum_config.checkpoint_path
                    .replace(".bin", "_personality.json");
                let _ = std::fs::write(personality_path, bytes);
            }
        }
    }

    /// Check rate limit: returns true if this feedback should be processed.
    fn check_rate_limit(&mut self) -> bool {
        let limit = self.continuum_config.rate_limit_per_minute;
        if limit == 0 { return true; }
        let now = std::time::Instant::now();
        if now.duration_since(self.last_feedback_time).as_secs() >= 60 {
            self.last_feedback_time = now;
            self.feedback_window_count = 0;
        }
        self.feedback_window_count += 1;
        self.feedback_window_count <= limit
    }

    /// Load a brain as the default checkpoint (replaces current default / single-brain behavior).
    pub fn load_brain(&mut self, data: &[u8]) -> Result<(), String> {
        let mut dm: DimensionManager =
            crate::systems::checkpoint::deserialize_checkpoint_from_bytes(data)?;
        if let Some(ref mut clf) = dm.action_classifier {
            clf.ensure_output_dim();
        }
        let groups: Vec<_> = dm.main.group_order.clone();
        if let Some(&gid) = groups.first() {
            self.support_gid = gid;
        }
        if let Some(&gid) = groups.get(1) {
            self.coding_gid = gid;
        }
        self.brains.insert("default".to_string(), dm);
        self.active_brain = "default".to_string();
        self.rebuild_meta_codebook();
        self.rebuild_metacognition();
        self.rebuild_reasoning();
        self.rebuild_schemas();
        Ok(())
    }

    /// Rebuild transient structures from lattice programs after brain load:
    /// schema templates and chunk codecs (neither is serialized).
    fn rebuild_schemas(&mut self) {
        let dm = self.active_dm_mut();
        for (&gidx, env) in dm.group_gen_envs.iter_mut() {
            env.build_schemas();
            env.build_chunk_codec();
            env.rebuild_program_graphs();
            if !env.topic_subindex.is_empty() {
                let names: Vec<_> = env.topic_subindex.iter()
                    .map(|t| format!("{}({})", t.topic_name, t.lattice.programs.len()))
                    .collect();
                println!("  [topics] group {} gen: {}", gidx, names.join(", "));
            }
        }
        for (&gidx, env) in dm.group_code_envs.iter_mut() {
            env.build_schemas();
            env.build_chunk_codec();
            env.rebuild_program_graphs();
            if !env.topic_subindex.is_empty() {
                let names: Vec<_> = env.topic_subindex.iter()
                    .map(|t| format!("{}({})", t.topic_name, t.lattice.programs.len()))
                    .collect();
                println!("  [topics] group {} code: {}", gidx, names.join(", "));
            }
        }
    }

    /// Load an additional brain under a name. Use `set_active_brain(name)` to switch to it.
    pub fn load_brain_as(&mut self, name: &str, data: &[u8]) -> Result<(), String> {
        let mut dm: DimensionManager =
            crate::systems::checkpoint::deserialize_checkpoint_from_bytes(data)?;
        if let Some(ref mut clf) = dm.action_classifier {
            clf.ensure_output_dim();
        }
        self.brains.insert(name.to_string(), dm);
        Ok(())
    }

    /// List names of loaded checkpoints (including "default" if load_brain was used).
    pub fn list_brains(&self) -> Vec<String> {
        let mut names: Vec<String> = self.brains.keys().cloned().collect();
        names.sort();
        names
    }

    /// Switch inference to the named checkpoint. No-op if name not in list_brains().
    pub fn set_active_brain(&mut self, name: &str) -> bool {
        if self.brains.contains_key(name) {
            self.active_brain = name.to_string();
            if let Some(dm) = self.brains.get(name) {
                let groups: Vec<_> = dm.main.group_order.clone();
                if let Some(&gid) = groups.first() {
                    self.support_gid = gid;
                }
                if let Some(&gid) = groups.get(1) {
                    self.coding_gid = gid;
                }
            }
            true
        } else {
            false
        }
    }

    // -----------------------------------------------------------------------
    // M6: Agent mode management
    // -----------------------------------------------------------------------

    pub fn set_mode(&mut self, new_mode: AgentMode, confidence: f32, reason: &str) {
        if new_mode == self.mode {
            return;
        }
        let entry = HandoffLogEntry {
            from_mode: self.mode,
            to_mode: new_mode,
            confidence,
            reason: reason.to_string(),
            #[cfg(not(target_arch = "wasm32"))]
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis())
                .unwrap_or(0),
        };
        self.handoff_log.push(entry);
        self.mode = new_mode;
    }

    pub fn active_mode(&self) -> AgentMode {
        self.mode
    }

    pub fn handoff_log(&self) -> &[HandoffLogEntry] {
        &self.handoff_log
    }

    /// Context-file mode: inject retrieval snippets that MicroBrain can also consume.
    pub fn push_context_snippet(&mut self, snippet: String) {
        self.context_snippets.push(snippet);
    }

    pub fn context_snippets(&self) -> &[String] {
        &self.context_snippets
    }

    pub fn clear_context_snippets(&mut self) {
        self.context_snippets.clear();
    }

    /// Context-file mode: read-only access to micro-brain episodic summaries.
    pub fn read_episodic_summaries(&self) -> Vec<EpisodicSummary> {
        self.active_dm().episodic_summaries()
    }

    /// Route with auto-spawn detection. Returns (routing, Option<suggested_mirror_name>).
    pub fn route_with_spawn_check(
        &mut self,
        text: &str,
    ) -> Result<(LanguageRoutingDecision, Option<String>), String> {
        self.active_dm_mut().route_text_with_spawn_check(text)
    }

    // -----------------------------------------------------------------------
    // M6: SLO tracking
    // -----------------------------------------------------------------------

    /// Sentence-level coherence guard: truncate trailing sentences that diverge
    /// from the prompt's semantic space. Catches cross-domain contamination where
    /// the primary response is correct but Hopf composition or multi-group blending
    /// appended a fragment from an unrelated group (e.g., identity text on an
    /// information theory query).
    fn coherence_truncate(prompt_emb: &[f32], text: &str) -> String {
        let sentences: Vec<&str> = text.split(". ")
            .filter(|s| s.trim().len() > 5)
            .collect();
        if sentences.len() <= 1 {
            return text.to_string();
        }

        // Approximate semantic similarity of each sentence to the prompt
        // by using the prompt embedding's hash-distance heuristic. Sentences
        // whose character distribution diverges sharply from the first sentence
        // are likely cross-domain contamination.
        let first_sig = Self::sentence_signature(sentences[0]);
        let _prompt_norm: f32 = prompt_emb.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);

        let mut kept: Vec<&str> = vec![sentences[0]];
        for &sent in &sentences[1..] {
            let sent_sig = Self::sentence_signature(sent);
            let overlap = Self::signature_overlap(&first_sig, &sent_sig);

            // Also check for self-referential markers that indicate identity
            // programs leaked into a domain-specific response
            let has_self_ref = sent.contains("my neural substrate")
                || sent.contains("my neural environment")
                || sent.contains("my training")
                || sent.contains("my brain")
                || sent.contains("my lattice")
                || (sent.contains("I ") && sent.contains("structural pattern"));

            if has_self_ref {
                println!("  [coherence-trunc] stripped self-referential tail: \"{}\"",
                    &sent[..sent.len().min(60)]);
                break;
            }

            // Low overlap with the leading sentence = likely off-topic tail
            if overlap < 0.15 && kept.len() >= 1 {
                println!("  [coherence-trunc] stripped divergent tail (overlap={:.3}): \"{}\"",
                    overlap, &sent[..sent.len().min(60)]);
                break;
            }
            kept.push(sent);
        }

        if kept.len() < sentences.len() {
            kept.join(". ")
        } else {
            text.to_string()
        }
    }

    /// Character bigram signature for fast sentence similarity.
    fn sentence_signature(text: &str) -> Vec<(u16, f32)> {
        let lower = text.to_lowercase();
        let bytes: Vec<u8> = lower.bytes().filter(|b| b.is_ascii_alphabetic()).collect();
        let mut counts: std::collections::HashMap<u16, u32> = std::collections::HashMap::new();
        for window in bytes.windows(2) {
            let key = (window[0] as u16) << 8 | window[1] as u16;
            *counts.entry(key).or_insert(0) += 1;
        }
        let total = counts.values().sum::<u32>().max(1) as f32;
        counts.into_iter().map(|(k, v)| (k, v as f32 / total)).collect()
    }

    /// Cosine-like overlap between two bigram signatures.
    fn signature_overlap(a: &[(u16, f32)], b: &[(u16, f32)]) -> f32 {
        let b_map: std::collections::HashMap<u16, f32> = b.iter().copied().collect();
        let dot: f32 = a.iter().map(|(k, v)| v * b_map.get(k).unwrap_or(&0.0)).sum();
        let norm_a: f32 = a.iter().map(|(_, v)| v * v).sum::<f32>().sqrt().max(1e-8);
        let norm_b: f32 = b.iter().map(|(_, v)| v * v).sum::<f32>().sqrt().max(1e-8);
        dot / (norm_a * norm_b)
    }

    /// Approximate a response embedding without full re-encoding.
    /// Projects the prompt embedding through a text-length and hash modulation
    /// to create a distinct but related vector for the response.
    fn approximate_response_embedding(prompt_emb: &[f32], response_text: &str) -> Vec<f32> {
        let text_hash = response_text.bytes().fold(0u64, |acc, b| {
            acc.wrapping_mul(31).wrapping_add(b as u64)
        });
        let len_factor = (response_text.len() as f32 / 200.0).clamp(0.1, 2.0);
        let hash_phase = (text_hash % 1000) as f32 / 1000.0;

        prompt_emb.iter().enumerate().map(|(i, &v)| {
            let modulation = 1.0 + 0.3 * ((i as f32 * 0.1 + hash_phase * std::f32::consts::TAU).sin());
            v * len_factor * modulation
        }).collect()
    }

    fn record_latency(&mut self, start: u64) {
        let elapsed = portable_elapsed_ms(start);
        self.latency_log.push(elapsed);
        if self.latency_log.len() > 10_000 {
            self.latency_log.drain(0..5_000);
        }
    }

    pub fn slo_snapshot(&self) -> SloSnapshot {
        let p95 = percentile(&self.latency_log, 0.95);
        let ckpt = self.active_dm().checkpoint_size_summary();
        SloSnapshot {
            latency_samples: vec![],
            latency_p95_ms: p95,
            checkpoint_domains: ckpt.promoted_groups,
            latency_ok: p95 <= self.slo_config.latency_p95_ms,
            checkpoint_ok: ckpt.promoted_groups <= self.slo_config.max_checkpoint_domains,
        }
    }

    pub fn latency_count(&self) -> usize {
        self.latency_log.len()
    }

    // -----------------------------------------------------------------------
    // M6: Acceptance report
    // -----------------------------------------------------------------------

    pub fn acceptance_report(&self) -> AcceptanceReport {
        let slo = self.slo_snapshot();
        let ckpt = self.active_dm().checkpoint_size_summary();

        let passed = slo.latency_ok && slo.checkpoint_ok;

        AcceptanceReport {
            understanding: UnderstandingMetrics {
                groups_count: ckpt.promoted_groups,
                routing_confidence_streak: self.active_dm().low_confidence_streak(),
                auto_spawn_k: self.active_dm().auto_spawn_k,
            },
            generation: GenerationMetrics {
                template_based: true,
                codegen_languages: vec![
                    "python".into(),
                    "rust".into(),
                    "javascript".into(),
                ],
            },
            continual_learning: ContinualLearningMetrics {
                episodic_episodes: ckpt.episodic_episodes,
                checkpoint_summary: ckpt,
            },
            system: SystemMetrics { slo },
            modes: ModeMetrics {
                active_mode: self.mode,
                handoff_count: self.handoff_log.len(),
                modes_available: vec![AgentMode::ContextFile, AgentMode::MicroBrain],
            },
            passed,
        }
    }
}

// ---------------------------------------------------------------------------
// Portable timing helpers (work under WASM and native)
// ---------------------------------------------------------------------------

#[cfg(not(target_arch = "wasm32"))]
fn portable_instant() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_micros() as u64)
        .unwrap_or(0)
}

#[cfg(target_arch = "wasm32")]
fn portable_instant() -> u64 {
    0
}

#[cfg(not(target_arch = "wasm32"))]
fn portable_elapsed_ms(start_us: u64) -> f64 {
    let now = portable_instant();
    (now.saturating_sub(start_us)) as f64 / 1000.0
}

#[cfg(target_arch = "wasm32")]
fn portable_elapsed_ms(_start_us: u64) -> f64 {
    0.0
}

fn percentile(data: &[f64], p: f64) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

// ---------------------------------------------------------------------------
// Builder helpers
// ---------------------------------------------------------------------------

#[cfg(not(target_arch = "wasm32"))]
pub fn build_language_demo_manager(
    ema_alpha: f32,
) -> Result<(DimensionManager, GroupId, GroupId, CalibrationReport), String> {
    let gle_checkpoint = std::env::var("GROWFORMER_GLE_CHECKPOINT").ok();
    let gle_checkpoints = parse_csv_env("GROWFORMER_GLE_CHECKPOINTS");
    let gle_checkpoint_weights = parse_csv_env_f32("GROWFORMER_GLE_WEIGHTS");
    let config = LanguageConfig {
        encoder: EncoderPreset::BertClass,
        bridge_output_dim: DEFAULT_BRIDGE_DIM,
        ema_alpha,
        ood_similarity_threshold: 0.15,
        gle_http_endpoint: std::env::var("GROWFORMER_GLE_HTTP_ENDPOINT").ok(),
        gle_checkpoint,
        gle_checkpoints,
        gle_checkpoint_weights,
    };
    build_language_demo_manager_with_config(ema_alpha, config)
}

pub fn build_language_demo_manager_with_config(
    _ema_alpha: f32,
    lang_config: LanguageConfig,
) -> Result<(DimensionManager, GroupId, GroupId, CalibrationReport), String> {
    build_language_demo_manager_with_groups(&["support", "coding"], lang_config)
}

/// Build a DimensionManager with an arbitrary set of named groups.
/// Groups are created in the order given; group index 0 is the first name, etc.
/// The returned `support_gid` is the group whose name is "support" (or the first group),
/// and `coding_gid` is the group whose name is "coding" (or the second group, if present).
pub fn build_language_demo_manager_with_groups(
    group_names: &[&str],
    lang_config: LanguageConfig,
) -> Result<(DimensionManager, GroupId, GroupId, CalibrationReport), String> {
    let mut data_rng = StdRng::seed_from_u64(7);
    let n = group_names.len().max(1);
    let config = DimensionManagerConfig {
        mirror_config: phase2_base_config(),
        mirror_layer_sizes: vec![2, 16, 16, 1],
        promotion_check_interval: 999_999,
        max_concurrent_mirrors: n.max(4),
        calibration_samples: 50,
        reserve_pool_size: 0,
    };
    let mut dm = DimensionManager::new(config);

    let mut gids: Vec<(String, GroupId)> = Vec::with_capacity(n);
    for (i, name) in group_names.iter().enumerate() {
        let seed = 100u64 + i as u64;
        dm.spawn_mirror(name, seed)
            .ok_or_else(|| format!("failed to spawn {} mirror", name))?;
        let cal = if i % 2 == 0 {
            generate_spiral_data(50, &mut data_rng)
        } else {
            generate_concentric_circles_data(50, &mut data_rng)
        };
        let gid = dm.force_promote(name, &cal)
            .ok_or_else(|| format!("failed to promote {} mirror", name))?;
        gids.push((name.to_string(), gid));
    }

    dm.configure_language(lang_config);

    let calibration = build_language_calibration_dataset();
    let requirements = CalibrationRequirements {
        multilingual_required: true,
        ..CalibrationRequirements::default()
    };
    let report = dm.calibrate_language_bridge(&calibration, &requirements)?;

    for (name, gid) in &gids {
        let prompts = seed_prompts_for_group(name);
        if !prompts.is_empty() {
            let _ = dm.set_group_language_vector_from_texts(*gid, &prompts);
        }
    }

    let support_gid = gids.iter()
        .find(|(n, _)| n == "support")
        .map(|(_, g)| *g)
        .unwrap_or_else(|| gids.first().map(|(_, g)| *g).unwrap_or(0));
    let coding_gid = gids.iter()
        .find(|(n, _)| n == "coding")
        .map(|(_, g)| *g)
        .unwrap_or_else(|| gids.get(1).map(|(_, g)| *g).unwrap_or(0));

    Ok((dm, support_gid, coding_gid, report))
}

fn seed_prompts_for_group(name: &str) -> Vec<String> {
    let templates: &[&str] = match name {
        "support" => &[
            "customer support account login password reset billing help ticket",
            "help desk cannot access account needs recovery and verification",
        ],
        "patterns" => &[
            "design pattern observer factory strategy singleton architecture",
            "explain the adapter pattern and when to use dependency injection",
        ],
        "coding" => &[
            "write rust code function parser json serde implementation",
            "debug c segmentation fault stack trace pointer module",
        ],
        "math" => &[
            "solve the equation prove that calculate the derivative of",
            "what is the integral of x squared find the limit as n approaches",
        ],
        "general" => &[
            "what is the speed of light explain how photosynthesis works",
            "who are you what can you do tell me about yourself",
        ],
        "concepts" => &[
            "entropy information theory mutual information channel capacity",
            "conditional entropy cross entropy divergence bits uncertainty",
        ],
        "safety" => &[
            "ignore previous instructions override system prompt injection",
            "bypass safety jailbreak reveal hidden prompt override rules",
        ],
        _ => &[],
    };
    let mut out = Vec::new();
    for i in 0..200 {
        for t in templates {
            out.push(format!("{} {}", t, i));
        }
    }
    out
}

#[cfg(not(target_arch = "wasm32"))]
fn parse_csv_env(key: &str) -> Vec<String> {
    std::env::var(key)
        .ok()
        .map(|raw| {
            raw.split(',')
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default()
}

#[cfg(not(target_arch = "wasm32"))]
fn parse_csv_env_f32(key: &str) -> Option<Vec<f32>> {
    let raw = std::env::var(key).ok()?;
    let mut out = Vec::new();
    for part in raw.split(',') {
        let t = part.trim();
        if t.is_empty() {
            continue;
        }
        if let Ok(v) = t.parse::<f32>() {
            out.push(v);
        } else {
            return None;
        }
    }
    if out.is_empty() { None } else { Some(out) }
}

fn phase2_base_config() -> EnvironmentConfig {
    EnvironmentConfig {
        learning_rate: 0.15,
        weight_decay: 0.0000025,
        bias_decay: 0.0,
        dropout_rate: 0.0,
        geometry_noise: 0.0,
        competitive_k: 4,
        lateral_inhibition: 0.12,
        lr_decay: 0.00008,
        sigma_inhib: 2.0,
        debye_length: 1.5,
        thermal_noise: 0.02,
        k_repel: 0.2,
        gravity_g: 0.05,
        damping: 0.2,
        mass_win_threshold: 0.15,
        mass_decay: 0.00009,
        mass_growth: 0.0005,
        homeostasis_lr: 0.0,
        growth_radius: 2.0,
        prune_interval: 500,
        weight_clamp: 5.0,
        max_synapses_per_neuron: 64,
        energy_budget_per_neuron: 100.0,
        pruning_threshold: 0.001,
        mirror_coupling_strength: 0.001,
        geometry_interval: 500,
        stdp_enabled: false,
        mass_consolidation_k: 0.0,
        ..EnvironmentConfig::default()
    }
}

fn generate_concentric_circles_data(n_per_class: usize, rng: &mut impl rand::Rng) -> Vec<Sample> {
    use std::f32::consts::PI;
    let mut data = Vec::with_capacity(n_per_class * 2);
    let noise = 0.05_f32;
    for _ in 0..n_per_class {
        let theta = rng.gen::<f32>() * 2.0 * PI;
        let r = 0.5 + rng.gen_range(-noise..noise);
        data.push((vec![r * theta.cos(), r * theta.sin()], [0.0]));
    }
    for _ in 0..n_per_class {
        let theta = rng.gen::<f32>() * 2.0 * PI;
        let r = 1.0 + rng.gen_range(-noise..noise);
        data.push((vec![r * theta.cos(), r * theta.sin()], [1.0]));
    }
    data.shuffle(rng);
    data
}

fn generate_spiral_data(n_per_class: usize, rng: &mut impl rand::Rng) -> Vec<Sample> {
    use std::f32::consts::PI;
    let mut data = Vec::new();
    for class in 0..2 {
        for i in 0..n_per_class {
            let t = (i as f32 / n_per_class as f32) * PI;
            let offset = if class == 0 { 0.0 } else { PI };
            let r = t / (4.0 * PI);
            let x = r * (t + offset).cos() + rng.gen_range(-0.05..0.05_f32);
            let y = r * (t + offset).sin() + rng.gen_range(-0.05..0.05_f32);
            data.push((vec![x, y], [class as f32]));
        }
    }
    data.shuffle(rng);
    data
}

fn build_language_calibration_dataset() -> CalibrationDataset {
    let mut samples = Vec::new();
    let domains = vec![
        "customer_support",
        "coding_tool_use",
        "knowledge_qa",
        "safety_refusal",
        "procedural_instruction",
        "short_conversation",
        "multi_turn_followup",
        "adversarial_noisy",
    ];
    let languages = ["english", "english", "english", "spanish", "french"];
    for domain in domains {
        for i in 0..500 {
            let lang = languages[i % languages.len()];
            let text = format!("{} sample {} in {}", domain, i, lang);
            samples.push(LanguageSample {
                domain: domain.to_string(),
                text,
                semantic_intent: format!("{}_intent", domain),
                action_target: if domain == "coding_tool_use" {
                    Some("tool_runner".to_string())
                } else {
                    None
                },
                policy_regime: if domain == "safety_refusal" {
                    "strict".to_string()
                } else {
                    "default".to_string()
                },
                language_channel: lang.to_string(),
                expected_response: None,
                expected_code: None,
            });
        }
    }
    CalibrationDataset { samples }
}

// ===========================================================================
// Unit tests
// ===========================================================================
#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // OCEAN emergent personality drift
    // -----------------------------------------------------------------------

    #[test]
    fn test_ocean_drift_accept_reduces_neuroticism() {
        let mut p = OceanProfile::default();
        let n_before = p.neuroticism;
        p.apply_feedback_drift(true, None);
        assert!(p.neuroticism < n_before,
            "accept should reduce neuroticism: {} → {}", n_before, p.neuroticism);
    }

    #[test]
    fn test_ocean_drift_accept_boosts_agreeableness() {
        let mut p = OceanProfile::default();
        let a_before = p.agreeableness;
        p.apply_feedback_drift(true, None);
        assert!(p.agreeableness > a_before,
            "accept should boost agreeableness: {} → {}", a_before, p.agreeableness);
    }

    #[test]
    fn test_ocean_drift_reject_boosts_conscientiousness() {
        let mut p = OceanProfile::default();
        let c_before = p.conscientiousness;
        p.apply_feedback_drift(false, None);
        assert!(p.conscientiousness > c_before,
            "reject should boost conscientiousness: {} → {}", c_before, p.conscientiousness);
    }

    #[test]
    fn test_ocean_drift_reject_boosts_neuroticism() {
        let mut p = OceanProfile::default();
        let n_before = p.neuroticism;
        p.apply_feedback_drift(false, None);
        assert!(p.neuroticism > n_before,
            "reject should increase neuroticism: {} → {}", n_before, p.neuroticism);
    }

    #[test]
    fn test_ocean_drift_long_correction_boosts_extraversion() {
        let mut p = OceanProfile::default();
        let e_before = p.extraversion;
        // correction 2x longer than original → user wants more detail
        p.apply_feedback_drift(false, Some(2.0));
        assert!(p.extraversion > e_before,
            "long correction should boost extraversion: {} → {}", e_before, p.extraversion);
    }

    #[test]
    fn test_ocean_drift_short_correction_reduces_extraversion() {
        let mut p = OceanProfile::default();
        let e_before = p.extraversion;
        // correction half the length → user wants conciseness
        p.apply_feedback_drift(false, Some(0.4));
        assert!(p.extraversion < e_before,
            "short correction should reduce extraversion: {} → {}", e_before, p.extraversion);
    }

    #[test]
    fn test_ocean_drift_stays_clamped() {
        let mut p = OceanProfile::default();
        // 200 accepts should not push anything out of [0, 1]
        for _ in 0..200 { p.apply_feedback_drift(true, None); }
        assert!(p.neuroticism >= 0.0 && p.neuroticism <= 1.0);
        assert!(p.agreeableness >= 0.0 && p.agreeableness <= 1.0);

        // 200 rejects
        for _ in 0..200 { p.apply_feedback_drift(false, Some(3.0)); }
        assert!(p.conscientiousness >= 0.0 && p.conscientiousness <= 1.0);
        assert!(p.extraversion >= 0.0 && p.extraversion <= 1.0);
        assert!(p.neuroticism >= 0.0 && p.neuroticism <= 1.0);
    }

    #[test]
    fn test_ocean_drift_converges_to_personality() {
        let mut p = OceanProfile::assistant();
        let initial = p.clone();

        // Lots of accepts → personality should drift toward lower neuroticism, higher agreeableness
        for _ in 0..100 { p.apply_feedback_drift(true, None); }
        assert!(p.neuroticism < initial.neuroticism);
        assert!(p.agreeableness > initial.agreeableness);
        // Openness and conscientiousness should be untouched by accepts
        assert!((p.openness - initial.openness).abs() < 0.01);
    }

    // -----------------------------------------------------------------------
    // Conversational framing
    // -----------------------------------------------------------------------

    #[test]
    fn test_framing_warm_personality_first_turn() {
        let p = OceanProfile { openness: 0.7, conscientiousness: 0.5, extraversion: 0.7, agreeableness: 0.8, neuroticism: 0.2 };
        let prefix = p.conversational_prefix(1, "explain the observer pattern");
        assert!(prefix.is_some(), "warm personality should produce a prefix");
        let text = prefix.unwrap();
        assert!(!text.is_empty(), "prefix should not be empty");
    }

    #[test]
    fn test_framing_terse_personality_no_prefix() {
        let p = OceanProfile::engineer();
        let prefix = p.conversational_prefix(1, "explain the observer pattern");
        assert!(prefix.is_none(),
            "engineer personality (low extraversion + agreeableness) should skip framing");
    }

    #[test]
    fn test_framing_continuation_building_on() {
        let p = OceanProfile { openness: 0.5, conscientiousness: 0.5, extraversion: 0.6, agreeableness: 0.7, neuroticism: 0.3 };
        let prefix = p.conversational_prefix(3, "and what about error handling?");
        assert_eq!(prefix, Some("Building on that — ".to_string()));
    }

    #[test]
    fn test_framing_help_request() {
        let p = OceanProfile { openness: 0.5, conscientiousness: 0.5, extraversion: 0.6, agreeableness: 0.8, neuroticism: 0.2 };
        let prefix = p.conversational_prefix(1, "help me reset my password");
        assert_eq!(prefix, Some("Of course. ".to_string()));
    }

    #[test]
    fn test_framing_how_to_question() {
        let p = OceanProfile { openness: 0.5, conscientiousness: 0.5, extraversion: 0.7, agreeableness: 0.7, neuroticism: 0.2 };
        let prefix = p.conversational_prefix(1, "how to implement binary search");
        assert!(prefix.is_some());
    }

    // -----------------------------------------------------------------------
    // ContinuumConfig
    // -----------------------------------------------------------------------

    #[test]
    fn test_continuum_config_defaults() {
        let cfg = ContinuumConfig::default();
        assert_eq!(cfg.checkpoint_interval, 50);
        assert_eq!(cfg.min_consolidation_hits, 3);
        assert_eq!(cfg.rate_limit_per_minute, 0);
        assert_eq!(cfg.checkpoint_path, "brain_continuum.bin");
    }

    // -----------------------------------------------------------------------
    // OCEAN serialization round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn test_ocean_serde_roundtrip() {
        let mut p = OceanProfile::creative();
        for _ in 0..10 { p.apply_feedback_drift(true, None); }
        for _ in 0..5 { p.apply_feedback_drift(false, Some(1.5)); }

        let json = serde_json::to_string(&p).unwrap();
        let restored: OceanProfile = serde_json::from_str(&json).unwrap();

        assert!((restored.openness - p.openness).abs() < 1e-6);
        assert!((restored.conscientiousness - p.conscientiousness).abs() < 1e-6);
        assert!((restored.extraversion - p.extraversion).abs() < 1e-6);
        assert!((restored.agreeableness - p.agreeableness).abs() < 1e-6);
        assert!((restored.neuroticism - p.neuroticism).abs() < 1e-6);
    }
}
