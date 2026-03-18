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
use crate::dimension::action::{ActionType, ActionPayload};
#[cfg(not(target_arch = "wasm32"))]
use crate::dimension::EncoderPreset;
use crate::spectral::{ProjectModel, EntityKind, HybridEmbedder};
use crate::dimension::tool::{ToolRegistry, ToolSchema, ToolCallInfo, ToolResult};
use crate::types::{EnvironmentConfig, GroupId, Sample};

// ---------------------------------------------------------------------------
// M6: Agent Modes
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentMode {
    ContextFile,
    MicroBrain,
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
}

impl Default for ConversationContext {
    fn default() -> Self {
        Self { history: Vec::new(), max_turns: 20 }
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
    }

    pub fn turn_count(&self) -> usize {
        self.history.iter().filter(|t| t.role == TurnRole::User).count()
    }

    pub fn is_empty(&self) -> bool {
        self.history.is_empty()
    }

    pub fn clear(&mut self) {
        self.history.clear();
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
}

impl LanguageService {
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new_default() -> Result<Self, String> {
        let (dm, support_gid, coding_gid, report) = build_language_demo_manager(0.2)?;
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
        })
    }

    pub fn new_with_config(config: LanguageConfig) -> Result<Self, String> {
        let (dm, support_gid, coding_gid, report) = build_language_demo_manager_with_config(0.2, config)?;
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
        let start = portable_instant();

        if Self::is_identity_query(text) {
            let action = self.active_dm_mut().route_text_to_action_stateless(text)?;
            self.record_latency(start);
            return Ok((action, self.identity_response()));
        }

        // Tool call interception: if the registry matches a tool, return a
        // ToolCall action with the call info. The caller executes the tool
        // and optionally calls generation_with_tool_result for a composed response.
        if let Some(tool_call) = self.tool_registry.match_tool(text) {
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
        let dm = self.active_dm_mut();
        let action = dm.route_text_to_action_stateless(text)?;

        let encoded = dm.language_runtime.encode_and_bridge(text).ok();
        let group_idx = action.target_group_id
            .and_then(|gid| dm.main.group_order.iter().position(|&g| g == gid));

        let resp = if let Some((_, ref bridged)) = encoded {
            // Apply OCEAN personality conditioning to the routed vector
            let mut conditioned = bridged.routed_vector.clone();
            personality.condition_vector(&mut conditioned);
            let routed = &conditioned;

            // --- Level 3: Check episodic memory for cached composition ---
            let _cached_groups = Self::retrieve_cached_composition(dm, routed);

            // Apply OCEAN Hopf diversity bonus to all gen envs
            let div_bonus = personality.hopf_diversity_bonus();
            for env in dm.group_gen_envs.values_mut() {
                env.diversity_bonus = div_bonus;
            }

            // --- Level 1: Competitive multi-head inference with E8 composition ---
            // Run routed group first. If effective confidence < 0.9, run all
            // groups, collect E8 contribution vectors, and blend in E8 space.
            use crate::dimension::group_gen::{
                E8Contribution, e8_blend_quantum, e8_compose_sentences_quantum,
                e8_select_best, compute_q,
            };

            let primary = group_idx.and_then(|gidx| {
                dm.group_gen_envs.get_mut(&gidx).map(|env| {
                    let (text, conf, e8) = env.generate_with_e8(routed, 300, 0.8);
                    E8Contribution { group_idx: gidx, lattice_point: e8, text, confidence: conf }
                })
            });

            let (best_text, best_conf, best_gidx) = match primary {
                Some(ref c) if c.confidence >= 0.9 && c.text.len() > 5 => {
                    (c.text.clone(), c.confidence, c.group_idx)
                }
                primary_result => {
                    let mut contributions: Vec<E8Contribution> = Vec::new();
                    if let Some(c) = primary_result {
                        contributions.push(c);
                    }

                    let other_keys: Vec<usize> = dm.group_gen_envs.keys()
                        .filter(|&&k| Some(k) != group_idx)
                        .copied().collect();
                    for gidx in other_keys {
                        if let Some(env) = dm.group_gen_envs.get_mut(&gidx) {
                            let (text, conf, e8) = env.generate_with_e8(routed, 300, 0.8);
                            if text.len() > 5 {
                                contributions.push(E8Contribution {
                                    group_idx: gidx, lattice_point: e8, text, confidence: conf,
                                });
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
                        let q = compute_q(routed, &contributions);
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
                                Self::cache_composition(dm, routed, &involved, comp_conf);
                            }
                        }

                        (best_t, best_c, best_g)
                    }
                }
            };

            if best_text.len() > 5 {
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

        self.record_latency(start);
        Ok((action, resp))
    }

    /// Conversational generation: uses conversation context + personality.
    /// Tracks multi-turn history and applies EMA modulation based on OCEAN.
    pub fn converse(&mut self, user_text: &str) -> Result<(ActionJson, GeneratedResponse), String> {
        self.conversation.push_user(user_text);

        // Modulate EMA alpha based on personality before encoding
        let base_alpha = self.active_dm().language_runtime.config.ema_alpha;
        let modulated_alpha = self.personality.modulated_ema_alpha(base_alpha);
        self.active_dm_mut().language_runtime.smoother.alpha = modulated_alpha;

        // Build context-augmented prompt: recent history + current message.
        // The EMA smoother in the bridge already blends temporal context,
        // but explicit history prepending improves semantic grounding.
        let context_prompt = if self.conversation.turn_count() > 1 {
            let ctx = self.conversation.context_window(3);
            format!("{} | user: {}", ctx, user_text)
        } else {
            user_text.to_string()
        };

        let (action, resp) = self.generation(&context_prompt)?;

        self.conversation.push_agent(&resp.text);

        // Store turn context for Continuum feedback
        self.last_turn = Some(TurnContext {
            message: user_text.to_string(),
            group_id: action.target_group_id,
            output: resp.text.clone(),
        });

        Ok((action, resp))
    }

    /// Reset conversation context (new session).
    pub fn reset_conversation(&mut self) {
        self.conversation.clear();
        self.active_dm_mut().language_runtime.smoother.reset();
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
        let dm = self.active_dm_mut();
        let action = dm.route_text_to_action_stateless(text)?;

        let encoded = dm.language_runtime.encode_and_bridge(text).ok();
        let group_idx = action.target_group_id
            .and_then(|gid| dm.main.group_order.iter().position(|&g| g == gid));

        let code = if let Some((_, ref bridged)) = encoded {
            let routed = &bridged.routed_vector;
            let lang = match action.payload {
                Some(crate::dimension::action::ActionPayload::CodingAssist { ref language_hint, .. }) =>
                    language_hint.clone(),
                _ => "python".to_string(),
            };

            // --- Level 1: Competitive multi-head inference for code ---
            let primary = group_idx.and_then(|gidx| {
                dm.group_code_envs.get_mut(&gidx).map(|env| {
                    let (code, conf) = env.generate(routed, 500, 0.7);
                    (code, conf, gidx)
                })
            });

            let (best_code, _best_conf, best_gidx) = match primary {
                Some((ref c, conf, gidx)) if conf >= 0.9 && c.len() > 5 => {
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
                        if let Some(env) = dm.group_code_envs.get_mut(&gidx) {
                            let (c, cf) = env.generate(routed, 500, 0.7);
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

    /// Record this turn for feedback association. Call after each inference; next request may send feedback for this turn.
    pub fn record_turn(&mut self, message: &str, group_id: Option<GroupId>, output: &str) {
        self.last_turn = Some(TurnContext {
            message: message.to_string(),
            group_id,
            output: output.to_string(),
        });
    }

    /// Consume feedback for the last turn. Training step not yet implemented; for now only clears last_turn when outcome indicates learning.
    pub fn submit_feedback(&mut self, feedback: &Feedback) -> Result<(), String> {
        let _ = self.last_turn.take();
        match feedback.outcome {
            FeedbackOutcome::Accept => {}
            FeedbackOutcome::Reject | FeedbackOutcome::Correct => {
                // TODO(CONTINUUM): run one or a few training steps (router / head) with small LR; see docs/CONTINUUM.md
            }
        }
        Ok(())
    }

    /// Load a brain as the default checkpoint (replaces current default / single-brain behavior).
    pub fn load_brain(&mut self, data: &[u8]) -> Result<(), String> {
        let dm: DimensionManager =
            crate::systems::checkpoint::deserialize_checkpoint_from_bytes(data)?;
        let groups: Vec<_> = dm.main.group_order.clone();
        if let Some(&gid) = groups.first() {
            self.support_gid = gid;
        }
        if let Some(&gid) = groups.get(1) {
            self.coding_gid = gid;
        }
        self.brains.insert("default".to_string(), dm);
        self.active_brain = "default".to_string();
        Ok(())
    }

    /// Load an additional brain under a name. Use `set_active_brain(name)` to switch to it.
    pub fn load_brain_as(&mut self, name: &str, data: &[u8]) -> Result<(), String> {
        let dm: DimensionManager =
            crate::systems::checkpoint::deserialize_checkpoint_from_bytes(data)?;
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
        bridge_output_dim: 64,
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
    let mut data_rng = StdRng::seed_from_u64(7);
    let config = DimensionManagerConfig {
        mirror_config: phase2_base_config(),
        mirror_layer_sizes: vec![2, 16, 16, 1],
        promotion_check_interval: 999_999,
        max_concurrent_mirrors: 4,
        calibration_samples: 50,
        reserve_pool_size: 0,
    };
    let mut dm = DimensionManager::new(config);

    dm.spawn_mirror("support", 100)
        .ok_or_else(|| "failed to spawn support mirror".to_string())?;
    dm.spawn_mirror("coding", 101)
        .ok_or_else(|| "failed to spawn coding mirror".to_string())?;
    let cal_support = generate_spiral_data(50, &mut data_rng);
    let cal_coding = generate_concentric_circles_data(50, &mut data_rng);
    let support_gid = dm
        .force_promote("support", &cal_support)
        .ok_or_else(|| "failed to promote support mirror".to_string())?;
    let coding_gid = dm
        .force_promote("coding", &cal_coding)
        .ok_or_else(|| "failed to promote coding mirror".to_string())?;

    dm.configure_language(lang_config);

    let calibration = build_language_calibration_dataset();
    let requirements = CalibrationRequirements {
        multilingual_required: true,
        ..CalibrationRequirements::default()
    };
    let report = dm.calibrate_language_bridge(&calibration, &requirements)?;

    let mut support_prompts = Vec::new();
    let mut coding_prompts = Vec::new();
    for i in 0..200 {
        support_prompts.push(format!(
            "customer support account login password reset billing help ticket {}",
            i
        ));
        support_prompts.push(format!(
            "help desk cannot access account needs recovery and verification {}",
            i
        ));
        coding_prompts.push(format!(
            "write rust code function parser json serde implementation {}",
            i
        ));
        coding_prompts.push(format!(
            "debug c segmentation fault stack trace pointer module {}",
            i
        ));
    }
    dm.set_group_language_vector_from_texts(support_gid, &support_prompts)?;
    dm.set_group_language_vector_from_texts(coding_gid, &coding_prompts)?;

    Ok((dm, support_gid, coding_gid, report))
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
