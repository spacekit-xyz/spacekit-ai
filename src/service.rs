use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

use crate::dimension::{
    generate_code_from_action, render_action_template, ActionJson, CalibrationDataset, CalibrationReport,
    CalibrationRequirements, CheckpointSizeSummary, CodeGeneration, DimensionManager, DimensionManagerConfig,
    EpisodicSummary, GeneratedResponse, LanguageConfig, LanguageRoutingDecision, LanguageSample,
};
#[cfg(not(target_arch = "wasm32"))]
use crate::dimension::EncoderPreset;
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
// LanguageService — core shared service
// ---------------------------------------------------------------------------

pub struct LanguageService {
    pub dm: DimensionManager,
    pub support_gid: GroupId,
    pub coding_gid: GroupId,
    pub calibration: CalibrationReport,
    pub mode: AgentMode,
    pub slo_config: SloConfig,
    latency_log: Vec<f64>,
    handoff_log: Vec<HandoffLogEntry>,
    context_snippets: Vec<String>,
}

impl LanguageService {
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new_default() -> Result<Self, String> {
        let (dm, support_gid, coding_gid, report) = build_language_demo_manager(0.2)?;
        Ok(Self {
            dm,
            support_gid,
            coding_gid,
            calibration: report,
            mode: AgentMode::MicroBrain,
            slo_config: SloConfig::default(),
            latency_log: Vec::new(),
            handoff_log: Vec::new(),
            context_snippets: Vec::new(),
        })
    }

    pub fn new_with_config(config: LanguageConfig) -> Result<Self, String> {
        let (dm, support_gid, coding_gid, report) = build_language_demo_manager_with_config(0.2, config)?;
        Ok(Self {
            dm,
            support_gid,
            coding_gid,
            calibration: report,
            mode: AgentMode::MicroBrain,
            slo_config: SloConfig::default(),
            latency_log: Vec::new(),
            handoff_log: Vec::new(),
            context_snippets: Vec::new(),
        })
    }

    // -----------------------------------------------------------------------
    // Core inference (unchanged API, now records latency)
    // -----------------------------------------------------------------------

    pub fn action(&mut self, text: &str) -> Result<ActionJson, String> {
        let start = portable_instant();
        let result = self.dm.route_text_to_action(text);
        self.record_latency(start);
        result
    }

    pub fn generation(&mut self, text: &str) -> Result<(ActionJson, GeneratedResponse), String> {
        let start = portable_instant();
        let action = self.dm.route_text_to_action(text)?;
        let resp = render_action_template(&action);
        self.record_latency(start);
        Ok((action, resp))
    }

    pub fn codegen(&mut self, text: &str) -> Result<(ActionJson, Option<CodeGeneration>), String> {
        let start = portable_instant();
        let action = self.dm.route_text_to_action(text)?;
        let code = generate_code_from_action(&action, text);
        self.record_latency(start);
        Ok((action, code))
    }

    pub fn load_gle_students_from_bytes(&mut self, data: &[&[u8]]) -> Result<usize, String> {
        self.dm.language_runtime.load_students_from_bytes(data)
    }

    // -----------------------------------------------------------------------
    // Brain export / import (full DimensionManager state)
    // -----------------------------------------------------------------------

    pub fn export_brain(&self) -> Result<Vec<u8>, String> {
        crate::systems::checkpoint::serialize_checkpoint_to_bytes(&self.dm)
    }

    pub fn load_brain(&mut self, data: &[u8]) -> Result<(), String> {
        let dm: DimensionManager =
            crate::systems::checkpoint::deserialize_checkpoint_from_bytes(data)?;
        self.dm = dm;
        let groups: Vec<_> = self.dm.main.group_order.clone();
        if let Some(&gid) = groups.first() {
            self.support_gid = gid;
        }
        if let Some(&gid) = groups.get(1) {
            self.coding_gid = gid;
        }
        Ok(())
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
        self.dm.episodic_summaries()
    }

    /// Route with auto-spawn detection. Returns (routing, Option<suggested_mirror_name>).
    pub fn route_with_spawn_check(
        &mut self,
        text: &str,
    ) -> Result<(LanguageRoutingDecision, Option<String>), String> {
        self.dm.route_text_with_spawn_check(text)
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
        let ckpt = self.dm.checkpoint_size_summary();
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
        let ckpt = self.dm.checkpoint_size_summary();

        let passed = slo.latency_ok && slo.checkpoint_ok;

        AcceptanceReport {
            understanding: UnderstandingMetrics {
                groups_count: ckpt.promoted_groups,
                routing_confidence_streak: self.dm.low_confidence_streak(),
                auto_spawn_k: self.dm.auto_spawn_k,
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
            });
        }
    }
    CalibrationDataset { samples }
}
