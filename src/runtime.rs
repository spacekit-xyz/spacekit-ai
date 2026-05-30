//! Portable inference runtime — works on native and wasm32.
//!
//! Wraps [`LanguageService`] with a platform-agnostic API: load a brain from
//! bytes, prompt, converse, codegen. No filesystem, no stdin, no rayon.

use serde::{Deserialize, Serialize};

use crate::dimension::action::ActionJson;
use crate::dimension::generation::GeneratedResponse;
use crate::dimension::tool::{ToolCallInfo, ToolResult};
use crate::service::AgentRuntimeState;
use crate::dimension::LanguageConfig;
use crate::service::{AgentMode, LanguageService, OceanProfile};

// ───────────────────────────────────────────────────────────────────────────
// Response types (owned, serialisable, no borrow lifetimes)
// ───────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeResponse {
    pub text: String,
    pub confidence: f32,
    pub template_id: String,
    pub action_type: String,
    pub action_confidence: f32,
    pub target_group: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeOutput {
    pub language: String,
    pub kind: String,
    pub code: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrainInfo {
    pub agent_name: String,
    pub agent_creator: String,
    pub num_groups: usize,
    pub has_router: bool,
    pub has_classifier: bool,
    pub gen_envs: usize,
    pub code_envs: usize,
    /// From [`crate::brain::BrainPackageHeader::inference_profile`] when present.
    #[serde(default)]
    pub inference_profile: Option<String>,
    /// True when a [`crate::brain::BrainPackage::plugins_blob`] was present and parsed.
    #[serde(default)]
    pub has_inference_plugins: bool,
}

// ───────────────────────────────────────────────────────────────────────────
// Runtime
// ───────────────────────────────────────────────────────────────────────────

pub struct Runtime {
    pub svc: LanguageService,
}

impl Runtime {
    /// Bootstrap from serialised brain bytes (`.bin` file contents).
    pub fn from_brain_bytes(data: &[u8]) -> Result<Self, String> {
        let config = LanguageConfig::default();
        let mut svc =
            LanguageService::new_with_config(config).map_err(|e| format!("init: {}", e))?;
        svc.load_brain(data)?;
        Ok(Self { svc })
    }

    /// Bootstrap with a custom [`LanguageConfig`], then load brain bytes.
    pub fn from_brain_bytes_with_config(
        data: &[u8],
        config: LanguageConfig,
    ) -> Result<Self, String> {
        let mut svc =
            LanguageService::new_with_config(config).map_err(|e| format!("init: {}", e))?;
        svc.load_brain(data)?;
        Ok(Self { svc })
    }

    /// Create an empty runtime (no brain loaded yet). Call [`Self::load_brain`] later.
    pub fn empty() -> Result<Self, String> {
        let config = LanguageConfig::default();
        let svc =
            LanguageService::new_with_config(config).map_err(|e| format!("init: {}", e))?;
        Ok(Self { svc })
    }

    /// Hot-swap the brain without rebuilding the runtime.
    pub fn load_brain(&mut self, data: &[u8]) -> Result<(), String> {
        self.svc.load_brain(data)
    }

    /// Export current brain state as bytes (for caching / saving).
    pub fn export_brain(&mut self) -> Result<Vec<u8>, String> {
        self.svc.export_brain()
    }

    // ─── Metadata ────────────────────────────────────────────────────────

    pub fn brain_info(&self) -> BrainInfo {
        let dm = self.svc.active_dm();
        BrainInfo {
            agent_name: self.svc.agent_name.clone(),
            agent_creator: self.svc.agent_creator.clone(),
            num_groups: dm.main.group_order.len(),
            has_router: dm.observer.learned_router.is_some(),
            has_classifier: dm.action_classifier.is_some(),
            gen_envs: dm.group_gen_envs.len(),
            code_envs: dm.group_code_envs.len(),
            inference_profile: self
                .svc
                .brain_package_header
                .as_ref()
                .and_then(|h| h.inference_profile.clone()),
            has_inference_plugins: self.svc.brain_plugins_manifest.is_some(),
        }
    }

    // ─── Inference ───────────────────────────────────────────────────────

    /// Single-shot prompt: routes, generates text, optionally generates code.
    pub fn prompt(&mut self, text: &str) -> Result<RuntimeResponse, String> {
        let (action, resp) = self.svc.generation(text)?;
        Ok(pack_response(&action, &resp))
    }

    /// Conversational turn: uses multi-turn context, personality, anaphora
    /// resolution, topic-shift detection.
    pub fn converse(&mut self, text: &str) -> Result<RuntimeResponse, String> {
        let (action, resp) = self.svc.converse(text)?;
        let validated = self.validate_response(resp);
        Ok(pack_response(&action, &validated))
    }

    /// Code generation for the given prompt.
    pub fn codegen(&mut self, text: &str) -> Result<Option<CodeOutput>, String> {
        let (_action, code) = self.svc.codegen(text)?;
        Ok(code.map(|c| CodeOutput {
            language: c.language,
            kind: c.kind,
            code: c.code,
        }))
    }

    /// Paramecium lattice-only inference (zero-synapse, wave-based).
    pub fn paramecium(&mut self, text: &str) -> Result<RuntimeResponse, String> {
        let (action, resp) = self.svc.paramecium_respond(text)?;
        Ok(pack_response(&action, &resp))
    }

    // ─── Tool dispatch ───────────────────────────────────────────────────

    /// Check whether the prompt triggers a registered tool. Returns `None`
    /// when no tool matches — the caller decides how to execute it.
    pub fn try_tool_call(&self, text: &str) -> Option<ToolCallInfo> {
        self.svc.try_tool_call(text)
    }

    /// After executing a tool, feed the result back to get a composed response.
    pub fn respond_with_tool_result(
        &mut self,
        original_text: &str,
        result: &ToolResult,
    ) -> Result<RuntimeResponse, String> {
        let (action, resp) = self.svc.generation_with_tool_result(original_text, result)?;
        Ok(pack_response(&action, &resp))
    }

    // ─── Conversation management ─────────────────────────────────────────

    pub fn reset_conversation(&mut self) {
        self.svc.reset_conversation();
    }

    pub fn turn_count(&self) -> usize {
        self.svc.conversation.turn_count()
    }

    // ─── Personality ─────────────────────────────────────────────────────

    pub fn set_personality(&mut self, profile: OceanProfile) {
        self.svc.personality = profile;
    }

    pub fn personality(&self) -> &OceanProfile {
        &self.svc.personality
    }

    /// Enable stochastic top-k retrieval on all generation environments.
    /// Temperature controls sampling sharpness (0.85 typical for chat).
    pub fn enable_stochastic_retrieval(&mut self, temperature: f32) {
        self.svc.enable_stochastic_retrieval(temperature);
    }

    pub fn apply_loaded_generation_config(&mut self) {
        self.svc.apply_loaded_generation_config();
    }

    // TODO: petstate needs to be generalized petstate is too specific
    /// Set agent state from a JSON string. The state modulates the conversation
    /// context prefix injected before generation (arbitrary dimensions + profile).
    pub fn set_agent_state_from_json(&mut self, json_str: &str) -> Result<(), String> {
        let state: AgentRuntimeState = serde_json::from_str(json_str)
            .map_err(|e| format!("parse agent state JSON: {}", e))?;
        self.svc.agent_state = Some(state);
        Ok(())
    }

    /// Validate a generated response against the `[response_shaping]` rules.
    /// Returns the original response if validation passes or is disabled,
    /// otherwise returns a truncated/cleaned version.
    fn validate_response(
        &self,
        mut resp: crate::dimension::GeneratedResponse,
    ) -> crate::dimension::GeneratedResponse {
        let loaded = crate::inference::inference_toml::inference_toml_loaded();
        let shaping = loaded.response_shaping();
        let validation = loaded.validation_config();

        if !validation.enabled {
            return resp;
        }

        // Length bounds
        if resp.text.len() > shaping.max_response_chars {
            let truncated = &resp.text[..shaping.max_response_chars];
            if let Some(last_period) = truncated.rfind(". ") {
                resp.text = truncated[..=last_period].to_string();
            } else if let Some(last_space) = truncated.rfind(' ') {
                resp.text = truncated[..last_space].to_string();
            } else {
                resp.text = truncated.to_string();
            }
        }

        // Forbidden phrases
        let text_lower = resp.text.to_ascii_lowercase();
        for phrase in &shaping.forbidden_phrases {
            if text_lower.contains(&phrase.to_ascii_lowercase()) {
                resp.confidence *= 0.3;
                crate::infer_trace!("  [validation] forbidden phrase hit: {:?}", phrase);
                break;
            }
        }

        // Forbid asterisks (narrative action markers)
        if shaping.forbid_asterisks && resp.text.contains('*') {
            resp.text = resp.text.replace('*', "");
        }

        resp
    }

    pub fn set_identity(&mut self, name: &str, creator: &str) {
        self.svc.set_identity(name, creator);
    }

    // ─── Mode ────────────────────────────────────────────────────────────

    pub fn set_mode(&mut self, mode: AgentMode, confidence: f32, reason: &str) {
        self.svc.set_mode(mode, confidence, reason);
    }

    pub fn active_mode(&self) -> AgentMode {
        self.svc.active_mode()
    }
}

// ─── helpers ─────────────────────────────────────────────────────────────

fn pack_response(action: &ActionJson, resp: &GeneratedResponse) -> RuntimeResponse {
    RuntimeResponse {
        text: resp.text.clone(),
        confidence: resp.confidence,
        template_id: resp.template_id.clone(),
        action_type: format!("{:?}", action.action_type),
        action_confidence: action.confidence,
        target_group: action.target_group_id,
    }
}
