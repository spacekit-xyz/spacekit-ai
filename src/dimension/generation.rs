//! M4 constrained NLG: deterministic template rendering from action JSON.

use serde::{Deserialize, Serialize};

use super::action::{ActionJson, ActionPayload, ActionType};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedResponse {
    pub text: String,
    pub template_id: String,
    pub traceable: bool,
    /// Generation confidence (prototype cosine similarity). 1.0 = high confidence,
    /// 0.0 = no prototypes or no match. Always generates, never falls back.
    #[serde(default = "default_confidence")]
    pub confidence: f32,
}

fn default_confidence() -> f32 { 1.0 }

pub fn render_action_template(action: &ActionJson) -> GeneratedResponse {
    match (&action.action_type, &action.payload) {
        (ActionType::SupportTicket, Some(ActionPayload::SupportTicket { issue_type, priority })) => {
            let text = format!(
                "[SupportTicket] Triage started for issue_type={} priority={}. Next step: collect account identifier and recent error details.",
                issue_type, priority
            );
            GeneratedResponse {
                text,
                template_id: "m4_template_support_v1".to_string(),
                traceable: true,
                confidence: 1.0,
            }
        }
        (ActionType::CodingAssist, Some(ActionPayload::CodingAssist { task, language_hint })) => {
            let text = format!(
                "[CodingAssist] Task={} language_hint={}. Next step: provide minimal repro and failing logs/tests.",
                task, language_hint
            );
            GeneratedResponse {
                text,
                template_id: "m4_template_coding_v1".to_string(),
                traceable: true,
                confidence: 1.0,
            }
        }
        (ActionType::GeneralAssist, Some(ActionPayload::GeneralAssist { topic })) => {
            let text = format!(
                "[GeneralAssist] Topic={}. Next step: respond concisely with policy-safe guidance.",
                topic
            );
            GeneratedResponse {
                text,
                template_id: "m4_template_general_v1".to_string(),
                traceable: true,
                confidence: 1.0,
            }
        }
        (ActionType::ToolCall, Some(ActionPayload::ToolCall { tool_name, arguments })) => {
            let args_str = arguments.iter()
                .map(|(k, v)| format!("{}={}", k, v))
                .collect::<Vec<_>>()
                .join(", ");
            let text = format!(
                "[ToolCall] tool={} args={{{}}}. Awaiting tool execution result.",
                tool_name, args_str
            );
            GeneratedResponse {
                text,
                template_id: "m4_template_tool_call_v1".to_string(),
                traceable: true,
                confidence: 1.0,
            }
        }
        (ActionType::Fallback, Some(ActionPayload::Fallback { fallback_code })) => {
            let text = format!(
                "[Fallback] fallback_code={}. Clarify intent or hand off safely.",
                fallback_code
            );
            GeneratedResponse {
                text,
                template_id: "m4_template_fallback_v1".to_string(),
                traceable: true,
                confidence: 1.0,
            }
        }
        _ => GeneratedResponse {
            text: "[Fallback] fallback_code=SCHEMA_MISMATCH. Clarify intent or hand off safely.".to_string(),
            template_id: "m4_template_schema_mismatch_v1".to_string(),
            traceable: false,
            confidence: 1.0,
        },
    }
}

