//! Deterministic action schema for M3 (intent -> action JSON).

use serde::{Deserialize, Serialize};

use crate::types::GroupId;

use super::language::LanguageRoutingDecision;
use super::main_dim::MainDimension;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ActionType {
    SupportTicket,
    CodingAssist,
    GeneralAssist,
    ToolCall,
    Fallback,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionJson {
    pub action_type: ActionType,
    pub target_group_id: Option<GroupId>,
    pub group_task_name: Option<String>,
    pub confidence: f32,
    pub margin: f32,
    pub reason: String,
    pub payload: Option<ActionPayload>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ActionPayload {
    SupportTicket {
        issue_type: String,
        priority: String,
    },
    CodingAssist {
        task: String,
        language_hint: String,
    },
    GeneralAssist {
        topic: String,
    },
    ToolCall {
        tool_name: String,
        arguments: std::collections::HashMap<String, String>,
    },
    Fallback {
        fallback_code: String,
    },
}

impl ActionJson {
    /// Ensures payload shape matches action_type.
    pub fn is_valid(&self) -> bool {
        matches!(
            (&self.action_type, &self.payload),
            (ActionType::SupportTicket, Some(ActionPayload::SupportTicket { .. }))
                | (ActionType::CodingAssist, Some(ActionPayload::CodingAssist { .. }))
                | (ActionType::GeneralAssist, Some(ActionPayload::GeneralAssist { .. }))
                | (ActionType::ToolCall, Some(ActionPayload::ToolCall { .. }))
                | (ActionType::Fallback, Some(ActionPayload::Fallback { .. }))
        )
    }
}

pub fn action_from_routing(
    main: &MainDimension,
    routing: &LanguageRoutingDecision,
    text: &str,
) -> ActionJson {
    let lower = text.to_ascii_lowercase();
    let keyword_hint = infer_action_type_from_text(&lower);
    if routing.rejected_as_ood || routing.chosen_group_id.is_none() {
        if let Some(hint) = keyword_hint {
            if hint != ActionType::GeneralAssist {
                return ActionJson {
                    action_type: hint.clone(),
                    target_group_id: None,
                    group_task_name: None,
                    confidence: routing.confidence,
                    margin: routing.margin,
                    reason: "keyword_override_after_reject".to_string(),
                    payload: payload_for_action_type(hint, &lower),
                };
            }
        }
        return ActionJson {
            action_type: ActionType::Fallback,
            target_group_id: None,
            group_task_name: None,
            confidence: routing.confidence,
            margin: routing.margin,
            reason: "ood_or_ambiguous".to_string(),
            payload: Some(ActionPayload::Fallback {
                fallback_code: "OOD_OR_AMBIGUOUS".to_string(),
            }),
        };
    }
    let gid = routing.chosen_group_id.unwrap_or_default();
    let task = main.groups.get(&gid).map(|g| g.task_name.clone());
    let mut action_type = match task.as_deref().map(|s| s.to_ascii_lowercase()) {
        Some(t) if t.contains("support") => ActionType::SupportTicket,
        Some(t) if t.contains("coding") || t.contains("code") => ActionType::CodingAssist,
        _ => ActionType::GeneralAssist,
    };
    if let Some(hint) = keyword_hint {
        if hint != ActionType::GeneralAssist {
            action_type = hint;
        }
    }
    let payload = payload_for_action_type(action_type.clone(), &lower);
    ActionJson {
        action_type,
        target_group_id: Some(gid),
        group_task_name: task,
        confidence: routing.confidence,
        margin: routing.margin,
        reason: "routed".to_string(),
        payload,
    }
}

fn payload_for_action_type(action_type: ActionType, lower: &str) -> Option<ActionPayload> {
    match action_type {
        ActionType::SupportTicket => Some(ActionPayload::SupportTicket {
            issue_type: infer_support_issue_type(lower),
            priority: infer_priority(lower),
        }),
        ActionType::CodingAssist => Some(ActionPayload::CodingAssist {
            task: infer_coding_task(lower),
            language_hint: infer_language_hint(lower),
        }),
        ActionType::GeneralAssist => Some(ActionPayload::GeneralAssist {
            topic: infer_topic(lower),
        }),
        ActionType::ToolCall => Some(ActionPayload::ToolCall {
            tool_name: infer_tool_name(lower),
            arguments: std::collections::HashMap::new(),
        }),
        ActionType::Fallback => Some(ActionPayload::Fallback {
            fallback_code: "UNSPECIFIED".to_string(),
        }),
    }
}

fn infer_action_type_from_text(text: &str) -> Option<ActionType> {
    let tool_terms = [
        "calculate",
        "compute",
        "search for",
        "look up",
        "find information",
        "run this",
        "execute this",
        "eval this",
        "run the code",
        "execute the code",
        "read file",
        "show file",
        "read the file",
        "show me the file",
    ];
    let tool_hits = tool_terms.iter().filter(|t| text.contains(**t)).count();
    if tool_hits > 0 {
        return Some(ActionType::ToolCall);
    }

    let support_terms = [
        "account",
        "login",
        "password",
        "billing",
        "refund",
        "subscription",
        "customer",
        "help desk",
        "ticket",
        "unlock",
        "sign in",
        "access",
    ];
    let coding_terms = [
        "code",
        "debug",
        "bug",
        "stack trace",
        "parser",
        "rust",
        "sql",
        "python",
        "c++",
        "segmentation fault",
        "implement",
        "optimize",
        "refactor",
        "unit test",
        "test",
        "pytest",
        "jest",
        "javascript",
        "typescript",
        "node",
        "node.js",
        "dom",
        "compile",
        "middleware",
        "server",
    ];
    let support_hits = support_terms.iter().filter(|t| text.contains(**t)).count();
    let coding_hits = coding_terms.iter().filter(|t| text.contains(**t)).count();
    if support_hits == 0 && coding_hits == 0 {
        None
    } else if coding_hits > support_hits {
        Some(ActionType::CodingAssist)
    } else {
        Some(ActionType::SupportTicket)
    }
}

fn infer_tool_name(text: &str) -> String {
    if text.contains("calculate") || text.contains("compute") {
        "calculator".to_string()
    } else if text.contains("search for") || text.contains("look up") || text.contains("find information") {
        "web_search".to_string()
    } else if text.contains("run this") || text.contains("execute this") || text.contains("eval this")
        || text.contains("run the code") || text.contains("execute the code") {
        "code_runner".to_string()
    } else if text.contains("read file") || text.contains("show file")
        || text.contains("read the file") || text.contains("show me the file") {
        "file_reader".to_string()
    } else {
        "unknown".to_string()
    }
}

fn infer_support_issue_type(text: &str) -> String {
    if text.contains("password") || text.contains("login") || text.contains("account") {
        "account_access".to_string()
    } else if text.contains("billing") || text.contains("refund") || text.contains("subscription") {
        "billing".to_string()
    } else {
        "general_support".to_string()
    }
}

fn infer_priority(text: &str) -> String {
    if text.contains("urgent") || text.contains("asap") || text.contains("immediately") {
        "high".to_string()
    } else {
        "normal".to_string()
    }
}

fn infer_coding_task(text: &str) -> String {
    if text.contains("debug") || text.contains("error") || text.contains("fault") {
        "debug".to_string()
    } else if text.contains("test") || text.contains("pytest") || text.contains("jest") {
        "test".to_string()
    } else if text.contains("refactor") {
        "refactor".to_string()
    } else if text.contains("optimize") || text.contains("performance") {
        "optimize".to_string()
    } else {
        "implement".to_string()
    }
}

fn infer_language_hint(text: &str) -> String {
    let tokens: Vec<&str> = text
        .split(|c: char| !c.is_ascii_alphanumeric() && c != '+')
        .filter(|t| !t.is_empty())
        .collect();
    let has_token = |needle: &str| tokens.iter().any(|t| t.eq_ignore_ascii_case(needle));
    if text.contains("rust") {
        "rust".to_string()
    } else if text.contains("typescript") || has_token("ts") || has_token("tsx") {
        "typescript".to_string()
    } else if text.contains("javascript")
        || text.contains("node.js")
        || text.contains("nodejs")
        || has_token("js")
        || has_token("jsx")
        || has_token("jest")
        || has_token("npm")
    {
        "javascript".to_string()
    } else if text.contains("sql") {
        "sql".to_string()
    } else if text.contains("python") {
        "python".to_string()
    } else if text.contains("c++") || text.contains(" c ") {
        "c".to_string()
    } else {
        "unknown".to_string()
    }
}

fn infer_topic(text: &str) -> String {
    if text.contains("policy") || text.contains("safety") {
        "policy".to_string()
    } else if text.contains("question") || text.contains("explain") {
        "qa".to_string()
    } else {
        "general".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dimension::main_dim::MainDimension;
    use crate::dimension::embedding::GroupEmbedding;
    use crate::environment::NeuralEnvironment;
    use crate::types::EnvironmentConfig;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn tool_call_detected_from_text() {
        let result = infer_action_type_from_text("calculate 347 * 892");
        assert_eq!(result, Some(ActionType::ToolCall));

        let result = infer_action_type_from_text("search for rust async patterns");
        assert_eq!(result, Some(ActionType::ToolCall));

        let result = infer_action_type_from_text("run this python script please");
        assert_eq!(result, Some(ActionType::ToolCall));

        let result = infer_action_type_from_text("read file src/main.rs");
        assert_eq!(result, Some(ActionType::ToolCall));

        // "observer" contains "server" (a coding term), so this routes to CodingAssist
        let result = infer_action_type_from_text("explain the observer pattern");
        assert_eq!(result, Some(ActionType::CodingAssist));

        let result = infer_action_type_from_text("what is the meaning of life");
        assert_eq!(result, None);
    }

    #[test]
    fn tool_call_action_is_valid() {
        let action = ActionJson {
            action_type: ActionType::ToolCall,
            target_group_id: None,
            group_task_name: None,
            confidence: 1.0,
            margin: 1.0,
            reason: "tool_match".to_string(),
            payload: Some(ActionPayload::ToolCall {
                tool_name: "calculator".to_string(),
                arguments: std::collections::HashMap::from([
                    ("expression".to_string(), "2+2".to_string()),
                ]),
            }),
        };
        assert!(action.is_valid());
        assert_eq!(action.action_type, ActionType::ToolCall);
    }

    #[test]
    fn tool_call_payload_from_type() {
        let payload = payload_for_action_type(ActionType::ToolCall, "calculate 100 / 5");
        match payload {
            Some(ActionPayload::ToolCall { tool_name, .. }) => {
                assert_eq!(tool_name, "calculator");
            }
            _ => panic!("expected ToolCall payload"),
        }
    }

    #[test]
    fn ood_with_tool_keywords_routes_to_tool() {
        let main = MainDimension::new();
        let routing = LanguageRoutingDecision {
            chosen_group_id: None,
            best_similarity: 0.0,
            second_similarity: 0.0,
            margin: 0.0,
            confidence: 0.2,
            rejected_as_ood: true,
        };
        let a = action_from_routing(&main, &routing, "calculate 2 + 2");
        assert_eq!(a.action_type, ActionType::ToolCall);
        assert!(a.is_valid());
    }

    #[test]
    fn ood_routing_yields_fallback() {
        let main = MainDimension::new();
        let routing = LanguageRoutingDecision {
            chosen_group_id: None,
            best_similarity: 0.0,
            second_similarity: 0.0,
            margin: 0.0,
            confidence: 0.2,
            rejected_as_ood: true,
        };
        let a = action_from_routing(&main, &routing, "what is weather");
        assert_eq!(a.action_type, ActionType::Fallback);
        assert!(a.is_valid());
    }

    #[test]
    fn support_task_maps_to_support_action() {
        let mut main = MainDimension::new();
        let mut rng = StdRng::seed_from_u64(7);
        let mut env = NeuralEnvironment::new(EnvironmentConfig::default());
        env.build_layers(&[2, 4, 1], &mut rng);
        env.freeze_all();
        let emb = GroupEmbedding {
            group_id: 0,
            vector: vec![0.0; 4],
            task_name: "support".into(),
            accuracy: 1.0,
            intrinsic_dim: None,
            description: None,
            metatags: vec![],
            tag_vector: vec![],
            language_vector: vec![],
        };
        main.register_group(0, "support".into(), env, emb, 1.0, 0);
        let routing = LanguageRoutingDecision {
            chosen_group_id: Some(0),
            best_similarity: 0.9,
            second_similarity: 0.1,
            margin: 0.8,
            confidence: 0.9,
            rejected_as_ood: false,
        };
        let a = action_from_routing(&main, &routing, "help with password reset");
        assert_eq!(a.action_type, ActionType::SupportTicket);
        assert!(a.is_valid());
    }
}
