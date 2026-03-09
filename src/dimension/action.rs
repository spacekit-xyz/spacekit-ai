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
                | (ActionType::Fallback, Some(ActionPayload::Fallback { .. }))
        )
    }
}

pub fn action_from_routing(
    main: &MainDimension,
    routing: &LanguageRoutingDecision,
    text: &str,
) -> ActionJson {
    if routing.rejected_as_ood || routing.chosen_group_id.is_none() {
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
    let action_type = match task.as_deref().map(|s| s.to_ascii_lowercase()) {
        Some(t) if t.contains("support") => ActionType::SupportTicket,
        Some(t) if t.contains("coding") || t.contains("code") => ActionType::CodingAssist,
        _ => ActionType::GeneralAssist,
    };
    let lower = text.to_ascii_lowercase();
    let payload = match action_type {
        ActionType::SupportTicket => Some(ActionPayload::SupportTicket {
            issue_type: infer_support_issue_type(&lower),
            priority: infer_priority(&lower),
        }),
        ActionType::CodingAssist => Some(ActionPayload::CodingAssist {
            task: infer_coding_task(&lower),
            language_hint: infer_language_hint(&lower),
        }),
        ActionType::GeneralAssist => Some(ActionPayload::GeneralAssist {
            topic: infer_topic(&lower),
        }),
        ActionType::Fallback => Some(ActionPayload::Fallback {
            fallback_code: "UNSPECIFIED".to_string(),
        }),
    };
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
    } else if text.contains("optimize") || text.contains("performance") {
        "optimize".to_string()
    } else {
        "implement".to_string()
    }
}

fn infer_language_hint(text: &str) -> String {
    if text.contains("rust") {
        "rust".to_string()
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
