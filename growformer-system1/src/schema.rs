//! Typed questions: **Choice** (one of N options), **Score** (ordered integer
//! scale) and **Noul** (binary truth probability).
//!
//! Every question maps to a fixed, ordered list of *outcomes*; all backends
//! produce one probability per outcome, so answers are schema-safe by
//! construction (no free text is ever parsed).

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The three decision primitives.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum QuestionKind {
    /// Select one of the enumerated options.
    Choice { options: Vec<String> },
    /// Ordered integer scale `min..=max` (e.g. 0–5). Optional per-point labels
    /// (`labels[i]` describes `min + i`) are shown to the open-model backend.
    Score {
        min: i32,
        max: i32,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        labels: Vec<String>,
    },
    /// Binary truth: probability that the statement/question is true.
    Noul,
}

/// A typed question asked about a state.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Question {
    /// Stable identifier (`"urgency"`, `"route"`). Used as the training label key.
    pub id: String,
    /// The question as a human would ask it ("How urgent is this ticket?").
    pub text: String,
    #[serde(flatten)]
    pub kind: QuestionKind,
    /// Optional guidance for annotators / the open-model prompt.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Answers below this calibrated confidence are flagged `abstain: true`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_confidence: Option<f32>,
    /// Derive training labels from an existing JSONL field instead of a
    /// `labels` object (e.g. reuse a project's `semantic_intent`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<LabelSource>,
}

/// Where training labels for a question come from when a row has no
/// `labels.<id>` entry: `row[field]`, optionally mapped through `map`
/// (values missing from `map` are skipped).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LabelSource {
    pub field: String,
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub map: std::collections::BTreeMap<String, Value>,
}

/// `"choice" | "score" | "noul"`.
pub fn kind_name(kind: &QuestionKind) -> &'static str {
    match kind {
        QuestionKind::Choice { .. } => "choice",
        QuestionKind::Score { .. } => "score",
        QuestionKind::Noul => "noul",
    }
}

impl Question {
    pub fn choice(id: &str, text: &str, options: &[&str]) -> Self {
        Self {
            id: id.into(),
            text: text.into(),
            kind: QuestionKind::Choice {
                options: options.iter().map(|s| s.to_string()).collect(),
            },
            description: None,
            min_confidence: None,
            source: None,
        }
    }

    pub fn score(id: &str, text: &str, min: i32, max: i32) -> Self {
        Self {
            id: id.into(),
            text: text.into(),
            kind: QuestionKind::Score {
                min,
                max,
                labels: Vec::new(),
            },
            description: None,
            min_confidence: None,
            source: None,
        }
    }

    pub fn noul(id: &str, text: &str) -> Self {
        Self {
            id: id.into(),
            text: text.into(),
            kind: QuestionKind::Noul,
            description: None,
            min_confidence: None,
            source: None,
        }
    }

    /// Ordered outcome labels. Choice → options; Score → `"min".."max"`;
    /// Noul → `["true", "false"]`.
    pub fn outcomes(&self) -> Vec<String> {
        match &self.kind {
            QuestionKind::Choice { options } => options.clone(),
            QuestionKind::Score { min, max, .. } => (*min..=*max).map(|v| v.to_string()).collect(),
            QuestionKind::Noul => vec!["true".into(), "false".into()],
        }
    }

    pub fn n_outcomes(&self) -> usize {
        match &self.kind {
            QuestionKind::Choice { options } => options.len(),
            QuestionKind::Score { min, max, .. } => (max - min + 1).max(0) as usize,
            QuestionKind::Noul => 2,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.id.trim().is_empty() {
            return Err("question id must not be empty".into());
        }
        match &self.kind {
            QuestionKind::Choice { options } => {
                if options.len() < 2 {
                    return Err(format!("{}: choice needs at least 2 options", self.id));
                }
                let mut seen = std::collections::HashSet::new();
                for o in options {
                    if !seen.insert(o.to_lowercase()) {
                        return Err(format!("{}: duplicate option {o:?}", self.id));
                    }
                }
            }
            QuestionKind::Score { min, max, labels } => {
                if max <= min {
                    return Err(format!("{}: score needs max > min", self.id));
                }
                if max - min > 100 {
                    return Err(format!("{}: score range limited to 101 points", self.id));
                }
                if !labels.is_empty() && labels.len() != (max - min + 1) as usize {
                    return Err(format!(
                        "{}: score labels must have one entry per point ({})",
                        self.id,
                        max - min + 1
                    ));
                }
            }
            QuestionKind::Noul => {}
        }
        Ok(())
    }

    /// Map a labelled value to its outcome index. Accepts option strings
    /// (case-insensitive) for Choice, numbers or numeric strings for Score, and
    /// booleans / `yes|no|true|false|1|0` for Noul.
    pub fn outcome_index(&self, value: &Value) -> Option<usize> {
        match &self.kind {
            QuestionKind::Choice { options } => {
                let s = value.as_str()?.trim().to_lowercase();
                options.iter().position(|o| o.to_lowercase() == s)
            }
            QuestionKind::Score { min, max, .. } => {
                let v = match value {
                    Value::Number(n) => n.as_f64()?,
                    Value::String(s) => s.trim().parse::<f64>().ok()?,
                    _ => return None,
                };
                let r = v.round() as i64;
                if r < *min as i64 || r > *max as i64 {
                    return None;
                }
                Some((r - *min as i64) as usize)
            }
            QuestionKind::Noul => {
                let t = match value {
                    Value::Bool(b) => *b,
                    Value::Number(n) => n.as_f64()? >= 0.5,
                    Value::String(s) => match s.trim().to_lowercase().as_str() {
                        "true" | "yes" | "y" | "1" => true,
                        "false" | "no" | "n" | "0" => false,
                        _ => return None,
                    },
                    _ => return None,
                };
                Some(if t { 0 } else { 1 })
            }
        }
    }
}

/// A named set of questions (the brain's schema).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct QuestionSet {
    #[serde(default)]
    pub questions: Vec<Question>,
}

impl QuestionSet {
    pub fn get(&self, id: &str) -> Option<&Question> {
        self.questions.iter().find(|q| q.id == id)
    }

    pub fn validate(&self) -> Result<(), String> {
        let mut ids = std::collections::HashSet::new();
        for q in &self.questions {
            q.validate()?;
            if !ids.insert(q.id.as_str()) {
                return Err(format!("duplicate question id {:?}", q.id));
            }
        }
        Ok(())
    }

    /// Parse TOML (`[[questions]] id=… type="choice" options=[…]`) or JSON.
    pub fn parse(text: &str) -> Result<Self, String> {
        let t = text.trim_start();
        // JSON object `{"questions": [...]}` or array `[{...}]`; otherwise TOML
        // (note TOML's `[[questions]]` also starts with '[').
        let set: QuestionSet = if let Ok(set) = serde_json::from_str::<QuestionSet>(t) {
            set
        } else if let Ok(qs) = serde_json::from_str::<Vec<Question>>(t) {
            QuestionSet { questions: qs }
        } else if t.starts_with('{') {
            serde_json::from_str(t).map_err(|e| e.to_string())?
        } else {
            toml::from_str(t).map_err(|e| e.to_string())?
        };
        set.validate()?;
        Ok(set)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parse_toml_schema() {
        let s = QuestionSet::parse(
            r#"
[[questions]]
id = "route"
text = "Which team should handle this?"
type = "choice"
options = ["Billing", "Tech Support", "Cancellation"]

[[questions]]
id = "urgency"
text = "How urgent is this?"
type = "score"
min = 0
max = 4

[[questions]]
id = "is_spam"
text = "Is this spam?"
type = "noul"
min_confidence = 0.8
"#,
        )
        .unwrap();
        assert_eq!(s.questions.len(), 3);
        assert_eq!(
            s.get("urgency").unwrap().outcomes(),
            vec!["0", "1", "2", "3", "4"]
        );
        assert_eq!(s.get("is_spam").unwrap().min_confidence, Some(0.8));
    }

    #[test]
    fn outcome_mapping() {
        let c = Question::choice("r", "?", &["Billing", "Tech"]);
        assert_eq!(c.outcome_index(&json!("tech")), Some(1));
        assert_eq!(c.outcome_index(&json!("other")), None);
        let s = Question::score("u", "?", 1, 5);
        assert_eq!(s.outcome_index(&json!(3)), Some(2));
        assert_eq!(s.outcome_index(&json!("5")), Some(4));
        assert_eq!(s.outcome_index(&json!(9)), None);
        let n = Question::noul("n", "?");
        assert_eq!(n.outcome_index(&json!(true)), Some(0));
        assert_eq!(n.outcome_index(&json!("no")), Some(1));
    }

    #[test]
    fn rejects_bad_schemas() {
        assert!(Question::choice("x", "?", &["a"]).validate().is_err());
        assert!(Question::score("x", "?", 3, 3).validate().is_err());
        assert!(QuestionSet::parse(
            r#"[{"id":"a","text":"?","type":"noul"},{"id":"a","text":"?","type":"noul"}]"#
        )
        .is_err());
    }
}
