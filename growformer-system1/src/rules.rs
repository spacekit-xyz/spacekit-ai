//! Rules: turn typed answers into one action.
//!
//! ```toml
//! default_action = "allow"
//!
//! [[rules]]
//! action = "block"
//! any = true                                   # any condition (default: all)
//! when = [{ id = "toxicity", min = 3 },        # score: expected value ≥ 3
//!         { id = "abuse_type", is = ["threat", "hate_speech"] }]
//!
//! [[rules]]
//! action = "human_review"
//! when = [{ id = "abuse_type", abstain = true }]   # the model wasn't sure
//! ```
//!
//! Conditions on one answer:
//! - `is = [...]`: the answer's value is one of these (choice options, or
//!   `"true"` / `"false"` for a noul).
//! - `min` / `max`: score → expected value; noul → P(true); choice →
//!   confidence of the chosen option.
//! - `min_confidence`: the answer's confidence is at least this.
//! - `abstain = true | false`: the answer did / didn't abstain.
//!
//! A condition on a question the request didn't ask is false, so rules for
//! other question groups never fire. For a question that was asked but
//! couldn't be answered, only `abstain = true` holds. The first matching rule
//! wins; otherwise `default_action` (if set).

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::decide::Answer;
use crate::schema::{QuestionKind, QuestionSet};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Condition {
    pub id: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub is: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_confidence: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub abstain: Option<bool>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Rule {
    pub action: String,
    #[serde(default)]
    pub when: Vec<Condition>,
    /// Match when any condition holds (default: all must hold).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub any: bool,
    /// Shown in the response, e.g. for logs and reviewers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

/// The action a request resolved to.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ActionResult {
    pub action: String,
    /// 1-based index of the matching rule; `None` for the default action.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rule: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

impl Rule {
    pub fn validate(&self, qs: &QuestionSet) -> Result<(), String> {
        if self.action.trim().is_empty() {
            return Err("empty action".into());
        }
        for c in &self.when {
            let q = qs
                .get(&c.id)
                .ok_or_else(|| format!("unknown question {:?}", c.id))?;
            if !c.is.is_empty() {
                let allowed: Vec<String> = match &q.kind {
                    QuestionKind::Choice { options } => options.clone(),
                    QuestionKind::Noul => vec!["true".into(), "false".into()],
                    QuestionKind::Score { .. } => {
                        return Err(format!(
                            "{}: `is` applies to choice / noul questions; use min / max for scores",
                            c.id
                        ))
                    }
                };
                for v in &c.is {
                    if !allowed.iter().any(|a| a == v) {
                        return Err(format!("{}: {v:?} is not one of {allowed:?}", c.id));
                    }
                }
            }
            if c.is.is_empty()
                && c.min.is_none()
                && c.max.is_none()
                && c.min_confidence.is_none()
                && c.abstain.is_none()
            {
                return Err(format!("{}: condition has no test", c.id));
            }
        }
        Ok(())
    }
}

fn holds(c: &Condition, answers: &[Answer]) -> bool {
    let Some(a) = answers.iter().find(|a| a.id == c.id) else {
        return false;
    };
    let answered = !a.value.is_null();
    if let Some(want) = c.abstain {
        if a.abstain != want {
            return false;
        }
    }
    if !answered {
        // Only an `abstain = true` test can hold for an unanswered question.
        return c.abstain == Some(true)
            && c.is.is_empty()
            && c.min.is_none()
            && c.max.is_none()
            && c.min_confidence.is_none();
    }
    if !c.is.is_empty() {
        let v = match &a.value {
            Value::String(s) => s.clone(),
            Value::Bool(b) => b.to_string(),
            other => other.to_string(),
        };
        if !c.is.contains(&v) {
            return false;
        }
    }
    if c.min.is_some() || c.max.is_some() {
        let x = match a.kind.as_str() {
            "score" => a.value.as_f64(),
            "noul" => a.probability.map(f64::from),
            _ => Some(a.confidence as f64),
        };
        let Some(x) = x else { return false };
        if c.min.map(|m| x < m - 1e-9).unwrap_or(false)
            || c.max.map(|m| x > m + 1e-9).unwrap_or(false)
        {
            return false;
        }
    }
    if let Some(mc) = c.min_confidence {
        if a.confidence < mc {
            return false;
        }
    }
    true
}

/// First matching rule's action, else the default action, else `None`.
pub fn apply(qs: &QuestionSet, answers: &[Answer]) -> Option<ActionResult> {
    for (i, r) in qs.rules.iter().enumerate() {
        let ok = if r.when.is_empty() {
            true
        } else if r.any {
            r.when.iter().any(|c| holds(c, answers))
        } else {
            r.when.iter().all(|c| holds(c, answers))
        };
        if ok {
            return Some(ActionResult {
                action: r.action.clone(),
                rule: Some(i + 1),
                reason: r.reason.clone(),
            });
        }
    }
    qs.default_action.as_ref().map(|a| ActionResult {
        action: a.clone(),
        rule: None,
        reason: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decide::make_answer;

    fn qs() -> QuestionSet {
        QuestionSet::parse(
            r#"
default_action = "allow"

[groups]
chat = ["toxicity", "abuse"]

[[questions]]
id = "toxicity"
text = "How toxic?"
type = "score"
min = 0
max = 4

[[questions]]
id = "abuse"
text = "Abuse type?"
type = "choice"
options = ["none", "insult", "threat"]

[[questions]]
id = "churn"
text = "Is the player about to quit?"
type = "noul"

[[rules]]
action = "block"
any = true
reason = "severe toxicity or a threat"
when = [{ id = "toxicity", min = 3 }, { id = "abuse", is = ["threat"] }]

[[rules]]
action = "retention_offer"
when = [{ id = "churn", min = 0.7 }]

[[rules]]
action = "human_review"
when = [{ id = "abuse", abstain = true }]
"#,
        )
        .unwrap()
    }

    fn ans(qs: &QuestionSet, id: &str, p: &[f32], min_conf: Option<f32>) -> Answer {
        make_answer(qs.get(id).unwrap(), p, "head", min_conf)
    }

    #[test]
    fn first_matching_rule_wins() {
        let q = qs();
        let calm = vec![
            ans(&q, "toxicity", &[0.9, 0.1, 0.0, 0.0, 0.0], None),
            ans(&q, "abuse", &[0.9, 0.05, 0.05], None),
            ans(&q, "churn", &[0.2, 0.8], None),
        ];
        assert_eq!(apply(&q, &calm).unwrap().action, "allow");
        let threat = vec![
            ans(&q, "toxicity", &[0.5, 0.5, 0.0, 0.0, 0.0], None),
            ans(&q, "abuse", &[0.1, 0.1, 0.8], None),
            ans(&q, "churn", &[0.9, 0.1], None),
        ];
        let a = apply(&q, &threat).unwrap();
        assert_eq!((a.action.as_str(), a.rule), ("block", Some(1)));
        let quitting = vec![ans(&q, "churn", &[0.75, 0.25], None)];
        assert_eq!(apply(&q, &quitting).unwrap().action, "retention_offer");
        // Unsure about abuse → review.
        let unsure = vec![ans(&q, "abuse", &[0.4, 0.35, 0.25], Some(0.6))];
        assert_eq!(apply(&q, &unsure).unwrap().action, "human_review");
        // Questions that weren't asked never trigger a rule.
        assert_eq!(apply(&q, &[]).unwrap().action, "allow");
    }

    #[test]
    fn rules_are_checked_against_the_schema() {
        let bad = r#"
[[questions]]
id = "t"
text = "?"
type = "score"
min = 0
max = 4
[[rules]]
action = "x"
when = [{ id = "t", is = ["3"] }]
"#;
        assert!(QuestionSet::parse(bad).unwrap_err().contains("min / max"));
        let unknown = bad.replace(r#"id = "t", is = ["3"]"#, r#"id = "nope", min = 1"#);
        assert!(QuestionSet::parse(&unknown)
            .unwrap_err()
            .contains("unknown question"));
        assert_eq!(qs().groups["chat"], vec!["toxicity", "abuse"]);
    }
}
