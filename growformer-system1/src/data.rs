//! Labelled rows for training / evaluation / distillation.
//!
//! JSONL, one object per line:
//! ```json
//! {"state": "...or an object...", "labels": {"route": "Billing", "urgency": 3, "is_spam": false}}
//! {"state": {...}, "soft": {"route": {"Billing": 0.7, "Tech Support": 0.3}}}
//! {"text": "...", "semantic_intent": "negative_mild"}      // via Question.source
//! ```
//! `state` falls back to `text`. `soft` (distilled probabilities) wins over `labels`.

use serde_json::Value;

use crate::schema::{Question, QuestionSet};
use crate::state::state_to_text;

/// One parsed row: state text + per-question target distribution.
#[derive(Clone, Debug)]
pub struct LabeledRow {
    pub state: Value,
    pub text: String,
    /// `(question index, target distribution over outcomes, gold argmax)`
    pub targets: Vec<(usize, Vec<f32>, usize)>,
}

fn target_for(q: &Question, row: &Value) -> Option<(Vec<f32>, usize)> {
    let k = q.n_outcomes();
    // 1. soft distribution
    if let Some(soft) = row
        .get("soft")
        .and_then(|s| s.get(&q.id))
        .and_then(|s| s.as_object())
    {
        let outcomes = q.outcomes();
        let mut t = vec![0.0f32; k];
        for (name, p) in soft {
            let idx = outcomes
                .iter()
                .position(|o| o.eq_ignore_ascii_case(name))
                .or_else(|| q.outcome_index(&Value::String(name.clone())));
            if let (Some(i), Some(p)) = (idx, p.as_f64()) {
                t[i] += p.max(0.0) as f32;
            }
        }
        let s: f32 = t.iter().sum();
        if s > 0.0 {
            t.iter_mut().for_each(|v| *v /= s);
            let g = crate::calibration::argmax(&t).0;
            return Some((t, g));
        }
    }
    // 2. explicit label
    let mut value = row.get("labels").and_then(|l| l.get(&q.id)).cloned();
    // 3. derived from a field
    if value.is_none() {
        if let Some(src) = &q.source {
            if let Some(raw) = row.get(&src.field) {
                value = if src.map.is_empty() {
                    Some(raw.clone())
                } else {
                    raw.as_str().and_then(|s| src.map.get(s)).cloned()
                };
            }
        }
    }
    let idx = q.outcome_index(&value?)?;
    let mut t = vec![0.0f32; k];
    t[idx] = 1.0;
    Some((t, idx))
}

/// Parse JSONL text into rows that carry at least one usable label.
/// Returns `(rows, skipped_lines)`.
pub fn parse_labeled_jsonl(
    text: &str,
    questions: &QuestionSet,
) -> Result<(Vec<LabeledRow>, usize), String> {
    let mut rows = Vec::new();
    let mut skipped = 0usize;
    for (n, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let v: Value = serde_json::from_str(line).map_err(|e| format!("line {}: {e}", n + 1))?;
        let state = v
            .get("state")
            .or_else(|| v.get("text"))
            .cloned()
            .unwrap_or(Value::Null);
        let text = state_to_text(&state);
        if text.trim().is_empty() {
            skipped += 1;
            continue;
        }
        let targets: Vec<_> = questions
            .questions
            .iter()
            .enumerate()
            .filter_map(|(qi, q)| target_for(q, &v).map(|(t, g)| (qi, t, g)))
            .collect();
        if targets.is_empty() {
            skipped += 1;
            continue;
        }
        rows.push(LabeledRow {
            state,
            text,
            targets,
        });
    }
    Ok((rows, skipped))
}

/// Deterministic split: every `1/fraction`-th row (by a hash of its text) goes to held-out.
pub fn split_holdout(
    rows: &[LabeledRow],
    holdout_frac: f64,
    seed: u64,
) -> (Vec<usize>, Vec<usize>) {
    let mut train = Vec::new();
    let mut test = Vec::new();
    for (i, r) in rows.iter().enumerate() {
        let h = fnv1a(r.text.as_bytes()) ^ seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let u = (splitmix(h) >> 11) as f64 / (1u64 << 53) as f64;
        if u < holdout_frac {
            test.push(i);
        } else {
            train.push(i);
        }
    }
    (train, test)
}

fn fnv1a(b: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &x in b {
        h ^= x as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

fn splitmix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_labels_soft_and_source() {
        let qs = QuestionSet::parse(
            r#"
[[questions]]
id = "sent"
text = "Sentiment?"
type = "choice"
options = ["positive", "negative"]
source = { field = "semantic_intent", map = { positive_mild = "positive", negative_strong = "negative" } }

[[questions]]
id = "spam"
text = "Spam?"
type = "noul"
"#,
        )
        .unwrap();
        let jsonl = r#"{"text":"great","semantic_intent":"positive_mild"}
{"state":{"body":"buy now"},"labels":{"spam":true}}
{"state":"meh","soft":{"sent":{"positive":0.2,"negative":0.6}}}
{"text":"skip me","semantic_intent":"unmapped"}"#;
        let (rows, skipped) = parse_labeled_jsonl(jsonl, &qs).unwrap();
        assert_eq!(rows.len(), 3);
        assert_eq!(skipped, 1);
        assert_eq!(rows[0].targets[0].2, 0);
        assert_eq!(rows[1].targets[0].0, 1);
        assert_eq!(rows[1].text, "body: buy now");
        let soft = &rows[2].targets[0].1;
        assert!((soft[1] - 0.75).abs() < 1e-6);
    }
}
