//! Turn a request `state` (text, JSON object or array) into the text both
//! backends read. Objects become `key.path: value` lines so field names carry
//! meaning ("priority: high" ≠ "subject: high").

use serde_json::Value;

/// Flatten `state` to text. Plain strings pass through unchanged.
pub fn state_to_text(state: &Value) -> String {
    match state {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        other => {
            let mut lines = Vec::new();
            flatten("", other, &mut lines);
            lines.join("\n")
        }
    }
}

fn flatten(prefix: &str, v: &Value, out: &mut Vec<String>) {
    match v {
        Value::Object(map) => {
            for (k, child) in map {
                let p = if prefix.is_empty() {
                    k.clone()
                } else {
                    format!("{prefix}.{k}")
                };
                flatten(&p, child, out);
            }
        }
        Value::Array(items) => {
            if items.iter().all(|x| !x.is_object() && !x.is_array()) {
                let joined: Vec<String> = items.iter().map(scalar).collect();
                out.push(label(prefix, &joined.join(", ")));
            } else {
                for (i, child) in items.iter().enumerate() {
                    flatten(&format!("{prefix}[{i}]"), child, out);
                }
            }
        }
        other => out.push(label(prefix, &scalar(other))),
    }
}

fn scalar(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => "null".into(),
        other => other.to_string(),
    }
}

fn label(prefix: &str, value: &str) -> String {
    if prefix.is_empty() {
        value.to_string()
    } else {
        format!("{prefix}: {value}")
    }
}

/// Keep at most `max_chars` characters (head and tail kept, middle elided),
/// so very long states stay within a backend's budget.
pub fn truncate_middle(text: &str, max_chars: usize) -> String {
    let n = text.chars().count();
    if max_chars == 0 || n <= max_chars {
        return text.to_string();
    }
    let head = max_chars * 2 / 3;
    let tail = max_chars.saturating_sub(head + 5);
    let h: String = text.chars().take(head).collect();
    let t: String = text.chars().skip(n - tail).collect();
    format!("{h}\n…\n{t}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn flattens_objects() {
        let t = state_to_text(&json!({
            "ticket": {"subject": "Refund please", "priority": "high"},
            "tags": ["billing", "vip"],
            "history": [{"from": "user", "text": "hi"}]
        }));
        assert!(t.contains("ticket.subject: Refund please"));
        assert!(t.contains("ticket.priority: high"));
        assert!(t.contains("tags: billing, vip"));
        assert!(t.contains("history[0].text: hi"));
    }

    #[test]
    fn strings_pass_through_and_truncate() {
        assert_eq!(state_to_text(&json!("hello")), "hello");
        let long = "x".repeat(1000);
        assert!(truncate_middle(&long, 100).chars().count() <= 100);
    }
}
