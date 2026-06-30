//! JSONL examples aligned with Growformer fintech / causal corpora (`text`, `semantic_intent`).

use serde::Deserialize;
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct Example {
    pub text: String,
    pub label: String,
}

#[derive(Debug, Deserialize)]
struct RowSparse {
    text: String,
    semantic_intent: String,
}

fn collect_jsonl_paths(dir: &Path) -> Result<Vec<PathBuf>, String> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| format!("read_dir {}: {}", dir.display(), e))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .map(|x| x.eq_ignore_ascii_case("jsonl"))
                .unwrap_or(false)
        })
        .collect();
    paths.sort();
    Ok(paths)
}

/// Names like `inference_guardrails.jsonl` live next to training corpora but are not labeled examples.
fn jsonl_filename_skipped_for_training(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower.contains("guardrails")
        || lower.starts_with("eval_")
        || lower.starts_with("inference_")
        || lower.starts_with("holdout_")
}

fn jsonl_paths_in_dir(dir: &Path) -> Result<Vec<PathBuf>, String> {
    let all = collect_jsonl_paths(dir)?;
    let train_prefixed: Vec<PathBuf> = all
        .iter()
        .filter(|p| {
            p.file_name()
                .and_then(|s| s.to_str())
                .map(|n| n.starts_with("train_"))
                .unwrap_or(false)
        })
        .cloned()
        .collect();

    let paths = if !train_prefixed.is_empty() {
        train_prefixed
    } else {
        all
            .into_iter()
            .filter(|p| {
                p.file_name()
                    .and_then(|s| s.to_str())
                    .map(|n| !jsonl_filename_skipped_for_training(n))
                    .unwrap_or(true)
            })
            .collect()
    };

    Ok(paths)
}

/// Load all `*.jsonl` under `dir`, collect unique `semantic_intent` labels (stable sort).
pub fn load_jsonl_dir(dir: &Path) -> Result<(Vec<Example>, Vec<String>), String> {
    let paths = jsonl_paths_in_dir(dir)?;
    if paths.is_empty() {
        return Err(format!(
            "no training .jsonl files in {} (use train_*.jsonl or rename; skipped guardrails/inference/eval)",
            dir.display()
        ));
    }

    let mut raw: Vec<Example> = Vec::new();
    for p in &paths {
        load_jsonl_file(p, &mut raw)?;
    }

    let mut label_set: BTreeMap<String, ()> = BTreeMap::new();
    for ex in &raw {
        label_set.insert(ex.label.clone(), ());
    }
    let labels: Vec<String> = label_set.into_keys().collect();

    Ok((raw, labels))
}

fn load_jsonl_file(path: &Path, out: &mut Vec<Example>) -> Result<(), String> {
    let f = File::open(path).map_err(|e| format!("open {}: {}", path.display(), e))?;
    let reader = BufReader::new(f);
    for (line_no, line) in reader.lines().enumerate() {
        let line = line.map_err(|e| format!("{}:{}: {}", path.display(), line_no + 1, e))?;
        let trim = line.trim();
        if trim.is_empty() {
            continue;
        }
        let v: Value = serde_json::from_str(trim)
            .map_err(|e| format!("{}:{}: {}", path.display(), line_no + 1, e))?;

        let text = v
            .get("text")
            .and_then(|x| x.as_str())
            .ok_or_else(|| format!("{}:{}: missing text", path.display(), line_no + 1))?
            .to_string();

        let label = if let Some(si) = v.get("semantic_intent").and_then(|x| x.as_str()) {
            si.to_string()
        } else if let Some(c) = v.get("causal").and_then(|c| c.get("causal_type")) {
            c.as_str()
                .unwrap_or("")
                .to_string()
        } else {
            return Err(format!(
                "{}:{}: missing semantic_intent",
                path.display(),
                line_no + 1
            ));
        };

        if label.is_empty() {
            return Err(format!(
                "{}:{}: empty label",
                path.display(),
                line_no + 1
            ));
        }

        out.push(Example { text, label });
    }
    Ok(())
}

/// Strict schema parse (used in tests).
pub fn parse_row(line: &str) -> Result<Example, String> {
    let r: RowSparse =
        serde_json::from_str(line.trim()).map_err(|e| e.to_string())?;
    Ok(Example {
        text: r.text,
        label: r.semantic_intent,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_minimal_line() {
        let line = r#"{"text":"hi","semantic_intent":"neutral"}"#;
        let ex = parse_row(line).unwrap();
        assert_eq!(ex.label, "neutral");
    }

    #[test]
    fn skips_guardrails_filename_rule() {
        assert!(jsonl_filename_skipped_for_training("inference_guardrails.jsonl"));
        assert!(!jsonl_filename_skipped_for_training("train_sentiment_fintech.jsonl"));
    }
}
