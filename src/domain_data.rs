//! JSONL examples aligned with Growformer fintech / causal corpora (`text`, `semantic_intent`).
//!
//! Chatbot training prefers **clean** assistant lines (user-facing), not classifier
//! rationales ("not first-person", "third-party headline", …).

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
    /// Preferred assistant target for chatbot training (`expected_response`, etc.).
    pub response: Option<String>,
}

#[derive(Debug, Clone, Copy)]
pub struct ChatWriteOptions {
    /// Strip meta rationales; keep short user-facing assistant lines.
    pub clean: bool,
    /// Soft cap on assistant characters after cleaning (0 = no cap).
    pub max_assistant_chars: usize,
}

impl Default for ChatWriteOptions {
    fn default() -> Self {
        Self {
            clean: true,
            max_assistant_chars: 160,
        }
    }
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
        all.into_iter()
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
            c.as_str().unwrap_or("").to_string()
        } else {
            return Err(format!(
                "{}:{}: missing semantic_intent",
                path.display(),
                line_no + 1
            ));
        };

        if label.is_empty() {
            return Err(format!("{}:{}: empty label", path.display(), line_no + 1));
        }

        let response = v
            .get("expected_response")
            .or_else(|| v.get("response"))
            .or_else(|| v.get("answer"))
            .or_else(|| v.get("assistant"))
            .and_then(|x| x.as_str())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());

        out.push(Example {
            text,
            label,
            response,
        });
    }
    Ok(())
}

/// Map semantic_intent / topic ids to a short chat polarity tag.
pub fn polarity_tag(label: &str) -> &'static str {
    let l = label.to_ascii_lowercase();
    if l.contains("negative_strong")
        || l.contains("capitulation")
        || l.contains("etf_delay_bearish")
        || l.contains("fee_complaint")
        || l.contains("mortgage_rate_complaint")
    {
        "NEGATIVE (strong)"
    } else if l.contains("negative")
        || l.contains("bearish")
        || l.contains("cautiously_negative")
        || l.contains("copium")
    {
        "NEGATIVE (mild)"
    } else if l.contains("positive_strong") || l.contains("euphoric") || l.contains("hopium") {
        "POSITIVE (strong)"
    } else if l.contains("positive") || l.contains("bullish") || l.contains("cautiously_positive") {
        "POSITIVE (mild)"
    } else if l.contains("mixed") {
        "MIXED"
    } else {
        "NEUTRAL"
    }
}

fn is_meta_rationale_clause(clause: &str) -> bool {
    let c = clause.to_ascii_lowercase();
    const NEEDLES: &[&str] = &[
        "first-person",
        "third-party",
        "not a consumer",
        "not praise",
        "not gratitude",
        "not consumer",
        "not a bank",
        "not crypto macro",
        "not resigned",
        "not neutral product",
        "analytical / reporting",
        "reporting tone",
        "desk read",
        "wire",
        "op-ed",
        "synthetic tickers",
        "without first-person",
        "not a bullish",
        "not personal",
        "not as personal",
        "classifier",
        "over-fit",
        "anchor in-domain",
    ];
    NEEDLES.iter().any(|n| c.contains(n))
}

/// Strip classifier rationales; keep a short user-facing assistant line.
pub fn clean_assistant_reply(raw: &str, label: &str, max_chars: usize) -> String {
    let raw = raw.trim();
    if raw.is_empty() {
        return format!("{} — {}", polarity_tag(label), "Noted.");
    }

    // Split "TAG — body" when present.
    let (tag, body) = if let Some((a, b)) = raw.split_once(" — ") {
        (a.trim().to_string(), b.trim().to_string())
    } else if let Some((a, b)) = raw.split_once(" - ") {
        // rare hyphen variant
        if a.len() < 40 {
            (a.trim().to_string(), b.trim().to_string())
        } else {
            (polarity_tag(label).to_string(), raw.to_string())
        }
    } else {
        (polarity_tag(label).to_string(), raw.to_string())
    };

    // Drop meta clauses after `;` / `.` when they look like rationales.
    let mut keep: Vec<&str> = Vec::new();
    for part in body.split(';') {
        let p = part.trim();
        if p.is_empty() {
            continue;
        }
        if is_meta_rationale_clause(p) {
            continue;
        }
        // Also drop trailing "not X" sentence fragments inside the clause.
        if p.to_ascii_lowercase().starts_with("not ") && p.len() < 80 {
            continue;
        }
        keep.push(p);
    }
    let mut body = keep.join("; ");
    if body.is_empty() || is_meta_rationale_clause(&body) {
        // Fall back: first sentence of original body without meta, or generic.
        let first = body
            .split(['.', '!'])
            .map(str::trim)
            .find(|s| !s.is_empty() && !is_meta_rationale_clause(s))
            .unwrap_or("");
        body = if first.is_empty() {
            "See retrieved domain memory for details.".into()
        } else {
            first.to_string()
        };
    }

    // Prefer first sentence for chat brevity.
    if let Some(end) = body.find(". ") {
        if end + 1 < body.len() {
            let rest = &body[end + 2..];
            if is_meta_rationale_clause(rest) || rest.len() > 40 {
                body = body[..=end].to_string();
            }
        }
    }

    let mut out = format!("{} — {}", tag.trim(), body.trim().trim_end_matches('.'));
    if !out.ends_with('.') {
        out.push('.');
    }
    if max_chars > 0 && out.chars().count() > max_chars {
        let mut s: String = out.chars().take(max_chars.saturating_sub(1)).collect();
        s.push('…');
        return s;
    }
    out
}

pub fn assistant_target(ex: &Example, opts: ChatWriteOptions) -> String {
    let raw = ex
        .response
        .as_deref()
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .unwrap_or_else(|| format!("{} — {}", polarity_tag(&ex.label), ex.text.trim()));
    if opts.clean {
        clean_assistant_reply(&raw, &ex.label, opts.max_assistant_chars)
    } else {
        raw
    }
}

/// Write example texts as TinyStories-style `<|endoftext|>`-separated `.txt`
/// for `tinystories tokenize` / `encode`.
pub fn write_eot_txt(examples: &[Example], out: &Path) -> Result<(), String> {
    use std::io::Write;
    let mut f = File::create(out).map_err(|e| format!("create {}: {e}", out.display()))?;
    for (i, ex) in examples.iter().enumerate() {
        let t = ex.text.trim();
        if t.is_empty() {
            continue;
        }
        if i > 0 {
            writeln!(f, "<|endoftext|>").map_err(|e| e.to_string())?;
        }
        writeln!(f, "{t}").map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// Load one or more JSONL dirs and write a combined EOT `.txt`.
pub fn jsonl_dirs_to_eot_txt(dirs: &[PathBuf], out: &Path) -> Result<usize, String> {
    let mut all = Vec::new();
    for d in dirs {
        let (ex, _) = load_jsonl_dir(d)?;
        all.extend(ex);
    }
    if all.is_empty() {
        return Err("no examples loaded".into());
    }
    write_eot_txt(&all, out)?;
    Ok(all.len())
}

/// One User/Assistant turn in chat markers, for chatbot LM training.
pub fn example_to_chat_block(ex: &Example, system: Option<&str>, opts: ChatWriteOptions) -> String {
    use crate::chat::{MARK_ASSISTANT, MARK_SYSTEM, MARK_USER};
    let assistant = assistant_target(ex, opts);
    let mut block = String::new();
    if let Some(sys) = system {
        let s = sys.trim();
        if !s.is_empty() {
            block.push_str(MARK_SYSTEM);
            block.push('\n');
            block.push_str(s);
            block.push('\n');
        }
    }
    block.push_str(MARK_USER);
    block.push('\n');
    block.push_str(ex.text.trim());
    block.push('\n');
    block.push_str(MARK_ASSISTANT);
    block.push('\n');
    block.push_str(assistant.trim());
    block.push('\n');
    block
}

/// Write chat-formatted EOT corpus (aligned with [`crate::chat`] markers).
pub fn write_chat_eot_txt(
    examples: &[Example],
    out: &Path,
    system: Option<&str>,
    opts: ChatWriteOptions,
) -> Result<usize, String> {
    use std::io::Write;
    let mut f = File::create(out).map_err(|e| format!("create {}: {e}", out.display()))?;
    let mut n = 0usize;
    for ex in examples {
        if ex.text.trim().is_empty() {
            continue;
        }
        if n > 0 {
            writeln!(f, "<|endoftext|>").map_err(|e| e.to_string())?;
        }
        write!(f, "{}", example_to_chat_block(ex, system, opts)).map_err(|e| e.to_string())?;
        n += 1;
    }
    if n == 0 {
        return Err("no examples written".into());
    }
    Ok(n)
}

pub fn jsonl_dirs_to_chat_eot_txt(
    dirs: &[PathBuf],
    out: &Path,
    system: Option<&str>,
    opts: ChatWriteOptions,
) -> Result<usize, String> {
    let mut all = Vec::new();
    for d in dirs {
        let (ex, _) = load_jsonl_dir(d)?;
        all.extend(ex);
    }
    if all.is_empty() {
        return Err("no examples loaded".into());
    }
    write_chat_eot_txt(&all, out, system, opts)
}

/// Strict schema parse (used in tests).
pub fn parse_row(line: &str) -> Result<Example, String> {
    let r: RowSparse = serde_json::from_str(line.trim()).map_err(|e| e.to_string())?;
    Ok(Example {
        text: r.text,
        label: r.semantic_intent,
        response: None,
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
        assert!(jsonl_filename_skipped_for_training(
            "inference_guardrails.jsonl"
        ));
        assert!(!jsonl_filename_skipped_for_training(
            "train_sentiment_fintech.jsonl"
        ));
    }

    #[test]
    fn cleans_meta_rationale() {
        let raw = "NEUTRAL — Corporate finance event: tender offer at record valuation. Factual reporting, not sentiment.";
        let out = clean_assistant_reply(raw, "neutral", 160);
        assert!(out.starts_with("NEUTRAL"));
        assert!(!out.to_ascii_lowercase().contains("first-person"));
        assert!(!out.to_ascii_lowercase().contains("third-party"));
        // Should keep the substance, drop trailing meta if split worked
        assert!(out.contains("tender offer") || out.contains("Corporate finance"));
    }

    #[test]
    fn cleans_third_party_clause() {
        let raw = "NEGATIVE (mild) — Bearish tape; third-party headline about holder outcomes; not gratitude toward support staff.";
        let out = clean_assistant_reply(raw, "negative_mild", 160);
        assert!(out.contains("NEGATIVE"));
        assert!(!out.to_ascii_lowercase().contains("third-party"));
        assert!(!out.to_ascii_lowercase().contains("gratitude"));
    }
}
