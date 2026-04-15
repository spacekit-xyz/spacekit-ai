//! Optional **guardrail** rules in **JSONL** (one JSON object per line).
//!
//! Use this for product- or domain-specific inference policy without bloating the shared
//! `inference_sentiment_core.toml` / domain packs. Numeric gates and universal `[rules]` lists stay in TOML; guardrails
//! append **after** merged TOML rules (same evaluation order: TOML `headline_lexical_topic` / misfire
//! first, then all JSONL lines in load order).
//!
//! ## Schema (`v` optional, default 1; unknown `v` skips the line)
//!
//! **Lexical topic** (CNF on normalized intent, same as `[[rules.headline_lexical_topic]]`):
//! ```json
//! {"kind":"lexical_topic","topic":"neutral","intent":[["where will"],["year","years"]],"inclusion_redirect":false}
//! ```
//! Optional keys mirror TOML: `after_pr_wire`, `min_trim_len`, `exclude_first_person`, `require_crypto_lexicon`,
//! `unless_any_cnf`, `requires_mixed_positive_outcome_cue`, `require_question_mark` (omit or use serde defaults).
//!
//! **Lattice misfire** (intent CNF ∧ response side — same as `[[rules.lattice_misfire]]`):
//! ```json
//! {"kind":"lattice_misfire","intent":[["mainnet","blockchain"]],"response_any":["retail filers"],"response":[["filing"],["compliance"]]}
//! ```
//!
//! ## Resolution (native only; wasm loads none)
//! 1. [`set_inference_guardrails_jsonl_path`] (CLI / host), else `GROWFORMER_INFERENCE_GUARDRAILS_JSONL`, else
//! 2. Every existing file among `data/sentiment/inference_guardrails.jsonl` and
//!    `data/fintech/inference_guardrails.jsonl` under cwd / exe / `../data/...` (merged in that order).

use serde_json::Value;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use super::inference_toml::{HeadlineLexicalTopicRule, LatticeMisfireRule};

/// What was appended from JSONL on the last native merge (for training logs / diagnostics).
#[derive(Debug, Clone, Default)]
pub struct GuardrailsDiskSummary {
    pub headline_rows_appended: usize,
    pub misfire_rows_appended: usize,
    /// One line per file actually read (path + counts).
    pub log_lines: Vec<String>,
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Default)]
struct CliGuardrailsPath {
    primary: Option<PathBuf>,
}

#[cfg(not(target_arch = "wasm32"))]
static CLI_GUARDRAILS: OnceLock<CliGuardrailsPath> = OnceLock::new();

/// Register guardrails JSONL path from CLI or host (before first `inference_toml_loaded()`).
#[cfg(not(target_arch = "wasm32"))]
pub fn set_inference_guardrails_jsonl_path(primary: Option<PathBuf>) {
    if primary.is_none() {
        return;
    }
    let _ = CLI_GUARDRAILS.set(CliGuardrailsPath { primary });
}

#[cfg(target_arch = "wasm32")]
pub fn set_inference_guardrails_jsonl_path(_primary: Option<PathBuf>) {}

const GUARDRAILS_REL_PATHS: &[&str] = &[
    "data/sentiment/inference_guardrails.jsonl",
    "data/fintech/inference_guardrails.jsonl",
];

#[cfg(not(target_arch = "wasm32"))]
fn guardrails_candidate_files() -> Vec<PathBuf> {
    let mut v = Vec::new();
    if let Ok(cwd) = std::env::current_dir() {
        for rel in GUARDRAILS_REL_PATHS {
            v.push(cwd.join(rel));
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            for rel in GUARDRAILS_REL_PATHS {
                v.push(dir.join(rel));
                v.push(dir.join("../").join(rel));
            }
        }
    }
    v
}

/// Append guardrail rules into runtime (TOML rules remain first). Returns append counts + log lines.
#[cfg(not(target_arch = "wasm32"))]
pub fn merge_guardrails_into_runtime(
    headline: &mut Vec<HeadlineLexicalTopicRule>,
    misfire: &mut Vec<LatticeMisfireRule>,
) -> GuardrailsDiskSummary {
    let (gh, gm, log_lines) = load_guardrails_merged();
    let headline_rows_appended = gh.len();
    let misfire_rows_appended = gm.len();
    headline.extend(gh);
    misfire.extend(gm);
    GuardrailsDiskSummary {
        headline_rows_appended,
        misfire_rows_appended,
        log_lines,
    }
}

#[cfg(target_arch = "wasm32")]
pub fn merge_guardrails_into_runtime(
    _headline: &mut Vec<HeadlineLexicalTopicRule>,
    _misfire: &mut Vec<LatticeMisfireRule>,
) -> GuardrailsDiskSummary {
    GuardrailsDiskSummary::default()
}

#[cfg(not(target_arch = "wasm32"))]
fn load_guardrails_merged() -> (
    Vec<HeadlineLexicalTopicRule>,
    Vec<LatticeMisfireRule>,
    Vec<String>,
) {
    let mut headlines = Vec::new();
    let mut misfires = Vec::new();
    let mut log_lines: Vec<String> = Vec::new();

    let mut paths: Vec<PathBuf> = Vec::new();
    let mut configured_primary_missing = false;
    if let Some(p) = CLI_GUARDRAILS.get().and_then(|c| c.primary.clone()) {
        if p.is_file() {
            paths.push(p);
        } else {
            configured_primary_missing = true;
            eprintln!(
                "[inference-guardrails] configured JSONL path is not a readable file: {}",
                p.display()
            );
        }
    } else if let Ok(env) = std::env::var("GROWFORMER_INFERENCE_GUARDRAILS_JSONL") {
        if !env.is_empty() {
            let p = PathBuf::from(&env);
            if p.is_file() {
                paths.push(p);
            } else {
                eprintln!(
                    "[inference-guardrails] GROWFORMER_INFERENCE_GUARDRAILS_JSONL={} is not a readable file",
                    env
                );
            }
        }
    } else {
        paths.extend(guardrails_candidate_files());
    }

    let mut seen = std::collections::HashSet::new();
    for p in paths {
        if !seen.insert(p.clone()) {
            continue;
        }
        if !p.is_file() {
            continue;
        }
        let nh0 = headlines.len();
        let nm0 = misfires.len();
        match parse_guardrails_file(&p, &mut headlines, &mut misfires) {
            Ok(()) => {
                let dh = headlines.len() - nh0;
                let dm = misfires.len() - nm0;
                log_lines.push(format!(
                    "{} (+{} headline_lexical_topic, +{} lattice_misfire)",
                    p.display(),
                    dh,
                    dm
                ));
            }
            Err(e) => eprintln!(
                "[inference-guardrails] skip {}: {}",
                p.display(),
                e
            ),
        }
    }

    if log_lines.is_empty() && !configured_primary_missing {
        // Default discovery: show which relative paths were checked (cwd / exe roots).
        let tried: Vec<String> = guardrails_candidate_files()
            .into_iter()
            .map(|p| format!("{} ({})", p.display(), if p.is_file() { "ok" } else { "missing" }))
            .collect();
        if !tried.is_empty() {
            log_lines.push(format!(
                "no JSONL rules merged (tried: {})",
                tried.join("; ")
            ));
        }
    }

    (headlines, misfires, log_lines)
}

#[cfg(not(target_arch = "wasm32"))]
fn parse_guardrails_file(
    path: &Path,
    headlines: &mut Vec<HeadlineLexicalTopicRule>,
    misfires: &mut Vec<LatticeMisfireRule>,
) -> Result<(), String> {
    let s = std::fs::read_to_string(path)
        .map_err(|e| format!("read: {}", e))?;
    for (line_no, line) in s.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        parse_guardrail_line(line, line_no + 1, path, headlines, misfires)?;
    }
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
fn parse_guardrail_line(
    line: &str,
    line_no: usize,
    path: &Path,
    headlines: &mut Vec<HeadlineLexicalTopicRule>,
    misfires: &mut Vec<LatticeMisfireRule>,
) -> Result<(), String> {
    let v: Value = serde_json::from_str(line).map_err(|e| {
        format!(
            "{}:{}: invalid JSON: {}",
            path.display(),
            line_no,
            e
        )
    })?;
    let kind = v
        .get("kind")
        .and_then(|x| x.as_str())
        .unwrap_or("")
        .to_string();
    if kind == "comment" || kind == "meta" {
        return Ok(());
    }
    if let Some(vn) = v.get("v").and_then(|x| x.as_u64()) {
        if vn != 1 {
            eprintln!(
                "[inference-guardrails] {}:{}: unsupported v={}, skip",
                path.display(),
                line_no,
                vn
            );
            return Ok(());
        }
    }

    match kind.as_str() {
        "lexical_topic" => {
            let topic = v
                .get("topic")
                .and_then(|x| x.as_str())
                .ok_or_else(|| {
                    format!(
                        "{}:{}: lexical_topic missing topic",
                        path.display(),
                        line_no
                    )
                })?
                .to_string();
            let intent = json_to_cnf(v.get("intent").ok_or_else(|| {
                format!(
                    "{}:{}: lexical_topic missing intent",
                    path.display(),
                    line_no
                )
            })?)?;
            if intent.is_empty() {
                return Err(format!(
                    "{}:{}: lexical_topic empty intent",
                    path.display(),
                    line_no
                ));
            }
            let inclusion_redirect = v
                .get("inclusion_redirect")
                .and_then(|x| x.as_bool())
                .unwrap_or(false);
            let min_trim_len = v.get("min_trim_len").and_then(|x| x.as_u64()).map(|n| n as usize);
            let exclude_first_person = v
                .get("exclude_first_person")
                .and_then(|x| x.as_bool())
                .unwrap_or(false);
            let require_crypto_lexicon = v
                .get("require_crypto_lexicon")
                .and_then(|x| x.as_bool())
                .unwrap_or(false);
            let unless_any_cnf = json_unless_any_cnf(v.get("unless_any_cnf"))?;
            let after_pr_wire = v
                .get("after_pr_wire")
                .and_then(|x| x.as_bool())
                .unwrap_or(false);
            let requires_mixed_positive_outcome_cue = v
                .get("requires_mixed_positive_outcome_cue")
                .and_then(|x| x.as_bool())
                .unwrap_or(false);
            let require_question_mark = v
                .get("require_question_mark")
                .and_then(|x| x.as_bool())
                .unwrap_or(false);
            headlines.push(HeadlineLexicalTopicRule {
                topic,
                inclusion_redirect,
                intent,
                min_trim_len,
                exclude_first_person,
                require_crypto_lexicon,
                unless_any_cnf,
                after_pr_wire,
                requires_mixed_positive_outcome_cue,
                require_question_mark,
            });
        }
        "lattice_misfire" => {
            let intent = json_to_cnf(v.get("intent").unwrap_or(&Value::Array(vec![])))?;
            if intent.is_empty() {
                return Err(format!(
                    "{}:{}: lattice_misfire empty intent",
                    path.display(),
                    line_no
                ));
            }
            let response_any: Vec<String> = v
                .get("response_any")
                .and_then(|x| x.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|i| i.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default();
            let response = v
                .get("response")
                .map(json_to_cnf)
                .transpose()?
                .unwrap_or_default();
            if response_any.is_empty() && response.is_empty() {
                return Err(format!(
                    "{}:{}: lattice_misfire needs response_any and/or response",
                    path.display(),
                    line_no
                ));
            }
            misfires.push(LatticeMisfireRule {
                intent,
                response_any,
                response,
            });
        }
        "" => {}
        other => {
            eprintln!(
                "[inference-guardrails] {}:{}: unknown kind {:?}, skip",
                path.display(),
                line_no,
                other
            );
        }
    }
    Ok(())
}

fn json_unless_any_cnf(val: Option<&Value>) -> Result<Vec<Vec<Vec<String>>>, String> {
    let Some(v) = val else {
        return Ok(Vec::new());
    };
    let arr = v
        .as_array()
        .ok_or_else(|| "unless_any_cnf must be a JSON array of CNFs".to_string())?;
    let mut out = Vec::new();
    for item in arr {
        out.push(json_to_cnf(item)?);
    }
    Ok(out)
}

fn json_to_cnf(val: &Value) -> Result<Vec<Vec<String>>, String> {
    let arr = val
        .as_array()
        .ok_or_else(|| "intent/response must be a JSON array of string arrays".to_string())?;
    let mut out = Vec::new();
    for g in arr {
        let garr = g
            .as_array()
            .ok_or_else(|| "each intent group must be a JSON array of strings".to_string())?;
        let mut ors = Vec::new();
        for s in garr {
            let st = s
                .as_str()
                .ok_or_else(|| "intent phrase must be a string".to_string())?;
            ors.push(st.to_string());
        }
        out.push(ors);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_lexical_and_misfire_lines() {
        let mut h = Vec::new();
        let mut m = Vec::new();
        parse_guardrail_line(
            r#"{"kind":"lexical_topic","topic":"neutral","intent":[["where will"],["years"]]}"#,
            1,
            Path::new("test.jsonl"),
            &mut h,
            &mut m,
        )
        .unwrap();
        parse_guardrail_line(
            r#"{"kind":"lattice_misfire","intent":[["foo"]],"response_any":["bar"]}"#,
            2,
            Path::new("test.jsonl"),
            &mut h,
            &mut m,
        )
        .unwrap();
        assert_eq!(h.len(), 1);
        assert_eq!(h[0].topic, "neutral");
        assert_eq!(m.len(), 1);
        assert_eq!(m[0].response_any, vec!["bar".to_string()]);
    }
}
