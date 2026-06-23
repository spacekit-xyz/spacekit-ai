//! Decompose conversational JSONL into typed fragment libraries.
//!
//! Vocal tokens, coda modifiers, and prompt→intent routing are read from
//! `[fragment_compose]` in the agent's inference TOML (same source as runtime).
//!
//! Usage:
//!   cargo run --example decompose_fragments -- \
//!     --data-dir /path/to/luna/data \
//!     --output luna_fragments_v2.jsonl
//!
//! Optional:
//!   --inference-toml /path/to/inference_pets.toml  (default: {data-dir}/inference_pets.toml)
//!   --agent-name Luna                               (for agent_name_greeting intent rules)
//!   --merge-hand /path/to/luna_fragments_v1.jsonl   (keep hand-authored rows)
//!   --min-count 2                                     (drop one-off fragments)

use growformer::inference::FragmentComposeConfig;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

/// JSONL training row (subset of fields we read).
#[derive(serde::Deserialize)]
struct TrainingRow {
    task_id: Option<String>,
    text: Option<String>,
    semantic_intent: Option<String>,
    expected_response: Option<String>,
    #[serde(default)]
    pet: Option<PetBlock>,
}

#[derive(serde::Deserialize, Default)]
struct PetBlock {
    #[serde(default)]
    archetype: Option<String>,
    #[serde(default)]
    ocean: Option<HashMap<String, f32>>,
    #[serde(default)]
    state: Option<HashMap<String, f32>>,
    #[serde(default)]
    graph_anchors: Vec<String>,
}

#[derive(Clone, Debug)]
struct DraftFragment {
    text: String,
    voice: &'static str,
    role: &'static str,
    intent_affinity: HashSet<String>,
    state_samples: Vec<HashMap<String, f32>>,
    vocalization: Option<String>,
    archetype: Option<String>,
    source_ids: Vec<String>,
    body_slot: Option<String>,
    intent_exclude: Vec<String>,
}

#[derive(Clone, Hash, Eq, PartialEq)]
struct FragmentKey {
    text_norm: String,
    voice: &'static str,
    role: &'static str,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut data_dir = PathBuf::from("/Users/astor/Projects/2026/spacekit/spacekit-projects/luna/data");
    let mut output = PathBuf::from("luna_fragments_v2.jsonl");
    let mut merge_hand: Option<PathBuf> = None;
    let mut min_count = 1usize;
    let mut inference_toml: Option<PathBuf> = None;
    let mut agent_name = String::new();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--data-dir" if i + 1 < args.len() => {
                data_dir = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            "--output" if i + 1 < args.len() => {
                output = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            "--inference-toml" if i + 1 < args.len() => {
                inference_toml = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--agent-name" if i + 1 < args.len() => {
                agent_name = args[i + 1].clone();
                i += 2;
            }
            "--merge-hand" if i + 1 < args.len() => {
                merge_hand = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--min-count" if i + 1 < args.len() => {
                min_count = args[i + 1].parse().unwrap_or(1);
                i += 2;
            }
            "--help" | "-h" => {
                eprintln!(
                    "usage: decompose_fragments [--data-dir DIR] [--output FILE] \
                     [--inference-toml FILE] [--agent-name NAME] \
                     [--merge-hand FILE] [--min-count N]"
                );
                std::process::exit(0);
            }
            other => {
                eprintln!("unknown arg: {other}");
                std::process::exit(2);
            }
        }
    }

    let toml_path = inference_toml.unwrap_or_else(|| data_dir.join("inference_pets.toml"));
    let vocab = FragmentComposeConfig::load_from_inference_toml_path(&toml_path).unwrap_or_else(|e| {
        eprintln!("failed to load fragment vocab from {}: {e}", toml_path.display());
        std::process::exit(1);
    });
    vocab.validate_for_decompose().unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1);
    });
    eprintln!(
        "loaded {} vocalizations from {}",
        vocab.vocalizations.len(),
        toml_path.display()
    );

    let input_files = select_input_files(&data_dir);
    if input_files.is_empty() {
        eprintln!("no input JSONL files found in {}", data_dir.display());
        std::process::exit(1);
    }

    let mut pool: HashMap<FragmentKey, DraftFragment> = HashMap::new();
    let mut rows_read = 0usize;
    let mut rows_skipped = 0usize;

    for path in &input_files {
        let content = std::fs::read_to_string(path).unwrap_or_else(|e| {
            eprintln!("read {}: {e}", path.display());
            std::process::exit(1);
        });
        for (line_no, line) in content.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let row: TrainingRow = match serde_json::from_str(line) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("  skip {}:{}: {e}", path.file_name().unwrap().to_string_lossy(), line_no + 1);
                    rows_skipped += 1;
                    continue;
                }
            };
            let response = match row.expected_response.as_deref() {
                Some(r) if !r.trim().is_empty() => r,
                _ => continue,
            };
            // Skip rows with code targets — not free-text chat.
            rows_read += 1;
            decompose_row(&row, response, &mut pool, &vocab, &agent_name);
        }
    }

    let mut fragments: Vec<DraftFragment> = pool
        .into_values()
        .filter(|f| f.source_ids.len() >= min_count)
        .collect();
    fragments.sort_by(|a, b| a.text.cmp(&b.text));

    let mut out_lines: Vec<String> = Vec::new();
    let mut seen_text: HashSet<String> = HashSet::new();

    if let Some(hand_path) = merge_hand {
        if let Ok(s) = std::fs::read_to_string(&hand_path) {
            for line in s.lines() {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
                    if let Some(text) = v.get("text").and_then(|t| t.as_str()) {
                        let norm = normalize(text);
                        if seen_text.insert(norm) {
                            out_lines.push(line.to_string());
                        }
                    }
                }
            }
            eprintln!("merged hand library from {} ({} lines kept)", hand_path.display(), out_lines.len());
        }
    }

    for f in &fragments {
        let norm = normalize(&f.text);
        if !seen_text.insert(norm) {
            continue;
        }
        out_lines.push(serialize_fragment(f, &vocab));
    }

    if let Some(parent) = output.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).ok();
        }
    }
    std::fs::write(&output, out_lines.join("\n") + "\n").unwrap_or_else(|e| {
        eprintln!("write {}: {e}", output.display());
        std::process::exit(1);
    });

    let voices: HashMap<&str, usize> = fragments.iter().fold(HashMap::new(), |mut m, f| {
        *m.entry(f.voice).or_default() += 1;
        m
    });

    println!("Decomposed {} training rows ({} skipped malformed)", rows_read, rows_skipped);
    println!("Input files ({}):", input_files.len());
    for p in &input_files {
        println!("  {}", p.display());
    }
    println!(
        "Wrote {} fragments to {} (voices: {:?})",
        out_lines.len(),
        output.display(),
        voices
    );
}

fn select_input_files(data_dir: &Path) -> Vec<PathBuf> {
    let prefix = if data_dir.join("pete_seed_v2.jsonl").is_file() {
        "pete_"
    } else {
        "luna_"
    };

    let include_suffixes = [
        "conversational_v1.jsonl",
        "state_variants_v1.jsonl",
        "multiturn_v1.jsonl",
        "opener_samples_v1.jsonl",
        "gratitude_comfort_v1.jsonl",
        "comfort_topics_v1.jsonl",
        "comfort_arcs_v1.jsonl",
        "coverage_v1.jsonl",
        "seed_v2.jsonl",
    ];
    let luna_only = ["expansion_v3.jsonl", "expansion_v4.jsonl", "expansion_v5.jsonl"];
    let pete_only = ["expansion_v1.jsonl", "expansion_v2.jsonl", "comfort_bible_v1.jsonl", "lore_v1.jsonl"];

    let exclude: HashSet<String> = [
        format!("{prefix}ood.jsonl"),
        format!("{prefix}fragments_v1.jsonl"),
        format!("{prefix}fragments_v2.jsonl"),
        "inference_guardrails.jsonl".into(),
    ]
    .into_iter()
    .collect();

    let mut files = Vec::new();
    for suffix in include_suffixes {
        let p = data_dir.join(format!("{prefix}{suffix}"));
        if p.is_file() {
            files.push(p);
        }
    }
    for suffix in if prefix == "pete_" {
        &pete_only[..]
    } else {
        &luna_only[..]
    } {
        let p = data_dir.join(format!("{prefix}{suffix}"));
        if p.is_file() {
            files.push(p);
        }
    }
    // Also pick up any other {prefix}*.jsonl not explicitly excluded.
    if let Ok(entries) = std::fs::read_dir(data_dir) {
        for ent in entries.flatten() {
            let p = ent.path();
            if p.extension().and_then(|e| e.to_str()) != Some("jsonl") {
                continue;
            }
            let fname = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if !fname.starts_with(prefix) || exclude.contains(fname) || fname.contains("fragments") {
                continue;
            }
            if !files.iter().any(|f| f == &p) {
                files.push(p);
            }
        }
    }
    files.sort();
    files
}

fn decompose_row(
    row: &TrainingRow,
    response: &str,
    pool: &mut HashMap<FragmentKey, DraftFragment>,
    vocab: &FragmentComposeConfig,
    agent_name: &str,
) {
    let user_text = row.text.clone().unwrap_or_default().to_ascii_lowercase();
    let intent = vocab
        .prompt_intent_override(&user_text, agent_name)
        .map(|h| h.intent)
        .or_else(|| row.semantic_intent.clone())
        .unwrap_or_else(|| "open_ended_chat".to_string());

    let archetype = row.pet.as_ref().and_then(|p| p.archetype.clone());
    let graph_anchors: Vec<String> = row
        .pet
        .as_ref()
        .map(|p| p.graph_anchors.clone())
        .unwrap_or_default();
    let state = row
        .pet
        .as_ref()
        .and_then(|p| p.state.clone())
        .unwrap_or_default();
    let source_id = row
        .task_id
        .clone()
        .unwrap_or_else(|| "unknown".to_string());

    let sentences = split_sentences(response, vocab);
    if sentences.is_empty() {
        return;
    }

    for (idx, sent) in sentences.iter().enumerate() {
        let text = sent.trim();
        if text.len() < 2 {
            continue;
        }
        let role = classify_role(text, idx, sentences.len(), vocab);
        let voice = vocab.classify_voice(text, role);
        let body_slot = vocab.classify_body_slot(text, role);
        let vocalization = if role == "coda" {
            vocab.detect_vocalization(text)
        } else {
            None
        };

        let key = FragmentKey {
            text_norm: normalize(text),
            voice,
            role,
        };

        pool.entry(key.clone())
            .and_modify(|f| {
                if intent != "lore_qa" {
                    f.intent_affinity.insert(intent.clone());
                }
                for a in &graph_anchors {
                    if !is_meta_anchor(a) {
                        f.intent_affinity.insert(a.clone());
                    }
                }
                tag_prompt_affinities(&user_text, &mut f.intent_affinity, vocab, agent_name);
                if !state.is_empty() {
                    f.state_samples.push(state.clone());
                }
                if f.archetype.is_none() {
                    f.archetype = archetype.clone();
                }
                if f.body_slot.is_none() {
                    f.body_slot = body_slot.clone();
                }
                f.source_ids.push(source_id.clone());
            })
            .or_insert_with(|| {
                let mut intents = HashSet::new();
                if intent != "lore_qa" {
                    intents.insert(intent.clone());
                }
                for a in &graph_anchors {
                    if !is_meta_anchor(a) {
                        intents.insert(a.clone());
                    }
                }
                tag_prompt_affinities(&user_text, &mut intents, vocab, agent_name);
                DraftFragment {
                    text: text.to_string(),
                    voice,
                    role,
                    intent_affinity: intents,
                    state_samples: if state.is_empty() {
                        Vec::new()
                    } else {
                        vec![state.clone()]
                    },
                    vocalization,
                    archetype: archetype.clone(),
                    source_ids: vec![source_id.clone()],
                    body_slot,
                    intent_exclude: Vec::new(),
                }
            });
    }
}

/// Split on sentence boundaries; keep trailing vocalization clauses separate.
fn split_sentences(response: &str, vocab: &FragmentComposeConfig) -> Vec<String> {
    let mut out = Vec::new();
    let mut buf = String::new();
    for ch in response.chars() {
        buf.push(ch);
        if ch == '.' || ch == '!' || ch == '?' {
            let piece = buf.trim().to_string();
            if !piece.is_empty() {
                out.push(piece);
            }
            buf.clear();
        }
    }
    let tail = buf.trim();
    if !tail.is_empty() {
        out.push(tail.to_string());
    }
    // Merge ultra-short non-vocal fragments with previous (e.g. "Sniff.").
    let mut merged = Vec::new();
    for s in out {
        let lower = s.to_ascii_lowercase();
        let is_voc = vocab.starts_with_vocalization(&lower);
        if !is_voc && s.len() < 12 && !merged.is_empty() {
            if let Some(prev) = merged.pop() {
                merged.push(format!("{prev} {s}"));
                continue;
            }
        }
        merged.push(s);
    }
    merged
}

fn tag_prompt_affinities(
    user_text: &str,
    intents: &mut HashSet<String>,
    vocab: &FragmentComposeConfig,
    agent_name: &str,
) {
    let hint = vocab.match_intent(user_text, agent_name);
    intents.insert(hint.intent);
    for a in &hint.anchors {
        if !is_meta_anchor(a) {
            intents.insert(a.clone());
        }
    }
}

fn is_meta_anchor(a: &str) -> bool {
    matches!(
        a,
        "cheerful_companion"
            | "siamese"
            | "cat"
            | "open_ended_chat"
            | "contented"
            | "playful_state"
            | "curious_state"
            | "routine_adherence"
            | "proactive_luna"
            | "vocal_communication"
    )
}

fn classify_role(
    text: &str,
    idx: usize,
    total: usize,
    vocab: &FragmentComposeConfig,
) -> &'static str {
    let lower = text.to_ascii_lowercase();
    if vocab.is_pure_vocal_coda(&lower) {
        return "coda";
    }
    if idx == 0 && vocab.is_opener(&lower) {
        return "opener";
    }
    if idx + 1 == total && text.len() < 40 && words(&lower) <= 6 {
        // Short trailing non-vocal line — still body unless it's clearly a coda.
        if lower.contains("go on") || lower.contains("help") || lower.contains("give") {
            return "body";
        }
    }
    "body"
}

fn words(s: &str) -> usize {
    s.split_whitespace().count()
}

fn normalize(s: &str) -> String {
    s.to_ascii_lowercase()
        .chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace())
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Merge observed state samples into inclusive gates with slack.
fn merge_state_gate(
    samples: &[HashMap<String, f32>],
    dims: &[String],
) -> HashMap<String, [f32; 2]> {
    if samples.is_empty() {
        return HashMap::new();
    }
    let mut gate = HashMap::new();
    for dim in dims {
        let vals: Vec<f32> = samples
            .iter()
            .filter_map(|s| s.get(dim).copied())
            .collect();
        if vals.is_empty() {
            continue;
        }
        let min_v = vals.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_v = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        // Expand slightly so gates generalize beyond exact training points.
        let lo = (min_v - 0.12).clamp(0.0, 1.0);
        let hi = (max_v + 0.12).clamp(0.0, 1.0);
        if hi - lo >= 0.08 {
            gate.insert(dim.clone(), [lo, hi]);
        }
    }
    gate
}

fn serialize_fragment(f: &DraftFragment, vocab: &FragmentComposeConfig) -> String {
    let mut intents: Vec<String> = f.intent_affinity.iter().cloned().collect();
    intents.sort();
    let state_gate = merge_state_gate(&f.state_samples, &vocab.decompose.state_gate_dims);
    let id_base = normalize(&f.text)
        .replace(' ', "_")
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == '_')
        .take(48)
        .collect::<String>();
    let fragment_id = format!("auto_{}_{}", f.voice, id_base);

    let mut obj = serde_json::Map::new();
    obj.insert("fragment_id".into(), fragment_id.into());
    obj.insert("voice".into(), f.voice.into());
    obj.insert("text".into(), f.text.clone().into());
    obj.insert("role".into(), f.role.into());
    obj.insert("intent_affinity".into(), serde_json::json!(intents));
    obj.insert("ocean_affinity".into(), serde_json::json!({}));
    obj.insert(
        "state_gate".into(),
        serde_json::to_value(&state_gate).unwrap_or(serde_json::json!({})),
    );
    if let Some(ref v) = f.vocalization {
        obj.insert("vocalization".into(), v.clone().into());
    }
    if let Some(ref a) = f.archetype {
        obj.insert("archetype".into(), a.clone().into());
    }
    if let Some(ref slot) = f.body_slot {
        obj.insert("body_slot".into(), slot.clone().into());
    }
    if !f.intent_exclude.is_empty() {
        obj.insert(
            "intent_exclude".into(),
            serde_json::json!(f.intent_exclude),
        );
    }
    obj.insert("weight".into(), serde_json::json!(1.0));
    serde_json::to_string(&obj).unwrap()
}
