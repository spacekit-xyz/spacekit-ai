//! A/B harness for the homeostatic DriveField.
//!
//! Proves (or disproves) that flipping Luna's drive state changes her behavior in
//! a coherent, measurable way — without retraining and without breaking coherence.
//!
//! It runs the SAME prompt list through one loaded brain under three arms:
//!   - `off`    : drive field disabled (current/base behavior, the control)
//!   - `hungry` : hunger≈0.95, energy≈0.85, social≈0.20  (high dopamine + NE seeking)
//!   - `sated`  : hunger≈0.05, energy≈0.45, social≈0.95  (high serotonin, calm)
//!
//! For each arm it prints the response per prompt, then a summary:
//!   - unique responses per arm (variety)
//!   - cross-arm divergence: fraction of prompts where hungry ≠ sated
//!   - a coherence check (no fragmented "garble" leaked through)
//!
//! Usage:
//!   cargo run --release --example drive_field_ab -- \
//!       <brain.bin> <inference.toml>
//!
//! Defaults point at the Luna v3 brain + inference_pets.toml.

use std::collections::HashSet;
use std::path::PathBuf;

use growformer::drive_field::DriveState;
use growformer::runtime::Runtime;

const DEFAULT_BRAIN: &str =
    "/Users/astor/Projects/2026/spacekit/spacekit-projects/pets/agent/luna-v3.bin";
const DEFAULT_TOML: &str =
    "/Users/astor/Projects/2026/spacekit/spacekit-projects/pets/data/inference_pets.toml";
const PETS_DATA: &str = "/Users/astor/Projects/2026/spacekit/spacekit-projects/pets/data";

const PROMPTS: &[&str] = &[
    "Hey Luna",
    "what do you want?",
    "are you hungry?",
    "want to play?",
    "what are you looking at?",
    "come here",
    "good girl",
    "are you tired?",
    "lets go for a walk",
    "tell me about your day",
];

// Clearly out-of-training prompts (none of these are cat scenarios). Used to
// measure OOD collapse: a healthy system should still SPREAD across coherent
// responses rather than returning the same canned line every time.
const OOD_PROMPTS: &[&str] = &[
    "what is the capital of France?",
    "explain quantum entanglement",
    "write me a poem about taxes",
    "who won the world cup in 1998?",
    "can you debug my python code?",
    "what is the meaning of life?",
    "recommend a good restaurant",
    "how do interest rates work?",
    "what is the weather in Tokyo?",
    "who let the dogs out?",
];

fn ensure_drive_flag(toml: &str, enabled: bool) -> String {
    // Replace/insert `drive_field = <bool>` inside the existing [generation] table.
    let want = format!("drive_field = {}", enabled);
    if toml.contains("drive_field") {
        // Swap the value on the existing line.
        toml.lines()
            .map(|l| {
                if l.trim_start().starts_with("drive_field") {
                    want.clone()
                } else {
                    l.to_string()
                }
            })
            .collect::<Vec<_>>()
            .join("\n")
    } else {
        // Insert right after the real [generation] header line (not a comment mention).
        let mut inserted = false;
        toml.lines()
            .flat_map(|l| {
                if !inserted && l.trim() == "[generation]" {
                    inserted = true;
                    vec![l.to_string(), want.clone()]
                } else {
                    vec![l.to_string()]
                }
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

fn state_json(s: &DriveState) -> String {
    format!(
        r#"{{"dimensions":{{"hunger":{},"energy":{},"social":{}}},"minutes_idle":0,"turn":0}}"#,
        s.hunger, s.energy, s.social
    )
}

/// Crude fragmentation detector mirroring the runtime garble guard, so the harness
/// independently confirms coherence rather than trusting the runtime's own gate.
fn looks_garbled(t: &str) -> bool {
    let low = t.to_ascii_lowercase();
    // Mid-sentence double dot that is not an ellipsis.
    let bytes = low.as_bytes();
    for i in 1..bytes.len().saturating_sub(1) {
        if bytes[i] == b'.' && bytes[i + 1] == b'.' {
            let is_ellipsis = i + 2 < bytes.len() && bytes[i + 2] == b'.';
            if !is_ellipsis {
                return true;
            }
        }
    }
    // Long run of single-letter "words" like "I front I, you back paws. I. to."
    let toks: Vec<&str> = low.split_whitespace().collect();
    let mut run = 0;
    for w in &toks {
        let clean = w.trim_matches(|c: char| !c.is_alphabetic());
        if clean.len() <= 1 {
            run += 1;
            if run >= 3 {
                return true;
            }
        } else {
            run = 0;
        }
    }
    false
}

fn run_arm(rt: &mut Runtime, label: &str, state: Option<DriveState>) -> Vec<String> {
    // Reset conversation + (re)seed drive state from agent_state.
    if let Some(ref s) = state {
        rt.set_agent_state_from_json(&state_json(s))
            .expect("set agent state");
    }
    rt.reset_conversation();

    println!("\n========== arm: {} ==========", label);
    if let Some(ref s) = state {
        let nm = s.map_neuromodulators();
        let m = s.field_modulation();
        println!(
            "drive: hunger={:.2} energy={:.2} social={:.2}",
            s.hunger, s.energy, s.social
        );
        println!(
            "neuromod: {}  |  temp×{:.2} nov×{:.2} decay={:.2} state-blend×{:.2}",
            nm.summary(),
            m.temperature_scale,
            m.novelty_scale,
            m.context_decay,
            m.state_blend_scale
        );
    } else {
        println!("drive field: DISABLED (control)");
    }

    let mut out = Vec::new();
    for p in PROMPTS {
        let resp = rt.converse(p).expect("converse");
        let flag = if looks_garbled(&resp.text) {
            "  <GARBLE>"
        } else {
            ""
        };
        println!("  [{}] {}{}", p, resp.text, flag);
        out.push(resp.text);
    }
    out
}

fn main() {
    let mut args = std::env::args().skip(1);
    let brain_path = PathBuf::from(args.next().unwrap_or_else(|| DEFAULT_BRAIN.to_string()));
    let toml_path = PathBuf::from(args.next().unwrap_or_else(|| DEFAULT_TOML.to_string()));

    eprintln!("loading brain: {}", brain_path.display());
    let bytes = std::fs::read(&brain_path).expect("read brain bin");
    let mut rt = Runtime::from_brain_bytes(&bytes).expect("load brain");

    // Load the same routing artifacts the website injects, so retrieval is
    // representative (without the TopicGraph everything collapses to one group).
    let base = std::fs::read_to_string(format!("{}/knowledge_graph.toml", PETS_DATA))
        .expect("read topic graph base");
    let overlay =
        std::fs::read_to_string(format!("{}/knowledge_graph_pet_overlay.toml", PETS_DATA)).ok();
    let base_graph =
        growformer::topic_graph::TopicGraph::from_toml(&base).expect("parse topic graph base");
    let final_graph = match overlay {
        Some(ov) => {
            let og = growformer::topic_graph::TopicGraph::from_toml_quiet(&ov)
                .expect("parse topic graph overlay");
            base_graph.merge_overlay(og)
        }
        None => base_graph,
    };
    growformer::growformer_lang::init_topic_graph_direct(final_graph).expect("init topic graph");
    if let Ok(grounding) =
        std::fs::read_to_string(format!("{}/pet_world_grounding.toml", PETS_DATA))
    {
        growformer::inference::world_grounding::load_grounding_graph_from_str(&grounding)
            .expect("load grounding");
    }
    eprintln!("topic graph + grounding loaded");

    let base_toml = std::fs::read_to_string(&toml_path).expect("read inference toml");

    // --- Control arm: drive field OFF ---
    let off_toml = ensure_drive_flag(&base_toml, false);
    growformer::inference::inference_toml::reload_inference_toml_from_str(&off_toml)
        .expect("reload toml (off)");
    rt.apply_loaded_generation_config();
    let off = run_arm(&mut rt, "off (control)", None);

    // --- Drive field ON: hungry vs sated ---
    let on_toml = ensure_drive_flag(&base_toml, true);
    growformer::inference::inference_toml::reload_inference_toml_from_str(&on_toml)
        .expect("reload toml (on)");
    rt.apply_loaded_generation_config();

    let hungry = DriveState {
        hunger: 0.95,
        energy: 0.85,
        social: 0.20,
    };
    let sated = DriveState {
        hunger: 0.05,
        energy: 0.45,
        social: 0.95,
    };

    let hungry_out = run_arm(&mut rt, "hungry", Some(hungry));
    let sated_out = run_arm(&mut rt, "sated", Some(sated));

    // --- OOD collapse probe: each out-of-training prompt asked TWICE in a row.
    // Healthy behavior = coherent + varied; collapse = same canned line every time.
    rt.reset_conversation();
    println!("\n========== arm: OOD (out-of-training, each asked 2x) ==========");
    let mut ood_out: Vec<String> = Vec::new();
    let mut ood_repeat_collapses = 0;
    for p in OOD_PROMPTS {
        let a = rt.converse(p).expect("converse").text;
        let b = rt.converse(p).expect("converse").text;
        let same = if a == b {
            ood_repeat_collapses += 1;
            "  <REPEAT-COLLAPSE>"
        } else {
            ""
        };
        println!("  [{}]\n     1: {}\n     2: {}{}", p, a, b, same);
        ood_out.push(a);
        ood_out.push(b);
    }

    // --- Summary ---
    let uniq = |v: &[String]| v.iter().cloned().collect::<HashSet<_>>().len();
    let garble = |v: &[String]| v.iter().filter(|t| looks_garbled(t)).count();

    let divergence = hungry_out
        .iter()
        .zip(sated_out.iter())
        .filter(|(a, b)| a != b)
        .count();

    println!("\n================ SUMMARY ================");
    println!("prompts                : {}", PROMPTS.len());
    println!(
        "unique responses       : off={} hungry={} sated={}",
        uniq(&off),
        uniq(&hungry_out),
        uniq(&sated_out)
    );
    println!(
        "garbled responses       : off={} hungry={} sated={}",
        garble(&off),
        garble(&hungry_out),
        garble(&sated_out)
    );
    println!(
        "hungry≠sated divergence : {}/{} ({:.0}%)",
        divergence,
        PROMPTS.len(),
        100.0 * divergence as f32 / PROMPTS.len() as f32
    );
    println!(
        "OOD: {} prompts ×2  | unique responses={} | repeat-collapses={}/{} | garbled={}",
        OOD_PROMPTS.len(),
        uniq(&ood_out),
        ood_repeat_collapses,
        OOD_PROMPTS.len(),
        garble(&ood_out),
    );
    let total_garble = garble(&off) + garble(&hungry_out) + garble(&sated_out);
    println!(
        "verdict                 : {}",
        if divergence > 0 && total_garble == 0 {
            "PASS — drive state shifts behavior, coherence intact"
        } else if total_garble > 0 {
            "FAIL — garble leaked"
        } else {
            "NEUTRAL — no behavioral divergence detected"
        }
    );
}
