//! A/B harness for the ReflectiveField (Identity ⊕ Activity ⊕ Drive composition).
//!
//! The DriveField proved *state* shifts behavior. The ReflectiveField's new claim
//! is that the agent's **activity** (conversation momentum) becomes a first-class
//! voice in the response — so the SAME probe prompt should land differently when
//! asked cold (turn 1) versus deep in a themed conversation (turn N).
//!
//! It runs one loaded brain under two arms (drive field ON in both so the
//! reflective policy has neuromodulators to couple to):
//!   - `reflective OFF` : scattered legacy blend constants (control)
//!   - `reflective ON`  : unified neuromodulator-coupled composition
//!
//! For each arm it walks a themed multi-turn script, re-asking a fixed probe at
//! the start and after the conversation has built momentum, then reports:
//!   - context sensitivity : probe@cold ≠ probe@warm  (activity is doing work)
//!   - OFF vs ON divergence : fraction of script turns where the arms differ
//!   - coherence           : no fragmented garble leaked through either arm
//!
//! Usage:
//!   cargo run --release --example reflective_field_ab -- <brain.bin> <inference.toml>

use std::collections::HashSet;
use std::path::PathBuf;

use growformer::drive_field::DriveState;
use growformer::runtime::Runtime;

const DEFAULT_BRAIN: &str =
    "/Users/astor/Projects/2026/spacekit/spacekit-projects/pets/agent/luna-v3.bin";
const DEFAULT_TOML: &str =
    "/Users/astor/Projects/2026/spacekit/spacekit-projects/pets/data/inference_pets.toml";
const PETS_DATA: &str = "/Users/astor/Projects/2026/spacekit/spacekit-projects/pets/data";

/// A fixed probe asked cold, then again after the conversation builds momentum.
const PROBE: &str = "what are you thinking about?";

/// A themed multi-turn conversation (play / hunting momentum) between the two probes.
const SCRIPT: &[&str] = &[
    "want to play?",
    "I'm waving the feather wand",
    "go get it!",
    "good hunting",
    "you caught it",
];

fn ensure_flags(toml: &str, drive: bool, reflective: bool) -> String {
    let set_line = |toml: String, key: &str, val: bool| -> String {
        let want = format!("{} = {}", key, val);
        if toml.lines().any(|l| l.trim_start().starts_with(key)) {
            toml.lines()
                .map(|l| {
                    if l.trim_start().starts_with(key) {
                        want.clone()
                    } else {
                        l.to_string()
                    }
                })
                .collect::<Vec<_>>()
                .join("\n")
        } else {
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
    };
    let t = set_line(toml.to_string(), "drive_field", drive);
    set_line(t, "reflective_field", reflective)
}

fn state_json(s: &DriveState) -> String {
    format!(
        r#"{{"dimensions":{{"hunger":{},"energy":{},"social":{}}},"minutes_idle":0,"turn":0}}"#,
        s.hunger, s.energy, s.social
    )
}

/// Crude fragmentation detector mirroring the runtime garble guard.
fn looks_garbled(t: &str) -> bool {
    let low = t.to_ascii_lowercase();
    let bytes = low.as_bytes();
    for i in 1..bytes.len().saturating_sub(1) {
        if bytes[i] == b'.' && bytes[i + 1] == b'.' {
            let is_ellipsis = i + 2 < bytes.len() && bytes[i + 2] == b'.';
            if !is_ellipsis {
                return true;
            }
        }
    }
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

/// Candidate target topics for the comfort "emotional landing", in preference
/// order. The harness picks the first one that exists in the loaded brain.
const GOAL_TOPIC_CANDIDATES: &[&str] = &[
    "emotional_support",
    "comfort",
    "bonding_request",
    "bonding_moment",
    "affection_display",
];

/// Returns (probe@cold, script responses, probe@warm).
fn run_arm(rt: &mut Runtime, label: &str) -> (String, Vec<String>, String) {
    // Same mild baseline state in both arms so the only variable is the policy.
    let state = DriveState {
        hunger: 0.4,
        energy: 0.7,
        social: 0.5,
    };
    rt.set_agent_state_from_json(&state_json(&state))
        .expect("set agent state");
    rt.reset_conversation();

    println!("\n========== arm: {} ==========", label);

    let cold = rt.converse(PROBE).expect("converse").text;
    println!("  probe@cold : {}", cold);

    let mut script_out = Vec::new();
    for p in SCRIPT {
        let resp = rt.converse(p).expect("converse").text;
        let flag = if looks_garbled(&resp) {
            "  <GARBLE>"
        } else {
            ""
        };
        println!("    [{}] {}{}", p, resp, flag);
        script_out.push(resp);
    }

    let warm = rt.converse(PROBE).expect("converse").text;
    let warm_flag = if cold != warm {
        "  <CONTEXT-SHIFT>"
    } else {
        "  <no shift>"
    };
    println!("  probe@warm : {}{}", warm, warm_flag);

    (cold, script_out, warm)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let brain_path = PathBuf::from(args.next().unwrap_or_else(|| DEFAULT_BRAIN.to_string()));
    let toml_path = PathBuf::from(args.next().unwrap_or_else(|| DEFAULT_TOML.to_string()));

    eprintln!("loading brain: {}", brain_path.display());
    let bytes = std::fs::read(&brain_path).expect("read brain bin");
    let mut rt = Runtime::from_brain_bytes(&bytes).expect("load brain");

    // Load the same routing artifacts the website injects.
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

    // --- Control: reflective OFF (drive ON) ---
    let off_toml = ensure_flags(&base_toml, true, false);
    growformer::inference::inference_toml::reload_inference_toml_from_str(&off_toml)
        .expect("reload toml (reflective off)");
    rt.apply_loaded_generation_config();
    let (off_cold, off_script, off_warm) = run_arm(&mut rt, "reflective OFF (control)");

    // --- Treatment: reflective ON (drive ON) ---
    let on_toml = ensure_flags(&base_toml, true, true);
    growformer::inference::inference_toml::reload_inference_toml_from_str(&on_toml)
        .expect("reload toml (reflective on)");
    rt.apply_loaded_generation_config();
    rt.set_goal(None, 0.0).expect("clear goal");
    let (on_cold, on_script, on_warm) = run_arm(&mut rt, "reflective ON (no goal)");

    // --- Treatment: reflective ON + retrocausal goal-attractor (comfort landing) ---
    let topics = rt.available_topics();
    eprintln!("available topics ({}): {}", topics.len(), topics.join(", "));
    let goal_topic = GOAL_TOPIC_CANDIDATES
        .iter()
        .find(|c| topics.iter().any(|t| t.eq_ignore_ascii_case(c)))
        .copied()
        .or_else(|| topics.first().map(|s| s.as_str()))
        .expect("at least one topic available");
    rt.set_goal(Some(goal_topic), 0.6).expect("set goal");
    println!(
        "\n[goal-attractor target topic: \"{}\" pull=0.60]",
        goal_topic
    );
    let (goal_cold, goal_script, goal_warm) =
        run_arm(&mut rt, &format!("reflective ON + goal({})", goal_topic));
    rt.set_goal(None, 0.0).expect("clear goal");

    // --- Summary ---
    let uniq = |v: &[String]| v.iter().cloned().collect::<HashSet<_>>().len();
    let garble = |v: &[String]| v.iter().filter(|t| looks_garbled(t)).count();

    let off_ctx_shift = off_cold != off_warm;
    let on_ctx_shift = on_cold != on_warm;

    let script_divergence = off_script
        .iter()
        .zip(on_script.iter())
        .filter(|(a, b)| a != b)
        .count();

    let mut off_all = off_script.clone();
    off_all.push(off_cold);
    off_all.push(off_warm);
    let mut on_all = on_script.clone();
    on_all.push(on_cold.clone());
    on_all.push(on_warm.clone());
    let mut goal_all = goal_script.clone();
    goal_all.push(goal_cold.clone());
    goal_all.push(goal_warm.clone());

    // How many turns did the comfort goal change vs the no-goal ON arm?
    let mut on_seq = on_script.clone();
    on_seq.insert(0, on_cold);
    on_seq.push(on_warm);
    let mut goal_seq = goal_script.clone();
    goal_seq.insert(0, goal_cold);
    goal_seq.push(goal_warm);
    let goal_shift = on_seq
        .iter()
        .zip(goal_seq.iter())
        .filter(|(a, b)| a != b)
        .count();

    println!("\n================ SUMMARY ================");
    println!("script turns               : {}", SCRIPT.len());
    println!(
        "context-shift (probe warm≠cold) : OFF={} ON={}",
        off_ctx_shift, on_ctx_shift
    );
    println!(
        "OFF vs ON script divergence     : {}/{} ({:.0}%)",
        script_divergence,
        SCRIPT.len(),
        100.0 * script_divergence as f32 / SCRIPT.len() as f32
    );
    println!(
        "goal-attractor shift (vs no-goal): {}/{} turns changed toward comfort landing",
        goal_shift,
        on_seq.len()
    );
    println!(
        "unique responses                : OFF={} ON={} ON+goal={}",
        uniq(&off_all),
        uniq(&on_all),
        uniq(&goal_all)
    );
    println!(
        "garbled responses               : OFF={} ON={} ON+goal={}",
        garble(&off_all),
        garble(&on_all),
        garble(&goal_all)
    );

    let total_garble = garble(&off_all) + garble(&on_all) + garble(&goal_all);
    let verdict = if total_garble > 0 {
        "FAIL — garble leaked"
    } else if on_ctx_shift && (script_divergence > 0 || !off_ctx_shift) {
        "PASS — activity (momentum) shapes responses, coherence intact"
    } else if script_divergence > 0 {
        "PARTIAL — arms differ but momentum sensitivity unproven"
    } else {
        "NEUTRAL — no measurable effect"
    };
    println!("verdict                         : {}", verdict);
}
