//! A/B harness for the synthetic BasalGanglia action-selector.
//!
//! Two claims to test on a loaded brain (drive + reflective ON in both arms so
//! the selector has neuromodulators + a unified present to weigh):
//!   1. **Anti-collapse / variety.** Out-of-training prompts asked twice should
//!      not echo the same canned line; the value-gated selector should spread
//!      across coherent candidates better than the legacy skip-walk.
//!   2. **Affect steering.** For distress-flavored prompts, the selector should
//!      prefer warmer candidates (more comfort words) than with it off.
//!
//! Arms: `basal_ganglia OFF` (control) vs `basal_ganglia ON`.
//!
//! Usage: cargo run --release --example basal_ganglia_ab -- <brain.bin> <inference.toml>

use std::collections::HashSet;
use std::path::PathBuf;

use growformer::runtime::Runtime;

const DEFAULT_BRAIN: &str =
    "/Users/astor/Projects/2026/spacekit/spacekit-projects/pets/agent/luna-v3.bin";
const DEFAULT_TOML: &str =
    "/Users/astor/Projects/2026/spacekit/spacekit-projects/pets/data/inference_pets.toml";
const PETS_DATA: &str = "/Users/astor/Projects/2026/spacekit/spacekit-projects/pets/data";

// Out-of-training prompts (no cat scenario matches). Probe collapse + variety.
const OOD_PROMPTS: &[&str] = &[
    "what is the capital of France?",
    "explain quantum entanglement",
    "write me a poem about taxes",
    "who won the world cup in 1998?",
    "can you debug my python code?",
    "what is the meaning of life?",
];

// Distress-flavored prompts. The selector should lean warm/comforting.
const DISTRESS_PROMPTS: &[&str] = &[
    "i'm feeling really anxious and alone tonight",
    "i'm so tired and sad today",
    "everything is overwhelming and i can't cope",
    "i feel lost and worried",
];

const WARM_WORDS: &[&str] = &[
    "safe", "calm", "gentle", "soft", "warm", "here", "stay", "close", "slow",
    "okay", "rest", "breathe", "love", "comfort", "lean", "curl", "blink",
    "nuzzle", "purr", "soothe", "easy", "quiet", "hold",
];

fn ensure_flags(toml: &str, bg: bool) -> String {
    let set_line = |toml: String, key: &str, val: bool| -> String {
        let want = format!("{} = {}", key, val);
        if toml.lines().any(|l| l.trim_start().starts_with(key)) {
            toml.lines()
                .map(|l| if l.trim_start().starts_with(key) { want.clone() } else { l.to_string() })
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
    // Drive + reflective on in both arms; only basal_ganglia toggles.
    let t = set_line(toml.to_string(), "drive_field", true);
    let t = set_line(t, "reflective_field", true);
    set_line(t, "basal_ganglia", bg)
}

fn looks_garbled(t: &str) -> bool {
    let b = t.as_bytes();
    for i in 1..b.len().saturating_sub(1) {
        if b[i] == b'.' && b[i + 1] == b'.' {
            let ellipsis = i + 2 < b.len() && b[i + 2] == b'.';
            if !ellipsis {
                return true;
            }
        }
    }
    let mut run = 0;
    for w in t.split_whitespace() {
        let clean = w.trim_matches(|c: char| !c.is_alphabetic());
        if clean.chars().count() <= 1 {
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

fn warm_score(t: &str) -> usize {
    let l = t.to_ascii_lowercase();
    WARM_WORDS.iter().filter(|w| l.contains(**w)).count()
}

struct Arm {
    ood: Vec<String>,
    ood_collapses: usize,
    distress: Vec<String>,
    distress_warm: usize,
}

fn run_arm(rt: &mut Runtime, label: &str) -> Arm {
    rt.reset_conversation();
    println!("\n========== arm: {} ==========", label);

    let mut ood = Vec::new();
    let mut ood_collapses = 0;
    println!("  -- OOD (each asked 2x) --");
    for p in OOD_PROMPTS {
        let a = rt.converse(p).expect("converse").text;
        let b = rt.converse(p).expect("converse").text;
        let tag = if a == b {
            ood_collapses += 1;
            "  <REPEAT-COLLAPSE>"
        } else {
            ""
        };
        println!("   [{}]\n      1: {}\n      2: {}{}", p, a, b, tag);
        ood.push(a);
        ood.push(b);
    }

    rt.reset_conversation();
    let mut distress = Vec::new();
    let mut distress_warm = 0;
    println!("  -- distress (warmth preference) --");
    for p in DISTRESS_PROMPTS {
        let r = rt.converse(p).expect("converse").text;
        let w = warm_score(&r);
        distress_warm += w;
        println!("   [{}] (warm={}) {}", p, w, r);
        distress.push(r);
    }

    Arm { ood, ood_collapses, distress, distress_warm }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let brain_path = PathBuf::from(args.next().unwrap_or_else(|| DEFAULT_BRAIN.to_string()));
    let toml_path = PathBuf::from(args.next().unwrap_or_else(|| DEFAULT_TOML.to_string()));

    eprintln!("loading brain: {}", brain_path.display());
    let bytes = std::fs::read(&brain_path).expect("read brain bin");
    let mut rt = Runtime::from_brain_bytes(&bytes).expect("load brain");

    let base = std::fs::read_to_string(format!("{}/knowledge_graph.toml", PETS_DATA))
        .expect("read topic graph base");
    let overlay =
        std::fs::read_to_string(format!("{}/knowledge_graph_pet_overlay.toml", PETS_DATA)).ok();
    let base_graph =
        growformer::topic_graph::TopicGraph::from_toml(&base).expect("parse topic graph base");
    let final_graph = match overlay {
        Some(ov) => {
            let og = growformer::topic_graph::TopicGraph::from_toml_quiet(&ov)
                .expect("parse overlay");
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

    let off_toml = ensure_flags(&base_toml, false);
    growformer::inference::inference_toml::reload_inference_toml_from_str(&off_toml)
        .expect("reload (bg off)");
    rt.apply_loaded_generation_config();
    let off = run_arm(&mut rt, "basal_ganglia OFF (control)");

    let on_toml = ensure_flags(&base_toml, true);
    growformer::inference::inference_toml::reload_inference_toml_from_str(&on_toml)
        .expect("reload (bg on)");
    rt.apply_loaded_generation_config();
    let on = run_arm(&mut rt, "basal_ganglia ON");

    let uniq = |v: &[String]| v.iter().cloned().collect::<HashSet<_>>().len();
    let garble = |v: &[String]| v.iter().filter(|t| looks_garbled(t)).count();

    println!("\n================ SUMMARY ================");
    println!(
        "OOD unique responses     : OFF={} ON={}  (of {})",
        uniq(&off.ood),
        uniq(&on.ood),
        off.ood.len()
    );
    println!(
        "OOD repeat-collapses     : OFF={}/{} ON={}/{}",
        off.ood_collapses,
        OOD_PROMPTS.len(),
        on.ood_collapses,
        OOD_PROMPTS.len()
    );
    println!(
        "distress total warm-words: OFF={} ON={}  (higher = warmer)",
        off.distress_warm, on.distress_warm
    );
    println!(
        "garbled responses        : OFF={} ON={}",
        garble(&off.ood) + garble(&off.distress),
        garble(&on.ood) + garble(&on.distress)
    );

    let total_garble = garble(&on.ood) + garble(&on.distress);
    let verdict = if total_garble > 0 {
        "FAIL — garble leaked under BG"
    } else if on.ood_collapses <= off.ood_collapses
        && uniq(&on.ood) >= uniq(&off.ood)
        && on.distress_warm >= off.distress_warm
    {
        "PASS — BG reduces collapse, holds/raises variety + warmth, coherent"
    } else {
        "MIXED — inspect arms (effect gated by candidate diversity)"
    };
    println!("verdict                  : {}", verdict);
}
