//! Demo: compose free-text responses from a fragment library across runtime states.
//!
//! Usage:
//!   cargo run --example fragment_compose_demo -- <path/to/luna_fragments_v1.jsonl>
//!
//! Prints, for a few (intent, state) scenarios, several seeded compositions so
//! you can see state-gated, OCEAN-weighted variety emerge from typed fragments
//! instead of a single canned response.

use std::collections::HashMap;

use growformer::fragment_composer::{ComposeContext, FragmentComposer};
use growformer::reflective_field::ReflectiveWeights;

fn state(pairs: &[(&str, f32)]) -> HashMap<String, f32> {
    pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
}

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: fragment_compose_demo <fragments.jsonl>");
        std::process::exit(2);
    });

    let (lib, skipped) = FragmentComposer::from_path(&path).unwrap_or_else(|e| {
        eprintln!("failed to load {path}: {e}");
        std::process::exit(1);
    });
    println!("loaded {} fragments ({} skipped)\n", lib.fragments.len(), skipped);

    // OCEAN for the cheerful, extraverted Luna.
    let ocean = [0.7, 0.5, 0.8, 0.7, 0.4];

    let scenarios: &[(&str, &str, HashMap<String, f32>)] = &[
        (
            "greeting_check_in / HUNGRY",
            "greeting_check_in",
            state(&[("hunger", 0.9), ("energy", 0.55), ("mood", 0.5)]),
        ),
        (
            "greeting_check_in / SLEEPY",
            "greeting_check_in",
            state(&[("hunger", 0.3), ("energy", 0.12), ("mood", 0.6)]),
        ),
        (
            "greeting_check_in / ZOOMIES",
            "greeting_check_in",
            state(&[("hunger", 0.25), ("energy", 0.95), ("mood", 0.9)]),
        ),
        (
            "open_ended_chat / CONTENT",
            "open_ended_chat",
            state(&[("hunger", 0.2), ("energy", 0.6), ("mood", 0.85)]),
        ),
    ];

    for (label, intent, st) in scenarios {
        println!("=== {label} ===");
        for seed in 0..3u64 {
            let ctx = ComposeContext {
                intent: intent.to_string(),
                graph_anchors: vec![],
                ocean,
                state: st.clone(),
                weights: ReflectiveWeights::default(),
                archetype: Some("cheerful_companion".to_string()),
                seed: seed.wrapping_mul(0x9e3779b97f4a7c15).wrapping_add(1),
            };
            match lib.compose(&ctx) {
                Some(r) => println!(
                    "  [{} voices] {}\n     ({})",
                    r.voices_used,
                    r.text,
                    r.fragment_ids.join(", ")
                ),
                None => println!("  (no eligible composition)"),
            }
        }
        println!();
    }
}
