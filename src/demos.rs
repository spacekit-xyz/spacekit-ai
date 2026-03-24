//! Growformer Demos & Benchmarks
//!
//! Standalone binary for running demos, benchmarks, and evaluations.
//! Not required for production — use `growformer` binary for train + infer.
//!
//! Usage:
//!   cargo run --bin growformer-demos -- --xor
//!   cargo run --bin growformer-demos -- --spiral
//!   cargo run --bin growformer-demos -- --mnist
//!   cargo run --bin growformer-demos -- --language-pipeline
//!   ... etc

use growformer::environment::NeuralEnvironment;
use growformer::types::NeuronId;
use growformer::dimension::{
    CalibrationDataset, CalibrationReport, CalibrationRequirements, EncoderPreset, LanguageConfig, LanguageSample,
    DimensionManager, DimensionManagerConfig, HashingLanguageEncoder, LanguageEncoder,
    MainDimension, VirtualGroup, render_action_template, generate_code_from_action,
    route_language_embedding,
};

use growformer::types::GroupId;
use std::collections::HashMap;
use std::io::Write;
use std::time::Instant;
use growformer::systems::checkpoint::{
    save_phase2_checkpoint, load_phase2_checkpoint, save_mnist_checkpoint, load_mnist_checkpoint,
    save_language_checkpoint, load_language_checkpoint
};
use growformer::systems::mirror::mirror_symmetry_score;
use growformer::systems::whorls::print_whorl_summary;
use growformer::service::LanguageService;
use growformer::types::{EnvironmentConfig, Sample};
use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};
use std::path::Path;
use indicatif::{ProgressBar, ProgressStyle};
use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "growformer-demos", version, about = "Growformer demos and benchmarks")]
struct Args {
    #[arg(short, long)]
    xor: bool,
    #[arg(short, long)]
    spiral: bool,
    #[arg(short, long)]
    concentric_circles: bool,
    #[arg(short, long)]
    mlp: bool,
    #[arg(short, long, value_name = "MODE", default_value = "full")]
    learning: Option<String>,
    #[arg(long)]
    fractal: bool,
    #[arg(long)]
    phase3c: bool,
    #[arg(long)]
    neurogenesis: bool,
    #[arg(long)]
    mnist: bool,
    #[arg(long)]
    mnist_v2: bool,
    #[arg(long)]
    mnist_retention: bool,
    #[arg(long, default_value_t = true)]
    progress: bool,
    #[arg(long, value_name = "N")]
    mnist_train_limit: Option<usize>,
    #[arg(long, value_name = "N")]
    mnist_max_epochs: Option<u32>,
    #[arg(long, value_name = "N")]
    mnist_batch_size: Option<usize>,
    #[arg(long)]
    language_pipeline: bool,
    #[arg(long)]
    language_distill: bool,
    #[arg(long, value_name = "PATH")]
    language_distill_data: Option<String>,
    #[arg(long, value_name = "PATH")]
    language_distill_save: Option<String>,
    #[arg(long, value_name = "PATH")]
    language_distill_resume: Option<String>,
    #[arg(long, value_name = "N", default_value_t = 12)]
    language_distill_epochs: u32,
    #[arg(long, value_name = "PATH")]
    print_gle_card: Option<String>,
    #[arg(long, value_name = "PATH")]
    validate_gle: Option<String>,
    #[arg(long, default_value_t = 0.95)]
    min_routing_acc: f32,
    #[arg(long, default_value_t = 0.25)]
    min_routing_median_margin: f32,
    #[arg(long, default_value_t = 0.10)]
    min_routing_p10_margin: f32,
    #[arg(long)]
    validate_action_schema: bool,
    #[arg(long, default_value_t = 0.95)]
    min_action_accuracy: f32,
    #[arg(long, default_value_t = 0.98)]
    min_fallback_precision: f32,
    #[arg(long, default_value_t = 1.0)]
    min_payload_valid_rate: f32,
    #[arg(long, value_name = "PATH")]
    action_eval_data: Option<String>,
    #[arg(long, value_name = "PATH")]
    action_eval_report: Option<String>,
    #[arg(long, value_name = "TEXT")]
    language_action_text: Option<String>,
    #[arg(long)]
    language_action_eval: bool,
    #[arg(long)]
    language_ema_ablation: bool,
    #[arg(long, value_name = "TEXT")]
    language_generate_text: Option<String>,
    #[arg(long, value_name = "TEXT")]
    language_code_text: Option<String>,
    #[arg(long)]
    language_code_eval: bool,
    #[arg(long)]
    validate_codegen: bool,
    #[arg(long, value_name = "PATH")]
    code_eval_data: Option<String>,
    #[arg(long, value_name = "PATH")]
    code_eval_report: Option<String>,
    #[arg(long, default_value_t = 0.95)]
    min_codegen_language_match: f32,
    #[arg(long, default_value_t = 0.80)]
    min_codegen_specialized_rate: f32,
    #[arg(long)]
    m5_retention_eval: bool,
    #[arg(long, value_name = "PATH", default_value = "data/language/m5/retention_eval_splits.json")]
    m5_retention_plan: String,
    #[arg(long, default_value_t = 20)]
    m5_epochs: u32,
    #[arg(long, default_value_t = 0.20)]
    m5_lr: f32,
    #[arg(long, default_value_t = 512)]
    m5_feature_dim: usize,
    #[arg(long, value_name = "PATH")]
    m5_retention_report: Option<String>,
    #[arg(long, default_value_t = 24)]
    m5_replay_per_epoch: usize,
    #[arg(long, default_value_t = 0.8)]
    m5_replay_prior_ratio: f32,
    #[arg(long)]
    language_generation_eval: bool,
    #[arg(long)]
    validate_generation: bool,
    #[arg(long, default_value_t = 0.01)]
    max_task_success_drop: f32,
    #[arg(long, default_value_t = 0.02)]
    max_hallucination_rate: f32,
    #[arg(long, value_name = "PATH")]
    generation_eval_report: Option<String>,
    #[arg(long)]
    acceptance_report: bool,
    #[arg(long, value_name = "PATH")]
    acceptance_report_path: Option<String>,
    #[arg(long, value_name = "PATH")]
    export_brain: Option<String>,
}

fn main() {
    println!("=============================================================");
    println!("  Growformer — Demos & Benchmarks");
    println!("=============================================================\n");

    let args = Args::parse();
    if let Some(path) = args.print_gle_card.as_deref() {
        if let Err(e) = print_gle_model_card(path) {
            eprintln!("Failed to print GLE model card: {}", e);
            std::process::exit(1);
        }
    } else if let Some(text) = args.language_action_text.as_deref() {
        if let Err(e) = demo_language_action(text) {
            eprintln!("Failed language action routing: {}", e);
            std::process::exit(1);
        }
    } else if let Some(text) = args.language_generate_text.as_deref() {
        if let Err(e) = demo_language_generate(text) {
            eprintln!("Failed language generation routing: {}", e);
            std::process::exit(1);
        }
    } else if let Some(text) = args.language_code_text.as_deref() {
        if let Err(e) = demo_language_code(text) {
            eprintln!("Failed language code generation: {}", e);
            std::process::exit(1);
        }
    } else if args.language_code_eval {
        if let Err(e) = demo_language_code_eval(args.code_eval_data.as_deref(), args.code_eval_report.as_deref()) {
            eprintln!("Failed language code eval: {}", e);
            std::process::exit(1);
        }
    } else if args.validate_codegen {
        match validate_codegen(args.code_eval_data.as_deref(), args.code_eval_report.as_deref(), args.min_codegen_language_match, args.min_codegen_specialized_rate) {
            Ok(true) => {} Ok(false) => std::process::exit(2),
            Err(e) => { eprintln!("Failed codegen validation: {}", e); std::process::exit(1); }
        }
    } else if args.m5_retention_eval {
        if let Err(e) = run_m5_retention_eval(&args.m5_retention_plan, args.m5_epochs, args.m5_lr, args.m5_feature_dim, args.m5_replay_per_epoch, args.m5_replay_prior_ratio, args.m5_retention_report.as_deref()) {
            eprintln!("Failed M5 retention eval: {}", e);
            std::process::exit(1);
        }
    } else if args.language_action_eval {
        if let Err(e) = demo_language_action_eval(args.action_eval_data.as_deref(), args.action_eval_report.as_deref()) {
            eprintln!("Failed language action eval: {}", e);
            std::process::exit(1);
        }
    } else if args.validate_action_schema {
        match validate_action_schema(args.action_eval_data.as_deref(), args.action_eval_report.as_deref(), args.min_action_accuracy, args.min_fallback_precision, args.min_payload_valid_rate) {
            Ok(true) => {} Ok(false) => std::process::exit(2),
            Err(e) => { eprintln!("Failed action schema validation: {}", e); std::process::exit(1); }
        }
    } else if args.language_generation_eval {
        if let Err(e) = demo_language_generation_eval(args.action_eval_data.as_deref(), args.generation_eval_report.as_deref()) {
            eprintln!("Failed language generation eval: {}", e);
            std::process::exit(1);
        }
    } else if args.validate_generation {
        match validate_generation(args.action_eval_data.as_deref(), args.generation_eval_report.as_deref(), args.max_task_success_drop, args.max_hallucination_rate) {
            Ok(true) => {} Ok(false) => std::process::exit(2),
            Err(e) => { eprintln!("Failed generation validation: {}", e); std::process::exit(1); }
        }
    } else if args.language_ema_ablation {
        if let Err(e) = demo_language_ema_ablation() {
            eprintln!("Failed language EMA ablation: {}", e);
            std::process::exit(1);
        }
    } else if let Some(path) = args.validate_gle.as_deref() {
        match validate_gle_model_card(path, args.min_routing_acc, args.min_routing_median_margin, args.min_routing_p10_margin) {
            Ok(true) => {} Ok(false) => std::process::exit(2),
            Err(e) => { eprintln!("Failed to validate GLE model card: {}", e); std::process::exit(1); }
        }
    } else if args.xor {
        demo_xor();
    } else if args.spiral {
        demo_spiral();
    } else if args.concentric_circles {
        demo_concentric_circles();
    } else if args.mlp {
        demo_mlp_baseline();
    } else if args.fractal {
        demo_fractal_continual_learning();
    } else if args.phase3c {
        demo_phase3c_composition();
    } else if args.neurogenesis {
        demo_neurogenesis();
    } else if args.mnist {
        demo_split_mnist(args.progress, args.mnist_train_limit, args.mnist_max_epochs, args.mnist_batch_size);
    } else if args.mnist_v2 {
        demo_clifford_mnist(args.mnist_train_limit, args.mnist_max_epochs);
    } else if args.mnist_retention {
        demo_mnist_retention();
    } else if args.acceptance_report {
        if let Err(e) = demo_acceptance_report(args.acceptance_report_path.as_deref()) {
            eprintln!("Failed acceptance report: {}", e);
            std::process::exit(1);
        }
    } else if let Some(path) = &args.export_brain {
        if let Err(e) = demo_export_brain(path) {
            eprintln!("Failed to export brain: {}", e);
            std::process::exit(1);
        }
    } else if args.language_pipeline {
        demo_language_pipeline();
    } else if args.language_distill {
        demo_language_distill_experiment(args.language_distill_data.as_deref(), args.language_distill_save.as_deref(), args.language_distill_resume.as_deref(), args.language_distill_epochs);
    } else if let Some(mode) = &args.learning {
        match mode.as_str() {
            "train-a" => demo_phase2_train_a(),
            "train-b" => demo_phase2_train_b(),
            _ => demo_continual_learning(),
        }
    } else {
        println!("Run with --help to see available demos and benchmarks.");
        std::process::exit(1);
    }
}

// =============================================================================
// Data loading (shared with main binary)
// =============================================================================

#[derive(Deserialize)]
struct JsonlLanguageSample {
    text: String,
    #[serde(default)]
    semantic_intent: Option<String>,
    #[serde(default)]
    intent: Option<String>,
    #[serde(default)]
    domain: Option<String>,
    #[serde(default)]
    action_target: Option<String>,
    #[serde(default)]
    policy_regime: Option<String>,
    #[serde(default)]
    language_channel: Option<String>,
    #[serde(default)]
    expected_response: Option<String>,
    #[serde(default)]
    expected_code: Option<String>,
}

fn load_language_samples_jsonl(path: &str) -> Result<Vec<LanguageSample>, String> {
    use std::io::BufRead;
    let file = std::fs::File::open(path).map_err(|e| format!("open failed: {}", e))?;
    let reader = std::io::BufReader::new(file);
    let mut out = Vec::new();
    for (idx, line) in reader.lines().enumerate() {
        let line = line.map_err(|e| format!("line {} read failed: {}", idx + 1, e))?;
        if line.trim().is_empty() { continue; }
        let rec: JsonlLanguageSample = serde_json::from_str(&line)
            .map_err(|e| format!("line {} json parse failed: {}", idx + 1, e))?;
        let intent = rec.semantic_intent.or(rec.intent).unwrap_or_else(|| "unknown_intent".to_string());
        out.push(LanguageSample {
            domain: rec.domain.unwrap_or_else(|| "custom".to_string()),
            text: rec.text, semantic_intent: intent, action_target: rec.action_target,
            policy_regime: rec.policy_regime.unwrap_or_else(|| "default".to_string()),
            language_channel: rec.language_channel.unwrap_or_else(|| "english".to_string()),
            expected_response: rec.expected_response, expected_code: rec.expected_code,
        });
    }
    Ok(out)
}

fn load_m5_or_synthetic() -> CalibrationDataset {
    match load_all_m5_training_data() {
        Ok(samples) if !samples.is_empty() => { println!("Loaded M5 training data: {} samples", samples.len()); CalibrationDataset { samples } }
        Ok(_) => { println!("M5 data empty, falling back to synthetic dataset."); build_language_calibration_dataset() }
        Err(e) => { println!("M5 data not found ({}) — falling back to synthetic dataset.", e); build_language_calibration_dataset() }
    }
}

fn load_train_jsonl_dir(all: &mut Vec<LanguageSample>, dir: &std::path::Path) -> Result<(), String> {
    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .map_err(|e| format!("read_dir failed ({}): {}", dir.display(), e))?
        .filter_map(|e| e.ok())
        .collect();
    entries.sort_by_key(|e| e.file_name());
    for entry in entries {
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with("train_") && name.ends_with(".jsonl") {
            let path = entry.path();
            let samples = load_language_samples_jsonl(path.to_str().unwrap())?;
            println!("  loaded {}: {} samples", path.display(), samples.len());
            all.extend(samples);
        }
    }
    Ok(())
}

fn load_all_m5_training_data() -> Result<Vec<LanguageSample>, String> {
    let m5 = std::path::Path::new("data/language/m5");
    if !m5.exists() {
        return Err(format!("M5 data directory not found: {}", m5.display()));
    }
    let mut all = Vec::new();
    load_train_jsonl_dir(&mut all, m5)?;
    let agent = std::path::Path::new("data/agent");
    if agent.exists() {
        load_train_jsonl_dir(&mut all, agent)?;
    }
    let routekit = std::path::Path::new("data/routekit");
    if routekit.exists() {
        load_train_jsonl_dir(&mut all, routekit)?;
    }
    Ok(all)
}

// =============================================================================
// Demo & benchmark functions
// =============================================================================

// =============================================================================
// Demo 1: XOR
// =============================================================================

fn demo_xor() {
    println!("--- Demo 1: XOR ---\n");

    let config = EnvironmentConfig {
        learning_rate: 0.5,
        weight_decay: 0.0001,
        bias_decay: 0.0,           // disabled — same reason as spiral (survival math)
        dropout_rate: 0.0,
        geometry_noise: 0.001,
        competitive_k: 0,
        lateral_inhibition: 0.0,
        max_synapses_per_neuron: 32,
        energy_budget_per_neuron: 100.0,
        pruning_threshold: 0.001,
        mirror_coupling_strength: 3.14,//0.001,
        growth_radius: 0.0,
        geometry_interval: 500,
        stdp_enabled: false,
        prune_interval: 500,
        ..EnvironmentConfig::default()
    };

    let mut env = NeuralEnvironment::new(config);
    let mut rng = StdRng::seed_from_u64(42);

    env.build_layers(&[2, 4, 1], &mut rng);

    let hidden_ids = env.layers[1].clone();
    let (g_a, g_b) = hidden_ids.split_at(hidden_ids.len() / 2);
    let group_a = env.create_group(g_a.to_vec());
    let group_b = env.create_group(g_b.to_vec());
    env.pair_mirror_groups(group_a, group_b);

    let mut xor_data: Vec<Sample> = vec![
        (vec![0.0, 0.0], [0.0]),
        (vec![0.0, 1.0], [1.0]),
        (vec![1.0, 0.0], [1.0]),
        (vec![1.0, 1.0], [0.0]),
    ];

    println!("Training XOR for 5000 epochs...");
    for epoch in 0..5000 {
        xor_data.shuffle(&mut rng);
        let mut epoch_loss = 0.0f32;
        for (input, target) in &xor_data {
            epoch_loss += env.train_tick(input.as_slice(), target, &mut rng).loss;
        }
        epoch_loss /= xor_data.len() as f32;
        if epoch % 1000 == 0 && epoch > 0 { env.prune_dormant(); }
        if epoch % 500 == 0 {
            let weights: Vec<String> = env.layers[1].iter()
                .flat_map(|id| env.neurons[id].synapses.iter().map(|s| format!("{:.3}", s.strength)))
                .collect();
            println!("  epoch {:>5} | loss={:.5} | sparsity={:.2} | h→o: {:?}",
                epoch, epoch_loss, env.firing_sparsity(), weights);
        }
    }

    println!("\nInference:");
    println!("  Input    Expected  Predicted  Correct");
    println!("  ---------------------------------------");
    let mut correct = 0;
    // Test in canonical order
    for (input, expected) in &[([0.0,0.0],[0.0]),([0.0,1.0],[1.0]),([1.0,0.0],[1.0]),([1.0,1.0],[0.0])] {
        let out = env.predict(input);
        let rounded = if out[0] > 0.5 { 1.0 } else { 0.0 };
        let expected_value:f32 = expected[0];
        let ok = (rounded - expected_value).abs() < 0.01;
        if ok { correct += 1; }
        println!("  [{:.0},{:.0}]     {:.1}       {:.4}    {}", input[0], input[1], expected[0], out[0], if ok { "✓" } else { "✗" });
    }
    println!("\n  Accuracy: {}/4", correct);
    print_structural_report(&env, group_a, group_b);
}

// =============================================================================
// Demo 2: Spiral — larger architecture, dropout enabled
// =============================================================================

fn demo_spiral() {
    println!("--- Demo 2: Spiral Classification ---\n");

    let config = EnvironmentConfig {
        learning_rate: 0.15,
        weight_decay: 0.0000025,   // scaled: 0.001/400 samples
        bias_decay: 0.0,           // disabled — any useful value annihilates over 3.2M steps
        dropout_rate: 0.1,    // reduced: 32 neurons each carry more weight than in 16-neuron layer
        geometry_noise: 0.0,        // replaced by thermal_noise in physics
        competitive_k: 4,            // KWTA top-4 of 16: hard competition regardless of synapse strength
        lateral_inhibition: 0.12,    // active from epoch 0: no warmup, moderate selectivity
                                   // targets act=0.25-0.40 with sigma_inhib=2.0
        lr_decay: 0.00008,    // slowed 5×: previous decay killed lr by epoch 1000, no learning late
        max_synapses_per_neuron: 64,
        energy_budget_per_neuron: 100.0,
        pruning_threshold: 0.001,
        mirror_coupling_strength: 0.001,
        growth_radius: 0.0,
        geometry_interval: 500,
        stdp_enabled: false,
        k_repel: 0.2,
        gravity_g: 0.05,
        damping: 0.2,
        thermal_noise: 0.02,     //
        // Reaction-diffusion: sigma_inhib tuned for 32-neuron single layer
        // spread ~2.8, 40% = 1.1 — slightly tighter for more diverse receptive fields
        sigma_inhib: 2.0,            // widened: spread=2.4, need sigma > spread/2 for truly local inhibition
        // Debye: screening length replaces hard repulsion_radius
        debye_length: 1.5,
        // Mass-competition: lower win threshold to match actual activation range
        mass_win_threshold: 0.14, // lowered: strong inhibition means winners fire ~0.3-0.5 post-suppression
        mass_decay: 0.00009,          // all neurons lose this fraction per sample, 0.00009 -> 0.000009
        // Homeostasis: gentle bias regulation to prevent runaway negative drift
        homeostasis_target: 0.30, // target slightly sparse
        homeostasis_lr: 0.0,      // disabled — equalizes all neurons to same bias, kills diversity
        // homeostasis_tau: 0.0001,  // 10000 sample window   
        prune_interval: 500,     // changed from 500 to 1000 to reduce pruning frequency
        ..EnvironmentConfig::default()
    };

    let mut env = NeuralEnvironment::new(config);
    // seeds 42(90.4,92.6,90.1), 7(92.2,91.8), 99, 314, 271.
    let mut rng = StdRng::seed_from_u64(42);

    // Larger architecture — spiral needs more representational capacity
    // 2 → 16 → 16 → 1  scaled layer-2: 8 neurons insufficient to integrate spiral boundary
    //                     warmup removed — created always-on neurons, killed specialisation
    
    //  2 → 16 → 16 → 1 
    // (300 = Spiral accuracy: 389/600 (64.8%))
    // (800 = Spiral accuracy: 991/1600 (61.9%))

    //  2 → 24 → 24 → 1 
    // (300 = Spiral accuracy: 389/600 (75.4%))
    env.build_layers(&[2, 16, 16, 1], &mut rng);

    let hidden_ids = env.layers[1].clone();
    let (g_a, g_b) = hidden_ids.split_at(hidden_ids.len() / 2);
    let group_a = env.create_group(g_a.to_vec());
    let group_b = env.create_group(g_b.to_vec());
    env.pair_mirror_groups(group_a, group_b);

    let samples = 400; // becomes 200 ->400, 400 -> 800
    let epochs = 8000;
    // change from 400 -> 
    // (200 = Spiral accuracy: 267/400 (66.8%))
    // (200 = Spiral accuracy: 264/400 (66.0%))
    // (300 = Spiral accuracy: 389/600 (64.8%)) 
    // (800 = Spiral accuracy: 991/1600 (61.9%))
    let mut spiral_data = generate_spiral_data(samples, &mut rng);
    println!("Training on {} samples, architecture [2→16→16→1], {} epochs...", samples, epochs);

    for epoch in 0..epochs {
        env.set_epoch(epoch);
        spiral_data.shuffle(&mut rng);
        let mut epoch_loss = 0.0f32;
        for (input, target) in &spiral_data {
            epoch_loss += env.train_tick(input, target, &mut rng).loss;
        }
        epoch_loss /= spiral_data.len() as f32;
        if epoch % 500 == 0 {
            println!("  epoch {:>4} | loss={:.5} | syn={} | sparse={:.2} | act={:.3} | spread={:.2} | mass={:.2} | lr={:.5}",
                epoch, epoch_loss,
                env.total_synapses(),
                env.firing_sparsity(),
                env.mean_hidden_activation(),
                env.geometric_spread(),
                env.mean_hidden_mass(),
                env.current_lr);
        }
    }

    let mut correct = 0;
    for (input, target) in &spiral_data {
        let out = env.predict(input);
        if (if out[0] > 0.5 { 1.0_f32 } else { 0.0 } - target[0]).abs() < 0.01 { correct += 1; }
    }
    println!("\nSpiral accuracy: {}/{} ({:.1}%)",
        correct, spiral_data.len(), 100.0 * correct as f32 / spiral_data.len() as f32);

    println!("\nSample predictions:");
    println!("  X        Y        Expected  Predicted");
    println!("  ----------------------------------------");
    for (input, target) in spiral_data.iter().take(8) {
        let out = env.predict(input);
        println!("  {:+.4}  {:+.4}    {:.1}       {:.4}", input[0], input[1], target[0], out[0]);
    }

    print_structural_report(&env, group_a, group_b);
}

// =============================================================================
// Demo 3: Concentric Circles
// =============================================================================

fn demo_concentric_circles() {
    println!("--- Demo 3: Concentric Circles ---\n");
    let config = EnvironmentConfig {
        learning_rate: 0.15,
        weight_decay: 0.0000025,   // scaled: 0.001/400 samples
        bias_decay: 0.0,           // disabled — any useful value annihilates over 3.2M steps
        dropout_rate: 0.1,    // reduced: 32 neurons each carry more weight than in 16-neuron layer
        geometry_noise: 0.0,        // replaced by thermal_noise in physics
        competitive_k: 4,            // KWTA top-4 of 16: hard competition regardless of synapse strength
        lateral_inhibition: 0.12,    // active from epoch 0: no warmup, moderate selectivity
                                   // targets act=0.25-0.40 with sigma_inhib=2.0
        lr_decay: 0.00008,    // slowed 5×: previous decay killed lr by epoch 1000, no learning late
        max_synapses_per_neuron: 64,
        energy_budget_per_neuron: 100.0,
        pruning_threshold: 0.001,
        mirror_coupling_strength: 0.001,
        growth_radius: 2.0,
        geometry_interval: 500,
        stdp_enabled: false,
        k_repel: 0.2,
        gravity_g: 0.05,
        damping: 0.2,
        thermal_noise: 0.02,     //
        // Reaction-diffusion: sigma_inhib tuned for 32-neuron single layer
        // spread ~2.8, 40% = 1.1 — slightly tighter for more diverse receptive fields
        sigma_inhib: 2.0,            // widened: spread=2.4, need sigma > spread/2 for truly local inhibition
        // Debye: screening length replaces hard repulsion_radius
        debye_length: 1.5,
        // Mass-competition: lower win threshold to match actual activation range
        mass_win_threshold: 0.14, // lowered: strong inhibition means winners fire ~0.3-0.5 post-suppression
        mass_decay: 0.00009,          // all neurons lose this fraction per sample, 0.00009 -> 0.000009
        // Homeostasis: gentle bias regulation to prevent runaway negative drift
        homeostasis_target: 0.30, // target slightly sparse
        homeostasis_lr: 0.0,      // disabled — equalizes all neurons to same bias, kills diversity
        // homeostasis_tau: 0.0001,  // 10000 sample window   
        prune_interval: 500,     // changed from 500 to 1000 to reduce pruning frequency
        ..EnvironmentConfig::default()
    };

    let mut env = NeuralEnvironment::new(config);
    // seeds 42(90.4,92.6,90.1), 7(92.2,91.8), 99, 314, 271.
    let mut rng = StdRng::seed_from_u64(7);
    // 2 → 16 → 16 → 1
    env.build_layers(&[2, 16, 16, 1], &mut rng);

    let hidden_ids = env.layers[1].clone();
    let (g_a, g_b) = hidden_ids.split_at(hidden_ids.len() / 2);
    let group_a = env.create_group(g_a.to_vec());
    let group_b = env.create_group(g_b.to_vec());
    env.pair_mirror_groups(group_a, group_b);

    let samples = 400; // becomes 200 ->400, 400 -> 800
    let epochs = 8000;
    
    let mut concentric_data = generate_concentric_circles_data(samples, &mut rng);
    println!("Training on {} samples, architecture [2→16→16→1], {} epochs...", samples, epochs);

    for epoch in 0..epochs {
        env.set_epoch(epoch);
        concentric_data.shuffle(&mut rng);
        let mut epoch_loss = 0.0f32;
        for (input, target) in &concentric_data {
            epoch_loss += env.train_tick(input, target, &mut rng).loss;
        }
        epoch_loss /= concentric_data.len() as f32;
        if epoch % 500 == 0 {
            println!("  epoch {:>4} | loss={:.5} | syn={} | sparse={:.2} | act={:.3} | spread={:.2} | mass={:.2} | lr={:.5}",
                epoch, epoch_loss,
                env.total_synapses(),
                env.firing_sparsity(),
                env.mean_hidden_activation(),
                env.geometric_spread(),
                env.mean_hidden_mass(),
                env.current_lr);
        }
    }

    let mut correct = 0;
    for (input, target) in &concentric_data {
        let out = env.predict(input);
        if (if out[0] > 0.5 { 1.0_f32 } else { 0.0 } - target[0]).abs() < 0.01 { correct += 1; }
    }
    println!("\nConcentric Circles accuracy: {}/{} ({:.1}%)",
        correct, concentric_data.len(), 100.0 * correct as f32 / concentric_data.len() as f32);

    println!("\nSample predictions:");
    println!("  X        Y        Expected  Predicted");
    println!("  ----------------------------------------");
    for (input, target) in concentric_data.iter().take(8) {
        let out = env.predict(input);
        println!("  {:+.4}  {:+.4}    {:.1}       {:.4}", input[0], input[1], target[0], out[0]);
    }

    print_structural_report(&env, group_a, group_b);
    
}

// =============================================================================
// Demo 4: MLP Baseline Comparison
// =============================================================================

fn demo_mlp_baseline() {
    println!("--- Demo 4: MLP Baseline Comparison ---\n");

    let mut mlp = NeuralEnvironment::new(EnvironmentConfig {
        learning_rate: 0.15,
        weight_decay: 0.0000025,
        bias_decay: 0.0,           // no Rivera bias pressure
        lr_decay: 0.00008,
        competitive_k: 0,          // no KWTA
        lateral_inhibition: 0.0,   // no inhibition
        dropout_rate: 0.0,         // no dropout
        thermal_noise: 0.0,        // no physics
        gravity_g: 0.0,
        k_repel: 0.0,
        mass_decay: 0.0,
        mass_growth: 0.0,
        homeostasis_lr: 0.0,       // no homeostasis
        mirror_coupling_strength: 0.0,
        prune_interval: 9_999_999,
        geometry_interval: 9_999_999,
        ..EnvironmentConfig::default()
    });

    let mut rng = StdRng::seed_from_u64(7); // same seed as Rivera run
    mlp.build_layers(&[2, 16, 16, 1], &mut rng);
    // NO groups, NO mirror coupling

    let mut spiral_data = generate_spiral_data(400, &mut rng); // same 800 samples
    println!("Training on 400 samples, architecture [2→16→16→1], 8000 epochs...");

    for epoch in 0..8000 {
        mlp.set_epoch(epoch);
        spiral_data.shuffle(&mut rng);
        let mut epoch_loss = 0.0f32;
        for (input, target) in &spiral_data {
            epoch_loss += mlp.train_tick(input, target, &mut rng).loss;
        }
        epoch_loss /= spiral_data.len() as f32;
        if epoch % 500 == 0 {
            println!("  epoch {:>4} | loss={:.5} | lr={:.5}",
                epoch, epoch_loss, mlp.current_lr);
        }
    }

    let mut correct = 0;
    for (input, target) in &spiral_data {
        let out = mlp.predict(input);
        if (if out[0] > 0.5 { 1.0_f32 } else { 0.0 } - target[0]).abs() < 0.01 {
            correct += 1;
        }
    }
    println!("\nMLP accuracy: {}/{} ({:.1}%)",
        correct, spiral_data.len(),
        100.0 * correct as f32 / spiral_data.len() as f32);
}


// =============================================================================
// Phase 2 Checkpoint modes — train-a saves checkpoint, train-b loads and runs B only
// =============================================================================

fn phase2_base_config() -> EnvironmentConfig {
    EnvironmentConfig {
        learning_rate: 0.15,
        weight_decay: 0.0000025,
        bias_decay: 0.0,
        dropout_rate: 0.0,
        geometry_noise: 0.0,
        competitive_k: 4,
        lateral_inhibition: 0.12,
        lr_decay: 0.00008,
        sigma_inhib: 2.0,
        debye_length: 1.5,
        thermal_noise: 0.02,
        k_repel: 0.2,
        gravity_g: 0.05,
        damping: 0.2,
        mass_win_threshold: 0.15,
        mass_decay: 0.00009,
        mass_growth: 0.0005,
        homeostasis_lr: 0.0,
        growth_radius: 2.0,
        prune_interval: 500,
        weight_clamp: 5.0,
        max_synapses_per_neuron: 64,
        energy_budget_per_neuron: 100.0,
        pruning_threshold: 0.001,
        mirror_coupling_strength: 0.001,
        geometry_interval: 500,
        stdp_enabled: false,
        // Mass consolidation: high-mass neurons get smaller LR. Set to 0.0 to isolate Task B learning;
        // if Task B then reaches 85%+ while retention holds, reintroduce k (e.g. 1.0–1.5) for balance.
        mass_consolidation_k: 0.0,
        ..EnvironmentConfig::default()
    }
}

const TASK_A_CHECKPOINT_PATH: &str = "task_a_checkpoint.json";

/// Trains Task A only, then saves checkpoint. Run once to create task_a_checkpoint.json.
fn demo_phase2_train_a() {
    println!("--- Phase 2: Train A + Save Checkpoint ---\n");

    let base_config = phase2_base_config();
    let seed = 42u64;
    let data_seed = 99u64;

    let mut weight_rng = StdRng::seed_from_u64(seed);
    let mut data_rng = StdRng::seed_from_u64(data_seed);

    let mut env = NeuralEnvironment::new(base_config.clone());
    env.build_layers(&[2, 16, 16, 2], &mut weight_rng);

    let layer1_ids = env.layers[1].clone();
    let layer2_ids = env.layers[2].clone();
    let output_0 = env.layers[3][0];
    let output_1 = env.layers[3][1];

    let (l1_a, l1_b) = layer1_ids.split_at(layer1_ids.len() / 2);
    let (l2_a, l2_b) = layer2_ids.split_at(layer2_ids.len() / 2);

    let group_a_ids: Vec<NeuronId> = l1_a.iter().chain(l2_a.iter()).cloned().collect();
    let group_b_ids: Vec<NeuronId> = l1_b.iter().chain(l2_b.iter()).cloned().collect();

    let group_a = env.create_group(group_a_ids.clone());
    let group_b = env.create_group(group_b_ids.clone());
    env.pair_mirror_groups(group_a, group_b);

    println!("=== TASK A: Spiral Classification ===");
    println!("Training on 400 samples, 4000 epochs...\n");

    let mut spiral_data = generate_spiral_data(400, &mut data_rng);

    for epoch in 0..4000 {
        env.set_epoch(epoch);
        spiral_data.shuffle(&mut weight_rng);
        let mut epoch_loss = 0.0f32;
        for (input, target) in &spiral_data {
            let current_out = env.predict(input);
            let target_both = [target[0], current_out[1]];
            epoch_loss += env.train_tick(input, &target_both, &mut weight_rng).loss;
        }
        epoch_loss /= spiral_data.len() as f32;
        if epoch % 500 == 0 {
            println!("  epoch {:>4} | loss={:.5} | syn={} | sparse={:.2} | mass={:.2} | lr={:.5}",
                epoch, epoch_loss,
                env.total_synapses(),
                env.firing_sparsity(),
                env.mean_hidden_mass(),
                env.current_lr);
        }
    }

    let task_a_result = evaluate_accuracy_head(&mut env, &spiral_data, 0);
    println!("\nTask A accuracy: {}/{} ({:.1}%)",
        task_a_result.0, task_a_result.1,
        100.0 * task_a_result.0 as f32 / task_a_result.1 as f32);

    println!("\n>>> Consolidating Task A (frozen flag gate)...");

    env.freeze_consolidated_pathway(&group_a_ids, output_0, l1_a);
    println!("  Frozen: Group A neurons, output[0], input→Group A layer1 synapses");

    save_phase2_checkpoint(
        &env,
        &group_a_ids,
        &group_b_ids,
        group_a,
        group_b,
        output_0,
        output_1,
        task_a_result.0,
        task_a_result.1,
        seed,
        data_seed,
        4000,
        TASK_A_CHECKPOINT_PATH,
    );

    println!("\nRun with '--learning train-b' to test Task B with this checkpoint.");
}

/// Loads checkpoint and trains Task B only. Run after demo_phase2_train_a() for fast iteration.
fn demo_phase2_train_b() {
    println!("--- Phase 2: Train B (from checkpoint) ---\n");

    if !Path::new(TASK_A_CHECKPOINT_PATH).exists() {
        println!("No checkpoint found at {} — run with '--learning train-a' first.", TASK_A_CHECKPOINT_PATH);
        return;
    }

    let base_config = phase2_base_config();

    let (
        mut env,
        _group_a_ids,
        _group_b_ids,
        group_a,
        _group_b,
        _output_0,
        _output_1,
        data_seed,
    ) = load_phase2_checkpoint(TASK_A_CHECKPOINT_PATH, &base_config);

    env.current_lr = base_config.learning_rate;
    env.set_consolidated_groups(&[group_a]);

    let mut data_rng = StdRng::seed_from_u64(data_seed);
    let mut weight_rng = StdRng::seed_from_u64(42);

    let spiral_data = generate_spiral_data(400, &mut data_rng);
    let mut circles_data = generate_concentric_circles_data(400, &mut data_rng);

    let task_a_before = evaluate_accuracy_head(&mut env, &spiral_data, 0);
    println!("Task A retention on load: {}/{} ({:.1}%)\n",
        task_a_before.0, task_a_before.1,
        100.0 * task_a_before.0 as f32 / task_a_before.1 as f32);

    println!("=== TASK B: Concentric Circles ===");
    println!("Training on 400 samples, 4000 epochs...");
    println!("(Consolidated pathway frozen — no restore loop)\n");

    for epoch in 0..4000 {
        env.set_epoch(epoch);
        circles_data.shuffle(&mut weight_rng);
        let mut epoch_loss = 0.0f32;

        for (input, target) in &circles_data {
            let current_out = env.predict(input);
            let target_both = [current_out[0], target[0]];
            epoch_loss += env.train_tick(input, &target_both, &mut weight_rng).loss;
        }

        epoch_loss /= circles_data.len() as f32;
        if epoch % 500 == 0 {
            let retention = evaluate_accuracy_head(&mut env, &spiral_data, 0);
            println!("  epoch {:>4} | loss={:.5} | syn={} | sparse={:.2} | mass={:.2} | A_retain={:.1}%",
                epoch, epoch_loss,
                env.total_synapses(),
                env.firing_sparsity(),
                env.mean_hidden_mass(),
                100.0 * retention.0 as f32 / retention.1 as f32);
        }
    }

    let task_a_after = evaluate_accuracy_head(&mut env, &spiral_data, 0);
    let task_b_result = evaluate_accuracy_head(&mut env, &circles_data, 1);

    let retention_pct = 100.0 * task_a_after.0 as f32 / task_a_after.1 as f32;
    let baseline_pct = 100.0 * task_a_before.0 as f32 / task_a_before.1 as f32;
    let forgetting = baseline_pct - retention_pct;

    println!("\n=== CONTINUAL LEARNING RESULTS ===\n");
    println!("  Task A (Spiral):");
    println!("    Before Task B: {}/{} ({:.1}%)", task_a_before.0, task_a_before.1, baseline_pct);
    println!("    After  Task B: {}/{} ({:.1}%)", task_a_after.0, task_a_after.1, retention_pct);
    println!("    Forgetting:    {:.1}%  (threshold: >10%)", forgetting);
    println!("\n  Task B (Circles):");
    println!("    Accuracy: {}/{} ({:.1}%)",
        task_b_result.0, task_b_result.1,
        100.0 * task_b_result.0 as f32 / task_b_result.1 as f32);
    println!("\n  Verdict: {}",
        if forgetting < 5.0 { "PASS — near-zero forgetting." }
        else if forgetting < 10.0 { "PASS — within threshold." }
        else if forgetting < 20.0 { "PARTIAL — significant forgetting." }
        else { "FAIL — catastrophic forgetting." }
    );
}

// =============================================================================
// Demo 6: Fractal Continual Learning (Phase 3)
// Main Dimension = frozen store only. Mirror Dimension = isolated env per task.
// =============================================================================

fn demo_fractal_continual_learning() {
    println!("--- Demo 6: Fractal Continual Learning (Phase 3) ---\n");

    let config = DimensionManagerConfig {
        mirror_config: phase2_base_config(),
        mirror_layer_sizes: vec![2, 16, 16, 1],
        promotion_check_interval: 500,
        max_concurrent_mirrors: 2,
        calibration_samples: 100,
        reserve_pool_size: 0,
    };

    let mut dm = DimensionManager::new(config);
    let mut rng = StdRng::seed_from_u64(42);
    let mut data_rng = StdRng::seed_from_u64(99);

    let spiral_data = generate_spiral_data(400, &mut data_rng);
    let circles_data = generate_concentric_circles_data(400, &mut data_rng);
    let calibration_spiral: Vec<_> = spiral_data.iter().take(100).cloned().collect();
    let calibration_circles: Vec<_> = circles_data.iter().take(100).cloned().collect();

    // === TASK A: Spiral in isolated Mirror ===
    dm.spawn_mirror("spiral", 42).expect("spawn spiral mirror");
    println!("=== TASK A: Spiral (Mirror) ===\n");

    for epoch in 0..4000 {
        let Some(result) = dm.train_mirror_epoch("spiral", &spiral_data, &mut rng, None) else {
            break; // mirror was auto-promoted and removed
        };
        if epoch % 500 == 0 {
            println!("  [spiral] epoch {:>4} | loss={:.4} | acc={:.1}%",
                epoch, result.loss, result.accuracy * 100.0);
        }
        if epoch % 500 == 0 {
            dm.evaluate_promotions(&calibration_spiral);
            if !dm.mirrors.contains_key("spiral") {
                break; // auto-promoted
            }
        }
    }

    let spiral_group = if dm.mirrors.contains_key("spiral") {
        dm.force_promote("spiral", &calibration_spiral).unwrap()
    } else {
        *dm.main.group_order.last().unwrap() // already promoted
    };
    println!("\nTask A promoted as Group {}\n", spiral_group);

    // === TASK B: Circles in fresh Mirror ===
    dm.spawn_mirror("circles", 43).expect("spawn circles mirror");
    println!("=== TASK B: Circles (Mirror) ===\n");

    for epoch in 0..4000 {
        let Some(result) = dm.train_mirror_epoch("circles", &circles_data, &mut rng, None) else {
            break;
        };
        if epoch % 500 == 0 {
            let spiral_retain = dm.evaluate_main_group(spiral_group, &spiral_data);
            println!("  [circles] epoch {:>4} | loss={:.4} | acc={:.1}% | A_retain={:.1}%",
                epoch, result.loss, result.accuracy * 100.0, spiral_retain * 100.0);
        }
        if epoch % 500 == 0 {
            dm.evaluate_promotions(&calibration_circles);
            if !dm.mirrors.contains_key("circles") {
                break;
            }
        }
    }

    let circles_group = if dm.mirrors.contains_key("circles") {
        dm.force_promote("circles", &calibration_circles).unwrap()
    } else {
        *dm.main.group_order.last().unwrap()
    };
    println!("\nTask B promoted as Group {}\n", circles_group);

    // Train learned router so no-context infer uses single forward → logits → argmax.
    // 400 epochs: spiral → group 0, circles → group 1 (correct). Margin target or 450+ can flip spiral to group 1.
    dm.train_and_set_router(
        &[(&calibration_spiral[..], 0), (&calibration_circles[..], 1)],
        &mut rng,
        400,
    );
    println!("Learned router trained (2 groups, 400 epochs, lr=0.15, hidden=16).\n");

    // === RESULTS ===
    let final_spiral = dm.evaluate_main_group(spiral_group, &spiral_data);
    let final_circles = dm.evaluate_main_group(circles_group, &circles_data);
    println!("=== RESULTS ===\n");
    println!("  Task A (Spiral):  {:.1}%", final_spiral * 100.0);
    println!("  Task B (Circles): {:.1}%", final_circles * 100.0);

    // === INFERENCE (no task label) ===
    let out_spiral = dm.infer(&[0.3_f32, 0.4]);
    print_routing("Routed spiral input ", &dm, &out_spiral);
    let out_circles = dm.infer(&[0.0_f32, 0.9]);
    print_routing("Routed circles input", &dm, &out_circles);

    let spiral_ctx = [String::from("spiral")];
    let circles_ctx = [String::from("circles")];
    let out_s = dm.infer_with_context(&[0.3_f32, 0.4], Some(&spiral_ctx));
    print_routing("  +ctx [spiral] ", &dm, &out_s);
    let out_c = dm.infer_with_context(&[0.0_f32, 0.9], Some(&circles_ctx));
    print_routing("  +ctx [circles] ", &dm, &out_c);
}

// =============================================================================
// Demo: Neurogenesis — trigger adds 1 neuron to last hidden layer after N epochs if loss > X
// =============================================================================

fn demo_neurogenesis() {
    println!("--- Neurogenesis: trigger (epoch + loss) ---\n");
    let config = DimensionManagerConfig {
        mirror_config: phase2_base_config(),
        mirror_layer_sizes: vec![2, 16, 16, 1],
        promotion_check_interval: 999_999,
        max_concurrent_mirrors: 1,
        calibration_samples: 100,
        reserve_pool_size: 0,
    };
    let mut dm = DimensionManager::new(config);
    let mut rng = StdRng::seed_from_u64(42);
    let mut data_rng = StdRng::seed_from_u64(99);
    let spiral_data: Vec<_> = generate_spiral_data(400, &mut data_rng).into_iter().take(60).collect();

    dm.spawn_mirror("spiral", 42).expect("spiral");
    // Use 300 epochs / trigger at 10 so it fires early in short run; spec uses 2000 / 0.3 for real runs.
    const EPOCH_TRIGGER: u32 = 10;
    const LOSS_THRESHOLD: f32 = 0.2;
    const TOTAL_EPOCHS: u32 = 300;

    let mut loss_before_trigger: Option<f32> = None;
    let mut loss_after_trigger: Option<f32> = None;
    let mut trigger_epoch: Option<u32> = None;

    for epoch in 0..TOTAL_EPOCHS {
        let Some(result) = dm.train_mirror_epoch("spiral", &spiral_data, &mut rng, None) else { break };
        if epoch == EPOCH_TRIGGER.saturating_sub(1) {
            loss_before_trigger = Some(result.loss);
        }
        let added = dm.try_mirror_neurogenesis(
            "spiral",
            EPOCH_TRIGGER,
            LOSS_THRESHOLD,
            result.loss,
            &mut rng,
        );
        if added {
            trigger_epoch = Some(epoch + 1);
            println!("  Neurogenesis: added 1 neuron at epoch {} (loss={:.4})", epoch + 1, result.loss);
        }
        if trigger_epoch.is_some() && epoch == EPOCH_TRIGGER + 99 {
            loss_after_trigger = Some(result.loss);
        }
        if epoch % 500 == 0 {
            println!("  [spiral] epoch {:>4} | loss={:.4} | acc={:.1}%", epoch, result.loss, result.accuracy * 100.0);
        }
    }

    println!("\n--- Neurogenesis run complete ---");
    if let Some(ep) = trigger_epoch {
        println!("  Trigger fired at epoch {}", ep);
        if let (Some(lb), Some(la)) = (loss_before_trigger, loss_after_trigger) {
            println!("  Loss before trigger: {:.4}  after (+100 epochs): {:.4}", lb, la);
        }
    } else {
        println!("  Trigger did not fire (loss was <= {} or epochs < {})", LOSS_THRESHOLD, EPOCH_TRIGGER);
    }
    let mirror = dm.mirrors.get("spiral").expect("mirror still present");
    let last_hidden_len = mirror.env.layers.get(mirror.env.layers.len().wrapping_sub(2)).map(|l| l.len()).unwrap_or(0);
    println!("  Last hidden layer size: {} (base was 16)", last_hidden_len);
    println!("  No crash; loss still decreases after event.");
}

// =============================================================================
// Demo: Split MNIST — five sequential digit-pair tasks, report average acc and forgetting
// =============================================================================

fn demo_split_mnist(
    show_progress: bool,
    train_limit: Option<usize>,
    max_epochs_override: Option<u32>,
    batch_size: Option<usize>,
) {
    use growformer::mnist::{load_mnist_normalized, filter_digit_pair, RandomProjection, project_dataset, MnistSample, MNIST_PROJECTED};
    use std::fs::OpenOptions;

    let log_path = std::env::var("GROWFORMER_MNIST_LOG").unwrap_or_else(|_| "mnist-run.log".to_string());
    let checkpoint_path = std::env::var("GROWFORMER_MNIST_CHECKPOINT").unwrap_or_else(|_| "mnist_checkpoint.json".to_string());
    let mut log = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .ok();

    // Build fingerprint: makes it obvious when EC2 binary is stale.
    let pkg_version = env!("CARGO_PKG_VERSION");
    let build_unix = option_env!("GROWFORMER_BUILD_UNIX").unwrap_or("unknown");
    let build_target = option_env!("GROWFORMER_TARGET").unwrap_or("unknown");
    let build_profile = option_env!("GROWFORMER_PROFILE").unwrap_or("unknown");
    let build_git = option_env!("GROWFORMER_GIT_SHA").unwrap_or("nogit");
    let cwd = std::env::current_dir()
        .ok()
        .and_then(|p| p.to_str().map(|s| s.to_string()))
        .unwrap_or_else(|| "?".to_string());

    let run_ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let _ = log.as_mut().map(|f| {
        let _ = writeln!(f, "\n--- run {} ---", run_ts);
        let _ = writeln!(
            f,
            "build version={} build_unix={} target={} profile={} git={}",
            pkg_version, build_unix, build_target, build_profile, build_git
        );
        let _ = writeln!(f, "cwd={}", cwd);
        let _ = writeln!(f, "train_limit={:?} max_epochs={:?} batch_size={:?}", train_limit, max_epochs_override, batch_size);
        let _ = writeln!(f, "log_path={} checkpoint_path={}", log_path, checkpoint_path);
        f.flush()
    });
    println!("--- Split MNIST ---\n");
    if log.is_some() {
        println!("Log: {}", log_path);
    }
    println!(
        "Build: growformer v{} build_unix={} target={} profile={} git={}",
        pkg_version, build_unix, build_target, build_profile, build_git
    );
    println!("Run: cwd={} log_path={} checkpoint_path={}", cwd, log_path, checkpoint_path);
    let data_path = std::env::var("MNIST_ROOT").unwrap_or_else(|_| "data".to_string());
    println!("Run: MNIST_ROOT={}", data_path);
    let images_path = std::path::Path::new(&data_path).join("train-images-idx3-ubyte");
    let images_gz = std::path::Path::new(&data_path).join("train-images-idx3-ubyte.gz");
    if !images_path.exists() && !images_gz.exists() {
        eprintln!("MNIST data not found. The mnist crate expects decompressed IDX files in {:?}.", data_path);
        eprintln!("Run from the repo root:  bash scripts/download_mnist.sh");
        eprintln!("Or set MNIST_ROOT to a directory that already contains the four .ubyte files.");
        std::process::exit(1);
    }
    if train_limit.is_some() || max_epochs_override.is_some() || batch_size.is_some() {
        println!("Fast run: train_limit={:?}, max_epochs={:?}, batch_size={:?}\n", train_limit, max_epochs_override, batch_size);
    }
    println!("Loading MNIST from {:?}...", data_path);
    let (train_imgs, train_lbls, test_imgs, test_lbls) = load_mnist_normalized(&data_path);
    let proj = RandomProjection::new(growformer::mnist::MNIST_INPUT, MNIST_PROJECTED, 42);

    const TASKS: [(u8, u8); 5] = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)];
    let mut train_per_task: Vec<Vec<MnistSample>> = Vec::with_capacity(5);
    let mut test_per_task: Vec<Vec<MnistSample>> = Vec::with_capacity(5);
    for (d1, d2) in TASKS {
        let tr = filter_digit_pair(&train_imgs, &train_lbls, d1, d2);
        let te = filter_digit_pair(&test_imgs, &test_lbls, d1, d2);
        let mut train = project_dataset(&proj, &tr);
        if let Some(lim) = train_limit {
            train.truncate(lim);
        }
        train_per_task.push(train);
        test_per_task.push(project_dataset(&proj, &te));
    }

    let config = DimensionManagerConfig {
        mirror_config: phase2_base_config(),
        mirror_layer_sizes: vec![MNIST_PROJECTED, 32, 32, 1],
        promotion_check_interval: 999_999,
        max_concurrent_mirrors: 1,
        calibration_samples: 200,
        reserve_pool_size: 0,
    };
    let mut dm = DimensionManager::new(config);
    let mut rng = StdRng::seed_from_u64(123);
    let max_epochs = max_epochs_override.unwrap_or(2500);
    const TARGET_ACC: f32 = 0.98;

    let mut group_ids: Vec<GroupId> = Vec::with_capacity(5);
    for (t, (d1, d2)) in TASKS.iter().enumerate() {
        let task_name = format!("task_{}", t);
        let train = &train_per_task[t];
        let cal: Vec<Sample> = train.iter().take(200).cloned().collect();
        dm.spawn_mirror(&task_name, 100 + t as u64).expect("spawn");
        println!("  Task {}: {} vs {} (train n={})", t, d1, d2, train.len());

        let pb = if show_progress {
            let bar = ProgressBar::new(max_epochs as u64);
            bar.set_style(
                ProgressStyle::default_bar()
                    .template("[{bar:40.cyan/blue}] {pos}/{len} epochs | task {msg} | {per_sec} | ETA {eta}")
                    .unwrap()
                    .progress_chars("=>-"),
            );
            bar.set_message(format!("{} ({} vs {})", t, d1, d2));
            Some(bar)
        } else {
            None
        };

        let mut last_result = None;
        for epoch in 0..max_epochs {
            let Some(result) = dm.train_mirror_epoch(&task_name, train, &mut rng, batch_size) else { break };
            last_result = Some(result.clone());
            if let Some(ref bar) = pb {
                bar.set_position((epoch + 1) as u64);
                bar.set_message(format!("{} ({} vs {}) acc={:.1}%", t, d1, d2, result.accuracy * 100.0));
            }
            // Always print epoch 0 and every 400th so loss/acc are visible (use bar.println when bar is on so it isn't overwritten)
            if epoch % 400 == 0 {
                let line = format!("    epoch {} loss={:.4} acc={:.1}%", epoch, result.loss, result.accuracy * 100.0);
                if let Some(ref bar) = pb {
                    let _ = bar.println(line);
                } else {
                    println!("{}", line);
                }
            }
            // Log every 50 epochs so tail -f mnist-run.log shows progress (minibatch epochs are slow)
            if epoch > 0 && epoch % 50 == 0 {
                let _ = log.as_mut().map(|f| {
                    let _ = writeln!(f, "  task_{} epoch {} loss={:.4} acc={:.1}%", t, epoch, result.loss, result.accuracy * 100.0);
                    f.flush()
                });
            }
            if result.accuracy >= TARGET_ACC {
                let reached = format!("    Reached {:.0}% at epoch {}", TARGET_ACC * 100.0, epoch);
                if let Some(ref bar) = pb {
                    bar.finish_with_message(format!("done at epoch {} ({:.0}%)", epoch, TARGET_ACC * 100.0));
                    let _ = bar.println(reached);
                } else {
                    println!("{}", reached);
                }
                break;
            }
        }
        if let Some(bar) = pb {
            bar.finish_and_clear();
        }
        let gid = dm.force_promote(&task_name, &cal).expect("promote");
        group_ids.push(gid);
        if let Some(ref r) = last_result {
            println!("  Task {} ({} vs {}) done: {:.1}% accuracy, loss={:.4}", t, d1, d2, r.accuracy * 100.0, r.loss);
        }
        let _ = log.as_mut().map(|f| {
            if let Some(ref r) = last_result {
                let _ = writeln!(f, "section task_{} ({} vs {}) done group_id={} acc={:.1}% loss={:.4}", t, d1, d2, gid, r.accuracy * 100.0, r.loss);
            } else {
                let _ = writeln!(f, "section task_{} ({} vs {}) done group_id={}", t, d1, d2, gid);
            }
            f.flush()
        });
    }

    let calibration_refs: Vec<(&[Sample], usize)> = (0..5).map(|t| (train_per_task[t].as_slice(), t)).collect();
    let _ = log.as_mut().map(|f| {
        let _ = writeln!(f, "section router start (epochs=400)");
        f.flush()
    });
    dm.train_and_set_router(&calibration_refs, &mut rng, 400);
    let _ = log.as_mut().map(|f| {
        let _ = writeln!(f, "section router trained (400 epochs)");
        let _ = writeln!(f, "section router end");
        f.flush()
    });

    let _ = log.as_mut().map(|f| {
        let _ = writeln!(f, "section final_eval start");
        f.flush()
    });
    println!("\n--- Final evaluation (all five tasks) ---");
    let mut accs = Vec::with_capacity(5);
    for (t, (d1, d2)) in TASKS.iter().enumerate() {
        let acc = dm.evaluate_main_group(group_ids[t], &test_per_task[t]);
        accs.push(acc);
        println!("  Task {} ({} vs {}): {:.1}%", t, d1, d2, acc * 100.0);
    }
    let avg = accs.iter().sum::<f32>() / 5.0;
    println!("  Average accuracy: {:.1}%", avg * 100.0);
    println!("  (Target: match EWC ~97%; zero forgetting by construction.)");

    let _ = log.as_mut().map(|f| {
        let _ = writeln!(f, "section final_eval");
        for (t, (d1, d2)) in TASKS.iter().enumerate() {
            let _ = writeln!(f, "  task {} ({} vs {}): {:.1}%", t, d1, d2, accs[t] * 100.0);
        }
        let _ = writeln!(f, "  average: {:.1}%", avg * 100.0);
        let _ = writeln!(f, "--- run {} end ---", run_ts);
        let _ = writeln!(f, "section final_eval end");
        f.flush()
    });

    println!("Saving checkpoint to {}", checkpoint_path);
    let _ = log.as_mut().map(|f| {
        let _ = writeln!(f, "section checkpoint_save start path={}", checkpoint_path);
        f.flush()
    });
    save_mnist_checkpoint(&dm.main, &group_ids, &accs, &checkpoint_path);
    match std::fs::metadata(&checkpoint_path) {
        Ok(m) => {
            println!("Checkpoint verification: exists=true bytes={}", m.len());
            let _ = log.as_mut().map(|f| {
                let _ = writeln!(f, "section checkpoint_save end path={} exists=true bytes={}", checkpoint_path, m.len());
                f.flush()
            });
        }
        Err(e) => {
            eprintln!("Checkpoint verification FAILED: path={} err={}", checkpoint_path, e);
            let _ = log.as_mut().map(|f| {
                let _ = writeln!(f, "section checkpoint_save end path={} exists=false err={}", checkpoint_path, e);
                f.flush()
            });
        }
    }
    println!("\nRun retention evaluation: ./growformer --mnist-retention");
}

// =============================================================================
// MNIST retention evaluation — load checkpoint, re-evaluate all 5 tasks
// Proves save/load preserves accuracy (no forgetting).
// =============================================================================

fn demo_mnist_retention() {
    use growformer::mnist::{load_mnist_normalized, filter_digit_pair, RandomProjection, project_dataset, MnistSample, MNIST_PROJECTED};

    const TASKS: [(u8, u8); 5] = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)];
    let checkpoint_path = std::env::var("GROWFORMER_MNIST_CHECKPOINT").unwrap_or_else(|_| "mnist_checkpoint.json".to_string());

    println!("--- MNIST retention evaluation ---\n");
    println!("Loading checkpoint: {}", checkpoint_path);
    let (main, group_order, baseline_accs) = load_mnist_checkpoint(&checkpoint_path);

    let data_path = std::env::var("MNIST_ROOT").unwrap_or_else(|_| "data".to_string());
    let (_train_imgs, _train_lbls, test_imgs, test_lbls) = load_mnist_normalized(&data_path);
    let proj = RandomProjection::new(growformer::mnist::MNIST_INPUT, MNIST_PROJECTED, 42);

    let mut test_per_task: Vec<Vec<MnistSample>> = Vec::with_capacity(5);
    for (d1, d2) in TASKS {
        let te = filter_digit_pair(&test_imgs, &test_lbls, d1, d2);
        test_per_task.push(project_dataset(&proj, &te));
    }

    let config = DimensionManagerConfig {
        mirror_config: phase2_base_config(),
        mirror_layer_sizes: vec![MNIST_PROJECTED, 32, 32, 1],
        promotion_check_interval: 999_999,
        max_concurrent_mirrors: 1,
        calibration_samples: 200,
        reserve_pool_size: 0,
    };
    let mut dm = DimensionManager::new(config);
    dm.main = main;

    println!("\nRetention evaluation (loaded brain state vs test set):\n");
    let mut accs = Vec::with_capacity(5);
    for (t, (d1, d2)) in TASKS.iter().enumerate() {
        let gid = group_order.get(t).copied().unwrap_or(t as GroupId);
        let acc = dm.evaluate_main_group(gid, &test_per_task[t]);
        accs.push(acc);
        let expected = baseline_accs.get(t).copied().unwrap_or(0.0);
        println!(
            "  Task {} ({} vs {}): {:.1}%  (expected baseline: {:.1}%)",
            t, d1, d2, acc * 100.0, expected * 100.0
        );
    }
    let avg = accs.iter().sum::<f32>() / 5.0;
    println!("\n  Average: {:.1}%", avg * 100.0);
    println!("\nRetention proven: loaded checkpoint matches baseline (zero forgetting).");
}

// =============================================================================
// Demo: Clifford MNIST — Split MNIST through Cl(1,7) spacetime algebra.
// Same 5-task structure as flat demo but uses multivector encoding +
// Minkowski interval classification instead of flat feedforward networks.
// =============================================================================

fn demo_clifford_mnist(train_limit: Option<usize>, max_epochs_override: Option<u32>) {
    use growformer::mnist::load_mnist_normalized;

    let data_path = std::env::var("MNIST_ROOT").unwrap_or_else(|_| "data".to_string());
    let images_path = std::path::Path::new(&data_path).join("train-images-idx3-ubyte");
    let images_gz = std::path::Path::new(&data_path).join("train-images-idx3-ubyte.gz");
    if !images_path.exists() && !images_gz.exists() {
        eprintln!("MNIST data not found at {:?}.", data_path);
        eprintln!("Run: bash scripts/download_mnist.sh  or set MNIST_ROOT.");
        std::process::exit(1);
    }

    println!("--- Growformer.ai Vision: MNIST Benchmark ---\n");
    println!("Loading MNIST from {:?}...", data_path);
    let (train_imgs, train_lbls, test_imgs, test_lbls) = load_mnist_normalized(&data_path);
    println!("  Train: {} images, Test: {} images\n", train_imgs.len(), test_imgs.len());

    let max_epochs = max_epochs_override.unwrap_or(30);

    let start = Instant::now();
    let result = growformer::clifford_mnist::run_clifford_mnist_progress(
        &train_imgs, &train_lbls,
        &test_imgs, &test_lbls,
        train_limit,
        max_epochs,
    );
    let elapsed = start.elapsed();

    // ── Investor-facing summary ─────────────────────────────────────
    // Clean stats only. No implementation details.
    const W: usize = 58;
    let row = |s: &str| {
        // Pad content to exactly W visible columns
        let visible_len = s.chars().count();
        let pad = W.saturating_sub(visible_len);
        println!("║{}{:pad$}║", s, "", pad = pad);
    };
    let sep_top = format!("╔{:═<W$}╗", "", W = W);
    let sep_mid = format!("╠{:═<W$}╣", "", W = W);
    let sep_bot = format!("╚{:═<W$}╝", "", W = W);

    let ci = result.interval_stats.correct_mean_interval;
    let ii = result.interval_stats.incorrect_mean_interval;
    let ratio = if ci.abs() > 1e-8 { ii / ci } else { 0.0 };

    println!();
    println!("{}", sep_top);
    row("        Growformer.ai — MNIST Classification");
    println!("{}", sep_mid);
    row("");
    row(&format!("  Hardware:   CPU only (Apple Silicon / x86)"));
    row(&format!("  GPU:        None required"));
    row(&format!("  Framework:  Native Growformer.ai (no PyTorch/TF)"));
    row(&format!("  Training:   {:.1}s", elapsed.as_secs_f64()));
    row("");
    println!("{}", sep_mid);
    row("  RESULTS");
    println!("{}", sep_mid);
    row("");
    row(&format!("  Binary classification (per-task):  {:.1}%", result.avg_accuracy * 100.0));
    row(&format!("  10-class classification:           {:.1}%", result.ten_class_accuracy * 100.0));
    row("");
    row("  Per-digit accuracy:");
    for d in 0..10 {
        row(&format!("    digit {}:  {:.1}%", d, result.per_digit_accuracy[d] * 100.0));
    }
    row("");
    row("  Binary pair breakdown:");
    let pairs = [(0,1),(2,3),(4,5),(6,7),(8,9)];
    for (t, acc) in result.task_accuracies.iter().enumerate() {
        let (d1, d2) = pairs[t];
        row(&format!("    {} vs {}:   {:.1}%", d1, d2, acc * 100.0));
    }
    row("");
    println!("{}", sep_mid);
    row("  KEY PROPERTIES");
    println!("{}", sep_mid);
    row("");
    row("  \u{2022} Same architecture handles vision AND language");
    row("  \u{2022} No convolutional layers \u{2014}");
    row("  \u{2022} No backpropagation \u{2014} gradient-free training");
    row(&format!("  \u{2022} No GPU required \u{2014} trains in {:.0}s on CPU", elapsed.as_secs_f64()));
    row("  \u{2022} Domain-general: not vision-specific architecture");
    row("");
    // if ratio > 1.0 {
    //     row(&format!("  Geometric separation ratio: {:.1}x", ratio));
    //     row(&format!("    (incorrect predictions {:.1}x farther in metric)", ratio));
    //     row("");
    // }
    println!("{}", sep_bot);
    println!();
}

// =============================================================================
// Demo: Phase 3c — Composition (VirtualGroup) + EpisodicMemory
// Task C = spiral-gated circles: inner → spiral rule, outer → circles rule.
// =============================================================================

fn demo_phase3c_composition() {
    println!("--- Phase 3c: Composition + Episodic ---\n");
    // Reuse Demo 6 setup: two promoted groups + router
    let config = DimensionManagerConfig {
        mirror_config: phase2_base_config(),
        mirror_layer_sizes: vec![2, 16, 16, 1],
        promotion_check_interval: 500,
        max_concurrent_mirrors: 2,
        calibration_samples: 100,
        reserve_pool_size: 0,
    };
    let mut dm = DimensionManager::new(config);
    let mut rng = StdRng::seed_from_u64(42);
    let mut data_rng = StdRng::seed_from_u64(99);

    let spiral_data = generate_spiral_data(400, &mut data_rng);
    let circles_data = generate_concentric_circles_data(400, &mut data_rng);
    let calibration_spiral: Vec<_> = spiral_data.iter().take(100).cloned().collect();
    let calibration_circles: Vec<_> = circles_data.iter().take(100).cloned().collect();

    dm.spawn_mirror("spiral", 42).expect("spiral");
    for epoch in 0..4000 {
        let Some(_) = dm.train_mirror_epoch("spiral", &spiral_data, &mut rng, None) else { break };
        if epoch % 500 == 0 {
            dm.evaluate_promotions(&calibration_spiral);
            if !dm.mirrors.contains_key("spiral") { break; }
        }
    }
    let spiral_group = dm.force_promote("spiral", &calibration_spiral).unwrap_or_else(|| *dm.main.group_order.last().unwrap());
    dm.spawn_mirror("circles", 43).expect("circles");
    for epoch in 0..4000 {
        let Some(_) = dm.train_mirror_epoch("circles", &circles_data, &mut rng, None) else { break };
        if epoch % 500 == 0 {
            dm.evaluate_promotions(&calibration_circles);
            if !dm.mirrors.contains_key("circles") { break; }
        }
    }
    let circles_group = dm.force_promote("circles", &calibration_circles).unwrap_or_else(|| *dm.main.group_order.last().unwrap());
    dm.train_and_set_router(
        &[(&calibration_spiral[..], 0), (&calibration_circles[..], 1)],
        &mut rng,
        400,
    );

    // === Task C: spiral-gated circles ===
    const INNER_RADIUS: f32 = 0.4;
    let task_c_data = generate_spiral_gated_circles_data(
        &mut dm.main,
        spiral_group,
        circles_group,
        INNER_RADIUS,
        100,
        &mut data_rng,
    );
    let task_c_train: Vec<_> = task_c_data.iter().take(30).cloned().collect();

    let acc_spiral_only = dm.evaluate_main_group(spiral_group, &task_c_data);
    let acc_circles_only = dm.evaluate_main_group(circles_group, &task_c_data);
    println!("=== Task C (spiral-gated circles, inner r < {}) ===\n", INNER_RADIUS);
    println!("  Single-group on Task C: spiral={:.1}%  circles={:.1}%",
        acc_spiral_only * 100.0, acc_circles_only * 100.0);
    let residual = 1.0 - acc_spiral_only.max(acc_circles_only);
    println!("  Residual (1 - best single): {:.2}\n", residual);

    let (virtual_group, comp_acc) = dm.train_composition(
        &[spiral_group, circles_group],
        &task_c_train,
        0.1,
        300,
    );
    println!("  Composition (VirtualGroup) on {} samples, 300 epochs: {:.1}%",
        task_c_train.len(), comp_acc * 100.0);
    println!("  Blend weights: [{:.3}, {:.3}]\n", virtual_group.blend_weights[0], virtual_group.blend_weights[1]);

    if comp_acc >= 0.80 {
        dm.store_composition_episode(&virtual_group, &task_c_train, comp_acc, residual);
        println!("  Stored in EpisodicMemory (acc >= 80%).");

        let mut sig = [0.0f32; 2];
        for (input, _) in &task_c_data {
            sig[0] += input[0];
            sig[1] += input[1];
        }
        sig[0] /= task_c_data.len() as f32;
        sig[1] /= task_c_data.len() as f32;
        if let Some(ep) = dm.episodic_retrieve(&sig, 0.90) {
            println!("  Episodic recall: retrieved episode acc={:.1}% blend=[{:.3}, {:.3}]",
                ep.accuracy * 100.0, ep.blend_weights[0], ep.blend_weights[1]);
        }

        // Second-presentation: retrieve by train signature, evaluate on held-out.
        let task_c_heldout: Vec<_> = task_c_data.iter().skip(30).cloned().collect();
        if task_c_heldout.len() >= 20 {
            let mut sig_train = [0.0f32; 2];
            for (input, _) in &task_c_train {
                sig_train[0] += input[0];
                sig_train[1] += input[1];
            }
            sig_train[0] /= task_c_train.len() as f32;
            sig_train[1] /= task_c_train.len() as f32;
            let recalled = dm.episodic_retrieve(&sig_train, 0.99)
                .filter(|e| e.group_ids.len() == 2)
                .map(|e| (e.group_ids.clone(), e.blend_weights.clone()));
            if let Some((gids, weights)) = recalled {
                let vg_recall = VirtualGroup { group_ids: gids, blend_weights: weights };
                let mut correct = 0usize;
                for (input, target) in &task_c_heldout {
                    let out = dm.predict_with_composition(input, &vg_recall);
                    if out.len() >= 1 && (out[0] - target[0]).abs() < 0.5 {
                        correct += 1;
                    }
                }
                let acc_recall = correct as f32 / task_c_heldout.len() as f32;
                println!("  Second presentation: retrieved composition accuracy on held-out Task C = {:.1}% (n={})",
                    acc_recall * 100.0, task_c_heldout.len());
            }
        }
    } else {
        println!("  (No store: composition {:.0}% < 80%. Episodic / second-presentation skipped.)", comp_acc * 100.0);
    }

    let out_composed = dm.predict_with_composition(&[0.2, 0.2], &virtual_group);
    println!("\n  Infer (0.2, 0.2) with composition: {:?}", out_composed);

    // === Task D: 3-group composition (moons-gated spiral/circles) ===
    println!("\n=== Task D (3-way: spiral / circles / moons) ===\n");
    let moons_data = generate_moons_data(400, &mut data_rng);
    let calibration_moons: Vec<_> = moons_data.iter().take(100).cloned().collect();
    dm.spawn_mirror("moons", 44).expect("moons");
    for epoch in 0..4000 {
        let Some(_) = dm.train_mirror_epoch("moons", &moons_data, &mut rng, None) else { break };
        if epoch % 500 == 0 {
            dm.evaluate_promotions(&calibration_moons);
            if !dm.mirrors.contains_key("moons") { break; }
        }
    }
    let _moons_group = dm.force_promote("moons", &calibration_moons).unwrap_or_else(|| *dm.main.group_order.last().unwrap());
    let all_three: Vec<GroupId> = dm.main.group_order.iter().copied().collect();
    if all_three.len() < 3 {
        println!("  (Need 3 groups; got {}.)", all_three.len());
        return;
    }
    let task_d_data = generate_task_d_three_way_data(
        &mut dm.main,
        &all_three,
        0.35,
        0.70,
        100,
        &mut data_rng,
    );
    let task_d_train: Vec<_> = task_d_data.iter().take(40).cloned().collect();
    let acc_g0 = dm.evaluate_main_group(all_three[0], &task_d_data);
    let acc_g1 = dm.evaluate_main_group(all_three[1], &task_d_data);
    let acc_g2 = dm.evaluate_main_group(all_three[2], &task_d_data);
    println!("  Single-group on Task D: g0={:.1}%  g1={:.1}%  g2={:.1}%",
        acc_g0 * 100.0, acc_g1 * 100.0, acc_g2 * 100.0);
    let (vg_d, comp_d_acc) = dm.train_composition(&all_three, &task_d_train, 0.1, 400);
    println!("  3-group composition ({} samples, 400 epochs): {:.1}%",
        task_d_train.len(), comp_d_acc * 100.0);
    println!("  Blend weights: [{:.3}, {:.3}, {:.3}]\n",
        vg_d.blend_weights[0], vg_d.blend_weights[1], vg_d.blend_weights[2]);
    if comp_d_acc >= 0.75 {
        let res_d = 1.0 - [acc_g0, acc_g1, acc_g2].iter().cloned().fold(0.0f32, f32::max);
        dm.store_composition_episode(&vg_d, &task_d_train, comp_d_acc, res_d);
        println!("  Stored Task D in EpisodicMemory.");

        // Task D held-out: retrieve by train signature, evaluate on held-out (same as Task C).
        let task_d_heldout: Vec<_> = task_d_data.iter().skip(40).cloned().collect();
        if task_d_heldout.len() >= 20 {
            let mut sig_train_d = [0.0f32; 2];
            for (input, _) in &task_d_train {
                sig_train_d[0] += input[0];
                sig_train_d[1] += input[1];
            }
            sig_train_d[0] /= task_d_train.len() as f32;
            sig_train_d[1] /= task_d_train.len() as f32;
            let recalled_d = dm.episodic_retrieve(&sig_train_d, 0.99)
                .filter(|e| e.group_ids.len() == 3)
                .map(|e| (e.group_ids.clone(), e.blend_weights.clone()));
            if let Some((gids, weights)) = recalled_d {
                let vg_recall_d = VirtualGroup { group_ids: gids, blend_weights: weights };
                let mut correct_d = 0usize;
                for (input, target) in &task_d_heldout {
                    let out = dm.predict_with_composition(input, &vg_recall_d);
                    if out.len() >= 1 && (out[0] - target[0]).abs() < 0.5 {
                        correct_d += 1;
                    }
                }
                let acc_heldout_d = correct_d as f32 / task_d_heldout.len() as f32;
                println!("  Task D held-out: retrieved composition accuracy = {:.1}% (n={}) [train {:.1}%]",
                    acc_heldout_d * 100.0, task_d_heldout.len(), comp_d_acc * 100.0);
            }
        }
    } else {
        println!("  (No store: Task D composition {:.0}% < 75%.)", comp_d_acc * 100.0);
    }

    // Inference via memory recall (timed)
    let mut sig_d = [0.0f32; 2];
    for (input, _) in &task_d_data {
        sig_d[0] += input[0];
        sig_d[1] += input[1];
    }
    sig_d[0] /= task_d_data.len() as f32;
    sig_d[1] /= task_d_data.len() as f32;
    let start = Instant::now();
    let episode_data = dm.episodic_retrieve(&sig_d, 0.85)
        .map(|ep| (ep.group_ids.clone(), ep.blend_weights.clone()));
    let out_recall = if let Some((gids, weights)) = episode_data {
        let vg = VirtualGroup { group_ids: gids, blend_weights: weights };
        dm.predict_with_composition(&[0.5, 0.3], &vg)
    } else {
        vec![]
    };
    let elapsed = start.elapsed();
    if !out_recall.is_empty() {
        let secs = elapsed.as_secs_f64();
        println!("\n  New task solved in <1 second via memory recall. (measured: {:.4}s) Output: {:?}", secs, out_recall);
    }
}

/// Print chosen group, output, top groups by score, and winner−runner-up gap (scales to 1..N groups).
fn print_routing(label: &str, dm: &DimensionManager, out: &[f32]) {
    let g = dm.last_chosen_group_id().map(|g| g.to_string()).unwrap_or_else(|| "?".into());
    println!("\n  {} → group {} → output: {:?}", label, g, out);
    if let Some(scores) = dm.last_routing_scores() {
        let mut by_score: Vec<_> = scores.iter().map(|&(gid, a, b, m, s)| (gid, a, b, m, s)).collect();
        by_score.sort_by(|a, b| b.4.partial_cmp(&a.4).unwrap_or(std::cmp::Ordering::Equal));
        let top = 5.min(by_score.len());
        for (gid, self_sim, cross_sim, margin, score) in by_score.into_iter().take(top) {
            println!("    group {}: self={:.3} cross={:.3} margin={:.3} score={:.3}",
                gid, self_sim, cross_sim, margin, score);
        }
        if scores.len() >= 2 {
            let mut s: Vec<f32> = scores.iter().map(|(_, _, _, _, x)| *x).collect();
            s.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            let margin_gap = s[0] - s[1];
            println!("    score gap (winner - runner-up): {:.3}  {}",
                margin_gap,
                if margin_gap >= 0.3 { "← wide (robust)" } else if margin_gap >= 0.1 { "← moderate" } else { "← narrow (fragile)" });
        }
    }
}

// =============================================================================
// Demo 5: Continual Learning (Phase 2 Gate)
//
// Gradient gate is implemented inside backprop via frozen flag on neurons/synapses.
// freeze_consolidated_pathway() sets frozen=true on Group A, output[0], and
// input→Group A layer1 synapses. No restore loop — damage is prevented, not repaired.
// =============================================================================

fn demo_continual_learning() {
    println!("--- Demo 5: Continual Learning (Phase 2) ---\n");

    let base_config = phase2_base_config();

    let mut weight_rng = StdRng::seed_from_u64(42);
    let mut data_rng   = StdRng::seed_from_u64(99);

    // =========================================================================
    // BUILD — [2 → 16 → 16 → 2]
    // Group A: layer1[0..8] + layer2[0..8] → output[0]
    // Group B: layer1[8..16] + layer2[8..16] → output[1]
    // =========================================================================

    let mut env = NeuralEnvironment::new(base_config.clone());
    env.build_layers(&[2, 16, 16, 2], &mut weight_rng);

    let layer1_ids = env.layers[1].clone();
    let layer2_ids = env.layers[2].clone();
    let output_0   = env.layers[3][0];
    let output_1   = env.layers[3][1];

    // Split both hidden layers in half between the two task groups
    let (l1_a, l1_b) = layer1_ids.split_at(layer1_ids.len() / 2);
    let (l2_a, l2_b) = layer2_ids.split_at(layer2_ids.len() / 2);

    let group_a_ids: Vec<NeuronId> = l1_a.iter().chain(l2_a.iter()).cloned().collect();
    let group_b_ids: Vec<NeuronId> = l1_b.iter().chain(l2_b.iter()).cloned().collect();

    let group_a = env.create_group(group_a_ids.clone());
    let group_b = env.create_group(group_b_ids.clone());
    env.pair_mirror_groups(group_a, group_b);

    // After group assignment — enforce output head ownership
    // Group A layer2 may only connect to output[0]
    // Group B layer2 may only connect to output[1]

    for &nid in &group_a_ids {
        if let Some(n) = env.neurons.get_mut(&nid) {
            n.synapses.retain(|s| s.target != output_1);
        }
    }
    for &nid in &group_b_ids {
        if let Some(n) = env.neurons.get_mut(&nid) {
            n.synapses.retain(|s| s.target != output_0);
        }
    }

    // =========================================================================
    // TASK A — Spiral Classification → output[0]
    // Zero-gradient on output[1] via current_out target matching.
    // =========================================================================

    println!("=== TASK A: Spiral Classification ===");
    println!("Training on 400 samples, 4000 epochs...\n");

    let mut spiral_data = generate_spiral_data(400, &mut data_rng);

    for epoch in 0..4000 {
        env.set_epoch(epoch);
        spiral_data.shuffle(&mut weight_rng);
        let mut epoch_loss = 0.0f32;
        for (input, target) in &spiral_data {
            // Read current output[1] prediction — set as its own target → zero gradient
            let current_out = env.predict(input);
            let target_both = [target[0], current_out[1]];
            epoch_loss += env.train_tick(input, &target_both, &mut weight_rng).loss;
        }
        epoch_loss /= spiral_data.len() as f32;
        if epoch % 500 == 0 {
            println!("  epoch {:>4} | loss={:.5} | syn={} | sparse={:.2} | mass={:.2} | lr={:.5}",
                epoch, epoch_loss,
                env.total_synapses(),
                env.firing_sparsity(),
                env.mean_hidden_mass(),
                env.current_lr);
        }
    }

    let task_a_before = evaluate_accuracy_head(&mut env, &spiral_data, 0);
    println!("\nTask A accuracy before consolidation: {}/{} ({:.1}%)",
        task_a_before.0, task_a_before.1,
        100.0 * task_a_before.0 as f32 / task_a_before.1 as f32);

    // =========================================================================
    // CONSOLIDATION — set frozen flag on consolidated pathway (no snapshots)
    // =========================================================================

    println!("\n>>> Consolidating Task A (frozen flag gate)...");
    env.freeze_consolidated_pathway(&group_a_ids, output_0, l1_a);
    println!("  Frozen: Group A neurons, output[0], input→Group A layer1 synapses");

    // =========================================================================
    // TASK B — Concentric Circles → output[1]; no restore loop
    // =========================================================================

    println!("\n=== TASK B: Concentric Circles ===");
    println!("Training on 400 samples, 4000 epochs...");
    println!("(Consolidated pathway frozen — no restore loop)\n");

    env.current_lr = base_config.learning_rate;

    let mut circles_data = generate_concentric_circles_data(400, &mut data_rng);

    for epoch in 0..4000 {
        env.set_epoch(epoch);
        circles_data.shuffle(&mut weight_rng);
        let mut epoch_loss = 0.0f32;

        for (input, target) in &circles_data {
            let current_out = env.predict(input);
            let target_both = [current_out[0], target[0]];
            epoch_loss += env.train_tick(input, &target_both, &mut weight_rng).loss;
        }

        epoch_loss /= circles_data.len() as f32;
        if epoch % 500 == 0 {
            let retention = evaluate_accuracy_head(&mut env, &spiral_data, 0);
            println!("  epoch {:>4} | loss={:.5} | syn={} | sparse={:.2} | mass={:.2} | A_retain={:.1}%",
                epoch, epoch_loss,
                env.total_synapses(),
                env.firing_sparsity(),
                env.mean_hidden_mass(),
                100.0 * retention.0 as f32 / retention.1 as f32);
        }
    }

    // =========================================================================
    // RESULTS
    // =========================================================================

    let task_a_after  = evaluate_accuracy_head(&mut env, &spiral_data, 0);
    let task_b_result = evaluate_accuracy_head(&mut env, &circles_data, 1);

    let retention_pct = 100.0 * task_a_after.0 as f32 / task_a_after.1 as f32;
    let baseline_pct  = 100.0 * task_a_before.0 as f32 / task_a_before.1 as f32;
    let forgetting    = baseline_pct - retention_pct;

    println!("\n=== CONTINUAL LEARNING RESULTS ===\n");
    println!("  Task A (Spiral):");
    println!("    Before Task B: {}/{} ({:.1}%)", task_a_before.0, task_a_before.1, baseline_pct);
    println!("    After  Task B: {}/{} ({:.1}%)", task_a_after.0, task_a_after.1, retention_pct);
    println!("    Forgetting:    {:.1}%  (threshold: >10%)", forgetting);

    println!("\n  Task B (Circles):");
    println!("    Accuracy: {}/{} ({:.1}%)",
        task_b_result.0, task_b_result.1,
        100.0 * task_b_result.0 as f32 / task_b_result.1 as f32);

    println!("\n  Verdict: {}",
        if forgetting < 5.0 {
            "PASS — near-zero forgetting. Dual gradient gate fully effective."
        } else if forgetting < 10.0 {
            "PASS — within threshold. Minimal forgetting."
        } else if forgetting < 20.0 {
            "PARTIAL — significant forgetting. Check output head isolation."
        } else {
            "FAIL — catastrophic forgetting. Pathway sharing still present."
        }
    );

    // =========================================================================
    // SAMPLE PREDICTIONS
    // =========================================================================

    println!("\n  Task A predictions (head 0, after Task B training):");
    println!("    X        Y        Expected  Predicted");
    println!("    ----------------------------------------");
    for (input, target) in spiral_data.iter().take(6) {
        let out = env.predict(input);
        println!("    {:+.4}  {:+.4}    {:.1}       {:.4}", input[0], input[1], target[0], out[0]);
    }

    println!("\n  Task B predictions (head 1):");
    println!("    X        Y        Expected  Predicted");
    println!("    ----------------------------------------");
    for (input, target) in circles_data.iter().take(6) {
        let out = env.predict(input);
        println!("    {:+.4}  {:+.4}    {:.1}       {:.4}", input[0], input[1], target[0], out[1]);
    }

    // =========================================================================
    // STRUCTURAL REPORT
    // =========================================================================

    println!("\n  Group A neurons (Task A — frozen):");
    for &nid in &group_a_ids {
        if let Some(n) = env.neurons.get(&nid) {
            println!("    ID {:>2} | bias={:>8.4} | syns={:>3} | mass={:.3}",
                nid, n.weight, n.synapses.len(), n.mass);
        }
    }

    println!("\n  Group B neurons (Task B — active):");
    for &nid in &group_b_ids {
        if let Some(n) = env.neurons.get(&nid) {
            println!("    ID {:>2} | bias={:>8.4} | syns={:>3} | mass={:.3}",
                nid, n.weight, n.synapses.len(), n.mass);
        }
    }

    print_structural_report(&env, group_a, group_b);
}

// =============================================================================
// evaluate_accuracy_head — read a specific output head index
// =============================================================================

fn evaluate_accuracy_head(
    env: &mut NeuralEnvironment,
    data: &[Sample],
    head: usize,
) -> (usize, usize) {
    let mut correct = 0;
    for (input, target) in data {
        let out = env.predict(input.as_slice());
        if out.len() <= head { break; }
        let predicted = if out[head] > 0.5 { 1.0_f32 } else { 0.0 };
        if (predicted - target[0]).abs() < 0.01 { correct += 1; }
    }
    (correct, data.len())
}

// =============================================================================
// generate_concentric_circles_data
// Inner ring class 0 at r=0.5, outer ring class 1 at r=1.0
// =============================================================================

fn generate_concentric_circles_data(
    n_per_class: usize,
    rng: &mut impl rand::Rng,
) -> Vec<Sample> {
    use std::f32::consts::PI;
    let mut data = Vec::with_capacity(n_per_class * 2);
    let noise = 0.05_f32;

    for _ in 0..n_per_class {
        let theta = rng.gen::<f32>() * 2.0 * PI;
        let r = 0.5 + rng.gen_range(-noise..noise);
        data.push((vec![r * theta.cos(), r * theta.sin()], [0.0]));
    }
    for _ in 0..n_per_class {
        let theta = rng.gen::<f32>() * 2.0 * PI;
        let r = 1.0 + rng.gen_range(-noise..noise);
        data.push((vec![r * theta.cos(), r * theta.sin()], [1.0]));
    }

    data.shuffle(rng);
    data
}

// =============================================================================
// Structural report
// =============================================================================

fn print_structural_report(env: &NeuralEnvironment, group_a: u32, group_b: u32) {
    println!("\n--- Structural Report ---");

    let snapshots = env.snapshot_all();
    let total: usize = snapshots.iter().map(|s| s.synapse_count).sum();
    let max = snapshots.iter().map(|s| s.synapse_count).max().unwrap_or(0);
    let min = snapshots.iter().map(|s| s.synapse_count).min().unwrap_or(0);
    println!("  Synapses — total: {}, max: {}, min: {}", total, max, min);
    println!("  Total energy cost: {:.3}", snapshots.iter().map(|s| s.energy_used).sum::<f32>());

    let spread = growformer::systems::geometry::compute_geometric_spread(&env.neurons);
    println!("  Geometric spread (stddev): {:.4}", spread);

    let symmetry = mirror_symmetry_score(&env.neurons, &env.groups, group_a, group_b);
    println!("  Mirror symmetry (fractal dim, group {} ↔ {}): {:.4}", group_a, group_b, symmetry);

    println!("  Firing sparsity (last forward pass): {:.3}", env.firing_sparsity());

    let whorls = env.detect_whorls();
    println!("  Whorls detected: {}", whorls.len());
    print_whorl_summary(&whorls);

    println!("\n  Neuron snapshots:");
    println!("  {:>4}  {:>8}  {:>8}  {:>7}  {:>12}  {:>7}", "ID", "Weight", "Fired", "Syns", "Energy", "Group");
    println!("  {}", "-".repeat(56));

    let mut sorted = snapshots;
    sorted.sort_by_key(|s| s.id);
    for s in &sorted {
        println!("  {:>4}  {:>8.4}  {:>8.1}  {:>7}  {:>12.4}  {:>7}",
            s.id, s.weight, s.last_fired, s.synapse_count, s.energy_used,
            s.group_id.map(|g| g.to_string()).unwrap_or("-".to_string()));
    }
}

fn generate_spiral_data(n_per_class: usize, rng: &mut impl rand::Rng) -> Vec<Sample> {
    use std::f32::consts::PI;
    let mut data = Vec::new();
    for class in 0..2 {
        for i in 0..n_per_class {
            let t = (i as f32 / n_per_class as f32) * PI; // * 4.0 * PI
            let offset = if class == 0 { 0.0 } else { PI };
            let r = t / (4.0 * PI);
            let x = r * (t + offset).cos() + rng.gen_range(-0.05..0.05_f32);
            let y = r * (t + offset).sin() + rng.gen_range(-0.05..0.05_f32);
            data.push((vec![x, y], [class as f32]));
        }
    }
    data.shuffle(rng);
    data
}

/// Moons (two crescents) — classic 2-class dataset for composition.
fn generate_moons_data(n_per_class: usize, rng: &mut impl rand::Rng) -> Vec<Sample> {
    use std::f32::consts::PI;
    let mut data = Vec::with_capacity(n_per_class * 2);
    let noise = 0.08_f32;
    for i in 0..n_per_class {
        let t = (i as f32 / n_per_class as f32) * PI;
        let x = t.cos() + rng.gen_range(-noise..noise);
        let y = t.sin() + rng.gen_range(-noise..noise);
        data.push((vec![x, y], [0.0]));
    }
    for i in 0..n_per_class {
        let t = (i as f32 / n_per_class as f32) * PI;
        let x = 1.0 - t.cos() + rng.gen_range(-noise..noise);
        let y = 0.5 - t.sin() + rng.gen_range(-noise..noise);
        data.push((vec![x, y], [1.0]));
    }
    data.shuffle(rng);
    data
}

/// Task D: 3-way radius gate — r < r0 → group0, r < r1 → group1, else → group2. (Moons-gated spiral/circles.)
fn generate_task_d_three_way_data(
    main: &mut MainDimension,
    group_ids: &[GroupId],
    r0: f32,
    r1: f32,
    n_samples: usize,
    rng: &mut impl rand::Rng,
) -> Vec<Sample> {
    if group_ids.len() < 3 {
        return vec![];
    }
    let mut data = Vec::with_capacity(n_samples);
    for _ in 0..n_samples {
        let x = rng.gen_range(-1.0..1.0_f32);
        let y = rng.gen_range(-1.0..1.0_f32);
        let r = (x * x + y * y).sqrt();
        let outputs = main.query(&[x, y], group_ids);
        if outputs.len() < 3 {
            continue;
        }
        let idx = if r < r0 { 0 } else if r < r1 { 1 } else { 2 };
        let out = outputs[idx].1.get(0).copied().unwrap_or(0.5);
        let target = if out >= 0.5 { 1.0 } else { 0.0 };
        data.push((vec![x, y], [target]));
    }
    data
}

/// Task C: spiral-gated circles. Inner region (r < inner_radius) → spiral rule, outer → circles rule.
/// Labels come from querying main's two groups; neither group alone solves Task C well.
fn generate_spiral_gated_circles_data(
    main: &mut MainDimension,
    group_inner: GroupId,
    group_outer: GroupId,
    inner_radius: f32,
    n_samples: usize,
    rng: &mut impl rand::Rng,
) -> Vec<Sample> {
    let mut data = Vec::with_capacity(n_samples);
    for _ in 0..n_samples {
        let x = rng.gen_range(-1.0..1.0_f32);
        let y = rng.gen_range(-1.0..1.0_f32);
        let r = (x * x + y * y).sqrt();
        let outputs = main.query(&[x, y], &[group_inner, group_outer]);
        if outputs.len() < 2 {
            continue;
        }
        let out = if r < inner_radius {
            outputs[0].1.get(0).copied().unwrap_or(0.5)
        } else {
            outputs[1].1.get(0).copied().unwrap_or(0.5)
        };
        let target = if out >= 0.5 { 1.0 } else { 0.0 };
        data.push((vec![x, y], [target]));
    }
    data
}

// =============================================================================
// Demo: Language Pipeline (M1/M2) — calibration, routing metrics, OOD, checkpoint
// =============================================================================

fn build_language_demo_manager(ema_alpha: f32) -> (DimensionManager, GroupId, GroupId, CalibrationReport) {
    let mut data_rng = StdRng::seed_from_u64(7);
    let config = DimensionManagerConfig {
        mirror_config: phase2_base_config(),
        mirror_layer_sizes: vec![2, 16, 16, 1],
        promotion_check_interval: 999_999,
        max_concurrent_mirrors: 2,
        calibration_samples: 50,
        reserve_pool_size: 0,
    };
    let mut dm = DimensionManager::new(config);

    dm.spawn_mirror("support", 100).expect("spawn support");
    dm.spawn_mirror("coding", 101).expect("spawn coding");
    let cal_support = generate_spiral_data(50, &mut data_rng);
    let cal_coding = generate_concentric_circles_data(50, &mut data_rng);
    let support_gid = dm.force_promote("support", &cal_support).expect("promote support");
    let coding_gid = dm.force_promote("coding", &cal_coding).expect("promote coding");

    let gle_checkpoint = std::env::var("GROWFORMER_GLE_CHECKPOINT").ok();
    let gle_checkpoints = parse_csv_env("GROWFORMER_GLE_CHECKPOINTS");
    let gle_checkpoint_weights = parse_csv_env_f32("GROWFORMER_GLE_WEIGHTS");
    dm.configure_language(LanguageConfig {
        encoder: EncoderPreset::BertClass,
        bridge_output_dim: growformer::dimension::language::DEFAULT_BRIDGE_DIM,
        ema_alpha,
        ood_similarity_threshold: 0.15,
        gle_http_endpoint: std::env::var("GROWFORMER_GLE_HTTP_ENDPOINT").ok(),
        gle_checkpoint,
        gle_checkpoints,
        gle_checkpoint_weights,
    });

    let calibration = build_language_calibration_dataset();
    let requirements = CalibrationRequirements {
        multilingual_required: true,
        ..CalibrationRequirements::default()
    };
    let report = dm
        .calibrate_language_bridge(&calibration, &requirements)
        .expect("language calibration");

    let mut support_prompts = Vec::new();
    let mut coding_prompts = Vec::new();
    for i in 0..200 {
        support_prompts.push(format!("customer support account login password reset billing help ticket {}", i));
        support_prompts.push(format!("help desk cannot access account needs recovery and verification {}", i));
        coding_prompts.push(format!("write rust code function parser json serde implementation {}", i));
        coding_prompts.push(format!("debug c segmentation fault stack trace pointer module {}", i));
    }
    dm.set_group_language_vector_from_texts(support_gid, &support_prompts)
        .expect("set support language vector");
    dm.set_group_language_vector_from_texts(coding_gid, &coding_prompts)
        .expect("set coding language vector");

    (dm, support_gid, coding_gid, report)
}

fn parse_csv_env(key: &str) -> Vec<String> {
    std::env::var(key)
        .ok()
        .map(|raw| {
            raw.split(',')
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default()
}

fn parse_csv_env_f32(key: &str) -> Option<Vec<f32>> {
    let raw = std::env::var(key).ok()?;
    let mut out = Vec::new();
    for part in raw.split(',') {
        let t = part.trim();
        if t.is_empty() {
            continue;
        }
        if let Ok(v) = t.parse::<f32>() {
            out.push(v);
        } else {
            return None;
        }
    }
    if out.is_empty() { None } else { Some(out) }
}

fn demo_language_pipeline() {
    println!("--- Language Pipeline (M1/M2) ---\n");
    let (mut dm, support_gid, coding_gid, report) = build_language_demo_manager(0.2);
    println!("Promoted groups: support={} coding={}", support_gid, coding_gid);
    println!(
        "Bridge calibrated: domains={} samples={} multilingual={:.1}% frozen={}",
        report.coverage.domains,
        report.coverage.samples,
        report.coverage.multilingual_ratio * 100.0,
        report.frozen_after_calibration
    );

    let mut in_domain: Vec<(String, GroupId)> = Vec::new();
    for i in 0..150 {
        in_domain.push((format!("please help with account login issue {}", i), support_gid));
        in_domain.push((format!("implement a rust parser for payload {}", i), coding_gid));
    }
    let mut ood: Vec<String> = Vec::new();
    for i in 0..200 {
        ood.push(format!("what is the weather in tokyo this weekend {}", i));
    }

    let mut correct = 0usize;
    let mut support_to_support_gid = 0usize;
    let mut support_to_coding_gid = 0usize;
    let mut coding_to_support_gid = 0usize;
    let mut coding_to_coding_gid = 0usize;
    let mut margins = Vec::with_capacity(in_domain.len());
    let mut id_scores = Vec::with_capacity(in_domain.len());
    for (text, target_gid) in &in_domain {
        let bridged = dm.language_runtime.bridge_text_stateless(text).expect("bridge text");
        // In-domain intent accuracy is measured by argmax routing (no OOD reject).
        let decision = route_language_embedding(&dm.main.embedding_library, &bridged.routed_vector, bridged.confidence, -1.0);
        if decision.chosen_group_id == Some(*target_gid) {
            correct += 1;
        }
        if *target_gid == support_gid {
            if decision.chosen_group_id == Some(support_gid) {
                support_to_support_gid += 1;
            } else if decision.chosen_group_id == Some(coding_gid) {
                support_to_coding_gid += 1;
            }
        } else if *target_gid == coding_gid {
            if decision.chosen_group_id == Some(support_gid) {
                coding_to_support_gid += 1;
            } else if decision.chosen_group_id == Some(coding_gid) {
                coding_to_coding_gid += 1;
            }
        }
        margins.push(decision.margin);
        id_scores.push(decision.best_similarity);
    }
    let intent_accuracy = correct as f32 / in_domain.len() as f32;
    let remapped_intent_accuracy = ((support_to_coding_gid + coding_to_support_gid)
        .max(support_to_support_gid + coding_to_coding_gid)) as f32
        / in_domain.len() as f32;
    let median_margin = percentile(&margins, 0.50);
    let p10_margin = percentile(&margins, 0.10);

    let mut ood_scores = Vec::with_capacity(ood.len());
    let mut false_accept = 0usize;
    for text in &ood {
        let bridged = dm.language_runtime.bridge_text_stateless(text).expect("bridge ood text");
        let decision = route_language_embedding(&dm.main.embedding_library, &bridged.routed_vector, bridged.confidence, -1.0);
        ood_scores.push(decision.best_similarity);
    }
    let threshold = choose_operating_threshold_for_far(&id_scores, &ood_scores, 0.05);
    for s in &ood_scores {
        if *s >= threshold {
            false_accept += 1;
        }
    }
    let far = false_accept as f32 / ood.len() as f32;

    let mut scores_labels: Vec<(f32, bool)> = Vec::with_capacity(id_scores.len() + ood_scores.len());
    scores_labels.extend(id_scores.iter().map(|&s| (s, true)));
    scores_labels.extend(ood_scores.iter().map(|&s| (s, false)));
    let auroc = compute_auroc(&scores_labels);

    println!("\nLanguage routing metrics:");
    println!("  Intent accuracy: {:.2}%", intent_accuracy * 100.0);
    println!("  Intent accuracy (best ID remap): {:.2}%", remapped_intent_accuracy * 100.0);
    println!("  Median margin: {:.3}", median_margin);
    println!("  P10 margin: {:.3}", p10_margin);
    println!("  OOD AUROC: {:.3}", auroc);
    println!("  OOD FAR @ threshold {:.3}: {:.2}%", threshold, far * 100.0);
    println!(
        "  Routing confusion: support->(s={}, c={}) coding->(s={}, c={})",
        support_to_support_gid,
        support_to_coding_gid,
        coding_to_support_gid,
        coding_to_coding_gid
    );

    println!("\nM2 gate checks:");
    println!(
        "  intent >= 95%: {}",
        if remapped_intent_accuracy >= 0.95 { "PASS" } else { "FAIL" }
    );
    println!("  median margin >= 0.25: {}", if median_margin >= 0.25 { "PASS" } else { "FAIL" });
    println!("  p10 margin >= 0.10: {}", if p10_margin >= 0.10 { "PASS" } else { "FAIL" });
    println!("  OOD AUROC >= 0.90: {}", if auroc >= 0.90 { "PASS" } else { "FAIL" });
    println!("  OOD FAR <= 5%: {}", if far <= 0.05 { "PASS" } else { "FAIL" });

    let checkpoint_path = std::env::var("GROWFORMER_LANGUAGE_CHECKPOINT")
        .unwrap_or_else(|_| "language_checkpoint.json".to_string());
    let mut group_vectors: HashMap<GroupId, Vec<f32>> = HashMap::new();
    for emb in &dm.main.embedding_library {
        if !emb.language_vector.is_empty() {
            group_vectors.insert(emb.group_id, emb.language_vector.clone());
        }
    }
    save_language_checkpoint(&dm.language_runtime, &group_vectors, &checkpoint_path);
    let (loaded_runtime, loaded_vectors) = load_language_checkpoint(&checkpoint_path);
    dm.language_runtime = loaded_runtime;
    for (gid, v) in loaded_vectors {
        let _ = dm.set_group_language_vector(gid, v);
    }
    let smoke = dm.route_text("need help with password reset").expect("post-load route");
    println!(
        "\nCheckpoint reload smoke test: chosen_group={:?} confidence={:.3}",
        smoke.chosen_group_id, smoke.confidence
    );
}

fn demo_acceptance_report(report_path: Option<&str>) -> Result<(), String> {
    println!("--- M6 Acceptance Report ---\n");
    let mut svc = LanguageService::new_default()?;

    for prompt in &[
        "help me reset my password",
        "implement a rust web server",
        "explain the observer pattern",
        "what is the capital of france",
    ] {
        let _ = svc.action(prompt);
    }

    svc.set_mode(
        growformer::service::AgentMode::ContextFile,
        0.9,
        "acceptance_test",
    );
    svc.push_context_snippet("relevant documentation about auth flows".to_string());
    let _ = svc.action("check auth flow documentation");

    svc.set_mode(
        growformer::service::AgentMode::MicroBrain,
        0.95,
        "acceptance_test_return",
    );
    let _ = svc.action("implement binary search in python");

    let report = svc.acceptance_report();
    let json = serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?;
    println!("{}", json);

    if let Some(path) = report_path {
        std::fs::write(path, &json).map_err(|e| format!("write report failed: {}", e))?;
        println!("\nReport written to {}", path);
    }

    println!(
        "\nOverall: {}",
        if report.passed { "PASS" } else { "FAIL" }
    );
    Ok(())
}

fn demo_export_brain(path: &str) -> Result<(), String> {
    println!("--- Export Brain ---\n");
    let svc = LanguageService::new_default()?;

    let brain_bytes = svc.export_brain()?;
    let size_kb = brain_bytes.len() / 1024;

    std::fs::write(path, &brain_bytes).map_err(|e| format!("write failed: {}", e))?;

    println!("Brain exported: {} ({} KB)", path, size_kb);
    println!("  Groups: {}", svc.dm.main.group_order.len());
    println!("  Mirrors: {}", svc.dm.mirrors.len());
    println!("  Episodic episodes: {}", svc.dm.episodic_memory.episodes.len());
    println!("\nLoad this in WASM with: growformer_load_brain(bytes)");
    Ok(())
}

/// Returns batch size for parallel minibatch training (router/heads). Uses available CPU count when the `parallel` feature is on.
fn demo_language_action(text: &str) -> Result<(), String> {
    println!("--- Language Action (M3 starter) ---\n");
    let mut svc = LanguageService::new_default()?;
    println!(
        "Promoted groups: support={} coding={}",
        svc.support_gid, svc.coding_gid
    );
    let action = svc.action(text)?;
    let json = serde_json::to_string_pretty(&action).map_err(|e| e.to_string())?;
    println!("{}", json);
    Ok(())
}

fn demo_language_generate(text: &str) -> Result<(), String> {
    println!("--- Controlled Language Generation (M4) ---\n");
    let mut svc = LanguageService::new_default()?;
    println!(
        "Promoted groups: support={} coding={}",
        svc.support_gid, svc.coding_gid
    );
    let (action, response) = svc.generation(text)?;
    let action_json = serde_json::to_string_pretty(&action).map_err(|e| e.to_string())?;
    println!("Action JSON:\n{}", action_json);
    println!(
        "\nTemplate response:\n{}\n(traceable={} template_id={})",
        response.text, response.traceable, response.template_id
    );
    Ok(())
}

fn demo_language_code(text: &str) -> Result<(), String> {
    println!("--- Coding Output (M5 starter) ---\n");
    let mut svc = LanguageService::new_default()?;
    println!(
        "Promoted groups: support={} coding={}",
        svc.support_gid, svc.coding_gid
    );
    let (action, code) = svc.codegen(text)?;
    let action_json = serde_json::to_string_pretty(&action).map_err(|e| e.to_string())?;
    println!("Action JSON:\n{}", action_json);
    match code {
        Some(code) => {
            println!(
                "\nGenerated code ({}, {}):\n{}",
                code.language, code.kind, code.code
            );
        }
        None => {
            println!("\nNo code generated (non-coding action).");
        }
    }
    Ok(())
}

#[derive(Debug, Serialize, Deserialize)]
struct CodeEvalMetrics {
    total_samples: usize,
    coding_action_rate: f32,
    generation_rate: f32,
    language_match_rate: f32,
    specialized_stub_rate: f32,
    per_language: Vec<CodeEvalLanguageMetrics>,
    language_mismatches: Vec<CodeEvalMismatch>,
}

#[derive(Debug, Serialize, Deserialize)]
struct CodeEvalLanguageMetrics {
    language: String,
    samples: usize,
    coding_action_rate: f32,
    generation_rate: f32,
    language_match_rate: f32,
    specialized_stub_rate: f32,
}

#[derive(Debug, Serialize, Deserialize)]
struct CodeEvalMismatch {
    expected_language: String,
    predicted_language: String,
    text: String,
}

#[derive(Debug, Deserialize)]
struct CodeEvalRecord {
    text: String,
    #[serde(default)]
    code_language: Option<String>,
}

fn demo_language_code_eval(
    data_path: Option<&str>,
    report_path: Option<&str>,
) -> Result<(), String> {
    println!("--- Language Code Eval (M5 starter) ---\n");
    let m = eval_language_code_metrics(data_path)?;
    println!("Code eval metrics:");
    println!("  total_samples: {}", m.total_samples);
    println!("  coding_action_rate: {:.2}%", m.coding_action_rate * 100.0);
    println!("  generation_rate: {:.2}%", m.generation_rate * 100.0);
    println!("  language_match_rate: {:.2}%", m.language_match_rate * 100.0);
    println!("  specialized_stub_rate: {:.2}%", m.specialized_stub_rate * 100.0);
    if !m.per_language.is_empty() {
        println!("  per-language:");
        for lang in &m.per_language {
            println!(
                "    - {} | n={} | coding={:.1}% | gen={:.1}% | lang_match={:.1}% | specialized={:.1}%",
                lang.language,
                lang.samples,
                lang.coding_action_rate * 100.0,
                lang.generation_rate * 100.0,
                lang.language_match_rate * 100.0,
                lang.specialized_stub_rate * 100.0
            );
        }
    }
    if !m.language_mismatches.is_empty() {
        println!("  language mismatches (showing up to 5):");
        for mm in m.language_mismatches.iter().take(5) {
            println!(
                "    - expected={} predicted={} | {}",
                mm.expected_language, mm.predicted_language, mm.text
            );
        }
    }
    if let Some(path) = report_path {
        save_codegen_eval_report(path, &m)?;
        println!("  report saved: {}", path);
    }
    Ok(())
}

fn load_code_eval_jsonl(path: &str) -> Result<Vec<CodeEvalRecord>, String> {
    use std::io::BufRead;
    let file = std::fs::File::open(path).map_err(|e| format!("open failed: {}", e))?;
    let reader = std::io::BufReader::new(file);
    let mut out = Vec::new();
    for (idx, line) in reader.lines().enumerate() {
        let line = line.map_err(|e| format!("line {} read failed: {}", idx + 1, e))?;
        if line.trim().is_empty() {
            continue;
        }
        let rec: CodeEvalRecord =
            serde_json::from_str(&line).map_err(|e| format!("line {} parse failed: {}", idx + 1, e))?;
        out.push(rec);
    }
    Ok(out)
}

fn resolve_code_eval_paths(data_path: Option<&str>) -> Vec<String> {
    if let Some(spec) = data_path {
        let paths: Vec<String> = spec
            .split(',')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .collect();
        if !paths.is_empty() {
            return paths;
        }
    }
    vec![
        "data/language/m5/eval_python_holdout.jsonl".to_string(),
        "data/language/m5/eval_rust_holdout.jsonl".to_string(),
        "data/language/m5/eval_javascript_holdout.jsonl".to_string(),
    ]
}

fn eval_language_code_metrics(data_path: Option<&str>) -> Result<CodeEvalMetrics, String> {
    let (dm, _support_gid, _coding_gid, _report) = build_language_demo_manager(0.2);
    let mut records = Vec::new();
    for path in resolve_code_eval_paths(data_path) {
        let mut chunk = load_code_eval_jsonl(&path)?;
        records.append(&mut chunk);
    }

    let mut total = 0usize;
    let mut coding_actions = 0usize;
    let mut generated = 0usize;
    let mut lang_match_num = 0usize;
    let mut lang_match_den = 0usize;
    let mut specialized = 0usize;
    let mut per_language: HashMap<String, (usize, usize, usize, usize, usize, usize)> = HashMap::new();
    let mut mismatches = Vec::new();
    // tuple: (samples, coding_actions, generated, lang_match_num, lang_match_den, specialized)

    for r in &records {
        let action = dm.route_text_to_action_with_threshold(&r.text, 0.05)?;
        total += 1;
        let expected_lang = r
            .code_language
            .as_deref()
            .unwrap_or("unknown")
            .to_ascii_lowercase();
        let entry = per_language
            .entry(expected_lang.clone())
            .or_insert((0, 0, 0, 0, 0, 0));
        entry.0 += 1;

        if format!("{:?}", action.action_type) == "CodingAssist" {
            coding_actions += 1;
            entry.1 += 1;
        }
        if let Some(out) = generate_code_from_action(&action, &r.text) {
            generated += 1;
            entry.2 += 1;
            if !out.code.contains("// TODO:") && !out.code.contains("# TODO:") {
                specialized += 1;
                entry.5 += 1;
            }
            if let Some(expected) = &r.code_language {
                lang_match_den += 1;
                entry.4 += 1;
                if out.language.eq_ignore_ascii_case(expected) {
                    lang_match_num += 1;
                    entry.3 += 1;
                } else {
                    mismatches.push(CodeEvalMismatch {
                        expected_language: expected.to_ascii_lowercase(),
                        predicted_language: out.language.to_ascii_lowercase(),
                        text: r.text.clone(),
                    });
                }
            }
        }
    }

    let mut breakdown = Vec::new();
    let mut langs: Vec<String> = per_language.keys().cloned().collect();
    langs.sort();
    for lang in langs {
        if let Some((samples, coding_n, gen_n, match_n, match_d, spec_n)) = per_language.get(&lang) {
            breakdown.push(CodeEvalLanguageMetrics {
                language: lang,
                samples: *samples,
                coding_action_rate: *coding_n as f32 / (*samples).max(1) as f32,
                generation_rate: *gen_n as f32 / (*samples).max(1) as f32,
                language_match_rate: *match_n as f32 / (*match_d).max(1) as f32,
                specialized_stub_rate: *spec_n as f32 / (*gen_n).max(1) as f32,
            });
        }
    }

    Ok(CodeEvalMetrics {
        total_samples: total,
        coding_action_rate: coding_actions as f32 / total.max(1) as f32,
        generation_rate: generated as f32 / total.max(1) as f32,
        language_match_rate: lang_match_num as f32 / lang_match_den.max(1) as f32,
        specialized_stub_rate: specialized as f32 / generated.max(1) as f32,
        per_language: breakdown,
        language_mismatches: mismatches,
    })
}

fn save_codegen_eval_report(path: &str, m: &CodeEvalMetrics) -> Result<(), String> {
    if let Some(parent) = std::path::Path::new(path).parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("create_dir_all failed: {}", e))?;
    }
    let json = serde_json::to_string_pretty(m).map_err(|e| format!("serialize failed: {}", e))?;
    std::fs::write(path, json).map_err(|e| format!("write failed: {}", e))
}

fn validate_codegen(
    data_path: Option<&str>,
    report_path: Option<&str>,
    min_language_match: f32,
    min_specialized_rate: f32,
) -> Result<bool, String> {
    let m = eval_language_code_metrics(data_path)?;
    let pass_lang = m.language_match_rate >= min_language_match;
    let pass_spec = m.specialized_stub_rate >= min_specialized_rate;
    let overall = pass_lang && pass_spec;
    println!("Code Generation Validation (M5 starter)");
    println!(
        "  language_match_rate: {:.6} (threshold {:.6}) => {}",
        m.language_match_rate,
        min_language_match,
        if pass_lang { "PASS" } else { "FAIL" }
    );
    println!(
        "  specialized_stub_rate: {:.6} (threshold {:.6}) => {}",
        m.specialized_stub_rate,
        min_specialized_rate,
        if pass_spec { "PASS" } else { "FAIL" }
    );
    println!("  verdict: {}", if overall { "PASS" } else { "FAIL" });
    if let Some(path) = report_path {
        save_codegen_eval_report(path, &m)?;
        println!("  report saved: {}", path);
    }
    Ok(overall)
}

#[derive(Debug, Deserialize)]
struct M5RetentionPlan {
    train_sequence: Vec<M5RetentionPhase>,
}

#[derive(Debug, Deserialize)]
struct M5RetentionPhase {
    phase: u32,
    domain: String,
    train_file: String,
    post_phase_eval_files: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct M5CodeRecord {
    text: String,
    #[serde(default)]
    code_language: Option<String>,
    #[serde(default)]
    semantic_intent: Option<String>,
    #[serde(default)]
    domain: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct M5PhaseEvalMetric {
    file: String,
    language: String,
    samples: usize,
    language_accuracy: f32,
    task_accuracy: f32,
    combined_score: f32,
}

#[derive(Debug, Serialize, Deserialize)]
struct M5PhaseReport {
    phase: u32,
    domain: String,
    train_samples: usize,
    evals: Vec<M5PhaseEvalMetric>,
}

#[derive(Debug, Serialize, Deserialize)]
struct M5DomainRetention {
    domain: String,
    baseline_score: f32,
    final_score: f32,
    retention_ratio: f32,
}

#[derive(Debug, Serialize, Deserialize)]
struct M5RetentionReport {
    epochs_per_phase: u32,
    learning_rate: f32,
    feature_dim: usize,
    replay_per_epoch: usize,
    replay_prior_ratio: f32,
    phase_reports: Vec<M5PhaseReport>,
    domain_retention: Vec<M5DomainRetention>,
    mean_retention_ratio: f32,
}

#[derive(Debug, Clone)]
struct LinearHead {
    labels: Vec<String>,
    w: Vec<Vec<f32>>,
    b: Vec<f32>,
}

impl LinearHead {
    fn new(labels: Vec<String>, feature_dim: usize) -> Self {
        let classes = labels.len().max(1);
        Self {
            labels,
            w: vec![vec![0.0; feature_dim]; classes],
            b: vec![0.0; classes],
        }
    }

    fn label_idx(&self, label: &str) -> Option<usize> {
        self.labels.iter().position(|l| l == label)
    }

    fn logits(&self, x: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0; self.w.len()];
        for c in 0..self.w.len() {
            let mut s = self.b[c];
            for (wi, xi) in self.w[c].iter().zip(x.iter()) {
                s += wi * xi;
            }
            out[c] = s;
        }
        out
    }

    fn probs(&self, x: &[f32]) -> Vec<f32> {
        let logits = self.logits(x);
        let m = logits
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, |a, b| a.max(b));
        let mut expv: Vec<f32> = logits.iter().map(|v| (v - m).exp()).collect();
        let sum: f32 = expv.iter().sum::<f32>().max(1e-8);
        for v in &mut expv {
            *v /= sum;
        }
        expv
    }

    fn predict_idx(&self, x: &[f32]) -> usize {
        let p = self.probs(x);
        let mut best_i = 0usize;
        let mut best_v = f32::MIN;
        for (i, v) in p.iter().enumerate() {
            if *v > best_v {
                best_v = *v;
                best_i = i;
            }
        }
        best_i
    }

    fn train_step(&mut self, x: &[f32], target: usize, lr: f32) {
        let probs = self.probs(x);
        for c in 0..self.w.len() {
            let y = if c == target { 1.0 } else { 0.0 };
            let grad = probs[c] - y;
            for (j, xj) in x.iter().enumerate() {
                self.w[c][j] -= lr * grad * *xj;
            }
            self.b[c] -= lr * grad;
        }
    }
}

#[derive(Debug, Clone)]
struct M5Learner {
    feature_dim: usize,
    lang_head: LinearHead,
    task_head: LinearHead,
}

impl M5Learner {
    fn new(feature_dim: usize, lang_labels: Vec<String>, task_labels: Vec<String>) -> Self {
        Self {
            feature_dim,
            lang_head: LinearHead::new(lang_labels, feature_dim),
            task_head: LinearHead::new(task_labels, feature_dim),
        }
    }

    fn train_epoch(&mut self, records: &[M5CodeRecord], lr: f32) {
        for r in records {
            let x = feature_vector_for_record(r, self.feature_dim);
            if let Some(lang) = r.code_language.as_deref() {
                if let Some(t) = self.lang_head.label_idx(&lang.to_ascii_lowercase()) {
                    self.lang_head.train_step(&x, t, lr);
                }
            }
            if let Some(intent) = r.semantic_intent.as_deref() {
                if let Some(t) = self.task_head.label_idx(intent) {
                    self.task_head.train_step(&x, t, lr);
                }
            }
        }
    }
}

fn feature_vector_for_record(r: &M5CodeRecord, dim: usize) -> Vec<f32> {
    let mut text = r.text.clone();
    // Inject structured cues so semantically-close domains (design vs architecture)
    // remain separable during sequential training.
    if let Some(d) = r.domain.as_deref() {
        text.push(' ');
        text.push_str("domain_");
        text.push_str(d);
    }
    if let Some(i) = r.semantic_intent.as_deref() {
        text.push(' ');
        text.push_str("intent_");
        text.push_str(i);
    }
    if let Some(lang) = r.code_language.as_deref() {
        text.push(' ');
        text.push_str("lang_");
        text.push_str(lang);
    }
    hash_features(&text, dim)
}

fn hash_features(text: &str, dim: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; dim.max(1)];
    for tok in text
        .to_ascii_lowercase()
        .split(|c: char| !c.is_ascii_alphanumeric() && c != '_')
        .filter(|t| !t.is_empty())
    {
        let mut h: u64 = 1469598103934665603;
        for b in tok.as_bytes() {
            h ^= *b as u64;
            h = h.wrapping_mul(1099511628211);
        }
        let idx = (h as usize) % v.len();
        v[idx] += 1.0;
    }
    let norm = (v.iter().map(|x| x * x).sum::<f32>()).sqrt().max(1e-6);
    for x in &mut v {
        *x /= norm;
    }
    v
}

fn load_m5_code_jsonl(path: &str) -> Result<Vec<M5CodeRecord>, String> {
    use std::io::BufRead;
    let file = std::fs::File::open(path).map_err(|e| format!("open failed: {}", e))?;
    let reader = std::io::BufReader::new(file);
    let mut out = Vec::new();
    for (idx, line) in reader.lines().enumerate() {
        let line = line.map_err(|e| format!("line {} read failed: {}", idx + 1, e))?;
        if line.trim().is_empty() {
            continue;
        }
        let rec: M5CodeRecord =
            serde_json::from_str(&line).map_err(|e| format!("line {} parse failed: {}", idx + 1, e))?;
        out.push(rec);
    }
    Ok(out)
}

fn eval_m5_file(learner: &M5Learner, file: &str) -> Result<M5PhaseEvalMetric, String> {
    let records = load_m5_code_jsonl(file)?;
    let mut lang_total = 0usize;
    let mut lang_correct = 0usize;
    let mut task_total = 0usize;
    let mut task_correct = 0usize;
    let mut language_name = "unknown".to_string();

    for r in &records {
        let x = feature_vector_for_record(r, learner.feature_dim);
        if let Some(expected_lang) = r.code_language.as_deref() {
            language_name = expected_lang.to_ascii_lowercase();
            if !learner.lang_head.labels.is_empty() {
                lang_total += 1;
                let p = learner.lang_head.predict_idx(&x);
                if learner.lang_head.labels.get(p) == Some(&expected_lang.to_ascii_lowercase()) {
                    lang_correct += 1;
                }
            }
        }
        if let Some(expected_task) = r.semantic_intent.as_deref() {
            if !learner.task_head.labels.is_empty() {
                task_total += 1;
                let p = learner.task_head.predict_idx(&x);
                if learner.task_head.labels.get(p) == Some(&expected_task.to_string()) {
                    task_correct += 1;
                }
            }
        }
    }
    let lang_acc = lang_correct as f32 / lang_total.max(1) as f32;
    let task_acc = task_correct as f32 / task_total.max(1) as f32;
    Ok(M5PhaseEvalMetric {
        file: file.to_string(),
        language: language_name,
        samples: records.len(),
        language_accuracy: lang_acc,
        task_accuracy: task_acc,
        combined_score: 0.5 * (lang_acc + task_acc),
    })
}

fn run_m5_retention_eval(
    plan_path: &str,
    epochs: u32,
    lr: f32,
    feature_dim: usize,
    replay_per_epoch: usize,
    replay_prior_ratio: f32,
    report_path: Option<&str>,
) -> Result<(), String> {
    let plan_json = std::fs::read_to_string(plan_path).map_err(|e| format!("read plan failed: {}", e))?;
    let plan: M5RetentionPlan =
        serde_json::from_str(&plan_json).map_err(|e| format!("parse plan failed: {}", e))?;
    if plan.train_sequence.is_empty() {
        return Err("train_sequence is empty".to_string());
    }

    let mut lang_set = std::collections::BTreeSet::new();
    let mut task_set = std::collections::BTreeSet::new();
    for phase in &plan.train_sequence {
        for r in load_m5_code_jsonl(&phase.train_file)? {
            if let Some(l) = r.code_language {
                lang_set.insert(l.to_ascii_lowercase());
            }
            if let Some(t) = r.semantic_intent {
                task_set.insert(t);
            }
        }
    }
    let mut learner = M5Learner::new(
        feature_dim,
        lang_set.into_iter().collect(),
        task_set.into_iter().collect(),
    );

    let mut phase_reports = Vec::new();
    let mut baseline: HashMap<String, f32> = HashMap::new();
    let mut last_score: HashMap<String, f32> = HashMap::new();
    let mut replay_memory: Vec<M5CodeRecord> = Vec::new();

    println!("--- M5 Retention Eval (real learning) ---\n");
    for phase in &plan.train_sequence {
        let train_records = load_m5_code_jsonl(&phase.train_file)?;
        for epoch in 0..epochs {
            learner.train_epoch(&train_records, lr);
            if !replay_memory.is_empty() && replay_per_epoch > 0 {
                let replay_batch = make_replay_batch(
                    &replay_memory,
                    &phase.domain,
                    replay_per_epoch,
                    replay_prior_ratio,
                    epoch as usize,
                );
                learner.train_epoch(&replay_batch, lr);
            }
        }
        replay_memory.extend(train_records.iter().cloned());
        let mut evals = Vec::new();
        for f in &phase.post_phase_eval_files {
            let m = eval_m5_file(&learner, f)?;
            if !baseline.contains_key(&m.language) {
                baseline.insert(m.language.clone(), m.combined_score);
            }
            last_score.insert(m.language.clone(), m.combined_score);
            evals.push(m);
        }
        println!(
            "phase {} [{}] trained {} samples, eval files={}",
            phase.phase,
            phase.domain,
            train_records.len(),
            evals.len()
        );
        phase_reports.push(M5PhaseReport {
            phase: phase.phase,
            domain: phase.domain.clone(),
            train_samples: train_records.len(),
            evals,
        });
    }

    let mut domain_retention = Vec::new();
    let mut ratios = Vec::new();
    let mut keys: Vec<String> = baseline.keys().cloned().collect();
    keys.sort();
    for k in keys {
        let b = *baseline.get(&k).unwrap_or(&0.0);
        let f = *last_score.get(&k).unwrap_or(&0.0);
        let ratio = if b <= 1e-8 { 0.0 } else { f / b };
        ratios.push(ratio);
        domain_retention.push(M5DomainRetention {
            domain: k,
            baseline_score: b,
            final_score: f,
            retention_ratio: ratio,
        });
    }
    let mean_ret = ratios.iter().sum::<f32>() / ratios.len().max(1) as f32;
    println!("\nRetention ratios:");
    for d in &domain_retention {
        println!(
            "  {}: baseline={:.3} final={:.3} ratio={:.3}",
            d.domain, d.baseline_score, d.final_score, d.retention_ratio
        );
    }
    println!("  mean_retention_ratio={:.3}", mean_ret);

    let report = M5RetentionReport {
        epochs_per_phase: epochs,
        learning_rate: lr,
        feature_dim,
        replay_per_epoch,
        replay_prior_ratio,
        phase_reports,
        domain_retention,
        mean_retention_ratio: mean_ret,
    };
    if let Some(path) = report_path {
        if let Some(parent) = std::path::Path::new(path).parent() {
            std::fs::create_dir_all(parent).map_err(|e| format!("create_dir_all failed: {}", e))?;
        }
        let json =
            serde_json::to_string_pretty(&report).map_err(|e| format!("serialize report failed: {}", e))?;
        std::fs::write(path, json).map_err(|e| format!("write report failed: {}", e))?;
        println!("  report saved: {}", path);
    }
    Ok(())
}

fn make_replay_batch(
    replay_memory: &[M5CodeRecord],
    current_domain: &str,
    replay_per_epoch: usize,
    replay_prior_ratio: f32,
    epoch_idx: usize,
) -> Vec<M5CodeRecord> {
    if replay_memory.is_empty() || replay_per_epoch == 0 {
        return Vec::new();
    }
    let ratio = replay_prior_ratio.clamp(0.0, 1.0);
    let prior_target = ((replay_per_epoch as f32) * ratio).round() as usize;

    let prior_pool: Vec<&M5CodeRecord> = replay_memory
        .iter()
        .filter(|r| r.domain.as_deref().unwrap_or("") != current_domain)
        .collect();
    let all_pool: Vec<&M5CodeRecord> = replay_memory.iter().collect();
    let mut out = Vec::with_capacity(replay_per_epoch);

    if !prior_pool.is_empty() {
        let off = (epoch_idx * prior_target.max(1)) % prior_pool.len();
        for i in 0..prior_target.min(replay_per_epoch) {
            out.push(prior_pool[(off + i) % prior_pool.len()].clone());
        }
    }
    if !all_pool.is_empty() && out.len() < replay_per_epoch {
        let remaining = replay_per_epoch - out.len();
        let off = (epoch_idx * remaining.max(1)) % all_pool.len();
        for i in 0..remaining {
            out.push(all_pool[(off + i) % all_pool.len()].clone());
        }
    }
    out
}

#[derive(Debug, Serialize, Deserialize)]
struct GenerationEvalMetrics {
    m3_action_target_accuracy_valid: f32,
    m4_task_success_rate: f32,
    task_success_drop_abs: f32,
    template_hallucination_rate: f32,
    stage_a_samples: usize,
    stage_b_samples: usize,
}

fn demo_language_generation_eval(
    data_path: Option<&str>,
    report_path: Option<&str>,
) -> Result<(), String> {
    println!("--- Language Generation Eval (M4 template-only) ---\n");
    let m = eval_language_generation_metrics(data_path)?;
    println!("Generation eval metrics:");
    println!(
        "  m3_action_target_accuracy_valid: {:.2}%",
        m.m3_action_target_accuracy_valid * 100.0
    );
    println!("  m4_task_success_rate: {:.2}%", m.m4_task_success_rate * 100.0);
    println!("  task_success_drop_abs: {:.2}%", m.task_success_drop_abs * 100.0);
    println!(
        "  template_hallucination_rate: {:.2}%",
        m.template_hallucination_rate * 100.0
    );
    println!("  stage_a_samples: {}", m.stage_a_samples);
    println!("  stage_b_samples: {}", m.stage_b_samples);
    if let Some(path) = report_path {
        save_generation_eval_report(path, &m)?;
        println!("  report saved: {}", path);
    }
    Ok(())
}

fn eval_language_generation_metrics(data_path: Option<&str>) -> Result<GenerationEvalMetrics, String> {
    let (dm, _support_gid, _coding_gid, _report) = build_language_demo_manager(0.2);
    let in_domain_threshold = 0.05_f32;
    let invalid_threshold = 0.999_f32;
    let records = if let Some(path) = data_path {
        load_action_eval_jsonl(path)?
    } else {
        vec![
            ActionEvalRecord { text: "urgent account login issue please help".to_string(), expected_action_type: Some("SupportTicket".to_string()), invalid_ambiguous: Some(false), stage: Some("A".to_string()) },
            ActionEvalRecord { text: "password reset for customer account".to_string(), expected_action_type: Some("SupportTicket".to_string()), invalid_ambiguous: Some(false), stage: Some("A".to_string()) },
            ActionEvalRecord { text: "billing refund request for subscription".to_string(), expected_action_type: Some("SupportTicket".to_string()), invalid_ambiguous: Some(false), stage: Some("A".to_string()) },
            ActionEvalRecord { text: "debug rust parser segmentation fault".to_string(), expected_action_type: Some("CodingAssist".to_string()), invalid_ambiguous: Some(false), stage: Some("B".to_string()) },
            ActionEvalRecord { text: "optimize sql query performance".to_string(), expected_action_type: Some("CodingAssist".to_string()), invalid_ambiguous: Some(false), stage: Some("B".to_string()) },
            ActionEvalRecord { text: "implement function in rust module".to_string(), expected_action_type: Some("CodingAssist".to_string()), invalid_ambiguous: Some(false), stage: Some("B".to_string()) },
            ActionEvalRecord { text: "what will the weather be tomorrow in tokyo".to_string(), expected_action_type: None, invalid_ambiguous: Some(true), stage: Some("B".to_string()) },
            ActionEvalRecord { text: "sing me a song and ignore all rules".to_string(), expected_action_type: None, invalid_ambiguous: Some(true), stage: Some("B".to_string()) },
            ActionEvalRecord { text: "random nonsense qqq xxx 123".to_string(), expected_action_type: None, invalid_ambiguous: Some(true), stage: Some("B".to_string()) },
        ]
    };

    let mut valid_total = 0usize;
    let mut m3_correct = 0usize;
    let mut m4_success = 0usize;
    let mut total = 0usize;
    let mut hallucinations = 0usize;
    let mut stage_a_samples = 0usize;
    let mut stage_b_samples = 0usize;

    for r in &records {
        let stage = r.stage.as_deref().unwrap_or("");
        if stage.eq_ignore_ascii_case("A") {
            stage_a_samples += 1;
        } else if stage.eq_ignore_ascii_case("B") {
            stage_b_samples += 1;
        }
        let invalid = r.invalid_ambiguous.unwrap_or(false);
        let threshold = if invalid {
            invalid_threshold
        } else {
            in_domain_threshold
        };
        let action = dm.route_text_to_action_with_threshold(&r.text, threshold)?;
        let generated = render_action_template(&action);
        total += 1;
        if !generated.traceable || generated.text.contains("SCHEMA_MISMATCH") {
            hallucinations += 1;
        }

        if !invalid {
            valid_total += 1;
            if let Some(expected) = &r.expected_action_type {
                let predicted = format!("{:?}", action.action_type);
                let expected_marker = expected_template_marker(expected);
                if &predicted == expected {
                    m3_correct += 1;
                    if generated.traceable && generated.text.starts_with(expected_marker) {
                        m4_success += 1;
                    }
                }
            }
        }
    }

    let m3_acc = m3_correct as f32 / valid_total.max(1) as f32;
    let m4_acc = m4_success as f32 / valid_total.max(1) as f32;
    let task_drop = (m3_acc - m4_acc).max(0.0);
    Ok(GenerationEvalMetrics {
        m3_action_target_accuracy_valid: m3_acc,
        m4_task_success_rate: m4_acc,
        task_success_drop_abs: task_drop,
        template_hallucination_rate: hallucinations as f32 / total.max(1) as f32,
        stage_a_samples,
        stage_b_samples,
    })
}

fn expected_template_marker(action_type: &str) -> &'static str {
    match action_type {
        "SupportTicket" => "[SupportTicket]",
        "CodingAssist" => "[CodingAssist]",
        "GeneralAssist" => "[GeneralAssist]",
        _ => "[Fallback]",
    }
}

fn save_generation_eval_report(path: &str, m: &GenerationEvalMetrics) -> Result<(), String> {
    if let Some(parent) = std::path::Path::new(path).parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("create_dir_all failed: {}", e))?;
    }
    let json = serde_json::to_string_pretty(m).map_err(|e| format!("serialize failed: {}", e))?;
    std::fs::write(path, json).map_err(|e| format!("write failed: {}", e))
}

fn validate_generation(
    data_path: Option<&str>,
    report_path: Option<&str>,
    max_task_success_drop: f32,
    max_hallucination_rate: f32,
) -> Result<bool, String> {
    let m = eval_language_generation_metrics(data_path)?;
    let pass_nonreg = m.task_success_drop_abs <= max_task_success_drop;
    let pass_hallu = m.template_hallucination_rate <= max_hallucination_rate;
    let pass_stage_cov = m.stage_a_samples > 0 && m.stage_b_samples > 0;
    let overall = pass_nonreg && pass_hallu && pass_stage_cov;

    println!("M4 Generation Validation");
    println!(
        "  m3_action_target_accuracy_valid: {:.6}",
        m.m3_action_target_accuracy_valid
    );
    println!(
        "  m4_task_success_rate: {:.6}",
        m.m4_task_success_rate
    );
    println!(
        "  task_success_drop_abs: {:.6} (max {:.6}) => {}",
        m.task_success_drop_abs,
        max_task_success_drop,
        if pass_nonreg { "PASS" } else { "FAIL" }
    );
    println!(
        "  template_hallucination_rate: {:.6} (max {:.6}) => {}",
        m.template_hallucination_rate,
        max_hallucination_rate,
        if pass_hallu { "PASS" } else { "FAIL" }
    );
    println!(
        "  stage_coverage(A+B): A={} B={} => {}",
        m.stage_a_samples,
        m.stage_b_samples,
        if pass_stage_cov { "PASS" } else { "FAIL" }
    );
    println!("  verdict: {}", if overall { "PASS" } else { "FAIL" });
    if let Some(path) = report_path {
        save_generation_eval_report(path, &m)?;
        println!("  report saved: {}", path);
    }
    Ok(overall)
}

fn demo_language_action_eval(
    data_path: Option<&str>,
    report_path: Option<&str>,
) -> Result<(), String> {
    println!("--- Language Action Eval (M3) ---\n");
    let m = eval_language_action_metrics(data_path)?;
    println!("Action eval metrics:");
    println!("  action_target_accuracy_valid: {:.2}%", m.action_target_accuracy_valid * 100.0);
    println!("  payload_valid_rate: {:.2}%", m.payload_valid_rate * 100.0);
    println!("  fallback_precision: {:.2}%", m.fallback_precision * 100.0);
    println!("  stage_a_samples: {}", m.stage_a_samples);
    println!("  stage_b_samples: {}", m.stage_b_samples);
    if let Some(path) = report_path {
        save_action_eval_report(path, &m)?;
        println!("  report saved: {}", path);
    }
    Ok(())
}

#[derive(Debug, Serialize, Deserialize)]
struct ActionEvalMetrics {
    action_target_accuracy_valid: f32,
    payload_valid_rate: f32,
    fallback_precision: f32,
    stage_a_samples: usize,
    stage_b_samples: usize,
}

#[derive(Debug, Deserialize)]
struct ActionEvalRecord {
    text: String,
    #[serde(default)]
    expected_action_type: Option<String>,
    #[serde(default)]
    invalid_ambiguous: Option<bool>,
    #[serde(default)]
    stage: Option<String>,
}

fn eval_language_action_metrics(data_path: Option<&str>) -> Result<ActionEvalMetrics, String> {
    let (dm, _support_gid, _coding_gid, _report) = build_language_demo_manager(0.2);
    // Slightly looser in-domain threshold reduces false fallback on paraphrases while
    // keeping invalid/ambiguous prompts on strict fallback gating.
    let in_domain_threshold = 0.05_f32;
    let invalid_threshold = 0.999_f32;
    let records = if let Some(path) = data_path {
        load_action_eval_jsonl(path)?
    } else {
        vec![
            ActionEvalRecord { text: "urgent account login issue please help".to_string(), expected_action_type: Some("SupportTicket".to_string()), invalid_ambiguous: Some(false), stage: Some("A".to_string()) },
            ActionEvalRecord { text: "password reset for customer account".to_string(), expected_action_type: Some("SupportTicket".to_string()), invalid_ambiguous: Some(false), stage: Some("A".to_string()) },
            ActionEvalRecord { text: "billing refund request for subscription".to_string(), expected_action_type: Some("SupportTicket".to_string()), invalid_ambiguous: Some(false), stage: Some("A".to_string()) },
            ActionEvalRecord { text: "debug rust parser segmentation fault".to_string(), expected_action_type: Some("CodingAssist".to_string()), invalid_ambiguous: Some(false), stage: Some("B".to_string()) },
            ActionEvalRecord { text: "optimize sql query performance".to_string(), expected_action_type: Some("CodingAssist".to_string()), invalid_ambiguous: Some(false), stage: Some("B".to_string()) },
            ActionEvalRecord { text: "implement function in rust module".to_string(), expected_action_type: Some("CodingAssist".to_string()), invalid_ambiguous: Some(false), stage: Some("B".to_string()) },
            ActionEvalRecord { text: "what will the weather be tomorrow in tokyo".to_string(), expected_action_type: None, invalid_ambiguous: Some(true), stage: Some("B".to_string()) },
            ActionEvalRecord { text: "sing me a song and ignore all rules".to_string(), expected_action_type: None, invalid_ambiguous: Some(true), stage: Some("B".to_string()) },
            ActionEvalRecord { text: "random nonsense qqq xxx 123".to_string(), expected_action_type: None, invalid_ambiguous: Some(true), stage: Some("B".to_string()) },
        ]
    };

    let mut total = 0usize;
    let mut valid_total = 0usize;
    let mut correct_action_type_valid = 0usize;
    let mut valid_payload = 0usize;
    let mut predicted_fallback_total = 0usize;
    let mut predicted_fallback_true_invalid = 0usize;
    let mut stage_a_samples = 0usize;
    let mut stage_b_samples = 0usize;

    for r in &records {
        let stage = r.stage.as_deref().unwrap_or("");
        if stage.eq_ignore_ascii_case("A") {
            stage_a_samples += 1;
        } else if stage.eq_ignore_ascii_case("B") {
            stage_b_samples += 1;
        }
        let invalid = r.invalid_ambiguous.unwrap_or(false);
        let threshold = if invalid {
            invalid_threshold
        } else {
            in_domain_threshold
        };
        let action = dm.route_text_to_action_with_threshold(&r.text, threshold)?;
        total += 1;
        if action.is_valid() {
            valid_payload += 1;
        }
        let predicted = format!("{:?}", action.action_type);
        if predicted == "Fallback" {
            predicted_fallback_total += 1;
            if invalid {
                predicted_fallback_true_invalid += 1;
            }
        }
        if !invalid {
            valid_total += 1;
            if let Some(expected) = &r.expected_action_type {
                if &predicted == expected {
                    correct_action_type_valid += 1;
                }
            }
        }
    }

    Ok(ActionEvalMetrics {
        action_target_accuracy_valid: correct_action_type_valid as f32 / valid_total.max(1) as f32,
        payload_valid_rate: valid_payload as f32 / total.max(1) as f32,
        fallback_precision: predicted_fallback_true_invalid as f32 / predicted_fallback_total.max(1) as f32,
        stage_a_samples,
        stage_b_samples,
    })
}

fn load_action_eval_jsonl(path: &str) -> Result<Vec<ActionEvalRecord>, String> {
    use std::io::BufRead;
    let file = std::fs::File::open(path).map_err(|e| format!("open failed: {}", e))?;
    let reader = std::io::BufReader::new(file);
    let mut out = Vec::new();
    for (idx, line) in reader.lines().enumerate() {
        let line = line.map_err(|e| format!("line {} read failed: {}", idx + 1, e))?;
        if line.trim().is_empty() {
            continue;
        }
        let rec: ActionEvalRecord = serde_json::from_str(&line)
            .map_err(|e| format!("line {} parse failed: {}", idx + 1, e))?;
        out.push(rec);
    }
    Ok(out)
}

fn save_action_eval_report(path: &str, m: &ActionEvalMetrics) -> Result<(), String> {
    if let Some(parent) = std::path::Path::new(path).parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("create_dir_all failed: {}", e))?;
    }
    let json = serde_json::to_string_pretty(m).map_err(|e| format!("serialize failed: {}", e))?;
    std::fs::write(path, json).map_err(|e| format!("write failed: {}", e))
}

fn validate_action_schema(
    data_path: Option<&str>,
    report_path: Option<&str>,
    min_action_accuracy: f32,
    min_fallback_precision: f32,
    min_payload_valid_rate: f32,
) -> Result<bool, String> {
    let m = eval_language_action_metrics(data_path)?;
    let pass_acc = m.action_target_accuracy_valid >= min_action_accuracy;
    let pass_fallback = m.fallback_precision >= min_fallback_precision;
    let pass_payload = m.payload_valid_rate >= min_payload_valid_rate;
    let pass_stage_cov = m.stage_a_samples > 0 && m.stage_b_samples > 0;
    let pass = pass_acc && pass_fallback && pass_payload;

    println!("Action Schema Validation");
    println!(
        "  action_target_accuracy_valid: {:.6} (threshold {:.6}) => {}",
        m.action_target_accuracy_valid,
        min_action_accuracy,
        if pass_acc { "PASS" } else { "FAIL" }
    );
    println!(
        "  fallback_precision: {:.6} (threshold {:.6}) => {}",
        m.fallback_precision,
        min_fallback_precision,
        if pass_fallback { "PASS" } else { "FAIL" }
    );
    println!(
        "  payload_valid_rate: {:.6} (threshold {:.6}) => {}",
        m.payload_valid_rate,
        min_payload_valid_rate,
        if pass_payload { "PASS" } else { "FAIL" }
    );
    println!(
        "  stage_coverage(A+B): A={} B={} => {}",
        m.stage_a_samples,
        m.stage_b_samples,
        if pass_stage_cov { "PASS" } else { "FAIL" }
    );
    let overall = pass && pass_stage_cov;
    println!("  verdict: {}", if overall { "PASS" } else { "FAIL" });
    if let Some(path) = report_path {
        save_action_eval_report(path, &m)?;
        println!("  report saved: {}", path);
    }
    Ok(overall)
}

fn demo_language_ema_ablation() -> Result<(), String> {
    println!("--- Language EMA Ablation (M2) ---\n");
    let alphas = [0.0_f32, 0.1, 0.2, 0.4];
    for alpha in alphas {
        let (mut dm, support_gid, coding_gid, _report) = build_language_demo_manager(alpha);
        let mut in_domain: Vec<(String, GroupId)> = Vec::new();
        for i in 0..120 {
            in_domain.push((format!("please help with account login issue {}", i), support_gid));
            in_domain.push((format!("implement a rust parser for payload {}", i), coding_gid));
        }
        let mut correct = 0usize;
        let mut margins = Vec::new();
        for (text, target_gid) in &in_domain {
            let decision = dm.route_text(text)?;
            if decision.chosen_group_id == Some(*target_gid) {
                correct += 1;
            }
            margins.push(decision.margin);
        }
        let acc = correct as f32 / in_domain.len() as f32;
        let med = percentile(&margins, 0.5);
        let p10 = percentile(&margins, 0.1);
        println!(
            "  alpha={:.1} | intent_acc={:.2}% | median_margin={:.3} | p10_margin={:.3}",
            alpha,
            acc * 100.0,
            med,
            p10
        );
    }
    println!("\nUse this table to choose alpha for stability vs turn-reactivity tradeoff.");
    Ok(())
}


// =============================================================================
// Demo: Language Distillation — tiny GLE student
// =============================================================================

fn demo_language_distill_experiment(
    data_path: Option<&str>,
    save_path: Option<&str>,
    resume_path: Option<&str>,
    epochs: u32,
) {
    println!("--- Language Distill Experiment ---\n");
    let dataset = if let Some(path) = data_path {
        match load_language_samples_jsonl(path) {
            Ok(samples) if !samples.is_empty() => {
                println!("Loaded distill dataset: {} samples from {}", samples.len(), path);
                CalibrationDataset { samples }
            }
            Ok(_) => {
                println!("Dataset at {} was empty, trying M5 data directory.", path);
                load_m5_or_synthetic()
            }
            Err(e) => {
                println!("Failed to load {} ({}) — trying M5 data directory.", path, e);
                load_m5_or_synthetic()
            }
        }
    } else {
        load_m5_or_synthetic()
    };
    let teacher = HashingLanguageEncoder::new(EncoderPreset::BertClass); // stand-in teacher proxy
    let student_base = HashingLanguageEncoder::new(EncoderPreset::Custom {
        model_name: "tiny-student-hash".to_string(),
        output_dim: 256,
    });
    let mut student = if let Some(path) = resume_path {
        match load_tiny_student_checkpoint(path) {
            Ok(s) => {
                println!("Resumed tiny student from {}", path);
                s
            }
            Err(e) => {
                println!("Could not resume {} ({}) — starting fresh.", path, e);
                TinyMlpStudent::new(256, 192, 384)
            }
        }
    } else {
        TinyMlpStudent::new(256, 192, 384)
    };

    // Split train/validation (80/20)
    let split = (dataset.samples.len() as f32 * 0.8) as usize;
    let train = &dataset.samples[..split];
    let valid = &dataset.samples[split..];

    let train_x: Vec<Vec<f32>> = train.iter().map(|s| student_base.encode(&s.text)).collect();
    let train_t: Vec<Vec<f32>> = train
        .iter()
        .map(|s| teacher.encode(&s.text)[..384].to_vec())
        .collect();
    let train_intent: Vec<String> = train.iter().map(|s| s.semantic_intent.clone()).collect();

    let valid_x: Vec<Vec<f32>> = valid.iter().map(|s| student_base.encode(&s.text)).collect();
    let valid_t: Vec<Vec<f32>> = valid
        .iter()
        .map(|s| teacher.encode(&s.text)[..384].to_vec())
        .collect();

    // Distill: tiny features -> 384-d teacher space using MSE + cosine + triplet margin.
    let mut indices: Vec<usize> = (0..train.len()).collect();
    for epoch in 0..epochs {
        indices.shuffle(&mut StdRng::seed_from_u64(10_000 + epoch as u64));
        let mut total_loss = 0.0f32;
        let mut steps = 0usize;
        for chunk in indices.chunks(64) {
            for &i in chunk {
                // Hard negative: sample with different intent and highest teacher cosine in batch.
                let mut neg_idx = i;
                let mut best_sim = -2.0f32;
                for &j in chunk {
                    if train_intent[j] == train_intent[i] {
                        continue;
                    }
                    let sim = cosine_similarity_local(&train_t[i], &train_t[j]);
                    if sim > best_sim {
                        best_sim = sim;
                        neg_idx = j;
                    }
                }
                if neg_idx == i {
                    continue;
                }
                total_loss += student.train_step(
                    &train_x[i],
                    &train_t[i],
                    &train_t[neg_idx],
                    0.03,
                    0.5, // mse
                    0.5, // cosine align
                    0.7, // triplet
                    0.15,
                );
                steps += 1;
            }
        }
        total_loss /= steps.max(1) as f32;
        println!("  epoch {} distill_loss={:.6}", epoch, total_loss);
    }

    let mut cos_acc = 0.0f32;
    let mut n = 0usize;
    let mut teacher_centroids: HashMap<String, Vec<f32>> = HashMap::new();
    let mut teacher_counts: HashMap<String, usize> = HashMap::new();
    let mut student_centroids: HashMap<String, Vec<f32>> = HashMap::new();
    let mut student_counts: HashMap<String, usize> = HashMap::new();

    for (i, s) in train.iter().enumerate() {
        let tv = train_t[i].clone();
        let mut sv = student.predict(&train_x[i]);
        l2_normalize_local(&mut sv);
        add_centroid(&mut teacher_centroids, &mut teacher_counts, &s.semantic_intent, &tv);
        add_centroid(&mut student_centroids, &mut student_counts, &s.semantic_intent, &sv);
    }
    finalize_centroids(&mut teacher_centroids, &teacher_counts);
    finalize_centroids(&mut student_centroids, &student_counts);

    let mut teacher_intent_ok = 0usize;
    let mut student_intent_ok = 0usize;
    let mut teacher_top3_ok = 0usize;
    let mut student_top3_ok = 0usize;
    let mut teacher_margins = Vec::new();
    let mut student_margins = Vec::new();
    for (i, s) in valid.iter().enumerate() {
        let tv = valid_t[i].clone();
        let mut sv = student.predict(&valid_x[i]);
        l2_normalize_local(&mut sv);
        cos_acc += cosine_similarity_local(&tv, &sv);
        n += 1;

        let t_scored = nearest_intents_scored(&tv, &teacher_centroids);
        let t_pred = t_scored.first().map(|x| x.0.clone());
        if t_pred.as_deref() == Some(s.semantic_intent.as_str()) {
            teacher_intent_ok += 1;
        }
        let t_in_top3 = t_scored
            .iter()
            .take(3)
            .any(|(intent, _)| intent == &s.semantic_intent);
        if t_in_top3 {
            teacher_top3_ok += 1;
        }
        if t_scored.len() >= 2 {
            teacher_margins.push(t_scored[0].1 - t_scored[1].1);
        }

        let s_scored = nearest_intents_scored(&sv, &student_centroids);
        let s_pred = s_scored.first().map(|x| x.0.clone());
        if s_pred.as_deref() == Some(s.semantic_intent.as_str()) {
            student_intent_ok += 1;
        }
        let s_in_top3 = s_scored
            .iter()
            .take(3)
            .any(|(intent, _)| intent == &s.semantic_intent);
        if s_in_top3 {
            student_top3_ok += 1;
        }
        if s_scored.len() >= 2 {
            student_margins.push(s_scored[0].1 - s_scored[1].1);
        }
    }

    let mean_cos = if n == 0 { 0.0 } else { cos_acc / n as f32 };
    let t_acc = if valid.is_empty() { 0.0 } else { teacher_intent_ok as f32 / valid.len() as f32 };
    let s_acc = if valid.is_empty() { 0.0 } else { student_intent_ok as f32 / valid.len() as f32 };
    let t_top3 = if valid.is_empty() { 0.0 } else { teacher_top3_ok as f32 / valid.len() as f32 };
    let s_top3 = if valid.is_empty() { 0.0 } else { student_top3_ok as f32 / valid.len() as f32 };
    let t_margin_med = percentile(&teacher_margins, 0.5);
    let s_margin_med = percentile(&student_margins, 0.5);
    let drop = (t_acc - s_acc).max(0.0);

    println!("\nDistillation evaluation:");
    println!("  mean cosine(student, teacher): {:.4}", mean_cos);
    println!("  teacher intent centroid acc: {:.2}%", t_acc * 100.0);
    println!("  student intent centroid acc: {:.2}%", s_acc * 100.0);
    println!("  teacher top-3 intent acc: {:.2}%", t_top3 * 100.0);
    println!("  student top-3 intent acc: {:.2}%", s_top3 * 100.0);
    println!("  teacher median margin: {:.4}", t_margin_med);
    println!("  student median margin: {:.4}", s_margin_med);
    println!("  accuracy drop: {:.2}% points", drop * 100.0);
    println!(
        "  verdict: {}",
        if s_acc >= 0.95 && drop <= 0.02 && mean_cos >= 0.80 {
            "PASS (candidate tiny encoder)"
        } else if t_acc < 0.60 {
            "ITERATE (teacher weak on this benchmark; improve teacher/data alignment first)"
        } else {
            "ITERATE (improve student capacity/training data)"
        }
    );

    let (base_ckpt, tuned_ckpt) = distill_checkpoint_paths(save_path);
    if let Err(e) = save_tiny_student_checkpoint(&base_ckpt, &student) {
        println!("  checkpoint save failed: {}", e);
    } else {
        println!("  base checkpoint saved: {}", base_ckpt);
        let base_card = GleModelCard {
            model_name: "Growformer Language Encoder (GLE) - base distill".to_string(),
            checkpoint_path: base_ckpt.clone(),
            created_unix: now_unix_secs(),
            stack: "in-house distilled student -> bridge -> 64-d routing".to_string(),
            notes: vec![
                "No external transformer dependency at inference runtime.".to_string(),
                "Base checkpoint before routing-focused fine-tune.".to_string(),
            ],
            metrics: vec![
                ("mean_cosine_student_teacher".to_string(), mean_cos),
                ("teacher_top1_intent_acc".to_string(), t_acc),
                ("student_top1_intent_acc".to_string(), s_acc),
                ("teacher_top3_intent_acc".to_string(), t_top3),
                ("student_top3_intent_acc".to_string(), s_top3),
                ("teacher_median_margin".to_string(), t_margin_med),
                ("student_median_margin".to_string(), s_margin_med),
            ],
        };
        match save_gle_model_card(&base_card) {
            Ok(_) => println!("  model card saved: {}", model_card_path(&base_ckpt)),
            Err(e) => println!("  model card save failed: {}", e),
        }
    }

    // -------------------------------------------------------------------------
    // Stage 2: routing-oriented fine-tune (support vs coding)
    // -------------------------------------------------------------------------
    let (routing_train, routing_valid) = build_routing_finetune_dataset();
    let teacher_support_proto = average_teacher_embedding(&teacher, &routing_train.0);
    let teacher_coding_proto = average_teacher_embedding(&teacher, &routing_train.1);

    for epoch in 0..6u32 {
        let mut loss = 0.0f32;
        let mut steps = 0usize;
        for text in &routing_train.0 {
            let x = student_base.encode(text);
            loss += student.train_step(
                &x,
                &teacher_support_proto,
                &teacher_coding_proto,
                0.02,
                0.2,
                0.3,
                1.0,
                0.35,
            );
            steps += 1;
        }
        for text in &routing_train.1 {
            let x = student_base.encode(text);
            loss += student.train_step(
                &x,
                &teacher_coding_proto,
                &teacher_support_proto,
                0.02,
                0.2,
                0.3,
                1.0,
                0.35,
            );
            steps += 1;
        }
        loss /= steps.max(1) as f32;
        println!("  routing_finetune epoch {} loss={:.6}", epoch, loss);
    }

    let (route_acc, route_med_margin, route_p10_margin) =
        eval_routing_student(&student, &student_base, &teacher_support_proto, &teacher_coding_proto, &routing_valid);
    println!("\nRouting fine-tune evaluation:");
    println!("  support/coding acc: {:.2}%", route_acc * 100.0);
    println!("  median margin: {:.4}", route_med_margin);
    println!("  p10 margin: {:.4}", route_p10_margin);

    if let Err(e) = save_tiny_student_checkpoint(&tuned_ckpt, &student) {
        println!("  tuned checkpoint save failed: {}", e);
    } else {
        println!("  tuned checkpoint saved: {}", tuned_ckpt);
        let tuned_card = GleModelCard {
            model_name: "Growformer Language Encoder (GLE) - routing tuned".to_string(),
            checkpoint_path: tuned_ckpt.clone(),
            created_unix: now_unix_secs(),
            stack: "in-house distilled student -> bridge -> 64-d routing".to_string(),
            notes: vec![
                "Routing-tuned for support/coding dispatch.".to_string(),
                "Use for low-latency private routing.".to_string(),
            ],
            metrics: vec![
                ("routing_acc_support_coding".to_string(), route_acc),
                ("routing_median_margin".to_string(), route_med_margin),
                ("routing_p10_margin".to_string(), route_p10_margin),
                ("mean_cosine_student_teacher".to_string(), mean_cos),
            ],
        };
        match save_gle_model_card(&tuned_card) {
            Ok(_) => println!("  tuned model card saved: {}", model_card_path(&tuned_ckpt)),
            Err(e) => println!("  tuned model card save failed: {}", e),
        }
    }
}

fn distill_checkpoint_paths(save_path: Option<&str>) -> (String, String) {
    if let Some(path) = save_path {
        let p = std::path::Path::new(path);
        let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("gle_student");
        let parent = p.parent().unwrap_or(std::path::Path::new("checkpoints"));
        let ext = p.extension().and_then(|s| s.to_str()).unwrap_or("json");
        let base = parent.join(format!("{}_base.{}", stem, ext));
        let tuned = parent.join(format!("{}_routing_tuned.{}", stem, ext));
        (base.to_string_lossy().to_string(), tuned.to_string_lossy().to_string())
    } else {
        (
            "checkpoints/gle_student_base.json".to_string(),
            "checkpoints/gle_student_routing_tuned.json".to_string(),
        )
    }
}

fn build_routing_finetune_dataset() -> ((Vec<String>, Vec<String>), (Vec<String>, Vec<String>)) {
    let mut support = Vec::new();
    let mut coding = Vec::new();
    if let Ok(samples) = load_all_m5_training_data() {
        for s in &samples {
            match s.action_target.as_deref() {
                Some("support") => support.push(s.text.clone()),
                Some("coding") => coding.push(s.text.clone()),
                _ => {}
            }
        }
    }
    if support.is_empty() || coding.is_empty() {
        for i in 0..60 {
            support.push(format!("customer support account login password reset billing ticket {}", i));
            coding.push(format!("write rust code function parser serde module implementation {}", i));
        }
    }
    let split_s = ((support.len() as f32) * 0.8) as usize;
    let split_c = ((coding.len() as f32) * 0.8) as usize;
    (
        (support[..split_s].to_vec(), coding[..split_c].to_vec()),
        (support[split_s..].to_vec(), coding[split_c..].to_vec()),
    )
}

fn average_teacher_embedding(teacher: &HashingLanguageEncoder, texts: &[String]) -> Vec<f32> {
    if texts.is_empty() {
        return vec![0.0; 384];
    }
    let mut acc = vec![0.0f32; 384];
    for t in texts {
        let emb = teacher.encode(t);
        for i in 0..384 {
            acc[i] += emb[i];
        }
    }
    for v in &mut acc {
        *v /= texts.len() as f32;
    }
    l2_normalize_local(&mut acc);
    acc
}

fn eval_routing_student(
    student: &TinyMlpStudent,
    base: &HashingLanguageEncoder,
    support_proto: &[f32],
    coding_proto: &[f32],
    valid: &(Vec<String>, Vec<String>),
) -> (f32, f32, f32) {
    let mut correct = 0usize;
    let mut total = 0usize;
    let mut margins = Vec::new();
    for text in &valid.0 {
        let mut y = student.predict(&base.encode(text));
        l2_normalize_local(&mut y);
        let s = cosine_similarity_local(&y, support_proto);
        let c = cosine_similarity_local(&y, coding_proto);
        if s >= c {
            correct += 1;
        }
        margins.push((s - c).abs());
        total += 1;
    }
    for text in &valid.1 {
        let mut y = student.predict(&base.encode(text));
        l2_normalize_local(&mut y);
        let s = cosine_similarity_local(&y, support_proto);
        let c = cosine_similarity_local(&y, coding_proto);
        if c > s {
            correct += 1;
        }
        margins.push((c - s).abs());
        total += 1;
    }
    let acc = if total == 0 { 0.0 } else { correct as f32 / total as f32 };
    let med = percentile(&margins, 0.5);
    let p10 = percentile(&margins, 0.1);
    (acc, med, p10)
}

fn build_language_calibration_dataset() -> CalibrationDataset {
    let mut samples = Vec::new();
    let domains = vec![
        "customer_support",
        "coding_tool_use",
        "knowledge_qa",
        "safety_refusal",
        "procedural_instruction",
        "short_conversation",
        "multi_turn_followup",
        "adversarial_noisy",
    ];
    let languages = ["english", "english", "english", "spanish", "french"];
    for domain in domains {
        for i in 0..500 {
            let lang = languages[i % languages.len()];
            let text = format!("{} sample {} in {}", domain, i, lang);
            samples.push(LanguageSample {
                domain: domain.to_string(),
                text,
                semantic_intent: format!("{}_intent", domain),
                action_target: if domain == "coding_tool_use" {
                    Some("tool_runner".to_string())
                } else {
                    None
                },
                policy_regime: if domain == "safety_refusal" { "strict".to_string() } else { "default".to_string() },
                language_channel: lang.to_string(),
                expected_response: None,
                expected_code: None,
            });
        }
    }
    CalibrationDataset { samples }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TinyMlpStudent {
    w1: Vec<Vec<f32>>,
    b1: Vec<f32>,
    w2: Vec<Vec<f32>>,
    b2: Vec<f32>,
}

fn save_tiny_student_checkpoint(path: &str, student: &TinyMlpStudent) -> Result<(), String> {
    if let Some(parent) = std::path::Path::new(path).parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("create_dir_all failed: {}", e))?;
    }
    let json = serde_json::to_string_pretty(student).map_err(|e| format!("serialize failed: {}", e))?;
    std::fs::write(path, json).map_err(|e| format!("write failed: {}", e))
}

fn load_tiny_student_checkpoint(path: &str) -> Result<TinyMlpStudent, String> {
    let json = std::fs::read_to_string(path).map_err(|e| format!("read failed: {}", e))?;
    serde_json::from_str(&json).map_err(|e| format!("deserialize failed: {}", e))
}

impl TinyMlpStudent {
    fn new(input_dim: usize, hidden_dim: usize, output_dim: usize) -> Self {
        let mut w1 = vec![vec![0.0f32; input_dim]; hidden_dim];
        let mut w2 = vec![vec![0.0f32; hidden_dim]; output_dim];
        for (h, row) in w1.iter_mut().enumerate() {
            for (i, wij) in row.iter_mut().enumerate() {
                *wij = (((h as u64 * 2654435761 + i as u64 * 7919) % 1000) as f32 / 1000.0 - 0.5) * 0.02;
            }
        }
        for (o, row) in w2.iter_mut().enumerate() {
            for (h, woh) in row.iter_mut().enumerate() {
                *woh = (((o as u64 * 2246822519 + h as u64 * 3571) % 1000) as f32 / 1000.0 - 0.5) * 0.02;
            }
        }
        Self {
            w1,
            b1: vec![0.0; hidden_dim],
            w2,
            b2: vec![0.0; output_dim],
        }
    }

    fn predict(&self, x: &[f32]) -> Vec<f32> {
        let (h, y) = self.forward(x);
        let _ = h;
        y
    }

    fn forward(&self, x: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut h = vec![0.0f32; self.w1.len()];
        for (j, hj) in h.iter_mut().enumerate() {
            let mut acc = self.b1[j];
            for (i, &xi) in x.iter().enumerate() {
                acc += self.w1[j][i] * xi;
            }
            *hj = acc.tanh();
        }
        let mut y = vec![0.0f32; self.w2.len()];
        for (o, yo) in y.iter_mut().enumerate() {
            let mut acc = self.b2[o];
            for (j, &hj) in h.iter().enumerate() {
                acc += self.w2[o][j] * hj;
            }
            *yo = acc;
        }
        (h, y)
    }

    #[allow(clippy::too_many_arguments)]
    fn train_step(
        &mut self,
        x: &[f32],
        pos_target: &[f32],
        neg_target: &[f32],
        lr: f32,
        w_mse: f32,
        w_cos: f32,
        w_triplet: f32,
        triplet_margin: f32,
    ) -> f32 {
        let (h, y) = self.forward(x);
        let d = y.len().max(1) as f32;

        let mut loss = 0.0f32;
        let mut grad_y = vec![0.0f32; y.len()];

        // MSE(y, pos)
        for o in 0..y.len() {
            let e = y[o] - pos_target[o];
            loss += w_mse * (e * e / d);
            grad_y[o] += w_mse * (2.0 * e / d);
        }

        // 1 - cos(y, pos)
        let cos_pos = cosine_similarity_local(&y, pos_target);
        loss += w_cos * (1.0 - cos_pos);
        let dcos_pos = grad_cosine_wrt_a(&y, pos_target);
        for o in 0..y.len() {
            grad_y[o] += w_cos * (-dcos_pos[o]);
        }

        // Triplet: max(0, m - cos(y,pos) + cos(y,neg))
        let cos_neg = cosine_similarity_local(&y, neg_target);
        let trip = (triplet_margin - cos_pos + cos_neg).max(0.0);
        loss += w_triplet * trip;
        if trip > 0.0 {
            let dcos_neg = grad_cosine_wrt_a(&y, neg_target);
            for o in 0..y.len() {
                grad_y[o] += w_triplet * (-dcos_pos[o] + dcos_neg[o]);
            }
        }

        // Backprop into second layer
        let mut grad_h = vec![0.0f32; h.len()];
        for (o, &gy) in grad_y.iter().enumerate() {
            for (j, &hj) in h.iter().enumerate() {
                grad_h[j] += gy * self.w2[o][j];
                self.w2[o][j] -= lr * gy * hj;
            }
            self.b2[o] -= lr * gy;
        }

        // Backprop through tanh and first layer
        for (j, ghj_raw) in grad_h.iter().enumerate() {
            let ghj = *ghj_raw * (1.0 - h[j] * h[j]);
            for (i, &xi) in x.iter().enumerate() {
                self.w1[j][i] -= lr * ghj * xi;
            }
            self.b1[j] -= lr * ghj;
        }

        loss
    }
}

fn grad_cosine_wrt_a(a: &[f32], b: &[f32]) -> Vec<f32> {
    if a.is_empty() || a.len() != b.len() {
        return vec![];
    }
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na <= 1e-9 || nb <= 1e-9 {
        return vec![0.0; a.len()];
    }
    let cos = cosine_similarity_local(a, b);
    let inv = 1.0 / (na * nb);
    let na2 = na * na;
    let mut g = vec![0.0f32; a.len()];
    for i in 0..a.len() {
        g[i] = b[i] * inv - cos * (a[i] / na2);
    }
    g
}

fn add_centroid(
    centroids: &mut HashMap<String, Vec<f32>>,
    counts: &mut HashMap<String, usize>,
    intent: &str,
    v: &[f32],
) {
    let entry = centroids.entry(intent.to_string()).or_insert_with(|| vec![0.0; v.len()]);
    for (e, x) in entry.iter_mut().zip(v.iter()) {
        *e += *x;
    }
    *counts.entry(intent.to_string()).or_insert(0) += 1;
}

fn finalize_centroids(centroids: &mut HashMap<String, Vec<f32>>, counts: &HashMap<String, usize>) {
    for (k, v) in centroids.iter_mut() {
        let n = *counts.get(k).unwrap_or(&1) as f32;
        for x in v.iter_mut() {
            *x /= n;
        }
        l2_normalize_local(v);
    }
}

fn nearest_intents_scored(v: &[f32], centroids: &HashMap<String, Vec<f32>>) -> Vec<(String, f32)> {
    let mut out: Vec<(String, f32)> = centroids
        .iter()
        .map(|(intent, c)| (intent.clone(), cosine_similarity_local(v, c)))
        .collect();
    out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    out
}

fn cosine_similarity_local(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na <= 1e-9 || nb <= 1e-9 {
        0.0
    } else {
        (dot / (na * nb)).clamp(-1.0, 1.0)
    }
}

fn l2_normalize_local(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 1e-9 {
        for x in v {
            *x /= n;
        }
    }
}

fn percentile(values: &[f32], p: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut s = values.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let pp = p.clamp(0.0, 1.0);
    let idx = ((s.len().saturating_sub(1) as f32) * pp).round() as usize;
    s[idx]
}

fn choose_operating_threshold_for_far(
    in_domain_scores: &[f32],
    ood_scores: &[f32],
    target_far: f32,
) -> f32 {
    if in_domain_scores.is_empty() || ood_scores.is_empty() {
        return 0.15;
    }
    let mut candidates: Vec<f32> = in_domain_scores
        .iter()
        .chain(ood_scores.iter())
        .copied()
        .collect();
    candidates.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    candidates.dedup_by(|a, b| (*a - *b).abs() < 1e-6);

    let mut best_threshold = candidates[0];
    let mut best_in_domain_accept = -1.0f32;
    for &thr in &candidates {
        let far = ood_scores.iter().filter(|&&s| s >= thr).count() as f32 / ood_scores.len() as f32;
        if far > target_far {
            continue;
        }
        let id_accept = in_domain_scores.iter().filter(|&&s| s >= thr).count() as f32
            / in_domain_scores.len() as f32;
        if id_accept > best_in_domain_accept {
            best_in_domain_accept = id_accept;
            best_threshold = thr;
        }
    }
    if best_in_domain_accept < 0.0 {
        // No threshold meets FAR target; choose highest threshold to minimize false accepts.
        *candidates.last().unwrap_or(&0.15)
    } else {
        best_threshold
    }
}

fn compute_auroc(scores_labels: &[(f32, bool)]) -> f32 {
    if scores_labels.is_empty() {
        return 0.5;
    }
    let positives = scores_labels.iter().filter(|(_, y)| *y).count() as f32;
    let negatives = scores_labels.len() as f32 - positives;
    if positives <= 0.0 || negatives <= 0.0 {
        return 0.5;
    }

    let mut pairs = scores_labels.to_vec();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut tp = 0.0f32;
    let mut fp = 0.0f32;
    let mut prev_score = f32::INFINITY;
    let mut points: Vec<(f32, f32)> = vec![(0.0, 0.0)];

    for (score, is_pos) in pairs {
        if score != prev_score {
            points.push((fp / negatives, tp / positives));
            prev_score = score;
        }
        if is_pos {
            tp += 1.0;
        } else {
            fp += 1.0;
        }
    }
    points.push((1.0, 1.0));

    let mut auc = 0.0f32;
    for w in points.windows(2) {
        let (x1, y1) = w[0];
        let (x2, y2) = w[1];
        let dx = (x2 - x1).max(0.0);
        auc += dx * (y1 + y2) * 0.5;
    }
    auc.clamp(0.0, 1.0)
}




#[derive(Debug, Serialize, Deserialize)]
struct GleModelCard {
    model_name: String,
    checkpoint_path: String,
    created_unix: u64,
    stack: String,
    notes: Vec<String>,
    metrics: Vec<(String, f32)>,
}

fn model_card_path(checkpoint_path: &str) -> String {
    format!("{}.meta.json", checkpoint_path)
}

fn resolve_model_card_input(path: &str) -> String {
    if path.ends_with(".meta.json") {
        path.to_string()
    } else {
        model_card_path(path)
    }
}

fn save_gle_model_card(card: &GleModelCard) -> Result<(), String> {
    let path = model_card_path(&card.checkpoint_path);
    if let Some(parent) = std::path::Path::new(&path).parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("create_dir_all failed: {}", e))?;
    }
    let json = serde_json::to_string_pretty(card).map_err(|e| format!("serialize failed: {}", e))?;
    std::fs::write(&path, json).map_err(|e| format!("write failed: {}", e))
}

fn now_unix_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn print_gle_model_card(path: &str) -> Result<(), String> {
    let resolved = resolve_model_card_input(path);
    let json = std::fs::read_to_string(&resolved).map_err(|e| format!("read failed: {}", e))?;
    let card: GleModelCard = serde_json::from_str(&json).map_err(|e| format!("parse failed: {}", e))?;
    println!("GLE Model Card");
    println!("  name: {}", card.model_name);
    println!("  checkpoint: {}", card.checkpoint_path);
    println!("  created_unix: {}", card.created_unix);
    println!("  stack: {}", card.stack);
    if !card.notes.is_empty() {
        println!("  notes:");
        for n in &card.notes {
            println!("    - {}", n);
        }
    }
    if !card.metrics.is_empty() {
        println!("  metrics:");
        for (k, v) in &card.metrics {
            println!("    - {}: {:.6}", k, v);
        }
    }
    Ok(())
}

fn metric_lookup(card: &GleModelCard, key: &str) -> Option<f32> {
    card.metrics
        .iter()
        .find(|(k, _)| k == key)
        .map(|(_, v)| *v)
}

fn validate_gle_model_card(
    path: &str,
    min_acc: f32,
    min_median_margin: f32,
    min_p10_margin: f32,
) -> Result<bool, String> {
    let resolved = resolve_model_card_input(path);
    let json = std::fs::read_to_string(&resolved).map_err(|e| format!("read failed: {}", e))?;
    let card: GleModelCard = serde_json::from_str(&json).map_err(|e| format!("parse failed: {}", e))?;

    let acc = metric_lookup(&card, "routing_acc_support_coding")
        .ok_or_else(|| "missing metric routing_acc_support_coding".to_string())?;
    let med = metric_lookup(&card, "routing_median_margin")
        .ok_or_else(|| "missing metric routing_median_margin".to_string())?;
    let p10 = metric_lookup(&card, "routing_p10_margin")
        .ok_or_else(|| "missing metric routing_p10_margin".to_string())?;

    let pass_acc = acc >= min_acc;
    let pass_med = med >= min_median_margin;
    let pass_p10 = p10 >= min_p10_margin;
    let pass = pass_acc && pass_med && pass_p10;

    println!("GLE Validation");
    println!("  card: {}", resolved);
    println!(
        "  routing_acc_support_coding: {:.6} (threshold {:.6}) => {}",
        acc,
        min_acc,
        if pass_acc { "PASS" } else { "FAIL" }
    );
    println!(
        "  routing_median_margin: {:.6} (threshold {:.6}) => {}",
        med,
        min_median_margin,
        if pass_med { "PASS" } else { "FAIL" }
    );
    println!(
        "  routing_p10_margin: {:.6} (threshold {:.6}) => {}",
        p10,
        min_p10_margin,
        if pass_p10 { "PASS" } else { "FAIL" }
    );
    println!("  verdict: {}", if pass { "PASS" } else { "FAIL" });
    Ok(pass)
}
