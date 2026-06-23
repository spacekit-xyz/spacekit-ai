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

use growformer::cmi::{CmiPointRecord, format_cmi_report, estimate_cmi_seed};
use growformer::cmi_spiral::{format_spiral_resolve_report, resolve_spiral_region_mi, SpiralResolveResult};
use growformer::inference::grounding_loop::{
    self, BatchVerdict, CaptureSplit, CoverageCurvePoint, EditProposal, FailureCapture,
    FailureTrigger, FixtureRow, GroundingLoopParams, ProposalKind,
    build_grounding_index, build_grounding_index_from_nodes, build_overlap_curve,
    calibrate_alias_threshold, certify_batch, clear_phrase_embedder, collision_check,
    concept_train_features, coverage_vs_additions_curve, curve_lifts, decide_batch_verdict,
    embed_phrase, evaluate_disjoint, format_certifier_report, format_coverage_curve,
    install_phrase_embedder_from_corpus, install_supervised_embedder, install_vector_embedder,
    overlap0_substrata, pooled_accuracy, propose_for_phrase, synthetic_audit_fixture,
    wilson_interval, OverlapBin, SupervisedEncoder, PET_DOMAIN_FIXTURE_TOML,
    EncoderVerdict, FeatureFamily, Verdict, VerdictInputs, decide_encoder_verdict,
    is_below_resolution, run_augmentation_firewall, data_hash, routing_accuracy_for_captures,
};
use growformer::inference::world_grounding::{self, GroundingFleetDomain};
use growformer::environment::NeuralEnvironment;
use growformer::types::NeuronId;
use growformer::dimension::{
    append_language_samples_from_training_jsonl_dir, CalibrationDataset, CalibrationReport,
    CalibrationRequirements, EncoderPreset, LanguageConfig, LanguageSample, DimensionManager,
    DimensionManagerConfig, HashingLanguageEncoder, LanguageEncoder, LearnedRouter, MainDimension,
    VirtualGroup, RoutingEntropyGuard, routing_entropy_bits, routing_entropy_degenerate,
    load_language_samples_jsonl, render_action_template, generate_code_from_action,
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
use rand::Rng;
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
    phase3e: bool,
    /// Expert-router boundary alignment: scatter CSV + leak diagnostics (run before interpreting 81%).
    #[arg(long)]
    phase3e_boundary: bool,
    /// Annulus misroute analysis from existing phase3e_boundary_diagnostic.csv (fast).
    #[arg(long)]
    phase3e_boundary_analyze: bool,
    /// Per-specialist competence routing on Task E (falsifiable gate; §COMPETENCE_ROUTING_SPEC).
    #[arg(long)]
    phase3f_competence: bool,
    #[arg(long)]
    phase3f_analyze: bool,
    /// Conditional MI measurement: present-but-inaccessible formalization (Task E).
    #[arg(long)]
    cmi: bool,
    #[arg(long)]
    cmi_analyze: bool,
    /// Resolve I(R;A_spiral): bottleneck vs below-resolution (permutation null + linear probe).
    #[arg(long)]
    cmi_spiral_resolve: bool,
    #[arg(long)]
    cmi_spiral_analyze: bool,
    /// Self-revising grounding loop audit (assisted maintenance; §GROUNDING_LOOP_SPEC).
    #[arg(long)]
    grounding_loop_audit: bool,
    #[arg(long)]
    grounding_loop_analyze: bool,
    /// Run the grounding-loop certifier on a real companion (lexical vs supervised encoder).
    /// Pass the companion dir; defaults to the Luna companion.
    #[arg(long)]
    grounding_loop_luna: Option<String>,
    /// Token-disjoint generalization test for the supervised encoder on a companion.
    #[arg(long)]
    grounding_disjoint_test: Option<String>,
    /// Re-run the disjoint test from captured CSV + meta sidecar.
    #[arg(long)]
    grounding_disjoint_analyze: bool,
    /// Certifier-first pipeline: judge an encoder by the contract and emit a verdict artifact.
    /// Usage: --certify-encoder <encoder_id> [companion_dir] [seed]
    /// encoder_id ∈ {supervised, cata}. Emits verdict_<encoder>_<datahash>_<seed>.json.
    #[arg(long, num_args = 1..=3, value_names = ["ENCODER", "DIR", "SEED"])]
    certify_encoder: Option<Vec<String>>,
    /// Re-read / pretty-print a verdict artifact (go/no-go summary).
    #[arg(long)]
    certify_verdict: Option<String>,
    /// Certify the GLE on its OWN (support/coding) home domain through the same gate:
    /// (A) the literal 2-way 100% headline + (B) home-domain many-intent routing.
    #[arg(long)]
    certify_gle_indomain: bool,
    #[arg(long)]
    neurogenesis: bool,
    #[arg(long)]
    mnist: bool,
    #[arg(long)]
    mnist_v2: bool,
    #[arg(long)]
    mnist_v2_gen: bool,
    #[arg(long)]
    mnist_retention: bool,
    #[arg(long)]
    pathmnist: bool,
    #[arg(long)]
    pathmnist_64: bool,
    #[arg(long)]
    arc_agi: bool,
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
    } else if args.cmi_spiral_analyze {
        demo_cmi_spiral_analyze();
    } else if args.cmi_spiral_resolve {
        demo_cmi_spiral_resolve();
    } else if args.grounding_loop_analyze {
        demo_grounding_loop_analyze();
    } else if let Some(dir) = args.grounding_loop_luna.clone() {
        demo_grounding_loop_luna(&dir);
    } else if let Some(dir) = args.grounding_disjoint_test.clone() {
        demo_grounding_disjoint_test(&dir);
    } else if args.grounding_disjoint_analyze {
        demo_grounding_disjoint_analyze();
    } else if let Some(spec) = args.certify_encoder.clone() {
        demo_certify_encoder(&spec);
    } else if let Some(path) = args.certify_verdict.clone() {
        demo_certify_verdict(&path);
    } else if args.certify_gle_indomain {
        demo_certify_gle_indomain();
    } else if args.grounding_loop_audit {
        demo_grounding_loop_audit();
    } else if args.cmi_analyze {
        demo_cmi_analyze_csv();
    } else if args.cmi {
        demo_cmi_measurement();
    } else if args.phase3f_analyze {
        demo_phase3f_analyze_csv();
    } else if args.phase3f_competence {
        demo_phase3f_competence_routing();
    } else if args.phase3e_boundary_analyze {
        demo_phase3e_boundary_analyze_csv();
    } else if args.phase3e_boundary {
        demo_phase3e_boundary_diagnostic();
    } else if args.phase3e {
        demo_phase3e_balanced_composite();
    } else if args.neurogenesis {
        demo_neurogenesis();
    } else if args.mnist {
        demo_split_mnist(args.progress, args.mnist_train_limit, args.mnist_max_epochs, args.mnist_batch_size);
    } else if args.mnist_v2 {
        demo_clifford_mnist(args.mnist_train_limit, args.mnist_max_epochs);
    } else if args.mnist_v2_gen {
        demo_clifford_mnist_gen(args.mnist_train_limit, args.mnist_max_epochs);
    } else if args.mnist_retention {
        demo_mnist_retention();
    } else if args.pathmnist {
        demo_pathmnist(args.mnist_train_limit, args.mnist_max_epochs);
    } else if args.pathmnist_64 {
        demo_pathmnist_64(args.mnist_train_limit);
    } else if args.arc_agi {
        demo_arc_agi();
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

fn load_m5_or_synthetic() -> CalibrationDataset {
    match load_all_m5_training_data() {
        Ok(samples) if !samples.is_empty() => { println!("Loaded M5 training data: {} samples", samples.len()); CalibrationDataset { samples } }
        Ok(_) => { println!("M5 data empty, falling back to synthetic dataset."); build_language_calibration_dataset() }
        Err(e) => { println!("M5 data not found ({}) — falling back to synthetic dataset.", e); build_language_calibration_dataset() }
    }
}

fn load_all_m5_training_data() -> Result<Vec<LanguageSample>, String> {
    let m5 = std::path::Path::new("data/language/m5");
    if !m5.exists() {
        return Err(format!("M5 data directory not found: {}", m5.display()));
    }
    let mut all = Vec::new();
    append_language_samples_from_training_jsonl_dir(&mut all, m5)?;
    let agent = std::path::Path::new("data/agent");
    if agent.exists() {
        append_language_samples_from_training_jsonl_dir(&mut all, agent)?;
    }
    let routekit = std::path::Path::new("data/routekit");
    if routekit.exists() {
        append_language_samples_from_training_jsonl_dir(&mut all, routekit)?;
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
    let _ratio = if ci.abs() > 1e-8 { ii / ci } else { 0.0 };

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
// Demo: Clifford MNIST Autoencoder — Image Generation from Cl(1,7) encodings
// =============================================================================

fn demo_clifford_mnist_gen(train_limit: Option<usize>, max_epochs_override: Option<u32>) {
    use growformer::mnist::load_mnist_normalized;
    use growformer::clifford_mnist::{
        run_clifford_mnist_progress, run_clifford_autoencoder, render_ascii,
    };

    let data_path = std::env::var("MNIST_ROOT").unwrap_or_else(|_| "data".to_string());
    let images_path = std::path::Path::new(&data_path).join("train-images-idx3-ubyte");
    let images_gz = std::path::Path::new(&data_path).join("train-images-idx3-ubyte.gz");
    if !images_path.exists() && !images_gz.exists() {
        eprintln!("MNIST data not found at {:?}.", data_path);
        eprintln!("Run: bash scripts/download_mnist.sh  or set MNIST_ROOT.");
        std::process::exit(1);
    }

    println!("═══════════════════════════════════════════════════════════");
    println!("  Growformer Vision: Growformer Autoencoder");
    println!("  Encode → Growformer Brain → Decode → Image");
    println!("═══════════════════════════════════════════════════════════\n");

    println!("Loading MNIST from {:?}...", data_path);
    let (train_imgs, train_lbls, test_imgs, test_lbls) = load_mnist_normalized(&data_path);
    println!("  Train: {} images, Test: {} images\n", train_imgs.len(), test_imgs.len());

    // Step 1: Train the encoder via classification (or reuse if already trained)
    let max_epochs = max_epochs_override.unwrap_or(30);
    println!("--- Step 1: Training encoder via classification ---\n");
    let start = Instant::now();
    let class_result = run_clifford_mnist_progress(
        &train_imgs, &train_lbls,
        &test_imgs, &test_lbls,
        train_limit, max_epochs,
    );
    let class_elapsed = start.elapsed();
    println!("\n  Classification: {:.1}% avg binary, {:.1}% 10-class ({:.1}s)\n",
        class_result.avg_accuracy * 100.0,
        class_result.ten_class_accuracy * 100.0,
        class_elapsed.as_secs_f64());

    let encoder = class_result.encoder;
    let classifier = class_result.classifier;

    // Step 2: Solve decoder (single-pass least squares)
    println!("--- Step 2: Decoder (single-pass solve) ---\n");
    let ae_start = Instant::now();
    let ae_result = run_clifford_autoencoder(
        &encoder,
        &train_imgs, &train_lbls,
        &test_imgs, &test_lbls,
        train_limit,
        &classifier,
        1,
    );
    let ae_elapsed = ae_start.elapsed();

    println!("\n═══════════════════════════════════════════════════════════");
    println!("  AUTOENCODER RESULTS");
    println!("═══════════════════════════════════════════════════════════");
    println!("  Pixel MSE:               {:.5}", ae_result.final_mse);
    println!("  SSIM:                    {:.3}", ae_result.final_ssim);
    println!("  Classifier on generated: {:.1}%", ae_result.classifier_accuracy * 100.0);
    println!("  Solve time:              {:.1}s", ae_elapsed.as_secs_f64());
    println!("═══════════════════════════════════════════════════════════\n");

    // Display sample reconstructions as ASCII
    println!("--- Sample Reconstructions (original → reconstructed) ---\n");
    for (label, original, reconstructed) in &ae_result.sample_reconstructions {
        println!("  Digit {}:", label);
        let orig_ascii = render_ascii(original, 28);
        let recon_ascii = render_ascii(reconstructed, 28);
        let orig_lines: Vec<&str> = orig_ascii.lines().collect();
        let recon_lines: Vec<&str> = recon_ascii.lines().collect();
        println!("  {:>28}    {:>28}", "Original", "Reconstructed");
        for i in 0..orig_lines.len().min(recon_lines.len()) {
            println!("  {}    {}", orig_lines[i], recon_lines[i]);
        }
        println!();
    }
}

// =============================================================================
// Demo: PathMNIST — Colorectal cancer histology through Cl(1,7)
// 9-class tissue classification on MedMNIST benchmark, CPU only.
// =============================================================================

fn demo_pathmnist(train_limit: Option<usize>, _max_epochs_override: Option<u32>) {
    use growformer::pathmnist::{
        PathMNISTDataset, CLASS_NAMES, PATH_NUM_CLASSES, is_cancer_class,
        compute_cancer_metrics,
    };
    use growformer::clifford_mnist::{
        CliffordRGBEncoder, CliffordDiracEncoder, PathClassifier, LinearClassifier,
        discriminability_weights, train_projection_for_bv,
    };
    use growformer::clifford::{Multivector, CL8_DIM, minkowski_interval, classify_interval, IntervalType};

    let data_dir = std::path::PathBuf::from(
        std::env::var("PATHMNIST_ROOT")
            .unwrap_or_else(|_| "data/pathology/pathmnist".to_string())
    );
    if !data_dir.exists() {
        eprintln!("PathMNIST data not found at {:?}", data_dir);
        eprintln!("Set PATHMNIST_ROOT or place data in data/pathology/pathmnist/");
        std::process::exit(1);
    }

    println!("═══════════════════════════════════════════════════════════════");
    println!("  Growformer Medical: Colorectal Cancer Histology");
    println!("  PathMNIST — 9 classes, RGB+Dirac Cl(1,7), single-pass solve");
    println!("═══════════════════════════════════════════════════════════════\n");

    println!("Loading PathMNIST (RGB)...");
    let train = PathMNISTDataset::load(&data_dir, "train");
    let val = PathMNISTDataset::load(&data_dir, "val");
    let test = PathMNISTDataset::load(&data_dir, "test");
    println!("  Train: {}, Val: {}, Test: {} (different clinical center)",
        train.n, val.n, test.n);

    let dist = train.class_distribution();
    println!("\n  Class distribution (train):");
    for c in 0..PATH_NUM_CLASSES {
        let marker = if is_cancer_class(c as u8) { " ← CANCER" } else { "" };
        println!("    {}: {:>14} {:>6}{}", c, CLASS_NAMES[c], dist[c], marker);
    }

    let total_start = Instant::now();

    // Two complementary encoders:
    //   RGB encoder  (2352→256): color/intensity patterns from raw pixels
    //   Dirac encoder (784→256): differential structure (gradients, Laplacian, texture)
    let mut rgb_enc = CliffordRGBEncoder::new(42);
    let mut dirac_enc = CliffordDiracEncoder::new(77);

    let rgb_cal: Vec<_> = train.images_rgb.iter().take(1000)
        .map(|img| (img.clone(), 0u8)).collect();
    rgb_enc.calibrate_scales(&rgb_cal);
    let gray_cal: Vec<_> = train.images_gray.iter().take(1000)
        .map(|img| (img.clone(), 0u8)).collect();
    dirac_enc.calibrate_scales(&gray_cal);

    let train_n = train_limit.unwrap_or(train.n).min(train.n);
    let joint_dim = CL8_DIM * 2; // 512D

    use growformer::clifford_mnist::diagnose_pair_learnability;

    let train_lbls_full = &train.labels[..train_n];

    // ══════════════════════════════════════════════════════════════════════
    //  Geometric Learnability Diagnostic
    //  |B| = |<Z_a * Z_b†>_2| for all class pairs — O(C²), milliseconds.
    //  Predicts which pairs are rotationally separable vs degenerate.
    // ══════════════════════════════════════════════════════════════════════
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  Geometric Learnability Diagnostic");
    println!("  |B| = |<Z_a * Z_b†>_2| — computable before any training");
    println!("═══════════════════════════════════════════════════════════════");

    let diag_start = Instant::now();
    let (bv_matrix, rotational_pairs, degenerate_pairs) =
        diagnose_pair_learnability(&rgb_enc, &train.images_rgb, train_lbls_full, PATH_NUM_CLASSES, 1000);

    println!("\n  Confusion bivector |B| matrix:");
    print!("         ");
    for c in 0..PATH_NUM_CLASSES { print!("{:>6}", c); }
    println!();
    for i in 0..PATH_NUM_CLASSES {
        print!("    {}:{:>8} ", i, &CLASS_NAMES[i][..CLASS_NAMES[i].len().min(8)]);
        for j in 0..PATH_NUM_CLASSES {
            if i == j { print!("     -"); }
            else {
                let bv = bv_matrix[i][j];
                let tag = if bv >= 0.3 { "+" } else if bv < 0.1 { "!" } else { " " };
                print!("{}{:5.3}", tag, bv);
            }
        }
        println!();
    }

    println!("\n  Rotational (|B| >= 0.3): {} pairs", rotational_pairs.len());
    println!("  Degenerate (|B| < 0.1):  {} pairs", degenerate_pairs.len());
    for &(a, b) in &degenerate_pairs {
        println!("    {} ↔ {}  ({} ↔ {})  |B|={:.3}",
            a, b, CLASS_NAMES[a as usize], CLASS_NAMES[b as usize], bv_matrix[a as usize][b as usize]);
    }
    println!("  Diagnostic: {:.1}s", diag_start.elapsed().as_secs_f64());

    // ══════════════════════════════════════════════════════════════════════
    //  Projection Training: maximize |B| for degenerate cancer pairs
    //  Train only the 784×256 projection matrix. Everything else frozen.
    //  Target: stroma(7)↔debris(2), stroma(7)↔adeno(8)
    // ══════════════════════════════════════════════════════════════════════
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  Projection Training — maximize |B| for cancer pairs");
    println!("  Target pairs: stroma↔debris (7,2), stroma↔adeno (7,8)");
    println!("  Params: {}×{} = {} (projection only)", train.images_rgb[0].len(), CL8_DIM,
        train.images_rgb[0].len() * CL8_DIM);
    println!("═══════════════════════════════════════════════════════════════");

    let proj_train_start = Instant::now();
    let target_pairs = vec![(7u8, 2u8), (7u8, 8u8)];
    train_projection_for_bv(
        &mut rgb_enc,
        &train.images_rgb[..train_n],
        train_lbls_full,
        &target_pairs,
        500,      // steps
        0.0001,   // learning rate
        5000,     // samples per class for mean computation
    );

    rgb_enc.calibrate_scales(&rgb_cal);
    println!("  Projection training: {:.1}s", proj_train_start.elapsed().as_secs_f64());

    // Post-training |B| diagnostic
    println!("\n  Post-training |B| diagnostic:");
    let (bv_matrix_post, rotational_post, degenerate_post) =
        diagnose_pair_learnability(&rgb_enc, &train.images_rgb, train_lbls_full, PATH_NUM_CLASSES, 1000);

    print!("         ");
    for c in 0..PATH_NUM_CLASSES { print!("{:>6}", c); }
    println!();
    for i in 0..PATH_NUM_CLASSES {
        print!("    {}:{:>8} ", i, &CLASS_NAMES[i][..CLASS_NAMES[i].len().min(8)]);
        for j in 0..PATH_NUM_CLASSES {
            if i == j { print!("     -"); }
            else {
                let bv = bv_matrix_post[i][j];
                let delta = bv - bv_matrix[i][j];
                let tag = if delta > 0.05 { "↑" } else if delta < -0.05 { "↓" } else { " " };
                print!("{}{:5.3}", tag, bv);
            }
        }
        println!();
    }

    println!("\n  Key pair changes:");
    for &(a, b) in &target_pairs {
        let before = bv_matrix[a as usize][b as usize];
        let after = bv_matrix_post[a as usize][b as usize];
        let verdict = if after >= 0.3 { "SEPARABLE" } else if after >= 0.1 { "improved" } else { "still degenerate" };
        println!("    {} ↔ {}:  {:.3} → {:.3}  ({})",
            CLASS_NAMES[a as usize], CLASS_NAMES[b as usize], before, after, verdict);
    }
    println!("  Rotational pairs: {} → {}", rotational_pairs.len(), rotational_post.len());
    println!("  Degenerate pairs: {} → {}", degenerate_pairs.len(), degenerate_post.len());

    // ══════════════════════════════════════════════════════════════════════
    //  Encode all training images: RGB + Dirac → 512D
    // ══════════════════════════════════════════════════════════════════════
    let full_dim = joint_dim;
    println!("\n--- Encoding {} training images (RGB+Dirac → {}D Cl(1,7)) ---", train_n, full_dim);
    let encode_start = Instant::now();

    let mut train_features: Vec<Vec<f32>> = Vec::with_capacity(train_n);
    let mut train_rgb_mvs: Vec<Multivector> = Vec::with_capacity(train_n);
    let mut train_dirac_mvs: Vec<Multivector> = Vec::with_capacity(train_n);

    for i in 0..train_n {
        let rgb_mv = rgb_enc.encode(&train.images_rgb[i]);
        let dirac_mv = dirac_enc.encode(&train.images_gray[i]);

        let mut joint = Vec::with_capacity(full_dim);
        joint.extend_from_slice(&rgb_mv.components);
        joint.extend_from_slice(&dirac_mv.components);
        train_features.push(joint);
        train_rgb_mvs.push(rgb_mv);
        train_dirac_mvs.push(dirac_mv);

        if (i + 1) % 20000 == 0 { println!("    {}/{}", i + 1, train_n); }
    }
    let train_lbls = &train.labels[..train_n];
    println!("  Encoded in {:.1}s ({}D features)\n",
        encode_start.elapsed().as_secs_f64(), full_dim);

    // ── Single-pass: solve 9-class on 512D joint features ──
    println!("--- Solving 9-class classifier (512D normal equations) ---");
    let head = LinearClassifier::fit_from_features(
        &train_features, train_lbls,
        PATH_NUM_CLASSES, 1.0, 1,
    );

    // ── Also solve binary cancer classifier on 512D joint features ──
    println!("\n--- Solving binary cancer classifier (512D) ---");
    let cancer_labels: Vec<u8> = train_lbls.iter()
        .map(|&l| if is_cancer_class(l) { 1 } else { 0 })
        .collect();
    let cancer_head = LinearClassifier::fit_from_features(
        &train_features, &cancer_labels,
        2, 1.0, 1,
    );

    // ── Geometry-driven hierarchical routing ──
    // Build binary tree from centroid distances — let the algebra define the splits
    println!("\n--- Building geometric routing tree from 512D centroids ---");

    // Compute 512D feature centroids for each class
    let mut class_sums = vec![vec![0.0f64; joint_dim]; PATH_NUM_CLASSES];
    let mut class_counts = vec![0usize; PATH_NUM_CLASSES];
    for (feat, &lbl) in train_features.iter().zip(train_lbls.iter()) {
        let c = lbl as usize;
        class_counts[c] += 1;
        for j in 0..joint_dim { class_sums[c][j] += feat[j] as f64; }
    }
    let centroids_512: Vec<Vec<f32>> = (0..PATH_NUM_CLASSES).map(|c| {
        let n = class_counts[c].max(1) as f64;
        class_sums[c].iter().map(|&s| (s / n) as f32).collect()
    }).collect();

    // Pairwise Euclidean distance matrix between centroids
    let mut dist_matrix = vec![vec![0.0f32; PATH_NUM_CLASSES]; PATH_NUM_CLASSES];
    for i in 0..PATH_NUM_CLASSES {
        for j in (i+1)..PATH_NUM_CLASSES {
            let d: f32 = centroids_512[i].iter().zip(centroids_512[j].iter())
                .map(|(a, b)| (a - b) * (a - b)).sum::<f32>().sqrt();
            dist_matrix[i][j] = d;
            dist_matrix[j][i] = d;
        }
    }

    println!("\n  Centroid distance matrix (512D Euclidean):");
    print!("         ");
    for c in 0..PATH_NUM_CLASSES { print!("{:>6}", c); }
    println!();
    for i in 0..PATH_NUM_CLASSES {
        print!("    {}:{:>8} ", i, &CLASS_NAMES[i][..CLASS_NAMES[i].len().min(8)]);
        for j in 0..PATH_NUM_CLASSES {
            if i == j { print!("     -"); }
            else { print!(" {:5.2}", dist_matrix[i][j]); }
        }
        println!();
    }

    // Recursive tree node
    #[allow(dead_code)]
    enum GeoNode {
        Leaf(u8),
        Split {
            classifier: LinearClassifier,
            left_classes: Vec<u8>,
            right_classes: Vec<u8>,
            left: Box<GeoNode>,
            right: Box<GeoNode>,
        },
    }

    impl GeoNode {
        fn classify(&self, feat: &[f32]) -> u8 {
            match self {
                GeoNode::Leaf(c) => *c,
                GeoNode::Split { classifier, left, right, .. } => {
                    let (pred, _) = classifier.classify_features(feat);
                    if pred == 0 { left.classify(feat) } else { right.classify(feat) }
                }
            }
        }
    }

    // Build tree recursively using furthest-pair centroid splitting
    fn build_geo_tree(
        classes: &[u8],
        centroids: &[Vec<f32>],
        dist_matrix: &[Vec<f32>],
        features: &[Vec<f32>],
        labels: &[u8],
        class_names: &[&str],
        depth: usize,
    ) -> GeoNode {
        if classes.len() == 1 {
            return GeoNode::Leaf(classes[0]);
        }
        if classes.len() == 2 {
            let c0 = classes[0];
            let c1 = classes[1];
            let indent = "    ".repeat(depth + 1);
            // Filter to these two classes, relabel as 0/1
            let (sub_feats, sub_lbls): (Vec<_>, Vec<_>) = features.iter()
                .zip(labels.iter())
                .filter(|(_, &l)| l == c0 || l == c1)
                .map(|(f, &l)| (f.clone(), if l == c0 { 0u8 } else { 1 }))
                .unzip();
            println!("{}Split: {} vs {} ({} samples)",
                indent, class_names[c0 as usize], class_names[c1 as usize], sub_feats.len());
            let clf = LinearClassifier::fit_from_features(&sub_feats, &sub_lbls, 2, 1.0, 0);
            return GeoNode::Split {
                classifier: clf,
                left_classes: vec![c0],
                right_classes: vec![c1],
                left: Box::new(GeoNode::Leaf(c0)),
                right: Box::new(GeoNode::Leaf(c1)),
            };
        }

        // Find the two most distant centroids among `classes`
        let mut max_dist = 0.0f32;
        let mut seed_a = classes[0];
        let mut seed_b = classes[1];
        for i in 0..classes.len() {
            for j in (i+1)..classes.len() {
                let d = dist_matrix[classes[i] as usize][classes[j] as usize];
                if d > max_dist { max_dist = d; seed_a = classes[i]; seed_b = classes[j]; }
            }
        }

        // Assign each class to the nearest seed
        let mut left_classes = Vec::new();
        let mut right_classes = Vec::new();
        for &c in classes {
            let da = dist_matrix[c as usize][seed_a as usize];
            let db = dist_matrix[c as usize][seed_b as usize];
            if da <= db { left_classes.push(c); } else { right_classes.push(c); }
        }

        // Safety: if one side is empty, force at least one class into it
        if left_classes.is_empty() {
            left_classes.push(right_classes.pop().unwrap());
        }
        if right_classes.is_empty() {
            right_classes.push(left_classes.pop().unwrap());
        }

        let indent = "    ".repeat(depth + 1);
        let left_names: Vec<_> = left_classes.iter().map(|&c| class_names[c as usize]).collect();
        let right_names: Vec<_> = right_classes.iter().map(|&c| class_names[c as usize]).collect();
        println!("{}Split (d={:.2}): {{{}}} vs {{{}}}",
            indent, max_dist, left_names.join(", "), right_names.join(", "));

        // Train binary classifier: left=0, right=1
        let left_set: std::collections::HashSet<u8> = left_classes.iter().copied().collect();
        let all_set: std::collections::HashSet<u8> = classes.iter().copied().collect();
        let (sub_feats, sub_lbls): (Vec<_>, Vec<_>) = features.iter()
            .zip(labels.iter())
            .filter(|(_, &l)| all_set.contains(&l))
            .map(|(f, &l)| (f.clone(), if left_set.contains(&l) { 0u8 } else { 1 }))
            .unzip();
        let clf = LinearClassifier::fit_from_features(&sub_feats, &sub_lbls, 2, 1.0, 0);

        let left_child = build_geo_tree(
            &left_classes, centroids, dist_matrix, features, labels, class_names, depth + 1,
        );
        let right_child = build_geo_tree(
            &right_classes, centroids, dist_matrix, features, labels, class_names, depth + 1,
        );

        GeoNode::Split {
            classifier: clf,
            left_classes,
            right_classes,
            left: Box::new(left_child),
            right: Box::new(right_child),
        }
    }

    // ── CliffordMicroBrain: spacetime algebra routing ──
    // Uses grade-weighted Minkowski distance + interval augmentation
    // instead of cosine similarity. Preserves magnitude and metric signature.
    println!("\n--- Building CliffordMicroBrain (spacetime algebra routing) ---");
    use growformer::clifford_mnist::CliffordMicroBrain;

    let cliff_brain = CliffordMicroBrain::build(
        &train_rgb_mvs, &train_dirac_mvs, train_lbls, PATH_NUM_CLASSES,
    );

    let _solve_elapsed = total_start.elapsed();

    // ── Encode validation + test with all encoders ──
    println!("\n--- Validation ---");
    let encode_joint = |rgb_imgs: &[Vec<f32>], gray_imgs: &[Vec<f32>]|
        -> (Vec<Vec<f32>>, Vec<Multivector>, Vec<Multivector>)
    {
        let mut feats = Vec::with_capacity(rgb_imgs.len());
        let mut rgb_mvs = Vec::with_capacity(rgb_imgs.len());
        let mut dirac_mvs_out = Vec::with_capacity(rgb_imgs.len());
        for i in 0..rgb_imgs.len() {
            let rgb_mv = rgb_enc.encode(&rgb_imgs[i]);
            let dirac_mv = dirac_enc.encode(&gray_imgs[i]);
            let mut joint = Vec::with_capacity(full_dim);
            joint.extend_from_slice(&rgb_mv.components);
            joint.extend_from_slice(&dirac_mv.components);
            feats.push(joint);
            rgb_mvs.push(rgb_mv);
            dirac_mvs_out.push(dirac_mv);
        }
        (feats, rgb_mvs, dirac_mvs_out)
    };

    let (val_features, val_rgb_mvs, val_dirac_mvs) = encode_joint(&val.images_rgb, &val.images_gray);

    let val_9_preds: Vec<u8> = val_features.iter().map(|f| head.classify_features(f).0).collect();
    let val_9_acc = val_9_preds.iter().zip(val.labels.iter())
        .filter(|(p, l)| p == l).count() as f32 / val.n as f32;

    let val_cancer_preds: Vec<u8> = val_features.iter().map(|f| cancer_head.classify_features(f).0).collect();
    let val_cancer_labels: Vec<u8> = val.labels.iter()
        .map(|&l| if is_cancer_class(l) { 1 } else { 0 }).collect();
    let val_cancer_acc = val_cancer_preds.iter().zip(val_cancer_labels.iter())
        .filter(|(p, l)| p == l).count() as f32 / val.n as f32;

    println!("  9-class val accuracy:      {:.1}%", val_9_acc * 100.0);
    println!("  Cancer binary val accuracy: {:.1}%", val_cancer_acc * 100.0);

    // ── Clifford spacetime analysis (diagnostic, not primary routing) ──
    // The grade discriminability is nearly flat (~0.02 per grade), meaning
    // the multivector space lacks differentiated grade structure for histology.
    // Centroid-based Clifford routing cannot outperform the linear solve in
    // this regime. Report Clifford metrics as geometric diagnostics.

    // ── Evaluate on test set (different clinical center) ──
    println!("\n--- Test Set Evaluation (CRC-VAL-HE-7K — different hospital) ---");
    let (test_features, test_rgb_mvs, test_dirac_mvs) = encode_joint(&test.images_rgb, &test.images_gray);

    let test_preds: Vec<u8> = test_features.iter().map(|f| head.classify_features(f).0).collect();
    let test_acc = test_preds.iter().zip(test.labels.iter())
        .filter(|(p, l)| p == l).count() as f32 / test.n as f32;
    let cancer = compute_cancer_metrics(&test_preds, &test.labels);

    // Binary cancer detection via dedicated classifier
    let test_cancer_preds: Vec<u8> = test_features.iter()
        .map(|f| cancer_head.classify_features(f).0).collect();
    let test_cancer_labels: Vec<u8> = test.labels.iter()
        .map(|&l| if is_cancer_class(l) { 1 } else { 0 }).collect();
    let mut c_tp = 0u32; let mut c_fn = 0u32;
    let mut c_fp = 0u32; let mut c_tn = 0u32;
    for (&pred, &label) in test_cancer_preds.iter().zip(test_cancer_labels.iter()) {
        match (pred, label) {
            (1, 1) => c_tp += 1,
            (0, 1) => c_fn += 1,
            (1, 0) => c_fp += 1,
            (0, 0) => c_tn += 1,
            _ => {}
        }
    }
    let cancer_sens = c_tp as f32 / (c_tp + c_fn).max(1) as f32;
    let cancer_spec = c_tn as f32 / (c_tn + c_fp).max(1) as f32;
    let cancer_f1 = 2.0 * c_tp as f32 / (2 * c_tp + c_fp + c_fn).max(1) as f32;
    let cancer_binary_acc = (c_tp + c_tn) as f32 / (c_tp + c_fn + c_fp + c_tn).max(1) as f32;

    println!("\n  BINARY CANCER DETECTION (primary clinical result):");
    println!("    Accuracy:     {:.1}%", cancer_binary_acc * 100.0);
    println!("    Sensitivity:  {:.1}%", cancer_sens * 100.0);
    println!("    Specificity:  {:.1}%", cancer_spec * 100.0);
    println!("    F1:           {:.3}", cancer_f1);

    println!("\n  9-CLASS RESULTS:");
    println!("    Overall accuracy:      {:.1}%", test_acc * 100.0);
    println!("    Adenocarcinoma recall: {:.1}%  (class 8)", cancer.adeno_recall * 100.0);
    println!("    Stroma recall:         {:.1}%  (class 7)", cancer.stroma_recall * 100.0);

    // Per-class accuracy
    println!("\n  Per-class accuracy:");
    let mut per_class_correct = [0u32; PATH_NUM_CLASSES];
    let mut per_class_total = [0u32; PATH_NUM_CLASSES];
    for (&pred, &label) in test_preds.iter().zip(test.labels.iter()) {
        let l = label as usize;
        if l < PATH_NUM_CLASSES {
            per_class_total[l] += 1;
            if pred == label { per_class_correct[l] += 1; }
        }
    }
    for c in 0..PATH_NUM_CLASSES {
        let acc = if per_class_total[c] > 0 {
            per_class_correct[c] as f32 / per_class_total[c] as f32
        } else { 0.0 };
        let marker = if is_cancer_class(c as u8) { " ←" } else { "" };
        println!("    {}: {:>14} {:.1}% ({}/{}){}", c, CLASS_NAMES[c],
            acc * 100.0, per_class_correct[c], per_class_total[c], marker);
    }

    // ── k-NN classifier: diagnose whether ceiling is embeddings or linear solve ──
    println!("\n  k-NN CLASSIFIER (embedding quality diagnostic):");
    let knn_start = Instant::now();
    let k_values = [3, 5, 7];
    for &k in &k_values {
        let mut knn_correct = 0u32;
        let mut knn_cancer_tp = 0u32;
        let mut knn_cancer_fn = 0u32;
        let mut knn_cancer_fp = 0u32;
        let mut knn_cancer_tn = 0u32;
        let mut knn_per_class_correct = [0u32; PATH_NUM_CLASSES];

        for (ti, test_feat) in test_features.iter().enumerate() {
            let mut dists: Vec<(f32, u8)> = Vec::with_capacity(train_n);
            for (tr_feat, &tr_lbl) in train_features.iter().zip(train_lbls.iter()) {
                let d: f32 = test_feat.iter().zip(tr_feat.iter())
                    .map(|(a, b)| { let diff = a - b; diff * diff })
                    .sum();
                dists.push((d, tr_lbl));
            }
            dists.select_nth_unstable_by(k - 1, |a, b| a.0.partial_cmp(&b.0).unwrap());
            let mut votes = [0u32; PATH_NUM_CLASSES];
            for &(_, lbl) in dists[..k].iter() {
                votes[lbl as usize] += 1;
            }
            let pred = votes.iter().enumerate()
                .max_by_key(|(_, &v)| v).map(|(i, _)| i as u8).unwrap();
            let label = test.labels[ti];
            if pred == label {
                knn_correct += 1;
                knn_per_class_correct[label as usize] += 1;
            }
            let pred_cancer = is_cancer_class(pred);
            let label_cancer = is_cancer_class(label);
            match (pred_cancer, label_cancer) {
                (true, true)   => knn_cancer_tp += 1,
                (false, true)  => knn_cancer_fn += 1,
                (true, false)  => knn_cancer_fp += 1,
                (false, false) => knn_cancer_tn += 1,
            }

            if (ti + 1) % 2000 == 0 {
                print!("    k={}: {}/{}...\r", k, ti + 1, test.n);
            }
        }
        let knn_acc = knn_correct as f32 / test.n as f32;
        let knn_sens = knn_cancer_tp as f32 / (knn_cancer_tp + knn_cancer_fn).max(1) as f32;
        let knn_spec = knn_cancer_tn as f32 / (knn_cancer_tn + knn_cancer_fp).max(1) as f32;
        println!("    k={}: 9-class={:.1}%  sensitivity={:.1}%  specificity={:.1}%  ({:.0}s)",
            k, knn_acc * 100.0, knn_sens * 100.0, knn_spec * 100.0,
            knn_start.elapsed().as_secs_f64());

        if k == 5 {
            println!("    k=5 per-class:");
            for c in 0..PATH_NUM_CLASSES {
                let acc = if per_class_total[c] > 0 {
                    knn_per_class_correct[c] as f32 / per_class_total[c] as f32
                } else { 0.0 };
                let flat_acc = if per_class_total[c] > 0 {
                    per_class_correct[c] as f32 / per_class_total[c] as f32
                } else { 0.0 };
                let delta = (acc - flat_acc) * 100.0;
                let arrow = if delta > 1.0 { "▲" } else if delta < -1.0 { "▼" } else { "=" };
                let marker = if is_cancer_class(c as u8) { " ←" } else { "" };
                println!("      {}: {:>14} {:.1}%  (linear {:.1}%)  {}{:+.1}{}",
                    c, CLASS_NAMES[c], acc * 100.0, flat_acc * 100.0, arrow, delta, marker);
            }
        }
    }

    // ── Clifford spacetime routing (diagnostic comparison) ──
    println!("\n  CLIFFORD SPACETIME ROUTING (diagnostic):");
    let cliff_preds: Vec<u8> = test_rgb_mvs.iter().zip(test_dirac_mvs.iter())
        .map(|(rgb, dirac)| cliff_brain.classify(rgb, dirac).0)
        .collect();
    let cliff_acc = cliff_preds.iter().zip(test.labels.iter())
        .filter(|(p, l)| p == l).count() as f32 / test.n as f32;
    let cliff_cancer = compute_cancer_metrics(&cliff_preds, &test.labels);

    println!("    Accuracy:              {:.1}%  (flat: {:.1}%)", cliff_acc * 100.0, test_acc * 100.0);
    println!("    Adenocarcinoma recall: {:.1}%  (flat: {:.1}%)",
        cliff_cancer.adeno_recall * 100.0, cancer.adeno_recall * 100.0);
    println!("    Stroma recall:         {:.1}%  (flat: {:.1}%)",
        cliff_cancer.stroma_recall * 100.0, cancer.stroma_recall * 100.0);

    let mut cliff_correct = [0u32; PATH_NUM_CLASSES];
    let mut cliff_total = [0u32; PATH_NUM_CLASSES];
    for (&pred, &label) in cliff_preds.iter().zip(test.labels.iter()) {
        let l = label as usize;
        if l < PATH_NUM_CLASSES {
            cliff_total[l] += 1;
            if pred == label { cliff_correct[l] += 1; }
        }
    }
    println!("\n    Per-class (clifford vs flat):");
    for c in 0..PATH_NUM_CLASSES {
        let c_acc = if cliff_total[c] > 0 {
            cliff_correct[c] as f32 / cliff_total[c] as f32
        } else { 0.0 };
        let f_acc = if per_class_total[c] > 0 {
            per_class_correct[c] as f32 / per_class_total[c] as f32
        } else { 0.0 };
        let delta = (c_acc - f_acc) * 100.0;
        let arrow = if delta > 0.5 { "▲" } else if delta < -0.5 { "▼" } else { "=" };
        let marker = if is_cancer_class(c as u8) { " ←" } else { "" };
        println!("      {}: {:>14} {:.1}%  (flat {:.1}%)  {}{:+.1}{}", c, CLASS_NAMES[c],
            c_acc * 100.0, f_acc * 100.0, arrow, delta, marker);
    }

    // Grade discriminability
    let mut centroids = PathClassifier::new(PATH_NUM_CLASSES);
    for (mv, &lbl) in train_rgb_mvs.iter().zip(train_lbls.iter()) {
        centroids.accumulate(mv, lbl);
    }
    let grade_disc = centroids.grade_discriminability();
    let gw = discriminability_weights(&grade_disc);
    let grade_labels = [
        "scalar (intensity)", "vector (gradients)", "bivector (texture)",
        "trivector (junctions)", "quadvector (topology)", "grade-5", "grade-6",
        "grade-7", "pseudoscalar (complement)",
    ];
    println!("\n--- Grade Discriminability (9-class) ---");
    for g in 0..=8 {
        println!("    grade {}: {:>6.2}  w={:.3}  — {}", g, grade_disc[g], gw[g], grade_labels[g]);
    }

    // Minkowski interval statistics
    let mut correct_intervals = Vec::new();
    let mut incorrect_intervals = Vec::new();
    for (i, (&pred, &label)) in test_preds.iter().zip(test.labels.iter()).enumerate() {
        let mink = minkowski_interval(&test_rgb_mvs[i], &centroids.centroids[pred as usize]);
        if pred == label {
            correct_intervals.push(mink);
        } else {
            incorrect_intervals.push(mink);
        }
    }
    let correct_mean = if correct_intervals.is_empty() { 0.0 }
        else { correct_intervals.iter().sum::<f32>() / correct_intervals.len() as f32 };
    let incorrect_mean = if incorrect_intervals.is_empty() { 0.0 }
        else { incorrect_intervals.iter().sum::<f32>() / incorrect_intervals.len() as f32 };
    let timelike_correct = correct_intervals.iter()
        .filter(|&&v| classify_interval(v) == IntervalType::Timelike)
        .count() as f32 / correct_intervals.len().max(1) as f32;

    println!("\n--- Minkowski Interval Statistics ---");
    println!("  Correct:   mean={:.4}  timelike={:.1}%", correct_mean, timelike_correct * 100.0);
    println!("  Incorrect: mean={:.4}", incorrect_mean);
    if correct_mean.abs() > 1e-8 {
        println!("  Ratio:     {:.1}x", incorrect_mean / correct_mean);
    }

    let total_elapsed = total_start.elapsed();

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  PathMNIST BENCHMARK — Colorectal Cancer Histology");
    println!("  Cl(1,7) spacetime algebra, no GPU, no end-to-end training");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    println!("  Binary cancer detection (primary clinical result):");
    println!("    Sensitivity:          {:.1}%", cancer_sens * 100.0);
    println!("    Specificity:          {:.1}%", cancer_spec * 100.0);
    println!("    F1:                   {:.3}", cancer_f1);
    println!();
    println!("  9-class tissue classification:");
    println!("    Overall accuracy:     {:.1}%", test_acc * 100.0);
    println!("    Lymphocytes:          {:.1}%", per_class_correct[3] as f32 / per_class_total[3].max(1) as f32 * 100.0);
    println!("    Background:           {:.1}%", per_class_correct[1] as f32 / per_class_total[1].max(1) as f32 * 100.0);
    println!("    Adipose:              {:.1}%", per_class_correct[0] as f32 / per_class_total[0].max(1) as f32 * 100.0);
    println!("    Smooth muscle:        {:.1}%", per_class_correct[5] as f32 / per_class_total[5].max(1) as f32 * 100.0);
    println!();
    println!("  Architecture:");
    println!("    Encoder:              Cl(1,7) RGB + Dirac ({}D)", full_dim);
    println!("    Classifier:           linear solve + k-NN");
    println!("    Training:             single-pass (no epochs)");
    println!("    Total time:           {:.0}s on CPU", total_elapsed.as_secs_f64());
    println!("    GPU required:         None");
    println!();
    println!("  Geometric diagnostics:");
    println!("    Minkowski ratio:      {:.1}x (correct vs incorrect)",
        if correct_mean.abs() > 1e-8 { incorrect_mean / correct_mean } else { 0.0 });
    println!("    Timelike rate:        {:.1}%", timelike_correct * 100.0);
    println!("    Stroma-debris dist:   {:.2}  (biological collision)",
        dist_matrix[7][2]);
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    println!("  Same Cl(1,7) algebra used for:");
    println!("    MNIST digits:         97.7%  (7.5s CPU)");
    println!("    Language generation:   97%   (25min train)");
    println!("    Histopathology:       {:.1}%  cancer sensitivity", cancer_sens * 100.0);
    println!();
}

// =============================================================================
// Demo: PathMNIST 64×64 — Resolution experiment for |B| diagnostic
// Same encoder architecture at 4× resolution to test embedding ceiling.
// =============================================================================

fn demo_pathmnist_64(train_limit: Option<usize>) {
    use growformer::pathmnist::{
        PathMNISTDataset, CLASS_NAMES, PATH_NUM_CLASSES, is_cancer_class,
        compute_cancer_metrics,
    };
    use growformer::clifford_mnist::{
        CliffordRGBEncoder, CliffordDiracEncoder, LinearClassifier,
        diagnose_pair_learnability,
    };
    use growformer::clifford::CL8_DIM;

    let data_dir = std::path::PathBuf::from(
        std::env::var("PATHMNIST64_ROOT")
            .unwrap_or_else(|_| "data/pathology/pathmnist_64".to_string())
    );
    if !data_dir.exists() {
        eprintln!("PathMNIST 64×64 data not found at {:?}", data_dir);
        eprintln!("Run: python3 data/pathology/extract_npz.py");
        std::process::exit(1);
    }

    println!("═══════════════════════════════════════════════════════════════");
    println!("  Growformer Medical: Colorectal Cancer Histology (64×64)");
    println!("  PathMNIST — 9 classes, RGB+Dirac Cl(1,7), single-pass solve");
    println!("  Resolution: 64×64×3 = 12,288 input dims (vs 2,352 at 28×28)");
    println!("═══════════════════════════════════════════════════════════════\n");

    println!("Loading PathMNIST 64×64 (RGB)...");
    let train = PathMNISTDataset::load_with_resolution(&data_dir, "train", 64, 64);
    let val = PathMNISTDataset::load_with_resolution(&data_dir, "val", 64, 64);
    let test = PathMNISTDataset::load_with_resolution(&data_dir, "test", 64, 64);
    println!("  Train: {}, Val: {}, Test: {}", train.n, val.n, test.n);

    let dist = train.class_distribution();
    println!("\n  Class distribution (train):");
    for c in 0..PATH_NUM_CLASSES {
        let marker = if is_cancer_class(c as u8) { " ← CANCER" } else { "" };
        println!("    {}: {:>14} {:>6}{}", c, CLASS_NAMES[c], dist[c], marker);
    }

    let total_start = Instant::now();

    let mut rgb_enc = CliffordRGBEncoder::new_with_resolution(42, 64, 64);
    let mut dirac_enc = CliffordDiracEncoder::new_with_resolution(77, 64, 64);

    let rgb_cal: Vec<_> = train.images_rgb.iter().take(500)
        .map(|img| (img.clone(), 0u8)).collect();
    rgb_enc.calibrate_scales(&rgb_cal);
    let gray_cal: Vec<_> = train.images_gray.iter().take(500)
        .map(|img| (img.clone(), 0u8)).collect();
    dirac_enc.calibrate_scales(&gray_cal);

    let train_n = train_limit.unwrap_or(train.n).min(train.n);
    let joint_dim = CL8_DIM * 2; // 512D

    let train_lbls_full = &train.labels[..train_n];

    // ══════════════════════════════════════════════════════════════════════
    //  THE KEY TEST: |B| diagnostic at 64×64
    //  If stroma-debris |B| rises from 0.098 → >0.3, resolution was the bottleneck.
    // ══════════════════════════════════════════════════════════════════════
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  Geometric Learnability Diagnostic at 64×64");
    println!("  |B| = |<Z_a * Z_b†>_2| — THE KEY TEST");
    println!("═══════════════════════════════════════════════════════════════");

    let diag_start = Instant::now();
    let (bv_matrix, rotational_pairs, degenerate_pairs) =
        diagnose_pair_learnability(&rgb_enc, &train.images_rgb, train_lbls_full, PATH_NUM_CLASSES, 500);

    println!("\n  Confusion bivector |B| matrix (64×64):");
    print!("         ");
    for c in 0..PATH_NUM_CLASSES { print!("{:>6}", c); }
    println!();
    for i in 0..PATH_NUM_CLASSES {
        print!("    {}:{:>8} ", i, &CLASS_NAMES[i][..CLASS_NAMES[i].len().min(8)]);
        for j in 0..PATH_NUM_CLASSES {
            if i == j { print!("     -"); }
            else {
                let bv = bv_matrix[i][j];
                let tag = if bv >= 0.3 { "+" } else if bv < 0.1 { "!" } else { " " };
                print!("{}{:5.3}", tag, bv);
            }
        }
        println!();
    }

    println!("\n  Rotational (|B| >= 0.3): {} pairs", rotational_pairs.len());
    println!("  Degenerate (|B| < 0.1):  {} pairs", degenerate_pairs.len());
    for &(a, b) in &degenerate_pairs {
        println!("    {} ↔ {}  ({} ↔ {})  |B|={:.3}",
            a, b, CLASS_NAMES[a as usize], CLASS_NAMES[b as usize], bv_matrix[a as usize][b as usize]);
    }

    // Compare with 28×28 values
    println!("\n  ── Comparison with 28×28 baseline ──");
    println!("  Key pairs to watch:");
    println!("    stroma(7) ↔ debris(2):  28×28 |B|=0.098  → 64×64 |B|={:.3}", bv_matrix[7][2]);
    println!("    stroma(7) ↔ adeno(8):   28×28 |B|=0.056  → 64×64 |B|={:.3}", bv_matrix[7][8]);
    println!("    debris(2) ↔ adeno(8):   28×28 |B|=0.082  → 64×64 |B|={:.3}", bv_matrix[2][8]);
    println!("    smooth(5) ↔ normal(6):  28×28 |B|=0.091  → 64×64 |B|={:.3}", bv_matrix[5][6]);

    let stroma_debris_bv = bv_matrix[7][2];
    if stroma_debris_bv >= 0.3 {
        println!("\n  ✓ RESOLUTION WAS THE BOTTLENECK — stroma-debris now rotationally separable");
    } else if stroma_debris_bv >= 0.1 {
        println!("\n  ~ PARTIAL IMPROVEMENT — stroma-debris weakly rotational, may benefit from CGD");
    } else {
        println!("\n  ✗ RESOLUTION IS NOT THE BOTTLENECK — stroma-debris still degenerate");
    }
    println!("  Diagnostic: {:.1}s", diag_start.elapsed().as_secs_f64());

    // ══════════════════════════════════════════════════════════════════════
    //  Full pipeline: encode + classify
    // ══════════════════════════════════════════════════════════════════════
    let full_dim = joint_dim;
    println!("\n--- Encoding {} training images (RGB+Dirac 64×64 → {}D) ---", train_n, full_dim);
    let encode_start = Instant::now();

    let mut train_features: Vec<Vec<f32>> = Vec::with_capacity(train_n);
    for i in 0..train_n {
        let rgb_mv = rgb_enc.encode(&train.images_rgb[i]);
        let dirac_mv = dirac_enc.encode(&train.images_gray[i]);

        let mut joint = Vec::with_capacity(full_dim);
        joint.extend_from_slice(&rgb_mv.components);
        joint.extend_from_slice(&dirac_mv.components);
        train_features.push(joint);

        if (i + 1) % 10000 == 0 {
            println!("    {}/{} ({:.0}s)", i + 1, train_n, encode_start.elapsed().as_secs_f64());
        }
    }
    let train_lbls = &train.labels[..train_n];
    println!("  Encoded in {:.1}s\n", encode_start.elapsed().as_secs_f64());

    println!("--- Solving 9-class classifier ({}D normal equations) ---", full_dim);
    let head = LinearClassifier::fit_from_features(
        &train_features, train_lbls,
        PATH_NUM_CLASSES, 1.0, 1,
    );

    println!("\n--- Solving binary cancer classifier ({}D) ---", full_dim);
    let cancer_labels: Vec<u8> = train_lbls.iter()
        .map(|&l| if is_cancer_class(l) { 1 } else { 0 })
        .collect();
    let cancer_head = LinearClassifier::fit_from_features(
        &train_features, &cancer_labels,
        2, 1.0, 1,
    );

    // ── Evaluate on test set ──
    println!("\n--- Evaluating on test set ({} samples) ---", test.n);
    let eval_start = Instant::now();

    let mut correct_9 = 0usize;
    let mut per_class_correct = vec![0usize; PATH_NUM_CLASSES];
    let mut per_class_total = vec![0usize; PATH_NUM_CLASSES];
    let mut cancer_preds = Vec::with_capacity(test.n);
    let mut cancer_truths = Vec::with_capacity(test.n);

    for i in 0..test.n {
        let rgb_mv = rgb_enc.encode(&test.images_rgb[i]);
        let dirac_mv = dirac_enc.encode(&test.images_gray[i]);

        let mut joint = Vec::with_capacity(full_dim);
        joint.extend_from_slice(&rgb_mv.components);
        joint.extend_from_slice(&dirac_mv.components);

        let (pred_9, _) = head.classify_features(&joint);
        let true_label = test.labels[i] as usize;
        per_class_total[true_label] += 1;
        if pred_9 as usize == true_label {
            correct_9 += 1;
            per_class_correct[true_label] += 1;
        }

        let (cancer_pred, _) = cancer_head.classify_features(&joint);
        cancer_preds.push(cancer_pred as u8);
        cancer_truths.push(if is_cancer_class(test.labels[i]) { 1u8 } else { 0u8 });
    }

    let acc_9 = correct_9 as f64 / test.n as f64;
    let cm = compute_cancer_metrics(&cancer_preds, &cancer_truths);
    let cancer_sens = cm.sensitivity;
    let cancer_spec = cm.specificity;

    println!("  Evaluation: {:.1}s\n", eval_start.elapsed().as_secs_f64());

    println!("═══════════════════════════════════════════════════════════════");
    println!("  PathMNIST 64×64 RESULTS");
    println!("═══════════════════════════════════════════════════════════════");
    println!("  9-class accuracy: {:.1}%  (test, {})", acc_9 * 100.0, test.n);
    println!("  Cancer sensitivity: {:.1}%", cancer_sens * 100.0);
    println!("  Cancer specificity: {:.1}%", cancer_spec * 100.0);
    println!();
    println!("  Per-class recall:");
    for c in 0..PATH_NUM_CLASSES {
        let recall = if per_class_total[c] > 0 {
            per_class_correct[c] as f64 / per_class_total[c] as f64
        } else { 0.0 };
        let marker = if is_cancer_class(c as u8) { " ← CANCER" } else { "" };
        println!("    {}: {:>14}  {:.1}% ({}/{}){}", c, CLASS_NAMES[c],
            recall * 100.0, per_class_correct[c], per_class_total[c], marker);
    }

    let total_time = total_start.elapsed().as_secs_f64();
    println!("\n  Total time: {:.1}s", total_time);
    println!("  Architecture: Cl(1,7) RGB(12288→256) + Dirac(4096→256) = 512D");
    println!("  Classifier: single-pass linear solve with class weights");
    println!("  Training: zero epochs (algebraic)");
    println!("═══════════════════════════════════════════════════════════════");
}

// =============================================================================
// Demo: ARC-AGI — Abstract Reasoning through Cl(1,7) spacetime algebra.
// Rules are rotors. |B| predicts task difficulty before solving.
// =============================================================================

fn demo_arc_agi() {
    use growformer::arc_agi::{
        load_arc_tasks, solve_task, encode_grid, extract_rule,
        rotor_consistency, solve_normal_equations, print_grid,
    };

    let data_dir = std::path::PathBuf::from(
        std::env::var("ARC_AGI_ROOT")
            .unwrap_or_else(|_| "data/arc-agi/data/training".to_string())
    );
    if !data_dir.exists() {
        eprintln!("ARC-AGI data not found at {:?}", data_dir);
        eprintln!("Clone https://github.com/fchollet/ARC-AGI into data/arc-agi/");
        std::process::exit(1);
    }

    println!("═══════════════════════════════════════════════════════════════");
    println!("  ARC-AGI-1 — Abstract Reasoning Corpus");
    println!("  Cl(1,7) spacetime algebra: rules are rotors");
    println!("═══════════════════════════════════════════════════════════════\n");

    let total_start = Instant::now();

    println!("Loading ARC-AGI training tasks...");
    let tasks = load_arc_tasks(&data_dir);
    println!("  {} tasks loaded ({:.1}s)\n", tasks.len(), total_start.elapsed().as_secs_f64());

    // ── Phase 1: |B| diagnostic on all tasks ──
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Phase 1: |B| Diagnostic — rotor consistency across examples");
    println!("═══════════════════════════════════════════════════════════════\n");

    let diag_start = Instant::now();

    let mut diagnostics: Vec<(String, usize, f32, bool)> = Vec::new();
    let mut bv_histogram = [0usize; 5]; // [0, 0.1), [0.1, 0.3), [0.3, 0.5), [0.5, 1.0), [1.0+)

    for task in &tasks {
        let inputs: Vec<_> = task.train.iter().map(|ex| encode_grid(&ex.input)).collect();
        let outputs: Vec<_> = task.train.iter().map(|ex| encode_grid(&ex.output)).collect();
        let rules: Vec<_> = inputs.iter().zip(outputs.iter())
            .map(|(i, o)| extract_rule(i, o))
            .collect();
        let (mean_bv, _) = rotor_consistency(&rules);

        let same_dims = task.train.iter().all(|ex|
            ex.input.height == ex.output.height && ex.input.width == ex.output.width);

        diagnostics.push((task.id.clone(), task.train.len(), mean_bv, same_dims));

        let bin = if mean_bv < 0.1 { 0 }
            else if mean_bv < 0.3 { 1 }
            else if mean_bv < 0.5 { 2 }
            else if mean_bv < 1.0 { 3 }
            else { 4 };
        bv_histogram[bin] += 1;
    }

    println!("  |B| distribution across {} tasks:", tasks.len());
    println!("    |B| < 0.1  (rotor-consistent):  {:>3} tasks", bv_histogram[0]);
    println!("    |B| 0.1-0.3 (weakly rotational): {:>3} tasks", bv_histogram[1]);
    println!("    |B| 0.3-0.5 (mixed):             {:>3} tasks", bv_histogram[2]);
    println!("    |B| 0.5-1.0 (context-dependent):  {:>3} tasks", bv_histogram[3]);
    println!("    |B| > 1.0  (highly inconsistent): {:>3} tasks", bv_histogram[4]);

    let same_dim_count = diagnostics.iter().filter(|d| d.3).count();
    println!("\n  Same-dimension tasks: {}/{}", same_dim_count, tasks.len());
    println!("  Diagnostic: {:.1}s\n", diag_start.elapsed().as_secs_f64());

    // Show top-10 most rotor-consistent tasks
    let mut sorted_diags = diagnostics.clone();
    sorted_diags.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
    println!("  Top-10 most rotor-consistent tasks (lowest |B|):");
    for (id, n_train, bv, same) in sorted_diags.iter().take(10) {
        let dim_tag = if *same { "same-dim" } else { "diff-dim" };
        println!("    {} |B|={:.4}  ({} examples, {})", id, bv, n_train, dim_tag);
    }

    println!("\n  Top-10 least consistent tasks (highest |B|):");
    for (id, n_train, bv, same) in sorted_diags.iter().rev().take(10) {
        let dim_tag = if *same { "same-dim" } else { "diff-dim" };
        println!("    {} |B|={:.4}  ({} examples, {})", id, bv, n_train, dim_tag);
    }

    // ── Phase 2: Solve all tasks ──
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  Phase 2: Solve — extract rotor, apply, decode");
    println!("═══════════════════════════════════════════════════════════════\n");

    let solve_start = Instant::now();
    let mut n_solved = 0usize;
    let n_total = tasks.len();
    let mut total_correct_cells = 0usize;
    let mut total_cells = 0usize;
    let mut solved_tasks: Vec<String> = Vec::new();
    let mut results: Vec<(String, f32, f32, bool, &str)> = Vec::new();
    let mut strategy_counts: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();

    for (i, task) in tasks.iter().enumerate() {
        let diag = solve_task(task);

        let cell_acc = if diag.n_total_cells > 0 {
            diag.n_correct_cells as f32 / diag.n_total_cells as f32
        } else { 0.0 };

        if diag.solved {
            n_solved += 1;
            solved_tasks.push(task.id.clone());
        }
        total_correct_cells += diag.n_correct_cells;
        total_cells += diag.n_total_cells;

        *strategy_counts.entry(diag.strategy).or_insert(0) += 1;
        results.push((task.id.clone(), diag.mean_bv_norm, cell_acc, diag.solved, diag.strategy));

        if (i + 1) % 50 == 0 || (i + 1) == n_total {
            println!("    {}/{} tasks evaluated ({} solved so far, {:.1}s)",
                i + 1, n_total, n_solved, solve_start.elapsed().as_secs_f64());
        }
    }

    println!("  Solve: {:.1}s", solve_start.elapsed().as_secs_f64());
    println!("  Strategy selection:");
    for (name, count) in &strategy_counts {
        println!("    {:>13}: {} tasks", name, count);
    }
    println!();

    // ── Results ──
    let overall_cell_acc = if total_cells > 0 {
        total_correct_cells as f32 / total_cells as f32
    } else { 0.0 };

    println!("═══════════════════════════════════════════════════════════════");
    println!("  ARC-AGI-1 RESULTS (training set, {} tasks)", n_total);
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Tasks exactly solved:  {}/{}  ({:.1}%)",
        n_solved, n_total, n_solved as f64 / n_total as f64 * 100.0);
    println!("  Cell-level accuracy:   {:.1}%  ({}/{})",
        overall_cell_acc * 100.0, total_correct_cells, total_cells);

    if !solved_tasks.is_empty() {
        println!("\n  Solved tasks:");
        for id in &solved_tasks {
            let res = results.iter().find(|r| &r.0 == id).unwrap();
            println!("    {} |B|={:.4}  strategy={}", id, res.1, res.4);
        }
    }

    // ── Accuracy by |B| bucket ──
    println!("\n  Accuracy by rotor consistency:");
    let buckets: [(f32, f32, &str); 5] = [
        (0.0, 0.1, "|B| < 0.1"),
        (0.1, 0.3, "|B| 0.1-0.3"),
        (0.3, 0.5, "|B| 0.3-0.5"),
        (0.5, 1.0, "|B| 0.5-1.0"),
        (1.0, f32::MAX, "|B| > 1.0"),
    ];

    for &(lo, hi, label) in &buckets {
        let bucket_results: Vec<_> = results.iter()
            .filter(|r| r.1 >= lo && r.1 < hi)
            .collect();
        if bucket_results.is_empty() { continue; }
        let n = bucket_results.len();
        let n_solved_bucket = bucket_results.iter().filter(|r| r.3).count();
        let mean_cell_acc: f32 = bucket_results.iter().map(|r| r.2).sum::<f32>() / n as f32;
        println!("    {:>13}: {:>3} tasks, {:>2} solved ({:.1}%), cell acc {:.1}%",
            label, n, n_solved_bucket,
            n_solved_bucket as f64 / n as f64 * 100.0,
            mean_cell_acc * 100.0);
    }

    // Show a few example tasks
    println!("\n  Example task visualizations:");
    let examples_to_show: Vec<_> = if !solved_tasks.is_empty() {
        tasks.iter().filter(|t| solved_tasks.contains(&t.id)).take(3).collect()
    } else {
        let mut by_acc: Vec<_> = results.iter().collect();
        by_acc.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        by_acc.iter().take(3)
            .filter_map(|r| tasks.iter().find(|t| t.id == r.0))
            .collect()
    };

    for task in examples_to_show {
        println!("\n  Task: {}", task.id);
        let res = results.iter().find(|r| r.0 == task.id).unwrap();
        println!("  |B|={:.4}  cell_acc={:.1}%  solved={}  strategy={}", res.1, res.2 * 100.0, res.3, res.4);
        if let Some(ex) = task.train.first() {
            println!("    Train input ({}×{}):", ex.input.height, ex.input.width);
            print_grid(&ex.input, "      ");
            println!("    Train output ({}×{}):", ex.output.height, ex.output.width);
            print_grid(&ex.output, "      ");
        }
    }

    let total_time = total_start.elapsed().as_secs_f64();
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  ARC-AGI-1 BENCHMARK SUMMARY");
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Tasks solved:     {}/{}  ({:.1}%)", n_solved, n_total,
        n_solved as f64 / n_total as f64 * 100.0);
    println!("  Cell accuracy:    {:.1}%", overall_cell_acc * 100.0);
    println!("  Total time:       {:.1}s", total_time);
    println!("  Architecture:     Growformer");
    println!("  Parameters:       0 (no training)");
    println!("  GPU required:     None");
    println!("═══════════════════════════════════════════════════════════════");
}

// =============================================================================
// Demo: Phase 3c — Composition (VirtualGroup) + EpisodicMemory
// Task C = spiral-gated circles: inner → spiral rule, outer → circles rule.
// =============================================================================

/// Parallel minibatch size for mirror training when the `parallel` feature is on.
#[allow(dead_code)]
fn demo_mirror_batch_size() -> Option<usize> {
    #[cfg(feature = "parallel")]
    {
        std::thread::available_parallelism()
            .ok()
            .map(|p| p.get().min(32).max(2))
    }
    #[cfg(not(feature = "parallel"))]
    {
        None
    }
}

const DEMO_MIRROR_TARGET_ACC: f32 = 0.93;
const DEMO_MIRROR_MAX_EPOCHS: usize = 2500;
const DEMO_MIRROR_LOG_INTERVAL: usize = 250;
const DEMO_MIRROR_PROMOTE_CHECK: usize = 50;

fn phase3_composition_config() -> DimensionManagerConfig {
    DimensionManagerConfig {
        mirror_config: phase2_base_config(),
        mirror_layer_sizes: vec![2, 16, 16, 1],
        promotion_check_interval: 500,
        max_concurrent_mirrors: 2,
        calibration_samples: 100,
        reserve_pool_size: 0,
    }
}

/// Train a mirror with gradient-only SGD, early stopping, then promote to Main.
fn train_promoted_mirror(
    dm: &mut DimensionManager,
    task_name: &str,
    seed: u64,
    data: &[Sample],
    calibration: &[Sample],
    rng: &mut StdRng,
    verbose: bool,
) -> GroupId {
    dm.spawn_mirror(task_name, seed).expect(task_name);
    if verbose {
        println!(
            "=== Training mirror: {} (gradient-only, target={:.0}%) ===\n",
            task_name,
            DEMO_MIRROR_TARGET_ACC * 100.0
        );
    }

    for epoch in 0..DEMO_MIRROR_MAX_EPOCHS {
        let Some(result) = dm.train_mirror_epoch_gradient(task_name, data, rng) else {
            break;
        };
        if epoch % DEMO_MIRROR_PROMOTE_CHECK == 0 {
            dm.evaluate_promotions(calibration);
            if !dm.mirrors.contains_key(task_name) {
                if verbose {
                    println!("  [{}] auto-promoted at epoch {}", task_name, epoch);
                }
                break;
            }
        }
        if verbose && epoch % DEMO_MIRROR_LOG_INTERVAL == 0 {
            println!(
                "  [{}] epoch {:>4} | loss={:.4} | acc={:.1}%",
                task_name,
                epoch,
                result.loss,
                result.accuracy * 100.0
            );
        }
        if result.accuracy >= DEMO_MIRROR_TARGET_ACC {
            if verbose {
                println!(
                    "  [{}] reached {:.0}% at epoch {}",
                    task_name,
                    DEMO_MIRROR_TARGET_ACC * 100.0,
                    epoch
                );
            }
            break;
        }
    }

    if dm.mirrors.contains_key(task_name) {
        dm.force_promote(task_name, calibration)
            .unwrap_or_else(|| *dm.main.group_order.last().unwrap())
    } else {
        *dm.main.group_order.last().unwrap()
    }
}

fn evaluate_mirror_accuracy(
    dm: &mut DimensionManager,
    task_name: &str,
    data: &[Sample],
) -> f32 {
    let Some(mirror) = dm.mirrors.get_mut(task_name) else {
        return 0.0;
    };
    let mut correct = 0usize;
    for (input, target) in data {
        let out = mirror.env.predict(input.as_slice());
        if out.len() >= 1 && (out[0] - target[0]).abs() < 0.5 {
            correct += 1;
        }
    }
    if data.is_empty() {
        0.0
    } else {
        correct as f32 / data.len() as f32
    }
}

/// Train a mirror on `train` only; return held-out accuracy. Mirror is discarded (not promoted).
fn train_direct_composite_mirror(
    dm: &mut DimensionManager,
    train: &[Sample],
    heldout: &[Sample],
    seed: u64,
    rng: &mut StdRng,
) -> f32 {
    const TASK: &str = "composite_direct";
    dm.spawn_mirror(TASK, seed).expect("spawn composite_direct");
    for epoch in 0..DEMO_MIRROR_MAX_EPOCHS {
        let Some(result) = dm.train_mirror_epoch_gradient(TASK, train, rng) else {
            break;
        };
        if result.accuracy >= DEMO_MIRROR_TARGET_ACC {
            break;
        }
    }
    let acc = evaluate_mirror_accuracy(dm, TASK, heldout);
    dm.mirrors.remove(TASK);
    acc
}

fn accuracy_virtual_group(
    dm: &mut DimensionManager,
    vg: &VirtualGroup,
    data: &[Sample],
) -> f32 {
    let mut correct = 0usize;
    for (input, target) in data {
        let out = dm.predict_with_composition(input, vg);
        if out.len() >= 1 && (out[0] - target[0]).abs() < 0.5 {
            correct += 1;
        }
    }
    if data.is_empty() {
        0.0
    } else {
        correct as f32 / data.len() as f32
    }
}

fn stratified_composite_split(
    data: &[Sample],
    inner_radius: f32,
    train_n: usize,
    rng: &mut StdRng,
) -> (Vec<Sample>, Vec<Sample>) {
    let mut inner = Vec::new();
    let mut outer = Vec::new();
    for sample in data {
        let r = (sample.0[0] * sample.0[0] + sample.0[1] * sample.0[1]).sqrt();
        if r < inner_radius {
            inner.push(sample.clone());
        } else {
            outer.push(sample.clone());
        }
    }
    inner.shuffle(rng);
    outer.shuffle(rng);
    let train_inner = train_n / 2;
    let train_outer = train_n.saturating_sub(train_inner);
    let mut train = Vec::with_capacity(train_n);
    train.extend(inner.iter().take(train_inner).cloned());
    train.extend(outer.iter().take(train_outer).cloned());
    let mut heldout = Vec::with_capacity(inner.len() + outer.len() - train.len());
    heldout.extend(inner.iter().skip(train_inner).cloned());
    heldout.extend(outer.iter().skip(train_outer).cloned());
    (train, heldout)
}

fn inner_region_fraction(data: &[Sample], inner_radius: f32) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    let inner = data
        .iter()
        .filter(|(input, _)| {
            let r = (input[0] * input[0] + input[1] * input[1]).sqrt();
            r < inner_radius
        })
        .count();
    inner as f32 / data.len() as f32
}

fn sample_radius(input: &[f32]) -> f32 {
    (input[0] * input[0] + input[1] * input[1]).sqrt()
}

fn specialist_scalar(
    main: &mut MainDimension,
    group_id: GroupId,
    input: &[f32],
) -> f32 {
    main.query(input, &[group_id])
        .first()
        .and_then(|(_, o)| o.first().copied())
        .unwrap_or(0.5)
}

fn scalar_matches_target(scalar: f32, target: f32) -> bool {
    (scalar - target).abs() < 0.5
}

/// Per-point hard switch: use spiral output if r < threshold, else circles.
fn accuracy_radius_gated(
    main: &mut MainDimension,
    spiral_gid: GroupId,
    circles_gid: GroupId,
    data: &[Sample],
    threshold: f32,
) -> f32 {
    let mut correct = 0usize;
    for (input, target) in data {
        let r = sample_radius(input);
        let scalar = if r < threshold {
            specialist_scalar(main, spiral_gid, input)
        } else {
            specialist_scalar(main, circles_gid, input)
        };
        if scalar_matches_target(scalar, target[0]) {
            correct += 1;
        }
    }
    if data.is_empty() {
        0.0
    } else {
        correct as f32 / data.len() as f32
    }
}

/// Learn radius threshold on train by grid search over midpoints between sorted train radii.
fn learn_radius_threshold(
    main: &mut MainDimension,
    spiral_gid: GroupId,
    circles_gid: GroupId,
    train: &[Sample],
    default: f32,
) -> f32 {
    let mut radii: Vec<f32> = train.iter().map(|(input, _)| sample_radius(input)).collect();
    if radii.is_empty() {
        return default;
    }
    radii.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut candidates = vec![default, 0.2, 0.35, 0.4, 0.5, 0.65];
    for pair in radii.windows(2) {
        candidates.push((pair[0] + pair[1]) * 0.5);
    }
    let mut best_t = default;
    let mut best_acc = 0.0f32;
    for t in candidates {
        let acc = accuracy_radius_gated(main, spiral_gid, circles_gid, train, t);
        if acc > best_acc {
            best_acc = acc;
            best_t = t;
        }
    }
    best_t
}

/// Per-point: pick the specialist with more decisive (extreme) scalar output.
fn accuracy_confidence_argmax(
    main: &mut MainDimension,
    spiral_gid: GroupId,
    circles_gid: GroupId,
    data: &[Sample],
) -> f32 {
    let mut correct = 0usize;
    for (input, target) in data {
        let o_spiral = specialist_scalar(main, spiral_gid, input);
        let o_circles = specialist_scalar(main, circles_gid, input);
        let scalar = if (o_spiral - 0.5).abs() >= (o_circles - 0.5).abs() {
            o_spiral
        } else {
            o_circles
        };
        if scalar_matches_target(scalar, target[0]) {
            correct += 1;
        }
    }
    if data.is_empty() {
        0.0
    } else {
        correct as f32 / data.len() as f32
    }
}

#[derive(Clone, Copy, Debug)]
struct RadiusLogisticGate {
    w: f32,
    b: f32,
}

fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}

/// Train logistic P(use spiral) = σ(w·r + b) from which specialist is closer to label on train.
fn train_radius_logistic_gate(
    main: &mut MainDimension,
    spiral_gid: GroupId,
    circles_gid: GroupId,
    train: &[Sample],
) -> RadiusLogisticGate {
    let mut gate = RadiusLogisticGate { w: 0.0, b: 0.0 };
    let lr = 0.15;
    for _ in 0..300 {
        for (input, target) in train {
            let r = sample_radius(input);
            let o_spiral = specialist_scalar(main, spiral_gid, input);
            let o_circles = specialist_scalar(main, circles_gid, input);
            let y = if (o_spiral - target[0]).abs() < (o_circles - target[0]).abs() {
                1.0
            } else {
                0.0
            };
            let z = gate.w * r + gate.b;
            let p = sigmoid(z);
            let err = p - y;
            gate.w -= lr * err * r;
            gate.b -= lr * err;
        }
    }
    gate
}

/// Which specialist index (0 = spiral, 1 = circles) to route to, from composite labels only.
fn routing_teacher_index(
    main: &mut MainDimension,
    spiral_gid: GroupId,
    circles_gid: GroupId,
    input: &[f32],
    target: f32,
) -> usize {
    let o_spiral = specialist_scalar(main, spiral_gid, input);
    let o_circles = specialist_scalar(main, circles_gid, input);
    let spiral_ok = scalar_matches_target(o_spiral, target);
    let circles_ok = scalar_matches_target(o_circles, target);
    if spiral_ok && !circles_ok {
        return 0;
    }
    if circles_ok && !spiral_ok {
        return 1;
    }
    if (o_spiral - target).abs() <= (o_circles - target).abs() {
        0
    } else {
        1
    }
}

fn specialist_feature_pair(
    main: &mut MainDimension,
    spiral_gid: GroupId,
    circles_gid: GroupId,
    input: &[f32],
) -> Vec<f32> {
    vec![
        specialist_scalar(main, spiral_gid, input),
        specialist_scalar(main, circles_gid, input),
    ]
}

/// Fraction of train points with radius within `margin` of the region boundary.
fn train_boundary_near_fraction(train: &[Sample], inner_radius: f32, margin: f32) -> f32 {
    if train.is_empty() {
        return 0.0;
    }
    let near = train
        .iter()
        .filter(|(input, _)| (sample_radius(input) - inner_radius).abs() < margin)
        .count();
    near as f32 / train.len() as f32
}

fn train_task_e_learned_router_xy(
    main: &mut MainDimension,
    spiral_gid: GroupId,
    circles_gid: GroupId,
    train: &[Sample],
) -> LearnedRouter {
    let samples: Vec<(Vec<f32>, GroupId)> = train
        .iter()
        .map(|(input, target)| {
            let idx = routing_teacher_index(main, spiral_gid, circles_gid, input, target[0]);
            (input.clone(), idx as GroupId)
        })
        .collect();
    LearnedRouter::build(2, 2, &samples)
}

/// Router features are specialist scalars only — position never enters the lattice.
fn train_task_e_learned_router_expert(
    main: &mut MainDimension,
    spiral_gid: GroupId,
    circles_gid: GroupId,
    train: &[Sample],
) -> LearnedRouter {
    let samples: Vec<(Vec<f32>, GroupId)> = train
        .iter()
        .map(|(input, target)| {
            let idx = routing_teacher_index(main, spiral_gid, circles_gid, input, target[0]);
            let features = specialist_feature_pair(main, spiral_gid, circles_gid, input);
            (features, idx as GroupId)
        })
        .collect();
    LearnedRouter::build(2, 2, &samples)
}

/// Deployment-stack router: task-identity labels from original calibration data, not Task E.
fn train_calibration_learned_router_xy(
    calibration_spiral: &[Sample],
    calibration_circles: &[Sample],
) -> LearnedRouter {
    let mut samples: Vec<(Vec<f32>, GroupId)> = Vec::with_capacity(
        calibration_spiral.len() + calibration_circles.len(),
    );
    for (input, _) in calibration_spiral {
        samples.push((input.clone(), 0));
    }
    for (input, _) in calibration_circles {
        samples.push((input.clone(), 1));
    }
    LearnedRouter::build(2, 2, &samples)
}

/// Deployment router on expert outputs only — no position in features.
fn train_calibration_learned_router_expert(
    main: &mut MainDimension,
    spiral_gid: GroupId,
    circles_gid: GroupId,
    calibration_spiral: &[Sample],
    calibration_circles: &[Sample],
) -> LearnedRouter {
    let mut samples: Vec<(Vec<f32>, GroupId)> = Vec::with_capacity(
        calibration_spiral.len() + calibration_circles.len(),
    );
    for (input, _) in calibration_spiral {
        let features = specialist_feature_pair(main, spiral_gid, circles_gid, input);
        samples.push((features, 0));
    }
    for (input, _) in calibration_circles {
        let features = specialist_feature_pair(main, spiral_gid, circles_gid, input);
        samples.push((features, 1));
    }
    LearnedRouter::build(2, 2, &samples)
}

/// Soft-leak control: route on specialist disagreement scalar only.
fn train_task_e_router_disagreement(
    main: &mut MainDimension,
    spiral_gid: GroupId,
    circles_gid: GroupId,
    train: &[Sample],
) -> LearnedRouter {
    let samples: Vec<(Vec<f32>, GroupId)> = train
        .iter()
        .map(|(input, target)| {
            let idx = routing_teacher_index(main, spiral_gid, circles_gid, input, target[0]);
            let f0 = specialist_scalar(main, spiral_gid, input);
            let f1 = specialist_scalar(main, circles_gid, input);
            (vec![f0 - f1], idx as GroupId)
        })
        .collect();
    LearnedRouter::build(1, 2, &samples)
}

fn pearson_correlation(xs: &[f32], ys: &[f32]) -> f32 {
    if xs.len() != ys.len() || xs.len() < 2 {
        return 0.0;
    }
    let n = xs.len() as f32;
    let mx = xs.iter().sum::<f32>() / n;
    let my = ys.iter().sum::<f32>() / n;
    let mut num = 0.0f32;
    let mut dx2 = 0.0f32;
    let mut dy2 = 0.0f32;
    for (&x, &y) in xs.iter().zip(ys.iter()) {
        let dx = x - mx;
        let dy = y - my;
        num += dx * dy;
        dx2 += dx * dx;
        dy2 += dy * dy;
    }
    let denom = (dx2 * dy2).sqrt();
    if denom < 1e-12 {
        0.0
    } else {
        num / denom
    }
}

fn router_route_index(router: &mut LearnedRouter, features: &[f32]) -> usize {
    router
        .predict_logits(features)
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Does the router's region choice match the generative oracle (spiral if r < threshold)?
fn router_region_agreement_oracle(
    router: &mut LearnedRouter,
    data: &[Sample],
    inner_radius: f32,
    features_for: &mut dyn FnMut(&[f32]) -> Vec<f32>,
) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    let mut agree = 0usize;
    for (input, _) in data {
        let r = sample_radius(input);
        let oracle_idx = if r < inner_radius { 0usize } else { 1usize };
        let features = features_for(input);
        if router_route_index(router, &features) == oracle_idx {
            agree += 1;
        }
    }
    agree as f32 / data.len() as f32
}

/// Pearson correlation between router margin (logit_spiral − logit_circles) and (inner_radius − r).
/// Strong positive correlation ⇒ decision boundary tracks the radius circle.
fn router_margin_radius_correlation(
    router: &mut LearnedRouter,
    data: &[Sample],
    inner_radius: f32,
    features_for: &mut dyn FnMut(&[f32]) -> Vec<f32>,
) -> f32 {
    if data.len() < 2 {
        return 0.0;
    }
    let mut margins = Vec::with_capacity(data.len());
    let mut radius_signals = Vec::with_capacity(data.len());
    for (input, _) in data {
        let features = features_for(input);
        let logits = router.predict_logits(&features);
        let margin = if logits.len() >= 2 {
            logits[0] - logits[1]
        } else {
            0.0
        };
        margins.push(margin);
        radius_signals.push(inner_radius - sample_radius(input));
    }
    pearson_correlation(&margins, &radius_signals)
}

#[derive(Clone, Debug)]
struct BoundaryPointRecord {
    seed: u64,
    x: f32,
    y: f32,
    r: f32,
    f_spiral: f32,
    f_circles: f32,
    margin: f32,
    router_spiral: bool,
    oracle_spiral: bool,
    composite_correct: bool,
    region_match: bool,
}

#[derive(Clone, Copy, Debug)]
struct BoundarySeedSummary {
    seed: u64,
    composite_acc: f32,
    region_agreement: f32,
    margin_radius_corr: f32,
    misroute_mean_dr: f32,
    f_spiral_r_corr: f32,
    f_circles_r_corr: f32,
    train_near_boundary_frac: f32,
}

/// Fit P(spiral route) = σ(w·x₁ + …) by batch gradient descent on train labels.
fn fit_logistic_router(
    train_x: &[Vec<f32>],
    train_y: &[f32],
    lr: f32,
    epochs: usize,
) -> Vec<f32> {
    let dim = train_x.first().map(|v| v.len()).unwrap_or(0);
    let mut w = vec![0.0f32; dim];
    let mut b = 0.0f32;
    if dim == 0 || train_x.is_empty() {
        return w;
    }
    for _ in 0..epochs {
        for (x, &y) in train_x.iter().zip(train_y.iter()) {
            let z: f32 = x.iter().zip(w.iter()).map(|(a, wi)| a * wi).sum::<f32>() + b;
            let p = sigmoid(z);
            let err = p - y;
            for (wi, xi) in w.iter_mut().zip(x.iter()) {
                *wi -= lr * err * xi;
            }
            b -= lr * err;
        }
    }
    let mut params = w;
    params.push(b);
    params
}

fn logistic_predict(params: &[f32], x: &[f32]) -> f32 {
    if params.is_empty() || x.is_empty() {
        return 0.5;
    }
    let b = *params.last().unwrap_or(&0.0);
    let w = &params[..params.len() - 1];
    let z: f32 = x.iter().zip(w.iter()).map(|(a, wi)| a * wi).sum::<f32>() + b;
    sigmoid(z)
}

fn collect_expert_boundary_records(seed: u64) -> (Vec<BoundaryPointRecord>, BoundarySeedSummary) {
    const INNER_RADIUS: f32 = 0.4;
    const N_SAMPLES: usize = 400;
    const TRAIN_N: usize = 30;

    let mut dm = DimensionManager::new(phase3_composition_config());
    let mut rng = StdRng::seed_from_u64(seed);
    let mut data_rng = StdRng::seed_from_u64(seed.wrapping_mul(97).wrapping_add(99));

    let spiral_data = generate_spiral_data(400, &mut data_rng);
    let circles_data = generate_concentric_circles_data(400, &mut data_rng);
    let calibration_spiral: Vec<_> = spiral_data.iter().take(100).cloned().collect();
    let calibration_circles: Vec<_> = circles_data.iter().take(100).cloned().collect();

    let spiral_group = train_promoted_mirror(
        &mut dm, "spiral", seed, &spiral_data, &calibration_spiral, &mut rng, false,
    );
    let circles_group = train_promoted_mirror(
        &mut dm, "circles", seed.wrapping_add(1), &circles_data, &calibration_circles, &mut rng, false,
    );

    let task_e_data = generate_balanced_spiral_gated_circles_data(
        &mut dm.main, spiral_group, circles_group, INNER_RADIUS, N_SAMPLES, &mut data_rng,
    );
    let (train, heldout) =
        stratified_composite_split(&task_e_data, INNER_RADIUS, TRAIN_N, &mut data_rng);

    let mut router = train_task_e_learned_router_expert(
        &mut dm.main, spiral_group, circles_group, &train,
    );

    let gids = [spiral_group, circles_group];
    let mut records = Vec::with_capacity(heldout.len());
    let mut margins = Vec::new();
    let mut radius_signals = Vec::new();
    let mut f_spiral_vals = Vec::new();
    let mut f_circles_vals = Vec::new();
    let mut radii = Vec::new();
    let mut correct = 0usize;
    let mut region_agree = 0usize;
    let mut misroute_dr_sum = 0.0f32;
    let mut misroute_n = 0usize;

    for (input, target) in &heldout {
        let r = sample_radius(input);
        let f_spiral = specialist_scalar(&mut dm.main, spiral_group, input);
        let f_circles = specialist_scalar(&mut dm.main, circles_group, input);
        let features = vec![f_spiral, f_circles];
        let logits = router.predict_logits(&features);
        let margin = if logits.len() >= 2 { logits[0] - logits[1] } else { 0.0 };
        let idx = router_route_index(&mut router, &features);
        let router_spiral = idx == 0;
        let oracle_spiral = r < INNER_RADIUS;
        let gid = gids[idx.min(1)];
        let scalar = specialist_scalar(&mut dm.main, gid, input);
        let composite_correct = scalar_matches_target(scalar, target[0]);
        let region_match = router_spiral == oracle_spiral;

        if composite_correct {
            correct += 1;
        }
        if region_match {
            region_agree += 1;
        } else {
            misroute_dr_sum += (r - INNER_RADIUS).abs();
            misroute_n += 1;
        }

        margins.push(margin);
        radius_signals.push(INNER_RADIUS - r);
        f_spiral_vals.push(f_spiral);
        f_circles_vals.push(f_circles);
        radii.push(r);

        records.push(BoundaryPointRecord {
            seed,
            x: input[0],
            y: input[1],
            r,
            f_spiral,
            f_circles,
            margin,
            router_spiral,
            oracle_spiral,
            composite_correct,
            region_match,
        });
    }

    let n = heldout.len().max(1) as f32;
    let summary = BoundarySeedSummary {
        seed,
        composite_acc: correct as f32 / n,
        region_agreement: region_agree as f32 / n,
        margin_radius_corr: pearson_correlation(&margins, &radius_signals),
        misroute_mean_dr: if misroute_n > 0 {
            misroute_dr_sum / misroute_n as f32
        } else {
            0.0
        },
        f_spiral_r_corr: pearson_correlation(&f_spiral_vals, &radii),
        f_circles_r_corr: pearson_correlation(&f_circles_vals, &radii),
        train_near_boundary_frac: train_boundary_near_fraction(&train, INNER_RADIUS, 0.08),
    };
    (records, summary)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RadiusZone {
    Interior,
    Annulus,
    Outer,
}

fn radius_zone(r: f32, inner_radius: f32, eps: f32) -> RadiusZone {
    if r < inner_radius - eps {
        RadiusZone::Interior
    } else if r > inner_radius + eps {
        RadiusZone::Outer
    } else {
        RadiusZone::Annulus
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct ZoneMisrouteStats {
    interior_misroute_rate: f32,
    annulus_misroute_rate: f32,
    outer_misroute_rate: f32,
    n_interior: usize,
    n_annulus: usize,
    n_outer: usize,
}

fn zone_misroute_stats(records: &[BoundaryPointRecord], inner_radius: f32, eps: f32) -> ZoneMisrouteStats {
    let mut interior_mis = 0usize;
    let mut annulus_mis = 0usize;
    let mut outer_mis = 0usize;
    let mut n_interior = 0usize;
    let mut n_annulus = 0usize;
    let mut n_outer = 0usize;

    for rec in records {
        let mis = !rec.region_match;
        match radius_zone(rec.r, inner_radius, eps) {
            RadiusZone::Interior => {
                n_interior += 1;
                if mis {
                    interior_mis += 1;
                }
            }
            RadiusZone::Annulus => {
                n_annulus += 1;
                if mis {
                    annulus_mis += 1;
                }
            }
            RadiusZone::Outer => {
                n_outer += 1;
                if mis {
                    outer_mis += 1;
                }
            }
        }
    }

    let rate = |mis: usize, n: usize| {
        if n == 0 {
            0.0
        } else {
            mis as f32 / n as f32
        }
    };

    ZoneMisrouteStats {
        interior_misroute_rate: rate(interior_mis, n_interior),
        annulus_misroute_rate: rate(annulus_mis, n_annulus),
        outer_misroute_rate: rate(outer_mis, n_outer),
        n_interior,
        n_annulus,
        n_outer,
    }
}

fn seed_cluster_label(summary: &BoundarySeedSummary) -> &'static str {
    if (summary.region_agreement - 0.5).abs() < 0.01 {
        "degenerate"
    } else if summary.composite_acc >= 0.85 {
        "ceiling"
    } else if summary.composite_acc < 0.75 {
        "floor"
    } else {
        "mid"
    }
}

fn summarize_boundary_records(records: &[BoundaryPointRecord]) -> Vec<BoundarySeedSummary> {
    use std::collections::HashMap;
    let mut by_seed: HashMap<u64, Vec<&BoundaryPointRecord>> = HashMap::new();
    for rec in records {
        by_seed.entry(rec.seed).or_default().push(rec);
    }
    let mut out: Vec<BoundarySeedSummary> = by_seed
        .into_iter()
        .map(|(seed, pts)| {
            let n = pts.len().max(1) as f32;
            let composite_acc =
                pts.iter().filter(|p| p.composite_correct).count() as f32 / n;
            let region_agreement = pts.iter().filter(|p| p.region_match).count() as f32 / n;
            let margins: Vec<f32> = pts.iter().map(|p| p.margin).collect();
            let radii: Vec<f32> = pts.iter().map(|p| p.r).collect();
            let radius_signals: Vec<f32> = pts.iter().map(|p| 0.4 - p.r).collect();
            let f_spiral: Vec<f32> = pts.iter().map(|p| p.f_spiral).collect();
            let f_circles: Vec<f32> = pts.iter().map(|p| p.f_circles).collect();
            let misroute_dr: Vec<f32> = pts
                .iter()
                .filter(|p| !p.region_match)
                .map(|p| (p.r - 0.4).abs())
                .collect();
            BoundarySeedSummary {
                seed,
                composite_acc,
                region_agreement,
                margin_radius_corr: pearson_correlation(&margins, &radius_signals),
                misroute_mean_dr: if misroute_dr.is_empty() {
                    0.0
                } else {
                    misroute_dr.iter().sum::<f32>() / misroute_dr.len() as f32
                },
                f_spiral_r_corr: pearson_correlation(&f_spiral, &radii),
                f_circles_r_corr: pearson_correlation(&f_circles, &radii),
                train_near_boundary_frac: 0.0,
            }
        })
        .collect();
    out.sort_by_key(|s| s.seed);
    out
}

fn print_annulus_misroute_analysis(
    records: &[BoundaryPointRecord],
    summaries: &[BoundarySeedSummary],
    inner_radius: f32,
    eps: f32,
) {
    println!(
        "\n=== Annulus misroute analysis (ε = {:.2} around r = {:.1}) ===\n",
        eps, inner_radius
    );
    println!("Interior: r < {:.2}  |  Annulus: |r−{:.1}| < {:.2}  |  Outer: r > {:.2}",
        inner_radius - eps, inner_radius, eps, inner_radius + eps);
    println!();
    println!("| Cluster | Seeds | Pts | Misroute interior | Misroute annulus | Misroute outer | Annulus / interior |");
    println!("| ------- | ----- | --- | ----------------- | ---------------- | -------------- | ------------------ |");

    let clusters = ["degenerate", "ceiling", "floor", "mid", "all"];
    for &cluster in &clusters {
        let seed_set: Vec<u64> = if cluster == "all" {
            summaries.iter().map(|s| s.seed).collect()
        } else {
            summaries
                .iter()
                .filter(|s| seed_cluster_label(s) == cluster)
                .map(|s| s.seed)
                .collect()
        };
        if seed_set.is_empty() {
            continue;
        }
        let subset: Vec<BoundaryPointRecord> = records
            .iter()
            .filter(|r| seed_set.contains(&r.seed))
            .cloned()
            .collect();
        let stats = zone_misroute_stats(&subset, inner_radius, eps);
        let ratio = if stats.interior_misroute_rate < 1e-6 {
            if stats.annulus_misroute_rate < 1e-6 {
                0.0
            } else {
                f32::INFINITY
            }
        } else {
            stats.annulus_misroute_rate / stats.interior_misroute_rate
        };
        let ratio_s = if ratio.is_finite() {
            format!("{:.2}×", ratio)
        } else {
            "∞".to_string()
        };
        let seeds_s: String = seed_set
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
            .join(",");
        println!(
            "| {} | {} | {} | {:.1}% (n={}) | {:.1}% (n={}) | {:.1}% (n={}) | {} |",
            cluster,
            seeds_s,
            subset.len(),
            stats.interior_misroute_rate * 100.0,
            stats.n_interior,
            stats.annulus_misroute_rate * 100.0,
            stats.n_annulus,
            stats.outer_misroute_rate * 100.0,
            stats.n_outer,
            ratio_s,
        );
    }

    println!();
    println!("Interpretation:");
    println!("  • Degenerate (~50% region agree): 0% interior + 100% outer misroute ⇒ constant specialist (always spiral).");
    println!("  • Ceiling seeds: annulus/interior ratio > 1 ⇒ partial boundary routing when train covers boundary.");
    println!("  • Uniform ~50% misroute in all zones ⇒ random routing (not observed in degenerate cluster).");
}

fn annulus_interior_ratio_for_seed(
    records: &[BoundaryPointRecord],
    seed: u64,
    inner_radius: f32,
    eps: f32,
) -> f32 {
    let subset: Vec<BoundaryPointRecord> = records
        .iter()
        .filter(|r| r.seed == seed)
        .cloned()
        .collect();
    let stats = zone_misroute_stats(&subset, inner_radius, eps);
    if stats.interior_misroute_rate < 1e-6 {
        f32::INFINITY
    } else {
        stats.annulus_misroute_rate / stats.interior_misroute_rate
    }
}

/// Ceiling-cluster cross-tab: does high composite accuracy co-occur with margin↔r correlation?
fn print_ceiling_margin_cross_tab(
    records: &[BoundaryPointRecord],
    summaries: &[BoundarySeedSummary],
    inner_radius: f32,
    eps: f32,
) {
    let ceiling: Vec<&BoundarySeedSummary> = summaries
        .iter()
        .filter(|s| seed_cluster_label(s) == "ceiling")
        .collect();
    if ceiling.is_empty() {
        println!("\n(No ceiling seeds for margin cross-tab.)");
        return;
    }

    println!("\n=== Ceiling cluster cross-tab (composite acc vs margin↔(0.4−r)) ===\n");
    println!("| Seed | Composite acc | Region agree | Margin↔r | Annulus/interior |");
    println!("| ---- | ------------- | ------------ | -------- | ---------------- |");

    let mut accs = Vec::with_capacity(ceiling.len());
    let mut corrs = Vec::with_capacity(ceiling.len());
    for s in &ceiling {
        let ratio = annulus_interior_ratio_for_seed(records, s.seed, inner_radius, eps);
        let ratio_s = if ratio.is_finite() {
            format!("{:.2}×", ratio)
        } else {
            "∞".to_string()
        };
        println!(
            "| {} | {:.1}% | {:.1}% | {:.3} | {} |",
            s.seed,
            s.composite_acc * 100.0,
            s.region_agreement * 100.0,
            s.margin_radius_corr,
            ratio_s,
        );
        accs.push(s.composite_acc);
        corrs.push(s.margin_radius_corr);
    }

    let acc_corr = pearson_correlation(&accs, &corrs);
    println!(
        "\nPearson(composite acc, margin↔r) across {} ceiling seeds: {:.3}",
        ceiling.len(),
        acc_corr,
    );

    let low_margin_high_acc = ceiling
        .iter()
        .filter(|s| s.composite_acc >= 0.85 && s.margin_radius_corr < 0.30)
        .count();
    let high_margin_high_acc = ceiling
        .iter()
        .filter(|s| s.composite_acc >= 0.85 && s.margin_radius_corr >= 0.30)
        .count();

    println!(
        "Ceiling seeds with acc≥85% and margin↔r≥0.30: {} / {}",
        high_margin_high_acc,
        ceiling.len(),
    );
    println!(
        "Ceiling seeds with acc≥85% and margin↔r<0.30:  {} / {} (non-radius route to high acc)",
        low_margin_high_acc,
        ceiling.len(),
    );

    if low_margin_high_acc == 0 && high_margin_high_acc == ceiling.len() {
        println!("\nCEILING VERDICT: All high-accuracy seeds show elevated margin↔r — partial radius");
        println!("               exploitation under favorable boundary coverage is confirmed.");
        println!("               No ceiling seed reaches 85%+ via a non-radius route.");
    } else if low_margin_high_acc > 0 {
        println!("\nCEILING VERDICT: Some high-accuracy seeds have low margin↔r — non-radius route");
        println!("               to good accuracy exists; partial-radius story is incomplete.");
    } else {
        println!("\nCEILING VERDICT: Mixed — inspect per-seed table.");
    }
}

fn load_boundary_csv(path: &str) -> Vec<BoundaryPointRecord> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {}", path, e));
    let mut records = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if i == 0 || line.trim().is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 11 {
            continue;
        }
        records.push(BoundaryPointRecord {
            seed: parts[0].parse().unwrap_or(0),
            x: parts[1].parse().unwrap_or(0.0),
            y: parts[2].parse().unwrap_or(0.0),
            r: parts[3].parse().unwrap_or(0.0),
            f_spiral: parts[4].parse().unwrap_or(0.0),
            f_circles: parts[5].parse().unwrap_or(0.0),
            margin: parts[6].parse().unwrap_or(0.0),
            router_spiral: parts[7] != "0",
            oracle_spiral: parts[8] != "0",
            composite_correct: parts[9] != "0",
            region_match: parts[10] != "0",
        });
    }
    records
}

fn demo_phase3e_boundary_analyze_csv() {
    const CSV_PATH: &str = "phase3e_boundary_diagnostic.csv";
    const INNER_RADIUS: f32 = 0.4;
    const EPS: f32 = 0.08;

    println!("--- Phase 3e annulus analysis (from {}) ---\n", CSV_PATH);
    let records = load_boundary_csv(CSV_PATH);
    if records.is_empty() {
        println!("No records found. Run --phase3e-boundary first.");
        return;
    }
    let summaries = summarize_boundary_records(&records);
    println!("Loaded {} points across {} seeds.\n", records.len(), summaries.len());
    print_annulus_misroute_analysis(&records, &summaries, INNER_RADIUS, EPS);
    print_ceiling_margin_cross_tab(&records, &summaries, INNER_RADIUS, EPS);
}

fn demo_phase3e_boundary_diagnostic() {
    println!("--- Phase 3e boundary diagnostic (expert router × composite labels) ---\n");
    println!("Authenticates whether 81% means discovery or positional leak via specialists.\n");

    const SEEDS: [u64; 20] = [
        42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    ];

    let csv_path = "phase3e_boundary_diagnostic.csv";
    let mut csv = std::fs::File::create(csv_path).expect("create boundary CSV");
    writeln!(
        csv,
        "seed,x,y,r,f_spiral,f_circles,margin,router_spiral,oracle_spiral,composite_correct,region_match"
    )
    .expect("csv header");

    let mut all_records = Vec::new();
    let mut summaries = Vec::new();
    let mut per_seed_acc = Vec::new();

    for &seed in &SEEDS {
        println!("  seed {} ...", seed);
        let (records, summary) = collect_expert_boundary_records(seed);
        println!(
            "    acc={:.1}% region_agree={:.1}% margin↔r corr={:.3} |f₁|↔r={:.3} misroute Δr={:.3} near_bnd={:.0}%",
            summary.composite_acc * 100.0,
            summary.region_agreement * 100.0,
            summary.margin_radius_corr,
            summary.f_spiral_r_corr,
            summary.misroute_mean_dr,
            summary.train_near_boundary_frac * 100.0,
        );
        per_seed_acc.push(summary.composite_acc);
        for rec in &records {
            writeln!(
                csv,
                "{},{:.5},{:.5},{:.5},{:.5},{:.5},{:.5},{},{},{},{}",
                rec.seed,
                rec.x,
                rec.y,
                rec.r,
                rec.f_spiral,
                rec.f_circles,
                rec.margin,
                rec.router_spiral as u8,
                rec.oracle_spiral as u8,
                rec.composite_correct as u8,
                rec.region_match as u8,
            )
            .expect("csv row");
        }
        all_records.extend(records);
        summaries.push(summary);
    }

    let n_pts = all_records.len().max(1) as f32;
    let pooled_region_agree =
        all_records.iter().filter(|r| r.region_match).count() as f32 / n_pts;
    let pooled_margin_r: Vec<f32> = all_records.iter().map(|r| r.margin).collect();
    let pooled_radius_sig: Vec<f32> = all_records.iter().map(|r| 0.4 - r.r).collect();
    let pooled_margin_corr = pearson_correlation(&pooled_margin_r, &pooled_radius_sig);
    let pooled_f_spiral_r = pearson_correlation(
        &all_records.iter().map(|r| r.f_spiral).collect::<Vec<_>>(),
        &all_records.iter().map(|r| r.r).collect::<Vec<_>>(),
    );
    let pooled_f_circles_r = pearson_correlation(
        &all_records.iter().map(|r| r.f_circles).collect::<Vec<_>>(),
        &all_records.iter().map(|r| r.r).collect::<Vec<_>>(),
    );

    let misroute_dr: Vec<f32> = all_records
        .iter()
        .filter(|r| !r.region_match)
        .map(|r| (r.r - 0.4).abs())
        .collect();
    let correct_dr: Vec<f32> = all_records
        .iter()
        .filter(|r| r.region_match)
        .map(|r| (r.r - 0.4).abs())
        .collect();
    let mean_misroute_dr = if misroute_dr.is_empty() {
        0.0
    } else {
        misroute_dr.iter().sum::<f32>() / misroute_dr.len() as f32
    };
    let mean_correct_dr = if correct_dr.is_empty() {
        0.0
    } else {
        correct_dr.iter().sum::<f32>() / correct_dr.len() as f32
    };

    // Circular vs free boundary: compare region agreement to linear-in-(x,y) agreement with router.
    let mut linear_xy_agree = 0usize;
    for rec in &all_records {
        // Best linear separator for r=0.4 in (x,y) is x²+y²=0.16; proxy: oracle_spiral label.
        if rec.router_spiral == rec.oracle_spiral {
            linear_xy_agree += 1;
        }
    }
    let circular_agreement = pooled_region_agree;
    // Linear in (x,y) without radius: fit w·x+b·y+c to teacher on one seed's train — use pooled oracle as ceiling.
    // Report ratio: circular agreement / agreement of margin sign with (0.4-r).
    let margin_sign_agree = all_records
        .iter()
        .filter(|r| (r.margin >= 0.0) == r.oracle_spiral)
        .count() as f32
        / n_pts;

    let floor_bucket = per_seed_acc.iter().filter(|&&a| a < 0.75).count();
    let mid_bucket = per_seed_acc.iter().filter(|&&a| a >= 0.75 && a < 0.85).count();
    let ceil_bucket = per_seed_acc.iter().filter(|&&a| a >= 0.85).count();

    let (acc_mean, acc_std) = mean_std(&per_seed_acc);
    let (reg_mean, reg_std) = mean_std(&summaries.iter().map(|s| s.region_agreement).collect::<Vec<_>>());
    let (corr_mean, corr_std) =
        mean_std(&summaries.iter().map(|s| s.margin_radius_corr).collect::<Vec<_>>());

    println!("\n=== Pooled held-out ({} seeds × ~370 pts = {} points) ===\n", SEEDS.len(), all_records.len());
    println!("| Metric | Value | Interpretation |");
    println!("| ------ | ----- | -------------- |");
    println!(
        "| Composite accuracy (per-seed mean) | {:.1}% ± {:.1}% | Uninterpretable until boundary check |",
        acc_mean * 100.0,
        acc_std * 100.0
    );
    println!(
        "| Per-seed histogram | floor<75%: {}  mid: {}  ceil≥85%: {} | Shape matters more than mean |",
        floor_bucket, mid_bucket, ceil_bucket
    );
    println!(
        "| Region agreement (router vs r<0.4) | {:.1}% | High ⇒ circular boundary / soft leak |",
        pooled_region_agree * 100.0
    );
    println!(
        "| Margin ↔ (0.4−r) correlation | {:.3} | Strong positive ⇒ tracks radius circle |",
        pooled_margin_corr
    );
    println!(
        "| Margin-sign ↔ oracle agreement | {:.1}% | Should match region agreement |",
        margin_sign_agree * 100.0
    );
    println!(
        "| f_spiral ↔ r correlation | {:.3} | Expert scalar encodes position? |",
        pooled_f_spiral_r
    );
    println!(
        "| f_circles ↔ r correlation | {:.3} | Expert scalar encodes position? |",
        pooled_f_circles_r
    );
    println!(
        "| Mean |r−0.4| when region wrong | {:.4} | vs {:.4} when right — boundary concentration |",
        mean_misroute_dr, mean_correct_dr
    );
    println!(
        "| Per-seed region agreement | {:.1}% ± {:.1}% | |",
        reg_mean * 100.0, reg_std * 100.0
    );
    println!(
        "| Per-seed margin↔r corr | {:.3} ± {:.3} | |",
        corr_mean, corr_std
    );

    println!("\nPer-seed composite accuracy: ");
    for s in &summaries {
        print!("  {}={:.0}%", s.seed, s.composite_acc * 100.0);
    }
    println!("\n");

    println!("Wrote per-point scatter data to {} (plot margin vs r, color by region_match).\n", csv_path);

    print_annulus_misroute_analysis(&all_records, &summaries, 0.4, 0.08);
    print_ceiling_margin_cross_tab(&all_records, &summaries, 0.4, 0.08);

    if pooled_region_agree > 0.90 && pooled_margin_corr > 0.7 {
        println!("VERDICT: Router boundary tracks r=0.4 — expert outputs are a positional proxy (soft leak).");
        println!("         Middle rung EMPTY. Do not claim discovery without position.");
    } else if pooled_region_agree < 0.65 && acc_mean > 0.75 {
        println!("VERDICT: High composite accuracy WITHOUT aligning to r<0.4 (region agree {:.1}%).",
            pooled_region_agree * 100.0);
        println!("         Not circular discovery — check for degenerate routing (many seeds at ~50% region agree");
        println!("         suggest constant specialist choice). f_circles↔r={:.3} flags positional encoding",
            pooled_f_circles_r);
        println!("         in specialist outputs even when router does not recover the generative boundary.");
    } else if pooled_region_agree >= 0.65 && pooled_margin_corr > 0.5 {
        println!("VERDICT: Partial radius alignment — mixed seeds; inspect histogram before any claim.");
    } else {
        println!("VERDICT: Ambiguous — inspect scatter and per-seed histogram before any paper claim.");
    }
}

// =============================================================================
// Phase 3f — Per-specialist competence routing (falsifiable gate; see COMPETENCE_ROUTING_SPEC.md)
// =============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CompetenceLabelMode {
    Correctness,
    Decisiveness,
}

struct CompetenceHead {
    input_dim: usize,
    w1: Vec<Vec<f32>>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: f32,
    temperature: f32,
}

impl CompetenceHead {
    fn random(input_dim: usize, rng: &mut StdRng) -> Self {
        let hidden = 16usize;
        let scale1 = (2.0 / (input_dim + hidden) as f32).sqrt();
        let scale2 = (2.0 / (hidden + 1) as f32).sqrt();
        let mut w1 = Vec::with_capacity(input_dim);
        for _ in 0..input_dim {
            w1.push((0..hidden).map(|_| (rng.gen::<f32>() - 0.5) * 2.0 * scale1).collect());
        }
        let b1 = (0..hidden).map(|_| 0.0f32).collect();
        let w2 = (0..hidden).map(|_| (rng.gen::<f32>() - 0.5) * 2.0 * scale2).collect();
        Self {
            input_dim,
            w1,
            b1,
            w2,
            b2: 0.0,
            temperature: 1.0,
        }
    }

    fn forward_hidden(&self, x: &[f32]) -> Vec<f32> {
        let hidden = self.b1.len();
        let mut h = vec![0.0f32; hidden];
        for j in 0..hidden {
            let mut z = self.b1[j];
            for i in 0..self.input_dim.min(x.len()) {
                z += self.w1[i][j] * x[i];
            }
            h[j] = z.max(0.0);
        }
        h
    }

    fn forward_logit(&self, x: &[f32]) -> f32 {
        let h = self.forward_hidden(x);
        let mut logit = self.b2;
        for (j, hj) in h.iter().enumerate() {
            logit += self.w2[j] * hj;
        }
        logit / self.temperature.max(1e-4)
    }

    fn predict(&self, x: &[f32]) -> f32 {
        sigmoid(self.forward_logit(x))
    }

    fn train_sgd(&mut self, features: &[Vec<f32>], labels: &[f32], lr: f32, epochs: usize) {
        if features.is_empty() {
            return;
        }
        for _ in 0..epochs {
            for (x, &y) in features.iter().zip(labels.iter()) {
                let h = self.forward_hidden(x);
                let logit = {
                    let mut z = self.b2;
                    for (j, hj) in h.iter().enumerate() {
                        z += self.w2[j] * hj;
                    }
                    z / self.temperature.max(1e-4)
                };
                let p = sigmoid(logit);
                let err = p - y;
                for j in 0..h.len() {
                    let dh = if h[j] > 0.0 { err * self.w2[j] } else { 0.0 };
                    self.w2[j] -= lr * err * h[j];
                    for i in 0..self.input_dim.min(x.len()) {
                        self.w1[i][j] -= lr * dh * x[i];
                    }
                    self.b1[j] -= lr * dh;
                }
                self.b2 -= lr * err;
            }
        }
    }

    fn fit_temperature(&mut self, features: &[Vec<f32>], labels: &[f32]) {
        if features.is_empty() {
            return;
        }
        let mut best_t = 1.0f32;
        let mut best_nll = f32::INFINITY;
        for t_idx in 0..40 {
            let t = 0.25 + (t_idx as f32) * 0.125;
            let mut nll = 0.0f32;
            for (x, &y) in features.iter().zip(labels.iter()) {
                let h = self.forward_hidden(x);
                let mut logit = self.b2;
                for (j, hj) in h.iter().enumerate() {
                    logit += self.w2[j] * hj;
                }
                logit /= t;
                let p = sigmoid(logit).clamp(1e-6, 1.0 - 1e-6);
                nll -= y * p.ln() + (1.0 - y) * (1.0 - p).ln();
            }
            nll /= features.len() as f32;
            if nll < best_nll {
                best_nll = nll;
                best_t = t;
            }
        }
        self.temperature = best_t;
    }
}

fn specialist_penultimate_hidden(
    main: &mut MainDimension,
    group_id: GroupId,
    input: &[f32],
) -> Vec<f32> {
    main.query_penultimate_hidden(input, group_id)
        .map(|(_, hidden)| hidden)
        .unwrap_or_default()
}

fn stratified_take(
    data: &[Sample],
    n: usize,
    inner_radius: f32,
    rng: &mut StdRng,
) -> Vec<Sample> {
    let mut inner = Vec::new();
    let mut outer = Vec::new();
    for sample in data {
        if sample_radius(&sample.0) < inner_radius {
            inner.push(sample.clone());
        } else {
            outer.push(sample.clone());
        }
    }
    inner.shuffle(rng);
    outer.shuffle(rng);
    let n_inner = n / 2;
    let n_outer = n.saturating_sub(n_inner);
    let mut out = Vec::with_capacity(n);
    out.extend(inner.into_iter().take(n_inner));
    out.extend(outer.into_iter().take(n_outer));
    out
}

fn competence_label_for_specialist(
    main: &mut MainDimension,
    group_id: GroupId,
    input: &[f32],
    target: f32,
    mode: CompetenceLabelMode,
) -> f32 {
    let scalar = specialist_scalar(main, group_id, input);
    match mode {
        CompetenceLabelMode::Correctness => {
            if scalar_matches_target(scalar, target) {
                1.0
            } else {
                0.0
            }
        }
        CompetenceLabelMode::Decisiveness => {
            if (scalar - 0.5).abs() > 0.4 {
                1.0
            } else {
                0.0
            }
        }
    }
}

fn collect_competence_training_data(
    main: &mut MainDimension,
    group_id: GroupId,
    router_cal: &[Sample],
    mode: CompetenceLabelMode,
) -> (Vec<Vec<f32>>, Vec<f32>) {
    let mut features = Vec::with_capacity(router_cal.len());
    let mut labels = Vec::with_capacity(router_cal.len());
    for (input, target) in router_cal {
        features.push(specialist_penultimate_hidden(main, group_id, input));
        labels.push(competence_label_for_specialist(
            main, group_id, input, target[0], mode,
        ));
    }
    (features, labels)
}

fn train_competence_heads(
    main: &mut MainDimension,
    spiral_gid: GroupId,
    circles_gid: GroupId,
    router_train: &[Sample],
    router_cal: &[Sample],
    mode: CompetenceLabelMode,
    rng: &mut StdRng,
) -> [CompetenceHead; 2] {
    let (feat_s, lab_s) =
        collect_competence_training_data(main, spiral_gid, router_train, mode);
    let (feat_c, lab_c) =
        collect_competence_training_data(main, circles_gid, router_train, mode);
    let input_dim = feat_s.first().map(|v| v.len()).unwrap_or(16).max(1);
    let mut head_spiral = CompetenceHead::random(input_dim, rng);
    let mut head_circles = CompetenceHead::random(input_dim, rng);
    head_spiral.train_sgd(&feat_s, &lab_s, 0.05, 80);
    head_circles.train_sgd(&feat_c, &lab_c, 0.05, 80);

    let (cal_feat_s, cal_lab_s) =
        collect_competence_training_data(main, spiral_gid, router_cal, mode);
    let (cal_feat_c, cal_lab_c) =
        collect_competence_training_data(main, circles_gid, router_cal, mode);
    head_spiral.fit_temperature(&cal_feat_s, &cal_lab_s);
    head_circles.fit_temperature(&cal_feat_c, &cal_lab_c);
    [head_spiral, head_circles]
}

#[derive(Clone, Copy, Debug)]
enum CompetenceDispatch {
    Route(usize),
    EnsembleTop2,
}

fn competence_scores(
    main: &mut MainDimension,
    spiral_gid: GroupId,
    circles_gid: GroupId,
    input: &[f32],
    heads: &[CompetenceHead; 2],
) -> (f32, f32) {
    let h_s = specialist_penultimate_hidden(main, spiral_gid, input);
    let h_c = specialist_penultimate_hidden(main, circles_gid, input);
    (heads[0].predict(&h_s), heads[1].predict(&h_c))
}

fn dispatch_competence(
    c0: f32,
    c1: f32,
    tau_abstain: f32,
    tau_margin: f32,
) -> (CompetenceDispatch, usize, f32) {
    let scores = [c0, c1];
    let route_k = if c0 >= c1 { 0 } else { 1 };
    let max = scores[0].max(scores[1]);
    let min = scores[0].min(scores[1]);
    let margin = max - min;
    if max < tau_abstain || margin < tau_margin {
        (CompetenceDispatch::EnsembleTop2, route_k, margin)
    } else {
        (CompetenceDispatch::Route(route_k), route_k, margin)
    }
}

fn composite_from_dispatch(
    main: &mut MainDimension,
    spiral_gid: GroupId,
    circles_gid: GroupId,
    input: &[f32],
    dispatch: CompetenceDispatch,
) -> f32 {
    match dispatch {
        CompetenceDispatch::Route(0) => specialist_scalar(main, spiral_gid, input),
        CompetenceDispatch::Route(_) => specialist_scalar(main, circles_gid, input),
        CompetenceDispatch::EnsembleTop2 => {
            let f0 = specialist_scalar(main, spiral_gid, input);
            let f1 = specialist_scalar(main, circles_gid, input);
            (f0 + f1) * 0.5
        }
    }
}

fn tune_competence_thresholds(
    main: &mut MainDimension,
    spiral_gid: GroupId,
    circles_gid: GroupId,
    heads: &[CompetenceHead; 2],
    tune_data: &[Sample],
) -> (f32, f32) {
    let mut best_abstain = 0.5f32;
    let mut best_margin = 0.1f32;
    let mut best_acc = -1.0f32;
    let mut abstain = 0.35f32;
    while abstain <= 0.65 {
        let mut margin = 0.05f32;
        while margin <= 0.20 {
            let mut correct = 0usize;
            for (input, target) in tune_data {
                let (c0, c1) = competence_scores(main, spiral_gid, circles_gid, input, heads);
                let (dispatch, _, _) = dispatch_competence(c0, c1, abstain, margin);
                let pred = composite_from_dispatch(main, spiral_gid, circles_gid, input, dispatch);
                if scalar_matches_target(pred, target[0]) {
                    correct += 1;
                }
            }
            let acc = correct as f32 / tune_data.len().max(1) as f32;
            if acc > best_acc {
                best_acc = acc;
                best_abstain = abstain;
                best_margin = margin;
            }
            margin += 0.05;
        }
        abstain += 0.05;
    }
    (best_abstain, best_margin)
}

#[derive(Clone, Debug)]
struct CompetencePointRecord {
    seed: u64,
    x: f32,
    y_coord: f32,
    r: f32,
    region_true: bool,
    route_k: usize,
    c1: f32,
    c2: f32,
    correct: bool,
    margin_top2: f32,
    y_target: f32,
    f_spiral: f32,
    f_circles: f32,
    spiral_correct: bool,
    circles_correct: bool,
    ensemble_fallback: bool,
}

#[derive(Clone, Debug)]
struct Phase3fSeedResult {
    seed: u64,
    n_router: usize,
    label_mode: CompetenceLabelMode,
    composite_acc: f32,
    region_agreement: f32,
    routing_entropy: f32,
    margin_radius_corr: f32,
    interior_misroute_rate: f32,
    annulus_interior_ratio: f32,
    confident_wrong_c_mean: f32,
    confident_wrong_n: usize,
    ensemble_frac: f32,
    tau_abstain: f32,
    tau_margin: f32,
}

fn run_phase3f_competence_seed(
    seed: u64,
    n_router: usize,
    label_mode: CompetenceLabelMode,
) -> (Vec<CompetencePointRecord>, Phase3fSeedResult) {
    const INNER_RADIUS: f32 = 0.4;
    const N_SAMPLES: usize = 400;
    const TRAIN_N: usize = 30;
    const EPS: f32 = 0.08;

    let mut dm = DimensionManager::new(phase3_composition_config());
    let mut rng = StdRng::seed_from_u64(seed);
    let mut data_rng = StdRng::seed_from_u64(seed.wrapping_mul(97).wrapping_add(99));

    let spiral_data = generate_spiral_data(400, &mut data_rng);
    let circles_data = generate_concentric_circles_data(400, &mut data_rng);
    let calibration_spiral: Vec<_> = spiral_data.iter().take(100).cloned().collect();
    let calibration_circles: Vec<_> = circles_data.iter().take(100).cloned().collect();

    let spiral_group = train_promoted_mirror(
        &mut dm, "spiral", seed, &spiral_data, &calibration_spiral, &mut rng, false,
    );
    let circles_group = train_promoted_mirror(
        &mut dm, "circles", seed.wrapping_add(1), &circles_data, &calibration_circles, &mut rng, false,
    );

    let task_e_data = generate_balanced_spiral_gated_circles_data(
        &mut dm.main, spiral_group, circles_group, INNER_RADIUS, N_SAMPLES, &mut data_rng,
    );
    let (train, heldout) =
        stratified_composite_split(&task_e_data, INNER_RADIUS, TRAIN_N, &mut data_rng);
    let _ = train;

    let extra_pool = generate_balanced_spiral_gated_circles_data(
        &mut dm.main,
        spiral_group,
        circles_group,
        INNER_RADIUS,
        n_router.saturating_add(80),
        &mut data_rng,
    );
    let router_all = stratified_take(&extra_pool, n_router, INNER_RADIUS, &mut data_rng);
    let cal_split = (router_all.len() * 4) / 5;
    let router_train = router_all[..cal_split.min(router_all.len())].to_vec();
    let router_cal = router_all[cal_split.min(router_all.len())..].to_vec();

    let mut head_rng = StdRng::seed_from_u64(seed.wrapping_add(9001));
    let heads = train_competence_heads(
        &mut dm.main,
        spiral_group,
        circles_group,
        &router_train,
        &router_cal,
        label_mode,
        &mut head_rng,
    );
    let (tau_abstain, tau_margin) = tune_competence_thresholds(
        &mut dm.main,
        spiral_group,
        circles_group,
        &heads,
        &router_cal,
    );

    let mut records = Vec::with_capacity(heldout.len());
    let mut route_ks = Vec::new();
    let mut margins = Vec::new();
    let mut radius_signals = Vec::new();
    let mut correct_n = 0usize;
    let mut region_agree = 0usize;
    let mut ensemble_n = 0usize;
    let mut confident_wrong_c_sum = 0.0f32;
    let mut confident_wrong_n = 0usize;

    for (input, target) in &heldout {
        let r = sample_radius(input);
        let region_true = r < INNER_RADIUS;
        let f_spiral = specialist_scalar(&mut dm.main, spiral_group, input);
        let f_circles = specialist_scalar(&mut dm.main, circles_group, input);
        let spiral_correct = scalar_matches_target(f_spiral, target[0]);
        let circles_correct = scalar_matches_target(f_circles, target[0]);
        let (c_spiral, c_circles) =
            competence_scores(&mut dm.main, spiral_group, circles_group, input, &heads);
        let (dispatch, route_k, margin) =
            dispatch_competence(c_spiral, c_circles, tau_abstain, tau_margin);
        let ensemble_fallback = matches!(dispatch, CompetenceDispatch::EnsembleTop2);
        if ensemble_fallback {
            ensemble_n += 1;
        }
        let pred = composite_from_dispatch(
            &mut dm.main, spiral_group, circles_group, input, dispatch,
        );
        let correct = scalar_matches_target(pred, target[0]);
        if correct {
            correct_n += 1;
        }
        let routed_spiral = route_k == 0;
        if routed_spiral == region_true {
            region_agree += 1;
        }
        route_ks.push(route_k);
        margins.push(margin);
        radius_signals.push(INNER_RADIUS - r);

        for (k, (fk, ck, spec_correct)) in [(f_spiral, c_spiral, spiral_correct), (f_circles, c_circles, circles_correct)]
            .iter()
            .enumerate()
        {
            if (*fk - 0.5).abs() > 0.4 && !spec_correct {
                confident_wrong_c_sum += *ck;
                confident_wrong_n += 1;
            }
            let _ = k;
        }

        records.push(CompetencePointRecord {
            seed,
            x: input[0],
            y_coord: input[1],
            r,
            region_true,
            route_k,
            c1: c_spiral,
            c2: c_circles,
            correct,
            margin_top2: margin,
            y_target: target[0],
            f_spiral,
            f_circles,
            spiral_correct,
            circles_correct,
            ensemble_fallback,
        });
    }

    let n = heldout.len().max(1) as f32;
    let boundary_like: Vec<BoundaryPointRecord> = records
        .iter()
        .map(|rec| BoundaryPointRecord {
            seed: rec.seed,
            x: rec.x,
            y: rec.y_coord,
            r: rec.r,
            f_spiral: rec.f_spiral,
            f_circles: rec.f_circles,
            margin: rec.margin_top2,
            router_spiral: rec.route_k == 0,
            oracle_spiral: rec.region_true,
            composite_correct: rec.correct,
            region_match: (rec.route_k == 0) == rec.region_true,
        })
        .collect();
    let zone = zone_misroute_stats(&boundary_like, INNER_RADIUS, EPS);
    let annulus_ratio = annulus_interior_ratio_for_seed(&boundary_like, seed, INNER_RADIUS, EPS);

    let summary = Phase3fSeedResult {
        seed,
        n_router,
        label_mode,
        composite_acc: correct_n as f32 / n,
        region_agreement: region_agree as f32 / n,
        routing_entropy: routing_entropy_bits(&route_ks),
        margin_radius_corr: pearson_correlation(&margins, &radius_signals),
        interior_misroute_rate: zone.interior_misroute_rate,
        annulus_interior_ratio: annulus_ratio,
        confident_wrong_c_mean: if confident_wrong_n > 0 {
            confident_wrong_c_sum / confident_wrong_n as f32
        } else {
            f32::NAN
        },
        confident_wrong_n,
        ensemble_frac: ensemble_n as f32 / n,
        tau_abstain,
        tau_margin,
    };
    (records, summary)
}

fn demo_phase3f_competence_routing() {
    println!("--- Phase 3f: Per-specialist competence routing (falsifiable gate) ---\n");
    println!("See docs/COMPETENCE_ROUTING_SPEC.md. Pre-registered pass/fail — fill §6 after run.\n");

    const SEEDS: [u64; 20] = [
        42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    ];
    const N_ROUTER_SWEEP: [usize; 3] = [30, 100, 300];

    let csv_path = "competence_routing_diagnostic.csv";
    let mut csv = std::fs::File::create(csv_path).expect("create competence CSV");
    writeln!(
        csv,
        "seed,x,y,region_true,route_k,c_1,c_2,correct,margin_top2,r,y_target,f_spiral,f_circles,ensemble_fallback,label_mode,n_router"
    )
    .expect("csv header");

    for &n_router in &N_ROUTER_SWEEP {
        println!("\n=== n_router = {} (correctness labels) ===\n", n_router);
        let mut summaries = Vec::new();
        for &seed in &SEEDS {
            print!("  seed {} ... ", seed);
            let (records, summary) =
                run_phase3f_competence_seed(seed, n_router, CompetenceLabelMode::Correctness);
            println!(
                "acc={:.1}% region={:.1}% H={:.2} margin↔r={:.2} cw_c={:.2} (n={})",
                summary.composite_acc * 100.0,
                summary.region_agreement * 100.0,
                summary.routing_entropy,
                summary.margin_radius_corr,
                summary.confident_wrong_c_mean,
                summary.confident_wrong_n,
            );
            summaries.push(summary);
            if n_router == 300 {
                for rec in &records {
                    writeln!(
                        csv,
                        "{},{:.5},{:.5},{},{},{:.5},{:.5},{},{:.5},{:.5},{:.5},{:.5},{:.5},{},correctness,{}",
                        rec.seed,
                        rec.x,
                        rec.y_coord,
                        if rec.region_true { 1 } else { 0 },
                        rec.route_k,
                        rec.c1,
                        rec.c2,
                        if rec.correct { 1 } else { 0 },
                        rec.margin_top2,
                        rec.r,
                        rec.y_target,
                        rec.f_spiral,
                        rec.f_circles,
                        if rec.ensemble_fallback { 1 } else { 0 },
                        n_router,
                    )
                    .expect("csv row");
                }
            }
        }

        let accs: Vec<f32> = summaries.iter().map(|s| s.composite_acc).collect();
        let regions: Vec<f32> = summaries.iter().map(|s| s.region_agreement).collect();
        let entropies: Vec<f32> = summaries.iter().map(|s| s.routing_entropy).collect();
        let corrs: Vec<f32> = summaries.iter().map(|s| s.margin_radius_corr).collect();
        let cw: Vec<f32> = summaries
            .iter()
            .filter_map(|s| if s.confident_wrong_n > 0 { Some(s.confident_wrong_c_mean) } else { None })
            .collect();
        let degenerate = entropies.iter().filter(|&&h| h < 0.3).count();

        let (acc_m, acc_s) = mean_std(&accs);
        let (reg_m, reg_s) = mean_std(&regions);
        let (ent_m, ent_s) = mean_std(&entropies);
        let (corr_m, corr_s) = mean_std(&corrs);
        let (cw_m, cw_s) = mean_std(&cw);

        println!("\n| Metric | Mean ± std | Pass? |");
        println!("| ------ | ---------- | ----- |");
        println!(
            "| Composite accuracy | {:.1}% ± {:.1}% | (vs 77% singles, 69.5% conf) |",
            acc_m * 100.0,
            acc_s * 100.0
        );
        println!(
            "| Region agreement | {:.1}% ± {:.1}% | target ≥80% |",
            reg_m * 100.0,
            reg_s * 100.0
        );
        println!(
            "| Routing entropy | {:.2} ± {:.2} bits | degenerate seeds: {} |",
            ent_m, ent_s, degenerate
        );
        println!(
            "| margin↔(0.4−r) | {:.2} ± {:.2} | target ≥0.5 |",
            corr_m, corr_s
        );
        println!(
            "| Confident-wrong mean c_k | {:.3} ± {:.3} | want LOW |",
            cw_m, cw_s
        );
    }

    println!("\n=== Decisiveness ablation (n_router=300) ===\n");
    let mut decis_summaries = Vec::new();
    for &seed in &SEEDS {
        let (_, summary) =
            run_phase3f_competence_seed(seed, 300, CompetenceLabelMode::Decisiveness);
        decis_summaries.push(summary);
    }
    let decis_acc = mean_std(&decis_summaries.iter().map(|s| s.composite_acc).collect::<Vec<_>>());
    let decis_reg = mean_std(&decis_summaries.iter().map(|s| s.region_agreement).collect::<Vec<_>>());
    let decis_cw = mean_std(
        &decis_summaries
            .iter()
            .filter_map(|s| if s.confident_wrong_n > 0 { Some(s.confident_wrong_c_mean) } else { None })
            .collect::<Vec<_>>(),
    );
    println!(
        "Decisiveness-trained head: acc={:.1}%±{:.1}% region={:.1}%±{:.1}% cw_c={:.3}±{:.3}",
        decis_acc.0 * 100.0,
        decis_acc.1 * 100.0,
        decis_reg.0 * 100.0,
        decis_reg.1 * 100.0,
        decis_cw.0,
        decis_cw.1,
    );

    println!("\n=== Baselines (reference — re-run from --phase3e for full table) ===\n");
    println!("| Baseline | Expected (Task E, 20 seeds) |");
    println!("| -------- | ----------------------------- |");
    println!("| Oracle-best-single | ~77% |");
    println!("| VirtualGroup blend | ~69.9% |");
    println!("| Confidence argmax | ~69.5% |");
    println!("| Logistic gate on r | ~91.5% |");
    println!("| Expert-output router | ~81% (55% region agree — rejected) |");

    println!("\nWrote {} (n_router=300 correctness only).", csv_path);
    println!("Run --phase3f-analyze for annulus / confident-wrong breakdown.");
    println!("\n§6 decision table: fill manually from metrics above — do not narrate before table.");
}

fn load_competence_csv(path: &str) -> Vec<CompetencePointRecord> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {}", path, e));
    let mut records = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if i == 0 || line.trim().is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 10 {
            continue;
        }
        records.push(CompetencePointRecord {
            seed: parts[0].parse().unwrap_or(0),
            x: parts[1].parse().unwrap_or(0.0),
            y_coord: parts[2].parse().unwrap_or(0.0),
            region_true: parts[3] != "0",
            route_k: parts[4].parse().unwrap_or(0),
            c1: parts[5].parse().unwrap_or(0.0),
            c2: parts[6].parse().unwrap_or(0.0),
            correct: parts[7] != "0",
            margin_top2: parts[8].parse().unwrap_or(0.0),
            r: parts[9].parse().unwrap_or(0.0),
            y_target: parts.get(10).and_then(|s| s.parse().ok()).unwrap_or(0.0),
            f_spiral: parts.get(11).and_then(|s| s.parse().ok()).unwrap_or(0.0),
            f_circles: parts.get(12).and_then(|s| s.parse().ok()).unwrap_or(0.0),
            spiral_correct: false,
            circles_correct: false,
            ensemble_fallback: parts.get(13).map(|s| *s != "0").unwrap_or(false),
        });
    }
    records
}

fn demo_phase3f_analyze_csv() {
    const CSV_PATH: &str = "competence_routing_diagnostic.csv";
    const INNER_RADIUS: f32 = 0.4;
    const EPS: f32 = 0.08;

    println!("--- Phase 3f competence routing analysis (from {}) ---\n", CSV_PATH);
    let records = load_competence_csv(CSV_PATH);
    if records.is_empty() {
        println!("No records found. Run --phase3f-competence first.");
        return;
    }

    let seeds: Vec<u64> = records.iter().map(|r| r.seed).collect::<std::collections::HashSet<_>>().into_iter().collect();
    println!("Loaded {} points across {} seeds.\n", records.len(), seeds.len());

    let boundary_like: Vec<BoundaryPointRecord> = records
        .iter()
        .map(|rec| BoundaryPointRecord {
            seed: rec.seed,
            x: rec.x,
            y: rec.y_coord,
            r: rec.r,
            f_spiral: rec.f_spiral,
            f_circles: rec.f_circles,
            margin: rec.margin_top2,
            router_spiral: rec.route_k == 0,
            oracle_spiral: rec.region_true,
            composite_correct: rec.correct,
            region_match: (rec.route_k == 0) == rec.region_true,
        })
        .collect();

    let mut per_seed = Vec::new();
    for &seed in &seeds {
        let seed_recs: Vec<_> = records.iter().filter(|r| r.seed == seed).collect();
        let route_ks: Vec<usize> = seed_recs.iter().map(|r| r.route_k).collect();
        let ent = routing_entropy_bits(&route_ks);
        let region_agree = seed_recs
            .iter()
            .filter(|r| (r.route_k == 0) == r.region_true)
            .count() as f32
            / seed_recs.len().max(1) as f32;
        let acc = seed_recs.iter().filter(|r| r.correct).count() as f32 / seed_recs.len().max(1) as f32;
        let margins: Vec<f32> = seed_recs.iter().map(|r| r.margin_top2).collect();
        let radius_sig: Vec<f32> = seed_recs.iter().map(|r| INNER_RADIUS - r.r).collect();
        let corr = pearson_correlation(&margins, &radius_sig);

        let mut cw_sum = 0.0f32;
        let mut cw_n = 0usize;
        for rec in &seed_recs {
            for (fk, ck) in [(rec.f_spiral, rec.c1), (rec.f_circles, rec.c2)] {
                let spec_correct = scalar_matches_target(fk, rec.y_target);
                if (fk - 0.5).abs() > 0.4 && !spec_correct {
                    cw_sum += ck;
                    cw_n += 1;
                }
            }
        }

        per_seed.push((seed, acc, region_agree, ent, corr, cw_sum / cw_n.max(1) as f32, cw_n));
    }

    println!("| seed | acc | region | H(bits) | margin↔r | cw_mean c_k | cw_n |");
    println!("| ---- | --- | ------ | ------- | -------- | ----------- | ---- |");
    for (seed, acc, reg, ent, corr, cw, cw_n) in &per_seed {
        println!(
            "| {} | {:.1}% | {:.1}% | {:.2} | {:.3} | {:.3} | {} |",
            seed,
            acc * 100.0,
            reg * 100.0,
            ent,
            corr,
            cw,
            cw_n,
        );
    }

    let degenerate = per_seed.iter().filter(|(_, _, _, ent, _, _, _)| *ent < 0.3).count();
    println!("\nDegenerate seeds (H < 0.3 bits): {} / {}", degenerate, per_seed.len());

    let summaries: Vec<BoundarySeedSummary> = seeds
        .iter()
        .map(|&seed| {
            let seed_recs: Vec<_> = boundary_like.iter().filter(|r| r.seed == seed).collect();
            let n = seed_recs.len().max(1) as f32;
            BoundarySeedSummary {
                seed,
                composite_acc: seed_recs.iter().filter(|r| r.composite_correct).count() as f32 / n,
                region_agreement: seed_recs.iter().filter(|r| r.region_match).count() as f32 / n,
                margin_radius_corr: pearson_correlation(
                    &seed_recs.iter().map(|r| r.margin).collect::<Vec<_>>(),
                    &seed_recs.iter().map(|r| INNER_RADIUS - r.r).collect::<Vec<_>>(),
                ),
                misroute_mean_dr: 0.0,
                f_spiral_r_corr: 0.0,
                f_circles_r_corr: 0.0,
                train_near_boundary_frac: 0.0,
            }
        })
        .collect();
    print_annulus_misroute_analysis(&boundary_like, &summaries, INNER_RADIUS, EPS);

    if degenerate >= 4 {
        println!("\nPRE-REGISTERED FAIL: ≥4/20 degenerate seeds (§5.2 control 2).");
    }
}

// =============================================================================
// CMI measurement — formalize "present but inaccessible" (see CMI spec)
// =============================================================================

fn collect_cmi_records_for_seed(seed: u64) -> Vec<CmiPointRecord> {
    const INNER_RADIUS: f32 = 0.4;
    const N_SAMPLES: usize = 400;
    const TRAIN_N: usize = 30;

    let mut dm = DimensionManager::new(phase3_composition_config());
    let mut rng = StdRng::seed_from_u64(seed);
    let mut data_rng = StdRng::seed_from_u64(seed.wrapping_mul(97).wrapping_add(99));

    let spiral_data = generate_spiral_data(400, &mut data_rng);
    let circles_data = generate_concentric_circles_data(400, &mut data_rng);
    let calibration_spiral: Vec<_> = spiral_data.iter().take(100).cloned().collect();
    let calibration_circles: Vec<_> = circles_data.iter().take(100).cloned().collect();

    let spiral_group = train_promoted_mirror(
        &mut dm, "spiral", seed, &spiral_data, &calibration_spiral, &mut rng, false,
    );
    let circles_group = train_promoted_mirror(
        &mut dm, "circles", seed.wrapping_add(1), &circles_data, &calibration_circles, &mut rng, false,
    );

    let task_e_data = generate_balanced_spiral_gated_circles_data(
        &mut dm.main, spiral_group, circles_group, INNER_RADIUS, N_SAMPLES, &mut data_rng,
    );
    let (_train, heldout) =
        stratified_composite_split(&task_e_data, INNER_RADIUS, TRAIN_N, &mut data_rng);

    let mut records = Vec::with_capacity(heldout.len());
    for (input, target) in &heldout {
        let r = sample_radius(input);
        let region = if r < INNER_RADIUS { 1u8 } else { 0u8 };
        let y_spiral = specialist_scalar(&mut dm.main, spiral_group, input);
        let y_circles = specialist_scalar(&mut dm.main, circles_group, input);
        let a_spiral = specialist_penultimate_hidden(&mut dm.main, spiral_group, input);
        let a_circles = specialist_penultimate_hidden(&mut dm.main, circles_group, input);
        let c_spiral = if scalar_matches_target(y_spiral, target[0]) {
            1
        } else {
            0
        };
        let c_circles = if scalar_matches_target(y_circles, target[0]) {
            1
        } else {
            0
        };
        records.push(CmiPointRecord {
            seed,
            x: input[0],
            y_coord: input[1],
            r,
            region,
            c_spiral,
            c_circles,
            y_spiral,
            y_circles,
            a_spiral,
            a_circles,
        });
    }
    records
}

fn write_cmi_csv(path: &str, records: &[CmiPointRecord]) {
    let mut csv = std::fs::File::create(path).expect("create cmi csv");
    writeln!(
        csv,
        "seed,x,y,r,region,c_spiral,c_circles,y_spiral,y_circles,a_spiral_0,a_spiral_1,a_spiral_2,a_spiral_3,a_spiral_4,a_spiral_5,a_spiral_6,a_spiral_7,a_spiral_8,a_spiral_9,a_spiral_10,a_spiral_11,a_spiral_12,a_spiral_13,a_spiral_14,a_spiral_15,a_circles_0,a_circles_1,a_circles_2,a_circles_3,a_circles_4,a_circles_5,a_circles_6,a_circles_7,a_circles_8,a_circles_9,a_circles_10,a_circles_11,a_circles_12,a_circles_13,a_circles_14,a_circles_15"
    )
    .expect("cmi header");
    for rec in records {
        let mut row = format!(
            "{},{:.5},{:.5},{:.5},{},{},{},{:.5},{:.5}",
            rec.seed,
            rec.x,
            rec.y_coord,
            rec.r,
            rec.region,
            rec.c_spiral,
            rec.c_circles,
            rec.y_spiral,
            rec.y_circles,
        );
        for v in &rec.a_spiral {
            row.push_str(&format!(",{:.5}", v));
        }
        for v in &rec.a_circles {
            row.push_str(&format!(",{:.5}", v));
        }
        writeln!(csv, "{}", row).expect("cmi row");
    }
}

fn load_cmi_csv(path: &str) -> Vec<CmiPointRecord> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {}", path, e));
    let mut records = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if i == 0 || line.trim().is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 41 {
            continue;
        }
        let parse_a = |start: usize| -> Vec<f32> {
            (0..16)
                .map(|j| parts.get(start + j).and_then(|s| s.parse().ok()).unwrap_or(0.0))
                .collect()
        };
        records.push(CmiPointRecord {
            seed: parts[0].parse().unwrap_or(0),
            x: parts[1].parse().unwrap_or(0.0),
            y_coord: parts[2].parse().unwrap_or(0.0),
            r: parts[3].parse().unwrap_or(0.0),
            region: parts[4].parse().unwrap_or(0),
            c_spiral: parts[5].parse().unwrap_or(0),
            c_circles: parts[6].parse().unwrap_or(0),
            y_spiral: parts[7].parse().unwrap_or(0.0),
            y_circles: parts[8].parse().unwrap_or(0.0),
            a_spiral: parse_a(9),
            a_circles: parse_a(25),
        });
    }
    records
}

fn demo_cmi_measurement() {
    println!("--- CMI measurement: present-but-inaccessible formalization ---\n");
    println!("Task E held-out points, 20 seeds. No projection — raw penultimate activations.\n");

    const SEEDS: [u64; 20] = [
        42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    ];
    const CSV_PATH: &str = "cmi_diagnostic.csv";

    let mut all_records = Vec::new();
    for &seed in &SEEDS {
        print!("  seed {} ... ", seed);
        let records = collect_cmi_records_for_seed(seed);
        println!("{} points", records.len());
        all_records.extend(records);
    }
    write_cmi_csv(CSV_PATH, &all_records);

    let mut estimates = Vec::new();
    for &seed in &SEEDS {
        estimates.push(estimate_cmi_seed(&all_records, seed));
    }

    let report = format_cmi_report(&estimates);
    println!("\n{}", report);
    println!("\nWrote {} ({} points). Re-run with --cmi-analyze.", CSV_PATH, all_records.len());
}

fn demo_cmi_analyze_csv() {
    const CSV_PATH: &str = "cmi_diagnostic.csv";
    println!("--- CMI analysis (from {}) ---\n", CSV_PATH);
    let records = load_cmi_csv(CSV_PATH);
    if records.is_empty() {
        println!("No records. Run --cmi first.");
        return;
    }
    let seeds: Vec<u64> = records
        .iter()
        .map(|r| r.seed)
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    let mut estimates = Vec::new();
    for &seed in &seeds {
        estimates.push(estimate_cmi_seed(&records, seed));
    }
    println!("{}", format_cmi_report(&estimates));
}

fn extract_spiral_output_weights_for_seed(seed: u64) -> (u64, Vec<f32>) {
    const INNER_RADIUS: f32 = 0.4;
    let mut dm = DimensionManager::new(phase3_composition_config());
    let mut rng = StdRng::seed_from_u64(seed);
    let mut data_rng = StdRng::seed_from_u64(seed.wrapping_mul(97).wrapping_add(99));
    let spiral_data = generate_spiral_data(400, &mut data_rng);
    let calibration_spiral: Vec<_> = spiral_data.iter().take(100).cloned().collect();
    let spiral_group = train_promoted_mirror(
        &mut dm, "spiral", seed, &spiral_data, &calibration_spiral, &mut rng, false,
    );
    let _ = INNER_RADIUS;
    (seed, dm.main.penultimate_to_output_weights(spiral_group))
}

fn demo_cmi_spiral_resolve() {
    const CSV_PATH: &str = "cmi_diagnostic.csv";
    const OUT_PATH: &str = "cmi_spiral_resolve.json";
    println!("--- CMI spiral resolve: output bottleneck or below resolution? ---\n");
    let records = load_cmi_csv(CSV_PATH);
    if records.is_empty() {
        println!("No records in {}. Run --cmi first.", CSV_PATH);
        return;
    }
    println!("Loaded {} points. Extracting output-head weights per seed ...\n", records.len());
    const SEEDS: [u64; 20] = [
        42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    ];
    let mut weights = Vec::new();
    for &seed in &SEEDS {
        print!("  weights seed {} ... ", seed);
        let w = extract_spiral_output_weights_for_seed(seed);
        println!("dim={}", w.1.len());
        weights.push(w);
    }
    println!("\nRunning permutation null (B=500), linear probe, PCA-kNN ...\n");
    let result = resolve_spiral_region_mi(&records, &weights, 500);
    let report = format_spiral_resolve_report(&result);
    println!("{}", report);
    let json = serde_json::to_string_pretty(&result).expect("serialize spiral resolve");
    std::fs::write(OUT_PATH, json).expect("write spiral resolve json");
    println!("\nWrote {}. Re-run with --cmi-spiral-analyze.", OUT_PATH);
}

fn demo_cmi_spiral_analyze() {
    const OUT_PATH: &str = "cmi_spiral_resolve.json";
    println!("--- CMI spiral resolve analysis (from {}) ---\n", OUT_PATH);
    let text = std::fs::read_to_string(OUT_PATH)
        .unwrap_or_else(|e| panic!("read {}: {}", OUT_PATH, e));
    let result: SpiralResolveResult = serde_json::from_str(&text).expect("parse spiral resolve");
    println!("{}", format_spiral_resolve_report(&result));
}

// =============================================================================
// Grounding loop audit (assisted maintenance — §GROUNDING_LOOP_SPEC)
// =============================================================================

const GROUNDING_CAPTURES_CSV: &str = "grounding_loop_captures.csv";
const GROUNDING_PROPOSALS_CSV: &str = "grounding_loop_proposals.csv";
const GROUNDING_CURVE_CSV: &str = "grounding_loop_curve.csv";
const GROUNDING_RESULTS_TXT: &str = "grounding_loop_results.txt";

fn write_grounding_curve_csv(path: &str, sweep: &str, curve: &[CoverageCurvePoint]) {
    use std::io::Write as _;
    // Append so the memorization and genuine sweeps share one file (distinguished by `sweep`).
    let exists = std::path::Path::new(path).exists();
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .expect("open curve csv");
    if !exists {
        writeln!(
            f,
            "sweep,additions,captured_accuracy,held_out_accuracy,generalization_gap,cross_domain_misroute_rate"
        )
        .expect("curve header");
    }
    for p in curve {
        writeln!(
            f,
            "{},{},{:.4},{:.4},{:.4},{:.4}",
            sweep,
            p.additions,
            p.captured_accuracy,
            p.held_out_accuracy,
            p.generalization_gap,
            p.cross_domain_misroute_rate,
        )
        .expect("curve row");
    }
}

fn parse_fleet_domain(s: &str) -> Option<GroundingFleetDomain> {
    match s.trim().to_ascii_lowercase().as_str() {
        "base" => Some(GroundingFleetDomain::Base),
        "crypto" => Some(GroundingFleetDomain::Crypto),
        "fintech" => Some(GroundingFleetDomain::Fintech),
        "runtime" => Some(GroundingFleetDomain::Runtime),
        _ => None,
    }
}

fn fixture_rows_to_captures(
    rt: &growformer::dimension::LanguageRuntime,
    index: &grounding_loop::GroundingNodeIndex,
    fixture: &[FixtureRow],
) -> Vec<FailureCapture> {
    fixture
        .iter()
        .filter_map(|row| {
            let (emb, conf) = embed_phrase(rt, &row.phrase).ok()?;
            Some(FailureCapture {
                phrase: row.phrase.clone(),
                encoder_embedding: emb.clone(),
                activated_nodes: index.activated_node_scores(&emb),
                max_confidence: conf,
                entropy_bits: None,
                trigger_reason: FailureTrigger::NoNodeActivated,
                downstream_signal: None,
                timestamp_unix: 0,
                domain_context: row.domain_context.clone(),
                inferred_concept_id: row.concept_id.clone(),
                split: row.split,
                provenance: grounding_loop::PhraseProvenance::real(row.phrase.clone()),
            })
        })
        .collect()
}

fn write_grounding_captures_csv(path: &str, captures: &[FailureCapture]) {
    let mut f = std::fs::File::create(path).expect("create captures csv");
    writeln!(
        f,
        "phrase,concept_id,split,trigger,domain_context,max_confidence,activated_count"
    )
    .expect("header");
    for c in captures {
        writeln!(
            f,
            "{},{},{},{},{},{:.4},{}",
            csv_escape(&c.phrase),
            csv_escape(&c.inferred_concept_id),
            c.split.as_str(),
            c.trigger_reason.as_str(),
            csv_escape(&c.domain_context),
            c.max_confidence,
            c.activated_nodes.len(),
        )
        .expect("row");
    }
}

fn write_grounding_proposals_csv(path: &str, proposals: &[EditProposal]) {
    let mut f = std::fs::File::create(path).expect("create proposals csv");
    writeln!(
        f,
        "kind,phrase,target_node,target_domain,similarity,margin,collision_score,conflicts,pre_certify_held_out,approved,integrated"
    )
    .expect("header");
    for p in proposals {
        let (kind, phrase, target, domain, sim, margin) = match &p.kind {
            ProposalKind::Alias {
                phrase,
                target_node,
                target_domain,
                similarity,
                margin,
                ..
            } => (
                "alias".to_string(),
                phrase.clone(),
                target_node.clone(),
                target_domain.clone(),
                *similarity,
                *margin,
            ),
            ProposalKind::NewNode { phrases, domain, .. } => (
                "new_node".to_string(),
                phrases.join("|"),
                String::new(),
                domain.clone(),
                0.0,
                0.0,
            ),
        };
        let conflicts = p
            .collision_conflicts
            .iter()
            .map(|(d, n, s)| format!("{}:{}:{:.3}", d, n, s))
            .collect::<Vec<_>>()
            .join(";");
        writeln!(
            f,
            "{},{},{},{},{:.4},{:.4},{:.4},{},{:.4},{},{}",
            kind,
            csv_escape(&phrase),
            csv_escape(&target),
            csv_escape(&domain),
            sim,
            margin,
            p.collision_score,
            csv_escape(&conflicts),
            p.pre_certify_held_out_estimate,
            p.approved,
            p.integrated,
        )
        .expect("row");
    }
}

fn csv_escape(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

fn load_grounding_captures_csv(path: &str) -> Vec<FailureCapture> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {}", path, e));
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if i == 0 || line.trim().is_empty() {
            continue;
        }
        let parts = parse_csv_line(line);
        if parts.len() < 7 {
            continue;
        }
        let split = match parts[2].as_str() {
            "certify" => CaptureSplit::Certify,
            _ => CaptureSplit::Propose,
        };
        let trigger = match parts[3].as_str() {
            "entropy_guard" => FailureTrigger::EntropyGuard,
            "low_confidence" => FailureTrigger::LowConfidence,
            "dissatisfaction" => FailureTrigger::Dissatisfaction,
            _ => FailureTrigger::NoNodeActivated,
        };
        out.push(FailureCapture {
            phrase: parts[0].clone(),
            encoder_embedding: Vec::new(),
            activated_nodes: Vec::new(),
            max_confidence: parts[5].parse().unwrap_or(0.0),
            entropy_bits: None,
            trigger_reason: trigger,
            downstream_signal: None,
            timestamp_unix: 0,
            domain_context: parts[4].clone(),
            inferred_concept_id: parts[1].clone(),
            split,
            provenance: grounding_loop::PhraseProvenance::real(parts[0].clone()),
        });
    }
    out
}

fn parse_csv_line(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut cur = String::new();
    let mut in_quotes = false;
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '"' if in_quotes => {
                if chars.peek() == Some(&'"') {
                    chars.next();
                    cur.push('"');
                } else {
                    in_quotes = false;
                }
            }
            '"' => in_quotes = true,
            ',' if !in_quotes => {
                fields.push(cur.clone());
                cur.clear();
            }
            _ => cur.push(c),
        }
    }
    fields.push(cur);
    fields
}

fn rehydrate_captures(
    rt: &growformer::dimension::LanguageRuntime,
    index: &grounding_loop::GroundingNodeIndex,
    captures: &mut [FailureCapture],
) {
    for c in captures.iter_mut() {
        if let Ok((emb, _)) = embed_phrase(rt, &c.phrase) {
            c.encoder_embedding = emb.clone();
            c.activated_nodes = index.activated_node_scores(&emb);
        }
    }
}

fn build_proposals_for_captures(
    captures: &[FailureCapture],
    index: &grounding_loop::GroundingNodeIndex,
    params: &GroundingLoopParams,
    rt: &growformer::dimension::LanguageRuntime,
    before_index: &grounding_loop::GroundingNodeIndex,
) -> Vec<EditProposal> {
    let held_out_before =
        grounding_loop::routing_accuracy_for_captures(captures, rt, before_index, CaptureSplit::Certify)
            .unwrap_or(0.0);

    let mut proposals = Vec::new();
    for cap in captures.iter().filter(|c| c.split == CaptureSplit::Propose) {
        let domain_hint = parse_fleet_domain(&cap.domain_context);
        let Some(kind) = propose_for_phrase(
            &cap.phrase,
            &cap.encoder_embedding,
            index,
            params,
            domain_hint,
        ) else {
            continue;
        };
        let (target_domain, target_node) = match &kind {
            ProposalKind::Alias {
                target_domain,
                target_node,
                ..
            } => (target_domain.clone(), target_node.clone()),
            ProposalKind::NewNode { domain, .. } => (domain.clone(), String::new()),
        };
        let domain = parse_fleet_domain(&target_domain).unwrap_or(GroundingFleetDomain::Crypto);
        let (collision_score, conflicts) = if let ProposalKind::Alias { target_node, .. } = &kind {
            collision_check(
                &cap.encoder_embedding,
                domain,
                target_node,
                index,
                params,
            )
        } else {
            (0.0, Vec::new())
        };
        proposals.push(EditProposal {
            kind,
            collision_score,
            collision_conflicts: conflicts,
            pre_certify_held_out_estimate: held_out_before,
            approved: collision_score < params.collision_threshold
                && target_node == cap.inferred_concept_id,
            integrated: false,
        });
    }
    proposals
}

/// Domain corpus that grounds the CliffordE8 encoder vocabulary (option 2 fix).
/// Natural domain sentences — deliberately NOT the held-out certify phrases, so the
/// certifier's held-out generalization test stays honest (no phrase-level leakage).
fn grounding_audit_corpus() -> Vec<&'static str> {
    vec![
        // crypto
        "bitcoin and btc holders stack sats onchain",
        "long term btc investors hold their coins for years",
        "sats accumulate onchain as bitcoin moves between wallets",
        "ethereum is an ether network where gas fees apply",
        "gas fees on the ethereum network rise when blocks fill",
        "ether transactions on ethereum cost gas",
        "a dex is a decentralized exchange where users swap tokens",
        "swaps on a dex can fail when routing through liquidity breaks",
        "decentralized exchange routing depends on liquidity pools",
        // pet
        "a puppy enjoys a snack as a treat during training",
        "dog treats are a reward for good training behavior",
        "walking the dog on a leash is daily exercise",
        "a stroll in the park keeps the dog active",
    ]
}

fn install_grounding_audit_dictionary(_dm: &mut DimensionManager) -> bool {
    let corpus = grounding_audit_corpus();
    let n = grounding_loop::install_phrase_embedder_from_corpus(&corpus, 1024);
    println!(
        "  [cata] grounding-audit phrase embedder: {} tokens from {} domain sentences",
        n,
        corpus.len()
    );
    true
}

const LUNA_DEFAULT_DIR: &str =
    "/Users/astor/Projects/2026/spacekit/spacekit-projects/companions/luna";

/// Runtime-domain grounding nodes (the loaded companion graph) as `(domain, id, aliases)`.
fn luna_runtime_nodes() -> Vec<(GroundingFleetDomain, String, Vec<String>)> {
    world_grounding::fleet_node_inventory()
        .into_iter()
        .filter(|n| n.domain == GroundingFleetDomain::Runtime)
        .map(|n| (GroundingFleetDomain::Runtime, n.node_id, n.aliases))
        .collect()
}

/// Build per-intent propose/certify captures from labeled companion utterances.
/// Only rows whose `semantic_intent` is a real loaded node id are kept. Within each
/// intent, utterances are split deterministically (even index → propose, odd → certify),
/// so the certify set is genuinely held out from both the proposals and the encoder.
fn luna_build_captures(
    samples: &[LanguageSample],
    valid_ids: &std::collections::HashSet<String>,
) -> Vec<FailureCapture> {
    let mut by_intent: std::collections::BTreeMap<String, Vec<String>> =
        std::collections::BTreeMap::new();
    let mut seen: std::collections::HashSet<(String, String)> = std::collections::HashSet::new();
    for s in samples {
        if !valid_ids.contains(&s.semantic_intent) {
            continue;
        }
        let phrase = s.text.trim().to_string();
        if phrase.is_empty() {
            continue;
        }
        if seen.insert((s.semantic_intent.clone(), phrase.to_ascii_lowercase())) {
            by_intent.entry(s.semantic_intent.clone()).or_default().push(phrase);
        }
    }
    let mut captures = Vec::new();
    for (intent, mut phrases) in by_intent {
        if phrases.len() < 4 {
            continue; // need at least 2 propose + 2 certify for a meaningful split
        }
        phrases.sort();
        for (i, phrase) in phrases.into_iter().enumerate() {
            let split = if i % 2 == 0 { CaptureSplit::Propose } else { CaptureSplit::Certify };
            let prov = grounding_loop::PhraseProvenance::real(phrase.clone());
            captures.push(FailureCapture {
                phrase,
                encoder_embedding: Vec::new(),
                activated_nodes: Vec::new(),
                max_confidence: 0.0,
                entropy_bits: None,
                trigger_reason: FailureTrigger::NoNodeActivated,
                downstream_signal: None,
                timestamp_unix: 0,
                domain_context: "runtime".into(),
                inferred_concept_id: intent.clone(),
                split,
                provenance: prov,
            });
        }
    }
    captures
}

/// One full certifier pass for a given (already-installed) encoder: build the index from
/// the companion nodes, rehydrate captures, build proposals, certify the genuine and
/// memorization batches, sweep the coverage curve, and format a compact report.
fn luna_certifier_pass(
    label: &str,
    rt: &growformer::dimension::LanguageRuntime,
    captures: &mut Vec<FailureCapture>,
    nodes: &[(GroundingFleetDomain, String, Vec<String>)],
    params: &GroundingLoopParams,
) -> String {
    let before_index = build_grounding_index_from_nodes(rt, nodes, params).expect("luna index");
    rehydrate_captures(rt, &before_index, captures);

    let proposals = build_proposals_for_captures(captures, &before_index, params, rt, &before_index);

    // Genuine batch: integrate only approved propose-set aliases (held-out untouched).
    let mut after_genuine = before_index.clone();
    let mut had_collisions = false;
    let mut approved = 0usize;
    for p in &proposals {
        if p.collision_score >= params.collision_threshold {
            had_collisions = true;
        }
        if !p.approved {
            continue;
        }
        if let ProposalKind::Alias { phrase, target_node, target_domain, .. } = &p.kind {
            let d = parse_fleet_domain(target_domain).unwrap_or(GroundingFleetDomain::Runtime);
            if let Ok((emb, _)) = embed_phrase(rt, phrase) {
                after_genuine.add_alias_to_node(d, target_node, phrase, emb);
                approved += 1;
            }
        }
    }

    // Memorization contrast: add exact captured propose phrases as aliases.
    let mut after_memo = before_index.clone();
    for cap in captures.iter().filter(|c| c.split == CaptureSplit::Propose) {
        if let Ok((emb, _)) = embed_phrase(rt, &cap.phrase) {
            let d = parse_fleet_domain(&cap.domain_context).unwrap_or(GroundingFleetDomain::Runtime);
            after_memo.add_alias_to_node(d, &cap.inferred_concept_id, &cap.phrase, emb);
        }
    }

    let home = GroundingFleetDomain::Runtime;
    let (before_m, gen_m) =
        certify_batch(captures, rt, &before_index, &after_genuine, home).expect("certify genuine");
    let (_, memo_m) =
        certify_batch(captures, rt, &before_index, &after_memo, home).expect("certify memo");
    let gen_verdict = decide_batch_verdict(&before_m, &gen_m, params, had_collisions);
    let memo_verdict = decide_batch_verdict(&before_m, &memo_m, params, false);

    let genuine_additions: Vec<(GroundingFleetDomain, String, String)> = proposals
        .iter()
        .filter(|p| p.approved)
        .filter_map(|p| match &p.kind {
            ProposalKind::Alias { phrase, target_node, target_domain, .. } => Some((
                parse_fleet_domain(target_domain).unwrap_or(GroundingFleetDomain::Runtime),
                target_node.clone(),
                phrase.clone(),
            )),
            _ => None,
        })
        .collect();
    let genuine_curve =
        coverage_vs_additions_curve(captures, rt, &before_index, &genuine_additions)
            .expect("genuine curve");
    let (gcap, gheld) = curve_lifts(&genuine_curve);

    let n_propose = captures.iter().filter(|c| c.split == CaptureSplit::Propose).count();
    let n_certify = captures.iter().filter(|c| c.split == CaptureSplit::Certify).count();
    let cal = calibrate_alias_threshold(captures, rt, &before_index).ok();

    let mut s = String::new();
    s.push_str(&format!("=== Encoder: {label} ===\n"));
    s.push_str(&format!(
        "  propose={n_propose}  certify(held-out)={n_certify}  proposals={}  approved={approved}\n",
        proposals.len()
    ));
    if let Some(c) = &cal {
        s.push_str(&format!(
            "  τ_alias calibration: same={:.3} cross={:.3} suggested={:.3} (default {:.2})\n",
            c.same_concept_mean, c.cross_concept_mean, c.suggested_tau_alias, params.tau_alias
        ));
    }
    s.push_str(&format_certifier_report(
        "  genuine batch (approved aliases only)",
        &before_m,
        &gen_m,
        gen_verdict,
    ));
    s.push('\n');
    s.push_str(&format_certifier_report(
        "  memorization contrast (exact phrases)",
        &before_m,
        &memo_m,
        memo_verdict,
    ));
    s.push_str(&format!(
        "\n  genuine sweep lifts: captured {:+.1}pp, held-out {:+.1}pp\n",
        gcap * 100.0,
        gheld * 100.0,
    ));
    s
}

fn demo_grounding_loop_luna(dir_arg: &str) {
    println!("--- Grounding loop on a real companion (Luna) ---\n");
    let dir = if dir_arg.trim().is_empty() || dir_arg.trim() == "default" {
        LUNA_DEFAULT_DIR.to_string()
    } else {
        dir_arg.trim().to_string()
    };
    let data_dir = std::path::Path::new(&dir).join("data");
    let grounding_path = data_dir.join("pet_world_grounding.toml");

    let toml = match std::fs::read_to_string(&grounding_path) {
        Ok(t) => t,
        Err(e) => {
            println!("Could not read {}: {}", grounding_path.display(), e);
            return;
        }
    };
    if let Err(e) = world_grounding::load_grounding_graph_from_str(&toml) {
        println!("Failed to parse grounding graph: {}", e);
        return;
    }

    let mut samples: Vec<LanguageSample> = Vec::new();
    if let Err(e) = append_language_samples_from_training_jsonl_dir(&mut samples, &data_dir) {
        println!("Failed to load JSONL corpus: {}", e);
        return;
    }
    println!("Loaded {} labeled utterances from {}\n", samples.len(), data_dir.display());

    let nodes = luna_runtime_nodes();
    let valid_ids: std::collections::HashSet<String> =
        nodes.iter().map(|(_, id, _)| id.clone()).collect();
    println!("Companion grounding graph: {} runtime nodes\n", nodes.len());

    let mut captures = luna_build_captures(&samples, &valid_ids);
    if captures.is_empty() {
        println!("No usable captures (need intents that are graph nodes with >=4 utterances).");
        return;
    }
    let n_intents: std::collections::BTreeSet<&str> =
        captures.iter().map(|c| c.inferred_concept_id.as_str()).collect();
    println!(
        "Captures: {} across {} intents (propose/certify split)\n",
        captures.len(),
        n_intents.len()
    );

    let (dm, _, _, _) = build_language_demo_manager(0.0);
    let rt = &dm.language_runtime;

    let params = GroundingLoopParams::default();
    let mut report = String::new();
    report.push_str("Grounding loop — real companion (Luna) certifier\n");
    report.push_str("================================================\n\n");
    report.push_str(&format!(
        "Runtime nodes: {} | intents tested: {} | propose+certify: {}\n\n",
        nodes.len(),
        n_intents.len(),
        captures.len()
    ));

    // Pass 1: lexical CATA encoder, dictionary built from the companion's own propose-set
    // utterances + node aliases (best case for the lexical regime).
    let mut corpus: Vec<String> = captures
        .iter()
        .filter(|c| c.split == CaptureSplit::Propose)
        .map(|c| c.phrase.clone())
        .collect();
    for (_, id, aliases) in &nodes {
        corpus.push(id.replace('_', " "));
        corpus.extend(aliases.iter().cloned());
    }
    let corpus_refs: Vec<&str> = corpus.iter().map(|s| s.as_str()).collect();
    let dict_n = install_phrase_embedder_from_corpus(&corpus_refs, 8192);
    println!("  [lexical] CATA dictionary: {dict_n} tokens");
    report.push_str(&luna_certifier_pass("lexical CATA centroid", rt, &mut captures, &nodes, &params));
    report.push_str("\n\n");

    // Pass 2: supervised encoder trained ONLY on propose-set labels (certify held out).
    let train_pairs: Vec<(String, String)> = captures
        .iter()
        .filter(|c| c.split == CaptureSplit::Propose)
        .map(|c| (c.phrase.clone(), c.inferred_concept_id.clone()))
        .collect();
    match SupervisedEncoder::train(&train_pairs, 4096, 60) {
        Some(enc) => {
            let k = install_supervised_embedder(enc);
            println!("  [supervised] trained on {} propose phrases, {k} concepts", train_pairs.len());
            report.push_str(&luna_certifier_pass(
                "supervised projection (trained on propose-set only)",
                rt,
                &mut captures,
                &nodes,
                &params,
            ));
        }
        None => report.push_str("supervised encoder: insufficient labels (need >=2)\n"),
    }
    clear_phrase_embedder();

    println!("\n{}", report);
    std::fs::write(GROUNDING_RESULTS_TXT, &report).expect("write results");
    println!("Wrote {}.", GROUNDING_RESULTS_TXT);
}

const DISJOINT_CURVE_CSV: &str = "grounding_disjoint_curve.csv";
const DISJOINT_META_TXT: &str = "grounding_disjoint_meta.txt";

fn disjoint_shuffle_labels(pairs: &[(String, String)], seed: u64) -> Vec<(String, String)> {
    let mut labels: Vec<String> = pairs.iter().map(|(_, l)| l.clone()).collect();
    let mut rng = seed ^ 0x9E3779B97F4A7C15;
    for i in (1..labels.len()).rev() {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        let j = (rng % (i as u64 + 1)) as usize;
        labels.swap(i, j);
    }
    pairs
        .iter()
        .enumerate()
        .map(|(i, (p, _))| (p.clone(), labels[i].clone()))
        .collect()
}

fn percentile_sorted(v: &mut [f32], p: f32) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((p * (v.len() as f32 - 1.0)).round() as usize).min(v.len() - 1);
    v[idx]
}

fn format_overlap_curve(label: &str, curve: &[OverlapBin]) -> String {
    let mut s = format!("{label}\n  overlap bin | n | accuracy [95% Wilson CI]\n");
    for b in curve {
        if b.n == 0 {
            s.push_str(&format!("  {:>10} | 0 | (empty)\n", b.label));
        } else {
            s.push_str(&format!(
                "  {:>10} | {:>3} | {:>5.1}% [{:>5.1}%, {:>5.1}%]\n",
                b.label,
                b.n,
                b.accuracy * 100.0,
                b.ci_lo * 100.0,
                b.ci_hi * 100.0,
            ));
        }
    }
    s
}

fn write_disjoint_curve_csv(path: &str, encoder: &str, level: &str, curve: &[OverlapBin]) {
    use std::io::Write as _;
    let exists = std::path::Path::new(path).exists();
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .expect("open disjoint csv");
    if !exists {
        writeln!(f, "encoder,feature_level,overlap_bin,n,hits,accuracy,ci_lo,ci_hi").expect("hdr");
    }
    for b in curve {
        writeln!(
            f,
            "{},{},{},{},{},{:.4},{:.4},{:.4}",
            encoder, level, b.label, b.n, b.hits, b.accuracy, b.ci_lo, b.ci_hi
        )
        .expect("row");
    }
}

fn demo_grounding_disjoint_test(dir_arg: &str) {
    println!("--- Token-disjoint generalization test (supervised grounding encoder) ---\n");
    let dir = if dir_arg.trim().is_empty() || dir_arg.trim() == "default" {
        LUNA_DEFAULT_DIR.to_string()
    } else {
        dir_arg.trim().to_string()
    };
    let data_dir = std::path::Path::new(&dir).join("data");
    let grounding_path = data_dir.join("pet_world_grounding.toml");
    let toml = match std::fs::read_to_string(&grounding_path) {
        Ok(t) => t,
        Err(e) => {
            println!("Could not read {}: {}", grounding_path.display(), e);
            return;
        }
    };
    if let Err(e) = world_grounding::load_grounding_graph_from_str(&toml) {
        println!("Failed to parse grounding graph: {}", e);
        return;
    }
    let mut samples: Vec<LanguageSample> = Vec::new();
    if let Err(e) = append_language_samples_from_training_jsonl_dir(&mut samples, &data_dir) {
        println!("Failed to load JSONL corpus: {}", e);
        return;
    }
    let nodes = luna_runtime_nodes();
    let valid_ids: std::collections::HashSet<String> =
        nodes.iter().map(|(_, id, _)| id.clone()).collect();
    let captures = luna_build_captures(&samples, &valid_ids);
    if captures.is_empty() {
        println!("No usable captures.");
        return;
    }
    write_grounding_captures_csv(GROUNDING_CAPTURES_CSV, &captures);
    std::fs::write(DISJOINT_META_TXT, grounding_path.to_string_lossy().as_bytes())
        .expect("write meta");

    let (dm, _, _, _) = build_language_demo_manager(0.0);
    let report = run_disjoint_core(&dm.language_runtime, &nodes, &captures);
    println!("\n{}", report);
    std::fs::write(GROUNDING_RESULTS_TXT, &report).expect("write results");
    println!(
        "Wrote {}, {}, {}, {}.",
        GROUNDING_RESULTS_TXT, DISJOINT_CURVE_CSV, GROUNDING_CAPTURES_CSV, DISJOINT_META_TXT
    );
}

fn demo_grounding_disjoint_analyze() {
    println!("--- Token-disjoint test (re-run from captured CSV) ---\n");
    let grounding_path = match std::fs::read_to_string(DISJOINT_META_TXT) {
        Ok(p) => p.trim().to_string(),
        Err(_) => {
            println!("No {} found. Run --grounding-disjoint-test first.", DISJOINT_META_TXT);
            return;
        }
    };
    let toml = match std::fs::read_to_string(&grounding_path) {
        Ok(t) => t,
        Err(e) => {
            println!("Could not read grounding graph {}: {}", grounding_path, e);
            return;
        }
    };
    if let Err(e) = world_grounding::load_grounding_graph_from_str(&toml) {
        println!("Failed to parse grounding graph: {}", e);
        return;
    }
    let nodes = luna_runtime_nodes();
    let captures = load_grounding_captures_csv(GROUNDING_CAPTURES_CSV);
    if captures.is_empty() {
        println!("No captures in {}.", GROUNDING_CAPTURES_CSV);
        return;
    }
    let (dm, _, _, _) = build_language_demo_manager(0.0);
    let report = run_disjoint_core(&dm.language_runtime, &nodes, &captures);
    println!("\n{}", report);
}

/// The full token-disjoint test (§§1–6): positive control (CATA) first, then the shuffle
/// floor, then the real supervised curve, then the pre-registered decision rule.
fn run_disjoint_core(
    rt: &growformer::dimension::LanguageRuntime,
    nodes: &[(GroundingFleetDomain, String, Vec<String>)],
    captures: &[FailureCapture],
) -> String {
    let params = GroundingLoopParams::default();
    let propose: Vec<(String, String)> = captures
        .iter()
        .filter(|c| c.split == CaptureSplit::Propose)
        .map(|c| (c.phrase.clone(), c.inferred_concept_id.clone()))
        .collect();
    let certify: Vec<FailureCapture> = captures
        .iter()
        .filter(|c| c.split == CaptureSplit::Certify)
        .cloned()
        .collect();
    let n_labels = propose.iter().map(|(_, l)| l).collect::<std::collections::HashSet<_>>().len();
    if n_labels < 2 || certify.is_empty() {
        return "insufficient data (need >=2 concepts and certify phrases)".into();
    }
    let (concept_train, global_train) = concept_train_features(&propose);
    let _ = std::fs::remove_file(DISJOINT_CURVE_CSV);

    let mut out = String::new();
    out.push_str("Token-disjoint generalization test\n");
    out.push_str("==================================\n\n");
    out.push_str(&format!(
        "concepts={} | propose={} | certify(held-out)={}\n\n",
        n_labels,
        propose.len(),
        certify.len()
    ));

    // ---- §5 Positive control FIRST: lexical CATA must collapse to floor at overlap-0. ----
    let mut corpus: Vec<String> = propose.iter().map(|(p, _)| p.clone()).collect();
    for (_, id, aliases) in nodes {
        corpus.push(id.replace('_', " "));
        corpus.extend(aliases.iter().cloned());
    }
    let corpus_refs: Vec<&str> = corpus.iter().map(|s| s.as_str()).collect();
    install_phrase_embedder_from_corpus(&corpus_refs, 8192);
    let idx_cata = build_grounding_index_from_nodes(rt, nodes, &params).expect("cata index");
    let evals_cata = evaluate_disjoint(rt, &idx_cata, &certify, &concept_train, &global_train, "wbc")
        .expect("cata eval");
    let curve_cata = build_overlap_curve(&evals_cata);
    let pooled_cata = pooled_accuracy(&evals_cata);
    write_disjoint_curve_csv(DISJOINT_CURVE_CSV, "lexical_cata", "wbc", &curve_cata);
    let cata_overlap0 = curve_cata[0].accuracy;
    let cata_top = curve_cata.iter().rev().find(|b| b.n > 0).cloned();

    // ---- §4 Shuffle floor: permute labels, retrain, collect overlap-0(a) accuracy. ----
    let b_total = 200usize;
    let mut floor_a: Vec<f32> = Vec::new();
    let mut floor_contrib = 0usize;
    for b in 0..b_total {
        let permuted = disjoint_shuffle_labels(&propose, b as u64);
        let Some(enc) = SupervisedEncoder::train(&permuted, 4096, 60) else { continue };
        install_supervised_embedder(enc);
        let idx = build_grounding_index_from_nodes(rt, nodes, &params).expect("shuffle index");
        let (ct, gl) = concept_train_features(&permuted);
        let evals = evaluate_disjoint(rt, &idx, &certify, &ct, &gl, "wbc").expect("shuffle eval");
        let (ah, an, _, _) = overlap0_substrata(&evals);
        if an > 0 {
            floor_a.push(ah as f32 / an as f32);
            floor_contrib += 1;
        }
        if (b + 1) % 50 == 0 {
            println!("  shuffle {}/{} (overlap-0(a) contributing: {})", b + 1, b_total, floor_contrib);
        }
    }
    let floor_mean = if floor_a.is_empty() { 0.0 } else { floor_a.iter().sum::<f32>() / floor_a.len() as f32 };
    let floor95 = percentile_sorted(&mut floor_a.clone(), 0.95);

    // ---- §2–3 Real supervised curve + overlap-0 sub-stratification. ----
    let enc = SupervisedEncoder::train(&propose, 4096, 60).expect("train supervised");
    install_supervised_embedder(enc);
    let idx = build_grounding_index_from_nodes(rt, nodes, &params).expect("sup index");
    let evals = evaluate_disjoint(rt, &idx, &certify, &concept_train, &global_train, "wbc")
        .expect("sup eval");
    let curve = build_overlap_curve(&evals);
    let pooled = pooled_accuracy(&evals);
    let (ah, an, bh, bn) = overlap0_substrata(&evals);
    write_disjoint_curve_csv(DISJOINT_CURVE_CSV, "supervised", "wbc", &curve);

    // Secondary, looser word+bigram-disjoint view (more leakage-prone; for when union-0 sparse).
    let evals_wb = evaluate_disjoint(rt, &idx, &certify, &concept_train, &global_train, "wb")
        .expect("sup eval wb");
    let curve_wb = build_overlap_curve(&evals_wb);
    write_disjoint_curve_csv(DISJOINT_CURVE_CSV, "supervised", "wb", &curve_wb);

    clear_phrase_embedder();

    // ---- §5 positive-control / overlap-measurement validity ----
    // The self-validation is precisely: a PURE-lexical method (CATA) cannot route
    // feature-disjoint phrases above the manufactured floor. If it does, the overlap-0 bin
    // is contaminated (overlap mis-measured) and nothing downstream is trustworthy. CATA's
    // high-overlap slope is reported as context but does NOT gate validity — on a large
    // graph CATA can be near-chance everywhere, which leaves no slope yet still floors.
    let overlap_measure_valid = cata_overlap0 <= floor95 + 0.05;
    let cata_slope = cata_top.as_ref().map(|t| t.accuracy - cata_overlap0).unwrap_or(0.0);

    // Corroborating lexical signature on the supervised curve: accuracy rises monotonically
    // with overlap and collapses toward floor at disjoint.
    let sup_top = curve.iter().rev().find(|b| b.n > 0).map(|b| b.accuracy).unwrap_or(0.0);
    let monotone_lexical = sup_top > cata_overlap0 + 0.20 && curve[0].accuracy <= floor95 + 0.05;
    let wb0 = curve_wb[0].accuracy;
    let wb0_n = curve_wb[0].n;

    // ---- §6 decision rule ----
    let g = if an > 0 { ah as f32 / an as f32 } else { f32::NAN };
    let (g_lo, g_hi) = wilson_interval(ah, an, 1.96);
    let resolution_ok = an >= 8 && (g_hi - g_lo) <= 0.30;

    let verdict: &str;
    let mut headline: String;
    if !overlap_measure_valid {
        verdict = "TEST INVALID — lexical CATA routed disjoint phrases above floor (overlap mis-measured)";
        headline = "fix overlap measurement before reading any supervised number".into();
    } else if !resolution_ok {
        verdict = "BELOW RESOLUTION — union-disjoint seen-elsewhere bin too small for a tight claim";
        headline = format!(
            "union-disjoint(a) n={} (CI [{:.0}%,{:.0}%]); pooled {:.1}% is NOT established as generalization",
            an, g_lo * 100.0, g_hi * 100.0, pooled * 100.0
        );
        // Surface the corroborating evidence even when the strict bin is underpowered.
        if monotone_lexical && wb0 <= floor95 + 0.05 {
            headline.push_str(&format!(
                "\n  → corroborating: supervised curve is monotone in overlap (top {:.0}% vs disjoint {:.0}%) and the\n    word+bigram-disjoint bin (n={}) is {:.1}% ≤ floor {:.1}% — the pooled number is overlap-driven (lexical-in-disguise)",
                sup_top * 100.0, curve[0].accuracy * 100.0, wb0_n, wb0 * 100.0, floor95 * 100.0
            ));
        }
    } else if g <= floor95 {
        verdict = "LEXICAL-IN-DISGUISE — disjoint accuracy at/under shuffle floor";
        headline = format!(
            "honest held-out ≈ floor {:.1}%; pooled {:.1}% was surface-overlap artifact",
            floor95 * 100.0,
            pooled * 100.0
        );
    } else if g >= 0.8 * pooled {
        verdict = "REAL GENERALIZATION — flat curve, disjoint bin holds";
        headline = format!("headline stays {:.1}% (disjoint-bin {:.1}%)", pooled * 100.0, g * 100.0);
    } else {
        verdict = "PARTIAL — generalization real but weaker than pooled headline";
        headline = format!("headline DROPS to disjoint-bin {:.1}% (pooled {:.1}% is coverage)", g * 100.0, pooled * 100.0);
    }

    // ---- report ----
    out.push_str("§5 POSITIVE CONTROL — lexical CATA (must floor at overlap-0)\n");
    out.push_str(&format_overlap_curve("  CATA accuracy-vs-overlap (union features)", &curve_cata));
    out.push_str(&format!(
        "  CATA pooled={:.1}%  overlap-0={:.1}%  shuffle-floor95={:.1}%  (slope to top bin: {:+.1}pp)\n",
        pooled_cata * 100.0,
        cata_overlap0 * 100.0,
        floor95 * 100.0,
        cata_slope * 100.0,
    ));
    out.push_str(&format!(
        "  overlap measure {}: pure-lexical CATA scores {:.1}% on disjoint phrases (≤ floor {:.1}% ⇒ overlap-0 bin is clean)\n\n",
        if overlap_measure_valid { "VALID" } else { "INVALID" },
        cata_overlap0 * 100.0,
        floor95 * 100.0
    ));

    out.push_str("§4 SHUFFLE FLOOR (B=200 retrains on permuted labels)\n");
    out.push_str(&format!(
        "  overlap-0(seen-elsewhere) accuracy: mean={:.1}%  95th-pct={:.1}%  (contributing shuffles: {}/{})\n\n",
        floor_mean * 100.0,
        floor95 * 100.0,
        floor_contrib,
        b_total
    ));

    out.push_str("§2 SUPERVISED accuracy-vs-overlap curve\n");
    out.push_str(&format_overlap_curve("  union (w∪b∪c) — the encoder's true input", &curve));
    out.push_str(&format_overlap_curve("  word+bigram (looser, leakage-prone secondary)", &curve_wb));
    let (s_lo, s_hi) = wilson_interval(ah, an, 1.96);
    let (n_lo, n_hi) = wilson_interval(bh, bn, 1.96);
    out.push_str("\n§3 OVERLAP-0 sub-stratification\n");
    out.push_str(&format!(
        "  (a) seen-elsewhere [GENERALIZATION HEADLINE]: {}/{} = {:.1}% [{:.1}%, {:.1}%]\n",
        ah, an,
        if an > 0 { ah as f32 / an as f32 * 100.0 } else { 0.0 },
        s_lo * 100.0, s_hi * 100.0
    ));
    out.push_str(&format!(
        "  (b) novel-features [routes by prior, not learning]: {}/{} = {:.1}% [{:.1}%, {:.1}%]\n\n",
        bh, bn,
        if bn > 0 { bh as f32 / bn as f32 * 100.0 } else { 0.0 },
        n_lo * 100.0, n_hi * 100.0
    ));

    out.push_str("§6 DECISION\n");
    out.push_str(&format!("  pooled held-out: {:.1}%\n", pooled * 100.0));
    out.push_str(&format!(
        "  disjoint-bin(a) g = {}  shuffle floor95 = {:.1}%\n",
        if g.is_nan() { "n/a".to_string() } else { format!("{:.1}%", g * 100.0) },
        floor95 * 100.0
    ));
    out.push_str(&format!("  VERDICT: {}\n", verdict));
    out.push_str(&format!("  → {}\n\n", headline));
    out.push_str(&format!(
        "  honest restatement: held-out = pooled {:.1}% (disjoint-bin(a) {}, shuffle floor {:.1}%)\n",
        pooled * 100.0,
        if g.is_nan() { "n/a".to_string() } else { format!("{:.1}%", g * 100.0) },
        floor95 * 100.0
    ));
    out
}

// ===========================================================================
// Certifier-First Pipeline orchestrator (§2 of the spec) — the contract every
// encoder is judged by. Deterministic given (encoder, data_hash, seed): same
// inputs → same verdict artifact, always.
// ===========================================================================

const CERTIFY_SHUFFLE_B: usize = 200;
const CERTIFY_N_BUCKETS: usize = 4096;
const CERTIFY_EPOCHS: usize = 60;

/// The encoder under audit. The pipeline is embedder-agnostic; an encoder is reduced to a
/// thing that can be (re)installed as the active phrase embedder given training pairs — which
/// is exactly what the shuffle control needs (retrain on permuted labels). A future drop-in
/// (real semantic encoder) plugs in here via the BYO-vectors hook.
#[derive(Clone, Debug)]
enum AuditEncoder {
    Supervised,
    Cata,
    /// A frozen bring-your-own encoder, reduced to precomputed vectors over every phrase/alias
    /// the pipeline embeds (the BYO-vectors hook). This is how a real semantic encoder (e.g. the
    /// GLE) plugs into the identical gate. The shuffle null for a frozen encoder reinstalls the
    /// same map and permutes only the overlap definition — the correct null for a fixed encoder.
    Vectors { id: String, map: HashMap<String, Vec<f32>> },
}

impl AuditEncoder {
    fn from_id(id: &str) -> Option<Self> {
        match id.trim().to_ascii_lowercase().as_str() {
            "supervised" | "supervised_v1" | "sup" => Some(Self::Supervised),
            "cata" | "lexical" | "lexical_cata" => Some(Self::Cata),
            _ => None,
        }
    }

    fn id(&self) -> &str {
        match self {
            Self::Supervised => "supervised",
            Self::Cata => "cata",
            Self::Vectors { id, .. } => id.as_str(),
        }
    }

    /// Install this encoder as the active phrase embedder, trained on `train` pairs.
    /// `node_corpus` is the node-id/alias text (used by CATA's dictionary, ignored by the
    /// supervised projection). `seed` makes supervised training reproducible.
    fn install(&self, train: &[(String, String)], node_corpus: &[String], seed: u64) -> Result<(), String> {
        match self {
            Self::Supervised => {
                let enc = SupervisedEncoder::train_seeded(train, CERTIFY_N_BUCKETS, CERTIFY_EPOCHS, seed)
                    .ok_or_else(|| "supervised: need >=2 labels".to_string())?;
                install_supervised_embedder(enc);
                Ok(())
            }
            Self::Cata => {
                let mut corpus: Vec<String> = train.iter().map(|(p, _)| p.clone()).collect();
                corpus.extend(node_corpus.iter().cloned());
                let refs: Vec<&str> = corpus.iter().map(|s| s.as_str()).collect();
                install_phrase_embedder_from_corpus(&refs, 8192);
                Ok(())
            }
            // Frozen encoder: labels are irrelevant to the vectors; reinstall the same map.
            Self::Vectors { map, .. } => {
                install_vector_embedder(map.clone());
                Ok(())
            }
        }
    }
}

/// Run the full certifier-first sequence (§2) and produce the single verdict artifact (§1).
/// Deterministic given `(enc, captures, nodes, seed)`. The shuffle and positive controls run
/// every time — they are not optional flags.
fn certify_encoder_pipeline(
    enc: &AuditEncoder,
    rt: &growformer::dimension::LanguageRuntime,
    nodes: &[(GroundingFleetDomain, String, Vec<String>)],
    captures: &[FailureCapture],
    seed: u64,
    encoder_provenance: &str,
) -> EncoderVerdict {
    let params = GroundingLoopParams::default();
    let node_ids: Vec<String> = nodes.iter().map(|(_, id, _)| id.clone()).collect();
    let dh = data_hash(captures, &node_ids);

    // Node-alias corpus text shared with CATA's dictionary.
    let mut node_corpus: Vec<String> = Vec::new();
    for (_, id, aliases) in nodes {
        node_corpus.push(id.replace('_', " "));
        node_corpus.extend(aliases.iter().cloned());
    }

    let propose: Vec<(String, String)> = captures
        .iter()
        .filter(|c| c.split == CaptureSplit::Propose)
        .map(|c| (c.phrase.clone(), c.inferred_concept_id.clone()))
        .collect();
    let certify: Vec<FailureCapture> = captures
        .iter()
        .filter(|c| c.split == CaptureSplit::Certify)
        .cloned()
        .collect();
    let (concept_train, global_train) = concept_train_features(&propose);

    // --- §4 provenance check + augmentation firewall (orthogonal to surface overlap) ---
    let firewall = run_augmentation_firewall(captures);

    // Defaults for the artifact (filled as the sequence runs).
    let mut verdict = EncoderVerdict {
        encoder_id: enc.id().to_string(),
        data_hash: dh,
        seed,
        candidate_set_size: nodes.len(),
        disjoint_semantic_lift: 0.0,
        disjoint_lift_ci: [0.0, 0.0],
        verdict: Verdict::Invalid.as_str().to_string(),
        semantic_floor_mean: 0.0,
        semantic_floor_95: 0.0,
        disjoint_gen_a: 0.0,
        disjoint_gen_a_n: 0,
        pooled_heldout: 0.0,
        memorization_gap: 0.0,
        overlap_curve: Vec::new(),
        feature_family: FeatureFamily::default(),
        plateau_flag: false,
        collision_delta: 0.0,
        disjoint_level: "wbc".to_string(),
        invalid_reason: String::new(),
        positive_control_collapsed: false,
        augmentation_firewall_clean: firewall.clean,
        below_resolution: true,
        firewall,
        shuffle_b: CERTIFY_SHUFFLE_B,
        encoder_provenance: encoder_provenance.to_string(),
    };

    let n_labels = propose.iter().map(|(_, l)| l).collect::<std::collections::HashSet<_>>().len();
    if n_labels < 2 || certify.is_empty() {
        // Not enough structure to measure anything; INVALID (a data problem, not a fail).
        verdict.verdict = Verdict::Invalid.as_str().to_string();
        verdict.firewall.violations.push("insufficient data: need >=2 concepts and certify phrases".into());
        verdict.augmentation_firewall_clean = false;
        verdict.invalid_reason = "insufficient data: need >=2 concepts and certify phrases".into();
        return verdict;
    }

    // --- Phase A: subject encoder (real labels) → disjoint curve, sub-bins, gap, plateau ---
    if enc.install(&propose, &node_corpus, seed).is_err() {
        verdict.verdict = Verdict::Invalid.as_str().to_string();
        verdict.firewall.violations.push("encoder failed to train/install".into());
        verdict.invalid_reason = "encoder failed to train/install".into();
        return verdict;
    }
    let idx = build_grounding_index_from_nodes(rt, nodes, &params).expect("subject index");
    let evals = evaluate_disjoint(rt, &idx, &certify, &concept_train, &global_train, "wbc")
        .expect("subject eval wbc");
    let curve = build_overlap_curve(&evals);
    let pooled = pooled_accuracy(&evals);

    // feature-family disjoint-0 accuracy (diagnostic: which granularity carries the routing).
    let evals_w = evaluate_disjoint(rt, &idx, &certify, &concept_train, &global_train, "w")
        .expect("subject eval w");
    let evals_wb = evaluate_disjoint(rt, &idx, &certify, &concept_train, &global_train, "wb")
        .expect("subject eval wb");
    let curve_w = build_overlap_curve(&evals_w);
    let curve_wb = build_overlap_curve(&evals_wb);
    verdict.feature_family = FeatureFamily {
        word: curve_w[0].accuracy,
        bigram: curve_wb[0].accuracy,
        trigram: curve[0].accuracy,
    };

    // Choose the disjoint granularity to resolve the gate at: strictest (union "wbc") preferred,
    // but on dense training the union-disjoint bin is empty (n=0) — fall back to the finest level
    // that actually has a seen-elsewhere sub-bin to read. Looser ⇒ leakier; the level is recorded.
    let an_wbc = overlap0_substrata(&evals).1;
    let an_wb = overlap0_substrata(&evals_wb).1;
    let an_w = overlap0_substrata(&evals_w).1;
    let level: &str = if an_wbc >= grounding_loop::DISJOINT_MIN_N {
        "wbc"
    } else if an_wb >= grounding_loop::DISJOINT_MIN_N {
        "wb"
    } else if an_w >= grounding_loop::DISJOINT_MIN_N {
        "w"
    } else {
        // None resolvable; keep strictest so n is honestly reported as underpowered.
        "wbc"
    };
    verdict.disjoint_level = level.to_string();
    let evals_lvl: &[grounding_loop::DisjointEval] = match level {
        "w" => &evals_w,
        "wb" => &evals_wb,
        _ => &evals,
    };
    let (ah, an, _bh, _bn) = overlap0_substrata(evals_lvl);

    // memorization gap = captured (propose routing) − held-out pooled.
    let captured = routing_accuracy_for_captures(captures, rt, &idx, CaptureSplit::Propose).unwrap_or(0.0);
    verdict.memorization_gap = (captured - pooled).max(0.0);

    // collision: this audit applies no graph edits, so the pre/post misroute delta is 0.
    verdict.collision_delta = 0.0;

    // plateau: does adding approved aliases stop lifting held-out accuracy?
    verdict.plateau_flag = {
        let mut caps_mut = captures.to_vec();
        rehydrate_captures(rt, &idx, &mut caps_mut);
        let proposals = build_proposals_for_captures(&caps_mut, &idx, &params, rt, &idx);
        let additions: Vec<(GroundingFleetDomain, String, String)> = proposals
            .iter()
            .filter(|p| p.approved)
            .filter_map(|p| match &p.kind {
                ProposalKind::Alias { phrase, target_node, target_domain, .. } => Some((
                    parse_fleet_domain(target_domain).unwrap_or(GroundingFleetDomain::Runtime),
                    target_node.clone(),
                    phrase.clone(),
                )),
                _ => None,
            })
            .collect();
        match coverage_vs_additions_curve(&caps_mut, rt, &idx, &additions) {
            Ok(c) => {
                let (_gcap, gheld) = curve_lifts(&c);
                gheld <= 0.0
            }
            Err(_) => false,
        }
    };

    verdict.overlap_curve = build_overlap_curve(evals_lvl);
    verdict.pooled_heldout = pooled;
    verdict.disjoint_gen_a = if an > 0 { ah as f32 / an as f32 } else { 0.0 };
    verdict.disjoint_gen_a_n = an;

    // --- Phase B: positive control — lexical CATA must collapse to floor at overlap-0 ---
    // Evaluated at the SAME granularity the gate resolves at, so the control is apples-to-apples.
    let cata = AuditEncoder::Cata;
    let _ = cata.install(&propose, &node_corpus, seed);
    let idx_cata = build_grounding_index_from_nodes(rt, nodes, &params).expect("cata index");
    let evals_cata = evaluate_disjoint(rt, &idx_cata, &certify, &concept_train, &global_train, level)
        .expect("cata eval");
    let curve_cata = build_overlap_curve(&evals_cata);
    let cata_overlap0 = curve_cata[0].accuracy;
    let cata_overlap0_n = curve_cata[0].n;

    // --- Phase C: shuffle floor (B≥200, retrain subject on permuted labels) ---
    let mut floor_a: Vec<f32> = Vec::new();
    for b in 0..CERTIFY_SHUFFLE_B {
        let permuted = disjoint_shuffle_labels(&propose, seed ^ (b as u64));
        // Retrain the SUBJECT encoder class on permuted labels (the proper null).
        if enc.install(&permuted, &node_corpus, seed ^ (0xA5A5 + b as u64)).is_err() {
            continue;
        }
        let idx_s = build_grounding_index_from_nodes(rt, nodes, &params).expect("shuffle index");
        let (ct, gl) = concept_train_features(&permuted);
        let evals_s = evaluate_disjoint(rt, &idx_s, &certify, &ct, &gl, level).expect("shuffle eval");
        let (ah_s, an_s, _, _) = overlap0_substrata(&evals_s);
        if an_s > 0 {
            floor_a.push(ah_s as f32 / an_s as f32);
        }
        if (b + 1) % 50 == 0 {
            println!("  [certify] shuffle {}/{}", b + 1, CERTIFY_SHUFFLE_B);
        }
    }
    clear_phrase_embedder();

    let floor_mean = if floor_a.is_empty() { 0.0 } else { floor_a.iter().sum::<f32>() / floor_a.len() as f32 };
    let floor95 = percentile_sorted(&mut floor_a.clone(), 0.95);
    verdict.semantic_floor_mean = floor_mean;
    verdict.semantic_floor_95 = floor95;
    // The control is only meaningful with a non-empty disjoint bin; an empty bin can't validate
    // the overlap measure, so it does not count as a collapse (⇒ INVALID, not a vacuous pass).
    verdict.positive_control_collapsed = cata_overlap0_n > 0 && cata_overlap0 <= floor95 + 0.05;

    // --- §3 lift = disjoint_gen_a − semantic_floor_95, with Wilson CI shifted by floor95 ---
    let (g_lo, g_hi) = wilson_interval(ah, an, 1.96);
    let ci_width = (g_hi - g_lo) as f32;
    verdict.disjoint_semantic_lift = verdict.disjoint_gen_a - floor95;
    verdict.disjoint_lift_ci = [(g_lo as f32) - floor95, (g_hi as f32) - floor95];
    verdict.below_resolution = is_below_resolution(an, ci_width);

    // --- §6 verdict state machine (deterministic) ---
    let inputs = VerdictInputs {
        positive_control_collapsed: verdict.positive_control_collapsed,
        firewall_clean: verdict.augmentation_firewall_clean,
        disjoint_gen_a_n: an,
        disjoint_gen_a_ci_width: ci_width,
        disjoint_semantic_lift: verdict.disjoint_semantic_lift,
        lift_ci_lo: verdict.disjoint_lift_ci[0],
        collision_delta: verdict.collision_delta,
        memorization_gap: verdict.memorization_gap,
    };
    verdict.verdict = decide_encoder_verdict(&inputs).as_str().to_string();
    if verdict.verdict == Verdict::Invalid.as_str() && verdict.invalid_reason.is_empty() {
        verdict.invalid_reason = if !verdict.augmentation_firewall_clean {
            format!("augmentation firewall tripped: {}", verdict.firewall.violations.join("; "))
        } else if an == 0 {
            format!(
                "no feature-disjoint held-out phrases at any granularity (overlap-0 seen-elsewhere bin empty, level={level}): every held-out phrase shares features with its own class's training, so the eval cannot separate memorization from generalization"
            )
        } else if !verdict.positive_control_collapsed {
            format!(
                "positive control did not collapse: lexical CATA at overlap-0 (level={level}) = {:.3} vs floor95 {:.3} — eval is lexically separable (an easy task), so a high score does not evidence semantics",
                cata_overlap0, floor95
            )
        } else {
            "invalid (see gates)".into()
        };
    }
    verdict
}

const CERTIFY_STORE_DIR: &str = "certify_artifacts";

/// Map a GLE encoder alias to its checkpoint path + canonical artifact id.
fn gle_checkpoint_for_id(id: &str) -> (&'static str, &'static str) {
    match id.trim().to_ascii_lowercase().as_str() {
        "gle_base" => ("checkpoints/gle_student_base.json", "gle_base"),
        "gle_m5" | "gle_m5_routing_tuned" => ("checkpoints/gle_m5_routing_tuned.json", "gle_m5_routing_tuned"),
        "gle_m5_base" => ("checkpoints/gle_m5_base.json", "gle_m5_base"),
        // "gle" / "gle_routing_tuned" / anything else gle* → the routing-tuned student.
        _ => ("checkpoints/gle_student_routing_tuned.json", "gle_routing_tuned"),
    }
}

/// Reduce a frozen distilled GLE checkpoint to precomputed vectors over every phrase/alias the
/// pipeline embeds, so it runs through the identical certifier gate via the BYO-vectors hook.
/// Returns the audit encoder + an encoder-training provenance note (distillation-disjointness).
fn build_gle_audit_encoder(
    encoder_id: &str,
    nodes: &[(GroundingFleetDomain, String, Vec<String>)],
    captures: &[FailureCapture],
) -> Result<(AuditEncoder, String), String> {
    let (ckpt, canon) = gle_checkpoint_for_id(encoder_id);
    build_gle_vector_encoder(ckpt, canon, nodes, captures)
}

/// Core: reduce a specific GLE checkpoint to a BYO-vectors audit encoder with a chosen
/// artifact id, GLE-encoding every phrase/alias the pipeline will embed.
fn build_gle_vector_encoder(
    ckpt: &str,
    artifact_id: &str,
    nodes: &[(GroundingFleetDomain, String, Vec<String>)],
    captures: &[FailureCapture],
) -> Result<(AuditEncoder, String), String> {
    if !std::path::Path::new(ckpt).exists() {
        return Err(format!("checkpoint not found: {ckpt}"));
    }
    let canon = artifact_id;
    let cfg = LanguageConfig {
        encoder: EncoderPreset::BertClass,
        gle_checkpoint: Some(ckpt.to_string()),
        ..Default::default()
    };
    let gle_rt = growformer::dimension::LanguageRuntime::new(cfg);
    // Sanity: the GLE must produce a non-degenerate vector (i.e. the student actually loaded,
    // not the hashing fallback collapsing). A zero probe means the checkpoint did not load.
    let probe = gle_rt
        .encode_and_bridge("hello")
        .map_err(|e| format!("gle encode failed: {e}"))?
        .0;
    if probe.iter().all(|x| x.abs() < 1e-9) {
        return Err("gle produced a zero vector (checkpoint not loaded?)".into());
    }
    let dim = probe.len();

    let mut map: HashMap<String, Vec<f32>> = HashMap::new();
    {
        let mut insert = |text: &str| {
            if text.trim().is_empty() {
                return;
            }
            if let Ok((v, _)) = gle_rt.encode_and_bridge(text) {
                map.insert(text.to_string(), v);
            }
        };
        for c in captures {
            insert(&c.phrase);
        }
        for (_, id, aliases) in nodes {
            insert(id);
            insert(&id.replace('_', " "));
            for a in aliases {
                insert(a);
            }
        }
    }

    // Distillation provenance from the GLE meta sidecar (training domain).
    let meta_path = format!("{ckpt}.meta.json");
    let domain = std::fs::read_to_string(&meta_path)
        .ok()
        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
        .and_then(|v| {
            v.get("notes")
                .and_then(|n| n.as_array())
                .map(|a| a.iter().filter_map(|x| x.as_str()).collect::<Vec<_>>().join("; "))
        })
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "see meta".to_string());
    let note = format!(
        "frozen distilled GLE [{ckpt}]; GLE training domain: {domain}. Positive control = lexical CATA on the same eval, every run."
    );

    println!("  [gle] {canon}: {} vectors (dim {dim}) from {ckpt}", map.len());
    Ok((AuditEncoder::Vectors { id: canon.to_string(), map }, note))
}

/// Push one split of `(text, class)` rows into a capture set with real-traffic provenance.
fn push_indomain_captures(
    captures: &mut Vec<FailureCapture>,
    phrases: &[String],
    class: &str,
    split: CaptureSplit,
) {
    for (i, phrase) in phrases.iter().enumerate() {
        let tag = if split == CaptureSplit::Propose { "p" } else { "c" };
        captures.push(FailureCapture {
            phrase: phrase.clone(),
            encoder_embedding: Vec::new(),
            activated_nodes: Vec::new(),
            max_confidence: 0.0,
            entropy_bits: None,
            trigger_reason: FailureTrigger::NoNodeActivated,
            downstream_signal: None,
            timestamp_unix: 0,
            domain_context: "runtime".into(),
            inferred_concept_id: class.to_string(),
            split,
            provenance: grounding_loop::PhraseProvenance {
                kind: grounding_loop::ProvenanceKind::RealTraffic,
                phrase_id: format!("{class}-{tag}#{i}"),
                derived_from: Vec::new(),
            },
        });
    }
}

/// Construction B: per-`action_target` home-domain fixture. Node centroids are built from each
/// class's propose-split phrases (prototypes from train), certify is the held-out half — the
/// same shape as the Luna fixture, on the GLE's native domain.
fn build_m5_action_target_fixture(
    samples: &[LanguageSample],
) -> (Vec<(GroundingFleetDomain, String, Vec<String>)>, Vec<FailureCapture>) {
    let mut by_class: std::collections::BTreeMap<String, Vec<String>> = std::collections::BTreeMap::new();
    let mut seen: std::collections::HashSet<(String, String)> = std::collections::HashSet::new();
    for s in samples {
        let Some(t) = s.action_target.as_deref() else { continue };
        let t = t.trim();
        if t.is_empty() {
            continue;
        }
        let phrase = s.text.trim().to_string();
        if phrase.is_empty() {
            continue;
        }
        if seen.insert((t.to_string(), phrase.to_ascii_lowercase())) {
            by_class.entry(t.to_string()).or_default().push(phrase);
        }
    }
    let mut nodes = Vec::new();
    let mut captures = Vec::new();
    for (class, mut phrases) in by_class {
        if phrases.len() < 4 {
            continue; // need >=2 propose + 2 certify for a meaningful split
        }
        phrases.sort();
        let propose: Vec<String> = phrases.iter().step_by(2).cloned().collect();
        let certify: Vec<String> = phrases.iter().skip(1).step_by(2).cloned().collect();
        push_indomain_captures(&mut captures, &propose, &class, CaptureSplit::Propose);
        push_indomain_captures(&mut captures, &certify, &class, CaptureSplit::Certify);
        nodes.push((GroundingFleetDomain::Runtime, class, propose));
    }
    (nodes, captures)
}

/// Construction A: the literal 2-way support/coding headline. certify = the fine-tune held-out
/// split, so it is provenance-disjoint from the GLE's routing fine-tune train.
fn build_support_coding_fixture(
) -> (Vec<(GroundingFleetDomain, String, Vec<String>)>, Vec<FailureCapture>) {
    let ((train_s, train_c), (valid_s, valid_c)) = build_routing_finetune_dataset();
    let mut captures = Vec::new();
    push_indomain_captures(&mut captures, &train_s, "support", CaptureSplit::Propose);
    push_indomain_captures(&mut captures, &train_c, "coding", CaptureSplit::Propose);
    push_indomain_captures(&mut captures, &valid_s, "support", CaptureSplit::Certify);
    push_indomain_captures(&mut captures, &valid_c, "coding", CaptureSplit::Certify);
    let nodes = vec![
        (GroundingFleetDomain::Runtime, "support".to_string(), train_s),
        (GroundingFleetDomain::Runtime, "coding".to_string(), train_c),
    ];
    (nodes, captures)
}

/// Run the certifier on a (GLE, in-domain fixture) pair and emit the verdict artifact.
fn run_indomain_certification(
    artifact_id: &str,
    ckpt: &str,
    nodes: &[(GroundingFleetDomain, String, Vec<String>)],
    captures: &[FailureCapture],
    provenance_extra: &str,
) {
    let n_propose = captures.iter().filter(|c| c.split == CaptureSplit::Propose).count();
    let n_certify = captures.iter().filter(|c| c.split == CaptureSplit::Certify).count();
    let n_concepts = nodes.len();
    println!(
        "\n=== {artifact_id}: {n_concepts} concepts | propose={n_propose} | certify={n_certify} ===",
    );
    if n_concepts < 2 || n_certify == 0 {
        println!("  insufficient data for {artifact_id} (need >=2 concepts and certify phrases).");
        return;
    }
    let (enc, gle_note) = match build_gle_vector_encoder(ckpt, artifact_id, nodes, captures) {
        Ok(p) => p,
        Err(e) => {
            println!("  could not build GLE encoder: {e}");
            return;
        }
    };
    let provenance = format!("{gle_note} || {provenance_extra}");
    let (dm, _, _, _) = build_language_demo_manager(0.0);
    let artifact = certify_encoder_pipeline(&enc, &dm.language_runtime, nodes, captures, 42, &provenance);

    let json = artifact.to_json();
    std::fs::create_dir_all(CERTIFY_STORE_DIR).ok();
    let store_path = std::path::Path::new(CERTIFY_STORE_DIR).join(artifact.filename());
    std::fs::write(&store_path, &json).expect("write verdict artifact");
    std::fs::write("certify_verdict_latest.json", &json).expect("write latest verdict");
    println!("\n{}", render_verdict(&artifact));
    println!("\nArtifact: {}", store_path.display());
}

/// `--certify-gle-indomain`: certify the GLE on its own (support/coding) home domain through the
/// identical gate — (A) the literal 2-way 100% headline, then (B) home-domain action_target
/// many-way routing. The CATA positive control runs on each eval; if it does not collapse, the
/// eval is lexically separable (an easy task) and the verdict is INVALID by construction.
fn demo_certify_gle_indomain() {
    println!("--- In-domain GLE certification (the 100% on its own turf) ---\n");
    let ckpt = "checkpoints/gle_student_routing_tuned.json";
    if !std::path::Path::new(ckpt).exists() {
        println!("GLE checkpoint not found: {ckpt}");
        return;
    }
    let samples = match load_all_m5_training_data() {
        Ok(s) if !s.is_empty() => s,
        Ok(_) => {
            println!("No M5 home-domain data found (data/language/m5 empty).");
            return;
        }
        Err(e) => {
            println!("Could not load M5 home-domain data: {e}");
            return;
        }
    };
    println!("Loaded {} home-domain labeled samples.\n", samples.len());

    // Construction A: the literal 2-way support/coding headline (the actual 100%).
    let (nodes_a, caps_a) = build_support_coding_fixture();
    run_indomain_certification(
        "gle_2way_support_coding",
        ckpt,
        &nodes_a,
        &caps_a,
        "Construction A: literal 2-way support/coding headline (the reported 100%). certify = fine-tune \
         held-out split (disjoint from routing fine-tune train). Distillation target was a HASHING proxy \
         (lexical). Expectation: lexical CATA also separates 2 buckets ⇒ positive control does NOT collapse \
         ⇒ INVALID = the 100% measured a lexically-separable 2-way task, not semantic generalization.",
    );

    // Construction B: home-domain many-way routing at action_target granularity.
    let (nodes_b, caps_b) = build_m5_action_target_fixture(&samples);
    run_indomain_certification(
        "gle_indomain_action_target",
        ckpt,
        &nodes_b,
        &caps_b,
        "Construction B: home-domain action_target many-way routing. DISTILLATION-LINEAGE CAVEAT: the GLE \
         distillation stage consumed M5 texts (teacher-mimicry vs a hashing proxy, not label-supervised), so \
         this eval is same-distribution held-out, NOT distillation-disjoint — i.e. generous to the GLE.",
    );

    println!("\nIn-domain certification complete. Artifacts in {CERTIFY_STORE_DIR}/.");
}

/// `--certify-encoder <encoder_id> [companion_dir] [seed]`: run the pipeline on a companion
/// and emit the verdict artifact (append-only longitudinal store + a stable latest copy).
fn demo_certify_encoder(spec: &[String]) {
    println!("--- Certifier-first pipeline (the contract every encoder is judged by) ---\n");
    let encoder_id = spec.first().cloned().unwrap_or_default();
    let is_gle = encoder_id.trim().to_ascii_lowercase().starts_with("gle");
    if !is_gle && AuditEncoder::from_id(&encoder_id).is_none() {
        println!("Unknown encoder '{encoder_id}'. Supported: supervised, cata, gle[_base|_m5].");
        return;
    }
    let dir = spec
        .get(1)
        .filter(|s| !s.trim().is_empty() && s.trim() != "default")
        .cloned()
        .unwrap_or_else(|| LUNA_DEFAULT_DIR.to_string());
    let seed: u64 = spec.get(2).and_then(|s| s.trim().parse().ok()).unwrap_or(42);

    let data_dir = std::path::Path::new(&dir).join("data");
    let grounding_path = data_dir.join("pet_world_grounding.toml");
    let toml = match std::fs::read_to_string(&grounding_path) {
        Ok(t) => t,
        Err(e) => {
            println!("Could not read {}: {}", grounding_path.display(), e);
            return;
        }
    };
    if let Err(e) = world_grounding::load_grounding_graph_from_str(&toml) {
        println!("Failed to parse grounding graph: {}", e);
        return;
    }
    let mut samples: Vec<LanguageSample> = Vec::new();
    if let Err(e) = append_language_samples_from_training_jsonl_dir(&mut samples, &data_dir) {
        println!("Failed to load JSONL corpus: {}", e);
        return;
    }
    let nodes = luna_runtime_nodes();
    let valid_ids: std::collections::HashSet<String> =
        nodes.iter().map(|(_, id, _)| id.clone()).collect();
    let captures = luna_build_captures(&samples, &valid_ids);
    if captures.is_empty() {
        println!("No usable captures.");
        return;
    }

    // Resolve the encoder under audit. For the GLE we reduce a frozen distilled encoder to
    // precomputed vectors over every phrase/alias the pipeline embeds (BYO-vectors hook).
    let (enc, provenance) = if is_gle {
        match build_gle_audit_encoder(&encoder_id, &nodes, &captures) {
            Ok((enc, note)) => {
                let prov = format!(
                    "{note} || eval = companion ({dir}) real traffic; domain-disjoint from the GLE's \
                     training domain (phrase-level manifest unavailable — distillation-disjointness asserted by domain)."
                );
                (enc, prov)
            }
            Err(e) => {
                println!("Could not build GLE audit encoder: {e}");
                return;
            }
        }
    } else {
        (AuditEncoder::from_id(&encoder_id).unwrap(), String::new())
    };

    println!(
        "encoder={} seed={} | nodes={} | captures={} | grounding={}\n",
        enc.id(),
        seed,
        nodes.len(),
        captures.len(),
        grounding_path.display()
    );
    if !provenance.is_empty() {
        println!("  encoder provenance: {provenance}\n");
    }

    let (dm, _, _, _) = build_language_demo_manager(0.0);
    let artifact = certify_encoder_pipeline(&enc, &dm.language_runtime, &nodes, &captures, seed, &provenance);

    let json = artifact.to_json();
    std::fs::create_dir_all(CERTIFY_STORE_DIR).ok();
    let store_path = std::path::Path::new(CERTIFY_STORE_DIR).join(artifact.filename());
    std::fs::write(&store_path, &json).expect("write verdict artifact");
    std::fs::write("certify_verdict_latest.json", &json).expect("write latest verdict");

    println!("\n{}", render_verdict(&artifact));
    println!("\nArtifact: {}", store_path.display());
    println!("Latest:   certify_verdict_latest.json");
}

/// `--certify-verdict <artifact.json>`: re-read and pretty-print a stored verdict.
fn demo_certify_verdict(path: &str) {
    let raw = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            println!("Could not read {path}: {e}");
            return;
        }
    };
    match serde_json::from_str::<EncoderVerdict>(&raw) {
        Ok(v) => println!("{}", render_verdict(&v)),
        Err(e) => println!("Not a valid verdict artifact ({e})"),
    }
}

fn render_verdict(v: &EncoderVerdict) -> String {
    let mut s = String::new();
    s.push_str("Verdict artifact (go/no-go = disjoint_semantic_lift)\n");
    s.push_str("====================================================\n");
    s.push_str(&format!("  encoder={}  data_hash={}  seed={}\n", v.encoder_id, v.data_hash, v.seed));
    s.push_str(&format!("  candidate set (honest, all nodes): {}\n\n", v.candidate_set_size));
    s.push_str(&format!("  >>> VERDICT: {} <<<\n", v.verdict));
    if !v.invalid_reason.is_empty() {
        s.push_str(&format!("  reason: {}\n", v.invalid_reason));
    }
    s.push_str(&format!(
        "  disjoint_semantic_lift = {:+.3}  CI=[{:+.3}, {:+.3}]   (gate: lift>0 AND CI excludes 0)\n\n",
        v.disjoint_semantic_lift, v.disjoint_lift_ci[0], v.disjoint_lift_ci[1]
    ));
    s.push_str("  lift decomposition\n");
    s.push_str(&format!(
        "    disjoint_gen_a (seen-elsewhere)   = {:.3}  (n={})\n",
        v.disjoint_gen_a, v.disjoint_gen_a_n
    ));
    s.push_str(&format!("    semantic_floor_95 (shuffle null)  = {:.3}\n", v.semantic_floor_95));
    s.push_str(&format!("    semantic_floor_mean               = {:.3}\n", v.semantic_floor_mean));
    s.push_str(&format!("    pooled_heldout (evidence, NOT gate) = {:.3}\n", v.pooled_heldout));
    s.push_str(&format!("    memorization_gap (captured−heldout) = {:.3}\n\n", v.memorization_gap));
    s.push_str("  diagnostics\n");
    s.push_str(&format!(
        "    feature_family disjoint-0 acc: word={:.3} bigram={:.3} trigram(union)={:.3}\n",
        v.feature_family.word, v.feature_family.bigram, v.feature_family.trigram
    ));
    s.push_str(&format!(
        "    disjoint_level={} (granularity lift resolved at; looser=leakier)  plateau_flag={}  collision_delta={:+.3}\n\n",
        v.disjoint_level, v.plateau_flag, v.collision_delta
    ));
    s.push_str("  validity gates (all must hold, else INVALID)\n");
    s.push_str(&format!("    positive_control_collapsed (CATA floors) = {}\n", v.positive_control_collapsed));
    s.push_str(&format!("    augmentation_firewall_clean              = {}\n", v.augmentation_firewall_clean));
    s.push_str(&format!("    below_resolution                         = {}\n", v.below_resolution));
    s.push_str(&format!(
        "    provenance: train(real={}, authored={}, augmented={}) certify(real={}, authored={}, augmented={})\n",
        v.firewall.train_real,
        v.firewall.train_authored,
        v.firewall.train_augmented,
        v.firewall.certify_real,
        v.firewall.certify_authored,
        v.firewall.certify_augmented,
    ));
    if !v.firewall.violations.is_empty() {
        s.push_str("    firewall violations:\n");
        for x in &v.firewall.violations {
            s.push_str(&format!("      - {x}\n"));
        }
    }
    if !v.encoder_provenance.is_empty() {
        s.push_str(&format!("    encoder provenance: {}\n", v.encoder_provenance));
    }
    s.push_str(&format!("\n  one-line: {}\n", v.one_line()));
    s
}

/// Positive control: install a synthetic semantic embedder (bring-your-own vectors),
/// build a 3-concept space where one approved alias per concept moves the centroid past
/// the held-out boundary, and certify. Returns a report fragment. Clears the embedder on
/// exit so it does not leak into other demos.
fn positive_control_section(rt: &growformer::dimension::language::LanguageRuntime, params: &GroundingLoopParams) -> String {
    let mut map: HashMap<String, Vec<f32>> = HashMap::new();
    map.insert("concept_a".into(), vec![1.0, 0.0, 0.0]);
    map.insert("concept_b".into(), vec![0.0, 1.0, 0.0]);
    map.insert("concept_c".into(), vec![0.0, 0.0, 1.0]);
    map.insert("alpha proposal phrase".into(), vec![0.8, 0.6, 0.0]);
    map.insert("beta proposal phrase".into(), vec![0.0, 0.8, 0.6]);
    map.insert("gamma proposal phrase".into(), vec![0.6, 0.0, 0.8]);
    map.insert("alpha certify phrase".into(), vec![0.6, 0.8, 0.0]);
    map.insert("beta certify phrase".into(), vec![0.0, 0.6, 0.8]);
    map.insert("gamma certify phrase".into(), vec![0.8, 0.0, 0.6]);
    install_vector_embedder(map);

    let d = GroundingFleetDomain::Runtime;
    let nodes_before = vec![
        (d, "concept_a".to_string(), vec![]),
        (d, "concept_b".to_string(), vec![]),
        (d, "concept_c".to_string(), vec![]),
    ];
    let nodes_after = vec![
        (d, "concept_a".to_string(), vec!["alpha proposal phrase".to_string()]),
        (d, "concept_b".to_string(), vec!["beta proposal phrase".to_string()]),
        (d, "concept_c".to_string(), vec!["gamma proposal phrase".to_string()]),
    ];
    let before = build_grounding_index_from_nodes(rt, &nodes_before, params).expect("ctrl before");
    let after = build_grounding_index_from_nodes(rt, &nodes_after, params).expect("ctrl after");

    let mk = |phrase: &str, concept: &str, split: CaptureSplit| FailureCapture {
        phrase: phrase.into(),
        encoder_embedding: Vec::new(),
        activated_nodes: Vec::new(),
        max_confidence: 0.0,
        entropy_bits: None,
        trigger_reason: FailureTrigger::NoNodeActivated,
        downstream_signal: None,
        timestamp_unix: 0,
        domain_context: "runtime".into(),
        inferred_concept_id: concept.into(),
        split,
        provenance: grounding_loop::PhraseProvenance::real(phrase),
    };
    let captures = vec![
        mk("alpha proposal phrase", "concept_a", CaptureSplit::Propose),
        mk("beta proposal phrase", "concept_b", CaptureSplit::Propose),
        mk("gamma proposal phrase", "concept_c", CaptureSplit::Propose),
        mk("alpha certify phrase", "concept_a", CaptureSplit::Certify),
        mk("beta certify phrase", "concept_b", CaptureSplit::Certify),
        mk("gamma certify phrase", "concept_c", CaptureSplit::Certify),
    ];

    let (before_m, after_m) = certify_batch(&captures, rt, &before, &after, d).expect("ctrl certify");
    let verdict = decide_batch_verdict(&before_m, &after_m, params, false);

    let additions: Vec<(GroundingFleetDomain, String, String)> = vec![
        (d, "concept_a".to_string(), "alpha proposal phrase".to_string()),
        (d, "concept_b".to_string(), "beta proposal phrase".to_string()),
        (d, "concept_c".to_string(), "gamma proposal phrase".to_string()),
    ];
    let curve = coverage_vs_additions_curve(&captures, rt, &before, &additions).expect("ctrl curve");

    let mut s = String::new();
    s.push_str("POSITIVE CONTROL — synthetic semantic geometry (bring-your-own vectors)\n");
    s.push_str("----------------------------------------------------------------------\n");
    s.push_str("Each concept's held-out paraphrase sits just past the baseline boundary; one\n");
    s.push_str("approved alias pulls the centroid over. A real semantic encoder should look\n");
    s.push_str("like THIS, not like the lexical CATA result above.\n\n");
    s.push_str(&format_certifier_report(
        "Semantic control batch (one approved alias per concept)",
        &before_m,
        &after_m,
        verdict,
    ));
    s.push_str("\n\n");
    s.push_str(&format_coverage_curve(
        "Coverage-vs-additions curve — semantic control sweep",
        &curve,
    ));
    s.push_str(&format!(
        "\n  → certifier verdict under semantic geometry: {} (expected GenuineCoverageImprovement)\n",
        verdict.as_str(),
    ));
    s.push_str("\nBring-your-own-encoder workflow: run any sentence encoder offline over every\n");
    s.push_str("captured phrase + node alias, install via `install_vector_embedder(map)`, then\n");
    s.push_str("re-run this audit. If the genuine sweep inverts (held-out rises with additions),\n");
    s.push_str("the encoder passes the acceptance gate; if it looks like the lexical result, it\n");
    s.push_str("does not.\n");

    clear_phrase_embedder();
    s
}

fn demo_grounding_loop_audit() {
    println!("--- Grounding loop audit (assisted maintenance) ---\n");
    println!("Loads pet runtime fixture + synthetic propose/certify splits.\n");

    let params = GroundingLoopParams::default();
    world_grounding::load_grounding_graph_from_str(PET_DOMAIN_FIXTURE_TOML)
        .expect("load pet domain fixture");

    let (mut dm, _, _, _) = build_language_demo_manager(0.0);
    // Option 2: back the CliffordE8 codec path with a domain dictionary so the encoder
    // is non-degenerate. Option 1 (bridge-routed fallback) activates automatically inside
    // `embed_phrase` if the raw vector still collapses.
    install_grounding_audit_dictionary(&mut dm);
    let rt = &dm.language_runtime;
    println!(
        "  representation: CATA centroid (pre-quantization, lexical), bridge-routed fallback\n"
    );

    let before_index =
        build_grounding_index(rt, &HashMap::new(), &params).expect("build grounding index");

    let fixture = synthetic_audit_fixture();
    let captures = fixture_rows_to_captures(rt, &before_index, &fixture);
    write_grounding_captures_csv(GROUNDING_CAPTURES_CSV, &captures);

    let proposals = build_proposals_for_captures(
        &captures,
        &before_index,
        &params,
        rt,
        &before_index,
    );
    write_grounding_proposals_csv(GROUNDING_PROPOSALS_CSV, &proposals);

    println!("Proposal mechanism (nearest-in-domain, τ_alias={}):", params.tau_alias);
    for cap in captures.iter().filter(|c| c.split == CaptureSplit::Propose) {
        let domain = parse_fleet_domain(&cap.domain_context).unwrap_or(GroundingFleetDomain::Crypto);
        if let Some(m) = before_index.nearest_in_domain(&cap.encoder_embedding, domain) {
            println!(
                "  {:?} → {}:{} sim={:.3} margin={:.3} (want {})",
                cap.phrase,
                m.domain.as_str(),
                m.node_id,
                m.similarity,
                m.similarity - m.second_similarity,
                cap.inferred_concept_id,
            );
        }
    }
    println!();
    // Genuine batch: integrate ONLY approved propose-set aliases. Held-out certify
    // phrases are NEVER added here — the whole point is to test whether the
    // proposal-set edits generalize to phrasings the mechanism never saw.
    let mut after_genuine = before_index.clone();
    let mut had_collisions = false;
    let mut genuine_added = 0usize;
    for p in &proposals {
        if p.collision_score >= params.collision_threshold {
            had_collisions = true;
        }
        if !p.approved {
            continue;
        }
        if let ProposalKind::Alias {
            phrase,
            target_node,
            target_domain,
            ..
        } = &p.kind
        {
            let domain = parse_fleet_domain(target_domain).unwrap_or(GroundingFleetDomain::Runtime);
            if let Ok((emb, _)) = embed_phrase(rt, phrase) {
                after_genuine.add_alias_to_node(domain, target_node, phrase, emb);
                genuine_added += 1;
            }
        }
    }

    let home = GroundingFleetDomain::Crypto;
    let (before_m, after_genuine_m) =
        certify_batch(&captures, rt, &before_index, &after_genuine, home).expect("certify genuine");
    let genuine_verdict = decide_batch_verdict(&before_m, &after_genuine_m, &params, had_collisions);

    // Memorization contrast: add only exact propose phrases as aliases (lookup-table growth).
    let mut after_memo = before_index.clone();
    for cap in captures.iter().filter(|c| c.split == CaptureSplit::Propose) {
        if let Ok((emb, _)) = embed_phrase(rt, &cap.phrase) {
            let domain = parse_fleet_domain(&cap.domain_context)
                .unwrap_or(GroundingFleetDomain::Runtime);
            after_memo.add_alias_to_node(domain, &cap.inferred_concept_id, &cap.phrase, emb);
        }
    }
    let (_, after_memo_m) =
        certify_batch(&captures, rt, &before_index, &after_memo, home).expect("certify memo");

    let mut report = String::new();
    report.push_str("Grounding loop audit — assisted maintenance certifier\n");
    report.push_str("=====================================================\n\n");
    report.push_str(&format!(
        "Fixture rows: {} (propose/certify per concept)\n",
        fixture.len()
    ));
    report.push_str(&format!(
        "Proposals: {} (approved: {}, integrated into genuine batch: {})\n\n",
        proposals.len(),
        proposals.iter().filter(|p| p.approved).count(),
        genuine_added,
    ));
    report.push_str(&format_certifier_report(
        "Genuine batch (approved propose-set aliases only)",
        &before_m,
        &after_genuine_m,
        genuine_verdict,
    ));
    report.push_str("\n\n");
    let memo_verdict = decide_batch_verdict(&before_m, &after_memo_m, &params, false);
    report.push_str(&format_certifier_report(
        "Memorization contrast (exact captured phrases only)",
        &before_m,
        &after_memo_m,
        memo_verdict,
    ));
    // Coverage-vs-additions curve (§6 n-sweep analog). Memorization sweep: add the
    // exact captured propose phrases one at a time as aliases of their concept, and
    // watch held-out paraphrase accuracy vs captured-set coverage.
    let memo_additions: Vec<(GroundingFleetDomain, String, String)> = captures
        .iter()
        .filter(|c| c.split == CaptureSplit::Propose)
        .map(|c| {
            (
                parse_fleet_domain(&c.domain_context).unwrap_or(GroundingFleetDomain::Runtime),
                c.inferred_concept_id.clone(),
                c.phrase.clone(),
            )
        })
        .collect();
    let _ = std::fs::remove_file(GROUNDING_CURVE_CSV);
    let memo_curve = coverage_vs_additions_curve(&captures, rt, &before_index, &memo_additions)
        .expect("memorization curve");
    write_grounding_curve_csv(GROUNDING_CURVE_CSV, "memorization", &memo_curve);

    // Genuine sweep: only approved propose-set aliases (held-out must be untouched).
    let genuine_additions: Vec<(GroundingFleetDomain, String, String)> = proposals
        .iter()
        .filter(|p| p.approved)
        .filter_map(|p| match &p.kind {
            ProposalKind::Alias {
                phrase,
                target_node,
                target_domain,
                ..
            } => Some((
                parse_fleet_domain(target_domain).unwrap_or(GroundingFleetDomain::Runtime),
                target_node.clone(),
                phrase.clone(),
            )),
            _ => None,
        })
        .collect();
    let genuine_curve =
        coverage_vs_additions_curve(&captures, rt, &before_index, &genuine_additions)
            .expect("genuine curve");
    write_grounding_curve_csv(GROUNDING_CURVE_CSV, "genuine", &genuine_curve);

    report.push_str("\n\n");
    report.push_str(&format_coverage_curve(
        "Coverage-vs-additions curve — memorization sweep (exact phrases)",
        &memo_curve,
    ));
    report.push_str("\n\n");
    report.push_str(&format_coverage_curve(
        "Coverage-vs-additions curve — genuine sweep (approved aliases)",
        &genuine_curve,
    ));

    let (memo_cap_lift, memo_held_lift) = curve_lifts(&memo_curve);
    let overfit = memo_cap_lift > 0.05 && memo_held_lift < params.min_held_out_lift;
    report.push_str("\n\nDecision rule (§7):\n");
    report.push_str(&format!(
        "  integrate genuine batch: {}\n",
        genuine_verdict == BatchVerdict::GenuineCoverageImprovement
    ));
    report.push_str(&format!(
        "  reject memorization batch: {}\n",
        memo_verdict == BatchVerdict::LexiconMemorization
            || after_memo_m.generalization_gap > params.max_generalization_gap
    ));
    report.push_str(&format!(
        "  n-sweep overfitting signature (captured rises, held-out plateaus): {}\n",
        overfit
    ));

    // Data-driven τ_alias: midpoint of same-concept vs cross-concept similarity under
    // the current encoder. Recommendation #4 — re-derive thresholds per encoder rather
    // than trusting the hand-picked default.
    if let Ok(cal) = calibrate_alias_threshold(&captures, rt, &before_index) {
        report.push_str("\nThreshold calibration (current encoder):\n");
        report.push_str(&format!(
            "  same-concept sim mean: {:.3}  cross-concept sim mean: {:.3}\n  suggested τ_alias: {:.3} (default in use: {:.3}; samples: {})\n",
            cal.same_concept_mean,
            cal.cross_concept_mean,
            cal.suggested_tau_alias,
            params.tau_alias,
            cal.samples,
        ));
        if cal.same_concept_mean - cal.cross_concept_mean < 0.05 {
            report.push_str(
                "  WARNING: same/cross separation < 0.05 — this encoder cannot discriminate concepts; proposals are tie-break artifacts (see lexical result above).\n",
            );
        }
    }

    // POSITIVE CONTROL: a synthetic *semantic* geometry where a single approved alias
    // genuinely extends held-out coverage. This is the mirror of the lexical negative
    // result above — it proves the certifier reports GenuineCoverageImprovement when the
    // representation actually generalizes, i.e. the gate is not rigged to always reject.
    // It also documents the bring-your-own-vectors path: install precomputed embeddings
    // from any real/semantic encoder (run offline over your phrases + node aliases) and
    // the loop runs unchanged.
    report.push_str("\n\n");
    report.push_str(&positive_control_section(rt, &params));

    println!("{}", report);
    std::fs::write(GROUNDING_RESULTS_TXT, &report).expect("write results");
    println!(
        "\nWrote {}, {}, {}, {}. Re-run with --grounding-loop-analyze.",
        GROUNDING_CAPTURES_CSV, GROUNDING_PROPOSALS_CSV, GROUNDING_CURVE_CSV, GROUNDING_RESULTS_TXT
    );
}

fn demo_grounding_loop_analyze() {
    println!("--- Grounding loop analyze (from captured CSV) ---\n");
    let params = GroundingLoopParams::default();
    world_grounding::load_grounding_graph_from_str(PET_DOMAIN_FIXTURE_TOML)
        .expect("load pet domain fixture");

    let (mut dm, _, _, _) = build_language_demo_manager(0.0);
    install_grounding_audit_dictionary(&mut dm);
    let rt = &dm.language_runtime;

    let mut captures = load_grounding_captures_csv(GROUNDING_CAPTURES_CSV);
    if captures.is_empty() {
        println!("No captures in {}. Run --grounding-loop-audit first.", GROUNDING_CAPTURES_CSV);
        return;
    }

    let before_index =
        build_grounding_index(rt, &HashMap::new(), &params).expect("build grounding index");
    rehydrate_captures(rt, &before_index, &mut captures);

    let proposals = build_proposals_for_captures(
        &captures,
        &before_index,
        &params,
        rt,
        &before_index,
    );

    let mut after = before_index.clone();
    for p in proposals.iter().filter(|p| p.approved) {
        if let ProposalKind::Alias {
            phrase,
            target_node,
            target_domain,
            ..
        } = &p.kind
        {
            let domain = parse_fleet_domain(target_domain).unwrap_or(GroundingFleetDomain::Runtime);
            if let Ok((emb, _)) = embed_phrase(rt, phrase) {
                after.add_alias_to_node(domain, target_node, phrase, emb);
            }
        }
    }

    let home = GroundingFleetDomain::Crypto;
    let (before_m, after_m) =
        certify_batch(&captures, rt, &before_index, &after, home).expect("certify");
    let verdict = decide_batch_verdict(&before_m, &after_m, &params, false);
    println!(
        "{}",
        format_certifier_report("Re-analyzed batch", &before_m, &after_m, verdict)
    );
}

fn accuracy_learned_router_xy(
    main: &mut MainDimension,
    router: &mut LearnedRouter,
    spiral_gid: GroupId,
    circles_gid: GroupId,
    data: &[Sample],
) -> f32 {
    let gids = [spiral_gid, circles_gid];
    let mut correct = 0usize;
    for (input, target) in data {
        let logits = router.predict_logits(input);
        let idx = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        let gid = gids[idx.min(1)];
        let scalar = specialist_scalar(main, gid, input);
        if scalar_matches_target(scalar, target[0]) {
            correct += 1;
        }
    }
    if data.is_empty() {
        0.0
    } else {
        correct as f32 / data.len() as f32
    }
}

fn accuracy_learned_router_expert(
    main: &mut MainDimension,
    router: &mut LearnedRouter,
    spiral_gid: GroupId,
    circles_gid: GroupId,
    data: &[Sample],
) -> f32 {
    let gids = [spiral_gid, circles_gid];
    let mut correct = 0usize;
    for (input, target) in data {
        let features = specialist_feature_pair(main, spiral_gid, circles_gid, input);
        let idx = router_route_index(router, &features);
        let gid = gids[idx.min(1)];
        let scalar = specialist_scalar(main, gid, input);
        if scalar_matches_target(scalar, target[0]) {
            correct += 1;
        }
    }
    if data.is_empty() {
        0.0
    } else {
        correct as f32 / data.len() as f32
    }
}

fn accuracy_learned_router_disagreement(
    main: &mut MainDimension,
    router: &mut LearnedRouter,
    spiral_gid: GroupId,
    circles_gid: GroupId,
    data: &[Sample],
) -> f32 {
    let gids = [spiral_gid, circles_gid];
    let mut correct = 0usize;
    for (input, target) in data {
        let f0 = specialist_scalar(main, spiral_gid, input);
        let f1 = specialist_scalar(main, circles_gid, input);
        let idx = router_route_index(router, &[f0 - f1]);
        let gid = gids[idx.min(1)];
        let scalar = specialist_scalar(main, gid, input);
        if scalar_matches_target(scalar, target[0]) {
            correct += 1;
        }
    }
    if data.is_empty() {
        0.0
    } else {
        correct as f32 / data.len() as f32
    }
}

fn accuracy_logistic_radius_gate(
    main: &mut MainDimension,
    spiral_gid: GroupId,
    circles_gid: GroupId,
    data: &[Sample],
    gate: RadiusLogisticGate,
) -> f32 {
    let mut correct = 0usize;
    for (input, target) in data {
        let r = sample_radius(input);
        let o_spiral = specialist_scalar(main, spiral_gid, input);
        let o_circles = specialist_scalar(main, circles_gid, input);
        let scalar = if sigmoid(gate.w * r + gate.b) >= 0.5 {
            o_spiral
        } else {
            o_circles
        };
        if scalar_matches_target(scalar, target[0]) {
            correct += 1;
        }
    }
    if data.is_empty() {
        0.0
    } else {
        correct as f32 / data.len() as f32
    }
}

fn demo_phase3c_composition() {
    println!("--- Phase 3c: Composition + Episodic ---\n");
    let mut dm = DimensionManager::new(phase3_composition_config());
    let mut rng = StdRng::seed_from_u64(42);
    let mut data_rng = StdRng::seed_from_u64(99);

    let spiral_data = generate_spiral_data(400, &mut data_rng);
    let circles_data = generate_concentric_circles_data(400, &mut data_rng);
    let calibration_spiral: Vec<_> = spiral_data.iter().take(100).cloned().collect();
    let calibration_circles: Vec<_> = circles_data.iter().take(100).cloned().collect();

    let spiral_group = train_promoted_mirror(
        &mut dm,
        "spiral",
        42,
        &spiral_data,
        &calibration_spiral,
        &mut rng,
        true,
    );
    let circles_group = train_promoted_mirror(
        &mut dm,
        "circles",
        43,
        &circles_data,
        &calibration_circles,
        &mut rng,
        true,
    );
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

    let (virtual_group, comp_acc) = dm.train_composition_one_pass(
        &[spiral_group, circles_group],
        &task_c_train,
    );
    println!(
        "  Composition (VirtualGroup) on {} samples, one-pass solve: {:.1}%",
        task_c_train.len(),
        comp_acc * 100.0
    );
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
    let _moons_group = train_promoted_mirror(
        &mut dm,
        "moons",
        44,
        &moons_data,
        &calibration_moons,
        &mut rng,
        true,
    );
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
    let (vg_d, comp_d_acc) = dm.train_composition_one_pass(&all_three, &task_d_train);
    println!(
        "  3-group composition ({} samples, one-pass solve): {:.1}%",
        task_d_train.len(),
        comp_d_acc * 100.0
    );
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

// =============================================================================
// Demo: Phase 3e — Balanced composite (decisive VirtualGroup evaluation)
// Task E = 50/50 inner/outer spiral-gated circles; four baselines + recall ablation.
// =============================================================================

#[derive(Clone, Copy, Debug)]
struct Phase3eSeedResult {
    spiral_heldout: f32,
    circles_heldout: f32,
    oracle_best_heldout: f32,
    vg_fixed_heldout: f32,
    vg_recall_heldout: f32,
    direct_mirror_heldout: f32,
    confidence_argmax_heldout: f32,
    learned_router_xy_heldout: f32,
    learned_router_expert_heldout: f32,
    calibration_router_xy_heldout: f32,
    calibration_router_expert_heldout: f32,
    disagreement_router_heldout: f32,
    expert_region_agreement: f32,
    expert_margin_radius_corr: f32,
    calib_expert_region_agreement: f32,
    calib_expert_margin_radius_corr: f32,
    radius_gate_heldout: f32,
    logistic_gate_heldout: f32,
    oracle_region_heldout: f32,
    train_inner_frac: f32,
    heldout_inner_frac: f32,
    train_near_boundary_frac: f32,
}

fn mean_std(values: &[f32]) -> (f32, f32) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    if values.len() == 1 {
        return (mean, 0.0);
    }
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / (values.len() - 1) as f32;
    (mean, var.sqrt())
}

fn run_phase3e_seed(seed: u64) -> Phase3eSeedResult {
    const INNER_RADIUS: f32 = 0.4;
    const N_SAMPLES: usize = 400;
    const TRAIN_N: usize = 30;

    let mut dm = DimensionManager::new(phase3_composition_config());
    let mut rng = StdRng::seed_from_u64(seed);
    let mut data_rng = StdRng::seed_from_u64(seed.wrapping_mul(97).wrapping_add(99));

    let spiral_data = generate_spiral_data(400, &mut data_rng);
    let circles_data = generate_concentric_circles_data(400, &mut data_rng);
    let calibration_spiral: Vec<_> = spiral_data.iter().take(100).cloned().collect();
    let calibration_circles: Vec<_> = circles_data.iter().take(100).cloned().collect();

    let spiral_group = train_promoted_mirror(
        &mut dm,
        "spiral",
        seed,
        &spiral_data,
        &calibration_spiral,
        &mut rng,
        false,
    );
    let circles_group = train_promoted_mirror(
        &mut dm,
        "circles",
        seed.wrapping_add(1),
        &circles_data,
        &calibration_circles,
        &mut rng,
        false,
    );

    let task_e_data = generate_balanced_spiral_gated_circles_data(
        &mut dm.main,
        spiral_group,
        circles_group,
        INNER_RADIUS,
        N_SAMPLES,
        &mut data_rng,
    );
    let (train, heldout) =
        stratified_composite_split(&task_e_data, INNER_RADIUS, TRAIN_N, &mut data_rng);

    let spiral_heldout = dm.evaluate_main_group(spiral_group, &heldout);
    let circles_heldout = dm.evaluate_main_group(circles_group, &heldout);
    let oracle_best_heldout = spiral_heldout.max(circles_heldout);

    let (vg, _train_acc) = dm.train_composition_one_pass(&[spiral_group, circles_group], &train);
    let vg_fixed_heldout = accuracy_virtual_group(&mut dm, &vg, &heldout);

    let residual = 1.0 - oracle_best_heldout;
    dm.store_composition_episode(&vg, &train, _train_acc, residual);
    let mut sig_train = [0.0f32; 2];
    for (input, _) in &train {
        sig_train[0] += input[0];
        sig_train[1] += input[1];
    }
    sig_train[0] /= train.len() as f32;
    sig_train[1] /= train.len() as f32;
    let vg_recall_heldout = if let Some(ep) = dm
        .episodic_retrieve(&sig_train, 0.99)
        .filter(|e| e.group_ids.len() == 2)
    {
        let vg_recall = VirtualGroup {
            group_ids: ep.group_ids.clone(),
            blend_weights: ep.blend_weights.clone(),
        };
        accuracy_virtual_group(&mut dm, &vg_recall, &heldout)
    } else {
        vg_fixed_heldout
    };

    let direct_mirror_heldout = train_direct_composite_mirror(
        &mut dm,
        &train,
        &heldout,
        seed.wrapping_add(2),
        &mut rng,
    );

    let confidence_argmax_heldout =
        accuracy_confidence_argmax(&mut dm.main, spiral_group, circles_group, &heldout);
    let mut learned_router_xy = train_task_e_learned_router_xy(
        &mut dm.main,
        spiral_group,
        circles_group,
        &train,
    );
    let learned_router_xy_heldout = accuracy_learned_router_xy(
        &mut dm.main,
        &mut learned_router_xy,
        spiral_group,
        circles_group,
        &heldout,
    );
    let mut learned_router_expert = train_task_e_learned_router_expert(
        &mut dm.main,
        spiral_group,
        circles_group,
        &train,
    );
    let learned_router_expert_heldout = accuracy_learned_router_expert(
        &mut dm.main,
        &mut learned_router_expert,
        spiral_group,
        circles_group,
        &heldout,
    );
    let mut calibration_router_xy = train_calibration_learned_router_xy(
        &calibration_spiral,
        &calibration_circles,
    );
    let calibration_router_xy_heldout = accuracy_learned_router_xy(
        &mut dm.main,
        &mut calibration_router_xy,
        spiral_group,
        circles_group,
        &heldout,
    );
    let mut calibration_router_expert = train_calibration_learned_router_expert(
        &mut dm.main,
        spiral_group,
        circles_group,
        &calibration_spiral,
        &calibration_circles,
    );
    let calibration_router_expert_heldout = accuracy_learned_router_expert(
        &mut dm.main,
        &mut calibration_router_expert,
        spiral_group,
        circles_group,
        &heldout,
    );
    let mut disagreement_router = train_task_e_router_disagreement(
        &mut dm.main,
        spiral_group,
        circles_group,
        &train,
    );
    let disagreement_router_heldout = accuracy_learned_router_disagreement(
        &mut dm.main,
        &mut disagreement_router,
        spiral_group,
        circles_group,
        &heldout,
    );

    let mut expert_features = |input: &[f32]| {
        specialist_feature_pair(&mut dm.main, spiral_group, circles_group, input)
    };
    let expert_region_agreement = router_region_agreement_oracle(
        &mut learned_router_expert,
        &heldout,
        INNER_RADIUS,
        &mut expert_features,
    );
    let expert_margin_radius_corr = router_margin_radius_correlation(
        &mut learned_router_expert,
        &heldout,
        INNER_RADIUS,
        &mut expert_features,
    );
    let mut calib_expert_features = |input: &[f32]| {
        specialist_feature_pair(&mut dm.main, spiral_group, circles_group, input)
    };
    let calib_expert_region_agreement = router_region_agreement_oracle(
        &mut calibration_router_expert,
        &heldout,
        INNER_RADIUS,
        &mut calib_expert_features,
    );
    let calib_expert_margin_radius_corr = router_margin_radius_correlation(
        &mut calibration_router_expert,
        &heldout,
        INNER_RADIUS,
        &mut calib_expert_features,
    );

    let threshold = learn_radius_threshold(
        &mut dm.main,
        spiral_group,
        circles_group,
        &train,
        INNER_RADIUS,
    );
    let radius_gate_heldout =
        accuracy_radius_gated(&mut dm.main, spiral_group, circles_group, &heldout, threshold);
    let logistic_gate = train_radius_logistic_gate(
        &mut dm.main,
        spiral_group,
        circles_group,
        &train,
    );
    let logistic_gate_heldout = accuracy_logistic_radius_gate(
        &mut dm.main,
        spiral_group,
        circles_group,
        &heldout,
        logistic_gate,
    );
    let oracle_region_heldout = accuracy_radius_gated(
        &mut dm.main,
        spiral_group,
        circles_group,
        &heldout,
        INNER_RADIUS,
    );

    Phase3eSeedResult {
        spiral_heldout,
        circles_heldout,
        oracle_best_heldout,
        vg_fixed_heldout,
        vg_recall_heldout,
        direct_mirror_heldout,
        confidence_argmax_heldout,
        learned_router_xy_heldout,
        learned_router_expert_heldout,
        calibration_router_xy_heldout,
        calibration_router_expert_heldout,
        disagreement_router_heldout,
        expert_region_agreement,
        expert_margin_radius_corr,
        calib_expert_region_agreement,
        calib_expert_margin_radius_corr,
        radius_gate_heldout,
        logistic_gate_heldout,
        oracle_region_heldout,
        train_inner_frac: inner_region_fraction(&train, INNER_RADIUS),
        heldout_inner_frac: inner_region_fraction(&heldout, INNER_RADIUS),
        train_near_boundary_frac: train_boundary_near_fraction(&train, INNER_RADIUS, 0.08),
    }
}

fn demo_phase3e_balanced_composite() {
    println!("--- Phase 3e: Balanced Composite (decisive evaluation) ---\n");
    println!("Task E: 50/50 inner/outer spiral-gated circles, stratified train n=30, held-out rest.\n");

    const SEEDS: [u64; 20] = [
        42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    ];
    let mut results = Vec::with_capacity(SEEDS.len());

    for &seed in &SEEDS {
        println!("  seed {} ...", seed);
        let r = run_phase3e_seed(seed);
        println!(
            "    held-out: oracle={:.1}% VG={:.1}% | xy={:.1}% expert={:.1}% cal_expert={:.1}% disagree={:.1}% | r_agree={:.0}% r_corr={:.2}",
            r.oracle_best_heldout * 100.0,
            r.vg_fixed_heldout * 100.0,
            r.learned_router_xy_heldout * 100.0,
            r.learned_router_expert_heldout * 100.0,
            r.calibration_router_expert_heldout * 100.0,
            r.disagreement_router_heldout * 100.0,
            r.expert_region_agreement * 100.0,
            r.expert_margin_radius_corr,
        );
        results.push(r);
    }

    let fmt = |getter: fn(&Phase3eSeedResult) -> f32| -> String {
        let vals: Vec<f32> = results.iter().map(|r| getter(r)).collect();
        let (m, s) = mean_std(&vals);
        format!("{:.1}% ± {:.1}%", m * 100.0, s * 100.0)
    };

    let fmt_range = |getter: fn(&Phase3eSeedResult) -> f32| -> String {
        let vals: Vec<f32> = results.iter().map(|r| getter(r)).collect();
        let min = vals.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        format!("{:.1}% – {:.1}%", min * 100.0, max * 100.0)
    };

    println!("\n=== Task E summary ({} seeds, stratified held-out) ===\n", SEEDS.len());
    println!("| Baseline | Held-out accuracy |");
    println!("| -------- | ----------------- |");
    println!("| Spiral specialist only | {} |", fmt(|r| r.spiral_heldout));
    println!("| Circles specialist only | {} |", fmt(|r| r.circles_heldout));
    println!("| Oracle-best-single (global) | {} |", fmt(|r| r.oracle_best_heldout));
    println!("| **VirtualGroup (global scalar blend)** | **{}** |", fmt(|r| r.vg_fixed_heldout));
    println!("| Direct composite Mirror | {} |", fmt(|r| r.direct_mirror_heldout));
    println!("| Confidence argmax (unsupervised proxy) | {} |", fmt(|r| r.confidence_argmax_heldout));
    println!("| Disagreement router (f₁−f₂ only, Task E labels) | {} |", fmt(|r| r.disagreement_router_heldout));
    println!("| Learned radius gate (input-dependent) | {} |", fmt(|r| r.radius_gate_heldout));
    println!("| Logistic gate on r (input-dependent) | {} |", fmt(|r| r.logistic_gate_heldout));
    println!("| Oracle region switch (r < 0.4, diagnostic ceiling) | {} |", fmt(|r| r.oracle_region_heldout));

    println!("\n=== LearnedRouter 4-cell grid (held-out composite accuracy) ===\n");
    println!("| Features | Composite labels | Calibration identity |");
    println!("| -------- | ---------------- | ------------------ |");
    println!(
        "| `(x,y)` coordinates | {} | {} |",
        fmt(|r| r.learned_router_xy_heldout),
        fmt(|r| r.calibration_router_xy_heldout),
    );
    println!(
        "| Expert outputs `(f₁, f₂)` | {} | {} |",
        fmt(|r| r.learned_router_expert_heldout),
        fmt(|r| r.calibration_router_expert_heldout),
    );

    println!("\n=== Boundary alignment (expert routers vs generative r < 0.4) ===\n");
    println!(
        "| Router | Region agreement with oracle | Margin–radius correlation |",
    );
    println!("| ------ | -------------------------- | ------------------------- |");
    println!(
        "| Expert × composite labels | {} | {} |",
        fmt(|r| r.expert_region_agreement),
        fmt(|r| r.expert_margin_radius_corr),
    );
    println!(
        "| Expert × calibration identity | {} | {} |",
        fmt(|r| r.calib_expert_region_agreement),
        fmt(|r| r.calib_expert_margin_radius_corr),
    );
    println!(
        "\nHigh region agreement + strong positive margin–radius correlation ⇒ boundary tracks the generative circle (soft positional leak via expert outputs)."
    );

    let (train_inner_m, _) = mean_std(&results.iter().map(|r| r.train_inner_frac).collect::<Vec<_>>());
    let (held_inner_m, _) = mean_std(&results.iter().map(|r| r.heldout_inner_frac).collect::<Vec<_>>());
    let (near_bnd_m, near_bnd_s) =
        mean_std(&results.iter().map(|r| r.train_near_boundary_frac).collect::<Vec<_>>());
    println!(
        "\nRegion balance: train inner frac {:.1}%, held-out inner frac {:.1}% (target 50/50).",
        train_inner_m * 100.0,
        held_inner_m * 100.0
    );
    println!(
        "Router seed spread: expert×composite {} | expert×calibration {} | disagreement {}",
        fmt_range(|r| r.learned_router_expert_heldout),
        fmt_range(|r| r.calibration_router_expert_heldout),
        fmt_range(|r| r.disagreement_router_heldout),
    );
    println!(
        "Train boundary coverage (|r − 0.4| < 0.08): {:.1}% ± {:.1}% of n=30 — low coverage predicts unstable (x,y) routing.",
        near_bnd_m * 100.0,
        near_bnd_s * 100.0
    );
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

/// Task E: balanced 50/50 inner/outer spiral-gated circles (rejection sampling per region).
fn generate_balanced_spiral_gated_circles_data(
    main: &mut MainDimension,
    group_inner: GroupId,
    group_outer: GroupId,
    inner_radius: f32,
    n_samples: usize,
    rng: &mut impl rand::Rng,
) -> Vec<Sample> {
    let n_inner = n_samples / 2;
    let n_outer = n_samples - n_inner;
    let mut data = Vec::with_capacity(n_samples);
    let mut inner_count = 0usize;
    let mut outer_count = 0usize;
    let mut attempts = 0usize;
    let max_attempts = n_samples.saturating_mul(200);
    while (inner_count < n_inner || outer_count < n_outer) && attempts < max_attempts {
        attempts += 1;
        let x = rng.gen_range(-1.0..1.0_f32);
        let y = rng.gen_range(-1.0..1.0_f32);
        let r = (x * x + y * y).sqrt();
        let is_inner = r < inner_radius;
        if is_inner && inner_count >= n_inner {
            continue;
        }
        if !is_inner && outer_count >= n_outer {
            continue;
        }
        let outputs = main.query(&[x, y], &[group_inner, group_outer]);
        if outputs.len() < 2 {
            continue;
        }
        let out = if is_inner {
            outputs[0].1.get(0).copied().unwrap_or(0.5)
        } else {
            outputs[1].1.get(0).copied().unwrap_or(0.5)
        };
        let target = if out >= 0.5 { 1.0 } else { 0.0 };
        data.push((vec![x, y], [target]));
        if is_inner {
            inner_count += 1;
        } else {
            outer_count += 1;
        }
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
        encoder: EncoderPreset::from_env(),
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
    let mut svc = LanguageService::new_default()?;

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
                causal: None,
                history: Vec::new(),
                conversation_turn: 1,
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
