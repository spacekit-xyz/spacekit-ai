use growformer::dimension::{
    LanguageSample,
    action_target_to_type,
};
use growformer::dimension::language::{
    sentiment_lattice_index_body_with_causal, should_use_sentiment_joint_index, CausalAnnotation,
    DEFAULT_BRIDGE_DIM,
};
use growformer::clifford::GroupRotor;
use growformer::dimension::group_gen::{AlgebraicCodebook, HopfCompositionTable};
use growformer::dimension::paramecium::InfraciliaryLattice;
use growformer::reasoning::{CognitiveMap, ReasoningEngine};
use growformer::spectral::TokenDictionary;
use std::collections::HashMap;
use growformer::service::LanguageService;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand::rngs::StdRng;
use serde::Deserialize;
use clap::{Parser, Subcommand};
use std::path::{Path, PathBuf};

mod train_progress;

fn bump_train_phase(ui: &Option<train_progress::TrainUi>, i: &mut u64, label: &str) {
    if let Some(u) = ui {
        u.set_major_phase(*i, label.to_string());
    }
    *i += 1;
}

#[derive(Parser, Debug)]
#[command(
    name = "growformer",
    version,
    about = "Growformer — train and run specialized neural brains"
)]
struct CliRoot {
    #[command(subcommand)]
    command: Option<Commands>,
    #[command(flatten)]
    args: Args,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Write a starter *.gf.toml project manifest (paths, inference TOML, train defaults).
    Init {
        /// Output path (e.g. scripts/sentiment-analysis.gf.toml)
        #[arg(value_name = "PATH")]
        output: Option<PathBuf>,
        /// Default [project].name in the template
        #[arg(long, value_name = "TEXT")]
        name: Option<String>,
    },
}

#[derive(Parser, Debug)]
struct Args {
    /// Train the full neural brain end-to-end: encoder, router, classifier, generation heads.
    #[arg(long)]
    train_brain: bool,

    /// Run a quick validation of brain training (cap samples and epochs, then assert inference).
    #[arg(long)]
    validate_brain_training: bool,

    /// When validating or for quick runs: max training samples (0 = no cap).
    #[arg(long, value_name = "N", default_value_t = 0)]
    brain_max_samples: usize,

    /// When validating: max generation epochs (0 = use full schedule).
    #[arg(long, value_name = "N", default_value_t = 0)]
    brain_quick_gen_epochs: u32,

    /// Generation epochs for per-group NeuralEnvironment training (0 = auto).
    #[arg(long, value_name = "N", default_value_t = 0)]
    brain_gen_epochs: u32,

    /// Parallel replicas per generation task (different seeds, keep best).
    #[arg(long, value_name = "K", default_value_t = 1)]
    brain_gen_replicas: u32,

    /// Training epochs for brain training (router + classifier).
    #[arg(long, value_name = "N", default_value_t = 30)]
    brain_epochs: u32,

    /// Auto-configure all training parameters from the dataset.
    #[arg(long)]
    auto: bool,

    /// Output path for trained brain binary (default: brain.bin, or from --project).
    #[arg(long, value_name = "PATH")]
    brain_output: Option<String>,

    /// Brain package / agent display name (embedded in exported brain header and "who are you" replies).
    #[arg(long, value_name = "TEXT")]
    brain_name: Option<String>,

    /// Short description embedded in the exported brain package header.
    #[arg(long, value_name = "TEXT")]
    brain_description: Option<String>,

    /// Author or org string embedded as `BrainPackageHeader.author` (default: swtch.ai).
    #[arg(long, value_name = "TEXT")]
    brain_author: Option<String>,

    /// Run inference on a trained brain.
    #[arg(long)]
    infer: bool,

    /// Path to brain.bin for --infer / --retrain-gen (default: brain.bin, or [infer].brain in --project).
    #[arg(long, value_name = "PATH")]
    brain: Option<String>,

    /// Prompt for single-shot inference. Omit for interactive mode. If the text contains `$`
    /// (e.g. dollar amounts), use **single quotes** in the shell or use `--prompt-file` —
    /// double quotes cause the shell to expand `$` and drop the amount before it reaches the binary.
    /// For many prompts, use `--prompts-file` instead.
    #[arg(long, value_name = "TEXT", conflicts_with = "prompt_file", conflicts_with = "prompts_file")]
    prompt: Option<String>,

    /// Read the inference prompt from a UTF-8 file (exact text; avoids shell `$` expansion).
    #[arg(long, value_name = "PATH", conflicts_with = "prompt", conflicts_with = "prompts_file")]
    prompt_file: Option<PathBuf>,

    /// Batch inference: one prompt per non-empty line (plain text; optional matching `"`/`'`/`“…”` wrappers are stripped). Lines starting with `#` are comments. Use with `--infer` (`-v` for traces).
    #[arg(long, value_name = "PATH", conflicts_with = "prompt", conflicts_with = "prompt_file")]
    prompts_file: Option<PathBuf>,

    /// Retrain only the gen env for a specific group index (loads existing brain, retrains one group, re-exports).
    #[arg(long, value_name = "GROUP_IDX")]
    retrain_gen: Option<usize>,

    /// Custom data directory containing train_*.jsonl files. When set, ONLY this
    /// directory is loaded (skips default m5/agent/routekit). Use for focused micro-brains.
    #[arg(long, value_name = "DIR")]
    data_dir: Option<String>,

    /// UTF-8 TOML embedded in the exported brain as the inference plugins manifest (format v2).
    /// Top-level tables, e.g. `[sentiment]`, `[language_detection]`, `[badwords]`.
    #[arg(long, value_name = "PATH")]
    brain_plugins_toml: Option<PathBuf>,

    /// Growformer project manifest (*.gf.toml): merges paths and metadata before train/infer.
    #[arg(long, value_name = "PATH")]
    project: Option<PathBuf>,

    /// Inference shortcut rules + numeric gates TOML (same as [inference].toml in a *.gf.toml).
    #[arg(long, value_name = "PATH")]
    inference_toml: Option<PathBuf>,

    /// Baseline inference TOML for merging empty `[rules]` arrays (same as [inference].defaults_toml).
    #[arg(long, value_name = "PATH")]
    inference_defaults_toml: Option<PathBuf>,

    /// Enable MetaCodebook (Stage 2b) and code lattice training from `expected_code` in JSONL. Off by default; use for a standalone code brain (see `scripts/code.gf.toml`).
    #[arg(long)]
    train_code_lattice: bool,

    /// Disable stderr progress bars (plain logs only; use for CI and when capturing stderr).
    #[arg(long)]
    no_progress: bool,

    /// Verbose inference: routing traces, metacognition, topic graph, lattice shortcuts (`--infer` is quiet by default).
    #[arg(long, short = 'v')]
    verbose: bool,
}

#[derive(Default)]
struct GfOverlay {
    train_auto: Option<bool>,
    /// Mirrors `[train].code_brain` in *.gf.toml
    train_code_lattice: Option<bool>,
    brain_epochs: Option<u32>,
    brain_gen_epochs: Option<u32>,
    brain_gen_replicas: Option<u32>,
}

fn apply_gf_project(args: &mut Args, project_path: &Path) -> Result<GfOverlay, String> {
    let mut overlay = GfOverlay::default();
    let gf = growformer::project_gf::read_project_file(project_path)?;
    if gf.schema_version != 1 {
        return Err(format!(
            "unsupported schema_version {} in {} (expected 1)",
            gf.schema_version,
            project_path.display()
        ));
    }
    let base = growformer::project_gf::manifest_base_dir(project_path);

    if let Some(p) = &gf.project {
        if args.brain_name.is_none() {
            args.brain_name = p.name.clone();
        }
        if args.brain_description.is_none() {
            args.brain_description = p.description.clone();
        }
        if args.brain_author.is_none() {
            args.brain_author = p.author.clone();
        }
    }

    if let Some(tr) = &gf.train {
        overlay.train_auto = tr.auto;
        overlay.train_code_lattice = tr.code_brain;
        overlay.brain_epochs = tr.brain_epochs;
        overlay.brain_gen_epochs = tr.brain_gen_epochs;
        overlay.brain_gen_replicas = tr.brain_gen_replicas;

        if args.data_dir.is_none() {
            if let Some(d) = &tr.data_dir {
                args.data_dir = Some(
                    growformer::project_gf::resolve_against(&base, d)
                        .to_string_lossy()
                        .into_owned(),
                );
            }
        }
        if args.brain_plugins_toml.is_none() {
            if let Some(p) = &tr.brain_plugins_toml {
                args.brain_plugins_toml = Some(growformer::project_gf::resolve_against(&base, p));
            }
        }
        if args.brain_output.is_none() {
            if let Some(o) = &tr.brain_output {
                args.brain_output = Some(
                    growformer::project_gf::resolve_against(&base, o)
                        .to_string_lossy()
                        .into_owned(),
                );
            }
        }
    }

    if let Some(inf) = &gf.inference {
        if args.inference_toml.is_none() {
            if let Some(t) = inf.toml.as_deref() {
                args.inference_toml = Some(growformer::project_gf::resolve_against(&base, t));
            }
        }
        if args.inference_defaults_toml.is_none() {
            if let Some(t) = inf.defaults_toml.as_deref() {
                args.inference_defaults_toml =
                    Some(growformer::project_gf::resolve_against(&base, t));
            }
        }
    }

    if let Some(inf) = &gf.infer {
        if args.brain.is_none() {
            if let Some(b) = &inf.brain {
                args.brain = Some(
                    growformer::project_gf::resolve_against(&base, b)
                        .to_string_lossy()
                        .into_owned(),
                );
            }
        }
    }

    Ok(overlay)
}

fn run_gf_init(output: Option<PathBuf>, name: Option<String>) -> Result<(), String> {
    let path = output.unwrap_or_else(|| PathBuf::from("Growformer.gf.toml"));
    let default_name = name.unwrap_or_else(|| "MyBrain".to_string());
    let body = growformer::project_gf::init_template(&default_name);
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("create_dir_all {}: {}", parent.display(), e))?;
        }
    }
    std::fs::write(&path, body.as_bytes())
        .map_err(|e| format!("write {}: {}", path.display(), e))?;
    println!("Wrote {}", path.display());
    Ok(())
}

/// How to drive `run_inference`: REPL, one prompt, or many.
enum InferMode {
    Interactive,
    Single(String),
    Batch(Vec<String>),
}

/// Remove one layer of matching ASCII or curly quotes so list files can use `“...”` wrappers.
fn strip_outer_wrapping_quotes(line: &str) -> String {
    let t = line.trim();
    let chars: Vec<char> = t.chars().collect();
    if chars.len() < 2 {
        return t.to_string();
    }
    let first = chars[0];
    let last = *chars.last().unwrap();
    let paired = matches!(
        (first, last),
        ('"', '"')
            | ('\u{201c}', '\u{201d}')
            | ('\u{2018}', '\u{2019}')
            | ('\'', '\'')
    );
    if !paired {
        return t.to_string();
    }
    chars[1..chars.len() - 1].iter().collect::<String>().trim().to_string()
}

fn resolve_infer_mode(args: &Args) -> Result<InferMode, String> {
    if let Some(path) = &args.prompts_file {
        let s = std::fs::read_to_string(path)
            .map_err(|e| format!("--prompts-file {}: {}", path.display(), e))?;
        let lines: Vec<String> = s
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .map(strip_outer_wrapping_quotes)
            .filter(|l| !l.is_empty())
            .collect();
        if lines.is_empty() {
            return Err(format!(
                "--prompts-file {}: no prompts (use non-empty lines; lines starting with `#` are comments)",
                path.display()
            ));
        }
        return Ok(InferMode::Batch(lines));
    }
    if let Some(path) = &args.prompt_file {
        let s = std::fs::read_to_string(path)
            .map_err(|e| format!("--prompt-file {}: {}", path.display(), e))?;
        return Ok(InferMode::Single(s.trim().to_string()));
    }
    match &args.prompt {
        Some(p) => Ok(InferMode::Single(p.clone())),
        None => Ok(InferMode::Interactive),
    }
}

fn main() {
    let cli = CliRoot::parse();

    if let Some(cmd) = &cli.command {
        match cmd {
            Commands::Init { output, name } => {
                if let Err(e) = run_gf_init(output.clone(), name.clone()) {
                    eprintln!("{}", e);
                    std::process::exit(1);
                }
                return;
            }
        }
    }

    let mut args = cli.args;

    let gf_overlay = match args.project.clone() {
        Some(ref p) => match apply_gf_project(&mut args, p) {
            Ok(o) => Some(o),
            Err(e) => {
                eprintln!("{}", e);
                std::process::exit(1);
            }
        },
        None => None,
    };

    growformer::inference::set_inference_toml_cli_paths(
        args.inference_toml.clone(),
        args.inference_defaults_toml.clone(),
    );

    let brain_out = args
        .brain_output
        .clone()
        .unwrap_or_else(|| "brain.bin".to_string());
    let brain_path = args.brain.clone().unwrap_or_else(|| "brain.bin".to_string());
    let auto = args.auto || gf_overlay.as_ref().and_then(|o| o.train_auto).unwrap_or(false);
    let brain_epochs = gf_overlay
        .as_ref()
        .and_then(|o| o.brain_epochs)
        .unwrap_or(args.brain_epochs);
    let brain_gen_epochs = gf_overlay
        .as_ref()
        .and_then(|o| o.brain_gen_epochs)
        .unwrap_or(args.brain_gen_epochs);
    let brain_gen_replicas = gf_overlay
        .as_ref()
        .and_then(|o| o.brain_gen_replicas)
        .unwrap_or(args.brain_gen_replicas);
    let train_code_lattice = args.train_code_lattice
        || gf_overlay
            .as_ref()
            .and_then(|o| o.train_code_lattice)
            .unwrap_or(false);

    // Quiet inference by default: suppress diagnostic `infer_trace!` lines until `--verbose`.
    let infer_quiet = args.infer && !args.verbose;
    growformer::infer_log::set_infer_trace_quiet(infer_quiet);

    // Initialize topic knowledge graph + optional sentiment NL overlay (same directory).
    let kg_path = "data/knowledge_graph.toml";
    if let Err(e) = growformer::growformer_lang::try_init_topic_graph_bundle(kg_path) {
        eprintln!("Warning: failed to load topic graph: {}", e);
    }

    if args.train_brain || args.validate_brain_training {
        println!("=============================================================");
        println!("  Growformer — Brain Training");
        println!("=============================================================\n");

        let max_samples = if args.validate_brain_training && args.brain_max_samples == 0 {
            100
        } else {
            args.brain_max_samples
        };
        let quick_gen_epochs = if args.validate_brain_training && args.brain_quick_gen_epochs == 0 {
            15
        } else {
            args.brain_quick_gen_epochs
        };
        if let Err(e) = train_brain(
            brain_epochs,
            &brain_out,
            max_samples,
            quick_gen_epochs,
            brain_gen_epochs,
            brain_gen_replicas,
            args.validate_brain_training,
            auto,
            args.brain_name.as_deref(),
            args.brain_description.as_deref(),
            args.brain_author.as_deref(),
            args.data_dir.as_deref(),
            args.brain_plugins_toml.as_deref(),
            train_code_lattice,
            args.no_progress,
        ) {
            eprintln!("Failed to train brain: {}", e);
            std::process::exit(1);
        }
    } else if let Some(group_idx) = args.retrain_gen {
        println!("=============================================================");
        println!("  Growformer — Retrain Gen Group {}", group_idx);
        println!("=============================================================\n");
        if let Err(e) = retrain_single_gen(
            group_idx,
            &brain_path,
            &brain_out,
            brain_gen_epochs,
            brain_gen_replicas,
            auto,
            args.brain_name.as_deref(),
            args.brain_description.as_deref(),
            args.brain_author.as_deref(),
            args.brain_plugins_toml.as_deref(),
        ) {
            eprintln!("Retrain failed: {}", e);
            std::process::exit(1);
        }
    } else if args.infer {
        let infer_mode = match resolve_infer_mode(&args) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("{}", e);
                std::process::exit(1);
            }
        };
        if let Err(e) = run_inference(&brain_path, infer_mode) {
            eprintln!("Inference failed: {}", e);
            std::process::exit(1);
        }
    } else {
        println!("Growformer — train and run specialized neural brains\n");
        println!("Usage:");
        println!("  Project:   cargo run --release -- init [PATH] [--name MyBrain]");
        println!("  Train:     cargo run --release -- --train-brain --project scripts/foo.gf.toml");
        println!("  Code brain: cargo run --release -- --train-brain --project scripts/code.gf.toml");
        println!("           or --train-code-lattice (MetaCodebook + expected_code lattices)");
        println!("  Train:     cargo run --release -- --train-brain [--auto] [--inference-toml path]");
        println!("  Retrain:   cargo run --release -- --retrain-gen 1 [--auto]");
        println!("  Infer:     cargo run --release -- --infer [--project scripts/foo.gf.toml] [-v]");
        println!("             --prompts-file path.txt  (one prompt per line; `#` comments)");
        println!("  Demos:     cargo run --bin growformer-demos -- --help");
        println!("\nRun with --help for all options.");
        std::process::exit(1);
    }
}

// =============================================================================
// Retrain a single gen group: loads existing brain, retrains one group, re-exports
// =============================================================================

/// Load TOML from `--brain-plugins-toml` and store for the next [`LanguageService::export_brain`].
fn apply_optional_brain_plugins_toml(
    svc: &mut LanguageService,
    path: Option<&std::path::Path>,
) -> Result<(), String> {
    let Some(path) = path else {
        return Ok(());
    };
    let s = std::fs::read_to_string(path)
        .map_err(|e| format!("--brain-plugins-toml {}: {}", path.display(), e))?;
    let m = growformer::inference::parse_plugins_manifest_bytes(s.as_bytes())?;
    if !m.has_embeddable_content() {
        println!(
            "  [brain-plugins] note: {} has no non-empty plugin tables (export may still add defaults for sentiment-shaped brains)",
            path.display()
        );
    }
    svc.set_brain_plugins_manifest(Some(m));
    println!("  [brain-plugins] will embed manifest from {}", path.display());
    Ok(())
}

fn apply_brain_package_cli(
    svc: &mut LanguageService,
    name: Option<&str>,
    description: Option<&str>,
    author: Option<&str>,
) {
    let mut n = svc.agent_name.clone();
    let mut c = svc.agent_creator.clone();
    if let Some(x) = name {
        n = x.to_string();
    }
    if let Some(x) = author {
        c = x.to_string();
    }
    if name.is_some() || author.is_some() {
        svc.set_identity(&n, &c);
    }
    if let Some(d) = description {
        svc.set_brain_package_description(d);
    }
}

fn retrain_single_gen(
    target_group: usize,
    brain_path: &str,
    output_path: &str,
    gen_epochs_override: u32,
    gen_replicas: u32,
    auto: bool,
    brain_name: Option<&str>,
    brain_description: Option<&str>,
    brain_author: Option<&str>,
    brain_plugins_toml: Option<&std::path::Path>,
) -> Result<(), String> {
    let data = std::fs::read(brain_path)
        .map_err(|e| format!("Failed to read {}: {}", brain_path, e))?;

    let mut svc = LanguageService::new_default()?;
    svc.load_brain(&data)?;
    apply_brain_package_cli(&mut svc, brain_name, brain_description, brain_author);
    apply_optional_brain_plugins_toml(&mut svc, brain_plugins_toml)?;
    println!("Loaded brain from {} ({} KB)", brain_path, data.len() / 1024);

    let dm = svc.active_dm();
    let num_groups = dm.main.group_order.len();
    println!("  Groups: {}, gen envs: {:?}", num_groups, dm.group_gen_envs.keys().collect::<Vec<_>>());

    if !dm.group_gen_envs.contains_key(&target_group) {
        return Err(format!("Group {} not found in brain (available: {:?})",
            target_group, dm.group_gen_envs.keys().collect::<Vec<_>>()));
    }

    // Load training data
    let samples = load_all_m5_training_data()?;
    println!("Loaded {} training samples", samples.len());

    // Build group lookup from the loaded brain's group names
    let brain_group_names: Vec<String> = dm.main.group_order.iter()
        .map(|gid| dm.main.groups.get(gid).map(|g| g.task_name.clone()).unwrap_or_default())
        .collect();
    let group_lookup = build_group_lookup(&brain_group_names);
    let group_map = |s: &LanguageSample| -> usize {
        action_target_to_group(s.action_target.as_deref(), &group_lookup, num_groups)
    };

    // Embed only the target group's gen data, applying the same Clifford + understanding
    // conditioning used at inference time so lattice cosine similarity is high.
    println!("\n--- Computing embeddings for group {} ---", target_group);
    let runtime = &svc.dm.language_runtime;
    let mut gen_pairs: Vec<(Vec<f32>, String, String)> = Vec::new();
    for s in &samples {
        if group_map(s) != target_group { continue; }
        if let Some(r) = s.expected_response.as_deref() {
            match runtime.encode_and_bridge(&s.text) {
                Ok((raw, bridged)) => {
                    let cond = svc.dm.adapt_for_group_clifford(
                        target_group, &bridged.routed_vector, &raw,
                        growformer::dimension::group_gen::GEN_COND_DIM,
                    );
                    let lattice_text = if should_use_sentiment_joint_index(s) {
                        sentiment_lattice_index_body_with_causal(&s.text, r, s.causal.as_ref())
                    } else {
                        r.to_string()
                    };
                    gen_pairs.push((cond, lattice_text, s.semantic_intent.clone()));
                }
                Err(_) => {
                    let lattice_text = if should_use_sentiment_joint_index(s) {
                        sentiment_lattice_index_body_with_causal(&s.text, r, s.causal.as_ref())
                    } else {
                        r.to_string()
                    };
                    gen_pairs.push((
                        vec![0.0; growformer::dimension::group_gen::GEN_COND_DIM],
                        lattice_text,
                        s.semantic_intent.clone(),
                    ));
                }
            }
        }
    }
    println!("  {} gen samples for group {}", gen_pairs.len(), target_group);

    if gen_pairs.is_empty() {
        return Err(format!("No gen training data for group {}", target_group));
    }

    // Auto-config
    let auto_cfg = if auto {
        let profile = profile_training_data(&samples, &group_map, num_groups);
        Some(auto_configure(&profile, false))
    } else {
        None
    };

    let _gen_epochs: usize = if gen_epochs_override > 0 {
        gen_epochs_override as usize
    } else if let Some(ac) = &auto_cfg {
        ac.gen_epochs
    } else {
        1500
    };
    let _k_replicas = if let Some(ac) = &auto_cfg { ac.replicas } else { (gen_replicas as usize).max(1) };
    let gen_overrides = auto_cfg.as_ref().map(|ac| {
        growformer::dimension::group_gen::GenEnvOverrides {
            max_tokens: Some(ac.max_tokens),
            hidden: Some(ac.gen_hidden),
            k: Some(ac.gen_k),
            max_synapses: Some(ac.max_synapses),
            energy_budget: Some(ac.energy_budget),
            hex_mode: Some(true),
            ..Default::default()
        }
    });

    // let early_stop_window = auto_cfg.as_ref().map(|ac| ac.early_stop_window).unwrap_or(0);
    // let early_stop_min_imp = auto_cfg.as_ref().map(|ac| ac.early_stop_min_improvement).unwrap_or(0.0);
    // let early_stop_min_ep = auto_cfg.as_ref().map(|ac| ac.early_stop_min_epochs).unwrap_or(0);

    // Build dictionary, codebook, Hopf table for the target group
    use growformer::dimension::group_gen::{bits_for_dict, MAX_TOKENS};
    let effective_max_tokens = gen_overrides.as_ref().and_then(|o| o.max_tokens).unwrap_or(MAX_TOKENS);

    let texts: Vec<&str> = gen_pairs.iter().map(|(_, t, _)| t.as_str()).collect();
    let embs: Vec<&[f32]> = gen_pairs.iter().map(|(e, _, _)| e.as_slice()).collect();

    let max_dict = if texts.len() >= 100 { 2048 } else { 1024 };
    let dict = TokenDictionary::build(&texts, max_dict);
    let bits = bits_for_dict(dict.len());
    println!("  dict: {} tokens from {} texts, {} bits/token, output={}",
        dict.len(), texts.len(), bits, effective_max_tokens * bits);

    let base_archetypes = 16usize;
    let gen_max_arch = if texts.len() > 150 {
        base_archetypes * 6
    } else if texts.len() > 50 {
        base_archetypes * 2
    } else {
        base_archetypes
    };
    let cb = AlgebraicCodebook::build(&texts, &dict, gen_max_arch, Some(&embs));
    let mode = if cb.has_prototypes() { "SLOT-ONLY" } else { "FULL" };
    println!("  codebook: {} archetypes (max={}), {} slots max, {} total/{} slot bits [{}]",
        cb.archetypes.len(), gen_max_arch, cb.max_slot_count, cb.total_bits, cb.slot_only_bits, mode);

    // Build Hopf composition table
    let hopf_segments = 3;
    let hopf = if cb.has_prototypes() && cb.archetypes.len() >= 2 {
        let seqs: Vec<Vec<u16>> = texts.iter().map(|t| dict.encode(t)).collect();
        let mut clusters = vec![vec![]; cb.archetypes.len()];
        for (i, seq) in seqs.iter().enumerate() {
            let (ai, _) = cb.match_best(seq);
            clusters[ai].push(i);
        }
        clusters.retain(|c| !c.is_empty());
        let hopf = HopfCompositionTable::build(&cb, Some(&embs), &clusters, hopf_segments);
        println!("  Hopf table: {} fragments, {} transition entries",
            hopf.fragments.len(), hopf.transition.len());
        Some(hopf)
    } else {
        None
    };

    // Build IndexedGenEnv: one-pass Paramecium lattice development
    use growformer::dimension::group_gen::IndexedGenEnv as RetrainIndexedGenEnv;
    println!("\n--- Building indexed gen g{} ({} pairs, Paramecium lattice) ---", target_group, gen_pairs.len());

    let mut indexed_env = RetrainIndexedGenEnv::from_tagged_parts(
        dict.clone(),
        cb.clone(),
        hopf.unwrap_or_default(),
        &gen_pairs,
        0.85,
    );
    indexed_env.freeze();
    println!("  Indexed gen g{}: {} lattice programs, frozen", target_group, indexed_env.program_count());

    // Replace the gen env in the brain
    let dm = svc.active_dm_mut();
    dm.group_gen_envs.insert(target_group, indexed_env);
    println!("\n  Replaced gen env for group {}", target_group);

    // Re-export
    println!("\n--- Exporting Brain ---");
    let brain_bytes = svc.export_brain()?;
    let size_kb = brain_bytes.len() / 1024;
    std::fs::write(output_path, &brain_bytes).map_err(|e| format!("write failed: {}", e))?;
    println!("Brain exported: {} ({} KB)", output_path, size_kb);

    Ok(())
}

// =============================================================================
// Inference: load brain.bin and run prompts
// =============================================================================

fn run_inference(brain_path: &str, mode: InferMode) -> Result<(), String> {
    let data = std::fs::read(brain_path)
        .map_err(|e| format!("Failed to read {}: {}", brain_path, e))?;

    let mut rt = growformer::runtime::Runtime::from_brain_bytes(&data)?;

    let info = rt.brain_info();
    let trace = growformer::infer_log::infer_trace_enabled();
    if trace {
        println!("Brain loaded: {}", brain_path);
        println!("  Agent: {} (by {})", info.agent_name, info.agent_creator);
        println!("  Groups: {}", info.num_groups);
        println!("  Router: {}", info.has_router);
        println!("  Classifier: {}", info.has_classifier);
        println!("  Gen envs: {} groups", info.gen_envs);
        println!("  Code envs: {} groups", info.code_envs);
        if let Some(ref p) = info.inference_profile {
            println!("  Inference profile: {}", p);
        }
        if info.has_inference_plugins {
            println!("  Inference plugins: embedded TOML manifest (see BrainPluginsManifest)");
        }

        let dm = rt.svc.active_dm();
        for (gidx, env) in &dm.group_gen_envs {
            let hopf_info = if env.hopf_table.is_some() { "hopf=yes" } else { "hopf=no" };
            let cb_info = env.codebook.as_ref().map(|cb| format!("proto={} arch={}", cb.has_prototypes(), cb.archetypes.len())).unwrap_or_else(|| "no-codebook".to_string());
            let sub_names: Vec<&str> = env.topic_subindex.iter().map(|t| t.topic_name.as_str()).collect();
            println!("    gen[{}]: {} tokens, {} progs, {} topics {:?}, {}, {}",
                gidx, env.dictionary.len(), env.program_count(), env.topic_subindex.len(), sub_names, hopf_info, cb_info);
        }
        for (gidx, env) in &dm.group_code_envs {
            println!("    code[{}]: {} tokens in dict, {} lattice programs",
                gidx, env.dictionary.len(), env.program_count());
        }
    } else {
        println!(
            "Brain loaded: {} · {} ({} group{})",
            brain_path,
            info.agent_name,
            info.num_groups,
            if info.num_groups == 1 { "" } else { "s" }
        );
    }

    match mode {
        InferMode::Interactive => run_conversation_repl(&mut rt),
        InferMode::Single(prompt_text) => run_single_prompt(&mut rt, &prompt_text, false),
        InferMode::Batch(prompts) => {
            let trace = growformer::infer_log::infer_trace_enabled();
            for (i, prompt_text) in prompts.iter().enumerate() {
                if trace {
                    println!("\n--- prompt {} / {} ---", i + 1, prompts.len());
                    println!("{}", prompt_text);
                } else {
                    println!("\n[{}] {}", i + 1, prompt_text);
                }
                run_single_prompt(&mut rt, prompt_text, true);
            }
        }
    }

    Ok(())
}

/// `skip_duplicate_prompt_echo`: when true, the caller already printed the prompt (batch mode).
fn run_single_prompt(rt: &mut growformer::runtime::Runtime, prompt: &str, skip_duplicate_prompt_echo: bool) {
    let trace = growformer::infer_log::infer_trace_enabled();
    if trace && !skip_duplicate_prompt_echo {
        println!("\n--- prompt ---\n{}", prompt);
    }
    match rt.prompt(prompt) {
        Ok(resp) => {
            if trace {
                println!(
                    "  route: {} (conf={:.2}) group={:?}",
                    resp.action_type, resp.action_confidence, resp.target_group
                );
            }
            if !resp.text.is_empty() {
                if trace {
                    println!("  gen (conf={:.2}): {}", resp.confidence, resp.text);
                } else {
                    println!("{}", resp.text);
                }
            }
        }
        Err(e) => eprintln!("  gen error: {}", e),
    }

    match rt.codegen(prompt) {
        Ok(Some(code)) if !code.code.is_empty() => {
            println!("  code [{}]: {}", code.kind, code.code);
        }
        Ok(_) => {}
        Err(e) => eprintln!("  code error: {}", e),
    }
}

fn execute_tool(call: &growformer::dimension::tool::ToolCallInfo) -> growformer::dimension::tool::ToolResult {
    growformer::tools_builtin::execute_tool(call)
}

fn run_conversation_repl(rt: &mut growformer::runtime::Runtime) {
    let svc = &mut rt.svc;
    let ocean = svc.personality.as_vec();
    println!("\n=== Growformer Conversation REPL ===");
    println!("  Agent: {} (by {})", svc.agent_name, svc.agent_creator);
    println!("  Personality [O={:.1} C={:.1} E={:.1} A={:.1} N={:.1}]",
        ocean[0], ocean[1], ocean[2], ocean[3], ocean[4]);
    println!();
    println!("Commands:");
    println!("  /personality <preset>   Switch: assistant, creative, engineer, analyst");
    println!("  /ocean O C E A N        Set custom OCEAN values (0.0-1.0)");
    println!("  /reset                  Clear conversation history");
    println!("  /history                Show conversation history");
    println!("  /single <prompt>        Single-shot (no conversation context)");
    println!("  /status                 Show brain + personality info");
    println!("  /index <path>           Index project directory into Leech lattice");
    println!("  /project [file]         Show project model / related entities");
    println!("  /paramecium <prompt>    Lattice-only inference (no neural substrate)");
    println!("  quit | exit             Exit");
    println!();

    let stdin = std::io::stdin();
    loop {
        eprint!("[turn {}] > ", rt.turn_count() + 1);
        let _ = std::io::Write::flush(&mut std::io::stderr());
        let mut line = String::new();
        match stdin.read_line(&mut line) {
            Ok(0) => break,
            Ok(_) => {
                let trimmed = line.trim();
                if trimmed.is_empty() { continue; }
                if trimmed == "quit" || trimmed == "exit" { break; }

                if let Some(cmd) = trimmed.strip_prefix('/') {
                    handle_repl_command(rt, cmd);
                    continue;
                }

                if let Some(tool_call) = rt.try_tool_call(trimmed) {
                    let result = execute_tool(&tool_call);
                    let status = if result.success { "ok" } else { "error" };
                    println!("  [tool: {} ({})]", tool_call.tool_name, status);
                    if !result.output.is_empty() {
                        for line in result.output.lines().take(20) {
                            println!("  | {}", line);
                        }
                        if result.output.lines().count() > 20 {
                            println!("  | ... (truncated)");
                        }
                    }

                    match rt.respond_with_tool_result(trimmed, &result) {
                        Ok(resp) if !resp.text.is_empty() && !resp.text.starts_with("[tool_call:") => {
                            println!();
                            println!("  {}", resp.text);
                        }
                        _ => {}
                    }
                } else {
                    match rt.converse(trimmed) {
                        Ok(resp) => {
                            if growformer::infer_log::infer_trace_enabled() {
                                if resp.target_group.is_some() {
                                    eprint!(
                                        "  [route: {} g={} conf={:.2}] ",
                                        resp.action_type,
                                        resp.target_group.unwrap(),
                                        resp.action_confidence
                                    );
                                }
                                println!();
                            }
                            if !resp.text.is_empty() {
                                if growformer::infer_log::infer_trace_enabled() {
                                    println!("  {} (conf={:.2})", resp.text, resp.confidence);
                                } else {
                                    println!("{}", resp.text);
                                }
                            }

                            match rt.codegen(trimmed) {
                                Ok(Some(code)) if !code.code.is_empty() => {
                                    println!("  code [{}]: {}", code.kind, code.code);
                                }
                                _ => {}
                            }
                        }
                        Err(e) => eprintln!("  error: {}", e),
                    }
                }
                println!();
            }
            Err(e) => {
                eprintln!("Read error: {}", e);
                break;
            }
        }
    }
}

fn handle_repl_command(rt: &mut growformer::runtime::Runtime, cmd: &str) {
    use growformer::service::OceanProfile;

    let parts: Vec<&str> = cmd.split_whitespace().collect();
    match parts.first().copied() {
        Some("personality") | Some("p") => {
            let profile = match parts.get(1).copied() {
                Some("assistant") => {
                    println!("  Personality: assistant (balanced, professional)");
                    Some(OceanProfile::assistant())
                }
                Some("creative") => {
                    println!("  Personality: creative (open, enthusiastic)");
                    Some(OceanProfile::creative())
                }
                Some("engineer") => {
                    println!("  Personality: engineer (precise, structured)");
                    Some(OceanProfile::engineer())
                }
                Some("analyst") => {
                    println!("  Personality: analyst (cautious, thorough)");
                    Some(OceanProfile::analyst())
                }
                _ => {
                    println!("  Usage: /personality <assistant|creative|engineer|analyst>");
                    None
                }
            };
            if let Some(p) = profile {
                rt.set_personality(p);
            }
            let v = rt.personality().as_vec();
            println!("  [O={:.1} C={:.1} E={:.1} A={:.1} N={:.1}]", v[0], v[1], v[2], v[3], v[4]);
        }
        Some("ocean") => {
            if parts.len() == 6 {
                let vals: Vec<f32> = parts[1..6].iter()
                    .filter_map(|s| s.parse::<f32>().ok())
                    .collect();
                if vals.len() == 5 {
                    rt.set_personality(OceanProfile {
                        openness: vals[0].clamp(0.0, 1.0),
                        conscientiousness: vals[1].clamp(0.0, 1.0),
                        extraversion: vals[2].clamp(0.0, 1.0),
                        agreeableness: vals[3].clamp(0.0, 1.0),
                        neuroticism: vals[4].clamp(0.0, 1.0),
                    });
                    let v = rt.personality().as_vec();
                    println!("  Custom OCEAN: [O={:.1} C={:.1} E={:.1} A={:.1} N={:.1}]",
                        v[0], v[1], v[2], v[3], v[4]);
                } else {
                    println!("  Usage: /ocean 0.5 0.7 0.5 0.6 0.3");
                }
            } else {
                let v = rt.personality().as_vec();
                println!("  Current: [O={:.1} C={:.1} E={:.1} A={:.1} N={:.1}]", v[0], v[1], v[2], v[3], v[4]);
                println!("  Usage: /ocean <O> <C> <E> <A> <N>  (each 0.0-1.0)");
            }
        }
        Some("reset") => {
            rt.reset_conversation();
            println!("  Conversation cleared.");
        }
        Some("history") | Some("h") => {
            let svc = &rt.svc;
            if svc.conversation.is_empty() {
                println!("  (no conversation history)");
            } else {
                for (i, turn) in svc.conversation.history.iter().enumerate() {
                    let role = match turn.role {
                        growformer::service::TurnRole::User => "user",
                        growformer::service::TurnRole::Agent => "agent",
                    };
                    let display = if turn.text.len() > 80 {
                        format!("{}...", &turn.text[..77])
                    } else {
                        turn.text.clone()
                    };
                    println!("  [{}] {}: {}", i + 1, role, display);
                }
            }
        }
        Some("single") | Some("s") => {
            let prompt = parts[1..].join(" ");
            if prompt.is_empty() {
                println!("  Usage: /single <prompt text>");
            } else {
                run_single_prompt(rt, &prompt, false);
            }
        }
        Some("status") => {
            let svc = &rt.svc;
            let dm = svc.active_dm();
            println!("  Agent: {} (by {})", svc.agent_name, svc.agent_creator);
            let v = svc.personality.as_vec();
            println!("  Personality: [O={:.1} C={:.1} E={:.1} A={:.1} N={:.1}]", v[0], v[1], v[2], v[3], v[4]);
            println!("  Conversation turns: {}", svc.conversation.turn_count());
            println!("  EMA alpha: {:.2} (base) -> {:.2} (modulated)",
                dm.language_runtime.config.ema_alpha,
                svc.personality.modulated_ema_alpha(dm.language_runtime.config.ema_alpha));
            println!("  Hopf diversity bonus: {:.2}", svc.personality.hopf_diversity_bonus());
            println!("  Groups: {}, Gen envs: {}, Code envs: {}",
                dm.main.group_order.len(), dm.group_gen_envs.len(), dm.group_code_envs.len());
            println!("  Project: {}", svc.project_status());
        }
        Some("index") => {
            let path = parts.get(1).copied().unwrap_or(".");
            let path_buf = std::path::Path::new(path);
            if !path_buf.exists() {
                println!("  Path not found: {}", path_buf.display());
            } else {
                let svc = &mut rt.svc;
                let mut count = 0usize;
                index_directory(svc, path_buf, &mut count);
                println!("  Indexed {} files (hybrid AST-lite + semantic + relational)", count);

                let git_dir = find_git_root(path_buf);
                if let Some(ref git_root) = git_dir {
                    match std::process::Command::new("git")
                        .args(["log", "--name-only", "--pretty=format:---", "-200"])
                        .current_dir(git_root)
                        .output()
                    {
                        Ok(output) if output.status.success() => {
                            let log = String::from_utf8_lossy(&output.stdout);
                            svc.load_git_history(&log);
                            println!("  Loaded git history for edit correlation");
                        }
                        _ => {
                            println!("  (no git history available — dims 12-15 will be zero)");
                        }
                    }
                }

                println!("  {}", svc.project_status());
            }
        }
        Some("project") => {
            let svc = &rt.svc;
            if svc.project_model.entity_count() == 0 {
                println!("  No project indexed. Use /index <path> first.");
            } else {
                println!("  {}", svc.project_status());
                if let Some(query_path) = parts.get(1).copied() {
                    let related = svc.project_model.context_for_file(query_path, 5);
                    if related.is_empty() {
                        println!("  No related entities found for: {}", query_path);
                    } else {
                        println!("  Related to {}:", query_path);
                        for e in &related {
                            println!("    {:?} {} ({})", e.kind, e.name, e.path);
                        }
                    }
                }
            }
        }
        Some("paramecium") | Some("pm") => {
            let prompt = parts[1..].join(" ");
            if prompt.is_empty() {
                let svc = &rt.svc;
                let status = if svc.paramecium.is_some() {
                    let p = svc.paramecium.as_ref().unwrap();
                    format!("{} programs, {} bytes", p.program_count(), p.memory_bytes())
                } else {
                    "not built yet (will auto-build on first use)".to_string()
                };
                println!("  Paramecium: {}", status);
                println!("  Usage: /paramecium <prompt>  or  /pm <prompt>");
            } else {
                match rt.paramecium(&prompt) {
                    Ok(resp) => {
                        println!("  [paramecium: conf={:.2}]", resp.action_confidence);
                        println!();
                        if !resp.text.is_empty() {
                            println!("  {}", resp.text);
                        } else {
                            println!("  (empty response — lattice may need more programs)");
                        }
                    }
                    Err(e) => eprintln!("  paramecium error: {}", e),
                }
            }
        }
        _ => {
            println!("  Unknown command. Available:");
            println!("    /personality, /ocean, /reset, /history, /single, /status");
            println!("    /index <path>, /project [file], /paramecium <prompt>");
        }
    }
}

fn find_git_root(start: &std::path::Path) -> Option<std::path::PathBuf> {
    let mut dir = if start.is_file() { start.parent()? } else { start };
    loop {
        if dir.join(".git").exists() { return Some(dir.to_path_buf()); }
        dir = dir.parent()?;
    }
}

fn index_directory(svc: &mut LanguageService, dir: &std::path::Path, count: &mut usize) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if name.starts_with('.') || name == "target" || name == "node_modules" { continue; }
            index_directory(svc, &path, count);
        } else if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            let indexable = matches!(ext, "rs" | "py" | "ts" | "tsx" | "js" | "jsx" |
                "go" | "c" | "cpp" | "h" | "hpp" | "java" | "rb" | "toml" | "yaml" | "yml" | "md");
            if indexable {
                if let Ok(content) = std::fs::read_to_string(&path) {
                    let path_str = path.to_string_lossy();
                    svc.index_file(&path_str, &content);
                    *count += 1;
                }
            }
        }
    }
}

// =============================================================================
// Training pipeline
// =============================================================================

fn brain_parallel_batch_size() -> Option<usize> {
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

// ====================================================================
// Auto-configuration: data profiling + parameter derivation
// ====================================================================

#[derive(Debug, Clone)]
struct GroupProfile {
    gen_count: usize,
    code_count: usize,
    max_response_tokens: usize,
    max_code_tokens: usize,
    avg_response_tokens: f32,
    avg_code_tokens: f32,
    unique_intents: usize,
}

#[derive(Debug, Clone)]
struct DataProfile {
    total_samples: usize,
    num_groups: usize,
    groups: HashMap<usize, GroupProfile>,
    global_max_response_tokens: usize,
    global_max_code_tokens: usize,
    has_code: bool,
    class_imbalance_ratio: f32,
}

#[derive(Debug, Clone)]
struct AutoConfig {
    max_tokens: usize,
    gen_hidden: usize,
    gen_k: usize,
    gen_epochs: usize,
    router_epochs: usize,
    classifier_epochs: usize,
    classifier_lr: f32,
    max_synapses: usize,
    energy_budget: f32,
    replicas: usize,
    early_stop_window: usize,
    early_stop_min_improvement: f32,
    early_stop_min_epochs: usize,
}

fn profile_training_data(
    samples: &[LanguageSample],
    group_map_fn: &dyn Fn(&LanguageSample) -> usize,
    num_groups: usize,
) -> DataProfile {
    use growformer::spectral::tokenize;
    let mut groups: HashMap<usize, GroupProfile> = HashMap::new();
    let mut intent_sets: HashMap<usize, std::collections::HashSet<String>> = HashMap::new();

    for s in samples {
        let gidx = group_map_fn(s);
        let gp = groups.entry(gidx).or_insert(GroupProfile {
            gen_count: 0, code_count: 0,
            max_response_tokens: 0, max_code_tokens: 0,
            avg_response_tokens: 0.0, avg_code_tokens: 0.0,
            unique_intents: 0,
        });
        intent_sets.entry(gidx).or_default().insert(s.semantic_intent.clone());

        if let Some(r) = s.expected_response.as_deref() {
            let n_tok = tokenize(r).len();
            gp.gen_count += 1;
            gp.avg_response_tokens += n_tok as f32;
            if n_tok > gp.max_response_tokens { gp.max_response_tokens = n_tok; }
        }
        if let Some(c) = s.expected_code.as_deref() {
            if !c.is_empty() && c != "null" {
                let n_tok = tokenize(c).len();
                gp.code_count += 1;
                gp.avg_code_tokens += n_tok as f32;
                if n_tok > gp.max_code_tokens { gp.max_code_tokens = n_tok; }
            }
        }
    }

    let mut global_max_resp = 0usize;
    let mut global_max_code = 0usize;
    let mut has_code = false;

    for (gidx, gp) in groups.iter_mut() {
        if gp.gen_count > 0 { gp.avg_response_tokens /= gp.gen_count as f32; }
        if gp.code_count > 0 {
            gp.avg_code_tokens /= gp.code_count as f32;
            has_code = true;
        }
        gp.unique_intents = intent_sets.get(gidx).map_or(0, |s| s.len());
        if gp.max_response_tokens > global_max_resp { global_max_resp = gp.max_response_tokens; }
        if gp.max_code_tokens > global_max_code { global_max_code = gp.max_code_tokens; }
    }

    let group_counts: Vec<usize> = groups.values().map(|g| g.gen_count + g.code_count).collect();
    let max_count = *group_counts.iter().max().unwrap_or(&1) as f32;
    let min_count = *group_counts.iter().min().unwrap_or(&1) as f32;
    let imbalance = if min_count > 0.0 { max_count / min_count } else { 10.0 };

    DataProfile {
        total_samples: samples.len(),
        num_groups,
        groups,
        global_max_response_tokens: global_max_resp,
        global_max_code_tokens: global_max_code,
        has_code,
        class_imbalance_ratio: imbalance,
    }
}

fn auto_configure(profile: &DataProfile, include_code_tasks: bool) -> AutoConfig {
    let global_max_tok = if include_code_tasks && profile.has_code {
        profile
            .global_max_response_tokens
            .max(profile.global_max_code_tokens)
    } else {
        profile.global_max_response_tokens
    };
    let max_tokens = ((global_max_tok as f32 * 1.3).ceil() as usize)
        .max(32)
        .min(256);
    let max_tokens = next_power_of_two_or_round(max_tokens);

    // Hidden layer: scale with output dimension
    // Estimate output_dim ~ max_tokens * 11 bits (worst case dict size)
    let est_output = max_tokens * 11;
    let gen_hidden = (est_output / 3).clamp(128, 512);
    let gen_k = (gen_hidden / 4).clamp(32, 128);

    // Epochs: more samples = fewer epochs needed per sample (diminishing returns)
    // Base: 2000 for <=100 samples, scale down for larger datasets
    let base_epochs = if profile.total_samples <= 50 {
        3000usize
    } else if profile.total_samples <= 200 {
        2000
    } else if profile.total_samples <= 500 {
        1500
    } else {
        1000
    };

    // Compensate for class imbalance: more epochs if heavily imbalanced
    let imbalance_mult = if profile.class_imbalance_ratio > 3.0 { 1.3 } else { 1.0 };
    let gen_epochs = (base_epochs as f64 * imbalance_mult) as usize;

    // Router/classifier: scale with number of classes
    let router_epochs = (profile.num_groups * 250).clamp(600, 1500);
    let classifier_epochs = (profile.num_groups * 200).clamp(500, 1200);
    let classifier_lr = if profile.total_samples > 300 { 0.02 } else { 0.03 };

    let max_synapses = if est_output > 800 { 250 } else { 200 };
    let energy_budget = if est_output > 800 { 30.0 } else { 25.0 };

    let available_cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let num_tasks = profile.groups.len()
        * if include_code_tasks && profile.has_code {
            2
        } else {
            1
        };
    let replicas = if available_cpus > num_tasks * 2 {
        ((available_cpus / num_tasks.max(1)) as usize).clamp(1, 4)
    } else {
        1
    };

    // Convergence monitor: stop if loss improvement < threshold over window
    let early_stop_window = 100;
    let early_stop_min_improvement = 0.003;
    let early_stop_min_epochs = (gen_epochs / 3).max(200);

    println!("\n=== Auto-Configuration ===");
    println!(
        "  Data: {} samples, {} groups, code_in_data={}, include_code_tasks={}",
        profile.total_samples,
        profile.num_groups,
        profile.has_code,
        include_code_tasks
    );
    println!(
        "  Max tokens in data: response={}, code={}",
        profile.global_max_response_tokens,
        profile.global_max_code_tokens
    );
    println!("  Class imbalance ratio: {:.1}", profile.class_imbalance_ratio);
    println!("  -> MAX_TOKENS={}", max_tokens);
    println!("  -> GEN_HIDDEN={}, GEN_K={}", gen_hidden, gen_k);
    println!("  -> gen_epochs={} (base={}, imbalance_mult={:.1})", gen_epochs, base_epochs, imbalance_mult);
    println!("  -> router_epochs={}, classifier_epochs={}, classifier_lr={}", router_epochs, classifier_epochs, classifier_lr);
    println!("  -> max_synapses={}, energy_budget={}", max_synapses, energy_budget);
    println!("  -> replicas={} (cpus={}, tasks={})", replicas, available_cpus, num_tasks);
    println!("  -> early_stop: window={}, min_improvement={:.4}, min_epochs={}", early_stop_window, early_stop_min_improvement, early_stop_min_epochs);

    AutoConfig {
        max_tokens,
        gen_hidden,
        gen_k,
        gen_epochs,
        router_epochs,
        classifier_epochs,
        classifier_lr,
        max_synapses,
        energy_budget,
        replicas,
        early_stop_window,
        early_stop_min_improvement,
        early_stop_min_epochs,
    }
}

fn next_power_of_two_or_round(n: usize) -> usize {
    for &p in &[32, 48, 64, 96, 128, 160, 192, 256] {
        if n <= p { return p; }
    }
    256
}

fn truncate_to_char_boundary(s: &str, max_bytes: usize) -> usize {
    if max_bytes >= s.len() { return s.len(); }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) { end -= 1; }
    end
}

/// Discover the unique group names from the training data's action_target field.
/// Returns a stable, sorted list of group names. The order defines group indices.
fn discover_group_names(samples: &[growformer::dimension::LanguageSample]) -> Vec<String> {
    let mut seen = std::collections::BTreeSet::new();
    for s in samples {
        if let Some(at) = s.action_target.as_deref() {
            seen.insert(at.to_string());
        }
    }
    // Canonical ordering: support first, coding second, then the rest alphabetically.
    // This ensures support_gid=0 and coding_gid=1 align with LanguageService expectations.
    let mut ordered: Vec<String> = Vec::with_capacity(seen.len());
    if seen.remove("support") { ordered.push("support".to_string()); }
    if seen.remove("coding") { ordered.push("coding".to_string()); }
    for name in seen {
        ordered.push(name);
    }
    ordered
}

/// Build a lookup table from action_target name → group index, matching
/// the canonical ordering produced by `discover_group_names`.
fn build_group_lookup(group_names: &[String]) -> HashMap<String, usize> {
    group_names.iter().enumerate().map(|(i, n)| (n.clone(), i)).collect()
}

fn action_target_to_group(target: Option<&str>, lookup: &HashMap<String, usize>, num_groups: usize) -> usize {
    target
        .and_then(|t| lookup.get(t).copied())
        .unwrap_or(0)
        .min(num_groups.saturating_sub(1))
}

fn train_brain(
    epochs: u32,
    output_path: &str,
    max_samples: usize,
    quick_gen_epochs: u32,
    gen_epochs_override: u32,
    gen_replicas: u32,
    validate: bool,
    auto: bool,
    brain_name: Option<&str>,
    brain_description: Option<&str>,
    brain_author: Option<&str>,
    data_dir: Option<&str>,
    brain_plugins_toml: Option<&std::path::Path>,
    train_code_lattice: bool,
    no_progress: bool,
) -> Result<(), String> {
    println!("=== Full Neural Brain Training ===\n");
    if validate {
        println!("(validate mode: capped samples and gen epochs, will assert inference)\n");
    }
    let ui = train_progress::TrainUi::try_new(no_progress);
    let mut major_phase: u64 = 0;
    bump_train_phase(&ui, &mut major_phase, "Load data & augmentation");

    let mut rng = StdRng::seed_from_u64(42);

    let mut samples = if let Some(dir) = data_dir {
        let path = std::path::Path::new(dir);
        if !path.exists() {
            return Err(format!("Data directory not found: {}", dir));
        }
        println!("--- Custom data directory: {} ---", dir);
        let mut all = Vec::new();
        load_train_jsonl_dir(&mut all, path)?;
        all
    } else {
        load_all_m5_training_data()?
    };
    println!("Loaded {} training samples", samples.len());
    if max_samples > 0 && samples.len() > max_samples {
        samples.shuffle(&mut rng);
        samples.truncate(max_samples);
        let support_n = samples.iter().filter(|s| s.action_target.as_deref() == Some("support")).count();
        let coding_n = samples.iter().filter(|s| s.action_target.as_deref() == Some("coding")).count();
        let other_n = samples.len() - support_n - coding_n;
        println!("  shuffled and truncated to {} (support={}, coding={}, other={})", samples.len(), support_n, coding_n, other_n);
    }

    // Build saliency lexicon from knowledge graph for training augmentation.
    let saliency_lexicon = growformer::growformer_lang::topic_graph().map(|graph| {
        let keywords = graph.all_keywords();
        let lexicon = growformer::training_objectives::SaliencyLexicon::from_keywords(keywords);
        println!("  [saliency] Built lexicon with {} keywords for training augmentation", lexicon.keyword_count());
        lexicon
    });

    // Augment training data with salient span masking + RTD.
    // Masked augments reinforce salient-keyword representations.
    // RTD negatives are stored separately for contrastive development.
    if let Some(ref lexicon) = saliency_lexicon {
        let original_count = samples.len();
        let mut augmented_samples = Vec::new();
        for (i, sample) in samples.iter().enumerate() {
            // Salient masking injects "[MASK]" into expected_response; for sentiment
            // that trains the generator to emit mask tokens at inference.
            let sentiment = sample.action_target.as_deref() == Some("sentiment")
                || sample.domain.eq_ignore_ascii_case("sentiment");
            if sentiment {
                continue;
            }
            if let Some(ref response) = sample.expected_response {
                let masked_versions = growformer::training_objectives::mask_salient_spans(
                    response, lexicon, 2, (i as u64).wrapping_mul(0x517cc1b727220a95),
                );
                for masked_resp in masked_versions {
                    let mut aug = sample.clone();
                    aug.expected_response = Some(masked_resp);
                    augmented_samples.push(aug);
                }
            }
        }
        if !augmented_samples.is_empty() {
            println!("  [salient-mask] Generated {} augmented samples from {} originals",
                augmented_samples.len(), original_count);
            samples.extend(augmented_samples);
        }

        // RTD (Replaced Token Detection): create corrupted versions of responses
        // as hard negatives. The corrupted texts are added as additional training
        // samples that the lattice should NOT match to the original query.
        let rtd_dict = TokenDictionary::build(
            &samples.iter().filter_map(|s| s.expected_response.as_deref()).collect::<Vec<_>>(),
            4096,
        );
        let mut rtd_augmented = Vec::new();
        for (i, sample) in samples.iter().enumerate() {
            let sentiment = sample.action_target.as_deref() == Some("sentiment")
                || sample.domain.eq_ignore_ascii_case("sentiment");
            if sentiment {
                continue;
            }
            if let Some(ref response) = sample.expected_response {
                if let Some((corrupted, _mask)) = growformer::training_objectives::replace_salient_tokens(
                    response, lexicon, &rtd_dict, 0.15,
                    (i as u64).wrapping_mul(0x94d049bb133111eb),
                ) {
                    let mut neg = sample.clone();
                    neg.expected_response = Some(corrupted);
                    rtd_augmented.push(neg);
                }
            }
        }
        if !rtd_augmented.is_empty() {
            println!("  [rtd] Generated {} RTD hard-negative samples", rtd_augmented.len());
            samples.extend(rtd_augmented);
        }
    }

    // Discover groups dynamically from the training data's action_target values.
    let discovered_group_names = discover_group_names(&samples);
    println!("Discovered {} groups from data: {:?}", discovered_group_names.len(), discovered_group_names);
    let group_name_refs: Vec<&str> = discovered_group_names.iter().map(|s| s.as_str()).collect();
    let mut svc = LanguageService::new_with_groups(&group_name_refs)?;
    apply_brain_package_cli(&mut svc, brain_name, brain_description, brain_author);
    apply_optional_brain_plugins_toml(&mut svc, brain_plugins_toml)?;

    // Compute both raw and bridged embeddings.
    // Bridged vectors for routing/classification; raw vectors for generation conditioning.
    // When the `parallel` feature is enabled, embedding computation runs in parallel across cores.
    bump_train_phase(&ui, &mut major_phase, "Compute embeddings");
    if let Some(u) = &ui {
        u.detail_spinner("encoding samples (parallel)…");
    }
    println!("\n--- Computing embeddings ---");
    let runtime = &svc.dm.language_runtime;
    let results: Vec<_> = growformer::maybe_par_iter!(samples)
        .map(|s| runtime.encode_and_bridge(&s.text))
        .collect();
    let mut raw_embeddings: Vec<Vec<f32>> = Vec::with_capacity(samples.len());
    let mut bridged_embeddings: Vec<Vec<f32>> = Vec::with_capacity(samples.len());
    for res in results {
        match res {
            Ok((raw, bridged)) => {
                raw_embeddings.push(raw);
                bridged_embeddings.push(bridged.routed_vector);
            }
            Err(_) => {
                raw_embeddings.push(vec![0.0; 384]);
                bridged_embeddings.push(vec![0.0; DEFAULT_BRIDGE_DIM]);
            }
        }
    }
    let raw_dim = raw_embeddings.first().map_or(384, |e| e.len());
    let bridge_dim = bridged_embeddings.first().map_or(DEFAULT_BRIDGE_DIM, |e| e.len());
    println!("  {} samples: raw={}d, bridged={}d", samples.len(), raw_dim, bridge_dim);
    if let Some(u) = &ui {
        u.detail_finish_clear();
    }

    // ---------------------------------------------------------------
    // Paramecium pre-training: build lattice from embeddings to provide
    // curriculum signals and group-discovery diagnostics for training.
    // ---------------------------------------------------------------
    bump_train_phase(&ui, &mut major_phase, "Paramecium pre-lattice");
    println!("\n--- Paramecium Pre-Training Lattice ---");
    let fallback_dict = svc.dm.gen_dictionary.clone()
        .unwrap_or_else(|| TokenDictionary::build(
            &samples.iter().filter_map(|s| s.expected_response.as_deref()).collect::<Vec<_>>(),
            1024,
        ));
    let mut pre_lattice = InfraciliaryLattice::new(fallback_dict.clone());
    let lattice_pairs: Vec<(Vec<f32>, String)> = bridged_embeddings.iter()
        .zip(samples.iter())
        .filter_map(|(emb, s)| {
            s.expected_response.as_deref().map(|r| (emb.clone(), r.to_string()))
        })
        .collect();
    pre_lattice.develop(&lattice_pairs, 0.85);
    println!("  lattice programs: {} from {} samples", pre_lattice.program_count(), lattice_pairs.len());
    println!("  lattice memory: {} bytes", pre_lattice.memory_bytes());

    let (discovered_groups, _assignments) = pre_lattice.discover_groups(&bridged_embeddings, 0.85);
    println!("  discovered natural clusters: {} (hand-labeled groups: {})",
        discovered_groups, svc.dm.main.group_order.len());

    let novelty_scores: Vec<f32> = bridged_embeddings.iter()
        .map(|emb| {
            let (conf, _) = pre_lattice.novelty_score(emb);
            conf
        })
        .collect();
    let avg_novelty: f32 = novelty_scores.iter().sum::<f32>() / novelty_scores.len().max(1) as f32;
    let min_novelty = novelty_scores.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_novelty = novelty_scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    println!("  novelty scores: avg={:.3} min={:.3} max={:.3}", avg_novelty, min_novelty, max_novelty);

    bump_train_phase(&ui, &mut major_phase, "Data profile, auto-config & router");
    let num_groups = svc.dm.main.group_order.len();
    let group_lookup = build_group_lookup(&discovered_group_names);
    let group_map = |s: &LanguageSample| -> usize {
        action_target_to_group(s.action_target.as_deref(), &group_lookup, num_groups)
    };

    // ---------------------------------------------------------------
    // Auto-configuration: profile data and derive parameters
    // ---------------------------------------------------------------
    let (auto_cfg, data_profile) = if auto {
        let profile = profile_training_data(&samples, &group_map, num_groups.max(1));
        println!(
            "  Data: {} samples, {} groups, expected_code present in data={}",
            profile.total_samples,
            profile.num_groups,
            profile.has_code
        );
        if train_code_lattice {
            println!("  Auto-config: code tasks included (replicas/gen budget may reflect gen+code).");
        } else {
            println!("  Auto-config: code tasks excluded (expected_code ignored even if present in JSONL).");
        }
        for (gidx, gp) in &profile.groups {
            println!("  group {}: gen={} code={} max_resp_tok={} max_code_tok={} avg_resp_tok={:.0} intents={}",
                gidx, gp.gen_count, gp.code_count, gp.max_response_tokens,
                gp.max_code_tokens, gp.avg_response_tokens, gp.unique_intents);
        }
        (
            Some(auto_configure(&profile, train_code_lattice)),
            Some(profile),
        )
    } else {
        (None, None)
    };

    if !train_code_lattice {
        println!("\n  [train] MetaCodebook (2b) + code lattice: OFF. Enable with --train-code-lattice or [train].code_brain = true (see scripts/code.gf.toml).");
    }

    // ---------------------------------------------------------------
    // Stage 1: Train LearnedRouter on bridged embeddings
    // Oversample minority class to balance support vs coding.
    // ---------------------------------------------------------------
    println!("\n--- Stage 1: LearnedRouter Training ---");
    let router_samples_raw: Vec<(Vec<f32>, usize)> = bridged_embeddings.iter().zip(samples.iter())
        .map(|(emb, s)| (emb.clone(), group_map(s)))
        .collect();
    let mut class_counts: HashMap<usize, usize> = HashMap::new();
    for (_, g) in &router_samples_raw { *class_counts.entry(*g).or_default() += 1; }
    let max_class_count = class_counts.values().copied().max().unwrap_or(1);
    let mut router_samples: Vec<(Vec<f32>, usize)> = Vec::new();
    let mut class_buckets: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, (_, g)) in router_samples_raw.iter().enumerate() {
        class_buckets.entry(*g).or_default().push(i);
    }
    for (&gidx, indices) in &class_buckets {
        let count = class_counts[&gidx];
        let oversample = max_class_count / count.max(1);
        let remainder = max_class_count % count.max(1);
        for _ in 0..oversample {
            for &i in indices {
                router_samples.push(router_samples_raw[i].clone());
            }
        }
        for &i in indices.iter().take(remainder) {
            router_samples.push(router_samples_raw[i].clone());
        }
    }
    println!("  Router training: {} raw samples, {} after oversampling (balanced)",
        router_samples_raw.len(), router_samples.len());
    let router_epochs = if let Some(ac) = &auto_cfg { ac.router_epochs } else { (epochs as usize).max(500) };
    let (router_loss, router_acc) = svc.dm.train_language_router(
        &router_samples,
        router_epochs,
        &mut rng,
        brain_parallel_batch_size(),
    );
    println!("  Router loss={:.4} accuracy={:.1}% (Paramecium one-pass, {} programs)",
        router_loss, router_acc * 100.0,
        svc.dm.observer.learned_router.as_ref().map(|r| r.program_count()).unwrap_or(0));

    // ---------------------------------------------------------------
    // Stage 2: Train ActionClassifier
    // ---------------------------------------------------------------
    bump_train_phase(&ui, &mut major_phase, "Action classifier");
    println!("\n--- Stage 2: ActionClassifier Training ---");
    let action_samples: Vec<(Vec<f32>, growformer::dimension::action::ActionType)> = bridged_embeddings
        .iter()
        .zip(samples.iter())
        .map(|(emb, s)| {
            let at = action_target_to_type(s.action_target.as_deref().unwrap_or("coding"));
            (emb.clone(), at)
        })
        .collect();
    let (clf_epochs, clf_lr) = if let Some(ac) = &auto_cfg { (ac.classifier_epochs, ac.classifier_lr) } else { (500, 0.03) };
    let (clf_loss, clf_acc) = svc.dm.train_action_classifier(&action_samples, clf_epochs, clf_lr);
    println!("  Classifier loss={:.4} accuracy={:.1}% (Paramecium one-pass, {} programs)",
        clf_loss, clf_acc * 100.0,
        svc.dm.action_classifier.as_ref().map(|c| c.program_count()).unwrap_or(0));

    // ---------------------------------------------------------------
    // Stage 2b: GrowformerLang MetaCodebook (code-brain / --train-code-lattice only)
    // ---------------------------------------------------------------
    bump_train_phase(&ui, &mut major_phase, "MetaCodebook (2b)");
    if train_code_lattice {
        println!("\n--- Stage 2b: GrowformerLang MetaCodebook ---");
        use growformer::growformer_lang::{infer_concept, detect_language, MetaCodebook};
        let meta_samples: Vec<_> = bridged_embeddings
            .iter()
            .zip(samples.iter())
            .map(|(emb, s)| {
                let concept = infer_concept(
                    &s.text,
                    Some(&s.semantic_intent),
                    s.action_target.as_deref(),
                );
                let lang = detect_language(&s.text);
                let gidx = group_map(s);
                (emb.clone(), concept, lang, gidx)
            })
            .collect();
        let mcb = MetaCodebook::build(&meta_samples);
        mcb.print_summary();
        svc.meta_codebook = Some(mcb);
    } else {
        println!("\n--- Stage 2b: GrowformerLang MetaCodebook [skipped] ---");
    }

    // ---------------------------------------------------------------
    // Stages 3+4: Per-group generation envs (Growformer substrate)
    // Each group gets its own NeuralEnvironment for text AND code generation.
    // Conditioning = bridged_embedding — routing already selected the group.
    // All groups train in parallel (structurally isolated, no shared state).
    // ---------------------------------------------------------------
    let num_groups = svc.dm.main.group_order.len().max(1);
    let gen_epochs: usize = if quick_gen_epochs > 0 {
        quick_gen_epochs as usize
    } else if gen_epochs_override > 0 {
        gen_epochs_override as usize
    } else if let Some(ac) = &auto_cfg {
        ac.gen_epochs
    } else {
        (epochs as usize * 50).max(500)
    };
    let k_replicas = if let Some(ac) = &auto_cfg { ac.replicas } else { (gen_replicas as usize).max(1) };
    let gen_overrides = auto_cfg.as_ref().map(|ac| {
        growformer::dimension::group_gen::GenEnvOverrides {
            max_tokens: Some(ac.max_tokens),
            hidden: Some(ac.gen_hidden),
            k: Some(ac.gen_k),
            max_synapses: Some(ac.max_synapses),
            energy_budget: Some(ac.energy_budget),
            hex_mode: Some(true),
            ..Default::default()
        }
    });
    println!("\n--- Stages 3+4: Per-Group Generation Envs (single-pass token prediction) ---");
    println!("  gen_epochs={}, num_groups={}, replicas_per_task={}", gen_epochs, num_groups, k_replicas);

    // Partition training data by (group, kind), carrying novelty scores per sample.
    // Each pair now carries (bridged, raw, target) so per-group adapters can specialize conditioning.
    let mut gen_by_group: HashMap<usize, Vec<(Vec<f32>, String)>> = HashMap::new();
    let mut gen_raw_by_group: HashMap<usize, Vec<&[f32]>> = HashMap::new();
    let mut gen_topic_by_group: HashMap<usize, Vec<&str>> = HashMap::new();
    let mut gen_novelty_by_group: HashMap<usize, Vec<f32>> = HashMap::new();
    let mut code_by_group: HashMap<usize, Vec<(&[f32], &str)>> = HashMap::new();
    let mut code_raw_by_group: HashMap<usize, Vec<&[f32]>> = HashMap::new();
    let mut code_topic_by_group: HashMap<usize, Vec<&str>> = HashMap::new();
    let mut code_novelty_by_group: HashMap<usize, Vec<f32>> = HashMap::new();
    for (i, ((bridged, raw), s)) in bridged_embeddings.iter().zip(raw_embeddings.iter()).zip(samples.iter()).enumerate() {
        let gidx = group_map(s);
        let nov = novelty_scores.get(i).copied().unwrap_or(0.0);
        if let Some(r) = s.expected_response.as_deref() {
            let lattice_text = if should_use_sentiment_joint_index(s) {
                sentiment_lattice_index_body_with_causal(&s.text, r, s.causal.as_ref())
            } else {
                r.to_string()
            };
            gen_by_group
                .entry(gidx)
                .or_default()
                .push((bridged.clone(), lattice_text));
            gen_raw_by_group.entry(gidx).or_default().push(raw.as_slice());
            gen_topic_by_group.entry(gidx).or_default().push(s.semantic_intent.as_str());
            gen_novelty_by_group.entry(gidx).or_default().push(nov);
        }
        if train_code_lattice {
            if let Some(c) = s.expected_code.as_deref() {
                if !c.is_empty() && c != "null" {
                    code_by_group.entry(gidx).or_default().push((bridged.as_slice(), c));
                    code_raw_by_group.entry(gidx).or_default().push(raw.as_slice());
                    code_topic_by_group.entry(gidx).or_default().push(s.semantic_intent.as_str());
                    code_novelty_by_group.entry(gidx).or_default().push(nov);
                }
            }
        }
    }

    // Create per-group low-rank adapters for generation conditioning.
    use growformer::dimension::language::{GroupAdapter, DEFAULT_ADAPTER_RANK};
    let mut group_adapters: HashMap<usize, GroupAdapter> = HashMap::new();
    for &gidx in gen_by_group.keys().chain(code_by_group.keys()) {
        if !group_adapters.contains_key(&gidx) {
            group_adapters.insert(gidx, GroupAdapter::new(raw_dim, bridge_dim, DEFAULT_ADAPTER_RANK));
        }
    }
    let adapter_params: usize = group_adapters.values().map(|a| a.param_count()).sum();
    println!("  per-group adapters: {} groups, rank={}, {}d->{}d, {} params total",
        group_adapters.len(), DEFAULT_ADAPTER_RANK, raw_dim, bridge_dim, adapter_params);

    // Stage 2.5: Understanding Layer + MetaBrain — Paramecium one-pass build.
    // All classifiers use Paramecium lattice develop(), zero iterative epochs.
    bump_train_phase(&ui, &mut major_phase, "Understanding + MetaBrain");
    println!("\n--- Stage 2.5: Understanding Layer + MetaBrain (Paramecium one-pass) ---");
    {
        use growformer::understanding::UnderstandingLayer;
        use growformer::micro_brain::{MetaBrain, MicroBrain, MicroBrainRole};
        use growformer::dimension::action_classifier::NUM_ACTION_TYPES;

        let understanding_samples: Vec<(&[f32], &str)> = raw_embeddings.iter()
            .zip(samples.iter())
            .map(|(raw, s)| (raw.as_slice(), s.semantic_intent.as_str()))
            .collect();
        if understanding_samples.is_empty() {
            println!("  [skip] no understanding samples");
        } else {
            let mut ul = UnderstandingLayer::build(&understanding_samples, raw_dim);
            ul.freeze();
            println!("  understanding layer: {} topics, {} verbs, Paramecium one-pass, frozen=true",
                ul.topic_count(), ul.verb_count());

            let mut mb_rng = rand::rngs::StdRng::seed_from_u64(314);
            let mut mb = MetaBrain::build(
                raw_dim,
                ul.topic_count(),
                ul.topic_names.clone(),
                ul.topic_embeddings.clone(),
                ul.verb_embeddings.clone(),
                NUM_ACTION_TYPES,
                &mut mb_rng,
            );

            // One-pass action brain from action_samples
            let action_data: Vec<(&[f32], usize)> = action_samples.iter()
                .map(|(emb, at)| {
                    let idx = match at {
                        growformer::dimension::action::ActionType::SupportTicket => 0,
                        growformer::dimension::action::ActionType::CodingAssist => 1,
                        growformer::dimension::action::ActionType::GeneralAssist => 2,
                        growformer::dimension::action::ActionType::ToolCall => 3,
                        growformer::dimension::action::ActionType::Fallback => 4,
                    };
                    (emb.as_slice(), idx)
                })
                .collect();
            mb.action_brain = MicroBrain::build_from_data(
                MicroBrainRole::Action, raw_dim, NUM_ACTION_TYPES,
                vec!["support".into(), "coding".into(), "general".into(), "tool".into(), "fallback".into()],
                &action_data,
            );
            println!("  action brain: {} programs (one-pass)", mb.action_brain.lattice.program_count());

            // One-pass topic + verb brains
            let mut topic_map: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
            for (i, name) in ul.topic_names.iter().enumerate() {
                topic_map.insert(name.clone(), i);
            }
            let verb_map: std::collections::HashMap<String, usize> = ul.verb_names.iter().enumerate()
                .map(|(i, v)| (v.clone(), i)).collect();

            let topic_data: Vec<(&[f32], usize)> = understanding_samples.iter()
                .filter_map(|&(raw, intent)| topic_map.get(intent).map(|&idx| (raw, idx)))
                .collect();
            mb.topic_brain = MicroBrain::build_from_data(
                MicroBrainRole::Topic, raw_dim, ul.topic_count(),
                ul.topic_names.clone(), &topic_data,
            );

            let verb_data: Vec<(&[f32], usize)> = understanding_samples.iter()
                .map(|&(raw, intent)| {
                    let verb = growformer::understanding::intent_to_verb(intent);
                    let idx = verb_map.get(verb).copied().unwrap_or(0);
                    (raw, idx)
                })
                .collect();
            mb.verb_brain = MicroBrain::build_from_data(
                MicroBrainRole::Verb, raw_dim, ul.verb_count(),
                ul.verb_names.clone(), &verb_data,
            );
            println!("  topic brain: {} programs, verb brain: {} programs (one-pass)",
                mb.topic_brain.lattice.program_count(), mb.verb_brain.lattice.program_count());

            // One-pass coordinator: develop from all (raw → conditioning) pairs
            let coord_pairs: Vec<(Vec<f32>, Vec<f32>)> = raw_embeddings.iter()
                .zip(bridged_embeddings.iter())
                .map(|(raw, bridged)| {
                    let (_, _, topic_logits) = mb.topic_brain.predict(raw);
                    let (_, _, verb_logits) = mb.verb_brain.predict(raw);
                    let (_, _, action_logits) = mb.action_brain.predict(raw);
                    let (_, tc, _) = mb.topic_brain.predict(raw);
                    let (_, vc, _) = mb.verb_brain.predict(raw);
                    let (_, ac, _) = mb.action_brain.predict(raw);
                    let mut ci = Vec::new();
                    ci.extend_from_slice(&topic_logits);
                    ci.extend_from_slice(&verb_logits);
                    ci.extend_from_slice(&action_logits);
                    ci.push(tc); ci.push(vc); ci.push(ac);
                    let mut target = bridged.clone();
                    target.resize(growformer::dimension::group_gen::GEN_COND_DIM, 0.0);
                    (ci, target)
                })
                .collect();
            mb.coordinator.develop(&coord_pairs);
            println!("  coordinator: {} centroids (one-pass)", mb.coordinator.centroids.len());

            mb.freeze();
            println!("  MetaBrain: frozen, topic={}cls verb={}cls action={}cls",
                mb.topic_brain.output_dim, mb.verb_brain.output_dim, mb.action_brain.output_dim);

            svc.dm.understanding = Some(ul);
            svc.dm.meta_brain = Some(mb);
        }
    }

    // Build per-group dictionaries from each group's own data.
    // Larger groups get larger dictionaries (2048 max), smaller groups get 1024.
    bump_train_phase(&ui, &mut major_phase, "Dictionaries & codebooks");
    use growformer::dimension::group_gen::{bits_for_dict, MAX_TOKENS};
    let effective_max_tokens = gen_overrides.as_ref().and_then(|o| o.max_tokens).unwrap_or(MAX_TOKENS);
    let mut gen_dicts: HashMap<usize, TokenDictionary> = HashMap::new();
    let mut code_dicts: HashMap<usize, TokenDictionary> = HashMap::new();
    let pairs_threshold_for_large_dict: usize = 100;
    for (&gidx, pairs) in &gen_by_group {
        if pairs.is_empty() { continue; }
        let texts: Vec<&str> = pairs.iter().map(|(_emb, text)| text.as_str()).collect();
        let max_dict = if pairs.len() >= pairs_threshold_for_large_dict { 2048 } else { 1024 };
        let dict = TokenDictionary::build(&texts, max_dict);
        let bits = bits_for_dict(dict.len());
        println!("  gen dict[g{}]: {} tokens from {} texts, {} bits/token, output={}",
            gidx, dict.len(), texts.len(), bits, effective_max_tokens * bits);
        gen_dicts.insert(gidx, dict);
    }
    for (&gidx, pairs) in &code_by_group {
        if pairs.is_empty() { continue; }
        let texts: Vec<&str> = pairs.iter().map(|(_emb, text)| *text).collect();
        let max_dict = if pairs.len() >= pairs_threshold_for_large_dict { 2048 } else { 1024 };
        let dict = TokenDictionary::build(&texts, max_dict);
        let bits = bits_for_dict(dict.len());
        println!("  code dict[g{}]: {} tokens from {} texts, {} bits/token, output={}",
            gidx, dict.len(), texts.len(), bits, effective_max_tokens * bits);
        code_dicts.insert(gidx, dict);
    }
    // Build algebraic codebooks: factorize each group's response space into
    // archetypes + slots, reducing prediction from ~1900 to ~80-120 bits.
    // Scale archetypes with data size: larger groups need more archetypes
    // to avoid collapsing diverse texts into too few buckets.
    let base_archetypes = 32usize;
    let mut gen_codebooks: HashMap<usize, AlgebraicCodebook> = HashMap::new();
    let mut code_codebooks: HashMap<usize, AlgebraicCodebook> = HashMap::new();
    for (&gidx, pairs) in &gen_by_group {
        if pairs.is_empty() { continue; }
        let texts: Vec<&str> = pairs.iter().map(|(_emb, text)| text.as_str()).collect();
        let embs: Vec<&[f32]> = pairs.iter().map(|(emb, _text)| emb.as_slice()).collect();
        let dict = gen_dicts.get(&gidx).unwrap();
        let intent_count = data_profile.as_ref()
            .and_then(|p| p.groups.get(&gidx))
            .map(|g| g.unique_intents)
            .unwrap_or(0);
        let gen_max_arch = if texts.len() > 150 || intent_count > 40 {
            base_archetypes * 4  // 128 for large/high-diversity groups
        } else if texts.len() > 50 || intent_count > 12 {
            base_archetypes * 3  // 96 for medium groups or high intent diversity
        } else {
            base_archetypes      // 32 for small groups
        };
        let cb = AlgebraicCodebook::build(&texts, dict, gen_max_arch, Some(&embs));
        let mode = if cb.has_prototypes() { "SLOT-ONLY" } else { "FULL" };
        println!("  gen codebook[g{}]: {} archetypes (max={}), {} slots max, {} total/{} slot bits (was {}) [{}]",
            gidx, cb.archetypes.len(), gen_max_arch, cb.max_slot_count, cb.total_bits, cb.slot_only_bits,
            effective_max_tokens * bits_for_dict(dict.len()), mode);
        gen_codebooks.insert(gidx, cb);
    }
    for (&gidx, pairs) in &code_by_group {
        if pairs.is_empty() { continue; }
        let texts: Vec<&str> = pairs.iter().map(|(_emb, text)| *text).collect();
        let embs: Vec<&[f32]> = pairs.iter().map(|(emb, _text)| *emb).collect();
        let dict = code_dicts.get(&gidx).unwrap();
        let code_max_arch = base_archetypes * 2;
        let cb = AlgebraicCodebook::build_syntax_aware(&texts, dict, code_max_arch, Some(&embs));
        let mode = if cb.has_prototypes() { "SYNTAX+SLOT-ONLY" } else { "SYNTAX-AWARE" };
        println!("  code codebook[g{}]: {} archetypes, {} slots max, {} total/{} slot bits (was {}) [{}]",
            gidx, cb.archetypes.len(), cb.max_slot_count, cb.total_bits, cb.slot_only_bits,
            effective_max_tokens * bits_for_dict(dict.len()), mode);
        code_codebooks.insert(gidx, cb);
    }

    // Build Hopf composition tables for compositional OOD generation.
    // Re-derive cluster assignments from match_best (fast, avoids changing build signatures).
    let hopf_segments = 3;
    let mut gen_hopf: HashMap<usize, HopfCompositionTable> = HashMap::new();
    let mut code_hopf: HashMap<usize, HopfCompositionTable> = HashMap::new();
    for (&gidx, pairs) in &gen_by_group {
        if pairs.is_empty() { continue; }
        if let Some(cb) = gen_codebooks.get(&gidx) {
            if !cb.has_prototypes() || cb.archetypes.len() < 2 { continue; }
            let dict = gen_dicts.get(&gidx).unwrap();
            let embs: Vec<&[f32]> = pairs.iter().map(|(emb, _)| emb.as_slice()).collect();
            let mut clusters: Vec<Vec<usize>> = vec![vec![]; cb.archetypes.len()];
            for (i, (_emb, text)) in pairs.iter().enumerate() {
                let ids = dict.encode(text.as_str());
                let (arch_idx, _) = cb.match_best(&ids);
                if arch_idx < clusters.len() { clusters[arch_idx].push(i); }
            }
            let hopf = HopfCompositionTable::build(cb, Some(&embs), &clusters, hopf_segments);
            println!("  gen hopf[g{}]: {} segments, {} response_len", gidx, hopf.num_segments, hopf.response_length);
            gen_hopf.insert(gidx, hopf);
        }
    }
    for (&gidx, pairs) in &code_by_group {
        if pairs.is_empty() { continue; }
        if let Some(cb) = code_codebooks.get(&gidx) {
            if !cb.has_prototypes() || cb.archetypes.len() < 2 { continue; }
            let dict = code_dicts.get(&gidx).unwrap();
            let embs: Vec<&[f32]> = pairs.iter().map(|(emb, _)| *emb).collect();
            let mut clusters: Vec<Vec<usize>> = vec![vec![]; cb.archetypes.len()];
            for (i, (_emb, text)) in pairs.iter().enumerate() {
                let ids = dict.encode(text);
                let (arch_idx, _) = cb.match_best(&ids);
                if arch_idx < clusters.len() { clusters[arch_idx].push(i); }
            }
            let hopf = HopfCompositionTable::build(cb, Some(&embs), &clusters, hopf_segments);
            println!("  code hopf[g{}]: {} segments, {} response_len", gidx, hopf.num_segments, hopf.response_length);
            code_hopf.insert(gidx, hopf);
        }
    }

    // Store first available dict as fallback for service inference
    if let Some(d) = gen_dicts.values().next() {
        svc.dm.gen_dictionary = Some(d.clone());
    }
    if let Some(d) = code_dicts.values().next() {
        svc.dm.code_dictionary = Some(d.clone());
    }

    // Seed ArchetypeBrain from all groups' codebook prototypes.
    // One program per (group_idx, archetype_idx) across all groups.
    {
        use growformer::micro_brain::ArchetypeBrain;
        let mut archetype_entries: Vec<(usize, usize, Vec<f32>, Vec<u16>)> = Vec::new();
        for (&gidx, cb) in &gen_codebooks {
            if !cb.has_prototypes() { continue; }
            if gen_dicts.get(&gidx).is_none() { continue; }
            for (ai, arch) in cb.archetypes.iter().enumerate() {
                let proto = cb.archetype_prototypes.get(ai)
                    .cloned().unwrap_or_else(|| vec![0.0f32; bridge_dim]);
                let tokens: Vec<u16> = arch.fixed.iter()
                    .map(|&(_, tok)| tok).collect();
                archetype_entries.push((gidx, ai, proto, tokens));
            }
        }
        if !archetype_entries.is_empty() {
            let ref_dict = gen_dicts.values().next().unwrap().clone();
            let ab = ArchetypeBrain::build(&archetype_entries, ref_dict);
            println!("\n  ArchetypeBrain: {} programs across {} groups",
                ab.program_count(), gen_codebooks.len());
            if let Some(ref mut mb) = svc.dm.meta_brain {
                mb.archetype_brain = Some(ab);
            }
        }
    }

    // ---------------------------------------------------------------
    // Stages 3+4: Indexed generation via Paramecium lattice
    //
    // Knowledge is indexed in one pass (codebook + lattice develop).
    // No NeuralEnvironment, no backprop, no iterative training.
    // ---------------------------------------------------------------
    bump_train_phase(&ui, &mut major_phase, "Indexed generation lattices");
    println!("\n--- Stages 3+4: Indexed Generation (Paramecium lattice) ---");

    use growformer::dimension::group_gen::IndexedGenEnv;
    let spawn_threshold = 0.97;
    let min_code_for_index = 10usize;
    let mut index_jobs: u64 = 0;
    for gidx in 0..num_groups {
        if gen_by_group.get(&gidx).map_or(false, |p| !p.is_empty()) {
            index_jobs += 1;
        }
        if code_by_group.get(&gidx).map_or(false, |p| p.len() >= min_code_for_index) {
            index_jobs += 1;
        }
    }
    if let Some(u) = &ui {
        if index_jobs > 0 {
            u.detail_bar(index_jobs, "building per-group lattices");
        }
    }
    for gidx in 0..num_groups {
        if let Some(p) = gen_by_group.get(&gidx) {
            if !p.is_empty() {
                println!("  task: group {} gen — {} pairs", gidx, p.len());
            }
        }
        if let Some(p) = code_by_group.get(&gidx) {
            if !p.is_empty() {
                println!("  task: group {} code — {} pairs", gidx, p.len());
            }
        }
    }

    // Install per-group adapters and rotors BEFORE building gen envs so we can
    // apply the same Clifford + understanding conditioning used at inference time.
    for &gidx in gen_by_group.keys().chain(code_by_group.keys()) {
        if !svc.dm.group_adapters.contains_key(&gidx) {
            let adapter = group_adapters.get(&gidx).cloned()
                .unwrap_or_else(|| GroupAdapter::new(raw_dim, bridge_dim, DEFAULT_ADAPTER_RANK));
            svc.dm.group_adapters.insert(gidx, adapter);
            svc.dm.group_rotors.insert(gidx, GroupRotor::new());
        }
    }

    // Build IndexedGenEnv for each group (one-pass Paramecium lattice).
    // Conditioning vectors are built with the same Clifford + understanding path
    // used at inference time so that cosine similarity is high during generation.
    let t_index_start = std::time::Instant::now();
    for gidx in 0..num_groups {
        if let Some(pairs) = gen_by_group.get(&gidx) {
            if pairs.is_empty() { continue; }
            let raw_vecs = gen_raw_by_group.get(&gidx).unwrap();
            let topic_names = gen_topic_by_group.get(&gidx).unwrap();
            let dict = gen_dicts.get(&gidx).unwrap().clone();
            let cb = gen_codebooks.get(&gidx).cloned().unwrap_or_else(|| {
                let text_refs: Vec<&str> = pairs.iter().map(|(_, t)| t.as_str()).collect();
                AlgebraicCodebook::build(&text_refs, &dict, 32, None)
            });
            let hopf = gen_hopf.get(&gidx).cloned().unwrap_or_default();
            let training_pairs: Vec<(Vec<f32>, String, String)> = pairs.iter()
                .zip(raw_vecs.iter())
                .zip(topic_names.iter())
                .map(|(((bridged, text), raw), topic_name)| {
                    let cond = svc.dm.adapt_for_group_clifford(
                        gidx, bridged.as_slice(), raw, growformer::dimension::group_gen::GEN_COND_DIM,
                    );
                    (cond, text.clone(), (*topic_name).to_string())
                })
                .collect();
            let mut env = IndexedGenEnv::from_tagged_parts(dict, cb, hopf, &training_pairs, spawn_threshold);
            env.freeze();
            println!(
                "  gen[g{}]: {} lattice programs, {} topic sub-lattices, frozen",
                gidx, env.program_count(), env.topic_subindex.len()
            );
            svc.dm.group_gen_envs.insert(gidx, env);
            if let Some(u) = &ui {
                if index_jobs > 0 {
                    u.detail_inc(1);
                }
            }
        }

        if let Some(pairs) = code_by_group.get(&gidx) {
            let min_code_samples = 10;
            if pairs.len() < min_code_samples { continue; }
            let raw_vecs = code_raw_by_group.get(&gidx).unwrap();
            let topic_names = code_topic_by_group.get(&gidx).unwrap();
            let dict = code_dicts.get(&gidx).unwrap().clone();
            let cb = code_codebooks.get(&gidx).cloned().unwrap_or_else(|| AlgebraicCodebook::build_syntax_aware(
                &pairs.iter().map(|(_, t)| *t).collect::<Vec<_>>(), &dict, 32, None,
            ));
            let hopf = code_hopf.get(&gidx).cloned().unwrap_or_default();
            let training_pairs: Vec<(Vec<f32>, String, String)> = pairs.iter()
                .zip(raw_vecs.iter())
                .zip(topic_names.iter())
                .map(|(((bridged, text), raw), topic_name)| {
                    let cond = svc.dm.adapt_for_group_clifford(
                        gidx, bridged, raw, growformer::dimension::group_gen::GEN_COND_DIM,
                    );
                    (cond, text.to_string(), (*topic_name).to_string())
                })
                .collect();
            let mut env = IndexedGenEnv::from_tagged_parts(dict, cb, hopf, &training_pairs, spawn_threshold);
            env.freeze();
            println!(
                "  code[g{}]: {} lattice programs, {} topic sub-lattices, frozen",
                gidx, env.program_count(), env.topic_subindex.len()
            );
            svc.dm.group_code_envs.insert(gidx, env);
            if let Some(u) = &ui {
                if index_jobs > 0 {
                    u.detail_inc(1);
                }
            }
        }
    }
    let t_index_elapsed = t_index_start.elapsed();
    if let Some(u) = &ui {
        u.detail_finish_clear();
    }
    println!("  Indexed {} gen + {} code groups in {:?}",
        svc.dm.group_gen_envs.len(), svc.dm.group_code_envs.len(), t_index_elapsed);

    // Contrastive refinement: push apart program centroids within each group's
    // topic sub-lattices that are too similar but represent different content.
    bump_train_phase(&ui, &mut major_phase, "Contrastive lattice refinement");
    println!("\n--- Contrastive Lattice Refinement ---");
    let contrastive_margin = 0.92;
    let contrastive_rate = 0.05;
    let mut total_repulsions = 0;
    for (&gidx, env) in svc.dm.group_gen_envs.iter_mut() {
        let mut group_repulsions = 0;
        for topic in &mut env.topic_subindex {
            let r = topic.lattice.contrastive_refine(contrastive_margin, contrastive_rate);
            group_repulsions += r;
        }
        env.lattice.contrastive_refine(contrastive_margin, contrastive_rate);
        if group_repulsions > 0 {
            println!("  gen[g{}]: {} contrastive repulsions", gidx, group_repulsions);
        }
        total_repulsions += group_repulsions;
    }
    for (&gidx, env) in svc.dm.group_code_envs.iter_mut() {
        let mut group_repulsions = 0;
        for topic in &mut env.topic_subindex {
            let r = topic.lattice.contrastive_refine(contrastive_margin, contrastive_rate);
            group_repulsions += r;
        }
        if group_repulsions > 0 {
            println!("  code[g{}]: {} contrastive repulsions", gidx, group_repulsions);
        }
        total_repulsions += group_repulsions;
    }
    println!("  Total: {} contrastive repulsions (margin={}, rate={})",
        total_repulsions, contrastive_margin, contrastive_rate);

    // ---------------------------------------------------------------
    // STA-CALM Orchestrated Training: per-grade semantic pressure
    // Runs 3-phase pipeline on lattice programs to activate the
    // Cl(1,7) grade structure with semantic meaning.
    // ---------------------------------------------------------------
    bump_train_phase(&ui, &mut major_phase, "STA-CALM orchestration");
    println!("\n--- STA-CALM Orchestrated Training ---");
    {
        let t_sta = std::time::Instant::now();
        let mut total_programs = 0usize;
        for (&gidx, env) in svc.dm.group_gen_envs.iter_mut() {
            let vocab_size = env.dictionary.tokens.len();
            let mut orchestrator = growformer::training_objectives::TrainingOrchestrator::new(vocab_size);

            let mut programs: Vec<(String, Vec<u16>, Vec<f32>)> = Vec::new();
            for topic in &env.topic_subindex {
                for prog in &topic.lattice.programs {
                    programs.push((
                        topic.topic_name.clone(),
                        prog.token_sequence.clone(),
                        prog.ema_centroid.clone(),
                    ));
                }
            }

            if programs.len() < 4 { continue; }

            let diags = orchestrator.run_full_pipeline(
                &mut programs,
                3,  // phase 1: grade pretraining epochs
                2,  // phase 2: rotor predictor epochs
                2,  // phase 3: joint fine-tuning epochs
            );

            // Write back adjusted centroids to the lattice programs
            let mut prog_idx = 0;
            for topic in &mut env.topic_subindex {
                for prog in &mut topic.lattice.programs {
                    if prog_idx < programs.len() {
                        prog.ema_centroid = programs[prog_idx].2.clone();
                        prog_idx += 1;
                    }
                }
            }

            total_programs += prog_idx;
            if let Some(last) = diags.last() {
                println!("  gen[g{}]: {} programs, final loss={:.4}, rotor_conf={:.3}",
                    gidx, prog_idx, last.avg_total_loss, last.rotor_prediction_confidence);
            }
        }
        println!("  STA-CALM: {} total programs trained in {:?}", total_programs, t_sta.elapsed());
    }

    // Register per-group structural fingerprints (grade-2 bivectors in Cl(8))
    // for understanding-based routing on novel/OOD inputs.
    bump_train_phase(&ui, &mut major_phase, "Structural fingerprints");
    println!("\n--- Computing Structural Fingerprints ---");
    for gidx in 0..num_groups {
        let mut all_raw: Vec<&[f32]> = Vec::new();
        if let Some(raws) = gen_raw_by_group.get(&gidx) {
            all_raw.extend(raws.iter());
        }
        if let Some(raws) = code_raw_by_group.get(&gidx) {
            all_raw.extend(raws.iter());
        }
        if !all_raw.is_empty() {
            svc.dm.register_group_fingerprint(gidx, &all_raw);
            println!("  group {}: fingerprint from {} embeddings", gidx, all_raw.len());
        }
    }

    // ---------------------------------------------------------------
    // Build cognitive map — hippocampal relational graph across all groups
    // ---------------------------------------------------------------
    bump_train_phase(&ui, &mut major_phase, "Cognitive map");
    println!("\n--- Building Cognitive Map (Reasoning Engine) ---");
    let cog_map = CognitiveMap::build(&svc.dm.group_gen_envs, &svc.dm.group_rotors);
    println!("  nodes: {}, edges: {} (cross-group structural links)", cog_map.node_count(), cog_map.edge_count());
    let group_dicts: HashMap<usize, TokenDictionary> = svc.dm.group_gen_envs.iter()
        .map(|(&gidx, env)| (gidx, env.dictionary.clone()))
        .collect();
    let reasoning_engine = ReasoningEngine::new(cog_map, group_dicts);
    svc.reasoning = Some(reasoning_engine);
    println!("  ReasoningEngine: active, settling_rounds=4, system2_max_steps={}", svc.system2_config.max_steps);

    // ---------------------------------------------------------------
    // Build MetaCognition: reflective quality gate on generation output.
    // Trains from (prompt_embedding, response_embedding, topic) triples
    // extracted from the lattice programs — learns what good (prompt, response)
    // pairs look like in embedding space.
    // ---------------------------------------------------------------
    {
        use growformer::metacognition::MetaCognition;
        bump_train_phase(&ui, &mut major_phase, "MetaCognition");
        println!("\n--- Building MetaCognition (Reflection Brain) ---");
        let mut mc = MetaCognition::with_defaults();
        let mut pair_count = 0u64;
        for (&_gidx, env) in svc.dm.group_gen_envs.iter() {
            let default_topic = if env.topic_subindex.is_empty() {
                "general"
            } else {
                env.topic_subindex[0].topic_name.as_str()
            };
            for prog in &env.lattice.programs {
                let topic = env.topic_label_for_program_centroid(&prog.ema_centroid, default_topic);
                mc.absorb_pair(&prog.ema_centroid, &prog.ema_centroid, topic.as_str());
                pair_count += 1;
            }
        }
        for (&_gidx, env) in svc.dm.group_code_envs.iter() {
            let default_topic = if env.topic_subindex.is_empty() {
                "code_general"
            } else {
                env.topic_subindex[0].topic_name.as_str()
            };
            for prog in &env.lattice.programs {
                let topic = env.topic_label_for_program_centroid(&prog.ema_centroid, default_topic);
                mc.absorb_pair(&prog.ema_centroid, &prog.ema_centroid, topic.as_str());
                pair_count += 1;
            }
        }
        println!(
            "  MetaCognition: {} pairs absorbed, {} topic centroids, ready={}",
            pair_count, mc.topic_count(), mc.is_ready()
        );
        svc.metacognition = Some(mc);
    }

    // ---------------------------------------------------------------
    // Cloze games: fill-in-the-blank learning to teach slot inference.
    // Programs learn to "own" specific slot fills through centroid drift.
    // Multiple rounds: each round refines program ownership.
    // ---------------------------------------------------------------
    {
        use growformer::cloze;
        bump_train_phase(&ui, &mut major_phase, "Cloze learning");
        println!("\n--- Learning With Games (Fill-in-the-Blank) ---");
        let cloze_rounds = 6;
        let k_voters = 7;
        let reward_rate = 0.10;
        let punish_rate = 0.06;
        let mut total_stats = cloze::ClozeStats::default();
        let cap_u64 = cloze::DEFAULT_MAX_CLOZE_TASKS_PER_GROUP as u64;
        let mut cloze_units: u64 = 0;
        for (_g, env) in &svc.dm.group_gen_envs {
            if env.codebook.as_ref().filter(|cb| !cb.archetypes.is_empty()).is_some() {
                cloze_units += (env.lattice.programs.len() as u64).min(cap_u64) * (cloze_rounds as u64);
            }
        }
        for (_g, env) in &svc.dm.group_code_envs {
            if env.codebook.as_ref().filter(|cb| !cb.archetypes.is_empty()).is_some() {
                cloze_units += (env.lattice.programs.len() as u64).min(cap_u64) * (cloze_rounds as u64);
            }
        }
        if let Some(u) = &ui {
            if cloze_units > 0 {
                u.detail_bar(cloze_units.max(1), "cloze (fill-in-the-blank)");
            }
        }
        let eprint_cloze = ui.is_none();
        // One IndexedGenEnv per routing group; each has its own AlgebraicCodebook (`env.codebook`:
        // archetypes + slots — not related to `group_code_envs`). Cloze needs that structure per group.
        for (gidx, env) in svc.dm.group_gen_envs.iter_mut() {
            let Some(algebraic) = env.codebook.as_ref().filter(|cb| !cb.archetypes.is_empty()) else {
                continue;
            };
            println!(
                "  cloze[g{}]: generating tasks from {} programs (this can take a bit on large lattices)",
                gidx,
                env.lattice.programs.len()
            );
            // Build cloze tasks from the lattice programs (distilled training data).
            let training_pairs: Vec<(Vec<f32>, String)> = env.lattice.programs.iter()
                .map(|prog| {
                    let text = prog.display_text(&env.dictionary);
                    (prog.ema_centroid.clone(), text)
                })
                .collect();
            let mut tasks = cloze::generate_cloze_tasks(algebraic, &env.dictionary, &training_pairs);
            if tasks.is_empty() { continue; }
            if tasks.len() > cloze::DEFAULT_MAX_CLOZE_TASKS_PER_GROUP {
                println!(
                    "  cloze[g{}]: capping {} → {} tasks (each task walks the full lattice)",
                    gidx,
                    tasks.len(),
                    cloze::DEFAULT_MAX_CLOZE_TASKS_PER_GROUP
                );
                tasks.shuffle(&mut rng);
                tasks.truncate(cloze::DEFAULT_MAX_CLOZE_TASKS_PER_GROUP);
            }
            // Unfreeze for cloze learning, re-freeze after.
            env.frozen = false;
            let mut round_stats = cloze::ClozeStats::default();
            for _round in 0..cloze_rounds {
                let stats = cloze::play_cloze_round(
                    env,
                    &tasks,
                    k_voters,
                    reward_rate,
                    punish_rate,
                    || {
                        if let Some(u) = &ui {
                            if cloze_units > 0 {
                                u.detail_inc(1);
                            }
                        }
                    },
                    eprint_cloze,
                );
                round_stats.games_played += stats.games_played;
                round_stats.total_slots += stats.total_slots;
                round_stats.correct_slots += stats.correct_slots;
                round_stats.reward_applied += stats.reward_applied;
                round_stats.punishment_applied += stats.punishment_applied;
            }
            env.frozen = true;
            println!("  cloze[g{}]: {} rounds x {} tasks, {}", gidx, cloze_rounds, tasks.len(), round_stats);
            total_stats.games_played += round_stats.games_played;
            total_stats.total_slots += round_stats.total_slots;
            total_stats.correct_slots += round_stats.correct_slots;
            total_stats.reward_applied += round_stats.reward_applied;
            total_stats.punishment_applied += round_stats.punishment_applied;
        }
        // Same cloze pass for code-output groups (each env has its own algebraic codebook).
        for (gidx, env) in svc.dm.group_code_envs.iter_mut() {
            let Some(algebraic) = env.codebook.as_ref().filter(|cb| !cb.archetypes.is_empty()) else {
                continue;
            };
            println!(
                "  cloze[code_g{}]: generating tasks from {} programs",
                gidx,
                env.lattice.programs.len()
            );
            let training_pairs: Vec<(Vec<f32>, String)> = env.lattice.programs.iter()
                .map(|prog| {
                    let text = prog.display_text(&env.dictionary);
                    (prog.ema_centroid.clone(), text)
                })
                .collect();
            let mut tasks = cloze::generate_cloze_tasks(algebraic, &env.dictionary, &training_pairs);
            if tasks.is_empty() { continue; }
            if tasks.len() > cloze::DEFAULT_MAX_CLOZE_TASKS_PER_GROUP {
                println!(
                    "  cloze[code_g{}]: capping {} → {} tasks",
                    gidx,
                    tasks.len(),
                    cloze::DEFAULT_MAX_CLOZE_TASKS_PER_GROUP
                );
                tasks.shuffle(&mut rng);
                tasks.truncate(cloze::DEFAULT_MAX_CLOZE_TASKS_PER_GROUP);
            }
            env.frozen = false;
            let mut round_stats = cloze::ClozeStats::default();
            for _round in 0..cloze_rounds {
                let stats = cloze::play_cloze_round(
                    env,
                    &tasks,
                    k_voters,
                    reward_rate,
                    punish_rate,
                    || {
                        if let Some(u) = &ui {
                            if cloze_units > 0 {
                                u.detail_inc(1);
                            }
                        }
                    },
                    eprint_cloze,
                );
                round_stats.games_played += stats.games_played;
                round_stats.total_slots += stats.total_slots;
                round_stats.correct_slots += stats.correct_slots;
                round_stats.reward_applied += stats.reward_applied;
                round_stats.punishment_applied += stats.punishment_applied;
            }
            env.frozen = true;
            println!("  cloze[code_g{}]: {} rounds × {} tasks, {}", gidx, cloze_rounds, tasks.len(), round_stats);
            total_stats.games_played += round_stats.games_played;
            total_stats.total_slots += round_stats.total_slots;
            total_stats.correct_slots += round_stats.correct_slots;
            total_stats.reward_applied += round_stats.reward_applied;
            total_stats.punishment_applied += round_stats.punishment_applied;
        }
        println!("  TOTAL: {}", total_stats);
        if let Some(u) = &ui {
            u.detail_finish_clear();
        }
    }

    // ---------------------------------------------------------------
    // Build final paramecium from trained codebooks.
    // This captures the learned archetype structure so the lattice is
    // available for curriculum-guided continuum learning at runtime.
    // ---------------------------------------------------------------
    bump_train_phase(&ui, &mut major_phase, "Paramecium + export brain package");
    println!("\n--- Building Post-Training Paramecium ---");
    svc.build_paramecium();
    if let Some(ref pm) = svc.paramecium {
        println!("  programs: {}, memory: {} bytes", pm.program_count(), pm.memory_bytes());
    }

    // ---------------------------------------------------------------
    // Export BrainPackage: binary envelope (metadata JSON + DimensionManager JSON + personality JSON).
    // Router, classifier, generation heads, group gen/code envs, etc. live inside the checkpoint.
    // ---------------------------------------------------------------
    println!("\n--- Exporting Brain Package ---");
    let brain_bytes = svc.export_brain()?;
    let size_kb = brain_bytes.len() / 1024;
    std::fs::write(output_path, &brain_bytes).map_err(|e| format!("write failed: {}", e))?;
    println!("Brain exported: {} ({} KB)", output_path, size_kb);
    println!("  Groups: {}", svc.dm.main.group_order.len());
    println!("  Router: {}", svc.dm.observer.learned_router.is_some());
    println!("  ActionClassifier: {}", svc.dm.action_classifier.is_some());
    println!("  GenerationHead (legacy): {}", svc.dm.generation_head.is_some());
    println!("  CodegenHead (legacy): {}", svc.dm.codegen_head.is_some());
    println!("  GroupGenEnvs: {} groups", svc.dm.group_gen_envs.len());
    println!("  GroupCodeEnvs: {} groups", svc.dm.group_code_envs.len());
    for (gidx, env) in &svc.dm.group_gen_envs {
        println!("    gen[{}]: {} lattice programs, frozen={}", gidx, env.program_count(), env.frozen);
    }
    for (gidx, env) in &svc.dm.group_code_envs {
        println!("    code[{}]: {} lattice programs, frozen={}", gidx, env.program_count(), env.frozen);
    }

    println!("\n--- Post-Training Inference Check ---");
    let skip_prompts = true; // Set to true to skip prompts
    if !skip_prompts {


        let test_prompts = [
            "help me reset my password",
            "implement binary search in Python",
            "explain the observer pattern",
            "design a microservices architecture in Rust",
            "my account is locked after too many failed attempts",
            "write an addition function in Rust",
            "explain the factory pattern using a restaurant analogy",
            // Circle+spiral composition tests: novel combinations of trained skills
            "write a subtraction function in Rust",
            "write a multiplication function in Rust",
            "implement a stack using an enum in Rust",
            "explain how to combine iterators with error handling",
            "what is the pattern for a struct with methods in Rust",
        ];
        for prompt in &test_prompts {
            println!("\n  prompt: {:?}", prompt);
            let action_result = svc.dm.route_text_to_action_stateless(prompt);
            let is_coding = matches!(
                &action_result,
                Ok(ref a) if matches!(a.action_type, growformer::dimension::action::ActionType::CodingAssist)
            );
            if let Ok(ref action) = action_result {
                println!("  action: {:?} (conf={:.2}) group={:?}", action.action_type, action.confidence, action.target_group_id);
            }
            if let Ok((_, resp)) = svc.generation(prompt) {
                let r = &resp.text;
                let r_end = truncate_to_char_boundary(r, 200);
                println!("  gen [{}] (conf={:.2}): {:?}", resp.template_id, resp.confidence, &r[..r_end]);
            }
            if is_coding {
                if let Ok((_, Some(code))) = svc.codegen(prompt) {
                    let c = &code.code;
                    let c_end = truncate_to_char_boundary(c, 200);
                    println!("  code [{}]: {:?}", code.kind, &c[..c_end]);
                }
            }
        }

        if validate {
            use growformer::dimension::action::ActionType;
            let checks: [(&str, ActionType); 3] = [
                ("help me reset my password", ActionType::SupportTicket),
                ("implement binary search in Python", ActionType::CodingAssist),
                ("explain the observer pattern", ActionType::CodingAssist),
            ];
            for (prompt, expected) in &checks {
                let action = svc.dm.route_text_to_action_stateless(prompt).map_err(|e| format!("route_text_to_action failed: {}", e))?;
                if action.action_type != *expected {
                    return Err(format!(
                        "validate: prompt {:?} expected action {:?}, got {:?}",
                        prompt, expected, action.action_type
                    ));
                }
            }
            println!("\n  Validate: action routing checks passed.");
        }    
    }
    println!("\n=== Brain training complete ===");
    if let Some(u) = &ui {
        u.finish_ok("Brain training complete");
    }
    Ok(())
}

// =============================================================================
// Data loading
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
    #[serde(default)]
    causal: Option<CausalAnnotation>,
}

fn load_language_samples_jsonl(path: &str) -> Result<Vec<LanguageSample>, String> {
    use std::io::BufRead;
    let file = std::fs::File::open(path).map_err(|e| format!("open failed: {}", e))?;
    let reader = std::io::BufReader::new(file);
    let mut out = Vec::new();
    for (idx, line) in reader.lines().enumerate() {
        let line = line.map_err(|e| format!("line {} read failed: {}", idx + 1, e))?;
        if line.trim().is_empty() {
            continue;
        }
        let rec: JsonlLanguageSample = serde_json::from_str(&line)
            .map_err(|e| format!("line {} json parse failed: {}", idx + 1, e))?;
        let intent = rec
            .semantic_intent
            .or(rec.intent)
            .unwrap_or_else(|| "unknown_intent".to_string());
        out.push(LanguageSample {
            domain: rec.domain.unwrap_or_else(|| "custom".to_string()),
            text: rec.text,
            semantic_intent: intent,
            action_target: rec.action_target,
            policy_regime: rec.policy_regime.unwrap_or_else(|| "default".to_string()),
            language_channel: rec.language_channel.unwrap_or_else(|| "english".to_string()),
            expected_response: rec.expected_response,
            expected_code: rec.expected_code,
            causal: rec.causal,
        });
    }
    Ok(out)
}

/// Load all `train_*.jsonl` from a directory into `all`.
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
        println!("--- Agent behavioral data (data/agent) ---");
        load_train_jsonl_dir(&mut all, agent)?;
    }
    let routekit = std::path::Path::new("data/routekit");
    if routekit.exists() {
        println!("--- RouteKit routing data (data/routekit) ---");
        load_train_jsonl_dir(&mut all, routekit)?;
    }
    Ok(all)
}

