use growformer::dimension::{
    LanguageSample,
    GroupGenEnv, action_target_to_type,
};
use growformer::dimension::group_gen::AlgebraicCodebook;
use growformer::spectral::TokenDictionary;
use std::collections::HashMap;
use growformer::service::LanguageService;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand::rngs::StdRng;
use serde::Deserialize;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "growformer", version, about = "Growformer — train and run specialized neural brains")]
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

    /// Output path for trained brain binary.
    #[arg(long, value_name = "PATH", default_value = "brain.bin")]
    brain_output: String,

    /// Run inference on a trained brain.
    #[arg(long)]
    infer: bool,

    /// Path to brain.bin file to load for inference (default: brain.bin).
    #[arg(long, value_name = "PATH", default_value = "brain.bin")]
    brain: String,

    /// Prompt text for single-shot inference. Omit for interactive mode.
    #[arg(long, value_name = "TEXT")]
    prompt: Option<String>,
}

fn main() {
    let args = Args::parse();

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
            args.brain_epochs,
            &args.brain_output,
            max_samples,
            quick_gen_epochs,
            args.brain_gen_epochs,
            args.brain_gen_replicas,
            args.validate_brain_training,
            args.auto,
        ) {
            eprintln!("Failed to train brain: {}", e);
            std::process::exit(1);
        }
    } else if args.infer {
        if let Err(e) = run_inference(&args.brain, args.prompt.as_deref()) {
            eprintln!("Inference failed: {}", e);
            std::process::exit(1);
        }
    } else {
        println!("Growformer — train and run specialized neural brains\n");
        println!("Usage:");
        println!("  Train:  cargo run --release -- --train-brain [--auto]");
        println!("  Infer:  cargo run --release -- --infer [--prompt \"your question\"]");
        println!("  Demos:  cargo run --bin growformer-demos -- --help");
        println!("\nRun with --help for all options.");
        std::process::exit(1);
    }
}

// =============================================================================
// Inference: load brain.bin and run prompts
// =============================================================================

fn run_inference(brain_path: &str, prompt: Option<&str>) -> Result<(), String> {
    let data = std::fs::read(brain_path)
        .map_err(|e| format!("Failed to read {}: {}", brain_path, e))?;

    let mut svc = LanguageService::new_default()?;
    svc.load_brain(&data)?;

    let dm = svc.active_dm();
    let n_groups = dm.main.group_order.len();
    let n_gen = dm.group_gen_envs.len();
    let n_code = dm.group_code_envs.len();
    let has_router = dm.observer.learned_router.is_some();
    let has_classifier = dm.action_classifier.is_some();

    println!("Brain loaded: {}", brain_path);
    println!("  Groups: {}", n_groups);
    println!("  Router: {}", has_router);
    println!("  Classifier: {}", has_classifier);
    println!("  Gen envs: {} groups", n_gen);
    println!("  Code envs: {} groups", n_code);
    for (gidx, env) in &dm.group_gen_envs {
        println!("    gen[{}]: {} tokens in dict, {} neurons, {} synapses",
            gidx, env.dictionary.len(), env.total_neurons(), env.total_synapses());
    }
    for (gidx, env) in &dm.group_code_envs {
        println!("    code[{}]: {} tokens in dict, {} neurons, {} synapses",
            gidx, env.dictionary.len(), env.total_neurons(), env.total_synapses());
    }

    if let Some(prompt_text) = prompt {
        run_single_prompt(&mut svc, prompt_text);
        return Ok(());
    }

    println!("\nInteractive mode. Type a prompt and press Enter. Ctrl+C to exit.\n");
    let stdin = std::io::stdin();
    loop {
        eprint!("> ");
        let _ = std::io::Write::flush(&mut std::io::stderr());
        let mut line = String::new();
        match stdin.read_line(&mut line) {
            Ok(0) => break,
            Ok(_) => {
                let trimmed = line.trim();
                if trimmed.is_empty() { continue; }
                if trimmed == "quit" || trimmed == "exit" { break; }
                run_single_prompt(&mut svc, trimmed);
                println!();
            }
            Err(e) => {
                eprintln!("Read error: {}", e);
                break;
            }
        }
    }

    Ok(())
}

fn run_single_prompt(svc: &mut LanguageService, prompt: &str) {
    if let Ok(action) = svc.active_dm_mut().route_text_to_action_stateless(prompt) {
        println!("  route: {:?} (conf={:.2}) group={:?}",
            action.action_type, action.confidence, action.target_group_id);
    }

    match svc.generation(prompt) {
        Ok((_, resp)) => {
            if !resp.text.is_empty() {
                println!("  gen: {}", resp.text);
            }
        }
        Err(e) => eprintln!("  gen error: {}", e),
    }

    match svc.codegen(prompt) {
        Ok((_, Some(code))) => {
            if !code.code.is_empty() {
                println!("  code [{}]: {}", code.kind, code.code);
            }
        }
        Ok((_, None)) => {}
        Err(e) => eprintln!("  code error: {}", e),
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

fn auto_configure(profile: &DataProfile) -> AutoConfig {
    let global_max_tok = profile.global_max_response_tokens.max(profile.global_max_code_tokens);
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
    let router_epochs = (profile.num_groups * 150).clamp(300, 1000);
    let classifier_epochs = (profile.num_groups * 120).clamp(300, 800);
    let classifier_lr = if profile.total_samples > 300 { 0.02 } else { 0.03 };

    let max_synapses = if est_output > 800 { 250 } else { 200 };
    let energy_budget = if est_output > 800 { 30.0 } else { 25.0 };

    let available_cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let num_tasks = profile.groups.len() * if profile.has_code { 2 } else { 1 };
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
    println!("  Data: {} samples, {} groups, code={}", profile.total_samples, profile.num_groups, profile.has_code);
    println!("  Max tokens in data: response={}, code={}", profile.global_max_response_tokens, profile.global_max_code_tokens);
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

fn train_brain(
    epochs: u32,
    output_path: &str,
    max_samples: usize,
    quick_gen_epochs: u32,
    gen_epochs_override: u32,
    gen_replicas: u32,
    validate: bool,
    auto: bool,
) -> Result<(), String> {
    println!("=== Full Neural Brain Training ===\n");
    if validate {
        println!("(validate mode: capped samples and gen epochs, will assert inference)\n");
    }
    let mut rng = StdRng::seed_from_u64(42);

    let mut samples = load_all_m5_training_data()?;
    println!("Loaded {} training samples", samples.len());
    if max_samples > 0 && samples.len() > max_samples {
        samples.shuffle(&mut rng);
        samples.truncate(max_samples);
        let support_n = samples.iter().filter(|s| s.action_target.as_deref() == Some("support")).count();
        let coding_n = samples.iter().filter(|s| s.action_target.as_deref() == Some("coding")).count();
        let other_n = samples.len() - support_n - coding_n;
        println!("  shuffled and truncated to {} (support={}, coding={}, other={})", samples.len(), support_n, coding_n, other_n);
    }

    let mut svc = LanguageService::new_default()?;

    // Compute both raw (384-dim) and bridged (64-dim) embeddings.
    // Bridged vectors for routing/classification; raw vectors for generation conditioning.
    // When the `parallel` feature is enabled, embedding computation runs in parallel across cores.
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
                bridged_embeddings.push(vec![0.0; 64]);
            }
        }
    }
    let raw_dim = raw_embeddings.first().map_or(384, |e| e.len());
    let bridge_dim = bridged_embeddings.first().map_or(64, |e| e.len());
    println!("  {} samples: raw={}d, bridged={}d", samples.len(), raw_dim, bridge_dim);

    let num_groups = svc.dm.main.group_order.len();
    let group_map = |s: &LanguageSample| -> usize {
        match s.action_target.as_deref() {
            Some("support") => 0,
            Some("coding") => 1.min(num_groups - 1),
            _ => 1.min(num_groups - 1),
        }
    };

    // ---------------------------------------------------------------
    // Auto-configuration: profile data and derive parameters
    // ---------------------------------------------------------------
    let auto_cfg = if auto {
        let profile = profile_training_data(&samples, &group_map, num_groups.max(1));
        for (gidx, gp) in &profile.groups {
            println!("  group {}: gen={} code={} max_resp_tok={} max_code_tok={} avg_resp_tok={:.0} intents={}",
                gidx, gp.gen_count, gp.code_count, gp.max_response_tokens,
                gp.max_code_tokens, gp.avg_response_tokens, gp.unique_intents);
        }
        Some(auto_configure(&profile))
    } else {
        None
    };

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
    println!("  Router loss={:.4} accuracy={:.1}% ({} epochs)", router_loss, router_acc * 100.0, router_epochs);

    // ---------------------------------------------------------------
    // Stage 2: Train ActionClassifier
    // ---------------------------------------------------------------
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
    println!("  Classifier loss={:.4} accuracy={:.1}%", clf_loss, clf_acc * 100.0);

    // ---------------------------------------------------------------
    // Stages 3+4: Per-group generation envs (Growformer substrate)
    // Each group gets its own NeuralEnvironment for text AND code generation.
    // Conditioning = bridged_embedding (64d) — routing already selected the group.
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
        }
    });
    println!("\n--- Stages 3+4: Per-Group Generation Envs (single-pass token prediction) ---");
    println!("  gen_epochs={}, num_groups={}, replicas_per_task={}", gen_epochs, num_groups, k_replicas);

    // Partition training data by (group, kind)
    let mut gen_by_group: HashMap<usize, Vec<(&[f32], &str)>> = HashMap::new();
    let mut code_by_group: HashMap<usize, Vec<(&[f32], &str)>> = HashMap::new();
    for (bridged, s) in bridged_embeddings.iter().zip(samples.iter()) {
        let gidx = group_map(s);
        if let Some(r) = s.expected_response.as_deref() {
            gen_by_group.entry(gidx).or_default().push((bridged.as_slice(), r));
        }
        if let Some(c) = s.expected_code.as_deref() {
            if !c.is_empty() && c != "null" {
                code_by_group.entry(gidx).or_default().push((bridged.as_slice(), c));
            }
        }
    }

    // Build per-group dictionaries from each group's own data.
    // Larger groups get larger dictionaries (2048 max), smaller groups get 1024.
    use growformer::dimension::group_gen::{bits_for_dict, MAX_TOKENS};
    let effective_max_tokens = gen_overrides.as_ref().and_then(|o| o.max_tokens).unwrap_or(MAX_TOKENS);
    let mut gen_dicts: HashMap<usize, TokenDictionary> = HashMap::new();
    let mut code_dicts: HashMap<usize, TokenDictionary> = HashMap::new();
    let pairs_threshold_for_large_dict: usize = 100;
    for (&gidx, pairs) in &gen_by_group {
        if pairs.is_empty() { continue; }
        let texts: Vec<&str> = pairs.iter().map(|(_emb, text)| *text).collect();
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
    let base_archetypes = 16usize;
    let mut gen_codebooks: HashMap<usize, AlgebraicCodebook> = HashMap::new();
    let mut code_codebooks: HashMap<usize, AlgebraicCodebook> = HashMap::new();
    for (&gidx, pairs) in &gen_by_group {
        if pairs.is_empty() { continue; }
        let texts: Vec<&str> = pairs.iter().map(|(_emb, text)| *text).collect();
        let dict = gen_dicts.get(&gidx).unwrap();
        let gen_max_arch = if texts.len() > 150 {
            base_archetypes * 3  // 48 for large gen groups
        } else if texts.len() > 50 {
            base_archetypes * 2  // 32 for medium groups
        } else {
            base_archetypes      // 16 for small groups
        };
        let cb = AlgebraicCodebook::build(&texts, dict, gen_max_arch);
        println!("  gen codebook[g{}]: {} archetypes (max={}), {} slots max, {} total bits (was {})",
            gidx, cb.archetypes.len(), gen_max_arch, cb.max_slot_count, cb.total_bits,
            effective_max_tokens * bits_for_dict(dict.len()));
        gen_codebooks.insert(gidx, cb);
    }
    for (&gidx, pairs) in &code_by_group {
        if pairs.is_empty() { continue; }
        let texts: Vec<&str> = pairs.iter().map(|(_emb, text)| *text).collect();
        let dict = code_dicts.get(&gidx).unwrap();
        let code_max_arch = base_archetypes * 2; // code has more unique structural patterns
        let cb = AlgebraicCodebook::build_syntax_aware(&texts, dict, code_max_arch);
        println!("  code codebook[g{}]: {} archetypes, {} slots max, {} total bits (was {}) [SYNTAX-AWARE]",
            gidx, cb.archetypes.len(), cb.max_slot_count, cb.total_bits,
            effective_max_tokens * bits_for_dict(dict.len()));
        code_codebooks.insert(gidx, cb);
    }

    // Store first available dict as fallback for service inference
    if let Some(d) = gen_dicts.values().next() {
        svc.dm.gen_dictionary = Some(d.clone());
    }
    if let Some(d) = code_dicts.values().next() {
        svc.dm.code_dictionary = Some(d.clone());
    }

    // Pruning warmup: stop pruning after 30% of training ticks.
    // Each epoch = num_pairs ticks (one train_tick per sample).
    // Warmup fraction: 30% — enough for the substrate to self-organize,
    // then freeze structure so learning doesn't erode capacity.
    let prune_warmup_frac = 0.3;

    let early_stop_window = auto_cfg.as_ref().map(|ac| ac.early_stop_window).unwrap_or(0);
    let early_stop_min_imp = auto_cfg.as_ref().map(|ac| ac.early_stop_min_improvement).unwrap_or(0.0);
    let early_stop_min_ep = auto_cfg.as_ref().map(|ac| ac.early_stop_min_epochs).unwrap_or(0);

    struct GenTask<'a> {
        gidx: usize,
        kind: &'static str,
        replica: usize,
        pairs: &'a [(&'a [f32], &'a str)],
        seed: u64,
        dictionary: TokenDictionary,
        codebook: Option<AlgebraicCodebook>,
        prune_stop_tick: u64,
        task_epochs: usize,
        overrides: Option<growformer::dimension::group_gen::GenEnvOverrides>,
        es_window: usize,
        es_min_imp: f32,
        es_min_ep: usize,
    }
    let mut tasks: Vec<GenTask> = Vec::new();
    let mut base_tasks = 0usize;
    for gidx in 0..num_groups {
        if let Some(p) = gen_by_group.get(&gidx) {
            if !p.is_empty() {
                let task_epochs = if p.len() > 150 {
                    (gen_epochs as f64 * 2.0) as usize
                } else {
                    gen_epochs
                };
                let total_ticks = task_epochs as u64 * p.len() as u64;
                let stop_tick = (total_ticks as f64 * prune_warmup_frac) as u64;
                let dict = gen_dicts.get(&gidx).unwrap().clone();
                let cb = gen_codebooks.get(&gidx).cloned();
                for r in 0..k_replicas {
                    tasks.push(GenTask {
                        gidx, kind: "gen", replica: r,
                        pairs: p.as_slice(),
                        seed: 100 + gidx as u64 * 1000 + r as u64,
                        dictionary: dict.clone(),
                        codebook: cb.clone(),
                        prune_stop_tick: stop_tick,
                        task_epochs,
                        overrides: gen_overrides.clone(),
                        es_window: early_stop_window,
                        es_min_imp: early_stop_min_imp,
                        es_min_ep: early_stop_min_ep,
                    });
                }
                base_tasks += 1;
            }
        }
        if let Some(p) = code_by_group.get(&gidx) {
            if !p.is_empty() {
                let task_epochs = gen_epochs;
                let total_ticks = task_epochs as u64 * p.len() as u64;
                let stop_tick = (total_ticks as f64 * prune_warmup_frac) as u64;
                let dict = code_dicts.get(&gidx).unwrap().clone();
                let cb = code_codebooks.get(&gidx).cloned();
                for r in 0..k_replicas {
                    tasks.push(GenTask {
                        gidx, kind: "code", replica: r,
                        pairs: p.as_slice(),
                        seed: 200 + gidx as u64 * 1000 + r as u64,
                        dictionary: dict.clone(),
                        codebook: cb.clone(),
                        prune_stop_tick: stop_tick,
                        task_epochs,
                        overrides: gen_overrides.clone(),
                        es_window: early_stop_window,
                        es_min_imp: early_stop_min_imp,
                        es_min_ep: early_stop_min_ep,
                    });
                }
                base_tasks += 1;
            }
        }
    }
    println!("  {} base tasks × {} replicas = {} total threads",
        base_tasks, k_replicas, tasks.len());
    println!("  pruning warmup: {:.0}% of training ticks", prune_warmup_frac * 100.0);
    for gidx in 0..num_groups {
        if let Some(p) = gen_by_group.get(&gidx) {
            if !p.is_empty() { println!("  task: group {} gen — {} pairs", gidx, p.len()); }
        }
        if let Some(p) = code_by_group.get(&gidx) {
            if !p.is_empty() { println!("  task: group {} code — {} pairs", gidx, p.len()); }
        }
    }

    let results: Vec<(usize, &str, usize, f32, GroupGenEnv)> = std::thread::scope(|s| {
        let handles: Vec<_> = tasks.iter().map(|task| {
            s.spawn(move || {
                let mut task_rng = StdRng::seed_from_u64(task.seed);
                let ov = task.overrides.as_ref();
                let default_ov = growformer::dimension::group_gen::GenEnvOverrides::default();
                let ov_ref = ov.unwrap_or(&default_ov);
                let mut env = if let Some(cb) = task.codebook.clone() {
                    GroupGenEnv::new_algebraic(task.dictionary.clone(), cb, ov_ref, &mut task_rng)
                } else if let Some(ov) = &task.overrides {
                    GroupGenEnv::new_with_overrides(task.dictionary.clone(), ov, &mut task_rng)
                } else {
                    GroupGenEnv::new(task.dictionary.clone(), &mut task_rng)
                };
                env.set_prune_stop_tick(task.prune_stop_tick);
                let te = task.task_epochs;
                let log_interval = (te / 50).max(1);
                let mode_str = if env.codebook.is_some() { " ALGEBRAIC" } else { "" };
                println!("    [{} g{} r{}]{} output_dim={} epochs={} prune_stop_tick={}{}",
                    task.kind, task.gidx, task.replica, mode_str,
                    env.output_dim, te, task.prune_stop_tick,
                    if task.es_window > 0 { format!(" early_stop(win={} min_imp={:.4} min_ep={})", task.es_window, task.es_min_imp, task.es_min_ep) } else { String::new() });
                let n_pairs = task.pairs.len();
                let mut indices: Vec<usize> = (0..n_pairs).collect();
                let mut sample_losses: Vec<f32> = vec![0.0; n_pairs];
                let replay_frac = 0.15;
                let replay_start_epoch = 50;
                let replay_count = (n_pairs as f32 * replay_frac).ceil() as usize;

                // Convergence monitor state
                let mut loss_history: Vec<f32> = Vec::new();
                let mut stopped_early = false;

                for epoch in 0..te {
                    indices.shuffle(&mut task_rng);
                    let mut total_loss = 0.0f32;
                    for &i in &indices {
                        let l = env.train_step(task.pairs[i].0, task.pairs[i].1, &mut task_rng);
                        sample_losses[i] = l;
                        total_loss += l;
                    }
                    if epoch >= replay_start_epoch && replay_count > 0 {
                        let mut ranked: Vec<(usize, f32)> = sample_losses.iter().copied().enumerate().collect();
                        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                        for &(idx, _loss) in ranked.iter().take(replay_count) {
                            env.train_step(task.pairs[idx].0, task.pairs[idx].1, &mut task_rng);
                        }
                    }
                    let last_avg = total_loss / n_pairs.max(1) as f32;
                    loss_history.push(last_avg);

                    // Early stopping: check if loss has plateaued
                    if task.es_window > 0 && epoch >= task.es_min_ep && loss_history.len() >= task.es_window {
                        let recent = &loss_history[loss_history.len() - task.es_window..];
                        let old_avg = recent[..task.es_window / 2].iter().sum::<f32>() / (task.es_window / 2) as f32;
                        let new_avg = recent[task.es_window / 2..].iter().sum::<f32>() / (task.es_window - task.es_window / 2) as f32;
                        let improvement = (old_avg - new_avg) / old_avg.max(1e-8);
                        if improvement < task.es_min_imp {
                            println!("    [{} g{} r{}] early stop at epoch {}/{}: loss={:.4} improvement={:.5} < {:.4}",
                                task.kind, task.gidx, task.replica, epoch, te, last_avg, improvement, task.es_min_imp);
                            stopped_early = true;
                            break;
                        }
                    }

                    if epoch % log_interval == 0 || epoch == te - 1 {
                        if k_replicas > 1 {
                            println!("    [{} g{} r{}] epoch {}/{} loss={:.4} synapses={}",
                                task.kind, task.gidx, task.replica, epoch, te,
                                last_avg, env.total_synapses());
                        } else {
                            println!("    [{} g{}] epoch {}/{} loss={:.4} synapses={}",
                                task.kind, task.gidx, epoch, te,
                                last_avg, env.total_synapses());
                        }
                    }
                }
                let actual_epochs = loss_history.len();
                if stopped_early {
                    println!("    [{} g{} r{}] converged after {} of {} epochs",
                        task.kind, task.gidx, task.replica, actual_epochs, te);
                }
                let mut eval_loss = 0.0f32;
                for &(cond, target) in task.pairs {
                    eval_loss += env.eval_loss(cond, target);
                }
                let final_loss = eval_loss / task.pairs.len().max(1) as f32;
                env.freeze();
                println!("    [{} g{} r{}] frozen: {} neurons, {} synapses, eval_loss={:.4}",
                    task.kind, task.gidx, task.replica,
                    env.total_neurons(), env.total_synapses(), final_loss);
                (task.gidx, task.kind, task.replica, final_loss, env)
            })
        }).collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    // Select best replica per (group, kind) by lowest eval loss
    let mut best: HashMap<(usize, &str), (f32, GroupGenEnv)> = HashMap::new();
    for (gidx, kind, replica, loss, env) in results {
        let key = (gidx, kind);
        let is_better = match best.get(&key) {
            Some((prev_loss, _)) => loss < *prev_loss,
            None => true,
        };
        if is_better {
            if k_replicas > 1 {
                println!("    [{} g{}] best replica so far: r{} loss={:.4}", kind, gidx, replica, loss);
            }
            best.insert(key, (loss, env));
        }
    }

    for ((gidx, kind), (_loss, env)) in best {
        match kind {
            "gen" => { svc.dm.group_gen_envs.insert(gidx, env); }
            "code" => { svc.dm.group_code_envs.insert(gidx, env); }
            _ => {}
        }
    }

    // ---------------------------------------------------------------
    // Export brain
    // ---------------------------------------------------------------
    println!("\n--- Exporting Brain ---");
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
        println!("    gen[{}]: {} neurons, {} synapses, frozen={}", gidx, env.total_neurons(), env.total_synapses(), env.frozen);
    }
    for (gidx, env) in &svc.dm.group_code_envs {
        println!("    code[{}]: {} neurons, {} synapses, frozen={}", gidx, env.total_neurons(), env.total_synapses(), env.frozen);
    }

    println!("\n--- Post-Training Inference Check ---");
    let test_prompts = [
        "help me reset my password",
        "implement binary search in Python",
        "explain the observer pattern",
        "design a microservices architecture in Rust",
        "my account is locked after too many failed attempts",
    ];
    for prompt in &test_prompts {
        println!("\n  prompt: {:?}", prompt);
        if let Ok(action) = svc.dm.route_text_to_action_stateless(prompt) {
            println!("  action: {:?} (conf={:.2}) group={:?}", action.action_type, action.confidence, action.target_group_id);
        }
        if let Ok((_, resp)) = svc.generation(prompt) {
            let r = &resp.text;
            println!("  gen [{}]: {:?}", resp.template_id, &r[..r.len().min(200)]);
        }
        if let Ok((_, Some(code))) = svc.codegen(prompt) {
            let c = &code.code;
            println!("  code [{}]: {:?}", code.kind, &c[..c.len().min(200)]);
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

    println!("\n=== Brain training complete ===");
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
        });
    }
    Ok(out)
}

fn load_all_m5_training_data() -> Result<Vec<LanguageSample>, String> {
    let dir = std::path::Path::new("data/language/m5");
    if !dir.exists() {
        return Err(format!("M5 data directory not found: {}", dir.display()));
    }
    let mut all = Vec::new();
    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .map_err(|e| format!("read_dir failed: {}", e))?
        .filter_map(|e| e.ok())
        .collect();
    entries.sort_by_key(|e| e.file_name());
    for entry in entries {
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with("train_") && name.ends_with(".jsonl") {
            let path = entry.path();
            let samples = load_language_samples_jsonl(path.to_str().unwrap())?;
            println!("  loaded {}: {} samples", name, samples.len());
            all.extend(samples);
        }
    }
    Ok(all)
}

