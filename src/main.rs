use growformer::dimension::{
    LanguageSample,
    GroupGenEnv, action_target_to_type,
};
use growformer::dimension::language::DEFAULT_BRIDGE_DIM;
use growformer::dimension::group_gen::{AlgebraicCodebook, HopfCompositionTable};
use growformer::dimension::paramecium::InfraciliaryLattice;
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

    /// Retrain only the gen env for a specific group index (loads existing brain, retrains one group, re-exports).
    #[arg(long, value_name = "GROUP_IDX")]
    retrain_gen: Option<usize>,
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
    } else if let Some(group_idx) = args.retrain_gen {
        println!("=============================================================");
        println!("  Growformer — Retrain Gen Group {}", group_idx);
        println!("=============================================================\n");
        if let Err(e) = retrain_single_gen(
            group_idx,
            &args.brain,
            &args.brain_output,
            args.brain_gen_epochs,
            args.brain_gen_replicas,
            args.auto,
        ) {
            eprintln!("Retrain failed: {}", e);
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
        println!("  Train:     cargo run --release -- --train-brain [--auto]");
        println!("  Retrain:   cargo run --release -- --retrain-gen 1 [--auto]");
        println!("  Infer:     cargo run --release -- --infer [--prompt \"your question\"]");
        println!("  Demos:     cargo run --bin growformer-demos -- --help");
        println!("\nRun with --help for all options.");
        std::process::exit(1);
    }
}

// =============================================================================
// Retrain a single gen group: loads existing brain, retrains one group, re-exports
// =============================================================================

fn retrain_single_gen(
    target_group: usize,
    brain_path: &str,
    output_path: &str,
    gen_epochs_override: u32,
    gen_replicas: u32,
    auto: bool,
) -> Result<(), String> {
    let data = std::fs::read(brain_path)
        .map_err(|e| format!("Failed to read {}: {}", brain_path, e))?;

    let mut svc = LanguageService::new_default()?;
    svc.load_brain(&data)?;
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

    let group_map = |s: &LanguageSample| -> usize {
        match s.action_target.as_deref() {
            Some("support") | Some("general") => 0,
            Some("patterns") => 1.min(num_groups - 1),
            Some("coding") => if num_groups >= 3 { 2.min(num_groups - 1) } else { 1.min(num_groups - 1) },
            Some("reasoning") => if num_groups >= 4 { 3.min(num_groups - 1) } else { 1.min(num_groups - 1) },
            _ => 0,
        }
    };

    // Embed only the target group's gen data
    println!("\n--- Computing embeddings for group {} ---", target_group);
    let runtime = &svc.dm.language_runtime;
    let mut gen_pairs: Vec<(Vec<f32>, String)> = Vec::new();
    for s in &samples {
        if group_map(s) != target_group { continue; }
        if let Some(r) = s.expected_response.as_deref() {
            match runtime.encode_and_bridge(&s.text) {
                Ok((_raw, bridged)) => {
                    gen_pairs.push((bridged.routed_vector, r.to_string()));
                }
                Err(_) => {
                    gen_pairs.push((vec![0.0; DEFAULT_BRIDGE_DIM], r.to_string()));
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
        Some(auto_configure(&profile))
    } else {
        None
    };

    let gen_epochs: usize = if gen_epochs_override > 0 {
        gen_epochs_override as usize
    } else if let Some(ac) = &auto_cfg {
        ac.gen_epochs
    } else {
        1500
    };
    let k_replicas = if let Some(ac) = &auto_cfg { ac.replicas } else { (gen_replicas as usize).max(1) };
    let gen_overrides = auto_cfg.as_ref().map(|ac| {
        growformer::dimension::group_gen::GenEnvOverrides {
            max_tokens: Some(ac.max_tokens),
            hidden: Some(ac.gen_hidden),
            k: Some(ac.gen_k),
            max_synapses: Some(ac.max_synapses),
            energy_budget: Some(ac.energy_budget),
            ..Default::default()
        }
    });

    let early_stop_window = auto_cfg.as_ref().map(|ac| ac.early_stop_window).unwrap_or(0);
    let early_stop_min_imp = auto_cfg.as_ref().map(|ac| ac.early_stop_min_improvement).unwrap_or(0.0);
    let early_stop_min_ep = auto_cfg.as_ref().map(|ac| ac.early_stop_min_epochs).unwrap_or(0);

    // Build dictionary, codebook, Hopf table for the target group
    use growformer::dimension::group_gen::{bits_for_dict, MAX_TOKENS};
    let effective_max_tokens = gen_overrides.as_ref().and_then(|o| o.max_tokens).unwrap_or(MAX_TOKENS);

    let texts: Vec<&str> = gen_pairs.iter().map(|(_, t)| t.as_str()).collect();
    let embs: Vec<&[f32]> = gen_pairs.iter().map(|(e, _)| e.as_slice()).collect();

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

    // Train replicas
    println!("\n--- Training gen g{} ({} epochs, {} replicas) ---",
        target_group, gen_epochs, k_replicas);

    let mut best_env: Option<GroupGenEnv> = None;
    let mut best_loss = f32::MAX;

    let log_interval = gen_epochs / 25 + 1;
    let replay_frac = 0.15;
    let replay_start_epoch = 50;

    for replica in 0..k_replicas as usize {
        let mut task_rng = StdRng::seed_from_u64(42 + replica as u64 * 1000 + target_group as u64 * 100);
        let ov = gen_overrides.clone().unwrap_or_default();
        let mut env = GroupGenEnv::new_algebraic(dict.clone(), cb.clone(), &ov, &mut task_rng);
        if let Some(ref h) = hopf {
            env.hopf_table = Some(h.clone());
        }

        let n_pairs = gen_pairs.len();
        let mut indices: Vec<usize> = (0..n_pairs).collect();
        let mut sample_losses: Vec<f32> = vec![0.0; n_pairs];
        let replay_count = (n_pairs as f32 * replay_frac).ceil() as usize;
        let mut loss_history: Vec<f32> = Vec::new();
        let mut stopped_early = false;

        for epoch in 0..gen_epochs {
            indices.shuffle(&mut task_rng);
            let mut total_loss = 0.0f32;

            for &idx in &indices {
                let l = env.train_step(&gen_pairs[idx].0, &gen_pairs[idx].1, &mut task_rng);
                sample_losses[idx] = l;
                total_loss += l;
            }

            // Priority replay: re-train on highest-loss samples
            if epoch >= replay_start_epoch && replay_count > 0 {
                let mut ranked: Vec<(usize, f32)> = sample_losses.iter().copied().enumerate().collect();
                ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                for &(idx, _) in ranked.iter().take(replay_count) {
                    env.train_step(&gen_pairs[idx].0, &gen_pairs[idx].1, &mut task_rng);
                }
            }

            let last_avg = total_loss / n_pairs.max(1) as f32;
            loss_history.push(last_avg);

            // Early stopping (with min loss floor)
            if early_stop_window > 0 && epoch >= early_stop_min_ep && loss_history.len() >= early_stop_window {
                let recent = &loss_history[loss_history.len() - early_stop_window..];
                let old_avg = recent[..early_stop_window / 2].iter().sum::<f32>() / (early_stop_window / 2) as f32;
                let new_avg = recent[early_stop_window / 2..].iter().sum::<f32>() / (early_stop_window - early_stop_window / 2) as f32;
                let improvement = (old_avg - new_avg) / old_avg.max(1e-8);
                if improvement < early_stop_min_imp && new_avg < 0.02 {
                    println!("    [gen g{} r{}] early stop at epoch {}/{}: loss={:.4} improvement={:.5}",
                        target_group, replica, epoch, gen_epochs, last_avg, improvement);
                    stopped_early = true;
                    break;
                }
                if improvement < early_stop_min_imp && new_avg >= 0.02 && epoch % log_interval == 0 {
                    println!("    [gen g{} r{}] epoch {}/{}: plateau (imp={:.5}) but loss={:.4} > 0.02, continuing...",
                        target_group, replica, epoch, gen_epochs, improvement, new_avg);
                }
            }

            if epoch % log_interval == 0 || epoch == gen_epochs - 1 {
                println!("    [gen g{} r{}] epoch {}/{} loss={:.4} synapses={}",
                    target_group, replica, epoch, gen_epochs, last_avg, env.env.total_synapses());
            }
        }

        let actual_epochs = loss_history.len();
        if stopped_early {
            println!("    [gen g{} r{}] converged after {} of {} epochs", target_group, replica, actual_epochs, gen_epochs);
        }

        // Eval loss
        let mut eval_loss = 0.0f32;
        for i in 0..n_pairs {
            eval_loss += env.eval_loss(&gen_pairs[i].0, &gen_pairs[i].1);
        }
        eval_loss /= n_pairs.max(1) as f32;

        env.freeze();
        println!("    [gen g{} r{}] frozen: {} neurons, {} synapses, eval_loss={:.4}",
            target_group, replica, env.env.layers.iter().map(|l| l.len()).sum::<usize>(),
            env.env.total_synapses(), eval_loss);

        if eval_loss < best_loss {
            best_loss = eval_loss;
            best_env = Some(env);
            println!("    [gen g{}] best replica so far: r{} loss={:.4}", target_group, replica, eval_loss);
        }
    }

    // Replace the gen env in the brain
    let dm = svc.active_dm_mut();
    if let Some(env) = best_env {
        dm.group_gen_envs.insert(target_group, env);
        println!("\n  Replaced gen env for group {} (eval_loss={:.4})", target_group, best_loss);
    } else {
        return Err("No successful replica".to_string());
    }

    // Re-export
    println!("\n--- Exporting Brain ---");
    let brain_bytes = svc.export_brain()?;
    let size_kb = brain_bytes.len() / 1024;
    std::fs::write(output_path, &brain_bytes).map_err(|e| format!("write failed: {}", e))?;
    println!("Brain exported: {} ({} KB)", output_path, size_kb);

    // Quick inference check
    println!("\n--- Post-Retrain Inference Check ---\n");
    let test_prompts = [
        "help me reset my password",
        "implement binary search in Python",
        "explain the observer pattern",
        "design a microservices architecture in Rust",
        "my account is locked after too many failed attempts",
    ];
    for prompt in &test_prompts {
        match svc.generation(prompt) {
            Ok((action, resp)) => {
                println!("  prompt: {:?}", prompt);
                println!("  action: {:?} (conf={:.2}) group={:?}",
                    action.action_type, action.confidence, action.target_group_id);
                println!("  gen [{}] (conf={:.2}): {:?}\n",
                    resp.template_id, resp.confidence,
                    &resp.text[..resp.text.len().min(200)]);
            }
            Err(e) => println!("  {:?} → ERROR: {}\n", prompt, e),
        }
    }

    Ok(())
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
        let hopf_info = if env.hopf_table.is_some() { "hopf=yes" } else { "hopf=no" };
        let cb_info = env.codebook.as_ref().map(|cb| format!("proto={} arch={}", cb.has_prototypes(), cb.archetypes.len())).unwrap_or_else(|| "no-codebook".to_string());
        println!("    gen[{}]: {} tokens in dict, {} neurons, {} synapses, {}, {}",
            gidx, env.dictionary.len(), env.total_neurons(), env.total_synapses(), hopf_info, cb_info);
    }
    for (gidx, env) in &dm.group_code_envs {
        println!("    code[{}]: {} tokens in dict, {} neurons, {} synapses",
            gidx, env.dictionary.len(), env.total_neurons(), env.total_synapses());
    }

    if let Some(prompt_text) = prompt {
        run_single_prompt(&mut svc, prompt_text);
        return Ok(());
    }

    run_conversation_repl(&mut svc);

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
                println!("  gen (conf={:.2}): {}", resp.confidence, resp.text);
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

// ---------------------------------------------------------------------------
// Tool execution — inline executors for the 4 built-in tools
// ---------------------------------------------------------------------------

fn execute_tool(call: &growformer::dimension::tool::ToolCallInfo) -> growformer::dimension::tool::ToolResult {
    use growformer::dimension::tool::ToolResult;
    match call.tool_name.as_str() {
        "calculator" => {
            let expr = call.arguments.get("expression").map(|s| s.as_str()).unwrap_or("");
            let result = eval_arithmetic(expr);
            ToolResult { tool_name: "calculator".into(), success: result.is_ok(), output: result.unwrap_or_else(|e| e) }
        }
        "file_reader" => {
            let path = call.arguments.get("path").map(|s| s.as_str()).unwrap_or("");
            match std::fs::read_to_string(path) {
                Ok(content) => {
                    let preview: String = content.lines().take(50).collect::<Vec<_>>().join("\n");
                    let total = content.lines().count();
                    let output = if total > 50 {
                        format!("{}\n... ({} more lines)", preview, total - 50)
                    } else {
                        preview
                    };
                    ToolResult { tool_name: "file_reader".into(), success: true, output }
                }
                Err(e) => ToolResult { tool_name: "file_reader".into(), success: false, output: e.to_string() },
            }
        }
        "code_runner" => {
            let code = call.arguments.get("code").map(|s| s.as_str()).unwrap_or("");
            let lang = call.arguments.get("language").map(|s| s.as_str()).unwrap_or("python");
            let (cmd, args) = match lang {
                "python" => ("python3", vec!["-c", code]),
                "bash" | "shell" => ("bash", vec!["-c", code]),
                "ruby" => ("ruby", vec!["-e", code]),
                "node" | "javascript" => ("node", vec!["-e", code]),
                _ => ("python3", vec!["-c", code]),
            };
            match std::process::Command::new(cmd)
                .args(&args)
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped())
                .output()
            {
                Ok(output) => {
                    let stdout = String::from_utf8_lossy(&output.stdout);
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    let text = if stdout.is_empty() { stderr.to_string() } else { stdout.to_string() };
                    let truncated = if text.len() > 2000 { format!("{}...(truncated)", &text[..2000]) } else { text };
                    ToolResult { tool_name: "code_runner".into(), success: output.status.success(), output: truncated }
                }
                Err(e) => ToolResult { tool_name: "code_runner".into(), success: false, output: e.to_string() },
            }
        }
        "web_search" => {
            let query = call.arguments.get("query").map(|s| s.as_str()).unwrap_or("");
            ToolResult {
                tool_name: "web_search".into(),
                success: false,
                output: format!("Web search not yet available. Query: {}", query),
            }
        }
        _ => ToolResult {
            tool_name: call.tool_name.clone(),
            success: false,
            output: format!("Unknown tool: {}", call.tool_name),
        },
    }
}

fn eval_arithmetic(expr: &str) -> Result<String, String> {
    let clean: String = expr.chars().filter(|c| c.is_ascii_digit() || " .+-*/()%".contains(*c)).collect();
    if clean.is_empty() { return Err("empty expression".into()); }
    let tokens = tokenize_math(&clean)?;
    let result = eval_tokens(&tokens)?;
    Ok(format!("{}", result))
}

#[derive(Debug, Clone)]
enum MathToken { Num(f64), Op(char), LParen, RParen }

fn tokenize_math(expr: &str) -> Result<Vec<MathToken>, String> {
    let mut tokens = Vec::new();
    let mut chars = expr.chars().peekable();
    while let Some(&ch) = chars.peek() {
        if ch.is_whitespace() { chars.next(); continue; }
        if ch.is_ascii_digit() || ch == '.' {
            let mut num = String::new();
            while let Some(&c) = chars.peek() {
                if c.is_ascii_digit() || c == '.' { num.push(c); chars.next(); } else { break; }
            }
            tokens.push(MathToken::Num(num.parse::<f64>().map_err(|e| e.to_string())?));
        } else if "+-*/%".contains(ch) {
            tokens.push(MathToken::Op(ch)); chars.next();
        } else if ch == '(' { tokens.push(MathToken::LParen); chars.next();
        } else if ch == ')' { tokens.push(MathToken::RParen); chars.next();
        } else { chars.next(); }
    }
    Ok(tokens)
}

fn eval_tokens(tokens: &[MathToken]) -> Result<f64, String> {
    let mut pos = 0;
    let result = parse_expr(tokens, &mut pos)?;
    Ok(result)
}

fn parse_expr(tokens: &[MathToken], pos: &mut usize) -> Result<f64, String> {
    let mut left = parse_term(tokens, pos)?;
    while *pos < tokens.len() {
        match tokens[*pos] {
            MathToken::Op('+') => { *pos += 1; left += parse_term(tokens, pos)?; }
            MathToken::Op('-') => { *pos += 1; left -= parse_term(tokens, pos)?; }
            _ => break,
        }
    }
    Ok(left)
}

fn parse_term(tokens: &[MathToken], pos: &mut usize) -> Result<f64, String> {
    let mut left = parse_factor(tokens, pos)?;
    while *pos < tokens.len() {
        match tokens[*pos] {
            MathToken::Op('*') => { *pos += 1; left *= parse_factor(tokens, pos)?; }
            MathToken::Op('/') => { *pos += 1; let r = parse_factor(tokens, pos)?; if r == 0.0 { return Err("division by zero".into()); } left /= r; }
            MathToken::Op('%') => { *pos += 1; let r = parse_factor(tokens, pos)?; if r == 0.0 { return Err("modulo by zero".into()); } left %= r; }
            _ => break,
        }
    }
    Ok(left)
}

fn parse_factor(tokens: &[MathToken], pos: &mut usize) -> Result<f64, String> {
    if *pos >= tokens.len() { return Err("unexpected end of expression".into()); }
    match &tokens[*pos] {
        MathToken::Num(n) => { let v = *n; *pos += 1; Ok(v) }
        MathToken::Op('-') => { *pos += 1; let v = parse_factor(tokens, pos)?; Ok(-v) }
        MathToken::LParen => {
            *pos += 1;
            let v = parse_expr(tokens, pos)?;
            if *pos < tokens.len() && matches!(tokens[*pos], MathToken::RParen) { *pos += 1; }
            Ok(v)
        }
        _ => Err(format!("unexpected token: {:?}", tokens[*pos])),
    }
}

fn run_conversation_repl(svc: &mut LanguageService) {
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
        eprint!("[turn {}] > ", svc.conversation.turn_count() + 1);
        let _ = std::io::Write::flush(&mut std::io::stderr());
        let mut line = String::new();
        match stdin.read_line(&mut line) {
            Ok(0) => break,
            Ok(_) => {
                let trimmed = line.trim();
                if trimmed.is_empty() { continue; }
                if trimmed == "quit" || trimmed == "exit" { break; }

                if let Some(cmd) = trimmed.strip_prefix('/') {
                    handle_repl_command(svc, cmd);
                    continue;
                }

                // Check for tool call first — execute inline if matched
                if let Some(tool_call) = svc.try_tool_call(trimmed) {
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

                    // Feed tool result back for a composed conversational response
                    match svc.generation_with_tool_result(trimmed, &result) {
                        Ok((_action, resp)) => {
                            if !resp.text.is_empty() && !resp.text.starts_with("[tool_call:") {
                                println!();
                                println!("  {}", resp.text);
                            }
                        }
                        Err(_) => {}
                    }
                } else {
                    match svc.converse(trimmed) {
                        Ok((action, resp)) => {
                            if let Some(gid) = action.target_group_id {
                                eprint!("  [route: {:?} g={} conf={:.2}] ",
                                    action.action_type, gid, action.confidence);
                            }
                            println!();
                            if !resp.text.is_empty() {
                                println!("  {} (conf={:.2})", resp.text, resp.confidence);
                            }

                            match svc.codegen(trimmed) {
                                Ok((_, Some(code))) if !code.code.is_empty() => {
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

fn handle_repl_command(svc: &mut LanguageService, cmd: &str) {
    use growformer::service::OceanProfile;

    let parts: Vec<&str> = cmd.split_whitespace().collect();
    match parts.first().copied() {
        Some("personality") | Some("p") => {
            match parts.get(1).copied() {
                Some("assistant") => {
                    svc.personality = OceanProfile::assistant();
                    println!("  Personality: assistant (balanced, professional)");
                }
                Some("creative") => {
                    svc.personality = OceanProfile::creative();
                    println!("  Personality: creative (open, enthusiastic)");
                }
                Some("engineer") => {
                    svc.personality = OceanProfile::engineer();
                    println!("  Personality: engineer (precise, structured)");
                }
                Some("analyst") => {
                    svc.personality = OceanProfile::analyst();
                    println!("  Personality: analyst (cautious, thorough)");
                }
                _ => {
                    println!("  Usage: /personality <assistant|creative|engineer|analyst>");
                }
            }
            let v = svc.personality.as_vec();
            println!("  [O={:.1} C={:.1} E={:.1} A={:.1} N={:.1}]", v[0], v[1], v[2], v[3], v[4]);
        }
        Some("ocean") => {
            if parts.len() == 6 {
                let vals: Vec<f32> = parts[1..6].iter()
                    .filter_map(|s| s.parse::<f32>().ok())
                    .collect();
                if vals.len() == 5 {
                    svc.personality = OceanProfile {
                        openness: vals[0].clamp(0.0, 1.0),
                        conscientiousness: vals[1].clamp(0.0, 1.0),
                        extraversion: vals[2].clamp(0.0, 1.0),
                        agreeableness: vals[3].clamp(0.0, 1.0),
                        neuroticism: vals[4].clamp(0.0, 1.0),
                    };
                    let v = svc.personality.as_vec();
                    println!("  Custom OCEAN: [O={:.1} C={:.1} E={:.1} A={:.1} N={:.1}]",
                        v[0], v[1], v[2], v[3], v[4]);
                } else {
                    println!("  Usage: /ocean 0.5 0.7 0.5 0.6 0.3");
                }
            } else {
                let v = svc.personality.as_vec();
                println!("  Current: [O={:.1} C={:.1} E={:.1} A={:.1} N={:.1}]", v[0], v[1], v[2], v[3], v[4]);
                println!("  Usage: /ocean <O> <C> <E> <A> <N>  (each 0.0-1.0)");
            }
        }
        Some("reset") => {
            svc.reset_conversation();
            println!("  Conversation cleared.");
        }
        Some("history") | Some("h") => {
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
                run_single_prompt(svc, &prompt);
            }
        }
        Some("status") => {
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
                let mut count = 0usize;
                index_directory(svc, path_buf, &mut count);
                println!("  Indexed {} files (hybrid AST-lite + semantic + relational)", count);

                // Try to load git history for edit correlation
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
                let status = if svc.paramecium.is_some() {
                    let p = svc.paramecium.as_ref().unwrap();
                    format!("{} programs, {} bytes", p.program_count(), p.memory_bytes())
                } else {
                    "not built yet (will auto-build on first use)".to_string()
                };
                println!("  Paramecium: {}", status);
                println!("  Usage: /paramecium <prompt>  or  /pm <prompt>");
            } else {
                match svc.paramecium_respond(&prompt) {
                    Ok((action, resp)) => {
                        println!("  [paramecium: {} conf={:.2}]", action.reason, action.confidence);
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
    let router_epochs = (profile.num_groups * 250).clamp(600, 1500);
    let classifier_epochs = (profile.num_groups * 200).clamp(500, 1200);
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

    // Compute both raw and bridged embeddings.
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
                bridged_embeddings.push(vec![0.0; DEFAULT_BRIDGE_DIM]);
            }
        }
    }
    let raw_dim = raw_embeddings.first().map_or(384, |e| e.len());
    let bridge_dim = bridged_embeddings.first().map_or(DEFAULT_BRIDGE_DIM, |e| e.len());
    println!("  {} samples: raw={}d, bridged={}d", samples.len(), raw_dim, bridge_dim);

    // ---------------------------------------------------------------
    // Paramecium pre-training: build lattice from embeddings to provide
    // curriculum signals and group-discovery diagnostics for training.
    // ---------------------------------------------------------------
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

    let num_groups = svc.dm.main.group_order.len();
    let group_map = |s: &LanguageSample| -> usize {
        match s.action_target.as_deref() {
            Some("support") | Some("general") => 0,
            Some("patterns") => 1.min(num_groups - 1),
            Some("coding") => if num_groups >= 3 { 2.min(num_groups - 1) } else { 1.min(num_groups - 1) },
            Some("reasoning") => if num_groups >= 4 { 3.min(num_groups - 1) } else { 1.min(num_groups - 1) },
            _ => 1.min(num_groups - 1),
        }
    };

    // ---------------------------------------------------------------
    // Auto-configuration: profile data and derive parameters
    // ---------------------------------------------------------------
    let (auto_cfg, data_profile) = if auto {
        let profile = profile_training_data(&samples, &group_map, num_groups.max(1));
        for (gidx, gp) in &profile.groups {
            println!("  group {}: gen={} code={} max_resp_tok={} max_code_tok={} avg_resp_tok={:.0} intents={}",
                gidx, gp.gen_count, gp.code_count, gp.max_response_tokens,
                gp.max_code_tokens, gp.avg_response_tokens, gp.unique_intents);
        }
        (Some(auto_configure(&profile)), Some(profile))
    } else {
        (None, None)
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
            ..Default::default()
        }
    });
    println!("\n--- Stages 3+4: Per-Group Generation Envs (single-pass token prediction) ---");
    println!("  gen_epochs={}, num_groups={}, replicas_per_task={}", gen_epochs, num_groups, k_replicas);

    // Partition training data by (group, kind), carrying novelty scores per sample.
    // Each pair now carries (bridged, raw, target) so per-group adapters can specialize conditioning.
    let mut gen_by_group: HashMap<usize, Vec<(&[f32], &str)>> = HashMap::new();
    let mut gen_raw_by_group: HashMap<usize, Vec<&[f32]>> = HashMap::new();
    let mut gen_novelty_by_group: HashMap<usize, Vec<f32>> = HashMap::new();
    let mut code_by_group: HashMap<usize, Vec<(&[f32], &str)>> = HashMap::new();
    let mut code_raw_by_group: HashMap<usize, Vec<&[f32]>> = HashMap::new();
    let mut code_novelty_by_group: HashMap<usize, Vec<f32>> = HashMap::new();
    for (i, ((bridged, raw), s)) in bridged_embeddings.iter().zip(raw_embeddings.iter()).zip(samples.iter()).enumerate() {
        let gidx = group_map(s);
        let nov = novelty_scores.get(i).copied().unwrap_or(0.0);
        if let Some(r) = s.expected_response.as_deref() {
            gen_by_group.entry(gidx).or_default().push((bridged.as_slice(), r));
            gen_raw_by_group.entry(gidx).or_default().push(raw.as_slice());
            gen_novelty_by_group.entry(gidx).or_default().push(nov);
        }
        if let Some(c) = s.expected_code.as_deref() {
            if !c.is_empty() && c != "null" {
                code_by_group.entry(gidx).or_default().push((bridged.as_slice(), c));
                code_raw_by_group.entry(gidx).or_default().push(raw.as_slice());
                code_novelty_by_group.entry(gidx).or_default().push(nov);
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
        let embs: Vec<&[f32]> = pairs.iter().map(|(emb, _text)| *emb).collect();
        let dict = gen_dicts.get(&gidx).unwrap();
        let intent_count = data_profile.as_ref()
            .and_then(|p| p.groups.get(&gidx))
            .map(|g| g.unique_intents)
            .unwrap_or(0);
        let gen_max_arch = if texts.len() > 150 || intent_count > 40 {
            base_archetypes * 6  // 96 for large/high-diversity groups
        } else if texts.len() > 50 || intent_count > 12 {
            base_archetypes * 3  // 48 for medium groups or high intent diversity
        } else {
            base_archetypes      // 16 for small groups
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
            let embs: Vec<&[f32]> = pairs.iter().map(|(emb, _)| *emb).collect();
            let mut clusters: Vec<Vec<usize>> = vec![vec![]; cb.archetypes.len()];
            for (i, (_emb, text)) in pairs.iter().enumerate() {
                let ids = dict.encode(text);
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
        raw_vecs: &'a [&'a [f32]],
        adapter: GroupAdapter,
        novelty: Vec<f32>,
        seed: u64,
        dictionary: TokenDictionary,
        codebook: Option<AlgebraicCodebook>,
        hopf: Option<HopfCompositionTable>,
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
                let hopf = gen_hopf.get(&gidx).cloned();
                let nov = gen_novelty_by_group.get(&gidx).cloned().unwrap_or_default();
                let raw = gen_raw_by_group.get(&gidx).map(|v| v.as_slice()).unwrap_or(&[]);
                let adapter = group_adapters.get(&gidx).cloned()
                    .unwrap_or_else(|| GroupAdapter::new(raw_dim, bridge_dim, DEFAULT_ADAPTER_RANK));
                for r in 0..k_replicas {
                    tasks.push(GenTask {
                        gidx, kind: "gen", replica: r,
                        pairs: p.as_slice(),
                        raw_vecs: raw,
                        adapter: adapter.clone(),
                        novelty: nov.clone(),
                        seed: 100 + gidx as u64 * 1000 + r as u64,
                        dictionary: dict.clone(),
                        codebook: cb.clone(),
                        hopf: hopf.clone(),
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
            let min_code_samples = 10;
            if p.len() >= min_code_samples {
                let task_epochs = gen_epochs;
                let total_ticks = task_epochs as u64 * p.len() as u64;
                let stop_tick = (total_ticks as f64 * prune_warmup_frac) as u64;
                let dict = code_dicts.get(&gidx).unwrap().clone();
                let cb = code_codebooks.get(&gidx).cloned();
                let hopf = code_hopf.get(&gidx).cloned();
                let nov = code_novelty_by_group.get(&gidx).cloned().unwrap_or_default();
                let raw = code_raw_by_group.get(&gidx).map(|v| v.as_slice()).unwrap_or(&[]);
                let adapter = group_adapters.get(&gidx).cloned()
                    .unwrap_or_else(|| GroupAdapter::new(raw_dim, bridge_dim, DEFAULT_ADAPTER_RANK));
                for r in 0..k_replicas {
                    tasks.push(GenTask {
                        gidx, kind: "code", replica: r,
                        pairs: p.as_slice(),
                        raw_vecs: raw,
                        adapter: adapter.clone(),
                        novelty: nov.clone(),
                        seed: 200 + gidx as u64 * 1000 + r as u64,
                        dictionary: dict.clone(),
                        codebook: cb.clone(),
                        hopf: hopf.clone(),
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

    let results: Vec<(usize, &str, usize, f32, GroupGenEnv, GroupAdapter)> = std::thread::scope(|s| {
        let handles: Vec<_> = tasks.iter().map(|task| {
            s.spawn(move || {
                let mut adapter = task.adapter.clone();
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
                if let Some(hopf) = task.hopf.clone() {
                    env.set_hopf_table(hopf);
                }
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

                // Paramecium curriculum: sort by ascending novelty (hardest-first).
                // In early epochs the model has no loss signal, so the lattice's
                // pre-training novelty scores drive sample ordering instead.
                let novelty_curriculum_end = replay_start_epoch;
                let mut novelty_order: Vec<usize> = (0..n_pairs).collect();
                if !task.novelty.is_empty() && task.novelty.len() == n_pairs {
                    novelty_order.sort_by(|&a, &b| {
                        task.novelty[a].partial_cmp(&task.novelty[b])
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                }

                // Convergence monitor state
                let mut loss_history: Vec<f32> = Vec::new();
                let mut eval_loss_history: Vec<f32> = Vec::new();
                let eval_check_interval = task.es_window.max(50);
                let mut best_eval_loss = f32::INFINITY;
                let mut stopped_early = false;
                let has_raw = task.raw_vecs.len() == task.pairs.len();

                for epoch in 0..te {
                    if epoch < novelty_curriculum_end && !task.novelty.is_empty() && task.novelty.len() == n_pairs {
                        // Curriculum phase: present novel (hard) samples first, then familiar.
                        // Add jitter so it's not identical every epoch.
                        let jitter = (epoch as f32 * 0.1).sin().abs();
                        let split = ((n_pairs as f32) * jitter * 0.3) as usize;
                        indices[..split].copy_from_slice(&novelty_order[..split.min(n_pairs)]);
                        indices[split..].shuffle(&mut task_rng);
                    } else {
                        indices.shuffle(&mut task_rng);
                    }
                    let adapter_lr = 0.001f32;
                    let adapter_warmup = 20;
                    let mut total_loss = 0.0f32;
                    for &i in &indices {
                        let (cond, h_raw) = if has_raw {
                            (adapter.adapt(task.pairs[i].0, task.raw_vecs[i]), Some(task.raw_vecs[i]))
                        } else {
                            (task.pairs[i].0.to_vec(), None)
                        };
                        let l = env.train_step(&cond, task.pairs[i].1, &mut task_rng);
                        sample_losses[i] = l;
                        total_loss += l;

                        // Train adapter via finite-difference gradient after warmup
                        if let Some(raw) = h_raw {
                            if epoch >= adapter_warmup && !adapter.frozen {
                                let base_loss = l;
                                let delta = adapter.forward(raw);
                                let eps = 0.01f32;
                                let mut grad = vec![0.0f32; delta.len()];
                                for d in 0..delta.len() {
                                    let mut perturbed = cond.clone();
                                    perturbed[d] += eps;
                                    let l_plus = env.eval_loss(&perturbed, task.pairs[i].1);
                                    grad[d] = (l_plus - base_loss) / eps;
                                }
                                adapter.train_step(raw, &grad, adapter_lr);
                            }
                        }
                    }
                    if epoch >= replay_start_epoch && replay_count > 0 {
                        let mut ranked: Vec<(usize, f32)> = sample_losses.iter().copied().enumerate()
                            .map(|(idx, loss)| {
                                let nov_weight = if idx < task.novelty.len() {
                                    1.0 - task.novelty[idx]
                                } else {
                                    0.0
                                };
                                (idx, loss + nov_weight * 0.1)
                            })
                            .collect();
                        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                        for &(idx, _score) in ranked.iter().take(replay_count) {
                            let cond = if has_raw {
                                adapter.adapt(task.pairs[idx].0, task.raw_vecs[idx])
                            } else {
                                task.pairs[idx].0.to_vec()
                            };
                            env.train_step(&cond, task.pairs[idx].1, &mut task_rng);
                        }
                    }
                    let last_avg = total_loss / n_pairs.max(1) as f32;
                    loss_history.push(last_avg);

                    // Early stopping: two conditions can trigger a stop.
                    //
                    // 1. Soft stop: loss plateaued AND below an adaptive floor.
                    //    Floor scales with output dimension — a 352-bit prediction task
                    //    can't reach the same absolute loss as a 140-bit task.
                    //    Floor = 0.02 * (output_dim / 140), clamped to [0.02, 0.15].
                    //
                    // 2. Hard stop: loss hasn't improved for 2× the early-stop window,
                    //    regardless of absolute value. The task is stuck.
                    let es_loss_floor = (0.02 * (env.output_dim as f32 / 140.0)).clamp(0.02, 0.15);
                    let hard_plateau_window = task.es_window * 2;
                    if task.es_window > 0 && epoch >= task.es_min_ep && loss_history.len() >= task.es_window {
                        let recent = &loss_history[loss_history.len() - task.es_window..];
                        let old_avg = recent[..task.es_window / 2].iter().sum::<f32>() / (task.es_window / 2) as f32;
                        let new_avg = recent[task.es_window / 2..].iter().sum::<f32>() / (task.es_window - task.es_window / 2) as f32;
                        let improvement = (old_avg - new_avg) / old_avg.max(1e-8);
                        if improvement < task.es_min_imp && new_avg < es_loss_floor {
                            println!("    [{} g{} r{}] early stop at epoch {}/{}: loss={:.4} improvement={:.5} < {:.4} (floor={:.3})",
                                task.kind, task.gidx, task.replica, epoch, te, last_avg, improvement, task.es_min_imp, es_loss_floor);
                            stopped_early = true;
                            break;
                        }
                        if improvement < task.es_min_imp && loss_history.len() >= hard_plateau_window {
                            let far_back = &loss_history[loss_history.len() - hard_plateau_window..];
                            let far_old = far_back[..hard_plateau_window / 2].iter().sum::<f32>() / (hard_plateau_window / 2) as f32;
                            let far_new = far_back[hard_plateau_window / 2..].iter().sum::<f32>() / (hard_plateau_window - hard_plateau_window / 2) as f32;
                            let long_improvement = (far_old - far_new) / far_old.max(1e-8);
                            if long_improvement < task.es_min_imp {
                                println!("    [{} g{} r{}] hard plateau stop at epoch {}/{}: loss={:.4} no improvement over {} epochs (floor={:.3})",
                                    task.kind, task.gidx, task.replica, epoch, te, last_avg, hard_plateau_window, es_loss_floor);
                                stopped_early = true;
                                break;
                            }
                        }
                        if improvement < task.es_min_imp && new_avg >= es_loss_floor {
                            println!("    [{} g{} r{}] epoch {}/{}: plateau (imp={:.5}) but loss={:.4} > floor {:.3}, continuing...",
                                task.kind, task.gidx, task.replica, epoch, te, improvement, new_avg, es_loss_floor);
                        }
                    }

                    // Eval-loss divergence check: detect overfitting by comparing
                    // train loss trend against periodic eval loss snapshots.
                    if task.es_window > 0 && epoch >= task.es_min_ep
                        && epoch % eval_check_interval == 0 && epoch > 0
                    {
                        let mut eval_sum = 0.0f32;
                        for (j, &(bridged, target)) in task.pairs.iter().enumerate() {
                            let cond = if has_raw {
                                adapter.adapt(bridged, task.raw_vecs[j])
                            } else {
                                bridged.to_vec()
                            };
                            eval_sum += env.eval_loss(&cond, target);
                        }
                        let eval_avg = eval_sum / n_pairs.max(1) as f32;
                        eval_loss_history.push(eval_avg);
                        if eval_avg < best_eval_loss {
                            best_eval_loss = eval_avg;
                        }
                        if eval_loss_history.len() >= 3 {
                            let recent_eval = eval_loss_history[eval_loss_history.len() - 1];
                            let prev_eval = eval_loss_history[eval_loss_history.len() - 2];
                            let older_eval = eval_loss_history[eval_loss_history.len() - 3];
                            let eval_rising = recent_eval > prev_eval && prev_eval > older_eval;
                            let train_falling = last_avg < loss_history[loss_history.len().saturating_sub(eval_check_interval + 1)];
                            if eval_rising && train_falling && recent_eval > best_eval_loss * 1.3 {
                                println!("    [{} g{} r{}] overfit stop at epoch {}/{}: train={:.4} eval={:.4} (best_eval={:.4}, rising for {} checks)",
                                    task.kind, task.gidx, task.replica, epoch, te, last_avg, recent_eval, best_eval_loss, eval_loss_history.len());
                                stopped_early = true;
                                break;
                            }
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
                for (j, &(bridged, target)) in task.pairs.iter().enumerate() {
                    let cond = if has_raw {
                        adapter.adapt(bridged, task.raw_vecs[j])
                    } else {
                        bridged.to_vec()
                    };
                    eval_loss += env.eval_loss(&cond, target);
                }
                let final_loss = eval_loss / task.pairs.len().max(1) as f32;
                env.freeze();
                adapter.freeze();
                println!("    [{} g{} r{}] frozen: {} neurons, {} synapses, eval_loss={:.4}, adapter_params={}",
                    task.kind, task.gidx, task.replica,
                    env.total_neurons(), env.total_synapses(), final_loss, adapter.param_count());
                (task.gidx, task.kind, task.replica, final_loss, env, adapter)
            })
        }).collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    // Select best replica per (group, kind) by lowest eval loss
    let mut best: HashMap<(usize, &str), (f32, GroupGenEnv, GroupAdapter)> = HashMap::new();
    for (gidx, kind, replica, loss, env, adapter) in results {
        let key = (gidx, kind);
        let is_better = match best.get(&key) {
            Some((prev_loss, _, _)) => loss < *prev_loss,
            None => true,
        };
        if is_better {
            if k_replicas > 1 {
                println!("    [{} g{}] best replica so far: r{} loss={:.4}", kind, gidx, replica, loss);
            }
            best.insert(key, (loss, env, adapter));
        }
    }

    for ((gidx, kind), (_loss, env, adapter)) in best {
        match kind {
            "gen" => { svc.dm.group_gen_envs.insert(gidx, env); }
            "code" => { svc.dm.group_code_envs.insert(gidx, env); }
            _ => {}
        }
        svc.dm.group_adapters.insert(gidx, adapter);
    }

    // ---------------------------------------------------------------
    // Build final paramecium from trained codebooks.
    // This captures the learned archetype structure so the lattice is
    // available for curriculum-guided continuum learning at runtime.
    // ---------------------------------------------------------------
    println!("\n--- Building Post-Training Paramecium ---");
    svc.build_paramecium();
    if let Some(ref pm) = svc.paramecium {
        println!("  programs: {}, memory: {} bytes", pm.program_count(), pm.memory_bytes());
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
            println!("  gen [{}] (conf={:.2}): {:?}", resp.template_id, resp.confidence, &r[..r.len().min(200)]);
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

