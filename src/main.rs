use growformer::dimension::{
    LanguageSample,
    GroupGenEnv, action_target_to_type,
};
use growformer::dimension::language::DEFAULT_BRIDGE_DIM;
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
                    gen_pairs.push((cond, r.to_string(), s.semantic_intent.clone()));
                }
                Err(_) => {
                    gen_pairs.push((
                        vec![0.0; growformer::dimension::group_gen::GEN_COND_DIM],
                        r.to_string(),
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
                let t_end = truncate_to_char_boundary(&resp.text, 200);
                println!("  gen [{}] (conf={:.2}): {:?}\n",
                    resp.template_id, resp.confidence,
                    &resp.text[..t_end]);
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
        println!("    gen[{}]: {} tokens in dict, {} lattice programs, {}, {}",
            gidx, env.dictionary.len(), env.program_count(), hopf_info, cb_info);
    }
    for (gidx, env) in &dm.group_code_envs {
        println!("    code[{}]: {} tokens in dict, {} lattice programs",
            gidx, env.dictionary.len(), env.program_count());
    }

    if let Some(prompt_text) = prompt {
        run_single_prompt(&mut svc, prompt_text);
        return Ok(());
    }

    run_conversation_repl(&mut svc);

    Ok(())
}

fn run_single_prompt(svc: &mut LanguageService, prompt: &str) {
    let action_result = svc.active_dm_mut().route_text_to_action_stateless(prompt);
    let is_coding = matches!(
        &action_result,
        Ok(ref a) if matches!(a.action_type, growformer::dimension::action::ActionType::CodingAssist)
    );
    if let Ok(ref action) = action_result {
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

    if is_coding {
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

    // Discover groups dynamically from the training data's action_target values.
    let discovered_group_names = discover_group_names(&samples);
    println!("Discovered {} groups from data: {:?}", discovered_group_names.len(), discovered_group_names);
    let group_name_refs: Vec<&str> = discovered_group_names.iter().map(|s| s.as_str()).collect();
    let mut svc = LanguageService::new_with_groups(&group_name_refs)?;

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
    let group_lookup = build_group_lookup(&discovered_group_names);
    let group_map = |s: &LanguageSample| -> usize {
        action_target_to_group(s.action_target.as_deref(), &group_lookup, num_groups)
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
    println!("  Router loss={:.4} accuracy={:.1}% (Paramecium one-pass, {} programs)",
        router_loss, router_acc * 100.0,
        svc.dm.observer.learned_router.as_ref().map(|r| r.program_count()).unwrap_or(0));

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
    println!("  Classifier loss={:.4} accuracy={:.1}% (Paramecium one-pass, {} programs)",
        clf_loss, clf_acc * 100.0,
        svc.dm.action_classifier.as_ref().map(|c| c.program_count()).unwrap_or(0));

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
            gen_by_group.entry(gidx).or_default().push((bridged.as_slice(), r));
            gen_raw_by_group.entry(gidx).or_default().push(raw.as_slice());
            gen_topic_by_group.entry(gidx).or_default().push(s.semantic_intent.as_str());
            gen_novelty_by_group.entry(gidx).or_default().push(nov);
        }
        if let Some(c) = s.expected_code.as_deref() {
            if !c.is_empty() && c != "null" {
                code_by_group.entry(gidx).or_default().push((bridged.as_slice(), c));
                code_raw_by_group.entry(gidx).or_default().push(raw.as_slice());
                code_topic_by_group.entry(gidx).or_default().push(s.semantic_intent.as_str());
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

    // Stage 2.5: Understanding Layer + MetaBrain — Paramecium one-pass build.
    // All classifiers use Paramecium lattice develop(), zero iterative epochs.
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
    let base_archetypes = 32usize;
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
    println!("\n--- Stages 3+4: Indexed Generation (Paramecium lattice) ---");

    use growformer::dimension::group_gen::IndexedGenEnv;
    let spawn_threshold = 0.97;
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
            let cb = gen_codebooks.get(&gidx).cloned().unwrap_or_else(|| AlgebraicCodebook::build(
                &pairs.iter().map(|(_, t)| *t).collect::<Vec<_>>(), &dict, 32, None,
            ));
            let hopf = gen_hopf.get(&gidx).cloned().unwrap_or_default();
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
                "  gen[g{}]: {} lattice programs, {} topic sub-lattices, frozen",
                gidx, env.program_count(), env.topic_subindex.len()
            );
            svc.dm.group_gen_envs.insert(gidx, env);
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
        }
    }
    let t_index_elapsed = t_index_start.elapsed();
    println!("  Indexed {} gen + {} code groups in {:?}",
        svc.dm.group_gen_envs.len(), svc.dm.group_code_envs.len(), t_index_elapsed);

    // (Legacy thread scope with NeuralEnvironment training removed —
    // replaced by one-pass Paramecium lattice indexing above)

    // Register per-group structural fingerprints (grade-2 bivectors in Cl(8))
    // for understanding-based routing on novel/OOD inputs.
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
    println!("\n--- Building Cognitive Map (Reasoning Engine) ---");
    let cog_map = CognitiveMap::build(&svc.dm.group_gen_envs, &svc.dm.group_rotors);
    println!("  nodes: {}, edges: {} (cross-group structural links)", cog_map.node_count(), cog_map.edge_count());
    let group_dicts: HashMap<usize, TokenDictionary> = svc.dm.group_gen_envs.iter()
        .map(|(&gidx, env)| (gidx, env.dictionary.clone()))
        .collect();
    let reasoning_engine = ReasoningEngine::new(cog_map, group_dicts);
    svc.reasoning = Some(reasoning_engine);
    println!("  ReasoningEngine: active, settling_rounds=4");

    // ---------------------------------------------------------------
    // Cloze games: fill-in-the-blank learning to teach slot inference.
    // Programs learn to "own" specific slot fills through centroid drift.
    // Multiple rounds: each round refines program ownership.
    // ---------------------------------------------------------------
    {
        use growformer::cloze;
        println!("\n--- Cloze Learning (Fill-in-the-Blank) ---");
        let cloze_rounds = 3;
        let k_voters = 5;
        let reward_rate = 0.08;
        let punish_rate = 0.04;
        let mut total_stats = cloze::ClozeStats::default();
        for (gidx, env) in svc.dm.group_gen_envs.iter_mut() {
            let codebook = match env.codebook.as_ref() {
                Some(cb) if !cb.archetypes.is_empty() => cb.clone(),
                _ => continue,
            };
            // Build cloze tasks from the lattice programs (distilled training data).
            let training_pairs: Vec<(Vec<f32>, String)> = env.lattice.programs.iter()
                .map(|prog| {
                    let text = env.dictionary.decode(&prog.token_sequence);
                    (prog.ema_centroid.clone(), text)
                })
                .collect();
            let tasks = cloze::generate_cloze_tasks(&codebook, &env.dictionary, &training_pairs);
            if tasks.is_empty() { continue; }
            // Unfreeze for cloze learning, re-freeze after.
            env.frozen = false;
            let mut round_stats = cloze::ClozeStats::default();
            for _round in 0..cloze_rounds {
                let stats = cloze::play_cloze_round(env, &tasks, k_voters, reward_rate, punish_rate);
                round_stats.games_played += stats.games_played;
                round_stats.total_slots += stats.total_slots;
                round_stats.correct_slots += stats.correct_slots;
                round_stats.reward_applied += stats.reward_applied;
                round_stats.punishment_applied += stats.punishment_applied;
            }
            env.frozen = true;
            println!("  cloze[g{}]: {} rounds × {} tasks, {}", gidx, cloze_rounds, tasks.len(), round_stats);
            total_stats.games_played += round_stats.games_played;
            total_stats.total_slots += round_stats.total_slots;
            total_stats.correct_slots += round_stats.correct_slots;
            total_stats.reward_applied += round_stats.reward_applied;
            total_stats.punishment_applied += round_stats.punishment_applied;
        }
        // Also run cloze on code environments.
        for (gidx, env) in svc.dm.group_code_envs.iter_mut() {
            let codebook = match env.codebook.as_ref() {
                Some(cb) if !cb.archetypes.is_empty() => cb.clone(),
                _ => continue,
            };
            let training_pairs: Vec<(Vec<f32>, String)> = env.lattice.programs.iter()
                .map(|prog| {
                    let text = env.dictionary.decode(&prog.token_sequence);
                    (prog.ema_centroid.clone(), text)
                })
                .collect();
            let tasks = cloze::generate_cloze_tasks(&codebook, &env.dictionary, &training_pairs);
            if tasks.is_empty() { continue; }
            env.frozen = false;
            let mut round_stats = cloze::ClozeStats::default();
            for _round in 0..cloze_rounds {
                let stats = cloze::play_cloze_round(env, &tasks, k_voters, reward_rate, punish_rate);
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
        println!("    gen[{}]: {} lattice programs, frozen={}", gidx, env.program_count(), env.frozen);
    }
    for (gidx, env) in &svc.dm.group_code_envs {
        println!("    code[{}]: {} lattice programs, frozen={}", gidx, env.program_count(), env.frozen);
    }

    println!("\n--- Post-Training Inference Check ---");
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

