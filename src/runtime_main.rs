//! Standalone inference binary — loads a brain and runs prompts or a REPL.
//!
//! Minimal dependencies: no training pipeline, no rayon, no indicatif.
//!
//! Usage:
//!   growformer-runtime brain.bin                     # interactive REPL
//!   growformer-runtime brain.bin "your prompt"       # single-shot
//!   growformer-runtime brain.bin --json "prompt"     # JSON output

use growformer::runtime::Runtime;
use growformer::service::OceanProfile;
use growformer::tools_builtin::execute_tool;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: growformer-runtime <brain.bin> [prompt | --json prompt]");
        std::process::exit(1);
    }

    let brain_path = &args[1];
    let data = std::fs::read(brain_path).unwrap_or_else(|e| {
        eprintln!("Failed to read {}: {}", brain_path, e);
        std::process::exit(1);
    });

    let mut rt = Runtime::from_brain_bytes(&data).unwrap_or_else(|e| {
        eprintln!("Failed to load brain: {}", e);
        std::process::exit(1);
    });

    let info = rt.brain_info();
    eprintln!("Brain loaded: {}", brain_path);
    eprintln!(
        "  Agent: {} (by {}), {} groups, {} gen envs, {} code envs",
        info.agent_name,
        info.agent_creator,
        info.num_groups,
        info.gen_envs,
        info.code_envs
    );

    let json_mode = args.iter().any(|a| a == "--json");
    let prompt: Option<String> = if json_mode {
        args.iter()
            .position(|a| a == "--json")
            .and_then(|i| args.get(i + 1))
            .cloned()
    } else {
        args.get(2).cloned()
    };

    if let Some(prompt_text) = prompt {
        run_single(&mut rt, &prompt_text, json_mode);
    } else {
        run_repl(&mut rt);
    }
}

fn run_single(rt: &mut Runtime, prompt: &str, json_mode: bool) {
    match rt.prompt(prompt) {
        Ok(resp) => {
            if json_mode {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&resp).unwrap_or_default()
                );
            } else {
                if !resp.text.is_empty() {
                    println!("{}", resp.text);
                }
            }
        }
        Err(e) => {
            eprintln!("error: {}", e);
            std::process::exit(1);
        }
    }

    match rt.codegen(prompt) {
        Ok(Some(code)) if !code.code.is_empty() => {
            if json_mode {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&code).unwrap_or_default()
                );
            } else {
                println!("[{}] {}", code.kind, code.code);
            }
        }
        _ => {}
    }
}

fn run_repl(rt: &mut Runtime) {
    let ocean = rt.personality().as_vec();
    println!("\n=== Growformer Runtime REPL ===");
    println!(
        "  Agent: {} (by {})",
        rt.brain_info().agent_name,
        rt.brain_info().agent_creator
    );
    println!(
        "  Personality [O={:.1} C={:.1} E={:.1} A={:.1} N={:.1}]",
        ocean[0], ocean[1], ocean[2], ocean[3], ocean[4]
    );
    println!();
    println!("Commands:");
    println!("  /personality <preset>   Switch: assistant, creative, engineer, analyst");
    println!("  /ocean O C E A N        Set custom OCEAN values (0.0-1.0)");
    println!("  /reset                  Clear conversation history");
    println!("  /single <prompt>        Single-shot (no conversation context)");
    println!("  /paramecium <prompt>    Lattice-only inference");
    println!("  /json <prompt>          Single-shot with JSON output");
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
                if trimmed.is_empty() {
                    continue;
                }
                if trimmed == "quit" || trimmed == "exit" {
                    break;
                }

                if let Some(cmd) = trimmed.strip_prefix('/') {
                    handle_command(rt, cmd);
                    continue;
                }

                if let Some(tool_call) = rt.try_tool_call(trimmed) {
                    let result = execute_tool(&tool_call);
                    let status = if result.success { "ok" } else { "error" };
                    println!("  [tool: {} ({})]", tool_call.tool_name, status);
                    if !result.output.is_empty() {
                        for l in result.output.lines().take(20) {
                            println!("  | {}", l);
                        }
                    }
                    match rt.respond_with_tool_result(trimmed, &result) {
                        Ok(resp) if !resp.text.is_empty() && !resp.text.starts_with("[tool_call:") => {
                            println!("\n  {}", resp.text);
                        }
                        _ => {}
                    }
                } else {
                    match rt.converse(trimmed) {
                        Ok(resp) => {
                            if !resp.text.is_empty() {
                                println!(
                                    "\n  {} (conf={:.2})",
                                    resp.text, resp.confidence
                                );
                            }
                            if let Ok(Some(code)) = rt.codegen(trimmed) {
                                if !code.code.is_empty() {
                                    println!("  [{}] {}", code.kind, code.code);
                                }
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

fn handle_command(rt: &mut Runtime, cmd: &str) {
    let parts: Vec<&str> = cmd.split_whitespace().collect();
    match parts.first().copied() {
        Some("personality") | Some("p") => {
            let profile = match parts.get(1).copied() {
                Some("assistant") => {
                    println!("  Personality: assistant");
                    Some(OceanProfile::assistant())
                }
                Some("creative") => {
                    println!("  Personality: creative");
                    Some(OceanProfile::creative())
                }
                Some("engineer") => {
                    println!("  Personality: engineer");
                    Some(OceanProfile::engineer())
                }
                Some("analyst") => {
                    println!("  Personality: analyst");
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
            println!(
                "  [O={:.1} C={:.1} E={:.1} A={:.1} N={:.1}]",
                v[0], v[1], v[2], v[3], v[4]
            );
        }
        Some("ocean") => {
            if parts.len() == 6 {
                let vals: Vec<f32> = parts[1..6]
                    .iter()
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
                    println!(
                        "  Custom OCEAN: [O={:.1} C={:.1} E={:.1} A={:.1} N={:.1}]",
                        v[0], v[1], v[2], v[3], v[4]
                    );
                } else {
                    println!("  Usage: /ocean 0.5 0.7 0.5 0.6 0.3");
                }
            } else {
                let v = rt.personality().as_vec();
                println!(
                    "  Current: [O={:.1} C={:.1} E={:.1} A={:.1} N={:.1}]",
                    v[0], v[1], v[2], v[3], v[4]
                );
            }
        }
        Some("reset") => {
            rt.reset_conversation();
            println!("  Conversation cleared.");
        }
        Some("single") | Some("s") => {
            let prompt = parts[1..].join(" ");
            if prompt.is_empty() {
                println!("  Usage: /single <prompt>");
            } else {
                run_single(rt, &prompt, false);
            }
        }
        Some("json") => {
            let prompt = parts[1..].join(" ");
            if prompt.is_empty() {
                println!("  Usage: /json <prompt>");
            } else {
                run_single(rt, &prompt, true);
            }
        }
        Some("paramecium") | Some("pm") => {
            let prompt = parts[1..].join(" ");
            if prompt.is_empty() {
                println!("  Usage: /paramecium <prompt>");
            } else {
                match rt.paramecium(&prompt) {
                    Ok(resp) => {
                        if !resp.text.is_empty() {
                            println!("  {}", resp.text);
                        } else {
                            println!("  (empty — lattice may need more programs)");
                        }
                    }
                    Err(e) => eprintln!("  paramecium error: {}", e),
                }
            }
        }
        _ => {
            println!("  Unknown command. Available:");
            println!("    /personality, /ocean, /reset, /single, /json, /paramecium");
        }
    }
}
