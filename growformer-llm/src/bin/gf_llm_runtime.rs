//! `gf-llm-runtime` — lean inference runtime for growformer-llm domain specialists.
//!
//! Two modes:
//!   * **single** — load one specialist (a `*.gfcard.json` manifest or a checkpoint
//!     JSON that embeds its tokenizer) and generate / REPL.
//!   * **fleet**  — load a directory of `*.gfcard.json` manifests and route each
//!     prompt to the specialist that owns the subject ("many brain micro-models").
//!
//! Build lean (no training deps, no growformer brain-memory):
//!   cargo build --release --bin gf-llm-runtime --no-default-features --features clifford-lm
//!
//! Examples:
//!   gf-llm-runtime model.gfcard.json --prompt "the market"
//!   gf-llm-runtime agent-data/lm.json --greedy --prompt "once upon a time"
//!   gf-llm-runtime --fleet ./specialists --prompt "quarterly earnings beat"
//!   gf-llm-runtime --fleet ./specialists            # routing REPL

use std::io::Write;
use std::path::{Path, PathBuf};

use clap::Parser;

use growformer_llm::fleet::{Specialist, SpecialistFleet};
use growformer_llm::model_card::{SpecialistManifest, CARD_EXT};
use growformer_llm::v2::sample::SampleConfig;

#[derive(Parser)]
#[command(name = "gf-llm-runtime")]
#[command(about = "Lean inference runtime for growformer-llm domain specialists", long_about = None)]
struct Cli {
    /// Single-model mode: a `*.gfcard.json` manifest or a checkpoint JSON.
    model: Option<PathBuf>,

    /// Fleet mode: a directory of `*.gfcard.json` manifests.
    #[arg(long)]
    fleet: Option<PathBuf>,

    /// Force a specialist by subject (fleet mode); bypasses the router.
    #[arg(long)]
    subject: Option<String>,

    /// Prompt to generate from. If omitted, start an interactive REPL.
    #[arg(long)]
    prompt: Option<String>,

    #[arg(long, default_value_t = 64)]
    max_new_tokens: usize,

    #[arg(long, default_value_t = 0.8)]
    temperature: f32,

    /// Repetition penalty (>1 discourages loops; 1.0 = off). Applied in both
    /// sampling and greedy modes.
    #[arg(long, default_value_t = 1.15)]
    repetition_penalty: f32,

    /// Greedy (argmax) decoding instead of sampling.
    #[arg(long, default_value_t = false)]
    greedy: bool,

    #[arg(long)]
    seed: Option<u64>,
}

fn sample_cfg(cli: &Cli) -> SampleConfig {
    if cli.greedy {
        SampleConfig {
            max_new_tokens: cli.max_new_tokens,
            repetition_penalty: cli.repetition_penalty,
            seed: cli.seed,
            ..SampleConfig::greedy()
        }
    } else {
        SampleConfig {
            temperature: cli.temperature,
            max_new_tokens: cli.max_new_tokens,
            repetition_penalty: cli.repetition_penalty,
            seed: cli.seed,
            ..SampleConfig::focused()
        }
    }
}

fn is_manifest(path: &Path) -> bool {
    path.file_name()
        .and_then(|n| n.to_str())
        .map(|n| n.ends_with(&format!(".{}", CARD_EXT)))
        .unwrap_or(false)
}

fn load_single(path: &Path) -> Result<Specialist, String> {
    if is_manifest(path) {
        let manifest = SpecialistManifest::load(path)?;
        let dir = path.parent().unwrap_or_else(|| Path::new("."));
        Specialist::from_manifest(dir, &manifest)
    } else {
        Specialist::load_checkpoint("model", path)
    }
}

/// Minimal REPL: print the prompt marker, read a line, hand the trimmed text to
/// `on_line`. Stops on EOF, an empty line, or `quit`/`exit`.
fn repl(mut on_line: impl FnMut(&str)) {
    let stdin = std::io::stdin();
    let mut buf = String::new();
    loop {
        eprint!("> ");
        let _ = std::io::stderr().flush();
        buf.clear();
        match stdin.read_line(&mut buf) {
            Ok(0) | Err(_) => break,
            Ok(_) => {}
        }
        let line = buf.trim();
        if line.is_empty() || line == "quit" || line == "exit" {
            break;
        }
        on_line(line);
    }
}

fn run_single(cli: &Cli, model: &Path) -> Result<(), String> {
    let spec = load_single(model)?;
    let cfg = sample_cfg(cli);
    eprintln!("[gf-llm-runtime] loaded specialist '{}'", spec.subject());

    if let Some(prompt) = &cli.prompt {
        println!("{}", spec.generate(prompt, &cfg));
        return Ok(());
    }

    eprintln!("Enter a prompt (blank line or 'quit' to exit):");
    repl(|line| println!("{}", spec.generate(line, &cfg)));
    Ok(())
}

fn run_fleet(cli: &Cli, dir: &Path) -> Result<(), String> {
    let fleet = SpecialistFleet::load_dir(dir)?;
    if fleet.is_empty() {
        return Err(format!(
            "no specialists found in {} (expected *.{} manifests)",
            dir.display(),
            CARD_EXT
        ));
    }
    eprintln!(
        "[gf-llm-runtime] fleet loaded: {} specialists {:?}",
        fleet.len(),
        fleet.subjects()
    );
    let cfg = sample_cfg(cli);

    let answer = |prompt: &str| -> Result<(String, String), String> {
        if let Some(subject) = &cli.subject {
            let text = fleet
                .generate_with(subject, prompt, &cfg)
                .ok_or_else(|| format!("no specialist '{subject}' in fleet"))?;
            Ok((subject.clone(), text))
        } else {
            fleet
                .generate(prompt, &cfg)
                .ok_or_else(|| "no specialist matched the prompt; try --subject".to_string())
        }
    };

    if let Some(prompt) = &cli.prompt {
        let (subject, text) = answer(prompt)?;
        println!("[{subject}] {text}");
        return Ok(());
    }

    eprintln!("Routing REPL — enter a prompt (blank line or 'quit' to exit):");
    repl(|line| match answer(line) {
        Ok((subject, text)) => println!("[{subject}] {text}"),
        Err(e) => eprintln!("  {e}"),
    });
    Ok(())
}

fn main() {
    let cli = Cli::parse();

    let result = match (&cli.model, &cli.fleet) {
        (Some(_), Some(_)) => Err("pass either a single MODEL or --fleet, not both".to_string()),
        (Some(model), None) => run_single(&cli, &model.clone()),
        (None, Some(dir)) => run_fleet(&cli, &dir.clone()),
        (None, None) => {
            Err("provide a MODEL (manifest or checkpoint) or --fleet <dir>".to_string())
        }
    };

    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
