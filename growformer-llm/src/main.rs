//! CLI: pooled classifier (`train` / `infer`) and v2 causal LM (`train-lm` / `generate`).

use std::io::Write;
use std::path::PathBuf;

use clap::{Parser, Subcommand};

use growformer_llm::train::{infer_one, load_infer_pack, train_classifier, TrainConfig};
use growformer_llm::v2::checkpoint::{load_lm_checkpoint, save_lm_checkpoint};
use growformer_llm::v2::data::{special, Dataset, Tokenizer};
use growformer_llm::v2::inference::InferenceCache;
use growformer_llm::v2::sample::{generate_stream, SampleConfig, TokenCallback};
use growformer_llm::v2::train_v2::{train_v2, ModelStateV2, TrainConfigV2};
use growformer_llm::world_grounding::GROUND_FEATURE_DIM;

#[derive(Parser)]
#[command(name = "growformer-llm")]
#[command(
    about = "Clifford LLM — classifier train/infer; v2 LM train-lm/generate",
    long_about = None
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Train mean-pooled byte classifier on all training `*.jsonl` in `--data-dir`.
    Train {
        #[arg(long)]
        data_dir: PathBuf,
        #[arg(long, default_value = "domain")]
        task: String,
        #[arg(long, default_value_t = 12)]
        epochs: usize,
        #[arg(long, default_value_t = 256)]
        max_seq_len: usize,
        #[arg(long, default_value_t = 16)]
        d_model: usize,
        #[arg(long)]
        checkpoint: PathBuf,
        #[arg(long)]
        grounding_toml: Vec<PathBuf>,
        #[arg(long, default_value_t = 3e-4f32)]
        lr: f32,
        #[arg(long, default_value_t = false)]
        no_grounding: bool,
    },
    /// Classification inference (JSON checkpoint from `train`).
    Infer {
        #[arg(long)]
        checkpoint: PathBuf,
        #[arg(long)]
        prompt: String,
        #[arg(long, default_value_t = 256)]
        max_seq_len: usize,
        #[arg(long)]
        grounding_toml: Vec<PathBuf>,
        #[arg(long, default_value_t = false)]
        no_grounding: bool,
    },
    /// v2: train causal LM on JSONL (`text`, `expected_response`, `split`) — full-graph backward.
    TrainLm {
        #[arg(long)]
        jsonl: PathBuf,
        #[arg(long)]
        checkpoint_out: PathBuf,
        #[arg(long, default_value_t = 128)]
        max_seq: usize,
        #[arg(long, default_value_t = 10)]
        epochs: usize,
        #[arg(long, default_value_t = 8)]
        d_model: usize,
        #[arg(long, default_value_t = 2)]
        n_heads: usize,
        #[arg(long, default_value_t = 32)]
        d_ff: usize,
        #[arg(long, default_value_t = 2)]
        n_blocks: usize,
        #[arg(long, default_value_t = 3e-4f32)]
        lr_max: f32,
        #[arg(long, default_value_t = false)]
        freeze_embeddings: bool,
    },
    /// v2: generate continuation after `BOS + prompt + SEP` using LM checkpoint.
    Generate {
        #[arg(long)]
        checkpoint: PathBuf,
        #[arg(long)]
        prompt: String,
        #[arg(long, default_value_t = 64)]
        max_new_tokens: usize,
        #[arg(long, default_value_t = 0.8)]
        temperature: f32,
        #[arg(long, default_value_t = false)]
        greedy: bool,
        #[arg(long)]
        seed: Option<u64>,
    },
}

fn default_grounding_paths() -> Vec<PathBuf> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    vec![
        root.join("data/inference/world_grounding.toml"),
        root.join("data/fintech/world_grounding_fintech.toml"),
    ]
}

/// Prints each decoded token (avoids HRTB issues with `FnMut` closures and `&str` callbacks).
struct StreamPrint;

impl TokenCallback for StreamPrint {
    fn on_token(&mut self, _token_id: usize, piece: &str) -> bool {
        print!("{} ", piece);
        let _ = std::io::stdout().flush();
        true
    }
}

fn resolve_grounding(user: &[PathBuf], no_grounding: bool) -> Vec<PathBuf> {
    if no_grounding {
        return Vec::new();
    }
    if user.is_empty() {
        default_grounding_paths()
    } else {
        user.to_vec()
    }
}

fn main() -> Result<(), String> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Train {
            data_dir,
            task,
            epochs,
            max_seq_len,
            d_model,
            checkpoint,
            grounding_toml,
            lr,
            no_grounding,
        } => {
            let grounding_paths = resolve_grounding(&grounding_toml, no_grounding);
            if grounding_paths.is_empty() {
                eprintln!(
                    "training without grounding TOML (zero features; ground_dim={})",
                    GROUND_FEATURE_DIM
                );
            }
            train_classifier(TrainConfig {
                data_dir,
                task_name: task,
                epochs,
                max_seq_len,
                d_model,
                checkpoint_path: checkpoint,
                grounding_paths,
                lr,
            })?;
        }
        Commands::Infer {
            checkpoint,
            prompt,
            max_seq_len,
            grounding_toml,
            no_grounding,
        } => {
            let paths = resolve_grounding(&grounding_toml, no_grounding);
            let pack = load_infer_pack(&checkpoint, &paths)?;
            let label = infer_one(&pack, &prompt, max_seq_len)?;
            println!("{label}");
        }
        Commands::TrainLm {
            jsonl,
            checkpoint_out,
            max_seq,
            epochs,
            d_model,
            n_heads,
            d_ff,
            n_blocks,
            lr_max,
            freeze_embeddings,
        } => {
            if d_model % n_heads != 0 {
                return Err(format!(
                    "d_model ({d_model}) must be divisible by n_heads ({n_heads})"
                ));
            }
            let mut tokenizer = Tokenizer::new();
            let dataset = Dataset::load_jsonl(&jsonl, &mut tokenizer, max_seq)?;
            if dataset.train.is_empty() {
                return Err(
                    "no training examples (need split=train or empty split and rows with expected_response)"
                        .into(),
                );
            }

            let mut cfg = TrainConfigV2::small(tokenizer.vocab_size());
            cfg.max_seq = max_seq;
            cfg.epochs = epochs;
            cfg.d_model = d_model;
            cfg.n_heads = n_heads;
            cfg.d_ff = d_ff;
            cfg.n_blocks = n_blocks;
            cfg.lr_max = lr_max;
            cfg.train_embeddings = !freeze_embeddings;
            cfg.total_steps = (cfg.epochs as u64)
                .saturating_mul(dataset.train.len() as u64)
                .max(cfg.warmup_steps + 1);

            let mut state = ModelStateV2::new(cfg);
            eprintln!(
                "[train-lm] vocab={} train={} d_model={} blocks={}",
                tokenizer.vocab_size(),
                dataset.train.len(),
                state.cfg.d_model,
                state.cfg.n_blocks
            );
            train_v2(&dataset, &mut state);
            save_lm_checkpoint(&checkpoint_out, &state, &tokenizer)?;
            eprintln!("[train-lm] wrote {}", checkpoint_out.display());
        }
        Commands::Generate {
            checkpoint,
            prompt,
            max_new_tokens,
            temperature,
            greedy,
            seed,
        } => {
            let (state, tokenizer) = load_lm_checkpoint(&checkpoint)?;
            let mut ids = vec![special::BOS];
            ids.extend(tokenizer.encode_words(&prompt));
            ids.push(special::SEP);

            let sample_cfg = if greedy {
                SampleConfig {
                    max_new_tokens,
                    seed,
                    ..SampleConfig::greedy()
                }
            } else {
                SampleConfig {
                    temperature,
                    max_new_tokens,
                    seed,
                    ..SampleConfig::focused()
                }
            };

            let mut cache = InferenceCache::new(
                state.cfg.n_blocks,
                state.cfg.max_seq,
                state.model.embedding[0].len(),
                state.cfg.dot_attention,
            );
            let _generated = generate_stream(
                &ids,
                &sample_cfg,
                &tokenizer,
                |tok_ids| cache.forward_extend(&state.alg, &state.model, tok_ids),
                StreamPrint,
            );
            println!();
        }
    }
    Ok(())
}
