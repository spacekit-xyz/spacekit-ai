//! TinyStories pipeline: BPE (`tokenize`), packed corpus (`encode`), v2 LM (`train`), sampling (`generate`).
//!
//! Checkpoints from `train` omit the word tokenizer JSON list — keep the `.tok` next to the `.json`
//! and pass both to `generate`.

use std::io::Write;
use std::path::{Path, PathBuf};

use clap::{Parser, Subcommand};

use growformer_llm::backprop::cross_entropy;
use growformer_llm::bpe::BpeTokenizer;
use growformer_llm::tinystories::{chunk_to_example, encode_corpus, load_tinystories_txt, PackedDataset};
use growformer_llm::param_budget::{log_param_match, matched_vanilla_d_model};
use growformer_llm::v2::checkpoint::{load_lm_state, save_lm_state};
use growformer_llm::v2::data::{special, TrainExample, N_SPECIAL};
use growformer_llm::v2::sample::{sample_next, softmax as logits_softmax, SampleConfig, SimpleRng};
use growformer_llm::v2::inference::InferenceCache;
use growformer_llm::v2::tape::model_forward_logits;
use growformer_llm::v2::train_v2::{
    corpus_semantic_init, train_step_v2, train_step_v2_accum, train_step_v2_head_only, ModelStateV2,
    TrainConfigV2,
};
use growformer_llm::v2::vanilla_checkpoint::{load_vanilla_state, save_vanilla_state};
use growformer_llm::v2::vanilla_train::{
    corpus_semantic_init_vanilla, eval_vanilla_lm_loss, train_step_vanilla_accum,
    VanillaModelState,
};
use growformer_llm::cl1::{append_cl1_ledger, load_frozen_specialist, load_heldout_tokens, run_cl1};
use growformer_llm::vanilla_llm::vanilla_forward_logits;

#[cfg(feature = "brain-memory")]
use growformer_llm::brain_infer_config::{battery_cases, scored_battery_cases, BrainInferConfig};
#[cfg(feature = "brain-memory")]
use growformer_llm::brain_memory::{
    brain_router_features, format_lm_memory_prefix, raw_lattice_report_json, BrainMemoryRuntime,
};

use growformer_ledger::results_ledger as ledger;

use spacekit_compressor::binary::{BinaryAlgorithm, BinaryCompressor};

#[derive(Parser)]
#[command(name = "tinystories")]
#[command(about = "TinyStories BPE + packed bin + v2 LM train/generate", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Train BPE on a TinyStories `.txt` and write `*.tok`.
    Tokenize {
        txt: PathBuf,
        #[arg(default_value_t = 2048)]
        target_vocab: u32,
        out_tok: PathBuf,
    },
    /// Pack BPE token ids to `*.bin` (CLIFTOKS).
    Encode {
        txt: PathBuf,
        tok: PathBuf,
        out_bin: PathBuf,
    },
    /// Random-chunk v2 training on packed bins (CPU-oriented defaults: d_model=16, n_blocks=4).
    Train {
        tok: PathBuf,
        train_bin: PathBuf,
        val_bin: PathBuf,
        #[arg(long, default_value = "agent-data/tinystories-lm.json")]
        checkpoint_out: PathBuf,
        #[arg(long, default_value_t = 128)]
        seq_len: usize,
        #[arg(long, default_value_t = 8000)]
        steps: u64,
        #[arg(long, default_value_t = 16)]
        d_model: usize,
        #[arg(long, default_value_t = 4)]
        n_heads: usize,
        #[arg(long, default_value_t = 64)]
        d_ff: usize,
        #[arg(long, default_value_t = 4)]
        n_blocks: usize,
        #[arg(long, default_value_t = 3e-4f32)]
        lr_max: f32,
        #[arg(long, default_value_t = 500)]
        sample_every: u64,
        #[arg(long, default_value_t = 32)]
        val_chunks: usize,
        /// Only train the output linear head (full forward tape; no block/embedding backward). Much faster; blocks stay frozen.
        #[arg(long, default_value_t = false)]
        head_only: bool,
        /// Inherit from a base checkpoint: load its weights + architecture, then fine-tune.
        /// Architecture flags (d_model/n_heads/d_ff/n_blocks) are taken from the base.
        #[arg(long)]
        init_from: Option<PathBuf>,
        /// Freeze the first N transformer blocks (shared base body); only the rest adapt.
        #[arg(long, default_value_t = 0)]
        freeze_blocks: usize,
        /// Freeze the embedding table (shared base representation).
        #[arg(long, default_value_t = false)]
        freeze_embeddings: bool,
        /// Weight tying: share the embedding table with the output head (recommended for small models).
        #[arg(long, default_value_t = false)]
        tie_embeddings: bool,
        /// Structured embedding init (deterministic unit-norm Gaussian per token, ported from growformer).
        #[arg(long, default_value_t = false)]
        structured_init: bool,
        /// Corpus-semantic embedding init: seed embeddings with random-indexing co-occurrence
        /// vectors from the training corpus (distributional prior). On by default for fresh
        /// training; overrides --structured-init. Use --no-semantic-init to disable.
        #[arg(long, default_value_t = false)]
        semantic_init: bool,
        /// Disable the default corpus-semantic embedding init (fall back to random/structured).
        #[arg(long, default_value_t = false)]
        no_semantic_init: bool,
        /// ±window for corpus-semantic co-occurrence accumulation.
        #[arg(long, default_value_t = 4)]
        semantic_window: usize,
        /// Gradient accumulation: average gradients over N microbatches per optimiser step (effective batch size).
        #[arg(long, default_value_t = 1)]
        grad_accum: usize,
        /// FFN-only ablation: param-matched dense real FFN (row 3) instead of Clifford geometric product.
        #[arg(long, default_value_t = false)]
        dense_ffn: bool,
        /// Attention score ablation (row 3b): dot product on Q/K multivectors instead of ⟨Q⊛K⟩₀.
        #[arg(long, default_value_t = false)]
        dot_attention: bool,
        /// Weight/init RNG seed (vary across ablation seeds).
        #[arg(long)]
        init_seed: Option<u64>,
        /// Row 2: param-matched vanilla transformer (real dot-attention baseline).
        #[arg(long, default_value_t = false)]
        vanilla: bool,
    },
    /// Split a packed bin into train + held-out shards (chronological 90/10 default).
    Split {
        src: PathBuf,
        train_out: PathBuf,
        held_out: PathBuf,
        #[arg(long, default_value_t = 0.9)]
        train_frac: f64,
    },
    /// Token-frequency baselines on a held-out shard (uniform + train-count unigram).
    Baselines {
        #[arg(long)]
        train_bin: PathBuf,
        eval_bin: PathBuf,
        #[arg(long)]
        tokenizer: PathBuf,
    },
    /// Prediction⇄compression: model bits/byte (= NLL/ln2 per byte) vs baselines.
    Eval {
        #[arg(long)]
        checkpoint: PathBuf,
        #[arg(long)]
        tokenizer: PathBuf,
        val_bin: PathBuf,
        /// Train shard for empirical unigram counts (held-out eval). Omit only for in-sample checks.
        #[arg(long)]
        train_bin: Option<PathBuf>,
        #[arg(long, default_value_t = 128)]
        seq_len: usize,
        /// Number of non-overlapping windows of `seq_len` tokens to evaluate.
        #[arg(long, default_value_t = 32)]
        windows: usize,
        /// Ledger run id (default: checkpoint file stem without extension).
        #[arg(long)]
        run_id: Option<String>,
        /// Append per-window bpt to this hash-chained ledger (held-out protocol).
        #[arg(long, default_value = "agent-data/results.jsonl")]
        ledger: PathBuf,
        /// Skip ledger append even when `--train-bin` is set.
        #[arg(long, default_value_t = false)]
        no_ledger: bool,
        /// Window selection tag for split_hash (must match across compared runs).
        #[arg(long, default_value = "first")]
        selection_tag: String,
        /// Row 2: evaluate a vanilla checkpoint (auto-detected from cfg.vanilla when omitted).
        #[arg(long, default_value_t = false)]
        vanilla: bool,
    },
    /// Verify SHA-256 chain integrity of `results.jsonl`.
    LedgerVerify {
        #[arg(long, default_value = "agent-data/results.jsonl")]
        ledger: PathBuf,
    },
    /// Render PRE_REGISTRATION §1.2 paired-SE verdict table from the ledger.
    LedgerTable {
        #[arg(long, default_value = "agent-data/results.jsonl")]
        ledger: PathBuf,
        #[arg(long, default_value = "row1b-v2")]
        baseline: String,
        /// Comma-separated candidate run ids (e.g. `row3b,row1b-v2`).
        #[arg(long, default_value = "row3b,row1b-v2")]
        candidates: String,
        #[arg(long, default_value_t = 0.05)]
        gate: f64,
    },
    /// CL-1 (Bet A): adjustable-cone routing over two frozen LM specialists.
    Cl1 {
        #[arg(long)]
        checkpoint_a: PathBuf,
        #[arg(long)]
        checkpoint_b: PathBuf,
        #[arg(long)]
        tokenizer: PathBuf,
        heldout_bin: PathBuf,
        #[arg(long, default_value_t = 128)]
        seq_len: usize,
        #[arg(long, default_value_t = 64)]
        windows: usize,
        #[arg(long, default_value_t = 30)]
        cal_windows: usize,
        #[arg(long, default_value = "cl1-row2-row3b")]
        run_id: String,
        #[arg(long, default_value = "agent-data/results.jsonl")]
        ledger: PathBuf,
        #[arg(long, default_value = "first")]
        selection_tag: String,
        #[arg(long, default_value_t = 42)]
        cone_seed: u64,
        #[arg(long, default_value_t = false)]
        no_ledger: bool,
    },
    /// Autoregressive continuation with BPE decode (`BOS` + encoded prompt).
    Generate {
        #[arg(long)]
        checkpoint: PathBuf,
        #[arg(long)]
        tokenizer: PathBuf,
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
        #[arg(long, default_value_t = 1.15)]
        repetition_penalty: f32,
    },
    /// Query a growformer brain.bin for routing/memory, then optionally continue with an LM.
    #[cfg(feature = "brain-memory")]
    BrainInfer {
        #[arg(long)]
        brain: PathBuf,
        #[arg(long)]
        prompt: String,
        /// Growformer project manifest (`*.gf.toml`) — loads inference TOML, guardrails, topic graph.
        #[arg(long, value_name = "PATH")]
        project: Option<PathBuf>,
        #[arg(long, value_name = "PATH")]
        inference_toml: Option<PathBuf>,
        #[arg(long, value_name = "PATH")]
        inference_defaults_toml: Option<PathBuf>,
        #[arg(long, value_name = "PATH")]
        guardrails_jsonl: Option<PathBuf>,
        #[arg(long, short = 'v', default_value_t = false)]
        verbose: bool,
        /// LM checkpoint for continuation (vanilla or Clifford; auto-detected).
        #[arg(long)]
        checkpoint: Option<PathBuf>,
        #[arg(long)]
        tokenizer: Option<PathBuf>,
        #[arg(long, default_value_t = 128)]
        max_new_tokens: usize,
        #[arg(long, default_value_t = 0.8)]
        temperature: f32,
        #[arg(long, default_value_t = false)]
        greedy: bool,
        #[arg(long)]
        seed: Option<u64>,
        #[arg(long, default_value_t = 1.15)]
        repetition_penalty: f32,
        /// Print brain routing + lattice memory only (no LM).
        #[arg(long, default_value_t = false)]
        brain_only: bool,
    },
    /// Pre-gate raw lattice retrieval diagnostic (no metacog, no grounding gate).
    #[cfg(feature = "brain-memory")]
    BrainRawDiag {
        #[arg(long, required_unless_present = "battery")]
        brain: Option<PathBuf>,
        #[arg(long, required_unless_present = "battery")]
        prompt: Option<String>,
        /// Growformer project manifest (`*.gf.toml`) for single-prompt mode.
        #[arg(long, value_name = "PATH")]
        project: Option<PathBuf>,
        #[arg(long, value_name = "PATH")]
        inference_toml: Option<PathBuf>,
        #[arg(long, value_name = "PATH")]
        inference_defaults_toml: Option<PathBuf>,
        #[arg(long, value_name = "PATH")]
        guardrails_jsonl: Option<PathBuf>,
        #[arg(long, short = 'v', default_value_t = false)]
        verbose: bool,
        #[arg(long, default_value_t = 5)]
        top_k: usize,
        #[arg(long, default_value_t = false)]
        json: bool,
        /// Run the 4-prompt fork-resolving battery (ignores --brain/--prompt).
        #[arg(long, default_value_t = false)]
        battery: bool,
        /// Use battery-retrained brains (includes new JSONL rows).
        #[arg(long, default_value_t = false)]
        battery_brains: bool,
        /// Include cases 1/4 (untrained neurokit sentiment-brain-v3.bin — diagnostic only).
        #[arg(long, default_value_t = false)]
        battery_all: bool,
    },
}

fn eval_lm_loss(state: &ModelStateV2, ex: &TrainExample) -> f32 {
    let logits = model_forward_logits(
        &state.alg,
        &state.model,
        &ex.full_ids,
        true,
        state.cfg.dot_attention,
    );
    let mask = ex.loss_mask();
    let mut total = 0.0f32;
    let mut n = 0usize;
    for t in 0..ex.len() {
        if !mask[t] || t + 1 >= ex.len() {
            continue;
        }
        let (loss, _) = cross_entropy(&logits[t], ex.full_ids[t + 1]);
        total += loss;
        n += 1;
    }
    if n == 0 {
        0.0
    } else {
        total / n as f32
    }
}

fn git_sha_short() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".into())
}

fn ledger_config_hash(cfg: &TrainConfigV2) -> String {
    let canonical = format!(
        "d_model={} n_heads={} d_ff={} n_blocks={} dense_ffn={} dot_attention={} tie={} vocab={} vanilla={} clifford_ref_d_model={}",
        cfg.d_model,
        cfg.n_heads,
        cfg.d_ff,
        cfg.n_blocks,
        cfg.dense_ffn,
        cfg.dot_attention,
        cfg.tie_embeddings,
        cfg.vocab_size,
        cfg.vanilla,
        cfg.clifford_ref_d_model,
    );
    ledger::compute_config_hash(&canonical)
}

fn peek_checkpoint_cfg(path: &Path) -> Result<TrainConfigV2, String> {
    let raw = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    #[derive(serde::Deserialize)]
    struct Peek {
        cfg: TrainConfigV2,
    }
    let p: Peek = serde_json::from_str(&raw).map_err(|e| e.to_string())?;
    Ok(p.cfg)
}

fn sample_prompt_vanilla(
    state: &VanillaModelState,
    bpe: &BpeTokenizer,
    cfg: &SampleConfig,
    seed: u64,
) {
    let mut ids: Vec<usize> = vec![special::BOS];
    ids.extend(bpe.encode("Once upon a time").iter().map(|&x| x as usize));
    let mut rng = SimpleRng::new(seed);
    print!("[sample] ");
    let _ = std::io::stdout().flush();
    for _ in 0..48 {
        let logits = vanilla_forward_logits(&state.model, &ids, true);
        let Some(last) = logits.last() else {
            break;
        };
        let next = sample_next(last, &ids, cfg, &mut rng);
        if cfg.stop_tokens.contains(&next) {
            break;
        }
        print!("{}", bpe.decode_one(next as u32));
        let _ = std::io::stdout().flush();
        ids.push(next);
    }
    println!();
}

fn default_run_id(checkpoint: &Path) -> String {
    checkpoint
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("eval")
        .to_string()
}

fn ensure_parent_dir(path: &Path) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(|e| {
                format!("could not create output directory {}: {e}", parent.display())
            })?;
        }
    }
    Ok(())
}

fn sample_prompt(state: &ModelStateV2, bpe: &BpeTokenizer, cfg: &SampleConfig, seed: u64) {
    let mut ids: Vec<usize> = vec![special::BOS];
    ids.extend(bpe.encode("Once upon a time").iter().map(|&x| x as usize));
    let mut rng = SimpleRng::new(seed);
    print!("[sample] ");
    let _ = std::io::stdout().flush();
    for _ in 0..48 {
        let logits = model_forward_logits(
            &state.alg,
            &state.model,
            &ids,
            true,
            state.cfg.dot_attention,
        );
        let Some(last) = logits.last() else {
            break;
        };
        let next = sample_next(last, &ids, cfg, &mut rng);
        if cfg.stop_tokens.contains(&next) {
            break;
        }
        print!("{}", bpe.decode_one(next as u32));
        let _ = std::io::stdout().flush();
        ids.push(next);
    }
    println!();
}

fn main() -> Result<(), String> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Tokenize {
            txt,
            target_vocab,
            out_tok,
        } => {
            let stories = load_tinystories_txt(&txt).map_err(|e| e.to_string())?;
            let mut tok = BpeTokenizer::new();
            tok.train(&stories, target_vocab, 2);
            ensure_parent_dir(&out_tok)?;
            tok.save(&out_tok)
                .map_err(|e| format!("write BPE tokenizer {}: {e}", out_tok.display()))?;
            eprintln!("[tokenize] wrote {} (vocab={})", out_tok.display(), tok.vocab_size());
        }
        Commands::Encode { txt, tok, out_bin } => {
            let stories = load_tinystories_txt(&txt).map_err(|e| e.to_string())?;
            if !tok.is_file() {
                return Err(format!(
                    "BPE tokenizer file not found: {}\n\
                     Hint: run tokenize first, e.g.\n\
                       cargo run --release --bin tinystories -- tokenize {} 2048 {}",
                    tok.display(),
                    txt.display(),
                    tok.display()
                ));
            }
            let tokenizer = BpeTokenizer::load(&tok).map_err(|e| {
                format!("read BPE tokenizer {}: {e}", tok.display())
            })?;
            ensure_parent_dir(&out_bin)?;
            encode_corpus(&stories, &tokenizer, &out_bin).map_err(|e| {
                format!("write packed corpus {}: {e}", out_bin.display())
            })?;
            eprintln!("[encode] wrote {}", out_bin.display());
        }
        Commands::Train {
            tok,
            train_bin,
            val_bin,
            checkpoint_out,
            seq_len,
            steps,
            d_model,
            n_heads,
            d_ff,
            n_blocks,
            lr_max,
            sample_every,
            val_chunks,
            head_only,
            init_from,
            freeze_blocks,
            freeze_embeddings,
            tie_embeddings,
            structured_init,
            semantic_init,
            no_semantic_init,
            semantic_window,
            grad_accum,
            dense_ffn,
            dot_attention,
            init_seed,
            vanilla,
        } => {
            // Corpus-semantic init is the validated default for fresh training
            // (≈37% lower val perplexity at equal steps). Opt out with
            // --no-semantic-init, or pick the random structured init explicitly.
            let do_semantic = semantic_init || (!structured_init && !no_semantic_init);
            if d_model % n_heads != 0 {
                return Err(format!(
                    "d_model ({d_model}) must be divisible by n_heads ({n_heads})"
                ));
            }
            let bpe = BpeTokenizer::load(&tok).map_err(|e| e.to_string())?;
            let vs = bpe.vocab_size() as usize;

            let train_ds = PackedDataset::load(&train_bin).map_err(|e| e.to_string())?;
            let val_ds = PackedDataset::load(&val_bin).map_err(|e| e.to_string())?;
            if train_ds.vocab_size != bpe.vocab_size() {
                return Err(format!(
                    "train bin vocab {} != tokenizer {}",
                    train_ds.vocab_size,
                    bpe.vocab_size()
                ));
            }
            if val_ds.vocab_size != bpe.vocab_size() {
                return Err(format!(
                    "val bin vocab {} != tokenizer {}",
                    val_ds.vocab_size,
                    bpe.vocab_size()
                ));
            }

            if vanilla {
                if init_from.is_some() {
                    return Err("--init-from is not supported with --vanilla (train row 2 fresh)".into());
                }
                if dense_ffn || dot_attention || head_only {
                    return Err("--vanilla row 2 does not use --dense-ffn, --dot-attention, or --head-only".into());
                }
                let clifford_ref = d_model;
                let matched_d = matched_vanilla_d_model(
                    vs, clifford_ref, d_ff, n_blocks, n_heads, tie_embeddings, 500,
                );
                log_param_match(vs, clifford_ref, matched_d, d_ff, n_blocks, tie_embeddings);

                let mut cfg = TrainConfigV2::small(vs);
                cfg.max_seq = seq_len;
                cfg.batch_size = 1;
                cfg.epochs = 1;
                cfg.d_model = matched_d;
                cfg.n_heads = n_heads;
                cfg.d_ff = d_ff;
                cfg.n_blocks = n_blocks;
                cfg.lr_max = lr_max;
                cfg.lr_min = 1e-5;
                cfg.warmup_steps = (steps / 20).max(50);
                cfg.total_steps = steps;
                cfg.log_every = 10;
                cfg.val_every = usize::MAX;
                cfg.train_embeddings = true;
                cfg.freeze_embeddings = freeze_embeddings;
                cfg.freeze_blocks = freeze_blocks;
                cfg.tie_embeddings = tie_embeddings;
                cfg.structured_init = structured_init && !do_semantic;
                cfg.grad_accum = grad_accum;
                cfg.vanilla = true;
                cfg.clifford_ref_d_model = clifford_ref;
                if let Some(s) = init_seed {
                    cfg.init_seed = s;
                }

                let mut state = VanillaModelState::new(cfg);
                if do_semantic {
                    eprintln!(
                        "[train] row 2 vanilla: corpus-semantic embedding init (window=±{semantic_window})"
                    );
                    corpus_semantic_init_vanilla(
                        &mut state.model,
                        &train_ds.tokens,
                        0x5EED ^ 0xE8E8,
                        semantic_window,
                        1.0,
                    );
                    if state.cfg.tie_embeddings {
                        state.model.sync_tied_head();
                    }
                }
                state.update_lr();

                let log_every_u64 = state.cfg.log_every as u64;
                let mut rng = SimpleRng::new(0xC0FFEE);
                eprintln!(
                    "[train] row 2 vanilla: d_model={matched_d} (matched from clifford_ref={clifford_ref}) \
                     vocab={vs} train_tokens={} val_tokens={} seq_len={seq_len} steps={steps}",
                    train_ds.n_tokens(),
                    val_ds.n_tokens()
                );
                if grad_accum > 1 {
                    eprintln!("[train] gradient accumulation: {grad_accum} microbatches/step");
                }

                let sample_cfg = SampleConfig {
                    temperature: 0.85,
                    top_p: Some(0.9),
                    repetition_penalty: 1.15,
                    max_new_tokens: 48,
                    stop_tokens: vec![special::EOS],
                    seed: Some(12345),
                    ..Default::default()
                };

                for step in 1u64..=steps {
                    let loss = if grad_accum > 1 {
                        let exs: Vec<TrainExample> = (0..grad_accum)
                            .map(|_| chunk_to_example(train_ds.random_chunk(seq_len, &mut rng)))
                            .collect();
                        train_step_vanilla_accum(&mut state, &exs)
                    } else {
                        let ex = chunk_to_example(train_ds.random_chunk(seq_len, &mut rng));
                        train_step_vanilla_accum(&mut state, &[ex])
                    };

                    if step % log_every_u64 == 0 || step == 1 {
                        let ppl = (loss.exp()).min(1e6);
                        eprintln!("[train] step={step} loss={loss:.4} ppl~={ppl:.1}");
                    }
                    if sample_every > 0 && step % sample_every == 0 {
                        sample_prompt_vanilla(&state, &bpe, &sample_cfg, step.wrapping_mul(991));
                    }
                    if step % 200 == 0 && val_chunks > 0 {
                        let mut vloss = 0.0f32;
                        for _ in 0..val_chunks {
                            let chunk = val_ds.random_chunk(seq_len, &mut rng);
                            let ex = chunk_to_example(chunk);
                            vloss += eval_vanilla_lm_loss(&state, &ex);
                        }
                        vloss /= val_chunks as f32;
                        eprintln!(
                            "[val] step={step} mean_nll={vloss:.4} ppl~={}",
                            (vloss.exp()).min(1e6)
                        );
                    }
                }

                save_vanilla_state(&checkpoint_out, &state)?;
                eprintln!("[train] wrote {}", checkpoint_out.display());
                return Ok(());
            }

            // Build the model state: either fresh, or inherited from a base
            // checkpoint (shared tokenizer + embeddings + body + head).
            let mut state = if let Some(base) = &init_from {
                let mut st = load_lm_state(base).map_err(|e| format!("init-from: {e}"))?;
                if st.cfg.vocab_size != vs {
                    return Err(format!(
                        "base vocab {} != tokenizer {} — base and expert must share the tokenizer",
                        st.cfg.vocab_size, vs
                    ));
                }
                // Inherit architecture from the base; override only training knobs.
                st.cfg.max_seq = seq_len;
                st.cfg.batch_size = 1;
                st.cfg.epochs = 1;
                st.cfg.lr_max = lr_max;
                st.cfg.lr_min = 1e-5;
                st.cfg.warmup_steps = (steps / 20).max(50);
                st.cfg.total_steps = steps;
                st.cfg.log_every = 10;
                st.cfg.val_every = usize::MAX;
                st.cfg.train_embeddings = !head_only;
                st.cfg.freeze_embeddings = freeze_embeddings;
                st.cfg.freeze_blocks = freeze_blocks;
                // Tying is an architectural property of the base; inherit it
                // unless the operator explicitly turns it on for this run.
                st.cfg.tie_embeddings = st.cfg.tie_embeddings || tie_embeddings;
                st.cfg.grad_accum = grad_accum;
                if dense_ffn != st.cfg.dense_ffn {
                    return Err(
                        "--dense-ffn must match the base checkpoint (retrain fresh for ablation row)"
                            .into(),
                    );
                }
                if dot_attention != st.cfg.dot_attention {
                    return Err(
                        "--dot-attention must match the base checkpoint (retrain fresh for ablation row)"
                            .into(),
                    );
                }
                if st.cfg.tie_embeddings {
                    st.model.sync_tied_head();
                }
                if semantic_init {
                    eprintln!("[train] note: semantic init ignored with --init-from (inherited embeddings kept)");
                }
                st.step = 0; // restart the LR schedule for fine-tuning
                eprintln!(
                    "[train] inheriting base {} (d_model={} n_heads={} d_ff={} n_blocks={})",
                    base.display(), st.cfg.d_model, st.cfg.n_heads, st.cfg.d_ff, st.cfg.n_blocks
                );
                st
            } else {
                let mut cfg = TrainConfigV2::small(vs);
                cfg.max_seq = seq_len;
                cfg.batch_size = 1;
                cfg.epochs = 1;
                cfg.d_model = d_model;
                cfg.n_heads = n_heads;
                cfg.d_ff = d_ff;
                cfg.n_blocks = n_blocks;
                cfg.lr_max = lr_max;
                cfg.lr_min = 1e-5;
                cfg.warmup_steps = (steps / 20).max(50);
                cfg.total_steps = steps;
                cfg.log_every = 10;
                cfg.val_every = usize::MAX;
                cfg.train_embeddings = !head_only;
                cfg.freeze_embeddings = freeze_embeddings;
                cfg.freeze_blocks = freeze_blocks;
                cfg.tie_embeddings = tie_embeddings;
                // Semantic init seeds embeddings post-construction (it needs the
                // corpus), so disable the random structured init when it's on.
                cfg.structured_init = structured_init && !do_semantic;
                cfg.grad_accum = grad_accum;
                cfg.dense_ffn = dense_ffn;
                cfg.dot_attention = dot_attention;
                if let Some(s) = init_seed {
                    cfg.init_seed = s;
                }
                let mut st = ModelStateV2::new(cfg);
                if dense_ffn {
                    use growformer_llm::matched_dense_ffn_hidden;
                    let h = matched_dense_ffn_hidden(d_model, d_ff);
                    eprintln!("[train] dense FFN ablation: matched hidden H={h} (d_model={d_model} d_ff={d_ff})");
                }
                if dot_attention {
                    eprintln!("[train] dot-attention ablation (row 3b): Q·K scores; Clifford Q/K/V/O + FFN unchanged");
                }
                if do_semantic {
                    eprintln!(
                        "[train] corpus-semantic embedding init (random indexing, window=±{semantic_window}; --no-semantic-init to disable)"
                    );
                    corpus_semantic_init(&mut st.model, &train_ds.tokens, 0x5EED ^ 0xE8E8, semantic_window, 1.0);
                    if st.cfg.tie_embeddings {
                        st.model.sync_tied_head();
                    }
                }
                st
            };

            if freeze_blocks > state.cfg.n_blocks {
                return Err(format!(
                    "freeze_blocks ({freeze_blocks}) > n_blocks ({})",
                    state.cfg.n_blocks
                ));
            }
            if (freeze_blocks > 0 || freeze_embeddings) && init_from.is_none() {
                eprintln!("[train] note: freezing requested without --init-from; freezing fresh random weights");
            }
            if freeze_blocks > 0 || freeze_embeddings {
                eprintln!(
                    "[train] freeze: embeddings={freeze_embeddings} blocks=[0..{freeze_blocks}) (adapting blocks [{freeze_blocks}..{}), final_norm, head)",
                    state.cfg.n_blocks
                );
            }
            state.update_lr();

            let log_every_u64 = state.cfg.log_every as u64;
            let mut rng = SimpleRng::new(0xC0FFEE);

            eprintln!(
                "[train] vocab={vs} train_tokens={} val_tokens={} seq_len={seq_len} steps={steps} head_only={head_only}",
                train_ds.n_tokens(),
                val_ds.n_tokens()
            );
            if head_only {
                eprintln!(
                    "[train] head-only mode: blocks + embeddings frozen; only output head updates (fast sanity run)"
                );
            }

            let sample_cfg = SampleConfig {
                temperature: 0.85,
                top_p: Some(0.9),
                repetition_penalty: 1.15,
                max_new_tokens: 48,
                stop_tokens: vec![special::EOS],
                seed: Some(12345),
                ..Default::default()
            };

            if grad_accum > 1 {
                eprintln!(
                    "[train] gradient accumulation: {grad_accum} microbatches/step (effective batch={grad_accum}, {} chunks total)",
                    steps * grad_accum as u64
                );
            }
            for step in 1u64..=steps {
                let loss = if head_only {
                    let ex = chunk_to_example(train_ds.random_chunk(seq_len, &mut rng));
                    train_step_v2_head_only(&mut state, &ex)
                } else if grad_accum > 1 {
                    let exs: Vec<TrainExample> = (0..grad_accum)
                        .map(|_| chunk_to_example(train_ds.random_chunk(seq_len, &mut rng)))
                        .collect();
                    train_step_v2_accum(&mut state, &exs)
                } else {
                    let ex = chunk_to_example(train_ds.random_chunk(seq_len, &mut rng));
                    train_step_v2(&mut state, &ex)
                };

                if step % log_every_u64 == 0 || step == 1 {
                    let ppl = (loss.exp()).min(1e6);
                    eprintln!("[train] step={step} loss={loss:.4} ppl~={ppl:.1}");
                }

                if sample_every > 0 && step % sample_every == 0 {
                    sample_prompt(&state, &bpe, &sample_cfg, step.wrapping_mul(991));
                }

                if step % 200 == 0 && val_chunks > 0 {
                    let mut vloss = 0.0f32;
                    for _ in 0..val_chunks {
                        let chunk = val_ds.random_chunk(seq_len, &mut rng);
                        let ex = chunk_to_example(chunk);
                        vloss += eval_lm_loss(&state, &ex);
                    }
                    vloss /= val_chunks as f32;
                    eprintln!(
                        "[val] step={step} mean_nll={vloss:.4} ppl~={}",
                        (vloss.exp()).min(1e6)
                    );
                }
            }

            save_lm_state(&checkpoint_out, &state)?;
            eprintln!("[train] wrote {}", checkpoint_out.display());
        }
        Commands::Split {
            src,
            train_out,
            held_out,
            train_frac,
        } => {
            let ds = PackedDataset::load(&src).map_err(|e| e.to_string())?;
            let (train, held) = ds.split_chronological(train_frac);
            train.write(&train_out).map_err(|e| e.to_string())?;
            held.write(&held_out).map_err(|e| e.to_string())?;
            eprintln!(
                "[split] train_frac={train_frac} train={} tokens → {}  held-out={} tokens → {}",
                train.n_tokens(),
                train_out.display(),
                held.n_tokens(),
                held_out.display()
            );
        }
        Commands::Baselines {
            train_bin,
            eval_bin,
            tokenizer,
        } => {
            let bpe = BpeTokenizer::load(&tokenizer).map_err(|e| e.to_string())?;
            let vocab = bpe.vocab_size() as usize;
            let train_ds = PackedDataset::load(&train_bin).map_err(|e| e.to_string())?;
            let eval_ds = PackedDataset::load(&eval_bin).map_err(|e| e.to_string())?;
            let (counts, total) = train_ds.unigram_counts(vocab);
            let (uni_nats, n_tok) =
                PackedDataset::unigram_nll_nats(&counts, total, &eval_ds.tokens, vocab);
            if n_tok == 0 {
                return Err("no text tokens in eval shard".into());
            }
            let uni_bpt = uni_nats / std::f64::consts::LN_2;
            let uni_ppl = uni_nats.exp();
            let mut uni_bytes = 0usize;
            for &t in &eval_ds.tokens {
                let id = t as usize;
                if id >= N_SPECIAL && id < vocab {
                    uni_bytes += bpe.vocab[id].len();
                }
            }
            let uni_bpb = (uni_bpt * n_tok as f64) / uni_bytes as f64;
            let uniform_nats = (vocab as f64).ln();
            let uniform_bpt = uniform_nats / std::f64::consts::LN_2;
            let uniform_bpb = (uniform_bpt * n_tok as f64) / uni_bytes as f64;

            println!("=== token baselines (full eval shard) ===");
            println!("train counts: {}  eval tokens: {}  eval bytes: {}", total, n_tok, uni_bytes);
            println!(
                "  uniform : ppl {:.1}  {:.3} bits/token  {:.4} bits/byte",
                (vocab as f64),
                uniform_bpt,
                uniform_bpb
            );
            println!(
                "  unigram : ppl {:.1}  {:.3} bits/token  {:.4} bits/byte  (MLE from train shard)",
                uni_ppl,
                uni_bpt,
                uni_bpb
            );
        }
        Commands::Eval {
            checkpoint,
            tokenizer,
            val_bin,
            train_bin,
            seq_len,
            windows,
            run_id,
            ledger,
            no_ledger,
            selection_tag,
            vanilla,
        } => {
            let peek_cfg = peek_checkpoint_cfg(&checkpoint)?;
            let use_vanilla = vanilla || peek_cfg.vanilla;
            if vanilla && !peek_cfg.vanilla {
                eprintln!("[eval] --vanilla flag set; loading as row-2 vanilla checkpoint");
            }

            let bpe = BpeTokenizer::load(&tokenizer).map_err(|e| e.to_string())?;
            let val_ds = PackedDataset::load(&val_bin).map_err(|e| e.to_string())?;
            let toks = &val_ds.tokens;
            let vocab = bpe.vocab_size() as usize;

            if use_vanilla {
                let state = load_vanilla_state(&checkpoint)?;
                if state.cfg.vocab_size != bpe.vocab_size() as usize {
                    return Err(format!(
                        "checkpoint vocab_size {} != BPE {}",
                        state.cfg.vocab_size,
                        bpe.vocab_size()
                    ));
                }
                let uniform_bpt = (vocab as f64).log2();
                let (uni_counts, uni_total) = if let Some(train_path) = &train_bin {
                    let train_ds = PackedDataset::load(train_path).map_err(|e| e.to_string())?;
                    train_ds.unigram_counts(vocab)
                } else {
                    eprintln!("[eval] warning: no --train-bin; empirical unigram uses eval shard (in-sample)");
                    val_ds.unigram_counts(vocab)
                };

                let mut model_bits = 0.0f64;
                let mut uniform_bits = 0.0f64;
                let mut unigram_bits = 0.0f64;
                let mut total_bytes = 0usize;
                let mut n_pred = 0usize;
                let mut text_bytes: Vec<u8> = Vec::new();
                let mut per_window_bpt: Vec<f64> = Vec::with_capacity(windows);

                for w in 0..windows {
                    let start = w * seq_len;
                    if start + 2 > toks.len() {
                        break;
                    }
                    let end = (start + seq_len).min(toks.len());
                    let window: Vec<usize> = toks[start..end].iter().map(|&x| x as usize).collect();
                    let logits = vanilla_forward_logits(&state.model, &window, true);
                    let mut window_bits = 0.0f64;
                    let mut window_pred = 0usize;
                    for p in 0..window.len().saturating_sub(1) {
                        let target = window[p + 1];
                        if target < N_SPECIAL {
                            continue;
                        }
                        let probs = logits_softmax(&logits[p]);
                        let pr = (probs[target] as f64).max(1e-12);
                        let bit = -pr.log2();
                        model_bits += bit;
                        window_bits += bit;
                        window_pred += 1;
                        uniform_bits += uniform_bpt;
                        let c = uni_counts[target] as f64;
                        let p_uni = (c / uni_total as f64).max(1e-12);
                        unigram_bits += -p_uni.log2();
                        let bytes = &bpe.vocab[target];
                        total_bytes += bytes.len();
                        text_bytes.extend_from_slice(bytes);
                        n_pred += 1;
                    }
                    if window_pred > 0 {
                        per_window_bpt.push(window_bits / window_pred as f64);
                    }
                }

                if total_bytes == 0 || n_pred == 0 {
                    return Err("no text tokens evaluated (val bin too small or all special)".into());
                }

                let model_bpb = model_bits / total_bytes as f64;
                let model_bpt = model_bits / n_pred as f64;
                let uniform_bpb = uniform_bits / total_bytes as f64;
                let unigram_bpb = unigram_bits / total_bytes as f64;
                let unigram_bpt = unigram_bits / n_pred as f64;
                let model_nats_per_tok = model_bpt * std::f64::consts::LN_2;
                let model_ppl = model_nats_per_tok.exp();
                let unigram_ppl = (unigram_bpt * std::f64::consts::LN_2).exp();

                let gz = BinaryCompressor::with_settings(BinaryAlgorithm::Gzip, 9)
                    .compress(&text_bytes)
                    .map_err(|e| e.to_string())?;
                let lz = BinaryCompressor::with_settings(BinaryAlgorithm::Lzma, 9)
                    .compress(&text_bytes)
                    .map_err(|e| e.to_string())?;
                let gz_bpb = gz.len() as f64 * 8.0 / total_bytes as f64;
                let lz_bpb = lz.len() as f64 * 8.0 / total_bytes as f64;

                println!("=== prediction ⇄ compression eval (row 2 vanilla) ===");
                println!(
                    "text: {} tokens, {} bytes ({} windows × {} tokens)",
                    n_pred, total_bytes, windows, seq_len
                );
                println!();
                println!("model (conditional CE; weights not amortized):");
                println!("  cross-entropy : {model_nats_per_tok:.4} nats/token");
                println!("  perplexity    : {model_ppl:.1}");
                println!("  bits/token    : {model_bpt:.4}");
                println!("  bits/byte     : {model_bpb:.4}");
                println!();
                println!("token baselines (same predicted tokens):");
                println!(
                    "  uniform       : {uniform_bpb:.4} bits/byte  ({uniform_bpt:.2} bits/token; floor log2({vocab}))"
                );
                println!(
                    "  unigram       : {unigram_bpb:.4} bits/byte  ({unigram_bpt:.2} bits/token; ppl {unigram_ppl:.1})"
                );
                println!();
                println!("byte baselines (same bytes; not token-aligned):");
                println!("  gzip -9       : {gz_bpb:.4} bits/byte");
                println!("  lzma -9       : {lz_bpb:.4} bits/byte");

                if !no_ledger && train_bin.is_some() && !per_window_bpt.is_empty() {
                    let split_hash = ledger::compute_split_hash(
                        &val_bin,
                        seq_len,
                        per_window_bpt.len(),
                        &selection_tag,
                    )
                    .map_err(|e| e.to_string())?;
                    let rid = run_id.unwrap_or_else(|| default_run_id(&checkpoint));
                    ensure_parent_dir(&ledger)?;
                    let rec = ledger::append_eval_record(
                        &ledger,
                        &rid,
                        'B',
                        &ledger_config_hash(&state.cfg),
                        state.cfg.init_seed,
                        &checkpoint.display().to_string(),
                        &split_hash,
                        seq_len,
                        per_window_bpt,
                        "held-out eval",
                        &git_sha_short(),
                    )
                    .map_err(|e| e.to_string())?;
                    eprintln!(
                        "[ledger] appended run_id={} mean_bpt={:.4} n_windows={} → {}",
                        rec.run_id,
                        rec.mean_bpt,
                        rec.n_windows,
                        ledger.display()
                    );
                }
                return Ok(());
            }

            let state = load_lm_state(&checkpoint)?;
            if state.cfg.vocab_size != bpe.vocab_size() as usize {
                return Err(format!(
                    "checkpoint vocab_size {} != BPE {}",
                    state.cfg.vocab_size,
                    bpe.vocab_size()
                ));
            }

            // Unigram baselines: uniform floor + empirical counts from train shard.
            let uniform_bpt = (vocab as f64).log2();
            let (uni_counts, uni_total) = if let Some(train_path) = &train_bin {
                let train_ds = PackedDataset::load(train_path).map_err(|e| e.to_string())?;
                train_ds.unigram_counts(vocab)
            } else {
                eprintln!("[eval] warning: no --train-bin; empirical unigram uses eval shard (in-sample)");
                val_ds.unigram_counts(vocab)
            };

            let mut model_bits = 0.0f64;
            let mut uniform_bits = 0.0f64;
            let mut unigram_bits = 0.0f64;
            let mut total_bytes = 0usize;
            let mut n_pred = 0usize;
            let mut text_bytes: Vec<u8> = Vec::new();
            let mut per_window_bpt: Vec<f64> = Vec::with_capacity(windows);

            for w in 0..windows {
                let start = w * seq_len;
                if start + 2 > toks.len() {
                    break;
                }
                let end = (start + seq_len).min(toks.len());
                let window: Vec<usize> = toks[start..end].iter().map(|&x| x as usize).collect();
                let logits = model_forward_logits(
                    &state.alg,
                    &state.model,
                    &window,
                    true,
                    state.cfg.dot_attention,
                );
                let mut window_bits = 0.0f64;
                let mut window_pred = 0usize;
                for p in 0..window.len().saturating_sub(1) {
                    let target = window[p + 1];
                    if target < N_SPECIAL {
                        continue;
                    }
                    let probs = logits_softmax(&logits[p]);
                    let pr = (probs[target] as f64).max(1e-12);
                    let bit = -pr.log2();
                    model_bits += bit;
                    window_bits += bit;
                    window_pred += 1;
                    uniform_bits += uniform_bpt;
                    let c = uni_counts[target] as f64;
                    let p_uni = (c / uni_total as f64).max(1e-12);
                    unigram_bits += -p_uni.log2();
                    let bytes = &bpe.vocab[target];
                    total_bytes += bytes.len();
                    text_bytes.extend_from_slice(bytes);
                    n_pred += 1;
                }
                if window_pred > 0 {
                    per_window_bpt.push(window_bits / window_pred as f64);
                }
            }

            if total_bytes == 0 || n_pred == 0 {
                return Err("no text tokens evaluated (val bin too small or all special)".into());
            }

            let model_bpb = model_bits / total_bytes as f64;
            let model_bpt = model_bits / n_pred as f64;
            let uniform_bpb = uniform_bits / total_bytes as f64;
            let unigram_bpb = unigram_bits / total_bytes as f64;
            let unigram_bpt = unigram_bits / n_pred as f64;
            let model_nats_per_tok = model_bpt * std::f64::consts::LN_2;
            let model_ppl = model_nats_per_tok.exp();
            let unigram_ppl = (unigram_bpt * std::f64::consts::LN_2).exp();

            // Classical baselines on the exact same byte stream (spacekit-compressor).
            let gz = BinaryCompressor::with_settings(BinaryAlgorithm::Gzip, 9)
                .compress(&text_bytes)
                .map_err(|e| e.to_string())?;
            let lz = BinaryCompressor::with_settings(BinaryAlgorithm::Lzma, 9)
                .compress(&text_bytes)
                .map_err(|e| e.to_string())?;
            let gz_bpb = gz.len() as f64 * 8.0 / total_bytes as f64;
            let lz_bpb = lz.len() as f64 * 8.0 / total_bytes as f64;

            println!("=== prediction ⇄ compression eval ===");
            println!(
                "text: {} tokens, {} bytes ({} windows × {} tokens)",
                n_pred, total_bytes, windows, seq_len
            );
            println!();
            println!("model (conditional CE; weights not amortized):");
            println!("  cross-entropy : {model_nats_per_tok:.4} nats/token");
            println!("  perplexity    : {model_ppl:.1}");
            println!("  bits/token    : {model_bpt:.4}");
            println!("  bits/byte     : {model_bpb:.4}");
            println!();
            println!("token baselines (same predicted tokens):");
            println!(
                "  uniform       : {uniform_bpb:.4} bits/byte  ({uniform_bpt:.2} bits/token; floor log2({vocab}))"
            );
            println!(
                "  unigram       : {unigram_bpb:.4} bits/byte  ({unigram_bpt:.2} bits/token; ppl {unigram_ppl:.1})"
            );
            println!();
            println!("byte baselines (same bytes; not token-aligned):");
            println!("  gzip -9       : {gz_bpb:.4} bits/byte");
            println!("  lzma -9       : {lz_bpb:.4} bits/byte");
            println!();
            let vs_uniform = model_bpt - uniform_bpt;
            let vs_unigram = model_bpt - unigram_bpt;
            println!(
                "vs uniform floor: {:+.2} bits/token ({:+.1}%); vs unigram: {:+.2} bits/token ({:+.1}%)",
                vs_uniform,
                100.0 * vs_uniform / uniform_bpt,
                vs_unigram,
                100.0 * vs_unigram / unigram_bpt
            );
            println!("headline metric: ppl {model_ppl:.0} (not bpb vs gzip)");
            if train_bin.is_some() {
                let (uni_nats, n_full) =
                    PackedDataset::unigram_nll_nats(&uni_counts, uni_total, toks, vocab);
                if n_full > 0 {
                    let full_uni_ppl = uni_nats.exp();
                    let full_uni_bpt = uni_nats / std::f64::consts::LN_2;
                    println!();
                    println!(
                        "full held-out shard unigram (MLE, train counts): ppl {full_uni_ppl:.1}  {full_uni_bpt:.3} bits/token"
                    );
                    println!(
                        "model vs unigram (ppl): {:.0} vs {:.0} ({:+.1}%)",
                        model_ppl,
                        full_uni_ppl,
                        100.0 * (model_ppl - full_uni_ppl) / full_uni_ppl
                    );
                }
            }
            println!(
                "Note: conditional model CE only — checkpoint weights excluded. \
                 gzip/lzma include codec overhead. Small corpora make gzip unreliable."
            );

            if !no_ledger && train_bin.is_some() && !per_window_bpt.is_empty() {
                let split_hash = ledger::compute_split_hash(
                    &val_bin,
                    seq_len,
                    per_window_bpt.len(),
                    &selection_tag,
                )
                .map_err(|e| e.to_string())?;
                let rid = run_id.unwrap_or_else(|| default_run_id(&checkpoint));
                ensure_parent_dir(&ledger)?;
                let rec = ledger::append_eval_record(
                    &ledger,
                    &rid,
                    'B',
                    &ledger_config_hash(&state.cfg),
                    state.cfg.init_seed,
                    &checkpoint.display().to_string(),
                    &split_hash,
                    seq_len,
                    per_window_bpt,
                    "held-out eval",
                    &git_sha_short(),
                )
                .map_err(|e| e.to_string())?;
                eprintln!(
                    "[ledger] appended run_id={} mean_bpt={:.4} n_windows={} → {}",
                    rec.run_id,
                    rec.mean_bpt,
                    rec.n_windows,
                    ledger.display()
                );
            }
        }
        Commands::LedgerVerify { ledger } => {
            match ledger::verify_chain(&ledger).map_err(|e| e.to_string())? {
                None => println!("[ledger] chain intact: {}", ledger.display()),
                Some(i) => {
                    return Err(format!(
                        "ledger tampered or corrupt at record index {i}: {}",
                        ledger.display()
                    ));
                }
            }
        }
        Commands::LedgerTable {
            ledger,
            baseline,
            candidates,
            gate,
        } => {
            let cands: Vec<&str> = candidates
                .split(',')
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .collect();
            let table = ledger::render_bet_b_table(&ledger, &baseline, &cands, gate)
                .map_err(|e| e.to_string())?;
            print!("{table}");
        }
        Commands::Cl1 {
            checkpoint_a,
            checkpoint_b,
            tokenizer,
            heldout_bin,
            seq_len,
            windows,
            cal_windows,
            run_id,
            ledger,
            selection_tag,
            cone_seed,
            no_ledger,
        } => {
            let _bpe = BpeTokenizer::load(&tokenizer).map_err(|e| e.to_string())?;
            let spec_a = load_frozen_specialist(&checkpoint_a)?;
            let spec_b = load_frozen_specialist(&checkpoint_b)?;
            let tokens = load_heldout_tokens(&heldout_bin)?;
            eprintln!(
                "[cl1] specialist A: {}  B: {}  cal={} eval={} windows",
                checkpoint_a.display(),
                checkpoint_b.display(),
                cal_windows,
                windows
            );
            let result = run_cl1(
                &spec_a,
                &spec_b,
                &tokens,
                seq_len,
                cal_windows,
                windows,
                cone_seed,
            );
            let split_hash = ledger::compute_split_hash(
                &heldout_bin,
                seq_len,
                result.per_window_routed_bpt.len(),
                &selection_tag,
            )
            .map_err(|e| e.to_string())?;
            println!("=== CL-1 preflight (standalone specialists) ===");
            println!("specialist A mean bpt: {:.4}", result.mean_bpt_a);
            println!("specialist B mean bpt: {:.4}", result.mean_bpt_b);
            println!(
                "specialist gap: {:.4} bpt  peer parity (≤{:.2}): {}",
                result.specialist_gap_bpt,
                growformer_llm::cl1::CL1_SPECIALIST_PARITY_BPT,
                if result.peer_specialists { "PASS" } else { "FAIL — imbalanced" }
            );
            println!(
                "per-window wins: A={} B={} / {}",
                result.wins_a, result.wins_b, result.eval_n
            );
            println!("best single specialist: {:.4}", result.mean_bpt_best_single);
            println!(
                "oracle per-window min: {:.4}  (gap vs best single: {:.4} bpt)",
                result.mean_bpt_oracle,
                result.oracle_gap_bpt
            );
            if result.no_complementarity {
                println!(
                    "PREFLIGHT STOP: oracle ≈ best single — no per-window complementarity; \
                     no router can beat the dominant specialist."
                );
            }
            if result.imbalanced_specialists {
                println!(
                    "PREFLIGHT: imbalanced specialists — not a routing test (dominant model wins)."
                );
            }
            println!();
            println!("=== CL-1 routed composite (Bet A) ===");
            println!("cal windows: {}  eval windows: {}", result.cal_n, result.eval_n);
            println!("routed composite:      {:.4}", result.mean_bpt_routed);
            println!("route A fraction: {:.3}", result.route_a_frac);
            println!(
                "degenerate (constant route): {}",
                if result.degenerate { "YES" } else { "no" }
            );
            let routing_interpretable =
                result.peer_specialists && result.complementarity_possible;
            println!(
                "routing interpretable: {} ({})",
                if routing_interpretable { "YES" } else { "NO" },
                if result.imbalanced_specialists {
                    "imbalanced specialists"
                } else if result.no_complementarity {
                    "oracle = best single"
                } else {
                    "peers with window-level disagreement"
                }
            );
            let pass_routed = result.mean_bpt_routed < result.mean_bpt_best_single;
            println!(
                "gate routed < best single: {} ({:.4} vs {:.4})",
                if pass_routed { "PASS" } else { "FAIL" },
                result.mean_bpt_routed,
                result.mean_bpt_best_single
            );
            if !no_ledger {
                ensure_parent_dir(&ledger)?;
                append_cl1_ledger(
                    &ledger,
                    &run_id,
                    &split_hash,
                    seq_len,
                    &result.per_window_routed_bpt,
                    &format!(
                        "CL-1 A={} B={} cal={}",
                        checkpoint_a.display(),
                        checkpoint_b.display(),
                        cal_windows
                    ),
                    &git_sha_short(),
                )?;
                eprintln!(
                    "[ledger] appended run_id={} mean_bpt={:.4} → {}",
                    run_id,
                    result.mean_bpt_routed,
                    ledger.display()
                );
            }
        }
        Commands::Generate {
            checkpoint,
            tokenizer,
            prompt,
            max_new_tokens,
            temperature,
            greedy,
            seed,
            repetition_penalty,
        } => {
            let state = load_lm_state(&checkpoint)?;
            let bpe = BpeTokenizer::load(&tokenizer).map_err(|e| e.to_string())?;
            if state.cfg.vocab_size != bpe.vocab_size() as usize {
                return Err(format!(
                    "checkpoint vocab_size {} != BPE {}",
                    state.cfg.vocab_size,
                    bpe.vocab_size()
                ));
            }

            let mut ids: Vec<usize> = vec![special::BOS];
            ids.extend(bpe.encode(&prompt).iter().map(|&x| x as usize));

            let sample_cfg = if greedy {
                SampleConfig {
                    max_new_tokens,
                    repetition_penalty,
                    seed,
                    stop_tokens: vec![special::EOS],
                    ..SampleConfig::greedy()
                }
            } else {
                SampleConfig {
                    temperature,
                    max_new_tokens,
                    repetition_penalty,
                    seed,
                    stop_tokens: vec![special::EOS],
                    ..SampleConfig::focused()
                }
            };

            let mut rng = SimpleRng::new(seed.unwrap_or(0xDECAFBAD));
            let mut cache = InferenceCache::new(
                state.cfg.n_blocks,
                state.cfg.max_seq,
                state.cfg.d_model,
                state.cfg.dot_attention,
            );
            for _ in 0..sample_cfg.max_new_tokens {
                let logits_rows = cache.forward_extend(&state.alg, &state.model, &ids);
                let Some(last) = logits_rows.last() else {
                    break;
                };
                let next = sample_next(last, &ids, &sample_cfg, &mut rng);
                if sample_cfg.stop_tokens.contains(&next) {
                    break;
                }
                print!("{}", bpe.decode_one(next as u32));
                let _ = std::io::stdout().flush();
                ids.push(next);
            }
            println!();
        }
        #[cfg(feature = "brain-memory")]
        Commands::BrainInfer {
            brain,
            prompt,
            project,
            inference_toml,
            inference_defaults_toml,
            guardrails_jsonl,
            verbose,
            checkpoint,
            tokenizer,
            max_new_tokens,
            temperature,
            greedy,
            seed,
            repetition_penalty,
            brain_only,
        } => {
            let infer_cfg = BrainInferConfig {
                project,
                inference_toml,
                inference_defaults_toml,
                guardrails_jsonl,
                verbose,
            };
            let mut mem = BrainMemoryRuntime::from_path_with_config(&brain, &infer_cfg)?;
            let info = mem.brain_info();
            let q = mem.query(&prompt)?;
            println!("=== brain memory unit ===");
            println!(
                "brain: {}  groups={}  router={}  gen_envs={}",
                brain.display(),
                info.num_groups,
                info.has_router,
                info.gen_envs
            );
            println!(
                "route: group={:?} margin={:.3} bridge_conf={:.3} ood={}",
                q.group_id, q.route_margin, q.bridge_confidence, q.route_rejected_ood
            );
            println!(
                "action: {} conf={:.3}  memory template={} conf={:.3}",
                q.action_type, q.action_confidence, q.memory_template_id, q.memory_confidence
            );
            println!("memory text:\n{}", q.memory_text.trim());
            let feats = brain_router_features(&q);
            println!(
                "router features [{:.3}, {:.3}, {:.3}, {:.3}, {:.3}, {:.3}, {:.3}, {:.3}]",
                feats[0], feats[1], feats[2], feats[3], feats[4], feats[5], feats[6], feats[7]
            );
            if brain_only {
                return Ok(());
            }
            let checkpoint = checkpoint.ok_or("--checkpoint required unless --brain-only")?;
            let tokenizer = tokenizer.ok_or("--tokenizer required unless --brain-only")?;
            let lm_prompt = format!("{}{}", format_lm_memory_prefix(&q), prompt);
            eprintln!("[brain-infer] LM prompt prefix: {} chars", lm_prompt.len());

            let peek_cfg = peek_checkpoint_cfg(&checkpoint)?;
            let bpe = BpeTokenizer::load(&tokenizer).map_err(|e| e.to_string())?;
            if peek_cfg.vocab_size != bpe.vocab_size() as usize {
                return Err(format!(
                    "checkpoint vocab {} != BPE {}",
                    peek_cfg.vocab_size,
                    bpe.vocab_size()
                ));
            }

            let mut ids: Vec<usize> = vec![special::BOS];
            ids.extend(bpe.encode(&lm_prompt).iter().map(|&x| x as usize));

            let sample_cfg = if greedy {
                SampleConfig {
                    max_new_tokens,
                    repetition_penalty,
                    seed,
                    stop_tokens: vec![special::EOS],
                    ..SampleConfig::greedy()
                }
            } else {
                SampleConfig {
                    temperature,
                    max_new_tokens,
                    repetition_penalty,
                    seed,
                    stop_tokens: vec![special::EOS],
                    ..SampleConfig::focused()
                }
            };
            let mut rng = SimpleRng::new(seed.unwrap_or(0xDECAFBAD));

            print!("=== LM continuation ===\n");
            if peek_cfg.vanilla {
                let state = load_vanilla_state(&checkpoint)?;
                for _ in 0..sample_cfg.max_new_tokens {
                    let logits_rows = vanilla_forward_logits(&state.model, &ids, true);
                    let Some(last) = logits_rows.last() else {
                        break;
                    };
                    let next = sample_next(last, &ids, &sample_cfg, &mut rng);
                    if sample_cfg.stop_tokens.contains(&next) {
                        break;
                    }
                    print!("{}", bpe.decode_one(next as u32));
                    let _ = std::io::stdout().flush();
                    ids.push(next);
                }
            } else {
                let state = load_lm_state(&checkpoint)?;
                let mut cache = InferenceCache::new(
                    state.cfg.n_blocks,
                    state.cfg.max_seq,
                    state.cfg.d_model,
                    state.cfg.dot_attention,
                );
                for _ in 0..sample_cfg.max_new_tokens {
                    let logits_rows = cache.forward_extend(&state.alg, &state.model, &ids);
                    let Some(last) = logits_rows.last() else {
                        break;
                    };
                    let next = sample_next(last, &ids, &sample_cfg, &mut rng);
                    if sample_cfg.stop_tokens.contains(&next) {
                        break;
                    }
                    print!("{}", bpe.decode_one(next as u32));
                    let _ = std::io::stdout().flush();
                    ids.push(next);
                }
            }
            println!();
        }
        #[cfg(feature = "brain-memory")]
        Commands::BrainRawDiag {
            brain,
            prompt,
            project,
            inference_toml,
            inference_defaults_toml,
            guardrails_jsonl,
            verbose,
            top_k,
            json,
            battery,
            battery_brains,
            battery_all,
        } => {
            if battery {
                if !battery_all {
                    for case in battery_cases(battery_brains) {
                        if let Some(reason) = case.skip_reason {
                            eprintln!("[battery] skip {}: {}", case.label, reason);
                        }
                    }
                }
                let cases: Vec<_> = if battery_all {
                    battery_cases(battery_brains).to_vec()
                } else {
                    scored_battery_cases(battery_brains).collect()
                };
                for case in cases {
                    if case.skip_reason.is_some() {
                        eprintln!(
                            "[battery] {} (untrained sentiment brain — diagnostic only, not scored)",
                            case.label
                        );
                    }
                    println!("========== {} ==========", case.label);
                    let infer_cfg = BrainInferConfig {
                        project: Some(case.project.clone()),
                        inference_toml: None,
                        inference_defaults_toml: None,
                        guardrails_jsonl: None,
                        verbose,
                    };
                    let mut mem =
                        BrainMemoryRuntime::from_path_with_config(&case.brain, &infer_cfg)?;
                    let report = mem.raw_lattice_diagnostic(case.prompt, top_k)?;
                    if json {
                        println!("{}", raw_lattice_report_json(&report)?);
                    } else {
                        print_raw_lattice_report(&report);
                    }
                    println!();
                }
            } else {
                let brain = brain.ok_or("--brain required unless --battery")?;
                let prompt = prompt.ok_or("--prompt required unless --battery")?;
                let infer_cfg = BrainInferConfig {
                    project,
                    inference_toml,
                    inference_defaults_toml,
                    guardrails_jsonl,
                    verbose,
                };
                let mut mem = BrainMemoryRuntime::from_path_with_config(&brain, &infer_cfg)?;
                let report = mem.raw_lattice_diagnostic(&prompt, top_k)?;
                if json {
                    println!("{}", raw_lattice_report_json(&report)?);
                } else {
                    print_raw_lattice_report(&report);
                }
            }
        }
    }
    Ok(())
}

#[cfg(feature = "brain-memory")]
fn print_raw_lattice_report(report: &growformer::dimension::group_gen::RawLatticeDiagnosticReport) {
    println!("prompt: {}", report.prompt);
    println!(
        "route: group={:?} topic_hint={:?} path={} forced_topic={:?}",
        report.group_idx, report.topic_hint, report.retrieval_path, report.forced_topic
    );
    println!("subject_keywords: {:?}", report.subject_keywords);
    if report.candidates.is_empty() {
        println!("candidates: (none — check retrieval_path)");
        return;
    }
    println!("top-{} pre-gate candidates:", report.candidates.len());
    for c in &report.candidates {
        println!(
            "  #{} prog={} score={:.3} topic={} witness={} hard_reject={} soft_reject={} graph_conf={} floor={}",
            c.rank,
            c.prog_idx,
            c.score,
            c.topic,
            c.witness_ok,
            c.hard_reject,
            c.soft_reject,
            c.graph_confident,
            c.above_score_floor
        );
        println!("      {}", c.text_preview.replace('\n', " "));
    }
}
