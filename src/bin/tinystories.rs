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
use growformer_llm::v2::checkpoint::{load_lm_state, save_lm_state};
use growformer_llm::v2::data::{special, TrainExample, N_SPECIAL};
use growformer_llm::v2::sample::{sample_next, softmax as logits_softmax, SampleConfig, SimpleRng};
use growformer_llm::v2::inference::InferenceCache;
use growformer_llm::v2::tape::model_forward_logits;
use growformer_llm::v2::train_v2::{
    corpus_semantic_init, train_step_v2, train_step_v2_accum, train_step_v2_head_only, ModelStateV2,
    TrainConfigV2,
};

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
    },
    /// Prediction⇄compression: model bits/byte (= NLL/ln2 per byte) vs gzip/LZMA baselines.
    Eval {
        #[arg(long)]
        checkpoint: PathBuf,
        #[arg(long)]
        tokenizer: PathBuf,
        val_bin: PathBuf,
        #[arg(long, default_value_t = 128)]
        seq_len: usize,
        /// Number of non-overlapping windows of `seq_len` tokens to evaluate.
        #[arg(long, default_value_t = 32)]
        windows: usize,
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
}

fn eval_lm_loss(state: &ModelStateV2, ex: &TrainExample) -> f32 {
    let logits = model_forward_logits(&state.alg, &state.model, &ex.full_ids, true);
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
        let logits = model_forward_logits(&state.alg, &state.model, &ids, true);
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
                let mut st = ModelStateV2::new(cfg);
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
        Commands::Eval {
            checkpoint,
            tokenizer,
            val_bin,
            seq_len,
            windows,
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
            let val_ds = PackedDataset::load(&val_bin).map_err(|e| e.to_string())?;
            let toks = &val_ds.tokens;

            // Accumulate the model's information content (in bits) over the
            // text tokens it predicts, plus the exact bytes those tokens encode.
            // Special tokens (PAD/UNK/BOS/SEP/EOS) carry no text bytes and are
            // skipped so the bits/byte figure is over real text only.
            let mut model_bits = 0.0f64;
            let mut total_bytes = 0usize;
            let mut n_pred = 0usize;
            let mut text_bytes: Vec<u8> = Vec::new();

            for w in 0..windows {
                let start = w * seq_len;
                if start + 2 > toks.len() {
                    break;
                }
                let end = (start + seq_len).min(toks.len());
                let window: Vec<usize> = toks[start..end].iter().map(|&x| x as usize).collect();
                let logits = model_forward_logits(&state.alg, &state.model, &window, true);
                for p in 0..window.len().saturating_sub(1) {
                    let target = window[p + 1];
                    if target < N_SPECIAL {
                        continue;
                    }
                    let probs = logits_softmax(&logits[p]);
                    let pr = (probs[target] as f64).max(1e-12);
                    model_bits += -pr.log2();
                    let bytes = &bpe.vocab[target];
                    total_bytes += bytes.len();
                    text_bytes.extend_from_slice(bytes);
                    n_pred += 1;
                }
            }

            if total_bytes == 0 || n_pred == 0 {
                return Err("no text tokens evaluated (val bin too small or all special)".into());
            }

            let model_bpb = model_bits / total_bytes as f64;
            let model_bpt = model_bits / n_pred as f64;
            let model_nats_per_tok = model_bpt * std::f64::consts::LN_2;
            let model_ppl = model_nats_per_tok.exp();

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
            println!("model (growformer-llm, conditional on trained weights):");
            println!("  cross-entropy : {model_nats_per_tok:.4} nats/token");
            println!("  perplexity    : {model_ppl:.2}");
            println!("  bits/token    : {model_bpt:.4}");
            println!("  bits/byte     : {model_bpb:.4}   (weights NOT amortized; see README)");
            println!();
            println!("classical baselines (same bytes; codec overhead included, no separate model file):");
            println!("  gzip -9       : {gz_bpb:.4} bits/byte");
            println!("  lzma -9       : {lz_bpb:.4} bits/byte");
            println!();
            println!(
                "Note: model bits/byte is conditional cross-entropy only — checkpoint \
                 weights are not included in the bit count. gzip/lzma include codec \
                 overhead on the same bytes. Not a like-for-like shipped-size comparison."
            );
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
    }
    Ok(())
}
