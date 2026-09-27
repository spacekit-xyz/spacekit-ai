//! TinyStories / domain LM pipeline: BPE (`tokenize`), packed corpus (`encode`),
//! vanilla LM by default (`train` / `generate` / `eval`), optional Clifford (`--clifford`).
//!
//! Checkpoints from `train` omit the word tokenizer JSON list — keep the `.tok` next to the `.json`
//! and pass both to `generate`.

use std::io::{Cursor, Write};
use std::path::{Path, PathBuf};

use clap::{Parser, Subcommand};
use flate2::{write::GzEncoder, Compression};

use growformer_llm::bpe::BpeTokenizer;
use growformer_llm::cross_entropy;
use growformer_llm::label_classifier::{
    coarse_sentiment, reply_label_head, LabelClassifier, LabelTrainConfig,
};
use growformer_llm::model_card::{Arch, ModelCard, SpecialistManifest, CARD_EXT};
use growformer_llm::param_budget::{log_param_match, matched_vanilla_d_model};
use growformer_llm::tinystories::{
    chunk_to_example, encode_corpus, load_tinystories_txt, PackedDataset,
};
use growformer_llm::v2::data::{special, TrainExample, N_SPECIAL};
use growformer_llm::v2::sample::{sample_next, SampleConfig, SimpleRng};
use growformer_llm::v2::vanilla_checkpoint::{
    load_vanilla_state, load_vanilla_state_for_resume, save_vanilla_optim, save_vanilla_state,
};
use growformer_llm::v2::vanilla_train::{
    corpus_semantic_init_vanilla, eval_vanilla_set, train_step_vanilla_accum, EarlyStopper,
    VanillaModelState,
};
use growformer_llm::vanilla_llm::vanilla_forward_logits;
use growformer_llm::TrainConfigV2;

#[cfg(feature = "clifford-lm")]
use growformer_llm::cl1::{
    append_cl1_ledger, load_frozen_specialist, load_heldout_tokens, run_cl1,
};
#[cfg(feature = "clifford-lm")]
use growformer_llm::v2::checkpoint::{load_lm_state, save_lm_state};
#[cfg(feature = "clifford-lm")]
use growformer_llm::v2::inference::InferenceCache;
#[cfg(feature = "clifford-lm")]
use growformer_llm::v2::tape::model_forward_logits;
#[cfg(feature = "clifford-lm")]
use growformer_llm::v2::train_v2::{
    corpus_semantic_init, train_step_v2, train_step_v2_accum, train_step_v2_head_only, ModelStateV2,
};

#[cfg(feature = "brain-memory")]
use growformer_llm::brain_infer_config::{battery_cases, heldout_battery_cases, BrainInferConfig};
#[cfg(feature = "brain-memory")]
use growformer_llm::brain_memory::{
    brain_router_features, format_lm_memory_prefix_with_source, raw_lattice_report_json,
    BrainMemoryRuntime, MemorySource,
};

use growformer_ledger::results_ledger as ledger;

fn gzip_bytes(input: &[u8]) -> Result<Vec<u8>, String> {
    let mut encoder = GzEncoder::new(Vec::new(), Compression::best());
    encoder
        .write_all(input)
        .map_err(|error| error.to_string())?;
    encoder.finish().map_err(|error| error.to_string())
}

fn lzma_bytes(input: &[u8]) -> Result<Vec<u8>, String> {
    let mut output = Vec::new();
    lzma_rs::lzma_compress(&mut Cursor::new(input), &mut output)
        .map_err(|error| error.to_string())?;
    Ok(output)
}

#[derive(Parser)]
#[command(name = "tinystories")]
#[command(about = "Growformer LM: train / eval / generate / chat (vanilla default)", long_about = None)]
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
    /// Convert Growformer JSONL dir(s) (`text` field) → TinyStories-style `.txt`.
    JsonlToTxt {
        /// One or more dirs containing `train_*.jsonl` (crypto/fintech/sentiment).
        #[arg(required = true, num_args = 1..)]
        dirs: Vec<PathBuf>,
        #[arg(long, short)]
        out: PathBuf,
        /// Emit `### User:` / `### Assistant:` turns (`expected_response` or polarity fallback).
        #[arg(long, default_value_t = false)]
        chat: bool,
        /// Clean assistant lines (strip meta rationales). Default on; pass `--chat-clean false` for raw.
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        chat_clean: bool,
        /// Soft cap on cleaned assistant characters.
        #[arg(long, default_value_t = 160)]
        max_assistant_chars: usize,
        /// Optional system prompt prepended to each chat turn (implies --chat). Prefer omit for train.
        #[arg(long)]
        system: Option<String>,
    },
    /// Interactive chatbot REPL (vanilla LM; optional brain compose / polish).
    Chat {
        /// LM checkpoint (required for compose=lm|polish).
        #[arg(long)]
        checkpoint: Option<PathBuf>,
        #[arg(long)]
        tokenizer: Option<PathBuf>,
        /// System prompt (default: concise domain assistant).
        #[arg(long)]
        system: Option<String>,
        #[arg(long, default_value_t = 64)]
        max_new_tokens: usize,
        #[arg(long, default_value_t = 0.4)]
        temperature: f32,
        /// Greedy decoding (default). Pass `--greedy false` to sample.
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        greedy: bool,
        #[arg(long)]
        seed: Option<u64>,
        #[arg(long, default_value_t = 1.2)]
        repetition_penalty: f32,
        /// Leave room under checkpoint max_seq for the reply.
        #[arg(long, default_value_t = 48)]
        reply_reserve: usize,
        /// `brain` = Path A memory as answer (default); `polish` = LM rewrites memory; `lm` = experimental LM chat.
        #[arg(long, default_value = "brain")]
        compose: String,
        /// Optional brain.bin — required for compose=brain|polish; optional for lm hybrid.
        #[cfg(feature = "brain-memory")]
        #[arg(long)]
        brain: Option<PathBuf>,
        #[cfg(feature = "brain-memory")]
        #[arg(long, value_name = "PATH")]
        project: Option<PathBuf>,
        #[cfg(feature = "brain-memory")]
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        hybrid: bool,
        /// Supervised label model (`gf-llm label-train`). When set, it picks the label and
        /// the brain retrieves the explanation from that label's sub-lattice.
        #[cfg(feature = "brain-memory")]
        #[arg(long, value_name = "PATH")]
        label_model: Option<PathBuf>,
        /// Minimum classifier probability to override the brain's own route (0 = always).
        #[cfg(feature = "brain-memory")]
        #[arg(long, default_value_t = 0.0)]
        label_min_conf: f32,
        /// Single-shot user message (no REPL). Useful for scripts.
        #[arg(long)]
        message: Option<String>,
    },
    /// Random-chunk / turn-aligned training on packed bins (CPU defaults: d_model=16, n_blocks=4).
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
        /// Train Clifford Cl(1,3) LM (Bet B research — requires feature clifford-lm).
        /// Default is vanilla transformer (product core).
        #[arg(long, default_value_t = false)]
        clifford: bool,
        /// Alias kept for older scripts; same as default (vanilla). Ignored if --clifford.
        #[arg(long, default_value_t = false)]
        vanilla: bool,
        /// Sample whole BOS→EOS documents (chat turns), PAD-pad to seq_len. Prefer for chat corpora.
        #[arg(long, default_value_t = false)]
        turn_aligned: bool,
        /// Vanilla: use --d-model exactly instead of param-matching it to a Clifford
        /// reference of that size (the Bet B comparison default). Recommended for product runs.
        #[arg(long, default_value_t = false)]
        no_param_match: bool,
        /// Vanilla: AdamW decoupled weight decay on attention/FFN/head matrices
        /// (never on LayerNorm, biases, embeddings).
        #[arg(long, default_value_t = 0.1)]
        weight_decay: f32,
        /// Vanilla: Adam β₂ (0.95 is steadier than 0.999 at batch size 1–8).
        #[arg(long, default_value_t = 0.95)]
        beta2: f32,
        /// Vanilla: global grad-norm clip across all trainable tensors (0 = off).
        #[arg(long, default_value_t = 1.0)]
        grad_clip: f32,
        /// Vanilla: LR floor for warmup start and cosine end.
        #[arg(long, default_value_t = 1e-5)]
        lr_min: f32,
        /// Vanilla: token-embedding multiplier before positions (0 = auto sqrt(d_model)).
        #[arg(long, default_value_t = 0.0)]
        embed_scale: f32,
        /// Vanilla: evaluate the fixed validation set every N steps (0 = off).
        #[arg(long, default_value_t = 200)]
        val_every: u64,
        /// Vanilla: stop after N consecutive validations without improvement (0 = never).
        #[arg(long, default_value_t = 0)]
        patience: usize,
        /// Vanilla: write the final weights to --checkpoint-out instead of the best-validation
        /// weights (by default the best goes to --checkpoint-out and the final to *.last.json).
        #[arg(long, default_value_t = false)]
        no_keep_best: bool,
        /// Vanilla: continue an interrupted run exactly (weights + Adam moments + step).
        /// Needs the `*.optim.json` sidecar written next to the checkpoint. --steps is the
        /// total schedule length, not additional steps.
        #[arg(long)]
        resume: Option<PathBuf>,
        /// Specialist card subject (defaults to the checkpoint file stem).
        #[arg(long)]
        subject: Option<String>,
        /// Comma-separated router keywords for the specialist card.
        #[arg(long)]
        keywords: Option<String>,
    },
    /// Train the supervised label classifier (TF-IDF + logistic regression) on JSONL
    /// `semantic_intent` labels. Same file-selection rules as `jsonl-to-txt`.
    LabelTrain {
        /// Dirs with training `*.jsonl` (e.g. `<project>/data`).
        dirs: Vec<PathBuf>,
        #[arg(long, short)]
        out: PathBuf,
        /// Inverse L2 strength (larger = weaker regularisation).
        #[arg(long, default_value_t = 10.0)]
        c: f32,
        #[arg(long, default_value_t = 400)]
        epochs: usize,
        /// Also report accuracy on a stratified held-out slice of this size (0 = off;
        /// the saved model is then trained on the remaining rows only).
        #[arg(long, default_value_t = 0.0)]
        holdout_frac: f64,
        #[arg(long, default_value_t = 7)]
        seed: u64,
    },
    /// Predict labels for a prompt with a trained label model.
    LabelPredict {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        prompt: String,
        #[arg(long, default_value_t = 3)]
        top: usize,
    },
    /// Stratified train/test split of labelled JSONL rows (per `semantic_intent`).
    JsonlSplit {
        dirs: Vec<PathBuf>,
        #[arg(long)]
        train_out: PathBuf,
        #[arg(long)]
        test_out: PathBuf,
        #[arg(long, default_value_t = 0.2)]
        test_frac: f64,
        #[arg(long, default_value_t = 7)]
        seed: u64,
    },
    /// Score a brain (and optionally a label model) on held-out JSONL rows.
    /// Loads the brain once; reports label accuracy, coarse sentiment accuracy and hedge rate.
    #[cfg(feature = "brain-memory")]
    BrainEval {
        #[arg(long)]
        brain: PathBuf,
        #[arg(long, value_name = "PATH")]
        project: Option<PathBuf>,
        /// Held-out rows (`gf-llm jsonl-split --test-out`).
        #[arg(long)]
        test: PathBuf,
        #[arg(long, value_name = "PATH")]
        label_model: Option<PathBuf>,
        #[arg(long, default_value_t = 0.0)]
        label_min_conf: f32,
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        hybrid: bool,
        /// Evaluate at most N rows (0 = all).
        #[arg(long, default_value_t = 0)]
        limit: usize,
        /// Write per-row results + summary JSON here.
        #[arg(long)]
        json_out: Option<PathBuf>,
        /// Print every row.
        #[arg(long, short = 'v', default_value_t = false)]
        verbose: bool,
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
    #[cfg(feature = "clifford-lm")]
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
        #[arg(long, required_unless_present_any = ["battery", "heldout"])]
        brain: Option<PathBuf>,
        #[arg(long, required_unless_present_any = ["battery", "heldout"])]
        prompt: Option<String>,
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
        /// Prefer raw lattice top-1 when scenario-topic rubric passes (HYBRID_DOMAIN_BRAIN).
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        hybrid: bool,
        /// Supervised label model (`gf-llm label-train`): picks the label, brain explains.
        #[arg(long, value_name = "PATH")]
        label_model: Option<PathBuf>,
        #[arg(long, default_value_t = 0.0)]
        label_min_conf: f32,
        /// Run scored SpaceKit battery (cases 2–3).
        #[arg(long, default_value_t = false)]
        battery: bool,
        /// Run pre-registered held-out paraphrase prompts.
        #[arg(long, default_value_t = false)]
        heldout: bool,
        #[arg(long, default_value_t = false)]
        battery_brains: bool,
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
        /// Skip headline routing; retrieve from this topic sub-lattice (routing isolation).
        #[arg(long, value_name = "TOPIC")]
        force_topic: Option<String>,
        /// Run the SpaceKit battery (cases 2–3; ignores --brain/--prompt).
        #[arg(long, default_value_t = false)]
        battery: bool,
        /// Use battery-retrained brains (includes new JSONL rows).
        #[arg(long, default_value_t = false)]
        battery_brains: bool,
    },
}

#[cfg(feature = "clifford-lm")]
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

/// Write `<checkpoint>.gfcard.json` for a vanilla checkpoint (arch, caps,
/// tokenizer sidecar, eval bits/byte when known).
#[allow(clippy::too_many_arguments)]
fn write_vanilla_card(
    checkpoint: &Path,
    tok: &Path,
    cfg: &TrainConfigV2,
    steps: u64,
    eval_bpb: Option<f32>,
    subject: Option<String>,
    keywords: Option<String>,
    base: Option<&Path>,
) -> Result<(), String> {
    let subject = subject.unwrap_or_else(|| {
        checkpoint
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("specialist")
            .to_string()
    });
    let mut card = ModelCard::new(subject);
    card.train_steps = steps;
    card.eval_bits_per_byte = eval_bpb;
    card.base_model = base.map(|b| b.display().to_string());
    if let Some(kw) = keywords {
        card.keywords = kw
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
    }
    if eval_bpb.is_some() {
        card.notes = "eval_bits_per_byte: fixed held-out chunks during training (quick); \
                      run `gf-llm eval` for the full held-out number"
            .into();
    }
    let weights = checkpoint
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("weights.json")
        .to_string();
    let mut manifest = SpecialistManifest::for_checkpoint(Arch::Vanilla, cfg, card, weights);
    let card_path = checkpoint.with_extension(CARD_EXT);
    let card_dir = card_path.parent().unwrap_or(Path::new("."));
    manifest.tokenizer_path = Some(relative_or_absolute(tok, card_dir));
    manifest.save(&card_path)?;
    eprintln!("[train] wrote specialist card {}", card_path.display());
    Ok(())
}

/// `path` relative to `dir` when it lives underneath it, else absolute.
fn relative_or_absolute(path: &Path, dir: &Path) -> String {
    let abs = |p: &Path| std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf());
    let (p, d) = (abs(path), abs(dir));
    match p.strip_prefix(&d) {
        Ok(rel) => rel.display().to_string(),
        Err(_) => p.display().to_string(),
    }
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
                format!(
                    "could not create output directory {}: {e}",
                    parent.display()
                )
            })?;
        }
    }
    Ok(())
}

#[cfg(feature = "clifford-lm")]
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
            eprintln!(
                "[tokenize] wrote {} (vocab={})",
                out_tok.display(),
                tok.vocab_size()
            );
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
            let tokenizer = BpeTokenizer::load(&tok)
                .map_err(|e| format!("read BPE tokenizer {}: {e}", tok.display()))?;
            ensure_parent_dir(&out_bin)?;
            encode_corpus(&stories, &tokenizer, &out_bin)
                .map_err(|e| format!("write packed corpus {}: {e}", out_bin.display()))?;
            eprintln!("[encode] wrote {}", out_bin.display());
        }
        Commands::JsonlToTxt {
            dirs,
            out,
            chat,
            chat_clean,
            max_assistant_chars,
            system,
        } => {
            ensure_parent_dir(&out)?;
            let chat = chat || system.is_some();
            let n = if chat {
                let opts = growformer_llm::domain_data::ChatWriteOptions {
                    clean: chat_clean,
                    max_assistant_chars,
                };
                let sys = system.as_deref();
                growformer_llm::domain_data::jsonl_dirs_to_chat_eot_txt(&dirs, &out, sys, opts)?
            } else {
                growformer_llm::domain_data::jsonl_dirs_to_eot_txt(&dirs, &out)?
            };
            eprintln!(
                "[jsonl-to-txt] wrote {} examples ({}) → {}",
                n,
                if chat {
                    if chat_clean {
                        "chat-clean"
                    } else {
                        "chat-raw"
                    }
                } else {
                    "plain"
                },
                out.display()
            );
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
            clifford,
            vanilla: _vanilla_alias,
            turn_aligned,
            no_param_match,
            weight_decay,
            beta2,
            grad_clip,
            lr_min,
            embed_scale,
            val_every,
            patience,
            no_keep_best,
            resume,
            subject,
            keywords,
        } => {
            let use_vanilla = !clifford;
            let sample_chunk = |ds: &PackedDataset, seq_len: usize, rng: &mut SimpleRng| {
                if turn_aligned {
                    ds.random_turn_chunk(seq_len, rng)
                } else {
                    ds.random_chunk(seq_len, rng)
                }
            };
            // Corpus-semantic init is the validated default for fresh training
            // (≈37% lower val perplexity at equal steps). Opt out with
            // --no-semantic-init, or pick the random structured init explicitly.
            let do_semantic = semantic_init || (!structured_init && !no_semantic_init);
            if !use_vanilla && d_model % n_heads != 0 {
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

            if use_vanilla {
                if dense_ffn || dot_attention || head_only {
                    return Err(
                        "vanilla train does not use --dense-ffn, --dot-attention, or --head-only"
                            .into(),
                    );
                }
                if init_from.is_some() && resume.is_some() {
                    return Err("use either --init-from (fine-tune) or --resume, not both".into());
                }
                let warmup = (steps / 20).max(50).min(steps.saturating_sub(1).max(1));
                // Training knobs applied to fresh, fine-tuned and resumed runs alike.
                let apply_knobs = |cfg: &mut TrainConfigV2| {
                    cfg.max_seq = seq_len;
                    cfg.batch_size = grad_accum.max(1);
                    cfg.epochs = 1;
                    cfg.lr_max = lr_max;
                    cfg.lr_min = lr_min;
                    cfg.warmup_steps = warmup;
                    cfg.total_steps = steps;
                    cfg.log_every = 10;
                    cfg.val_every = if val_every == 0 {
                        usize::MAX
                    } else {
                        val_every as usize
                    };
                    cfg.grad_clip = grad_clip;
                    cfg.weight_decay = weight_decay;
                    cfg.adam_beta2 = beta2;
                    cfg.grad_accum = grad_accum;
                    cfg.vanilla = true;
                };

                let mut state = if let Some(ck) = &resume {
                    let mut st = load_vanilla_state_for_resume(ck)?;
                    if st.cfg.vocab_size != vs {
                        return Err(format!(
                            "resume vocab {} != tokenizer {vs}",
                            st.cfg.vocab_size
                        ));
                    }
                    // Schedule/optimiser knobs must match the original run to resume exactly;
                    // only the total length may be extended.
                    st.cfg.total_steps = steps.max(st.step);
                    st.cfg.val_every = if val_every == 0 {
                        usize::MAX
                    } else {
                        val_every as usize
                    };
                    st.update_lr();
                    eprintln!(
                        "[train] resuming {} at step {} / {} (lr={:.2e})",
                        ck.display(),
                        st.step,
                        st.cfg.total_steps,
                        st.current_lr()
                    );
                    st
                } else if let Some(base) = &init_from {
                    let mut st = load_vanilla_state(base).map_err(|e| format!("init-from: {e}"))?;
                    if st.cfg.vocab_size != vs {
                        return Err(format!(
                            "base vocab {} != tokenizer {vs} — fine-tune with the base's .tok \
                             (re-encode the domain corpus with it)",
                            st.cfg.vocab_size
                        ));
                    }
                    if freeze_blocks > st.cfg.n_blocks {
                        return Err(format!(
                            "freeze_blocks ({freeze_blocks}) > n_blocks ({})",
                            st.cfg.n_blocks
                        ));
                    }
                    apply_knobs(&mut st.cfg);
                    st.cfg.freeze_blocks = freeze_blocks;
                    st.cfg.freeze_embeddings = freeze_embeddings;
                    st.cfg.train_embeddings = true;
                    st.step = 0; // restart the LR schedule for fine-tuning
                    st.reset_optimisers();
                    eprintln!(
                        "[train] fine-tuning base {} (d_model={} n_heads={} d_ff={} n_blocks={}) \
                         freeze: embeddings={} blocks=[0..{})",
                        base.display(),
                        st.cfg.d_model,
                        st.cfg.n_heads,
                        st.cfg.d_ff,
                        st.cfg.n_blocks,
                        freeze_embeddings,
                        freeze_blocks
                    );
                    st
                } else {
                    let d_model_used = if no_param_match {
                        d_model
                    } else {
                        let clifford_ref = d_model;
                        let matched_d = matched_vanilla_d_model(
                            vs,
                            clifford_ref,
                            d_ff,
                            n_blocks,
                            n_heads,
                            tie_embeddings,
                            500,
                        );
                        log_param_match(
                            vs,
                            clifford_ref,
                            matched_d,
                            d_ff,
                            n_blocks,
                            tie_embeddings,
                        );
                        eprintln!(
                            "[train] note: --d-model {d_model} is a Clifford reference; vanilla d_model \
                             set to {matched_d} to match its params. Pass --no-param-match to use \
                             --d-model as-is."
                        );
                        matched_d
                    };
                    if d_model_used % n_heads != 0 {
                        return Err(format!(
                            "d_model ({d_model_used}) must be divisible by n_heads ({n_heads})"
                        ));
                    }
                    if d_ff < 2 * d_model_used {
                        eprintln!(
                            "[train] warning: d_ff={d_ff} < 2·d_model={}; the FFN is a bottleneck \
                             (typical d_ff ≈ 4·d_model)",
                            2 * d_model_used
                        );
                    }
                    let mut cfg = TrainConfigV2::small(vs);
                    apply_knobs(&mut cfg);
                    cfg.d_model = d_model_used;
                    cfg.n_heads = n_heads;
                    cfg.d_ff = d_ff;
                    cfg.n_blocks = n_blocks;
                    cfg.train_embeddings = true;
                    cfg.freeze_embeddings = freeze_embeddings;
                    cfg.freeze_blocks = freeze_blocks.min(n_blocks);
                    cfg.tie_embeddings = tie_embeddings;
                    cfg.structured_init = structured_init && !do_semantic;
                    cfg.clifford_ref_d_model = if no_param_match { 0 } else { d_model };
                    cfg.embed_scale = embed_scale;
                    if let Some(s) = init_seed {
                        cfg.init_seed = s;
                    }
                    let mut st = VanillaModelState::new(cfg);
                    if do_semantic {
                        eprintln!(
                            "[train] row 2 vanilla: corpus-semantic embedding init (window=±{semantic_window})"
                        );
                        corpus_semantic_init_vanilla(
                            &mut st.model,
                            &train_ds.tokens,
                            0x5EED ^ 0xE8E8,
                            semantic_window,
                            1.0,
                        );
                        if st.cfg.tie_embeddings {
                            st.model.sync_tied_head();
                        }
                    }
                    st
                };
                if (freeze_blocks > 0 || freeze_embeddings)
                    && init_from.is_none()
                    && resume.is_none()
                {
                    eprintln!("[train] note: freezing requested without --init-from; freezing fresh random weights");
                }

                let log_every_u64 = state.cfg.log_every as u64;
                // Training stream RNG. On resume, offset by the step so the run doesn't
                // replay the chunks it already saw.
                let mut rng = SimpleRng::new(0xC0FFEE ^ state.step.wrapping_mul(0x9E37_79B9));
                eprintln!(
                    "[train] row 2 vanilla: d_model={} n_heads={} d_ff={} n_blocks={} embed_scale={:.3} \
                     vocab={vs} train_tokens={} val_tokens={} seq_len={seq_len} steps={} batch={} \
                     wd={} beta2={} clip={} turn_aligned={turn_aligned}",
                    state.cfg.d_model,
                    state.cfg.n_heads,
                    state.cfg.d_ff,
                    state.cfg.n_blocks,
                    state.model.embed_scale,
                    train_ds.n_tokens(),
                    val_ds.n_tokens(),
                    state.cfg.total_steps,
                    grad_accum.max(1),
                    state.cfg.weight_decay,
                    state.cfg.adam_beta2,
                    state.cfg.grad_clip,
                );

                // Fixed validation set, drawn once from its own RNG: every eval sees the
                // same chunks (comparable numbers) and evaluating never perturbs the
                // training sample stream.
                let val_set: Vec<TrainExample> = if val_chunks > 0 && val_every > 0 {
                    let mut vrng = SimpleRng::new(0x7A1_5EED);
                    (0..val_chunks)
                        .map(|_| chunk_to_example(sample_chunk(&val_ds, seq_len, &mut vrng)))
                        .collect()
                } else {
                    Vec::new()
                };
                let keep_best = !no_keep_best && !val_set.is_empty();
                let last_path = checkpoint_out.with_extension("last.json");
                let mut stopper = EarlyStopper::new(patience, 1e-4);
                let mut best_val_bpb: Option<f32> = None;
                let val_bytes: usize = val_set
                    .iter()
                    .map(|ex| {
                        let ids = &ex.full_ids;
                        (0..ids.len().saturating_sub(1))
                            .filter(|&t| ex.loss_mask()[t])
                            .map(|t| {
                                let id = ids[t + 1];
                                if id >= N_SPECIAL && id < vs {
                                    bpe.vocab[id].len()
                                } else {
                                    0
                                }
                            })
                            .sum::<usize>()
                    })
                    .sum();

                let sample_cfg = SampleConfig {
                    temperature: 0.85,
                    top_p: Some(0.9),
                    repetition_penalty: 1.15,
                    max_new_tokens: 48,
                    stop_tokens: vec![special::EOS],
                    seed: Some(12345),
                    ..Default::default()
                };

                let start_step = state.step + 1;
                let mut stopped_early = false;
                for step in start_step..=state.cfg.total_steps {
                    let exs: Vec<TrainExample> = (0..grad_accum.max(1))
                        .map(|_| chunk_to_example(sample_chunk(&train_ds, seq_len, &mut rng)))
                        .collect();
                    let loss = train_step_vanilla_accum(&mut state, &exs);

                    if step % log_every_u64 == 0 || step == start_step {
                        let ppl = (loss.exp()).min(1e6);
                        eprintln!(
                            "[train] step={step} loss={loss:.4} ppl~={ppl:.1} lr={:.2e} gnorm={:.3}",
                            state.current_lr(),
                            state.last_grad_norm
                        );
                    }
                    if sample_every > 0 && step % sample_every == 0 {
                        sample_prompt_vanilla(&state, &bpe, &sample_cfg, step.wrapping_mul(991));
                    }
                    let is_last = step == state.cfg.total_steps;
                    if !val_set.is_empty() && (step % val_every == 0 || is_last) {
                        let (vloss, vsum, _) = eval_vanilla_set(&state, &val_set);
                        let improved = stopper.observe(step, vloss);
                        let bpb = if val_bytes > 0 {
                            Some((vsum / std::f64::consts::LN_2 / val_bytes as f64) as f32)
                        } else {
                            None
                        };
                        eprintln!(
                            "[val] step={step} mean_nll={vloss:.4} ppl~={:.1}{}{}",
                            (vloss.exp()).min(1e6),
                            bpb.map(|b| format!(" bpb≈{b:.4}")).unwrap_or_default(),
                            if improved { "  (best)" } else { "" }
                        );
                        if improved && keep_best {
                            best_val_bpb = bpb;
                            save_vanilla_state(&checkpoint_out, &state)?;
                        }
                        if stopper.should_stop() {
                            eprintln!(
                                "[train] early stop at step {step}: no val improvement for {} evals \
                                 (best {:.4} @ step {})",
                                stopper.patience, stopper.best, stopper.best_step
                            );
                            stopped_early = true;
                            break;
                        }
                    }
                }

                // Final state (+ optimiser moments for --resume).
                let final_path = if keep_best {
                    &last_path
                } else {
                    &checkpoint_out
                };
                save_vanilla_state(final_path, &state)?;
                save_vanilla_optim(final_path, &state)?;
                if keep_best {
                    eprintln!(
                        "[train] best val {:.4} @ step {} → {}  |  final step {} → {}{}",
                        stopper.best,
                        stopper.best_step,
                        checkpoint_out.display(),
                        state.step,
                        last_path.display(),
                        if stopped_early {
                            " (stopped early)"
                        } else {
                            ""
                        }
                    );
                } else {
                    eprintln!("[train] wrote {}", checkpoint_out.display());
                }

                // Specialist card (Arch::Vanilla) so fleets/runtimes can discover it.
                let card_steps = if keep_best {
                    stopper.best_step
                } else {
                    state.step
                };
                write_vanilla_card(
                    &checkpoint_out,
                    &tok,
                    &state.cfg,
                    card_steps,
                    best_val_bpb,
                    subject.clone(),
                    keywords.clone(),
                    init_from.as_deref(),
                )?;
                return Ok(());
            }

            #[cfg(not(feature = "clifford-lm"))]
            {
                return Err(
                    "--clifford requires cargo feature clifford-lm \
                     (default build includes it; slim product: --no-default-features --features vanilla-lm,brain-memory)"
                        .into(),
                );
            }

            #[cfg(feature = "clifford-lm")]
            {
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
                        base.display(),
                        st.cfg.d_model,
                        st.cfg.n_heads,
                        st.cfg.d_ff,
                        st.cfg.n_blocks
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
                    cfg.vanilla = false;
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
                        corpus_semantic_init(
                            &mut st.model,
                            &train_ds.tokens,
                            0x5EED ^ 0xE8E8,
                            semantic_window,
                            1.0,
                        );
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
                "[train] vocab={vs} train_tokens={} val_tokens={} seq_len={seq_len} steps={steps} head_only={head_only} turn_aligned={turn_aligned}",
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
                        let ex = chunk_to_example(sample_chunk(&train_ds, seq_len, &mut rng));
                        train_step_v2_head_only(&mut state, &ex)
                    } else if grad_accum > 1 {
                        let exs: Vec<TrainExample> = (0..grad_accum)
                            .map(|_| chunk_to_example(sample_chunk(&train_ds, seq_len, &mut rng)))
                            .collect();
                        train_step_v2_accum(&mut state, &exs)
                    } else {
                        let ex = chunk_to_example(sample_chunk(&train_ds, seq_len, &mut rng));
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
                            let chunk = sample_chunk(&val_ds, seq_len, &mut rng);
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
            } // #[cfg(feature = "clifford-lm")]
        }
        Commands::LabelTrain {
            dirs,
            out,
            c,
            epochs,
            holdout_frac,
            seed,
        } => {
            let lines = growformer_llm::domain_data::load_labeled_lines(&dirs)?;
            let to_rows = |ls: &[String]| -> Result<Vec<(String, String)>, String> {
                ls.iter()
                    .map(|l| {
                        let v: serde_json::Value =
                            serde_json::from_str(l).map_err(|e| e.to_string())?;
                        let text = v
                            .get("text")
                            .and_then(|x| x.as_str())
                            .unwrap_or("")
                            .to_string();
                        let label = v
                            .get("semantic_intent")
                            .and_then(|x| x.as_str())
                            .or_else(|| {
                                v.get("causal")
                                    .and_then(|c| c.get("causal_type"))
                                    .and_then(|x| x.as_str())
                            })
                            .unwrap_or("")
                            .to_string();
                        Ok((text, label))
                    })
                    .collect()
            };
            let (train_lines, test_lines) = if holdout_frac > 0.0 {
                growformer_llm::domain_data::stratified_split(&lines, holdout_frac, seed)
            } else {
                (lines.iter().map(|(l, _)| l.clone()).collect(), Vec::new())
            };
            let train_rows = to_rows(&train_lines)?;
            let cfg = LabelTrainConfig {
                c,
                epochs,
                ..Default::default()
            };
            let t0 = std::time::Instant::now();
            let model = LabelClassifier::train(&train_rows, &cfg)?;
            eprintln!(
                "[label-train] {} rows, {} labels, trained in {:.1}s",
                train_rows.len(),
                model.labels.len(),
                t0.elapsed().as_secs_f32()
            );
            if !test_lines.is_empty() {
                let test_rows = to_rows(&test_lines)?;
                let (mut fine, mut coarse, mut n_coarse) = (0usize, 0usize, 0usize);
                for (t, l) in &test_rows {
                    let p = model.predict_top(t);
                    fine += (p.label == *l) as usize;
                    if let Some(g) = coarse_sentiment(l) {
                        n_coarse += 1;
                        coarse += (coarse_sentiment(&p.label) == Some(g)) as usize;
                    }
                }
                eprintln!(
                    "[label-train] held-out {}: label acc {:.3}{}",
                    test_rows.len(),
                    fine as f32 / test_rows.len() as f32,
                    if n_coarse > 0 {
                        format!(
                            "  coarse sentiment acc {:.3}",
                            coarse as f32 / n_coarse as f32
                        )
                    } else {
                        String::new()
                    }
                );
            }
            model.save(&out)?;
            eprintln!("[label-train] wrote {}", out.display());
        }
        Commands::LabelPredict { model, prompt, top } => {
            let m = LabelClassifier::load(&model)?;
            for s in m.predict(&prompt).into_iter().take(top.max(1)) {
                println!("{:.3}  {}", s.prob, s.label);
            }
        }
        Commands::JsonlSplit {
            dirs,
            train_out,
            test_out,
            test_frac,
            seed,
        } => {
            let lines = growformer_llm::domain_data::load_labeled_lines(&dirs)?;
            let (train, test) =
                growformer_llm::domain_data::stratified_split(&lines, test_frac, seed);
            for (path, rows) in [(&train_out, &train), (&test_out, &test)] {
                if let Some(parent) = path.parent() {
                    std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
                }
                let mut body = rows.join("\n");
                body.push('\n');
                std::fs::write(path, body).map_err(|e| format!("write {}: {e}", path.display()))?;
            }
            eprintln!(
                "[jsonl-split] {} rows → train {} ({}) / test {} ({})",
                lines.len(),
                train.len(),
                train_out.display(),
                test.len(),
                test_out.display()
            );
        }
        #[cfg(feature = "brain-memory")]
        Commands::BrainEval {
            brain,
            project,
            test,
            label_model,
            label_min_conf,
            hybrid,
            limit,
            json_out,
            verbose,
        } => {
            run_brain_eval(
                &brain,
                project.as_deref(),
                &test,
                label_model.as_deref(),
                label_min_conf,
                hybrid,
                limit,
                json_out.as_deref(),
                verbose,
            )?;
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
            println!(
                "train counts: {}  eval tokens: {}  eval bytes: {}",
                total, n_tok, uni_bytes
            );
            println!(
                "  uniform : ppl {:.1}  {:.3} bits/token  {:.4} bits/byte",
                (vocab as f64),
                uniform_bpt,
                uniform_bpb
            );
            println!(
                "  unigram : ppl {:.1}  {:.3} bits/token  {:.4} bits/byte  (MLE from train shard)",
                uni_ppl, uni_bpt, uni_bpb
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
                        // Log-space CE: exact even when p(target) underflows f32
                        // (the old 1e-12 floor capped a miss at ~40 bits and
                        // made bits/byte look better than it was).
                        let (nats, _) = cross_entropy(&logits[p], target);
                        let bit = nats as f64 / std::f64::consts::LN_2;
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
                    return Err(
                        "no text tokens evaluated (val bin too small or all special)".into(),
                    );
                }

                let model_bpb = model_bits / total_bytes as f64;
                let model_bpt = model_bits / n_pred as f64;
                let uniform_bpb = uniform_bits / total_bytes as f64;
                let unigram_bpb = unigram_bits / total_bytes as f64;
                let unigram_bpt = unigram_bits / n_pred as f64;
                let model_nats_per_tok = model_bpt * std::f64::consts::LN_2;
                let model_ppl = model_nats_per_tok.exp();
                let unigram_ppl = (unigram_bpt * std::f64::consts::LN_2).exp();

                let gz = gzip_bytes(&text_bytes)?;
                let lz = lzma_bytes(&text_bytes)?;
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

                // Record the held-out number on the specialist card, if one exists.
                let card_path = checkpoint.with_extension(CARD_EXT);
                if train_bin.is_some() && card_path.exists() {
                    match SpecialistManifest::load(&card_path) {
                        Ok(mut m) => {
                            m.card.eval_bits_per_byte = Some(model_bpb as f32);
                            m.card.notes = format!(
                                "eval_bits_per_byte: gf-llm eval, {} windows × {} tokens",
                                per_window_bpt.len(),
                                seq_len
                            );
                            m.save(&card_path)?;
                            eprintln!(
                                "[eval] updated {} eval_bits_per_byte={model_bpb:.4}",
                                card_path.display()
                            );
                        }
                        Err(e) => eprintln!("[eval] note: card not updated ({e})"),
                    }
                }

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

            #[cfg(not(feature = "clifford-lm"))]
            {
                return Err(
                    "checkpoint is Clifford (cfg.vanilla=false); rebuild with feature clifford-lm \
                     or train a vanilla checkpoint (default)"
                        .into(),
                );
            }

            #[cfg(feature = "clifford-lm")]
            {
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
                        // Log-space CE: exact even when p(target) underflows f32
                        // (the old 1e-12 floor capped a miss at ~40 bits and
                        // made bits/byte look better than it was).
                        let (nats, _) = cross_entropy(&logits[p], target);
                        let bit = nats as f64 / std::f64::consts::LN_2;
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
                    return Err(
                        "no text tokens evaluated (val bin too small or all special)".into(),
                    );
                }

                let model_bpb = model_bits / total_bytes as f64;
                let model_bpt = model_bits / n_pred as f64;
                let uniform_bpb = uniform_bits / total_bytes as f64;
                let unigram_bpb = unigram_bits / total_bytes as f64;
                let unigram_bpt = unigram_bits / n_pred as f64;
                let model_nats_per_tok = model_bpt * std::f64::consts::LN_2;
                let model_ppl = model_nats_per_tok.exp();
                let unigram_ppl = (unigram_bpt * std::f64::consts::LN_2).exp();

                // Classical baselines on the exact same byte stream.
                let gz = gzip_bytes(&text_bytes)?;
                let lz = lzma_bytes(&text_bytes)?;
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
            } // #[cfg(feature = "clifford-lm")]
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
        #[cfg(feature = "clifford-lm")]
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
                if result.peer_specialists {
                    "PASS"
                } else {
                    "FAIL — imbalanced"
                }
            );
            println!(
                "per-window wins: A={} B={} / {}",
                result.wins_a, result.wins_b, result.eval_n
            );
            println!("best single specialist: {:.4}", result.mean_bpt_best_single);
            println!(
                "oracle per-window min: {:.4}  (gap vs best single: {:.4} bpt)",
                result.mean_bpt_oracle, result.oracle_gap_bpt
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
            println!(
                "cal windows: {}  eval windows: {}",
                result.cal_n, result.eval_n
            );
            println!("routed composite:      {:.4}", result.mean_bpt_routed);
            println!("route A fraction: {:.3}", result.route_a_frac);
            println!(
                "degenerate (constant route): {}",
                if result.degenerate { "YES" } else { "no" }
            );
            let routing_interpretable = result.peer_specialists && result.complementarity_possible;
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
            let peek_cfg = peek_checkpoint_cfg(&checkpoint)?;
            let bpe = BpeTokenizer::load(&tokenizer).map_err(|e| e.to_string())?;
            if peek_cfg.vocab_size != bpe.vocab_size() as usize {
                return Err(format!(
                    "checkpoint vocab_size {} != BPE {}",
                    peek_cfg.vocab_size,
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
                #[cfg(not(feature = "clifford-lm"))]
                {
                    return Err("checkpoint is Clifford; rebuild with feature clifford-lm \
                         or use a vanilla checkpoint"
                        .into());
                }
                #[cfg(feature = "clifford-lm")]
                {
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
            }
            println!();
        }
        Commands::Chat {
            checkpoint,
            tokenizer,
            system,
            max_new_tokens,
            temperature,
            greedy,
            seed,
            repetition_penalty,
            reply_reserve,
            compose,
            #[cfg(feature = "brain-memory")]
            brain,
            #[cfg(feature = "brain-memory")]
            project,
            #[cfg(feature = "brain-memory")]
            hybrid,
            #[cfg(feature = "brain-memory")]
            label_model,
            #[cfg(feature = "brain-memory")]
            label_min_conf,
            message,
        } => {
            run_chat_repl(ChatReplArgs {
                checkpoint: checkpoint.as_deref(),
                tokenizer: tokenizer.as_deref(),
                system: system.as_deref(),
                max_new_tokens,
                temperature,
                greedy,
                seed,
                repetition_penalty,
                reply_reserve,
                compose: &compose,
                #[cfg(feature = "brain-memory")]
                brain: brain.as_deref(),
                #[cfg(feature = "brain-memory")]
                project: project.as_deref(),
                #[cfg(feature = "brain-memory")]
                hybrid,
                #[cfg(feature = "brain-memory")]
                label_model: label_model.as_deref(),
                #[cfg(feature = "brain-memory")]
                label_min_conf,
                message: message.as_deref(),
            })?;
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
            hybrid,
            label_model,
            label_min_conf,
            battery,
            heldout,
            battery_brains,
        } => {
            let labeler = match &label_model {
                Some(p) => Some(LabelClassifier::load(p)?),
                None => None,
            };
            if battery && heldout {
                return Err("use --battery or --heldout, not both".into());
            }
            if battery {
                for case in battery_cases(battery_brains) {
                    println!("========== {} ==========", case.label);
                    let infer_cfg = BrainInferConfig {
                        project: Some(case.project.clone()),
                        inference_toml: None,
                        inference_defaults_toml: None,
                        guardrails_jsonl: None,
                        verbose,
                    };
                    run_brain_infer_case(
                        &case.brain,
                        case.prompt,
                        &infer_cfg,
                        hybrid,
                        labeler.as_ref(),
                        label_min_conf,
                        brain_only,
                        checkpoint.as_deref(),
                        tokenizer.as_deref(),
                        max_new_tokens,
                        temperature,
                        greedy,
                        seed,
                        repetition_penalty,
                    )?;
                    println!();
                }
            } else if heldout {
                for case in heldout_battery_cases()? {
                    println!("========== {} ==========", case.label);
                    let infer_cfg = BrainInferConfig {
                        project: Some(case.project.clone()),
                        inference_toml: None,
                        inference_defaults_toml: None,
                        guardrails_jsonl: None,
                        verbose,
                    };
                    run_brain_infer_case(
                        &case.brain,
                        &case.prompt,
                        &infer_cfg,
                        hybrid,
                        labeler.as_ref(),
                        label_min_conf,
                        brain_only,
                        checkpoint.as_deref(),
                        tokenizer.as_deref(),
                        max_new_tokens,
                        temperature,
                        greedy,
                        seed,
                        repetition_penalty,
                    )?;
                    if let Some(ref topic) = case.expected_topic {
                        println!("expected_topic: {topic}");
                    }
                    println!();
                }
            } else {
                let brain = brain.ok_or("--brain required unless --battery or --heldout")?;
                let prompt = prompt.ok_or("--prompt required unless --battery or --heldout")?;
                let infer_cfg = BrainInferConfig {
                    project,
                    inference_toml,
                    inference_defaults_toml,
                    guardrails_jsonl,
                    verbose,
                };
                run_brain_infer_case(
                    &brain,
                    &prompt,
                    &infer_cfg,
                    hybrid,
                    labeler.as_ref(),
                    label_min_conf,
                    brain_only,
                    checkpoint.as_deref(),
                    tokenizer.as_deref(),
                    max_new_tokens,
                    temperature,
                    greedy,
                    seed,
                    repetition_penalty,
                )?;
            }
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
            force_topic,
            battery,
            battery_brains,
        } => {
            if battery {
                for case in battery_cases(battery_brains) {
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
                let report = if let Some(ref ft) = force_topic {
                    mem.raw_lattice_diagnostic_with_force_topic(&prompt, top_k, Some(ft))?
                } else {
                    mem.raw_lattice_diagnostic(&prompt, top_k)?
                };
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

struct ChatReplArgs<'a> {
    checkpoint: Option<&'a Path>,
    tokenizer: Option<&'a Path>,
    system: Option<&'a str>,
    max_new_tokens: usize,
    temperature: f32,
    greedy: bool,
    seed: Option<u64>,
    repetition_penalty: f32,
    reply_reserve: usize,
    compose: &'a str,
    #[cfg(feature = "brain-memory")]
    brain: Option<&'a Path>,
    #[cfg(feature = "brain-memory")]
    project: Option<&'a Path>,
    #[cfg(feature = "brain-memory")]
    hybrid: bool,
    #[cfg(feature = "brain-memory")]
    label_model: Option<&'a Path>,
    #[cfg(feature = "brain-memory")]
    label_min_conf: f32,
    message: Option<&'a str>,
}

fn run_chat_repl(args: ChatReplArgs<'_>) -> Result<(), String> {
    use growformer_llm::{default_chatbot_system, ChatTranscript};

    let compose = args.compose.trim().to_ascii_lowercase();
    if !matches!(compose.as_str(), "brain" | "polish" | "lm") {
        return Err(format!(
            "unknown --compose {compose:?} (expected brain|polish|lm)"
        ));
    }

    #[cfg(not(feature = "brain-memory"))]
    if matches!(compose.as_str(), "brain" | "polish") {
        return Err("compose=brain|polish requires feature brain-memory".into());
    }

    #[cfg(feature = "brain-memory")]
    if matches!(compose.as_str(), "brain" | "polish") && args.brain.is_none() {
        return Err("--brain is required for compose=brain|polish".into());
    }

    let needs_lm = matches!(compose.as_str(), "lm" | "polish");
    let (state, bpe, ctx_budget) = if needs_lm {
        let checkpoint = args
            .checkpoint
            .ok_or("--checkpoint required for compose=lm|polish")?;
        let tokenizer = args
            .tokenizer
            .ok_or("--tokenizer required for compose=lm|polish")?;
        let peek_cfg = peek_checkpoint_cfg(checkpoint)?;
        if !peek_cfg.vanilla {
            return Err(
                "chat currently supports vanilla checkpoints only (train without --clifford)"
                    .into(),
            );
        }
        let state = load_vanilla_state(checkpoint)?;
        let bpe = BpeTokenizer::load(tokenizer).map_err(|e| e.to_string())?;
        if state.cfg.vocab_size != bpe.vocab_size() as usize {
            return Err(format!(
                "checkpoint vocab {} != BPE {}",
                state.cfg.vocab_size,
                bpe.vocab_size()
            ));
        }
        let ctx_budget = state.cfg.max_seq.saturating_sub(args.reply_reserve.max(1));
        (Some(state), Some(bpe), ctx_budget)
    } else {
        (None, None, 0)
    };

    let sys = args.system.unwrap_or_else(|| default_chatbot_system());
    let mut transcript = ChatTranscript::with_system(sys);

    #[cfg(feature = "brain-memory")]
    let mut brain_rt = if let Some(path) = args.brain {
        let infer_cfg = BrainInferConfig {
            project: args.project.map(|p| p.to_path_buf()),
            ..BrainInferConfig::default()
        };
        Some(BrainMemoryRuntime::from_path_with_config(path, &infer_cfg)?)
    } else {
        None
    };
    #[cfg(feature = "brain-memory")]
    let labeler = match args.label_model {
        Some(p) => {
            let m = LabelClassifier::load(p)?;
            eprintln!(
                "[chat] label model: {} ({} labels)",
                p.display(),
                m.labels.len()
            );
            Some(m)
        }
        None => None,
    };

    let sample_cfg = if args.greedy {
        SampleConfig {
            max_new_tokens: args.max_new_tokens,
            repetition_penalty: args.repetition_penalty,
            seed: args.seed,
            stop_tokens: vec![special::EOS],
            ..SampleConfig::greedy()
        }
    } else {
        SampleConfig {
            temperature: args.temperature,
            max_new_tokens: args.max_new_tokens,
            repetition_penalty: args.repetition_penalty,
            seed: args.seed,
            stop_tokens: vec![special::EOS],
            ..SampleConfig::focused()
        }
    };
    let mut rng = SimpleRng::new(args.seed.unwrap_or(0xC0FFEE));

    eprintln!(
        "[chat] compose={compose} greedy={}  (quit / exit / :q)",
        args.greedy
    );
    if let Some(ref st) = state {
        eprintln!(
            "[chat] lm d_model={} max_seq={} ctx_budget={}",
            st.cfg.d_model, st.cfg.max_seq, ctx_budget
        );
    }
    if compose == "brain" {
        eprintln!("[chat] Path A: assistant reply = retrieved lattice memory (no LM)");
    } else if compose == "polish" {
        eprintln!("[chat] polish: LM rewrites brain memory (facts from Path A)");
    } else {
        eprintln!("[chat] lm mode is experimental — prefer compose=brain for product answers");
    }

    let oneshot = args.message.is_some();
    let mut one_shot = args.message.map(|s| s.to_string());
    loop {
        let user_line = if let Some(m) = one_shot.take() {
            m
        } else {
            eprint!("You> ");
            let _ = std::io::stderr().flush();
            let mut line = String::new();
            match std::io::stdin().read_line(&mut line) {
                Ok(0) => break,
                Ok(_) => {
                    let t = line.trim().to_string();
                    if t.is_empty() {
                        continue;
                    }
                    if matches!(t.as_str(), "quit" | "exit" | ":q" | "/quit") {
                        break;
                    }
                    t
                }
                Err(e) => return Err(format!("stdin: {e}")),
            }
        };

        let mut memory_text: Option<String> = None;
        #[cfg(feature = "brain-memory")]
        if let Some(rt) = brain_rt.as_mut() {
            let res = if let Some(lab) = labeler.as_ref() {
                rt.query_labeled(&user_line, lab, args.label_min_conf)
                    .map(|(q, src, top)| {
                        eprintln!("[chat] label={} p={:.2}", top.label, top.prob);
                        (q, src)
                    })
            } else if args.hybrid {
                rt.query_hybrid(&user_line)
            } else {
                rt.query(&user_line)
                    .map(|q| (q, MemorySource::FullGeneration))
            };
            match res {
                Ok((q, source)) => {
                    eprintln!(
                        "[chat] brain memory_source={} chars={}",
                        source.as_str(),
                        q.memory_text.len()
                    );
                    memory_text = Some(q.memory_text.trim().to_string());
                }
                // Keep the REPL alive on a per-prompt brain failure.
                Err(e) if !oneshot => {
                    eprintln!("[chat] brain query failed: {e}");
                }
                Err(e) => return Err(e),
            }
        }

        let reply = match compose.as_str() {
            "brain" => {
                let mem = memory_text
                    .clone()
                    .filter(|s| !s.is_empty())
                    .unwrap_or_else(|| "No lattice memory retrieved for this prompt.".into());
                print!("Assistant> {mem}\n");
                mem
            }
            "polish" => {
                let state = state.as_ref().unwrap();
                let bpe = bpe.as_ref().unwrap();
                let mem = memory_text.clone().unwrap_or_default();
                if mem.is_empty() {
                    let fallback = "No lattice memory retrieved for this prompt.".to_string();
                    print!("Assistant> {fallback}\n");
                    fallback
                } else {
                    // Facts come from brain; LM only rephrases.
                    let polish_user = format!(
                        "Rewrite in one short helpful sentence for the user (do not invent facts):\n{mem}"
                    );
                    let mut turn = ChatTranscript::with_system(sys);
                    turn.push_user(polish_user);
                    turn.truncate_to_token_budget(bpe, ctx_budget);
                    let prompt = turn.render_for_completion();
                    let generated = generate_vanilla_reply(
                        state,
                        bpe,
                        &prompt,
                        &sample_cfg,
                        &mut rng,
                        ctx_budget,
                    )?;
                    // Only accept the rewrite if it stays faithful to the brain's
                    // answer; a weak LM otherwise replaces a correct reply with noise.
                    let out = if polish_is_faithful(&mem, &generated) {
                        generated
                    } else {
                        if !generated.trim().is_empty() {
                            eprintln!(
                                "[chat] polish rejected (label/content drift) — using brain memory. LM said: {:?}",
                                generated.trim()
                            );
                        }
                        mem
                    };
                    print!("Assistant> {out}\n");
                    out
                }
            }
            _ => {
                // lm (experimental)
                let state = state.as_ref().unwrap();
                let bpe = bpe.as_ref().unwrap();
                let mut user_for_lm = user_line.clone();
                if let Some(ref mem) = memory_text {
                    if !mem.is_empty() {
                        user_for_lm = format!("[brain memory]\n{mem}\n\n{user_line}");
                    }
                }
                transcript.push_user(user_for_lm);
                transcript.truncate_to_token_budget(bpe, ctx_budget);
                let prompt = transcript.render_for_completion();
                print!("Assistant> ");
                let _ = std::io::stdout().flush();
                let out = generate_vanilla_reply_streaming(
                    state,
                    bpe,
                    &prompt,
                    &sample_cfg,
                    &mut rng,
                    state.cfg.max_seq,
                )?;
                println!();
                out
            }
        };

        transcript.push_assistant(reply.trim());
        if oneshot {
            break;
        }
    }
    Ok(())
}

/// Faithfulness gate for `--compose polish`: the rewrite must keep the brain's
/// label (e.g. `NEGATIVE (mild)`) and mostly reuse its words (no invented facts).
fn polish_is_faithful(memory: &str, rewrite: &str) -> bool {
    let out = rewrite.trim();
    if out.chars().count() < 8 {
        return false;
    }
    if let Some((label, _)) = memory.split_once(" — ") {
        let label = label.trim();
        if !label.is_empty()
            && !out
                .to_ascii_lowercase()
                .contains(&label.to_ascii_lowercase())
        {
            return false;
        }
    }
    let words = |t: &str| -> Vec<String> {
        t.split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.len() >= 3)
            .map(|w| w.to_ascii_lowercase())
            .collect()
    };
    let mem_words: std::collections::HashSet<String> = words(memory).into_iter().collect();
    let out_words = words(out);
    if out_words.len() < 2 {
        return false;
    }
    let shared = out_words.iter().filter(|w| mem_words.contains(*w)).count();
    shared >= 2 && shared as f32 / out_words.len() as f32 >= 0.6
}

fn generate_vanilla_reply(
    state: &VanillaModelState,
    bpe: &BpeTokenizer,
    prompt: &str,
    sample_cfg: &SampleConfig,
    rng: &mut SimpleRng,
    max_seq: usize,
) -> Result<String, String> {
    let mut ids: Vec<usize> = vec![special::BOS];
    ids.extend(bpe.encode(prompt).iter().map(|&x| x as usize));
    eprintln!("[chat] prompt_tokens={} (max_seq {})", ids.len(), max_seq);
    if ids.len() >= max_seq {
        return Err(format!(
            "prompt tokens {} ≥ max_seq {max_seq} — shorten system/history or raise --seq-len at train",
            ids.len()
        ));
    }
    let mut reply = String::new();
    for _ in 0..sample_cfg.max_new_tokens {
        if ids.len() >= max_seq {
            break;
        }
        let logits_rows = vanilla_forward_logits(&state.model, &ids, true);
        let Some(last) = logits_rows.last() else {
            break;
        };
        let next = sample_next(last, &ids, sample_cfg, rng);
        if sample_cfg.stop_tokens.contains(&next) {
            break;
        }
        let piece = bpe.decode_one(next as u32);
        let mut candidate = reply.clone();
        candidate.push_str(&piece);
        if let Some(cut) = growformer_llm::role_marker_cut(&candidate) {
            reply = candidate[..cut].to_string();
            break;
        }
        reply.push_str(&piece);
        ids.push(next);
    }
    Ok(reply.trim().to_string())
}

fn generate_vanilla_reply_streaming(
    state: &VanillaModelState,
    bpe: &BpeTokenizer,
    prompt: &str,
    sample_cfg: &SampleConfig,
    rng: &mut SimpleRng,
    max_seq: usize,
) -> Result<String, String> {
    let mut ids: Vec<usize> = vec![special::BOS];
    ids.extend(bpe.encode(prompt).iter().map(|&x| x as usize));
    eprintln!("[chat] prompt_tokens={} (max_seq {})", ids.len(), max_seq);
    if ids.len() >= max_seq {
        return Err(format!("prompt tokens {} ≥ max_seq {max_seq}", ids.len()));
    }
    let mut reply = String::new();
    for _ in 0..sample_cfg.max_new_tokens {
        if ids.len() >= max_seq {
            break;
        }
        let logits_rows = vanilla_forward_logits(&state.model, &ids, true);
        let Some(last) = logits_rows.last() else {
            break;
        };
        let next = sample_next(last, &ids, sample_cfg, rng);
        if sample_cfg.stop_tokens.contains(&next) {
            break;
        }
        let piece = bpe.decode_one(next as u32);
        let mut candidate = reply.clone();
        candidate.push_str(&piece);
        if let Some(cut) = growformer_llm::role_marker_cut(&candidate) {
            reply = candidate[..cut].to_string();
            break;
        }
        print!("{piece}");
        let _ = std::io::stdout().flush();
        reply.push_str(&piece);
        ids.push(next);
    }
    Ok(reply.trim().to_string())
}

#[cfg(feature = "brain-memory")]
fn run_brain_infer_case(
    brain: &Path,
    prompt: &str,
    infer_cfg: &BrainInferConfig,
    hybrid: bool,
    labeler: Option<&LabelClassifier>,
    label_min_conf: f32,
    brain_only: bool,
    checkpoint: Option<&Path>,
    tokenizer: Option<&Path>,
    max_new_tokens: usize,
    temperature: f32,
    greedy: bool,
    seed: Option<u64>,
    repetition_penalty: f32,
) -> Result<(), String> {
    let mut mem = BrainMemoryRuntime::from_path_with_config(brain, infer_cfg)?;
    let info = mem.brain_info();
    let (q, source) = if let Some(lab) = labeler {
        let (q, src, top) = mem.query_labeled(prompt, lab, label_min_conf)?;
        println!("label: {} p={:.3}", top.label, top.prob);
        (q, src)
    } else if hybrid {
        mem.query_hybrid(prompt)?
    } else {
        (mem.query(prompt)?, MemorySource::FullGeneration)
    };
    println!("=== brain memory unit ===");
    println!(
        "brain: {}  groups={}  router={}  gen_envs={}  memory_source={}",
        brain.display(),
        info.num_groups,
        info.has_router,
        info.gen_envs,
        source.as_str()
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
    let lm_prompt = format!(
        "{}{}",
        format_lm_memory_prefix_with_source(&q, source),
        prompt
    );
    eprintln!("[brain-infer] LM prompt prefix: {} chars", lm_prompt.len());

    let peek_cfg = peek_checkpoint_cfg(checkpoint)?;
    let bpe = BpeTokenizer::load(tokenizer).map_err(|e| e.to_string())?;
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
        let state = load_vanilla_state(checkpoint)?;
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
        #[cfg(not(feature = "clifford-lm"))]
        {
            return Err(
                "checkpoint is Clifford; rebuild with feature clifford-lm or use a vanilla checkpoint"
                    .into(),
            );
        }
        #[cfg(feature = "clifford-lm")]
        {
            let state = load_lm_state(checkpoint)?;
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
    }
    println!();
    Ok(())
}

/// Hedge replies the brain gives when its grounding gate declines.
#[cfg(feature = "brain-memory")]
fn is_hedge_reply(reply: &str) -> bool {
    let r = reply.to_ascii_lowercase();
    r.contains("don't have enough information")
        || r.contains("non-obvious from surface text")
        || r.contains("no lattice memory retrieved")
        || r.contains("no stored explanation")
}

#[cfg(feature = "brain-memory")]
#[allow(clippy::too_many_arguments)]
fn run_brain_eval(
    brain: &Path,
    project: Option<&Path>,
    test: &Path,
    label_model: Option<&Path>,
    label_min_conf: f32,
    hybrid: bool,
    limit: usize,
    json_out: Option<&Path>,
    verbose: bool,
) -> Result<(), String> {
    let raw = std::fs::read_to_string(test).map_err(|e| format!("read {}: {e}", test.display()))?;
    let mut rows: Vec<(String, String)> = Vec::new();
    for line in raw.lines().filter(|l| !l.trim().is_empty()) {
        let v: serde_json::Value = serde_json::from_str(line).map_err(|e| e.to_string())?;
        let text = v
            .get("text")
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .to_string();
        let label = v
            .get("semantic_intent")
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .to_string();
        rows.push((text, label));
    }
    if limit > 0 && rows.len() > limit {
        rows.truncate(limit);
    }
    let labeler = match label_model {
        Some(p) => Some(LabelClassifier::load(p)?),
        None => None,
    };
    let infer_cfg = BrainInferConfig {
        project: project.map(|p| p.to_path_buf()),
        ..BrainInferConfig::default()
    };
    let mut mem = BrainMemoryRuntime::from_path_with_config(brain, &infer_cfg)?;
    let mode = if labeler.is_some() {
        "label+brain"
    } else if hybrid {
        "brain (hybrid)"
    } else {
        "brain"
    };
    eprintln!("[brain-eval] {} rows, mode={mode}", rows.len());

    #[derive(Default)]
    struct Tally {
        n: usize,
        coarse_n: usize,
        coarse_ok: usize,
        label_ok: usize,
        hedged: usize,
        errors: usize,
    }
    let mut t = Tally::default();
    let mut per_row = Vec::new();
    let t0 = std::time::Instant::now();
    for (text, gold) in &rows {
        t.n += 1;
        let res = if let Some(lab) = labeler.as_ref() {
            mem.query_labeled(text, lab, label_min_conf)
                .map(|(q, src, top)| (q, src, Some(top)))
        } else if hybrid {
            mem.query_hybrid(text).map(|(q, s)| (q, s, None))
        } else {
            mem.query(text)
                .map(|q| (q, MemorySource::FullGeneration, None))
        };
        let (reply, source, predicted) = match res {
            Ok((q, src, top)) => (q.memory_text.trim().to_string(), src.as_str(), top),
            Err(e) => {
                t.errors += 1;
                (format!("ERROR: {e}"), "error", None)
            }
        };
        let hedged = is_hedge_reply(&reply);
        t.hedged += hedged as usize;
        if let Some(p) = &predicted {
            t.label_ok += (p.label == *gold) as usize;
        }
        let gold_c = coarse_sentiment(gold);
        let pred_c = coarse_sentiment(&reply_label_head(&reply));
        if let Some(g) = gold_c {
            t.coarse_n += 1;
            t.coarse_ok += (pred_c == Some(g)) as usize;
        }
        if verbose {
            println!(
                "[{}] gold={} ({}) pred={} {}\n    {}",
                source,
                gold,
                gold_c.unwrap_or("-"),
                pred_c.unwrap_or("-"),
                if pred_c.is_some() && pred_c == gold_c {
                    "✓"
                } else {
                    "✗"
                },
                reply.chars().take(140).collect::<String>()
            );
        }
        per_row.push(serde_json::json!({
            "text": text, "gold": gold, "gold_coarse": gold_c, "reply": reply,
            "reply_coarse": pred_c, "source": source, "hedged": hedged,
            "predicted_label": predicted.as_ref().map(|p| p.label.clone()),
            "predicted_prob": predicted.as_ref().map(|p| p.prob),
        }));
    }
    let pct = |a: usize, b: usize| if b == 0 { 0.0 } else { a as f64 / b as f64 };
    let summary = serde_json::json!({
        "mode": mode,
        "rows": t.n,
        "coarse_sentiment_acc": pct(t.coarse_ok, t.coarse_n),
        "coarse_rows": t.coarse_n,
        "label_acc": labeler.as_ref().map(|_| pct(t.label_ok, t.n)),
        "hedge_rate": pct(t.hedged, t.n),
        "errors": t.errors,
        "seconds": t0.elapsed().as_secs_f64(),
    });
    println!(
        "=== brain-eval ({mode}) ===\nrows {}  coarse sentiment acc {:.3} ({}/{})  hedge rate {:.3}{}  errors {}",
        t.n,
        pct(t.coarse_ok, t.coarse_n),
        t.coarse_ok,
        t.coarse_n,
        pct(t.hedged, t.n),
        if labeler.is_some() {
            format!("  label acc {:.3}", pct(t.label_ok, t.n))
        } else {
            String::new()
        },
        t.errors
    );
    if let Some(p) = json_out {
        let doc = serde_json::json!({ "summary": summary, "rows": per_row });
        std::fs::write(
            p,
            serde_json::to_string_pretty(&doc).map_err(|e| e.to_string())?,
        )
        .map_err(|e| format!("write {}: {e}", p.display()))?;
        eprintln!("[brain-eval] wrote {}", p.display());
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

#[cfg(test)]
mod polish_tests {
    use super::polish_is_faithful;

    #[test]
    fn keeps_faithful_rewrite() {
        let mem = "NEGATIVE (mild) — ETF timeline slip plus overnight BTC selloff; regulatory delay driving price weakness.";
        assert!(polish_is_faithful(
            mem,
            "NEGATIVE (mild) — the ETF delay and overnight BTC selloff point to price weakness."
        ));
    }

    #[test]
    fn rejects_noise_and_label_flip() {
        let mem = "NEGATIVE (mild) — ETF timeline slip plus overnight BTC selloff.";
        assert!(!polish_is_faithful(mem, "--ered s s"));
        assert!(!polish_is_faithful(
            mem,
            "POSITIVE (strong) — ETF timeline slip plus overnight BTC selloff."
        ));
        assert!(!polish_is_faithful(
            mem,
            "NEGATIVE (mild) — Tight be sider deal: misconfirsted for a ling"
        ));
    }
}
