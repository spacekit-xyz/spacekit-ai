//! Open-model backend (feature `candle`): scores each question's answer
//! tokens with **one forward pass** — no text generation.
//!
//! The prompt is `prefix(state)` + `suffix(question)`. The prefix is run once;
//! every question continues from a copy of that KV cache (Llama, Qwen3), so
//! asking N questions about a long state costs one state pass plus N short
//! suffixes. Qwen2 GGUFs (whose candle model can't be cloned) recompute the
//! prefix per question.
//!
//! Weights are GGUF (any quantisation candle supports); the tokenizer is the
//! model's Hugging Face `tokenizer.json`. Everything loads from bytes, so the
//! same code runs natively and in `wasm32-unknown-unknown`.

use std::io::Cursor;

use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use candle_transformers::models::{quantized_llama, quantized_qwen2};

use crate::qwen3 as quantized_qwen3;
use tokenizers::Tokenizer;

use crate::brain::{LmArch, LmConfig};
use crate::decide::OptionScorer;
use crate::prompt::DEFAULT_SYSTEM;
use crate::schema::Question;
use crate::state::truncate_middle;

enum Model {
    Llama(quantized_llama::ModelWeights),
    Qwen2(quantized_qwen2::ModelWeights),
    Qwen3(quantized_qwen3::ModelWeights),
}

impl Model {
    /// Shares the weights (candle tensors are reference-counted); only the KV
    /// cache is per copy. candle's quantized Qwen2 isn't `Clone`.
    fn share(&self) -> Option<Model> {
        match self {
            Model::Llama(m) => Some(Model::Llama(m.clone())),
            Model::Qwen3(m) => Some(Model::Qwen3(m.clone())),
            Model::Qwen2(_) => None,
        }
    }
}

fn ce(e: candle_core::Error) -> String {
    e.to_string()
}

/// Candle-backed [`OptionScorer`].
pub struct CandleScorer {
    model: Model,
    tokenizer: Tokenizer,
    device: Device,
    cfg: LmConfig,
    system: String,
    /// Share of the full next-token probability mass that landed on the
    /// allowed answer tokens, per question of the last call (≈1 when the model
    /// follows the format; low values mean the prompt/template is off).
    pub last_answer_mass: Vec<f32>,
}

// Command-line apps on macOS must link CoreGraphics, or Metal's device list
// (`MTLCreateSystemDefaultDevice`) comes back empty and candle panics in `new_metal(0)`.
#[cfg(all(feature = "metal", target_os = "macos"))]
#[link(name = "CoreGraphics", kind = "framework")]
extern "C" {}

/// Run `f`, turning a panic into `None` without printing it.
#[cfg(any(feature = "metal", feature = "cuda"))]
fn quiet<T>(f: impl FnOnce() -> T + std::panic::UnwindSafe) -> Option<T> {
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let r = std::panic::catch_unwind(f).ok();
    std::panic::set_hook(hook);
    r
}

/// Pick the compute device: CUDA / Metal when compiled in and available, else
/// CPU. `GF_S1_DEVICE=cpu|metal|cuda` overrides the choice.
pub fn default_device() -> Device {
    let want = std::env::var("GF_S1_DEVICE")
        .unwrap_or_default()
        .to_ascii_lowercase();
    if want == "cpu" {
        return Device::Cpu;
    }
    #[cfg(feature = "cuda")]
    if want.is_empty() || want == "cuda" {
        if let Some(Ok(d)) = quiet(|| Device::new_cuda(0)) {
            return d;
        }
    }
    #[cfg(feature = "metal")]
    if want.is_empty() || want == "metal" {
        if let Some(Ok(d)) = quiet(|| Device::new_metal(0)) {
            return d;
        }
    }
    if !want.is_empty() {
        eprintln!("[gf-system1] device {want:?} unavailable in this build; using CPU");
    } else if cfg!(any(feature = "metal", feature = "cuda")) {
        eprintln!("[gf-system1] no GPU device found; using CPU");
    }
    Device::Cpu
}

fn device_name(d: &Device) -> &'static str {
    match d {
        Device::Cpu => "cpu",
        Device::Cuda(_) => "cuda",
        Device::Metal(_) => "metal",
    }
}

impl CandleScorer {
    /// Load from in-memory GGUF + `tokenizer.json` (works in wasm).
    pub fn from_bytes(cfg: LmConfig, gguf: &[u8], tokenizer_json: &[u8]) -> Result<Self, String> {
        Self::from_reader(cfg, &mut Cursor::new(gguf), tokenizer_json)
    }

    /// Load from any seekable GGUF reader + `tokenizer.json` bytes.
    pub fn from_reader<R: std::io::Read + std::io::Seek>(
        cfg: LmConfig,
        reader: &mut R,
        tokenizer_json: &[u8],
    ) -> Result<Self, String> {
        let device = default_device();
        let content = gguf_file::Content::read(reader).map_err(|e| format!("gguf: {e}"))?;
        let model = match cfg.arch {
            LmArch::Llama => Model::Llama(
                quantized_llama::ModelWeights::from_gguf(content, reader, &device).map_err(ce)?,
            ),
            LmArch::Qwen2 => Model::Qwen2(
                quantized_qwen2::ModelWeights::from_gguf(content, reader, &device).map_err(ce)?,
            ),
            LmArch::Qwen3 => Model::Qwen3(
                quantized_qwen3::ModelWeights::from_gguf(content, reader, &device).map_err(ce)?,
            ),
        };
        let tokenizer =
            Tokenizer::from_bytes(tokenizer_json).map_err(|e| format!("tokenizer: {e}"))?;
        let system = cfg
            .system
            .clone()
            .unwrap_or_else(|| DEFAULT_SYSTEM.to_string());
        Ok(Self {
            model,
            tokenizer,
            device,
            cfg,
            system,
            last_answer_mass: Vec::new(),
        })
    }

    /// A second scorer on the same loaded weights (own KV cache), optionally
    /// with different prompt settings (`cfg` must name the same model and
    /// tokenizer). `None` for Qwen2, whose candle model can't share weights.
    pub fn fork_with(&self, cfg: Option<LmConfig>) -> Option<Self> {
        let cfg = cfg.unwrap_or_else(|| self.cfg.clone());
        let system = cfg
            .system
            .clone()
            .unwrap_or_else(|| DEFAULT_SYSTEM.to_string());
        Some(Self {
            model: self.model.share()?,
            tokenizer: self.tokenizer.clone(),
            device: self.device.clone(),
            cfg,
            system,
            last_answer_mass: Vec::new(),
        })
    }

    /// Load the files named in `cfg`, resolved against `base_dir`. The GGUF is
    /// read straight from disk (not buffered whole), so peak memory stays near
    /// the model's own size.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn from_files(cfg: LmConfig, base_dir: &std::path::Path) -> Result<Self, String> {
        let resolve = |p: &str| {
            let pb = std::path::PathBuf::from(p);
            if pb.is_absolute() {
                pb
            } else {
                base_dir.join(pb)
            }
        };
        let (m, t) = (resolve(&cfg.model), resolve(&cfg.tokenizer));
        let tok = std::fs::read(&t).map_err(|e| format!("read {}: {e}", t.display()))?;
        let file = std::fs::File::open(&m).map_err(|e| format!("open {}: {e}", m.display()))?;
        let mut reader = std::io::BufReader::new(file);
        Self::from_reader(cfg, &mut reader, &tok).map_err(|e| format!("{}: {e}", m.display()))
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>, String> {
        Ok(self
            .tokenizer
            .encode(text, false)
            .map_err(|e| e.to_string())?
            .get_ids()
            .to_vec())
    }

    /// Token ids that mean each answer: the answer's first token, plus the
    /// leading-space variant when it is a distinct token. Ids shared by more
    /// than one answer are dropped (they can't discriminate).
    fn answer_token_ids(&self, answers: &[String]) -> Result<Vec<Vec<u32>>, String> {
        let mut per: Vec<Vec<u32>> = Vec::with_capacity(answers.len());
        for a in answers {
            let mut ids = Vec::new();
            for variant in [a.clone(), format!(" {a}")] {
                if let Some(&id) = self.encode(&variant)?.first() {
                    if !ids.contains(&id) {
                        ids.push(id);
                    }
                }
            }
            per.push(ids);
        }
        let mut counts = std::collections::HashMap::new();
        for ids in &per {
            for id in ids {
                *counts.entry(*id).or_insert(0usize) += 1;
            }
        }
        for (ids, a) in per.iter_mut().zip(answers) {
            ids.retain(|id| counts[id] == 1);
            if ids.is_empty() {
                return Err(format!(
                    "answer {a:?} has no distinct token in this tokenizer"
                ));
            }
        }
        Ok(per)
    }

    fn input(&self, ids: &[u32]) -> Result<Tensor, String> {
        Tensor::new(ids, &self.device)
            .and_then(|t| t.unsqueeze(0))
            .map_err(ce)
    }

    /// Next-token logits after `prefix` + each suffix.
    fn next_logits(
        &mut self,
        prefix: &[u32],
        suffixes: &[Vec<u32>],
    ) -> Result<Vec<Vec<f32>>, String> {
        let to_vec = |t: Tensor| -> Result<Vec<f32>, String> {
            t.squeeze(0)
                .and_then(|t| t.to_dtype(candle_core::DType::F32))
                .and_then(|t| t.to_vec1::<f32>())
                .map_err(ce)
        };
        let mut out = Vec::with_capacity(suffixes.len());
        match &mut self.model {
            Model::Llama(m) => {
                let x = Tensor::new(prefix, &self.device)
                    .and_then(|t| t.unsqueeze(0))
                    .map_err(ce)?;
                m.forward(&x, 0).map_err(ce)?;
                for s in suffixes {
                    // candle's quantized Llama masks only same-length blocks, so
                    // continue the cached prefix one token at a time.
                    let mut mm = m.clone();
                    let mut last = None;
                    for (j, &tok) in s.iter().enumerate() {
                        let x = Tensor::new(&[tok], &self.device)
                            .and_then(|t| t.unsqueeze(0))
                            .map_err(ce)?;
                        last = Some(mm.forward(&x, prefix.len() + j).map_err(ce)?);
                    }
                    out.push(to_vec(last.ok_or("empty question suffix")?)?);
                }
            }
            Model::Qwen3(m) => {
                m.clear_kv_cache();
                let x = Tensor::new(prefix, &self.device)
                    .and_then(|t| t.unsqueeze(0))
                    .map_err(ce)?;
                m.forward(&x, 0).map_err(ce)?;
                for s in suffixes {
                    let mut mm = m.clone();
                    let x = Tensor::new(s.as_slice(), &self.device)
                        .and_then(|t| t.unsqueeze(0))
                        .map_err(ce)?;
                    out.push(to_vec(mm.forward(&x, prefix.len()).map_err(ce)?)?);
                }
                m.clear_kv_cache();
            }
            Model::Qwen2(m) => {
                for s in suffixes {
                    let mut ids = prefix.to_vec();
                    ids.extend_from_slice(s);
                    let x = Tensor::new(ids.as_slice(), &self.device)
                        .and_then(|t| t.unsqueeze(0))
                        .map_err(ce)?;
                    out.push(to_vec(m.forward(&x, 0).map_err(ce)?)?);
                }
            }
        }
        Ok(out)
    }

    /// Full recompute (no prefix reuse) — used by tests to check the cache path.
    #[doc(hidden)]
    pub fn next_logits_uncached(&mut self, ids: &[u32]) -> Result<Vec<f32>, String> {
        let x = self.input(ids)?;
        let t = match &mut self.model {
            Model::Llama(m) => m.clone().forward(&x, 0),
            Model::Qwen2(m) => m.forward(&x, 0),
            Model::Qwen3(m) => {
                let mut mm = m.clone();
                mm.clear_kv_cache();
                mm.forward(&x, 0)
            }
        }
        .map_err(ce)?;
        t.squeeze(0).and_then(|t| t.to_vec1::<f32>()).map_err(ce)
    }

    /// Probe features for a state (Qwen3 only): the mean of the residual
    /// stream after layer `round(layer_frac * (layers - 1))` over the state's
    /// tokens, concatenated with the final-normed hidden state of the last
    /// prompt token. One forward pass, no LM head.
    fn embed_state(&mut self, state_text: &str, layer_frac: f32) -> Result<Vec<f32>, String> {
        let state = truncate_middle(state_text, self.cfg.max_state_tokens.saturating_mul(4));
        let ids = self.encode(&self.cfg.template.prefix(&self.system, &state))?;
        if ids.is_empty() {
            return Err("empty prompt".into());
        }
        // Tokens before the state (system prompt, chat markers) are the same
        // for every state; average only from where the state starts.
        let marker = '\u{1}';
        let head = self
            .cfg
            .template
            .prefix(&self.system, &marker.to_string())
            .split(marker)
            .next()
            .unwrap_or_default()
            .to_string();
        let start = self.encode(&head)?.len().min(ids.len() - 1);
        let x = self.input(&ids)?;
        let Model::Qwen3(m) = &mut self.model else {
            return Err(format!(
                "probe heads need a Qwen3 model for now (this brain uses {:?})",
                self.cfg.arch
            ));
        };
        let tap_layer =
            ((layer_frac.clamp(0.0, 1.0) * (m.num_layers() - 1) as f32).round()) as usize;
        m.clear_kv_cache();
        let r = m.forward_hidden(&x, 0, tap_layer);
        m.clear_kv_cache();
        let (tap, last) = r.map_err(ce)?;
        let l = ids.len();
        let to_vec = |t: Tensor| -> Result<Vec<f32>, String> {
            t.flatten_all()
                .and_then(|t| t.to_dtype(candle_core::DType::F32))
                .and_then(|t| t.to_vec1::<f32>())
                .map_err(ce)
        };
        let mean = tap
            .narrow(1, start, l - start)
            .and_then(|t| t.mean(1))
            .map_err(ce)?;
        let mut v = to_vec(mean)?;
        v.extend(to_vec(last.narrow(1, l - 1, 1).map_err(ce)?)?);
        Ok(v)
    }

    #[doc(hidden)]
    pub fn encode_for_test(&self, text: &str) -> Result<Vec<u32>, String> {
        self.encode(text)
    }
}

fn logsumexp(xs: &[f32]) -> f32 {
    let m = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    m + xs.iter().map(|x| (x - m).exp()).sum::<f32>().ln()
}

impl OptionScorer for CandleScorer {
    fn describe(&self) -> String {
        format!(
            "candle {:?} on {} ({})",
            self.cfg.arch,
            device_name(&self.device),
            self.cfg.model
        )
    }

    fn embed(&mut self, state_text: &str, layer_frac: f32) -> Result<Vec<f32>, String> {
        self.embed_state(state_text, layer_frac)
    }

    fn fork(&self) -> Option<Box<dyn OptionScorer + Send>> {
        Some(Box::new(self.fork_with(None)?))
    }

    fn model_key(&self) -> String {
        let file = std::path::Path::new(&self.cfg.model)
            .file_name()
            .map(|f| f.to_string_lossy().into_owned())
            .unwrap_or_else(|| self.cfg.model.clone());
        format!("{:?}:{file}:{:?}", self.cfg.arch, self.cfg.template)
    }

    fn score(
        &mut self,
        state_text: &str,
        questions: &[&Question],
    ) -> Result<Vec<Vec<f32>>, String> {
        if questions.is_empty() {
            return Ok(Vec::new());
        }
        // ~4 characters per token keeps the state inside the token budget.
        let state = truncate_middle(state_text, self.cfg.max_state_tokens.saturating_mul(4));
        let prefix = self.encode(&self.cfg.template.prefix(&self.system, &state))?;
        let mut suffixes = Vec::with_capacity(questions.len());
        let mut answer_ids = Vec::with_capacity(questions.len());
        for q in questions {
            let qp = self.cfg.template.question(&self.system, q);
            suffixes.push(self.encode(&qp.suffix)?);
            answer_ids.push(self.answer_token_ids(&qp.answers)?);
        }
        if suffixes.iter().any(|s| s.is_empty()) {
            return Err("template produced an empty question suffix".into());
        }
        let logits = self.next_logits(&prefix, &suffixes)?;
        self.last_answer_mass.clear();
        let mut out = Vec::with_capacity(questions.len());
        for (z, ids) in logits.iter().zip(&answer_ids) {
            let per: Vec<f32> = ids
                .iter()
                .map(|v| {
                    let xs: Vec<f32> = v.iter().map(|&i| z[i as usize]).collect();
                    logsumexp(&xs)
                })
                .collect();
            let lz = logsumexp(z);
            self.last_answer_mass
                .push(per.iter().map(|x| (x - lz).exp()).sum());
            out.push(per);
        }
        Ok(out)
    }
}

/// Tiny random GGUF models + a byte-level tokenizer, for smoke-testing the
/// pipeline without downloading a model (`cargo run --example tiny_model`).
/// The outputs are meaningless; only the plumbing is real.
#[doc(hidden)]
pub mod testkit {
    use candle_core::quantized::{gguf_file, GgmlDType, QTensor};
    use candle_core::{Device, Tensor};

    pub const HIDDEN: usize = 32;
    pub const HEADS: usize = 4;
    pub const KV_HEADS: usize = 2;
    pub const FF: usize = 64;
    pub const LAYERS: usize = 2;

    /// Byte-level BPE tokenizer with no merges + the chat special tokens.
    pub fn tokenizer_json() -> Vec<u8> {
        use tokenizers::pre_tokenizers::byte_level::ByteLevel;
        let mut chars: Vec<String> = ByteLevel::alphabet()
            .into_iter()
            .map(|c| c.to_string())
            .collect();
        chars.sort();
        let vocab: serde_json::Map<String, serde_json::Value> = chars
            .iter()
            .enumerate()
            .map(|(i, c)| (c.clone(), serde_json::json!(i)))
            .collect();
        let specials = [
            "<|im_start|>",
            "<|im_end|>",
            "<think>",
            "</think>",
            "<|begin_of_text|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|eot_id|>",
        ];
        let added: Vec<serde_json::Value> = specials
            .iter()
            .enumerate()
            .map(|(i, t)| {
                serde_json::json!({"id": 256 + i, "content": t, "single_word": false, "lstrip": false,
                                   "rstrip": false, "normalized": false, "special": true})
            })
            .collect();
        let bl = serde_json::json!({"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true});
        serde_json::to_vec(&serde_json::json!({
            "version": "1.0", "truncation": null, "padding": null, "added_tokens": added,
            "normalizer": null, "pre_tokenizer": bl, "post_processor": null, "decoder": bl,
            "model": {"type": "BPE", "dropout": null, "unk_token": null, "continuing_subword_prefix": null,
                      "end_of_word_suffix": null, "fuse_unk": false, "byte_fallback": false,
                      "vocab": vocab, "merges": []}
        }))
        .unwrap()
    }

    pub fn vocab_size() -> usize {
        256 + 8
    }

    fn q(t: Tensor) -> QTensor {
        QTensor::quantize(&t, GgmlDType::F32).unwrap()
    }

    fn rnd(shape: &[usize], seed: u64) -> Tensor {
        // Deterministic pseudo-random weights (no RNG dependency).
        let n: usize = shape.iter().product();
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let v: Vec<f32> = (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.4
            })
            .collect();
        Tensor::from_vec(v, shape, &Device::Cpu).unwrap()
    }

    fn ones(n: usize) -> Tensor {
        Tensor::ones(n, candle_core::DType::F32, &Device::Cpu).unwrap()
    }

    /// Write a tiny random GGUF for `arch` ("llama" | "qwen2" | "qwen3").
    pub fn gguf(arch: &str) -> Vec<u8> {
        use gguf_file::Value as V;
        let hd = HIDDEN / HEADS;
        let vocab = vocab_size();
        let mut md: Vec<(String, V)> = vec![
            (format!("{arch}.attention.head_count"), V::U32(HEADS as u32)),
            (
                format!("{arch}.attention.head_count_kv"),
                V::U32(KV_HEADS as u32),
            ),
            (format!("{arch}.block_count"), V::U32(LAYERS as u32)),
            (format!("{arch}.embedding_length"), V::U32(HIDDEN as u32)),
            (
                format!("{arch}.attention.layer_norm_rms_epsilon"),
                V::F32(1e-5),
            ),
            (format!("{arch}.rope.freq_base"), V::F32(10000.0)),
            (format!("{arch}.context_length"), V::U32(4096)),
        ];
        if arch == "llama" {
            md.push(("llama.rope.dimension_count".into(), V::U32(hd as u32)));
        }
        if arch == "qwen3" {
            md.push(("qwen3.attention.key_length".into(), V::U32(hd as u32)));
        }
        let mut tensors: Vec<(String, QTensor)> = vec![
            ("token_embd.weight".into(), q(rnd(&[vocab, HIDDEN], 1))),
            ("output_norm.weight".into(), q(ones(HIDDEN))),
            ("output.weight".into(), q(rnd(&[vocab, HIDDEN], 2))),
        ];
        for l in 0..LAYERS {
            let p = format!("blk.{l}");
            let s = 100 + l as u64 * 20;
            tensors.push((
                format!("{p}.attn_q.weight"),
                q(rnd(&[HEADS * hd, HIDDEN], s)),
            ));
            tensors.push((
                format!("{p}.attn_k.weight"),
                q(rnd(&[KV_HEADS * hd, HIDDEN], s + 1)),
            ));
            tensors.push((
                format!("{p}.attn_v.weight"),
                q(rnd(&[KV_HEADS * hd, HIDDEN], s + 2)),
            ));
            tensors.push((
                format!("{p}.attn_output.weight"),
                q(rnd(&[HIDDEN, HEADS * hd], s + 3)),
            ));
            tensors.push((format!("{p}.ffn_gate.weight"), q(rnd(&[FF, HIDDEN], s + 4))));
            tensors.push((format!("{p}.ffn_up.weight"), q(rnd(&[FF, HIDDEN], s + 5))));
            tensors.push((format!("{p}.ffn_down.weight"), q(rnd(&[HIDDEN, FF], s + 6))));
            tensors.push((format!("{p}.attn_norm.weight"), q(ones(HIDDEN))));
            tensors.push((format!("{p}.ffn_norm.weight"), q(ones(HIDDEN))));
            if arch == "qwen2" {
                tensors.push((format!("{p}.attn_q.bias"), q(rnd(&[HEADS * hd], s + 7))));
                tensors.push((format!("{p}.attn_k.bias"), q(rnd(&[KV_HEADS * hd], s + 8))));
                tensors.push((format!("{p}.attn_v.bias"), q(rnd(&[KV_HEADS * hd], s + 9))));
            }
            if arch == "qwen3" {
                tensors.push((format!("{p}.attn_q_norm.weight"), q(ones(hd))));
                tensors.push((format!("{p}.attn_k_norm.weight"), q(ones(hd))));
            }
        }
        let md_ref: Vec<(&str, &V)> = md.iter().map(|(k, v)| (k.as_str(), v)).collect();
        let t_ref: Vec<(&str, &QTensor)> = tensors.iter().map(|(k, v)| (k.as_str(), v)).collect();
        let mut buf = std::io::Cursor::new(Vec::new());
        gguf_file::write(&mut buf, &md_ref, &t_ref).unwrap();
        buf.into_inner()
    }
}

#[cfg(test)]
mod tests {
    use super::testkit::*;
    use super::*;
    use crate::prompt::TemplatePreset;

    fn scorer(arch: LmArch, name: &str, template: TemplatePreset) -> CandleScorer {
        let cfg = LmConfig {
            arch,
            model: format!("{name}.gguf"),
            tokenizer: "tokenizer.json".into(),
            template,
            system: None,
            max_state_tokens: 512,
        };
        CandleScorer::from_bytes(cfg, &gguf(name), &tokenizer_json()).unwrap()
    }

    fn questions() -> Vec<Question> {
        vec![
            Question::choice("route", "Which team?", &["Billing", "Tech", "Sales"]),
            Question::score("urgency", "How urgent?", 0, 4),
            Question::noul("spam", "Is this spam?"),
        ]
    }

    /// The cached-prefix path must give the same answer logits as recomputing
    /// prefix + suffix from scratch.
    #[test]
    fn prefix_reuse_matches_full_recompute() {
        for (arch, name, tpl) in [
            (LmArch::Llama, "llama", TemplatePreset::Llama3),
            (LmArch::Qwen3, "qwen3", TemplatePreset::ChatmlNothink),
            (LmArch::Qwen2, "qwen2", TemplatePreset::Chatml),
        ] {
            let mut s = scorer(arch, name, tpl.clone());
            let qs = questions();
            let refs: Vec<&Question> = qs.iter().collect();
            let state = "ticket: I was charged twice\npriority: high";
            let scored = s.score(state, &refs).unwrap();
            assert_eq!(scored.len(), 3);
            assert_eq!(scored[0].len(), 3);
            assert_eq!(scored[1].len(), 5);
            assert_eq!(scored[2].len(), 2);
            // Recompute the first question without the cache.
            let prefix = s
                .encode_for_test(&tpl.prefix(DEFAULT_SYSTEM, state))
                .unwrap();
            let qp = tpl.question(DEFAULT_SYSTEM, &qs[0]);
            let mut ids = prefix.clone();
            ids.extend(s.encode_for_test(&qp.suffix).unwrap());
            let full = s.next_logits_uncached(&ids).unwrap();
            let ans = s.answer_token_ids(&qp.answers).unwrap();
            for (k, v) in ans.iter().enumerate() {
                let xs: Vec<f32> = v.iter().map(|&i| full[i as usize]).collect();
                let want = logsumexp(&xs);
                assert!(
                    (want - scored[0][k]).abs() < 1e-3,
                    "{name}: outcome {k}: cached {} vs full {want}",
                    scored[0][k]
                );
            }
            // Deterministic across calls.
            assert_eq!(
                s.score(state, &refs).unwrap(),
                scored,
                "{name} not deterministic"
            );
            assert_eq!(s.last_answer_mass.len(), 3);
        }
    }

    #[test]
    fn qwen3_embeddings_feed_probe_heads() {
        let mut s = scorer(LmArch::Qwen3, "qwen3", TemplatePreset::ChatmlNothink);
        let a = s.embed("I was charged twice", 0.5).unwrap();
        assert_eq!(a.len(), 2 * super::testkit::HIDDEN);
        assert_eq!(
            a,
            s.embed("I was charged twice", 0.5).unwrap(),
            "deterministic"
        );
        assert_ne!(a, s.embed("the app crashes", 0.5).unwrap());
        // Embedding must not disturb scoring (KV cache is cleared around it).
        let qs = questions();
        let refs: Vec<&Question> = qs.iter().collect();
        let before = s.score("x", &refs).unwrap();
        s.embed("y", 1.0).unwrap();
        assert_eq!(before, s.score("x", &refs).unwrap());
        // Other architectures say so instead of failing obscurely.
        let mut l = scorer(LmArch::Llama, "llama", TemplatePreset::Llama3);
        assert!(l.embed("x", 0.5).unwrap_err().contains("Qwen3"));
    }

    #[test]
    fn decider_routes_unknown_questions_to_the_model() {
        let b = crate::brain::System1Brain::new("lm-only", crate::schema::QuestionSet::default())
            .unwrap();
        let mut d = crate::decide::Decider::new(b).with_scorer(Box::new(scorer(
            LmArch::Qwen3,
            "qwen3",
            TemplatePreset::ChatmlNothink,
        )));
        let out = d
            .decide_json(
                r#"{"state":{"subject":"Refund","body":"charged twice"},
                    "questions":[{"id":"route","text":"Which team?","type":"choice","options":["Billing","Tech"]},
                                 {"id":"sev","text":"Severity?","type":"score","min":1,"max":5},
                                 {"id":"spam","text":"Spam?","type":"noul"}]}"#,
            )
            .unwrap();
        let resp: crate::decide::DecideResponse = serde_json::from_str(&out).unwrap();
        assert!(resp
            .answers
            .iter()
            .all(|a| a.source == "lm" && a.error.is_none()));
        for a in &resp.answers {
            let s: f32 = a.distribution.iter().map(|o| o.p).sum();
            assert!((s - 1.0).abs() < 1e-3);
        }
    }
}
