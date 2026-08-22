// sample.rs — Sampling strategies and streaming generation for inference
//
// Standalone module that operates on logits — the LLM's forward pass produces
// Vec<f32> per position; everything in this file works on those slices and is
// independent of how the model was built.
//
// Provided:
//   - Temperature scaling
//   - Top-k filtering (keep only top k logits)
//   - Top-p / nucleus filtering (keep smallest set with cumulative prob ≥ p)
//   - Repetition penalty (down-weight already-generated tokens)
//   - Multinomial sampling from a probability distribution
//   - Streaming generation loop with a user-supplied token callback

use super::data::{special, Tokenizer};

// ─── Sampling configuration ──────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct SampleConfig {
    /// Softmax temperature.  Higher = more diverse; 1.0 = unchanged; 0.0 = greedy.
    pub temperature: f32,
    /// Top-k filter.  None = disabled.
    pub top_k: Option<usize>,
    /// Top-p / nucleus filter.  None = disabled.  Common values: 0.9, 0.95.
    pub top_p: Option<f32>,
    /// Repetition penalty applied to tokens already in the context.
    /// 1.0 = no penalty; 1.1–1.3 typical for reducing loops.
    pub repetition_penalty: f32,
    /// Maximum new tokens to generate before forced stop.
    pub max_new_tokens: usize,
    /// Stop generation if any of these token ids is produced.
    pub stop_tokens: Vec<usize>,
    /// Optional fixed random seed for reproducible sampling.  None = use system time.
    pub seed: Option<u64>,
}

impl Default for SampleConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: None,
            top_p: Some(0.9),
            repetition_penalty: 1.0,
            max_new_tokens: 128,
            stop_tokens: vec![special::EOS],
            seed: None,
        }
    }
}

impl SampleConfig {
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_k: None,
            top_p: None,
            repetition_penalty: 1.0,
            ..Default::default()
        }
    }

    pub fn creative() -> Self {
        Self {
            temperature: 1.0,
            top_p: Some(0.95),
            repetition_penalty: 1.1,
            ..Default::default()
        }
    }

    pub fn focused() -> Self {
        Self {
            temperature: 0.7,
            top_p: Some(0.9),
            repetition_penalty: 1.15,
            ..Default::default()
        }
    }
}

// ─── Logit transforms ─────────────────────────────────────────────────────────

/// Apply temperature scaling: logits /= T.  T=0 returns the input unchanged
/// (caller will use argmax).
pub fn apply_temperature(logits: &mut [f32], temperature: f32) {
    if temperature <= 0.0 || (temperature - 1.0).abs() < 1e-6 {
        return;
    }
    for l in logits.iter_mut() {
        *l /= temperature;
    }
}

/// Penalise tokens already present in `context` by dividing their logit by
/// `penalty` (if positive) or multiplying by it (if negative — symmetric).
pub fn apply_repetition_penalty(logits: &mut [f32], context: &[usize], penalty: f32) {
    if (penalty - 1.0).abs() < 1e-6 {
        return;
    }
    for &tok in context {
        if tok < logits.len() {
            if logits[tok] > 0.0 {
                logits[tok] /= penalty;
            } else {
                logits[tok] *= penalty;
            }
        }
    }
}

/// Keep only the top-k logits — set the rest to NEG_INFINITY.
pub fn apply_top_k(logits: &mut [f32], k: usize) {
    if k == 0 || k >= logits.len() {
        return;
    }

    // Find the k-th largest value (anything below it is filtered out)
    let mut sorted: Vec<f32> = logits.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let cutoff = sorted[k - 1];

    for l in logits.iter_mut() {
        if *l < cutoff {
            *l = f32::NEG_INFINITY;
        }
    }
}

/// Top-p / nucleus filtering: keep the smallest set of tokens whose cumulative
/// probability is ≥ `p`.  Everything else is set to NEG_INFINITY.
pub fn apply_top_p(logits: &mut [f32], p: f32) {
    if p >= 1.0 || p <= 0.0 {
        return;
    }

    // Compute softmax over the logits we currently have
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let probs: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = probs.iter().sum();
    let probs: Vec<f32> = probs.iter().map(|&pr| pr / sum).collect();

    // Sort tokens by probability descending
    let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Find the cutoff index where cumulative prob crosses p
    let mut cum = 0.0f32;
    let mut keep = std::collections::HashSet::new();
    for (idx, prob) in indexed {
        keep.insert(idx);
        cum += prob;
        if cum >= p {
            break;
        }
    }

    // Mask out tokens not in the keep set
    for (i, l) in logits.iter_mut().enumerate() {
        if !keep.contains(&i) {
            *l = f32::NEG_INFINITY;
        }
    }
}

// ─── Sampling ─────────────────────────────────────────────────────────────────

/// Convert logits to a normalised probability distribution.
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum == 0.0 || !sum.is_finite() {
        return vec![1.0 / logits.len() as f32; logits.len()];
    }
    exps.iter().map(|&e| e / sum).collect()
}

/// Sample from a probability distribution using a simple LCG.
/// Returns the chosen token index.
pub fn multinomial(probs: &[f32], rng: &mut SimpleRng) -> usize {
    let r = rng.next_f32();
    let mut cum = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cum += p;
        if r < cum {
            return i;
        }
    }
    probs.len() - 1
}

/// Argmax — returns the token with the highest logit.  Used for greedy decoding.
pub fn argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Pick the next token given raw logits and a SampleConfig.
///
/// Applies (in order): repetition penalty, temperature, top-k, top-p, then samples.
/// `context` is the full sequence so far (for repetition penalty).
pub fn sample_next(
    logits: &[f32],
    context: &[usize],
    cfg: &SampleConfig,
    rng: &mut SimpleRng,
) -> usize {
    let mut l: Vec<f32> = logits.to_vec();

    apply_repetition_penalty(&mut l, context, cfg.repetition_penalty);

    // Temperature 0 → greedy decode after rep penalty
    if cfg.temperature <= 1e-6 {
        return argmax(&l);
    }

    apply_temperature(&mut l, cfg.temperature);
    if let Some(k) = cfg.top_k {
        apply_top_k(&mut l, k);
    }
    if let Some(p) = cfg.top_p {
        apply_top_p(&mut l, p);
    }

    let probs = softmax(&l);
    multinomial(&probs, rng)
}

// ─── Streaming generation ─────────────────────────────────────────────────────

/// Callback type for streaming generation.  Called once per new token with
/// (token_id, decoded_piece).  Return `true` to continue, `false` to stop early.
pub trait TokenCallback {
    fn on_token(&mut self, token_id: usize, piece: &str) -> bool;
}

impl<F: FnMut(usize, &str) -> bool> TokenCallback for F {
    fn on_token(&mut self, token_id: usize, piece: &str) -> bool {
        self(token_id, piece)
    }
}

/// Streaming generation.  Calls `callback` for each generated token; if the
/// callback returns false, stops early.
///
/// `forward_fn` — closure that takes the current token ids and returns logits.
///                Wrap your model.forward() call here.
///
/// Returns the full sequence of generated token ids (excluding the prompt).
pub fn generate_stream<F, C>(
    prompt_ids: &[usize],
    cfg: &SampleConfig,
    tokenizer: &Tokenizer,
    mut forward_fn: F,
    mut callback: C,
) -> Vec<usize>
where
    F: FnMut(&[usize]) -> Vec<Vec<f32>>,
    C: TokenCallback,
{
    let mut ids = prompt_ids.to_vec();
    let mut generated = Vec::new();
    let mut rng = SimpleRng::new(cfg.seed.unwrap_or_else(|| {
        // Fallback seed if none provided — uses ids as deterministic seed
        let mut s: u64 = 0xCAFEBABE;
        for &i in &ids {
            s = s.wrapping_mul(31).wrapping_add(i as u64);
        }
        s
    }));

    for _ in 0..cfg.max_new_tokens {
        let logits = forward_fn(&ids);
        let last = match logits.last() {
            Some(l) => l,
            None => break,
        };

        let next = sample_next(last, &ids, cfg, &mut rng);

        // Stop conditions
        if cfg.stop_tokens.contains(&next) {
            break;
        }

        // Decode this single token for the callback
        let piece = tokenizer
            .id_to_word
            .get(next)
            .map(|s| s.as_str())
            .unwrap_or("<UNK>");

        if !callback.on_token(next, piece) {
            break;
        }

        ids.push(next);
        generated.push(next);
    }

    generated
}

// ─── Deterministic RNG ────────────────────────────────────────────────────────

/// Linear congruential generator — small, deterministic, no dependencies.
/// Not cryptographically secure; perfectly fine for sampling.
pub struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    pub fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.state >> 32) as u32
    }

    pub fn next_f32(&mut self) -> f32 {
        (self.next_u32() as f32) / (u32::MAX as f32)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temperature_zero_is_greedy() {
        let logits = vec![1.0f32, 3.0, 2.0, 0.5];
        let cfg = SampleConfig::greedy();
        let mut rng = SimpleRng::new(42);
        let tok = sample_next(&logits, &[], &cfg, &mut rng);
        assert_eq!(tok, 1); // index of max
    }

    #[test]
    fn top_k_masks_low_logits() {
        let mut logits = vec![1.0, 5.0, 2.0, 4.0, 3.0];
        apply_top_k(&mut logits, 2);
        // Only top-2 (values 5.0 and 4.0) should remain finite
        let finite_count = logits.iter().filter(|&&l| l.is_finite()).count();
        assert_eq!(finite_count, 2);
    }

    #[test]
    fn top_p_keeps_high_probability_mass() {
        // Almost all mass on token 0
        let mut logits = vec![10.0, 0.0, 0.0, 0.0];
        apply_top_p(&mut logits, 0.5);
        // Only token 0 should survive at p=0.5
        assert!(logits[0].is_finite());
        assert_eq!(logits[1], f32::NEG_INFINITY);
    }

    #[test]
    fn repetition_penalty_reduces_repeated_logit() {
        let mut logits = vec![1.0f32, 2.0, 3.0];
        apply_repetition_penalty(&mut logits, &[1], 2.0);
        assert!(logits[1] < 2.0, "repeated token should be penalised");
        assert!((logits[0] - 1.0).abs() < 1e-6, "non-repeated unchanged");
    }

    #[test]
    fn multinomial_respects_distribution() {
        // Token 2 has 100% probability — should always be sampled
        let probs = vec![0.0, 0.0, 1.0, 0.0];
        let mut rng = SimpleRng::new(1);
        for _ in 0..10 {
            assert_eq!(multinomial(&probs, &mut rng), 2);
        }
    }

    #[test]
    fn streaming_stops_on_eos() {
        // Mock forward function that always emits EOS
        let cfg = SampleConfig {
            temperature: 0.0,
            max_new_tokens: 100,
            stop_tokens: vec![special::EOS],
            ..Default::default()
        };

        let tok = Tokenizer::new();
        // Vocab now has 5 special tokens
        let mut last_token = None;
        let forward = |_ids: &[usize]| -> Vec<Vec<f32>> {
            // logits where EOS (index 3) wins
            let mut row = vec![0.0; tok.vocab_size()];
            row[special::EOS] = 100.0;
            vec![row]
        };
        let cb = |t: usize, _p: &str| {
            last_token = Some(t);
            true
        };

        let generated = generate_stream(&[], &cfg, &tok, forward, cb);
        assert!(generated.is_empty(), "should stop immediately on EOS");
    }
}
