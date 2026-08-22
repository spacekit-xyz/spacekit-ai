// sample.rs — minimal RNG for NCA (extracted from growformer-llm `v2/sample.rs`).
//
// The full growformer sampling module pulls in tokenizer / logits helpers; NCA only
// needs deterministic floats for the stochastic fire mask.

/// Linear congruential generator — small, deterministic, no dependencies.
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
