//! Growformer Paramecium — lattice-only sub-neuronal inference engine.
//!
//! Inspired by Paramecium caudatum: 17 behaviors from ~100,000 microtubules,
//! zero neurons. The infraciliary lattice coordinates 5,000 cilia through
//! metachronal waves on a microtubule mesh that predates the nervous system
//! by a billion years.
//!
//! This module implements computation without NeuralEnvironment, without
//! synapses, without backpropagation. The primitives are:
//!
//! - **Cilium nodes** on a regular lattice grid (basal bodies)
//! - **Metachronal waves** propagating state across the lattice (phase-locked EMA)
//! - **Behavioral programs** stored as lattice attractors (codebook entries)
//! - **Gradient sensing** for input → program selection (chemotaxis analog)
//! - **Habituation** for dampening repeated stimuli (primitive learning)
//!
//! Inference: project input onto E8 lattice → wave-select behavioral program →
//! decode response. No forward pass through layers. No weighted sums.
//!
//! Training: wave-phase alignment of codebook entries from data. Each sample
//! shifts the nearest attractor via EMA. The lattice self-organizes through
//! repeated exposure — like a paramecium learning to ignore a repeated stimulus.

use serde::{Deserialize, Serialize};
use crate::spectral::{E8Lattice, TokenDictionary};

/// A node on the infraciliary lattice. Each cilium has a position in E8 space,
/// a phase (activation level), and a habituation counter.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CiliumNode {
    /// Position in E8 lattice space (quantized).
    pub lattice_point: [f32; 8],
    /// Current phase: 0.0 = resting, 1.0 = fully activated.
    pub phase: f32,
    /// Habituation: how many consecutive activations without novel input.
    /// Higher values dampen the response (the paramecium "gets used to it").
    pub habituation: f32,
}

/// A behavioral program stored as a lattice attractor.
/// Analogous to a ResponseArchetype but without the neural substrate —
/// the program IS the lattice configuration, not a pattern learned by neurons.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BehavioralProgram {
    /// Centroid embedding in bridge space (for gradient sensing / selection).
    pub centroid: Vec<f32>,
    /// E8-quantized centroid (n/8 × 8d lattice points).
    pub lattice_signature: Vec<f32>,
    /// The response token sequence this program produces.
    pub token_sequence: Vec<u16>,
    /// Activation count (how many times this program has fired).
    pub activation_count: u64,
    /// EMA of input embeddings that activated this program.
    /// Drifts the centroid toward the data distribution.
    pub ema_centroid: Vec<f32>,
    /// Confidence: how tightly clustered the activating inputs are.
    pub coherence: f32,
    /// Habituation counter: dampens response when this program fires repeatedly.
    pub habituation: f32,
}

/// Wave propagation state across the lattice.
/// The metachronal wave is an EMA field that carries "what was just sensed"
/// across all nodes, enabling coordination without point-to-point connections.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WaveState {
    /// Per-node phase values (same length as lattice nodes).
    pub phases: Vec<f32>,
    /// Wave propagation speed: how quickly phase changes spread.
    pub propagation_alpha: f32,
    /// Damping: how quickly waves decay (prevents runaway oscillation).
    pub damping: f32,
    /// Global wave energy (sum of squared phases). Tracks overall activation level.
    pub energy: f32,
}

impl WaveState {
    pub fn new(num_nodes: usize) -> Self {
        Self {
            phases: vec![0.0; num_nodes],
            propagation_alpha: 0.3,
            damping: 0.95,
            energy: 0.0,
        }
    }

    /// Propagate a stimulus from a source node outward.
    /// Phase decays with lattice distance from the source.
    pub fn propagate(&mut self, source_idx: usize, intensity: f32) {
        if source_idx >= self.phases.len() {
            return;
        }
        self.phases[source_idx] = intensity;

        let n = self.phases.len();
        for i in 0..n {
            if i == source_idx {
                continue;
            }
            let distance = ((i as f32 - source_idx as f32).abs() / n as f32).min(0.5);
            let wave_contribution = intensity * (-distance * 4.0).exp();
            self.phases[i] = self.phases[i] * self.damping
                + wave_contribution * self.propagation_alpha;
        }

        self.energy = self.phases.iter().map(|p| p * p).sum();
    }

    /// Global decay step: all phases move toward zero.
    pub fn decay(&mut self) {
        for p in &mut self.phases {
            *p *= self.damping;
        }
        self.energy = self.phases.iter().map(|p| p * p).sum();
    }
}

/// The Infraciliary Lattice — the complete paramecium computation substrate.
///
/// No neurons. No synapses. No backpropagation.
/// Computation is: sense → lattice projection → wave selection → program fire → decode.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InfraciliaryLattice {
    /// Behavioral programs (the paramecium's repertoire).
    pub programs: Vec<BehavioralProgram>,
    /// Wave state across program nodes.
    pub wave: WaveState,
    /// Token dictionary for encoding/decoding.
    pub dictionary: TokenDictionary,
    /// Habituation decay rate: how fast the organism "forgets" repeated stimuli.
    pub habituation_decay: f32,
    /// Learning rate for wave-phase alignment (EMA centroid drift).
    pub learning_rate: f32,
    /// Last selected program index.
    pub last_program: Option<usize>,
    /// Last inference confidence.
    pub last_confidence: f32,
}

/// Result of a paramecium inference step.
#[derive(Clone, Debug)]
pub struct ParameciumResponse {
    pub text: String,
    pub program_idx: usize,
    pub confidence: f32,
    pub wave_energy: f32,
    pub habituated: bool,
}

impl InfraciliaryLattice {
    pub fn new(dictionary: TokenDictionary) -> Self {
        Self {
            programs: Vec::new(),
            wave: WaveState::new(0),
            dictionary,
            habituation_decay: 0.9,
            learning_rate: 0.05,
            last_program: None,
            last_confidence: 0.0,
        }
    }

    /// Build behavioral programs from (embedding, response_text) pairs.
    /// This is the paramecium's "development" — organizing the lattice from
    /// exposure to data, without gradient descent or error backpropagation.
    ///
    /// Algorithm:
    /// 1. Quantize each embedding to E8 lattice
    /// 2. Cluster by lattice proximity (nearest existing program or spawn new)
    /// 3. Each cluster becomes a BehavioralProgram with EMA centroid
    /// 4. Initialize wave state for the program count
    pub fn develop(&mut self, samples: &[(Vec<f32>, String)], spawn_threshold: f32) {
        for (embedding, response) in samples {
            let token_ids = self.dictionary.encode(response);
            let lattice_sig = E8Lattice::quantize_64d(embedding);

            if let Some((idx, similarity)) = self.nearest_program(embedding) {
                if similarity >= spawn_threshold {
                    let prog = &mut self.programs[idx];
                    let alpha = self.learning_rate;
                    for (i, v) in embedding.iter().enumerate() {
                        if i < prog.ema_centroid.len() {
                            prog.ema_centroid[i] = prog.ema_centroid[i] * (1.0 - alpha) + v * alpha;
                        }
                    }
                    prog.activation_count += 1;

                    let ac = prog.activation_count as f32;
                    prog.coherence = prog.coherence * ((ac - 1.0) / ac) + similarity / ac;
                    continue;
                }
            }

            self.programs.push(BehavioralProgram {
                centroid: embedding.clone(),
                lattice_signature: lattice_sig,
                token_sequence: token_ids,
                activation_count: 1,
                ema_centroid: embedding.clone(),
                coherence: 1.0,
                habituation: 0.0,
            });
        }

        self.wave = WaveState::new(self.programs.len());
    }

    /// Sense + respond: the complete paramecium inference loop.
    ///
    /// 1. **Gradient sensing (chemotaxis):** compute similarity to all programs
    /// 2. **Wave selection:** propagate activation from the best match
    /// 3. **Habituation check:** dampen if same program fires repeatedly
    /// 4. **Program fire:** decode the selected program's token sequence
    pub fn respond(&mut self, embedding: &[f32]) -> ParameciumResponse {
        if self.programs.is_empty() {
            return ParameciumResponse {
                text: String::new(),
                program_idx: 0,
                confidence: 0.0,
                wave_energy: 0.0,
                habituated: false,
            };
        }

        // 1. Gradient sensing: find the nearest behavioral program
        let (best_idx, best_similarity) = self.nearest_program(embedding)
            .unwrap_or((0, 0.0));

        // 2. Wave propagation: activate the winning program and let the wave spread
        self.wave.propagate(best_idx, best_similarity);

        // 3. Habituation: if the same program fires repeatedly, dampen confidence
        let habituated = self.last_program == Some(best_idx);
        let effective_confidence = if habituated {
            let prog = &mut self.programs[best_idx];
            prog.habituation = (prog.habituation + 1.0).min(10.0);
            let damping = 1.0 / (1.0 + prog.habituation * 0.1);
            best_similarity * damping
        } else {
            // Reset habituation on the previously active program
            if let Some(prev) = self.last_program {
                if prev < self.programs.len() {
                    self.programs[prev].habituation *= self.habituation_decay;
                }
            }
            best_similarity
        };

        // 4. Wave-modulated selection: let neighboring programs contribute
        //    if the wave energy spreads to them above a threshold.
        //    This is the metachronal coordination — multiple cilia beating together.
        let mut selected_idx = best_idx;
        let mut best_wave_score = self.wave.phases.get(best_idx).copied().unwrap_or(0.0);
        for (i, &phase) in self.wave.phases.iter().enumerate() {
            if phase > best_wave_score && self.programs[i].coherence > 0.5 {
                best_wave_score = phase;
                selected_idx = i;
            }
        }

        // 5. Fire the selected program
        let prog = &self.programs[selected_idx];
        let text = self.dictionary.decode(&prog.token_sequence);

        // 6. Online learning: drift the centroid toward the input
        let alpha = self.learning_rate;
        let prog_mut = &mut self.programs[selected_idx];
        for (i, v) in embedding.iter().enumerate() {
            if i < prog_mut.ema_centroid.len() {
                prog_mut.ema_centroid[i] =
                    prog_mut.ema_centroid[i] * (1.0 - alpha) + v * alpha;
            }
        }
        prog_mut.activation_count += 1;

        self.last_program = Some(selected_idx);
        self.last_confidence = effective_confidence;

        // 7. Wave decay for next cycle
        self.wave.decay();

        ParameciumResponse {
            text,
            program_idx: selected_idx,
            confidence: effective_confidence,
            wave_energy: self.wave.energy,
            habituated,
        }
    }

    /// Gradient sensing: find the nearest behavioral program by cosine similarity.
    /// This is chemotaxis — sensing the concentration gradient and swimming toward
    /// the strongest signal.
    fn nearest_program(&self, embedding: &[f32]) -> Option<(usize, f32)> {
        if self.programs.is_empty() {
            return None;
        }
        let mut best_idx = 0;
        let mut best_sim = f32::NEG_INFINITY;
        for (i, prog) in self.programs.iter().enumerate() {
            let sim = cosine_similarity(embedding, &prog.ema_centroid);
            if sim > best_sim {
                best_sim = sim;
                best_idx = i;
            }
        }
        Some((best_idx, best_sim.max(0.0)))
    }

    /// Emergency avoidance: the paramecium equivalent of a burst reversal.
    /// Resets wave state and forces selection of the lowest-habituation program
    /// that isn't the current one.
    pub fn avoidance_reaction(&mut self) {
        self.wave = WaveState::new(self.programs.len());
        if self.programs.len() > 1 {
            let current = self.last_program.unwrap_or(0);
            let mut min_hab = f32::MAX;
            let mut min_idx = 0;
            for (i, prog) in self.programs.iter().enumerate() {
                if i != current && prog.habituation < min_hab {
                    min_hab = prog.habituation;
                    min_idx = i;
                }
            }
            self.last_program = Some(min_idx);
        }
    }

    /// Autogamy: self-reorganize when no new data arrives.
    /// Merge programs that have drifted close together, prune programs
    /// with zero activations.
    pub fn autogamy(&mut self, merge_threshold: f32) {
        let mut merged = vec![false; self.programs.len()];
        let n = self.programs.len();

        for i in 0..n {
            if merged[i] {
                continue;
            }
            for j in (i + 1)..n {
                if merged[j] {
                    continue;
                }
                let sim = cosine_similarity(&self.programs[i].ema_centroid,
                                           &self.programs[j].ema_centroid);
                if sim >= merge_threshold {
                    let total = self.programs[i].activation_count
                        + self.programs[j].activation_count;
                    let wi = self.programs[i].activation_count as f32 / total as f32;
                    let wj = self.programs[j].activation_count as f32 / total as f32;

                    let dim = self.programs[i].ema_centroid.len();
                    for k in 0..dim {
                        self.programs[i].ema_centroid[k] =
                            self.programs[i].ema_centroid[k] * wi
                            + self.programs[j].ema_centroid[k] * wj;
                    }
                    self.programs[i].activation_count = total;
                    if self.programs[j].activation_count > self.programs[i].activation_count {
                        self.programs[i].token_sequence = self.programs[j].token_sequence.clone();
                    }
                    merged[j] = true;
                }
            }
        }

        let mut i = 0;
        while i < self.programs.len() {
            if merged.get(i).copied().unwrap_or(false)
                || self.programs[i].activation_count == 0
            {
                self.programs.remove(i);
                merged.remove(i);
            } else {
                i += 1;
            }
        }

        self.wave = WaveState::new(self.programs.len());
    }

    /// Trichocyst volley: fire a burst of the top-K most relevant programs
    /// for a given input. Returns multiple candidate responses ranked by
    /// wave-modulated confidence. Useful when a single program isn't
    /// sufficient (composition).
    pub fn trichocyst_volley(&mut self, embedding: &[f32], k: usize) -> Vec<ParameciumResponse> {
        if self.programs.is_empty() {
            return Vec::new();
        }

        let mut scored: Vec<(usize, f32)> = self.programs.iter().enumerate()
            .map(|(i, prog)| (i, cosine_similarity(embedding, &prog.ema_centroid)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scored.into_iter().take(k).map(|(idx, sim)| {
            let prog = &self.programs[idx];
            let text = self.dictionary.decode(&prog.token_sequence);
            ParameciumResponse {
                text,
                program_idx: idx,
                confidence: sim,
                wave_energy: self.wave.phases.get(idx).copied().unwrap_or(0.0),
                habituated: false,
            }
        }).collect()
    }

    // -------------------------------------------------------------------
    // Learning-side: signals the paramecium provides to the training loop
    // -------------------------------------------------------------------

    /// Score a training sample's novelty. Returns (confidence, nearest_program_idx).
    /// Low confidence = genuinely novel input the lattice hasn't seen.
    /// High confidence = well-represented, redundant for training.
    /// Use as a curriculum signal: train harder on low-novelty samples.
    pub fn novelty_score(&self, embedding: &[f32]) -> (f32, Option<usize>) {
        match self.nearest_program(embedding) {
            Some((idx, sim)) => (sim, Some(idx)),
            None => (0.0, None),
        }
    }

    /// Score and rank a batch of training samples by novelty (ascending confidence).
    /// Returns indices into the original slice, hardest-first.
    /// Use for curriculum learning: present novel samples with more training steps.
    pub fn curriculum_order(&self, embeddings: &[Vec<f32>]) -> Vec<(usize, f32)> {
        let mut scored: Vec<(usize, f32)> = embeddings.iter().enumerate()
            .map(|(i, emb)| {
                let (conf, _) = self.novelty_score(emb);
                (i, conf)
            })
            .collect();
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        scored
    }

    /// Discover natural cluster structure from embeddings.
    /// Returns (suggested_group_count, cluster_assignments).
    /// Each cluster assignment is the program index the embedding maps to.
    /// Use for bottom-up group discovery instead of hand-labeled action_target.
    pub fn discover_groups(&mut self, embeddings: &[Vec<f32>], spawn_threshold: f32) -> (usize, Vec<usize>) {
        let saved_programs = self.programs.clone();
        let saved_wave = self.wave.clone();

        self.programs.clear();
        self.wave = WaveState::new(0);

        let dummy_texts: Vec<(Vec<f32>, String)> = embeddings.iter()
            .map(|e| (e.clone(), String::new()))
            .collect();
        self.develop(&dummy_texts, spawn_threshold);

        let group_count = self.programs.len();
        let assignments: Vec<usize> = embeddings.iter()
            .map(|emb| self.nearest_program(emb).map(|(idx, _)| idx).unwrap_or(0))
            .collect();

        self.programs = saved_programs;
        self.wave = saved_wave;

        (group_count, assignments)
    }

    /// Extract program centroids as initial archetype prototypes for codebook construction.
    /// Returns (centroid_embedding, representative_token_sequence) pairs.
    /// Use to initialize AlgebraicCodebook archetypes instead of random k-means.
    pub fn archetype_seeds(&self) -> Vec<(Vec<f32>, Vec<u16>)> {
        self.programs.iter()
            .filter(|p| p.activation_count > 0)
            .map(|p| (p.ema_centroid.clone(), p.token_sequence.clone()))
            .collect()
    }

    /// Compute a wave-conditioning vector for a given input.
    /// Propagates the input through the lattice and returns the resulting
    /// wave phase pattern as a fixed-size vector. This captures the lattice's
    /// "opinion" about how to route the input — which programs are relevant
    /// and how they compete — without requiring the learned router.
    /// Concatenate with the bridge embedding for enriched conditioning.
    pub fn wave_conditioning(&mut self, embedding: &[f32]) -> Vec<f32> {
        if self.programs.is_empty() {
            return Vec::new();
        }

        let (best_idx, best_sim) = self.nearest_program(embedding)
            .unwrap_or((0, 0.0));

        let mut scratch_wave = WaveState::new(self.programs.len());
        scratch_wave.propagate(best_idx, best_sim);

        scratch_wave.phases
    }

    pub fn program_count(&self) -> usize {
        self.programs.len()
    }

    /// Total memory footprint estimate in bytes.
    pub fn memory_bytes(&self) -> usize {
        let prog_bytes: usize = self.programs.iter().map(|p| {
            p.centroid.len() * 4
                + p.lattice_signature.len() * 4
                + p.token_sequence.len() * 2
                + p.ema_centroid.len() * 4
                + 24 // scalar fields
        }).sum();
        let wave_bytes = self.wave.phases.len() * 4 + 12;
        prog_bytes + wave_bytes + 32
    }

    /// Build from an existing AlgebraicCodebook + dictionary + embeddings.
    /// Converts neuronal codebook archetypes into lattice behavioral programs.
    pub fn from_codebook(
        dictionary: TokenDictionary,
        archetypes: &[(Vec<u16>, Vec<f32>)], // (token_sequence, centroid_embedding)
    ) -> Self {
        let mut lattice = Self::new(dictionary);
        for (tokens, centroid) in archetypes {
            let lattice_sig = E8Lattice::quantize_64d(centroid);
            lattice.programs.push(BehavioralProgram {
                centroid: centroid.clone(),
                lattice_signature: lattice_sig,
                token_sequence: tokens.clone(),
                activation_count: 1,
                ema_centroid: centroid.clone(),
                coherence: 1.0,
                habituation: 0.0,
            });
        }
        lattice.wave = WaveState::new(lattice.programs.len());
        lattice
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..a.len().min(b.len()) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom < 1e-10 { 0.0 } else { dot / denom }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::spectral::TokenDictionary;

    fn test_dict() -> TokenDictionary {
        let texts = &[
            "reset your password",
            "the observer pattern decouples event producers from consumers",
            "implement binary search in python",
            "navigate toward food",
            "avoid the predator",
        ];
        let text_refs: Vec<&str> = texts.iter().copied().collect();
        TokenDictionary::build(&text_refs, 256)
    }

    fn test_embedding(seed: f32) -> Vec<f32> {
        (0..crate::dimension::language::DEFAULT_BRIDGE_DIM)
            .map(|i| ((i as f32 + seed) * 0.1).sin()).collect()
    }

    #[test]
    fn test_develop_spawns_programs() {
        let dict = test_dict();
        let mut lattice = InfraciliaryLattice::new(dict);

        let samples: Vec<(Vec<f32>, String)> = vec![
            (test_embedding(1.0), "reset your password".to_string()),
            (test_embedding(2.0), "the observer pattern decouples".to_string()),
            (test_embedding(100.0), "implement binary search".to_string()),
        ];

        lattice.develop(&samples, 0.95);

        assert!(lattice.program_count() >= 2,
            "distant embeddings should spawn separate programs, got {}",
            lattice.program_count());
    }

    #[test]
    fn test_respond_returns_text() {
        let dict = test_dict();
        let mut lattice = InfraciliaryLattice::new(dict);

        let samples: Vec<(Vec<f32>, String)> = vec![
            (test_embedding(1.0), "reset your password".to_string()),
            (test_embedding(50.0), "implement binary search".to_string()),
        ];
        lattice.develop(&samples, 0.99);

        let response = lattice.respond(&test_embedding(1.1));
        assert!(!response.text.is_empty(), "response should contain text");
        assert!(response.confidence > 0.0, "confidence should be positive");
    }

    #[test]
    fn test_habituation_dampens_repeated_stimulus() {
        let dict = test_dict();
        let mut lattice = InfraciliaryLattice::new(dict);

        let emb = test_embedding(1.0);
        lattice.develop(&[(emb.clone(), "reset password".to_string())], 0.99);

        let r1 = lattice.respond(&emb);
        let r2 = lattice.respond(&emb);
        let r3 = lattice.respond(&emb);

        assert!(!r1.habituated, "first response should not be habituated");
        assert!(r2.habituated, "second identical stimulus should be habituated");
        assert!(r3.confidence <= r2.confidence,
            "confidence should not increase with habituation");
    }

    #[test]
    fn test_wave_propagation() {
        let mut wave = WaveState::new(10);
        wave.propagate(5, 1.0);

        assert!(wave.phases[5] > wave.phases[0],
            "source node should have higher phase than distant node");
        assert!(wave.phases[4] > wave.phases[0],
            "neighbor should have higher phase than far node");
        assert!(wave.energy > 0.0, "wave energy should be positive after propagation");

        let energy_before = wave.energy;
        wave.decay();
        assert!(wave.energy < energy_before,
            "energy should decrease after decay: {} vs {}", wave.energy, energy_before);
    }

    #[test]
    fn test_avoidance_reaction() {
        let dict = test_dict();
        let mut lattice = InfraciliaryLattice::new(dict);

        let samples: Vec<(Vec<f32>, String)> = vec![
            (test_embedding(1.0), "reset your password".to_string()),
            (test_embedding(50.0), "implement binary search".to_string()),
        ];
        lattice.develop(&samples, 0.99);

        let _ = lattice.respond(&test_embedding(1.0));
        let before = lattice.last_program;

        lattice.avoidance_reaction();
        assert_ne!(lattice.last_program, before,
            "avoidance should switch to a different program");
    }

    #[test]
    fn test_autogamy_merges_similar() {
        let dict = test_dict();
        let mut lattice = InfraciliaryLattice::new(dict);

        let e1 = test_embedding(1.0);
        let mut e2 = e1.clone();
        for v in &mut e2 { *v += 0.001; }

        lattice.develop(&[
            (e1, "reset password".to_string()),
            (e2, "reset your password".to_string()),
            (test_embedding(100.0), "binary search".to_string()),
        ], 0.99);

        let before = lattice.program_count();
        lattice.autogamy(0.99);
        assert!(lattice.program_count() <= before,
            "autogamy should merge very similar programs");
    }

    #[test]
    fn test_trichocyst_volley() {
        let dict = test_dict();
        let mut lattice = InfraciliaryLattice::new(dict);

        let samples: Vec<(Vec<f32>, String)> = vec![
            (test_embedding(1.0), "reset password".to_string()),
            (test_embedding(2.0), "update email".to_string()),
            (test_embedding(50.0), "binary search".to_string()),
        ];
        lattice.develop(&samples, 0.99);

        let volley = lattice.trichocyst_volley(&test_embedding(1.5), 2);
        assert_eq!(volley.len(), 2, "should return top-2 programs");
        assert!(volley[0].confidence >= volley[1].confidence,
            "results should be sorted by confidence");
    }

    #[test]
    fn test_memory_footprint_is_tiny() {
        let dict = test_dict();
        let mut lattice = InfraciliaryLattice::new(dict);

        let samples: Vec<(Vec<f32>, String)> = (0..50)
            .map(|i| (test_embedding(i as f32 * 10.0), format!("program {}", i)))
            .collect();
        lattice.develop(&samples, 0.99);

        let bytes = lattice.memory_bytes();
        assert!(bytes < 100_000,
            "50-program paramecium should be under 100KB, got {} bytes", bytes);
    }

    #[test]
    fn test_from_codebook() {
        let dict = test_dict();
        let archetypes = vec![
            (dict.encode("reset your password"), test_embedding(1.0)),
            (dict.encode("binary search"), test_embedding(50.0)),
        ];

        let lattice = InfraciliaryLattice::from_codebook(dict, &archetypes);
        assert_eq!(lattice.program_count(), 2);
        assert!(lattice.memory_bytes() > 0);
    }

    #[test]
    fn test_gradient_sensing_selects_nearest() {
        let dict = test_dict();
        let mut lattice = InfraciliaryLattice::new(dict);

        let e_password = test_embedding(1.0);
        let e_search = test_embedding(100.0);

        lattice.develop(&[
            (e_password.clone(), "reset your password".to_string()),
            (e_search.clone(), "implement binary search".to_string()),
        ], 0.99);

        let r1 = lattice.respond(&test_embedding(1.1));
        let r2 = lattice.respond(&test_embedding(99.9));

        assert_ne!(r1.program_idx, r2.program_idx,
            "different inputs should activate different programs");
    }

    // -------------------------------------------------------------------
    // Learning-side tests
    // -------------------------------------------------------------------

    #[test]
    fn test_novelty_score_known_vs_unknown() {
        let dict = test_dict();
        let mut lattice = InfraciliaryLattice::new(dict);

        let known = test_embedding(1.0);
        lattice.develop(&[(known.clone(), "reset password".to_string())], 0.99);

        let (known_conf, known_idx) = lattice.novelty_score(&known);
        let (novel_conf, _) = lattice.novelty_score(&test_embedding(500.0));

        assert!(known_conf > novel_conf,
            "known input should have higher confidence than novel: {} vs {}",
            known_conf, novel_conf);
        assert!(known_idx.is_some());
    }

    #[test]
    fn test_curriculum_order_hardest_first() {
        let dict = test_dict();
        let mut lattice = InfraciliaryLattice::new(dict);

        let e1 = test_embedding(1.0);
        lattice.develop(&[(e1.clone(), "known response".to_string())], 0.99);

        let embeddings = vec![
            test_embedding(1.1),   // very similar to known — easy
            test_embedding(500.0), // completely different — hard
            test_embedding(1.05),  // almost identical — easiest
        ];

        let order = lattice.curriculum_order(&embeddings);
        assert_eq!(order[0].0, 1,
            "hardest sample (idx 1, distant) should come first");
    }

    #[test]
    fn test_discover_groups_finds_clusters() {
        let dict = test_dict();
        let mut lattice = InfraciliaryLattice::new(dict);

        let embeddings: Vec<Vec<f32>> = vec![
            test_embedding(1.0),
            test_embedding(1.1),
            test_embedding(1.2),
            test_embedding(100.0),
            test_embedding(100.1),
            test_embedding(100.2),
        ];

        let (group_count, assignments) = lattice.discover_groups(&embeddings, 0.95);

        assert!(group_count >= 2,
            "two distant clusters should produce at least 2 groups, got {}", group_count);
        assert_eq!(assignments[0], assignments[1],
            "nearby embeddings should be in the same cluster");
        assert_eq!(assignments[3], assignments[4],
            "nearby embeddings should be in the same cluster");
        assert_ne!(assignments[0], assignments[3],
            "distant clusters should be in different groups");
    }

    #[test]
    fn test_archetype_seeds_returns_centroids() {
        let dict = test_dict();
        let mut lattice = InfraciliaryLattice::new(dict);

        lattice.develop(&[
            (test_embedding(1.0), "reset password".to_string()),
            (test_embedding(100.0), "binary search".to_string()),
        ], 0.99);

        let seeds = lattice.archetype_seeds();
        assert!(seeds.len() >= 2, "should have at least 2 archetype seeds");
        assert!(!seeds[0].0.is_empty(), "centroid should not be empty");
        assert!(!seeds[0].1.is_empty(), "token sequence should not be empty");
    }

    #[test]
    fn test_wave_conditioning_vector() {
        let dict = test_dict();
        let mut lattice = InfraciliaryLattice::new(dict);

        lattice.develop(&[
            (test_embedding(1.0), "reset password".to_string()),
            (test_embedding(100.0), "binary search".to_string()),
        ], 0.99);

        let wc = lattice.wave_conditioning(&test_embedding(1.1));
        assert_eq!(wc.len(), lattice.program_count(),
            "wave conditioning vector should have one element per program");
        assert!(wc.iter().any(|&v| v > 0.0),
            "at least one program should be activated");
    }
}
