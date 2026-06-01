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
///
/// Multi-timescale state modeled on Paramecium cell biology:
///
/// **Persistent (gene expression):** `centroid`, `token_sequence`, `quality_score`,
///   `reliability` — durable configuration that survives serialization.
///
/// **Medium-term (post-translational):** `session_centroid_drift`, `session_access_count`,
///   `session_quality_sum` — session-scoped state that accumulates across inference
///   turns within a session but resets on load. Enables in-context learning.
///
/// **Short-term (ionic/membrane):** `activation_level`, `refractory` —
///   per-inference volatile state that decays between turns. Prevents the
///   nearest-neighbor monoculture problem.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BehavioralProgram {
    /// Centroid embedding in bridge space (for gradient sensing / selection).
    pub centroid: Vec<f32>,
    /// E8-quantized centroid (n/8 × 8d lattice points).
    pub lattice_signature: Vec<f32>,
    /// The response token sequence this program produces.
    pub token_sequence: Vec<u16>,
    /// Exact training or correction line for display and lexical retrieval.
    /// When set, avoids [`TokenDictionary::encode`] mapping unknown words to the
    /// nearest dictionary token by edit distance, which garbles proper nouns.
    #[serde(default)]
    pub verbatim_display_text: Option<String>,
    /// Activation count (how many times this program has fired during training).
    pub activation_count: u64,
    /// EMA of input embeddings that activated this program.
    /// Drifts the centroid toward the data distribution.
    pub ema_centroid: Vec<f32>,
    /// Confidence: how tightly clustered the activating inputs are.
    pub coherence: f32,
    /// Habituation counter: dampens response when this program fires repeatedly.
    pub habituation: f32,

    // === Persistent state (gene expression / long-term config) ===

    /// Accumulated quality score from MetaCognition feedback.
    /// Programs that consistently pass quality checks build higher scores,
    /// biasing future retrieval toward proven-reliable responses.
    /// Range: [-1.0, 1.0], initialized at 0.0 (neutral).
    #[serde(default)]
    pub quality_score: f32,
    /// Reliability: ratio of successful MetaCognition passes to total retrievals.
    /// EMA-smoothed so recent performance weighs more than distant history.
    #[serde(default = "default_reliability")]
    pub reliability: f32,
    /// Total inference retrievals (lifetime, across all sessions).
    #[serde(default)]
    pub total_retrievals: u64,

    // === Medium-term state (post-translational / session-scoped) ===

    /// Session centroid drift: accumulated embedding bias from repeated queries
    /// in the current session. Enables in-context adaptation without modifying
    /// the persistent centroid. Applied as an additive offset during retrieval.
    #[serde(skip)]
    pub session_drift: Vec<f32>,
    /// Access count within the current session.
    #[serde(skip)]
    pub session_hits: u32,
    /// Cumulative quality feedback within this session (for session-level stats).
    #[serde(skip)]
    pub session_quality_sum: f32,

    // === Short-term state (ionic / membrane potential) ===

    /// Activation level: decays between inference turns. Recently-fired programs
    /// have high activation, enabling refractory-period suppression.
    #[serde(skip)]
    pub activation_level: f32,
    /// Refractory flag: when true, this program was just fired and should be
    /// suppressed in the next retrieval to force compositional diversity.
    #[serde(skip)]
    pub refractory: bool,
}

impl BehavioralProgram {
    /// Human-visible program body and preferred surface for lexical retrieval.
    pub fn display_text(&self, dictionary: &TokenDictionary) -> String {
        if let Some(ref v) = self.verbatim_display_text {
            if !v.is_empty() {
                return v.clone();
            }
        }
        dictionary.decode(&self.token_sequence)
    }
}

fn default_reliability() -> f32 { 0.5 }

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
    /// Novelty pressure factor for retrieval bias (1.0 = default, 2.0 = chat/companion).
    /// Set by the host after brain load based on inference_profile.
    #[serde(default = "default_novelty_factor")]
    pub novelty_factor: f32,
}

fn default_novelty_factor() -> f32 { 1.0 }

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
            novelty_factor: 1.0,
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
                verbatim_display_text: Some(response.clone()),
                activation_count: 1,
                ema_centroid: embedding.clone(),
                coherence: 1.0,
                habituation: 0.0,
                quality_score: 0.0,
                reliability: 0.5,
                total_retrievals: 0,
                session_drift: Vec::new(),
                session_hits: 0,
                session_quality_sum: 0.0,
                activation_level: 0.0,
                refractory: false,
            });
        }

        self.wave = WaveState::new(self.programs.len());
    }

    /// Contrastive refinement: push apart program centroids that are
    /// too similar. Called after `develop()` to sharpen decision boundaries.
    ///
    /// `margin`: minimum cosine similarity to trigger repulsion (e.g. 0.85)
    /// `rate`: how much to push apart per repulsion step (e.g. 0.05)
    /// Returns the number of repulsion operations performed.
    pub fn contrastive_refine(&mut self, margin: f32, rate: f32) -> usize {
        let n = self.programs.len();
        if n < 2 { return 0; }

        let mut repulsions = 0;
        for i in 0..n {
            for j in (i + 1)..n {
                let sim = {
                    let a = &self.programs[i].ema_centroid;
                    let b = &self.programs[j].ema_centroid;
                    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
                    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
                    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
                    if na < 1e-8 || nb < 1e-8 { 0.0 } else { dot / (na * nb) }
                };

                if sim > margin {
                    let tokens_match = self.programs[i].token_sequence == self.programs[j].token_sequence;
                    if tokens_match { continue; }

                    let dim = self.programs[i].ema_centroid.len().min(self.programs[j].ema_centroid.len());
                    for d in 0..dim {
                        let delta = self.programs[i].ema_centroid[d] - self.programs[j].ema_centroid[d];
                        self.programs[i].ema_centroid[d] += delta * rate;
                        self.programs[j].ema_centroid[d] -= delta * rate;
                    }
                    repulsions += 1;
                }
            }
        }
        repulsions
    }

    /// Develop with RTD negative repulsion: process pairs where some are
    /// marked as negatives (from replaced token detection). Negatives
    /// push AWAY from their nearest program instead of attracting.
    pub fn develop_with_negatives(
        &mut self,
        positives: &[(Vec<f32>, String)],
        negatives: &[Vec<f32>],
        spawn_threshold: f32,
        repulsion_rate: f32,
    ) {
        self.develop(positives, spawn_threshold);

        for neg_embedding in negatives {
            if let Some((idx, similarity)) = self.nearest_program(neg_embedding) {
                if similarity > 0.5 {
                    let prog = &mut self.programs[idx];
                    let dim = prog.ema_centroid.len().min(neg_embedding.len());
                    for d in 0..dim {
                        let delta = prog.ema_centroid[d] - neg_embedding[d];
                        prog.ema_centroid[d] += delta * repulsion_rate;
                    }
                }
            }
        }
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
        let text = prog.display_text(&self.dictionary);

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
            let text = prog.display_text(&self.dictionary);
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

    // ===================================================================
    // Continuum: multi-timescale state management
    // ===================================================================

    /// Begin a new session: reset all session-scoped and volatile state.
    /// Call this at the start of a conversation or inference batch.
    /// Persistent state (quality_score, reliability) is preserved.
    pub fn begin_session(&mut self) {
        for prog in &mut self.programs {
            prog.session_drift = vec![0.0f32; prog.ema_centroid.len()];
            prog.session_hits = 0;
            prog.session_quality_sum = 0.0;
            prog.activation_level = 0.0;
            prog.refractory = false;
        }
    }

    /// Record that a program was retrieved during inference.
    /// Updates short-term activation, session counters, and persistent retrievals.
    /// Also accumulates session centroid drift toward the query embedding.
    pub fn on_retrieval(&mut self, program_idx: usize, query_embedding: &[f32]) {
        if program_idx >= self.programs.len() { return; }

        let prog = &mut self.programs[program_idx];
        prog.total_retrievals += 1;
        prog.session_hits += 1;
        prog.activation_level = 1.0;
        prog.refractory = true;

        // Session centroid drift: gently pull toward the query that activated us.
        // This is the "second messenger" — transient but accumulating within a session.
        let drift_alpha = 0.1;
        if prog.session_drift.is_empty() {
            prog.session_drift = vec![0.0f32; prog.ema_centroid.len()];
        }
        let dim = prog.session_drift.len().min(query_embedding.len());
        for i in 0..dim {
            let delta = query_embedding[i] - prog.ema_centroid[i];
            prog.session_drift[i] += drift_alpha * delta;
        }
    }

    /// Apply MetaCognition feedback to a program after quality evaluation.
    /// `accepted`: whether MetaCognition accepted the response.
    /// `quality`: the MetaCognition quality score [0.0, 1.0].
    ///
    /// This is the "gene expression" pathway — persistent changes that
    /// bias future retrieval toward proven-reliable programs.
    pub fn apply_feedback(&mut self, program_idx: usize, accepted: bool, quality: f32) {
        if program_idx >= self.programs.len() { return; }

        let prog = &mut self.programs[program_idx];

        // Quality score: EMA toward +1 (accepted) or -1 (rejected)
        let feedback = if accepted { quality.clamp(0.0, 1.0) } else { -quality.clamp(0.0, 1.0) };
        let alpha = 0.15;
        prog.quality_score = (prog.quality_score * (1.0 - alpha) + feedback * alpha)
            .clamp(-1.0, 1.0);

        // Reliability: ratio of acceptances, EMA-smoothed
        let hit = if accepted { 1.0f32 } else { 0.0 };
        let rel_alpha = 0.1;
        prog.reliability = (prog.reliability * (1.0 - rel_alpha) + hit * rel_alpha)
            .clamp(0.0, 1.0);

        // Session quality tracking
        prog.session_quality_sum += if accepted { quality } else { -quality };
    }

    /// Decay short-term state between inference turns.
    /// Activation levels decay toward zero; refractory flags clear.
    /// Call this between turns in a multi-turn conversation.
    pub fn decay_activations(&mut self) {
        let decay_rate = 0.6; // membrane potential decay per turn
        for prog in &mut self.programs {
            prog.activation_level *= decay_rate;
            if prog.activation_level < 0.05 {
                prog.activation_level = 0.0;
                prog.refractory = false;
            }
        }
    }

    /// Get the effective centroid for retrieval, incorporating session drift.
    /// This is the "working centroid" that adapts to in-context queries.
    pub fn effective_centroid(&self, program_idx: usize) -> Vec<f32> {
        if program_idx >= self.programs.len() {
            return Vec::new();
        }
        let prog = &self.programs[program_idx];
        if prog.session_drift.is_empty() || prog.session_drift.iter().all(|&v| v == 0.0) {
            return prog.ema_centroid.clone();
        }
        prog.ema_centroid.iter()
            .zip(prog.session_drift.iter())
            .map(|(&base, &drift)| base + drift)
            .collect()
    }

    /// Compute a retrieval bias for a program based on its multi-timescale state.
    /// Returns a multiplier in [0.3, 1.5] that adjusts the raw similarity score:
    ///   > 1.0 = boost (high quality, high reliability, low activation)
    ///   < 1.0 = suppress (low quality, refractory, over-activated)
    pub fn retrieval_bias(&self, program_idx: usize) -> f32 {
        self.retrieval_bias_with_novelty(program_idx, self.novelty_factor)
    }

    /// Retrieval bias with configurable novelty pressure.
    /// `novelty_factor` scales the refractory/activation penalties:
    ///   1.0 = default (sentiment brains), 2.0+ = strong novelty (chat brains).
    pub fn retrieval_bias_with_novelty(&self, program_idx: usize, novelty_factor: f32) -> f32 {
        if program_idx >= self.programs.len() { return 1.0; }
        let prog = &self.programs[program_idx];

        // Persistent: quality and reliability boost/suppress
        let quality_factor = 1.0 + prog.quality_score * 0.15; // [-0.85, 1.15]
        let reliability_factor = 0.7 + prog.reliability * 0.6;  // [0.7, 1.3]

        // Short-term: refractory suppression prevents monoculture
        // novelty_factor amplifies suppression for chat brains (e.g. 2.0 → 0.4 penalty)
        let base_refractory = 0.3 * novelty_factor;
        let refractory_penalty = if prog.refractory {
            (1.0 - base_refractory).max(0.2)
        } else { 1.0 };
        let activation_suppress = (prog.activation_level * 0.3 * novelty_factor).min(0.5);
        let activation_penalty = 1.0 - activation_suppress;

        (quality_factor * reliability_factor * refractory_penalty * activation_penalty)
            .clamp(0.3, 1.5)
    }

    /// Commit session drift to persistent centroid (end-of-session consolidation).
    /// Only commits if the program was accessed enough times in the session
    /// to indicate genuine in-context learning, not noise.
    pub fn consolidate_session(&mut self, min_session_hits: u32) {
        let consolidation_alpha = 0.02; // very gentle persistent drift
        for prog in &mut self.programs {
            if prog.session_hits >= min_session_hits && !prog.session_drift.is_empty() {
                let avg_quality = if prog.session_hits > 0 {
                    prog.session_quality_sum / prog.session_hits as f32
                } else { 0.0 };

                // Only consolidate if session quality was net-positive
                if avg_quality > 0.0 {
                    let dim = prog.ema_centroid.len().min(prog.session_drift.len());
                    for i in 0..dim {
                        prog.ema_centroid[i] += prog.session_drift[i] * consolidation_alpha;
                    }
                }
            }
            // Reset session state regardless
            prog.session_drift.clear();
            prog.session_hits = 0;
            prog.session_quality_sum = 0.0;
            prog.activation_level = 0.0;
            prog.refractory = false;
        }
    }

    /// Inject a user correction into the lattice. Degrades the wrong program
    /// and either reinforces a nearby existing program or spawns a new one
    /// with the correction text and query embedding.
    pub fn inject_correction(
        &mut self,
        wrong_program_idx: Option<usize>,
        embedding: &[f32],
        correction_text: &str,
    ) {
        // Degrade the wrong program
        if let Some(idx) = wrong_program_idx {
            if idx < self.programs.len() {
                self.apply_feedback(idx, false, 0.8);
            }
        }

        // Check if an existing program is close enough to absorb the correction
        let token_ids = self.dictionary.encode(correction_text);
        if let Some((nearest_idx, similarity)) = self.nearest_program(embedding) {
            if similarity >= 0.92 {
                // Reinforce existing neighbor: shift centroid toward query
                let prog = &mut self.programs[nearest_idx];
                let alpha = self.learning_rate;
                for (i, v) in embedding.iter().enumerate() {
                    if i < prog.ema_centroid.len() {
                        prog.ema_centroid[i] = prog.ema_centroid[i] * (1.0 - alpha) + v * alpha;
                    }
                }
                prog.token_sequence = token_ids;
                prog.verbatim_display_text = Some(correction_text.to_string());
                prog.activation_count += 1;
                self.apply_feedback(nearest_idx, true, 0.9);
                return;
            }
        }

        // Spawn new program for the correction
        let lattice_sig = E8Lattice::quantize_64d(embedding);
        self.programs.push(BehavioralProgram {
            centroid: embedding.to_vec(),
            lattice_signature: lattice_sig,
            token_sequence: token_ids,
            verbatim_display_text: Some(correction_text.to_string()),
            activation_count: 1,
            ema_centroid: embedding.to_vec(),
            coherence: 1.0,
            habituation: 0.0,
            quality_score: 0.3,
            reliability: 0.7,
            total_retrievals: 0,
            session_drift: Vec::new(),
            session_hits: 0,
            session_quality_sum: 0.0,
            activation_level: 0.0,
            refractory: false,
        });
        self.wave = WaveState::new(self.programs.len());
    }

    /// Total memory footprint estimate in bytes.
    pub fn memory_bytes(&self) -> usize {
        let prog_bytes: usize = self.programs.iter().map(|p| {
            p.centroid.len() * 4
                + p.lattice_signature.len() * 4
                + p.token_sequence.len() * 2
                + p.ema_centroid.len() * 4
                + p.verbatim_display_text.as_ref().map(|s| s.len()).unwrap_or(0)
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
                verbatim_display_text: None,
                activation_count: 1,
                ema_centroid: centroid.clone(),
                coherence: 1.0,
                habituation: 0.0,
                quality_score: 0.0,
                reliability: 0.5,
                total_retrievals: 0,
                session_drift: Vec::new(),
                session_hits: 0,
                session_quality_sum: 0.0,
                activation_level: 0.0,
                refractory: false,
            });
        }
        lattice.wave = WaveState::new(lattice.programs.len());
        lattice
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for i in 0..a.len().min(b.len()) {
        let ai = a[i] as f64;
        let bi = b[i] as f64;
        dot += ai * bi;
        na += ai * ai;
        nb += bi * bi;
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom < 1e-20 { 0.0 } else { (dot / denom) as f32 }
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

    // ===================================================================
    // Continuum foundation: multi-timescale state management
    // ===================================================================

    fn make_two_program_lattice() -> InfraciliaryLattice {
        let dict = test_dict();
        let mut lattice = InfraciliaryLattice::new(dict);
        lattice.develop(&[
            (test_embedding(1.0), "reset your password".to_string()),
            (test_embedding(100.0), "implement binary search".to_string()),
        ], 0.99);
        assert!(lattice.program_count() >= 2);
        lattice
    }

    #[test]
    fn test_display_text_returns_verbatim_when_dict_folds_tokens() {
        // Vocabulary too small: encode() maps unknown headline tokens by edit distance.
        let dict = TokenDictionary::build(&["hello", "world"], 50);
        let mut lattice = InfraciliaryLattice::new(dict.clone());
        let line = "Kalshi wins in Arizona".to_string();
        lattice.develop(&[(test_embedding(42.0), line.clone())], 0.99);
        assert_eq!(lattice.program_count(), 1);
        let prog = &lattice.programs[0];
        assert_eq!(prog.verbatim_display_text.as_deref(), Some(line.as_str()));
        assert_eq!(prog.display_text(&dict), line);
        assert_ne!(
            dict.decode(&prog.token_sequence),
            line,
            "decode(token_sequence) should not round-trip OOV training text"
        );
    }

    #[test]
    fn test_new_programs_have_neutral_state() {
        let lattice = make_two_program_lattice();
        for prog in &lattice.programs {
            assert_eq!(prog.quality_score, 0.0, "new programs start with neutral quality");
            assert!((prog.reliability - 0.5).abs() < 0.01, "new programs start with 0.5 reliability");
            assert_eq!(prog.total_retrievals, 0, "no retrievals yet");
            assert_eq!(prog.activation_level, 0.0, "no activation yet");
            assert!(!prog.refractory, "not refractory");
        }
    }

    #[test]
    fn test_begin_session_resets_volatile_state() {
        let mut lattice = make_two_program_lattice();

        // Simulate some activity
        lattice.on_retrieval(0, &test_embedding(1.1));
        assert_eq!(lattice.programs[0].activation_level, 1.0);
        assert!(lattice.programs[0].refractory);
        assert_eq!(lattice.programs[0].session_hits, 1);

        // Begin new session
        lattice.begin_session();
        for prog in &lattice.programs {
            assert_eq!(prog.activation_level, 0.0, "activation reset");
            assert!(!prog.refractory, "refractory reset");
            assert_eq!(prog.session_hits, 0, "session hits reset");
            assert!(!prog.session_drift.is_empty(), "session drift initialized");
            assert!(prog.session_drift.iter().all(|&v| v == 0.0), "session drift zeroed");
        }
    }

    #[test]
    fn test_on_retrieval_updates_all_timescales() {
        let mut lattice = make_two_program_lattice();
        lattice.begin_session();

        let query = test_embedding(1.1);
        lattice.on_retrieval(0, &query);

        let prog = &lattice.programs[0];
        assert_eq!(prog.total_retrievals, 1, "persistent retrieval count incremented");
        assert_eq!(prog.session_hits, 1, "session hit count incremented");
        assert_eq!(prog.activation_level, 1.0, "activation set to 1.0");
        assert!(prog.refractory, "refractory flag set");
        assert!(prog.session_drift.iter().any(|&v| v != 0.0),
            "session drift should shift toward query");

        // Second program should be untouched
        let other = &lattice.programs[1];
        assert_eq!(other.total_retrievals, 0);
        assert_eq!(other.session_hits, 0);
        assert_eq!(other.activation_level, 0.0);
    }

    #[test]
    fn test_session_drift_accumulates_toward_queries() {
        let mut lattice = make_two_program_lattice();
        lattice.begin_session();

        let base_centroid = lattice.programs[0].ema_centroid.clone();

        // Hit program 0 multiple times with slightly different queries
        for offset in &[1.05, 1.1, 1.15] {
            lattice.on_retrieval(0, &test_embedding(*offset));
        }

        assert_eq!(lattice.programs[0].session_hits, 3);

        // Effective centroid should be shifted from base
        let effective = lattice.effective_centroid(0);
        let drift_magnitude: f32 = effective.iter()
            .zip(base_centroid.iter())
            .map(|(e, b)| (e - b) * (e - b))
            .sum::<f32>()
            .sqrt();

        assert!(drift_magnitude > 0.0,
            "effective centroid should drift from base after 3 hits");

        // Persistent centroid should be unchanged
        assert_eq!(lattice.programs[0].ema_centroid, base_centroid,
            "persistent centroid must not change from session drift");
    }

    #[test]
    fn test_decay_activations_reduces_levels() {
        let mut lattice = make_two_program_lattice();
        lattice.begin_session();

        lattice.on_retrieval(0, &test_embedding(1.1));
        assert_eq!(lattice.programs[0].activation_level, 1.0);
        assert!(lattice.programs[0].refractory);

        // One decay step
        lattice.decay_activations();
        let level_after_1 = lattice.programs[0].activation_level;
        assert!(level_after_1 < 1.0 && level_after_1 > 0.0,
            "activation should decay but not vanish: {}", level_after_1);

        // Several more decay steps should clear it
        for _ in 0..10 {
            lattice.decay_activations();
        }
        assert_eq!(lattice.programs[0].activation_level, 0.0,
            "activation should reach zero after many decay steps");
        assert!(!lattice.programs[0].refractory,
            "refractory should clear when activation drops below threshold");
    }

    #[test]
    fn test_apply_feedback_positive_increases_quality() {
        let mut lattice = make_two_program_lattice();

        let q_before = lattice.programs[0].quality_score;
        let r_before = lattice.programs[0].reliability;

        lattice.apply_feedback(0, true, 0.8);

        assert!(lattice.programs[0].quality_score > q_before,
            "positive feedback should increase quality: {} → {}",
            q_before, lattice.programs[0].quality_score);
        assert!(lattice.programs[0].reliability > r_before,
            "accept should increase reliability: {} → {}",
            r_before, lattice.programs[0].reliability);
    }

    #[test]
    fn test_apply_feedback_negative_decreases_quality() {
        let mut lattice = make_two_program_lattice();

        // Give initial positive feedback so quality is above zero
        lattice.apply_feedback(0, true, 0.9);
        lattice.apply_feedback(0, true, 0.9);
        let q_before = lattice.programs[0].quality_score;
        let r_before = lattice.programs[0].reliability;

        lattice.apply_feedback(0, false, 0.7);

        assert!(lattice.programs[0].quality_score < q_before,
            "negative feedback should decrease quality: {} → {}",
            q_before, lattice.programs[0].quality_score);
        assert!(lattice.programs[0].reliability < r_before,
            "reject should decrease reliability: {} → {}",
            r_before, lattice.programs[0].reliability);
    }

    #[test]
    fn test_apply_feedback_quality_clamped() {
        let mut lattice = make_two_program_lattice();

        // Lots of positive feedback
        for _ in 0..100 {
            lattice.apply_feedback(0, true, 1.0);
        }
        assert!(lattice.programs[0].quality_score <= 1.0,
            "quality must be clamped to 1.0");
        assert!(lattice.programs[0].reliability <= 1.0,
            "reliability must be clamped to 1.0");

        // Lots of negative feedback
        for _ in 0..200 {
            lattice.apply_feedback(0, false, 1.0);
        }
        assert!(lattice.programs[0].quality_score >= -1.0,
            "quality must be clamped to -1.0");
        assert!(lattice.programs[0].reliability >= 0.0,
            "reliability must be clamped to 0.0");
    }

    #[test]
    fn test_retrieval_bias_neutral_for_fresh_programs() {
        let lattice = make_two_program_lattice();
        let bias = lattice.retrieval_bias(0);
        // quality=0 → factor 1.0, reliability=0.5 → factor 1.0,
        // no activation, no refractory
        assert!((bias - 1.0).abs() < 0.15,
            "fresh program bias should be near 1.0, got {}", bias);
    }

    #[test]
    fn test_retrieval_bias_rewards_high_quality() {
        let mut lattice = make_two_program_lattice();

        // Build up quality on program 0
        for _ in 0..20 {
            lattice.apply_feedback(0, true, 0.9);
        }

        let bias_good = lattice.retrieval_bias(0);
        let bias_neutral = lattice.retrieval_bias(1);

        assert!(bias_good > bias_neutral,
            "high-quality program should have higher bias: {} vs {}",
            bias_good, bias_neutral);
    }

    #[test]
    fn test_retrieval_bias_suppresses_refractory() {
        let mut lattice = make_two_program_lattice();
        lattice.begin_session();

        let bias_before = lattice.retrieval_bias(0);

        lattice.on_retrieval(0, &test_embedding(1.1));
        let bias_after = lattice.retrieval_bias(0);

        assert!(bias_after < bias_before,
            "refractory program should have lower bias: {} → {}",
            bias_before, bias_after);
    }

    #[test]
    fn test_refractory_forces_diversity() {
        let mut lattice = make_two_program_lattice();
        lattice.begin_session();

        // Fire program 0
        lattice.on_retrieval(0, &test_embedding(1.1));

        // Program 0 is refractory, program 1 is not
        let bias_0 = lattice.retrieval_bias(0);
        let bias_1 = lattice.retrieval_bias(1);

        assert!(bias_0 < bias_1,
            "refractory program 0 should have lower bias than fresh program 1: {} vs {}",
            bias_0, bias_1);
    }

    #[test]
    fn test_consolidate_session_commits_drift() {
        let mut lattice = make_two_program_lattice();
        lattice.begin_session();

        let original_centroid = lattice.programs[0].ema_centroid.clone();

        // Simulate a session with multiple positive hits
        for _ in 0..5 {
            lattice.on_retrieval(0, &test_embedding(1.2));
            lattice.apply_feedback(0, true, 0.8);
        }

        assert_eq!(lattice.programs[0].session_hits, 5);

        // Consolidate with min_hits=3 (should commit)
        lattice.consolidate_session(3);

        let consolidated_centroid = &lattice.programs[0].ema_centroid;
        let drift: f32 = consolidated_centroid.iter()
            .zip(original_centroid.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            .sqrt();

        assert!(drift > 0.0,
            "consolidation should shift persistent centroid slightly");

        // Session state should be reset after consolidation
        assert_eq!(lattice.programs[0].session_hits, 0);
        assert!(lattice.programs[0].session_drift.is_empty());
        assert_eq!(lattice.programs[0].activation_level, 0.0);
    }

    #[test]
    fn test_consolidate_session_skips_low_hit_programs() {
        let mut lattice = make_two_program_lattice();
        lattice.begin_session();

        let original_centroid = lattice.programs[0].ema_centroid.clone();

        // Only 1 hit (below min_hits threshold of 3)
        lattice.on_retrieval(0, &test_embedding(1.5));
        lattice.apply_feedback(0, true, 0.9);

        lattice.consolidate_session(3);

        assert_eq!(lattice.programs[0].ema_centroid, original_centroid,
            "low-hit programs should not have their centroid modified");
    }

    #[test]
    fn test_consolidate_session_skips_negative_quality() {
        let mut lattice = make_two_program_lattice();
        lattice.begin_session();

        let original_centroid = lattice.programs[0].ema_centroid.clone();

        // Multiple hits but all negative feedback
        for _ in 0..5 {
            lattice.on_retrieval(0, &test_embedding(1.2));
            lattice.apply_feedback(0, false, 0.5);
        }

        lattice.consolidate_session(3);

        assert_eq!(lattice.programs[0].ema_centroid, original_centroid,
            "programs with net-negative session quality should not consolidate drift");
    }

    #[test]
    fn test_effective_centroid_without_session_is_base() {
        let lattice = make_two_program_lattice();
        let effective = lattice.effective_centroid(0);
        assert_eq!(effective, lattice.programs[0].ema_centroid,
            "without session, effective centroid should equal base centroid");
    }

    #[test]
    fn test_effective_centroid_with_session_differs() {
        let mut lattice = make_two_program_lattice();
        lattice.begin_session();

        lattice.on_retrieval(0, &test_embedding(1.5));
        lattice.on_retrieval(0, &test_embedding(1.5));

        let effective = lattice.effective_centroid(0);
        let base = &lattice.programs[0].ema_centroid;

        let diff: f32 = effective.iter()
            .zip(base.iter())
            .map(|(e, b)| (e - b).abs())
            .sum();

        assert!(diff > 0.0,
            "effective centroid should differ from base after session hits");
    }

    #[test]
    fn test_multi_session_lifecycle() {
        let mut lattice = make_two_program_lattice();

        // === Session 1: positive experience with program 0 ===
        lattice.begin_session();
        for _ in 0..5 {
            lattice.on_retrieval(0, &test_embedding(1.1));
            lattice.apply_feedback(0, true, 0.85);
            lattice.decay_activations();
        }
        lattice.consolidate_session(3);

        let q_after_s1 = lattice.programs[0].quality_score;
        let r_after_s1 = lattice.programs[0].reliability;
        assert!(q_after_s1 > 0.0, "quality should be positive after good session");
        assert!(r_after_s1 > 0.5, "reliability should increase from 0.5");
        assert_eq!(lattice.programs[0].total_retrievals, 5);

        // === Session 2: negative experience ===
        lattice.begin_session();
        for _ in 0..4 {
            lattice.on_retrieval(0, &test_embedding(1.3));
            lattice.apply_feedback(0, false, 0.6);
            lattice.decay_activations();
        }
        lattice.consolidate_session(3);

        let q_after_s2 = lattice.programs[0].quality_score;
        let r_after_s2 = lattice.programs[0].reliability;
        assert!(q_after_s2 < q_after_s1, "quality should decrease after bad session");
        assert!(r_after_s2 < r_after_s1, "reliability should decrease");
        assert_eq!(lattice.programs[0].total_retrievals, 9);

        // Program 1 should be completely unaffected
        assert_eq!(lattice.programs[1].quality_score, 0.0);
        assert_eq!(lattice.programs[1].total_retrievals, 0);
    }

    #[test]
    fn test_retrieval_bias_range() {
        let mut lattice = make_two_program_lattice();
        lattice.begin_session();

        const MIN_BIAS: f32 = 0.3;
        const MAX_BIAS: f32 = 1.5;

        // Test every combination of states
        // Fresh
        let b = lattice.retrieval_bias(0);
        assert!(b >= MIN_BIAS && b <= MAX_BIAS, "bias out of range: {}", b);

        // After positive feedback
        for _ in 0..50 { lattice.apply_feedback(0, true, 1.0); }
        let b = lattice.retrieval_bias(0);
        assert!(b >= MIN_BIAS && b <= MAX_BIAS, "bias out of range after positive: {}", b);

        // After retrieval (refractory)
        lattice.on_retrieval(0, &test_embedding(1.0));
        let b = lattice.retrieval_bias(0);
        assert!(b >= MIN_BIAS && b <= MAX_BIAS, "bias out of range when refractory: {}", b);

        // After lots of negative feedback
        for _ in 0..100 { lattice.apply_feedback(0, false, 1.0); }
        let b = lattice.retrieval_bias(0);
        assert!(b >= MIN_BIAS && b <= MAX_BIAS, "bias out of range after negative: {}", b);
    }

    #[test]
    fn test_serde_preserves_persistent_drops_volatile() {
        let mut lattice = make_two_program_lattice();
        lattice.begin_session();

        // Accumulate persistent state
        lattice.on_retrieval(0, &test_embedding(1.1));
        lattice.apply_feedback(0, true, 0.9);
        lattice.apply_feedback(0, true, 0.9);

        let q_before = lattice.programs[0].quality_score;
        let r_before = lattice.programs[0].reliability;
        let retrievals_before = lattice.programs[0].total_retrievals;

        // Serialize and deserialize via JSON
        let serialized = serde_json::to_string(&lattice).expect("serialize");
        let restored: InfraciliaryLattice = serde_json::from_str(&serialized).expect("deserialize");

        // Persistent state preserved
        assert!((restored.programs[0].quality_score - q_before).abs() < 1e-6,
            "quality_score should survive serialization");
        assert!((restored.programs[0].reliability - r_before).abs() < 1e-6,
            "reliability should survive serialization");
        assert_eq!(restored.programs[0].total_retrievals, retrievals_before,
            "total_retrievals should survive serialization");

        // Volatile state dropped (serde(skip))
        assert_eq!(restored.programs[0].activation_level, 0.0,
            "activation_level should be zero after deserialization");
        assert!(!restored.programs[0].refractory,
            "refractory should be false after deserialization");
        assert!(restored.programs[0].session_drift.is_empty(),
            "session_drift should be empty after deserialization");
        assert_eq!(restored.programs[0].session_hits, 0,
            "session_hits should be zero after deserialization");
    }

    // ===================================================================
    // Continuum: inject_correction (online learning from user feedback)
    // ===================================================================

    /// Produces an embedding orthogonal to the sin-based test_embedding by
    /// alternating sign and using cos, avoiding periodicity collisions.
    fn distant_embedding() -> Vec<f32> {
        (0..crate::dimension::language::DEFAULT_BRIDGE_DIM)
            .map(|i| {
                let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
                sign * ((i as f32) * 0.73 + 3.14).cos()
            }).collect()
    }

    #[test]
    fn test_inject_correction_spawns_new_program() {
        let mut lattice = make_two_program_lattice();
        let before = lattice.program_count();

        let far = distant_embedding();
        lattice.inject_correction(
            Some(0),
            &far,
            "completely new response",
        );

        assert_eq!(lattice.program_count(), before + 1,
            "distant correction should spawn a new program");

        let new_prog = lattice.programs.last().unwrap();
        assert!(new_prog.quality_score > 0.0,
            "new correction program should start with positive quality");
        assert!(new_prog.reliability > 0.5,
            "new correction program should start with above-average reliability");
    }

    #[test]
    fn test_inject_correction_degrades_wrong_program() {
        let mut lattice = make_two_program_lattice();

        lattice.apply_feedback(0, true, 0.9);
        lattice.apply_feedback(0, true, 0.9);
        let q_before = lattice.programs[0].quality_score;

        let far = distant_embedding();
        lattice.inject_correction(
            Some(0),
            &far,
            "corrected response",
        );

        assert!(lattice.programs[0].quality_score < q_before,
            "wrong program should be degraded by correction");
    }

    #[test]
    fn test_inject_correction_reinforces_nearby() {
        let mut lattice = make_two_program_lattice();
        let before = lattice.program_count();

        let near_emb = test_embedding(1.001);
        lattice.inject_correction(
            Some(1),
            &near_emb,
            "updated nearby response",
        );

        assert_eq!(lattice.program_count(), before,
            "correction near existing program should reinforce, not spawn");
    }

    #[test]
    fn test_inject_correction_none_wrong_idx() {
        let mut lattice = make_two_program_lattice();
        let before = lattice.program_count();

        let far = distant_embedding();
        lattice.inject_correction(
            None,
            &far,
            "new knowledge",
        );

        assert_eq!(lattice.program_count(), before + 1,
            "should still spawn even without a wrong program index");
    }
}
