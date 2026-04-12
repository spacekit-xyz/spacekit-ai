//! Reasoning Engine — hippocampal-prefrontal circuit for compositional inference.
//!
//! The brain does not retrieve single memories and return them verbatim.
//! It activates *multiple* regions simultaneously, allows them to compete
//! and cooperate through wave propagation, and composes a novel response
//! from the settled activation pattern.
//!
//! This module implements that process using Growformer primitives:
//!
//! - **Cognitive Map** (hippocampus): a graph of programs across all groups,
//!   connected by structural similarity (Cl(1,7) bivector cosine). Enables
//!   traversal from one concept to structurally related concepts in other
//!   domains.
//!
//! - **Multi-group Activation** (parietal integration): queries all groups
//!   simultaneously, collecting a "spread of activation" across the full
//!   knowledge base.
//!
//! - **Wave Settling** (prefrontal working memory): iterative wave propagation
//!   on the cognitive map — activated nodes boost neighbors, competing
//!   hypotheses inhibit each other, and the pattern converges to a coherent
//!   "thought."
//!
//! - **Transfer Rotor Composition** (frontal reasoning): the Cl(1,7) transfer
//!   rotor maps structure from domain A to domain B — genuine analogical
//!   reasoning expressed as geometry.
//!
//! - **Fragment Assembly** (motor output): the settled activation pattern
//!   selects fragments from multiple groups, composed via the Hopf
//!   composition table into a novel response.
//!
//! Zero backpropagation. One-pass construction. Continual-learning compatible.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::clifford::{
    embed_bridge_vector, structural_fingerprint, structural_similarity,
    apply_group_rotor, extract_conditioning, transfer_rotor,
    Multivector, Rotor, GroupRotor,
};
use crate::dimension::group_gen::IndexedGenEnv;
use crate::dimension::paramecium::InfraciliaryLattice;
use crate::spectral::TokenDictionary;

/// A node in the cognitive map: one program from one group.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CognitiveNode {
    pub group_idx: usize,
    pub program_idx: usize,
    pub fingerprint: [f32; 28],
    pub centroid: Vec<f32>,
    pub token_sequence: Vec<u16>,
    /// Mirrors [`crate::dimension::paramecium::BehavioralProgram::verbatim_display_text`].
    #[serde(default)]
    pub verbatim_display_text: Option<String>,
}

impl CognitiveNode {
    pub fn display_text(&self, dict: &TokenDictionary) -> String {
        if let Some(ref v) = self.verbatim_display_text {
            if !v.is_empty() {
                return v.clone();
            }
        }
        dict.decode(&self.token_sequence)
    }
}

/// An edge connecting two nodes in the cognitive map.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CognitiveEdge {
    pub source: usize,
    pub target: usize,
    pub structural_sim: f32,
}

/// The cognitive map: a graph of programs across all groups, connected
/// by structural similarity in Cl(8) bivector space.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CognitiveMap {
    pub nodes: Vec<CognitiveNode>,
    pub edges: Vec<CognitiveEdge>,
    /// adjacency[node_idx] = [(neighbor_idx, weight)]
    pub adjacency: Vec<Vec<(usize, f32)>>,
    /// Mapping from (group_idx, program_idx) → node index.
    pub index: HashMap<(usize, usize), usize>,
}

const EDGE_THRESHOLD: f32 = 0.25;
const MAX_EDGES_PER_NODE: usize = 12;

impl CognitiveMap {
    /// Build the cognitive map from all group gen envs.
    /// One-pass: compute pairwise structural similarity, keep edges above threshold.
    pub fn build(
        group_envs: &HashMap<usize, IndexedGenEnv>,
        group_rotors: &HashMap<usize, GroupRotor>,
    ) -> Self {
        let mut nodes = Vec::new();
        let mut index = HashMap::new();

        for (&gidx, env) in group_envs {
            let rotor = group_rotors.get(&gidx).map(|gr| gr.rotor());
            for (pidx, prog) in env.lattice.programs.iter().enumerate() {
                let mv = embed_bridge_vector(&prog.ema_centroid);
                let fp = structural_fingerprint(&mv);
                let node_idx = nodes.len();
                index.insert((gidx, pidx), node_idx);
                nodes.push(CognitiveNode {
                    group_idx: gidx,
                    program_idx: pidx,
                    fingerprint: fp,
                    centroid: prog.ema_centroid.clone(),
                    token_sequence: prog.token_sequence.clone(),
                    verbatim_display_text: prog.verbatim_display_text.clone(),
                });
            }
        }

        let n = nodes.len();
        let mut adjacency: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n];
        let mut edges = Vec::new();

        for i in 0..n {
            let mv_i = Multivector::from_bivector_fp(&nodes[i].fingerprint);
            let mut scored: Vec<(usize, f32)> = Vec::new();
            for j in 0..n {
                if i == j { continue; }
                // Cross-group edges are more valuable for reasoning
                if nodes[i].group_idx == nodes[j].group_idx { continue; }
                let mv_j = Multivector::from_bivector_fp(&nodes[j].fingerprint);
                let sim = structural_similarity(&mv_i, &mv_j);
                if sim >= EDGE_THRESHOLD {
                    scored.push((j, sim));
                }
            }
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            scored.truncate(MAX_EDGES_PER_NODE);
            for &(j, sim) in &scored {
                edges.push(CognitiveEdge { source: i, target: j, structural_sim: sim });
                adjacency[i].push((j, sim));
            }
        }

        CognitiveMap { nodes, edges, adjacency, index }
    }

    pub fn node_count(&self) -> usize { self.nodes.len() }
    pub fn edge_count(&self) -> usize { self.edges.len() }
}

/// An activation during multi-group query.
#[derive(Clone, Debug)]
pub struct Activation {
    pub node_idx: usize,
    pub group_idx: usize,
    pub similarity: f32,
    pub text: String,
}

/// Result of reasoning over a prompt.
#[derive(Clone, Debug)]
pub struct ReasoningResult {
    pub text: String,
    pub confidence: f32,
    pub source_groups: Vec<usize>,
    pub wave_rounds: usize,
    pub fragments_used: usize,
}

/// The reasoning engine: performs multi-group activation, wave settling,
/// and compositional assembly.
#[derive(Clone, Serialize, Deserialize)]
pub struct ReasoningEngine {
    pub cognitive_map: CognitiveMap,
    /// Per-group dictionaries for decoding programs from different groups.
    pub group_dictionaries: HashMap<usize, TokenDictionary>,
    /// Settling rounds (default 4, like prefrontal thalamocortical loops).
    pub settling_rounds: usize,
    /// Top-K activations per group during initial query.
    pub activations_per_group: usize,
    /// Lateral inhibition strength (competition between hypotheses).
    pub inhibition_strength: f32,
    /// Spreading activation decay per hop.
    pub spread_decay: f32,
}

const DEFAULT_SETTLING_ROUNDS: usize = 4;
const DEFAULT_K_PER_GROUP: usize = 3;
const DEFAULT_INHIBITION: f32 = 0.15;
const DEFAULT_SPREAD_DECAY: f32 = 0.6;

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-10 || nb < 1e-10 { return 0.0; }
    dot / (na * nb)
}

impl ReasoningEngine {
    pub fn new(
        cognitive_map: CognitiveMap,
        group_dictionaries: HashMap<usize, TokenDictionary>,
    ) -> Self {
        Self {
            cognitive_map,
            group_dictionaries,
            settling_rounds: DEFAULT_SETTLING_ROUNDS,
            activations_per_group: DEFAULT_K_PER_GROUP,
            inhibition_strength: DEFAULT_INHIBITION,
            spread_decay: DEFAULT_SPREAD_DECAY,
        }
    }

    /// Phase 1: Multi-group activation.
    /// Query all groups simultaneously, find top-K nearest programs per group.
    fn activate_all_groups(
        &self,
        cond: &[f32],
        group_envs: &HashMap<usize, IndexedGenEnv>,
    ) -> Vec<Activation> {
        let mut activations = Vec::new();
        for (&gidx, env) in group_envs {
            let dict = self.group_dictionaries.get(&gidx);
            let mut scored: Vec<(usize, f32)> = env.lattice.programs.iter().enumerate()
                .map(|(i, prog)| (i, cosine_sim(cond, &prog.ema_centroid)))
                .collect();
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            for &(pidx, sim) in scored.iter().take(self.activations_per_group) {
                if sim < 0.1 { break; }
                let text = dict
                    .map(|d| env.lattice.programs[pidx].display_text(d))
                    .unwrap_or_default();
                let node_idx = self.cognitive_map.index
                    .get(&(gidx, pidx))
                    .copied()
                    .unwrap_or(0);
                activations.push(Activation {
                    node_idx,
                    group_idx: gidx,
                    similarity: sim,
                    text,
                });
            }
        }
        activations
    }

    /// Phase 2: Wave settling on the cognitive map.
    /// Iterative propagation: activated nodes boost neighbors, lateral
    /// inhibition suppresses competing hypotheses.
    fn settle_wave(
        &self,
        initial_activations: &[Activation],
    ) -> Vec<f32> {
        let n = self.cognitive_map.nodes.len();
        if n == 0 { return Vec::new(); }

        let mut energy = vec![0.0f32; n];

        // Seed from initial activations
        for act in initial_activations {
            if act.node_idx < n {
                energy[act.node_idx] = act.similarity;
            }
        }

        // Iterative settling
        for _round in 0..self.settling_rounds {
            let prev = energy.clone();

            // Spreading activation: each node boosts its neighbors
            for i in 0..n {
                if prev[i] < 0.05 { continue; }
                for &(neighbor, edge_weight) in &self.cognitive_map.adjacency[i] {
                    let spread = prev[i] * edge_weight * self.spread_decay;
                    energy[neighbor] = (energy[neighbor] + spread).min(1.0);
                }
            }

            // Lateral inhibition: nodes in the same group compete
            let mut group_max: HashMap<usize, f32> = HashMap::new();
            for (i, e) in energy.iter().enumerate() {
                let gidx = self.cognitive_map.nodes[i].group_idx;
                let entry = group_max.entry(gidx).or_insert(0.0);
                if *e > *entry { *entry = *e; }
            }
            for (i, e) in energy.iter_mut().enumerate() {
                let gidx = self.cognitive_map.nodes[i].group_idx;
                let max_in_group = group_max.get(&gidx).copied().unwrap_or(0.0);
                if max_in_group > 0.0 && *e < max_in_group {
                    let suppression = (max_in_group - *e) * self.inhibition_strength;
                    *e = (*e - suppression).max(0.0);
                }
            }

            // Normalize to prevent runaway
            let max_e = energy.iter().cloned().fold(0.0f32, f32::max);
            if max_e > 1.0 {
                for e in &mut energy { *e /= max_e; }
            }
        }

        energy
    }

    /// Phase 3: Compose response from settled activation pattern.
    /// Takes the top activated nodes from different groups and blends
    /// their response fragments using transfer rotors.
    fn compose_from_settled(
        &self,
        cond: &[f32],
        energy: &[f32],
        group_envs: &HashMap<usize, IndexedGenEnv>,
        group_rotors: &HashMap<usize, GroupRotor>,
    ) -> ReasoningResult {
        if energy.is_empty() || self.cognitive_map.nodes.is_empty() {
            return ReasoningResult {
                text: String::new(),
                confidence: 0.0,
                source_groups: Vec::new(),
                wave_rounds: self.settling_rounds,
                fragments_used: 0,
            };
        }

        // Score nodes by BOTH wave energy AND direct input relevance.
        // This prevents a high-connectivity hub from winning regardless of input.
        let mut scored: Vec<(usize, f32)> = energy.iter().enumerate()
            .filter(|(_, &e)| e > 0.1)
            .map(|(i, &e)| {
                let node = &self.cognitive_map.nodes[i];
                let input_relevance = cosine_sim(cond, &node.centroid).max(0.0);
                // Blend: 40% wave energy, 60% input relevance
                let combined = 0.4 * e + 0.6 * input_relevance;
                (i, combined)
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Pick the best node per group (up to 3 groups)
        let mut best_per_group: Vec<(usize, f32)> = Vec::new();
        let mut seen_groups = std::collections::HashSet::new();
        for &(node_idx, combined_score) in &scored {
            let gidx = self.cognitive_map.nodes[node_idx].group_idx;
            if seen_groups.insert(gidx) {
                best_per_group.push((node_idx, combined_score));
                if best_per_group.len() >= 3 { break; }
            }
        }

        if best_per_group.is_empty() {
            return ReasoningResult {
                text: String::new(),
                confidence: 0.0,
                source_groups: Vec::new(),
                wave_rounds: self.settling_rounds,
                fragments_used: 0,
            };
        }

        // If only one group activated strongly, fall back to direct response
        if best_per_group.len() == 1 {
            let (node_idx, e) = best_per_group[0];
            let node = &self.cognitive_map.nodes[node_idx];
            let text = self.group_dictionaries.get(&node.group_idx)
                .map(|d| node.display_text(d))
                .unwrap_or_default();
            return ReasoningResult {
                text,
                confidence: e,
                source_groups: vec![node.group_idx],
                wave_rounds: self.settling_rounds,
                fragments_used: 1,
            };
        }

        // Multi-group composition via transfer rotors
        let source_groups: Vec<usize> = best_per_group.iter()
            .map(|(ni, _)| self.cognitive_map.nodes[*ni].group_idx)
            .collect();

        // The primary node is the strongest activation
        let (primary_node_idx, primary_energy) = best_per_group[0];
        let primary_node = &self.cognitive_map.nodes[primary_node_idx];
        let primary_text = self.group_dictionaries.get(&primary_node.group_idx)
            .map(|d| primary_node.display_text(d))
            .unwrap_or_default();

        // For each secondary group, extract transferred knowledge
        let mut transferred_fragments: Vec<(String, f32)> = Vec::new();
        let primary_rotor = group_rotors.get(&primary_node.group_idx)
            .map(|gr| gr.rotor())
            .unwrap_or_else(Rotor::identity);

        for &(node_idx, e) in best_per_group.iter().skip(1) {
            let node = &self.cognitive_map.nodes[node_idx];
            let secondary_rotor = group_rotors.get(&node.group_idx)
                .map(|gr| gr.rotor())
                .unwrap_or_else(Rotor::identity);

            // Compute transfer rotor: maps secondary domain → primary domain
            let xfer = transfer_rotor(&secondary_rotor, &primary_rotor);

            // Apply transfer to the secondary program's centroid
            let secondary_mv = embed_bridge_vector(&node.centroid);
            let transferred_mv = apply_group_rotor(&secondary_mv, &xfer);

            // Measure alignment after transfer — how well does the transferred
            // concept fit the original query?
            let input_mv = embed_bridge_vector(cond);
            let alignment = structural_similarity(&transferred_mv, &input_mv)
                .max(0.0);

            let text = self.group_dictionaries.get(&node.group_idx)
                .map(|d| node.display_text(d))
                .unwrap_or_default();

            if alignment > 0.05 && !text.is_empty() {
                transferred_fragments.push((text, e * alignment));
            }
        }

        // Assemble: primary response + transferred insights
        let mut parts: Vec<(String, f32)> = vec![(primary_text, primary_energy)];
        parts.extend(transferred_fragments);
        parts.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Compose by sentence-level interleaving
        let composed = self.interleave_fragments(&parts, cond);
        let avg_confidence = parts.iter().map(|(_, e)| e).sum::<f32>() / parts.len() as f32;
        let fragments_used = parts.len();

        ReasoningResult {
            text: composed,
            confidence: avg_confidence.clamp(0.0, 1.0),
            source_groups,
            wave_rounds: self.settling_rounds,
            fragments_used,
        }
    }

    /// Interleave fragments from multiple groups into a coherent response.
    /// Splits each fragment into sentences, scores each by relevance to
    /// the input, and selects the most relevant non-redundant sentences.
    fn interleave_fragments(&self, fragments: &[(String, f32)], cond: &[f32]) -> String {
        let mut sentences: Vec<ScoredSentence> = Vec::new();

        for (frag_idx, (text, weight)) in fragments.iter().enumerate() {
            for sent in text.split(". ").chain(text.split(".\n")) {
                let sent = sent.trim().trim_end_matches('.');
                if sent.len() < 5 { continue; }
                sentences.push(ScoredSentence {
                    text: sent.to_string(),
                    weight: *weight,
                    fragment_idx: frag_idx,
                });
            }
        }

        if sentences.is_empty() {
            return fragments.first().map(|(t, _)| t.clone()).unwrap_or_default();
        }

        // Score by weight and deduplicate
        sentences.sort_by(|a, b| b.weight.partial_cmp(&a.weight).unwrap_or(std::cmp::Ordering::Equal));

        let mut selected: Vec<String> = Vec::new();
        let mut total_len = 0;
        let max_len = 600;

        for sent in &sentences {
            if total_len + sent.text.len() > max_len { break; }
            // Rough dedup: skip if a selected sentence is very similar
            let duplicate = selected.iter().any(|s| {
                let overlap = sent.text.split_whitespace()
                    .filter(|w| s.contains(w))
                    .count();
                let total = sent.text.split_whitespace().count().max(1);
                overlap as f32 / total as f32 > 0.7
            });
            if !duplicate {
                selected.push(sent.text.clone());
                total_len += sent.text.len();
            }
        }

        if selected.is_empty() {
            return fragments.first().map(|(t, _)| t.clone()).unwrap_or_default();
        }

        selected.join(". ") + "."
    }

    /// Full reasoning pipeline: activate → settle → compose.
    pub fn reason(
        &self,
        cond: &[f32],
        group_envs: &HashMap<usize, IndexedGenEnv>,
        group_rotors: &HashMap<usize, GroupRotor>,
    ) -> ReasoningResult {
        let activations = self.activate_all_groups(cond, group_envs);
        let energy = self.settle_wave(&activations);
        self.compose_from_settled(cond, &energy, group_envs, group_rotors)
    }

    /// Check if the reasoning engine should be invoked.
    /// Only fires when there's genuine cross-domain ambiguity: the top two
    /// groups must be very close in activation AND both must be individually
    /// strong.  This prevents triggering on clear single-domain prompts.
    pub fn should_reason(
        &self,
        cond: &[f32],
        _primary_confidence: f32,
        group_envs: &HashMap<usize, IndexedGenEnv>,
    ) -> bool {
        let mut group_scores: Vec<(usize, f32)> = group_envs.iter()
            .map(|(&gidx, env)| {
                let best_sim = env.lattice.programs.iter()
                    .map(|p| cosine_sim(cond, &p.ema_centroid))
                    .fold(0.0f32, f32::max);
                (gidx, best_sim)
            })
            .collect();
        group_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        if group_scores.len() < 2 {
            return false;
        }

        let top = group_scores[0].1;
        let second = group_scores[1].1;

        // Both must be individually strong (above 0.35)
        if top < 0.35 || second < 0.35 {
            return false;
        }

        // The gap between top and second must be small (ratio > 0.85).
        // This means the prompt genuinely activates two domains nearly equally.
        if top > 0.01 {
            let ambiguity = second / top;
            ambiguity > 0.85
        } else {
            false
        }
    }
}

struct ScoredSentence {
    text: String,
    weight: f32,
    fragment_idx: usize,
}

// Helper: reconstruct a Multivector from a 28d bivector fingerprint
impl Multivector {
    pub fn from_bivector_fp(fp: &[f32; 28]) -> Self {
        let mut mv = Self::zero();
        for (i, &v) in fp.iter().enumerate() {
            mv.components[crate::clifford::GRADE_OFFSETS[2] + i] = v;
        }
        mv
    }
}

// =========================================================================
// System 2 — Deliberate Multi-Step Reasoning
// =========================================================================
//
// Extends the ReasoningEngine from fixed-round wave settling (proto-System 2)
// to variable-length deliberate chaining with intermediate state.
//
// Each step is a single lattice query, transfer rotor application, or
// fragment composition — no backprop, no autoregressive token loop.
// The inner loop chains *structural operations*, not token predictions.
//
// Biological analog: prefrontal working memory + inner speech.

/// An entry in the working memory buffer.
#[derive(Clone, Debug)]
pub struct WorkingMemoryEntry {
    /// Which cognitive map node this came from (if any).
    pub node_idx: Option<usize>,
    /// The group this entry belongs to.
    pub group_idx: usize,
    /// Embedding in bridge space.
    pub embedding: Vec<f32>,
    /// Decoded text fragment.
    pub text: String,
    /// Activation strength (how relevant this entry is to the goal).
    pub activation: f32,
    /// Which reasoning step produced this entry.
    pub step_produced: usize,
}

/// The working memory buffer: a bounded scratchpad of activated programs
/// and partial conclusions maintained across reasoning steps.
#[derive(Clone, Debug)]
pub struct WorkingMemory {
    pub entries: Vec<WorkingMemoryEntry>,
    pub capacity: usize,
    /// The original goal embedding (prompt conditioning vector).
    pub goal: Vec<f32>,
    /// Running coherence score: how well the current buffer addresses the goal.
    pub coherence: f32,
}

impl WorkingMemory {
    pub fn new(goal: Vec<f32>, capacity: usize) -> Self {
        Self {
            entries: Vec::new(),
            capacity,
            goal,
            coherence: 0.0,
        }
    }

    /// Insert an entry, evicting the weakest if at capacity.
    pub fn insert(&mut self, entry: WorkingMemoryEntry) {
        if self.entries.len() >= self.capacity {
            if let Some((min_idx, _)) = self.entries.iter().enumerate()
                .min_by(|(_, a), (_, b)| a.activation.partial_cmp(&b.activation)
                    .unwrap_or(std::cmp::Ordering::Equal))
            {
                if self.entries[min_idx].activation < entry.activation {
                    self.entries.swap_remove(min_idx);
                } else {
                    return; // new entry is weaker than everything in buffer
                }
            }
        }
        self.entries.push(entry);
    }

    /// Compute overall coherence: average cosine similarity of all entries
    /// to the goal embedding.
    pub fn update_coherence(&mut self) {
        if self.entries.is_empty() || self.goal.is_empty() {
            self.coherence = 0.0;
            return;
        }
        let total: f32 = self.entries.iter()
            .map(|e| cosine_sim(&e.embedding, &self.goal).max(0.0) * e.activation)
            .sum();
        self.coherence = total / self.entries.len() as f32;
    }

    /// Assemble working memory into a composite embedding by weighted average.
    pub fn composite_embedding(&self) -> Vec<f32> {
        if self.entries.is_empty() {
            return self.goal.clone();
        }
        let dim = self.entries[0].embedding.len();
        let mut composite = vec![0.0f32; dim];
        let mut total_weight = 0.0f32;
        for entry in &self.entries {
            for (i, v) in entry.embedding.iter().enumerate() {
                if i < dim {
                    composite[i] += v * entry.activation;
                }
            }
            total_weight += entry.activation;
        }
        if total_weight > 0.0 {
            for v in &mut composite {
                *v /= total_weight;
            }
        }
        composite
    }
}

/// The action a reasoning step can take.
#[derive(Clone, Debug)]
pub enum StepAction {
    /// Retrieve a related program from a specific group.
    Retrieve { group_idx: usize },
    /// Apply a transfer rotor to map knowledge from one domain to another.
    Transfer { source_group: usize, target_group: usize },
    /// Compose current working memory entries into a partial conclusion.
    Compose,
    /// Working memory is coherent enough; stop reasoning.
    Terminate,
}

/// Configuration for System 2 deliberate reasoning.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct System2Config {
    /// Maximum reasoning steps before forced termination.
    pub max_steps: usize,
    /// Working memory capacity (number of entries).
    pub wm_capacity: usize,
    /// Coherence threshold: terminate when working memory coherence exceeds this.
    pub coherence_threshold: f32,
    /// Minimum activation for a retrieved program to enter working memory.
    pub min_activation: f32,
    /// Transfer alignment threshold: only accept transferred knowledge above this.
    pub transfer_threshold: f32,
}

impl Default for System2Config {
    fn default() -> Self {
        Self {
            max_steps: 6,
            wm_capacity: 8,
            coherence_threshold: 0.65,
            min_activation: 0.15,
            transfer_threshold: 0.10,
        }
    }
}

/// Result of System 2 deliberate reasoning.
#[derive(Clone, Debug)]
pub struct System2Result {
    pub text: String,
    pub confidence: f32,
    pub source_groups: Vec<usize>,
    pub steps_taken: usize,
    pub working_memory_size: usize,
    pub final_coherence: f32,
    pub terminated_by: System2Termination,
}

#[derive(Clone, Debug)]
pub enum System2Termination {
    CoherenceReached,
    MaxSteps,
    NoProgress,
}

impl ReasoningEngine {
    /// System 2: deliberate multi-step reasoning with working memory.
    ///
    /// Unlike the fixed-round wave settling (System 1.5), this performs
    /// variable-length chaining where each step is a conscious decision:
    /// retrieve, transfer, compose, or terminate.
    pub fn reason_deliberate(
        &self,
        cond: &[f32],
        group_envs: &HashMap<usize, IndexedGenEnv>,
        group_rotors: &HashMap<usize, GroupRotor>,
        config: &System2Config,
    ) -> System2Result {
        let mut wm = WorkingMemory::new(cond.to_vec(), config.wm_capacity);

        // Step 0: Seed working memory with initial multi-group activations
        let activations = self.activate_all_groups(cond, group_envs);
        for act in &activations {
            if act.similarity >= config.min_activation {
                let node = self.cognitive_map.nodes.get(act.node_idx);
                let embedding = node.map(|n| n.centroid.clone()).unwrap_or_default();
                wm.insert(WorkingMemoryEntry {
                    node_idx: Some(act.node_idx),
                    group_idx: act.group_idx,
                    embedding,
                    text: act.text.clone(),
                    activation: act.similarity,
                    step_produced: 0,
                });
            }
        }
        wm.update_coherence();

        println!(
            "  [system2] seeded wm with {} entries, initial coherence={:.3}",
            wm.entries.len(),
            wm.coherence
        );

        let mut steps_taken = 0;
        let mut last_coherence = wm.coherence;
        let mut stall_count = 0;
        let terminated_by;

        loop {
            steps_taken += 1;

            if wm.coherence >= config.coherence_threshold {
                terminated_by = System2Termination::CoherenceReached;
                println!(
                    "  [system2] step {} → TERMINATE (coherence {:.3} >= {:.3})",
                    steps_taken, wm.coherence, config.coherence_threshold
                );
                break;
            }

            if steps_taken > config.max_steps {
                terminated_by = System2Termination::MaxSteps;
                println!("  [system2] step {} → TERMINATE (max steps)", steps_taken);
                break;
            }

            // Detect stalling: if coherence hasn't improved for 2 steps, stop
            if (wm.coherence - last_coherence).abs() < 0.01 {
                stall_count += 1;
                if stall_count >= 2 {
                    terminated_by = System2Termination::NoProgress;
                    println!("  [system2] step {} → TERMINATE (no progress)", steps_taken);
                    break;
                }
            } else {
                stall_count = 0;
            }
            last_coherence = wm.coherence;

            // Decide next action based on working memory state
            let action = self.select_step_action(&wm, group_envs, group_rotors, config);

            match action {
                StepAction::Retrieve { group_idx } => {
                    self.step_retrieve(&mut wm, group_idx, group_envs, config, steps_taken);
                }
                StepAction::Transfer { source_group, target_group } => {
                    self.step_transfer(
                        &mut wm,
                        source_group,
                        target_group,
                        group_envs,
                        group_rotors,
                        config,
                        steps_taken,
                    );
                }
                StepAction::Compose => {
                    self.step_compose(&mut wm, steps_taken);
                }
                StepAction::Terminate => {
                    terminated_by = System2Termination::CoherenceReached;
                    println!("  [system2] step {} → operator chose TERMINATE", steps_taken);
                    break;
                }
            }

            wm.update_coherence();
        }

        // Assemble final response from working memory
        let (text, confidence) = self.assemble_from_wm(&wm, cond);
        let source_groups: Vec<usize> = wm.entries.iter()
            .map(|e| e.group_idx)
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        System2Result {
            text,
            confidence,
            source_groups,
            steps_taken,
            working_memory_size: wm.entries.len(),
            final_coherence: wm.coherence,
            terminated_by,
        }
    }

    /// The StepOperator: given the current working memory state, decide what
    /// reasoning action to take next.
    ///
    /// Strategy:
    /// - If working memory has entries from only 1 group → Retrieve from
    ///   the most structurally related group (diversify)
    /// - If working memory has entries from 2+ groups → Transfer between
    ///   the two most activated groups (synthesize)
    /// - If transfer has been tried and coherence is moderate → Compose
    /// - If coherence is high enough → Terminate
    fn select_step_action(
        &self,
        wm: &WorkingMemory,
        group_envs: &HashMap<usize, IndexedGenEnv>,
        group_rotors: &HashMap<usize, GroupRotor>,
        config: &System2Config,
    ) -> StepAction {
        if wm.coherence >= config.coherence_threshold {
            return StepAction::Terminate;
        }

        // Count distinct groups in working memory
        let mut group_activations: HashMap<usize, f32> = HashMap::new();
        for entry in &wm.entries {
            let act = group_activations.entry(entry.group_idx).or_insert(0.0);
            *act = act.max(entry.activation);
        }

        let distinct_groups = group_activations.len();

        if distinct_groups <= 1 {
            // Need diversity: find a group not yet in working memory
            // that has high structural similarity to an existing entry
            let existing_group = wm.entries.first().map(|e| e.group_idx).unwrap_or(0);
            let best_related = self.find_related_group(existing_group, &group_activations, group_envs);
            if let Some(target_group) = best_related {
                println!("  [system2] → RETRIEVE from group {} (diversify)", target_group);
                return StepAction::Retrieve { group_idx: target_group };
            }
            // No related groups available; try compose with what we have
            return StepAction::Compose;
        }

        // Multiple groups present — check if we should transfer or compose
        let mut sorted_groups: Vec<(usize, f32)> = group_activations.into_iter().collect();
        sorted_groups.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        if sorted_groups.len() >= 2 {
            let (g1, _) = sorted_groups[0];
            let (g2, _) = sorted_groups[1];

            // Check if we've already done a transfer between these groups
            let has_transfer = wm.entries.iter().any(|e| {
                e.step_produced > 0
                    && e.text.starts_with("[transferred:")
            });

            if !has_transfer {
                println!("  [system2] → TRANSFER from group {} to group {}", g2, g1);
                return StepAction::Transfer {
                    source_group: g2,
                    target_group: g1,
                };
            }
        }

        println!("  [system2] → COMPOSE (multi-group assembly)");
        StepAction::Compose
    }

    /// Find a group structurally related to `source_group` that isn't already
    /// heavily represented in working memory.
    fn find_related_group(
        &self,
        source_group: usize,
        existing_groups: &HashMap<usize, f32>,
        group_envs: &HashMap<usize, IndexedGenEnv>,
    ) -> Option<usize> {
        // Use cognitive map edges to find structurally related groups
        let source_nodes: Vec<usize> = self.cognitive_map.nodes.iter().enumerate()
            .filter(|(_, n)| n.group_idx == source_group)
            .map(|(i, _)| i)
            .collect();

        let mut group_scores: HashMap<usize, f32> = HashMap::new();
        for &node_idx in &source_nodes {
            for &(neighbor, weight) in &self.cognitive_map.adjacency[node_idx] {
                let neighbor_group = self.cognitive_map.nodes[neighbor].group_idx;
                if !existing_groups.contains_key(&neighbor_group) {
                    let score = group_scores.entry(neighbor_group).or_insert(0.0);
                    *score = score.max(weight);
                }
            }
        }

        // Pick the most related group that we have an env for
        group_scores.into_iter()
            .filter(|(g, _)| group_envs.contains_key(g))
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(g, _)| g)
    }

    /// Retrieve step: query a specific group using the composite working memory
    /// embedding and add the best match to working memory.
    fn step_retrieve(
        &self,
        wm: &mut WorkingMemory,
        group_idx: usize,
        group_envs: &HashMap<usize, IndexedGenEnv>,
        config: &System2Config,
        step: usize,
    ) {
        let query = wm.composite_embedding();
        let env = match group_envs.get(&group_idx) {
            Some(e) => e,
            None => return,
        };

        let dict = self.group_dictionaries.get(&group_idx);

        let mut best_idx = 0;
        let mut best_sim = -1.0f32;
        for (i, prog) in env.lattice.programs.iter().enumerate() {
            let sim = cosine_sim(&query, &prog.ema_centroid);
            if sim > best_sim {
                best_sim = sim;
                best_idx = i;
            }
        }

        if best_sim >= config.min_activation {
            let prog = &env.lattice.programs[best_idx];
            let text = dict.map(|d| prog.display_text(d)).unwrap_or_default();
            let node_idx = self.cognitive_map.index.get(&(group_idx, best_idx)).copied();

            println!(
                "  [system2] step {} RETRIEVE: group={}, sim={:.3}, text_len={}",
                step, group_idx, best_sim, text.len()
            );

            wm.insert(WorkingMemoryEntry {
                node_idx,
                group_idx,
                embedding: prog.ema_centroid.clone(),
                text,
                activation: best_sim,
                step_produced: step,
            });
        }
    }

    /// Transfer step: apply a transfer rotor to map knowledge from source_group
    /// to target_group, creating a new working memory entry that represents
    /// the analogical transfer.
    fn step_transfer(
        &self,
        wm: &mut WorkingMemory,
        source_group: usize,
        target_group: usize,
        _group_envs: &HashMap<usize, IndexedGenEnv>,
        group_rotors: &HashMap<usize, GroupRotor>,
        config: &System2Config,
        step: usize,
    ) {
        let source_rotor = group_rotors.get(&source_group)
            .map(|gr| gr.rotor())
            .unwrap_or_else(Rotor::identity);
        let target_rotor = group_rotors.get(&target_group)
            .map(|gr| gr.rotor())
            .unwrap_or_else(Rotor::identity);
        let xfer = transfer_rotor(&source_rotor, &target_rotor);

        // Find the strongest source entry in working memory
        let source_entry = wm.entries.iter()
            .filter(|e| e.group_idx == source_group)
            .max_by(|a, b| a.activation.partial_cmp(&b.activation)
                .unwrap_or(std::cmp::Ordering::Equal))
            .cloned();

        if let Some(entry) = source_entry {
            let source_mv = embed_bridge_vector(&entry.embedding);
            let transferred_mv = apply_group_rotor(&source_mv, &xfer);

            // Extract the transferred embedding back to bridge space
            let transferred_emb = extract_conditioning(&transferred_mv, entry.embedding.len());

            // Measure alignment with the goal
            let alignment = cosine_sim(&transferred_emb, &wm.goal).max(0.0);

            if alignment >= config.transfer_threshold {
                println!(
                    "  [system2] step {} TRANSFER: {}→{}, alignment={:.3}",
                    step, source_group, target_group, alignment
                );

                wm.insert(WorkingMemoryEntry {
                    node_idx: None,
                    group_idx: target_group,
                    embedding: transferred_emb,
                    text: format!("[transferred: {}→{}] {}", source_group, target_group, entry.text),
                    activation: alignment,
                    step_produced: step,
                });
            }
        }
    }

    /// Compose step: merge working memory entries into a partial conclusion.
    /// Uses sentence-level interleaving weighted by activation strength.
    fn step_compose(&self, wm: &mut WorkingMemory, step: usize) {
        if wm.entries.len() < 2 {
            return;
        }

        let fragments: Vec<(String, f32)> = wm.entries.iter()
            .map(|e| (e.text.clone(), e.activation))
            .collect();

        let composed = self.interleave_fragments(&fragments, &wm.goal);
        if composed.is_empty() {
            return;
        }

        // The composed text's embedding is the composite of working memory
        let composite = wm.composite_embedding();
        let activation = cosine_sim(&composite, &wm.goal).max(0.0);

        println!(
            "  [system2] step {} COMPOSE: {} entries → {:.0} chars, activation={:.3}",
            step,
            fragments.len(),
            composed.len() as f32,
            activation
        );

        wm.insert(WorkingMemoryEntry {
            node_idx: None,
            group_idx: wm.entries[0].group_idx,
            embedding: composite,
            text: composed,
            activation,
            step_produced: step,
        });
    }

    /// Assemble final response from working memory.
    /// Selects the highest-activation entries and produces the final text.
    fn assemble_from_wm(&self, wm: &WorkingMemory, cond: &[f32]) -> (String, f32) {
        if wm.entries.is_empty() {
            return (String::new(), 0.0);
        }

        // If there's a composed entry (from a Compose step), prefer it
        if let Some(composed) = wm.entries.iter()
            .filter(|e| e.step_produced > 0 && e.text.len() > 20)
            .max_by(|a, b| a.activation.partial_cmp(&b.activation)
                .unwrap_or(std::cmp::Ordering::Equal))
        {
            let mut text = composed.text.clone();
            // Strip transfer markers from final output
            while let Some(end) = text.find("] ") {
                if text.starts_with("[transferred:") {
                    text = text[end + 2..].to_string();
                } else {
                    break;
                }
            }
            return (text, composed.activation.clamp(0.0, 1.0));
        }

        // Otherwise, assemble from individual entries
        let mut sorted: Vec<&WorkingMemoryEntry> = wm.entries.iter().collect();
        sorted.sort_by(|a, b| b.activation.partial_cmp(&a.activation)
            .unwrap_or(std::cmp::Ordering::Equal));

        let fragments: Vec<(String, f32)> = sorted.iter()
            .take(4)
            .map(|e| (e.text.clone(), e.activation))
            .collect();

        let text = self.interleave_fragments(&fragments, cond);
        let confidence = sorted.first().map(|e| e.activation).unwrap_or(0.0);
        (text, confidence.clamp(0.0, 1.0))
    }

    /// System 2 engagement check: should we invoke deliberate reasoning?
    ///
    /// Triggers when:
    /// - Primary generation confidence is in the "uncertain middle" (0.30–0.65)
    /// - Multiple groups are co-activated (cross-domain query)
    /// - The topic hint suggests a complex/compositional question
    pub fn should_reason_deliberate(
        &self,
        cond: &[f32],
        primary_confidence: f32,
        group_envs: &HashMap<usize, IndexedGenEnv>,
        topic_hint: Option<&str>,
    ) -> bool {
        self.should_reason_deliberate_ext(cond, primary_confidence, group_envs, topic_hint, false)
    }

    /// Extended version with broad query awareness.
    /// When `is_broad` is true, the confidence ceiling is raised to 0.85
    /// (broad queries often get high confidence on a *wrong* single program).
    pub fn should_reason_deliberate_ext(
        &self,
        cond: &[f32],
        primary_confidence: f32,
        group_envs: &HashMap<usize, IndexedGenEnv>,
        topic_hint: Option<&str>,
        is_broad: bool,
    ) -> bool {
        let confidence_ceiling = if is_broad { 0.85 } else { 0.65 };

        if primary_confidence > confidence_ceiling {
            return false;
        }

        if primary_confidence < 0.10 {
            return false;
        }

        // Check for cross-domain co-activation
        let mut group_scores: Vec<(usize, f32)> = group_envs.iter()
            .map(|(&gidx, env)| {
                let best_sim = env.lattice.programs.iter()
                    .map(|p| cosine_sim(cond, &p.ema_centroid))
                    .fold(0.0f32, f32::max);
                (gidx, best_sim)
            })
            .collect();
        group_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        if group_scores.len() < 2 {
            return false;
        }

        let top = group_scores[0].1;
        let second = group_scores[1].1;

        // Both groups must be meaningfully activated
        if top < 0.25 || second < 0.20 {
            return false;
        }

        // Cross-domain ambiguity: top two groups are close
        let ambiguity = if top > 0.01 { second / top } else { 0.0 };

        // Complex topic hints suggest multi-step reasoning is valuable
        let complex_topic = topic_hint.map(|t| {
            t.contains("compare")
                || t.contains("difference")
                || t.contains("combine")
                || t.contains("migrate")
                || t.contains("refactor")
                || t.contains("trade")
        }).unwrap_or(false);

        ambiguity > 0.70 || complex_topic
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cognitive_map_empty() {
        let map = CognitiveMap::build(&HashMap::new(), &HashMap::new());
        assert_eq!(map.node_count(), 0);
        assert_eq!(map.edge_count(), 0);
    }

    #[test]
    fn test_wave_settling_converges() {
        let map = CognitiveMap {
            nodes: vec![
                CognitiveNode { group_idx: 0, program_idx: 0, fingerprint: [0.1; 28], centroid: vec![1.0, 0.0], token_sequence: vec![1], verbatim_display_text: None },
                CognitiveNode { group_idx: 1, program_idx: 0, fingerprint: [0.1; 28], centroid: vec![0.0, 1.0], token_sequence: vec![2], verbatim_display_text: None },
                CognitiveNode { group_idx: 2, program_idx: 0, fingerprint: [0.0; 28], centroid: vec![0.5, 0.5], token_sequence: vec![3], verbatim_display_text: None },
            ],
            edges: vec![
                CognitiveEdge { source: 0, target: 1, structural_sim: 0.8 },
            ],
            adjacency: vec![
                vec![(1, 0.8)],
                vec![(0, 0.8)],
                vec![],
            ],
            index: HashMap::new(),
        };

        let engine = ReasoningEngine::new(map, HashMap::new());
        let activations = vec![
            Activation { node_idx: 0, group_idx: 0, similarity: 0.9, text: "hello".into() },
        ];
        let energy = engine.settle_wave(&activations);
        assert_eq!(energy.len(), 3);
        assert!(energy[0] > 0.5, "Primary activation should be strong");
        assert!(energy[1] > 0.0, "Connected neighbor should activate via spreading");
        assert!(energy[2] < energy[1], "Unconnected node should be weaker");
    }
}
