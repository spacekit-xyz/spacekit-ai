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
//!   connected by structural similarity (Cl(8) bivector cosine). Enables
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
//! - **Transfer Rotor Composition** (frontal reasoning): the Cl(8) transfer
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
                let text = dict.map(|d| d.decode(&env.lattice.programs[pidx].token_sequence))
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

        // Collect top activated nodes, ensuring cross-group diversity
        let mut scored: Vec<(usize, f32)> = energy.iter().enumerate()
            .filter(|(_, &e)| e > 0.1)
            .map(|(i, &e)| (i, e))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Pick the best node per group (up to 3 groups)
        let mut best_per_group: Vec<(usize, f32)> = Vec::new();
        let mut seen_groups = std::collections::HashSet::new();
        for &(node_idx, e) in &scored {
            let gidx = self.cognitive_map.nodes[node_idx].group_idx;
            if seen_groups.insert(gidx) {
                best_per_group.push((node_idx, e));
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
                .map(|d| d.decode(&node.token_sequence))
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
            .map(|d| d.decode(&primary_node.token_sequence))
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
                .map(|d| d.decode(&node.token_sequence))
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
    /// Returns true when the primary classifier has low confidence (the prompt
    /// doesn't cleanly fit one group) or when multiple groups activate strongly.
    pub fn should_reason(
        &self,
        cond: &[f32],
        primary_confidence: f32,
        group_envs: &HashMap<usize, IndexedGenEnv>,
    ) -> bool {
        if primary_confidence > 0.75 {
            return false;
        }

        // Check if multiple groups activate above threshold
        let mut active_groups = 0;
        for (&_gidx, env) in group_envs {
            let best_sim = env.lattice.programs.iter()
                .map(|p| cosine_sim(cond, &p.ema_centroid))
                .fold(0.0f32, f32::max);
            if best_sim > 0.3 {
                active_groups += 1;
            }
        }

        active_groups >= 2
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
                CognitiveNode { group_idx: 0, program_idx: 0, fingerprint: [0.1; 28], centroid: vec![1.0, 0.0], token_sequence: vec![1] },
                CognitiveNode { group_idx: 1, program_idx: 0, fingerprint: [0.1; 28], centroid: vec![0.0, 1.0], token_sequence: vec![2] },
                CognitiveNode { group_idx: 2, program_idx: 0, fingerprint: [0.0; 28], centroid: vec![0.5, 0.5], token_sequence: vec![3] },
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
