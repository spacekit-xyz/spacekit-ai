//! Cloze (fill-in-the-blank) learning engine.
//!
//! Teaches the system to INFER slot content from input semantics rather than
//! copy from retrieved text. The Paramecium lattice programs learn to "own"
//! specific slot fills — one program owns addition-slot-fills, another owns
//! subtraction-slot-fills — through geometric centroid drift.
//!
//! Reward signal: causal fingerprint alignment between proposed fill and
//! ground truth. No backpropagation. One-pass per game.
//!
//! Three phases of play:
//!   Phase 1 — Structural cloze: blank out operators, type names, keywords
//!   Phase 2 — Semantic cloze: blank out clauses and phrases
//!   Phase 3 — Compositional cloze: novel templates combining two skills

use crate::clifford::{embed_bridge_vector, causal_fingerprint, BOOST_BIVECTOR_COUNT};
use crate::dimension::group_gen::{AlgebraicCodebook, IndexedGenEnv};
use crate::dimension::paramecium::BehavioralProgram;
use crate::spectral::TokenDictionary;

/// A single cloze game: a template with blanked slots and the known-good fills.
#[derive(Clone, Debug)]
pub struct ClozeTask {
    /// The conditioning vector (input embedding) for this task.
    pub cond: Vec<f32>,
    /// Index of the archetype that provides the template structure.
    pub archetype_idx: usize,
    /// Ground-truth slot values (indices into each slot's vocab).
    pub ground_truth_slots: Vec<usize>,
    /// The original text this task was derived from (for diagnostics).
    pub source_text: String,
}

/// Result of a single cloze game round.
#[derive(Clone, Debug)]
pub struct ClozeResult {
    pub task_idx: usize,
    pub proposed_slots: Vec<usize>,
    pub ground_truth_slots: Vec<usize>,
    pub slot_correct: Vec<bool>,
    pub accuracy: f32,
}

/// Cumulative statistics from a cloze training session.
#[derive(Clone, Debug, Default)]
pub struct ClozeStats {
    pub games_played: usize,
    pub total_slots: usize,
    pub correct_slots: usize,
    pub reward_applied: usize,
    pub punishment_applied: usize,
}

impl std::fmt::Display for ClozeStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let acc = if self.total_slots > 0 {
            self.correct_slots as f32 / self.total_slots as f32
        } else { 0.0 };
        write!(f, "games={}, slots={}/{} ({:.1}%), rewards={}, punishments={}",
            self.games_played, self.correct_slots, self.total_slots,
            acc * 100.0, self.reward_applied, self.punishment_applied)
    }
}

/// Generate cloze tasks from a group's training data and codebook.
///
/// For each training sample, we:
/// 1. Match it to its best archetype (structural template)
/// 2. Extract the ground-truth slot values
/// 3. Pair with the sample's conditioning vector
pub fn generate_cloze_tasks(
    codebook: &AlgebraicCodebook,
    dictionary: &TokenDictionary,
    training_pairs: &[(Vec<f32>, String)],
) -> Vec<ClozeTask> {
    if codebook.archetypes.is_empty() {
        return Vec::new();
    }

    let mut tasks = Vec::with_capacity(training_pairs.len());
    for (cond, text) in training_pairs {
        let token_ids = dictionary.encode(text);
        let (arch_idx, slot_values) = codebook.match_best(&token_ids);

        let arch = &codebook.archetypes[arch_idx];
        if arch.slots.is_empty() {
            continue;
        }

        tasks.push(ClozeTask {
            cond: cond.clone(),
            archetype_idx: arch_idx,
            ground_truth_slots: slot_values,
            source_text: text.clone(),
        });
    }
    tasks
}

/// Infer slot values from the input conditioning vector using lattice programs.
///
/// Each lattice program "owns" a region of embedding space. For a given input,
/// the K nearest programs vote on each slot position. The vote is weighted by
/// cosine similarity — programs closer to the input have stronger votes.
///
/// This is the key mechanism that replaces "copy from retrieved text" with
/// "infer from input semantics."
pub fn infer_slots(
    cond: &[f32],
    archetype_idx: usize,
    codebook: &AlgebraicCodebook,
    dictionary: &TokenDictionary,
    programs: &[BehavioralProgram],
    k_voters: usize,
) -> Vec<usize> {
    let arch = match codebook.archetypes.get(archetype_idx) {
        Some(a) => a,
        None => return Vec::new(),
    };
    if arch.slots.is_empty() || programs.is_empty() {
        return vec![0; codebook.max_slot_count];
    }

    // Find K nearest programs by cosine similarity to input.
    let mut scored: Vec<(usize, f32)> = programs.iter().enumerate()
        .map(|(i, prog)| {
            let sim = cosine_sim(cond, &prog.ema_centroid).max(0.0);
            (i, sim)
        })
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let voters: Vec<(usize, f32)> = scored.into_iter().take(k_voters).collect();

    // For each slot, voters decode their token_sequence at that position
    // and vote for the matching vocab index, weighted by similarity.
    let mut inferred = Vec::with_capacity(codebook.max_slot_count);
    for (slot_idx, slot) in arch.slots.iter().enumerate() {
        if slot.vocab.is_empty() {
            inferred.push(0);
            continue;
        }
        let mut votes = vec![0.0f32; slot.vocab.len()];

        for &(prog_idx, weight) in &voters {
            let prog = &programs[prog_idx];
            let actual_tok = prog.token_sequence.get(slot.position).copied().unwrap_or(0);
            if let Some(vocab_idx) = slot.vocab.iter().position(|&t| t == actual_tok) {
                votes[vocab_idx] += weight;
            }
        }

        // Also incorporate causal alignment as a tiebreaker:
        // if two candidates have similar vote totals, prefer the one whose
        // owning programs have closer causal fingerprints to the input.
        let input_mv = embed_bridge_vector(cond);
        let input_cf = causal_fingerprint(&input_mv);

        for (vocab_idx, vote) in votes.iter_mut().enumerate() {
            if *vote < 1e-8 { continue; }
            let tok = slot.vocab[vocab_idx];
            // Find the program whose token_sequence at this position matches
            // this vocab token and has the highest similarity.
            let best_causal_alignment = voters.iter()
                .filter(|&&(pi, _)| {
                    programs[pi].token_sequence.get(slot.position).copied() == Some(tok)
                })
                .map(|&(pi, _)| {
                    let prog_mv = embed_bridge_vector(&programs[pi].ema_centroid);
                    let prog_cf = causal_fingerprint(&prog_mv);
                    causal_cosine(&input_cf, &prog_cf)
                })
                .fold(0.0f32, f32::max);

            // Blend: 70% vote weight + 30% causal alignment
            *vote = 0.70 * *vote + 0.30 * best_causal_alignment * *vote;
        }

        let winner = votes.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        inferred.push(winner);
    }

    inferred.resize(codebook.max_slot_count, 0);
    inferred
}

/// Play a round of cloze games and apply reward/punishment to lattice programs.
///
/// Reward: when a program's slot-region vote was correct, drift its centroid
/// toward the input embedding (strengthening ownership).
///
/// Punishment: when a program voted for the wrong fill, drift its centroid
/// AWAY from the input (weakening ownership of this region).
///
/// The drift magnitude is proportional to the program's similarity to the input,
/// so nearby programs learn faster than distant ones.
pub fn play_cloze_round(
    env: &mut IndexedGenEnv,
    tasks: &[ClozeTask],
    k_voters: usize,
    reward_rate: f32,
    punish_rate: f32,
) -> ClozeStats {
    let mut stats = ClozeStats::default();

    let codebook = match env.codebook.as_ref() {
        Some(cb) if !cb.archetypes.is_empty() => cb.clone(),
        _ => return stats,
    };

    for task in tasks {
        let arch = match codebook.archetypes.get(task.archetype_idx) {
            Some(a) => a,
            None => continue,
        };
        if arch.slots.is_empty() { continue; }

        // Infer slots from input conditioning vector.
        let proposed = infer_slots(
            &task.cond,
            task.archetype_idx,
            &codebook,
            &env.dictionary,
            &env.lattice.programs,
            k_voters,
        );

        // Score each slot.
        let mut slot_correct = Vec::with_capacity(arch.slots.len());
        for (slot_idx, slot) in arch.slots.iter().enumerate() {
            let proposed_val = proposed.get(slot_idx).copied().unwrap_or(0);
            let truth_val = task.ground_truth_slots.get(slot_idx).copied().unwrap_or(0);
            let correct = proposed_val == truth_val;
            slot_correct.push(correct);
            stats.total_slots += 1;
            if correct { stats.correct_slots += 1; }
        }

        // Apply reward/punishment to the K-nearest programs.
        let mut scored: Vec<(usize, f32)> = env.lattice.programs.iter().enumerate()
            .map(|(i, prog)| (i, cosine_sim(&task.cond, &prog.ema_centroid).max(0.0)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        for &(prog_idx, sim) in scored.iter().take(k_voters) {
            if sim < 0.05 { continue; }
            let prog = &env.lattice.programs[prog_idx];
            let prog_token_at_slots: Vec<Option<u16>> = arch.slots.iter()
                .map(|s| prog.token_sequence.get(s.position).copied())
                .collect();

            let mut did_reward = false;
            let mut did_punish = false;

            for (slot_idx, slot) in arch.slots.iter().enumerate() {
                let truth_val = task.ground_truth_slots.get(slot_idx).copied().unwrap_or(0);
                let truth_tok = slot.vocab.get(truth_val).copied().unwrap_or(0);
                let prog_tok = prog_token_at_slots[slot_idx].unwrap_or(0);

                if prog_tok == truth_tok {
                    did_reward = true;
                } else if prog_tok != 0 {
                    did_punish = true;
                }
            }

            let prog = &mut env.lattice.programs[prog_idx];
            let dim = prog.ema_centroid.len().min(task.cond.len());

            if did_reward && !did_punish {
                // Reward: drift toward the input (strengthen ownership)
                let alpha = reward_rate * sim;
                for i in 0..dim {
                    prog.ema_centroid[i] = prog.ema_centroid[i] * (1.0 - alpha) + task.cond[i] * alpha;
                }
                stats.reward_applied += 1;
            } else if did_punish && !did_reward {
                // Punishment: drift AWAY from the input (weaken ownership)
                let alpha = punish_rate * sim;
                for i in 0..dim {
                    prog.ema_centroid[i] = prog.ema_centroid[i] * (1.0 + alpha) - task.cond[i] * alpha;
                }
                // Re-normalize to prevent centroid explosion
                let norm = prog.ema_centroid.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
                let original_norm = prog.centroid.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
                let target_norm = original_norm; // maintain original scale
                for v in &mut prog.ema_centroid {
                    *v *= target_norm / norm;
                }
                stats.punishment_applied += 1;
            }
        }

        stats.games_played += 1;
    }

    stats
}

/// Encode inferred slot values into slot bits for decode_with_archetype.
pub fn encode_inferred_slot_bits(
    inferred_slots: &[usize],
    codebook: &AlgebraicCodebook,
) -> Vec<f32> {
    let mut bits = vec![0.0f32; codebook.slot_only_bits];
    let mut offset = 0;
    for (slot_idx, &val) in inferred_slots.iter().enumerate() {
        let sbits = codebook.slot_bit_widths.get(slot_idx).copied().unwrap_or(0);
        for i in 0..sbits {
            if offset + i < bits.len() {
                bits[offset + i] = if (val >> i) & 1 == 1 { 1.0 } else { 0.0 };
            }
        }
        offset += sbits;
    }
    bits
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-12 || nb < 1e-12 { 0.0 } else { dot / (na * nb) }
}

fn causal_cosine(a: &[f32; BOOST_BIVECTOR_COUNT], b: &[f32; BOOST_BIVECTOR_COUNT]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-10 || nb < 1e-10 { 0.0 } else { (dot / (na * nb)).max(0.0) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spectral::TokenDictionary;
    use crate::dimension::group_gen::AlgebraicCodebook;
    use crate::dimension::paramecium::BehavioralProgram;

    fn make_test_dict() -> TokenDictionary {
        TokenDictionary::build(
            &["add two numbers", "subtract two numbers", "multiply two numbers"],
            50,
        )
    }

    #[test]
    fn test_cosine_sim_self() {
        let v = vec![1.0, 2.0, 3.0];
        let sim = cosine_sim(&v, &v);
        assert!((sim - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_sim_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let sim = cosine_sim(&a, &b);
        assert!(sim.abs() < 1e-5);
    }

    #[test]
    fn test_encode_inferred_slot_bits_roundtrip() {
        let codebook = AlgebraicCodebook {
            archetypes: vec![],
            archetype_bits: 2,
            max_slot_count: 3,
            slot_bit_widths: vec![4, 4, 4],
            total_bits: 14,
            archetype_prototypes: vec![],
            slot_only_bits: 12,
        };

        let inferred = vec![5, 2, 7];
        let bits = encode_inferred_slot_bits(&inferred, &codebook);
        assert_eq!(bits.len(), 12);

        // Verify slot 0 = 5 = 0b0101 → bits[0..4] = [1, 0, 1, 0]
        assert_eq!(bits[0], 1.0); // bit 0
        assert_eq!(bits[1], 0.0); // bit 1
        assert_eq!(bits[2], 1.0); // bit 2
        assert_eq!(bits[3], 0.0); // bit 3
    }

    #[test]
    fn test_cloze_stats_display() {
        let stats = ClozeStats {
            games_played: 10,
            total_slots: 30,
            correct_slots: 20,
            reward_applied: 15,
            punishment_applied: 5,
        };
        let s = format!("{}", stats);
        assert!(s.contains("66.7%"));
        assert!(s.contains("games=10"));
    }
}
