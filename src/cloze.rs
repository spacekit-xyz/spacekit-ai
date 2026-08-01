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

use crate::clifford::{causal_fingerprint, embed_bridge_vector, BOOST_BIVECTOR_COUNT};
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
        } else {
            0.0
        };
        write!(
            f,
            "games={}, slots={}/{} ({:.1}%), rewards={}, punishments={}",
            self.games_played,
            self.correct_slots,
            self.total_slots,
            acc * 100.0,
            self.reward_applied,
            self.punishment_applied
        )
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

    let voters: Vec<(usize, f32)> = top_k_programs_by_cosine(programs, cond, k_voters);

    // Causal tiebreaker: one Cl(8) embed + fingerprint for the query, and one per
    // voter centroid. Doing this inside the slot×vocab loops was O(slots × vocab × voters)
    // full embeds per task and made cloze appear hung on large lattices / vocabs.
    let input_mv = embed_bridge_vector(cond);
    let input_cf = causal_fingerprint(&input_mv);
    let voter_cf: Vec<[f32; BOOST_BIVECTOR_COUNT]> = voters
        .iter()
        .map(|&(pi, _)| {
            let prog_mv = embed_bridge_vector(&programs[pi].ema_centroid);
            causal_fingerprint(&prog_mv)
        })
        .collect();

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

        // Blend vote totals with causal alignment (precomputed fingerprints only).
        for (vocab_idx, vote) in votes.iter_mut().enumerate() {
            if *vote < 1e-8 {
                continue;
            }
            let tok = slot.vocab[vocab_idx];
            let best_causal_alignment = voters
                .iter()
                .enumerate()
                .filter(|(_, &(pi, _))| {
                    programs[pi].token_sequence.get(slot.position).copied() == Some(tok)
                })
                .map(|(j, _)| causal_cosine(&input_cf, &voter_cf[j]))
                .fold(0.0f32, f32::max);

            // Blend: 70% vote weight + 30% causal alignment
            *vote = 0.70 * *vote + 0.30 * best_causal_alignment * *vote;
        }

        let winner = votes
            .iter()
            .enumerate()
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
///
/// `on_each_task` is invoked once after each task (e.g. progress bar tick). Use `|| {}` if unused.
/// When `eprint_progress` is true, periodic lines go to stderr (skipped when using a GUI bar).
pub fn play_cloze_round<F>(
    env: &mut IndexedGenEnv,
    tasks: &[ClozeTask],
    k_voters: usize,
    reward_rate: f32,
    punish_rate: f32,
    mut on_each_task: F,
    eprint_progress: bool,
) -> ClozeStats
where
    F: FnMut(),
{
    let mut stats = ClozeStats::default();

    let Some(codebook) = env.codebook.as_ref().filter(|cb| !cb.archetypes.is_empty()) else {
        return stats;
    };

    let n_tasks = tasks.len();
    let n_progs = env.lattice.programs.len();
    let progress_every = if n_tasks <= 50 {
        10
    } else if n_tasks <= 200 {
        25
    } else {
        100
    };

    for (ti, task) in tasks.iter().enumerate() {
        if eprint_progress && (ti == 0 || (ti > 0 && ti % progress_every == 0)) {
            eprintln!(
                "    cloze progress: {}/{} tasks ({} lattice programs)",
                ti, n_tasks, n_progs
            );
        }
        let arch = match codebook.archetypes.get(task.archetype_idx) {
            Some(a) => a,
            None => continue,
        };
        if arch.slots.is_empty() {
            continue;
        }

        // Infer slots from input conditioning vector.
        let proposed = infer_slots(
            &task.cond,
            task.archetype_idx,
            codebook,
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
            if correct {
                stats.correct_slots += 1;
            }
        }

        let nearest = top_k_programs_by_cosine(&env.lattice.programs, &task.cond, k_voters);

        for &(prog_idx, sim) in &nearest {
            if sim < 0.05 {
                continue;
            }
            let prog = &env.lattice.programs[prog_idx];
            let prog_token_at_slots: Vec<Option<u16>> = arch
                .slots
                .iter()
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
                    prog.ema_centroid[i] =
                        prog.ema_centroid[i] * (1.0 - alpha) + task.cond[i] * alpha;
                }
                stats.reward_applied += 1;
            } else if did_punish && !did_reward {
                // Contrastive punishment (two forces):
                //   1. Push AWAY from the incorrect input (repulsive)
                //   2. Pull TOWARD own original centroid (attractive anchor)
                //
                // This creates within-group separation: each program retreats
                // to its correct home while being repelled from incorrect queries.
                let repel_alpha = punish_rate * sim;
                let attract_alpha = punish_rate * 0.5; // gentler pull toward home

                for i in 0..dim {
                    // Repel from incorrect input
                    prog.ema_centroid[i] =
                        prog.ema_centroid[i] * (1.0 + repel_alpha) - task.cond[i] * repel_alpha;
                }

                // Attract toward original centroid (home base)
                let home_dim = prog.centroid.len().min(dim);
                for i in 0..home_dim {
                    prog.ema_centroid[i] = prog.ema_centroid[i] * (1.0 - attract_alpha)
                        + prog.centroid[i] * attract_alpha;
                }

                // Re-normalize to prevent centroid explosion
                let norm = prog
                    .ema_centroid
                    .iter()
                    .map(|x| x * x)
                    .sum::<f32>()
                    .sqrt()
                    .max(1e-8);
                let original_norm = prog
                    .centroid
                    .iter()
                    .map(|x| x * x)
                    .sum::<f32>()
                    .sqrt()
                    .max(1e-8);
                let target_norm = original_norm;
                for v in &mut prog.ema_centroid {
                    *v *= target_norm / norm;
                }
                stats.punishment_applied += 1;
            }
        }

        stats.games_played += 1;
        on_each_task();
    }

    stats
}

/// Upper bound on cloze tasks per group (each task scans the whole lattice for top-k voters).
pub const DEFAULT_MAX_CLOZE_TASKS_PER_GROUP: usize = 2000;

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

/// Compute the field gradient ∇F at query point `cond`.
///
/// Each program is a source J_i at position x_i with response embedding r_i.
/// The field gradient at x is:
///
///   ∇F(x) = Σ_i  w_i · (x - x_i) / |x - x_i|²
///
/// where w_i = cosine_sim(cond, centroid_i) is the source strength.
///
/// The gradient is a vector in embedding space that points in the direction
/// the response field is changing fastest. For "subtraction" near the "addition"
/// program, ∇F points AWAY from addition TOWARD subtraction.
///
/// Returns: (gradient_vector, gradient_magnitude)
pub fn compute_field_gradient(cond: &[f32], programs: &[BehavioralProgram]) -> (Vec<f32>, f32) {
    let dim = cond.len();
    let mut gradient = vec![0.0f32; dim];

    if programs.is_empty() {
        return (gradient, 0.0);
    }

    let mut weight_sum = 0.0f32;

    for prog in programs {
        let centroid = &prog.ema_centroid;
        let min_dim = dim.min(centroid.len());

        // Displacement: x - x_i
        let mut disp = vec![0.0f32; dim];
        let mut disp_norm_sq = 0.0f32;
        for i in 0..min_dim {
            disp[i] = cond[i] - centroid[i];
            disp_norm_sq += disp[i] * disp[i];
        }

        if disp_norm_sq < 1e-10 {
            continue; // Skip exact matches (zero displacement = no gradient contribution)
        }

        // Source strength: how relevant is this program to the query
        let sim = cosine_sim(cond, centroid).max(0.0);
        if sim < 0.01 {
            continue;
        }

        // Green's function weight: 1/r² * source_strength
        let green_weight = sim / disp_norm_sq;

        // Accumulate: gradient += weight * displacement_direction
        for i in 0..dim {
            gradient[i] += green_weight * disp[i];
        }
        weight_sum += green_weight;
    }

    // Normalize by total weight to get directional gradient
    if weight_sum > 1e-10 {
        for v in &mut gradient {
            *v /= weight_sum;
        }
    }

    let magnitude = gradient.iter().map(|x| x * x).sum::<f32>().sqrt();
    (gradient, magnitude)
}

/// Infer slot values using the field gradient ∇F for directional discrimination.
///
/// Unlike `infer_slots` which uses proximity voting (same for add/sub),
/// this uses the gradient to bias votes TOWARD programs that the field
/// is flowing toward and AWAY from programs the field is flowing from.
///
/// For "subtraction" query near "addition" centroid:
///   - ∇F points from addition → subtraction
///   - Programs aligned with ∇F get vote bonus (subtraction program)
///   - Programs anti-aligned with ∇F get vote penalty (addition program)
pub fn infer_slots_with_gradient(
    cond: &[f32],
    gradient: &[f32],
    gradient_mag: f32,
    archetype_idx: usize,
    codebook: &AlgebraicCodebook,
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

    // Score programs by BOTH proximity AND gradient alignment.
    let mut scored: Vec<(usize, f32)> = programs
        .iter()
        .enumerate()
        .map(|(i, prog)| {
            let sim = cosine_sim(cond, &prog.ema_centroid).max(0.0);

            // Gradient alignment: does this program lie in the direction ∇F points?
            // Compute: dot(centroid_i - cond, gradient) — positive means the program
            // is in the direction the field is flowing TO.
            let gradient_alignment = if gradient_mag > 1e-6 {
                let min_dim = cond.len().min(prog.ema_centroid.len()).min(gradient.len());
                let dot: f32 = (0..min_dim)
                    .map(|j| (prog.ema_centroid[j] - cond[j]) * gradient[j])
                    .sum();
                // Normalize by gradient magnitude
                (dot / gradient_mag).clamp(-1.0, 1.0)
            } else {
                0.0
            };

            // Blend: proximity (60%) + gradient alignment (40%)
            // gradient_alignment ranges [-1, 1], map to [0, 1]
            let grad_score = (gradient_alignment + 1.0) / 2.0;
            let combined = 0.60 * sim + 0.40 * grad_score;
            (i, combined)
        })
        .collect();

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let voters: Vec<(usize, f32)> = scored.into_iter().take(k_voters).collect();

    // Vote on each slot position, weighted by the gradient-aware score.
    let mut inferred = Vec::with_capacity(codebook.max_slot_count);
    for (_slot_idx, slot) in arch.slots.iter().enumerate() {
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

        let winner = votes
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        inferred.push(winner);
    }

    inferred.resize(codebook.max_slot_count, 0);
    inferred
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let dot: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as f64) * (y as f64))
        .sum();
    let na: f64 = a
        .iter()
        .map(|&x| (x as f64) * (x as f64))
        .sum::<f64>()
        .sqrt();
    let nb: f64 = b
        .iter()
        .map(|&x| (x as f64) * (x as f64))
        .sum::<f64>()
        .sqrt();
    if na < 1e-20 || nb < 1e-20 {
        0.0
    } else {
        (dot / (na * nb)) as f32
    }
}

/// Top-`k` programs by cosine similarity to `cond` in **O(programs.len() × k)** time.
/// Avoids sorting all programs per cloze task (previously O(n log n) per task → training appeared hung).
fn top_k_programs_by_cosine(
    programs: &[BehavioralProgram],
    cond: &[f32],
    mut k: usize,
) -> Vec<(usize, f32)> {
    use std::cmp::Ordering;
    k = k.min(programs.len());
    if k == 0 {
        return Vec::new();
    }
    let mut buf: Vec<(usize, f32)> = Vec::with_capacity(k + 1);
    for (i, prog) in programs.iter().enumerate() {
        let mut sim = cosine_sim(cond, &prog.ema_centroid);
        if !sim.is_finite() {
            continue;
        }
        sim = sim.max(0.0);
        buf.push((i, sim));
        if buf.len() <= k {
            continue;
        }
        let (min_j, _) = buf
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
            .unwrap();
        buf.swap_remove(min_j);
    }
    buf.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    buf
}

fn causal_cosine(a: &[f32; BOOST_BIVECTOR_COUNT], b: &[f32; BOOST_BIVECTOR_COUNT]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-10 || nb < 1e-10 {
        0.0
    } else {
        (dot / (na * nb)).max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dimension::group_gen::AlgebraicCodebook;
    use crate::dimension::paramecium::BehavioralProgram;
    use crate::spectral::TokenDictionary;

    fn make_test_dict() -> TokenDictionary {
        TokenDictionary::build(
            &[
                "add two numbers",
                "subtract two numbers",
                "multiply two numbers",
            ],
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
