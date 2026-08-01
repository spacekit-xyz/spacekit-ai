//! Synthetic basal ganglia — value-weighted action selection over candidate responses.
//!
//! The basal ganglia are the brain's *action-selection engine*: given many
//! possible actions, they release **one** by weighing cortical input (fit),
//! identity, current state, and reward — gated by dopamine (Go), serotonin
//! (NoGo / patience), and norepinephrine (urgency / decisiveness).
//!
//! Our retrieval already *generates* candidates (`stochastic_ordered_candidates`)
//! and several scattered heuristics *select* among them (stochastic top-k
//! sampling, refractory anti-repeat, OOD exploration temperature, the coherence
//! gate). This module **consolidates** that scatter into one principled selector
//! that sits between the candidate set and the final response:
//!
//! ```text
//!   candidates (idx, text, retrieval_score)        ← cortical "what fits the prompt"
//!        │
//!        ▼  value V(c) = fit + affect + identity − repeat − garble − verbosity
//!        │  gated by neuromodulators (DA Go-temp, 5HT NoGo-floor, NE sharpen)
//!        ▼
//!   one selected action
//! ```
//!
//! Where the [`crate::reflective_field`] biases the *input* (which programs score
//! high), the basal ganglia evaluates the *output* (the actual candidate texts),
//! so it can directly veto a dominant-but-wrong program — the failure mode that
//! caused OOD collapse.

use crate::drive_field::Neuromodulators;

/// Lexicons are intentionally generic English affect words (not domain-specific),
/// so the selector ports across agents. They tag the *tone* of a candidate.
const WARM_WORDS: &[&str] = &[
    "safe", "calm", "gentle", "soft", "warm", "here", "stay", "close", "slow", "okay", "rest",
    "breathe", "love", "comfort", "lean", "curl", "blink", "nuzzle", "purr", "soothe", "easy",
    "quiet", "hold",
];
const PLAYFUL_WORDS: &[&str] = &[
    "play", "chase", "pounce", "zoom", "hunt", "leap", "dash", "bounce", "toy", "wand", "feather",
    "fun", "go", "catch", "spring", "trill", "chirp", "wiggle",
];
/// Words in the *user's* message that signal distress (→ favor warm candidates).
const USER_DISTRESS: &[&str] = &[
    "scared",
    "anxious",
    "sad",
    "tired",
    "exhausted",
    "hate",
    "alone",
    "lonely",
    "cry",
    "crying",
    "hurt",
    "stressed",
    "overwhelmed",
    "afraid",
    "worried",
    "upset",
    "depressed",
    "miss you",
    "lost",
    "can't",
    "cannot cope",
    "down",
];
/// Words in the user's message that signal excitement (→ favor playful candidates).
const USER_EXCITEMENT: &[&str] = &[
    "play",
    "fun",
    "yay",
    "awesome",
    "let's",
    "lets",
    "chase",
    "excited",
    "treat",
    "walk",
    "go get",
    "good girl",
    "good boy",
    "wanna",
    "want to play",
];

#[inline]
fn lexicon_hits(text_lower: &str, lex: &[&str]) -> f32 {
    lex.iter().filter(|w| text_lower.contains(**w)).count() as f32
}

/// Inferred affect of the *user's* current message. Scalars in `[0, 1]`.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct UserAffect {
    pub distress: f32,
    pub excitement: f32,
}

impl UserAffect {
    /// Cheap lexical estimate from the incoming prompt.
    pub fn from_prompt(prompt: &str) -> Self {
        let l = prompt.to_ascii_lowercase();
        let d = lexicon_hits(&l, USER_DISTRESS);
        let e = lexicon_hits(&l, USER_EXCITEMENT);
        // Saturate: one strong cue is enough; more cues add diminishing weight.
        Self {
            distress: (d / 2.0).clamp(0.0, 1.0),
            excitement: (e / 2.0).clamp(0.0, 1.0),
        }
    }
}

/// Per-turn context the selector weighs each candidate against.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ActionContext {
    pub neuromods: Option<Neuromodulators>,
    pub user_affect: UserAffect,
    /// Identity trait vector `[O, C, E, A, N]`.
    pub ocean: [f32; 5],
    /// Preferred response length (chars); candidates far from it are penalized.
    pub target_chars: usize,
}

impl Default for ActionContext {
    fn default() -> Self {
        Self {
            neuromods: None,
            user_affect: UserAffect::default(),
            ocean: [0.5; 5],
            target_chars: 160,
        }
    }
}

/// Value-term weights. Positive terms reward; `repeat`/`garble`/`verbosity` penalize.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ValueWeights {
    pub fit: f32,
    pub affect: f32,
    pub identity: f32,
    pub repeat: f32,
    pub garble: f32,
    pub verbosity: f32,
}

impl Default for ValueWeights {
    fn default() -> Self {
        // Fit (prompt relevance) dominates; affect/identity refine; penalties guard.
        Self {
            fit: 1.0,
            affect: 0.6,
            identity: 0.25,
            repeat: 0.8,
            garble: 1.5,
            verbosity: 0.3,
        }
    }
}

/// A response candidate produced by retrieval.
#[derive(Debug, Clone)]
pub struct Candidate {
    pub idx: usize,
    pub text: String,
    pub retrieval_score: f32,
}

/// The selector. Flag-gated; carries weights + the current turn's context.
#[derive(Debug, Clone)]
pub struct BasalGanglia {
    pub enabled: bool,
    pub weights: ValueWeights,
    pub ctx: ActionContext,
}

impl BasalGanglia {
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            weights: ValueWeights::default(),
            ctx: ActionContext::default(),
        }
    }

    pub fn with_context(mut self, ctx: ActionContext) -> Self {
        self.ctx = ctx;
        self
    }

    /// Light fragmentation heuristic (mirrors the runtime garble guard) so the
    /// selector can down-weight broken candidates as part of the value, not as a
    /// separate gate.
    fn fragmented(text: &str) -> bool {
        let b = text.as_bytes();
        for i in 1..b.len().saturating_sub(1) {
            if b[i] == b'.' && b[i + 1] == b'.' {
                let ellipsis = i + 2 < b.len() && b[i + 2] == b'.';
                if !ellipsis {
                    return true;
                }
            }
        }
        let mut run = 0;
        for w in text.split_whitespace() {
            let clean = w.trim_matches(|c: char| !c.is_alphabetic());
            if clean.chars().count() <= 1 {
                run += 1;
                if run >= 3 {
                    return true;
                }
            } else {
                run = 0;
            }
        }
        false
    }

    fn affect_alignment(&self, text_lower: &str) -> f32 {
        let warm = (lexicon_hits(text_lower, WARM_WORDS) * 0.2).clamp(0.0, 1.0);
        let playful = (lexicon_hits(text_lower, PLAYFUL_WORDS) * 0.2).clamp(0.0, 1.0);
        let a = self.ctx.user_affect;
        // Reward tone that matches the user's state; mildly penalize the opposite
        // (playful banter at a distressed user, or flat calm at an excited one).
        a.distress * warm + a.excitement * playful
            - 0.4 * a.distress * playful
            - 0.2 * a.excitement * (1.0 - playful) * warm
    }

    fn identity_alignment(&self, text_lower: &str) -> f32 {
        let warm = (lexicon_hits(text_lower, WARM_WORDS) * 0.2).clamp(0.0, 1.0);
        let playful = (lexicon_hits(text_lower, PLAYFUL_WORDS) * 0.2).clamp(0.0, 1.0);
        let agreeableness = self.ctx.ocean[3] - 0.5;
        let extraversion = self.ctx.ocean[2] - 0.5;
        (agreeableness * warm + extraversion * playful).clamp(-0.5, 0.5)
    }

    fn repeat_penalty(text: &str, recent: &[String]) -> f32 {
        if recent.iter().any(|r| r == text) {
            1.0
        } else {
            0.0
        }
    }

    fn verbosity_penalty(len: usize, target: usize) -> f32 {
        if target == 0 {
            return 0.0;
        }
        ((len as f32 - target as f32).abs() / target as f32).clamp(0.0, 1.0)
    }

    /// Compute the scalar value of a candidate. `fit_norm` is the candidate's
    /// retrieval score min-max-normalized across the set into `[0, 1]`.
    fn value(&self, text: &str, fit_norm: f32, recent: &[String]) -> f32 {
        let l = text.to_ascii_lowercase();
        let w = self.weights;
        w.fit * fit_norm
            + w.affect * self.affect_alignment(&l)
            + w.identity * self.identity_alignment(&l)
            - w.repeat * Self::repeat_penalty(text, recent)
            - w.garble * if Self::fragmented(text) { 1.0 } else { 0.0 }
            - w.verbosity * Self::verbosity_penalty(text.len(), self.ctx.target_chars)
    }

    /// Neuromodulator-gated selection. Returns the *position in `candidates`* of
    /// the chosen action, or `None` if the set is empty. `seed` makes the
    /// dopamine-tempered sampling reproducible.
    pub fn select(&self, candidates: &[Candidate], recent: &[String], seed: u64) -> Option<usize> {
        if candidates.is_empty() {
            return None;
        }
        if candidates.len() == 1 {
            return Some(0);
        }

        // Normalize retrieval scores by the max (preserving real proportions, so a
        // tiny score gap stays tiny instead of being stretched to the full range).
        let hi = candidates
            .iter()
            .map(|c| c.retrieval_score)
            .fold(f32::NEG_INFINITY, f32::max)
            .max(1e-6);

        let values: Vec<f32> = candidates
            .iter()
            .map(|c| self.value(&c.text, (c.retrieval_score / hi).clamp(0.0, 1.0), recent))
            .collect();

        let (da, ser, ne) = match self.ctx.neuromods {
            Some(nm) => (
                nm.dopamine - 0.5,
                nm.serotonin - 0.5,
                nm.norepinephrine - 0.5,
            ),
            None => (0.0, 0.0, 0.0),
        };

        // NoGo floor: keep candidates within a window below the best. Dopamine
        // (seeking) widens it (consider more); serotonin (patience) narrows it
        // (commit to the clearly-best); so does norepinephrine (urgency).
        let vmax = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let window = (0.15 + 0.30 * da - 0.12 * ser - 0.10 * ne).clamp(0.03, 0.7);
        let survivors: Vec<usize> = (0..candidates.len())
            .filter(|&i| values[i] >= vmax - window)
            .collect();
        if survivors.len() == 1 {
            return Some(survivors[0]);
        }

        // Go-temperature: dopamine raises exploration, norepinephrine sharpens.
        let temp = (0.6 * (1.0 + 0.8 * da - 0.6 * ne)).clamp(0.05, 2.0);
        let smax = survivors
            .iter()
            .map(|&i| values[i])
            .fold(f32::NEG_INFINITY, f32::max);
        let weights: Vec<f32> = survivors
            .iter()
            .map(|&i| ((values[i] - smax) / temp).exp())
            .collect();
        let sum: f32 = weights.iter().sum::<f32>().max(1e-8);

        // Deterministic LCG draw from the seed.
        let mixed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let uniform = ((mixed >> 11) as f64) / ((1u64 << 53) as f64);

        let mut cum = 0.0f64;
        for (j, &i) in survivors.iter().enumerate() {
            cum += (weights[j] / sum) as f64;
            if uniform < cum {
                return Some(i);
            }
        }
        Some(*survivors.last().unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::drive_field::DriveState;

    fn cand(idx: usize, text: &str, score: f32) -> Candidate {
        Candidate {
            idx,
            text: text.to_string(),
            retrieval_score: score,
        }
    }

    #[test]
    fn distressed_user_selects_warm_over_playful() {
        let mut bg = BasalGanglia::new(true);
        bg.ctx.user_affect = UserAffect::from_prompt("i'm so anxious and tired and alone");
        assert!(bg.ctx.user_affect.distress > 0.0);
        // Two equally-retrievable candidates; tone differs.
        let cands = vec![
            cand(
                0,
                "Let's chase and pounce and zoom around, this is fun!",
                1.0,
            ),
            cand(
                1,
                "I am here with you. Stay close. Slow breathe. You are safe.",
                1.0,
            ),
        ];
        let pick = bg.select(&cands, &[], 42).unwrap();
        assert_eq!(pick, 1, "distressed user should land on the warm candidate");
    }

    #[test]
    fn excited_user_selects_playful() {
        let mut bg = BasalGanglia::new(true);
        bg.ctx.user_affect = UserAffect::from_prompt("let's play, go get the toy!");
        let cands = vec![
            cand(
                0,
                "I am here with you. Stay close and rest, you are safe.",
                1.0,
            ),
            cand(
                1,
                "I chase and pounce! Zoom! Catch the feather wand, so fun!",
                1.0,
            ),
        ];
        let pick = bg.select(&cands, &[], 7).unwrap();
        assert_eq!(pick, 1, "excited user should land on the playful candidate");
    }

    #[test]
    fn avoids_recent_repeat_when_alternative_exists() {
        let bg = BasalGanglia::new(true);
        let repeated = "The sun is in its spot. The birds are normal volume.";
        let cands = vec![
            cand(0, repeated, 1.0),
            cand(
                1,
                "I trot over and bump your ankle with a soft trill.",
                0.98,
            ),
        ];
        let recent = vec![repeated.to_string()];
        let pick = bg.select(&cands, &recent, 1).unwrap();
        assert_eq!(pick, 1, "should skip the just-used line for a fresh one");
    }

    #[test]
    fn rejects_garbled_candidate() {
        let bg = BasalGanglia::new(true);
        let cands = vec![
            cand(
                0,
                "I do is the and warm. I stretch. I front I.. you back. I. to.",
                1.2,
            ),
            cand(
                1,
                "I flick both ears forward and trot over with a trill.",
                0.9,
            ),
        ];
        let pick = bg.select(&cands, &[], 99).unwrap();
        assert_eq!(pick, 1, "garbled top-scorer must be vetoed");
    }

    #[test]
    fn dopamine_widens_serotonin_narrows_selection() {
        // High dopamine (hungry/seeking) should admit more survivors than high
        // serotonin (sated/content) for the same value spread.
        let seeking = DriveState {
            hunger: 0.95,
            energy: 0.8,
            social: 0.1,
        }
        .map_neuromodulators();
        let content = DriveState {
            hunger: 0.05,
            energy: 0.4,
            social: 0.95,
        }
        .map_neuromodulators();
        let cands = vec![
            cand(0, "alpha response one here", 1.00),
            cand(1, "beta response two here", 0.92),
            cand(2, "gamma response three here", 0.85),
        ];
        let mut bg_seek = BasalGanglia::new(true);
        bg_seek.ctx.neuromods = Some(seeking);
        let mut bg_cont = BasalGanglia::new(true);
        bg_cont.ctx.neuromods = Some(content);
        // Run many seeds; seeking should produce more variety in picks.
        let variety = |bg: &BasalGanglia| {
            let mut seen = std::collections::HashSet::new();
            for s in 0..200u64 {
                if let Some(p) = bg.select(&cands, &[], s.wrapping_mul(2654435761)) {
                    seen.insert(p);
                }
            }
            seen.len()
        };
        assert!(
            variety(&bg_seek) >= variety(&bg_cont),
            "dopamine-seeking should explore at least as much as content"
        );
    }

    #[test]
    fn single_or_empty_candidate_is_safe() {
        let bg = BasalGanglia::new(true);
        assert_eq!(bg.select(&[], &[], 0), None);
        let one = vec![cand(0, "only option", 1.0)];
        assert_eq!(bg.select(&one, &[], 0), Some(0));
    }
}
