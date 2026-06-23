//! FragmentComposer — free-text generation by composing typed sentence fragments.
//!
//! The retrieval/lattice path emits *whole* training sentences (or verbatim
//! nearest-neighbours), so distinct prompts collapse onto a small set of canned
//! responses. This module makes the *text* layer compose the same three voices
//! the [`crate::reflective_field::ReflectiveField`] already composes in
//! conditioning space:
//!
//! ```text
//!   Identity  — who I am            (stable persona / OCEAN, vocalizations)
//!   Activity  — what I'm doing      (action / momentum content)
//!   Drive     — how I feel now      (state-gated needs: hunger/energy/mood)
//! ```
//!
//! A response is assembled from a skeleton `[opener?] body+ [coda?]`, sampling
//! one fragment per slot from the eligible pool. Eligibility is gated by intent
//! affinity and runtime `state` ranges; selection is scored by OCEAN affinity ×
//! the live per-voice [`ReflectiveWeights`]. With a handful of fragments per
//! voice per intent this yields hundreds of coherent, state-varying outputs that
//! remain fully traceable to authored, policy-safe clauses.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use crate::reflective_field::ReflectiveWeights;

/// The three reflective voices — 1:1 with [`ReflectiveWeights`] fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Voice {
    /// Stable persona / OCEAN expression and vocalizations.
    Identity,
    /// What the agent is doing — action/momentum content.
    Activity,
    /// How the agent feels now — state-gated needs.
    Drive,
}

impl Voice {
    /// The live blend weight for this voice from the reflective field.
    pub fn weight(self, w: &ReflectiveWeights) -> f32 {
        match self {
            Voice::Identity => w.identity,
            Voice::Activity => w.activity,
            Voice::Drive => w.drive,
        }
    }
}

/// Position of a fragment within the assembled sentence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SlotRole {
    /// Conversational framing prefix (optional, at most one).
    Opener,
    /// Main clause(s) — one or more compose the body.
    Body,
    /// Trailing marker (optional, at most one), e.g. a vocalization.
    Coda,
}

fn default_weight() -> f32 {
    1.0
}

/// A single reusable sentence fragment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fragment {
    pub fragment_id: String,
    pub voice: Voice,
    pub text: String,
    pub role: SlotRole,
    /// `semantic_intent` / `graph_anchors` this fragment is eligible for.
    /// `"*"` matches any intent (e.g. generic vocalizations).
    #[serde(default)]
    pub intent_affinity: Vec<String>,
    /// Personality direction; scored against the agent's OCEAN profile.
    /// Keys are any of `"O"`,`"C"`,`"E"`,`"A"`,`"N"`; missing ⇒ 0.
    #[serde(default)]
    pub ocean_affinity: HashMap<String, f32>,
    /// Inclusive runtime-state ranges gating eligibility, e.g. `{"hunger":[0.65,1.0]}`.
    /// Empty ⇒ always eligible.
    #[serde(default)]
    pub state_gate: HashMap<String, [f32; 2]>,
    /// Typed vocalization marker (kept separate so "catness" is a mood-gated knob).
    #[serde(default)]
    pub vocalization: Option<String>,
    /// Archetype this fragment belongs to; `None` ⇒ archetype-agnostic.
    #[serde(default)]
    pub archetype: Option<String>,
    /// Prior on selection within its eligible pool.
    #[serde(default = "default_weight")]
    pub weight: f32,
    /// Optional sub-role for templated body assembly (`empathic`, `action`, …).
    #[serde(default)]
    pub body_slot: Option<String>,
    /// Intents for which this fragment must not be selected.
    #[serde(default)]
    pub intent_exclude: Vec<String>,
}

/// Per-intent negative filters resolved from inference TOML.
#[derive(Debug, Clone, Default)]
pub struct ComposeExcludes {
    pub body_slots: HashSet<String>,
    pub keywords: Vec<String>,
    pub fragment_intents: HashSet<String>,
}

impl ComposeExcludes {
    pub fn blocks_fragment(&self, intent: &str, fragment: &Fragment) -> bool {
        if fragment
            .intent_exclude
            .iter()
            .any(|ex| ex == intent || ex == "*")
        {
            return true;
        }
        if let Some(ref slot) = fragment.body_slot {
            if self.body_slots.contains(slot) {
                return true;
            }
        }
        if fragment.intent_affinity.iter().any(|a| self.fragment_intents.contains(a)) {
            return true;
        }
        if !self.keywords.is_empty() {
            let lower = fragment.text.to_ascii_lowercase();
            if self.keywords.iter().any(|kw| lower.contains(kw)) {
                return true;
            }
        }
        false
    }
}

impl Fragment {
    /// Whether this fragment is eligible given the intent, anchors, runtime
    /// state, and active archetype.
    pub fn is_eligible(&self, ctx: &ComposeContext) -> bool {
        self.is_eligible_for(ctx, false)
    }

    /// Opener slots require a direct intent (or `"*"`) match so shared graph
    /// anchors like `bonding_moment` do not bleed into unrelated intents.
    pub fn is_eligible_opener(&self, ctx: &ComposeContext) -> bool {
        self.is_eligible_for(ctx, true)
    }

    /// Lore Q&A body/stance shards tagged with a specific topic (favorite_meal, lore_origin, …)
    /// must match that topic anchor — not the broad `lore_qa` intent alone.
    fn requires_lore_topic_anchor(&self, ctx: &ComposeContext) -> bool {
        if ctx.intent != "lore_qa" {
            return false;
        }
        match self.body_slot.as_deref() {
            Some("lore") => true,
            Some("stance") => self.has_specific_topic_affinity(),
            _ => false,
        }
    }

    fn has_specific_topic_affinity(&self) -> bool {
        self.intent_affinity.iter().any(|a| is_lore_topic_anchor(a))
    }

    fn matches_lore_topic_anchor(&self, ctx: &ComposeContext) -> bool {
        self.intent_affinity.iter().any(|a| {
            is_lore_topic_anchor(a) && ctx.graph_anchors.iter().any(|g| g == a)
        })
    }

    fn is_eligible_for(&self, ctx: &ComposeContext, opener_strict: bool) -> bool {
        // Intent / anchor affinity.
        if !self.intent_affinity.is_empty() {
            if self.requires_lore_topic_anchor(ctx) {
                if !self.matches_lore_topic_anchor(ctx) {
                    return false;
                }
            } else {
                let direct = self
                    .intent_affinity
                    .iter()
                    .any(|a| a == "*" || a == &ctx.intent);
                let anchor = if opener_strict {
                    false
                } else {
                    self.intent_affinity.iter().any(|a| {
                        a != "*"
                            && !is_meta_anchor(a)
                            && ctx.graph_anchors.iter().any(|g| g == a)
                    })
                };
                if !direct && !anchor {
                    return false;
                }
            }
        }
        // Archetype gate.
        if let Some(ref af) = self.archetype {
            if let Some(ref cf) = ctx.archetype {
                if af != cf {
                    return false;
                }
            }
        }
        // Runtime-state gates. A gate *requires* knowledge of its dimension: if
        // the current state doesn't provide it, the fragment is ineligible rather
        // than assuming a (misleading) default. This prevents a stateless query
        // from spuriously satisfying every low-range gate via an implicit 0.0.
        for (dim, &[lo, hi]) in &self.state_gate {
            match ctx.state.get(dim) {
                Some(&v) if v >= lo && v <= hi => {}
                _ => return false,
            }
        }
        if ctx.excludes.blocks_fragment(&ctx.intent, self) {
            return false;
        }
        true
    }

    /// Selection score: prior × OCEAN alignment × live voice weight.
    pub fn score(&self, ctx: &ComposeContext) -> f32 {
        let voice_w = self.voice.weight(&ctx.weights).max(0.0);
        let ocean_align = 0.5 + 0.5 * self.ocean_cosine(&ctx.ocean);
        // Keep a small floor so a near-zero voice weight still allows fallback.
        self.weight.max(0.0) * ocean_align * (0.05 + voice_w)
    }

    /// Cosine between this fragment's OCEAN affinity and the agent profile.
    fn ocean_cosine(&self, ocean: &[f32; 5]) -> f32 {
        const KEYS: [&str; 5] = ["O", "C", "E", "A", "N"];
        let mut dot = 0.0f32;
        let mut na = 0.0f32;
        for (i, k) in KEYS.iter().enumerate() {
            let a = self.ocean_affinity.get(*k).copied().unwrap_or(0.0);
            // Center the profile around the neutral 0.5 so traits read as ±.
            let b = ocean[i] - 0.5;
            dot += a * b;
            na += a * a;
        }
        let nb: f32 = ocean.iter().map(|o| (o - 0.5) * (o - 0.5)).sum::<f32>();
        if na < 1e-9 || nb < 1e-9 {
            0.0
        } else {
            dot / (na.sqrt() * nb.sqrt())
        }
    }
}

/// Species/archetype tags and broad fallbacks — not useful for anchor-only matching.
fn is_meta_anchor(a: &str) -> bool {
    matches!(
        a,
        "cheerful_companion"
            | "siamese"
            | "cat"
            | "open_ended_chat"
            | "contented"
            | "playful_state"
            | "curious_state"
            | "routine_adherence"
            | "proactive_luna"
            | "vocal_communication"
            | "lore_qa"
    )
}

/// Topic-specific lore anchor (not the broad intent label).
fn is_lore_topic_anchor(a: &str) -> bool {
    !is_meta_anchor(a) && a != "lore_qa"
}

/// Inputs for a single composition turn.
#[derive(Debug, Clone)]
pub struct ComposeContext {
    pub intent: String,
    pub graph_anchors: Vec<String>,
    /// Agent OCEAN profile `[O, C, E, A, N]` in `[0, 1]`.
    pub ocean: [f32; 5],
    /// Runtime state dims (e.g. `hunger`, `energy`, `mood`) in `[0, 1]`.
    pub state: HashMap<String, f32>,
    /// Live per-voice blend weights from the reflective field.
    pub weights: ReflectiveWeights,
    /// Active archetype, if any.
    pub archetype: Option<String>,
    /// Deterministic seed (e.g. derived from conditioning + turn counter).
    pub seed: u64,
    /// When true, always try to add a second body from a different voice.
    pub force_second_body: bool,
    /// When false, skip the opener slot even if eligible openers exist.
    pub use_opener: bool,
    /// Ordered body sub-slots from a compose template; `None` ⇒ legacy sampling.
    pub body_slots: Option<Vec<String>>,
    /// Minimum body fragments required (from template or default 1).
    pub min_bodies: usize,
    /// When true, templated bodies must come from distinct voices.
    pub require_distinct_voices: bool,
    /// Negative affinity filters for the active intent.
    pub excludes: ComposeExcludes,
}

/// A composed response and the fragments that produced it (for tracing).
#[derive(Debug, Clone)]
pub struct ComposedResponse {
    pub text: String,
    pub fragment_ids: Vec<String>,
    /// Number of distinct voices represented (a coarse coherence/variety signal).
    pub voices_used: usize,
    /// Body fragments in the assembled line (excludes opener/coda).
    pub body_count: usize,
    /// Total characters across body fragments.
    pub body_chars: usize,
}

/// A loaded set of fragments plus the composition policy.
#[derive(Debug, Clone, Default)]
pub struct FragmentComposer {
    pub fragments: Vec<Fragment>,
}

impl FragmentComposer {
    pub fn new(fragments: Vec<Fragment>) -> Self {
        Self { fragments }
    }

    /// Parse a JSONL fragment library (one [`Fragment`] per non-empty line).
    /// Malformed lines are skipped; returns the composer plus the skip count.
    pub fn from_jsonl_str(s: &str) -> (Self, usize) {
        let mut fragments = Vec::new();
        let mut skipped = 0usize;
        for line in s.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            match serde_json::from_str::<Fragment>(line) {
                Ok(f) => fragments.push(f),
                Err(_) => skipped += 1,
            }
        }
        (Self { fragments }, skipped)
    }

    /// Load a JSONL fragment library from disk.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn from_path<P: AsRef<std::path::Path>>(path: P) -> std::io::Result<(Self, usize)> {
        let s = std::fs::read_to_string(path)?;
        Ok(Self::from_jsonl_str(&s))
    }

    pub fn is_empty(&self) -> bool {
        self.fragments.is_empty()
    }

    /// Compose a response for the given turn. Returns `None` when no fragment is
    /// eligible (caller should fall back to the existing generation path).
    ///
    /// Skeleton: `[opener?] body (+ a second body if a distinct voice exists) [coda?]`.
    /// Selection is deterministic given `ctx.seed`.
    pub fn compose(&self, ctx: &ComposeContext) -> Option<ComposedResponse> {
        let mut rng = ctx.seed | 1;

        // Eligible fragments partitioned by role.
        let eligible: Vec<&Fragment> = self
            .fragments
            .iter()
            .filter(|f| f.is_eligible(ctx))
            .collect();
        if eligible.is_empty() {
            return None;
        }

        let pool = |role: SlotRole| -> Vec<&Fragment> {
            eligible.iter().copied().filter(|f| f.role == role).collect()
        };

        let opener_pool: Vec<&Fragment> = if ctx.use_opener {
            self.fragments
                .iter()
                .filter(|f| f.role == SlotRole::Opener && f.is_eligible_opener(ctx))
                .collect()
        } else {
            Vec::new()
        };

        let mut chosen: Vec<&Fragment> = Vec::new();

        // Opener: optional framing prefix — only for greeting/reunion/comfort intents.
        if ctx.use_opener {
            if let Some(op) = Self::sample(&opener_pool, ctx, &mut rng) {
                chosen.push(op);
            }
        }

        let body_pool = pool(SlotRole::Body);
        let mut body_count = 0usize;
        let mut body_chars = 0usize;

        if let Some(ref slots) = ctx.body_slots {
            if !Self::compose_templated_bodies(
                slots,
                &body_pool,
                ctx,
                &mut rng,
                &mut chosen,
                &mut body_count,
                &mut body_chars,
            ) {
                return None;
            }
        } else if let Some(b1) = Self::sample(&body_pool, ctx, &mut rng) {
            chosen.push(b1);
            body_count += 1;
            body_chars += b1.text.len();
            let second_pool: Vec<&Fragment> = body_pool
                .iter()
                .copied()
                .filter(|f| f.voice != b1.voice && f.fragment_id != b1.fragment_id)
                .collect();
            if !second_pool.is_empty() {
                let add_p = if ctx.force_second_body {
                    1.0
                } else {
                    (ctx.weights.activity + ctx.weights.drive).clamp(0.0, 1.0)
                };
                if Self::next_uniform(&mut rng) < add_p {
                    if let Some(b2) = Self::sample(&second_pool, ctx, &mut rng) {
                        chosen.push(b2);
                        body_count += 1;
                        body_chars += b2.text.len();
                    }
                }
            }
        } else {
            return None;
        }

        if body_count < ctx.min_bodies.max(1) {
            return None;
        }

        // Coda: optional trailing marker (e.g. vocalization).
        if let Some(cd) = Self::sample(&pool(SlotRole::Coda), ctx, &mut rng) {
            chosen.push(cd);
        }

        let mut voices: Vec<Voice> = chosen.iter().map(|f| f.voice).collect();
        voices.sort_by_key(|v| *v as u8);
        voices.dedup();

        let text = chosen
            .iter()
            .map(|f| f.text.trim())
            .collect::<Vec<_>>()
            .join(" ");

        Some(ComposedResponse {
            text,
            fragment_ids: chosen.iter().map(|f| f.fragment_id.clone()).collect(),
            voices_used: voices.len(),
            body_count,
            body_chars,
        })
    }

    /// Fill ordered body sub-slots from a compose template.
    fn compose_templated_bodies<'a>(
        slots: &[String],
        body_pool: &[&'a Fragment],
        ctx: &ComposeContext,
        rng: &mut u64,
        chosen: &mut Vec<&'a Fragment>,
        body_count: &mut usize,
        body_chars: &mut usize,
    ) -> bool {
        let already = |f: &&Fragment, picked: &[&Fragment]| {
            picked.iter().any(|p| p.fragment_id == f.fragment_id)
        };

        let mut picked_bodies: Vec<&Fragment> = Vec::new();

        for slot in slots {
            let slot_pool: Vec<&Fragment> = body_pool
                .iter()
                .copied()
                .filter(|f| f.body_slot.as_deref() == Some(slot.as_str()))
                .filter(|f| !already(f, &picked_bodies))
                .filter(|f| {
                    !ctx.require_distinct_voices
                        || picked_bodies.iter().all(|p| p.voice != f.voice)
                })
                .collect();

            let fallback_pool: Vec<&Fragment> = body_pool
                .iter()
                .copied()
                .filter(|f| f.body_slot.is_none())
                .filter(|f| !already(f, &picked_bodies))
                .filter(|f| {
                    !ctx.require_distinct_voices
                        || picked_bodies.iter().all(|p| p.voice != f.voice)
                })
                .collect();

            let picked = if slot == "lore" {
                Self::sample(&slot_pool, ctx, rng)
            } else {
                Self::sample(&slot_pool, ctx, rng)
                    .or_else(|| Self::sample(&fallback_pool, ctx, rng))
            };
            if let Some(b) = picked {
                chosen.push(b);
                picked_bodies.push(b);
                *body_count += 1;
                *body_chars += b.text.len();
            }
        }

        *body_count > 0
    }

    /// Weighted, seeded sampling from a fragment pool by [`Fragment::score`].
    fn sample<'a>(
        pool: &[&'a Fragment],
        ctx: &ComposeContext,
        rng: &mut u64,
    ) -> Option<&'a Fragment> {
        if pool.is_empty() {
            return None;
        }
        let scores: Vec<f32> = pool.iter().map(|f| f.score(ctx).max(0.0)).collect();
        let total: f32 = scores.iter().sum();
        if total <= 1e-9 {
            // All-zero scores: fall back to uniform choice.
            let idx = (Self::next_u64(rng) as usize) % pool.len();
            return Some(pool[idx]);
        }
        let r = Self::next_uniform(rng) * total;
        let mut acc = 0.0f32;
        for (f, s) in pool.iter().zip(&scores) {
            acc += s;
            if r <= acc {
                return Some(f);
            }
        }
        pool.last().copied()
    }

    /// xorshift64 step.
    fn next_u64(state: &mut u64) -> u64 {
        let mut x = *state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *state = x;
        x
    }

    /// Uniform float in `[0, 1)`.
    fn next_uniform(state: &mut u64) -> f32 {
        ((Self::next_u64(state) >> 40) as f32) / (1u64 << 24) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frag(
        id: &str,
        voice: Voice,
        role: SlotRole,
        text: &str,
        intents: &[&str],
        gate: &[(&str, [f32; 2])],
    ) -> Fragment {
        Fragment {
            fragment_id: id.to_string(),
            voice,
            text: text.to_string(),
            role,
            intent_affinity: intents.iter().map(|s| s.to_string()).collect(),
            ocean_affinity: HashMap::new(),
            state_gate: gate.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
            vocalization: None,
            archetype: None,
            weight: 1.0,
            body_slot: None,
            intent_exclude: Vec::new(),
        }
    }

    fn ctx(state: &[(&str, f32)], seed: u64) -> ComposeContext {
        ComposeContext {
            intent: "open_ended_chat".to_string(),
            graph_anchors: vec![],
            ocean: [0.7, 0.5, 0.8, 0.7, 0.4],
            state: state.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
            weights: ReflectiveWeights::default(),
            archetype: None,
            seed,
            force_second_body: false,
            use_opener: true,
            body_slots: None,
            min_bodies: 1,
            require_distinct_voices: false,
            excludes: ComposeExcludes::default(),
        }
    }

    fn sample_library() -> FragmentComposer {
        FragmentComposer::new(vec![
            frag("id_gaze", Voice::Identity, SlotRole::Body,
                 "I sit and look at you.", &["open_ended_chat"], &[]),
            frag("act_knock", Voice::Activity, SlotRole::Body,
                 "I knock the pen off the table.", &["open_ended_chat"], &[("energy", [0.5, 1.0])]),
            frag("drive_hungry", Voice::Drive, SlotRole::Body,
                 "My stomach has opinions.", &["open_ended_chat"], &[("hunger", [0.65, 1.0])]),
            frag("coda_trill", Voice::Identity, SlotRole::Coda,
                 "Trill.", &["*"], &[]),
        ])
    }

    #[test]
    fn composes_some_text() {
        let lib = sample_library();
        let out = lib.compose(&ctx(&[("hunger", 0.3), ("energy", 0.8)], 42)).unwrap();
        assert!(!out.text.is_empty());
        assert!(out.fragment_ids.len() >= 1);
    }

    #[test]
    fn state_gate_excludes_hungry_fragment_when_sated() {
        let lib = sample_library();
        // Low hunger: the hungry drive fragment must never appear.
        for seed in 0..50u64 {
            let out = lib.compose(&ctx(&[("hunger", 0.1), ("energy", 0.9)], seed)).unwrap();
            assert!(
                !out.fragment_ids.iter().any(|id| id == "drive_hungry"),
                "hungry fragment leaked while sated: {:?}",
                out.fragment_ids
            );
        }
    }

    #[test]
    fn state_gate_allows_hungry_fragment_when_hungry() {
        let lib = sample_library();
        let appeared = (0..50u64).any(|seed| {
            lib.compose(&ctx(&[("hunger", 0.9), ("energy", 0.9)], seed))
                .map(|o| o.fragment_ids.iter().any(|id| id == "drive_hungry"))
                .unwrap_or(false)
        });
        assert!(appeared, "hungry fragment never appeared while hungry");
    }

    #[test]
    fn composition_is_deterministic_per_seed() {
        let lib = sample_library();
        let a = lib.compose(&ctx(&[("hunger", 0.9), ("energy", 0.9)], 7)).unwrap();
        let b = lib.compose(&ctx(&[("hunger", 0.9), ("energy", 0.9)], 7)).unwrap();
        assert_eq!(a.text, b.text);
    }

    #[test]
    fn varies_across_seeds() {
        let lib = sample_library();
        let mut texts = std::collections::HashSet::new();
        for seed in 0..30u64 {
            if let Some(o) = lib.compose(&ctx(&[("hunger", 0.9), ("energy", 0.9)], seed)) {
                texts.insert(o.text);
            }
        }
        assert!(texts.len() > 1, "composition collapsed to a single output");
    }

    #[test]
    fn empty_when_no_eligible_intent() {
        let lib = sample_library();
        let mut c = ctx(&[("hunger", 0.5)], 1);
        c.intent = "totally_unknown_intent".to_string();
        // Only the "*" coda is eligible; with no body, compose returns None.
        assert!(lib.compose(&c).is_none());
    }

    #[test]
    fn gate_requires_known_dimension() {
        // A fragment gating a dimension absent from ctx.state must be ineligible
        // (not assumed satisfied via an implicit default).
        let lib = FragmentComposer::new(vec![
            frag("needs_social", Voice::Drive, SlotRole::Body, "lonely.",
                 &["open_ended_chat"], &[("social", [0.5, 1.0])]),
            frag("ungated", Voice::Identity, SlotRole::Body, "here.",
                 &["open_ended_chat"], &[]),
        ]);
        for seed in 0..30u64 {
            // ctx provides only "hunger" — never "social".
            if let Some(o) = lib.compose(&ctx(&[("hunger", 0.5)], seed)) {
                assert!(
                    !o.fragment_ids.iter().any(|id| id == "needs_social"),
                    "fragment gating an unknown dimension leaked: {:?}",
                    o.fragment_ids
                );
            }
        }
    }

    #[test]
    fn opener_requires_direct_intent_not_shared_anchor() {
        let lib = FragmentComposer::new(vec![
            frag("open_lulu", Voice::Identity, SlotRole::Opener,
                 "Lulu — the soft name.", &["bonding_moment"], &[]),
            frag("body_school", Voice::Identity, SlotRole::Body,
                 "School is loud in your head.", &["school_stress"], &[]),
        ]);
        let mut c = ctx(&[("mood", 0.4)], 42);
        c.intent = "school_stress".to_string();
        c.graph_anchors = vec![
            "school_stress".into(),
            "emotional_support".into(),
            "bonding_moment".into(),
        ];
        c.use_opener = true;
        let out = lib.compose(&c).unwrap();
        assert!(!out.text.starts_with("Lulu"));
        assert!(out.text.contains("School"));
    }

    #[test]
    fn meta_anchor_does_not_unlock_open_ended_fragments() {
        let lib = FragmentComposer::new(vec![
            frag("school_only", Voice::Identity, SlotRole::Body,
                 "School is loud in your head.", &["school_stress"], &[]),
            frag("generic_helping", Voice::Identity, SlotRole::Body,
                 "That is helping.", &["open_ended_chat"], &[]),
        ]);
        let mut c = ctx(&[("mood", 0.4)], 99);
        c.intent = "school_stress".to_string();
        c.graph_anchors = vec!["school_stress".into(), "cheerful_companion".into()];
        let out = lib.compose(&c).unwrap();
        assert!(out.text.contains("School"));
        assert!(!out.text.contains("That is helping"));
    }

    #[test]
    fn skips_opener_when_disabled() {
        let lib = FragmentComposer::new(vec![
            frag("open_there", Voice::Identity, SlotRole::Opener,
                 "There you are.", &["open_ended_chat"], &[]),
            frag("body_here", Voice::Identity, SlotRole::Body,
                 "I sit and look at you.", &["open_ended_chat"], &[]),
        ]);
        let mut c = ctx(&[("hunger", 0.5)], 42);
        c.use_opener = false;
        let out = lib.compose(&c).unwrap();
        assert!(!out.text.starts_with("There you are"));
        assert!(out.text.contains("look at you"));
    }

    #[test]
    fn jsonl_roundtrip_parses() {
        let line = r#"{"fragment_id":"x","voice":"drive","text":"hi","role":"body","intent_affinity":["open_ended_chat"],"state_gate":{"hunger":[0.6,1.0]},"body_slot":"action"}"#;
        let (lib, skipped) = FragmentComposer::from_jsonl_str(line);
        assert_eq!(skipped, 0);
        assert_eq!(lib.fragments.len(), 1);
        assert_eq!(lib.fragments[0].voice, Voice::Drive);
        assert_eq!(lib.fragments[0].role, SlotRole::Body);
        assert_eq!(lib.fragments[0].body_slot.as_deref(), Some("action"));
    }

    #[test]
    fn compose_template_fills_body_slots() {
        let lib = FragmentComposer::new(vec![
            Fragment {
                fragment_id: "emp".into(),
                voice: Voice::Identity,
                text: "School is loud in your head.".into(),
                role: SlotRole::Body,
                intent_affinity: vec!["school_stress".into()],
                ocean_affinity: HashMap::new(),
                state_gate: HashMap::new(),
                vocalization: None,
                archetype: None,
                weight: 1.0,
                body_slot: Some("empathic".into()),
                intent_exclude: Vec::new(),
            },
            Fragment {
                fragment_id: "act".into(),
                voice: Voice::Activity,
                text: "I climb into your lap.".into(),
                role: SlotRole::Body,
                intent_affinity: vec!["school_stress".into()],
                ocean_affinity: HashMap::new(),
                state_gate: HashMap::new(),
                vocalization: None,
                archetype: None,
                weight: 1.0,
                body_slot: Some("action".into()),
                intent_exclude: Vec::new(),
            },
        ]);
        let mut c = ctx(&[("mood", 0.4)], 5);
        c.intent = "school_stress".into();
        c.body_slots = Some(vec!["empathic".into(), "action".into()]);
        c.min_bodies = 2;
        c.use_opener = false;
        let out = lib.compose(&c).unwrap();
        assert_eq!(out.body_count, 2);
        assert!(out.text.contains("School"));
        assert!(out.text.contains("lap"));
    }

    #[test]
    fn intent_exclude_blocks_fragment() {
        let lib = FragmentComposer::new(vec![
            Fragment {
                fragment_id: "meal".into(),
                voice: Voice::Drive,
                text: "Food time always.".into(),
                role: SlotRole::Body,
                intent_affinity: vec!["open_ended_chat".into()],
                ocean_affinity: HashMap::new(),
                state_gate: HashMap::new(),
                vocalization: None,
                archetype: None,
                weight: 1.0,
                body_slot: Some("mealtime".into()),
                intent_exclude: vec!["school_stress".into()],
            },
            Fragment {
                fragment_id: "emp".into(),
                voice: Voice::Identity,
                text: "I hear you.".into(),
                role: SlotRole::Body,
                intent_affinity: vec!["school_stress".into()],
                ocean_affinity: HashMap::new(),
                state_gate: HashMap::new(),
                vocalization: None,
                archetype: None,
                weight: 1.0,
                body_slot: Some("empathic".into()),
                intent_exclude: Vec::new(),
            },
        ]);
        let mut c = ctx(&[("mood", 0.4)], 9);
        c.intent = "school_stress".into();
        c.use_opener = false;
        let out = lib.compose(&c).unwrap();
        assert!(!out.text.contains("Food time"));
    }

    #[test]
    fn compose_excludes_block_by_keyword() {
        let lib = FragmentComposer::new(vec![
            Fragment {
                fragment_id: "bad".into(),
                voice: Voice::Identity,
                text: "You taste like you.".into(),
                role: SlotRole::Body,
                intent_affinity: vec!["emotional_support".into()],
                ocean_affinity: HashMap::new(),
                state_gate: HashMap::new(),
                vocalization: None,
                archetype: None,
                weight: 1.0,
                body_slot: None,
                intent_exclude: Vec::new(),
            },
            Fragment {
                fragment_id: "good".into(),
                voice: Voice::Identity,
                text: "I sit with you.".into(),
                role: SlotRole::Body,
                intent_affinity: vec!["emotional_support".into()],
                ocean_affinity: HashMap::new(),
                state_gate: HashMap::new(),
                vocalization: None,
                archetype: None,
                weight: 1.0,
                body_slot: None,
                intent_exclude: Vec::new(),
            },
        ]);
        let mut c = ctx(&[("mood", 0.4)], 11);
        c.intent = "emotional_support".into();
        c.use_opener = false;
        c.excludes.keywords = vec!["you taste".into()];
        let out = lib.compose(&c).unwrap();
        assert!(!out.text.contains("taste"));
    }

    #[test]
    fn lore_topic_anchor_blocks_cross_topic_body() {
        let lib = FragmentComposer::new(vec![
            Fragment {
                fragment_id: "stance".into(),
                voice: Voice::Identity,
                text: "Hmm. A question worth answering properly.".into(),
                role: SlotRole::Body,
                intent_affinity: vec!["lore_qa".into()],
                ocean_affinity: HashMap::new(),
                state_gate: HashMap::new(),
                vocalization: None,
                archetype: None,
                weight: 1.0,
                body_slot: Some("stance".into()),
                intent_exclude: Vec::new(),
            },
            Fragment {
                fragment_id: "meal".into(),
                voice: Voice::Identity,
                text: "Sushi rolls. Rice, fish, dignity.".into(),
                role: SlotRole::Body,
                intent_affinity: vec!["favorite_meal".into()],
                ocean_affinity: HashMap::new(),
                state_gate: HashMap::new(),
                vocalization: None,
                archetype: None,
                weight: 1.2,
                body_slot: Some("lore".into()),
                intent_exclude: Vec::new(),
            },
            Fragment {
                fragment_id: "nick".into(),
                voice: Voice::Identity,
                text: "LuLu is my nickname.".into(),
                role: SlotRole::Body,
                intent_affinity: vec!["lore_nickname".into(), "lore_qa".into()],
                ocean_affinity: HashMap::new(),
                state_gate: HashMap::new(),
                vocalization: None,
                archetype: None,
                weight: 1.0,
                body_slot: Some("lore".into()),
                intent_exclude: Vec::new(),
            },
        ]);
        let mut c = ctx(&[], 42);
        c.intent = "lore_qa".into();
        c.graph_anchors = vec!["lore_qa".into(), "favorite_meal".into()];
        c.body_slots = Some(vec!["stance".into(), "lore".into()]);
        c.min_bodies = 2;
        c.use_opener = false;
        let out = lib.compose(&c).unwrap();
        assert!(out.text.contains("Sushi rolls"));
        assert!(!out.text.contains("nickname"));
    }
}
