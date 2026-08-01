//! Turn-level context frame: mood + intent gradients → blended OCEAN for compose.
//!
//! This is the first rung toward compositional / pragmatic response structure:
//! understand the prompt as a soft distribution over intents and a mood field,
//! then answer through personality (OCEAN) shifted by that understanding — not
//! a single hard label alone.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::basal_ganglia::UserAffect;
use crate::inference::inference_toml::{FragmentComposeConfig, FragmentIntentHint};

/// Illocutionary force of the user turn (coarse speech-act).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpeechAct {
    Greet,
    Query,
    Inform,
    Express,
    ComfortSeek,
    Request,
    Offer,
    Refuse,
    Assert,
    Other,
}

impl SpeechAct {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Greet => "greet",
            Self::Query => "query",
            Self::Inform => "inform",
            Self::Express => "express",
            Self::ComfortSeek => "comfort_seek",
            Self::Request => "request",
            Self::Offer => "offer",
            Self::Refuse => "refuse",
            Self::Assert => "assert",
            Self::Other => "other",
        }
    }
}

/// Continuous mood field derived from vitals + prompt affect.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MoodGradient {
    /// −1 distressed … +1 content (from state mood + prompt).
    pub valence: f32,
    /// 0 calm … 1 activated (energy / excitement).
    pub arousal: f32,
    /// 0 withdrawn … 1 socially engaged.
    pub social: f32,
}

impl Default for MoodGradient {
    fn default() -> Self {
        Self {
            valence: 0.0,
            arousal: 0.4,
            social: 0.5,
        }
    }
}

/// Soft mass on one intent label.
#[derive(Debug, Clone, PartialEq)]
pub struct IntentMass {
    pub intent: String,
    pub mass: f32,
}

/// Snapshot used to understand then respond through OCEAN-conditioned compose.
#[derive(Debug, Clone)]
pub struct ContextFrame {
    pub speech_act: SpeechAct,
    pub mood: MoodGradient,
    pub intent_primary: String,
    pub intent_masses: Vec<IntentMass>,
    /// Base personality `[O, C, E, A, N]`.
    pub ocean_base: [f32; 5],
    /// Mood/intent-modulated personality for fragment scoring.
    pub ocean_blended: [f32; 5],
    pub anchors: Vec<String>,
    pub user_affect: UserAffect,
}

/// Compact prior-turn snapshot stored on [`crate::service::ConversationContext`].
#[derive(Debug, Clone, PartialEq)]
pub struct PersistedContextFrame {
    pub speech_act: SpeechAct,
    pub mood: MoodGradient,
    pub intent_primary: String,
    pub ocean_blended: [f32; 5],
}

/// Knobs under `[fragment_compose.context_frame]`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContextFrameConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// How strongly mood gradients shift OCEAN (0 = base personality only).
    #[serde(default = "default_mood_ocean_strength")]
    pub mood_ocean_strength: f32,
    /// How strongly primary/secondary intent shifts OCEAN.
    #[serde(default = "default_intent_ocean_strength")]
    pub intent_ocean_strength: f32,
    /// Mass reserved for secondary soft intents (rest stays on primary).
    #[serde(default = "default_secondary_intent_mass")]
    pub secondary_intent_mass: f32,
    /// How strongly the previous turn's mood/OCEAN linger into this frame (0 = none).
    #[serde(default = "default_discourse_mood_blend")]
    pub discourse_mood_blend: f32,
}

fn default_true() -> bool {
    true
}
fn default_mood_ocean_strength() -> f32 {
    0.28
}
fn default_intent_ocean_strength() -> f32 {
    0.18
}
fn default_secondary_intent_mass() -> f32 {
    0.35
}
fn default_discourse_mood_blend() -> f32 {
    0.22
}

impl Default for ContextFrameConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            mood_ocean_strength: default_mood_ocean_strength(),
            intent_ocean_strength: default_intent_ocean_strength(),
            secondary_intent_mass: default_secondary_intent_mass(),
            discourse_mood_blend: default_discourse_mood_blend(),
        }
    }
}

impl ContextFrame {
    /// Build a frame from prompt, hard intent hint, vitals, and base OCEAN.
    pub fn build(
        prompt: &str,
        hints: &FragmentIntentHint,
        state: &HashMap<String, f32>,
        ocean_base: [f32; 5],
        fc: &FragmentComposeConfig,
        cfg: &ContextFrameConfig,
    ) -> Self {
        let user_affect = UserAffect::from_prompt(prompt);
        let mood = mood_from_state_and_affect(state, &user_affect);
        let speech_act = speech_act_for_intent(&hints.intent, prompt);
        let intent_masses = soft_intent_masses(prompt, hints, fc, cfg.secondary_intent_mass);
        let ocean_blended = if cfg.enabled {
            blend_ocean(
                ocean_base,
                &mood,
                &hints.intent,
                speech_act,
                &user_affect,
                cfg.mood_ocean_strength,
                cfg.intent_ocean_strength,
            )
        } else {
            ocean_base
        };

        let mut anchors = hints.anchors.clone();
        for m in intent_masses.iter().skip(1) {
            if m.mass >= 0.12 && !anchors.iter().any(|a| a == &m.intent) {
                anchors.push(m.intent.clone());
            }
        }

        Self {
            speech_act,
            mood,
            intent_primary: hints.intent.clone(),
            intent_masses,
            ocean_base,
            ocean_blended,
            anchors,
            user_affect,
        }
    }

    pub fn summary(&self) -> String {
        format!(
            "act={} mood(v={:.2},a={:.2},s={:.2}) intent={} ocean_Δ=[{:.2},{:.2},{:.2},{:.2},{:.2}]",
            self.speech_act.as_str(),
            self.mood.valence,
            self.mood.arousal,
            self.mood.social,
            self.intent_primary,
            self.ocean_blended[0] - self.ocean_base[0],
            self.ocean_blended[1] - self.ocean_base[1],
            self.ocean_blended[2] - self.ocean_base[2],
            self.ocean_blended[3] - self.ocean_base[3],
            self.ocean_blended[4] - self.ocean_base[4],
        )
    }

    pub fn to_persisted(&self) -> PersistedContextFrame {
        PersistedContextFrame {
            speech_act: self.speech_act,
            mood: self.mood,
            intent_primary: self.intent_primary.clone(),
            ocean_blended: self.ocean_blended,
        }
    }

    /// Soften this turn's mood/OCEAN toward the prior turn (discourse continuity).
    pub fn blend_with_prior(&mut self, prior: &PersistedContextFrame, strength: f32) {
        let s = strength.clamp(0.0, 0.85);
        if s <= 0.0 {
            return;
        }
        let keep = 1.0 - s;
        self.mood.valence = self.mood.valence * keep + prior.mood.valence * s;
        self.mood.arousal = self.mood.arousal * keep + prior.mood.arousal * s;
        self.mood.social = self.mood.social * keep + prior.mood.social * s;
        for i in 0..5 {
            self.ocean_blended[i] =
                (self.ocean_blended[i] * keep + prior.ocean_blended[i] * s).clamp(0.05, 0.95);
        }
    }
}

fn mood_from_state_and_affect(state: &HashMap<String, f32>, affect: &UserAffect) -> MoodGradient {
    let mood = state.get("mood").copied().unwrap_or(0.65);
    let energy = state.get("energy").copied().unwrap_or(0.55);
    let hunger = state.get("hunger").copied().unwrap_or(0.3);
    // Map mood∈[0,1] → valence∈[−1,1]; distress pulls down, excitement pulls up slightly.
    let mut valence = (mood - 0.5) * 2.0;
    valence -= affect.distress * 0.55;
    valence += affect.excitement * 0.2;
    valence = valence.clamp(-1.0, 1.0);

    let mut arousal = energy * 0.7 + affect.excitement * 0.4 + hunger * 0.15;
    arousal = arousal.clamp(0.0, 1.0);

    // Social: content mood + low distress → engaged; distress alone → seeking comfort.
    let social =
        ((mood * 0.6) + (1.0 - affect.distress) * 0.25 + affect.excitement * 0.15).clamp(0.0, 1.0);

    MoodGradient {
        valence,
        arousal,
        social,
    }
}

fn speech_act_for_intent(intent: &str, prompt: &str) -> SpeechAct {
    let p = prompt.to_ascii_lowercase();
    match intent {
        "greeting_check_in" | "reunion_warm" | "reunion" => SpeechAct::Greet,
        "emotional_support" | "grounding_support" | "school_stress" | "comfort_seeking" => {
            SpeechAct::ComfortSeek
        }
        "status_check" | "lore_qa" | "food_preference" => {
            if p.contains('?') || p.starts_with("what") || p.starts_with("how") {
                SpeechAct::Query
            } else {
                SpeechAct::Inform
            }
        }
        "mealtime_request" | "bonding_request" | "play" | "play_invitation" => SpeechAct::Request,
        "feeding_ack" | "gratitude_simple" | "gratitude_comfort" => SpeechAct::Offer,
        "trigger_warning" | "training_command" | "hiding_behavior" => SpeechAct::Refuse,
        "storytelling" | "story_continue" | "mischief" => SpeechAct::Express,
        "identity_intro" => SpeechAct::Assert,
        "household_activity" | "owner_absence" => {
            if p.contains('?') {
                SpeechAct::Query
            } else {
                SpeechAct::Inform
            }
        }
        _ => {
            if p.contains('?') {
                SpeechAct::Query
            } else if affect_like_express(&p) {
                SpeechAct::Express
            } else {
                SpeechAct::Other
            }
        }
    }
}

fn affect_like_express(p: &str) -> bool {
    p.contains("i feel")
        || p.contains("i'm sad")
        || p.contains("im sad")
        || p.contains("rough day")
        || p.contains("miss you")
}

/// Soft intent distribution: primary keeps most mass; other matching rules share the rest.
fn soft_intent_masses(
    prompt: &str,
    primary: &FragmentIntentHint,
    fc: &FragmentComposeConfig,
    secondary_budget: f32,
) -> Vec<IntentMass> {
    let lower = prompt.to_ascii_lowercase();
    let secondary_budget = secondary_budget.clamp(0.0, 0.8);
    let mut scored: Vec<(String, f32)> = Vec::new();

    for rule in &fc.intent_rules {
        if rule.r#match == "fallback" || rule.intent == primary.intent {
            continue;
        }
        let hit = soft_rule_score(&lower, prompt, rule);
        if hit > 0.0 {
            scored.push((rule.intent.clone(), hit));
        }
    }
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.dedup_by(|a, b| a.0 == b.0);

    let top: Vec<(String, f32)> = scored.into_iter().take(3).collect();
    let hit_sum: f32 = top.iter().map(|(_, h)| h).sum::<f32>().max(1e-6);

    let mut out = vec![IntentMass {
        intent: primary.intent.clone(),
        mass: 1.0 - secondary_budget,
    }];
    for (intent, hit) in top {
        out.push(IntentMass {
            intent,
            mass: secondary_budget * (hit / hit_sum),
        });
    }
    // Renormalize
    let total: f32 = out.iter().map(|m| m.mass).sum::<f32>().max(1e-6);
    for m in &mut out {
        m.mass /= total;
    }
    out
}

fn soft_rule_score(
    lower: &str,
    _original: &str,
    rule: &crate::inference::inference_toml::FragmentIntentRuleToml,
) -> f32 {
    match rule.r#match.as_str() {
        "contains_any" | "starts_with_any" => {
            let hits = rule
                .patterns
                .iter()
                .filter(|p| {
                    let pat = p.to_ascii_lowercase();
                    !pat.is_empty() && lower.contains(&pat)
                })
                .count();
            if hits == 0 {
                0.0
            } else {
                (hits as f32).min(3.0)
            }
        }
        "greeting" | "agent_name_greeting" => {
            // Handled by hard match usually; light secondary only if short.
            if lower.len() < 24 {
                0.5
            } else {
                0.0
            }
        }
        _ => 0.0,
    }
}

/// Shift base OCEAN by mood field, speech act, and primary intent.
fn blend_ocean(
    base: [f32; 5],
    mood: &MoodGradient,
    intent: &str,
    act: SpeechAct,
    affect: &UserAffect,
    mood_strength: f32,
    intent_strength: f32,
) -> [f32; 5] {
    // Indices: O C E A N
    let mut d = [0.0f32; 5];

    // Mood → traits
    d[4] += (-mood.valence) * 0.45; // low valence → neuroticism up
    d[3] += mood.valence * 0.25 + affect.distress * 0.35; // distress seeks agreeableness (comfort)
    d[2] += mood.arousal * 0.35 - affect.distress * 0.2; // arousal → extraversion
    d[0] += mood.arousal * 0.15; // curiosity with activation
    d[1] += mood.valence * 0.1; // content → slightly more orderly

    for i in 0..5 {
        d[i] *= mood_strength;
    }

    // Intent / speech-act nudges
    let mut id = [0.0f32; 5];
    match act {
        SpeechAct::ComfortSeek | SpeechAct::Express => {
            id[3] += 0.35; // A
            id[4] += 0.15; // N (attune)
            id[2] -= 0.1; // quieter E
        }
        SpeechAct::Greet => {
            id[2] += 0.3;
            id[3] += 0.2;
        }
        SpeechAct::Query | SpeechAct::Inform => {
            id[1] += 0.25; // C — report-like
            id[0] += 0.1;
        }
        SpeechAct::Request => {
            id[2] += 0.2;
            id[0] += 0.15;
        }
        SpeechAct::Refuse => {
            id[1] += 0.2;
            id[4] += 0.15;
        }
        SpeechAct::Offer => {
            id[3] += 0.25;
            id[2] += 0.1;
        }
        SpeechAct::Assert => {
            id[1] += 0.2;
            id[2] += 0.1;
        }
        SpeechAct::Other => {}
    }
    match intent {
        "status_check" => {
            id[1] += 0.2;
            id[0] += 0.05;
        }
        "play" | "play_invitation" => {
            id[2] += 0.35;
            id[0] += 0.25;
            id[4] -= 0.1;
        }
        "bedtime_routine" => {
            id[2] -= 0.2;
            id[4] -= 0.1;
            id[3] += 0.15;
        }
        "emotional_support" | "grounding_support" => {
            id[3] += 0.3;
        }
        _ => {}
    }
    for i in 0..5 {
        d[i] += id[i] * intent_strength;
    }

    let mut out = base;
    for i in 0..5 {
        out[i] = (out[i] + d[i]).clamp(0.05, 0.95);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::inference_toml::{
        FragmentComposeConfig, FragmentIntentHint, FragmentIntentRuleToml,
    };

    fn hints(intent: &str) -> FragmentIntentHint {
        FragmentIntentHint {
            intent: intent.into(),
            anchors: vec![intent.into()],
            min_voices: 1,
            relaxed_parts: true,
        }
    }

    #[test]
    fn distress_raises_neuroticism_and_agreeableness() {
        let base = [0.7, 0.5, 0.8, 0.7, 0.3];
        let mut state = HashMap::new();
        state.insert("mood".into(), 0.35);
        state.insert("energy".into(), 0.4);
        let fc = FragmentComposeConfig::default();
        let cfg = ContextFrameConfig::default();
        let frame = ContextFrame::build(
            "I feel sad and overwhelmed",
            &hints("emotional_support"),
            &state,
            base,
            &fc,
            &cfg,
        );
        assert_eq!(frame.speech_act, SpeechAct::ComfortSeek);
        assert!(frame.mood.valence < 0.0);
        assert!(
            frame.ocean_blended[4] > base[4],
            "N should rise under distress"
        );
        assert!(
            frame.ocean_blended[3] >= base[3],
            "A should not drop under comfort-seek"
        );
    }

    #[test]
    fn status_check_query_is_query_act_and_raises_c() {
        let base = [0.7, 0.5, 0.8, 0.7, 0.3];
        let state = HashMap::from([("mood".into(), 0.72), ("energy".into(), 0.55)]);
        let fc = FragmentComposeConfig::default();
        let cfg = ContextFrameConfig::default();
        let frame = ContextFrame::build(
            "What's your current mood and what caused it?",
            &hints("status_check"),
            &state,
            base,
            &fc,
            &cfg,
        );
        assert_eq!(frame.speech_act, SpeechAct::Query);
        assert!(
            frame.ocean_blended[1] > base[1],
            "C should rise for status report"
        );
        assert!(frame.mood.valence > 0.0);
    }

    #[test]
    fn soft_intent_adds_secondary_mass() {
        let fc = FragmentComposeConfig {
            intent_rules: vec![
                FragmentIntentRuleToml {
                    id: "mood".into(),
                    intent: "status_check".into(),
                    anchors: vec!["status_check".into()],
                    min_voices: 1,
                    relaxed_parts: true,
                    r#match: "contains_any".into(),
                    patterns: vec!["current mood".into()],
                    max_len: None,
                },
                FragmentIntentRuleToml {
                    id: "sad".into(),
                    intent: "emotional_support".into(),
                    anchors: vec![],
                    min_voices: 1,
                    relaxed_parts: true,
                    r#match: "contains_any".into(),
                    patterns: vec!["feel sad".into()],
                    max_len: None,
                },
                FragmentIntentRuleToml {
                    id: "fb".into(),
                    intent: "open_ended_chat".into(),
                    anchors: vec![],
                    min_voices: 1,
                    relaxed_parts: false,
                    r#match: "fallback".into(),
                    patterns: vec![],
                    max_len: None,
                },
            ],
            ..Default::default()
        };
        let cfg = ContextFrameConfig::default();
        let frame = ContextFrame::build(
            "I feel sad — what's your current mood?",
            &hints("status_check"),
            &HashMap::from([("mood".into(), 0.5)]),
            [0.5; 5],
            &fc,
            &cfg,
        );
        assert!(frame.intent_masses.len() >= 2);
        assert!(frame
            .intent_masses
            .iter()
            .any(|m| m.intent == "emotional_support" && m.mass > 0.05));
        assert!(frame.anchors.iter().any(|a| a == "emotional_support"));
    }

    #[test]
    fn prior_frame_blends_mood_toward_distress() {
        let mut frame = ContextFrame::build(
            "hi",
            &hints("greeting_check_in"),
            &HashMap::from([("mood".into(), 0.8), ("energy".into(), 0.6)]),
            [0.5; 5],
            &FragmentComposeConfig::default(),
            &ContextFrameConfig::default(),
        );
        let prior = PersistedContextFrame {
            speech_act: SpeechAct::ComfortSeek,
            mood: MoodGradient {
                valence: -0.7,
                arousal: 0.5,
                social: 0.4,
            },
            intent_primary: "emotional_support".into(),
            ocean_blended: [0.5, 0.5, 0.4, 0.8, 0.7],
        };
        let before_v = frame.mood.valence;
        frame.blend_with_prior(&prior, 0.4);
        assert!(frame.mood.valence < before_v);
        assert!(frame.ocean_blended[4] > 0.5);
    }
}
