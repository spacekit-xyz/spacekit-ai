//! Homeostatic drive → neuromodulator → response-field modulation.
//!
//! OCEAN traits describe a *static* identity. Real behavior is state-dependent:
//! two agents with identical traits act differently depending on their drives
//! (hunger, energy, social need) and the neuromodulators those drives release.
//!
//! This module models that as a small, fully-deterministic pipeline:
//!
//! ```text
//!   drives (hunger/energy/social)
//!        │  map_neuromodulators()
//!        ▼
//!   neuromodulators (dopamine/serotonin/norepinephrine/acetylcholine)
//!        │  field_modulation()
//!        ▼
//!   gains on EXISTING generation knobs
//!     (retrieval temperature, novelty/refractory, context decay,
//!      agent-state blend, salience sensitivity)
//! ```
//!
//! Crucially it adds no parallel machinery, it re-weights knobs that already
//! shape retrieval. A homeostatic loop ([`DriveState::tick`] / [`DriveState::satisfy`])
//! makes the state causal: deficits grow over time and inputs ("food", "petting",
//! "nap") relax them, so the same prompt yields different behavior as state evolves.

/// Survival / social drives, each normalized to `[0, 1]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DriveState {
    /// 0.0 = fully sated, 1.0 = starving.
    pub hunger: f32,
    /// 0.0 = exhausted, 1.0 = energetic.
    pub energy: f32,
    /// 0.0 = lonely (needs interaction), 1.0 = socially satisfied.
    pub social: f32,
}

impl Default for DriveState {
    fn default() -> Self {
        // Mild baseline: a little hungry, decently rested, moderately social.
        Self { hunger: 0.3, energy: 0.6, social: 0.6 }
    }
}

/// Neuromodulator levels in `[0, 1]`. These are *derived* from drives, never set
/// directly, they are the chemistry that turns a drive deficit into a behavior.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Neuromodulators {
    /// Seeking / exploration / reward pursuit. Rises with unmet drives.
    pub dopamine: f32,
    /// Patience / contentment / stability. Rises when drives are satisfied.
    pub serotonin: f32,
    /// Alertness / urgency. Rises with acute deficits and available energy.
    pub norepinephrine: f32,
    /// Attention precision. Rises with energy and alertness.
    pub acetylcholine: f32,
}

/// Multiplicative / absolute adjustments applied to existing generation knobs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FieldModulation {
    /// Multiplies the base retrieval temperature (exploration vs. focus).
    pub temperature_scale: f32,
    /// Multiplies the lattice novelty factor (refractory / anti-monoculture pressure).
    pub novelty_scale: f32,
    /// Absolute conversation context decay (memory persistence / patience).
    pub context_decay: f32,
    /// Multiplies the agent-state conditioning blend (how strongly state biases retrieval).
    pub state_blend_scale: f32,
    /// Multiplies field-gradient sensitivity (alertness to "something changed").
    pub salience_gain: f32,
}

impl Default for FieldModulation {
    fn default() -> Self {
        Self {
            temperature_scale: 1.0,
            novelty_scale: 1.0,
            context_decay: 0.65,
            state_blend_scale: 1.0,
            salience_gain: 1.0,
        }
    }
}

#[inline]
fn clamp01(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

#[inline]
fn centered(x: f32) -> f32 {
    // Map [0,1] level to a [-0.5, 0.5] deviation around neutral.
    x - 0.5
}

impl DriveState {
    /// Construct from named agent-state dimensions (as carried in `AgentRuntimeState`).
    /// `social` is derived from idle time when not explicitly provided.
    pub fn from_dimensions(
        hunger: Option<f32>,
        energy: Option<f32>,
        social: Option<f32>,
        minutes_idle: f32,
    ) -> Self {
        let d = Self::default();
        let social = social.unwrap_or_else(|| {
            // Longer idle → lonelier. 0 min → 1.0, 120+ min → ~0.0.
            (1.0 - (minutes_idle / 120.0)).clamp(0.0, 1.0)
        });
        Self {
            hunger: clamp01(hunger.unwrap_or(d.hunger)),
            energy: clamp01(energy.unwrap_or(d.energy)),
            social: clamp01(social),
        }
    }

    /// Derive neuromodulator levels from the current drives. Pure and deterministic.
    pub fn map_neuromodulators(&self) -> Neuromodulators {
        let hunger = self.hunger;
        let energy = self.energy;
        let lonely = 1.0 - self.social;

        // Dopamine: seeking pressure — driven by unmet hunger and social deficit.
        let dopamine = clamp01(0.25 + 0.50 * hunger + 0.35 * lonely);
        // Serotonin: contentment — high when fed and socially satisfied.
        let serotonin = clamp01(0.30 + 0.40 * (1.0 - hunger) + 0.30 * self.social);
        // Norepinephrine: urgency / alertness — acute deficit plus available energy
        // (an exhausted agent cannot be "alert" even if hungry).
        let norepinephrine = clamp01(0.15 + 0.45 * hunger + 0.40 * energy);
        // Acetylcholine: attention precision — needs energy and some alertness.
        let acetylcholine = clamp01(0.25 + 0.50 * energy + 0.20 * norepinephrine);

        Neuromodulators { dopamine, serotonin, norepinephrine, acetylcholine }
    }

    /// Full pipeline: drives → neuromodulators → knob modulation.
    pub fn field_modulation(&self) -> FieldModulation {
        self.map_neuromodulators().field_modulation()
    }

    /// Advance drives by one conversation turn. Deficits grow slowly; idle time
    /// deepens loneliness. Energy ebbs gently (recovered via rest in [`satisfy`]).
    pub fn tick(&mut self, minutes_idle: f32) {
        self.hunger = clamp01(self.hunger + 0.04);
        self.energy = clamp01(self.energy - 0.015);
        let idle_loneliness = (minutes_idle / 240.0).clamp(0.0, 0.4);
        self.social = clamp01(self.social - 0.03 - idle_loneliness * 0.1);
    }

    /// Relax drives in response to a satisfying input. Keyword-driven so it works
    /// directly off the user's message (e.g. "here's a treat", "good girl", "nap time").
    /// Returns the set of drives that were affected (for tracing / tests).
    pub fn satisfy(&mut self, input: &str) -> &'static [&'static str] {
        let l = input.to_ascii_lowercase();
        let mut tags: Vec<&'static str> = Vec::new();

        let feeds = ["food", "treat", "feed", "dinner", "breakfast", "eat", "snack", "fish", "tuna"];
        if feeds.iter().any(|k| l.contains(k)) {
            self.hunger = clamp01(self.hunger - 0.6);
            tags.push("hunger");
        }

        let social = ["pet", "scratch", "cuddle", "love", "good girl", "good kitty",
            "hello", "hey", "hi ", "snuggle", "hold", "lap"];
        if social.iter().any(|k| l.contains(k)) {
            self.social = clamp01(self.social + 0.5);
            tags.push("social");
        }

        let play = ["play", "toy", "wand", "feather", "chase", "fetch"];
        if play.iter().any(|k| l.contains(k)) {
            self.energy = clamp01(self.energy - 0.10);
            self.social = clamp01(self.social + 0.30);
            tags.push("play");
        }

        let rest = ["nap", "sleep", "rest", "bed", "bedtime", "goodnight", "night"];
        if rest.iter().any(|k| l.contains(k)) {
            self.energy = clamp01(self.energy + 0.50);
            tags.push("energy");
        }

        // Leak the tags via a static mapping (small, fixed set) for trace strings.
        match tags.as_slice() {
            [] => &[],
            ["hunger"] => &["hunger"],
            ["social"] => &["social"],
            ["play"] => &["play"],
            ["energy"] => &["energy"],
            _ => &["multi"],
        }
    }
}

impl Neuromodulators {
    /// Map neuromodulator levels onto multiplicative/absolute knob gains.
    /// All outputs are clamped to safe ranges so a runaway state cannot break retrieval.
    pub fn field_modulation(&self) -> FieldModulation {
        let da = centered(self.dopamine);
        let ne = centered(self.norepinephrine);
        let ser = centered(self.serotonin);

        // Dopamine raises exploration; norepinephrine sharpens (focuses) it.
        let temperature_scale = (1.0 + 0.5 * da - 0.4 * ne).clamp(0.5, 1.8);
        // Dopamine drives novelty-seeking (stronger anti-repeat / diversity).
        let novelty_scale = (1.0 + 0.6 * da).clamp(0.6, 1.6);
        // Serotonin = patience = longer memory persistence.
        let context_decay = (0.65 + 0.25 * ser).clamp(0.40, 0.90);
        // Norepinephrine urgency pushes harder toward state-appropriate programs.
        let state_blend_scale = (1.0 + 0.8 * ne).clamp(0.5, 2.0);
        // Norepinephrine raises sensitivity to field change (salience).
        let salience_gain = (1.0 + 0.5 * ne).clamp(0.6, 1.6);

        FieldModulation {
            temperature_scale,
            novelty_scale,
            context_decay,
            state_blend_scale,
            salience_gain,
        }
    }

    /// One-line human-readable summary for trace logs.
    pub fn summary(&self) -> String {
        format!(
            "DA={:.2} 5HT={:.2} NE={:.2} ACh={:.2}",
            self.dopamine, self.serotonin, self.norepinephrine, self.acetylcholine
        )
    }
}

/// Live drive field held by the language service. Flag-gated; when disabled the
/// service uses unmodulated defaults so it is a clean A/B against current behavior.
#[derive(Debug, Clone)]
pub struct DriveField {
    pub state: DriveState,
    pub enabled: bool,
    /// Base retrieval temperature (before modulation). Captured once at init so
    /// per-turn scaling multiplies a stable base instead of compounding.
    pub base_temperature: f32,
    /// Base lattice novelty factor (before modulation).
    pub base_novelty: f32,
}

impl DriveField {
    pub fn new(state: DriveState, enabled: bool) -> Self {
        Self { state, enabled, base_temperature: 0.85, base_novelty: 1.0 }
    }

    pub fn with_bases(mut self, base_temperature: f32, base_novelty: f32) -> Self {
        self.base_temperature = base_temperature;
        self.base_novelty = base_novelty;
        self
    }

    /// Current modulation, or neutral defaults when disabled.
    pub fn modulation(&self) -> FieldModulation {
        if self.enabled {
            self.state.field_modulation()
        } else {
            FieldModulation::default()
        }
    }

    /// Modulated retrieval temperature for this turn.
    pub fn temperature(&self) -> f32 {
        (self.base_temperature * self.modulation().temperature_scale).max(0.01)
    }

    /// Modulated novelty factor for this turn.
    pub fn novelty(&self) -> f32 {
        (self.base_novelty * self.modulation().novelty_scale).max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hungry_lonely_is_high_dopamine_seeking() {
        let hungry = DriveState { hunger: 0.95, energy: 0.7, social: 0.1 };
        let nm = hungry.map_neuromodulators();
        assert!(nm.dopamine > 0.8, "hungry+lonely should spike dopamine: {}", nm.dopamine);
        assert!(nm.norepinephrine > 0.6, "hunger+energy should raise alertness: {}", nm.norepinephrine);
        assert!(nm.serotonin < 0.5, "unmet drives keep serotonin low: {}", nm.serotonin);
    }

    #[test]
    fn sated_content_is_high_serotonin_calm() {
        let sated = DriveState { hunger: 0.05, energy: 0.4, social: 0.95 };
        let nm = sated.map_neuromodulators();
        assert!(nm.serotonin > 0.8, "fed+social should raise serotonin: {}", nm.serotonin);
        assert!(nm.dopamine < 0.5, "satisfied drives lower seeking: {}", nm.dopamine);
    }

    #[test]
    fn hungry_explores_more_than_sated() {
        let hungry = DriveState { hunger: 0.95, energy: 0.8, social: 0.2 }.field_modulation();
        let sated = DriveState { hunger: 0.05, energy: 0.4, social: 0.95 }.field_modulation();
        // Hungry pushes harder on state-appropriate (food) programs.
        assert!(hungry.state_blend_scale > sated.state_blend_scale);
        // Sated is more patient (slower context decay → longer memory).
        assert!(sated.context_decay > hungry.context_decay);
    }

    #[test]
    fn feeding_relaxes_hunger() {
        let mut s = DriveState { hunger: 0.9, energy: 0.5, social: 0.5 };
        s.satisfy("here is a treat for you");
        assert!(s.hunger < 0.4, "feeding should drop hunger: {}", s.hunger);
    }

    #[test]
    fn petting_satisfies_social() {
        let mut s = DriveState { hunger: 0.5, energy: 0.5, social: 0.1 };
        s.satisfy("good girl, come scratch your chin");
        assert!(s.social > 0.5, "petting should raise social: {}", s.social);
    }

    #[test]
    fn rest_restores_energy() {
        let mut s = DriveState { hunger: 0.3, energy: 0.1, social: 0.6 };
        s.satisfy("time for a nap");
        assert!(s.energy > 0.5, "rest should restore energy: {}", s.energy);
    }

    #[test]
    fn tick_grows_deficits() {
        let mut s = DriveState { hunger: 0.3, energy: 0.6, social: 0.8 };
        s.tick(60.0);
        assert!(s.hunger > 0.3, "hunger grows over a turn");
        assert!(s.social < 0.8, "social need grows over a turn");
    }

    #[test]
    fn modulation_clamped_in_safe_range() {
        // Even an extreme state must keep knobs in bounds.
        for &(h, e, so) in &[(1.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)] {
            let m = DriveState { hunger: h, energy: e, social: so }.field_modulation();
            assert!(m.temperature_scale >= 0.5 && m.temperature_scale <= 1.8);
            assert!(m.novelty_scale >= 0.6 && m.novelty_scale <= 1.6);
            assert!(m.context_decay >= 0.40 && m.context_decay <= 0.90);
            assert!(m.state_blend_scale >= 0.5 && m.state_blend_scale <= 2.0);
            assert!(m.salience_gain >= 0.6 && m.salience_gain <= 1.6);
        }
    }

    #[test]
    fn disabled_field_is_neutral() {
        let f = DriveField::new(DriveState { hunger: 1.0, energy: 1.0, social: 0.0 }, false);
        assert_eq!(f.modulation(), FieldModulation::default());
    }
}
