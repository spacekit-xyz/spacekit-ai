//! ReflectiveField — unified present-state composition (Identity ⊕ Activity ⊕ Drive).
//!
//! Human self-reflection composes a response from three sources held in one
//! workspace:
//!
//! ```text
//!   Identity   — who I am          (stable priors / OCEAN)
//!   Activity   — what I've been doing (conversation momentum / trajectory)
//!   Drive/State— how I feel now     (neuromodulator-modulated needs)
//!        │
//!        ▼  compose()  (weights modulated by current neuromodulators)
//!   present-state bias on the generation-conditioning vector
//! ```
//!
//! Before this module, those three signals each biased retrieval through
//! *separate, independently-tuned constants* scattered across the generation
//! path (a 0.15/0.28 context blend, a 0.20 state blend, a 0.15 OCEAN scale).
//! Nothing coordinated them, so "Activity" was never a first-class peer of
//! identity and drive.
//!
//! [`ReflectiveField`] makes the composition a single coherent policy: one
//! object decides how strongly each source speaks *this* turn, with the balance
//! shifted by the current neuromodulators (the "listening to my own state"
//! step). Concretely:
//!   * serotonin (contentment)    → stronger, more stable **identity** expression
//!   * dopamine  (seeking)        → stronger pull from current **activity**/momentum
//!   * norepinephrine (urgency)   → **drive/state** dominates the present
//!
//! This is the substrate the retrocausal goal-attractor builds on: a
//! well-defined *present* state, so a desired *future* landing has a coherent
//! origin to be pulled toward.

use crate::drive_field::Neuromodulators;

#[inline]
fn norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

#[inline]
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let dot: f32 = (0..n).map(|i| a[i] * b[i]).sum();
    let na = norm(&a[..n]);
    let nb = norm(&b[..n]);
    if na < 1e-6 || nb < 1e-6 {
        0.0
    } else {
        dot / (na * nb)
    }
}

/// A retrocausal goal-attractor: a desired *future* "emotional landing" expressed
/// as a target direction in conditioning space, which biases the *present*
/// response toward trajectories that reach it.
///
/// "Retrocausal" is teleological, not literal: the future goal acts as an
/// attractor on the present (like a goal pulling behavior toward it), implemented
/// as a small geodesic step of the present-state vector toward the target each
/// turn. Applied repeatedly across a conversation, the agent gradually *lands* on
/// the intended affect (e.g. steering a distressed exchange toward comfort) rather
/// than reacting turn-by-turn with no destination.
#[derive(Debug, Clone)]
pub struct GoalAttractor {
    /// Target direction in conditioning space (e.g. the encoding of a goal phrase
    /// like "you are safe, I am here"). Need not be normalized.
    pub target: Vec<f32>,
    /// Per-turn pull strength in `[0, 1]`. Small values give a gradual landing;
    /// 0 disables, 1 snaps to the target in one step.
    pub pull: f32,
    /// Human-readable label for tracing (e.g. "comfort", "play").
    pub label: String,
}

impl GoalAttractor {
    pub fn new(target: Vec<f32>, pull: f32, label: impl Into<String>) -> Self {
        Self { target, pull: pull.clamp(0.0, 1.0), label: label.into() }
    }

    /// Take one geodesic-style step of `cond` toward the target, preserving the
    /// vector's magnitude (a rotation-like move, not a rescale). Returns the cosine
    /// alignment to the target *after* the step — i.e. progress toward the landing.
    pub fn apply(&self, cond: &mut [f32]) -> f32 {
        let n = cond.len().min(self.target.len());
        if n == 0 || self.pull <= 0.0 {
            return cosine(cond, &self.target);
        }
        let orig_norm = norm(&cond[..n]);
        for i in 0..n {
            cond[i] = (1.0 - self.pull) * cond[i] + self.pull * self.target[i];
        }
        // Renormalize the affected prefix back to the original magnitude so the
        // pull rotates the conditioning toward the goal without inflating it.
        let new_norm = norm(&cond[..n]);
        if new_norm > 1e-6 && orig_norm > 1e-6 {
            let s = orig_norm / new_norm;
            for x in cond[..n].iter_mut() {
                *x *= s;
            }
        }
        cosine(&cond[..n], &self.target[..n])
    }
}

/// How strongly each reflective source biases retrieval this turn. All in
/// blend-weight units (roughly `[0, 0.6]`); higher = that voice speaks louder.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReflectiveWeights {
    /// Identity (OCEAN) directional bias on the trait dims.
    pub identity: f32,
    /// Activity (conversation momentum) blend across the conditioning vector.
    pub activity: f32,
    /// Drive/state quantized bias on the state tail dims.
    pub drive: f32,
}

impl Default for ReflectiveWeights {
    fn default() -> Self {
        // Mirror the previously-scattered constants as the neutral baseline so an
        // unmodulated field reproduces the tuned behavior.
        Self { identity: 0.10, activity: 0.15, drive: 0.20 }
    }
}

/// The reflective composition policy. Flag-gated; when disabled the service
/// keeps the original scattered blends so this is a clean A/B.
#[derive(Debug, Clone)]
pub struct ReflectiveField {
    pub enabled: bool,
    pub base: ReflectiveWeights,
    /// Optional retrocausal goal-attractor: a desired future landing that pulls the
    /// composed present state toward it. `None` ⇒ purely reactive (present only).
    pub attractor: Option<GoalAttractor>,
}

impl ReflectiveField {
    pub fn new(enabled: bool) -> Self {
        Self { enabled, base: ReflectiveWeights::default(), attractor: None }
    }

    /// Set (or clear) the active goal-attractor for upcoming turns.
    pub fn set_attractor(&mut self, attractor: Option<GoalAttractor>) {
        self.attractor = attractor;
    }

    /// Resolve the per-turn weights, shifting the identity/activity/drive balance
    /// by the current neuromodulators. Pure and deterministic.
    pub fn weights(&self, nm: Option<Neuromodulators>, multi_turn: bool) -> ReflectiveWeights {
        let mut w = self.base;
        // Mid-conversation, recent momentum matters more (continuity).
        if multi_turn {
            w.activity = (w.activity + 0.13).min(0.6);
        }
        if let Some(nm) = nm {
            let da = nm.dopamine - 0.5;
            let ser = nm.serotonin - 0.5;
            let ne = nm.norepinephrine - 0.5;
            w.identity = (w.identity + 0.20 * ser).clamp(0.0, 0.5);
            w.activity = (w.activity + 0.25 * da).clamp(0.0, 0.6);
            w.drive = (w.drive + 0.40 * ne).clamp(0.0, 0.6);
        }
        w
    }

    /// Compose the unified present-state bias into `gen_conditioning` in a single
    /// coherent pass, returning the weights actually applied (for tracing/tests).
    ///
    /// * `ocean`       — identity trait vector `[O, C, E, A, N]`.
    /// * `activity`    — conversation momentum (e.g. `ConversationContext.context_embedding`).
    /// * `drive_vals`  — runtime state dimension values (HashMap iter order), tail-mapped.
    /// * `nm`          — current neuromodulators (None ⇒ unmodulated baseline).
    /// * `multi_turn`  — true mid-conversation (raises activity weight).
    /// * `turn`        — turn counter, drives a small variety signal.
    #[allow(clippy::too_many_arguments)]
    pub fn compose(
        &self,
        gen_conditioning: &mut [f32],
        ocean: [f32; 5],
        activity: &[f32],
        drive_vals: &[f32],
        nm: Option<Neuromodulators>,
        multi_turn: bool,
        turn: u32,
    ) -> ReflectiveWeights {
        let w = self.weights(nm, multi_turn);
        let dim = gen_conditioning.len();

        // --- Activity: blend conversation momentum across the whole vector ---
        if !activity.is_empty() && w.activity > 0.0 {
            let n = dim.min(activity.len());
            for i in 0..n {
                gen_conditioning[i] =
                    (1.0 - w.activity) * gen_conditioning[i] + w.activity * activity[i];
            }
        }

        // --- Drive/state: quantize runtime dims into the tail-16 slots ---
        if dim >= 16 && !drive_vals.is_empty() && w.drive > 0.0 {
            let slot_count = drive_vals.len().min(16);
            let base_offset = dim - 16;
            for (i, &val) in drive_vals.iter().enumerate().take(slot_count) {
                let idx = base_offset + i;
                let quantized = (val.clamp(0.0, 1.0) - 0.5) * 2.0;
                gen_conditioning[idx] =
                    (1.0 - w.drive) * gen_conditioning[idx] + w.drive * quantized;
            }
            // Small turn-indexed signal keeps repeated prompts from collapsing.
            if turn > 0 {
                let turn_signal = ((turn as f32) * 0.1).sin() * 0.15;
                gen_conditioning[base_offset] += turn_signal;
            }
        }

        // --- Identity: re-assert OCEAN directional bias on the trait tail ---
        if dim >= 10 && w.identity > 0.0 {
            for (i, &o) in ocean.iter().enumerate() {
                let idx = dim - 5 + i;
                gen_conditioning[idx] += (o - 0.5) * w.identity;
            }
        }

        // --- Retrocausal goal-attractor: pull the composed present toward the
        // desired future landing (teleological, applied as a geodesic step) ---
        if let Some(ref attractor) = self.attractor {
            let alignment = attractor.apply(gen_conditioning);
            crate::infer_trace!(
                "  [goal-attractor] pull→'{}' (pull={:.2}) alignment={:.3}",
                attractor.label,
                attractor.pull,
                alignment
            );
        }

        w
    }

    /// One-line human-readable summary for trace logs.
    pub fn summary(w: &ReflectiveWeights) -> String {
        format!(
            "identity={:.2} activity={:.2} drive={:.2}",
            w.identity, w.activity, w.drive
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::drive_field::DriveState;

    #[test]
    fn unmodulated_weights_match_baseline() {
        let f = ReflectiveField::new(true);
        let w = f.weights(None, false);
        assert_eq!(w, ReflectiveWeights::default());
    }

    #[test]
    fn multi_turn_raises_activity() {
        let f = ReflectiveField::new(true);
        let single = f.weights(None, false);
        let multi = f.weights(None, true);
        assert!(multi.activity > single.activity, "mid-conversation favors momentum");
    }

    #[test]
    fn contentment_strengthens_identity_urgency_strengthens_drive() {
        let f = ReflectiveField::new(true);
        // Sated/content: high serotonin, low norepinephrine.
        let sated = DriveState { hunger: 0.05, energy: 0.4, social: 0.95 }.map_neuromodulators();
        // Hungry/urgent: high norepinephrine, lower serotonin.
        let hungry = DriveState { hunger: 0.95, energy: 0.8, social: 0.2 }.map_neuromodulators();

        let ws = f.weights(Some(sated), false);
        let wh = f.weights(Some(hungry), false);

        assert!(ws.identity > wh.identity, "contentment → stronger identity");
        assert!(wh.drive > ws.drive, "urgency → drive dominates");
    }

    #[test]
    fn compose_shifts_conditioning() {
        let f = ReflectiveField::new(true);
        let mut a = vec![0.0f32; 32];
        let mut b = vec![0.0f32; 32];
        let ocean = [0.9, 0.5, 0.8, 0.7, 0.4];
        let activity = vec![1.0f32; 32];
        let drive_lonely = [0.9f32]; // hungry/needy
        let drive_sated = [0.1f32];
        let lonely_nm = DriveState { hunger: 0.9, energy: 0.7, social: 0.1 }.map_neuromodulators();
        let sated_nm = DriveState { hunger: 0.1, energy: 0.5, social: 0.9 }.map_neuromodulators();

        f.compose(&mut a, ocean, &activity, &drive_lonely, Some(lonely_nm), true, 3);
        f.compose(&mut b, ocean, &activity, &drive_sated, Some(sated_nm), true, 3);

        let diff: f32 = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).sum();
        assert!(diff > 0.01, "different drive states must produce different conditioning: {diff}");
    }

    #[test]
    fn attractor_pulls_toward_target() {
        let target = vec![1.0, 0.0, 0.0, 0.0];
        let att = GoalAttractor::new(target.clone(), 0.3, "x");
        let mut cond = vec![0.0, 1.0, 0.0, 0.0];
        let before = cosine(&cond, &target);
        let after = att.apply(&mut cond);
        assert!(after > before, "one step should raise alignment: {before} → {after}");
    }

    #[test]
    fn attractor_converges_over_turns() {
        let target = vec![0.0, 1.0, 0.0, 1.0];
        let att = GoalAttractor::new(target.clone(), 0.25, "land");
        let mut cond = vec![1.0, 0.0, 1.0, 0.0];
        let mut last = cosine(&cond, &target);
        for _ in 0..20 {
            last = att.apply(&mut cond);
        }
        assert!(last > 0.95, "iterated pull should land on target: {last}");
    }

    #[test]
    fn attractor_pull_zero_is_noop() {
        let att = GoalAttractor::new(vec![1.0, 1.0], 0.0, "off");
        let mut cond = vec![0.3, -0.7];
        let copy = cond.clone();
        att.apply(&mut cond);
        assert_eq!(cond, copy);
    }

    #[test]
    fn attractor_preserves_magnitude() {
        let att = GoalAttractor::new(vec![1.0, 2.0, -1.0, 0.5], 0.4, "m");
        let mut cond = vec![0.5, -0.5, 0.5, -0.5];
        let before = norm(&cond);
        att.apply(&mut cond);
        let after = norm(&cond);
        assert!((before - after).abs() < 1e-4, "magnitude preserved: {before} vs {after}");
    }

    #[test]
    fn weights_stay_bounded_under_extremes() {
        let f = ReflectiveField::new(true);
        for &(h, e, s) in &[(1.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)] {
            let nm = DriveState { hunger: h, energy: e, social: s }.map_neuromodulators();
            let w = f.weights(Some(nm), true);
            assert!(w.identity >= 0.0 && w.identity <= 0.5);
            assert!(w.activity >= 0.0 && w.activity <= 0.6);
            assert!(w.drive >= 0.0 && w.drive <= 0.6);
        }
    }
}
