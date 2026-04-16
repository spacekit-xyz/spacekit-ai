//! Map existing inference subsystems onto belief updates and blanket observations.

use super::blanket::{Observation, ReflectionTerminal};
use super::state::BeliefState;
use crate::metacognition::ReflectionOutcome;

/// Update internal belief after a MetaCognition reflect pass (Phase 2 hook).
pub fn belief_update_from_reflection(belief: &mut BeliefState, outcome: &ReflectionOutcome) {
    let q = reflection_quality(outcome);
    belief.last_quality = Some(q);
    match outcome {
        ReflectionOutcome::Retry { attempt, .. } => {
            belief.reflection_retries = belief.reflection_retries.max(*attempt + 1);
        }
        ReflectionOutcome::Accept { .. } | ReflectionOutcome::Degrade { .. } => {}
    }
}

fn reflection_quality(outcome: &ReflectionOutcome) -> f32 {
    match outcome {
        ReflectionOutcome::Accept { scores }
        | ReflectionOutcome::Retry { scores, .. }
        | ReflectionOutcome::Degrade { scores, .. } => scores.quality,
    }
}

/// Turn a reflection outcome into a spine [`Observation`] for logging or replay.
pub fn observation_from_reflection_outcome(outcome: &ReflectionOutcome) -> Observation {
    let quality = reflection_quality(outcome);
    let terminal = match outcome {
        ReflectionOutcome::Accept { .. } => ReflectionTerminal::Accepted,
        ReflectionOutcome::Retry { attempt, .. } => ReflectionTerminal::Retry {
            attempt: *attempt,
        },
        ReflectionOutcome::Degrade { .. } => ReflectionTerminal::Degraded,
    };
    Observation::ReflectionCycle { quality, terminal }
}
