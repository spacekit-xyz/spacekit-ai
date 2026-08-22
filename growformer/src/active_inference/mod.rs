//! Active Inference **spine** (Phases 0–3): explicit belief, Markov-blanket I/O, episode loop.
//!
//! ## Markov blanket (information boundary)
//! - **Inward**: only [`blanket::Observation`] updates the spine (user text, reflection summaries).
//! - **Outward**: only [`blanket::Action`] mutates the world through [`blanket::EnvironmentPort`].
//! - **Internal**: [`BeliefState`] plus your lattice / rules / MetaCognition engines stay inside;
//!   do not feed raw HTTP or file handles into belief; parse them into observations first.
//!
//! ## Phases delivered here
//! - **0**: Types and module boundary.
//! - **1**: [`ActiveInferenceSpine::run_episode`] (`observe → policy → act`).
//! - **2**: [`integration`] helpers for MetaCognition → belief / observation.
//! - **3**: [`blanket::EnvironmentPort`] + [`harness::QueuedEnvironment`] for tests.
//!
//! **Phase 4** (heavy external LLM proposal adapters) is **not** implemented by design.
//!
//! For a production-style policy over [`LanguageService`](crate::service::LanguageService), see
//! [`crate::service::RoutingGenerationMetacogEpisodePolicy`].

mod blanket;
mod harness;
mod integration;
mod spine;
mod state;

pub use blanket::{Action, EnvironmentError, EnvironmentPort, Observation, ReflectionTerminal};
pub use harness::QueuedEnvironment;
pub use integration::{belief_update_from_reflection, observation_from_reflection_outcome};
pub use spine::{
    ActiveInferenceSpine, EchoPolicy, EpisodePolicy, PolicyTurn, SpineConfig, SpineStepRecord,
};
pub use state::BeliefState;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metacognition::ReflectionScores;

    #[test]
    fn episode_echo_user_then_stops() {
        let mut env = QueuedEnvironment::from_observations([Observation::UserText("hello".into())]);
        let mut belief = BeliefState::new();
        let mut policy = EchoPolicy::new("> ");
        let spine = ActiveInferenceSpine::new(SpineConfig {
            max_steps: 8,
            ..Default::default()
        });
        let trace = spine
            .run_episode(&mut env, &mut belief, &mut policy)
            .expect("env ok");
        assert_eq!(trace.len(), 1);
        assert_eq!(belief.step, 1);
        assert_eq!(
            env.applied,
            vec![Action::Emit {
                text: "> hello".into()
            }]
        );
    }

    #[test]
    #[test]
    fn push_reflection_outcome_roundtrip() {
        let mut q = QueuedEnvironment::default();
        let outcome = crate::metacognition::ReflectionOutcome::Accept {
            scores: ReflectionScores {
                coherence: 0.5,
                relevance: 0.4,
                completeness: 0.3,
                quality: 0.44,
            },
        };
        q.push_reflection_outcome(&outcome);
        let obs = q.next_observation().expect("one obs");
        assert!(matches!(
            obs,
            Observation::ReflectionCycle {
                quality: q,
                terminal: ReflectionTerminal::Accepted
            } if (q - 0.44).abs() < 1e-5
        ));
    }

    fn reflection_outcome_maps_to_observation() {
        let outcome = crate::metacognition::ReflectionOutcome::Accept {
            scores: ReflectionScores {
                coherence: 0.5,
                relevance: 0.4,
                completeness: 0.3,
                quality: 0.44,
            },
        };
        let mut b = BeliefState::new();
        belief_update_from_reflection(&mut b, &outcome);
        assert!((b.last_quality.unwrap() - 0.44).abs() < 1e-5);

        let obs = observation_from_reflection_outcome(&outcome);
        assert!(matches!(
            obs,
            Observation::ReflectionCycle {
                quality: q,
                terminal: ReflectionTerminal::Accepted
            } if (q - 0.44).abs() < 1e-5
        ));
    }
}
