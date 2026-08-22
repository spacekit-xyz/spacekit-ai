//! In-memory [`EnvironmentPort`] for tests and offline episode replay.

use super::blanket::{Action, EnvironmentError, EnvironmentPort, Observation};
use super::integration::observation_from_reflection_outcome;
use crate::metacognition::ReflectionOutcome;
use std::collections::VecDeque;

#[derive(Clone, Debug, Default)]
pub struct QueuedEnvironment {
    queue: VecDeque<Observation>,
    pub applied: Vec<Action>,
}

impl QueuedEnvironment {
    pub fn from_observations<I: IntoIterator<Item = Observation>>(it: I) -> Self {
        Self {
            queue: it.into_iter().collect(),
            applied: Vec::new(),
        }
    }

    pub fn push(&mut self, obs: Observation) {
        self.queue.push_back(obs);
    }

    /// Append a MetaCognition outcome for offline replay (same mapping as live [`LanguageService`](crate::service::LanguageService) logging).
    pub fn push_reflection_outcome(&mut self, outcome: &ReflectionOutcome) {
        self.push(observation_from_reflection_outcome(outcome));
    }
}

impl EnvironmentPort for QueuedEnvironment {
    fn next_observation(&mut self) -> Option<Observation> {
        self.queue.pop_front()
    }

    fn apply(&mut self, action: &Action) -> Result<(), EnvironmentError> {
        self.applied.push(action.clone());
        Ok(())
    }
}
