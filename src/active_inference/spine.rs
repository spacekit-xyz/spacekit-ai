//! Episode loop: **observe → update belief → policy → act** until policy completes or cap hit.
//!
//! Phase 4 (external LLM proposal adapters) is intentionally out of scope.

use super::blanket::{Action, EnvironmentError, EnvironmentPort, Observation};
use super::state::BeliefState;

/// One step of the spine (auditable trace).
#[derive(Clone, Debug, PartialEq)]
pub struct SpineStepRecord {
    pub step_index: usize,
    pub observation_summary: String,
    pub actions_applied: Vec<String>,
}

/// Stops the episode when `true`.
pub trait EpisodePolicy {
    fn on_observation(
        &mut self,
        belief: &mut BeliefState,
        obs: &Observation,
    ) -> PolicyTurn;
}

/// What the spine should do after an observation.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PolicyTurn {
    /// Actions to apply through the environment port (in order).
    pub actions: Vec<Action>,
    /// If true, `run_episode` returns after applying actions (success).
    pub episode_complete: bool,
}

impl PolicyTurn {
    pub fn complete_after(actions: Vec<Action>) -> Self {
        Self {
            actions,
            episode_complete: true,
        }
    }

    pub fn continue_with(actions: Vec<Action>) -> Self {
        Self {
            actions,
            episode_complete: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SpineConfig {
    pub max_steps: usize,
}

impl Default for SpineConfig {
    fn default() -> Self {
        Self { max_steps: 64 }
    }
}

pub struct ActiveInferenceSpine {
    pub config: SpineConfig,
}

impl ActiveInferenceSpine {
    pub fn new(config: SpineConfig) -> Self {
        Self { config }
    }

    /// Run until policy sets `episode_complete`, environment returns no observation, or `max_steps`.
    pub fn run_episode<E: EnvironmentPort, P: EpisodePolicy>(
        &self,
        env: &mut E,
        belief: &mut BeliefState,
        policy: &mut P,
    ) -> Result<Vec<SpineStepRecord>, EnvironmentError> {
        let mut trace = Vec::new();
        let mut turns = 0usize;

        while turns < self.config.max_steps {
            let Some(obs) = env.next_observation() else {
                break;
            };

            belief.advance_step();
            let turn = policy.on_observation(belief, &obs);

            let mut applied_labels = Vec::new();
            for a in &turn.actions {
                env.apply(a)?;
                applied_labels.push(action_summary(a));
            }

            trace.push(SpineStepRecord {
                step_index: belief.step,
                observation_summary: observation_summary(&obs),
                actions_applied: applied_labels,
            });

            turns += 1;
            if turn.episode_complete {
                break;
            }
        }

        Ok(trace)
    }
}

fn observation_summary(obs: &Observation) -> String {
    match obs {
        Observation::UserText(s) => {
            let t = s.trim();
            if t.len() > 48 {
                format!("UserText({}…)", &t[..48])
            } else {
                format!("UserText({})", t)
            }
        }
        Observation::ReflectionCycle { quality, terminal } => format!(
            "Reflection(quality={:.3}, {:?})",
            quality, terminal
        ),
    }
}

fn action_summary(a: &Action) -> String {
    match a {
        Action::Emit { text } => {
            let t = text.trim();
            if t.len() > 40 {
                format!("Emit({}…)", &t[..40])
            } else {
                format!("Emit({})", t)
            }
        }
        Action::RecordTrace { message } => format!("RecordTrace({})", message),
        Action::NoOp => "NoOp".to_string(),
    }
}

/// Echo policy: emit a fixed prefix + user text then complete (tests / minimal harness).
#[derive(Clone, Debug)]
pub struct EchoPolicy {
    prefix: String,
}

impl EchoPolicy {
    pub fn new(prefix: impl Into<String>) -> Self {
        Self {
            prefix: prefix.into(),
        }
    }
}

impl EpisodePolicy for EchoPolicy {
    fn on_observation(&mut self, _belief: &mut BeliefState, obs: &Observation) -> PolicyTurn {
        match obs {
            Observation::UserText(u) => PolicyTurn::complete_after(vec![Action::Emit {
                text: format!("{}{}", self.prefix, u),
            }]),
            Observation::ReflectionCycle { .. } => {
                PolicyTurn::continue_with(vec![Action::RecordTrace {
                    message: "reflection_obs_seen".into(),
                }])
            }
        }
    }
}
