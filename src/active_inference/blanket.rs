//! Markov blanket: **inward** only via [`Observation`]; **outward** only via committed [`Action`].
//! Internal generative state stays in [`super::BeliefState`] and lattice/rule engines.

use std::fmt;

/// Inward boundary: evidence the spine may condition on (already parsed; no raw sockets here).
#[derive(Clone, Debug, PartialEq)]
pub enum Observation {
    /// User or upstream agent text.
    UserText(String),
    /// One pass through the reflective gate (from MetaCognition or tests).
    ReflectionCycle {
        quality: f32,
        terminal: ReflectionTerminal,
    },
}

/// How a single reflection evaluation ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReflectionTerminal {
    Accepted,
    Retry { attempt: usize },
    Degraded,
}

/// Outward boundary: effects on the world (emit text, log, tool calls in future).
#[derive(Clone, Debug, PartialEq)]
pub enum Action {
    /// Committed user-visible output.
    Emit { text: String },
    /// Bookkeeping only (no user-visible effect in default harness).
    RecordTrace { message: String },
    /// Explicit no-op (policy chose to wait or skip actuation).
    NoOp,
}

/// Environment applies actions and supplies observations (hybrid agent / IO shim).
pub trait EnvironmentPort {
    fn next_observation(&mut self) -> Option<Observation>;
    fn apply(&mut self, action: &Action) -> Result<(), EnvironmentError>;
}

#[derive(Clone, Debug, PartialEq)]
pub struct EnvironmentError(pub String);

impl fmt::Display for EnvironmentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl std::error::Error for EnvironmentError {}
