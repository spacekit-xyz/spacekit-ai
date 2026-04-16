//! Internal belief state updated by the spine (not raw environment payloads).

/// Carries step counters and recent reflective quality for policy hooks.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BeliefState {
    /// Monotonic spine steps (observations processed).
    pub step: usize,
    /// Last MetaCognition-style quality score, if any.
    pub last_quality: Option<f32>,
    /// How many reflection retries have been seen this episode.
    pub reflection_retries: usize,
}

impl BeliefState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn advance_step(&mut self) {
        self.step = self.step.saturating_add(1);
    }
}
