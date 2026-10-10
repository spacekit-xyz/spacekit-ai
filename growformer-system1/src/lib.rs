//! # Growformer System 1
//!
//! Fast, non-generative decisions: give it a **state** (text or JSON) and
//! **typed questions**, get back schema-safe answers with calibrated
//! probability distributions and confidence.
//!
//! * [`schema`] — the three primitives: **Choice**, **Score**, **Noul**.
//! * [`brain::System1Brain`] — the deployable artifact (`*.s1.json[.gz]`):
//!   questions, a shared featurizer, one fast head per trained question,
//!   per-question temperatures and router thresholds.
//! * [`decide::Decider`] — answers requests; each question goes to its head
//!   when the head is confident, otherwise to an open-model
//!   [`decide::OptionScorer`] (feature `candle`), which reads the shared state
//!   once and scores every question's answer tokens without generating text.
//! * [`eval`] — accuracy / Brier / ECE / latency, LM temperature fitting,
//!   router tuning and distillation (open model → soft labels → heads).
//!
//! Pure Rust, no C/C++ dependencies; builds for Linux, macOS, Windows and
//! `wasm32-unknown-unknown` (feature `wasm` for JS bindings).

pub mod brain;
pub mod calibration;
pub mod data;
pub mod decide;
pub mod eval;
pub mod features;
pub mod head;
pub mod prompt;
pub mod schema;
pub mod state;

#[cfg(feature = "candle")]
pub mod lm;

#[cfg(not(target_arch = "wasm32"))]
pub mod project;

#[cfg(feature = "wasm")]
pub mod wasm;

pub use brain::{LmArch, LmConfig, RouterConfig, System1Brain, TrainConfig};
pub use decide::{
    Answer, Backend, DecideRequest, DecideResponse, Decider, OptionScorer, QuestionRef,
};
pub use schema::{Question, QuestionKind, QuestionSet};
