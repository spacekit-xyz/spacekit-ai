/// Compatibility shim: dispatches to `rayon::par_iter` when the `parallel` feature
/// is enabled, falls back to sequential `iter` otherwise (WASM-safe).
#[cfg(feature = "parallel")]
#[macro_export]
macro_rules! maybe_par_iter {
    ($slice:expr) => {
        $slice.par_iter()
    };
}

#[cfg(not(feature = "parallel"))]
#[macro_export]
macro_rules! maybe_par_iter {
    ($slice:expr) => {
        $slice.iter()
    };
}

#[cfg(feature = "parallel")]
#[macro_export]
macro_rules! maybe_par_iter_mut {
    ($slice:expr) => {
        $slice.par_iter_mut()
    };
}

#[cfg(not(feature = "parallel"))]
#[macro_export]
macro_rules! maybe_par_iter_mut {
    ($slice:expr) => {
        $slice.iter_mut()
    };
}

pub mod clifford;
pub mod cloze;
pub mod coherence;
/// Categorical DAG, Pythagoras-node training scaffolding, and sentiment functor experiments.
#[cfg(feature = "categorical")]
pub mod category;
#[cfg(feature = "training")]
pub mod gradient_memory;
pub mod predictive_coder;
pub mod drive_field;
pub mod reflective_field;
pub mod basal_ganglia;
pub mod dimension;
pub mod growformer_lang;
pub mod text_keywords;
pub mod topic_graph;
pub mod micro_brain;
pub mod understanding;
pub mod environment;
#[cfg(all(not(target_arch = "wasm32"), feature = "cli"))]
pub mod mnist;
#[cfg(all(not(target_arch = "wasm32"), feature = "cli"))]
pub mod clifford_mnist;
#[cfg(all(not(target_arch = "wasm32"), feature = "cli"))]
pub mod pathmnist;
#[cfg(all(not(target_arch = "wasm32"), feature = "cli"))]
pub mod arc_agi;
#[cfg(all(not(target_arch = "wasm32"), feature = "cli"))]
pub mod arc_brain;
#[cfg(all(not(target_arch = "wasm32"), feature = "cli"))]
pub mod arc_dsl;
pub mod neuron;
pub mod systems;
pub mod brain;
pub mod types;
pub mod metacognition;
/// Active Inference spine: belief, Markov blanket I/O, episode loop (Phases 0–3).
pub mod active_inference;
pub mod reasoning;
pub mod inference;
pub mod infer_log;

/// Print inference diagnostics only when `infer_log::infer_trace_enabled()` (see CLI `--verbose` / quiet `--infer`).
#[macro_export]
macro_rules! infer_trace {
    ($($t:tt)*) => {
        if $crate::infer_log::infer_trace_enabled() {
            println!($($t)*);
        }
    };
}
#[cfg(all(not(target_arch = "wasm32"), feature = "cli"))]
pub mod project_gf;
/// Full Growformer train / merge / infer CLI (same as the `growformer` binary). Used by SpaceKit and tests.
#[cfg(all(not(target_arch = "wasm32"), feature = "cli"))]
mod cli_impl;

/// Entitlement gate for embedded distribution (SpaceKit CLI).
#[cfg(all(not(target_arch = "wasm32"), feature = "cli"))]
pub mod entitlement;

/// Run the full growformer CLI from an argv-like iterator.
///
/// `argv[0]` is the program name (ignored by clap except for help text).
/// Returns `Ok(())` on success or `Err(message)` on failure.
///
/// Standalone `growformer` binary calls this without entitlement enforcement.
/// SpaceKit must use [`run_cli_with_entitlement`].
#[cfg(all(not(target_arch = "wasm32"), feature = "cli"))]
pub fn run_cli<I, T>(argv: I) -> Result<(), String>
where
    I: IntoIterator<Item = T>,
    T: Into<std::ffi::OsString> + Clone,
{
    cli_impl::run_from_argv(argv)
}

/// Run growformer CLI with entitlement enforcement (library-embedded distribution).
#[cfg(all(not(target_arch = "wasm32"), feature = "cli"))]
pub fn run_cli_with_entitlement<I, T>(
    argv: I,
    ctx: entitlement::EntitlementContext,
) -> Result<(), String>
where
    I: IntoIterator<Item = T>,
    T: Into<std::ffi::OsString> + Clone,
{
    entitlement::set_context(ctx);
    entitlement::set_enforced(true);
    let result = cli_impl::run_from_argv(argv);
    entitlement::clear_context();
    result
}
pub mod service;
pub mod runtime;
pub mod spectral;
pub mod text_autoencoder;
#[cfg(feature = "training")]
pub mod training_objectives;
#[cfg(not(target_arch = "wasm32"))]
pub mod tools_builtin;
#[cfg(feature = "wasm-bindgen")]
pub mod wasm;
