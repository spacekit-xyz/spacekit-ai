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
pub mod reasoning;
pub mod inference;
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
