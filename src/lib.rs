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
pub mod gradient_memory;
pub mod predictive_coder;
pub mod dimension;
pub mod growformer_lang;
pub mod topic_graph;
pub mod micro_brain;
pub mod understanding;
pub mod environment;
#[cfg(all(not(target_arch = "wasm32"), feature = "cli"))]
pub mod mnist;
pub mod neuron;
pub mod systems;
pub mod types;
pub mod metacognition;
pub mod reasoning;
pub mod service;
pub mod spectral;
pub mod text_autoencoder;
#[cfg(feature = "wasm-bindgen")]
pub mod wasm;
