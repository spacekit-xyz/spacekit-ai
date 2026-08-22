// lib.rs — GrowFormser-NCA crate root
//
// `clifford_llm.rs` is copied from growformer-llm; this crate only exercises
// Multivector / Linear / FFN, so the full transformer types stay unused.
#![allow(dead_code)]
//
// Module layout:
//   clifford_llm — STA multivectors, geometric product, CliffordLinear / FFN
//   blade        — blade indices; REVERSE_SIGNS for compile-time algebra
//   cayley_const — compile-time Cayley table (CliffordAlgebraConst)
//   sample       — SimpleRng for stochastic NCA updates
//   nca          — Neural Cellular Automata scratchpad

pub mod clifford_llm;

pub mod blade;
pub mod cayley_const;
pub mod nca;
pub mod sample;

pub use clifford_llm::{CayleyEntry, CliffordAlgebra, CliffordFFN, CliffordLinear, Multivector};

pub use cayley_const::{CayleyCell, CliffordAlgebraConst, CAYLEY_STA};

// nca
pub use nca::{Cell as NcaCell, CliffordGridNCA, NcaCommand, NcaResponse};
