/// Growformer payment integration.
///
/// Core payment types (`PaymentNetwork`, `PaymentReceipt`, `FeeRouter`, etc.)
/// are now defined in the `spacekit-payments` crate. The x402-specific seller/buyer
/// binaries in `x402/` still use local copies of these types for build isolation.
///
/// Migration: Add `spacekit-payments = { path = "../../spacekit/spacekit-payments" }`
/// to the x402-module Cargo.toml and replace the local type definitions with
/// re-exports from `spacekit_payments`.
pub mod x402;