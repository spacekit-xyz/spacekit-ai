//! Runtime entitlement gate for embedded growformer (SpaceKit CLI).
//!
//! When enforcement is enabled, train / infer / merge entry points require an active
//! capability in the thread-local [`EntitlementContext`] (see GROWFORMER_SPEC.md).

use std::cell::RefCell;

/// Capability: end-to-end brain training.
pub const CAP_TRAIN: &str = "growformer.train";
/// Capability: inference (including batch / REPL).
pub const CAP_INFER: &str = "growformer.infer";
/// Capability: merge overlay brains.
pub const CAP_MERGE: &str = "growformer.merge";

/// Context supplied by SpaceKit before calling [`crate::run_cli_with_entitlement`].
#[derive(Debug, Clone)]
pub struct EntitlementContext {
    pub user_did: String,
    pub tier_name: String,
    pub active_capabilities: Vec<String>,
    /// Unix seconds; `0` = no expiry.
    pub expires_at: u64,
    pub quota_remaining: Option<u64>,
    pub on_chain_verified: bool,
}

impl EntitlementContext {
    pub fn has_active_entitlement_for(&self, capability: &str) -> bool {
        if !self.active_capabilities.iter().any(|c| c == capability) {
            return false;
        }
        if self.expires_at > 0 {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            if self.expires_at < now {
                return false;
            }
        }
        true
    }

    pub fn consume_quota(&self, _operation: &str) -> Result<(), String> {
        if let Some(0) = self.quota_remaining {
            return Err("growformer quota exhausted for this tier".to_string());
        }
        Ok(())
    }
}

thread_local! {
    static CURRENT: RefCell<Option<EntitlementContext>> = const { RefCell::new(None) };
    static ENFORCED: RefCell<bool> = const { RefCell::new(false) };
}

pub fn set_enforced(enforced: bool) {
    ENFORCED.with(|f| *f.borrow_mut() = enforced);
}

pub fn is_enforced() -> bool {
    ENFORCED.with(|f| *f.borrow())
}

pub fn set_context(ctx: EntitlementContext) {
    CURRENT.with(|c| *c.borrow_mut() = Some(ctx));
}

pub fn clear_context() {
    CURRENT.with(|c| *c.borrow_mut() = None);
    ENFORCED.with(|f| *f.borrow_mut() = false);
}

/// Gate a capability when enforcement is active (SpaceKit embed path).
pub fn require_capability(capability: &str) -> Result<(), String> {
    if !is_enforced() {
        return Ok(());
    }
    CURRENT.with(|cell| {
        let ctx = cell.borrow();
        let Some(ctx) = ctx.as_ref() else {
            return Err("growformer entitlement context missing (internal error)".to_string());
        };
        if !ctx.has_active_entitlement_for(capability) {
            return Err(format!(
                "No active entitlement for {} (tier: {}). \
                 Obtain access: spacekit content view --content-id <growformer_id> \
                 or spacekit content access --content-id <id>",
                capability, ctx.tier_name
            ));
        }
        ctx.consume_quota(capability)
    })
}
