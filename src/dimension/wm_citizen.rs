//! WM citizen records shared by DimensionManager and Phase 5a spike.

use serde::{Deserialize, Serialize};

use crate::types::GroupId;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum WmCitizenKind {
    ActingDisk,
    Composed,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WmCitizenRecord {
    pub group_id: GroupId,
    pub task_name: String,
    pub kind: WmCitizenKind,
    pub encoder_fingerprint: u64,
    /// Path to pinned JSON bundle.
    pub bundle_path: String,
}
