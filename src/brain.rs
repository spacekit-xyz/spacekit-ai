//! Brain package format: versioned on-disk envelope for the Growformer runtime.
//!
//! A **brain package** is the full portable unit: JSON [`BrainPackageHeader`],
//! JSON `DimensionManager` checkpoint, optional personality JSON, and optional
//! UTF-8 TOML **plugins** blob (see `crate::inference::BrainPluginsManifest`).
//! Legacy format v1 ends after personality; v2 appends `plugin_len: u32` + plugin bytes.

use serde::{Deserialize, Serialize};

/// Magic + little-endian `format_version` must match for the binary envelope.
pub const BRAIN_PACKAGE_MAGIC: &[u8; 8] = b"GWFBRPKG";
/// Header + checkpoint + personality only (no trailing plugin section).
pub const BRAIN_PACKAGE_FORMAT_VERSION: u32 = 1;
/// Same as v1, then `u32` plugin length + plugin bytes (UTF-8 TOML manifest).
pub const BRAIN_PACKAGE_FORMAT_VERSION_PLUGINS: u32 = 2;

/// Semantic version for display and provenance (not the binary format version).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct SemVer {
    pub major: u16,
    pub minor: u16,
    pub patch: u16,
}

impl Default for SemVer {
    fn default() -> Self {
        Self {
            major: 0,
            minor: 1,
            patch: 0,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct Identity {
    pub name: String,
    pub contact: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct BrainRef {
    pub id: String,
    pub version: SemVer,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct TopicRef {
    pub name: String,
}

/// Cryptographic provenance placeholder (empty unless you attach a signature later).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct Signature {
    pub bytes: Vec<u8>,
}

/// JSON header stored inside the binary brain package (before checkpoint bytes).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct BrainPackageHeader {
    /// Schema for this header struct (increment when header fields change).
    pub header_schema: u32,
    pub id: String,
    pub version: SemVer,
    pub name: String,
    pub description: String,
    pub author: Identity,
    pub base_brain: Option<BrainRef>,
    pub topics: Vec<TopicRef>,
    pub signature: Signature,
    pub merged_from: Vec<BrainRef>,
    /// Hint for inference-time plugins: e.g. `sentiment_lattice` (set on export for sentiment brains),
    /// or `off` / `none` to disable sentiment shortcuts even when the lattice shape matches.
    #[serde(default)]
    pub inference_profile: Option<String>,
}

impl Default for BrainPackageHeader {
    fn default() -> Self {
        Self {
            header_schema: 1,
            id: String::new(),
            version: SemVer::default(),
            name: "growformer".to_string(),
            description: String::new(),
            author: Identity::default(),
            base_brain: None,
            topics: vec![],
            signature: Signature::default(),
            merged_from: vec![],
            inference_profile: None,
        }
    }
}

/// Full logical unit for create / load / export: metadata, weights checkpoint, drift, and plugins.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BrainPackage {
    pub header: BrainPackageHeader,
    pub checkpoint: Vec<u8>,
    pub personality: Option<Vec<u8>>,
    /// UTF-8 TOML inference manifest (see `crate::inference::BrainPluginsManifest`); `None` → on-disk format v1.
    pub plugins_blob: Option<Vec<u8>>,
}

impl BrainPackage {
    /// Assemble a package in memory before [`Self::encode_to_bytes`].
    pub fn new(
        header: BrainPackageHeader,
        checkpoint: Vec<u8>,
        personality: Option<Vec<u8>>,
        plugins_blob: Option<Vec<u8>>,
    ) -> Self {
        Self {
            header,
            checkpoint,
            personality,
            plugins_blob,
        }
    }

    /// Serialize to the binary `.bin` envelope consumed by [`peel_brain_file_bytes`].
    pub fn encode_to_bytes(&self) -> Result<Vec<u8>, String> {
        encode_brain_package(
            &self.header,
            &self.checkpoint,
            self.personality.as_deref(),
            self.plugins_blob.as_deref().filter(|b| !b.is_empty()),
        )
    }

    /// Decode from file bytes (v1 or v2).
    pub fn decode_from_bytes(data: &[u8]) -> Result<Self, String> {
        decode_brain_package(data)
    }
}

fn u32_le(b: &[u8]) -> Result<u32, String> {
    if b.len() < 4 {
        return Err("brain package: truncated u32".to_string());
    }
    Ok(u32::from_le_bytes(b[..4].try_into().unwrap()))
}

fn push_u32_le(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

/// True if `data` begins with the brain package magic (not a raw JSON checkpoint).
pub fn is_brain_package_bytes(data: &[u8]) -> bool {
    data.len() >= BRAIN_PACKAGE_MAGIC.len() && &data[..BRAIN_PACKAGE_MAGIC.len()] == BRAIN_PACKAGE_MAGIC.as_slice()
}

/// Wrap checkpoint bytes (JSON `DimensionManager`), optional personality JSON, and optional plugins TOML.
///
/// Uses on-disk format v1 when `plugins_blob` is `None` or empty (backward compatible).
/// Non-empty plugins use format v2 (trailing `u32` length + bytes).
pub fn encode_brain_package(
    header: &BrainPackageHeader,
    checkpoint: &[u8],
    personality: Option<&[u8]>,
    plugins_blob: Option<&[u8]>,
) -> Result<Vec<u8>, String> {
    let header_json = serde_json::to_vec(header)
        .map_err(|e| format!("brain package header serialize failed: {}", e))?;
    if header_json.len() > u32::MAX as usize {
        return Err("brain package: header too large".to_string());
    }
    if checkpoint.len() > u32::MAX as usize {
        return Err("brain package: checkpoint too large".to_string());
    }
    let pers = personality.unwrap_or(&[]);
    if pers.len() > u32::MAX as usize {
        return Err("brain package: personality blob too large".to_string());
    }
    let plug: &[u8] = match plugins_blob {
        Some(b) if !b.is_empty() => b,
        _ => &[],
    };
    let use_v2 = !plug.is_empty();
    if plug.len() > u32::MAX as usize {
        return Err("brain package: plugins blob too large".to_string());
    }
    let format_ver = if use_v2 {
        BRAIN_PACKAGE_FORMAT_VERSION_PLUGINS
    } else {
        BRAIN_PACKAGE_FORMAT_VERSION
    };

    let mut cap = 8 + 4 + 4 + 4 + header_json.len() + checkpoint.len() + 4 + pers.len();
    if use_v2 {
        cap += 4 + plug.len();
    }
    let mut out = Vec::with_capacity(cap);
    out.extend_from_slice(BRAIN_PACKAGE_MAGIC.as_slice());
    push_u32_le(&mut out, format_ver);
    push_u32_le(&mut out, header_json.len() as u32);
    push_u32_le(&mut out, checkpoint.len() as u32);
    out.extend_from_slice(&header_json);
    out.extend_from_slice(checkpoint);
    push_u32_le(&mut out, pers.len() as u32);
    out.extend_from_slice(pers);
    if use_v2 {
        push_u32_le(&mut out, plug.len() as u32);
        out.extend_from_slice(plug);
    }
    Ok(out)
}

/// Decode a brain package file. Fails if magic/version is wrong or lengths are inconsistent.
pub fn decode_brain_package(data: &[u8]) -> Result<BrainPackage, String> {
    if data.len() < 8 + 4 + 4 + 4 {
        return Err("brain package: file too small".to_string());
    }
    if &data[..8] != BRAIN_PACKAGE_MAGIC.as_slice() {
        return Err("brain package: bad magic".to_string());
    }
    let format_ver = u32_le(&data[8..12])?;
    if format_ver != BRAIN_PACKAGE_FORMAT_VERSION && format_ver != BRAIN_PACKAGE_FORMAT_VERSION_PLUGINS {
        return Err(format!(
            "brain package: unsupported format version {} (expected {} or {})",
            format_ver,
            BRAIN_PACKAGE_FORMAT_VERSION,
            BRAIN_PACKAGE_FORMAT_VERSION_PLUGINS
        ));
    }
    let header_len = u32_le(&data[12..16])? as usize;
    let ckpt_len = u32_le(&data[16..20])? as usize;
    let base = 20usize;
    let end_header = base
        .checked_add(header_len)
        .ok_or_else(|| "brain package: header length overflow".to_string())?;
    let end_ckpt = end_header
        .checked_add(ckpt_len)
        .ok_or_else(|| "brain package: checkpoint length overflow".to_string())?;
    if data.len() < end_ckpt + 4 {
        return Err("brain package: truncated body".to_string());
    }
    let header_json = &data[base..end_header];
    let checkpoint = data[end_header..end_ckpt].to_vec();
    let pers_len = u32_le(&data[end_ckpt..end_ckpt + 4])? as usize;
    let pers_start = end_ckpt + 4;
    let pers_end = pers_start
        .checked_add(pers_len)
        .ok_or_else(|| "brain package: personality length overflow".to_string())?;
    let (personality, plugins_blob, expected_end) = if format_ver == BRAIN_PACKAGE_FORMAT_VERSION {
        if data.len() != pers_end {
            return Err(format!(
                "brain package: length mismatch (file {} bytes, expected {} for format v1)",
                data.len(),
                pers_end
            ));
        }
        let personality = if pers_len == 0 {
            None
        } else {
            Some(data[pers_start..pers_end].to_vec())
        };
        (personality, None, pers_end)
    } else {
        if data.len() < pers_end + 4 {
            return Err("brain package: truncated plugins length (format v2)".to_string());
        }
        let personality = if pers_len == 0 {
            None
        } else {
            Some(data[pers_start..pers_end].to_vec())
        };
        let plugin_len = u32_le(&data[pers_end..pers_end + 4])? as usize;
        let plug_start = pers_end + 4;
        let plug_end = plug_start
            .checked_add(plugin_len)
            .ok_or_else(|| "brain package: plugin length overflow".to_string())?;
        if data.len() != plug_end {
            return Err(format!(
                "brain package: length mismatch (file {} bytes, expected {} for format v2)",
                data.len(),
                plug_end
            ));
        }
        let plugins_blob = if plugin_len == 0 {
            None
        } else {
            Some(data[plug_start..plug_end].to_vec())
        };
        (personality, plugins_blob, plug_end)
    };
    debug_assert_eq!(data.len(), expected_end);
    let header: BrainPackageHeader = serde_json::from_slice(header_json)
        .map_err(|e| format!("brain package header deserialize failed: {}", e))?;
    Ok(BrainPackage {
        header,
        checkpoint,
        personality,
        plugins_blob,
    })
}

/// If `data` is a brain package, return checkpoint + optional personality; otherwise treat
/// `data` as a legacy raw checkpoint (JSON only).
pub fn peel_brain_file_bytes(data: &[u8]) -> Result<BrainPackage, String> {
    if is_brain_package_bytes(data) {
        decode_brain_package(data)
    } else {
        Ok(BrainPackage {
            header: BrainPackageHeader::default(),
            checkpoint: data.to_vec(),
            personality: None,
            plugins_blob: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn brain_package_roundtrip() {
        let mut h = BrainPackageHeader::default();
        h.id = "test-id".to_string();
        h.name = "unit-test".to_string();
        let ckpt = br#"{"smoke": true}"#;
        let pers = br#"{"O":0.5}"#;
        let bytes = encode_brain_package(&h, ckpt, Some(pers), None).unwrap();
        let p = decode_brain_package(&bytes).unwrap();
        assert_eq!(p.header.id, "test-id");
        assert_eq!(p.checkpoint, ckpt);
        assert_eq!(p.personality.as_deref(), Some(pers.as_slice()));
        assert!(p.plugins_blob.is_none());
    }

    #[test]
    fn brain_package_v2_plugins_roundtrip() {
        let mut h = BrainPackageHeader::default();
        h.id = "p2".to_string();
        let ckpt = br#"{"v":2}"#;
        let pers = br#"{}"#;
        let plugins = b"[sentiment]\nmeta_gk_margin = 0.06\n";
        let bytes = encode_brain_package(&h, ckpt, Some(pers), Some(plugins)).unwrap();
        let p = decode_brain_package(&bytes).unwrap();
        assert_eq!(p.plugins_blob.as_deref(), Some(plugins.as_slice()));
        let round = p.encode_to_bytes().unwrap();
        assert_eq!(round, bytes);
    }

    #[test]
    fn peel_legacy_raw_checkpoint() {
        let raw = br#"{"legacy": true}"#;
        let p = peel_brain_file_bytes(raw).unwrap();
        assert_eq!(p.checkpoint, raw);
        assert!(p.personality.is_none());
    }
}
