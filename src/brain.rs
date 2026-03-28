//! Brain package format: versioned on-disk envelope around the JSON `DimensionManager`
//! checkpoint (router, classifier, generation heads, group envs, etc.) plus optional
//! personality JSON.

use serde::{Deserialize, Serialize};

/// Magic + little-endian `format_version` must match for the binary envelope.
pub const BRAIN_PACKAGE_MAGIC: &[u8; 8] = b"GWFBRPKG";
pub const BRAIN_PACKAGE_FORMAT_VERSION: u32 = 1;

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
        }
    }
}

/// Full logical package: header + checkpoint (`DimensionManager` JSON) + optional personality JSON.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BrainPackage {
    pub header: BrainPackageHeader,
    pub checkpoint: Vec<u8>,
    pub personality: Option<Vec<u8>>,
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

/// Wrap checkpoint bytes (JSON `DimensionManager`) and optional personality JSON.
pub fn encode_brain_package(
    header: &BrainPackageHeader,
    checkpoint: &[u8],
    personality: Option<&[u8]>,
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

    let mut out = Vec::with_capacity(8 + 4 + 4 + 4 + header_json.len() + checkpoint.len() + 4 + pers.len());
    out.extend_from_slice(BRAIN_PACKAGE_MAGIC.as_slice());
    push_u32_le(&mut out, BRAIN_PACKAGE_FORMAT_VERSION);
    push_u32_le(&mut out, header_json.len() as u32);
    push_u32_le(&mut out, checkpoint.len() as u32);
    out.extend_from_slice(&header_json);
    out.extend_from_slice(checkpoint);
    push_u32_le(&mut out, pers.len() as u32);
    out.extend_from_slice(pers);
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
    if format_ver != BRAIN_PACKAGE_FORMAT_VERSION {
        return Err(format!(
            "brain package: unsupported format version {} (expected {})",
            format_ver, BRAIN_PACKAGE_FORMAT_VERSION
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
    if data.len() != pers_end {
        return Err(format!(
            "brain package: length mismatch (file {} bytes, expected {})",
            data.len(),
            pers_end
        ));
    }
    let personality = if pers_len == 0 {
        None
    } else {
        Some(data[pers_start..pers_end].to_vec())
    };
    let header: BrainPackageHeader = serde_json::from_slice(header_json)
        .map_err(|e| format!("brain package header deserialize failed: {}", e))?;
    Ok(BrainPackage {
        header,
        checkpoint,
        personality,
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
        let bytes = encode_brain_package(&h, ckpt, Some(pers)).unwrap();
        let p = decode_brain_package(&bytes).unwrap();
        assert_eq!(p.header.id, "test-id");
        assert_eq!(p.checkpoint, ckpt);
        assert_eq!(p.personality.as_deref(), Some(pers.as_slice()));
    }

    #[test]
    fn peel_legacy_raw_checkpoint() {
        let raw = br#"{"legacy": true}"#;
        let p = peel_brain_file_bytes(raw).unwrap();
        assert_eq!(p.checkpoint, raw);
        assert!(p.personality.is_none());
    }
}
