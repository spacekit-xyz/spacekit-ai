//! Brain package format: versioned on-disk envelope for the Growformer runtime.
//!
//! A **brain package** is the full portable unit: JSON [`BrainPackageHeader`],
//! JSON `DimensionManager` checkpoint, optional personality JSON, and optional
//! UTF-8 TOML **plugins** blob (see `crate::inference::BrainPluginsManifest`).
//! Legacy format v1 ends after personality; v2 appends `plugin_len: u32` + plugin bytes.
//! A separate `GWFCMPKG` v1 envelope can losslessly compress either inner format.

use serde::{Deserialize, Serialize};

/// Magic + little-endian `format_version` must match for the binary envelope.
pub const BRAIN_PACKAGE_MAGIC: &[u8; 8] = b"GWFBRPKG";
/// Header + checkpoint + personality only (no trailing plugin section).
pub const BRAIN_PACKAGE_FORMAT_VERSION: u32 = 1;
/// Same as v1, then `u32` plugin length + plugin bytes (UTF-8 TOML manifest).
pub const BRAIN_PACKAGE_FORMAT_VERSION_PLUGINS: u32 = 2;
/// Magic for the optional outer compression envelope.
pub const COMPRESSED_BRAIN_MAGIC: &[u8; 8] = b"GWFCMPKG";
/// Current outer compression-envelope version.
pub const COMPRESSED_BRAIN_FORMAT_VERSION: u32 = 1;
/// SpaceKit's default binary codec (gzip level 6).
const COMPRESSED_BRAIN_CODEC_GZIP: u8 = 1;
const COMPRESSED_BRAIN_HEADER_LEN: usize = 32;

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
    /// or `off` / `none` / `disabled` to disable lattice shortcuts **and** TOML headline / PR-wire /
    /// lattice-misfire guards in generation (see `crate::inference::plugins::lattice_shortcuts::sentiment_toml_lexical_guards_active`).
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

    /// Encode the package inside the versioned lossless compression envelope.
    #[cfg(feature = "brain-compression")]
    pub fn encode_to_compressed_bytes(&self) -> Result<Vec<u8>, String> {
        wrap_compressed_brain_bytes(&self.encode_to_bytes()?)
    }

    /// Decode from uncompressed v1/v2 bytes or a compressed outer envelope.
    pub fn decode_from_bytes(data: &[u8]) -> Result<Self, String> {
        peel_brain_file_bytes(data)
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

fn u64_le(b: &[u8]) -> Result<u64, String> {
    if b.len() < 8 {
        return Err("compressed brain package: truncated u64".to_string());
    }
    Ok(u64::from_le_bytes(b[..8].try_into().unwrap()))
}

/// True if `data` begins with the brain package magic (not a raw JSON checkpoint).
pub fn is_brain_package_bytes(data: &[u8]) -> bool {
    data.len() >= BRAIN_PACKAGE_MAGIC.len()
        && &data[..BRAIN_PACKAGE_MAGIC.len()] == BRAIN_PACKAGE_MAGIC.as_slice()
}

/// True if `data` begins with the optional outer compression-envelope magic.
pub fn is_compressed_brain_bytes(data: &[u8]) -> bool {
    data.len() >= COMPRESSED_BRAIN_MAGIC.len()
        && &data[..COMPRESSED_BRAIN_MAGIC.len()] == COMPRESSED_BRAIN_MAGIC.as_slice()
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
    if format_ver != BRAIN_PACKAGE_FORMAT_VERSION
        && format_ver != BRAIN_PACKAGE_FORMAT_VERSION_PLUGINS
    {
        return Err(format!(
            "brain package: unsupported format version {} (expected {} or {})",
            format_ver, BRAIN_PACKAGE_FORMAT_VERSION, BRAIN_PACKAGE_FORMAT_VERSION_PLUGINS
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

/// Wrap an encoded v1/v2 brain package in a versioned gzip envelope.
#[cfg(feature = "brain-compression")]
pub fn wrap_compressed_brain_bytes(data: &[u8]) -> Result<Vec<u8>, String> {
    if !is_brain_package_bytes(data) {
        return Err("compressed brain package: inner payload is not a brain package".to_string());
    }
    let compressor = spacekit_compressor::SpaceKitCompressor::new();
    let result = compressor
        .compress(data, spacekit_compressor::CompressionMode::Binary)
        .map_err(|e| format!("compressed brain package: compression failed: {}", e))?;
    let payload = result.compressed;
    let mut out = Vec::with_capacity(COMPRESSED_BRAIN_HEADER_LEN + payload.len());
    out.extend_from_slice(COMPRESSED_BRAIN_MAGIC);
    push_u32_le(&mut out, COMPRESSED_BRAIN_FORMAT_VERSION);
    out.push(COMPRESSED_BRAIN_CODEC_GZIP);
    out.extend_from_slice(&[0, 0, 0]);
    out.extend_from_slice(&(data.len() as u64).to_le_bytes());
    out.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    out.extend_from_slice(&payload);
    Ok(out)
}

#[cfg(feature = "brain-compression")]
fn decode_compressed_brain_bytes(data: &[u8]) -> Result<Vec<u8>, String> {
    if data.len() < COMPRESSED_BRAIN_HEADER_LEN {
        return Err("compressed brain package: file too small".to_string());
    }
    if !is_compressed_brain_bytes(data) {
        return Err("compressed brain package: bad magic".to_string());
    }
    let version = u32_le(&data[8..12])?;
    if version != COMPRESSED_BRAIN_FORMAT_VERSION {
        return Err(format!(
            "compressed brain package: unsupported format version {} (expected {})",
            version, COMPRESSED_BRAIN_FORMAT_VERSION
        ));
    }
    let codec = data[12];
    if data[13..16] != [0, 0, 0] {
        return Err("compressed brain package: reserved header bytes are non-zero".to_string());
    }
    let original_len = usize::try_from(u64_le(&data[16..24])?)
        .map_err(|_| "compressed brain package: original length is too large".to_string())?;
    let payload_len = usize::try_from(u64_le(&data[24..32])?)
        .map_err(|_| "compressed brain package: payload length is too large".to_string())?;
    let expected_len = COMPRESSED_BRAIN_HEADER_LEN
        .checked_add(payload_len)
        .ok_or_else(|| "compressed brain package: payload length overflow".to_string())?;
    if data.len() != expected_len {
        return Err(format!(
            "compressed brain package: length mismatch (file {} bytes, expected {})",
            data.len(),
            expected_len
        ));
    }
    let payload = &data[COMPRESSED_BRAIN_HEADER_LEN..];
    let decoded = match codec {
        COMPRESSED_BRAIN_CODEC_GZIP => {
            let compressor = spacekit_compressor::SpaceKitCompressor::new();
            compressor
                .decompress(payload, spacekit_compressor::CompressionMode::Binary)
                .map_err(|e| format!("compressed brain package: decompression failed: {}", e))?
        }
        other => {
            return Err(format!(
                "compressed brain package: unsupported codec {}",
                other
            ))
        }
    };
    if decoded.len() != original_len {
        return Err(format!(
            "compressed brain package: decoded length mismatch (got {}, expected {})",
            decoded.len(),
            original_len
        ));
    }
    if !is_brain_package_bytes(&decoded) {
        return Err(
            "compressed brain package: decoded payload has invalid inner magic".to_string(),
        );
    }
    Ok(decoded)
}

/// If `data` is a brain package, return checkpoint + optional personality; otherwise treat
/// `data` as a legacy raw checkpoint (JSON only).
pub fn peel_brain_file_bytes(data: &[u8]) -> Result<BrainPackage, String> {
    if is_compressed_brain_bytes(data) {
        #[cfg(feature = "brain-compression")]
        {
            let decoded = decode_compressed_brain_bytes(data)?;
            return decode_brain_package(&decoded);
        }
        #[cfg(not(feature = "brain-compression"))]
        {
            return Err(
                "compressed brain package: rebuild Growformer with the `brain-compression` feature"
                    .to_string(),
            );
        }
    } else if is_brain_package_bytes(data) {
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

    #[cfg(feature = "brain-compression")]
    #[test]
    fn compressed_brain_package_roundtrip() {
        let mut h = BrainPackageHeader::default();
        h.id = "compressed".to_string();
        let checkpoint = format!(r#"{{"weights":"{}"}}"#, "0123456789".repeat(1000));
        let pkg = BrainPackage::new(
            h,
            checkpoint.as_bytes().to_vec(),
            Some(br#"{"O":0.5}"#.to_vec()),
            Some(b"[sentiment]\nmeta_gk_margin = 0.06\n".to_vec()),
        );
        let plain = pkg.encode_to_bytes().unwrap();
        let compressed = pkg.encode_to_compressed_bytes().unwrap();
        assert!(is_compressed_brain_bytes(&compressed));
        assert!(compressed.len() < plain.len());
        let decoded = BrainPackage::decode_from_bytes(&compressed).unwrap();
        assert_eq!(decoded, pkg);
    }

    #[cfg(feature = "brain-compression")]
    #[test]
    fn compressed_brain_package_rejects_unknown_version() {
        let pkg = BrainPackage::new(
            BrainPackageHeader::default(),
            br#"{"v":1}"#.to_vec(),
            None,
            None,
        );
        let mut compressed = pkg.encode_to_compressed_bytes().unwrap();
        compressed[8..12].copy_from_slice(&99u32.to_le_bytes());
        assert!(BrainPackage::decode_from_bytes(&compressed)
            .unwrap_err()
            .contains("unsupported format version 99"));
    }

    #[test]
    fn peel_legacy_raw_checkpoint() {
        let raw = br#"{"legacy": true}"#;
        let p = peel_brain_file_bytes(raw).unwrap();
        assert_eq!(p.checkpoint, raw);
        assert!(p.personality.is_none());
    }
}
