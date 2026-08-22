//! Brain package format: versioned on-disk envelope for the Growformer runtime.
//!
//! A **brain package** is the full portable unit: JSON [`BrainPackageHeader`],
//! JSON `DimensionManager` checkpoint, optional personality JSON, and optional
//! UTF-8 TOML **plugins** blob (see `crate::inference::BrainPluginsManifest`).
//! Legacy format v1 ends after personality; v2 appends `plugin_len: u32` + plugin bytes.
//! A separate `GWFCMPKG` v1 envelope can losslessly compress either inner format.

#[cfg(feature = "brain-compression")]
use flate2::{read::GzDecoder, write::GzEncoder, Compression};
use serde::{Deserialize, Serialize};
use std::io::{self, Read, Seek, SeekFrom, Write};

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
pub const DEFAULT_BRAIN_IO_LIMIT_BYTES: u64 = 1024 * 1024 * 1024;

/// Hard limits applied while reading or writing brain artifacts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BrainIoLimits {
    pub max_file_bytes: u64,
    pub max_decoded_bytes: u64,
}

impl Default for BrainIoLimits {
    fn default() -> Self {
        Self {
            max_file_bytes: DEFAULT_BRAIN_IO_LIMIT_BYTES,
            max_decoded_bytes: DEFAULT_BRAIN_IO_LIMIT_BYTES,
        }
    }
}

/// Reader that fails before returning any byte past its configured limit.
pub struct LimitedReader<R> {
    inner: R,
    remaining: u64,
    label: &'static str,
}

impl<R> LimitedReader<R> {
    pub fn new(inner: R, limit: u64, label: &'static str) -> Self {
        Self {
            inner,
            remaining: limit,
            label,
        }
    }

    pub fn remaining(&self) -> u64 {
        self.remaining
    }
}

impl<R: Read> Read for LimitedReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }
        if self.remaining == 0 {
            let mut probe = [0u8; 1];
            return match self.inner.read(&mut probe)? {
                0 => Ok(0),
                _ => Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("{} exceeds configured byte limit", self.label),
                )),
            };
        }
        let allowed = usize::try_from(self.remaining.min(buf.len() as u64)).unwrap_or(buf.len());
        let read = self.inner.read(&mut buf[..allowed])?;
        self.remaining -= read as u64;
        Ok(read)
    }
}

#[cfg(feature = "brain-compression")]
struct LimitedWriter<'a, W> {
    inner: &'a mut W,
    remaining: u64,
    written: u64,
    label: &'static str,
}

#[cfg(feature = "brain-compression")]
impl<'a, W> LimitedWriter<'a, W> {
    fn new(inner: &'a mut W, limit: u64, label: &'static str) -> Self {
        Self {
            inner,
            remaining: limit,
            written: 0,
            label,
        }
    }

    fn written(&self) -> u64 {
        self.written
    }
}

#[cfg(feature = "brain-compression")]
impl<W: Write> Write for LimitedWriter<'_, W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if buf.len() as u64 > self.remaining {
            return Err(io::Error::new(
                io::ErrorKind::WriteZero,
                format!("{} exceeds configured byte limit", self.label),
            ));
        }
        let written = self.inner.write(buf)?;
        self.remaining -= written as u64;
        self.written += written as u64;
        Ok(written)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

/// Parsed package metadata plus a checkpoint deserialized directly from its bounded section.
#[derive(Debug)]
pub struct ParsedBrain<T> {
    pub header: BrainPackageHeader,
    pub checkpoint: T,
    pub personality: Option<Vec<u8>>,
    pub plugins_blob: Option<Vec<u8>>,
}

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

fn read_u32<R: Read>(reader: &mut R, context: &str) -> Result<u32, String> {
    let mut bytes = [0u8; 4];
    reader
        .read_exact(&mut bytes)
        .map_err(|e| format!("{}: truncated u32: {}", context, e))?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64<R: Read>(reader: &mut R, context: &str) -> Result<u64, String> {
    let mut bytes = [0u8; 8];
    reader
        .read_exact(&mut bytes)
        .map_err(|e| format!("{}: truncated u64: {}", context, e))?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_vec<R: Read>(reader: &mut R, len: u64, context: &str) -> Result<Vec<u8>, String> {
    let len = usize::try_from(len).map_err(|_| format!("{}: length is too large", context))?;
    let mut bytes = vec![0u8; len];
    reader
        .read_exact(&mut bytes)
        .map_err(|e| format!("{}: truncated body: {}", context, e))?;
    Ok(bytes)
}

fn ensure_eof<R: Read>(reader: &mut R, context: &str) -> Result<(), String> {
    let mut byte = [0u8; 1];
    match reader.read(&mut byte) {
        Ok(0) => Ok(()),
        Ok(_) => Err(format!("{}: trailing data", context)),
        Err(e) => Err(format!("{}: {}", context, e)),
    }
}

fn checked_total(parts: &[u64], limit: u64, context: &str) -> Result<u64, String> {
    let total = parts.iter().try_fold(0u64, |sum, part| {
        sum.checked_add(*part)
            .ok_or_else(|| format!("{}: declared length overflow", context))
    })?;
    if total > limit {
        return Err(format!(
            "{}: declared length {} exceeds configured {} byte limit",
            context, total, limit
        ));
    }
    Ok(total)
}

fn parse_plain_reader<R, T, F>(
    mut reader: R,
    prefix: [u8; 8],
    limits: BrainIoLimits,
    deserialize_checkpoint: F,
) -> Result<ParsedBrain<T>, String>
where
    R: Read,
    F: FnOnce(&mut dyn Read) -> Result<T, String>,
{
    if prefix != *BRAIN_PACKAGE_MAGIC {
        let chained = io::Cursor::new(prefix).chain(reader);
        let mut decoded =
            LimitedReader::new(chained, limits.max_decoded_bytes, "legacy brain checkpoint");
        let checkpoint = deserialize_checkpoint(&mut decoded)?;
        ensure_eof(&mut decoded, "legacy brain checkpoint")?;
        return Ok(ParsedBrain {
            header: BrainPackageHeader::default(),
            checkpoint,
            personality: None,
            plugins_blob: None,
        });
    }

    let format_ver = read_u32(&mut reader, "brain package")?;
    if format_ver != BRAIN_PACKAGE_FORMAT_VERSION
        && format_ver != BRAIN_PACKAGE_FORMAT_VERSION_PLUGINS
    {
        return Err(format!(
            "brain package: unsupported format version {} (expected {} or {})",
            format_ver, BRAIN_PACKAGE_FORMAT_VERSION, BRAIN_PACKAGE_FORMAT_VERSION_PLUGINS
        ));
    }
    let header_len = read_u32(&mut reader, "brain package")? as u64;
    let checkpoint_len = read_u32(&mut reader, "brain package")? as u64;
    checked_total(
        &[20, header_len, checkpoint_len, 4],
        limits.max_decoded_bytes,
        "brain package",
    )?;

    let header_json = read_vec(&mut reader, header_len, "brain package header")?;
    let header = serde_json::from_slice(&header_json)
        .map_err(|e| format!("brain package header deserialize failed: {}", e))?;

    let mut checkpoint_reader = reader.by_ref().take(checkpoint_len);
    let checkpoint = deserialize_checkpoint(&mut checkpoint_reader)?;
    if checkpoint_reader.limit() != 0 {
        return Err(format!(
            "brain package: checkpoint parser left {} declared bytes unread",
            checkpoint_reader.limit()
        ));
    }

    let personality_len = read_u32(&mut reader, "brain package personality")? as u64;
    checked_total(
        &[20, header_len, checkpoint_len, 4, personality_len],
        limits.max_decoded_bytes,
        "brain package",
    )?;
    let personality = if personality_len == 0 {
        None
    } else {
        Some(read_vec(
            &mut reader,
            personality_len,
            "brain package personality",
        )?)
    };

    let plugins_blob = if format_ver == BRAIN_PACKAGE_FORMAT_VERSION_PLUGINS {
        let plugin_len = read_u32(&mut reader, "brain package plugins")? as u64;
        checked_total(
            &[
                20,
                header_len,
                checkpoint_len,
                4,
                personality_len,
                4,
                plugin_len,
            ],
            limits.max_decoded_bytes,
            "brain package",
        )?;
        if plugin_len == 0 {
            None
        } else {
            Some(read_vec(&mut reader, plugin_len, "brain package plugins")?)
        }
    } else {
        None
    };
    ensure_eof(&mut reader, "brain package")?;
    Ok(ParsedBrain {
        header,
        checkpoint,
        personality,
        plugins_blob,
    })
}

/// Parse a package sequentially and deserialize its checkpoint from a bounded reader.
pub fn parse_brain_reader<R, T, F>(
    reader: R,
    limits: BrainIoLimits,
    deserialize_checkpoint: F,
) -> Result<ParsedBrain<T>, String>
where
    R: Read,
    F: FnOnce(&mut dyn Read) -> Result<T, String>,
{
    let mut file = LimitedReader::new(reader, limits.max_file_bytes, "brain file");
    let mut prefix = [0u8; 8];
    let mut prefix_len = 0;
    while prefix_len < prefix.len() {
        let read = file
            .read(&mut prefix[prefix_len..])
            .map_err(|e| format!("brain file: {}", e))?;
        if read == 0 {
            break;
        }
        prefix_len += read;
    }
    if prefix_len < prefix.len() {
        let chained = io::Cursor::new(prefix[..prefix_len].to_vec()).chain(file);
        let mut decoded =
            LimitedReader::new(chained, limits.max_decoded_bytes, "legacy brain checkpoint");
        let checkpoint = deserialize_checkpoint(&mut decoded)?;
        ensure_eof(&mut decoded, "legacy brain checkpoint")?;
        return Ok(ParsedBrain {
            header: BrainPackageHeader::default(),
            checkpoint,
            personality: None,
            plugins_blob: None,
        });
    }

    if prefix != *COMPRESSED_BRAIN_MAGIC {
        return parse_plain_reader(file, prefix, limits, deserialize_checkpoint);
    }

    #[cfg(not(feature = "brain-compression"))]
    {
        let _ = deserialize_checkpoint;
        Err(
            "compressed brain package: rebuild Growformer with the `brain-compression` feature"
                .to_string(),
        )
    }
    #[cfg(feature = "brain-compression")]
    {
        let version = read_u32(&mut file, "compressed brain package")?;
        if version != COMPRESSED_BRAIN_FORMAT_VERSION {
            return Err(format!(
                "compressed brain package: unsupported format version {} (expected {})",
                version, COMPRESSED_BRAIN_FORMAT_VERSION
            ));
        }
        let mut codec_reserved = [0u8; 4];
        file.read_exact(&mut codec_reserved)
            .map_err(|e| format!("compressed brain package: truncated header: {}", e))?;
        if codec_reserved[0] != COMPRESSED_BRAIN_CODEC_GZIP {
            return Err(format!(
                "compressed brain package: unsupported codec {}",
                codec_reserved[0]
            ));
        }
        if codec_reserved[1..] != [0, 0, 0] {
            return Err("compressed brain package: reserved header bytes are non-zero".to_string());
        }
        let original_len = read_u64(&mut file, "compressed brain package")?;
        let payload_len = read_u64(&mut file, "compressed brain package")?;
        checked_total(
            &[COMPRESSED_BRAIN_HEADER_LEN as u64, payload_len],
            limits.max_file_bytes,
            "compressed brain package",
        )?;
        if original_len > limits.max_decoded_bytes {
            return Err(format!(
                "compressed brain package: declared decoded length {} exceeds configured {} byte limit",
                original_len, limits.max_decoded_bytes
            ));
        }
        let mut payload = file.by_ref().take(payload_len);

        #[cfg(not(target_arch = "wasm32"))]
        {
            let mut decoded = tempfile::tempfile()
                .map_err(|e| format!("compressed brain package: temporary file failed: {}", e))?;
            let count = {
                let decoder = GzDecoder::new(&mut payload);
                let mut bounded =
                    LimitedReader::new(decoder, original_len, "decoded brain package");
                io::copy(&mut bounded, &mut decoded)
                    .map_err(|e| format!("compressed brain package: decompression failed: {}", e))?
            };
            if count != original_len {
                return Err(format!(
                    "compressed brain package: decoded length mismatch (got {}, expected {})",
                    count, original_len
                ));
            }
            if payload.limit() != 0 {
                return Err("compressed brain package: truncated payload".to_string());
            }
            ensure_eof(&mut file, "compressed brain package")?;
            decoded
                .seek(SeekFrom::Start(0))
                .map_err(|e| format!("compressed brain package: rewind failed: {}", e))?;
            let mut inner_prefix = [0u8; 8];
            decoded.read_exact(&mut inner_prefix).map_err(|e| {
                format!("compressed brain package: decoded payload truncated: {}", e)
            })?;
            if inner_prefix != *BRAIN_PACKAGE_MAGIC {
                return Err(
                    "compressed brain package: decoded payload has invalid inner magic".to_string(),
                );
            }
            parse_plain_reader(decoded, inner_prefix, limits, deserialize_checkpoint)
        }
        #[cfg(target_arch = "wasm32")]
        {
            let mut decoded = Vec::with_capacity(
                usize::try_from(original_len)
                    .map_err(|_| "compressed brain package: decoded length too large")?,
            );
            let count = {
                let decoder = GzDecoder::new(&mut payload);
                let mut bounded =
                    LimitedReader::new(decoder, original_len, "decoded brain package");
                io::copy(&mut bounded, &mut decoded)
                    .map_err(|e| format!("compressed brain package: decompression failed: {}", e))?
            };
            if count != original_len || payload.limit() != 0 {
                return Err("compressed brain package: decoded length mismatch".to_string());
            }
            ensure_eof(&mut file, "compressed brain package")?;
            if decoded.len() < 8 || decoded[..8] != *BRAIN_PACKAGE_MAGIC {
                return Err(
                    "compressed brain package: decoded payload has invalid inner magic".to_string(),
                );
            }
            let mut inner_prefix = [0u8; 8];
            inner_prefix.copy_from_slice(&decoded[..8]);
            parse_plain_reader(
                io::Cursor::new(&decoded[8..]),
                inner_prefix,
                limits,
                deserialize_checkpoint,
            )
        }
    }
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

/// Stream an uncompressed v1/v2 package to a seekable destination.
///
/// Length fields are backfilled after serialization, preserving the existing byte format
/// without first materializing the checkpoint or complete package in a `Vec`.
pub fn write_brain_package_seekable<W, F>(
    writer: &mut W,
    header: &BrainPackageHeader,
    personality: Option<&[u8]>,
    plugins_blob: Option<&[u8]>,
    limits: BrainIoLimits,
    write_checkpoint: F,
) -> Result<u64, String>
where
    W: Write + Seek,
    F: FnOnce(&mut W) -> Result<(), String>,
{
    let start = writer
        .stream_position()
        .map_err(|e| format!("brain package: position failed: {}", e))?;
    let plugins = plugins_blob.filter(|blob| !blob.is_empty());
    let version = if plugins.is_some() {
        BRAIN_PACKAGE_FORMAT_VERSION_PLUGINS
    } else {
        BRAIN_PACKAGE_FORMAT_VERSION
    };
    writer
        .write_all(BRAIN_PACKAGE_MAGIC)
        .and_then(|_| writer.write_all(&version.to_le_bytes()))
        .and_then(|_| writer.write_all(&0u32.to_le_bytes()))
        .and_then(|_| writer.write_all(&0u32.to_le_bytes()))
        .map_err(|e| format!("brain package: header write failed: {}", e))?;

    let header_start = writer
        .stream_position()
        .map_err(|e| format!("brain package: position failed: {}", e))?;
    serde_json::to_writer(&mut *writer, header)
        .map_err(|e| format!("brain package header serialize failed: {}", e))?;
    let checkpoint_start = writer
        .stream_position()
        .map_err(|e| format!("brain package: position failed: {}", e))?;
    write_checkpoint(writer)?;
    let checkpoint_end = writer
        .stream_position()
        .map_err(|e| format!("brain package: position failed: {}", e))?;

    let header_len = checkpoint_start
        .checked_sub(header_start)
        .ok_or_else(|| "brain package: header length underflow".to_string())?;
    let checkpoint_len = checkpoint_end
        .checked_sub(checkpoint_start)
        .ok_or_else(|| "brain package: checkpoint length underflow".to_string())?;
    let header_len =
        u32::try_from(header_len).map_err(|_| "brain package: header too large".to_string())?;
    let checkpoint_len = u32::try_from(checkpoint_len)
        .map_err(|_| "brain package: checkpoint too large".to_string())?;

    let personality = personality.unwrap_or(&[]);
    let personality_len = u32::try_from(personality.len())
        .map_err(|_| "brain package: personality blob too large".to_string())?;
    writer
        .write_all(&personality_len.to_le_bytes())
        .and_then(|_| writer.write_all(personality))
        .map_err(|e| format!("brain package: personality write failed: {}", e))?;
    if let Some(plugins) = plugins {
        let plugin_len = u32::try_from(plugins.len())
            .map_err(|_| "brain package: plugins blob too large".to_string())?;
        writer
            .write_all(&plugin_len.to_le_bytes())
            .and_then(|_| writer.write_all(plugins))
            .map_err(|e| format!("brain package: plugins write failed: {}", e))?;
    }
    let end = writer
        .stream_position()
        .map_err(|e| format!("brain package: position failed: {}", e))?;
    let total = end
        .checked_sub(start)
        .ok_or_else(|| "brain package: output length underflow".to_string())?;
    if total > limits.max_decoded_bytes {
        return Err(format!(
            "brain package: output length {} exceeds configured {} byte limit",
            total, limits.max_decoded_bytes
        ));
    }

    writer
        .seek(SeekFrom::Start(start + 12))
        .and_then(|_| writer.write_all(&header_len.to_le_bytes()))
        .and_then(|_| writer.write_all(&checkpoint_len.to_le_bytes()))
        .and_then(|_| writer.seek(SeekFrom::Start(end)).map(|_| ()))
        .map_err(|e| format!("brain package: length backfill failed: {}", e))?;
    Ok(total)
}

/// Stream a gzip compression envelope to a seekable destination.
#[cfg(feature = "brain-compression")]
pub fn write_compressed_brain_seekable<R: Read, W: Write + Seek>(
    reader: R,
    original_len: u64,
    writer: &mut W,
    limits: BrainIoLimits,
) -> Result<u64, String> {
    if original_len > limits.max_decoded_bytes {
        return Err(format!(
            "compressed brain package: decoded length {} exceeds configured {} byte limit",
            original_len, limits.max_decoded_bytes
        ));
    }
    let start = writer
        .stream_position()
        .map_err(|e| format!("compressed brain package: position failed: {}", e))?;
    writer
        .write_all(COMPRESSED_BRAIN_MAGIC)
        .and_then(|_| writer.write_all(&COMPRESSED_BRAIN_FORMAT_VERSION.to_le_bytes()))
        .and_then(|_| writer.write_all(&[COMPRESSED_BRAIN_CODEC_GZIP, 0, 0, 0]))
        .and_then(|_| writer.write_all(&original_len.to_le_bytes()))
        .and_then(|_| writer.write_all(&0u64.to_le_bytes()))
        .map_err(|e| format!("compressed brain package: header write failed: {}", e))?;
    let payload_limit = limits
        .max_file_bytes
        .checked_sub(COMPRESSED_BRAIN_HEADER_LEN as u64)
        .ok_or_else(|| "compressed brain package: file limit is smaller than header".to_string())?;
    let payload_len = {
        let mut limited = LimitedWriter::new(
            &mut *writer,
            payload_limit,
            "compressed brain package payload",
        );
        {
            let mut encoder = GzEncoder::new(&mut limited, Compression::default());
            let mut reader = reader;
            io::copy(&mut reader, &mut encoder)
                .and_then(|_| encoder.try_finish())
                .map_err(|e| format!("compressed brain package: compression failed: {}", e))?;
        }
        limited.written()
    };
    let end = writer
        .stream_position()
        .map_err(|e| format!("compressed brain package: position failed: {}", e))?;
    writer
        .seek(SeekFrom::Start(start + 24))
        .and_then(|_| writer.write_all(&payload_len.to_le_bytes()))
        .and_then(|_| writer.seek(SeekFrom::Start(end)).map(|_| ()))
        .map_err(|e| format!("compressed brain package: length backfill failed: {}", e))?;
    Ok(COMPRESSED_BRAIN_HEADER_LEN as u64 + payload_len)
}

/// Run a write into a same-directory temporary file, sync it, then atomically rename it.
#[cfg(not(target_arch = "wasm32"))]
pub fn atomic_write_path<P, T, F>(path: P, write: F) -> Result<T, String>
where
    P: AsRef<std::path::Path>,
    F: FnOnce(&mut std::fs::File) -> Result<T, String>,
{
    let path = path.as_ref();
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| std::path::Path::new("."));
    std::fs::create_dir_all(parent)
        .map_err(|e| format!("create brain output directory failed: {}", e))?;
    let mut temp = tempfile::NamedTempFile::new_in(parent)
        .map_err(|e| format!("create brain temporary file failed: {}", e))?;
    let result = write(temp.as_file_mut())?;
    temp.as_file_mut()
        .flush()
        .map_err(|e| format!("flush brain temporary file failed: {}", e))?;
    temp.as_file()
        .sync_all()
        .map_err(|e| format!("sync brain temporary file failed: {}", e))?;
    temp.persist(path)
        .map_err(|e| format!("atomic brain rename failed: {}", e.error))?;
    Ok(result)
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
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder
        .write_all(data)
        .map_err(|e| format!("compressed brain package: compression failed: {}", e))?;
    let payload = encoder
        .finish()
        .map_err(|e| format!("compressed brain package: compression failed: {}", e))?;
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

/// If `data` is a brain package, return checkpoint + optional personality; otherwise treat
/// `data` as a legacy raw checkpoint (JSON only).
pub fn peel_brain_file_bytes(data: &[u8]) -> Result<BrainPackage, String> {
    let limits = BrainIoLimits {
        max_file_bytes: data.len() as u64,
        max_decoded_bytes: DEFAULT_BRAIN_IO_LIMIT_BYTES.max(data.len() as u64),
    };
    let parsed = parse_brain_reader(io::Cursor::new(data), limits, |reader| {
        let mut checkpoint = Vec::new();
        reader
            .read_to_end(&mut checkpoint)
            .map_err(|e| format!("brain checkpoint read failed: {}", e))?;
        Ok(checkpoint)
    })?;
    Ok(BrainPackage {
        header: parsed.header,
        checkpoint: parsed.checkpoint,
        personality: parsed.personality,
        plugins_blob: parsed.plugins_blob,
    })
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

    #[test]
    fn seekable_writer_is_byte_compatible_and_reader_roundtrips() {
        let mut header = BrainPackageHeader::default();
        header.id = "stream".to_string();
        let checkpoint = br#"{"streaming":true}"#;
        let personality = br#"{"O":0.7}"#;
        let plugins = b"[rules]\n";
        let expected =
            encode_brain_package(&header, checkpoint, Some(personality), Some(plugins)).unwrap();
        let mut streamed = io::Cursor::new(Vec::new());
        write_brain_package_seekable(
            &mut streamed,
            &header,
            Some(personality),
            Some(plugins),
            BrainIoLimits::default(),
            |writer| writer.write_all(checkpoint).map_err(|e| e.to_string()),
        )
        .unwrap();
        assert_eq!(streamed.into_inner(), expected);

        let parsed = parse_brain_reader(
            io::Cursor::new(expected),
            BrainIoLimits::default(),
            |reader| {
                let value: serde_json::Value =
                    serde_json::from_reader(reader).map_err(|e| e.to_string())?;
                Ok(value)
            },
        )
        .unwrap();
        assert_eq!(parsed.checkpoint["streaming"], true);
        assert_eq!(parsed.personality.as_deref(), Some(personality.as_slice()));
        assert_eq!(parsed.plugins_blob.as_deref(), Some(plugins.as_slice()));
    }

    #[test]
    fn reader_rejects_truncation_trailing_data_and_declared_limit() {
        let bytes = encode_brain_package(
            &BrainPackageHeader::default(),
            br#"{"ok":true}"#,
            None,
            None,
        )
        .unwrap();
        let read_checkpoint = |reader: &mut dyn Read| {
            let mut data = Vec::new();
            reader.read_to_end(&mut data).map_err(|e| e.to_string())?;
            Ok(data)
        };
        assert!(parse_brain_reader(
            io::Cursor::new(&bytes[..bytes.len() - 1]),
            BrainIoLimits::default(),
            read_checkpoint,
        )
        .is_err());
        let mut trailing = bytes.clone();
        trailing.push(0);
        assert!(parse_brain_reader(
            io::Cursor::new(trailing),
            BrainIoLimits::default(),
            read_checkpoint,
        )
        .unwrap_err()
        .contains("trailing"));
        assert!(parse_brain_reader(
            io::Cursor::new(bytes),
            BrainIoLimits {
                max_file_bytes: 1024,
                max_decoded_bytes: 24,
            },
            read_checkpoint,
        )
        .unwrap_err()
        .contains("declared length"));
    }

    #[cfg(feature = "brain-compression")]
    #[test]
    fn compressed_reader_stops_at_declared_output_limit() {
        let checkpoint = format!(r#"{{"padding":"{}"}}"#, "x".repeat(128 * 1024));
        let plain = encode_brain_package(
            &BrainPackageHeader::default(),
            checkpoint.as_bytes(),
            None,
            None,
        )
        .unwrap();
        let mut compressed = wrap_compressed_brain_bytes(&plain).unwrap();
        compressed[16..24].copy_from_slice(&64u64.to_le_bytes());
        let error = parse_brain_reader(
            io::Cursor::new(compressed),
            BrainIoLimits::default(),
            |reader| {
                let mut data = Vec::new();
                reader.read_to_end(&mut data).map_err(|e| e.to_string())?;
                Ok(data)
            },
        )
        .unwrap_err();
        assert!(error.contains("decompression failed") || error.contains("decoded length"));
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn atomic_write_replaces_target_and_cleans_temporary_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("brain.bin");
        std::fs::write(&path, b"old").unwrap();
        atomic_write_path(&path, |file| {
            file.write_all(b"new brain").map_err(|e| e.to_string())?;
            Ok(())
        })
        .unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"new brain");
        let failed = atomic_write_path(&path, |file| {
            file.write_all(b"partial").map_err(|e| e.to_string())?;
            Err::<(), _>("simulated failure".to_string())
        });
        assert!(failed.is_err());
        assert_eq!(std::fs::read(&path).unwrap(), b"new brain");
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
    }
}
