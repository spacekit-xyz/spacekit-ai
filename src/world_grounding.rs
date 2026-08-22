//! Load world-grounding TOML graphs (same schema as Growformer `WorldGroundingFile`)
//! and turn token hits into a fixed-size feature vector for classifier conditioning.

use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Deserialize)]
struct WorldGroundingFile {
    #[allow(dead_code)]
    version: u32,
    #[serde(default)]
    nodes: Vec<NodeToml>,
}

#[derive(Debug, Deserialize)]
struct NodeToml {
    id: String,
    #[serde(default)]
    aliases: Vec<String>,
}

fn normalize_key(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .flat_map(|c| c.to_lowercase())
        .collect()
}

/// Bucketed multi-hot features from grounding vocabulary hits (reference-style “expansion”).
#[derive(Debug, Clone)]
pub struct GroundingExtractor {
    pub dim: usize,
    /// token -> first matching canonical node id (for debugging)
    lookup: HashMap<String, String>,
}

impl GroundingExtractor {
    pub fn from_toml_files(paths: &[impl AsRef<Path>]) -> Result<Self, String> {
        let mut lookup: HashMap<String, String> = HashMap::new();

        for p in paths {
            let path = p.as_ref();
            let raw = std::fs::read_to_string(path)
                .map_err(|e| format!("read {}: {}", path.display(), e))?;
            let file: WorldGroundingFile =
                toml::from_str(&raw).map_err(|e| format!("parse {}: {}", path.display(), e))?;
            if file.version != 1 {
                return Err(format!(
                    "{}: unsupported version {}",
                    path.display(),
                    file.version
                ));
            }
            for n in file.nodes {
                let id_norm = normalize_key(&n.id);
                if id_norm.is_empty() {
                    continue;
                }
                let canon = n.id.clone();
                lookup
                    .entry(id_norm.clone())
                    .or_insert_with(|| canon.clone());
                for a in n.aliases {
                    let an = normalize_key(&a);
                    if !an.is_empty() {
                        lookup.entry(an).or_insert_with(|| canon.clone());
                    }
                }
            }
        }

        Ok(Self {
            dim: GROUND_FEATURE_DIM,
            lookup,
        })
    }

    /// Lowercase word tokens (alphanumeric runs); matches Growformer-style lexical checks.
    pub fn tokenize(text: &str) -> Vec<String> {
        let lower: String = text.to_lowercase();
        let mut out = Vec::new();
        let mut cur = String::new();
        for ch in lower.chars() {
            if ch.is_ascii_alphanumeric() {
                cur.push(ch);
            } else if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
        }
        if !cur.is_empty() {
            out.push(cur);
        }
        out
    }

    /// L2-normalized bucket counts from activated grounding keys.
    pub fn features(&self, text: &str) -> Vec<f32> {
        let mut acc = vec![0.0f32; self.dim];
        for tok in Self::tokenize(text) {
            let n = normalize_key(&tok);
            if n.len() < 2 {
                continue;
            }
            if let Some(canon) = self.lookup.get(&n) {
                let h = fnv1a_bytes(canon.as_bytes()) % (self.dim as u64);
                acc[h as usize] += 1.0;
            }
        }
        let norm: f32 = acc.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-8 {
            for x in &mut acc {
                *x /= norm;
            }
        }
        acc
    }

    pub fn zero_features(dim: usize) -> Vec<f32> {
        vec![0.0f32; dim]
    }
}

/// Bucket dimension for grounding features (`features()` output length).
pub const GROUND_FEATURE_DIM: usize = 64;

fn fnv1a_bytes(bytes: &[u8]) -> u64 {
    const OFFSET: u64 = 14695981039346656037;
    const PRIME: u64 = 1099511628211;
    let mut h = OFFSET;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(PRIME);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn loads_sample_grounding() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let base = root.join("data/inference/world_grounding.toml");
        let fin = root.join("data/fintech/world_grounding_fintech.toml");
        let g = GroundingExtractor::from_toml_files(&[&base, &fin]).expect("parse");
        let f = g.features("The ETF rallied after earnings beat expectations.");
        assert!(f.iter().any(|&x| x > 0.0));
    }
}
