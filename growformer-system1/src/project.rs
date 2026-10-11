//! Native helpers: load deciders from disk, read labelled data from files or
//! folders, and the `[system1]` section of a Growformer project manifest.

use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::brain::{LmConfig, System1Brain};
use crate::data::{parse_labeled_jsonl, LabeledRow};
use crate::decide::Decider;
use crate::schema::QuestionSet;

/// `[system1]` section of a `*.gf.toml` (paths relative to the manifest).
///
/// ```toml
/// [system1]
/// questions = "system1/questions.toml"
/// data = ["data"]                      # labelled JSONL files or folders
/// brain_output = "agent/system1.s1.json.gz"
///
/// [system1.lm]                         # optional open model
/// arch = "qwen3"
/// model = "models/qwen3-0.6b-q8_0.gguf"
/// tokenizer = "models/qwen3-tokenizer.json"
/// template = "chatml-nothink"
/// ```
#[derive(Clone, Debug, Deserialize)]
pub struct System1Section {
    pub questions: String,
    #[serde(default)]
    pub data: Vec<String>,
    pub brain_output: String,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub lm: Option<LmConfig>,
}

#[derive(Deserialize)]
struct ManifestDoc {
    #[serde(default)]
    system1: Option<System1Section>,
    #[serde(default)]
    project: Option<ProjectMeta>,
}

#[derive(Deserialize)]
struct ProjectMeta {
    name: Option<String>,
}

/// Resolved project settings (absolute paths).
#[derive(Clone, Debug)]
pub struct System1Project {
    pub name: String,
    pub questions: PathBuf,
    pub data: Vec<PathBuf>,
    pub brain_output: PathBuf,
    pub lm: Option<LmConfig>,
    pub base: PathBuf,
}

pub fn read_project(manifest: &Path) -> Result<System1Project, String> {
    let raw = std::fs::read_to_string(manifest)
        .map_err(|e| format!("read {}: {e}", manifest.display()))?;
    let doc: ManifestDoc =
        toml::from_str(&raw).map_err(|e| format!("parse {}: {e}", manifest.display()))?;
    let s = doc
        .system1
        .ok_or_else(|| format!("{} has no [system1] section", manifest.display()))?;
    let base = manifest
        .parent()
        .map(|p| {
            if p.as_os_str().is_empty() {
                Path::new(".")
            } else {
                p
            }
        })
        .unwrap_or(Path::new("."))
        .to_path_buf();
    let lm = s.lm.map(|mut c| {
        c.model = base.join(&c.model).to_string_lossy().into_owned();
        c.tokenizer = base.join(&c.tokenizer).to_string_lossy().into_owned();
        c
    });
    Ok(System1Project {
        name: s
            .name
            .or(doc.project.and_then(|p| p.name))
            .unwrap_or_else(|| "system1".into()),
        questions: base.join(&s.questions),
        data: s.data.iter().map(|d| base.join(d)).collect(),
        brain_output: base.join(&s.brain_output),
        lm,
        base,
    })
}

pub fn read_questions(path: &Path) -> Result<QuestionSet, String> {
    let raw = std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    QuestionSet::parse(&raw).map_err(|e| format!("{}: {e}", path.display()))
}

fn skipped_name(name: &str) -> bool {
    let l = name.to_ascii_lowercase();
    l.contains("guardrails") || l.starts_with("inference_") || l.contains("_fragments")
}

/// JSONL files for a path: a file as-is; a folder's `*.jsonl` (minus
/// guardrail / inference / fragment files), sorted.
pub fn jsonl_files(path: &Path) -> Result<Vec<PathBuf>, String> {
    if path.is_file() {
        return Ok(vec![path.to_path_buf()]);
    }
    let mut v: Vec<PathBuf> = std::fs::read_dir(path)
        .map_err(|e| format!("read_dir {}: {e}", path.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension().map(|x| x == "jsonl").unwrap_or(false)
                && !p
                    .file_name()
                    .and_then(|n| n.to_str())
                    .map(skipped_name)
                    .unwrap_or(true)
        })
        .collect();
    v.sort();
    Ok(v)
}

/// Read labelled rows from files/folders. Returns `(rows, skipped_lines)`.
pub fn read_rows(paths: &[PathBuf], qs: &QuestionSet) -> Result<(Vec<LabeledRow>, usize), String> {
    let mut rows = Vec::new();
    let mut skipped = 0;
    for p in paths {
        for f in jsonl_files(p)? {
            let raw =
                std::fs::read_to_string(&f).map_err(|e| format!("read {}: {e}", f.display()))?;
            let (r, s) =
                parse_labeled_jsonl(&raw, qs).map_err(|e| format!("{}: {e}", f.display()))?;
            rows.extend(r);
            skipped += s;
        }
    }
    Ok((rows, skipped))
}

/// `path` relative to `base` when it is underneath it, else absolute.
pub fn relative_to(path: &Path, base: &Path) -> String {
    let abs = |p: &Path| std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf());
    let (p, b) = (abs(path), abs(base));
    match p.strip_prefix(&b) {
        Ok(rel) => rel.to_string_lossy().into_owned(),
        Err(_) => p.to_string_lossy().into_owned(),
    }
}

/// Load a brain and, when `with_lm` and the brain names an open model (and
/// the crate was built with `candle`), attach it. Model paths in the brain
/// are relative to the brain file.
pub fn load_decider(brain_path: &Path, with_lm: bool) -> Result<Decider, String> {
    let brain = System1Brain::load(brain_path)?;
    #[allow(unused_mut)]
    let mut d = Decider::new(brain);
    if with_lm {
        if let Some(cfg) = d.brain.lm.clone() {
            #[cfg(feature = "candle")]
            {
                use crate::decide::OptionScorer;
                let base = brain_path.parent().unwrap_or(Path::new("."));
                let t0 = web_time::Instant::now();
                let s = crate::lm::CandleScorer::from_files(cfg, base)?;
                eprintln!(
                    "[gf-system1] open model: {} (loaded in {:.1}s)",
                    s.describe(),
                    t0.elapsed().as_secs_f32()
                );
                d.set_scorer(Box::new(s));
            }
            #[cfg(not(feature = "candle"))]
            {
                let _ = cfg;
                return Err("this brain names an open model but the binary was built without the `candle` feature (use --no-lm or rebuild with --features candle)".into());
            }
        }
    }
    Ok(d)
}

/// On-disk cache of probe features, keyed by state fingerprint. A cache made
/// with a different model/layer (`key`) is ignored. Format: `S1EMB1`, key
/// length (u32 LE) + key bytes, dim (u32 LE), then `(u64 fingerprint, dim × f32)`
/// records, all little-endian.
pub struct EmbCache {
    key: String,
    dim: usize,
    map: std::collections::HashMap<u64, Vec<f32>>,
    dirty: bool,
}

const EMB_MAGIC: &[u8; 6] = b"S1EMB1";

impl EmbCache {
    pub fn load(path: &Path, key: &str) -> Self {
        let empty = Self {
            key: key.to_string(),
            dim: 0,
            map: Default::default(),
            dirty: false,
        };
        let Ok(b) = std::fs::read(path) else {
            return empty;
        };
        let u32_at = |i: usize| -> Option<usize> {
            Some(u32::from_le_bytes(b.get(i..i + 4)?.try_into().ok()?) as usize)
        };
        if b.len() < 10 || &b[..6] != EMB_MAGIC {
            return empty;
        }
        let Some(kl) = u32_at(6) else { return empty };
        if b.get(10..10 + kl) != Some(key.as_bytes()) {
            return empty;
        }
        let Some(dim) = u32_at(10 + kl) else {
            return empty;
        };
        let mut i = 14 + kl;
        let rec = 8 + 4 * dim;
        let mut map = std::collections::HashMap::new();
        while dim > 0 && i + rec <= b.len() {
            let fp = u64::from_le_bytes(b[i..i + 8].try_into().unwrap());
            let v = b[i + 8..i + rec]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            map.insert(fp, v);
            i += rec;
        }
        Self {
            key: key.to_string(),
            dim,
            map,
            dirty: false,
        }
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    pub fn get(&self, text: &str) -> Option<&Vec<f32>> {
        self.map.get(&crate::brain::row_fingerprint(text))
    }

    pub fn insert(&mut self, text: &str, v: Vec<f32>) {
        if self.dim == 0 {
            self.dim = v.len();
        }
        if v.len() == self.dim {
            self.map.insert(crate::brain::row_fingerprint(text), v);
            self.dirty = true;
        }
    }

    pub fn save(&mut self, path: &Path) -> Result<(), String> {
        if !self.dirty {
            return Ok(());
        }
        let mut b = Vec::with_capacity(14 + self.key.len() + self.map.len() * (8 + 4 * self.dim));
        b.extend_from_slice(EMB_MAGIC);
        b.extend_from_slice(&(self.key.len() as u32).to_le_bytes());
        b.extend_from_slice(self.key.as_bytes());
        b.extend_from_slice(&(self.dim as u32).to_le_bytes());
        for (fp, v) in &self.map {
            b.extend_from_slice(&fp.to_le_bytes());
            for x in v {
                b.extend_from_slice(&x.to_le_bytes());
            }
        }
        std::fs::write(path, b).map_err(|e| format!("write {}: {e}", path.display()))?;
        self.dirty = false;
        Ok(())
    }
}

/// Per-question outcome counts for labelled rows (`gf-system1 stats`).
pub fn label_counts(rows: &[LabeledRow], qs: &QuestionSet) -> Vec<(String, Vec<(String, usize)>)> {
    qs.questions
        .iter()
        .enumerate()
        .map(|(qi, q)| {
            let outcomes = q.outcomes();
            let mut c = vec![0usize; outcomes.len()];
            for r in rows {
                for (i, _, g) in &r.targets {
                    if *i == qi {
                        c[*g] += 1;
                    }
                }
            }
            (q.id.clone(), outcomes.into_iter().zip(c).collect())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_section_resolves_paths_and_trains() {
        let dir = std::env::temp_dir().join(format!("gf-s1-proj-{}", std::process::id()));
        std::fs::create_dir_all(dir.join("data")).unwrap();
        std::fs::write(
            dir.join("p.gf.toml"),
            r#"
[project]
name = "Demo"

[train]
data_dir = "data"

[system1]
questions = "q.toml"
data = ["data"]
brain_output = "agent/demo.s1.json.gz"
"#,
        )
        .unwrap();
        std::fs::write(
            dir.join("q.toml"),
            "[[questions]]\nid = \"pos\"\ntext = \"Positive?\"\ntype = \"noul\"\nsource = { field = \"semantic_intent\", map = { good = true, bad = false } }\n",
        )
        .unwrap();
        let mut lines = String::new();
        for i in 0..40 {
            let (t, l) = if i % 2 == 0 {
                ("great gains today", "good")
            } else {
                ("terrible losses today", "bad")
            };
            lines.push_str(&format!(
                "{{\"text\":\"{t} {i}\",\"semantic_intent\":\"{l}\"}}\n"
            ));
        }
        std::fs::write(dir.join("data/train_demo.jsonl"), lines).unwrap();
        std::fs::write(
            dir.join("data/inference_guardrails.jsonl"),
            "{\"oops\":1}\n",
        )
        .unwrap();

        let p = read_project(&dir.join("p.gf.toml")).unwrap();
        assert_eq!(p.name, "Demo");
        let qs = read_questions(&p.questions).unwrap();
        let (rows, _) = read_rows(&p.data, &qs).unwrap();
        assert_eq!(rows.len(), 40, "guardrail file must be skipped");
        let mut b = System1Brain::new(p.name.clone(), qs).unwrap();
        b.train_heads(&rows, &crate::brain::TrainConfig::default())
            .unwrap();
        b.save(&p.brain_output).unwrap();
        let mut d = load_decider(&p.brain_output, true).unwrap();
        let out = d.decide_json(r#"{"state":"great gains"}"#).unwrap();
        assert!(out.contains("\"value\":true"), "{out}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn emb_cache_round_trips_and_checks_key() {
        let p = std::env::temp_dir().join(format!("gf-s1-emb-{}", std::process::id()));
        let mut c = EmbCache::load(&p, "m:a");
        assert!(c.is_empty());
        c.insert("Hello  world", vec![1.0, 2.0, 3.0]);
        c.insert("other", vec![4.0, 5.0, 6.0]);
        c.save(&p).unwrap();
        let c2 = EmbCache::load(&p, "m:a");
        assert_eq!(c2.len(), 2);
        assert_eq!(c2.get("hello world"), Some(&vec![1.0, 2.0, 3.0]));
        assert!(EmbCache::load(&p, "m:b").is_empty());
        let _ = std::fs::remove_file(&p);
    }
}
