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

/// Loads brains for a [`crate::multi::Registry`], sharing one loaded open
/// model between every brain that names the same GGUF (and one
/// [`crate::multi::ModelPool`] between brains with identical prompt settings).
pub struct Loader {
    with_lm: bool,
    #[cfg_attr(not(feature = "candle"), allow(dead_code))]
    workers: usize,
    #[cfg_attr(not(feature = "candle"), allow(dead_code))]
    feature_cache: usize,
    #[cfg(feature = "candle")]
    models: std::sync::Mutex<std::collections::HashMap<String, crate::lm::CandleScorer>>,
    pools: std::sync::Mutex<
        std::collections::HashMap<String, std::sync::Arc<crate::multi::ModelPool>>,
    >,
}

impl Loader {
    /// `workers`: parallel model passes per pool (copies share the weights).
    pub fn new(with_lm: bool, workers: usize) -> Self {
        Self {
            with_lm,
            workers: workers.max(1),
            feature_cache: crate::multi::DEFAULT_FEATURE_CACHE,
            #[cfg(feature = "candle")]
            models: Default::default(),
            pools: Default::default(),
        }
    }

    /// Probe features kept per pool, in states (0 disables).
    pub fn feature_cache(mut self, states: usize) -> Self {
        self.feature_cache = states;
        self
    }

    /// Distinct model pools loaded so far.
    pub fn pool_count(&self) -> usize {
        self.pools.lock().map(|p| p.len()).unwrap_or(0)
    }

    /// Load a brain file; `name` defaults to the file stem (`crypto.s1.json.gz` → `crypto`).
    pub fn load(
        &self,
        name: Option<String>,
        path: &Path,
    ) -> Result<crate::multi::BrainEntry, String> {
        let brain = System1Brain::load(path)?;
        let name =
            name.unwrap_or_else(|| crate::multi::brain_name_from_path(&path.to_string_lossy()));
        let pool = match (&brain.lm, self.with_lm) {
            (Some(cfg), true) => Some(self.pool_for(cfg, path)?),
            _ => None,
        };
        Ok(crate::multi::BrainEntry {
            name,
            brain: std::sync::Arc::new(brain),
            pool,
            source: Some(path.to_string_lossy().into_owned()),
        })
    }

    /// Every `*.s1.json` / `*.s1.json.gz` in a folder, sorted by name.
    pub fn load_dir(&self, dir: &Path) -> Result<Vec<crate::multi::BrainEntry>, String> {
        let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
            .map_err(|e| format!("read_dir {}: {e}", dir.display()))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                let n = p.to_string_lossy();
                n.ends_with(".s1.json") || n.ends_with(".s1.json.gz")
            })
            .collect();
        files.sort();
        if files.is_empty() {
            return Err(format!("no *.s1.json(.gz) brains in {}", dir.display()));
        }
        files.iter().map(|f| self.load(None, f)).collect()
    }

    #[allow(unused_variables)]
    fn pool_for(
        &self,
        cfg: &LmConfig,
        brain_path: &Path,
    ) -> Result<std::sync::Arc<crate::multi::ModelPool>, String> {
        #[cfg(not(feature = "candle"))]
        {
            Err(format!(
                "{} names an open model but this binary was built without the `candle` feature (use --no-lm or rebuild with --features candle)",
                brain_path.display()
            ))
        }
        #[cfg(feature = "candle")]
        {
            use crate::decide::OptionScorer;
            use crate::multi::ModelPool;
            use std::sync::Arc;
            let base = brain_path.parent().unwrap_or(Path::new("."));
            let abs = |p: &str| {
                let pb = PathBuf::from(p);
                let pb = if pb.is_absolute() { pb } else { base.join(pb) };
                std::fs::canonicalize(&pb)
                    .unwrap_or(pb)
                    .to_string_lossy()
                    .into_owned()
            };
            let mut cfg = cfg.clone();
            cfg.model = abs(&cfg.model);
            cfg.tokenizer = abs(&cfg.tokenizer);
            let pool_key = serde_json::to_string(&cfg).map_err(|e| e.to_string())?;
            let mut pools = self.pools.lock().map_err(|_| "loader lock poisoned")?;
            if let Some(p) = pools.get(&pool_key) {
                return Ok(p.clone());
            }
            let weights_key = format!("{:?}|{}|{}", cfg.arch, cfg.model, cfg.tokenizer);
            let mut models = self.models.lock().map_err(|_| "loader lock poisoned")?;
            let scorer: Box<dyn OptionScorer + Send> = match models
                .get(&weights_key)
                .and_then(|m| m.fork_with(Some(cfg.clone())))
            {
                Some(s) => {
                    eprintln!(
                        "[gf-system1] sharing loaded {} with {}",
                        cfg.model,
                        brain_path.display()
                    );
                    Box::new(s)
                }
                None => {
                    let t0 = web_time::Instant::now();
                    let s = crate::lm::CandleScorer::from_files(cfg.clone(), Path::new("/"))?;
                    eprintln!(
                        "[gf-system1] open model: {} (loaded in {:.1}s)",
                        s.describe(),
                        t0.elapsed().as_secs_f32()
                    );
                    match s.fork_with(None) {
                        // Keep one copy to share with later brains.
                        Some(copy) => {
                            models.insert(weights_key, s);
                            Box::new(copy)
                        }
                        // Can't share (Qwen2): this pool owns it.
                        None => Box::new(s),
                    }
                }
            };
            let pool = Arc::new(
                ModelPool::new(scorer, self.workers).with_feature_cache(self.feature_cache),
            );
            pools.insert(pool_key, pool.clone());
            Ok(pool)
        }
    }
}

/// Append-only JSONL decision log: one line per answered request with every
/// answer's value, confidence, source and full probability distribution, plus
/// the action and latency. States are logged as a fingerprint unless
/// `include_state` (they may hold personal data).
pub struct JsonlLog {
    file: std::sync::Mutex<std::io::BufWriter<std::fs::File>>,
    include_state: bool,
}

impl JsonlLog {
    pub fn open(path: &Path, include_state: bool) -> Result<Self, String> {
        let f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|e| format!("open {}: {e}", path.display()))?;
        Ok(Self {
            file: std::sync::Mutex::new(std::io::BufWriter::new(f)),
            include_state,
        })
    }
}

impl crate::multi::DecisionSink for JsonlLog {
    fn record(&self, brain: &str, state: &serde_json::Value, r: &crate::decide::DecideResponse) {
        use std::io::Write;
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        let text = crate::state::state_to_text(state);
        let answers: serde_json::Map<String, serde_json::Value> = r
            .answers
            .iter()
            .map(|a| {
                let p: serde_json::Map<String, serde_json::Value> = a
                    .distribution
                    .iter()
                    .map(|o| (o.outcome.clone(), serde_json::json!(o.p)))
                    .collect();
                (
                    a.id.clone(),
                    serde_json::json!({
                        "value": a.value, "confidence": a.confidence, "abstain": a.abstain,
                        "source": a.source, "p": p, "error": a.error,
                    }),
                )
            })
            .collect();
        let mut line = serde_json::json!({
            "ts_ms": ts,
            "brain": brain,
            "state_fp": format!("{:016x}", crate::brain::row_fingerprint(&text)),
            "answers": answers,
            "action": r.action,
            "latency_ms": (r.latency_ms * 1000.0).round() / 1000.0,
        });
        if self.include_state {
            line["state"] = state.clone();
        }
        if let Ok(mut f) = self.file.lock() {
            let _ = writeln!(f, "{line}");
            let _ = f.flush();
        }
    }
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
    fn decisions_are_logged_with_probabilities_and_action() {
        use crate::multi::{BrainEntry, Registry};
        let (mut qs, rows) = crate::brain::tests::toy_rows();
        qs.default_action = Some("allow".into());
        let mut b = System1Brain::new("toy", qs).unwrap();
        b.train_heads(&rows, &crate::brain::TrainConfig::default())
            .unwrap();
        let reg = Registry::new();
        reg.insert(BrainEntry {
            name: "toy".into(),
            brain: std::sync::Arc::new(b),
            pool: None,
            source: None,
        });
        let p = std::env::temp_dir().join(format!("gf-s1-log-{}.jsonl", std::process::id()));
        let _ = std::fs::remove_file(&p);
        reg.set_sink(std::sync::Arc::new(JsonlLog::open(&p, false).unwrap()));
        reg.decide_json(r#"{"state":"card declined, my email is a@b.com"}"#)
            .unwrap();
        let line = std::fs::read_to_string(&p).unwrap();
        let v: serde_json::Value = serde_json::from_str(line.trim()).unwrap();
        assert_eq!(v["brain"], "toy");
        assert_eq!(v["action"]["action"], "allow");
        assert!(v["answers"]["route"]["p"]["Billing"].is_number());
        assert!(v.get("state").is_none(), "states are not logged by default");
        let _ = std::fs::remove_file(&p);
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
