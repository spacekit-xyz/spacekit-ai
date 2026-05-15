// ── training.rs ───────────────────────────────────────────────────────────────
// Revised minimal JSONL schema for Growformer.
//
// KEY CHANGE from prior version:
//   REMOVED: "entity_category": "day_of_week"  (hard symbolic label)
//   KEPT:    "sentiment"                         (target signal)
//   KEPT:    "plural"                            (morphological hint)
//   ADDED:   "aux_category" (Optional)           (WEAK supervision only,
//                                                 used in Stage 1, dropped
//                                                 in Stages 2 and 3)
//
// The model discovers that mondays/tuesdays/rain/meetings are structurally
// equivalent ITSELF via region formation — not by being told.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::Path;

// ── SentimentLabel ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SentimentLabel {
    PositiveStrong,
    PositiveMild,
    Neutral,
    NegativeMild,
    NegativeStrong,
    /// Ironic / surface-mismatch valence (matches `semantic_intent: "sarcastic"` in corpus).
    Sarcastic,
    /// Conflicting poles in one utterance (`semantic_intent: "mixed"`).
    Mixed,
}

impl SentimentLabel {
    pub fn score(&self) -> f32 {
        match self {
            Self::PositiveStrong =>  1.0,
            Self::PositiveMild   =>  0.5,
            Self::Neutral        =>  0.0,
            Self::NegativeMild   => -0.5,
            Self::NegativeStrong => -1.0,
            Self::Sarcastic      => -0.35,
            Self::Mixed          =>  0.08,
        }
    }

    pub fn one_hot(&self) -> Vec<f32> {
        let idx = self.class_index();
        let mut v = vec![0.0f32; Self::num_classes()];
        v[idx] = 1.0;
        v
    }

    pub fn num_classes() -> usize {
        7
    }

    /// Index matching `one_hot` / linear head layout (stable ordering).
    pub fn class_index(&self) -> usize {
        match self {
            Self::NegativeStrong => 0,
            Self::NegativeMild   => 1,
            Self::Neutral        => 2,
            Self::PositiveMild   => 3,
            Self::PositiveStrong => 4,
            Self::Sarcastic      => 5,
            Self::Mixed          => 6,
        }
    }

    pub fn from_class_index(i: usize) -> Option<Self> {
        match i {
            0 => Some(Self::NegativeStrong),
            1 => Some(Self::NegativeMild),
            2 => Some(Self::Neutral),
            3 => Some(Self::PositiveMild),
            4 => Some(Self::PositiveStrong),
            5 => Some(Self::Sarcastic),
            6 => Some(Self::Mixed),
            _ => None,
        }
    }
}

// ── AuxCategory ───────────────────────────────────────────────────────────────

/// WEAK supervision hint — Stage 1 only (λ=0.3), dropped in Stages 2 and 3.
/// Never treated as ground truth. Seeds region formation, nothing more.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuxCategory {
    Temporal,
    Weather,
    Event,
    Person,
    Place,
    Other,
}

impl AuxCategory {
    pub fn one_hot(&self) -> Vec<f32> {
        let idx = match self {
            Self::Temporal => 0,
            Self::Weather  => 1,
            Self::Event    => 2,
            Self::Person   => 3,
            Self::Place    => 4,
            Self::Other    => 5,
        };
        let mut v = vec![0.0f32; 6];
        v[idx] = 1.0;
        v
    }

    pub fn num_classes() -> usize { 6 }

    pub fn class_index(&self) -> usize {
        match self {
            Self::Temporal => 0,
            Self::Weather  => 1,
            Self::Event    => 2,
            Self::Person   => 3,
            Self::Place    => 4,
            Self::Other    => 5,
        }
    }

    pub fn from_class_index(i: usize) -> Option<Self> {
        match i {
            0 => Some(Self::Temporal),
            1 => Some(Self::Weather),
            2 => Some(Self::Event),
            3 => Some(Self::Person),
            4 => Some(Self::Place),
            5 => Some(Self::Other),
            _ => None,
        }
    }

    pub fn infer(entity: &str) -> Self {
        let e = entity.to_lowercase();
        let e = e.trim_end_matches('s');
        match e {
            "monday"|"tuesday"|"wednesday"|"thursday"|
            "friday"|"saturday"|"sunday"|
            "morning"|"evening"|"night"|"week"|"month" => Self::Temporal,
            "rain"|"snow"|"sun"|"cloud"|"storm"|"wind" => Self::Weather,
            "meeting"|"standup"|"call"|"presentation"|
            "workshop"|"sprint"                         => Self::Event,
            _ => Self::Other,
        }
    }

    /// Weak aux label using full sentence + tail token (reduces `Other` skew on `data/sentiment`).
    pub fn infer_from_context(full_text: &str, entity_tail: &str) -> Self {
        let t = full_text.to_lowercase();
        if t.contains("bitcoin")
            || t.contains("crypto")
            || t.contains("portfolio")
            || t.contains("stock")
            || t.contains("earnings")
            || t.contains("dividend")
            || t.contains("ipo")
            || t.contains("fed ")
            || t.contains("interest rate")
            || t.contains("mortgage")
            || t.contains("loan")
            || t.contains("salary")
            || t.contains("invoice")
        {
            return Self::Event;
        }
        if t.contains("restaurant")
            || t.contains("recipe")
            || t.contains("flight")
            || t.contains("hotel")
            || t.contains("vacation")
            || t.contains("beach")
            || t.contains("coffee")
        {
            return Self::Place;
        }
        if t.contains("therapist")
            || t.contains("boyfriend")
            || t.contains("girlfriend")
            || t.contains("spouse")
            || t.contains("mother")
            || t.contains("father")
            || t.contains("friend")
            || t.contains("family")
        {
            return Self::Person;
        }
        if t.contains("movie")
            || t.contains("album")
            || t.contains("concert")
            || t.contains("netflix")
            || t.contains("episode")
            || t.contains("sequel")
        {
            return Self::Event;
        }
        if t.contains("patch")
            || t.contains("respawn")
            || t.contains("steam")
            || t.contains("multiplayer")
            || t.contains("raid")
            || t.contains("dlc")
            || t.contains("fps")
        {
            return Self::Event;
        }
        Self::infer(entity_tail)
    }
}

// ── growformer/data/sentiment JSONL compatibility ─────────────────────────────

/// Map dataset `semantic_intent` strings (see `data/sentiment/*.jsonl`) onto [`SentimentLabel`]
/// (seven classes, including `sarcastic` and `mixed`).
pub fn semantic_intent_to_label(semantic_intent: &str) -> Result<SentimentLabel, String> {
    match semantic_intent.trim() {
        "positive_strong" => Ok(SentimentLabel::PositiveStrong),
        "positive_mild" => Ok(SentimentLabel::PositiveMild),
        "neutral" => Ok(SentimentLabel::Neutral),
        "negative_mild" => Ok(SentimentLabel::NegativeMild),
        "negative_strong" => Ok(SentimentLabel::NegativeStrong),
        "sarcastic" => Ok(SentimentLabel::Sarcastic),
        "mixed" => Ok(SentimentLabel::Mixed),
        other => Err(format!("unknown semantic_intent {:?}", other)),
    }
}

/// Heuristic plural hint: last alphanumeric token ends with `s` (length ≥ 3), not `ss`.
/// Many false positives (`this`, `glass`); combine with JSON `plural` via
/// [`TrainingBatch::reinforce_plural_with_heuristic`].
pub fn infer_plural_from_text(input: &str) -> bool {
    let t = input
        .split_whitespace()
        .last()
        .unwrap_or("")
        .trim_matches(|c: char| !c.is_alphanumeric());
    if t.len() < 3 {
        return false;
    }
    let lower = t.to_lowercase();
    if lower.ends_with("ss") {
        return false;
    }
    static FALSE_POS: &[&str] = &[
        "this", "thus", "yes", "bus", "us", "news", "his", "was", "has", "its", "ours", "hers",
    ];
    if FALSE_POS.contains(&lower.as_str()) {
        return false;
    }
    lower.ends_with('s')
}

#[derive(Deserialize)]
struct SentimentFileRow {
    text: String,
    semantic_intent: String,
    #[serde(default)]
    plural: bool,
    #[serde(default)]
    embedding: Option<Vec<f32>>,
}

/// Which `.jsonl` files to merge from a directory such as `data/sentiment`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SentimentJsonlSelection {
    /// Sample corpora: every `*.jsonl` except `inference_guardrails.jsonl` and `eval_*.jsonl`
    /// (same rule as `--data-dir` / brain training).
    TrainFilesOnly,
    /// Every `*.jsonl` except `inference_guardrails.jsonl` (includes `eval_*` holdouts — use only when intentional).
    AllJsonl,
}

// ── TrainingRecord ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingRecord {
    #[serde(alias = "text")]
    pub input: String,
    #[serde(alias = "semantic_intent")]
    pub sentiment: SentimentLabel,
    #[serde(default)]
    pub plural: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aux_category: Option<AuxCategory>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f32>>,
    /// Optional causal annotation for temporal ordering training (Cl(1,7) causal block).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub causal: Option<crate::dimension::language::CausalAnnotation>,
}

impl TrainingRecord {
    pub fn new(input: impl Into<String>, sentiment: SentimentLabel, plural: bool) -> Self {
        Self { input: input.into(), sentiment, plural, aux_category: None, embedding: None, causal: None }
    }

    pub fn with_aux(mut self, cat: AuxCategory) -> Self {
        self.aux_category = Some(cat);
        self
    }

    pub fn with_embedding(mut self, emb: Vec<f32>) -> Self {
        self.embedding = Some(emb);
        self
    }

    pub fn resolved_aux_category(&self) -> AuxCategory {
        self.aux_category.clone().unwrap_or_else(|| {
            let raw = self.input.split_whitespace().last().unwrap_or("");
            let entity = raw.trim_matches(|c: char| !c.is_alphanumeric());
            AuxCategory::infer_from_context(&self.input, entity)
        })
    }

    pub fn to_jsonl(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }
}

// ── TrainingBatch ─────────────────────────────────────────────────────────────

#[derive(Debug, Default)]
pub struct TrainingBatch {
    pub records: Vec<TrainingRecord>,
}

impl TrainingBatch {
    pub fn new() -> Self { Self::default() }
    pub fn push(&mut self, r: TrainingRecord) { self.records.push(r); }
    pub fn len(&self) -> usize { self.records.len() }
    pub fn is_empty(&self) -> bool { self.records.is_empty() }

    pub fn from_jsonl<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let reader = BufReader::new(File::open(path)?);
        let mut batch = Self::new();
        for (i, line) in reader.lines().enumerate() {
            let line = line?;
            let line = line.trim();
            if line.is_empty() || line.starts_with("//") { continue; }
            let record: TrainingRecord = serde_json::from_str(line)
                .map_err(|e| format!("Line {}: {}", i + 1, e))?;
            batch.push(record);
        }
        Ok(batch)
    }

    /// Load `growformer/data/sentiment`-style JSONL: `text` + `semantic_intent` (+ optional `plural`, `embedding`).
    /// Other keys are ignored by serde.
    pub fn append_from_sentiment_jsonl<P: AsRef<Path>>(
        &mut self,
        path: P,
    ) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
        let path = path.as_ref();
        let reader = BufReader::new(File::open(path)?);
        let mut n = 0usize;
        for (i, line) in reader.lines().enumerate() {
            let line = line.map_err(|e| format!("{} line {}: {}", path.display(), i + 1, e))?;
            let line = line.trim();
            if line.is_empty() || line.starts_with("//") {
                continue;
            }
            let row: SentimentFileRow = serde_json::from_str(line).map_err(|e| {
                format!("{} line {}: {}", path.display(), i + 1, e)
            })?;
            let sentiment = semantic_intent_to_label(&row.semantic_intent).map_err(|e| {
                format!("{} line {}: {}", path.display(), i + 1, e)
            })?;
            let mut rec = TrainingRecord::new(row.text, sentiment, row.plural);
            if let Some(emb) = row.embedding.filter(|e| !e.is_empty()) {
                rec = rec.with_embedding(emb);
            }
            self.push(rec);
            n += 1;
        }
        Ok(n)
    }

    /// Set `embedding` using [`crate::category::embedding::SentenceEmbedder`] when missing or empty.
    pub fn fill_missing_embeddings<E: crate::category::embedding::SentenceEmbedder>(
        &mut self,
        embedder: &E,
    ) {
        for r in &mut self.records {
            let need = r.embedding.as_ref().map_or(true, |e| e.is_empty());
            if need {
                r.embedding = Some(embedder.embed(&r.input));
            }
        }
    }

    /// `plural |= infer_plural_from_text` so JSON `false` can become true when the tail token looks plural.
    pub fn reinforce_plural_with_heuristic(&mut self) {
        for r in &mut self.records {
            r.plural = r.plural || infer_plural_from_text(&r.input);
        }
    }

    /// Same as [`Self::append_from_sentiment_jsonl`] but returns a fresh batch.
    pub fn from_sentiment_jsonl<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let mut b = TrainingBatch::new();
        b.append_from_sentiment_jsonl(path)?;
        Ok(b)
    }

    /// Merge multiple `*.jsonl` files from a directory (e.g. `data/sentiment`).
    pub fn from_sentiment_jsonl_dir<P: AsRef<Path>>(
        dir: P,
        selection: SentimentJsonlSelection,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let dir = dir.as_ref();
        let mut paths: Vec<_> = fs::read_dir(dir)
            .map_err(|e| format!("read_dir {}: {}", dir.display(), e))?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "jsonl"))
            .collect();
        paths.sort();
        let mut batch = TrainingBatch::new();
        for p in paths {
            let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
            let include = match selection {
                SentimentJsonlSelection::TrainFilesOnly => {
                    crate::dimension::language::is_brain_training_jsonl_filename(name)
                }
                SentimentJsonlSelection::AllJsonl => {
                    name.ends_with(".jsonl")
                        && !crate::dimension::language::is_inference_guardrails_jsonl_filename(name)
                }
            };
            if include {
                batch.append_from_sentiment_jsonl(&p)?;
            }
        }
        Ok(batch)
    }

    pub fn to_jsonl<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        use std::io::Write;
        let mut file = File::create(path)?;
        for r in &self.records { writeln!(file, "{}", r.to_jsonl()?)?; }
        Ok(())
    }

    pub fn by_inferred_category(&self) -> HashMap<String, Vec<&TrainingRecord>> {
        let mut map: HashMap<String, Vec<&TrainingRecord>> = HashMap::new();
        for r in &self.records {
            map.entry(format!("{:?}", r.resolved_aux_category())).or_default().push(r);
        }
        map
    }

    pub fn by_sentiment(&self) -> HashMap<String, Vec<&TrainingRecord>> {
        let mut map: HashMap<String, Vec<&TrainingRecord>> = HashMap::new();
        for r in &self.records {
            map.entry(format!("{:?}", r.sentiment)).or_default().push(r);
        }
        map
    }

    pub fn validate_coverage(&self, min_count: usize) -> Vec<String> {
        self.by_inferred_category()
            .iter()
            .filter(|(_, recs)| recs.len() < min_count)
            .map(|(cat, recs)| format!(
                "Sparse: inferred category '{}' has {} example(s) (min: {})",
                cat, recs.len(), min_count
            ))
            .collect()
    }

    pub fn coverage_report(&self) -> String {
        let by_cat  = self.by_inferred_category();
        let by_sent = self.by_sentiment();
        let mut lines = vec![format!("Training batch: {} records\n", self.len())];
        lines.push("Inferred category distribution (diagnostic, not supervision):".to_string());
        for (cat, recs) in &by_cat {
            lines.push(format!("  {:20} {} examples", cat, recs.len()));
        }
        lines.push("\nSentiment distribution:".to_string());
        for (sent, recs) in &by_sent {
            lines.push(format!("  {:20} {} examples", sent, recs.len()));
        }
        lines.join("\n")
    }

    pub fn sentiment_labels(&self) -> Vec<&SentimentLabel> {
        self.records.iter().map(|r| &r.sentiment).collect()
    }
}

// ── combined_loss ─────────────────────────────────────────────────────────────

pub fn combined_loss(
    task_loss: f32,
    sentiment_emb: &[f32],
    entity_emb: &[f32],
    lambda: f32,
) -> f32 {
    let dot: f32 = sentiment_emb.iter().zip(entity_emb.iter()).map(|(a, b)| a * b).sum();
    let ns = sentiment_emb.iter().map(|x| x * x).sum::<f32>().sqrt();
    let ne = entity_emb.iter().map(|x| x * x).sum::<f32>().sqrt();
    let dis = if ns == 0.0 || ne == 0.0 { 0.0 } else { (dot / (ns * ne)).abs() };
    task_loss + lambda * dis
}

// ── Example training batch ────────────────────────────────────────────────────

pub fn example_training_batch() -> TrainingBatch {
    use SentimentLabel::*;
    use AuxCategory::*;
    let mut b = TrainingBatch::new();

    // Days — no aux hints; model must discover temporal clustering itself
    for (day, neg, pos) in [
        ("mondays",    NegativeMild,   PositiveMild),
        ("tuesdays",   NegativeMild,   PositiveMild),
        ("wednesdays", Neutral,        PositiveMild),
        ("thursdays",  Neutral,        PositiveMild),
        ("fridays",    PositiveMild,   PositiveStrong),
        ("saturdays",  PositiveStrong, PositiveStrong),
        ("sundays",    PositiveMild,   PositiveStrong),
    ] {
        b.push(TrainingRecord::new(format!("I hate {}", day),          neg,           true));
        b.push(TrainingRecord::new(format!("I love {}", day),          pos,           true));
        b.push(TrainingRecord::new(format!("I really hate {}", day),   NegativeStrong,true));
        b.push(TrainingRecord::new(format!("{} are the worst", day),   NegativeStrong,true));
    }

    // Weather — weak aux hints present on some records to seed Stage 1
    for weather in ["rain", "snow", "storms", "sun"] {
        let plural = weather == "storms";
        b.push(TrainingRecord::new(format!("I hate {}", weather),  NegativeMild, plural).with_aux(Weather));
        b.push(TrainingRecord::new(format!("I love {}", weather),  PositiveMild, plural).with_aux(Weather));
    }

    // Events — no aux hints
    for event in ["meetings", "standups", "presentations", "workshops"] {
        b.push(TrainingRecord::new(format!("I hate {}", event),        NegativeMild,  true));
        b.push(TrainingRecord::new(format!("I really hate {}", event), NegativeStrong,true));
    }

    // Cross-category: same template, diverse entities — forces entity-agnostic morphism
    for (entity, sentiment, plural) in [
        ("traffic",   NegativeStrong, false),
        ("deadlines", NegativeStrong, true),
        ("coffee",    PositiveStrong, false),
        ("weekends",  PositiveStrong, true),
        ("commutes",  NegativeMild,   true),
        ("holidays",  PositiveStrong, true),
    ] {
        b.push(TrainingRecord::new(format!("I hate {}", entity), NegativeMild, plural));
        b.push(TrainingRecord::new(format!("I love {}", entity), sentiment,    plural));
    }

    b
}

#[cfg(test)]
mod sentiment_loader_tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn semantic_intent_maps_sarcastic_and_mixed() {
        assert_eq!(
            semantic_intent_to_label("sarcastic").unwrap(),
            SentimentLabel::Sarcastic
        );
        assert_eq!(
            semantic_intent_to_label("mixed").unwrap(),
            SentimentLabel::Mixed
        );
    }

    #[test]
    fn infer_plural_monday_vs_glass() {
        assert!(infer_plural_from_text("I hate mondays"));
        assert!(!infer_plural_from_text("the glass is cold"));
    }

    #[test]
    fn append_from_sentiment_jsonl_minimal_line() {
        let path = std::env::temp_dir().join(format!(
            "growformer_sent_{}.jsonl",
            std::process::id()
        ));
        std::fs::write(
            &path,
            "{\"text\":\"ok\",\"semantic_intent\":\"neutral\",\"extra\":1}\n",
        )
        .unwrap();
        let mut batch = TrainingBatch::new();
        let n = batch.append_from_sentiment_jsonl(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(n, 1);
        assert_eq!(batch.records[0].input, "ok");
        assert_eq!(batch.records[0].sentiment, SentimentLabel::Neutral);
    }

    #[test]
    fn load_data_sentiment_train_files() {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data/sentiment");
        if !dir.is_dir() {
            return;
        }
        let b = TrainingBatch::from_sentiment_jsonl_dir(&dir, SentimentJsonlSelection::TrainFilesOnly)
            .expect("load sentiment train jsonl");
        assert!(
            b.len() > 100,
            "expected many train rows from data/sentiment, got {}",
            b.len()
        );
    }
}
