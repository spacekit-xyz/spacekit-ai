//! Supervised label classifier for domain prompts (e.g. `semantic_intent`).
//!
//! TF-IDF features (word 1–2-grams + in-word character 2–5-grams, each block
//! L2-normalised) feeding a multinomial logistic regression trained with
//! full-batch Adam and L2 regularisation. Small, deterministic, CPU-only.
//!
//! Why this exists: on the crypto sentiment project the Growformer brain's own
//! topic step scores ~11 % on its *training* rows, so topic-scoped lattice
//! retrieval usually searches the wrong sub-lattice. This classifier picks the
//! label (≈64 % held-out on the same 302 rows vs 37 % for the brain), and the
//! brain then retrieves an explanation from that label's sub-lattice
//! (`BrainMemoryRuntime::query_labeled`).

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use serde::{Deserialize, Serialize};

const MODEL_FORMAT: u32 = 1;

// ─── Features ────────────────────────────────────────────────────────────────

fn words(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !(c.is_alphanumeric() || c == '$' || c == '%' || c == '\''))
        .map(|w| w.trim_matches('\''))
        .filter(|w| !w.is_empty())
        .map(|w| w.to_string())
        .collect()
}

fn word_terms(text: &str) -> Vec<String> {
    let ws = words(text);
    let mut out = ws.clone();
    for pair in ws.windows(2) {
        out.push(format!("{} {}", pair[0], pair[1]));
    }
    out
}

fn char_terms(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for w in words(text) {
        let padded: Vec<char> = format!(" {w} ").chars().collect();
        for n in 2..=5 {
            if padded.len() < n {
                continue;
            }
            for win in padded.windows(n) {
                out.push(win.iter().collect());
            }
        }
    }
    out
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct TfidfBlock {
    terms: Vec<String>,
    idf: Vec<f32>,
    #[serde(skip)]
    index: HashMap<String, u32>,
}

impl TfidfBlock {
    fn fit(docs: &[Vec<String>]) -> Self {
        let mut df: BTreeMap<&str, u32> = BTreeMap::new();
        for d in docs {
            let mut seen: Vec<&str> = d.iter().map(|s| s.as_str()).collect();
            seen.sort_unstable();
            seen.dedup();
            for t in seen {
                *df.entry(t).or_insert(0) += 1;
            }
        }
        let n = docs.len() as f32;
        let mut terms = Vec::with_capacity(df.len());
        let mut idf = Vec::with_capacity(df.len());
        for (t, c) in df {
            terms.push(t.to_string());
            idf.push(((1.0 + n) / (1.0 + c as f32)).ln() + 1.0);
        }
        let mut b = Self {
            terms,
            idf,
            index: HashMap::new(),
        };
        b.rebuild_index();
        b
    }

    fn rebuild_index(&mut self) {
        self.index = self
            .terms
            .iter()
            .enumerate()
            .map(|(i, t)| (t.clone(), i as u32))
            .collect();
    }

    /// Sublinear-tf × idf, L2-normalised; indices offset by `base`.
    fn transform(&self, terms: &[String], base: u32, out: &mut Vec<(u32, f32)>) {
        let mut tf: HashMap<u32, f32> = HashMap::new();
        for t in terms {
            if let Some(&i) = self.index.get(t) {
                *tf.entry(i).or_insert(0.0) += 1.0;
            }
        }
        let mut v: Vec<(u32, f32)> = tf
            .into_iter()
            .map(|(i, c)| (i, (1.0 + c.ln()) * self.idf[i as usize]))
            .collect();
        v.sort_unstable_by_key(|&(i, _)| i);
        let norm = v.iter().map(|&(_, x)| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for (i, x) in v {
                out.push((base + i, x / norm));
            }
        }
    }

    fn len(&self) -> usize {
        self.terms.len()
    }
}

// ─── Model ───────────────────────────────────────────────────────────────────

/// Training knobs.
#[derive(Clone, Debug)]
pub struct LabelTrainConfig {
    /// Inverse L2 strength (sklearn-style `C`; larger = weaker regularisation).
    pub c: f32,
    pub epochs: usize,
    pub lr: f32,
}

impl Default for LabelTrainConfig {
    fn default() -> Self {
        Self {
            c: 10.0,
            epochs: 400,
            lr: 0.05,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LabelClassifier {
    pub format: u32,
    pub labels: Vec<String>,
    word: TfidfBlock,
    chars: TfidfBlock,
    /// `[n_labels][n_features]`
    weights: Vec<Vec<f32>>,
    bias: Vec<f32>,
    /// Rows the model was fitted on (for the card / sanity checks).
    #[serde(default)]
    pub n_train: usize,
    /// Sentiment schemes only: a second head trained on coarse polarity
    /// (POS/NEG/NEU/MIXED/SARC). `predict_top` picks the coarse bucket with it and
    /// then the best fine label inside that bucket — ~+7 pts coarse accuracy on
    /// the crypto split vs taking the fine argmax directly.
    #[serde(default)]
    coarse_head: Option<Box<LabelClassifier>>,
}

/// One scored label.
#[derive(Clone, Debug, PartialEq)]
pub struct LabelScore {
    pub label: String,
    pub prob: f32,
}

fn softmax_in_place(z: &mut [f32]) {
    let m = z.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut s = 0.0f32;
    for v in z.iter_mut() {
        *v = (*v - m).exp();
        s += *v;
    }
    for v in z.iter_mut() {
        *v /= s;
    }
}

impl LabelClassifier {
    fn features(&self, text: &str) -> Vec<(u32, f32)> {
        let mut out = Vec::new();
        self.word.transform(&word_terms(text), 0, &mut out);
        self.chars
            .transform(&char_terms(text), self.word.len() as u32, &mut out);
        out
    }

    fn logits(&self, x: &[(u32, f32)]) -> Vec<f32> {
        self.weights
            .iter()
            .zip(&self.bias)
            .map(|(w, &b)| b + x.iter().map(|&(i, v)| w[i as usize] * v).sum::<f32>())
            .collect()
    }

    /// Fit on `(text, label)` pairs. Labels are sorted for a stable class order.
    pub fn train(rows: &[(String, String)], cfg: &LabelTrainConfig) -> Result<Self, String> {
        if rows.is_empty() {
            return Err("no training rows".into());
        }
        let mut labels: Vec<String> = rows.iter().map(|(_, l)| l.clone()).collect();
        labels.sort();
        labels.dedup();
        if labels.len() < 2 {
            return Err(format!("need at least 2 labels, found {:?}", labels));
        }
        let label_idx: HashMap<String, usize> = labels
            .iter()
            .enumerate()
            .map(|(i, l)| (l.clone(), i))
            .collect();

        let word = TfidfBlock::fit(&rows.iter().map(|(t, _)| word_terms(t)).collect::<Vec<_>>());
        let chars = TfidfBlock::fit(&rows.iter().map(|(t, _)| char_terms(t)).collect::<Vec<_>>());
        let n_feat = word.len() + chars.len();
        let k = labels.len();
        let mut model = Self {
            format: MODEL_FORMAT,
            labels,
            word,
            chars,
            weights: vec![vec![0.0; n_feat]; k],
            bias: vec![0.0; k],
            n_train: rows.len(),
            coarse_head: None,
        };

        let xs: Vec<Vec<(u32, f32)>> = rows.iter().map(|(t, _)| model.features(t)).collect();
        let ys: Vec<usize> = rows.iter().map(|(_, l)| label_idx[l]).collect();
        let n = rows.len() as f32;
        // sklearn: sum CE + ||W||²/(2C)  ⇔  mean CE + λ/2 ||W||² with λ = 1/(C·n)
        let lambda = 1.0 / (cfg.c * n);

        let (b1, b2, eps) = (0.9f32, 0.999f32, 1e-8f32);
        let mut mw = vec![vec![0.0f32; n_feat]; k];
        let mut vw = vec![vec![0.0f32; n_feat]; k];
        let mut mb = vec![0.0f32; k];
        let mut vb = vec![0.0f32; k];
        let mut gw = vec![vec![0.0f32; n_feat]; k];
        let mut gb = vec![0.0f32; k];

        for epoch in 1..=cfg.epochs {
            for row in gw.iter_mut() {
                row.iter_mut().for_each(|g| *g = 0.0);
            }
            gb.iter_mut().for_each(|g| *g = 0.0);
            for (x, &y) in xs.iter().zip(&ys) {
                let mut p = model.logits(x);
                softmax_in_place(&mut p);
                p[y] -= 1.0;
                for c in 0..k {
                    let d = p[c] / n;
                    if d == 0.0 {
                        continue;
                    }
                    gb[c] += d;
                    let row = &mut gw[c];
                    for &(i, v) in x {
                        row[i as usize] += d * v;
                    }
                }
            }
            let t = epoch as f32;
            let (bc1, bc2) = (1.0 - b1.powf(t), 1.0 - b2.powf(t));
            for c in 0..k {
                for j in 0..n_feat {
                    let g = gw[c][j] + lambda * model.weights[c][j];
                    mw[c][j] = b1 * mw[c][j] + (1.0 - b1) * g;
                    vw[c][j] = b2 * vw[c][j] + (1.0 - b2) * g * g;
                    model.weights[c][j] -=
                        cfg.lr * (mw[c][j] / bc1) / ((vw[c][j] / bc2).sqrt() + eps);
                }
                let g = gb[c];
                mb[c] = b1 * mb[c] + (1.0 - b1) * g;
                vb[c] = b2 * vb[c] + (1.0 - b2) * g * g;
                model.bias[c] -= cfg.lr * (mb[c] / bc1) / ((vb[c] / bc2).sqrt() + eps);
            }
        }

        // Coarse head when every label maps to a sentiment bucket and ≥2 buckets exist.
        let coarse_rows: Option<Vec<(String, String)>> = rows
            .iter()
            .map(|(t, l)| coarse_sentiment(l).map(|c| (t.clone(), c.to_string())))
            .collect();
        if let Some(cr) = coarse_rows {
            let mut buckets: Vec<&str> = cr.iter().map(|(_, c)| c.as_str()).collect();
            buckets.sort_unstable();
            buckets.dedup();
            if buckets.len() >= 2 && buckets.len() < model.labels.len() {
                model.coarse_head = Some(Box::new(Self::train(&cr, cfg)?));
            }
        }
        Ok(model)
    }

    /// Cosine similarity of two texts in this model's TF-IDF space (word + char
    /// blocks). Used to pick the stored example closest to a prompt.
    pub fn similarity(&self, a: &str, b: &str) -> f32 {
        let (fa, fb) = (self.features(a), self.features(b));
        let (mut i, mut j, mut dot) = (0usize, 0usize, 0.0f32);
        while i < fa.len() && j < fb.len() {
            match fa[i].0.cmp(&fb[j].0) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    dot += fa[i].1 * fb[j].1;
                    i += 1;
                    j += 1;
                }
            }
        }
        let na = fa.iter().map(|x| x.1 * x.1).sum::<f32>().sqrt();
        let nb = fb.iter().map(|x| x.1 * x.1).sum::<f32>().sqrt();
        if na > 0.0 && nb > 0.0 {
            dot / (na * nb)
        } else {
            0.0
        }
    }

    /// All labels with probabilities, best first.
    pub fn predict(&self, text: &str) -> Vec<LabelScore> {
        let x = self.features(text);
        let mut p = self.logits(&x);
        softmax_in_place(&mut p);
        let mut out: Vec<LabelScore> = self
            .labels
            .iter()
            .zip(p)
            .map(|(l, prob)| LabelScore {
                label: l.clone(),
                prob,
            })
            .collect();
        out.sort_by(|a, b| b.prob.partial_cmp(&a.prob).unwrap_or(std::cmp::Ordering::Equal));
        out
    }

    /// Best label. With a coarse head: best coarse bucket first, then the most
    /// probable fine label inside it (`prob` = the bucket's probability).
    pub fn predict_top(&self, text: &str) -> LabelScore {
        let fine = self.predict(text);
        if let Some(head) = &self.coarse_head {
            let bucket = head.predict(text).into_iter().next().expect("≥2 buckets");
            if let Some(best) = fine
                .iter()
                .find(|s| coarse_sentiment(&s.label) == Some(bucket.label.as_str()))
            {
                return LabelScore {
                    label: best.label.clone(),
                    prob: bucket.prob,
                };
            }
        }
        fine.into_iter().next().expect("≥2 labels")
    }

    /// True when the label set is a sentiment scheme (drives reply label prefixes).
    pub fn is_sentiment_scheme(&self) -> bool {
        self.labels
            .iter()
            .any(|l| l.contains("positive") || l.contains("negative"))
    }

    pub fn save(&self, path: &Path) -> Result<(), String> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        // Weights are rounded to 5 significant digits by f32 → JSON; keep compact.
        let json = serde_json::to_string(self).map_err(|e| e.to_string())?;
        std::fs::write(path, json).map_err(|e| format!("write {}: {e}", path.display()))
    }

    pub fn load(path: &Path) -> Result<Self, String> {
        let raw = std::fs::read_to_string(path)
            .map_err(|e| format!("read {}: {e}", path.display()))?;
        let mut m: Self = serde_json::from_str(&raw)
            .map_err(|e| format!("parse {}: {e}", path.display()))?;
        if m.format != MODEL_FORMAT {
            return Err(format!("unsupported label model format {}", m.format));
        }
        m.rebuild_indices();
        Ok(m)
    }

    fn rebuild_indices(&mut self) {
        self.word.rebuild_index();
        self.chars.rebuild_index();
        if let Some(h) = self.coarse_head.as_mut() {
            h.rebuild_indices();
        }
    }
}

// ─── Label helpers (sentiment display / coarse scoring) ──────────────────────

/// Human-facing label for a reply prefix: `negative_mild` → `NEGATIVE (mild)`,
/// `cautiously_positive` → `CAUTIOUSLY POSITIVE`, `hopium` → `HOPIUM`.
pub fn display_label(label: &str) -> String {
    for strength in ["strong", "mild"] {
        if let Some(base) = label.strip_suffix(&format!("_{strength}")) {
            return format!("{} ({strength})", base.replace('_', " ").to_uppercase());
        }
    }
    label.replace('_', " ").to_uppercase()
}

/// Coarse polarity bucket for a fine sentiment label or a reply's leading label.
/// Returns `POS | NEG | NEU | MIXED | SARC`, or `None` if it isn't a sentiment label.
pub fn coarse_sentiment(label_or_reply_head: &str) -> Option<&'static str> {
    let l = label_or_reply_head.to_lowercase();
    if l.contains("sarcas") {
        Some("SARC")
    } else if l.contains("mixed") {
        Some("MIXED")
    } else if l.contains("positive") || l.contains("hopium") || l.contains("euphori") {
        Some("POS")
    } else if l.contains("negative")
        || l.contains("capitulation")
        || l.contains("bearish")
        || l.contains("copium")
    {
        Some("NEG")
    } else if l.contains("neutral") || l.contains("confused") || l.contains("chop") {
        Some("NEU")
    } else {
        None
    }
}

/// The label a reply leads with (`"NEGATIVE (mild) — …"` → `"NEGATIVE (mild)"`).
pub fn reply_label_head(reply: &str) -> String {
    let r = reply.trim();
    let head = r.split(" — ").next().unwrap_or(r);
    head.chars().take(40).collect()
}

/// True when `reply` already starts with an upper-case label (`"MIXED — …"`).
pub fn reply_has_label(reply: &str) -> bool {
    let head = reply.trim().split(" — ").next().unwrap_or("");
    reply.contains(" — ")
        && !head.is_empty()
        && head.len() <= 40
        && head
            .chars()
            .filter(|c| c.is_alphabetic())
            .all(|c| c.is_uppercase() || "mildstrong".contains(c))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toy() -> Vec<(String, String)> {
        let pos = [
            "bitcoin rallied hard today",
            "eth pumped to a new high",
            "great gains across the market",
            "bullish breakout on huge volume",
        ];
        let neg = [
            "bitcoin crashed overnight",
            "eth dumped after the hack",
            "terrible losses across the market",
            "bearish breakdown on huge volume",
        ];
        pos.iter()
            .map(|t| (t.to_string(), "positive_mild".to_string()))
            .chain(neg.iter().map(|t| (t.to_string(), "negative_mild".to_string())))
            .collect()
    }

    #[test]
    fn learns_and_generalises_on_toy_data() {
        let m = LabelClassifier::train(&toy(), &LabelTrainConfig::default()).unwrap();
        assert_eq!(m.predict_top("solana rallied to a new high").label, "positive_mild");
        assert_eq!(m.predict_top("doge crashed after the dump").label, "negative_mild");
        let p = m.predict("anything");
        assert!((p.iter().map(|s| s.prob).sum::<f32>() - 1.0).abs() < 1e-4);
    }

    #[test]
    fn save_load_roundtrip() {
        let m = LabelClassifier::train(&toy(), &LabelTrainConfig::default()).unwrap();
        let path = std::env::temp_dir().join(format!("gf-label-{}.json", std::process::id()));
        m.save(&path).unwrap();
        let back = LabelClassifier::load(&path).unwrap();
        let a = m.predict("bitcoin crashed");
        let b = back.predict("bitcoin crashed");
        assert_eq!(a[0].label, b[0].label);
        assert!((a[0].prob - b[0].prob).abs() < 1e-5);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn roundtrip_keeps_coarse_head_working() {
        let mut rows = toy();
        rows.push(("btc ripped higher on etf news".into(), "positive_strong".into()));
        rows.push(("massive liquidation cascade wiped longs".into(), "negative_strong".into()));
        let m = LabelClassifier::train(&rows, &LabelTrainConfig::default()).unwrap();
        assert!(m.coarse_head.is_some());
        let path = std::env::temp_dir().join(format!("gf-label-c-{}.json", std::process::id()));
        m.save(&path).unwrap();
        let back = LabelClassifier::load(&path).unwrap();
        for t in ["bitcoin crashed overnight", "eth pumped to a new high", "random words"] {
            assert_eq!(m.predict_top(t), back.predict_top(t), "{t}");
        }
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn label_helpers() {
        assert_eq!(display_label("negative_mild"), "NEGATIVE (mild)");
        assert_eq!(display_label("cautiously_positive"), "CAUTIOUSLY POSITIVE");
        assert_eq!(coarse_sentiment("hopium"), Some("POS"));
        assert_eq!(coarse_sentiment("NEGATIVE (strong)"), Some("NEG"));
        assert_eq!(coarse_sentiment("greeting_check_in"), None);
        assert!(reply_has_label("MIXED — both sides"));
        assert!(reply_has_label("NEGATIVE (mild) — selloff"));
        assert!(!reply_has_label("Positive governance outcome; mixed sentiment"));
    }
}
