//! Shared text featurizer for the fast heads: TF-IDF over word 1–2-grams and
//! in-word character 2–5-grams, each block L2-normalised. Computed **once per
//! state** and reused by every question head (the "shared state" pattern).

use std::collections::{BTreeMap, HashMap};

use serde::{Deserialize, Serialize};

/// Sparse feature vector: sorted `(index, value)` pairs.
pub type SparseVec = Vec<(u32, f32)>;

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
struct Block {
    terms: Vec<String>,
    idf: Vec<f32>,
    #[serde(skip)]
    index: HashMap<String, u32>,
}

impl Block {
    fn fit(docs: &[Vec<String>], min_df: u32, max_terms: usize) -> Self {
        let mut df: BTreeMap<&str, u32> = BTreeMap::new();
        for d in docs {
            let mut seen: Vec<&str> = d.iter().map(|s| s.as_str()).collect();
            seen.sort_unstable();
            seen.dedup();
            for t in seen {
                *df.entry(t).or_insert(0) += 1;
            }
        }
        let mut kept: Vec<(&str, u32)> = df.into_iter().filter(|&(_, c)| c >= min_df).collect();
        if max_terms > 0 && kept.len() > max_terms {
            // Keep the most frequent terms; ties broken alphabetically (deterministic).
            kept.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
            kept.truncate(max_terms);
            kept.sort_by(|a, b| a.0.cmp(b.0));
        }
        let n = docs.len() as f32;
        let mut b = Self {
            terms: kept.iter().map(|(t, _)| t.to_string()).collect(),
            idf: kept
                .iter()
                .map(|&(_, c)| ((1.0 + n) / (1.0 + c as f32)).ln() + 1.0)
                .collect(),
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

    fn transform(&self, terms: &[String], base: u32, out: &mut SparseVec) {
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
            out.extend(v.into_iter().map(|(i, x)| (base + i, x / norm)));
        }
    }
}

/// Fitted featurizer (vocabulary + IDF).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Featurizer {
    word: Block,
    chars: Block,
}

/// Vocabulary limits (keep brains small enough to ship to the browser).
#[derive(Clone, Debug)]
pub struct FeaturizerConfig {
    pub min_df: u32,
    pub max_word_terms: usize,
    pub max_char_terms: usize,
}

impl Default for FeaturizerConfig {
    fn default() -> Self {
        Self {
            min_df: 1,
            max_word_terms: 50_000,
            max_char_terms: 100_000,
        }
    }
}

impl Featurizer {
    pub fn fit(texts: &[String], cfg: &FeaturizerConfig) -> Self {
        let w: Vec<Vec<String>> = texts.iter().map(|t| word_terms(t)).collect();
        let c: Vec<Vec<String>> = texts.iter().map(|t| char_terms(t)).collect();
        Self {
            word: Block::fit(&w, cfg.min_df, cfg.max_word_terms),
            chars: Block::fit(&c, cfg.min_df, cfg.max_char_terms),
        }
    }

    pub fn dim(&self) -> usize {
        self.word.terms.len() + self.chars.terms.len()
    }

    pub fn transform(&self, text: &str) -> SparseVec {
        let mut out = Vec::new();
        self.word.transform(&word_terms(text), 0, &mut out);
        self.chars
            .transform(&char_terms(text), self.word.terms.len() as u32, &mut out);
        out
    }

    /// Must be called after deserialising.
    pub fn rebuild_index(&mut self) {
        self.word.rebuild_index();
        self.chars.rebuild_index();
    }
}

/// Cosine similarity of two sparse vectors.
pub fn cosine(a: &SparseVec, b: &SparseVec) -> f32 {
    let (mut i, mut j, mut dot) = (0usize, 0usize, 0.0f32);
    while i < a.len() && j < b.len() {
        match a[i].0.cmp(&b[j].0) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                dot += a[i].1 * b[j].1;
                i += 1;
                j += 1;
            }
        }
    }
    let na = a.iter().map(|x| x.1 * x.1).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x.1 * x.1).sum::<f32>().sqrt();
    if na > 0.0 && nb > 0.0 {
        dot / (na * nb)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_transform_roundtrip() {
        let docs: Vec<String> = ["refund my order", "app keeps crashing", "cancel my plan"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let f = Featurizer::fit(&docs, &FeaturizerConfig::default());
        let a = f.transform("please refund my order");
        let b = f.transform("refund order");
        let c = f.transform("the app is crashing");
        assert!(cosine(&a, &b) > cosine(&a, &c));
        let json = serde_json::to_string(&f).unwrap();
        let mut back: Featurizer = serde_json::from_str(&json).unwrap();
        back.rebuild_index();
        assert_eq!(
            back.transform("refund my order"),
            f.transform("refund my order")
        );
    }
}
