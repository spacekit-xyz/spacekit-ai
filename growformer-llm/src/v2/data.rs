//! JSONL corpus + word tokenizer for causal LM training (`train_v2`).
//!
//! Format per record: `BOS + tokenized(text) + SEP + tokenized(expected_response) + EOS`.
//! Cross-entropy applies only to positions predicting tokens from `expected_response` through `EOS`.

use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Reserved token ids (fixed indices).
pub mod special {
    pub const PAD: usize = 0;
    pub const UNK: usize = 1;
    pub const BOS: usize = 2;
    pub const SEP: usize = 3;
    pub const EOS: usize = 4;
}

/// Exclusive upper bound for non-byte BPE ids: bytes start at this index (must match `BpeTokenizer`).
pub const N_SPECIAL: usize = 5;

#[derive(Debug, Clone)]
pub struct TrainExample {
    pub full_ids: Vec<usize>,
    loss_mask: Vec<bool>,
}

impl TrainExample {
    pub fn len(&self) -> usize {
        self.full_ids.len()
    }

    pub fn loss_mask(&self) -> &[bool] {
        &self.loss_mask
    }

    /// Packed LM chunk (e.g. TinyStories): CE on every position `t` that predicts `t+1`.
    pub fn lm_sequence(full_ids: Vec<usize>) -> Self {
        let n = full_ids.len();
        let mut loss_mask = vec![false; n];
        for t in 0..n.saturating_sub(1) {
            // Skip predicting from/into PAD (turn-aligned padding).
            if full_ids[t] == special::PAD || full_ids[t + 1] == special::PAD {
                continue;
            }
            loss_mask[t] = true;
        }
        Self {
            full_ids,
            loss_mask,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct RawRecord {
    #[serde(default)]
    pub task_id: String,
    pub text: String,
    #[serde(default)]
    pub semantic_intent: String,
    #[serde(default)]
    pub domain: String,
    #[serde(default)]
    pub action_target: String,
    #[serde(default)]
    pub policy_regime: String,
    #[serde(default)]
    pub language_channel: String,
    pub code_language: Option<String>,
    #[serde(default)]
    pub split: String,
    pub expected_response: String,
    pub expected_code: Option<String>,
}

#[derive(Debug, Clone)]
pub struct Dataset {
    pub train: Vec<TrainExample>,
    pub val: Vec<TrainExample>,
}

impl Dataset {
    pub fn load_jsonl(
        path: &Path,
        tokenizer: &mut Tokenizer,
        max_seq: usize,
    ) -> Result<Self, String> {
        let records = load_raw_records(path)?;
        tokenizer.fit(&records);
        let mut train = Vec::new();
        let mut val = Vec::new();
        for r in records {
            if r.expected_response.trim().is_empty() {
                continue;
            }
            let ex = encode_record(&r, tokenizer, max_seq)?;
            match r.split.to_lowercase().as_str() {
                "val" | "validation" | "dev" => val.push(ex),
                "train" | "" => train.push(ex),
                _ => {}
            }
        }
        Ok(Dataset { train, val })
    }

    pub fn shuffled_train(&self, seed: u64) -> Vec<TrainExample> {
        let mut v = self.train.clone();
        let mut s = seed;
        for i in (1..v.len()).rev() {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            let j = (s % (i as u64 + 1)) as usize;
            v.swap(i, j);
        }
        v
    }
}

fn tokenize_words(s: &str) -> Vec<String> {
    s.split_whitespace()
        .map(|w| w.to_lowercase())
        .filter(|w| !w.is_empty())
        .collect()
}

pub struct Tokenizer {
    word_to_id: HashMap<String, usize>,
    pub id_to_word: Vec<String>,
}

impl Tokenizer {
    pub fn new() -> Self {
        let specials = ["<PAD>", "<UNK>", "<BOS>", "<SEP>", "<EOS>"];
        let mut word_to_id = HashMap::new();
        let mut id_to_word = Vec::new();
        for (i, s) in specials.iter().enumerate() {
            word_to_id.insert((*s).to_string(), i);
            id_to_word.push((*s).to_string());
        }
        Self {
            word_to_id,
            id_to_word,
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.id_to_word.len()
    }

    pub fn fit(&mut self, records: &[RawRecord]) {
        for r in records {
            for w in tokenize_words(&r.text) {
                self.add_word(&w);
            }
            for w in tokenize_words(&r.expected_response) {
                self.add_word(&w);
            }
        }
    }

    fn add_word(&mut self, w: &str) {
        if self.word_to_id.contains_key(w) {
            return;
        }
        let id = self.id_to_word.len();
        self.word_to_id.insert(w.to_string(), id);
        self.id_to_word.push(w.to_string());
    }

    pub fn word_id(&self, w: &str) -> usize {
        *self.word_to_id.get(w).unwrap_or(&special::UNK)
    }

    /// Encode a whitespace-tokenized string (lowercased words).
    pub fn encode_words(&self, s: &str) -> Vec<usize> {
        tokenize_words(s)
            .into_iter()
            .map(|w| self.word_id(&w))
            .collect()
    }

    /// Restore tokenizer from checkpoint vocabulary (`id_to_word` order).
    pub fn from_vocab_list(id_to_word: Vec<String>) -> Self {
        let mut word_to_id = HashMap::with_capacity(id_to_word.len());
        for (i, w) in id_to_word.iter().enumerate() {
            word_to_id.insert(w.clone(), i);
        }
        Self {
            word_to_id,
            id_to_word,
        }
    }
}

impl Default for Tokenizer {
    fn default() -> Self {
        Self::new()
    }
}

pub fn encode_record(
    rec: &RawRecord,
    tok: &Tokenizer,
    max_seq: usize,
) -> Result<TrainExample, String> {
    let mut ids = vec![special::BOS];
    for w in tokenize_words(&rec.text) {
        ids.push(tok.word_id(&w));
    }
    ids.push(special::SEP);
    let response_start = ids.len();
    for w in tokenize_words(&rec.expected_response) {
        ids.push(tok.word_id(&w));
    }
    ids.push(special::EOS);

    if ids.len() > max_seq {
        return Err(format!(
            "encoded length {} exceeds max_seq {}; shorten text or increase max_seq",
            ids.len(),
            max_seq
        ));
    }

    let seq = ids.len();
    let mut loss_mask = vec![false; seq];
    for t in 0..seq.saturating_sub(1) {
        if (t + 1) >= response_start {
            loss_mask[t] = true;
        }
    }

    Ok(TrainExample {
        full_ids: ids,
        loss_mask,
    })
}

fn load_raw_records(path: &Path) -> Result<Vec<RawRecord>, String> {
    let f = File::open(path).map_err(|e| format!("open {}: {}", path.display(), e))?;
    let reader = BufReader::new(f);
    let mut out = Vec::new();
    for (ln, line) in reader.lines().enumerate() {
        let line = line.map_err(|e| format!("{}:{}: {}", path.display(), ln + 1, e))?;
        let trim = line.trim();
        if trim.is_empty() {
            continue;
        }
        let rec: RawRecord = serde_json::from_str(trim)
            .map_err(|e| format!("{}:{}: {}", path.display(), ln + 1, e))?;
        out.push(rec);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_sets_loss_on_response_only() {
        let mut tok = Tokenizer::new();
        let rec = RawRecord {
            task_id: "1".into(),
            text: "hello world".into(),
            semantic_intent: "x".into(),
            domain: "d".into(),
            action_target: "a".into(),
            policy_regime: "p".into(),
            language_channel: "en".into(),
            code_language: None,
            split: "train".into(),
            expected_response: "ok".into(),
            expected_code: None,
        };
        tok.fit(&[rec.clone()]);
        let ex = encode_record(&rec, &tok, 64).unwrap();
        assert!(ex.loss_mask[0] == false); // BOS predicts hello — masked false
                                           // First true should predict start of response
        let sep_ix = ex.full_ids.iter().position(|&x| x == special::SEP).unwrap();
        let r_start = sep_ix + 1;
        assert_eq!(r_start, 4); // BOS, hello, world, SEP → first response token at index 4
        assert!(ex.loss_mask[sep_ix]); // from SEP, predict first response token
    }
}
