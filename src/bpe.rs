//! Byte-pair encoding tokenizer for TinyStories-scale corpora.
//!
//! Reserved ids match [`crate::v2::data::special`] and [`crate::v2::data::N_SPECIAL`]:
//! PAD, UNK, BOS, SEP, EOS — then 256 byte tokens, then merge symbols.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use crate::v2::data::N_SPECIAL;

// ─── Merge rule ──────────────────────────────────────────────────────────────

/// When symbols `left` and `right` appear adjacent, replace them with `new_id`.
#[derive(Clone, Debug)]
pub struct Merge {
    pub left: u32,
    pub right: u32,
    pub new_id: u32,
}

// ─── BPE tokenizer ───────────────────────────────────────────────────────────

pub struct BpeTokenizer {
    pub vocab: Vec<Vec<u8>>,
    pub merges: Vec<Merge>,
    pub merge_map: HashMap<(u32, u32), u32>,
}

impl BpeTokenizer {
    pub fn new() -> Self {
        let mut vocab = Vec::with_capacity(N_SPECIAL + 256);

        for name in &["<PAD>", "<UNK>", "<BOS>", "<SEP>", "<EOS>"] {
            vocab.push(name.as_bytes().to_vec());
        }
        for b in 0u8..=255 {
            vocab.push(vec![b]);
        }
        Self {
            vocab,
            merges: Vec::new(),
            merge_map: HashMap::new(),
        }
    }

    pub fn vocab_size(&self) -> u32 {
        self.vocab.len() as u32
    }

    #[inline]
    pub fn byte_to_id(b: u8) -> u32 {
        N_SPECIAL as u32 + u32::from(b)
    }

    pub fn train(&mut self, texts: &[String], target_vocab: u32, min_pair_freq: usize) {
        let starting_vocab = self.vocab_size();
        let target_merges = target_vocab.saturating_sub(starting_vocab) as usize;

        let mut sequences: Vec<Vec<u32>> = texts
            .iter()
            .map(|t| t.bytes().map(Self::byte_to_id).collect())
            .collect();

        for merge_idx in 0..target_merges {
            let mut pair_counts: HashMap<(u32, u32), usize> = HashMap::new();
            for seq in &sequences {
                for window in seq.windows(2) {
                    *pair_counts.entry((window[0], window[1])).or_insert(0) += 1;
                }
            }

            let best = pair_counts
                .iter()
                .max_by_key(|(_, &c)| c)
                .map(|(&p, &c)| (p, c));

            let Some(((left, right), freq)) = best else {
                break;
            };
            if freq < min_pair_freq {
                break;
            }

            let new_id = self.vocab_size();
            let mut combined = self.vocab[left as usize].clone();
            combined.extend_from_slice(&self.vocab[right as usize]);
            self.vocab.push(combined);

            self.merges.push(Merge {
                left,
                right,
                new_id,
            });
            self.merge_map.insert((left, right), new_id);

            for seq in &mut sequences {
                let mut out = Vec::with_capacity(seq.len());
                let mut i = 0;
                while i < seq.len() {
                    if i + 1 < seq.len() && seq[i] == left && seq[i + 1] == right {
                        out.push(new_id);
                        i += 2;
                    } else {
                        out.push(seq[i]);
                        i += 1;
                    }
                }
                *seq = out;
            }

            if (merge_idx + 1) % 100 == 0 {
                eprintln!(
                    "[bpe] merges={} vocab={} last_pair_freq={}",
                    merge_idx + 1,
                    self.vocab_size(),
                    freq
                );
            }
        }
        eprintln!(
            "[bpe] trained {} merges, final vocab = {}",
            self.merges.len(),
            self.vocab_size()
        );
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut tokens: Vec<u32> = text.bytes().map(Self::byte_to_id).collect();

        for merge in &self.merges {
            let mut out = Vec::with_capacity(tokens.len());
            let mut i = 0;
            while i < tokens.len() {
                if i + 1 < tokens.len() && tokens[i] == merge.left && tokens[i + 1] == merge.right {
                    out.push(merge.new_id);
                    i += 2;
                } else {
                    out.push(tokens[i]);
                    i += 1;
                }
            }
            tokens = out;
        }
        tokens
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::with_capacity(ids.len() * 4);
        for &id in ids {
            if (id as usize) < N_SPECIAL {
                continue;
            }
            if let Some(piece) = self.vocab.get(id as usize) {
                bytes.extend_from_slice(piece);
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    pub fn decode_one(&self, id: u32) -> String {
        if (id as usize) < N_SPECIAL {
            return String::new();
        }
        self.vocab
            .get(id as usize)
            .map(|b| String::from_utf8_lossy(b).into_owned())
            .unwrap_or_default()
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let mut f = File::create(path)?;
        writeln!(f, "BPE v1 {} {}", self.vocab_size(), self.merges.len())?;
        for m in &self.merges {
            writeln!(f, "{} {} {}", m.left, m.right, m.new_id)?;
        }
        Ok(())
    }

    pub fn load<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let mut tok = Self::new();
        let f = File::open(path)?;
        let mut lines = BufReader::new(f).lines();

        let header = lines.next().ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "missing header")
        })??;
        if !header.starts_with("BPE v1") {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "not a BPE v1 file",
            ));
        }

        for line in lines {
            let line = line?;
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() != 3 {
                continue;
            }
            let left: u32 = parts[0].parse().unwrap_or(0);
            let right: u32 = parts[1].parse().unwrap_or(0);
            let new_id: u32 = parts[2].parse().unwrap_or(0);

            let mut combined = tok.vocab[left as usize].clone();
            combined.extend_from_slice(&tok.vocab[right as usize]);
            tok.vocab.push(combined);
            tok.merges.push(Merge {
                left,
                right,
                new_id,
            });
            tok.merge_map.insert((left, right), new_id);
        }
        Ok(tok)
    }
}

impl Default for BpeTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Streaming corpus reader ───────────────────────────────────────────────────

pub fn read_tinystories_jsonl<P: AsRef<Path>>(path: P) -> std::io::Result<Vec<String>> {
    let f = File::open(path)?;
    let mut stories = Vec::new();
    for line in BufReader::new(f).lines() {
        let line = line?;
        if let Some(story) = extract_story(&line) {
            stories.push(story);
        }
    }
    Ok(stories)
}

fn extract_story(line: &str) -> Option<String> {
    for key in &["story", "text", "content"] {
        let needle = format!("\"{key}\":");
        if let Some(pos) = line.find(&needle) {
            let after = line[pos + needle.len()..].trim_start();
            if after.starts_with('"') {
                let inner = &after[1..];
                let mut end = 0;
                let bytes = inner.as_bytes();
                while end < bytes.len() {
                    match bytes[end] {
                        b'\\' => end += 2,
                        b'"' => break,
                        _ => end += 1,
                    }
                }
                let raw = &inner[..end];
                return Some(unescape_json(raw));
            }
        }
    }
    None
}

fn unescape_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('n') => out.push('\n'),
                Some('t') => out.push('\t'),
                Some('r') => out.push('\r'),
                Some('"') => out.push('"'),
                Some('\\') => out.push('\\'),
                Some(other) => {
                    out.push('\\');
                    out.push(other);
                }
                None => {}
            }
        } else {
            out.push(c);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v2::data::N_SPECIAL;

    #[test]
    fn untrained_tokenizer_encodes_to_bytes() {
        let tok = BpeTokenizer::new();
        let ids = tok.encode("hi");
        assert_eq!(ids.len(), 2);
        assert_eq!(ids[0], BpeTokenizer::byte_to_id(b'h'));
        assert_eq!(ids[1], BpeTokenizer::byte_to_id(b'i'));
    }

    #[test]
    fn round_trip_after_training() {
        let texts: Vec<String> = vec![
            "the cat sat on the mat".into(),
            "the dog sat on the log".into(),
            "the cat and the dog".into(),
        ]
        .into_iter()
        .cycle()
        .take(60)
        .collect();

        let mut tok = BpeTokenizer::new();
        tok.train(&texts, N_SPECIAL as u32 + 256 + 20, 2);

        let original = "the cat";
        let ids = tok.encode(original);
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, original);
    }

    #[test]
    fn merges_reduce_token_count() {
        let texts: Vec<String> = (0..50).map(|_| "the the the".into()).collect();
        let mut tok = BpeTokenizer::new();
        let before = tok.encode("the").len();
        tok.train(&texts, N_SPECIAL as u32 + 256 + 10, 2);
        let after = tok.encode("the").len();
        assert!(
            after < before,
            "after BPE, 'the' should encode as fewer tokens"
        );
    }

    #[test]
    fn save_and_load_round_trip() {
        let texts: Vec<String> = (0..30).map(|_| "hello world".into()).collect();
        let mut tok = BpeTokenizer::new();
        tok.train(&texts, N_SPECIAL as u32 + 256 + 5, 2);

        let path = std::env::temp_dir().join("growformer_bpe_test.txt");
        tok.save(&path).unwrap();
        let loaded = BpeTokenizer::load(&path).unwrap();

        assert_eq!(loaded.vocab_size(), tok.vocab_size());
        assert_eq!(loaded.encode("hello"), tok.encode("hello"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn extract_story_handles_both_formats() {
        let a = r#"{"story": "Once upon a time."}"#;
        let b = r#"{"text": "The cat sat."}"#;
        assert_eq!(extract_story(a), Some("Once upon a time.".to_string()));
        assert_eq!(extract_story(b), Some("The cat sat.".to_string()));
    }
}
