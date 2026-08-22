//! TinyStories raw `.txt` loader, packed token binary (`CLIFTOKS`), and random-chunk sampling for v2 LM training.

use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::Path;

use crate::bpe::BpeTokenizer;
use crate::v2::data::{special, TrainExample, N_SPECIAL};
use crate::v2::sample::SimpleRng;

// ─── Raw text loader ──────────────────────────────────────────────────────────

/// Read TinyStories `<|endoftext|>`-separated `.txt` (one story per segment).
pub fn load_tinystories_txt<P: AsRef<Path>>(path: P) -> std::io::Result<Vec<String>> {
    let mut content = String::new();
    File::open(path)?.read_to_string(&mut content)?;

    let stories: Vec<String> = content
        .split("<|endoftext|>")
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    eprintln!("[tinystories] loaded {} stories", stories.len());
    Ok(stories)
}

// ─── Packed binary format ─────────────────────────────────────────────────────

const MAGIC: &[u8; 8] = b"CLIFTOKS";
const VERSION: u32 = 1;

/// Encode stories as: `BOS`, BPE ids, `EOS` per story.
pub fn encode_corpus<P: AsRef<Path>>(
    stories: &[String],
    tokenizer: &BpeTokenizer,
    output: P,
) -> std::io::Result<usize> {
    let mut writer = BufWriter::new(File::create(output)?);

    writer.write_all(MAGIC)?;
    writer.write_all(&VERSION.to_le_bytes())?;
    writer.write_all(&tokenizer.vocab_size().to_le_bytes())?;

    let mut total = 0usize;
    let report_every = (stories.len() / 50).max(1);

    for (i, story) in stories.iter().enumerate() {
        writer.write_all(&(special::BOS as u32).to_le_bytes())?;
        let ids = tokenizer.encode(story);
        for id in &ids {
            writer.write_all(&id.to_le_bytes())?;
        }
        writer.write_all(&(special::EOS as u32).to_le_bytes())?;
        total += ids.len() + 2;

        if i % report_every == 0 {
            eprintln!(
                "[encode_corpus] {}/{} stories ({} tokens so far)",
                i,
                stories.len(),
                total
            );
        }
    }
    eprintln!(
        "[encode_corpus] done: {} stories, {} tokens",
        stories.len(),
        total
    );
    Ok(total)
}

// ─── Packed dataset ────────────────────────────────────────────────────────────

pub struct PackedDataset {
    pub tokens: Vec<u32>,
    pub vocab_size: u32,
}

impl PackedDataset {
    pub fn load<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let mut f = File::open(path)?;

        let mut magic = [0u8; 8];
        f.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("bad magic: expected {MAGIC:?}, got {magic:?}"),
            ));
        }

        let mut buf4 = [0u8; 4];
        f.read_exact(&mut buf4)?;
        let version = u32::from_le_bytes(buf4);
        if version != VERSION {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("unsupported version {version}"),
            ));
        }
        f.read_exact(&mut buf4)?;
        let vocab_size = u32::from_le_bytes(buf4);

        let mut bytes = Vec::new();
        f.read_to_end(&mut bytes)?;
        if bytes.len() % 4 != 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "token stream not aligned to u32",
            ));
        }
        let tokens: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        eprintln!(
            "[packed] loaded {} tokens, vocab_size={}",
            tokens.len(),
            vocab_size
        );
        Ok(Self { tokens, vocab_size })
    }

    pub fn random_chunk(&self, seq_len: usize, rng: &mut SimpleRng) -> Vec<usize> {
        let max_start = self.tokens.len().saturating_sub(seq_len + 1);
        if max_start == 0 {
            return self
                .tokens
                .iter()
                .take(seq_len)
                .map(|&x| x as usize)
                .collect();
        }
        let start = (rng.next_u32() as usize) % max_start;
        self.tokens[start..start + seq_len]
            .iter()
            .map(|&x| x as usize)
            .collect()
    }

    /// Indices of `BOS` tokens (document starts from [`encode_corpus`]).
    pub fn doc_starts(&self) -> Vec<usize> {
        self.tokens
            .iter()
            .enumerate()
            .filter_map(|(i, &t)| (t as usize == special::BOS).then_some(i))
            .collect()
    }

    /// Sample a full document from BOS through EOS (or `seq_len`), left-aligned and
    /// PAD-padded. Keeps User→Assistant turns intact for chat corpora.
    pub fn random_turn_chunk(&self, seq_len: usize, rng: &mut SimpleRng) -> Vec<usize> {
        let starts = self.doc_starts();
        if starts.is_empty() || seq_len == 0 {
            return self.random_chunk(seq_len, rng);
        }
        let doc_i = (rng.next_u32() as usize) % starts.len();
        let start = starts[doc_i];
        let end_lim = starts.get(doc_i + 1).copied().unwrap_or(self.tokens.len());
        let mut out: Vec<usize> = Vec::with_capacity(seq_len);
        for &t in &self.tokens[start..end_lim] {
            if out.len() >= seq_len {
                break;
            }
            out.push(t as usize);
            if t as usize == special::EOS && out.len() > 1 {
                break;
            }
        }
        while out.len() < seq_len {
            out.push(special::PAD);
        }
        out
    }

    pub fn n_tokens(&self) -> usize {
        self.tokens.len()
    }

    /// Count text-token occurrences (ids ≥ `N_SPECIAL`) for unigram baselines.
    pub fn unigram_counts(&self, vocab_size: usize) -> (Vec<u64>, u64) {
        let mut counts = vec![0u64; vocab_size];
        let mut total = 0u64;
        for &t in &self.tokens {
            let id = t as usize;
            if id < N_SPECIAL as usize || id >= vocab_size {
                continue;
            }
            counts[id] += 1;
            total += 1;
        }
        (counts, total)
    }

    /// Mean NLL (nats/token) of `tokens` under an empirical unigram from `counts` / `total`.
    pub fn unigram_nll_nats(
        counts: &[u64],
        total: u64,
        tokens: &[u32],
        vocab_size: usize,
    ) -> (f64, usize) {
        if total == 0 {
            return (0.0, 0);
        }
        let mut nll = 0.0f64;
        let mut n = 0usize;
        for &t in tokens {
            let id = t as usize;
            if id < N_SPECIAL as usize || id >= vocab_size {
                continue;
            }
            let p = (counts[id] as f64 / total as f64).max(1e-12);
            nll += -p.ln();
            n += 1;
        }
        if n == 0 {
            (0.0, 0)
        } else {
            (nll / n as f64, n)
        }
    }

    /// Chronological split for held-out eval (first `train_frac` tokens → train).
    pub fn split_chronological(&self, train_frac: f64) -> (Self, Self) {
        let n = self.tokens.len();
        let split = ((n as f64) * train_frac.clamp(0.0, 1.0)) as usize;
        let split = split.clamp(1, n.saturating_sub(1));
        let train = Self {
            tokens: self.tokens[..split].to_vec(),
            vocab_size: self.vocab_size,
        };
        let held = Self {
            tokens: self.tokens[split..].to_vec(),
            vocab_size: self.vocab_size,
        };
        (train, held)
    }

    /// Write a CLIFTOKS bin (same format as [`encode_corpus`]).
    pub fn write<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let mut writer = BufWriter::new(File::create(path)?);
        writer.write_all(MAGIC)?;
        writer.write_all(&VERSION.to_le_bytes())?;
        writer.write_all(&self.vocab_size.to_le_bytes())?;
        for &t in &self.tokens {
            writer.write_all(&t.to_le_bytes())?;
        }
        Ok(())
    }
}

/// Random contiguous chunk → [`TrainExample`] with next-token loss on all non-final positions.
pub fn chunk_to_example(chunk: Vec<usize>) -> TrainExample {
    TrainExample::lm_sequence(chunk)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v2::data::N_SPECIAL;

    #[test]
    fn encode_decode_round_trip() {
        let texts: Vec<String> = (0..30).map(|_| "the cat sat".into()).collect();
        let mut tok = BpeTokenizer::new();
        tok.train(&texts, N_SPECIAL as u32 + 256 + 10, 2);

        let stories = vec!["the cat".to_string(), "sat".to_string()];

        let path = std::env::temp_dir().join("growformer_packed_test.bin");
        let total = encode_corpus(&stories, &tok, &path).unwrap();
        assert!(total > 0);

        let dataset = PackedDataset::load(&path).unwrap();
        assert_eq!(dataset.vocab_size, tok.vocab_size());
        assert_eq!(dataset.n_tokens(), total);

        assert_eq!(dataset.tokens[0], special::BOS as u32);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn random_chunk_returns_correct_length() {
        let path = std::env::temp_dir().join("growformer_packed_rand_test.bin");
        let mut tok = BpeTokenizer::new();
        let texts: Vec<String> = (0..30).map(|_| "abcdefg".into()).collect();
        tok.train(&texts, N_SPECIAL as u32 + 256 + 5, 2);
        let stories = vec!["abcdefg".repeat(20)];
        encode_corpus(&stories, &tok, &path).unwrap();

        let dataset = PackedDataset::load(&path).unwrap();
        let mut rng = SimpleRng::new(123);
        let chunk = dataset.random_chunk(32, &mut rng);
        assert_eq!(chunk.len(), 32);

        let _ = std::fs::remove_file(&path);
    }
}
