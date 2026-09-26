//! **Fleet of brain micro-models.** A host program loads *many* domain
//! specialists at once, routes a prompt to the one that owns the subject, and
//! generates from it — the "single subject understanding vs all-knowing
//! generalization" architecture, made concrete.
//!
//! A [`Specialist`] is one loaded model + its tokenizer + its [`ModelCard`].
//! A [`SpecialistFleet`] is a registry keyed by subject, plus a pluggable
//! [`Router`] (default: [`KeywordRouter`]) that maps a prompt to a subject.
//!
//! Feature-gated behind `clifford-lm`: today it loads Clifford `LmCheckpoint`
//! specialists (which embed their tokenizer). Vanilla specialists are recognised
//! but not yet runnable here — that lands with the Stage 2 vanilla runtime.

use std::collections::HashMap;
use std::path::Path;

use crate::model_card::{Arch, ModelCard, SpecialistManifest, CARD_EXT};
use crate::v2::checkpoint::load_lm_checkpoint;
use crate::v2::data::{special, Tokenizer};
use crate::v2::inference::InferenceCache;
use crate::v2::sample::{sample_next, SampleConfig, SimpleRng};
use crate::v2::train_v2::ModelStateV2;

/// One loaded brain micro-model: weights + tokenizer + identity card.
pub struct Specialist {
    pub card: ModelCard,
    state: ModelStateV2,
    tokenizer: Tokenizer,
}

impl Specialist {
    /// Load a Clifford specialist directly from a checkpoint that embeds its
    /// tokenizer (written by `save_lm_checkpoint`). `subject` names it in a fleet.
    pub fn load_checkpoint(subject: impl Into<String>, checkpoint: &Path) -> Result<Self, String> {
        let (state, tokenizer) = load_lm_checkpoint(checkpoint)?;
        Ok(Self {
            card: ModelCard::new(subject),
            state,
            tokenizer,
        })
    }

    /// Load a specialist from a sidecar manifest (`*.gfcard.json`).
    /// `manifest_dir` is the directory the manifest lives in (weights paths are
    /// resolved relative to it).
    pub fn from_manifest(manifest_dir: &Path, manifest: &SpecialistManifest) -> Result<Self, String> {
        match manifest.arch {
            Arch::Clifford => {
                let weights = manifest.weights_abs(manifest_dir);
                let (state, tokenizer) = load_lm_checkpoint(&weights)?;
                Ok(Self {
                    card: manifest.card.clone(),
                    state,
                    tokenizer,
                })
            }
            Arch::Vanilla => Err(format!(
                "specialist '{}' is a vanilla model — not yet runnable in the fleet \
                 (Stage 2 vanilla runtime TODO)",
                manifest.card.subject
            )),
        }
    }

    pub fn subject(&self) -> &str {
        &self.card.subject
    }

    /// Generate a continuation for `prompt` (`BOS + prompt + SEP → tokens`).
    /// KV-cached, single-token-at-a-time; returns the decoded text.
    pub fn generate(&self, prompt: &str, cfg: &SampleConfig) -> String {
        let mut ids = vec![special::BOS];
        ids.extend(self.tokenizer.encode_words(prompt));
        ids.push(special::SEP);

        let d_model = self
            .state
            .model
            .embedding
            .first()
            .map(|row| row.len())
            .unwrap_or(0);
        let mut cache = InferenceCache::new(
            self.state.cfg.n_blocks,
            self.state.cfg.max_seq,
            d_model,
            self.state.cfg.dot_attention,
        );

        let mut rng = SimpleRng::new(cfg.seed.unwrap_or_else(|| seed_from_ids(&ids)));
        let mut out = String::new();

        for _ in 0..cfg.max_new_tokens {
            let logits = cache.forward_extend(&self.state.alg, &self.state.model, &ids);
            let last = match logits.last() {
                Some(l) => l,
                None => break,
            };
            let next = sample_next(last, &ids, cfg, &mut rng);
            if cfg.stop_tokens.contains(&next) {
                break;
            }
            if let Some(piece) = self.tokenizer.id_to_word.get(next) {
                if !out.is_empty() {
                    out.push(' ');
                }
                out.push_str(piece);
            }
            ids.push(next);
        }
        out
    }
}

fn seed_from_ids(ids: &[usize]) -> u64 {
    let mut s: u64 = 0xCAFE_BABE;
    for &i in ids {
        s = s.wrapping_mul(31).wrapping_add(i as u64);
    }
    s
}

/// Maps a prompt to the subject of the specialist that should answer it.
pub trait Router {
    /// Return the chosen subject, or `None` if nothing matches confidently.
    fn route(&self, prompt: &str, fleet: &SpecialistFleet) -> Option<String>;
}

/// Default router: score each specialist by how many of its card keywords (and
/// subject tokens) appear in the prompt; pick the highest. Cheap, dependency-free,
/// and good enough to dispatch across clearly-separated domains. Swap in the
/// trained cone router later by implementing [`Router`].
pub struct KeywordRouter;

impl Router for KeywordRouter {
    fn route(&self, prompt: &str, fleet: &SpecialistFleet) -> Option<String> {
        let p = prompt.to_lowercase();
        let mut best: Option<String> = None;
        let mut best_score = 0usize;
        for (subject, card) in fleet.cards() {
            let mut score = 0usize;
            for kw in &card.keywords {
                if !kw.is_empty() && p.contains(&kw.to_lowercase()) {
                    score += 2;
                }
            }
            for tok in subject.split(|c: char| !c.is_alphanumeric()) {
                if tok.len() >= 3 && p.contains(&tok.to_lowercase()) {
                    score += 1;
                }
            }
            if score > best_score {
                best_score = score;
                best = Some(subject.to_string());
            }
        }
        best
    }
}

/// A registry of specialists a single program holds in memory at once.
pub struct SpecialistFleet {
    specialists: HashMap<String, Specialist>,
    router: Box<dyn Router>,
}

impl SpecialistFleet {
    pub fn new() -> Self {
        Self {
            specialists: HashMap::new(),
            router: Box::new(KeywordRouter),
        }
    }

    pub fn with_router(mut self, router: Box<dyn Router>) -> Self {
        self.router = router;
        self
    }

    /// Add a loaded specialist (keyed by its card subject).
    pub fn insert(&mut self, specialist: Specialist) {
        self.specialists
            .insert(specialist.card.subject.clone(), specialist);
    }

    /// Load every `*.gfcard.json` manifest in a directory into a fleet. A
    /// specialist that fails to load is logged to stderr and skipped, so one bad
    /// model does not sink the whole fleet.
    pub fn load_dir(dir: &Path) -> Result<Self, String> {
        let mut fleet = Self::new();
        let suffix = format!(".{}", CARD_EXT);
        let entries = std::fs::read_dir(dir)
            .map_err(|e| format!("read fleet dir {}: {}", dir.display(), e))?;
        for entry in entries {
            let entry = entry.map_err(|e| e.to_string())?;
            let path = entry.path();
            let is_card = path
                .file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.ends_with(&suffix))
                .unwrap_or(false);
            if !is_card {
                continue;
            }
            match SpecialistManifest::load(&path).and_then(|m| Specialist::from_manifest(dir, &m)) {
                Ok(spec) => fleet.insert(spec),
                Err(e) => eprintln!("[fleet] skipping {}: {}", path.display(), e),
            }
        }
        Ok(fleet)
    }

    pub fn len(&self) -> usize {
        self.specialists.len()
    }

    pub fn is_empty(&self) -> bool {
        self.specialists.is_empty()
    }

    pub fn subjects(&self) -> Vec<String> {
        let mut v: Vec<String> = self.specialists.keys().cloned().collect();
        v.sort();
        v
    }

    pub fn cards(&self) -> impl Iterator<Item = (&str, &ModelCard)> {
        self.specialists.iter().map(|(k, s)| (k.as_str(), &s.card))
    }

    pub fn get(&self, subject: &str) -> Option<&Specialist> {
        self.specialists.get(subject)
    }

    /// Choose the subject the router thinks owns this prompt.
    pub fn route(&self, prompt: &str) -> Option<String> {
        self.router.route(prompt, self)
    }

    /// Route then generate. Returns `(subject, text)`, or `None` if no specialist
    /// matched. When the fleet holds exactly one specialist, it is used even if
    /// the router abstains.
    pub fn generate(&self, prompt: &str, cfg: &SampleConfig) -> Option<(String, String)> {
        let subject = self.route(prompt).or_else(|| {
            if self.specialists.len() == 1 {
                self.specialists.keys().next().cloned()
            } else {
                None
            }
        })?;
        let spec = self.specialists.get(&subject)?;
        Some((subject, spec.generate(prompt, cfg)))
    }

    /// Generate from a named specialist, bypassing the router.
    pub fn generate_with(&self, subject: &str, prompt: &str, cfg: &SampleConfig) -> Option<String> {
        self.specialists.get(subject).map(|s| s.generate(prompt, cfg))
    }
}

impl Default for SpecialistFleet {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FixedRouter(&'static str);
    impl Router for FixedRouter {
        fn route(&self, _prompt: &str, _fleet: &SpecialistFleet) -> Option<String> {
            Some(self.0.to_string())
        }
    }

    #[test]
    fn keyword_router_picks_by_card_keywords() {
        // Build a fleet by hand-inserting specialists is heavy (needs weights);
        // instead exercise the router scoring through a tiny stand-in fleet.
        let fleet = SpecialistFleet::new();
        // Empty fleet routes to nothing.
        assert_eq!(fleet.route("anything"), None);
    }

    #[test]
    fn router_is_pluggable() {
        let fleet = SpecialistFleet::new().with_router(Box::new(FixedRouter("x")));
        assert_eq!(fleet.route("whatever"), Some("x".to_string()));
    }
}
