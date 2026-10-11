//! A `growformer-system1` brain as the label model for "label model picks,
//! brain explains" (`--label-model agent/x.s1.json.gz`).
//!
//! Compared with [`crate::label_classifier::LabelClassifier`], the System 1
//! heads are temperature-calibrated on held-out rows, so `--label-min-conf`
//! means what it says: at 0.6, about 60 % of the labels it lets through are
//! right. The same brain also answers the project's other typed questions
//! (`gf-system1 decide`).

use std::path::Path;

use growformer_system1::schema::{Question, QuestionKind};
use growformer_system1::{Decider, System1Brain};

use crate::label_classifier::{coarse_sentiment, LabelScore, Labeler};

/// A System 1 Choice question used as the label picker, optionally gated by a
/// coarse sentiment question (best bucket first, then the best fine label in it).
pub struct System1Labeler {
    decider: Decider,
    fine: String,
    fine_labels: Vec<String>,
    coarse: Option<String>,
    coarse_labels: Vec<String>,
}

fn choice_options(q: &Question) -> Option<&[String]> {
    match &q.kind {
        QuestionKind::Choice { options } => Some(options),
        _ => None,
    }
}

/// A Choice question whose options are distinct coarse sentiment buckets.
fn is_coarse_sentiment(q: &Question) -> bool {
    let Some(opts) = choice_options(q) else {
        return false;
    };
    let mut buckets: Vec<&str> = Vec::new();
    for o in opts {
        match coarse_sentiment(o) {
            Some(b) if !buckets.contains(&b) => buckets.push(b),
            _ => return false,
        }
    }
    buckets.len() >= 2
}

fn is_sentiment(labels: &[String]) -> bool {
    labels
        .iter()
        .any(|l| l.contains("positive") || l.contains("negative"))
}

impl System1Labeler {
    /// `path[#fine[,coarse]]`.
    pub fn load_spec(spec: &str) -> Result<Self, String> {
        let (path, frag) = match spec.split_once('#') {
            Some((p, f)) => (p, Some(f)),
            None => (spec, None),
        };
        let (fine, coarse) = match frag {
            Some(f) => {
                let mut it = f.split(',').map(str::trim).filter(|s| !s.is_empty());
                (it.next(), it.next())
            }
            None => (None, None),
        };
        let brain = System1Brain::load(Path::new(path))?;
        Self::from_brain(brain, fine, coarse)
    }

    /// Use question `fine` (or auto: the trained Choice question with the most
    /// options) and optional `coarse` (or auto for sentiment schemes).
    pub fn from_brain(
        brain: System1Brain,
        fine: Option<&str>,
        coarse: Option<&str>,
    ) -> Result<Self, String> {
        let trained = |q: &&Question| brain.heads.contains_key(&q.id);
        let fine_q = match fine {
            Some(id) => brain
                .questions
                .get(id)
                .ok_or_else(|| format!("System 1 brain has no question `{id}`"))?,
            None => brain
                .questions
                .questions
                .iter()
                .filter(trained)
                .filter(|q| choice_options(q).is_some())
                .max_by_key(|q| q.n_outcomes())
                .ok_or("System 1 brain has no trained choice question to use as labels")?,
        };
        let fine_labels = choice_options(fine_q)
            .ok_or_else(|| format!("question `{}` is not a choice question", fine_q.id))?
            .to_vec();
        if !brain.heads.contains_key(&fine_q.id) {
            return Err(format!(
                "question `{}` has no trained head (gf-system1 train)",
                fine_q.id
            ));
        }
        let coarse_q = match coarse {
            Some(id) => {
                let q = brain
                    .questions
                    .get(id)
                    .ok_or_else(|| format!("System 1 brain has no question `{id}`"))?;
                if !is_coarse_sentiment(q) || !brain.heads.contains_key(id) {
                    return Err(format!(
                        "coarse question `{id}` must be a trained choice question whose options are sentiment buckets"
                    ));
                }
                Some(q)
            }
            None if is_sentiment(&fine_labels) => brain
                .questions
                .questions
                .iter()
                .filter(trained)
                .filter(|q| q.id != fine_q.id)
                .find(|q| is_coarse_sentiment(q)),
            None => None,
        };
        let fine = fine_q.id.clone();
        let (coarse, coarse_labels) = match coarse_q {
            Some(q) => (
                Some(q.id.clone()),
                choice_options(q).unwrap_or_default().to_vec(),
            ),
            None => (None, Vec::new()),
        };
        Ok(Self {
            decider: Decider::new(brain),
            fine,
            fine_labels,
            coarse,
            coarse_labels,
        })
    }

    fn feats(&self, text: &str) -> growformer_system1::features::SparseVec {
        match &self.decider.brain.featurizer {
            Some(f) => f.transform(&self.decider.head_text(text)),
            None => Vec::new(),
        }
    }

    fn scored(labels: &[String], probs: Vec<f32>) -> Vec<LabelScore> {
        let mut out: Vec<LabelScore> = labels
            .iter()
            .zip(probs)
            .map(|(l, prob)| LabelScore {
                label: l.clone(),
                prob,
            })
            .collect();
        out.sort_by(|a, b| {
            b.prob
                .partial_cmp(&a.prob)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        out
    }

    pub fn fine_question(&self) -> &str {
        &self.fine
    }

    pub fn coarse_question(&self) -> Option<&str> {
        self.coarse.as_deref()
    }
}

impl Labeler for System1Labeler {
    fn predict(&self, text: &str) -> Vec<LabelScore> {
        let x = self.feats(text);
        let p = self
            .decider
            .head_probs(&x, &self.fine)
            .unwrap_or_else(|| vec![1.0 / self.fine_labels.len() as f32; self.fine_labels.len()]);
        Self::scored(&self.fine_labels, p)
    }

    fn predict_top(&self, text: &str) -> LabelScore {
        let x = self.feats(text);
        let fine = Self::scored(
            &self.fine_labels,
            self.decider
                .head_probs(&x, &self.fine)
                .unwrap_or_else(|| vec![0.0; self.fine_labels.len()]),
        );
        if let Some(cq) = &self.coarse {
            if let Some(cp) = self.decider.head_probs(&x, cq) {
                let bucket = Self::scored(&self.coarse_labels, cp)
                    .into_iter()
                    .next()
                    .expect("≥2 buckets");
                let b = coarse_sentiment(&bucket.label);
                if let Some(best) = fine.iter().find(|s| coarse_sentiment(&s.label) == b) {
                    return LabelScore {
                        label: best.label.clone(),
                        prob: bucket.prob,
                    };
                }
            }
        }
        fine.into_iter().next().expect("≥2 labels")
    }

    fn similarity(&self, a: &str, b: &str) -> f32 {
        growformer_system1::features::cosine(&self.feats(a), &self.feats(b))
    }

    fn is_sentiment_scheme(&self) -> bool {
        is_sentiment(&self.fine_labels)
    }

    fn describe(&self) -> String {
        let b = &self.decider.brain;
        let acc = |q: &str| {
            b.reports
                .get(q)
                .map(|r| format!(", held-out {:.1}%", r.calibrated.accuracy * 100.0))
                .unwrap_or_default()
        };
        match &self.coarse {
            Some(c) => format!(
                "System 1 brain `{}`: labels from `{}` ({} labels{}), bucket from `{}` ({} buckets{})",
                b.name,
                self.fine,
                self.fine_labels.len(),
                acc(&self.fine),
                c,
                self.coarse_labels.len(),
                acc(c)
            ),
            None => format!(
                "System 1 brain `{}`: labels from `{}` ({} labels{})",
                b.name,
                self.fine,
                self.fine_labels.len(),
                acc(&self.fine)
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use growformer_system1::data::parse_labeled_jsonl;
    use growformer_system1::{QuestionSet, TrainConfig};

    fn brain() -> System1Brain {
        let qs = QuestionSet::parse(
            r#"
[[questions]]
id = "sentiment"
text = "Overall sentiment?"
type = "choice"
options = ["positive", "negative"]
source = { field = "semantic_intent", map = { positive_strong = "positive", positive_mild = "positive", negative_strong = "negative", negative_mild = "negative" } }

[[questions]]
id = "intent"
text = "Fine label?"
type = "choice"
options = ["positive_strong", "positive_mild", "negative_strong", "negative_mild"]
source = { field = "semantic_intent" }
"#,
        )
        .unwrap();
        let mut jsonl = String::new();
        let rows = [
            (
                "moon soon, massive breakout, all time high",
                "positive_strong",
            ),
            ("decent gains, slightly up today", "positive_mild"),
            ("total crash, rug pull, everything lost", "negative_strong"),
            ("a bit down today, small dip", "negative_mild"),
        ];
        for i in 0..12 {
            for (t, l) in rows {
                jsonl.push_str(&format!(
                    "{{\"text\":\"{t} {i}\",\"semantic_intent\":\"{l}\"}}\n"
                ));
            }
        }
        let (rows, _) = parse_labeled_jsonl(&jsonl, &qs).unwrap();
        let mut b = System1Brain::new("t", qs).unwrap();
        b.train_heads(&rows, &TrainConfig::default()).unwrap();
        b
    }

    #[test]
    fn auto_picks_fine_and_coarse_questions() {
        let l = System1Labeler::from_brain(brain(), None, None).unwrap();
        assert_eq!(l.fine_question(), "intent");
        assert_eq!(l.coarse_question(), Some("sentiment"));
        assert!(l.is_sentiment_scheme());
        let top = l.predict_top("rug pull crash, everything lost");
        assert_eq!(coarse_sentiment(&top.label), Some("NEG"), "{top:?}");
        assert!(top.prob > 0.5 && top.prob <= 1.0);
        assert_eq!(l.predict("moon breakout").len(), 4);
        assert!(l.similarity("rug pull crash", "crash rug pull") > 0.5);
    }

    #[test]
    fn spec_fragment_and_errors() {
        let dir = std::env::temp_dir().join(format!("gf-llm-s1lab-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("t.s1.json.gz");
        brain().save(&p).unwrap();
        let s = p.to_string_lossy();
        let l = System1Labeler::load_spec(&format!("{s}#intent")).unwrap();
        assert_eq!(
            l.coarse_question(),
            Some("sentiment"),
            "auto coarse for sentiment"
        );
        let l = System1Labeler::load_spec(&format!("{s}#sentiment")).unwrap();
        assert_eq!(l.fine_question(), "sentiment");
        assert!(System1Labeler::load_spec(&format!("{s}#nope")).is_err());
        assert!(System1Labeler::load_spec(&format!("{s}#intent,intent")).is_err());
        let boxed =
            crate::label_classifier::load_labeler(Path::new(&format!("{s}#intent"))).unwrap();
        assert!(
            boxed.describe().contains("System 1"),
            "{}",
            boxed.describe()
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
