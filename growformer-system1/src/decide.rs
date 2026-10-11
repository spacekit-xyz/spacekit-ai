//! Request → typed answers. Routes each question to its fast head when the
//! head is confident, otherwise to the open-model scorer (if attached).

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use web_time::Instant;

use crate::brain::{score_values, System1Brain};
use crate::calibration::{argmax, probs_with_temperature};
use crate::schema::{kind_name, Question, QuestionKind};
use crate::state::{state_to_text, truncate_middle};

/// Anything that can score a question's outcomes for a state in one pass
/// (the candle open-model backend implements this; tests use fakes).
pub trait OptionScorer {
    fn describe(&self) -> String;
    /// Raw logits over each question's outcomes, in order. Implementations
    /// should process the shared state once and reuse it across questions.
    fn score(&mut self, state_text: &str, questions: &[&Question])
        -> Result<Vec<Vec<f32>>, String>;
    /// Hidden-state features for probe heads: one pass over the state, no
    /// question. `layer_frac` picks the tapped layer (0 = first, 1 = last).
    fn embed(&mut self, _state_text: &str, _layer_frac: f32) -> Result<Vec<f32>, String> {
        Err("this open model does not expose hidden states".into())
    }
    /// Identifies the model whose features a probe was trained on.
    fn model_key(&self) -> String {
        self.describe()
    }
}

/// A question by id (from the brain's schema) or defined inline.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum QuestionRef {
    Id(String),
    Inline(Question),
}

/// Force a backend for every question (useful for evaluation).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Backend {
    Head,
    Probe,
    Lm,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DecideRequest {
    /// Text, JSON object or array.
    pub state: Value,
    /// Empty = every question in the brain's schema.
    #[serde(default)]
    pub questions: Vec<QuestionRef>,
    /// Default abstain threshold for questions without their own `min_confidence`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_confidence: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub force: Option<Backend>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OutcomeProb {
    pub outcome: String,
    pub p: f32,
}

/// One typed answer.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Answer {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    /// Choice → option string; Score → expected value (probability-weighted
    /// mean); Noul → boolean (`probability >= 0.5`). `null` when unanswerable.
    pub value: Value,
    /// Noul: P(true).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub probability: Option<f32>,
    /// Score: standard deviation of the distribution.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub std: Option<f32>,
    /// Probability of the most likely outcome (calibrated).
    pub confidence: f32,
    /// True when confidence is below the question's / request's threshold,
    /// or no backend could answer.
    pub abstain: bool,
    pub distribution: Vec<OutcomeProb>,
    /// `"head"`, `"probe"`, `"lm"` or `"none"`.
    pub source: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DecideResponse {
    pub brain: String,
    pub answers: Vec<Answer>,
    pub latency_ms: f64,
}

/// Build an answer from calibrated probabilities.
pub fn make_answer(q: &Question, probs: &[f32], source: &str, min_conf: Option<f32>) -> Answer {
    let outcomes = q.outcomes();
    let (best, conf) = argmax(probs);
    let distribution = outcomes
        .iter()
        .zip(probs)
        .map(|(o, &p)| OutcomeProb {
            outcome: o.clone(),
            p: (p * 1e4).round() / 1e4,
        })
        .collect();
    let (value, probability, std) = match &q.kind {
        QuestionKind::Choice { options } => (json!(options[best]), None, None),
        QuestionKind::Score { .. } => {
            let vals = score_values(&q.kind).unwrap_or_default();
            let mean: f32 = probs.iter().zip(&vals).map(|(p, v)| p * v).sum();
            let var: f32 = probs
                .iter()
                .zip(&vals)
                .map(|(p, v)| p * (v - mean).powi(2))
                .sum();
            (
                json!(((mean as f64) * 1000.0).round() / 1000.0),
                None,
                Some((var.sqrt() * 1000.0).round() / 1000.0),
            )
        }
        QuestionKind::Noul => (
            json!(probs[0] >= 0.5),
            Some((probs[0] * 1e4).round() / 1e4),
            None,
        ),
    };
    let threshold = q.min_confidence.or(min_conf).unwrap_or(0.0);
    Answer {
        id: q.id.clone(),
        kind: kind_name(&q.kind).into(),
        value,
        probability,
        std,
        confidence: (conf * 1e4).round() / 1e4,
        abstain: conf < threshold,
        distribution,
        source: source.into(),
        error: None,
    }
}

fn unanswerable(q: &Question, why: &str) -> Answer {
    Answer {
        id: q.id.clone(),
        kind: kind_name(&q.kind).into(),
        value: Value::Null,
        probability: None,
        std: None,
        confidence: 0.0,
        abstain: true,
        distribution: Vec::new(),
        source: "none".into(),
        error: Some(why.into()),
    }
}

/// Brain + optional open-model scorer.
pub struct Decider {
    pub brain: System1Brain,
    scorer: Option<Box<dyn OptionScorer + Send>>,
}

impl Decider {
    pub fn new(brain: System1Brain) -> Self {
        Self {
            brain,
            scorer: None,
        }
    }

    pub fn with_scorer(mut self, scorer: Box<dyn OptionScorer + Send>) -> Self {
        self.scorer = Some(scorer);
        self
    }

    pub fn set_scorer(&mut self, scorer: Box<dyn OptionScorer + Send>) {
        self.scorer = Some(scorer);
    }

    pub fn has_scorer(&self) -> bool {
        self.scorer.is_some()
    }

    pub fn scorer_description(&self) -> Option<String> {
        self.scorer.as_ref().map(|s| s.describe())
    }

    /// Heads' state text (truncated for the featurizer).
    pub fn head_text(&self, state_text: &str) -> String {
        truncate_middle(state_text, self.brain.max_state_chars)
    }

    /// Calibrated head probabilities, or `None` if the question has no head.
    pub fn head_probs(&self, feats: &crate::features::SparseVec, qid: &str) -> Option<Vec<f32>> {
        let head = self.brain.heads.get(qid)?;
        let t = self.brain.head_temperature.get(qid).copied().unwrap_or(1.0);
        Some(probs_with_temperature(&head.logits(feats), t))
    }

    /// Raw open-model logits for several questions (shared state).
    pub fn lm_logits(
        &mut self,
        state_text: &str,
        qs: &[&Question],
    ) -> Result<Vec<Vec<f32>>, String> {
        let s = self
            .scorer
            .as_mut()
            .ok_or("no open-model scorer attached")?;
        let out = s.score(state_text, qs)?;
        if out.len() != qs.len() {
            return Err("scorer returned the wrong number of questions".into());
        }
        for (q, z) in qs.iter().zip(&out) {
            if z.len() != q.n_outcomes() {
                return Err(format!("scorer returned {} logits for {}", z.len(), q.id));
            }
        }
        Ok(out)
    }

    /// True when the brain has a probe head for `qid`, an open model is
    /// attached, and it is the model the probe was trained on.
    pub fn probe_ready(&self, qid: &str) -> bool {
        match (&self.brain.probe, &self.scorer) {
            (Some(p), Some(s)) => p.heads.contains_key(qid) && p.model_key == s.model_key(),
            _ => false,
        }
    }

    /// Standardised probe features for a state (one open-model pass).
    pub fn probe_features(&mut self, state_text: &str) -> Result<Vec<f32>, String> {
        let p = self.brain.probe.as_ref().ok_or("brain has no probe")?;
        let layer = p.layer_frac;
        let s = self
            .scorer
            .as_mut()
            .ok_or("no open-model scorer attached")?;
        let raw = s.embed(state_text, layer)?;
        let p = self.brain.probe.as_ref().expect("checked above");
        if raw.len() != p.dim {
            return Err(format!(
                "probe expects {} features, the model gave {}",
                p.dim,
                raw.len()
            ));
        }
        Ok(p.standardize(&raw))
    }

    /// The attached model's identity (see [`OptionScorer::model_key`]).
    pub fn scorer_model_key(&self) -> Option<String> {
        self.scorer.as_ref().map(|s| s.model_key())
    }

    /// Raw (unstandardised) probe features from the attached model.
    pub fn embed_raw(&mut self, state_text: &str, layer_frac: f32) -> Result<Vec<f32>, String> {
        self.scorer
            .as_mut()
            .ok_or("no open-model scorer attached")?
            .embed(state_text, layer_frac)
    }

    /// Calibrated probe probabilities for standardised features.
    pub fn probe_probs(&self, x_std: &[f32], qid: &str) -> Option<Vec<f32>> {
        self.brain.probe.as_ref()?.probs(qid, x_std)
    }

    fn resolve(&self, refs: &[QuestionRef]) -> Result<Vec<Question>, String> {
        if refs.is_empty() {
            return Ok(self.brain.questions.questions.clone());
        }
        refs.iter()
            .map(|r| match r {
                QuestionRef::Id(id) => self
                    .brain
                    .questions
                    .get(id)
                    .cloned()
                    .ok_or_else(|| format!("unknown question id {id:?}")),
                QuestionRef::Inline(q) => {
                    q.validate()?;
                    Ok(q.clone())
                }
            })
            .collect()
    }

    /// Answer every question about one state.
    pub fn decide(&mut self, req: &DecideRequest) -> Result<DecideResponse, String> {
        let t0 = Instant::now();
        let questions = self.resolve(&req.questions)?;
        let state_text = state_to_text(&req.state);
        let feats = self
            .brain
            .featurizer
            .as_ref()
            .map(|f| f.transform(&self.head_text(&state_text)));

        let mut answers: Vec<Option<Answer>> = vec![None; questions.len()];
        let mut to_probe: Vec<usize> = Vec::new();
        let mut to_lm: Vec<usize> = Vec::new();
        for (i, q) in questions.iter().enumerate() {
            // An inline question that reuses a trained id but changes its schema
            // must not be answered by that trained head or probe.
            let trained_ok = self
                .brain
                .questions
                .get(&q.id)
                .map(|bq| bq.kind == q.kind)
                .unwrap_or(false);
            let head = if trained_ok {
                feats.as_ref().and_then(|x| self.head_probs(x, &q.id))
            } else {
                None
            };
            let probe_ok = trained_ok && self.probe_ready(&q.id);
            // Fallback when the head is missing or unsure: the probe if there is
            // one, else asking the open model.
            let fallback = if probe_ok {
                Some(Backend::Probe)
            } else if self.has_scorer() {
                Some(Backend::Lm)
            } else {
                None
            };
            let backend = match req.force {
                Some(Backend::Head) if head.is_some() => Backend::Head,
                Some(Backend::Probe) if probe_ok => Backend::Probe,
                Some(Backend::Lm) if self.has_scorer() => Backend::Lm,
                _ => match (&head, fallback) {
                    (Some(p), Some(fb)) => {
                        if argmax(p).1 >= self.brain.router.threshold(&q.id) {
                            Backend::Head
                        } else {
                            fb
                        }
                    }
                    (Some(_), None) => Backend::Head,
                    (None, Some(fb)) => fb,
                    (None, None) => {
                        answers[i] = Some(unanswerable(
                            q,
                            "no trained head for this question and no open model attached",
                        ));
                        continue;
                    }
                },
            };
            match backend {
                Backend::Head => {
                    answers[i] = Some(make_answer(
                        q,
                        head.as_ref().expect("head chosen only when present"),
                        "head",
                        req.min_confidence,
                    ))
                }
                Backend::Probe => to_probe.push(i),
                Backend::Lm => to_lm.push(i),
            }
        }

        if !to_probe.is_empty() {
            match self.probe_features(&state_text) {
                Ok(x) => {
                    for &i in &to_probe {
                        let q = &questions[i];
                        let p = self.probe_probs(&x, &q.id).expect("probe_ready checked");
                        answers[i] = Some(make_answer(q, &p, "probe", req.min_confidence));
                    }
                }
                Err(e) => {
                    for &i in &to_probe {
                        answers[i] = Some(unanswerable(
                            &questions[i],
                            &format!("open model failed: {e}"),
                        ));
                    }
                }
            }
        }

        if !to_lm.is_empty() {
            let qs: Vec<&Question> = to_lm.iter().map(|&i| &questions[i]).collect();
            match self.lm_logits(&state_text, &qs) {
                Ok(logits) => {
                    for (&i, z) in to_lm.iter().zip(logits) {
                        let q = &questions[i];
                        let t = self.brain.lm_temperature.get(&q.id).copied().unwrap_or(1.0);
                        answers[i] = Some(make_answer(
                            q,
                            &probs_with_temperature(&z, t),
                            "lm",
                            req.min_confidence,
                        ));
                    }
                }
                Err(e) => {
                    for &i in &to_lm {
                        answers[i] = Some(unanswerable(
                            &questions[i],
                            &format!("open model failed: {e}"),
                        ));
                    }
                }
            }
        }

        Ok(DecideResponse {
            brain: self.brain.name.clone(),
            answers: answers
                .into_iter()
                .map(|a| a.expect("every question answered"))
                .collect(),
            latency_ms: t0.elapsed().as_secs_f64() * 1000.0,
        })
    }

    /// Convenience: JSON in, JSON out (used by the server and wasm bindings).
    pub fn decide_json(&mut self, request_json: &str) -> Result<String, String> {
        let req: DecideRequest =
            serde_json::from_str(request_json).map_err(|e| format!("bad request: {e}"))?;
        let resp = self.decide(&req)?;
        serde_json::to_string(&resp).map_err(|e| e.to_string())
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::brain::TrainConfig;

    /// Fake open model: prefers the last outcome, counts calls.
    pub struct FakeScorer {
        pub calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }

    impl OptionScorer for FakeScorer {
        fn describe(&self) -> String {
            "fake".into()
        }
        fn embed(&mut self, s: &str, _layer: f32) -> Result<Vec<f32>, String> {
            Ok(crate::probe::toy_embed(s, 64))
        }
        fn score(&mut self, _s: &str, qs: &[&Question]) -> Result<Vec<Vec<f32>>, String> {
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(qs
                .iter()
                .map(|q| {
                    let k = q.n_outcomes();
                    (0..k).map(|i| if i + 1 == k { 3.0 } else { 0.0 }).collect()
                })
                .collect())
        }
    }

    fn trained() -> System1Brain {
        let (qs, rows) = crate::brain::tests::toy_rows();
        let mut b = System1Brain::new("toy", qs).unwrap();
        b.train_heads(&rows, &TrainConfig::default()).unwrap();
        b
    }

    #[test]
    fn heads_answer_typed_json() {
        let mut d = Decider::new(trained());
        let out = d
            .decide_json(r#"{"state":"I was charged twice, refund ASAP this is urgent"}"#)
            .unwrap();
        let resp: DecideResponse = serde_json::from_str(&out).unwrap();
        assert_eq!(resp.answers.len(), 2);
        let route = &resp.answers[0];
        assert_eq!(route.kind, "choice");
        assert_eq!(route.value, json!("Billing"));
        assert_eq!(route.source, "head");
        let s: f32 = route.distribution.iter().map(|o| o.p).sum();
        assert!((s - 1.0).abs() < 1e-3);
        let urgent = &resp.answers[1];
        assert_eq!(urgent.kind, "noul");
        assert!(urgent.probability.is_some());
    }

    #[test]
    fn router_sends_low_confidence_and_new_questions_to_lm() {
        let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut b = trained();
        b.router.head_min_confidence = 1.01; // heads never confident enough
        let mut d = Decider::new(b).with_scorer(Box::new(FakeScorer {
            calls: calls.clone(),
        }));
        let req: DecideRequest = serde_json::from_value(json!({
            "state": {"subject": "app crashes", "body": "since the update"},
            "questions": [
                "route",
                {"id": "sev", "text": "Severity?", "type": "score", "min": 0, "max": 3}
            ]
        }))
        .unwrap();
        let resp = d.decide(&req).unwrap();
        assert!(resp.answers.iter().all(|a| a.source == "lm"));
        assert_eq!(
            calls.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "one shared-state pass"
        );
        let sev = &resp.answers[1];
        assert_eq!(sev.kind, "score");
        assert!(sev.value.as_f64().unwrap() > 2.0);
        assert!(sev.std.is_some());
    }

    #[test]
    fn no_backend_means_abstain_not_error() {
        let mut d = Decider::new(trained());
        let req: DecideRequest = serde_json::from_value(json!({
            "state": "hello",
            "questions": [{"id": "new", "text": "Is it a greeting?", "type": "noul"}],
            "min_confidence": 0.9
        }))
        .unwrap();
        let resp = d.decide(&req).unwrap();
        assert!(resp.answers[0].abstain);
        assert_eq!(resp.answers[0].source, "none");
        assert!(resp.answers[0].error.is_some());
    }

    #[test]
    fn probe_answers_when_trained_and_router_prefers_it() {
        use crate::probe::{toy_embed, Probe, ProbeTrainConfig};
        let (qs, rows) = crate::brain::tests::toy_rows();
        let mut b = System1Brain::new("toy", qs.clone()).unwrap();
        b.train_heads(&rows, &TrainConfig::default()).unwrap();
        let feats: Vec<Vec<f32>> = rows.iter().map(|r| toy_embed(&r.text, 64)).collect();
        b.probe = Some(
            Probe::train(
                &qs,
                &rows,
                &feats,
                "fake".into(),
                0.5,
                &ProbeTrainConfig::default(),
            )
            .unwrap(),
        );
        b.router.per_question.insert("route".into(), 1.01);
        let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut d = Decider::new(b).with_scorer(Box::new(FakeScorer {
            calls: calls.clone(),
        }));
        let out = d
            .decide(&DecideRequest {
                state: json!("card declined"),
                questions: vec![QuestionRef::Id("route".into())],
                min_confidence: None,
                force: None,
            })
            .unwrap();
        assert_eq!(out.answers[0].source, "probe");
        assert_eq!(out.answers[0].value, json!("Billing"));
        // The probe needs no per-question scoring call.
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 0);
        // A probe trained on another model is ignored.
        d.brain.probe.as_mut().unwrap().model_key = "other".into();
        let out = d
            .decide(&DecideRequest {
                state: json!("card declined"),
                questions: vec![QuestionRef::Id("route".into())],
                min_confidence: None,
                force: None,
            })
            .unwrap();
        assert_eq!(out.answers[0].source, "lm");
    }
}
