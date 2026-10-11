//! Evaluation, LM calibration, router tuning and distillation.

use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use web_time::Instant;

use crate::brain::score_values;
use crate::calibration::{argmax, fit_temperature, metrics, probs_with_temperature, Metrics};
use crate::data::LabeledRow;
use crate::decide::Decider;
use crate::schema::Question;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Latency {
    pub n: usize,
    pub mean_ms: f64,
    pub p50_ms: f64,
    pub p90_ms: f64,
    pub p99_ms: f64,
}

impl Latency {
    pub fn from_samples(mut ms: Vec<f64>) -> Self {
        if ms.is_empty() {
            return Self::default();
        }
        ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let q = |p: f64| ms[((ms.len() as f64 - 1.0) * p).round() as usize];
        Self {
            n: ms.len(),
            mean_ms: ms.iter().sum::<f64>() / ms.len() as f64,
            p50_ms: q(0.5),
            p90_ms: q(0.9),
            p99_ms: q(0.99),
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct QuestionEval {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub n: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub head: Option<Metrics>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub probe: Option<Metrics>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lm: Option<Metrics>,
    /// Head if confident, else the fallback (probe if trained, else LM), at the router threshold.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub routed: Option<Metrics>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub router_threshold: Option<f32>,
    /// Share of rows the head answered under routing.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub head_share: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lm_temperature: Option<f32>,
    /// `tune_router` was asked for but this question had too few rows.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub router_not_tuned: bool,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct EvalReport {
    pub rows: usize,
    pub questions: Vec<QuestionEval>,
    pub head_latency: Latency,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub probe_latency: Option<Latency>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lm_latency: Option<Latency>,
}

#[derive(Clone, Debug, Default)]
pub struct EvalOptions {
    /// Score with the open model too (needs a scorer).
    pub use_lm: bool,
    /// Fit and store per-question LM temperatures from these rows.
    pub fit_lm_temperature: bool,
    /// Pick per-question router thresholds that maximise routed accuracy and store them.
    pub tune_router: bool,
    /// Questions with fewer labelled rows than this keep their current router
    /// threshold (a threshold picked on a few dozen rows is mostly noise).
    pub min_router_rows: usize,
}

const THRESHOLDS: &[f32] = &[0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.01];

/// Evaluate on labelled rows. With `fit_lm_temperature` / `tune_router` the
/// decider's brain is updated in place (save it afterwards).
///
/// Fit and evaluate on **different** rows for honest numbers: `gf-system1 eval`
/// fits on one half and reports on the other when both flags are set.
pub fn evaluate(
    d: &mut Decider,
    rows: &[LabeledRow],
    opts: &EvalOptions,
) -> Result<EvalReport, String> {
    let qs: Vec<Question> = d.brain.questions.questions.clone();
    let use_lm = opts.use_lm && d.has_scorer();
    let nq = qs.len();
    // Per question: (gold, head probs, lm raw logits)
    let mut gold: Vec<Vec<usize>> = vec![Vec::new(); nq];
    let mut head_p: Vec<Vec<Option<Vec<f32>>>> = vec![Vec::new(); nq];
    let mut lm_z: Vec<Vec<Option<Vec<f32>>>> = vec![Vec::new(); nq];
    let mut probe_p: Vec<Vec<Option<Vec<f32>>>> = vec![Vec::new(); nq];
    let use_probe = qs.iter().any(|q| d.probe_ready(&q.id));
    let (mut head_ms, mut lm_ms, mut probe_ms) = (Vec::new(), Vec::new(), Vec::new());

    for r in rows {
        let t0 = Instant::now();
        let feats = d
            .brain
            .featurizer
            .as_ref()
            .map(|f| f.transform(&d.head_text(&r.text)));
        let mut hp: Vec<Option<Vec<f32>>> = vec![None; nq];
        for &(qi, _, _) in &r.targets {
            hp[qi] = feats.as_ref().and_then(|x| d.head_probs(x, &qs[qi].id));
        }
        head_ms.push(t0.elapsed().as_secs_f64() * 1000.0);

        let mut pp: Vec<Option<Vec<f32>>> = vec![None; nq];
        if use_probe {
            let t1 = Instant::now();
            let x = d.probe_features(&r.text)?;
            probe_ms.push(t1.elapsed().as_secs_f64() * 1000.0);
            for &(qi, _, _) in &r.targets {
                if d.probe_ready(&qs[qi].id) {
                    pp[qi] = d.probe_probs(&x, &qs[qi].id);
                }
            }
        }
        let mut lz: Vec<Option<Vec<f32>>> = vec![None; nq];
        if use_lm {
            let asked: Vec<usize> = r.targets.iter().map(|(qi, _, _)| *qi).collect();
            let refs: Vec<&Question> = asked.iter().map(|&qi| &qs[qi]).collect();
            let t1 = Instant::now();
            let z = d.lm_logits(&r.text, &refs)?;
            lm_ms.push(t1.elapsed().as_secs_f64() * 1000.0);
            for (qi, zz) in asked.into_iter().zip(z) {
                lz[qi] = Some(zz);
            }
        }
        for &(qi, _, g) in &r.targets {
            gold[qi].push(g);
            head_p[qi].push(hp[qi].take());
            probe_p[qi].push(pp[qi].take());
            lm_z[qi].push(lz[qi].take());
        }
    }

    let mut out = EvalReport {
        rows: rows.len(),
        head_latency: Latency::from_samples(head_ms),
        probe_latency: if use_probe {
            Some(Latency::from_samples(probe_ms))
        } else {
            None
        },
        lm_latency: if use_lm {
            Some(Latency::from_samples(lm_ms))
        } else {
            None
        },
        ..Default::default()
    };

    for (qi, q) in qs.iter().enumerate() {
        let n = gold[qi].len();
        let mut qe = QuestionEval {
            id: q.id.clone(),
            kind: crate::schema::kind_name(&q.kind).into(),
            n,
            ..Default::default()
        };
        if n == 0 {
            out.questions.push(qe);
            continue;
        }
        let sv = score_values(&q.kind);
        let g = &gold[qi];
        let head_all: Option<Vec<Vec<f32>>> = head_p[qi].iter().cloned().collect();
        if let Some(h) = &head_all {
            qe.head = Some(metrics(h, g, sv.as_deref(), q.min_confidence));
        }
        let probe_all: Option<Vec<Vec<f32>>> = probe_p[qi].iter().cloned().collect();
        if let Some(p) = &probe_all {
            qe.probe = Some(metrics(p, g, sv.as_deref(), q.min_confidence));
        }
        let lm_all: Option<Vec<Vec<f32>>> = lm_z[qi].iter().cloned().collect();
        let mut lm_probs: Option<Vec<Vec<f32>>> = None;
        if let Some(z) = &lm_all {
            if opts.fit_lm_temperature {
                let t = fit_temperature(z, g);
                d.brain.lm_temperature.insert(q.id.clone(), t);
            }
            let t = d.brain.lm_temperature.get(&q.id).copied().unwrap_or(1.0);
            qe.lm_temperature = Some(t);
            let lp: Vec<Vec<f32>> = z.iter().map(|zz| probs_with_temperature(zz, t)).collect();
            qe.lm = Some(metrics(&lp, g, sv.as_deref(), q.min_confidence));
            lm_probs = Some(lp);
        }
        // What the decider falls back to when the head is unsure.
        let fallback = probe_all.or(lm_probs);
        if let (Some(h), Some(fb)) = (&head_all, &fallback) {
            let route = |thr: f32| -> (Vec<Vec<f32>>, usize) {
                let mut used = 0usize;
                let p = h
                    .iter()
                    .zip(fb)
                    .map(|(hh, ff)| {
                        if argmax(hh).1 >= thr {
                            used += 1;
                            hh.clone()
                        } else {
                            ff.clone()
                        }
                    })
                    .collect();
                (p, used)
            };
            if opts.tune_router && n < opts.min_router_rows {
                qe.router_not_tuned = true;
            } else if opts.tune_router {
                // Best accuracy; ties go to the lower threshold (more head = faster).
                let mut best = (f64::MIN, 0.0f32);
                for &thr in THRESHOLDS {
                    let (p, _) = route(thr);
                    let acc = metrics(&p, g, None, None).accuracy;
                    if acc > best.0 + 1e-9 {
                        best = (acc, thr);
                    }
                }
                d.brain.router.per_question.insert(q.id.clone(), best.1);
            }
            let thr = d.brain.router.threshold(&q.id);
            let (p, used) = route(thr);
            qe.routed = Some(metrics(&p, g, sv.as_deref(), q.min_confidence));
            qe.router_threshold = Some(thr);
            qe.head_share = Some(used as f64 / n as f64);
        }
        out.questions.push(qe);
    }
    Ok(out)
}

/// Distil: label states with the open model and return training rows that
/// carry the full calibrated distribution (`soft`) plus the argmax (`labels`).
/// Train heads on the result with `gf-system1 train`.
pub fn distill(
    d: &mut Decider,
    states: &[Value],
    question_ids: &[String],
    source: DistillSource,
) -> Result<Vec<Value>, String> {
    if !d.has_scorer() {
        return Err("distillation needs an open model (--lm)".into());
    }
    let qs: Vec<Question> = if question_ids.is_empty() {
        d.brain.questions.questions.clone()
    } else {
        question_ids
            .iter()
            .map(|id| {
                d.brain
                    .questions
                    .get(id)
                    .cloned()
                    .ok_or_else(|| format!("unknown question {id}"))
            })
            .collect::<Result<_, _>>()?
    };
    let refs: Vec<&Question> = qs.iter().collect();
    let mut out = Vec::with_capacity(states.len());
    for s in states {
        let text = crate::state::state_to_text(s);
        // (question, calibrated probabilities, backend)
        let per: Vec<(&Question, Vec<f32>, String)> = match source {
            DistillSource::Lm => {
                let z = d.lm_logits(&text, &refs)?;
                qs.iter()
                    .zip(z)
                    .map(|(q, zz)| {
                        let t = d.brain.lm_temperature.get(&q.id).copied().unwrap_or(1.0);
                        (q, probs_with_temperature(&zz, t), "lm".to_string())
                    })
                    .collect()
            }
            DistillSource::Best => {
                let resp = d.decide(&crate::decide::DecideRequest {
                    state: s.clone(),
                    questions: qs
                        .iter()
                        .map(|q| crate::decide::QuestionRef::Id(q.id.clone()))
                        .collect(),
                    min_confidence: None,
                    force: None,
                })?;
                qs.iter()
                    .zip(resp.answers)
                    .filter(|(_, a)| !a.distribution.is_empty())
                    .map(|(q, a)| (q, a.distribution.iter().map(|o| o.p).collect(), a.source))
                    .collect()
            }
        };
        let (mut labels, mut soft, mut sources) = (Map::new(), Map::new(), Map::new());
        for (q, p, src) in per {
            let outcomes = q.outcomes();
            let (best, _) = argmax(&p);
            let label = match &q.kind {
                crate::schema::QuestionKind::Score { min, .. } => json!(min + best as i32),
                crate::schema::QuestionKind::Noul => json!(best == 0),
                _ => json!(outcomes[best]),
            };
            labels.insert(q.id.clone(), label);
            let dist: Map<String, Value> = outcomes
                .iter()
                .zip(&p)
                .map(|(o, &pp)| (o.clone(), json!(((pp as f64) * 1e4).round() / 1e4)))
                .collect();
            soft.insert(q.id.clone(), Value::Object(dist));
            sources.insert(q.id.clone(), json!(src));
        }
        out.push(json!({"state": s, "labels": labels, "soft": soft, "backend": sources}));
    }
    Ok(out)
}

/// Which backend labels the states in [`distill`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DistillSource {
    /// Ask the open model every question (zero-shot).
    Lm,
    /// Whatever `decide` would answer: probe or head where trained, else the open model.
    Best,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brain::{System1Brain, TrainConfig};
    use crate::decide::tests::FakeScorer;

    #[test]
    fn eval_reports_head_lm_and_routes() {
        let (qs, rows) = crate::brain::tests::toy_rows();
        let mut b = System1Brain::new("toy", qs).unwrap();
        b.train_heads(&rows, &TrainConfig::default()).unwrap();
        let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut d = Decider::new(b).with_scorer(Box::new(FakeScorer { calls }));
        let rep = evaluate(
            &mut d,
            &rows,
            &EvalOptions {
                use_lm: true,
                fit_lm_temperature: true,
                tune_router: true,
                min_router_rows: 0,
            },
        )
        .unwrap();
        let route = &rep.questions[0];
        assert!(route.head.as_ref().unwrap().accuracy > 0.9);
        // The fake LM always says "Tech": ~50% on balanced data.
        assert!(route.lm.as_ref().unwrap().accuracy < 0.6);
        // Router learns to trust the head (threshold low, head share high).
        assert!(
            route.routed.as_ref().unwrap().accuracy >= route.head.as_ref().unwrap().accuracy - 1e-9
        );
        assert!(route.head_share.unwrap() > 0.9);
        assert!(rep.lm_latency.is_some());
    }

    #[test]
    fn router_needs_enough_rows() {
        let (qs, rows) = crate::brain::tests::toy_rows();
        let mut b = System1Brain::new("toy", qs).unwrap();
        b.train_heads(&rows, &TrainConfig::default()).unwrap();
        assert!(b.trained_on(&rows[0].text));
        assert!(b.trained_on(&format!("  {}  ", rows[0].text.to_uppercase())));
        assert!(!b.trained_on("a state nobody has seen"));
        let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut d = Decider::new(b).with_scorer(Box::new(FakeScorer { calls }));
        let rep = evaluate(
            &mut d,
            &rows,
            &EvalOptions {
                use_lm: true,
                tune_router: true,
                min_router_rows: 10_000,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(rep.questions.iter().all(|q| q.router_not_tuned || q.n == 0));
        assert!(d.brain.router.per_question.is_empty());
    }

    #[test]
    fn distill_emits_soft_rows_that_train() {
        let (qs, _) = crate::brain::tests::toy_rows();
        let b = System1Brain::new("toy", qs.clone()).unwrap();
        let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut d = Decider::new(b).with_scorer(Box::new(FakeScorer { calls }));
        let states: Vec<Value> = (0..12).map(|i| json!(format!("ticket {i}"))).collect();
        let rows = distill(&mut d, &states, &[], DistillSource::Lm).unwrap();
        assert_eq!(rows.len(), 12);
        assert_eq!(rows[0]["labels"]["route"], json!("Tech"));
        assert_eq!(rows[0]["labels"]["urgent"], json!(false));
        let jsonl: Vec<String> = rows.iter().map(|r| r.to_string()).collect();
        let (parsed, _) = crate::data::parse_labeled_jsonl(&jsonl.join("\n"), &qs).unwrap();
        assert_eq!(parsed.len(), 12);
    }
}
