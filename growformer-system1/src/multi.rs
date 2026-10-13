//! Many brains, one open model.
//!
//! A brain's heads, probes and router are a few MB and never change while
//! serving, so they're shared read-only (`Arc`). The open model is the heavy
//! part (GBs): brains that name the same model share one [`ModelPool`], whose
//! workers are copies of the model that share its weights but each have their
//! own KV cache, so requests run in parallel without loading the model twice.
//!
//! The pool also caches probe features per state, so asking several brains
//! about the same state costs one model pass.
//!
//! ```text
//! request ─► Registry[name] ─► Arc<System1Brain>  (heads: answered at once, no lock)
//!                                   │ probe / open model
//!                                   ▼
//!                             ModelPool ─► worker k (shared weights, own cache)
//! ```

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use web_time::Instant;

use crate::brain::System1Brain;
use crate::decide::{decide_with, DecideRequest, DecideResponse, OptionScorer};
use crate::schema::Question;

type Worker = Mutex<Box<dyn OptionScorer + Send>>;

/// Least-recently-inserted cache of probe features, keyed by state + layer.
struct FeatureCache {
    cap: usize,
    map: HashMap<(u64, u32), Arc<Vec<f32>>>,
    order: VecDeque<(u64, u32)>,
}

impl FeatureCache {
    fn get(&self, k: &(u64, u32)) -> Option<Arc<Vec<f32>>> {
        self.map.get(k).cloned()
    }

    fn insert(&mut self, k: (u64, u32), v: Arc<Vec<f32>>) {
        if self.cap == 0 || self.map.contains_key(&k) {
            return;
        }
        while self.map.len() >= self.cap {
            match self.order.pop_front() {
                Some(old) => {
                    self.map.remove(&old);
                }
                None => break,
            }
        }
        self.order.push_back(k);
        self.map.insert(k, v);
    }
}

/// One loaded open model shared by every brain that uses it.
pub struct ModelPool {
    key: String,
    description: String,
    workers: Vec<Worker>,
    next: AtomicUsize,
    features: Mutex<FeatureCache>,
}

/// Default number of recent states whose probe features are kept.
pub const DEFAULT_FEATURE_CACHE: usize = 1024;

impl ModelPool {
    /// Wrap a loaded scorer, adding up to `workers - 1` copies that share its
    /// weights (fewer if the backend can't share).
    pub fn new(first: Box<dyn OptionScorer + Send>, workers: usize) -> Self {
        let key = first.model_key();
        let description = first.describe();
        let mut ws: Vec<Box<dyn OptionScorer + Send>> = Vec::new();
        for _ in 1..workers.max(1) {
            match first.fork() {
                Some(w) => ws.push(w),
                None => break,
            }
        }
        ws.insert(0, first);
        Self {
            key,
            description,
            workers: ws.into_iter().map(Mutex::new).collect(),
            next: AtomicUsize::new(0),
            features: Mutex::new(FeatureCache {
                cap: DEFAULT_FEATURE_CACHE,
                map: HashMap::new(),
                order: VecDeque::new(),
            }),
        }
    }

    /// Keep probe features for this many recent states (0 disables the cache).
    pub fn with_feature_cache(self, states: usize) -> Self {
        if let Ok(mut c) = self.features.lock() {
            c.cap = states;
        }
        self
    }

    pub fn model_key(&self) -> &str {
        &self.key
    }

    pub fn describe(&self) -> &str {
        &self.description
    }

    pub fn workers(&self) -> usize {
        self.workers.len()
    }

    /// Run `f` on a free worker; wait for the next one in turn if all are busy.
    fn with<R>(&self, f: impl FnOnce(&mut dyn OptionScorer) -> R) -> Result<R, String> {
        let n = self.workers.len();
        let start = self.next.fetch_add(1, Ordering::Relaxed);
        for i in 0..n {
            if let Ok(mut w) = self.workers[(start + i) % n].try_lock() {
                return Ok(f(w.as_mut()));
            }
        }
        let mut w = self.workers[start % n]
            .lock()
            .map_err(|_| "open-model worker poisoned by an earlier panic".to_string())?;
        Ok(f(w.as_mut()))
    }

    /// A per-request handle that implements [`OptionScorer`].
    pub fn handle(self: &Arc<Self>) -> PoolScorer {
        PoolScorer(self.clone())
    }
}

/// [`OptionScorer`] backed by a shared [`ModelPool`].
pub struct PoolScorer(Arc<ModelPool>);

impl OptionScorer for PoolScorer {
    fn describe(&self) -> String {
        self.0.description.clone()
    }

    fn model_key(&self) -> String {
        self.0.key.clone()
    }

    fn score(&mut self, state_text: &str, qs: &[&Question]) -> Result<Vec<Vec<f32>>, String> {
        self.0.with(|w| w.score(state_text, qs))?
    }

    fn embed(&mut self, state_text: &str, layer_frac: f32) -> Result<Vec<f32>, String> {
        let k = (
            crate::brain::row_fingerprint(state_text),
            layer_frac.to_bits(),
        );
        if let Some(v) = self.0.features.lock().ok().and_then(|c| c.get(&k)) {
            return Ok(v.as_ref().clone());
        }
        let v = self.0.with(|w| w.embed(state_text, layer_frac))??;
        if let Ok(mut c) = self.0.features.lock() {
            c.insert(k, Arc::new(v.clone()));
        }
        Ok(v)
    }

    fn fork(&self) -> Option<Box<dyn OptionScorer + Send>> {
        Some(Box::new(PoolScorer(self.0.clone())))
    }
}

/// A loaded brain and the model pool it uses (if any).
pub struct BrainEntry {
    pub name: String,
    pub brain: Arc<System1Brain>,
    pub pool: Option<Arc<ModelPool>>,
    /// Where it was loaded from (for reloads); `None` for in-memory brains.
    pub source: Option<String>,
}

impl BrainEntry {
    pub fn decide(&self, req: &DecideRequest) -> Result<DecideResponse, String> {
        let mut handle = self.pool.as_ref().map(|p| p.handle());
        decide_with(
            &self.brain,
            handle.as_mut().map(|h| h as &mut dyn OptionScorer),
            req,
        )
    }

    pub fn info(&self) -> Value {
        json!({
            "name": self.name,
            "brain": self.brain.name,
            "questions": self.brain.questions.questions.iter().map(|q| q.id.clone()).collect::<Vec<_>>(),
            "heads": self.brain.heads.keys().cloned().collect::<Vec<_>>(),
            "probes": self.brain.probe.as_ref().map(|p| p.heads.keys().cloned().collect::<Vec<_>>()).unwrap_or_default(),
            "open_model": self.pool.as_ref().map(|p| p.describe().to_string()),
            "source": self.source,
        })
    }
}

/// A decide request that names its brain(s). Without `brain` / `brains`,
/// the only loaded brain answers.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MultiRequest {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub brain: Option<String>,
    /// Ask several brains about the same state (one model pass is shared).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub brains: Vec<String>,
    /// Let a routing brain pick the brain: its first choice question whose
    /// options name loaded brains decides which brain answers the request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub route: Option<String>,
    #[serde(flatten)]
    pub request: DecideRequest,
}

/// Per-brain result of a multi-brain request.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum BrainResult {
    Ok(DecideResponse),
    Err { error: String },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MultiResponse {
    pub results: BTreeMap<String, BrainResult>,
    /// With `route`: the brain the routing brain picked.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub routed_to: Option<String>,
    pub latency_ms: f64,
}

/// What [`Registry::decide`] returns: one brain's response, or a
/// [`MultiResponse`] when the request named several brains.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RegistryResponse {
    One(DecideResponse),
    Many(MultiResponse),
}

/// Brains by name. Reads never block each other; loading or replacing a
/// brain swaps an `Arc`, so in-flight requests finish on the old one.
#[derive(Default)]
pub struct Registry {
    brains: RwLock<BTreeMap<String, Arc<BrainEntry>>>,
}

impl Registry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add or replace a brain.
    pub fn insert(&self, entry: BrainEntry) {
        if let Ok(mut b) = self.brains.write() {
            b.insert(entry.name.clone(), Arc::new(entry));
        }
    }

    pub fn remove(&self, name: &str) -> bool {
        self.brains
            .write()
            .map(|mut b| b.remove(name).is_some())
            .unwrap_or(false)
    }

    pub fn get(&self, name: &str) -> Option<Arc<BrainEntry>> {
        self.brains.read().ok()?.get(name).cloned()
    }

    pub fn names(&self) -> Vec<String> {
        self.brains
            .read()
            .map(|b| b.keys().cloned().collect())
            .unwrap_or_default()
    }

    pub fn entries(&self) -> Vec<Arc<BrainEntry>> {
        self.brains
            .read()
            .map(|b| b.values().cloned().collect())
            .unwrap_or_default()
    }

    pub fn len(&self) -> usize {
        self.brains.read().map(|b| b.len()).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The brain a request names, or the only loaded brain.
    pub fn resolve(&self, name: Option<&str>) -> Result<Arc<BrainEntry>, String> {
        match name {
            Some(n) => self.get(n).ok_or_else(|| {
                format!("no brain named {n:?} (loaded: {})", self.names().join(", "))
            }),
            None => {
                let b = self.brains.read().map_err(|_| "registry lock poisoned")?;
                match b.len() {
                    1 => Ok(b.values().next().cloned().expect("one entry")),
                    0 => Err("no brains loaded".into()),
                    _ => Err(format!(
                        "several brains are loaded; name one with \"brain\" ({})",
                        b.keys().cloned().collect::<Vec<_>>().join(", ")
                    )),
                }
            }
        }
    }

    /// One brain → its [`DecideResponse`]; `brains: [...]` → a [`MultiResponse`].
    pub fn decide(&self, req: &MultiRequest) -> Result<RegistryResponse, String> {
        if let Some(router) = &req.route {
            return self.decide_routed(router, req).map(RegistryResponse::Many);
        }
        if req.brains.is_empty() {
            let e = self.resolve(req.brain.as_deref())?;
            return e.decide(&req.request).map(RegistryResponse::One);
        }
        let t0 = Instant::now();
        let mut results = BTreeMap::new();
        for name in &req.brains {
            let r = self
                .resolve(Some(name))
                .and_then(|e| e.decide(&req.request));
            results.insert(
                name.clone(),
                match r {
                    Ok(r) => BrainResult::Ok(r),
                    Err(error) => BrainResult::Err { error },
                },
            );
        }
        Ok(RegistryResponse::Many(MultiResponse {
            results,
            routed_to: None,
            latency_ms: t0.elapsed().as_secs_f64() * 1000.0,
        }))
    }

    /// Ask the routing brain which brain should answer, then ask that brain.
    /// The routing brain's own question list is used for its step; the
    /// request's `questions` go to the chosen brain.
    fn decide_routed(&self, router: &str, req: &MultiRequest) -> Result<MultiResponse, String> {
        let t0 = Instant::now();
        let r = self.resolve(Some(router))?;
        let loaded = self.names();
        let q = r
            .brain
            .questions
            .questions
            .iter()
            .find(|q| match &q.kind {
                crate::schema::QuestionKind::Choice { options } => {
                    options.iter().any(|o| loaded.contains(o) && o != router)
                }
                _ => false,
            })
            .ok_or_else(|| {
                format!("routing brain {router:?} has no choice question whose options name loaded brains")
            })?;
        let route_req = DecideRequest {
            state: req.request.state.clone(),
            questions: vec![crate::decide::QuestionRef::Id(q.id.clone())],
            min_confidence: req.request.min_confidence,
            force: None,
        };
        let routed = r.decide(&route_req)?;
        let ans = &routed.answers[0];
        // Best option that names a loaded brain (the top option may not).
        let mut opts = ans.distribution.clone();
        opts.sort_by(|a, b| b.p.partial_cmp(&a.p).unwrap_or(std::cmp::Ordering::Equal));
        let chosen = opts
            .iter()
            .map(|o| o.outcome.clone())
            .find(|o| loaded.contains(o) && o != router)
            .ok_or("routing brain picked no loaded brain")?;
        let mut results = BTreeMap::new();
        results.insert(router.to_string(), BrainResult::Ok(routed.clone()));
        let out = self
            .resolve(Some(&chosen))
            .and_then(|e| e.decide(&req.request));
        results.insert(
            chosen.clone(),
            match out {
                Ok(r) => BrainResult::Ok(r),
                Err(error) => BrainResult::Err { error },
            },
        );
        Ok(MultiResponse {
            results,
            routed_to: Some(chosen),
            latency_ms: t0.elapsed().as_secs_f64() * 1000.0,
        })
    }

    pub fn decide_json(&self, request_json: &str) -> Result<String, String> {
        let req: MultiRequest =
            serde_json::from_str(request_json).map_err(|e| format!("bad request: {e}"))?;
        serde_json::to_string(&self.decide(&req)?).map_err(|e| e.to_string())
    }
}

/// Registry name for a brain file: `agent/crypto.s1.json.gz` → `crypto`.
pub fn brain_name_from_path(path: &str) -> String {
    let file = path.rsplit(['/', '\\']).next().unwrap_or(path).to_string();
    for ext in [".s1.json.gz", ".s1.json", ".json.gz", ".json"] {
        if let Some(stem) = file.strip_suffix(ext) {
            return stem.to_string();
        }
    }
    file
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brain::TrainConfig;
    use crate::decide::tests::FakeScorer;
    use crate::decide::QuestionRef;
    use crate::probe::{toy_embed, Probe, ProbeTrainConfig};
    use std::sync::atomic::AtomicUsize;

    /// Fake model that counts passes and can be forked (shared counter).
    struct CountingScorer {
        inner: FakeScorer,
        embeds: Arc<AtomicUsize>,
    }

    impl OptionScorer for CountingScorer {
        fn describe(&self) -> String {
            "counting".into()
        }
        fn model_key(&self) -> String {
            "fake".into()
        }
        fn score(&mut self, s: &str, qs: &[&Question]) -> Result<Vec<Vec<f32>>, String> {
            self.inner.score(s, qs)
        }
        fn embed(&mut self, s: &str, l: f32) -> Result<Vec<f32>, String> {
            self.embeds.fetch_add(1, Ordering::SeqCst);
            self.inner.embed(s, l)
        }
        fn fork(&self) -> Option<Box<dyn OptionScorer + Send>> {
            Some(Box::new(CountingScorer {
                inner: FakeScorer {
                    calls: self.inner.calls.clone(),
                },
                embeds: self.embeds.clone(),
            }))
        }
    }

    fn probed_brain(name: &str) -> System1Brain {
        let (qs, rows) = crate::brain::tests::toy_rows();
        let mut b = System1Brain::new(name, qs.clone()).unwrap();
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
        b
    }

    #[test]
    fn brains_share_one_model_and_one_pass_per_state() {
        let embeds = Arc::new(AtomicUsize::new(0));
        let pool = Arc::new(ModelPool::new(
            Box::new(CountingScorer {
                inner: FakeScorer {
                    calls: Arc::new(AtomicUsize::new(0)),
                },
                embeds: embeds.clone(),
            }),
            3,
        ));
        assert_eq!(pool.workers(), 3);
        let reg = Registry::new();
        for name in ["a", "b"] {
            reg.insert(BrainEntry {
                name: name.into(),
                brain: Arc::new(probed_brain(name)),
                pool: Some(pool.clone()),
                source: None,
            });
        }
        let out = reg
            .decide_json(
                r#"{"brains":["a","b","nope"],"state":"card declined","questions":["route"]}"#,
            )
            .unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["results"]["a"]["answers"][0]["source"], "probe");
        assert_eq!(v["results"]["b"]["answers"][0]["value"], "Billing");
        assert!(v["results"]["nope"]["error"]
            .as_str()
            .unwrap()
            .contains("no brain"));
        // Two brains, one state: one model pass.
        assert_eq!(embeds.load(Ordering::SeqCst), 1);
        // Several brains loaded and none named → a clear error.
        assert!(reg
            .decide_json(r#"{"state":"x"}"#)
            .unwrap_err()
            .contains("several brains"));
        let one = reg.decide_json(r#"{"brain":"a","state":"x"}"#).unwrap();
        assert!(one.contains("\"answers\""));
        assert!(reg.remove("b"));
        assert_eq!(reg.names(), vec!["a".to_string()]);
    }

    #[test]
    fn parallel_requests_on_shared_brains() {
        let pool = Arc::new(
            ModelPool::new(
                Box::new(CountingScorer {
                    inner: FakeScorer {
                        calls: Arc::new(AtomicUsize::new(0)),
                    },
                    embeds: Arc::new(AtomicUsize::new(0)),
                }),
                2,
            )
            .with_feature_cache(0),
        );
        let reg = Arc::new(Registry::new());
        reg.insert(BrainEntry {
            name: "a".into(),
            brain: Arc::new(probed_brain("a")),
            pool: Some(pool),
            source: None,
        });
        let threads: Vec<_> = (0..8)
            .map(|i| {
                let reg = reg.clone();
                std::thread::spawn(move || {
                    let req = MultiRequest {
                        brain: Some("a".into()),
                        brains: Vec::new(),
                        route: None,
                        request: DecideRequest {
                            state: json!(format!("app crashes {i}")),
                            questions: vec![QuestionRef::Id("route".into())],
                            min_confidence: None,
                            force: None,
                        },
                    };
                    match reg.decide(&req).unwrap() {
                        RegistryResponse::One(r) => r.answers[0].value.clone(),
                        RegistryResponse::Many(_) => panic!("one brain asked"),
                    }
                })
            })
            .collect();
        for t in threads {
            assert_eq!(t.join().unwrap(), json!("Tech"));
        }
    }

    #[test]
    fn routing_brain_picks_the_specialist() {
        let reg = Registry::new();
        for name in ["billing", "tech"] {
            reg.insert(BrainEntry {
                name: name.into(),
                brain: Arc::new(probed_brain(name)),
                pool: None,
                source: None,
            });
        }
        // A router whose answer names the brains (reuses the toy "route" data).
        let (_, rows) = crate::brain::tests::toy_rows();
        let qs = crate::schema::QuestionSet::parse(
            r#"[[questions]]
id = "brain"
text = "Which specialist?"
type = "choice"
options = ["billing", "tech"]
source = { field = "team", map = { Billing = "billing", Tech = "tech" } }
"#,
        )
        .unwrap();
        let jsonl: Vec<String> = rows
            .iter()
            .map(|r| {
                let team = if r.targets[0].2 == 0 {
                    "Billing"
                } else {
                    "Tech"
                };
                json!({"text": r.text, "team": team}).to_string()
            })
            .collect();
        let (rrows, _) = crate::data::parse_labeled_jsonl(&jsonl.join("\n"), &qs).unwrap();
        let mut router = System1Brain::new("router", qs).unwrap();
        router.train_heads(&rrows, &TrainConfig::default()).unwrap();
        reg.insert(BrainEntry {
            name: "router".into(),
            brain: Arc::new(router),
            pool: None,
            source: None,
        });
        let out = reg
            .decide_json(
                r#"{"route":"router","state":"app crashes on login","questions":["urgent"]}"#,
            )
            .unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["routed_to"], "tech", "{v}");
        assert_eq!(v["results"]["tech"]["answers"][0]["id"], "urgent");
        assert!(v["results"]["router"]["answers"][0]["distribution"].is_array());
    }

    #[test]
    fn names_from_paths() {
        assert_eq!(brain_name_from_path("agent/crypto.s1.json.gz"), "crypto");
        assert_eq!(brain_name_from_path("C:\\b\\support.s1.json"), "support");
        assert_eq!(brain_name_from_path("x.json"), "x");
    }
}
