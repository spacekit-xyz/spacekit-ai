//! Layer 0 — **typed world grounding graph** (MVP scaffold).
//!
//! Curated concept nodes and weighted edges expand **subject keywords** before lattice BM25 /
//! witness alignment (`GROWFORMER_CAUSAL_AI.md` — World grounding). This is **bearing**, not
//! answers: it does not emit sentiment labels by itself. `sentiment_bearing` edges additionally
//! feed a bounded valence scalar used to nudge the **retrieval** conditioning vector (Stage‑1
//! cosine in forced-topic program pick), not generation-wide classification.
//!
//! **PR / wire neutral headlines** (`LanguageService` PR-wire path): journalistic “raises” /
//! “profit” must not fire consumer valence. While that pass is active, [`PrWireHeadlineGuard`]
//! skips **all** expansion from **`promotion`** (career valence). For **`gain`** and
//! **`fundraising`**, PR-wire skips only **`sentiment_bearing`** edges so cap-table / FX
//! keywords (`financial_gain`, `venture_capital`, `ipo`, …) still aid neutral retrieval
//! without a consumer valence nudge. [`sentiment_bearing_from_intent`] stays zero on PR.
//!
//! Data: `data/inference/world_grounding.toml` (embedded at compile time, versioned).

use serde::Deserialize;
use std::cell::Cell;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::OnceLock;

use crate::spectral::tokenize;

thread_local! {
    /// Set during PR-wire neutral headline inference (see `LanguageService`).
    static PR_WIRE_HEADLINE_PASS: Cell<bool> = Cell::new(false);
}

fn pr_wire_headline_pass_active() -> bool {
    PR_WIRE_HEADLINE_PASS.with(|c| c.get())
}

/// RAII: while dropped at end of request scope, clears the PR-wire headline flag.
pub struct PrWireHeadlineGuard {
    armed: bool,
}

impl PrWireHeadlineGuard {
    /// When `active`, subsequent world-grounding keyword expansion omits `sentiment_bearing`
    /// targets and [`sentiment_bearing_from_intent`] returns `0.0`.
    pub fn bind(active: bool) -> Self {
        if active {
            PR_WIRE_HEADLINE_PASS.with(|c| c.set(true));
        }
        Self { armed: active }
    }
}

impl Drop for PrWireHeadlineGuard {
    fn drop(&mut self) {
        if self.armed {
            PR_WIRE_HEADLINE_PASS.with(|c| c.set(false));
        }
    }
}

#[derive(Debug, Deserialize)]
struct WorldGroundingFile {
    #[serde(default = "default_version")]
    version: u32,
    #[serde(default)]
    nodes: Vec<NodeToml>,
}

fn default_version() -> u32 {
    1
}

#[derive(Debug, Deserialize)]
struct NodeToml {
    id: String,
    #[serde(default)]
    aliases: Vec<String>,
    #[serde(default)]
    magnitude_sensitive: bool,
    #[serde(default)]
    causal_anchor: bool,
    #[serde(default)]
    edges: Vec<EdgeToml>,
}

#[derive(Debug, Deserialize)]
struct EdgeToml {
    #[serde(default)]
    kind: String,
    target: String,
    #[serde(default = "default_weight")]
    weight: f32,
    /// For `adjacent_to` with `weight = 0`: unlock only if this context id matched
    /// (`disambiguated_by` target on the same node, or activated root / token).
    #[serde(default)]
    requires_context: Option<String>,
}

fn default_weight() -> f32 {
    1.0
}

#[derive(Clone, Debug)]
struct WorldEdge {
    kind: String,
    target: String,
    weight: f32,
    requires_context: Option<String>,
}

#[derive(Clone, Debug)]
struct WorldNode {
    id: String,
    magnitude_sensitive: bool,
    #[allow(dead_code)]
    causal_anchor: bool,
    edges: Vec<WorldEdge>,
}

#[derive(Clone, Debug)]
struct WorldGraph {
    lookup: HashMap<String, usize>,
    nodes: Vec<WorldNode>,
}

fn normalize_key(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '_')
        .collect::<String>()
        .to_ascii_lowercase()
}

fn kind_multiplier(kind: &str) -> f32 {
    match kind.trim().to_ascii_lowercase().as_str() {
        "is_a" => 1.25,
        "sentiment_bearing" => 1.15,
        "causal_type" => 1.1,
        "domain" => 1.1,
        "adjacent_to" => 1.0,
        "disambiguated_by" => 0.0,
        _ => 1.0,
    }
}

fn padded_intent(intent_text: &str) -> String {
    format!(
        " {} ",
        intent_text
            .to_ascii_lowercase()
            .replace(['\n', '\r', '\t'], " ")
    )
}

fn intent_token_set(intent_text: &str) -> HashSet<String> {
    let mut s = HashSet::new();
    for tok in tokenize(intent_text) {
        let k = normalize_key(&tok);
        if k.len() > 1 {
            s.insert(k);
        }
    }
    s
}

/// True if context id `t` is activated as a root, appears as a token, or appears in padded text.
fn context_matched(g: &WorldGraph, roots: &[usize], intent_tokens: &HashSet<String>, padded: &str, t: &str) -> bool {
    let t = normalize_key(t);
    if t.is_empty() {
        return false;
    }
    for &rix in roots {
        if let Some(n) = g.nodes.get(rix) {
            if n.id == t {
                return true;
            }
        }
    }
    if intent_tokens.contains(&t) {
        return true;
    }
    let needle = format!(" {} ", t);
    padded.contains(&needle)
}

/// Targets of `disambiguated_by` edges on this node that matched the query.
fn disambig_matched_for_node(
    g: &WorldGraph,
    node_ix: usize,
    roots: &[usize],
    intent_tokens: &HashSet<String>,
    padded: &str,
) -> HashSet<String> {
    let mut out = HashSet::new();
    let Some(node) = g.nodes.get(node_ix) else {
        return out;
    };
    for e in &node.edges {
        if e.kind.to_ascii_lowercase().trim() != "disambiguated_by" {
            continue;
        }
        let key = normalize_key(&e.target);
        if key.is_empty() {
            continue;
        }
        if context_matched(g, roots, intent_tokens, padded, &key) {
            out.insert(key);
        }
    }
    out
}

fn load_graph() -> WorldGraph {
    let raw = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/data/inference/world_grounding.toml"
    ));
    let file: WorldGroundingFile =
        toml::from_str(raw).expect("parse embedded world_grounding.toml");
    assert_eq!(file.version, 1, "unsupported world_grounding.toml version");

    let crypto_raw = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/data/crypto/world_grounding_crypto.toml"
    ));
    let tradfi_raw = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/data/fintech/world_grounding_fintech.toml"
    ));

    let mut all_nodes = file.nodes;
    if let Ok(crypto_file) = toml::from_str::<WorldGroundingFile>(crypto_raw) {
        all_nodes.extend(crypto_file.nodes);
    }
    if let Ok(tradfi_file) = toml::from_str::<WorldGroundingFile>(tradfi_raw) {
        all_nodes.extend(tradfi_file.nodes);
    }

    let mut nodes: Vec<WorldNode> = Vec::new();
    let mut lookup: HashMap<String, usize> = HashMap::new();

    for n in all_nodes {
        let id_norm = normalize_key(&n.id);
        if id_norm.is_empty() {
            continue;
        }
        let idx = nodes.len();
        let mut edges: Vec<WorldEdge> = Vec::new();
        for e in n.edges {
            let t = normalize_key(&e.target);
            if t.is_empty() {
                continue;
            }
            let kind = e.kind.trim().to_ascii_lowercase();
            if kind.is_empty() {
                continue;
            }
            edges.push(WorldEdge {
                kind,
                target: t,
                weight: e.weight,
                requires_context: e.requires_context.as_ref().map(|s| normalize_key(s)).filter(|s| !s.is_empty()),
            });
        }
        nodes.push(WorldNode {
            id: id_norm.clone(),
            magnitude_sensitive: n.magnitude_sensitive,
            causal_anchor: n.causal_anchor,
            edges,
        });
        lookup.insert(id_norm, idx);
        for a in n.aliases {
            let an = normalize_key(&a);
            if !an.is_empty() {
                lookup.entry(an).or_insert(idx);
            }
        }
    }

    WorldGraph { lookup, nodes }
}

static GRAPH: OnceLock<WorldGraph> = OnceLock::new();

/// Domain-specific grounding graph loaded at runtime (e.g., pet_world_grounding.toml).
/// On WASM this is thread-local; on native it uses a RwLock.
#[cfg(target_arch = "wasm32")]
thread_local! {
    static DOMAIN_GRAPH: std::cell::RefCell<Option<WorldGraph>> = std::cell::RefCell::new(None);
}

#[cfg(not(target_arch = "wasm32"))]
static DOMAIN_GRAPH_NATIVE: std::sync::RwLock<Option<WorldGraph>> = std::sync::RwLock::new(None);

fn graph() -> &'static WorldGraph {
    GRAPH.get_or_init(load_graph)
}

fn parse_grounding_toml(toml_str: &str) -> Result<WorldGraph, String> {
    let file: WorldGroundingFile = toml::from_str(toml_str)
        .map_err(|e| format!("parse grounding TOML: {}", e))?;

    let mut nodes: Vec<WorldNode> = Vec::new();
    let mut lookup: HashMap<String, usize> = HashMap::new();

    for n in file.nodes {
        let id_norm = normalize_key(&n.id);
        if id_norm.is_empty() { continue; }
        let idx = nodes.len();
        let mut edges: Vec<WorldEdge> = Vec::new();
        for e in n.edges {
            let t = normalize_key(&e.target);
            if t.is_empty() { continue; }
            let kind = e.kind.trim().to_ascii_lowercase();
            if kind.is_empty() { continue; }
            edges.push(WorldEdge {
                kind,
                target: t,
                weight: e.weight,
                requires_context: e.requires_context.as_ref().map(|s| normalize_key(s)).filter(|s| !s.is_empty()),
            });
        }
        nodes.push(WorldNode {
            id: id_norm.clone(),
            magnitude_sensitive: n.magnitude_sensitive,
            causal_anchor: n.causal_anchor,
            edges,
        });
        lookup.insert(id_norm, idx);
        for a in n.aliases {
            let an = normalize_key(&a);
            if !an.is_empty() {
                lookup.entry(an).or_insert(idx);
            }
        }
    }

    Ok(WorldGraph { lookup, nodes })
}

/// Load a domain-specific grounding graph from a TOML string at runtime.
/// This augments the embedded base graph with domain-specific concepts.
pub fn load_grounding_graph_from_str(toml_str: &str) -> Result<(), String> {
    let graph = parse_grounding_toml(toml_str)?;
    #[cfg(target_arch = "wasm32")]
    {
        DOMAIN_GRAPH.with(|cell| {
            *cell.borrow_mut() = Some(graph);
        });
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let mut guard = DOMAIN_GRAPH_NATIVE.write()
            .map_err(|e| format!("lock domain graph: {}", e))?;
        *guard = Some(graph);
    }
    Ok(())
}

/// `raised` is a fundraising alias, but "Chase raised my mortgage rate" is a consumer complaint —
/// not a venture round / earnings headline.
fn fundraising_root_false_positive_on_consumer_rate(intent_text: &str) -> bool {
    let lower = intent_text.to_ascii_lowercase();
    let padded = format!(" {} ", lower.replace('\n', " "));
    let first_person = padded.contains(" my ")
        || padded.contains(" i ")
        || padded.contains(" me ")
        || lower.trim_start().starts_with("my ")
        || lower.trim_start().starts_with("i ");
    if !first_person {
        return false;
    }
    padded.contains("mortgage")
        || padded.contains("interest rate")
        || padded.contains(" my rate ")
        || padded.contains("without notice")
        || padded.contains("without warning")
        || padded.contains(" my apr")
        || padded.contains(" apr ")
}

fn filter_fundraising_root_from_indices(
    g: &WorldGraph,
    intent_text: &str,
    roots: &mut Vec<usize>,
    trace_prefix: &str,
) {
    if !fundraising_root_false_positive_on_consumer_rate(intent_text) {
        return;
    }
    let Some(&fundraising_ix) = g.lookup.get("fundraising") else {
        return;
    };
    let before = roots.len();
    roots.retain(|&ix| ix != fundraising_ix);
    if roots.len() != before {
        crate::infer_trace!(
            "  [{}] skip fundraising root on first-person mortgage/rate complaint",
            trace_prefix
        );
    }
}

fn activated_roots(intent_text: &str) -> Vec<usize> {
    let g = graph();
    let mut seen_idx = HashSet::new();
    let mut roots = Vec::new();
    for tok in tokenize(intent_text) {
        let k = normalize_key(&tok);
        if k.len() < 2 {
            continue;
        }
        if let Some(&ix) = g.lookup.get(&k) {
            if seen_idx.insert(ix) {
                roots.push(ix);
            }
        }
    }
    filter_fundraising_root_from_indices(g, intent_text, &mut roots, "world-ground");
    roots
}

/// Effective weight for expansion / continuation (kind scales auditable `weight`).
fn edge_effective(
    g: &WorldGraph,
    node_ix: usize,
    e: &WorldEdge,
    roots: &[usize],
    intent_tokens: &HashSet<String>,
    padded: &str,
) -> f32 {
    let km = kind_multiplier(&e.kind);
    if e.kind == "disambiguated_by" {
        return 0.0;
    }
    let dis = disambig_matched_for_node(g, node_ix, roots, intent_tokens, padded);

    // Zero-weight gated adjacent_to: unlock with explicit or any disambiguator match.
    if e.kind == "adjacent_to" && e.weight <= 1e-6 {
        let ok = match &e.requires_context {
            Some(req) => dis.contains(req),
            None => !dis.is_empty(),
        };
        if !ok {
            return 0.0;
        }
        return 0.55 * km;
    }

    e.weight.max(0.0) * km
}

/// Minimum effective weight to continue BFS into a neighbor node (stronger than emitting a keyword).
const MIN_TRAVERSE_EFFECTIVE: f32 = 0.42;

fn expand_from_roots(
    roots: &[usize],
    intent_text: &str,
    max_depth: u8,
    max_terms: usize,
) -> Vec<String> {
    let g = graph();
    let padded = padded_intent(intent_text);
    let intent_tokens = intent_token_set(intent_text);
    let mut out: Vec<String> = Vec::new();
    let mut seen_kw: HashSet<String> = HashSet::new();
    let mut q: VecDeque<(usize, u8)> = VecDeque::new();
    let mut visited_nodes: HashSet<usize> = HashSet::new();

    for &r in roots {
        if visited_nodes.insert(r) {
            q.push_back((r, 0));
        }
    }

    while let Some((ix, depth)) = q.pop_front() {
        if out.len() >= max_terms {
            break;
        }
        let node = match g.nodes.get(ix) {
            Some(n) => n,
            None => continue,
        };
        // PR headlines: never expand `promotion` (career / recognition spillover).
        if pr_wire_headline_pass_active() && node.id == "promotion" {
            continue;
        }
        let pr_skip_bearing_only =
            pr_wire_headline_pass_active() && (node.id == "gain" || node.id == "fundraising");
        if node.magnitude_sensitive {
            crate::infer_trace!(
                "  [world-ground] magnitude-sensitive node hit: {}",
                node.id
            );
        }
        for e in &node.edges {
            if pr_skip_bearing_only && e.kind == "sentiment_bearing" {
                continue;
            }
            let eff = edge_effective(g, ix, e, roots, &intent_tokens, &padded);
            if eff < 1e-4 {
                continue;
            }
            let t = &e.target;
            if t.len() < 2 {
                continue;
            }
            if seen_kw.insert(t.clone()) {
                out.push(t.clone());
            }
            if out.len() >= max_terms {
                break;
            }
            if depth < max_depth {
                if let Some(&nxt) = g.lookup.get(t.as_str()) {
                    if eff >= MIN_TRAVERSE_EFFECTIVE && visited_nodes.insert(nxt) {
                        q.push_back((nxt, depth + 1));
                    }
                }
            }
        }
    }
    out
}

/// Append Layer‑0 graph expansion keywords (deduped, min length 3) for BM25 / alignment.
pub fn extend_subject_keywords_with_world_graph(intent_text: &str, subject_kw: &mut Vec<String>) {
    let roots = activated_roots(intent_text);
    if roots.is_empty() && !has_domain_graph() {
        return;
    }
    if !roots.is_empty() {
        let expanded = expand_from_roots(&roots, intent_text, 2, 24);
        let mut added: Vec<String> = Vec::new();
        for kw in expanded {
            if kw.len() > 2 && !subject_kw.iter().any(|x| x == &kw) {
                subject_kw.push(kw.clone());
                added.push(kw);
            }
        }
        if !added.is_empty() {
            crate::infer_trace!(
                "  [world-ground] layer-0 graph: roots={:?} +keywords {:?}",
                roots
                    .iter()
                    .filter_map(|&i| graph().nodes.get(i).map(|n| n.id.as_str()))
                    .collect::<Vec<_>>(),
                added
            );
        }
    }
    // Also walk the domain-specific graph
    extend_subject_keywords_with_domain_graph(intent_text, subject_kw);
}

fn has_domain_graph() -> bool {
    #[cfg(target_arch = "wasm32")]
    {
        DOMAIN_GRAPH.with(|cell| cell.borrow().is_some())
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        DOMAIN_GRAPH_NATIVE.read().map(|g| g.is_some()).unwrap_or(false)
    }
}

/// Walk the domain-specific grounding graph for keyword expansion.
fn extend_subject_keywords_with_domain_graph(intent_text: &str, subject_kw: &mut Vec<String>) {
    #[cfg(target_arch = "wasm32")]
    {
        DOMAIN_GRAPH.with(|cell| {
            let borrow = cell.borrow();
            if let Some(ref dg) = *borrow {
                domain_graph_expand(dg, intent_text, subject_kw);
            }
        });
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        if let Ok(guard) = DOMAIN_GRAPH_NATIVE.read() {
            if let Some(ref dg) = *guard {
                domain_graph_expand(dg, intent_text, subject_kw);
            }
        }
    }
}

/// Expand keywords from a domain graph: find activated roots, walk edges, collect targets.
fn domain_graph_expand(dg: &WorldGraph, intent_text: &str, subject_kw: &mut Vec<String>) {
    let mut roots = Vec::new();
    let mut seen_idx = HashSet::new();
    for tok in tokenize(intent_text) {
        let k = normalize_key(&tok);
        if k.len() < 2 { continue; }
        if let Some(&ix) = dg.lookup.get(&k) {
            if seen_idx.insert(ix) {
                roots.push(ix);
            }
        }
    }
    filter_fundraising_root_from_indices(dg, intent_text, &mut roots, "domain-ground");
    if roots.is_empty() { return; }

    let skip_fundraising = fundraising_root_false_positive_on_consumer_rate(intent_text);
    let fundraising_ix = dg.lookup.get("fundraising").copied();

    let mut added: Vec<String> = Vec::new();
    let mut visited = HashSet::new();
    let mut queue: VecDeque<(usize, u8)> = roots.iter().map(|&r| (r, 0u8)).collect();

    while let Some((idx, depth)) = queue.pop_front() {
        if depth > 2 || !visited.insert(idx) { continue; }
        if skip_fundraising && fundraising_ix == Some(idx) {
            continue;
        }
        let node = &dg.nodes[idx];
        for e in &node.edges {
            let kind = e.kind.as_str();
            if kind == "disambiguated_by" { continue; }
            if skip_fundraising && kind == "sentiment_bearing" && e.target == "positive" {
                continue;
            }
            if let Some(ref ctx) = e.requires_context {
                if !roots.iter().any(|&r| dg.nodes[r].id == *ctx) { continue; }
            }
            if e.weight < 0.3 { continue; }
            let target_key = &e.target;
            if target_key.len() > 2
                && !subject_kw.iter().any(|x| x == target_key)
                && !added.iter().any(|x| x == target_key)
            {
                added.push(target_key.clone());
            }
            if let Some(&tix) = dg.lookup.get(target_key.as_str()) {
                if depth < 2 {
                    if skip_fundraising && fundraising_ix == Some(tix) {
                        continue;
                    }
                    queue.push_back((tix, depth + 1));
                }
            }
        }
    }

    for kw in &added {
        subject_kw.push(kw.clone());
    }
    if !added.is_empty() {
        crate::infer_trace!(
            "  [domain-ground] activated {:?} +keywords {:?}",
            roots.iter().filter_map(|&i| dg.nodes.get(i).map(|n| n.id.as_str())).collect::<Vec<_>>(),
            added
        );
    }
}

/// Aggregate `sentiment_bearing` edges from **activated graph roots** into `[-1, 1]`.
/// Positive targets move toward `+1`, negative toward `-1`; weights use the same
/// `edge_effective` gating as keyword expansion.
pub fn sentiment_bearing_from_intent(intent_text: &str) -> f32 {
    if pr_wire_headline_pass_active() {
        return 0.0;
    }
    let g = graph();
    let roots = activated_roots(intent_text);
    if roots.is_empty() {
        return 0.0;
    }
    let padded = padded_intent(intent_text);
    let intent_tokens = intent_token_set(intent_text);
    let mut num = 0.0f32;
    let mut den = 0.0f32;
    for &rix in &roots {
        let Some(node) = g.nodes.get(rix) else {
            continue;
        };
        for e in &node.edges {
            if e.kind != "sentiment_bearing" {
                continue;
            }
            let eff = edge_effective(g, rix, e, &roots, &intent_tokens, &padded);
            if eff < 1e-6 {
                continue;
            }
            let sign = match e.target.as_str() {
                "positive" => 1.0f32,
                "negative" => -1.0f32,
                "neutral" | "mixed" => 0.0f32,
                _ => continue,
            };
            num += sign * eff;
            den += eff;
        }
    }
    if den < 1e-6 {
        return 0.0;
    }
    (num / den).clamp(-1.0, 1.0)
}

/// Fleet slice for cross-domain collision checks (graphs are not merged for audit).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum GroundingFleetDomain {
    Base,
    Crypto,
    Fintech,
    Runtime,
}

impl GroundingFleetDomain {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Base => "base",
            Self::Crypto => "crypto",
            Self::Fintech => "fintech",
            Self::Runtime => "runtime",
        }
    }
}

/// One grounding node and its alias keys (audit / assisted-maintenance loop).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GroundingNodeInfo {
    pub domain: GroundingFleetDomain,
    pub node_id: String,
    pub aliases: Vec<String>,
}

fn nodes_with_aliases_from_graph(g: &WorldGraph, domain: GroundingFleetDomain) -> Vec<GroundingNodeInfo> {
    let mut aliases_per_node: Vec<Vec<String>> = vec![Vec::new(); g.nodes.len()];
    for (alias, &idx) in &g.lookup {
        let node_id = &g.nodes[idx].id;
        if alias != node_id {
            aliases_per_node[idx].push(alias.clone());
        }
    }
    g.nodes
        .iter()
        .enumerate()
        .map(|(i, n)| GroundingNodeInfo {
            domain,
            node_id: n.id.clone(),
            aliases: aliases_per_node[i].clone(),
        })
        .collect()
}

fn load_base_graph_only() -> WorldGraph {
    let raw = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/data/inference/world_grounding.toml"
    ));
    parse_grounding_toml(raw).expect("parse base world_grounding.toml")
}

fn load_crypto_graph_only() -> WorldGraph {
    let raw = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/data/crypto/world_grounding_crypto.toml"
    ));
    parse_grounding_toml(raw).expect("parse crypto world_grounding.toml")
}

fn load_fintech_graph_only() -> WorldGraph {
    let raw = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/data/fintech/world_grounding_fintech.toml"
    ));
    parse_grounding_toml(raw).expect("parse fintech world_grounding.toml")
}

fn domain_graph_clone() -> Option<WorldGraph> {
    #[cfg(target_arch = "wasm32")]
    {
        DOMAIN_GRAPH.with(|cell| cell.borrow().clone())
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        DOMAIN_GRAPH_NATIVE
            .read()
            .ok()
            .and_then(|g| g.as_ref().cloned())
    }
}

/// Activated root node ids on the merged fleet graph (token lookup).
pub fn activated_root_ids(intent_text: &str) -> Vec<String> {
    activated_roots(intent_text)
        .iter()
        .filter_map(|&i| graph().nodes.get(i).map(|n| n.id.clone()))
        .collect()
}

/// Activated root node ids on the optional runtime domain graph only.
pub fn activated_root_ids_in_domain_graph(intent_text: &str) -> Vec<String> {
    let Some(dg) = domain_graph_clone() else {
        return Vec::new();
    };
    let mut roots = Vec::new();
    let mut seen_idx = HashSet::new();
    for tok in tokenize(intent_text) {
        let k = normalize_key(&tok);
        if k.len() < 2 {
            continue;
        }
        if let Some(&ix) = dg.lookup.get(&k) {
            if seen_idx.insert(ix) {
                if let Some(n) = dg.nodes.get(ix) {
                    roots.push(n.id.clone());
                }
            }
        }
    }
    roots
}

/// Inventory of all loaded grounding graphs (per-domain slices, not merged).
pub fn fleet_node_inventory() -> Vec<GroundingNodeInfo> {
    let mut out = nodes_with_aliases_from_graph(&load_base_graph_only(), GroundingFleetDomain::Base);
    out.extend(nodes_with_aliases_from_graph(
        &load_crypto_graph_only(),
        GroundingFleetDomain::Crypto,
    ));
    out.extend(nodes_with_aliases_from_graph(
        &load_fintech_graph_only(),
        GroundingFleetDomain::Fintech,
    ));
    if let Some(dg) = domain_graph_clone() {
        out.extend(nodes_with_aliases_from_graph(&dg, GroundingFleetDomain::Runtime));
    }
    out
}

/// True when any fleet or runtime-domain graph is loaded.
pub fn has_runtime_domain_graph() -> bool {
    has_domain_graph()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graph_loads() {
        let g = graph();
        assert!(!g.nodes.is_empty());
        assert!(g.lookup.contains_key("bitcoin"));
        assert!(g.lookup.contains_key("btc"));
        assert!(g.lookup.contains_key("coiling"));
        assert!(g.lookup.contains_key("loss"));
    }

    #[test]
    fn launch_expands_product_release() {
        let mut kw: Vec<String> = vec!["morgan".to_string()];
        extend_subject_keywords_with_world_graph("Morgan Stanley launches Bitcoin ETF", &mut kw);
        assert!(
            kw.iter().any(|x| x == "product_release" || x == "go_live"),
            "kw={kw:?}"
        );
    }

    #[test]
    fn stablecoin_license_expansion() {
        let mut kw = Vec::new();
        extend_subject_keywords_with_world_graph(
            "HSBC granted Hong Kong stablecoin issuer licence",
            &mut kw,
        );
        assert!(kw.iter().any(|x| x == "regulatory_approval" || x == "issuer"));
    }

    #[test]
    fn loss_on_game_unlocks_gambling() {
        let mut kw = Vec::new();
        extend_subject_keywords_with_world_graph("I lost $5000 on the game last night", &mut kw);
        assert!(
            kw.contains(&"gambling".to_string()),
            "expected gambling in expansion, kw={kw:?}"
        );
    }

    #[test]
    fn because_adds_causal_frame() {
        let mut kw = Vec::new();
        extend_subject_keywords_with_world_graph("Prices fell because regulation tightened", &mut kw);
        assert!(kw.contains(&"direct_cause".to_string()), "kw={kw:?}");
    }

    #[test]
    fn sentiment_bearing_loss_negative() {
        let b = sentiment_bearing_from_intent("I totally lost my wallet on that trade");
        assert!(b < -0.15, "b={b}");
    }

    #[test]
    fn sentiment_bearing_profit_positive() {
        let b = sentiment_bearing_from_intent("record profits this quarter");
        assert!(b > 0.15, "b={b}");
    }

    #[test]
    fn pr_wire_guard_suppresses_valence_keywords_and_bearing() {
        let _g = super::PrWireHeadlineGuard::bind(true);
        let mut kw = Vec::new();
        super::extend_subject_keywords_with_world_graph(
            "Y Combinator grad Glimpse raises money_usd_35 led by a16z",
            &mut kw,
        );
        assert!(
            !kw.iter().any(|x| x == "positive" || x == "career" || x == "recognition"),
            "unexpected valence tokens in kw={kw:?}"
        );
        assert!(
            kw.iter().any(|x| x == "venture_capital" || x == "ipo" || x == "listing"),
            "PR-wire should still expand cap-table keywords (not sentiment), kw={kw:?}"
        );
        let b = super::sentiment_bearing_from_intent("raises money record profits");
        assert!(b.abs() < 1e-4, "b={b}");
    }

    #[test]
    fn pr_wire_guard_gain_keeps_financial_gain_not_positive() {
        let _g = super::PrWireHeadlineGuard::bind(true);
        let mut kw = Vec::new();
        super::extend_subject_keywords_with_world_graph(
            "Company reports record profits for Q4",
            &mut kw,
        );
        assert!(kw.contains(&"financial_gain".to_string()), "kw={kw:?}");
        assert!(!kw.iter().any(|x| x == "positive"), "kw={kw:?}");
    }

    #[test]
    fn consumer_mortgage_domain_overlay_skips_fundraising() {
        let overlay = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/data/fintech/world_grounding_fintech.toml"
        ));
        super::load_grounding_graph_from_str(overlay).expect("fintech overlay");
        let mut kw = Vec::new();
        super::extend_subject_keywords_with_world_graph(
            "Chase raised my mortgage rate without notice",
            &mut kw,
        );
        assert!(
            !kw.iter().any(|x| {
                matches!(
                    x.as_str(),
                    "positive" | "venture_capital" | "ipo" | "financial_gain" | "fundraising"
                )
            }),
            "domain overlay fundraising spillover, kw={kw:?}"
        );
        assert!(
            kw.iter().any(|x| x == "consumer_credit" || x == "interest_rate" || x == "mortgage"),
            "expected consumer mortgage keywords, kw={kw:?}"
        );
    }

    #[test]
    fn consumer_mortgage_rate_skips_fundraising_root() {
        let mut kw = Vec::new();
        super::extend_subject_keywords_with_world_graph(
            "Chase raised my mortgage rate without notice",
            &mut kw,
        );
        assert!(
            !kw.iter().any(|x| x == "positive" || x == "venture_capital" || x == "ipo"),
            "fundraising spillover on consumer complaint, kw={kw:?}"
        );
        let b = super::sentiment_bearing_from_intent("Chase raised my mortgage rate without notice");
        assert!(b < 0.15, "consumer rate hike should not nudge positive, b={b}");
    }

    #[test]
    fn fundraising_expands_venture_keywords() {
        let mut kw = Vec::new();
        super::extend_subject_keywords_with_world_graph(
            "Doss raises money_usd_55 in series_b led by top_tier_funds",
            &mut kw,
        );
        assert!(
            kw.iter().any(|x| x == "venture_capital" || x == "startup_growth" || x == "ipo"),
            "expected cap-table expansion, kw={kw:?}"
        );
    }

    #[test]
    fn range_compression_expands_from_coiling() {
        let mut kw = Vec::new();
        extend_subject_keywords_with_world_graph(
            "BTC is coiling tighter than ever before the breakout",
            &mut kw,
        );
        assert!(
            kw.iter().any(|x| {
                matches!(
                    x.as_str(),
                    "bitcoin" | "cryptocurrency" | "volatility" | "breakout" | "etf"
                )
            }),
            "kw={kw:?}"
        );
    }

    #[test]
    fn sprint_context_expands_professional() {
        let mut kw = Vec::new();
        extend_subject_keywords_with_world_graph(
            "the sprint retrospective was exhausting",
            &mut kw,
        );
        assert!(
            kw.iter().any(|x| x == "professional" || x == "meeting"),
            "kw={kw:?}"
        );
    }

    #[test]
    fn absence_expands_relationships_frame() {
        let mut kw = Vec::new();
        extend_subject_keywords_with_world_graph("she drifted away over winter", &mut kw);
        assert!(
            kw.iter().any(|x| x == "relationships" || x == "retrospective_framing"),
            "kw={kw:?}"
        );
    }
}
