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

fn graph() -> &'static WorldGraph {
    GRAPH.get_or_init(load_graph)
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
    if roots.is_empty() {
        return;
    }
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
