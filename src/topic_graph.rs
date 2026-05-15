//! TopicGraph — data-driven knowledge graph for topic and concept inference.
//!
//! Replaces hardcoded if-else keyword chains with a declarative TOML config.
//! Each node carries keyword matching rules, a topic name (for forced-topic
//! routing), and a MetaConcept (for meta-codebook routing). The graph is loaded
//! once at startup and queried per-prompt at inference time.

use std::collections::{HashMap, HashSet};
use serde::Deserialize;
use std::sync::{Mutex, OnceLock};

use crate::growformer_lang::MetaConcept;
use crate::text_keywords::keyword_matches_in_lower;

// ---------------------------------------------------------------------------
// TOML schema — deserialized directly from knowledge_graph.toml
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct KnowledgeGraphConfig {
    #[serde(default)]
    pub nodes: Vec<NodeConfig>,
    #[serde(default)]
    pub action_target_concepts: HashMap<String, String>,
    #[serde(default)]
    pub concept_keywords: HashMap<String, Vec<String>>,
}

#[derive(Debug, Deserialize)]
pub struct NodeConfig {
    pub topic: String,
    pub concept: String,
    #[serde(default)]
    pub category: String,
    #[serde(default = "default_priority")]
    pub priority: i32,
    #[serde(default)]
    pub rules: Vec<RuleConfig>,
}

fn default_priority() -> i32 { 10 }

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuleConfig {
    #[serde(default)]
    pub any: Vec<String>,
    #[serde(default)]
    pub all: Vec<String>,
    #[serde(default)]
    pub not: Vec<String>,
}

// ---------------------------------------------------------------------------
// Compiled graph — optimized for fast matching
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct TopicRule {
    pub any: Vec<String>,
    pub all: Vec<String>,
    pub not: Vec<String>,
}

impl TopicRule {
    /// A rule matches when:
    /// - ALL `all` keywords are present (AND)
    /// - At least one `any` keyword is present (OR) — or `any` is empty
    /// - NONE of the `not` keywords are present (NOT)
    pub fn matches(&self, lower: &str) -> bool {
        if !self.not.is_empty() && self.not.iter().any(|kw| keyword_matches_in_lower(lower, kw)) {
            return false;
        }
        let all_ok = self.all.is_empty() || self.all.iter().all(|kw| keyword_matches_in_lower(lower, kw));
        let any_ok = self.any.is_empty() || self.any.iter().any(|kw| keyword_matches_in_lower(lower, kw));
        all_ok && any_ok
    }

    fn specificity(&self) -> usize {
        self.all.len() + self.any.len() + self.not.len()
    }
}

#[derive(Debug, Clone)]
pub struct TopicNode {
    pub topic: String,
    pub concept: MetaConcept,
    pub category: String,
    pub priority: i32,
    pub rules: Vec<TopicRule>,
}

impl TopicNode {
    pub fn matches(&self, lower: &str) -> Option<usize> {
        for rule in &self.rules {
            if rule.matches(lower) {
                return Some(rule.specificity());
            }
        }
        None
    }
}

#[derive(Debug, Clone)]
pub struct TopicGraph {
    nodes: Vec<TopicNode>,
    action_target_map: HashMap<String, MetaConcept>,
    concept_keywords: HashMap<MetaConcept, Vec<String>>,
}

impl TopicGraph {
    /// Load from a TOML config file.
    pub fn from_toml(toml_str: &str) -> Result<Self, String> {
        Self::from_toml_impl(toml_str, true)
    }

    /// Parse TOML without the standard startup log line (for overlay merges).
    pub fn from_toml_quiet(toml_str: &str) -> Result<Self, String> {
        Self::from_toml_impl(toml_str, false)
    }

    fn from_toml_impl(toml_str: &str, log: bool) -> Result<Self, String> {
        let config: KnowledgeGraphConfig = toml::from_str(toml_str)
            .map_err(|e| format!("Failed to parse knowledge_graph.toml: {}", e))?;

        let mut nodes = Vec::with_capacity(config.nodes.len());
        for nc in &config.nodes {
            let concept = parse_concept(&nc.concept);
            let rules: Vec<TopicRule> = nc.rules.iter().map(|r| TopicRule {
                any: r.any.clone(),
                all: r.all.clone(),
                not: r.not.clone(),
            }).collect();

            if rules.is_empty() {
                return Err(format!("Node '{}' has no rules", nc.topic));
            }

            nodes.push(TopicNode {
                topic: nc.topic.clone(),
                concept,
                category: nc.category.clone(),
                priority: nc.priority,
                rules,
            });
        }

        // Sort by descending priority for fast first-match
        nodes.sort_by(|a, b| b.priority.cmp(&a.priority));

        let mut action_target_map = HashMap::new();
        for (k, v) in &config.action_target_concepts {
            action_target_map.insert(k.clone(), parse_concept(v));
        }

        let mut concept_keywords = HashMap::new();
        for (k, v) in &config.concept_keywords {
            concept_keywords.insert(parse_concept(k), v.clone());
        }

        if log && crate::infer_log::infer_trace_enabled() {
            println!("  [topic-graph] loaded {} nodes, {} action_target mappings, {} concept keyword sets",
                nodes.len(), action_target_map.len(), concept_keywords.len());
        }

        Ok(TopicGraph { nodes, action_target_map, concept_keywords })
    }

    /// Load from a file path.
    pub fn from_file(path: &str) -> Result<Self, String> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read {}: {}", path, e))?;
        Self::from_toml(&content)
    }

    /// Append nodes and merge maps from another graph (e.g. sentiment NL hints).
    pub fn merge_overlay(mut self, mut other: TopicGraph) -> Self {
        let added = other.nodes.len();
        self.nodes.append(&mut other.nodes);
        self.nodes.sort_by(|a, b| b.priority.cmp(&a.priority));

        for (k, v) in other.action_target_map {
            self.action_target_map.insert(k, v);
        }
        for (concept, kws) in other.concept_keywords {
            self.concept_keywords
                .entry(concept)
                .or_default()
                .extend(kws);
        }
        if crate::infer_log::infer_trace_enabled() {
            println!("  [topic-graph] merged overlay: +{} nodes (total {} nodes)",
                added, self.nodes.len());
        }
        self
    }

    pub fn node_count(&self) -> usize { self.nodes.len() }

    /// Collect all unique keywords from all rule `any`/`all` lists
    /// and `concept_keywords` arrays. Used as the saliency lexicon
    /// for salient span masking during training.
    pub fn all_keywords(&self) -> Vec<String> {
        let mut set = std::collections::HashSet::new();
        for node in &self.nodes {
            for rule in &node.rules {
                for kw in &rule.any {
                    set.insert(kw.to_ascii_lowercase());
                }
                for kw in &rule.all {
                    set.insert(kw.to_ascii_lowercase());
                }
            }
        }
        for keywords in self.concept_keywords.values() {
            for kw in keywords {
                set.insert(kw.to_ascii_lowercase());
            }
        }
        set.into_iter().collect()
    }

    // -----------------------------------------------------------------------
    // Topic inference (replaces infer_operation_topic)
    // -----------------------------------------------------------------------

    /// Match a query against all nodes and return the best topic.
    /// Returns None if no node matches.
    pub fn infer_topic(&self, text: &str) -> Option<String> {
        let lower = text.to_lowercase();
        let mut best: Option<(&TopicNode, usize)> = None;

        for node in &self.nodes {
            // Skip concept-only nodes (empty topic)
            if node.topic.is_empty() { continue; }

            if let Some(specificity) = node.matches(&lower) {
                let dominated = best.as_ref().map_or(false, |(prev, prev_spec)| {
                    prev.priority > node.priority
                        || (prev.priority == node.priority && *prev_spec >= specificity)
                });
                if !dominated {
                    best = Some((node, specificity));
                }
            }
        }

        best.map(|(node, _)| node.topic.clone())
    }

    // -----------------------------------------------------------------------
    // Concept inference (replaces infer_concept)
    // -----------------------------------------------------------------------

    /// Infer the MetaConcept for a query, with optional training-time overrides.
    pub fn infer_concept(
        &self,
        text: &str,
        semantic_intent: Option<&str>,
        action_target: Option<&str>,
    ) -> MetaConcept {
        let lower = text.to_lowercase();

        // Training-time: action_target is ground truth
        if let Some(target) = action_target {
            let t = target.to_lowercase();
            if !t.is_empty() {
                // Check direct matches first
                if let Some(concept) = self.action_target_map.get(&t) {
                    return concept.clone();
                }
                // Check substring matches
                for (key, concept) in &self.action_target_map {
                    if t.contains(key.as_str()) {
                        return concept.clone();
                    }
                }
            }
        }

        // Check concept_keywords arrays (broad InformationTheory, BinaryArithmetic, etc.)
        for (concept, keywords) in &self.concept_keywords {
            if keywords
                .iter()
                .any(|kw| keyword_matches_in_lower(&lower, kw.as_str()))
            {
                return concept.clone();
            }
        }

        // Match against all nodes (including concept-only nodes)
        let mut best: Option<(&TopicNode, usize)> = None;
        for node in &self.nodes {
            if let Some(specificity) = node.matches(&lower) {
                let dominated = best.as_ref().map_or(false, |(prev, prev_spec)| {
                    prev.priority > node.priority
                        || (prev.priority == node.priority && *prev_spec >= specificity)
                });
                if !dominated {
                    best = Some((node, specificity));
                }
            }
        }

        if let Some((node, _)) = best {
            return node.concept.clone();
        }

        // Fallback heuristics
        let intent = semantic_intent.unwrap_or("").to_lowercase();
        let target = action_target.unwrap_or("").to_lowercase();

        if intent.contains("coding") || target.contains("coding")
            || lower.contains("implement") || lower.contains("write a") || lower.contains("create a")
        {
            return MetaConcept::FunctionDefinition;
        }

        MetaConcept::GeneralKnowledge
    }

    // -----------------------------------------------------------------------
    // action_target → concept (replaces concept_from_action_target)
    // -----------------------------------------------------------------------

    pub fn concept_from_action_target(&self, target: &str) -> Option<MetaConcept> {
        let t = target.to_lowercase();
        // Broad coding categories should return None to allow keyword matching
        if t == "coding_general" || t == "coding" || t == "coding_patterns" {
            return None;
        }
        // Direct match
        if let Some(c) = self.action_target_map.get(&t) {
            return Some(c.clone());
        }
        // Substring match
        for (key, concept) in &self.action_target_map {
            if t.contains(key.as_str()) {
                return Some(concept.clone());
            }
        }
        None
    }
}

// ---------------------------------------------------------------------------
// MetaConcept parsing from string
// ---------------------------------------------------------------------------

// Log each unknown TOML concept string at most once per process (avoid spam on large overlays).
static UNKNOWN_CONCEPT_WARNINGS: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();

fn parse_concept(s: &str) -> MetaConcept {
    match s {
        "BinaryArithmetic" => MetaConcept::BinaryArithmetic,
        "UnaryOperation" => MetaConcept::UnaryOperation,
        "DataStructure" => MetaConcept::DataStructure,
        "SearchAlgorithm" => MetaConcept::SearchAlgorithm,
        "SortAlgorithm" => MetaConcept::SortAlgorithm,
        "PatternMatching" => MetaConcept::PatternMatching,
        "EnumAlgebraic" => MetaConcept::EnumAlgebraic,
        "StructDefinition" => MetaConcept::StructDefinition,
        "TraitInterface" => MetaConcept::TraitInterface,
        "ErrorHandling" => MetaConcept::ErrorHandling,
        "Iteration" => MetaConcept::Iteration,
        "AsyncConcurrency" => MetaConcept::AsyncConcurrency,
        "FunctionDefinition" => MetaConcept::FunctionDefinition,
        "Composition" => MetaConcept::Composition,
        "Testing" => MetaConcept::Testing,
        "Debugging" => MetaConcept::Debugging,
        "Refactoring" => MetaConcept::Refactoring,
        "Support" => MetaConcept::Support,
        "Conversation" => MetaConcept::Conversation,
        "PetCompanion" => MetaConcept::PetCompanion,
        "GeneralKnowledge" => MetaConcept::GeneralKnowledge,
        "InformationTheory" => MetaConcept::InformationTheory,
        "CausalReasoning" => MetaConcept::CausalReasoning,
        _ => {
            let set = UNKNOWN_CONCEPT_WARNINGS.get_or_init(|| Mutex::new(HashSet::new()));
            let first = set.lock().unwrap().insert(s.to_string());
            if first {
                eprintln!(
                    "[topic-graph] WARNING: unknown concept '{}', defaulting to GeneralKnowledge",
                    s
                );
            }
            MetaConcept::GeneralKnowledge
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_graph() -> TopicGraph {
        let toml = r#"
[[nodes]]
topic = "addition_operation"
concept = "BinaryArithmetic"
category = "arithmetic"
priority = 20
[[nodes.rules]]
any = ["addition", "add two"]

[[nodes]]
topic = "structural"
concept = "TraitInterface"
category = "design_pattern"
priority = 15
[[nodes.rules]]
all = ["decorator"]
not = ["python"]

[[nodes]]
topic = "decorator_operation"
concept = "FunctionDefinition"
category = "coding"
priority = 18
[[nodes.rules]]
all = ["decorator", "python"]

[[nodes]]
topic = "lru_cache_operation"
concept = "DataStructure"
category = "data_structure"
priority = 18
[[nodes.rules]]
any = ["lru cache"]
[[nodes.rules]]
all = ["lru", "cache"]

[[nodes]]
topic = ""
concept = "Support"
category = "concept_only"
priority = 10
[[nodes.rules]]
any = ["password", "account"]

[action_target_concepts]
arithmetic = "BinaryArithmetic"
support = "Support"

[concept_keywords]
BinaryArithmetic = ["add two", "calculator"]
"#;
        TopicGraph::from_toml(toml).unwrap()
    }

    #[test]
    fn test_infer_topic_arithmetic() {
        let g = test_graph();
        assert_eq!(g.infer_topic("write an addition function"), Some("addition_operation".into()));
    }

    #[test]
    fn test_infer_topic_decorator_rust() {
        let g = test_graph();
        assert_eq!(g.infer_topic("use decorator for extensibility"), Some("structural".into()));
    }

    #[test]
    fn test_infer_topic_decorator_python() {
        let g = test_graph();
        assert_eq!(g.infer_topic("write a python decorator"), Some("decorator_operation".into()));
    }

    #[test]
    fn test_infer_topic_lru() {
        let g = test_graph();
        assert_eq!(g.infer_topic("implement an LRU cache"), Some("lru_cache_operation".into()));
    }

    #[test]
    fn test_infer_topic_no_match() {
        let g = test_graph();
        assert_eq!(g.infer_topic("explain quantum mechanics"), None);
    }

    /// Paxos Labs (company) must not hit `consensus_algorithms` via bare "paxos" (regression).
    #[test]
    fn test_infer_topic_paxos_labs_skips_consensus_algorithms() {
        let toml = r#"
[[nodes]]
topic = "consensus_algorithms"
concept = "TraitInterface"
category = "architecture"
priority = 15
[[nodes.rules]]
any = ["raft", "zab", "distributed consensus", "consensus protocol", "consensus algorithm", "multi-paxos", "viewstamped replication", "byzantine fault", "pbft"]
[[nodes.rules]]
any = ["consensus", "raft", "paxos", "zab"]
not = ["paxos labs", "paxos trust", "paxos global", "paxos stablecoin", "paxos dollar", "paxos usd", "paxos inc", "paxos pax", "paxos gold"]
[[nodes.rules]]
all = ["paxos", "protocol"]
[[nodes.rules]]
all = ["paxos", "algorithm"]
[[nodes.rules]]
all = ["paxos", "replication"]
[[nodes.rules]]
all = ["paxos", "distributed"]
[[nodes.rules]]
all = ["paxos", "leader"]
[[nodes.rules]]
all = ["paxos", "quorum"]
[[nodes.rules]]
all = ["explain", "paxos"]
"#;
        let g = TopicGraph::from_toml(toml).unwrap();
        assert_eq!(
            g.infer_topic("Paxos Labs secured 12M USD for crypto yield platform Amplify"),
            None
        );
        assert_eq!(
            g.infer_topic("Explain the Paxos protocol for distributed consensus"),
            Some("consensus_algorithms".into())
        );
    }

    #[test]
    fn test_infer_concept_with_action_target() {
        let g = test_graph();
        assert_eq!(
            g.infer_concept("anything", None, Some("arithmetic")),
            MetaConcept::BinaryArithmetic,
        );
    }

    #[test]
    fn test_infer_concept_from_text() {
        let g = test_graph();
        assert_eq!(
            g.infer_concept("help me reset my password", None, None),
            MetaConcept::Support,
        );
    }

    #[test]
    fn test_infer_concept_keyword_array() {
        let g = test_graph();
        assert_eq!(
            g.infer_concept("add two numbers", None, None),
            MetaConcept::BinaryArithmetic,
        );
    }

    #[test]
    fn test_concept_from_action_target_broad() {
        let g = test_graph();
        assert!(g.concept_from_action_target("coding_general").is_none());
        assert_eq!(g.concept_from_action_target("support"), Some(MetaConcept::Support));
    }
}
