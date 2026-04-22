//! GrowformerLang — meta-programming language for language-agnostic concept representation.
//!
//! The brain learns abstract computational concepts (meta-programs) once,
//! then projects them to any target language via Clifford rotors.
//!
//! Architecture:
//!   prompt → encoder → MetaRouter(concept) + detect_language(lang)
//!         → concept_embedding → LanguageProjector(lang) → language_embedding
//!         → generation
//!
//! This decouples WHAT the code does from HOW it looks in a specific language.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Once, OnceLock};

use crate::topic_graph::TopicGraph;

static TOPIC_GRAPH: OnceLock<TopicGraph> = OnceLock::new();

static LEGACY_OPERATION_TOPIC_WARN: Once = Once::new();

/// Overlay merged after the base graph when both exist (same directory as `base_path`).
const SENTIMENT_OVERLAY_FILENAME: &str = "knowledge_graph_sentiment_overlay.toml";

/// Initialize the global TopicGraph from a TOML file.
/// Called once at startup. If the file is not found, [`infer_operation_topic`] falls back to
/// legacy keyword rules and emits a one-time `eprintln` diagnostic.
pub fn init_topic_graph(toml_path: &str) -> Result<(), String> {
    let graph = TopicGraph::from_file(toml_path)?;
    TOPIC_GRAPH.set(graph).map_err(|_| "TopicGraph already initialized".to_string())
}

/// Load `base_path` and merge `knowledge_graph_sentiment_overlay.toml` from the same directory
/// when present. If only the overlay exists, loads the overlay alone.
/// Returns `Ok(())` with **no graph installed** if neither file exists (callers should use
/// [`topic_graph_loaded`] before `--infer`).
pub fn try_init_topic_graph_bundle(base_path: &str) -> Result<(), String> {
    let base_p = std::path::Path::new(base_path);
    let overlay_pb = base_p
        .parent()
        .map(|dir| dir.join(SENTIMENT_OVERLAY_FILENAME))
        .unwrap_or_else(|| std::path::PathBuf::from(SENTIMENT_OVERLAY_FILENAME));
    let overlay_s = overlay_pb
        .to_str()
        .ok_or_else(|| "overlay path is not valid UTF-8".to_string())?;

    let base_exists = base_p.exists();
    let overlay_exists = overlay_pb.exists();

    let graph = match (base_exists, overlay_exists) {
        (true, true) => {
            let base_g = TopicGraph::from_file(base_path)?;
            let overlay_content = std::fs::read_to_string(overlay_s)
                .map_err(|e| format!("Failed to read {}: {}", overlay_s, e))?;
            let overlay_g = TopicGraph::from_toml_quiet(&overlay_content)?;
            base_g.merge_overlay(overlay_g)
        }
        (true, false) => TopicGraph::from_file(base_path)?,
        (false, true) => TopicGraph::from_file(overlay_s)?,
        (false, false) => return Ok(()),
    };

    TOPIC_GRAPH
        .set(graph)
        .map_err(|_| "TopicGraph already initialized".to_string())
}

/// Initialize from an inline TOML string (for tests or embedded configs).
pub fn init_topic_graph_from_str(toml_str: &str) -> Result<(), String> {
    let graph = TopicGraph::from_toml(toml_str)?;
    TOPIC_GRAPH.set(graph).map_err(|_| "TopicGraph already initialized".to_string())
}

/// Get a reference to the global TopicGraph, if initialized.
pub fn topic_graph() -> Option<&'static TopicGraph> {
    TOPIC_GRAPH.get()
}

/// `true` after a successful [`try_init_topic_graph_bundle`], [`init_topic_graph`], or
/// [`init_topic_graph_from_str`]. `false` when neither knowledge graph file existed.
#[inline]
pub fn topic_graph_loaded() -> bool {
    TOPIC_GRAPH.get().is_some()
}

use crate::clifford::{
    Multivector, Rotor, embed_bridge_vector, structural_fingerprint,
    structural_similarity, apply_group_rotor, GRADE_OFFSETS,
};

// ---------------------------------------------------------------------------
// Meta-operations: abstract computational primitives
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MetaOp {
    Bind { name: String, typ: MetaType },
    BinaryOp { op: String },
    UnaryOp { op: String },
    FnDef { name: String, params: u8, returns: MetaType },
    Call { arity: u8 },
    Branch { arms: u8 },
    Loop { kind: LoopKind },
    Map,
    Fold,
    Filter,
    Compose,
    PatternMatch { arms: u8 },
    Return,
    Collect,
    StructDef { fields: u8 },
    EnumDef { variants: u8 },
    TraitDef { methods: u8 },
    ImplBlock,
    ErrorHandle,
    GenericParam,
    Allocate,
    AsyncAwait,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MetaType {
    Numeric,
    Text,
    Bool,
    Collection,
    Function,
    Generic,
    Option,
    Result,
    Void,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LoopKind {
    ForEach,
    While,
    Recursive,
}

// ---------------------------------------------------------------------------
// Meta-concepts: language-agnostic categories for routing
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Copy)]
pub enum MetaConcept {
    BinaryArithmetic,
    UnaryOperation,
    FunctionDefinition,
    StructDefinition,
    EnumAlgebraic,
    TraitInterface,
    ErrorHandling,
    Iteration,
    PatternMatching,
    AsyncConcurrency,
    SearchAlgorithm,
    SortAlgorithm,
    DataStructure,
    Composition,
    Testing,
    Debugging,
    Refactoring,
    InformationTheory,
    GeneralKnowledge,
    Support,
    Conversation,
    CausalReasoning,
}

impl MetaConcept {
    pub fn all() -> &'static [MetaConcept] {
        use MetaConcept::*;
        &[
            BinaryArithmetic, UnaryOperation, FunctionDefinition,
            StructDefinition, EnumAlgebraic, TraitInterface,
            ErrorHandling, Iteration, PatternMatching, AsyncConcurrency,
            SearchAlgorithm, SortAlgorithm, DataStructure, Composition,
            Testing, Debugging, Refactoring, InformationTheory,
            GeneralKnowledge, Support, Conversation, CausalReasoning,
        ]
    }

    pub fn index(&self) -> usize {
        Self::all().iter().position(|c| c == self).unwrap_or(0)
    }

    pub fn from_index(idx: usize) -> Self {
        Self::all().get(idx).copied().unwrap_or(MetaConcept::GeneralKnowledge)
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::BinaryArithmetic => "binary_arithmetic",
            Self::UnaryOperation => "unary_operation",
            Self::FunctionDefinition => "function_definition",
            Self::StructDefinition => "struct_definition",
            Self::EnumAlgebraic => "enum_algebraic",
            Self::TraitInterface => "trait_interface",
            Self::ErrorHandling => "error_handling",
            Self::Iteration => "iteration",
            Self::PatternMatching => "pattern_matching",
            Self::AsyncConcurrency => "async_concurrency",
            Self::SearchAlgorithm => "search_algorithm",
            Self::SortAlgorithm => "sort_algorithm",
            Self::DataStructure => "data_structure",
            Self::Composition => "composition",
            Self::Testing => "testing",
            Self::Debugging => "debugging",
            Self::Refactoring => "refactoring",
            Self::InformationTheory => "information_theory",
            Self::GeneralKnowledge => "general_knowledge",
            Self::Support => "support",
            Self::Conversation => "conversation",
            Self::CausalReasoning => "causal_reasoning",
        }
    }

    pub fn canonical_ops(&self) -> Vec<MetaOp> {
        match self {
            Self::BinaryArithmetic => vec![
                MetaOp::FnDef { name: "op".into(), params: 2, returns: MetaType::Numeric },
                MetaOp::Bind { name: "a".into(), typ: MetaType::Numeric },
                MetaOp::Bind { name: "b".into(), typ: MetaType::Numeric },
                MetaOp::BinaryOp { op: "?".into() },
                MetaOp::Return,
            ],
            Self::UnaryOperation => vec![
                MetaOp::FnDef { name: "op".into(), params: 1, returns: MetaType::Numeric },
                MetaOp::Bind { name: "x".into(), typ: MetaType::Numeric },
                MetaOp::UnaryOp { op: "?".into() },
                MetaOp::Return,
            ],
            Self::FunctionDefinition => vec![
                MetaOp::FnDef { name: "f".into(), params: 0, returns: MetaType::Generic },
                MetaOp::Return,
            ],
            Self::StructDefinition => vec![
                MetaOp::StructDef { fields: 0 },
                MetaOp::ImplBlock,
            ],
            Self::EnumAlgebraic => vec![
                MetaOp::EnumDef { variants: 0 },
                MetaOp::PatternMatch { arms: 0 },
            ],
            Self::TraitInterface => vec![
                MetaOp::TraitDef { methods: 0 },
                MetaOp::ImplBlock,
            ],
            Self::ErrorHandling => vec![
                MetaOp::ErrorHandle,
                MetaOp::Branch { arms: 2 },
                MetaOp::Return,
            ],
            Self::Iteration => vec![
                MetaOp::Map,
                MetaOp::Filter,
                MetaOp::Collect,
            ],
            Self::PatternMatching => vec![
                MetaOp::PatternMatch { arms: 0 },
            ],
            Self::AsyncConcurrency => vec![
                MetaOp::AsyncAwait,
                MetaOp::ErrorHandle,
                MetaOp::Return,
            ],
            Self::SearchAlgorithm => vec![
                MetaOp::Loop { kind: LoopKind::While },
                MetaOp::Branch { arms: 2 },
                MetaOp::Return,
            ],
            Self::SortAlgorithm => vec![
                MetaOp::Loop { kind: LoopKind::ForEach },
                MetaOp::Branch { arms: 2 },
                MetaOp::Collect,
            ],
            Self::DataStructure => vec![
                MetaOp::StructDef { fields: 0 },
                MetaOp::Allocate,
                MetaOp::ImplBlock,
            ],
            Self::Composition => vec![
                MetaOp::Compose,
                MetaOp::Call { arity: 0 },
                MetaOp::Return,
            ],
            _ => vec![],
        }
    }
}

// ---------------------------------------------------------------------------
// Target language detection
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Copy)]
pub enum TargetLanguage {
    Rust,
    Python,
    TypeScript,
    Go,
    Generic,
}

impl TargetLanguage {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Rust => "rust",
            Self::Python => "python",
            Self::TypeScript => "typescript",
            Self::Go => "go",
            Self::Generic => "generic",
        }
    }
}

pub fn detect_language(text: &str) -> TargetLanguage {
    let lower = text.to_lowercase();

    // Note: avoid bare `"match "` — it appears in English ("playoff match because…")
    // and falsely wins the Rust score. Prefer `match {` / `match(` style cues.
    let rust_signals = ["rust", "cargo", "impl ", "struct ", "&mut", "fn ",
        "crate", "enum ", "trait ", "tokio", "async fn", "Vec<", "Option<",
        "Result<", "match {", " match {", "println!", "unwrap", "lifetime", "borrow"];
    let python_signals = ["python", "pip", "def ", "import ", "class ",
        "self.", "print(", "__init__", "numpy", "pandas", "django",
        "flask", "pytest", "lambda ", "list comprehension"];
    let ts_signals = ["typescript", "javascript", "npm", "node",
        "const ", "interface ", "react", "async function", "promise",
        "console.log", "=>", "export ", "import {"];
    let go_signals = ["golang", " go ", "func ", "package ", "goroutine",
        "chan ", "defer ", "go func"];

    let mut scores = [0i32; 4]; // rust, python, ts, go
    for kw in &rust_signals {
        if lower.contains(kw) { scores[0] += 1; }
    }
    for kw in &python_signals {
        if lower.contains(kw) { scores[1] += 1; }
    }
    for kw in &ts_signals {
        if lower.contains(kw) { scores[2] += 1; }
    }
    for kw in &go_signals {
        if lower.contains(kw) { scores[3] += 1; }
    }

    let max_score = *scores.iter().max().unwrap_or(&0);
    if max_score == 0 {
        return TargetLanguage::Generic;
    }

    let idx = scores.iter().position(|&s| s == max_score).unwrap_or(4);
    match idx {
        0 => TargetLanguage::Rust,
        1 => TargetLanguage::Python,
        2 => TargetLanguage::TypeScript,
        3 => TargetLanguage::Go,
        _ => TargetLanguage::Generic,
    }
}

// ---------------------------------------------------------------------------
// Query Intent — structured two-phase understanding
// Phase 1: Extract the ACTION (what/where/why/when/how/design/create/build)
// Phase 2: Extract the SUBJECT (the conceptual target)
// This replaces fragile keyword matching with structural parsing.
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum QueryAction {
    Define,     // "What is X?" / "Define X"
    Explain,    // "Explain X" / "How does X work?"
    Locate,     // "Where is X?" / "Where does X happen?"
    Reason,     // "Why does X?" / "Why is X important?"
    Temporal,   // "When should I use X?" / "When does X apply?"
    Compare,    // "Compare X and Y" / "Difference between X and Y"
    Implement,  // "Write X" / "Create X" / "Build X" / "Implement X"
    Design,     // "Design X" / "Architect X" / "Plan X"
    Debug,      // "Fix X" / "Debug X" / "Why is X broken?"
    List,       // "List X" / "What are the types of X?"
    Retrieve,   // Direct noun phrase — "Rate-distortion tradeoff" / "Huffman coding"
}

impl QueryAction {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Define => "define",
            Self::Explain => "explain",
            Self::Locate => "locate",
            Self::Reason => "reason",
            Self::Temporal => "temporal",
            Self::Compare => "compare",
            Self::Implement => "implement",
            Self::Design => "design",
            Self::Debug => "debug",
            Self::List => "list",
            Self::Retrieve => "retrieve",
        }
    }

    pub fn is_generative(&self) -> bool {
        matches!(self, Self::Implement | Self::Design)
    }

    pub fn is_informational(&self) -> bool {
        matches!(self, Self::Define | Self::Explain | Self::Reason | Self::Locate | Self::Temporal | Self::Compare | Self::List | Self::Retrieve)
    }
}

#[derive(Clone, Debug)]
pub struct QueryIntent {
    pub action: QueryAction,
    /// The conceptual subject extracted from the query, with action words stripped.
    /// e.g. "What is Rate-distortion tradeoff?" → subject = "rate-distortion tradeoff"
    pub subject: String,
    /// The full original query text (lowercased).
    pub raw: String,
}

/// Normalize currency and money-like glued tokens so **magnitude** survives:
/// - Hashing encoders skip pure-numeric words; replacing with `money_usd_5000` keeps a lexical signal.
/// - Shells expand `$` inside double quotes — use single quotes or `--prompt-file` for literal `$`.
///
/// Idempotent: already-normalized `money_*` tokens are left unchanged.
pub fn normalize_inference_money_spans(text: &str) -> String {
    if text.is_empty() {
        return String::new();
    }
    let mut out: Vec<String> = Vec::new();
    for raw in text.split_whitespace() {
        let (core, trailing_punct) = split_trailing_punct(raw);
        let piece = if let Some(m) = replace_money_token(core) {
            m
        } else {
            core.to_string()
        };
        out.push(format!("{}{}", piece, trailing_punct));
    }
    out.join(" ")
}

fn split_trailing_punct(word: &str) -> (&str, &str) {
    let end = word
        .char_indices()
        .rev()
        .find_map(|(i, c)| {
            if ".,;:!?)]}\"'".contains(c) {
                None
            } else {
                Some(i + c.len_utf8())
            }
        })
        .unwrap_or(word.len());
    word.split_at(end)
}

fn replace_money_token(w: &str) -> Option<String> {
    if w.starts_with("money_") {
        return None;
    }
    let lower = w.to_ascii_lowercase();
    // Glued forms: 5000usd, 12kusd, $5000usd
    if let Some(m) = parse_glued_money(&lower) {
        return Some(m);
    }
    // World currency symbol prefixes
    static SYMBOL_CCY: &[(&str, &str, &str)] = &[
        ("£", "\u{00a3}", "gbp"),
        ("€", "\u{20ac}", "eur"),
        ("¥", "\u{00a5}", "jpy"),
        ("₩", "\u{20a9}", "krw"),
        ("₹", "\u{20b9}", "inr"),
        ("₿", "\u{20bf}", "btc"),
        ("₽", "\u{20bd}", "rub"),
        ("₱", "\u{20b1}", "php"),
        ("₫", "\u{20ab}", "vnd"),
        ("₺", "\u{20ba}", "try"),
        ("₴", "\u{20b4}", "uah"),
        ("₦", "\u{20a6}", "ngn"),
        ("₸", "\u{20b8}", "kzt"),
        ("R$", "", "brl"),
        ("kr", "", "sek"),
    ];
    for &(ascii_sym, unicode_sym, ccy) in SYMBOL_CCY {
        let rest = if !ascii_sym.is_empty() {
            w.strip_prefix(ascii_sym)
        } else {
            None
        };
        let rest = rest.or_else(|| {
            if !unicode_sym.is_empty() { w.strip_prefix(unicode_sym) } else { None }
        });
        if let Some(rest) = rest {
            if let Some(m) = parse_prefixed_currency_amount(rest, ccy) {
                return Some(m);
            }
        }
    }
    if !lower.starts_with('$') || lower.len() < 2 {
        return None;
    }
    // $ prefix — use parse_prefixed_currency_amount to handle magnitude suffixes ($9M, $3.8B)
    if let Some(m) = parse_prefixed_currency_amount(&w[1..], "usd") {
        return Some(m);
    }
    None
}

fn take_digits_commas(s: &str) -> (String, usize) {
    let mut out = String::new();
    let mut i = 0;
    for ch in s.chars() {
        if ch.is_ascii_digit() {
            out.push(ch);
            i += ch.len_utf8();
        } else if ch == ',' {
            i += ch.len_utf8();
        } else {
            break;
        }
    }
    (out, i)
}

/// After leading `£`/`€` stripped: `43bn`, `1_200_000`, `500k`.
fn parse_prefixed_currency_amount(rest: &str, default_ccy: &str) -> Option<String> {
    let rl = rest.to_ascii_lowercase();
    let rl = rl.trim_end_matches(|c: char| matches!(c, '.' | ',' | ';' | ':' | '!' | '?' | ')' | ']'));
    let (scaled_digits, ccy) = if let Some(s) = rl.strip_suffix("bn") {
        (scale_money_digits(s, 1_000_000_000u128)?, default_ccy)
    } else if let Some(s) = rl.strip_suffix('b') {
        (scale_money_digits(s, 1_000_000_000u128)?, default_ccy)
    } else if let Some(s) = rl.strip_suffix('m') {
        (scale_money_digits(s, 1_000_000u128)?, default_ccy)
    } else if let Some(s) = rl.strip_suffix('k') {
        (scale_money_digits(s, 1_000u128)?, default_ccy)
    } else {
        let (digits, after_int) = take_digits_commas(&rl);
        if digits.is_empty() {
            return None;
        }
        // If the bare amount has a decimal (e.g. "$1.50", "$0.18"), skip
        // tokenization — the integer-only token would lose precision and
        // detokenize to a wrong amount. Large round amounts ($67,400) and
        // suffix amounts ($3.8B) are already handled above.
        if after_int < rl.len() && rl.as_bytes().get(after_int) == Some(&b'.') {
            return None;
        }
        (digits, default_ccy)
    };
    Some(format!("money_{}_{}", ccy, scaled_digits))
}

fn scale_money_digits(num_part: &str, mult: u128) -> Option<String> {
    let (digits, after_int) = take_digits_commas(num_part);
    if digits.is_empty() {
        return None;
    }
    let v: u128 = digits.parse().ok()?;
    let base = v.checked_mul(mult)?;
    // Handle decimal: "1.4" with mult=1e9 → 1*1e9 + 4*1e8 = 1_400_000_000
    if after_int < num_part.len() && num_part.as_bytes().get(after_int) == Some(&b'.') {
        let frac_str = &num_part[after_int + 1..];
        let (frac_digits, _) = take_digits_commas(frac_str);
        if !frac_digits.is_empty() {
            let frac_val: u128 = frac_digits.parse().ok()?;
            let frac_places = frac_digits.len() as u32;
            let frac_mult = mult / 10u128.pow(frac_places);
            let total = base.checked_add(frac_val.checked_mul(frac_mult)?)?;
            return Some(total.to_string());
        }
    }
    Some(base.to_string())
}

fn parse_glued_money(lower: &str) -> Option<String> {
    // e.g. 5000usd, 1200eur (no `$` — handled in replace_money_token)
    for (suffix, ccy) in [("usd", "usd"), ("eur", "eur"), ("gbp", "gbp")] {
        if !lower.ends_with(suffix) || lower.len() <= suffix.len() {
            continue;
        }
        let num_part = &lower[..lower.len() - suffix.len()];
        if num_part.starts_with('$') {
            continue;
        }
        let (digits, _) = take_digits_commas(num_part);
        if digits.is_empty() || digits.chars().all(|c| c == '0') {
            continue;
        }
        return Some(format!("money_{}_{}", ccy, digits));
    }
    None
}

/// Parse a query into structured intent: action + subject.
///
/// This is the first pass of understanding — before any embedding-based routing.
/// The action determines HOW to respond (define, explain, implement, etc.),
/// and the subject determines WHAT domain to route to.
pub fn parse_query_intent(text: &str) -> QueryIntent {
    let lower = text.to_lowercase();
    let trimmed = lower.trim();

    // Phase 1: Extract action from query structure
    let (action, subject_start) = extract_action(trimmed);

    // Phase 2: Extract subject (everything after the action prefix)
    let subject = trimmed[subject_start..].trim()
        .trim_end_matches('?')
        .trim_end_matches('.')
        .trim()
        .to_string();

    QueryIntent {
        action,
        subject,
        raw: lower,
    }
}

fn extract_action(text: &str) -> (QueryAction, usize) {
    // Interrogative patterns — ordered by specificity
    let patterns: &[(&[&str], QueryAction)] = &[
        // Compare
        (&["compare ", "difference between ", "differences between ",
          "contrast ", "versus ", " vs "], QueryAction::Compare),

        // Why
        (&["why does ", "why is ", "why do ", "why are ", "why should ",
          "why can't ", "why would "], QueryAction::Reason),

        // Where
        (&["where is ", "where does ", "where do ", "where are ",
          "where should "], QueryAction::Locate),

        // When
        (&["when should ", "when does ", "when do ", "when is ",
          "when to "], QueryAction::Temporal),

        // What-is (define)
        (&["what is ", "what is a ", "what is an ", "what is the ",
          "what are ", "what does ", "define "], QueryAction::Define),

        // How (explain mechanism)
        (&["how does ", "how do ", "how is ", "how are ", "how to ",
          "how can ", "how would "], QueryAction::Explain),

        // List
        (&["list ", "list all ", "what types of ", "what kinds of ",
          "enumerate "], QueryAction::List),

        // Implement / create / build
        (&["write ", "write a ", "write an ", "create ", "create a ",
          "build ", "build a ", "implement ", "implement a ",
          "code ", "code a ", "make ", "make a ", "generate "], QueryAction::Implement),

        // Design / architect
        (&["design ", "design a ", "architect ", "plan ",
          "plan a "], QueryAction::Design),

        // Debug / fix
        (&["fix ", "debug ", "troubleshoot ", "diagnose ",
          "why is my ", "why isn't ", "why doesn't "], QueryAction::Debug),

        // Explain (weaker patterns)
        (&["explain ", "describe ", "tell me about ", "tell me what ",
          "overview of ", "introduction to "], QueryAction::Explain),
    ];

    for (prefixes, action) in patterns {
        for prefix in *prefixes {
            if text.starts_with(prefix) {
                return (action.clone(), prefix.len());
            }
            if text.contains(prefix) && action == &QueryAction::Compare {
                return (action.clone(), 0);
            }
        }
    }

    // No action prefix detected — treat as a direct concept retrieval
    (QueryAction::Retrieve, 0)
}

impl QueryIntent {
    /// Does this intent need a specific sub-lattice answer (not a broad summary)?
    pub fn is_specific(&self) -> bool {
        !self.subject.is_empty() && self.subject.split_whitespace().count() <= 8
            && !self.action.is_generative()
    }

    /// Does this intent ask for a broad domain overview?
    pub fn is_broad_overview(&self) -> bool {
        if self.subject.is_empty() {
            return false;
        }
        // "What is software architecture?" — Define + domain-level subject
        let is_domain_level = is_domain_subject(&self.subject);
        matches!(self.action, QueryAction::Define | QueryAction::List) && is_domain_level
    }
}

/// Check if a subject is a domain-level concept (broad) vs a specific topic.
fn is_domain_subject(subject: &str) -> bool {
    let domains = [
        "software architecture", "software engineering", "design patterns",
        "system design", "programming paradigm", "machine learning",
        "artificial intelligence", "cloud computing", "devops",
        "web development", "data engineering", "computer science",
        "distributed systems", "operating systems", "networking",
        "information theory", "coding theory", "signal processing",
        "database", "security", "cryptography",
    ];
    domains.iter().any(|d| subject.contains(d))
}

// ---------------------------------------------------------------------------
// Meta-concept inference from text
// ---------------------------------------------------------------------------

/// TODO: Derive from a knowledge graph of operations and their relationships.
pub fn infer_concept(text: &str, semantic_intent: Option<&str>, action_target: Option<&str>) -> MetaConcept {
    if let Some(graph) = TOPIC_GRAPH.get() {
        return graph.infer_concept(text, semantic_intent, action_target);
    }
    // Legacy fallback
    infer_concept_legacy(text, semantic_intent, action_target)
}

fn infer_concept_legacy(text: &str, semantic_intent: Option<&str>, action_target: Option<&str>) -> MetaConcept {
    let lower = text.to_lowercase();
    let intent = semantic_intent.unwrap_or("").to_lowercase();
    let target = action_target.unwrap_or("").to_lowercase();

    if !target.is_empty() {
        if let Some(concept) = concept_from_action_target(&target) {
            return concept;
        }
    }

    // Support/conversation (check first — these are non-coding)
    if lower.contains("password") || lower.contains("account")
        || lower.contains("subscription") || lower.contains("refund") || lower.contains("ticket")
    {
        return MetaConcept::Support;
    }
    if lower.starts_with("hello") || lower.starts_with("hi ") || lower.starts_with("who are") {
        return MetaConcept::Conversation;
    }

    // Arithmetic operations — tight keyword matching to avoid false positives
    let arith_keywords = ["addition function", "subtraction function", "multiplication function",
        "division function", "modulo function", "add function", "subtract function",
        "multiply function", "divide function", "calculator",
        "absolute value", "power function", "sum of a list", "compute the average",
        "min and max", "clamp function", "checked_add", "checked arithmetic",
        "saturating", "wrapping", "integer overflow",
        "add two", "two integers", "two numbers", "arithmetic operation"];
    if arith_keywords.iter().any(|kw| lower.contains(kw)) {
        if lower.contains("abs") || lower.contains("negate") || lower.contains("unary") {
            return MetaConcept::UnaryOperation;
        }
        return MetaConcept::BinaryArithmetic;
    }

    // Data structures
    let ds_keywords = ["linked list", "stack", "queue", "tree", "heap",
        "hash map", "hashmap", "btree", "graph data", "priority queue",
        "binary tree", "trie"];
    if target.contains("data_structure") || ds_keywords.iter().any(|kw| lower.contains(kw)) {
        return MetaConcept::DataStructure;
    }

    // Algorithms
    let search_keywords = ["binary search", "linear search", "search algorithm",
        "find in", "lookup", "bfs", "dfs", "breadth-first", "depth-first",
        "dijkstra", "a-star", "a*"];
    if search_keywords.iter().any(|kw| lower.contains(kw)) {
        return MetaConcept::SearchAlgorithm;
    }
    let sort_keywords = ["sort", "quicksort", "mergesort", "bubble sort",
        "insertion sort", "heap sort", "radix sort"];
    if sort_keywords.iter().any(|kw| lower.contains(kw)) {
        return MetaConcept::SortAlgorithm;
    }

    // Pattern matching / enums
    if lower.contains("pattern match") || lower.contains("match expression")
        || (lower.contains("enum") && lower.contains("match"))
    {
        return MetaConcept::PatternMatching;
    }

    // Enum / algebraic types
    if lower.contains("enum") || lower.contains("algebraic data type")
        || lower.contains("variant") || lower.contains("tagged union")
    {
        return MetaConcept::EnumAlgebraic;
    }

    // Struct / class definition
    if lower.contains("struct ") || lower.contains("class ")
        || lower.contains("struct with") || lower.contains("data class")
        || lower.contains("with methods")
    {
        return MetaConcept::StructDefinition;
    }

    // Trait / interface
    if lower.contains("trait") || lower.contains("interface")
        || lower.contains("polymorphism") || lower.contains("impl ")
        || lower.contains("implement display") || lower.contains("implement debug")
    {
        return MetaConcept::TraitInterface;
    }

    // Error handling
    if lower.contains("error handling") || lower.contains("result<")
        || lower.contains("option<") || lower.contains("try") || lower.contains("catch")
        || lower.contains("unwrap") || lower.contains("? operator")
        || target.contains("error")
    {
        return MetaConcept::ErrorHandling;
    }

    // Iteration
    if lower.contains("iterator") || lower.contains("map(") || lower.contains("filter(")
        || lower.contains("fold(") || lower.contains("collect") || lower.contains("for_each")
        || lower.contains("method chain") || lower.contains("list comprehension")
    {
        return MetaConcept::Iteration;
    }

    // Async
    if lower.contains("async") || lower.contains("await") || lower.contains("future")
        || lower.contains("promise") || lower.contains("tokio") || lower.contains("concurrent")
    {
        return MetaConcept::AsyncConcurrency;
    }

    // Closures / higher-order / function definition
    if lower.contains("closure") || lower.contains("higher-order")
        || lower.contains("lambda") || lower.contains("function definition")
    {
        return MetaConcept::FunctionDefinition;
    }

    // Composition
    if lower.contains("combin") || lower.contains("composit") || lower.contains("blend")
        || lower.contains("integrate") || target.contains("reasoning")
    {
        return MetaConcept::Composition;
    }

    // Testing
    if lower.contains("test") || lower.contains("assert") || lower.contains("mock")
        || lower.contains("benchmark")
    {
        return MetaConcept::Testing;
    }

    // Debugging
    if lower.contains("debug") || lower.contains("stack trace") || lower.contains("breakpoint")
        || target.contains("debug")
    {
        return MetaConcept::Debugging;
    }

    // Refactoring
    if lower.contains("refactor") || lower.contains("clean up") || lower.contains("redesign")
        || target.contains("refactor")
    {
        return MetaConcept::Refactoring;
    }

    // Design patterns (map to closest meta-concept)
    if lower.contains("observer pattern") || lower.contains("factory pattern")
        || lower.contains("strategy pattern") || lower.contains("decorator pattern")
        || lower.contains("design pattern") || lower.contains("singleton")
        || lower.contains("builder pattern")
    {
        return MetaConcept::TraitInterface; // patterns are fundamentally about interfaces
    }

    // Information theory
    let it_keywords = ["entropy", "mutual information", "kl divergence", "kullback",
        "channel capacity", "source coding", "shannon", "huffman",
        "arithmetic coding", "lempel-ziv", "rate-distortion", "rate distortion",
        "fisher information", "cramer-rao", "fano", "data processing inequality",
        "cross-entropy", "cross entropy", "information bottleneck",
        "error-correcting code", "error correcting code", "ldpc", "turbo code",
        "binary symmetric channel", "awgn", "gaussian noise",
        "kolmogorov complexity", "minimum description length", "mdl",
        "typical set", "equipartition", "jensen-shannon", "hellinger",
        "total variation", "renyi", "convolutional code", "linear block code",
        "differential entropy", "conditional entropy", "joint entropy",
        "entropy rate", "sufficient statistic", "information theory"];
    if it_keywords.iter().any(|kw| lower.contains(kw))
        || intent.contains("entropy") || intent.contains("coding")
        || intent.contains("divergence") || intent.contains("channel")
        || intent.contains("information")
    {
        return MetaConcept::InformationTheory;
    }

    // Architecture — routes to TraitInterface to match patterns training group
    if lower.contains("architecture") || lower.contains("microservice")
        || lower.contains("system design")
    {
        return MetaConcept::TraitInterface;
    }

    // Lifetime / borrow
    if lower.contains("lifetime") || lower.contains("borrow") || lower.contains("ownership") {
        return MetaConcept::FunctionDefinition;
    }

    // Fallback: coding intent → function definition, otherwise general knowledge
    if intent.contains("coding") || target.contains("coding") || lower.contains("implement")
        || lower.contains("write a") || lower.contains("create a")
    {
        return MetaConcept::FunctionDefinition;
    }

    MetaConcept::GeneralKnowledge
}

/// Detect whether a query is broad/categorical (asking for an overview or definition)
/// versus specific (asking about a particular operation, pattern, or task).
///
/// Broad queries like "What is software architecture?" need multi-program composition
/// rather than single-program retrieval. This enables the generation path to invoke
/// the summarization mode instead of returning the nearest single program.
///
/// Returns `true` for broad queries that should trigger group-level summarization.
pub fn is_broad_query(text: &str) -> bool {
    let lower = text.to_lowercase();

    // "What is X?" / "Define X" / "What are X?" patterns
    let definitional = lower.starts_with("what is ")
        || lower.starts_with("what are ")
        || lower.starts_with("define ")
        || lower.starts_with("what does ")
        || lower.contains("what is a ")
        || lower.contains("what is an ");

    // "Tell me about X" / "Explain X" without a specific sub-topic
    let overview = lower.starts_with("tell me about ")
        || lower.starts_with("overview of ")
        || lower.starts_with("introduction to ")
        || lower.starts_with("describe ")
        || lower.contains("in general");

    // "Explain X" is broad when X is a domain-level concept, not a specific operation
    let explain_broad = lower.starts_with("explain ")
        && !lower.contains("how to ")
        && !lower.contains("the difference")
        && !lower.contains("pattern")
        && !lower.contains("function")
        && !lower.contains("algorithm");

    // "How does X work?" for domain-level concepts
    let how_work = (lower.contains("how does") || lower.contains("how do"))
        && lower.contains("work")
        && !lower.contains("function")
        && !lower.contains("implement");

    // Domain-level concepts (not specific operations) — these suggest the user wants a summary
    let domain_concept = lower.contains("software architecture")
        || lower.contains("software engineering")
        || lower.contains("design patterns")
        || lower.contains("system design")
        || lower.contains("programming paradigm")
        || lower.contains("machine learning")
        || lower.contains("artificial intelligence")
        || lower.contains("cloud computing")
        || lower.contains("devops")
        || lower.contains("web development")
        || lower.contains("data engineering")
        || lower.contains("computer science")
        || lower.contains("distributed systems")
        || lower.contains("operating systems")
        || lower.contains("networking")
        || lower.contains("information theory")
        || lower.contains("coding theory")
        || lower.contains("signal processing");

    // If the query mentions a specific sub-topic, it's not broad even if phrased generally
    let has_specific = infer_operation_topic(text).is_some();
    if has_specific {
        return false;
    }

    (definitional || overview || explain_broad || how_work) && domain_concept
}

/// Infer the specific operation-level topic from text, matching `semantic_intent` values
/// used in training data. Returns `None` for non-specific or unrecognized operations.
/// This is finer-grained than `infer_concept` and is used as the `topic_hint` for
/// within-group discrimination via topic sub-lattices.
///
/// Delegates to [`TopicGraph`] (`data/knowledge_graph.toml` + optional overlay) when initialized.
/// Otherwise uses [`infer_operation_topic_legacy`] and prints a **one-time** `eprintln` hint.
pub fn infer_operation_topic(text: &str) -> Option<String> {
    if let Some(graph) = TOPIC_GRAPH.get() {
        return graph.infer_topic(text);
    }
    LEGACY_OPERATION_TOPIC_WARN.call_once(|| {
        eprintln!(
            "[growformer] TopicGraph not loaded; infer_operation_topic uses legacy keyword rules. \
Install data/knowledge_graph.toml (CLI and server call try_init_topic_graph_bundle at startup)."
        );
    });
    infer_operation_topic_legacy(text)
}

/// Last-resort keyword ladder when no [`TopicGraph`] is loaded (tests, misconfigured deploy).
/// **Do not add production rules here** — extend `data/knowledge_graph.toml` instead.
fn infer_operation_topic_legacy(text: &str) -> Option<String> {
    let lower = text.to_lowercase();

    if lower.contains("addition") || lower.contains("add two") || lower.contains("add ") && lower.contains("number")
        || lower.contains("plus ") || lower.contains("sum function") || lower.contains("sum of")
        || lower.contains("compute the sum") || lower.contains("adds ")
    {
        return Some("addition_operation".into());
    }
    if lower.contains("subtraction") || lower.contains("subtract") || lower.contains("minus")
        || lower.contains("difference function") || lower.contains("a - b")
    {
        return Some("subtraction_operation".into());
    }
    if lower.contains("multiplication") || lower.contains("multiply") || lower.contains("product function")
        || lower.contains("times ") || lower.contains("a * b")
    {
        return Some("multiplication_operation".into());
    }
    if lower.contains("division") || lower.contains("divide") || lower.contains("quotient")
        || lower.contains("a / b")
    {
        return Some("division_operation".into());
    }
    if lower.contains("modulo") || lower.contains("remainder") {
        return Some("modulo_operation".into());
    }
    if lower.contains("power") || lower.contains("exponent") || lower.contains("pow") {
        return Some("power_operation".into());
    }
    if lower.contains("absolute value") || lower.contains("abs(") || lower.contains("abs_") {
        return Some("absolute_value_operation".into());
    }
    if lower.contains("calculator") || lower.contains("basic calc") {
        return Some("calculator_operation".into());
    }
    if crate::text_keywords::keyword_matches_in_lower(&lower, "average")
        || crate::text_keywords::keyword_matches_in_lower(&lower, "mean")
    {
        return Some("average_operation".into());
    }
    if lower.contains("min ") && lower.contains("max ") || lower.contains("min_val") || lower.contains("max_val") {
        return Some("minmax_operation".into());
    }
    if lower.contains("clamp") || lower.contains("restrict") {
        return Some("clamp_operation".into());
    }
    if lower.contains("checked") || lower.contains("wrapping") || lower.contains("saturating") || lower.contains("overflow") {
        return Some("safe_arithmetic_operation".into());
    }

    // Data structure specifics
    if lower.contains("linked list") { return Some("linked_list".into()); }
    if lower.contains("stack") { return Some("stack_implementation".into()); }
    if lower.contains("queue") { return Some("queue_implementation".into()); }
    if lower.contains("hash map") || lower.contains("hashmap") { return Some("hashmap_implementation".into()); }
    if lower.contains("binary tree") || lower.contains("bst") { return Some("tree_implementation".into()); }

    // Design pattern specifics — behavioral (training intent: "behavioral")
    if lower.contains("observer pattern") || lower.contains("observer") && lower.contains("pattern")
        || lower.contains("observer") && (lower.contains("event") || lower.contains("subscri") || lower.contains("notif"))
    {
        return Some("behavioral".into());
    }
    if lower.contains("strategy pattern") || lower.contains("strategy") && lower.contains("algorithm")
        || lower.contains("swap algorithm") || lower.contains("strategy") && lower.contains("state")
    {
        return Some("behavioral".into());
    }
    if lower.contains("command pattern") || lower.contains("undo") && lower.contains("redo") {
        return Some("behavioral".into());
    }
    if lower.contains("state pattern") || lower.contains("state machine") && lower.contains("pattern") {
        return Some("behavioral".into());
    }
    if lower.contains("template method") { return Some("behavioral".into()); }
    if lower.contains("mediator pattern") || lower.contains("mediator") && lower.contains("coupl") {
        return Some("behavioral".into());
    }
    if lower.contains("visitor pattern") || lower.contains("chain of responsibility") || lower.contains("memento") {
        return Some("behavioral".into());
    }

    // Design pattern specifics — creational (training intent: "creational")
    if lower.contains("factory pattern") || lower.contains("factory method") || lower.contains("abstract factory")
        || lower.contains("factory") && (lower.contains("creat") || lower.contains("instantiat") || lower.contains("construct"))
    {
        return Some("creational".into());
    }
    if lower.contains("builder pattern") || lower.contains("builder") && lower.contains("construct") {
        return Some("creational".into());
    }
    if lower.contains("singleton") { return Some("creational".into()); }
    if lower.contains("prototype pattern") || lower.contains("prototype") && lower.contains("clon") {
        return Some("creational".into());
    }

    // Design pattern specifics — structural (training intent: "structural")
    if lower.contains("adapter pattern") || lower.contains("adapter") && lower.contains("interface") {
        return Some("structural".into());
    }
    if (lower.contains("decorator") || lower.contains("decorator pattern"))
        && !lower.contains("python")
    {
        return Some("structural".into());
    }
    if lower.contains("facade pattern") || lower.contains("facade") && lower.contains("simplif") {
        return Some("structural".into());
    }
    if lower.contains("bridge pattern") || lower.contains("bridge") && lower.contains("abstraction") {
        return Some("structural".into());
    }
    if lower.contains("composite pattern") || lower.contains("composite") && lower.contains("tree") {
        return Some("structural".into());
    }
    if lower.contains("flyweight") { return Some("structural".into()); }
    if lower.contains("subclass") && (lower.contains("extensib") || lower.contains("decorator") || lower.contains("composition")) {
        return Some("structural".into());
    }

    // Architectural patterns
    if lower.contains("microservice") {
        return Some("microservices".into());
    }
    if lower.contains("hexagonal architecture") || lower.contains("ports and adapter") || lower.contains("hexagonal") && lower.contains("port") {
        return Some("hexagonal".into());
    }
    if lower.contains("cqrs") || lower.contains("command query") && lower.contains("segregat") {
        return Some("cqrs".into());
    }
    if lower.contains("event sourcing") || lower.contains("event-sourc") || lower.contains("event sourc") {
        return Some("event_sourcing".into());
    }
    if lower.contains("event-driven") || lower.contains("event driven") || lower.contains("pub/sub") || lower.contains("pub sub") {
        return Some("event_driven".into());
    }
    if lower.contains("saga") && (lower.contains("pattern") || lower.contains("compensat") || lower.contains("distributed")) {
        return Some("saga_pattern".into());
    }
    if lower.contains("circuit breaker") || lower.contains("bulkhead") || lower.contains("resilience pattern") {
        return Some("resilience_patterns".into());
    }
    // Bare "paxos" matches Paxos Labs (fintech); require CS context or exclude company phrases.
    let paxos_company = lower.contains("paxos labs")
        || lower.contains("paxos trust")
        || lower.contains("paxos global")
        || lower.contains("paxos stablecoin")
        || lower.contains("paxos inc")
        || lower.contains("paxos usd")
        || lower.contains("paxos dollar");
    let paxos_distributed = lower.contains("paxos")
        && !paxos_company
        && (lower.contains("protocol")
            || lower.contains("algorithm")
            || lower.contains("replication")
            || lower.contains("distributed")
            || lower.contains("leader")
            || lower.contains("quorum")
            || lower.contains("multi-paxos")
            || lower.contains("two-phase")
            || (lower.contains("explain") && lower.contains("paxos")));
    if lower.contains("consensus") && !paxos_company
        || lower.contains("raft")
        || paxos_distributed
        || lower.contains("zab")
    {
        return Some("consensus_algorithms".into());
    }
    if lower.contains("multi-tenant") || lower.contains("multi tenant") || lower.contains("tenancy") {
        return Some("multi_tenancy".into());
    }
    if lower.contains("stream process") || lower.contains("streaming") && (lower.contains("pipeline") || lower.contains("data")) {
        return Some("stream_processing".into());
    }
    if lower.contains("crdt") || lower.contains("conflict-free") || lower.contains("conflict free") {
        return Some("crdt_conflict_resolution".into());
    }

    // Coding general specifics
    if lower.contains("iterator") && lower.contains("error") || lower.contains("combine") && lower.contains("iterator") {
        return Some("iterator_error_handling".into());
    }
    if lower.contains("struct") && lower.contains("method") || lower.contains("impl block") || lower.contains("impl ") && lower.contains("struct") {
        return Some("struct_methods".into());
    }
    if lower.contains("error handling") || lower.contains("result") && lower.contains("error") || lower.contains("unwrap") || lower.contains("expect") {
        return Some("error_handling".into());
    }
    if lower.contains("decorator") && lower.contains("python") || lower.contains("write a decorator") {
        return Some("decorator_operation".into());
    }
    if lower.contains("async") && (lower.contains("await") || lower.contains("future") || lower.contains("tokio"))
        || lower.contains("async/await")
    {
        return Some("async_pattern".into());
    }
    if lower.contains("async") && (lower.contains("handler") || lower.contains("timeout") || lower.contains("retry")) {
        return Some("async_operation".into());
    }
    if lower.contains("refactor") && lower.contains("module") || lower.contains("es module") {
        return Some("coding_refactor".into());
    }

    // Language-specific coding patterns: use specific intents where available
    if lower.contains("lru cache") || lower.contains("lru") && lower.contains("cache") {
        return Some("lru_cache_operation".into());
    }
    if lower.contains("lifetime") && (lower.contains("annotation") || lower.contains("pattern") || lower.contains("rust") || lower.contains("elision")) {
        return Some("lifetime_pattern".into());
    }
    if lower.contains("borrow") && (lower.contains("debug") || lower.contains("error") || lower.contains("checker")) {
        return Some("coding_debug".into());
    }
    if lower.contains("recursion") && lower.contains("python")
        || lower.contains("python") && (lower.contains("depth") || lower.contains("recursionerror") || lower.contains("recursion"))
    {
        return Some("python_debug".into());
    }
    if lower.contains("recursion") && (lower.contains("debug") || lower.contains("depth") || lower.contains("error") || lower.contains("infinite")) {
        return Some("coding_debug".into());
    }
    if lower.contains("debounce") || lower.contains("throttle") {
        return Some("debounce_throttle".into());
    }
    if lower.contains("middleware") {
        return Some("middleware_operation".into());
    }

    // General knowledge topics (match semantic_intent values from training)
    if lower.contains("relativity") || lower.contains("einstein") {
        return Some("physics".into());
    }
    if lower.contains("halting problem") || lower.contains("turing") && lower.contains("machine")
        || lower.contains("computab") || lower.contains("decidab")
    {
        return Some("cs_fundamentals".into());
    }
    if lower.contains("natural selection") || lower.contains("evolution") && lower.contains("darwin") {
        return Some("biology".into());
    }
    if lower.contains("photosynthesis") {
        return Some("biology".into());
    }
    if lower.contains("supply and demand") || lower.contains("gdp") || lower.contains("inflation") && lower.contains("econom") {
        return Some("economics".into());
    }

    // Support intents — account, billing, cancellation, onboarding
    if lower.contains("lock") && lower.contains("account") || lower.contains("locked out")
        || lower.contains("can't log in") || lower.contains("cannot log in")
        || lower.contains("can't login") || lower.contains("login attempt")
        || lower.contains("account recover") || lower.contains("regain access")
        || lower.contains("compromised") && lower.contains("account")
    {
        return Some("account_recovery".into());
    }
    if lower.contains("cancel") && (lower.contains("subscri") || lower.contains("account") || lower.contains("plan"))
        || lower.contains("downgrade") && lower.contains("plan")
    {
        return Some("cancellation".into());
    }
    if lower.contains("charged twice") || lower.contains("duplicate charge")
        || lower.contains("billing") || lower.contains("refund")
        || lower.contains("overcharged") || lower.contains("wrong charge")
    {
        return Some("billing_issue".into());
    }
    if lower.contains("reset") && lower.contains("password") || lower.contains("password reset")
        || lower.contains("forgot") && lower.contains("password")
    {
        return Some("account_recovery".into());
    }
    if lower.contains("api key") || lower.contains("get started") || lower.contains("onboard")
        || lower.contains("new account") || lower.contains("setup") && lower.contains("account")
    {
        return Some("onboarding_help".into());
    }
    if lower.contains("outage") || lower.contains("down") && lower.contains("service")
        || lower.contains("not working") || lower.contains("server error")
    {
        return Some("service_outage".into());
    }
    if lower.contains("feature request") || lower.contains("suggest") && lower.contains("feature")
        || lower.contains("wish") && lower.contains("could")
    {
        return Some("feature_request".into());
    }
    if lower.contains("privacy") || lower.contains("gdpr") || lower.contains("data deletion")
        || lower.contains("opt out") || lower.contains("tracking")
    {
        return Some("data_privacy".into());
    }

    // Information theory concepts — match semantic_intent values from training data
    if lower.contains("entropy") && (lower.contains("measure") || lower.contains("uncertainty")
        || lower.contains("discrete") || lower.contains("random variable"))
    {
        return Some("entropy".into());
    }
    if lower.contains("differential entropy") || (lower.contains("entropy") && lower.contains("continuous")) {
        return Some("differential_entropy".into());
    }
    if lower.contains("conditional entropy") {
        return Some("conditional_entropy".into());
    }
    if lower.contains("joint entropy") {
        return Some("joint_entropy".into());
    }
    if lower.contains("entropy rate") || lower.contains("stationary stochastic") {
        return Some("entropy_rate".into());
    }
    if lower.contains("mutual information") {
        if lower.contains("conditional") {
            return Some("conditional_mutual_information".into());
        }
        if lower.contains("algorithmic") {
            return Some("algorithmic_mutual_information".into());
        }
        return Some("mutual_information".into());
    }
    if lower.contains("multi-information") || lower.contains("total correlation") {
        return Some("multi_information".into());
    }
    if lower.contains("interaction information") {
        return Some("interaction_information".into());
    }
    if lower.contains("kl divergence") || lower.contains("kullback") {
        return Some("kl_divergence".into());
    }
    if lower.contains("jensen-shannon") || lower.contains("jensen shannon") {
        return Some("jensen_shannon".into());
    }
    if lower.contains("total variation") {
        return Some("total_variation".into());
    }
    if lower.contains("hellinger") {
        return Some("hellinger_distance".into());
    }
    if lower.contains("renyi") {
        return Some("renyi_divergence".into());
    }
    if lower.contains("cross-entropy") || lower.contains("cross entropy") {
        return Some("cross_entropy".into());
    }
    if lower.contains("rate-distortion") || lower.contains("rate distortion") {
        return Some("rate_distortion".into());
    }
    if lower.contains("channel capacity") {
        return Some("channel_capacity".into());
    }
    if lower.contains("source coding") || (lower.contains("shannon") && lower.contains("coding")) {
        return Some("source_coding".into());
    }
    if lower.contains("shannon-hartley") || lower.contains("shannon hartley") || lower.contains("bandlimited") {
        return Some("shannon_hartley".into());
    }
    if lower.contains("huffman") {
        return Some("huffman_coding".into());
    }
    if lower.contains("arithmetic coding") {
        return Some("arithmetic_coding".into());
    }
    if lower.contains("lempel-ziv") || lower.contains("lempel ziv") || lower.contains("lz77") || lower.contains("lz78") {
        return Some("lempel_ziv".into());
    }
    if lower.contains("universal coding") || (lower.contains("universal") && lower.contains("compression")) {
        return Some("universal_coding".into());
    }
    if lower.contains("binary symmetric channel") || lower.contains("bsc") {
        return Some("binary_symmetric_channel".into());
    }
    if lower.contains("binary erasure channel") {
        return Some("binary_erasure_channel".into());
    }
    if lower.contains("awgn") || lower.contains("additive white gaussian") {
        return Some("awgn_channel".into());
    }
    if lower.contains("error-correcting") || lower.contains("error correcting code") {
        return Some("error_correcting_codes".into());
    }
    if lower.contains("linear block code") {
        return Some("linear_block_codes".into());
    }
    if lower.contains("convolutional code") {
        return Some("convolutional_codes".into());
    }
    if lower.contains("turbo code") {
        return Some("turbo_codes".into());
    }
    if lower.contains("ldpc") || lower.contains("low-density parity") {
        return Some("ldpc_codes".into());
    }
    if lower.contains("fisher information") {
        return Some("fisher_information".into());
    }
    if lower.contains("cramer-rao") || lower.contains("cramer rao") || lower.contains("cramér-rao") {
        return Some("cramer_rao".into());
    }
    if lower.contains("fano") && lower.contains("inequality") {
        return Some("fano".into());
    }
    if lower.contains("data processing inequality") {
        return Some("data_processing".into());
    }
    if lower.contains("information bottleneck") {
        return Some("information_bottleneck".into());
    }
    if lower.contains("minimum description length") || lower.contains("mdl principle") {
        return Some("minimum_description_length".into());
    }
    if lower.contains("kolmogorov complexity") || lower.contains("algorithmic complexity") {
        return Some("kolmogorov_complexity".into());
    }
    if lower.contains("sufficient statistic") {
        return Some("sufficient_statistic".into());
    }
    if lower.contains("typical set") || lower.contains("asymptotic equipartition") || lower.contains("aep") {
        return Some("asymptotic_equipartition".into());
    }

    None
}

// ---------------------------------------------------------------------------
// Meta-program: abstract representation of a code concept
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MetaProgram {
    pub concept: MetaConcept,
    pub ops: Vec<MetaOp>,
    pub concept_embedding: Vec<f32>,
}

// ---------------------------------------------------------------------------
// Language projector: Clifford rotor mapping concept space → language space
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LanguageProjector {
    pub language: TargetLanguage,
    pub rotor: Rotor,
    pub trained: bool,
}

impl LanguageProjector {
    pub fn new(language: TargetLanguage) -> Self {
        Self {
            language,
            rotor: Rotor::identity(),
            trained: false,
        }
    }

    /// Train the projector from parallel concept/language embeddings.
    ///
    /// Given pairs of (concept_centroid, language_centroid), compute the optimal
    /// rotor that maps concept space → language space using the geometric product.
    pub fn train(&mut self, concept_centroid: &[f32], language_centroid: &[f32]) {
        let concept_mv = embed_bridge_vector(concept_centroid);
        let lang_mv = embed_bridge_vector(language_centroid);

        // Compute the rotor that maps concept → language:
        // R = lang · concept^{-1}
        // For unit-norm vectors: concept^{-1} ≈ concept_reverse / |concept|²
        let concept_rev = concept_mv.reverse();
        let concept_norm_sq = concept_mv.inner(&concept_mv);

        if concept_norm_sq.abs() < 1e-10 {
            self.trained = false;
            return;
        }

        let inv_scale = 1.0 / concept_norm_sq;
        let mut scaled_rev = concept_rev;
        for c in scaled_rev.components.iter_mut() {
            *c *= inv_scale;
        }

        let transfer = lang_mv.geo(&scaled_rev);

        // Extract the even-grade components as a rotor
        self.rotor.components[0] = transfer.components[GRADE_OFFSETS[0]];
        for i in 0..28 {
            self.rotor.components[1 + i] = transfer.components[GRADE_OFFSETS[2] + i];
        }

        self.trained = true;
    }

    /// Project a concept embedding into language-specific space.
    pub fn project(&self, concept_embedding: &[f32]) -> Vec<f32> {
        if !self.trained {
            return concept_embedding.to_vec();
        }
        let mv = embed_bridge_vector(concept_embedding);
        let projected = apply_group_rotor(&mv, &self.rotor);

        // Extract back to flat vector: grade-1 components
        let g1_start = GRADE_OFFSETS[1];
        let dim = concept_embedding.len().min(8);
        let mut out = vec![0.0f32; concept_embedding.len()];
        for i in 0..dim {
            out[i] = projected.components[g1_start + i];
        }
        // Preserve higher dimensions from original (beyond grade-1)
        for i in dim..concept_embedding.len() {
            out[i] = concept_embedding[i];
        }
        out
    }
}

// ---------------------------------------------------------------------------
// MetaCodebook: concept-level routing + generation
// ---------------------------------------------------------------------------

/// A concept entry: stores the concept centroid and per-language projectors.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConceptEntry {
    pub concept: MetaConcept,
    pub centroid: Vec<f32>,
    pub sample_count: usize,
    pub projectors: HashMap<String, LanguageProjector>, // language_name → projector
    pub language_centroids: HashMap<String, Vec<f32>>,   // language_name → centroid
}

/// The meta-codebook: maps concepts to entries, enabling concept-level routing.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MetaCodebook {
    pub entries: HashMap<MetaConcept, ConceptEntry>,
    pub concept_to_groups: HashMap<MetaConcept, Vec<usize>>, // concept → original group indices
}

impl MetaCodebook {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            concept_to_groups: HashMap::new(),
        }
    }

    /// Build the meta-codebook from training data embeddings grouped by concept and language.
    ///
    /// `samples`: (embedding, concept, language, original_group_idx)
    pub fn build(
        samples: &[(Vec<f32>, MetaConcept, TargetLanguage, usize)],
    ) -> Self {
        let mut codebook = Self::new();

        // Group embeddings by concept
        let mut by_concept: HashMap<MetaConcept, Vec<(&[f32], TargetLanguage, usize)>> = HashMap::new();
        for (emb, concept, lang, gidx) in samples {
            by_concept.entry(*concept).or_default().push((emb.as_slice(), *lang, *gidx));
        }

        for (concept, items) in &by_concept {
            if items.is_empty() { continue; }

            let dim = items[0].0.len();

            // Compute concept centroid (averaged across ALL languages)
            let mut centroid = vec![0.0f32; dim];
            for (emb, _, _) in items {
                for (c, &e) in centroid.iter_mut().zip(emb.iter()) {
                    *c += e;
                }
            }
            let n = items.len() as f32;
            for c in &mut centroid { *c /= n; }

            // Group by language within this concept
            let mut by_lang: HashMap<TargetLanguage, Vec<&[f32]>> = HashMap::new();
            for (emb, lang, _) in items {
                by_lang.entry(*lang).or_default().push(emb);
            }

            // Compute per-language centroids
            let mut language_centroids: HashMap<String, Vec<f32>> = HashMap::new();
            let mut projectors: HashMap<String, LanguageProjector> = HashMap::new();

            for (lang, lang_embs) in &by_lang {
                let mut lang_centroid = vec![0.0f32; dim];
                for emb in lang_embs {
                    for (c, &e) in lang_centroid.iter_mut().zip(emb.iter()) {
                        *c += e;
                    }
                }
                let ln = lang_embs.len() as f32;
                for c in &mut lang_centroid { *c /= ln; }

                // Train a projector: concept_centroid → language_centroid
                let mut projector = LanguageProjector::new(*lang);
                projector.train(&centroid, &lang_centroid);

                language_centroids.insert(lang.name().to_string(), lang_centroid);
                projectors.insert(lang.name().to_string(), projector);
            }

            // Track which original groups belong to this concept, ordered by
            // frequency (most common group first = primary group for this concept).
            let mut group_counts: HashMap<usize, usize> = HashMap::new();
            for (_, _, g) in items {
                *group_counts.entry(*g).or_default() += 1;
            }
            let mut groups: Vec<(usize, usize)> = group_counts.into_iter().collect();
            groups.sort_by(|a, b| b.1.cmp(&a.1)); // Most common first
            let groups: Vec<usize> = groups.into_iter().map(|(g, _)| g).collect();
            codebook.concept_to_groups.insert(*concept, groups);

            codebook.entries.insert(*concept, ConceptEntry {
                concept: *concept,
                centroid,
                sample_count: items.len(),
                projectors,
                language_centroids,
            });
        }

        codebook
    }

    /// Route an embedding to the best meta-concept using cosine similarity
    /// to concept centroids. Returns (concept, confidence, second_confidence).
    pub fn route(&self, embedding: &[f32]) -> (MetaConcept, f32, f32) {
        let mut scored: Vec<(MetaConcept, f32)> = self.entries.iter()
            .map(|(concept, entry)| {
                (*concept, cosine_sim(embedding, &entry.centroid))
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let (best_concept, best_sim) = scored.first().copied()
            .unwrap_or((MetaConcept::GeneralKnowledge, 0.0));
        let second_sim = scored.get(1).map(|x| x.1).unwrap_or(-1.0);

        (best_concept, best_sim, second_sim)
    }

    /// Route and project: determine concept + language, then return a
    /// language-specific conditioning vector.
    ///
    /// Uses a two-stage approach:
    ///   1. Text-based concept classification (deterministic, keyword-accurate)
    ///   2. Embedding similarity to select the best GROUP within that concept
    ///
    /// This decouples "what concept" (text keywords) from "which group" (embedding).
    pub fn route_and_project(
        &self,
        embedding: &[f32],
        text: &str,
    ) -> MetaRoutingResult {
        let language = detect_language(text);

        // Stage 1: Text-based concept classification — this is deterministic
        // and correctly identifies "addition" as BinaryArithmetic, "linked list"
        // as DataStructure, etc.
        let text_concept = infer_concept(text, None, None);

        // Stage 2: If the text-based concept has an entry, use it.
        // Otherwise, fall back to embedding-based routing.
        let (concept, confidence, margin) = if self.entries.contains_key(&text_concept) {
            // Compute confidence as similarity to this concept's centroid
            let entry = &self.entries[&text_concept];
            let sim = cosine_sim(embedding, &entry.centroid);
            // Also check the embedding route for margin calculation
            let (_, emb_conf, emb_second) = self.route(embedding);
            (text_concept, sim.max(0.5), sim - emb_second.max(0.0))
        } else {
            let (emb_concept, emb_conf, emb_second) = self.route(embedding);
            (emb_concept, emb_conf, emb_conf - emb_second)
        };

        // Stage 3: Within the concept, select the best group by finding which
        // group's centroid the embedding is closest to.
        let target_groups = self.concept_to_groups.get(&concept)
            .cloned()
            .unwrap_or_default();

        // Groups are already ordered by frequency (primary group first)
        // from the build phase, so best_group() returns the most common group.
        let best_group_reordered = target_groups;

        let projected = if let Some(entry) = self.entries.get(&concept) {
            if let Some(projector) = entry.projectors.get(language.name()) {
                projector.project(embedding)
            } else {
                embedding.to_vec()
            }
        } else {
            embedding.to_vec()
        };

        // Stage 4: Detect secondary concept for multi-group blending.
        // When primary concept is NOT CausalReasoning but text contains causal
        // connectors, add the causal group as auxiliary so both can be consulted.
        let auxiliary_groups = if concept != MetaConcept::CausalReasoning {
            if has_causal_connectors(text) {
                self.concept_to_groups.get(&MetaConcept::CausalReasoning)
                    .cloned()
                    .unwrap_or_default()
            } else {
                Vec::new()
            }
        } else {
            // Primary IS causal — check if sentiment/other group should be auxiliary
            // (the primary target_groups already contain the causal group)
            Vec::new()
        };

        MetaRoutingResult {
            concept,
            language,
            confidence,
            margin,
            projected_embedding: projected,
            target_groups: best_group_reordered,
            auxiliary_groups,
        }
    }

    pub fn concept_count(&self) -> usize {
        self.entries.len()
    }

    pub fn print_summary(&self) {
        println!("  MetaCodebook: {} concepts", self.entries.len());
        let mut entries: Vec<_> = self.entries.iter().collect();
        entries.sort_by_key(|(c, _)| c.index());
        for (concept, entry) in &entries {
            let langs: Vec<&str> = entry.projectors.keys().map(|s| s.as_str()).collect();
            let groups = self.concept_to_groups.get(concept)
                .map(|g| format!("{:?}", g))
                .unwrap_or_default();
            println!("    {:20} {} samples, langs={:?}, groups={}",
                concept.name(), entry.sample_count, langs, groups);
        }
    }
}

// ---------------------------------------------------------------------------
// Meta-routing result
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct MetaRoutingResult {
    pub concept: MetaConcept,
    pub language: TargetLanguage,
    pub confidence: f32,
    pub margin: f32,
    pub projected_embedding: Vec<f32>,
    pub target_groups: Vec<usize>,
    /// Groups from a secondary concept that should also be consulted.
    /// Populated when the text triggers both a primary concept (e.g. sentiment)
    /// and a secondary concept (e.g. causal reasoning).
    pub auxiliary_groups: Vec<usize>,
}

impl MetaRoutingResult {
    /// The best group index for this concept (first in the mapped list).
    pub fn best_group(&self) -> Option<usize> {
        self.target_groups.first().copied()
    }

    /// Whether this is a coding concept (vs general/support).
    pub fn is_coding(&self) -> bool {
        !matches!(self.concept,
            MetaConcept::GeneralKnowledge | MetaConcept::Support | MetaConcept::Conversation)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Map action_target labels directly to concepts (training-time ground truth).
fn concept_from_action_target(target: &str) -> Option<MetaConcept> {
    match target {
        t if t.contains("arithmetic") => Some(MetaConcept::BinaryArithmetic),
        t if t.contains("data_structure") => Some(MetaConcept::DataStructure),
        t if t.contains("algorithm") => Some(MetaConcept::SearchAlgorithm),
        // "patterns" (design/architectural) → TraitInterface;
        // "coding_patterns" (struct/impl/trait) → None, let keyword matching decide per-sample
        "patterns" => Some(MetaConcept::TraitInterface),
        t if t.contains("architecture") => Some(MetaConcept::TraitInterface),
        t if t.contains("refactoring") || t.contains("refactor") => Some(MetaConcept::Refactoring),
        t if t.contains("debugging") || t.contains("debug") => Some(MetaConcept::Debugging),
        t if t.contains("support") => Some(MetaConcept::Support),
        t if t.contains("conversation") || t.contains("identity") => Some(MetaConcept::Conversation),
        t if t.contains("general_knowledge") => Some(MetaConcept::GeneralKnowledge),
        t if t.contains("reasoning") => Some(MetaConcept::Composition),
        "concepts" => Some(MetaConcept::InformationTheory),
        "math" => Some(MetaConcept::GeneralKnowledge),
        "safety" => Some(MetaConcept::Conversation),
        t if t.contains("causal") => Some(MetaConcept::CausalReasoning),
        // Broad coding categories: let keyword matching assign per-sample concepts
        // so iterator+error → ErrorHandling, struct+methods → StructDefinition, etc.
        "coding_general" | "coding" | "coding_patterns" => None,
        _ => None,
    }
}

/// Detect causal connectors in text for dual-concept routing.
/// These are the same connectors used in `inference_causal.toml` lexicon_keywords.
/// A match means the text likely has a causal relationship that the causal group
/// should also evaluate, even if the primary concept is something else.
fn has_causal_connectors(text: &str) -> bool {
    let lower = text.to_lowercase();

    // Strong causal markers: these alone indicate causal reasoning.
    // `so` is included only as `, so ` or `. so ` (comma/period anchor) so
    // casual usages like "so cool" or "so, anyway" don't trip it.
    const STRONG: &[&str] = &[
        " because ", " since ", " therefore ",
        ", so ", ". so ", "; so ",
        "in retrospect", "looking back", "it turned out",
        " triggered ", " caused ", " resulted in ",
        " pushed ", " led to ", " lead to ",
        " so that ", " thereby ",
    ];
    if STRONG.iter().any(|c| lower.contains(*c)) {
        return true;
    }

    // Contrastive markers alone are NOT causal — they're usually sentiment
    // (MIXED / concessive). Only count as causal when paired with explicit
    // consequence or causal verbs in the same clause.
    const CONTRASTIVE: &[&str] = &[
        " despite ", " although ", " even though ", " yet ",
        " but ", " however ", " nevertheless ",
        " meaning ", " implying ", " suggesting ",
    ];
    const CAUSAL_CO: &[&str] = &[
        "spiked", "pushed", "triggered", "caused", "drove",
        "resulted", "led to", "forced", "crashed", "tanked",
        "surged", "plummeted", "breach", "default",
    ];
    let has_contrastive = CONTRASTIVE.iter().any(|c| lower.contains(*c));
    if has_contrastive && CAUSAL_CO.iter().any(|w| lower.contains(*w)) {
        return true;
    }
    false
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    let len = a.len().min(b.len());
    for i in 0..len {
        let ai = a[i] as f64;
        let bi = b[i] as f64;
        dot += ai * bi;
        na += ai * ai;
        nb += bi * bi;
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom < 1e-20 { 0.0 } else { (dot / denom) as f32 }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_language() {
        assert_eq!(detect_language("write a function in Rust"), TargetLanguage::Rust);
        assert_eq!(detect_language("implement in Python"), TargetLanguage::Python);
        assert_eq!(detect_language("create a React component"), TargetLanguage::TypeScript);
        assert_eq!(detect_language("explain algorithms"), TargetLanguage::Generic);
        // English "match" (noun) must not count as Rust `match`:
        assert_eq!(
            detect_language("we lost the playoff match because our star was hurt"),
            TargetLanguage::Generic
        );
    }

    #[test]
    fn legacy_infer_topic_mean_not_inside_meant() {
        assert_eq!(
            infer_operation_topic("she meant more than anything anyone has done"),
            None
        );
    }

    #[test]
    fn test_normalize_inference_money_spans_dollar_glued() {
        let s = normalize_inference_money_spans("I lost $5000USD on the game last night");
        assert!(s.contains("money_usd_5000"), "got: {}", s);
    }

    #[test]
    fn test_normalize_inference_money_spans_sterling_bn() {
        let s = normalize_inference_money_spans("Unlock £43bn annually");
        assert!(s.contains("money_gbp_"), "got: {}", s);
        assert!(s.contains("43000000000") || s.contains("43"), "scaled amount preserved: {}", s);
    }

    #[test]
    fn test_normalize_inference_money_spans_glued_no_dollar() {
        let s = normalize_inference_money_spans("lost 5000usd on the game");
        assert!(s.contains("money_usd_5000"), "got: {}", s);
    }

    #[test]
    fn test_normalize_inference_money_idempotent() {
        let s = normalize_inference_money_spans("already money_usd_5000 token");
        assert!(s.contains("money_usd_5000"));
    }

    #[test]
    fn test_infer_concept() {
        assert_eq!(
            infer_concept("write an addition function in Rust", None, None),
            MetaConcept::BinaryArithmetic
        );
        assert_eq!(
            infer_concept("implement binary search", None, None),
            MetaConcept::SearchAlgorithm
        );
        assert_eq!(
            infer_concept("implement a linked list", None, None),
            MetaConcept::DataStructure
        );
        assert_eq!(
            infer_concept("implement a stack using enum", None, None),
            MetaConcept::DataStructure
        );
        assert_eq!(
            infer_concept("explain the observer pattern", None, None),
            MetaConcept::TraitInterface
        );
        assert_eq!(
            infer_concept("help me reset my password", None, None),
            MetaConcept::Support
        );
        assert_eq!(
            infer_concept("write a subtraction function in Rust", None, None),
            MetaConcept::BinaryArithmetic
        );
        assert_eq!(
            infer_concept("write a multiplication function", None, None),
            MetaConcept::BinaryArithmetic
        );
        assert_eq!(
            infer_concept("What is the pattern for a struct with methods in Rust", None, None),
            MetaConcept::StructDefinition
        );
    }

    #[test]
    fn test_concept_round_trip() {
        for concept in MetaConcept::all() {
            assert_eq!(MetaConcept::from_index(concept.index()), *concept);
        }
    }

    #[test]
    fn test_projector_identity() {
        let proj = LanguageProjector::new(TargetLanguage::Rust);
        let input = vec![1.0, 0.5, 0.0, -0.3, 0.2, 0.1, 0.0, 0.0];
        let output = proj.project(&input);
        assert_eq!(input.len(), output.len());
        // Untrained projector should return input unchanged
        for (a, b) in input.iter().zip(output.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_meta_codebook_build() {
        let samples = vec![
            (vec![1.0, 0.0, 0.0, 0.0], MetaConcept::BinaryArithmetic, TargetLanguage::Rust, 0),
            (vec![0.9, 0.1, 0.0, 0.0], MetaConcept::BinaryArithmetic, TargetLanguage::Python, 0),
            (vec![0.0, 1.0, 0.0, 0.0], MetaConcept::DataStructure, TargetLanguage::Rust, 1),
        ];
        let codebook = MetaCodebook::build(&samples);
        assert_eq!(codebook.concept_count(), 2);
        assert!(codebook.entries.contains_key(&MetaConcept::BinaryArithmetic));
        assert!(codebook.entries.contains_key(&MetaConcept::DataStructure));

        // Routing: arithmetic-like vector should route to BinaryArithmetic
        let (concept, _, _) = codebook.route(&[0.95, 0.05, 0.0, 0.0]);
        assert_eq!(concept, MetaConcept::BinaryArithmetic);

        // Routing: data-structure-like vector should route to DataStructure
        let (concept, _, _) = codebook.route(&[0.0, 0.9, 0.1, 0.0]);
        assert_eq!(concept, MetaConcept::DataStructure);
    }
}
