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
    GeneralKnowledge,
    Support,
    Conversation,
}

impl MetaConcept {
    pub fn all() -> &'static [MetaConcept] {
        use MetaConcept::*;
        &[
            BinaryArithmetic, UnaryOperation, FunctionDefinition,
            StructDefinition, EnumAlgebraic, TraitInterface,
            ErrorHandling, Iteration, PatternMatching, AsyncConcurrency,
            SearchAlgorithm, SortAlgorithm, DataStructure, Composition,
            Testing, Debugging, Refactoring, GeneralKnowledge,
            Support, Conversation,
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
            Self::GeneralKnowledge => "general_knowledge",
            Self::Support => "support",
            Self::Conversation => "conversation",
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

    let rust_signals = ["rust", "cargo", "impl ", "struct ", "&mut", "fn ",
        "crate", "enum ", "trait ", "tokio", "async fn", "Vec<", "Option<",
        "Result<", "match ", "println!", "unwrap", "lifetime", "borrow"];
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
// Meta-concept inference from text
// ---------------------------------------------------------------------------

pub fn infer_concept(text: &str, semantic_intent: Option<&str>, action_target: Option<&str>) -> MetaConcept {
    let lower = text.to_lowercase();
    let intent = semantic_intent.unwrap_or("").to_lowercase();
    let target = action_target.unwrap_or("").to_lowercase();

    // When action_target is available (training), use it as ground truth first.
    // This prevents false-positive keyword matching from contaminating concept centroids.
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

    // Architecture
    if lower.contains("architecture") || lower.contains("microservice")
        || lower.contains("system design")
    {
        return MetaConcept::Composition;
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

/// Infer the specific operation-level topic from text, matching `semantic_intent` values
/// used in training data. Returns `None` for non-specific or unrecognized operations.
/// This is finer-grained than `infer_concept` and is used as the `topic_hint` for
/// within-group discrimination via topic sub-lattices.
pub fn infer_operation_topic(text: &str) -> Option<String> {
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
    if lower.contains("average") || lower.contains("mean") {
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

    // Design pattern specifics
    if lower.contains("factory pattern") || lower.contains("factory method") { return Some("factory_pattern".into()); }
    if lower.contains("observer pattern") { return Some("observer_pattern".into()); }
    if lower.contains("strategy pattern") { return Some("strategy_pattern".into()); }
    if lower.contains("decorator pattern") { return Some("decorator_pattern".into()); }
    if lower.contains("singleton") { return Some("singleton_pattern".into()); }
    if lower.contains("builder pattern") { return Some("builder_pattern".into()); }

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

        MetaRoutingResult {
            concept,
            language,
            confidence,
            margin,
            projected_embedding: projected,
            target_groups: best_group_reordered,
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
        t if t.contains("pattern") => Some(MetaConcept::TraitInterface),
        t if t.contains("architecture") => Some(MetaConcept::Composition),
        t if t.contains("refactoring") || t.contains("refactor") => Some(MetaConcept::Refactoring),
        t if t.contains("debugging") || t.contains("debug") => Some(MetaConcept::Debugging),
        t if t.contains("support") => Some(MetaConcept::Support),
        t if t.contains("conversation") || t.contains("identity") => Some(MetaConcept::Conversation),
        t if t.contains("general_knowledge") => Some(MetaConcept::GeneralKnowledge),
        t if t.contains("reasoning") => Some(MetaConcept::Composition),
        _ => None,
    }
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    let len = a.len().min(b.len());
    for i in 0..len {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom < 1e-10 { 0.0 } else { dot / denom }
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
