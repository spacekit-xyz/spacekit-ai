//! Tool use — schema, registry, and matching for external tool invocation.
//!
//! The tool system identifies when a prompt requires an external action (run code,
//! search the web, read a file, compute a value) and produces a structured ToolCallInfo
//! that the caller can execute. Execution is external — the Growformer substrate
//! identifies the tool and extracts arguments; the runtime host performs the action.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Schema types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolParam {
    pub name: String,
    pub param_type: ParamType,
    pub description: String,
    pub required: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ParamType {
    String,
    Number,
    Boolean,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSchema {
    pub name: String,
    pub description: String,
    pub parameters: Vec<ToolParam>,
    /// Trigger phrases that indicate this tool should be invoked.
    pub triggers: Vec<String>,
}

// ---------------------------------------------------------------------------
// Call and result types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallInfo {
    pub tool_name: String,
    pub arguments: HashMap<String, String>,
    /// The raw text that triggered the tool call.
    pub source_text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub tool_name: String,
    pub success: bool,
    pub output: String,
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ToolRegistry {
    tools: Vec<ToolSchema>,
}

/// "How much … lost … fraud/scam" headlines carry digits (years, dollars) but are not math tasks.
fn fraud_loss_how_much_headline_not_math(lower: &str) -> bool {
    let how = lower.contains("how much");
    let fraudy = lower.contains("fraud")
        || lower.contains("scam")
        || lower.contains("victims")
        || (lower.contains("lost") && lower.contains("crypto"));
    let explicit_math = lower.contains("calculate")
        || lower.contains("compute ")
        || lower.contains("evaluate ")
        || lower.contains(" what is ")
        || lower.contains("what's ");
    how && fraudy && !explicit_math
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self { tools: Vec::new() }
    }

    pub fn register(&mut self, schema: ToolSchema) {
        if !self.tools.iter().any(|t| t.name == schema.name) {
            self.tools.push(schema);
        }
    }

    pub fn tools(&self) -> &[ToolSchema] {
        &self.tools
    }

    pub fn get(&self, name: &str) -> Option<&ToolSchema> {
        self.tools.iter().find(|t| t.name == name)
    }

    /// Match a prompt against registered tool triggers.
    /// Returns the best-matching tool and extracted arguments, or None.
    /// Calculator requires a numeric signal (digit or arithmetic operator) to avoid
    /// false positives on phrases like "what is the factory method pattern".
    pub fn match_tool(&self, text: &str) -> Option<ToolCallInfo> {
        let lower = text.to_ascii_lowercase();
        // Require actual arithmetic context: digits, or operators adjacent to spaces/digits.
        // Reject '/' inside words like "async/await", "tcp/ip", "read/write".
        let has_digit = lower.chars().any(|c| c.is_ascii_digit());
        let has_arith_op = lower.contains(" * ")
            || lower.contains(" + ")
            || lower.contains(" / ")
            || lower.contains(" minus ")
            || lower.contains(" plus ")
            || lower.contains(" times ");
        let has_numeric_signal = has_digit || has_arith_op;
        let mut best: Option<(usize, &ToolSchema)> = None;

        for schema in &self.tools {
            if schema.name == "calculator" && fraud_loss_how_much_headline_not_math(&lower) {
                continue;
            }
            if schema.name == "calculator" && !has_numeric_signal {
                continue;
            }
            let hits = schema
                .triggers
                .iter()
                .filter(|t| lower.contains(t.as_str()))
                .count();
            if hits > 0 {
                if best
                    .as_ref()
                    .map_or(true, |(prev_hits, _)| hits > *prev_hits)
                {
                    best = Some((hits, schema));
                }
            }
        }

        let (_, schema) = best?;
        let args = extract_arguments(schema, &lower);
        Some(ToolCallInfo {
            tool_name: schema.name.clone(),
            arguments: args,
            source_text: text.to_string(),
        })
    }

    /// Returns true if any registered tool matches the text.
    pub fn has_match(&self, text: &str) -> bool {
        let lower = text.to_ascii_lowercase();
        self.tools
            .iter()
            .any(|s| s.triggers.iter().any(|t| lower.contains(t.as_str())))
    }

    /// Create a registry pre-loaded with built-in tool schemas.
    pub fn with_builtins() -> Self {
        let mut reg = Self::new();
        reg.register(builtin_calculator());
        reg.register(builtin_web_search());
        reg.register(builtin_code_runner());
        reg.register(builtin_file_reader());
        reg
    }
}

// ---------------------------------------------------------------------------
// Argument extraction — keyword/pattern based
// ---------------------------------------------------------------------------

fn extract_arguments(schema: &ToolSchema, text: &str) -> HashMap<String, String> {
    let mut args = HashMap::new();
    match schema.name.as_str() {
        "calculator" => {
            if let Some(expr) = extract_math_expression(text) {
                args.insert("expression".to_string(), expr);
            }
        }
        "web_search" => {
            if let Some(query) = extract_after_keyword(
                text,
                &["search for", "look up", "find information about", "search"],
            ) {
                args.insert("query".to_string(), query);
            }
        }
        "code_runner" => {
            if let Some(code) = extract_code_block(text) {
                args.insert("code".to_string(), code);
            }
            if let Some(lang) = extract_language(text) {
                args.insert("language".to_string(), lang);
            }
        }
        "file_reader" => {
            if let Some(path) = extract_file_path(text) {
                args.insert("path".to_string(), path);
            }
        }
        _ => {
            for param in &schema.parameters {
                if param.required {
                    if let Some(val) = extract_after_keyword(text, &[&param.name]) {
                        args.insert(param.name.clone(), val);
                    }
                }
            }
        }
    }
    args
}

fn extract_math_expression(text: &str) -> Option<String> {
    let prefixes = ["calculate", "compute", "evaluate", "what is", "what's"];
    for prefix in prefixes {
        if let Some(pos) = text.find(prefix) {
            let rest = text[pos + prefix.len()..].trim();
            let expr: String = rest
                .chars()
                .take_while(|c| c.is_ascii_digit() || " +-*/.()%^".contains(*c))
                .collect();
            let expr = expr.trim().to_string();
            if !expr.is_empty() {
                return Some(expr);
            }
        }
    }
    None
}

fn extract_after_keyword(text: &str, keywords: &[&str]) -> Option<String> {
    for kw in keywords {
        if let Some(pos) = text.find(kw) {
            let rest = text[pos + kw.len()..].trim();
            if !rest.is_empty() {
                return Some(
                    rest.trim_matches(|c: char| c == '"' || c == '\'')
                        .to_string(),
                );
            }
        }
    }
    None
}

fn extract_code_block(text: &str) -> Option<String> {
    if let Some(start) = text.find("```") {
        let after = &text[start + 3..];
        let content_start = after.find('\n').map(|i| i + 1).unwrap_or(0);
        if let Some(end) = after[content_start..].find("```") {
            return Some(after[content_start..content_start + end].trim().to_string());
        }
    }
    extract_after_keyword(text, &["run ", "execute ", "eval "])
}

fn extract_language(text: &str) -> Option<String> {
    let langs = [
        "python",
        "rust",
        "javascript",
        "typescript",
        "ruby",
        "go",
        "java",
        "c++",
        "bash",
        "shell",
    ];
    for lang in langs {
        if text.contains(lang) {
            return Some(lang.to_string());
        }
    }
    None
}

fn extract_file_path(text: &str) -> Option<String> {
    let tokens: Vec<&str> = text.split_whitespace().collect();
    for token in &tokens {
        if token.contains('/') || token.contains('.') && !token.starts_with("http") {
            let clean = token.trim_matches(|c: char| c == '"' || c == '\'' || c == ',' || c == ';');
            if clean.contains('/') || clean.contains('.') {
                return Some(clean.to_string());
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Built-in tool schemas
// ---------------------------------------------------------------------------

fn builtin_calculator() -> ToolSchema {
    ToolSchema {
        name: "calculator".to_string(),
        description: "Evaluate arithmetic expressions".to_string(),
        parameters: vec![ToolParam {
            name: "expression".to_string(),
            param_type: ParamType::String,
            description: "The mathematical expression to evaluate".to_string(),
            required: true,
        }],
        triggers: vec![
            "calculate".to_string(),
            "compute".to_string(),
            "what is".to_string(),
            "how much".to_string(),
            "evaluate".to_string(),
            "multiply".to_string(),
            "divide".to_string(),
            "add".to_string(),
            "subtract".to_string(),
            "sum of".to_string(),
            "product of".to_string(),
        ],
    }
}

fn builtin_web_search() -> ToolSchema {
    ToolSchema {
        name: "web_search".to_string(),
        description: "Search the web for information".to_string(),
        parameters: vec![ToolParam {
            name: "query".to_string(),
            param_type: ParamType::String,
            description: "The search query".to_string(),
            required: true,
        }],
        triggers: vec![
            "search for".to_string(),
            "look up".to_string(),
            "find information".to_string(),
        ],
    }
}

fn builtin_code_runner() -> ToolSchema {
    ToolSchema {
        name: "code_runner".to_string(),
        description: "Execute a code snippet".to_string(),
        parameters: vec![
            ToolParam {
                name: "code".to_string(),
                param_type: ParamType::String,
                description: "The code to execute".to_string(),
                required: true,
            },
            ToolParam {
                name: "language".to_string(),
                param_type: ParamType::String,
                description: "Programming language".to_string(),
                required: false,
            },
        ],
        triggers: vec![
            "run this".to_string(),
            "execute this".to_string(),
            "eval this".to_string(),
            "run the code".to_string(),
            "execute the code".to_string(),
        ],
    }
}

fn builtin_file_reader() -> ToolSchema {
    ToolSchema {
        name: "file_reader".to_string(),
        description: "Read the contents of a file".to_string(),
        parameters: vec![ToolParam {
            name: "path".to_string(),
            param_type: ParamType::String,
            description: "File path to read".to_string(),
            required: true,
        }],
        triggers: vec![
            "read file".to_string(),
            "show file".to_string(),
            "cat ".to_string(),
            "read the file".to_string(),
            "show me the file".to_string(),
        ],
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_register_and_get() {
        let mut reg = ToolRegistry::new();
        assert!(reg.get("calculator").is_none());
        reg.register(builtin_calculator());
        assert!(reg.get("calculator").is_some());
        assert_eq!(reg.tools().len(), 1);
        reg.register(builtin_calculator());
        assert_eq!(reg.tools().len(), 1, "duplicate should be ignored");
    }

    #[test]
    fn test_match_calculator() {
        let reg = ToolRegistry::with_builtins();
        let call = reg.match_tool("calculate 347 * 892").unwrap();
        assert_eq!(call.tool_name, "calculator");
        assert_eq!(call.arguments.get("expression").unwrap(), "347 * 892");
    }

    #[test]
    fn test_match_web_search() {
        let reg = ToolRegistry::with_builtins();
        let call = reg.match_tool("search for rust async patterns").unwrap();
        assert_eq!(call.tool_name, "web_search");
        assert_eq!(call.arguments.get("query").unwrap(), "rust async patterns");
    }

    #[test]
    fn test_match_code_runner() {
        let reg = ToolRegistry::with_builtins();
        let call = reg
            .match_tool("run this python code: print('hello')")
            .unwrap();
        assert_eq!(call.tool_name, "code_runner");
        assert!(call
            .arguments
            .get("language")
            .map_or(false, |l| l == "python"));
    }

    #[test]
    fn test_match_file_reader() {
        let reg = ToolRegistry::with_builtins();
        let call = reg.match_tool("read file src/main.rs").unwrap();
        assert_eq!(call.tool_name, "file_reader");
        assert_eq!(call.arguments.get("path").unwrap(), "src/main.rs");
    }

    #[test]
    fn test_no_match() {
        let reg = ToolRegistry::with_builtins();
        assert!(reg.match_tool("explain the observer pattern").is_none());
    }

    #[test]
    fn calculator_skips_fraud_loss_how_much_headline() {
        let reg = ToolRegistry::with_builtins();
        let s = "Here's how much Michiganders lost in crypto fraud in 2025";
        assert!(reg.match_tool(s).is_none());
    }

    #[test]
    fn test_has_match() {
        let reg = ToolRegistry::with_builtins();
        assert!(reg.has_match("calculate 2 + 2"));
        assert!(!reg.has_match("explain recursion"));
    }

    #[test]
    fn test_builtins_registered() {
        let reg = ToolRegistry::with_builtins();
        assert_eq!(reg.tools().len(), 4);
        assert!(reg.get("calculator").is_some());
        assert!(reg.get("web_search").is_some());
        assert!(reg.get("code_runner").is_some());
        assert!(reg.get("file_reader").is_some());
    }

    #[test]
    fn test_extract_math_expression() {
        assert_eq!(
            extract_math_expression("calculate 2 + 3"),
            Some("2 + 3".to_string())
        );
        assert_eq!(
            extract_math_expression("what is 100 / 5"),
            Some("100 / 5".to_string())
        );
        assert_eq!(
            extract_math_expression("compute 3.14 * 2"),
            Some("3.14 * 2".to_string())
        );
        assert_eq!(extract_math_expression("explain recursion"), None);
    }

    #[test]
    fn test_extract_file_path() {
        assert_eq!(
            extract_file_path("read file src/main.rs"),
            Some("src/main.rs".to_string())
        );
        assert_eq!(
            extract_file_path("show me /tmp/data.txt please"),
            Some("/tmp/data.txt".to_string())
        );
    }

    #[test]
    fn test_tool_call_info_serialization() {
        let call = ToolCallInfo {
            tool_name: "calculator".to_string(),
            arguments: HashMap::from([("expression".to_string(), "2+2".to_string())]),
            source_text: "calculate 2+2".to_string(),
        };
        let json = serde_json::to_string(&call).unwrap();
        let back: ToolCallInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(back.tool_name, "calculator");
        assert_eq!(back.arguments["expression"], "2+2");
    }

    #[test]
    fn test_tool_result_serialization() {
        let result = ToolResult {
            tool_name: "calculator".to_string(),
            success: true,
            output: "4".to_string(),
        };
        let json = serde_json::to_string(&result).unwrap();
        let back: ToolResult = serde_json::from_str(&json).unwrap();
        assert!(back.success);
        assert_eq!(back.output, "4");
    }
}
