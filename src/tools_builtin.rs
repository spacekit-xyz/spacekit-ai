//! Native-only built-in tool executors (calculator, file_reader, code_runner, web_search).
//!
//! These require filesystem and subprocess access and are gated behind
//! `#[cfg(not(target_arch = "wasm32"))]` at the module level in `lib.rs`.

use crate::dimension::tool::{ToolCallInfo, ToolResult};

pub fn execute_tool(call: &ToolCallInfo) -> ToolResult {
    match call.tool_name.as_str() {
        "calculator" => {
            let expr = call
                .arguments
                .get("expression")
                .map(|s| s.as_str())
                .unwrap_or("");
            let result = eval_arithmetic(expr);
            ToolResult {
                tool_name: "calculator".into(),
                success: result.is_ok(),
                output: result.unwrap_or_else(|e| e),
            }
        }
        "file_reader" => {
            let path = call
                .arguments
                .get("path")
                .map(|s| s.as_str())
                .unwrap_or("");
            match std::fs::read_to_string(path) {
                Ok(content) => {
                    let preview: String =
                        content.lines().take(50).collect::<Vec<_>>().join("\n");
                    let total = content.lines().count();
                    let output = if total > 50 {
                        format!("{}\n... ({} more lines)", preview, total - 50)
                    } else {
                        preview
                    };
                    ToolResult {
                        tool_name: "file_reader".into(),
                        success: true,
                        output,
                    }
                }
                Err(e) => ToolResult {
                    tool_name: "file_reader".into(),
                    success: false,
                    output: e.to_string(),
                },
            }
        }
        "code_runner" => {
            let code = call
                .arguments
                .get("code")
                .map(|s| s.as_str())
                .unwrap_or("");
            let lang = call
                .arguments
                .get("language")
                .map(|s| s.as_str())
                .unwrap_or("python");
            let (cmd, args) = match lang {
                "python" => ("python3", vec!["-c", code]),
                "bash" | "shell" => ("bash", vec!["-c", code]),
                "ruby" => ("ruby", vec!["-e", code]),
                "node" | "javascript" => ("node", vec!["-e", code]),
                _ => ("python3", vec!["-c", code]),
            };
            match std::process::Command::new(cmd)
                .args(&args)
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped())
                .output()
            {
                Ok(output) => {
                    let stdout = String::from_utf8_lossy(&output.stdout);
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    let text = if stdout.is_empty() {
                        stderr.to_string()
                    } else {
                        stdout.to_string()
                    };
                    let truncated = if text.len() > 2000 {
                        format!("{}...(truncated)", &text[..2000])
                    } else {
                        text
                    };
                    ToolResult {
                        tool_name: "code_runner".into(),
                        success: output.status.success(),
                        output: truncated,
                    }
                }
                Err(e) => ToolResult {
                    tool_name: "code_runner".into(),
                    success: false,
                    output: e.to_string(),
                },
            }
        }
        "web_search" => {
            let query = call
                .arguments
                .get("query")
                .map(|s| s.as_str())
                .unwrap_or("");
            ToolResult {
                tool_name: "web_search".into(),
                success: false,
                output: format!("Web search not yet available. Query: {}", query),
            }
        }
        _ => ToolResult {
            tool_name: call.tool_name.clone(),
            success: false,
            output: format!("Unknown tool: {}", call.tool_name),
        },
    }
}

// ─── Arithmetic evaluator ────────────────────────────────────────────────

pub fn eval_arithmetic(expr: &str) -> Result<String, String> {
    let clean: String = expr
        .chars()
        .filter(|c| c.is_ascii_digit() || " .+-*/()%".contains(*c))
        .collect();
    if clean.is_empty() {
        return Err("empty expression".into());
    }
    let tokens = tokenize_math(&clean)?;
    let result = eval_tokens(&tokens)?;
    Ok(format!("{}", result))
}

#[derive(Debug, Clone)]
enum MathToken {
    Num(f64),
    Op(char),
    LParen,
    RParen,
}

fn tokenize_math(expr: &str) -> Result<Vec<MathToken>, String> {
    let mut tokens = Vec::new();
    let mut chars = expr.chars().peekable();
    while let Some(&ch) = chars.peek() {
        if ch.is_whitespace() {
            chars.next();
            continue;
        }
        if ch.is_ascii_digit() || ch == '.' {
            let mut num = String::new();
            while let Some(&c) = chars.peek() {
                if c.is_ascii_digit() || c == '.' {
                    num.push(c);
                    chars.next();
                } else {
                    break;
                }
            }
            tokens.push(MathToken::Num(
                num.parse::<f64>().map_err(|e| e.to_string())?,
            ));
        } else if "+-*/%".contains(ch) {
            tokens.push(MathToken::Op(ch));
            chars.next();
        } else if ch == '(' {
            tokens.push(MathToken::LParen);
            chars.next();
        } else if ch == ')' {
            tokens.push(MathToken::RParen);
            chars.next();
        } else {
            chars.next();
        }
    }
    Ok(tokens)
}

fn eval_tokens(tokens: &[MathToken]) -> Result<f64, String> {
    let mut pos = 0;
    parse_expr(tokens, &mut pos)
}

fn parse_expr(tokens: &[MathToken], pos: &mut usize) -> Result<f64, String> {
    let mut left = parse_term(tokens, pos)?;
    while *pos < tokens.len() {
        match tokens[*pos] {
            MathToken::Op('+') => {
                *pos += 1;
                left += parse_term(tokens, pos)?;
            }
            MathToken::Op('-') => {
                *pos += 1;
                left -= parse_term(tokens, pos)?;
            }
            _ => break,
        }
    }
    Ok(left)
}

fn parse_term(tokens: &[MathToken], pos: &mut usize) -> Result<f64, String> {
    let mut left = parse_factor(tokens, pos)?;
    while *pos < tokens.len() {
        match tokens[*pos] {
            MathToken::Op('*') => {
                *pos += 1;
                left *= parse_factor(tokens, pos)?;
            }
            MathToken::Op('/') => {
                *pos += 1;
                let r = parse_factor(tokens, pos)?;
                if r == 0.0 {
                    return Err("division by zero".into());
                }
                left /= r;
            }
            MathToken::Op('%') => {
                *pos += 1;
                let r = parse_factor(tokens, pos)?;
                if r == 0.0 {
                    return Err("modulo by zero".into());
                }
                left %= r;
            }
            _ => break,
        }
    }
    Ok(left)
}

fn parse_factor(tokens: &[MathToken], pos: &mut usize) -> Result<f64, String> {
    if *pos >= tokens.len() {
        return Err("unexpected end of expression".into());
    }
    match &tokens[*pos] {
        MathToken::Num(n) => {
            let v = *n;
            *pos += 1;
            Ok(v)
        }
        MathToken::Op('-') => {
            *pos += 1;
            let v = parse_factor(tokens, pos)?;
            Ok(-v)
        }
        MathToken::LParen => {
            *pos += 1;
            let v = parse_expr(tokens, pos)?;
            if *pos < tokens.len() && matches!(tokens[*pos], MathToken::RParen) {
                *pos += 1;
            }
            Ok(v)
        }
        _ => Err(format!("unexpected token: {:?}", tokens[*pos])),
    }
}
