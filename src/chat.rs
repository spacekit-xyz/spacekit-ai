//! Chat transcript helpers for chatbot builders.
//!
//! Prompt format (stable contract for train + infer):
//! ```text
//! ### System:
//! …
//! ### User:
//! …
//! ### Assistant:
//! …
//! ```
//!
//! Train on this format (`jsonl-to-chat-txt`) so generate/chat stay aligned.
//! For **domain** bots, prefer Path A brain retrieve+label first; use the LM for
//! fluent continuation / chitchat after a domain checkpoint exists.

use crate::bpe::BpeTokenizer;

/// Role markers — keep ASCII and short so small BPE vocabs learn them.
pub const MARK_SYSTEM: &str = "### System:";
pub const MARK_USER: &str = "### User:";
pub const MARK_ASSISTANT: &str = "### Assistant:";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatRole {
    System,
    User,
    Assistant,
}

impl ChatRole {
    pub fn marker(self) -> &'static str {
        match self {
            Self::System => MARK_SYSTEM,
            Self::User => MARK_USER,
            Self::Assistant => MARK_ASSISTANT,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::System,
            content: content.into(),
        }
    }
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::User,
            content: content.into(),
        }
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::Assistant,
            content: content.into(),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct ChatTranscript {
    pub messages: Vec<ChatMessage>,
}

impl ChatTranscript {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_system(system: impl Into<String>) -> Self {
        let mut t = Self::new();
        t.messages.push(ChatMessage::system(system));
        t
    }

    pub fn push(&mut self, msg: ChatMessage) {
        self.messages.push(msg);
    }

    pub fn push_user(&mut self, content: impl Into<String>) {
        self.push(ChatMessage::user(content));
    }

    pub fn push_assistant(&mut self, content: impl Into<String>) {
        self.push(ChatMessage::assistant(content));
    }

    /// Full transcript text (no trailing assistant cue).
    pub fn render(&self) -> String {
        let mut out = String::new();
        for (i, m) in self.messages.iter().enumerate() {
            if i > 0 {
                out.push('\n');
            }
            out.push_str(m.role.marker());
            out.push('\n');
            out.push_str(m.content.trim());
            out.push('\n');
        }
        out
    }

    /// Prompt for the next assistant turn (adds empty `### Assistant:` cue).
    pub fn render_for_completion(&self) -> String {
        let mut out = self.render();
        if !out.ends_with('\n') {
            out.push('\n');
        }
        out.push_str(MARK_ASSISTANT);
        out.push('\n');
        out
    }

    /// Drop oldest turns until BPE token count ≤ `max_tokens`.
    /// Never drops the latest user message; shrinks/drops system first, then older turns.
    pub fn truncate_to_token_budget(&mut self, bpe: &BpeTokenizer, max_tokens: usize) {
        if max_tokens == 0 {
            return;
        }
        for _ in 0..64 {
            let n = bpe.encode(&self.render_for_completion()).len();
            if n <= max_tokens {
                return;
            }
            // 1) Drop or shorten system to free budget for the latest user turn.
            if let Some(m) = self.messages.first_mut() {
                if m.role == ChatRole::System {
                    if m.content.len() > 24 {
                        m.content = "Be brief.".into();
                        continue;
                    }
                    self.messages.remove(0);
                    continue;
                }
            }
            // 2) Drop oldest non-latest message (keep final user turn).
            if self.messages.len() <= 1 {
                return;
            }
            let last_is_user = matches!(self.messages.last().map(|m| m.role), Some(ChatRole::User));
            let remove_at = if last_is_user && self.messages.len() >= 2 {
                self.messages.len() - 2
            } else {
                0
            };
            if remove_at < self.messages.len() {
                self.messages.remove(remove_at);
            } else {
                return;
            }
        }
    }
}

/// If generation drifted into a new role header, return cut index.
pub fn role_marker_cut(text: &str) -> Option<usize> {
    let markers = [MARK_SYSTEM, MARK_USER, MARK_ASSISTANT];
    let mut best: Option<usize> = None;
    for m in markers {
        if let Some(i) = text.find(m) {
            best = Some(best.map_or(i, |b| b.min(i)));
        }
    }
    best
}

/// Default system prompt for small domain chatbots (override per product).
/// Keep short — `max_seq` is often 128.
pub fn default_chatbot_system() -> &'static str {
    "Be brief. Prefer memory facts. If unsure, say so."
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuts_hallucinated_assistant_marker() {
        let s = "NEGATIVE (strong) — sold off.\n### Assistant:\nmore";
        assert_eq!(role_marker_cut(s), Some(s.find(MARK_ASSISTANT).unwrap()));
    }
}
