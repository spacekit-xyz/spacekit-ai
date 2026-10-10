//! Prompt layout for the open-model backend.
//!
//! The **state comes first** (prefix) and each question is a short suffix, so
//! the model processes the shared state once and every question reuses it.
//! The answer is read from the next-token distribution restricted to the
//! answer tokens (`A`/`B`/…, digits, or `Yes`/`No`) — no text is generated.

use serde::{Deserialize, Serialize};

use crate::schema::{Question, QuestionKind};

/// Chat-template preset (or a custom prefix/suffix pair).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TemplatePreset {
    /// Qwen2 / Qwen2.5 and other ChatML models.
    Chatml,
    /// Qwen3 with thinking disabled (empty `<think></think>` block).
    ChatmlNothink,
    /// Llama 3.x instruct.
    Llama3,
    /// Base (non-chat) models.
    Plain,
    /// `prefix` must contain `{state}`; `suffix` may use `{question}`,
    /// `{options}`, `{instruction}`; `{system}` is allowed in either.
    Custom { prefix: String, suffix: String },
}

pub const DEFAULT_SYSTEM: &str =
    "You are a fast, careful classifier. Read the state, then answer the \
question with exactly one of the allowed answers and nothing else.";

/// The pieces of one question's prompt.
#[derive(Clone, Debug, PartialEq)]
pub struct QuestionPrompt {
    /// Text appended after the shared prefix.
    pub suffix: String,
    /// Answer strings, one per outcome (same order as `Question::outcomes`).
    pub answers: Vec<String>,
}

const LETTERS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ";

/// Answer labels for a question's outcomes.
pub fn answer_labels(q: &Question) -> Vec<String> {
    match &q.kind {
        QuestionKind::Choice { options } => (0..options.len())
            .map(|i| (LETTERS[i % 26] as char).to_string())
            .collect(),
        QuestionKind::Score { min, max, .. } => {
            if *min >= 0 && *max <= 9 {
                (*min..=*max).map(|v| v.to_string()).collect()
            } else {
                (0..=(max - min) as usize)
                    .map(|i| (LETTERS[i % 26] as char).to_string())
                    .collect()
            }
        }
        QuestionKind::Noul => vec!["Yes".into(), "No".into()],
    }
}

fn options_block(q: &Question, answers: &[String]) -> (String, String) {
    match &q.kind {
        QuestionKind::Choice { options } => {
            let lines: Vec<String> = options
                .iter()
                .zip(answers)
                .map(|(o, a)| format!("{a}) {o}"))
                .collect();
            (
                format!("Options:\n{}", lines.join("\n")),
                "Answer with the letter of the best option only.".into(),
            )
        }
        QuestionKind::Score { min, max, labels } => {
            let digits = *min >= 0 && *max <= 9;
            let mut lines = Vec::new();
            for (i, v) in (*min..=*max).enumerate() {
                let desc = labels.get(i).map(|l| format!(" = {l}")).unwrap_or_default();
                if digits {
                    if !desc.is_empty() {
                        lines.push(format!("{v}{desc}"));
                    }
                } else {
                    lines.push(format!("{}) {v}{desc}", answers[i]));
                }
            }
            let block = if lines.is_empty() {
                format!("Scale: {min} (lowest) to {max} (highest).")
            } else {
                format!("Scale:\n{}", lines.join("\n"))
            };
            let instr = if digits {
                format!("Answer with a single number from {min} to {max} only.")
            } else {
                "Answer with the letter of the scale point only.".into()
            };
            (block, instr)
        }
        QuestionKind::Noul => (String::new(), "Answer Yes or No only.".into()),
    }
}

fn fill(
    template: &str,
    system: &str,
    state: &str,
    question: &str,
    options: &str,
    instruction: &str,
) -> String {
    template
        .replace("{system}", system)
        .replace("{state}", state)
        .replace("{question}", question)
        .replace("{options}", options)
        .replace("{instruction}", instruction)
}

impl TemplatePreset {
    fn parts(&self) -> (String, String) {
        match self {
            Self::Chatml => (
                "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\nState:\n{state}\n\n".into(),
                "Question: {question}\n{options}\n{instruction}<|im_end|>\n<|im_start|>assistant\n".into(),
            ),
            Self::ChatmlNothink => (
                "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\nState:\n{state}\n\n".into(),
                "Question: {question}\n{options}\n{instruction}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n".into(),
            ),
            Self::Llama3 => (
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nState:\n{state}\n\n".into(),
                "Question: {question}\n{options}\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n".into(),
            ),
            Self::Plain => (
                "{system}\n\nState:\n{state}\n\n".into(),
                "Question: {question}\n{options}\n{instruction}\nAnswer:".into(),
            ),
            Self::Custom { prefix, suffix } => (prefix.clone(), suffix.clone()),
        }
    }

    /// Shared prefix containing the state.
    pub fn prefix(&self, system: &str, state: &str) -> String {
        fill(&self.parts().0, system, state, "", "", "")
    }

    /// Per-question suffix and its answer labels.
    pub fn question(&self, system: &str, q: &Question) -> QuestionPrompt {
        let answers = answer_labels(q);
        let (options, instruction) = options_block(q, &answers);
        let mut question = q.text.trim().to_string();
        if let Some(d) = q.description.as_deref().filter(|d| !d.trim().is_empty()) {
            question = format!("{question}\nGuidance: {}", d.trim());
        }
        let options = if options.is_empty() {
            String::new()
        } else {
            options
        };
        let mut suffix = fill(
            &self.parts().1,
            system,
            "",
            &question,
            &options,
            &instruction,
        );
        // Collapse the blank line left by an empty options block.
        while suffix.contains("\n\n\n") {
            suffix = suffix.replace("\n\n\n", "\n\n");
        }
        if options.is_empty() {
            suffix = suffix.replacen(&format!("{question}\n\n"), &format!("{question}\n"), 1);
        }
        QuestionPrompt { suffix, answers }
    }

    /// Plain-text base models are prompted with `Answer:` and answer with a leading space.
    pub fn answers_have_leading_space(&self) -> bool {
        matches!(self, Self::Plain)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chatml_layout_puts_state_first() {
        let t = TemplatePreset::ChatmlNothink;
        let pre = t.prefix(DEFAULT_SYSTEM, "ticket: refund please");
        assert!(pre.starts_with("<|im_start|>system"));
        assert!(pre.ends_with("State:\nticket: refund please\n\n"));
        let q = Question::choice("route", "Which team?", &["Billing", "Tech Support"]);
        let qp = t.question(DEFAULT_SYSTEM, &q);
        assert_eq!(qp.answers, vec!["A", "B"]);
        assert!(qp.suffix.contains("A) Billing\nB) Tech Support"));
        assert!(qp.suffix.ends_with("<think>\n\n</think>\n\n"));
    }

    #[test]
    fn score_and_noul_answers() {
        let s = Question::score("u", "How urgent?", 0, 4);
        assert_eq!(answer_labels(&s), vec!["0", "1", "2", "3", "4"]);
        let wide = Question::score("w", "Rate", -2, 2);
        assert_eq!(answer_labels(&wide), vec!["A", "B", "C", "D", "E"]);
        let n = Question::noul("n", "Is it spam?");
        let qp = TemplatePreset::Chatml.question(DEFAULT_SYSTEM, &n);
        assert_eq!(qp.answers, vec!["Yes", "No"]);
        assert!(qp.suffix.contains("Is it spam?\nAnswer Yes or No only."));
    }
}
