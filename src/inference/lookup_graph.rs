//! Runtime JSON graph lookup (WordNet ego-network rows) when lattice retrieval misses.

use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::sync::{OnceLock, RwLock};

const DEFAULT_MAX_NEIGHBORS: usize = 32;

#[derive(Debug, Deserialize)]
struct GraphFile {
    lex: HashMap<String, Vec<String>>,
    edges: Vec<[String; 3]>,
}

struct LookupGraph {
    lex: HashMap<String, [String; 2]>,
    edges: Vec<[String; 3]>,
    adjacency: HashMap<String, HashSet<String>>,
    by_node: HashMap<String, Vec<[String; 3]>>,
}

static LOOKUP_GRAPH: OnceLock<RwLock<Option<LookupGraph>>> = OnceLock::new();

fn cell() -> &'static RwLock<Option<LookupGraph>> {
    LOOKUP_GRAPH.get_or_init(|| RwLock::new(None))
}

fn normalize_word(s: &str) -> String {
    s.trim().to_ascii_lowercase().replace('_', " ")
}

pub fn load_lookup_graph_from_str(json: &str) -> Result<(), String> {
    let parsed: GraphFile =
        serde_json::from_str(json).map_err(|e| format!("lookup graph JSON: {e}"))?;
    let mut lex = HashMap::new();
    for (word, entry) in parsed.lex {
        if entry.len() >= 2 {
            lex.insert(normalize_word(&word), [entry[0].clone(), entry[1].clone()]);
        }
    }
    let mut edges = Vec::with_capacity(parsed.edges.len());
    for e in parsed.edges {
        if e.len() != 3 {
            continue;
        }
        let a = normalize_word(&e[0]);
        let b = normalize_word(&e[1]);
        if a.is_empty() || b.is_empty() || a == b {
            continue;
        }
        if !lex.contains_key(&a) || !lex.contains_key(&b) {
            continue;
        }
        edges.push([a, b, e[2].clone()]);
    }

    let mut adjacency: HashMap<String, HashSet<String>> = lex
        .keys()
        .map(|w| (w.clone(), HashSet::from([w.clone()])))
        .collect();
    for [a, b, _] in &edges {
        if let Some(set) = adjacency.get_mut(a) {
            set.insert(b.clone());
        }
        if let Some(set) = adjacency.get_mut(b) {
            set.insert(a.clone());
        }
    }
    let mut by_node: HashMap<String, Vec<[String; 3]>> = HashMap::new();
    for edge in &edges {
        by_node
            .entry(edge[0].clone())
            .or_default()
            .push(edge.clone());
        by_node
            .entry(edge[1].clone())
            .or_default()
            .push(edge.clone());
    }

    let graph = LookupGraph {
        lex,
        edges,
        adjacency,
        by_node,
    };
    if let Ok(mut guard) = cell().write() {
        *guard = Some(graph);
    }
    Ok(())
}

pub fn lookup_graph_loaded() -> bool {
    cell()
        .read()
        .ok()
        .and_then(|g| g.as_ref().map(|_| true))
        .unwrap_or(false)
}

pub fn try_lookup(intent_text: &str, subject: &str) -> Option<String> {
    let guard = cell().read().ok()?;
    let graph = guard.as_ref()?;
    let word = normalize_word(subject);
    if word.len() <= 2 {
        return None;
    }
    // Only serve bare lemma prompts (not multi-sentence queries).
    if normalize_word(intent_text) != word
        && !intent_text.trim().eq_ignore_ascii_case(subject.trim())
    {
        return None;
    }
    Some(graph.payload_for(&word, DEFAULT_MAX_NEIGHBORS))
}

impl LookupGraph {
    fn payload_for(&self, word: &str, max_neighbors: usize) -> String {
        if !self.lex.contains_key(word) {
            return format!(r#"{{"center":"{word}","found":false,"lex":{{}},"edges":[]}}"#);
        }

        let priority = |rel: &str| match rel {
            "syn" => 0,
            "ant" => 1,
            "hyp" => 2,
            "part" => 3,
            "rel" => 4,
            _ => 9,
        };

        let vis = self
            .adjacency
            .get(word)
            .cloned()
            .unwrap_or_else(|| HashSet::from([word.to_string()]));
        let mut seen: HashSet<(String, String, String)> = HashSet::new();
        let mut edges: Vec<[String; 3]> = Vec::new();
        for node in &vis {
            for e in self.by_node.get(node).into_iter().flatten() {
                let key = (e[0].clone(), e[1].clone(), e[2].clone());
                if seen.contains(&key) || !vis.contains(&e[0]) || !vis.contains(&e[1]) {
                    continue;
                }
                seen.insert(key);
                edges.push(e.clone());
            }
        }
        edges.sort_by(|a, b| {
            priority(&a[2])
                .cmp(&priority(&b[2]))
                .then_with(|| a[0].cmp(&b[0]))
                .then_with(|| a[1].cmp(&b[1]))
        });
        if edges.len() > max_neighbors {
            let mut kept = HashSet::new();
            let mut trimmed = Vec::new();
            for e in edges {
                trimmed.push(e.clone());
                kept.insert(e[0].clone());
                kept.insert(e[1].clone());
                if trimmed.len() >= max_neighbors {
                    break;
                }
            }
            edges = trimmed;
            let mut lex = HashMap::new();
            for w in kept {
                if let Some(entry) = self.lex.get(&w) {
                    lex.insert(w, vec![entry[0].clone(), entry[1].clone()]);
                }
            }
            let [pos, gloss] = &self.lex[word];
            return serde_json::json!({
                "center": word,
                "found": true,
                "pos": pos,
                "definition": gloss,
                "lex": lex,
                "edges": edges,
            })
            .to_string();
        }

        let mut lex = HashMap::new();
        for w in &vis {
            if let Some(entry) = self.lex.get(w) {
                lex.insert(w.clone(), vec![entry[0].clone(), entry[1].clone()]);
            }
        }
        let [pos, gloss] = &self.lex[word];
        serde_json::json!({
            "center": word,
            "found": true,
            "pos": pos,
            "definition": gloss,
            "lex": lex,
            "edges": edges,
        })
        .to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_and_looks_up_center() {
        let json = r#"{"lex":{"cheerful":["adj","happy"],"happy":["adj","pleased"]},"edges":[["cheerful","happy","syn"]]}"#;
        load_lookup_graph_from_str(json).unwrap();
        let out = try_lookup("cheerful", "cheerful").unwrap();
        assert!(out.contains(r#""center":"cheerful""#));
        assert!(out.contains(r#""found":true"#));
    }
}
