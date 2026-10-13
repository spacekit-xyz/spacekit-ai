//! WASM bindings (feature `wasm`), following the same pattern as the
//! Growformer brain runtime: thread-local state driven by JSON strings.
//!
//! Several brains can be loaded by name; they share one open model (feature
//! `candle`) and one model pass per state.
//!
//! ```js
//! import init, * as s1 from "./growformer_system1.js";
//! await init();
//! s1.system1_load_brain_named("crypto", new Uint8Array(await (await fetch("crypto.s1.json.gz")).arrayBuffer()));
//! s1.system1_load_brain_named("support", supportBytes);
//! const one = JSON.parse(s1.system1_decide(JSON.stringify({ brain: "crypto", state: "BTC dumped" })));
//! const both = JSON.parse(s1.system1_decide(JSON.stringify({ brains: ["crypto", "support"], state: "…" })));
//! ```

use std::cell::RefCell;
use std::sync::Arc;

use wasm_bindgen::prelude::*;

use crate::brain::System1Brain;
use crate::multi::{BrainEntry, ModelPool, Registry};

thread_local! {
    static REG: Registry = Registry::new();
    static POOL: RefCell<Option<Arc<ModelPool>>> = const { RefCell::new(None) };
}

fn err(e: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&e.to_string())
}

fn insert(name: &str, brain: System1Brain) {
    let pool = POOL.with(|p| p.borrow().clone());
    REG.with(|r| {
        r.insert(BrainEntry {
            name: name.to_string(),
            brain: Arc::new(brain),
            pool,
            source: None,
        })
    });
}

/// Load a brain (`*.s1.json` / `*.s1.json.gz` bytes) under `name`,
/// replacing any brain with that name.
#[wasm_bindgen]
pub fn system1_load_brain_named(name: &str, bytes: &[u8]) -> Result<(), JsValue> {
    let brain = System1Brain::from_bytes(bytes).map_err(err)?;
    insert(name, brain);
    Ok(())
}

/// Load a brain as `"default"` (single-brain use).
#[wasm_bindgen]
pub fn system1_load_brain(bytes: &[u8]) -> Result<(), JsValue> {
    system1_load_brain_named("default", bytes)
}

/// Remove a brain; returns whether it was loaded.
#[wasm_bindgen]
pub fn system1_unload(name: &str) -> bool {
    REG.with(|r| r.remove(name))
}

#[wasm_bindgen]
pub fn system1_ready() -> bool {
    REG.with(|r| !r.is_empty())
}

/// Loaded brains as JSON (`[{name, questions, heads, probes, open_model}, …]`).
#[wasm_bindgen]
pub fn system1_brains() -> String {
    REG.with(|r| {
        serde_json::to_string(&r.entries().iter().map(|e| e.info()).collect::<Vec<_>>())
            .unwrap_or_else(|_| "[]".into())
    })
}

/// A brain's question schema as JSON (`name` may be omitted when one brain is loaded).
#[wasm_bindgen]
pub fn system1_questions(name: Option<String>) -> Result<String, JsValue> {
    REG.with(|r| {
        let e = r.resolve(name.as_deref()).map_err(err)?;
        serde_json::to_string(&e.brain.questions).map_err(err)
    })
}

/// Answer `{"brain"?: …, "brains"?: […], "state": …, "questions"?: […]}`
/// and return the response JSON.
#[wasm_bindgen]
pub fn system1_decide(request_json: &str) -> Result<String, JsValue> {
    REG.with(|r| r.decide_json(request_json).map_err(err))
}

/// Attach one open model, shared by every loaded brain (and brains loaded
/// later), from GGUF + `tokenizer.json` bytes. Settings (architecture,
/// template) come from `lm_config_json`, else from the first brain that names
/// an open model.
#[cfg(feature = "candle")]
#[wasm_bindgen]
pub fn system1_load_lm(
    gguf: &[u8],
    tokenizer_json: &[u8],
    lm_config_json: Option<String>,
) -> Result<(), JsValue> {
    let cfg: crate::brain::LmConfig = match lm_config_json {
        Some(j) => serde_json::from_str(&j).map_err(err)?,
        None => REG
            .with(|r| r.entries().iter().find_map(|e| e.brain.lm.clone()))
            .ok_or_else(|| err("no loaded brain names an open model; pass lm_config_json"))?,
    };
    let s = crate::lm::CandleScorer::from_bytes(cfg, gguf, tokenizer_json).map_err(err)?;
    let pool = Arc::new(ModelPool::new(Box::new(s), 1));
    POOL.with(|p| *p.borrow_mut() = Some(pool.clone()));
    REG.with(|r| {
        for e in r.entries() {
            r.insert(BrainEntry {
                name: e.name.clone(),
                brain: e.brain.clone(),
                pool: Some(pool.clone()),
                source: None,
            });
        }
    });
    Ok(())
}
