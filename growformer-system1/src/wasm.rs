//! WASM bindings (feature `wasm`), following the same pattern as the
//! Growformer brain runtime: a thread-local decider driven by JSON strings.
//!
//! ```js
//! import init, * as s1 from "./growformer_system1.js";
//! await init();
//! s1.system1_load_brain(new Uint8Array(await (await fetch("brain.s1.json.gz")).arrayBuffer()));
//! const out = JSON.parse(s1.system1_decide(JSON.stringify({ state: "refund please" })));
//! ```

use std::cell::RefCell;

use wasm_bindgen::prelude::*;

use crate::brain::System1Brain;
use crate::decide::Decider;

thread_local! {
    static DECIDER: RefCell<Option<Decider>> = const { RefCell::new(None) };
}

fn err(e: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&e.to_string())
}

/// Load a System 1 brain (`*.s1.json` or `*.s1.json.gz` bytes). Replaces any loaded brain.
#[wasm_bindgen]
pub fn system1_load_brain(bytes: &[u8]) -> Result<(), JsValue> {
    let brain = System1Brain::from_bytes(bytes).map_err(err)?;
    DECIDER.with(|c| *c.borrow_mut() = Some(Decider::new(brain)));
    Ok(())
}

#[wasm_bindgen]
pub fn system1_ready() -> bool {
    DECIDER.with(|c| c.borrow().is_some())
}

/// The loaded brain's question schema as JSON.
#[wasm_bindgen]
pub fn system1_questions() -> Result<String, JsValue> {
    DECIDER.with(|c| {
        let b = c.borrow();
        let d = b.as_ref().ok_or_else(|| err("no System 1 brain loaded"))?;
        serde_json::to_string(&d.brain.questions).map_err(err)
    })
}

/// Answer a request (`{"state": …, "questions": […]}`) and return the response JSON.
#[wasm_bindgen]
pub fn system1_decide(request_json: &str) -> Result<String, JsValue> {
    DECIDER.with(|c| {
        let mut b = c.borrow_mut();
        let d = b.as_mut().ok_or_else(|| err("no System 1 brain loaded"))?;
        d.decide_json(request_json).map_err(err)
    })
}

/// Attach an open model from GGUF + `tokenizer.json` bytes. Uses the brain's
/// `lm` settings (architecture, template); pass `lm_config_json` to override.
#[cfg(feature = "candle")]
#[wasm_bindgen]
pub fn system1_load_lm(
    gguf: &[u8],
    tokenizer_json: &[u8],
    lm_config_json: Option<String>,
) -> Result<(), JsValue> {
    DECIDER.with(|c| {
        let mut b = c.borrow_mut();
        let d = b
            .as_mut()
            .ok_or_else(|| err("load a brain before the open model"))?;
        let cfg = match lm_config_json {
            Some(j) => serde_json::from_str(&j).map_err(err)?,
            None => d
                .brain
                .lm
                .clone()
                .ok_or_else(|| err("brain has no `lm` settings; pass lm_config_json"))?,
        };
        let s = crate::lm::CandleScorer::from_bytes(cfg, gguf, tokenizer_json).map_err(err)?;
        d.set_scorer(Box::new(s));
        Ok(())
    })
}
