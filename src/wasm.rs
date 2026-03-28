//! WASM bindings for the Growformer inference runtime.
//!
//! Delegates to [`Runtime`] for all inference operations. The `Runtime` is
//! stored in a `thread_local!` cell and initialised by `growformer_init` or
//! `growformer_load_brain`.

use std::cell::RefCell;
use wasm_bindgen::prelude::*;

use crate::runtime::Runtime;
use crate::service::AgentMode;

thread_local! {
    static RT: RefCell<Option<Runtime>> = RefCell::new(None);
}

fn with_rt<F, R>(f: F) -> Result<R, JsValue>
where
    F: FnOnce(&mut Runtime) -> Result<R, String>,
{
    RT.with(|cell| {
        let mut opt = cell.borrow_mut();
        let rt = opt
            .as_mut()
            .ok_or_else(|| JsValue::from_str("growformer not initialized — call growformer_init() first"))?;
        f(rt).map_err(|e| JsValue::from_str(&e))
    })
}

fn with_rt_ref<F, R>(f: F) -> Result<R, JsValue>
where
    F: FnOnce(&Runtime) -> Result<R, String>,
{
    RT.with(|cell| {
        let opt = cell.borrow();
        let rt = opt
            .as_ref()
            .ok_or_else(|| JsValue::from_str("growformer not initialized — call growformer_init() first"))?;
        f(rt).map_err(|e| JsValue::from_str(&e))
    })
}

/// Initialise an empty runtime (no brain loaded). Call `growformer_load_brain`
/// afterwards to load a trained brain.
#[wasm_bindgen]
pub fn growformer_init() -> Result<(), JsValue> {
    #[cfg(feature = "wasm-bindgen")]
    console_error_panic_hook::set_once();

    let rt = Runtime::empty().map_err(|e| JsValue::from_str(&e))?;
    RT.with(|cell| {
        *cell.borrow_mut() = Some(rt);
    });
    Ok(())
}

#[wasm_bindgen]
pub fn growformer_ready() -> bool {
    RT.with(|cell| cell.borrow().is_some())
}

/// Load a trained brain (`.bin` bytes). Can be called before or after
/// `growformer_init`; if called without init it bootstraps a new runtime.
#[wasm_bindgen]
pub fn growformer_load_brain(data: &[u8]) -> Result<(), JsValue> {
    RT.with(|cell| {
        let mut opt = cell.borrow_mut();
        match opt.as_mut() {
            Some(rt) => rt.load_brain(data).map_err(|e| JsValue::from_str(&e)),
            None => {
                let rt = Runtime::from_brain_bytes(data).map_err(|e| JsValue::from_str(&e))?;
                *opt = Some(rt);
                Ok(())
            }
        }
    })
}

/// Export current brain state as bytes for caching.
#[wasm_bindgen]
pub fn growformer_export_brain() -> Result<Vec<u8>, JsValue> {
    with_rt_ref(|rt| rt.svc.export_brain())
}

/// Return brain metadata as JSON.
#[wasm_bindgen]
pub fn growformer_brain_info() -> Result<JsValue, JsValue> {
    with_rt_ref(|rt| {
        let info = rt.brain_info();
        serde_json::to_string(&info).map_err(|e| e.to_string())
    })
    .map(|json| JsValue::from_str(&json))
}

/// Route text to an action (intent classification only).
#[wasm_bindgen]
pub fn growformer_action(text: &str) -> Result<JsValue, JsValue> {
    with_rt(|rt| {
        let action = rt.svc.action(text)?;
        serde_json::to_string(&action).map_err(|e| e.to_string())
    })
    .map(|json| JsValue::from_str(&json))
}

/// Single-shot generation (route + generate text).
#[wasm_bindgen]
pub fn growformer_generation(text: &str) -> Result<JsValue, JsValue> {
    with_rt(|rt| {
        let resp = rt.prompt(text)?;
        serde_json::to_string(&resp).map_err(|e| e.to_string())
    })
    .map(|json| JsValue::from_str(&json))
}

/// Conversational generation (multi-turn context + personality).
#[wasm_bindgen]
pub fn growformer_converse(text: &str) -> Result<JsValue, JsValue> {
    with_rt(|rt| {
        let resp = rt.converse(text)?;
        serde_json::to_string(&resp).map_err(|e| e.to_string())
    })
    .map(|json| JsValue::from_str(&json))
}

/// Code generation.
#[wasm_bindgen]
pub fn growformer_codegen(text: &str) -> Result<JsValue, JsValue> {
    with_rt(|rt| {
        let code = rt.codegen(text)?;
        serde_json::to_string(&code).map_err(|e| e.to_string())
    })
    .map(|json| JsValue::from_str(&json))
}

/// Paramecium lattice-only inference.
#[wasm_bindgen]
pub fn growformer_paramecium(text: &str) -> Result<JsValue, JsValue> {
    with_rt(|rt| {
        let resp = rt.paramecium(text)?;
        serde_json::to_string(&resp).map_err(|e| e.to_string())
    })
    .map(|json| JsValue::from_str(&json))
}

/// Reset conversation history.
#[wasm_bindgen]
pub fn growformer_reset_conversation() -> Result<(), JsValue> {
    with_rt(|rt| {
        rt.reset_conversation();
        Ok(())
    })
}

#[wasm_bindgen]
pub fn growformer_set_mode(mode: &str, confidence: f32, reason: &str) -> Result<(), JsValue> {
    let m = match mode {
        "context_file" | "ContextFile" => AgentMode::ContextFile,
        "micro_brain" | "MicroBrain" => AgentMode::MicroBrain,
        "paramecium" | "Paramecium" => AgentMode::Paramecium,
        _ => return Err(JsValue::from_str("unknown mode; use ContextFile, MicroBrain, or Paramecium")),
    };
    with_rt(|rt| {
        rt.set_mode(m, confidence, reason);
        Ok(())
    })
}

#[wasm_bindgen]
pub fn growformer_active_mode() -> Result<String, JsValue> {
    with_rt_ref(|rt| Ok(format!("{:?}", rt.active_mode())))
}

#[wasm_bindgen]
pub fn growformer_push_context_snippet(snippet: &str) -> Result<(), JsValue> {
    with_rt(|rt| {
        rt.svc.push_context_snippet(snippet.to_string());
        Ok(())
    })
}

#[wasm_bindgen]
pub fn growformer_clear_context_snippets() -> Result<(), JsValue> {
    with_rt(|rt| {
        rt.svc.clear_context_snippets();
        Ok(())
    })
}

#[wasm_bindgen]
pub fn growformer_acceptance_report() -> Result<JsValue, JsValue> {
    with_rt_ref(|rt| {
        let report = rt.svc.acceptance_report();
        serde_json::to_string(&report).map_err(|e| e.to_string())
    })
    .map(|json| JsValue::from_str(&json))
}

#[wasm_bindgen]
pub fn growformer_load_checkpoints(data: &[u8]) -> Result<usize, JsValue> {
    with_rt(|rt| rt.svc.load_gle_students_from_bytes(&[data]))
}
