use std::cell::RefCell;
use wasm_bindgen::prelude::*;

use crate::dimension::LanguageConfig;
use crate::service::{AgentMode, LanguageService};

thread_local! {
    static SVC: RefCell<Option<LanguageService>> = RefCell::new(None);
}

fn with_svc<F, R>(f: F) -> Result<R, JsValue>
where
    F: FnOnce(&mut LanguageService) -> Result<R, String>,
{
    SVC.with(|cell| {
        let mut opt = cell.borrow_mut();
        let svc = opt
            .as_mut()
            .ok_or_else(|| JsValue::from_str("growformer not initialized — call growformer_init() first"))?;
        f(svc).map_err(|e| JsValue::from_str(&e))
    })
}

fn with_svc_ref<F, R>(f: F) -> Result<R, JsValue>
where
    F: FnOnce(&LanguageService) -> Result<R, String>,
{
    SVC.with(|cell| {
        let opt = cell.borrow();
        let svc = opt
            .as_ref()
            .ok_or_else(|| JsValue::from_str("growformer not initialized — call growformer_init() first"))?;
        f(svc).map_err(|e| JsValue::from_str(&e))
    })
}

#[wasm_bindgen]
pub fn growformer_init() -> Result<(), JsValue> {
    #[cfg(feature = "wasm-bindgen")]
    console_error_panic_hook::set_once();

    let config = LanguageConfig::default();
    let svc = LanguageService::new_with_config(config)
        .map_err(|e| JsValue::from_str(&e))?;
    SVC.with(|cell| {
        *cell.borrow_mut() = Some(svc);
    });
    Ok(())
}

#[wasm_bindgen]
pub fn growformer_ready() -> bool {
    SVC.with(|cell| cell.borrow().is_some())
}

#[wasm_bindgen]
pub fn growformer_action(text: &str) -> Result<JsValue, JsValue> {
    with_svc(|svc| {
        let action = svc.action(text)?;
        serde_json::to_string(&action).map_err(|e| e.to_string())
    })
    .map(|json| JsValue::from_str(&json))
}

#[wasm_bindgen]
pub fn growformer_generation(text: &str) -> Result<JsValue, JsValue> {
    with_svc(|svc| {
        let (action, resp) = svc.generation(text)?;
        let out = serde_json::json!({
            "action": action,
            "response": {
                "text": resp.text,
                "traceable": resp.traceable,
                "template_id": resp.template_id,
            }
        });
        serde_json::to_string(&out).map_err(|e| e.to_string())
    })
    .map(|json| JsValue::from_str(&json))
}

#[wasm_bindgen]
pub fn growformer_codegen(text: &str) -> Result<JsValue, JsValue> {
    with_svc(|svc| {
        let (action, code) = svc.codegen(text)?;
        let out = serde_json::json!({
            "action": action,
            "code": code.map(|c| serde_json::json!({
                "language": c.language,
                "kind": c.kind,
                "code": c.code,
            }))
        });
        serde_json::to_string(&out).map_err(|e| e.to_string())
    })
    .map(|json| JsValue::from_str(&json))
}

#[wasm_bindgen]
pub fn growformer_set_mode(mode: &str, confidence: f32, reason: &str) -> Result<(), JsValue> {
    let m = match mode {
        "context_file" | "ContextFile" => AgentMode::ContextFile,
        "micro_brain" | "MicroBrain" => AgentMode::MicroBrain,
        _ => return Err(JsValue::from_str("unknown mode; use ContextFile or MicroBrain")),
    };
    with_svc(|svc| {
        svc.set_mode(m, confidence, reason);
        Ok(())
    })
}

#[wasm_bindgen]
pub fn growformer_active_mode() -> Result<String, JsValue> {
    with_svc_ref(|svc| Ok(format!("{:?}", svc.active_mode())))
}

#[wasm_bindgen]
pub fn growformer_push_context_snippet(snippet: &str) -> Result<(), JsValue> {
    with_svc(|svc| {
        svc.push_context_snippet(snippet.to_string());
        Ok(())
    })
}

#[wasm_bindgen]
pub fn growformer_clear_context_snippets() -> Result<(), JsValue> {
    with_svc(|svc| {
        svc.clear_context_snippets();
        Ok(())
    })
}

#[wasm_bindgen]
pub fn growformer_acceptance_report() -> Result<JsValue, JsValue> {
    with_svc_ref(|svc| {
        let report = svc.acceptance_report();
        serde_json::to_string(&report).map_err(|e| e.to_string())
    })
    .map(|json| JsValue::from_str(&json))
}

#[wasm_bindgen]
pub fn growformer_load_checkpoints(data: &[u8]) -> Result<usize, JsValue> {
    with_svc(|svc| svc.load_gle_students_from_bytes(&[data]))
}

/// Load a full trained brain (DimensionManager) exported via --export-brain.
/// Replaces all neurons, groups, routing vectors, and episodic memory.
#[wasm_bindgen]
pub fn growformer_load_brain(data: &[u8]) -> Result<(), JsValue> {
    with_svc(|svc| svc.load_brain(data))
}

/// Export the current brain state as bytes (for saving/caching).
#[wasm_bindgen]
pub fn growformer_export_brain() -> Result<Vec<u8>, JsValue> {
    with_svc_ref(|svc| svc.export_brain())
}
