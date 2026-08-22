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
        let rt = opt.as_mut().ok_or_else(|| {
            JsValue::from_str("growformer not initialized — call growformer_init() first")
        })?;
        f(rt).map_err(|e| JsValue::from_str(&e))
    })
}

fn with_rt_ref<F, R>(f: F) -> Result<R, JsValue>
where
    F: FnOnce(&Runtime) -> Result<R, String>,
{
    RT.with(|cell| {
        let opt = cell.borrow();
        let rt = opt.as_ref().ok_or_else(|| {
            JsValue::from_str("growformer not initialized — call growformer_init() first")
        })?;
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
    with_rt(|rt| rt.svc.export_brain())
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
        _ => {
            return Err(JsValue::from_str(
                "unknown mode; use ContextFile, MicroBrain, or Paramecium",
            ))
        }
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

/// Return JSON summary of active inference rule counts (for diagnostics).
/// Gated behind `wasm-debug` feature to avoid leaking rule internals in production.
#[cfg(feature = "wasm-debug")]
#[wasm_bindgen]
pub fn growformer_inference_rules_info() -> Result<JsValue, JsValue> {
    let loaded = crate::inference::inference_toml::inference_toml_loaded();
    let rules = loaded.rules();
    Ok(JsValue::from_str(&rules.rules_summary_json()))
}

/// Diagnostic: dump raw encoder vec + bridge output for a given text.
/// Gated behind `wasm-debug` feature to avoid leaking embedding dimensions and structure.
#[cfg(feature = "wasm-debug")]
#[wasm_bindgen]
pub fn growformer_debug_embedding(text: &str) -> Result<JsValue, JsValue> {
    with_rt(|rt| {
        let dm = rt.svc.active_dm_mut();

        // Step 1: tokenize and dictionary lookup
        let dict = dm.language_runtime.preloaded_dictionary.as_ref();
        let (tok_ids, tok_count, dict_len) = if let Some(d) = dict {
            let ids = d.encode(text);
            (ids.clone(), ids.len(), d.len())
        } else {
            (vec![], 0usize, 0usize)
        };

        // Step 2: manually test ChunkCodec encoding
        let centroid_norm: f64 = if let Some(d) = dict {
            let codec = crate::text_autoencoder::ChunkCodec::new(d.len());
            let seq = codec.encode_text(text, d);
            let c = seq.centroid();
            c.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt()
        } else {
            0.0
        };

        // Step 3: full encode_and_bridge
        let (raw_enc, bridged) = dm.language_runtime.encode_and_bridge(text)?;
        let raw_first8: Vec<f32> = raw_enc.iter().take(8).copied().collect();
        let raw_norm: f64 = raw_enc.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
        let br_vec = &bridged.routed_vector;
        let br_first8: Vec<f32> = br_vec.iter().take(8).copied().collect();
        let br_norm: f64 = br_vec.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
        Ok(format!(
            "{{\"dict_size\":{},\"token_ids\":{:?},\"tok_count\":{},\"centroid_norm\":{:.10},\"raw_dim\":{},\"raw_norm\":{:.10},\"raw_first8\":{:?},\"bridge_dim\":{},\"bridge_confidence\":{:.6},\"bridge_norm\":{:.10},\"bridge_first8\":{:?}}}",
            dict_len, tok_ids, tok_count, centroid_norm,
            raw_enc.len(), raw_norm, raw_first8,
            br_vec.len(), bridged.confidence, br_norm, br_first8
        ))
    })
    .map(|s| JsValue::from_str(&s))
}

/// Diagnostic: check language runtime dictionary and encoder state.
#[cfg(feature = "wasm-debug")]
#[wasm_bindgen]
pub fn growformer_debug_dictionary() -> Result<JsValue, JsValue> {
    with_rt(|rt| {
        let dm = rt.svc.active_dm();
        let lr = &dm.language_runtime;
        let dict_tokens = lr.preloaded_dictionary.as_ref().map(|d| d.tokens.len()).unwrap_or(0);
        let dict_lookup = lr.preloaded_dictionary.as_ref().map(|d| d.lookup_len()).unwrap_or(0);
        let encoder_preset = format!("{:?}", lr.config.encoder);
        let bridge_out = lr.config.bridge_output_dim;
        Ok(format!(
            "{{\"encoder_preset\":\"{}\",\"preloaded_dict_tokens\":{},\"preloaded_dict_lookup\":{},\"bridge_out_dim\":{}}}",
            encoder_preset, dict_tokens, dict_lookup, bridge_out
        ))
    })
    .map(|s| JsValue::from_str(&s))
}

/// Load domain-specific inference TOML (e.g. `inference_fintech.toml` or
/// `inference_crypto.toml`) to replace the embedded core rules. The domain
/// document is merged with the compiled-in core baseline so that any rule
/// lists the domain file leaves empty are filled from core — matching
/// the native CLI merge behavior.
///
/// Call **before** the first inference so the rules are active. Calling again
/// replaces the previous rules (hot-swap is safe on WASM's single thread).
#[wasm_bindgen]
pub fn growformer_load_inference_toml(toml_str: &str) -> Result<(), JsValue> {
    crate::inference::inference_toml::reload_inference_toml_from_str(toml_str)
        .map_err(|e| JsValue::from_str(&e))?;

    with_rt(|rt| {
        rt.apply_loaded_generation_config();
        Ok(())
    })?;
    Ok(())
}

/// Enable stochastic top-k retrieval on all generation environments.
/// Temperature controls sampling sharpness: lower = more deterministic, higher = more varied.
/// Typical values: 0.7 (conservative) to 1.2 (creative). Call after `growformer_load_brain`.
#[wasm_bindgen]
pub fn growformer_enable_stochastic_retrieval(temperature: f32) -> Result<(), JsValue> {
    with_rt(|rt| {
        rt.enable_stochastic_retrieval(temperature);
        Ok(())
    })
}

/// Load a knowledge graph TOML (topic routing rules) into the global TopicGraph.
/// This is essential for WASM inference — without it, `infer_operation_topic` falls
/// back to legacy keyword rules that only know math operations, not domain-specific
/// topics (e.g., pet chat greeting/anxiety/play topics).
///
/// Accepts a base TOML string and an optional overlay TOML string.
/// Call after `growformer_load_brain` and before inference.
#[wasm_bindgen]
pub fn growformer_load_topic_graph(
    base_toml: &str,
    overlay_toml: Option<String>,
) -> Result<(), JsValue> {
    let base_graph = crate::topic_graph::TopicGraph::from_toml(base_toml)
        .map_err(|e| JsValue::from_str(&format!("topic graph base: {}", e)))?;
    let final_graph = if let Some(ref overlay) = overlay_toml {
        let overlay_graph = crate::topic_graph::TopicGraph::from_toml_quiet(overlay)
            .map_err(|e| JsValue::from_str(&format!("topic graph overlay: {}", e)))?;
        base_graph.merge_overlay(overlay_graph)
    } else {
        base_graph
    };
    crate::growformer_lang::init_topic_graph_direct(final_graph).map_err(|e| JsValue::from_str(&e))
}

/// Drop the in-memory topic graph so the next `growformer_load_topic_graph` installs fresh rules.
#[wasm_bindgen]
pub fn growformer_clear_topic_graph() {
    crate::growformer_lang::clear_topic_graph();
}

/// Load a world grounding graph TOML (concept nodes, typed edges, disambiguation).
/// This enables BM25 keyword expansion, anchor resolution, and disambiguation
/// during retrieval. Call after `growformer_load_brain` and before inference.
#[wasm_bindgen]
pub fn growformer_load_grounding_graph(toml_str: &str) -> Result<(), JsValue> {
    crate::inference::world_grounding::load_grounding_graph_from_str(toml_str)
        .map_err(|e| JsValue::from_str(&e))
}

/// Set agent runtime state for state-conditioned generation.
/// Accepts a JSON object with arbitrary string→float state dimensions
/// and an optional archetype/profile string. The state is blended into the
/// conditioning vector on subsequent `growformer_converse` calls.
///
/// Example: `{"dimensions": {"hunger": 0.4, "energy": 0.6, "mood": 0.7}, "profile": "cheerful_companion", "turn": 3}`
#[wasm_bindgen]
pub fn growformer_set_agent_state(state_json: &str) -> Result<(), JsValue> {
    with_rt(|rt| rt.set_agent_state_from_json(state_json))
}

/// Load a JSONL fragment library for chat-mode fragment composition.
/// Call after `growformer_load_inference_toml` (which enables `[fragment_compose]`).
/// Returns the number of fragments loaded.
#[wasm_bindgen]
pub fn growformer_load_fragments_jsonl(jsonl: &str) -> Result<usize, JsValue> {
    with_rt(|rt| Ok(rt.svc.load_fragments_from_str(jsonl)))
}

#[wasm_bindgen]
pub fn growformer_load_checkpoints(data: &[u8]) -> Result<usize, JsValue> {
    with_rt(|rt| rt.svc.load_gle_students_from_bytes(&[data]))
}
