//! WASM bindings for the Growformer inference runtime.
//!
//! Two APIs:
//!
//! - The `growformer_*` functions drive one brain, stored in a `thread_local!`
//!   cell and initialised by `growformer_init` or `growformer_load_brain`.
//! - `growformer_instance_create` and the `*_i` functions hold several brains,
//!   each with its own inference rules, topic graph and grounding graph
//!   ([`crate::brain_instances`]), side by side and alongside the functions.

use std::cell::RefCell;
use wasm_bindgen::prelude::*;

use crate::brain_instances::{self, BrainInstance};
use crate::runtime::Runtime;
use crate::service::AgentMode;

thread_local! {
    static RT: RefCell<Option<Runtime>> = RefCell::new(None);
}

fn with_rt<F, R>(f: F) -> Result<R, JsValue>
where
    F: FnOnce(&mut Runtime) -> Result<R, String>,
{
    brain_instances::activate_default();
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
    brain_instances::activate_default();
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

    brain_instances::activate_default();
    let rt = Runtime::empty().map_err(|e| JsValue::from_str(&e))?;
    RT.with(|cell| {
        *cell.borrow_mut() = Some(rt);
    });
    brain_instances::capture_default();
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
    brain_instances::activate_default();
    let out = RT.with(|cell| {
        let mut opt = cell.borrow_mut();
        match opt.as_mut() {
            Some(rt) => rt.load_brain(data).map_err(|e| JsValue::from_str(&e)),
            None => {
                let rt = Runtime::from_brain_bytes(data).map_err(|e| JsValue::from_str(&e))?;
                *opt = Some(rt);
                Ok(())
            }
        }
    });
    brain_instances::capture_default();
    out
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
    brain_instances::activate_default();
    crate::inference::inference_toml::reload_inference_toml_from_str(toml_str)
        .map_err(|e| JsValue::from_str(&e))?;

    with_rt(|rt| {
        rt.apply_loaded_generation_config();
        Ok(())
    })?;
    brain_instances::capture_default();
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
    brain_instances::activate_default();
    let base_graph = crate::topic_graph::TopicGraph::from_toml(base_toml)
        .map_err(|e| JsValue::from_str(&format!("topic graph base: {}", e)))?;
    let final_graph = if let Some(ref overlay) = overlay_toml {
        let overlay_graph = crate::topic_graph::TopicGraph::from_toml_quiet(overlay)
            .map_err(|e| JsValue::from_str(&format!("topic graph overlay: {}", e)))?;
        base_graph.merge_overlay(overlay_graph)
    } else {
        base_graph
    };
    let out = crate::growformer_lang::init_topic_graph_direct(final_graph)
        .map_err(|e| JsValue::from_str(&e));
    brain_instances::capture_default();
    out
}

/// Drop the in-memory topic graph so the next `growformer_load_topic_graph` installs fresh rules.
#[wasm_bindgen]
pub fn growformer_clear_topic_graph() {
    brain_instances::activate_default();
    crate::growformer_lang::clear_topic_graph();
    brain_instances::capture_default();
}

/// Load a world grounding graph TOML (concept nodes, typed edges, disambiguation).
/// This enables BM25 keyword expansion, anchor resolution, and disambiguation
/// during retrieval. Call after `growformer_load_brain` and before inference.
#[wasm_bindgen]
pub fn growformer_load_grounding_graph(toml_str: &str) -> Result<(), JsValue> {
    brain_instances::activate_default();
    let out = crate::inference::world_grounding::load_grounding_graph_from_str(toml_str)
        .map_err(|e| JsValue::from_str(&e));
    brain_instances::capture_default();
    out
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

// ───────────────────────────────────────────────────────────────────────────
// Several brains: instance API
// ───────────────────────────────────────────────────────────────────────────
//
// `growformer_instance_create` returns an id; the `*_i` functions take it as
// their first argument. Each instance keeps its own brain, conversation,
// inference rules, topic graph and grounding graph (`crate::brain_instances`).
// Id 0 is the single brain the functions above use.

use std::collections::BTreeMap;

thread_local! {
    static INSTANCES: RefCell<BTreeMap<u32, BrainInstance>> = RefCell::new(BTreeMap::new());
}

fn with_instance<R>(
    id: u32,
    f: impl FnOnce(&mut BrainInstance) -> Result<R, String>,
) -> Result<R, JsValue> {
    INSTANCES.with(|cell| {
        let mut map = cell.borrow_mut();
        let inst = map
            .get_mut(&id)
            .ok_or_else(|| JsValue::from_str(&format!("unknown growformer instance {}", id)))?;
        f(inst).map_err(|e| JsValue::from_str(&e))
    })
}

fn json_value<T: serde::Serialize>(value: &T) -> Result<JsValue, String> {
    serde_json::to_string(value)
        .map(|s| JsValue::from_str(&s))
        .map_err(|e| e.to_string())
}

/// What this build is: engine name and version, brain package formats it reads,
/// whether gzip-wrapped brains load, and whether the instance API exists.
#[wasm_bindgen]
pub fn growformer_engine_info() -> JsValue {
    let info = serde_json::json!({
        "engine": "growformer",
        "version": env!("CARGO_PKG_VERSION"),
        "brain_formats": [1, 2],
        "compressed_brains": cfg!(feature = "brain-compression"),
        "multi_instance": true,
    });
    JsValue::from_str(&info.to_string())
}

/// Create an empty instance (load a brain into it next). Returns its id (≥ 1).
#[wasm_bindgen]
pub fn growformer_instance_create() -> Result<u32, JsValue> {
    #[cfg(feature = "wasm-bindgen")]
    console_error_panic_hook::set_once();
    let inst = BrainInstance::empty().map_err(|e| JsValue::from_str(&e))?;
    let id =
        u32::try_from(inst.id()).map_err(|_| JsValue::from_str("too many growformer instances"))?;
    INSTANCES.with(|cell| cell.borrow_mut().insert(id, inst));
    Ok(id)
}

/// Free an instance. Returns false when it did not exist.
#[wasm_bindgen]
pub fn growformer_instance_destroy(id: u32) -> bool {
    INSTANCES
        .with(|cell| cell.borrow_mut().remove(&id))
        .is_some()
}

/// JSON array of live ids, including 0 (the single-brain API).
#[wasm_bindgen]
pub fn growformer_instance_list() -> JsValue {
    let mut ids = vec![0u32];
    INSTANCES.with(|cell| ids.extend(cell.borrow().keys().copied()));
    JsValue::from_str(&serde_json::to_string(&ids).unwrap_or_else(|_| "[0]".into()))
}

#[wasm_bindgen]
pub fn growformer_load_brain_i(id: u32, data: &[u8]) -> Result<(), JsValue> {
    with_instance(id, |i| i.load_brain(data))
}

#[wasm_bindgen]
pub fn growformer_ready_i(id: u32) -> Result<bool, JsValue> {
    with_instance(id, |i| Ok(i.with(|rt| rt.brain_info().num_groups > 0)))
}

#[wasm_bindgen]
pub fn growformer_load_inference_toml_i(id: u32, toml_str: &str) -> Result<(), JsValue> {
    with_instance(id, |i| i.load_inference_toml(toml_str))
}

#[wasm_bindgen]
pub fn growformer_load_topic_graph_i(
    id: u32,
    base_toml: &str,
    overlay_toml: Option<String>,
) -> Result<(), JsValue> {
    with_instance(id, |i| {
        i.load_topic_graph(base_toml, overlay_toml.as_deref())
    })
}

#[wasm_bindgen]
pub fn growformer_clear_topic_graph_i(id: u32) -> Result<(), JsValue> {
    with_instance(id, |i| {
        i.clear_topic_graph();
        Ok(())
    })
}

#[wasm_bindgen]
pub fn growformer_load_grounding_graph_i(id: u32, toml_str: &str) -> Result<(), JsValue> {
    with_instance(id, |i| i.load_grounding_graph(toml_str))
}

#[wasm_bindgen]
pub fn growformer_load_fragments_jsonl_i(id: u32, jsonl: &str) -> Result<usize, JsValue> {
    with_instance(id, |i| {
        Ok(i.with(|rt| rt.svc.load_fragments_from_str(jsonl)))
    })
}

#[wasm_bindgen]
pub fn growformer_set_agent_state_i(id: u32, state_json: &str) -> Result<(), JsValue> {
    with_instance(id, |i| {
        i.with(|rt| rt.set_agent_state_from_json(state_json))
    })
}

#[wasm_bindgen]
pub fn growformer_enable_stochastic_retrieval_i(id: u32, temperature: f32) -> Result<(), JsValue> {
    with_instance(id, |i| {
        i.with(|rt| rt.enable_stochastic_retrieval(temperature));
        Ok(())
    })
}

#[wasm_bindgen]
pub fn growformer_brain_info_i(id: u32) -> Result<JsValue, JsValue> {
    with_instance(id, |i| json_value(&i.with(|rt| rt.brain_info())))
}

#[wasm_bindgen]
pub fn growformer_generation_i(id: u32, text: &str) -> Result<JsValue, JsValue> {
    with_instance(id, |i| {
        i.with(|rt| rt.prompt(text)).and_then(|r| json_value(&r))
    })
}

#[wasm_bindgen]
pub fn growformer_converse_i(id: u32, text: &str) -> Result<JsValue, JsValue> {
    with_instance(id, |i| {
        i.with(|rt| rt.converse(text)).and_then(|r| json_value(&r))
    })
}

#[wasm_bindgen]
pub fn growformer_codegen_i(id: u32, text: &str) -> Result<JsValue, JsValue> {
    with_instance(id, |i| {
        i.with(|rt| rt.codegen(text)).and_then(|r| json_value(&r))
    })
}

#[wasm_bindgen]
pub fn growformer_reset_conversation_i(id: u32) -> Result<(), JsValue> {
    with_instance(id, |i| {
        i.with(|rt| rt.reset_conversation());
        Ok(())
    })
}

#[wasm_bindgen]
pub fn growformer_export_brain_i(id: u32) -> Result<Vec<u8>, JsValue> {
    with_instance(id, |i| i.with(|rt| rt.export_brain()))
}
