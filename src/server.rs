use axum::extract::State;
use axum::http::{HeaderMap, Method, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::routing::{get, post};
use axum::{Json, Router};
use growformer::service::{AgentMode, Feedback, LanguageService};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::convert::Infallible;
use std::fs::OpenOptions;
use std::io::Write;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Mutex;
use tower_http::cors::{Any, CorsLayer};
use uuid::Uuid;

#[derive(Clone)]
struct AppState {
    service: Arc<Mutex<LanguageService>>,
    auth_token: Option<String>,
    log_path: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChatRequest {
    session_id: Option<String>,
    mode: String,
    message: String,
    #[serde(default)]
    agent_mode: Option<String>,
    /// Optional: use this named checkpoint for this request (see GET /v1/brains).
    #[serde(default)]
    brain: Option<String>,
    #[serde(default)]
    context_snippets: Vec<String>,
    #[serde(default)]
    options: ChatOptions,
    /// Optional: feedback for the *previous* turn (outcome: accept | reject | correct; correction text for correct).
    #[serde(default)]
    feedback: Option<Feedback>,
}

#[derive(Debug, Default, Deserialize)]
struct ChatOptions {
    #[serde(default)]
    include_raw_stdout: bool,
}

#[derive(Debug, Serialize)]
struct ChatResponse {
    session_id: String,
    mode: String,
    agent_mode: String,
    latency_ms: u128,
    perf: ChatPerf,
    output: ChatOutput,
}

#[derive(Debug, Serialize)]
struct ChatOutput {
    text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    raw_stdout: Option<String>,
}

#[derive(Debug, Serialize)]
struct ChatPerf {
    child_max_rss_bytes: Option<u64>,
    child_user_s: Option<f32>,
    child_sys_s: Option<f32>,
}

#[derive(Debug, Serialize)]
struct HealthResponse {
    status: &'static str,
    runtime: &'static str,
    agent_mode: String,
    log_path: Option<String>,
}

#[tokio::main]
async fn main() {
    let addr: SocketAddr = std::env::var("GROWFORMER_NODE_ADDR")
        .unwrap_or_else(|_| "127.0.0.1:8080".to_string())
        .parse()
        .expect("invalid GROWFORMER_NODE_ADDR");
    let auth_token = std::env::var("GROWFORMER_NODE_TOKEN").ok();
    let log_path = std::env::var("GROWFORMER_NODE_LOG_PATH").ok();
    let mut service = LanguageService::new_default().expect("failed to initialize language service");

    // Auto-load trained brain(s)
    // - GROWFORMER_BRAIN_DIR: directory of .bin files → load each as name = filename stem, first = active
    // - GROWFORMER_BRAIN_PATH: single file → load as "default" (legacy)
    if let Ok(dir) = std::env::var("GROWFORMER_BRAIN_DIR") {
        let mut names: Vec<String> = Vec::new();
        if let Ok(entries) = std::fs::read_dir(&dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().map_or(false, |e| e == "bin") {
                    let name = path
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("unnamed")
                        .to_string();
                    if let Ok(data) = std::fs::read(&path) {
                        if service.load_brain_as(&name, &data).is_ok() {
                            names.push(name);
                        }
                    }
                }
            }
        }
        if !names.is_empty() {
            names.sort();
            let active = names.first().cloned().unwrap_or_default();
            service.set_active_brain(&active);
            println!(
                "Brains loaded from {}: {} (active: {})",
                dir,
                names.join(", "),
                active
            );
        }
    } else {
        let brain_path = std::env::var("GROWFORMER_BRAIN_PATH")
            .unwrap_or_else(|_| "brain.bin".to_string());
        if let Ok(data) = std::fs::read(&brain_path) {
            match service.load_brain(&data) {
                Ok(()) => {
                    let dm = service.active_dm();
                    let has_gen = dm.generation_head.is_some();
                    let has_code = dm.codegen_head.is_some();
                    let has_clf = dm.action_classifier.is_some();
                    let has_router = dm.observer.learned_router.is_some();
                    println!(
                        "Brain loaded: {} ({} KB) router={} classifier={} gen_head={} code_head={}",
                        brain_path,
                        data.len() / 1024,
                        has_router,
                        has_clf,
                        has_gen,
                        has_code
                    );
                }
                Err(e) => eprintln!("Warning: failed to load brain {}: {}", brain_path, e),
            }
        }
    }

    let state = Arc::new(AppState {
        service: Arc::new(Mutex::new(service)),
        auth_token,
        log_path,
    });
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods([Method::GET, Method::POST])
        .allow_headers(Any);
    let app = Router::new()
        .route("/v1/health", get(health))
        .route("/v1/brains", get(list_brains))
        .route("/v1/chat", post(chat))
        .route("/v1/chat/stream", post(chat_stream))
        .route("/v1/acceptance", get(acceptance))
        .route("/v1/mode", post(set_mode))
        .route("/v1/brain/save", post(brain_save))
        .layer(cors)
        .with_state(state.clone());

    println!("Growformer Node listening on http://{}", addr);
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("bind failed");
    axum::serve(listener, app).await.expect("server failed");
}

async fn health(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    let svc = state.service.lock().await;
    let mode_str = format!("{:?}", svc.active_mode());
    Json(HealthResponse {
        status: "ok",
        runtime: "in_process_lib",
        agent_mode: mode_str,
        log_path: state.log_path.clone(),
    })
}

#[derive(Debug, Serialize)]
struct BrainsResponse {
    brains: Vec<String>,
    active: String,
}

async fn list_brains(State(state): State<Arc<AppState>>) -> Json<BrainsResponse> {
    let svc = state.service.lock().await;
    let brains = svc.list_brains();
    let active = svc.active_brain.clone();
    Json(BrainsResponse { brains, active })
}

#[derive(Debug, Deserialize)]
struct BrainSaveRequest {
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    brain: Option<String>,
}

async fn brain_save(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<BrainSaveRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    authorize(&headers, &state)?;
    let path = req.path.or_else(|| std::env::var("GROWFORMER_BRAIN_PATH").ok())
        .unwrap_or_else(|| "brain.bin".to_string());
    let mut svc = state.service.lock().await;
    let prev_active = svc.active_brain.clone();
    if let Some(name) = &req.brain {
        if !svc.list_brains().contains(name) {
            return Err((StatusCode::BAD_REQUEST, format!("unknown brain '{}'", name)));
        }
        svc.set_active_brain(name);
    }
    let bytes = svc.export_brain().map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
    svc.set_active_brain(&prev_active);
    std::fs::write(&path, &bytes).map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(Json(json!({ "ok": true, "path": path, "bytes": bytes.len() })))
}

async fn acceptance(
    State(state): State<Arc<AppState>>,
) -> Json<serde_json::Value> {
    let svc = state.service.lock().await;
    let report = svc.acceptance_report();
    Json(serde_json::to_value(&report).unwrap_or_default())
}

#[derive(Debug, Deserialize)]
struct SetModeRequest {
    mode: String,
    confidence: Option<f32>,
    reason: Option<String>,
}

async fn set_mode(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<SetModeRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    authorize(&headers, &state)?;
    let new_mode = match req.mode.as_str() {
        "context_file" | "ContextFile" => AgentMode::ContextFile,
        "micro_brain" | "MicroBrain" => AgentMode::MicroBrain,
        other => {
            return Err((
                StatusCode::BAD_REQUEST,
                format!("unknown mode '{}'; use context_file|micro_brain", other),
            ))
        }
    };
    let mut svc = state.service.lock().await;
    svc.set_mode(
        new_mode,
        req.confidence.unwrap_or(1.0),
        req.reason.as_deref().unwrap_or("api_request"),
    );
    Ok(Json(json!({"ok": true, "mode": format!("{:?}", new_mode)})))
}

async fn chat(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<ChatRequest>,
) -> Result<Json<ChatResponse>, (StatusCode, String)> {
    authorize(&headers, &state)?;
    run_chat(state, req).await.map(Json)
}

async fn chat_stream(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<ChatRequest>,
) -> Result<Sse<impl futures_core::Stream<Item = Result<Event, Infallible>>>, (StatusCode, String)> {
    authorize(&headers, &state)?;
    let stream = async_stream::stream! {
        match run_chat(state, req).await {
            Ok(resp) => {
                let payload = serde_json::to_string(&resp)
                    .unwrap_or_else(|_| "{\"error\":\"serialize\"}".to_string());
                yield Ok(Event::default().event("message").data(payload));
            }
            Err((status, msg)) => {
                let payload = format!("{{\"status\":{},\"error\":{}}}", status.as_u16(), serde_json::to_string(&msg).unwrap_or_else(|_| "\"unknown\"".to_string()));
                yield Ok(Event::default().event("error").data(payload));
            }
        }
        yield Ok(Event::default().event("done").data("{\"ok\":true}"));
    };
    Ok(Sse::new(stream).keep_alive(KeepAlive::new().interval(std::time::Duration::from_secs(10))))
}

async fn run_chat(
    state: Arc<AppState>,
    req: ChatRequest,
) -> Result<ChatResponse, (StatusCode, String)> {
    let started = Instant::now();
    let session_id = req
        .session_id
        .clone()
        .unwrap_or_else(|| Uuid::new_v4().to_string());

    let mut svc = state.service.lock().await;

    if let Some(name) = &req.brain {
        svc.set_active_brain(name);
    }
    if let Some(am) = &req.agent_mode {
        let new_mode = match am.as_str() {
            "context_file" | "ContextFile" => Some(AgentMode::ContextFile),
            "micro_brain" | "MicroBrain" => Some(AgentMode::MicroBrain),
            _ => None,
        };
        if let Some(m) = new_mode {
            svc.set_mode(m, 1.0, "chat_request");
        }
    }

    for snippet in &req.context_snippets {
        svc.push_context_snippet(snippet.clone());
    }

    if let Some(ref feedback) = req.feedback {
        let _ = svc.submit_feedback(feedback);
    }

    let active_mode = svc.active_mode();

    let (action, text, raw_stdout) = match req.mode.as_str() {
        "action" => {
            let action = svc
                .action(&req.message)
                .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("inference failed: {}", e)))?;
            let pretty = serde_json::to_string_pretty(&action)
                .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("serialize failed: {}", e)))?;
            let text = format!("Action JSON:\n{}", pretty);
            (action, text.clone(), pretty)
        }
        "generation" => {
            let (action, generated) = svc
                .generation(&req.message)
                .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("inference failed: {}", e)))?;
            let action_json = serde_json::to_string_pretty(&action)
                .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("serialize failed: {}", e)))?;
            let text = format!(
                "Action JSON:\n{}\n\nTemplate response:\n{}",
                action_json, generated.text
            );
            (action, text.clone(), generated.text)
        }
        "codegen" => {
            let (action, code) = svc
                .codegen(&req.message)
                .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("inference failed: {}", e)))?;
            let action_json = serde_json::to_string_pretty(&action)
                .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("serialize failed: {}", e)))?;
            let text = match &code {
                Some(code) => format!(
                    "Action JSON:\n{}\n\nGenerated code ({}, {}):\n{}",
                    action_json, code.language, code.kind, code.code
                ),
                None => format!("Action JSON:\n{}\n\nNo code output", action_json),
            };
            (action, text.clone(), text)
        }
        other => {
            return Err((
                StatusCode::BAD_REQUEST,
                format!("unsupported mode '{}'; use action|generation|codegen", other),
            ))
        }
    };
    svc.record_turn(&req.message, action.target_group_id, &text);
    drop(svc);
    let latency_ms = started.elapsed().as_millis();
    let perf = ChatPerf {
        child_max_rss_bytes: None,
        child_user_s: None,
        child_sys_s: None,
    };

    if let Some(path) = &state.log_path {
        let line = json!({
            "ts_unix_ms": std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis())
                .unwrap_or(0),
            "session_id": session_id,
            "mode": req.mode,
            "agent_mode": format!("{:?}", active_mode),
            "message_chars": req.message.len(),
            "latency_ms": latency_ms,
            "perf": {
                "child_max_rss_bytes": perf.child_max_rss_bytes,
                "child_user_s": perf.child_user_s,
                "child_sys_s": perf.child_sys_s
            }
        })
        .to_string();
        let _ = append_jsonl(path, &line);
    }

    Ok(ChatResponse {
        session_id: session_id.to_string(),
        mode: req.mode.to_string(),
        agent_mode: format!("{:?}", active_mode),
        latency_ms,
        perf,
        output: ChatOutput {
            text,
            raw_stdout: if req.options.include_raw_stdout {
                Some(raw_stdout)
            } else {
                None
            },
        },
    })
}

fn authorize(headers: &HeaderMap, state: &AppState) -> Result<(), (StatusCode, String)> {
    let Some(token) = &state.auth_token else {
        return Ok(());
    };
    let auth = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    let expected = format!("Bearer {}", token);
    if auth == expected {
        Ok(())
    } else {
        Err((StatusCode::UNAUTHORIZED, "missing/invalid bearer token".to_string()))
    }
}

fn append_jsonl(path: &str, line: &str) -> std::io::Result<()> {
    if let Some(parent) = std::path::Path::new(path).parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut f = OpenOptions::new().create(true).append(true).open(path)?;
    writeln!(f, "{}", line)?;
    Ok(())
}
