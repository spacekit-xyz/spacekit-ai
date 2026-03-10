use axum::extract::State;
use axum::http::{HeaderMap, Method, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::convert::Infallible;
use std::net::SocketAddr;
use std::process::Command;
use std::sync::Arc;
use std::fs::OpenOptions;
use std::io::Write;
use std::time::Instant;
use tower_http::cors::{Any, CorsLayer};
use uuid::Uuid;

#[derive(Clone)]
struct AppState {
    growformer_bin: String,
    auth_token: Option<String>,
    log_path: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChatRequest {
    session_id: Option<String>,
    mode: String,
    message: String,
    #[serde(default)]
    options: ChatOptions,
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
    growformer_bin: String,
    log_path: Option<String>,
}

#[tokio::main]
async fn main() {
    let addr: SocketAddr = std::env::var("GROWFORMER_NODE_ADDR")
        .unwrap_or_else(|_| "127.0.0.1:8080".to_string())
        .parse()
        .expect("invalid GROWFORMER_NODE_ADDR");
    let bin = std::env::var("GROWFORMER_BIN").unwrap_or_else(|_| "target/debug/growformer".to_string());
    let auth_token = std::env::var("GROWFORMER_NODE_TOKEN").ok();
    let log_path = std::env::var("GROWFORMER_NODE_LOG_PATH").ok();

    let state = Arc::new(AppState {
        growformer_bin: bin,
        auth_token,
        log_path,
    });
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods([Method::GET, Method::POST])
        .allow_headers(Any);
    let app = Router::new()
        .route("/v1/health", get(health))
        .route("/v1/chat", post(chat))
        .route("/v1/chat/stream", post(chat_stream))
        .layer(cors)
        .with_state(state.clone());

    println!("Growformer Node listening on http://{}", addr);
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("bind failed");
    axum::serve(listener, app).await.expect("server failed");
}

async fn health(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        growformer_bin: state.growformer_bin.clone(),
        log_path: state.log_path.clone(),
    })
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

    let flag = match req.mode.as_str() {
        "action" => "--language-action-text",
        "generation" => "--language-generate-text",
        "codegen" => "--language-code-text",
        other => {
            return Err((
                StatusCode::BAD_REQUEST,
                format!("unsupported mode '{}'; use action|generation|codegen", other),
            ))
        }
    };

    let bin = state.growformer_bin.clone();
    let msg = req.message.clone();
    let output = tokio::task::spawn_blocking(move || run_growformer_with_metrics(&bin, flag, &msg))
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("join error: {}", e)))?
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("failed to execute '{}': {}", state.growformer_bin, e),
            )
        })?;

    if !output.output.status.success() {
        let stderr = String::from_utf8_lossy(&output.output.stderr).to_string();
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("growformer failed: {}", stderr),
        ));
    }

    let stdout = String::from_utf8_lossy(&output.output.stdout).to_string();
    let text = extract_primary_output(&stdout);
    let latency_ms = started.elapsed().as_millis();
    let perf = ChatPerf {
        child_max_rss_bytes: output.max_rss_bytes,
        child_user_s: output.user_s,
        child_sys_s: output.sys_s,
    };

    if let Some(path) = &state.log_path {
        let line = json!({
            "ts_unix_ms": std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis())
                .unwrap_or(0),
            "session_id": session_id,
            "mode": req.mode,
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
        latency_ms,
        perf,
        output: ChatOutput {
            text,
            raw_stdout: if req.options.include_raw_stdout {
                Some(stdout)
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

fn extract_primary_output(stdout: &str) -> String {
    if let Some(i) = stdout.find("Generated code") {
        return stdout[i..].to_string();
    }
    if let Some(i) = stdout.find("Template response:") {
        return stdout[i..].to_string();
    }
    if let Some(i) = stdout.find("Action JSON:") {
        return stdout[i..].to_string();
    }
    stdout.to_string()
}

struct ChildRunOutput {
    output: std::process::Output,
    max_rss_bytes: Option<u64>,
    user_s: Option<f32>,
    sys_s: Option<f32>,
}

fn run_growformer_with_metrics(bin: &str, flag: &str, msg: &str) -> std::io::Result<ChildRunOutput> {
    let timed = Command::new("/usr/bin/time")
        .arg("-lp")
        .arg(bin)
        .arg(flag)
        .arg(msg)
        .output();

    if let Ok(o) = timed {
        let stderr = String::from_utf8_lossy(&o.stderr).to_string();
        let (rss, user_s, sys_s) = parse_time_lp(&stderr);
        return Ok(ChildRunOutput {
            output: o,
            max_rss_bytes: rss,
            user_s,
            sys_s,
        });
    }

    let o = Command::new(bin).arg(flag).arg(msg).output()?;
    Ok(ChildRunOutput {
        output: o,
        max_rss_bytes: None,
        user_s: None,
        sys_s: None,
    })
}

fn parse_time_lp(stderr: &str) -> (Option<u64>, Option<f32>, Option<f32>) {
    let mut rss = None;
    let mut user_s = None;
    let mut sys_s = None;
    for line in stderr.lines() {
        let t = line.trim();
        if let Some(rest) = t.strip_prefix("user ") {
            user_s = rest.trim().parse::<f32>().ok();
        } else if let Some(rest) = t.strip_prefix("sys ") {
            sys_s = rest.trim().parse::<f32>().ok();
        } else if t.contains("maximum resident set size") {
            let n = t.split_whitespace().next().and_then(|x| x.parse::<u64>().ok());
            rss = n;
        }
    }
    (rss, user_s, sys_s)
}

fn append_jsonl(path: &str, line: &str) -> std::io::Result<()> {
    if let Some(parent) = std::path::Path::new(path).parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut f = OpenOptions::new().create(true).append(true).open(path)?;
    writeln!(f, "{}", line)?;
    Ok(())
}