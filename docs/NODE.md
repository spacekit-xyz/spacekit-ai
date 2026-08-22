# Growformer Node

`growformer-node` is the HTTP dev server for integrating Growformer with a frontend chat UI.

## Start

```bash
cargo run --bin growformer-node
```

Defaults:
- bind address: `127.0.0.1:8080`
- runtime: in-process Growformer library (`growformer::service::LanguageService`)

Optional env vars:
- `GROWFORMER_NODE_ADDR` (example: `0.0.0.0:8080`)
- `GROWFORMER_NODE_TOKEN` (optional bearer token for `/v1/chat*`)
- `GROWFORMER_NODE_LOG_PATH` (optional JSONL perf log path)
- `GROWFORMER_GLE_CHECKPOINT` (single local GLE checkpoint path)
- `GROWFORMER_GLE_CHECKPOINTS` (comma-separated local checkpoint list for ensemble)
- `GROWFORMER_GLE_WEIGHTS` (comma-separated weights for `GROWFORMER_GLE_CHECKPOINTS`)

## Endpoints

### `GET /v1/health`

Returns server status, runtime mode, and active agent mode.

```json
{
  "status": "ok",
  "runtime": "in_process_lib",
  "agent_mode": "MicroBrain",
  "log_path": null
}
```

### `POST /v1/chat`

Request JSON:

```json
{
  "session_id": "optional",
  "mode": "codegen",
  "message": "implement a web server in rust",
  "agent_mode": "micro_brain",
  "brain": "optional named brain (see GET /v1/brains)",
  "context_snippets": ["optional retrieval context"],
  "feedback": {
    "outcome": "accept | reject | correct",
    "correction": "optional string when outcome is correct"
  },
  "options": {
    "include_raw_stdout": false
  }
}
```

Optional `feedback` (Continuum): applies to the **previous** turn. Use `reject` or `correct` (with optional `correction` text) to signal a learning signal; training step is not yet wired (see `docs/CONTINUUM.md`).

Supported `mode`:
- `action` -> routes text and returns `Action JSON`
- `generation` -> routes text and returns template generation output
- `codegen` -> routes text and returns generated code output

Optional `agent_mode` (M6):
- `context_file` or `ContextFile` -> switches to context-file agent before processing
- `micro_brain` or `MicroBrain` -> switches to micro-brain agent before processing

Optional `context_snippets` (M6):
- Array of strings injected as retrieval context for the current request

Response JSON:

```json
{
  "session_id": "uuid",
  "mode": "codegen",
  "agent_mode": "MicroBrain",
  "latency_ms": 1450,
  "perf": {
    "child_max_rss_bytes": null,
    "child_user_s": null,
    "child_sys_s": null
  },
  "output": {
    "text": "Generated code (...) ...",
    "raw_stdout": "optional diagnostic text payload"
  }
}
```

### `POST /v1/chat/stream`

SSE variant of `/v1/chat`. Emits:
- `message` event with the full JSON response payload
- `done` event

### `GET /v1/brains`

Returns list of loaded brain names and the active brain. Optional env: `GROWFORMER_BRAIN_DIR` to auto-load all `*.bin` files.

```json
{
  "brains": ["brain", "default"],
  "active": "brain"
}
```

### `POST /v1/brain/save` (Continuum)

Persist the current in-memory state of the active brain (or a named brain) to disk. Requires auth token if configured.

Request JSON:

```json
{
  "path": "optional output path (default: GROWFORMER_BRAIN_PATH or brain.bin)",
  "brain": "optional brain name to save (default: active brain)"
}
```

Response:

```json
{"ok": true, "path": "brain.bin", "bytes": 12345}
```

### `POST /v1/mode` (M6)

Switch agent mode. Requires auth token if configured.

```json
{
  "mode": "context_file",
  "confidence": 0.9,
  "reason": "user requested retrieval mode"
}
```

Response:

```json
{"ok": true, "mode": "ContextFile"}
```

### `GET /v1/acceptance` (M6)

Returns the full M6 acceptance report: understanding metrics, generation metrics, continual learning metrics, system SLO snapshot, and mode handoff counts.

```json
{
  "understanding": {
    "groups_count": 2,
    "routing_confidence_streak": 0,
    "auto_spawn_k": 10
  },
  "generation": {
    "template_based": true,
    "codegen_languages": ["python", "rust", "javascript"]
  },
  "continual_learning": {
    "episodic_episodes": 0,
    "checkpoint_summary": {
      "promoted_groups": 2,
      "active_mirrors": 0,
      "episodic_episodes": 0
    }
  },
  "system": {
    "slo": {
      "latency_samples": [],
      "latency_p95_ms": 1.6,
      "checkpoint_domains": 2,
      "latency_ok": true,
      "checkpoint_ok": true
    }
  },
  "modes": {
    "active_mode": "MicroBrain",
    "handoff_count": 0,
    "modes_available": ["ContextFile", "MicroBrain"]
  },
  "passed": true
}
```

## React/TypeScript Example

```ts
type ChatReq = {
  session_id?: string;
  mode: "action" | "generation" | "codegen";
  message: string;
  agent_mode?: "context_file" | "micro_brain";
  context_snippets?: string[];
};

export async function chatWithGrowformer(req: ChatReq) {
  const res = await fetch("http://127.0.0.1:8080/v1/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function switchMode(mode: "context_file" | "micro_brain") {
  const res = await fetch("http://127.0.0.1:8080/v1/mode", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mode }),
  });
  return res.json();
}

export async function getAcceptanceReport() {
  const res = await fetch("http://127.0.0.1:8080/v1/acceptance");
  return res.json();
}
```

## Notes

- `growformer-node` runs inference directly through the shared library service.
- CLI and Node share the same language pipeline initialization path.
- The service instance is kept warm in-memory for request handling.
- Agent mode handoffs are logged to the JSONL perf log when `GROWFORMER_NODE_LOG_PATH` is set.
- Handoff log entries include `agent_mode` field alongside existing perf fields.
