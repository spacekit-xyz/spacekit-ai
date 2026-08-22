# World-Model SpaceKit Host

**Non-goal:** Luna / chat accuracy is **not** a WM certifier.

## Protocol

JSON lines (one request → one response). Implemented by [`WmHostSession`](../src/dimension/wm_open.rs).

| `op` | Body | Effect |
|------|------|--------|
| `load_bundle` | `{ "path": "..." }` | Load pinned `ComposedWmBundle`; verify encoder fingerprint |
| `step` | `{ "obs": [x,y,vx,vy] }` | `deploy_step` → route / abstain / energies |
| `fingerprint` | — | Current encoder pin |
| `status` | — | Loaded path + note |

Example:

```json
{"op":"load_bundle","path":"/tmp/bundle.json"}
{"op":"step","obs":[0.1,-0.2,0.0,0.05]}
{"op":"fingerprint"}
```

## SpaceKit wiring

- Growformer library API: `WmHostSession::handle` / `handle_json` (also acting / scene hosts).
- Demo / certifier: `cargo run --release --bin growformer-demos -- --phase3s-open-ladder`
- **Stdio JSONL host (product glue):**

```bash
# One JSON request per stdin line → one JSON response per stdout line
cargo run --release --bin growformer-demos -- --wm-host-stdio scene
cargo run --release --bin growformer-demos -- --wm-host-stdio acting
cargo run --release --bin growformer-demos -- --wm-host-stdio deploy

# Thin client helper
python3 scripts/wm_spacekit_client.py scene --bundle /tmp/scene_bundle_42.json
```

## Pin contract

Reload after “process restart” (new session + same file) must return the **same** `encoder_fingerprint`. Any silent encoder update is a kill.

## Acting host (Phase 3t / §8 F)

Separate session: `ActingHostSession` in [`wm_act.rs`](../src/dimension/wm_act.rs).

| `op` | Body | Effect |
|------|------|--------|
| `load_acting` | `{ "path": "..." }` | Load pinned `ActingWmBundle` |
| `act` | `{ "obs": [x,y,vx,vy] }` | Route → `plan_action` → discrete action |
| `fingerprint` | — | Encoder pin |

**Non-goal:** chat / Luna accuracy is not a WM certifier. Demo: `--phase3t-act-wm`.

## Scene-graph host (Phase 3w / WM-1 deploy)

Session: `SceneHostSession` in [`wm_scene_host.rs`](../src/dimension/wm_scene_host.rs). Bundle: `SceneWmBundle` (frozen scene encoder + energy/act adapters).

| `op` | Body | Effect |
|------|------|--------|
| `load_scene` | `{ "path": "..." }` | Load pinned scene bundle; verify encoder fingerprint |
| `step` | `{ "scene": { nodes, edges, regime_stable } }` | Energy route / abstain / propose latent |
| `act` | `{ "scene": ..., "block_idx": 1 }` | Route → `plan_action` → discrete nudge |
| `fingerprint` | — | Encoder pin |
| `status` | — | Loaded path |

```bash
cargo run --release --bin growformer-demos -- --phase3w-scene-host
```

**Pin contract:** new session + same bundle file must return the same `encoder_fingerprint`. Chat is not a certifier.
