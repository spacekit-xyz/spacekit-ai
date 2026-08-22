# Continuum — Online / Continual Learning Spec

This document specifies **train-while-on** (online continual learning) for Growformer: inference-time feedback, in-memory training steps, and optional persistence. It is implementable with the current architecture (DimensionManager, Mirror/Main, router, language pipeline) and does not require Phase 4 (Evolver).

---

## 1. Goals

| Goal | Description |
|------|-------------|
| **Train while on** | Run one or a few training steps on live traffic when a feedback signal is present, without a full offline retrain. |
| **Feedback-driven** | Training signal comes from explicit feedback (e.g. thumbs up/down, user correction) or from inferred success (e.g. user accepted suggestion). |
| **Controlled persistence** | Optionally write an updated brain checkpoint to disk on a schedule or on demand, so that restarts pick up the evolved state. |
| **No regression** | Mitigate catastrophic forgetting via small learning rates, optional replay of prior examples, and (where applicable) existing Main/Mirror isolation. |

**Out of scope for this spec:** Phase 4 "Evolver" (optimizing own physics / hyperparameters). That remains a separate, future phase. This spec only covers **parameter updates from feedback**, not meta-learning of architecture or physics.

---

## 2. Current State

- **Implemented:** Full two-layer feedback pipeline. `submit_feedback` in `LanguageService` updates both the **neural network path** (LearnedRouter + GroupGenEnv training steps) and the **Paramecium lattice path** (multi-timescale quality/reliability via `apply_quality_feedback`, correction injection via `inject_correction`). The chat API carries optional `feedback` for the previous turn. A dedicated `POST /v1/feedback` endpoint is also available. Auto-checkpoint to configurable path fires every N feedback events (default 50). Rate limiting prevents update thrash.
- **Paramecium Foundation:** Multi-timescale `BehavioralProgram` model with persistent state (quality_score, reliability, total_retrievals), session-scoped state (session_drift, session_hits), and volatile state (activation_level, refractory). Session lifecycle: `begin_session` (on conversation reset) → `decay_activations` (between turns) → `consolidate_session` (on next reset, commits drift for well-used positive programs). `effective_centroid()` applies session drift for in-context adaptation. `retrieval_bias()` modulates program scoring based on quality, reliability, and refractory status. MetaCognition Accept/Degrade feedback wired to lattice programs during inference.
- **Correction Injection:** `inject_correction` on `InfraciliaryLattice` degrades the wrong program and either reinforces a nearby existing program (cosine ≥ 0.92) or spawns a new program with the correction text. New correction programs start with boosted quality (0.3) and reliability (0.7).
- **Configuration:** `ContinuumConfig` struct with configurable checkpoint_interval, min_consolidation_hits, rate_limit_per_minute, and checkpoint_path. Replaces hardcoded constants.
- **Server:** Performs inference on the loaded brain and applies Continuum feedback (both neural and lattice) when feedback is present. `POST /v1/brain/save` persists the evolved brain on demand.
- **Existing pieces:** Episodic memory, retention evals, replay buffers, Mirror/Main promotion, and LearnedRouter are present. All wired to live feedback via `submit_feedback`.

**Region binding (inference).** Generation and codegen heads are conditioned on `[raw_embedding; action_type_one_hot; group_one_hot]` so the **routed group** selects the correct region-specific attractor. Implemented: `group_id_one_hot` in `action_classifier`, training and inference append group one-hot when the head’s `cond_dim` indicates region dims. Older brains (cond_dim = raw+4) receive no group dims and remain backward compatible.

---

## 3. Components

### 3.1 Feedback signal

- **Source:** Caller provides a feedback payload with the request (or a follow-up request) that indicates success/failure or correction.
- **Representation (recommended):**
  - **Explicit:** e.g. `feedback: { outcome: "accept" \| "reject" \| "correct", correction?: string }`.
  - **Implicit:** e.g. "user edited the suggested code" → treat as weak positive or as correction (text) for that turn.
- **Scope:** Per turn or per conversation; design should support both (e.g. feedback tied to `turn_id` or last N turns).

### 3.2 Training trigger

- **When to run a training step:** Only when feedback is present and indicates a learning signal (e.g. reject, or accept with optional correction). Optionally gate on rate limiting (e.g. at most one step per minute per user/session) to avoid thrash.
- **What to update:** One or more of:
  - **Router** (LearnedRouter): e.g. one gradient step so that the chosen group's logit increases (on accept) or decreases (on reject), or use correction to infer target group.
  - **Action classifier / language routing:** same idea — reinforce or correct the routing decision.
  - **Generation head (GenHead / CodegenHead):** if feedback includes correction text, one or a few steps on that example (e.g. input text → target output).
- **Learning rate:** Small (e.g. 1e–4 to 1e–3) for online steps to limit forgetting and instability.

### 3.3 Where training runs

- **Process:** In the same process as the server (LanguageService / DimensionManager). No separate "trainer" process required for minimal design.
- **Threading:** Training step should be short (e.g. one minibatch or one step). Can run on a dedicated thread or a thread pool to avoid blocking the request that delivered feedback; the next inference will then see the updated state.

### 3.4 Persistence

- **When to write:** (a) On demand (e.g. admin API: "save brain now"), and/or (b) on a schedule (e.g. every N minutes or every M feedback events), and/or (c) on graceful shutdown.
- **What to write:** Same format as current checkpoint/brain export (e.g. `brain.bin`), so existing load path can load the continuum-evolved brain after restart.
- **Where to write:** Configurable path (e.g. same `GROWFORMER_BRAIN_DIR` or a dedicated "continuum" output path). Overwrite or versioned filenames (e.g. `brain_<timestamp>.bin`) is a product choice.

---

## 4. API (Suggested)

### 4.1 Chat request with feedback

Extend chat request body to carry optional feedback for the **previous** turn (or current turn, depending on UX):

```json
{
  "message": "…",
  "brain": "my-brain",
  "feedback": {
    "outcome": "accept" | "reject" | "correct",
    "correction": "optional string for correct"
  }
}
```

- If `feedback` is present and outcome is not `accept` (or outcome is `correct` with text), server enqueues or runs a training step for that turn's routing/generation, then responds to the current `message` as usual.

### 4.2 Dedicated feedback endpoint (alternative or addition)

- `POST /v1/feedback` with body: `{ "turn_id": "…", "outcome": "…", "correction": "…" }` so that feedback can be sent asynchronously from the client after the user acts.

### 4.3 Persistence control

- `POST /v1/brain/save` (or `PUT /v1/brain/checkpoint`): persist the current in-memory state of the active brain to disk.
- Optional query or body param: `brain=name` when multiple brains are loaded.

---

## 5. Data flow (minimal)

1. **Inference:** Request → `LanguageService` → route → action/generation/codegen → response. Store minimal turn context (e.g. input text, chosen group, action, response) for that turn if feedback might arrive later.
2. **Feedback:** Request includes `feedback` (or dedicated `/v1/feedback`). Resolve turn (e.g. last turn or by `turn_id`).
3. **Training step:** Build a single-example (or small batch) from turn + feedback: e.g. (input_embedding, target_group) for router; (input_text, correction_text) for generation head. Run one (or a few) steps with small LR. Update DimensionManager / LearnedRouter / heads in memory.
4. **Persistence:** When trigger fires (schedule or on-demand), call existing checkpoint export for the active brain and write to disk.

---

## 6. Anti-forgetting and safety

- **Learning rate:** Use a small LR for online steps (e.g. 1e–4–1e–3); document in spec and make configurable.
- **Replay (optional):** If a replay buffer of prior (input, target) pairs is maintained, mix one or a few replay examples into each online step to reduce catastrophic forgetting of routing or generation.
- **Frozen Main:** Do not update Main Dimension groups from online feedback; only update router, action routing, or Mirror/generation heads as designed. Main remains consolidated and frozen.
- **Rate limiting:** Cap frequency of online steps per user/session/brain to avoid runaway updates.

---

## 7. Implementation checklist

- [x] **Feedback type:** Define `Feedback` struct and include in chat request (and/or dedicated endpoint).
- [x] **Turn context:** Retain per-turn state (input, routing decision, response, effective_gidx, program_idx) for feedback association.
- [x] **Router update:** `submit_feedback` runs `CONTINUUM_STEPS` (3) training steps on the `LearnedRouter` with the turn's embedding and group_id when feedback is Reject or Correct.
- [x] **Head update:** When feedback includes `correction` text, `submit_feedback` temporarily unfreezes the target group's `GroupGenEnv`, runs `CONTINUUM_STEPS` train_steps with (embedding, correction), then restores frozen state.
- [x] **Lattice feedback:** Accept → `apply_quality_feedback(positive)` on selected program. Reject → `apply_quality_feedback(negative)`. Correct with text → `inject_correction` (degrade wrong program + spawn/reinforce correction program in lattice).
- [x] **Session lifecycle:** `reset_conversation` consolidates outgoing session → begins fresh session. `converse` decays activations between turns. Retrieval uses `effective_centroid` + `retrieval_bias`.
- [x] **Trigger:** In server, feedback present and outcome indicates learning invokes training steps synchronously. Dedicated `POST /v1/feedback` endpoint also available.
- [x] **Persistence:** `POST /v1/brain/save` for on-demand. Auto-checkpoint every `checkpoint_interval` feedback events to configurable path.
- [x] **Config:** `ContinuumConfig` struct with configurable checkpoint_interval, min_consolidation_hits, rate_limit_per_minute, checkpoint_path. Rate limiting via sliding 60-second window.

---

## 8. Relation to Phase 4 (Evolver)

- **Phase 4 (Open-Ended Evolver)** in the skill is defined as "Optimizes own physics" — i.e. meta-learning of architecture or physics parameters (e.g. growth_radius, learning_rate) from experience. That is a **different** direction from continuum.
- **Continuum (this spec)** is **parameter learning from feedback** within the current architecture: same physics, same Mirror/Main, same router/heads; only the way training is triggered (feedback) and persisted (on a schedule or on demand) changes.
- **Recommendation:** Implement continuum as a **separate track** (e.g. "Continuum" or "M7: Online learning") that does not depend on Phase 4. Phase 4 can later use continuum as one of its **signals** (e.g. use feedback and retention metrics to decide how to adjust physics). So:
  - **Continuum first:** train-while-on, feedback, persistence (this spec).
  - **Phase 4 later:** Evolver that may consume continuum feedback and metrics to optimize its own physics.

---

## 9. References

- **Skill:** `docs/GrowformerSkill/SKILL.md` — Phase 3 (Mirror, Main, router), M5 retention, M6 agent modes; Phase 4 (Evolver) as future.
- **Server:** `src/server.rs` — chat and brain endpoints; extend with feedback and save.
- **Service:** `src/service.rs` — `LanguageService`; add feedback handling and optional training step.
- **Checkpoint:** `systems/checkpoint.rs` (or equivalent) — reuse for continuum persistence.
