# Growformer AI Operating System — Technical Specification

**Status:** Target architecture. Agent smart contracts exist and are operational; Growformer integration is planned once specialized brains/agents are trained per domain.

**Architecture:** The AI OS specification describes a sound architecture: three product pillars (SpaceKit, RouteKit, NeuroKit), event-driven meta-controller, specialized brains, policy/syscall layer, failure and fallback (RouteKit), optional consensus, end-to-end flow, brain deployment via runtime + Storage Node + Fact Package, and contract–package correlation. Implementation can proceed against this spec.

**Related:** [Growformer README](../README.md). [ROADMAP](ROADMAP.md). Agent smart contracts: [spacekit-standard-library/agents](https://github.com/spacekit-xyz/spacekit-standard-library) (Agents & AI).

---

## Ecosystem (three pillars)

From a **product** standpoint the stack is three pillars: **SpaceKit** (infra), **RouteKit** (model router), **NeuroKit** (AI/ML). Growformer is the AI/ML engine within NeuroKit.

| Pillar | Role | Where |
|--------|------|-------|
| **SpaceKit** | **Infrastructure** — servers, compute/storage/messaging nodes, WASM orchestration, service mesh, smart contracts, execution layer. Policy and syscall-style tooling are part of this stack. | [spacekit-standard-library](https://github.com/spacekit-xyz/spacekit-standard-library), **spacekit-simulator** (testnet/simulator: orchestration, node scheduling, WASM registry, gRPC/HTTP gateway). |
| **RouteKit** | **Model router** — single endpoint, task/intent classification, provider selection, health-based failover, cost tracking. Can use **NeuroKit (Growformer)** for intent-based routing (route by intent → domain → model/brain). | RouteKit relay (swtch.ai); intent validation and forwarding to SpaceKit. |
| **NeuroKit** | **AI/ML** — multidimensional neural training (Growformer), GLE, routing and action/generation/codegen heads, specialized brains per domain. | This repo (growformer); neurokit. |

Agent smart contracts live in SpaceKit (standard library). RouteKit handles model routing and failover; NeuroKit (Growformer) can back intent routing and domain-specific inference once brains are trained.

---

## 1. Scope and definitions

### 1.1 Purpose

This document specifies the **Growformer AI Operating System**: a meta-controller architecture in which a top-level Growformer instance orchestrates a network of specialized Growformer agents, each trained and responsible for a specific domain. The specification ties that architecture to:

- **Growformer** — the substrate for both the meta-controller and each specialized “brain” (see [Growformer README](../README.md)).
- **SpaceKit agent smart contracts** — the current on-chain agent layer (identity, forum, inference, market). These contracts do **not** use Growformer today; they will integrate once domain-specific brains are trained and deployed.

### 1.2 Definitions

| Term | Definition |
|------|------------|
| **Meta-controller / Supervisor** | Top-level Growformer that monitors environment, routes to specialists, enforces policy, and evaluates outcomes. The “brain that manages other brains.” |
| **Specialized brain / Sub-agent** | A Growformer checkpoint (router, action classifier, generation/codegen heads, group layout) trained for a narrow domain. Invocable and suspendable by the supervisor. |
| **Agent smart contract** | WASM contract in `spacekit-standard-library/agents` that implements agent identity, inference, intent, or market. Currently backed by host primitives (e.g. `spacekit_llm`, `microgpt_forward`), not Growformer. |
| **Domain** | A bounded capability area (e.g. codegen, moderation, support, analytics) for which one specialized brain is trained and responsible. |

### 1.3 Current vs target state

| Aspect | Current | Target (this spec) |
|--------|---------|-------------------|
| **Agent contracts** | Implemented: identity, forum, moderation, agent (LLM), microgpt, intent classifier, inference market, inference mesh. | Unchanged as execution and identity layer; **inference** may be backed by Growformer where a domain brain exists. |
| **Inference** | `spacekit_llm` (generic LLM) or `microgpt_forward` (next-token). No Growformer. | Optional use of Growformer: GLE → routing → action/generation/codegen per domain brain. |
| **Orchestration** | None; contracts are invoked directly. | Meta-controller (Growformer) receives events, selects specialist, enforces policy, routes to tools. |
| **Specialized brains** | Growformer supports multiple checkpoints (e.g. `GROWFORMER_BRAIN_DIR`); not yet mapped to agent contracts or domains. | One or more trained brains per domain; each brain is the inference backend for that domain’s contract or endpoint. |

---

## 2. Reference components

### 2.1 Growformer (substrate)

As per [Growformer README](../README.md):

- **What it is:** A multidimensional neural training environment (six dimensions: weight, geometry, timing, metabolic cost, connectivity, structural group). Dynamic graph, GLE language front-end, 64-d routing, action/generation/codegen heads.
- **Runtime:** CLI, Node HTTP server, and WASM library (`--no-default-features`, `wasm32-unknown-unknown`). Single or multiple brains (checkpoints) loadable by name.
- **API surface:** `LanguageService`: `action()`, `generation()`, `codegen()`, brain load/set active, M6 agent modes (ContextFile, MicroBrain).
- **Relevance to AI OS:** The meta-controller and each specialized agent are Growformer instances (or the same runtime with different checkpoints). Policy, routing, and tool-calling are implemented on top of this substrate.

### 2.2 Agent smart contracts (execution layer)

Location: **[spacekit-standard-library/agents](https://github.com/spacekit-xyz/spacekit-standard-library)** (repo path `agents/`).

| Contract / crate | Role | Inference today | Future (Growformer) |
|------------------|------|------------------|----------------------|
| **spacekit-spacetime** | Identity (`register_agent`, `is_agent`, `AgentProfile`), Forum (agent-only), Moderation. | N/A | Identity and forum remain; agents may be bound to domain brains. |
| **spacekit-agent** | Chat, analyze, summarize, code review, classify. | `spacekit_llm` host. | When a “general” or per-op domain brain exists, inference can be delegated to Growformer (e.g. routing + action/generation). |
| **spacekit-agent-microgpt** | Next-token / chat step. | `microgpt_forward` host. | Can be replaced or complemented by a Growformer-backed endpoint (e.g. codegen brain). |
| **spacekit-intent-classifier** | Intent classification. | `spacekit_llm`. | Natural fit for a Growformer routing/action brain (intent → action type → domain). |
| **spacekit-agent-inference-market** | Agent pricing, job creation, result submission. | N/A | Same; jobs may reference agent_id → domain → brain. |
| **spacekit-inference-mesh** | Mesh / discovery. | N/A | Can expose which agents/domains are backed by which brains. |

**Important:** None of these contracts currently call or depend on Growformer. Integration will require either (a) host functions that invoke Growformer (e.g. in a VM/runtime that supports both contracts and Growformer WASM), or (b) off-chain orchestration that uses Growformer and then submits results/state via contract calls.

---

## 3. AI OS architecture (target)

### 3.1 Meta-controller (supervisor)

The top-level Growformer acts as the **kernel** of the AI OS:

- **Stateful:** Maintains environment and session state.
- **Policy-driven:** Traits, heuristics, risk profiles, communication style, adaptation rules (scheduler + security + personality layer).
- **Event-reactive:** Consumes events (user input, system events, tool results) and decides *which* specialist should act.
- **Agent-orchestrating:** Invokes or suspends specialized brains; no direct execution of arbitrary LLMs outside this model.
- **Tool-routing:** Agents call a **syscall layer** (not raw APIs); WASM tooling is the intended abstraction (Palantir AIP–style but with WASM).
- **Memory-aware:** Episodic and working memory; M6 ContextFile / MicroBrain modes and shared-state contract apply.

#### 3.1.1 Event model

Events are the input stream to the meta-controller. Sources and consumers:

| Source | Description | Payload (minimal) |
|--------|-------------|-------------------|
| **User / client** | Message, command, or request. | `message`, `session_id`, optional `task` hint (RouteKit-aligned). |
| **SpaceKit Messaging Node** | Intent status, agent notifications, market/event propagation. | `intent_id`, `status` (`accepted` \| `submitted` \| `executed` \| `failed`), `timestamp`, optional receipt. |
| **SpaceKit simulator / orchestration** | Node lifecycle, deployment, service mesh. | Deployment events, node health; wire format TBD in SpaceKit docs. |
| **Tool / syscall result** | Outcome of an agent or adapter call. | `call_id`, `success`, `result` or `error_code`. |

**Consumer:** The meta-controller subscribes to these streams (or receives them via a single event bus), classifies and routes, and decides which specialist should act. Event schema and topic naming (e.g. `intent/status/{id}`, `actor/{id}/intents`) are defined in SpaceKit protocol and Messaging Node specs; see RouteKit README and SPEC-GAPS-AND-BUILD-PLAN for Messaging Node responsibilities.

### 3.2 Specialized sub-agents (brains)

Each domain is served by a **specialized Growformer brain**:

- Narrow, well-trained capability (one domain or a small set of related intents).
- Sandboxed execution (WASM, existing contract boundaries).
- Own memory and state (per-checkpoint state in Growformer terms).
- Invoked or suspended by the supervisor (process model of an OS).

These are **not** the current agent contracts themselves; they are the **inference backends** (Growformer checkpoints) that will back contract operations once trained. The contracts define *what* agents do (identity, chat, classify, jobs); Growformer brains define *how* they reason for their domain.

### 3.3 Policy and personality engine

- **Traits, heuristics, risk profiles, communication styles, adaptation rules.**
- Implemented above the Growformer substrate (e.g. in the meta-controller’s decision loop) and in the **SpaceKit infrastructure** layer (see [Ecosystem](#ecosystem-three-pillars)).
- **SpaceKit infra** (e.g. **spacekit-simulator**): orchestration, node scheduling, service mesh, WASM registry, key manager, port allocation. This stack provides the server-side environment where policy constraints and node lifecycle are enforced. Policy rules (who may invoke which agent, resource limits) can live in simulator/orchestrator config or in contracts (e.g. agent-scope). Maps to: OS scheduler + security subsystem + userland personality.

### 3.4 Tooling / syscall layer

- Agents do not call external APIs directly; they call a **syscall layer** owned by the platform.
- **SpaceKit infra** provides this in practice: WASM contracts use host functions (e.g. `spacekit_llm`, `spacekit_storage`); the simulator and Compute Node expose these. Target: WASM-based, OS-like (Palantir AIP–style but with WASM). Full syscall API surface is defined by SpaceKit VM and standard library host modules.

### 3.5 Distributed execution

- Agents can run locally or remotely; tasks can move across nodes; state can be maintained across environments.
- Contracts (identity, forum, market, mesh) provide the durable and discoverable layer; Growformer provides the reasoning layer.

### 3.6 Failure and fallback (RouteKit)

**RouteKit** is the model router that selects provider and model by task/intent, with automatic failover:

- **Intent-based routing:** Task types (chat, search, summarize, classify, code_review, analyze, status) align with spacekit-agent-microgpt and Kit agent ops. RouteKit can use **Growformer** for intent-based routing: route by intent → domain → model or domain brain.
- **Provider health:** Health graph (latency, error rate, traffic weight); traffic shifted on degradation; full failover when a provider is down.
- **Fallback when no Growformer brain:** If no specialized brain is trained for a domain, or routing confidence is low, RouteKit falls back to generic LLM (or microgpt) per existing config. Contracts continue to use `spacekit_llm` / `microgpt_forward` until a domain brain is registered and wired.
- **Relay unavailable:** Clients may submit eligible intents directly to chain (see SpaceKit/RouteKit architecture); bridge or match-required intents still require the relay.

So: **RouteKit** owns model routing, intent validation, and failover; **Growformer** (when integrated) backs intent→domain routing and domain-specific inference; **SpaceKit** owns execution and contracts.

### 3.7 End-to-end flow

1. **User or client** sends a message/request (optional: with task hint from microgpt or RouteKit).
2. **Event** is emitted (user input and/or Messaging Node intent status, etc.) and delivered to the **meta-controller** (supervisor Growformer or orchestration layer).
3. **Meta-controller** (optionally using **RouteKit** and/or **Growformer** for intent routing) decides **which domain** and **which specialist** should handle the request.
4. **RouteKit** (if in path): classifies task/intent, selects provider/model (or domain brain when Growformer is wired), handles failover; streams completion or forwards **SignedIntent** to SpaceKit.
5. **Specialist** (Growformer brain for that domain, or generic LLM via RouteKit): produces action, generation, or codegen; may call **syscall layer** (SpaceKit host functions / tools).
6. **Tool result** or **completion** is returned; optionally an **Intent** is composed, signed, and submitted via RouteKit → SpaceKit Compute Node (or direct to chain when relay is unavailable).
7. **SpaceKit** executes contracts (identity, forum, inference market, etc.); **Messaging Node** may push intent status and agent notifications back to clients and meta-controller.

```
User/Client → Event → Meta-controller → RouteKit (intent/model) → Specialist (Growformer or LLM)
       → Syscall/tools → Result / SignedIntent → RouteKit → SpaceKit (contracts) → Messaging Node
```

### 3.8 Consensus within Growformer (optional)

**Within** the Growformer / AI OS stack (not at the blockchain layer), **consensus** means agreeing on a result or action when multiple specialists, simulation runs, or event sources contribute. It is **not superfluous** when the system must be robust or auditable; it is **optional** so the default path remains single-specialist, single-run.

| Use case | Description |
|----------|-------------|
| **Multi-specialist agreement** | The meta-controller invokes two or more specialists (or the same brain with different seeds/temperatures); results are aggregated (e.g. majority vote, weighted blend, or “act only if N agree”). Fits high-stakes or safety-critical decisions. |
| **Simulation consensus** | Multiple rollouts or simulations (e.g. planning, world-model “what if”) produce outcomes; the policy layer chooses or agrees on one (e.g. majority outcome, highest value, or escalate if disagreement). Aligns with World Models (planning) and Continuum (outcome feedback). |
| **Event/source reconciliation** | Conflicting events or observations (e.g. from different tools or agents) are reconciled before the meta-controller commits an action; policy defines how to resolve (e.g. prefer source, require agreement, or defer). |

**Where it lives:** Consensus is a **policy option** implemented in or above the meta-controller: it decides *when* to run multiple specialists or simulations, collects outputs, and applies an aggregation rule (voting, blending, threshold). It does not replace single-specialist routing; it extends the flow when policy requires it (e.g. “for class X of intents, run three specialists and act only if at least two agree”). VirtualGroup already blends frozen specialists with learned weights; consensus is the complementary case where the system explicitly requires **agreement** (or a defined disagreement protocol) before committing.

---

## 4. Mapping to OS concepts

| OS concept | AI OS component |
|------------|------------------|
| **Kernel** | Supervisor Growformer (meta-controller). |
| **Process model** | Specialized brains: one “process” per domain brain, lifecycle controlled by supervisor. |
| **Scheduler** | Policy + personality engine (who runs when, with what constraints). |
| **Memory manager** | Growformer episodic/working memory + M6 shared-state contract; optional persistence. |
| **Syscall layer** | Tooling / syscall abstraction (WASM); agents call this, not raw APIs. |
| **Userland** | Agent smart contracts (identity, forum, inference, market) + application logic. |

---

## 5. Integration path: contracts ↔ Growformer

1. **Keep existing contracts.** Identity, forum, moderation, inference market, and mesh stay as the canonical agent and job layer.
2. **Train and register domain brains.** For each domain (e.g. codegen, intent classification, support, moderation), train a Growformer checkpoint (router, action, generation/codegen as needed) and register it (e.g. brain name, domain id, capabilities).
3. **Expose Growformer to the runtime.** In environments where both contracts and Growformer run:
   - Add host functions or an orchestration service that can run Growformer inference (GLE → routing → action/generation/codegen) for a given brain name.
   - Optionally: a “router” contract or endpoint that accepts input, calls Growformer to choose domain and action, then invokes the appropriate specialist or returns a result.
4. **Wire contracts to brains.** For each agent contract that today uses `spacekit_llm` or `microgpt_forward`:
   - Define which domains it serves (e.g. “general”, “codegen”, “intent”).
   - When a domain has a trained brain, route that contract’s inference to Growformer for that domain instead of (or in addition to) the generic LLM/microgpt.
5. **Meta-controller.** Introduce a top-level Growformer or equivalent process that subscribes to events, uses policy to select a domain/specialist, and invokes the right brain or contract. This can be off-chain first (orchestrator service) and later reflected on-chain via contracts where needed.

**Concrete stack:** **SpaceKit** = infra (simulator, nodes, contracts, syscall/host layer). **RouteKit** = model router with intent (task classification, provider health, failover; optional Growformer for intent→domain routing). **Growformer** = AI/ML in NeuroKit (domain brains). Integration: events from clients and Messaging Node → meta-controller; RouteKit handles model/brain selection and fallback; SpaceKit executes and persists.

#### 5.1 Brain deployment: contract vs runtime

Where does `brain.bin` live, and where does inference run?

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| **A. Embed in contract** | Bundle the checkpoint inside the agent contract WASM; inference runs inside the contract. | Self-contained; no external oracle. | Contract size and deploy cost scale with brain size; updating brain = redeploy; multiple brains = multiple or very large contracts; WASM memory/cost limits for larger brains. |
| **B. Load at runtime inside contract** | Contract fetches brain (e.g. from storage) and runs inference in-contract. | Brain not in bytecode. | Loading/parsing large checkpoints in WASM is heavy and memory-limited; same size/cost constraints as A for the actual inference. |
| **C. Inference in runtime (recommended)** | Contract calls a host function (e.g. `growformer_inference(brain_id, op, input)`); the **runtime** holds or resolves brain checkpoints and runs NeuroKit (Growformer) inference outside the contract sandbox; result returned to the contract. Same pattern as `spacekit_llm` / `llm_agent`. | Contract stays small; brains versioned and updated independently; one contract can invoke many brains by id/name; matches existing LLM oracle pattern; heavy compute in runtime (optimization, caching). | Contract depends on runtime to provide the brain; inference is an oracle (same trust model as current LLM agent). |

**Recommendation:** **C.** Do not embed `brain.bin` in the agent smart contract; do not run full inference inside the contract. The contract should **point to a brain** (by id or name); the **runtime** (Compute Node, simulator, or VM host) owns or resolves checkpoints and runs inference, returning results to the contract. This aligns with `llm_agent` and keeps contracts deployable while allowing many brains and updates without redeploying contracts.

**Brain storage:** `brain.bin` (and other NeuroKit checkpoints) can be stored on the **SpaceKit Storage Node** — the p2p content-addressed storage layer (spacekit-storage-node). The runtime resolves `brain_id` (or a content-addressed reference) by fetching the checkpoint from the Storage Node when needed; optionally the runtime caches locally after first fetch. This keeps brains off the contract, versionable and distributable via the same infra used for contract state, manifests, and other artifacts.

**Fact Package:** A brain checkpoint can be distributed as part of a **Fact Package** — a mechanism that lets a developer package content or data (e.g. models, configs, assets). A Fact Package may include one or more `brain.bin` artifacts plus metadata (brain name, domain, version); the runtime or deployment pipeline can consume Fact Packages to register and resolve brains from Storage Node or from a package manifest.

**Contract ↔ Fact Package:** The smart contract can hold a **correlation** to the Fact Package: e.g. an agent or brain registry contract stores a reference (content-addressed id, package manifest uri, or Fact Package id) so that the runtime knows which package to fetch for a given `brain_id` or agent. On-chain state then records which Fact Package (and thus which brain version) an agent or domain uses; updates are done by pointing the contract to a new package reference rather than redeploying the contract.

**Constraint:** Agent smart contracts do **not** use Growformer until the specialized brains for their domains are trained and the integration path above is implemented. Until then, contracts continue to use existing host primitives (`spacekit_llm`, `microgpt_forward`).

---

## 6. Why this is a differentiator (non-normative)

Most “AI agent frameworks” are stateless prompt wrappers, single-agent loops, or cron-like jobs with no policy engine, event system, or real autonomy. This architecture provides:

- **WASM sandboxing** (contracts + Growformer WASM).
- **AI smart contracts** (identity, forum, market, inference) as the execution and economic layer.
- **Persistent agent state** (Growformer checkpoints + episodic memory + contracts).
- **Event-driven scheduling** (meta-controller reacts to events and routes to specialists).
- **Multi-agent coordination** (supervisor + domain brains).
- **Distributed execution** (local/remote, multi-node, state across environments).

One-sentence summary:

> **A meta-agent that acts as an operating system for a network of specialized AI agents — a brain that manages other brains — each with its own capabilities, memory, and policies, backed by Growformer and executed via agent smart contracts.**

---

## 7. Open work (non-normative)

| Item | Owner | Notes |
|------|-------|-------|
| Name the top-level brain (meta-controller product name) | TBD | — |
| Define its responsibilities (RACI or capability matrix) | TBD | — |
| Refine OS mapping for implementation docs | TBD | kernel, scheduler, memory, syscall |
| Policy/syscall API surface in SpaceKit docs | TBD | spacekit-simulator, host modules |
| Consensus policy (when to require multi-specialist or multi-run agreement) | TBD | §3.8; optional within Growformer |
| Positioning (enterprise, government, developer) | TBD | — |
| Announcement copy (swtch.ai Growformer “AI OS”) | TBD | — |

---

## 8. References

- **Growformer:** [README](../README.md) — multidimensional neural environment, GLE, six dimensions, WASM, multiple brains, M6 agent modes.
- **Roadmap:** [ROADMAP](ROADMAP.md) — AI Operating System as future phase.
- **SpaceKit Standard Library (agents):** [spacekit-standard-library](https://github.com/spacekit-xyz/spacekit-standard-library) — `agents/`: SpaceTimeIdentity, SpaceTimeForum, SpaceTimeModeration, SpacekitAgent, spacekit-agent-microgpt, SpacekitIntentClassifier, SpaceKitInferenceMarket, SpaceKitInferenceMesh.
- **SpaceKit infra:** spacekit-simulator — orchestration, service mesh, WASM registry, node scheduling (servers/testnet).
- **RouteKit:** Model router (intent, provider health, failover); can use Growformer for intent-based routing; intent validation and forwarding to SpaceKit.
