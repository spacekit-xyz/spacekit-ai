# Growformer World Models — Technical Specification

**Status:** Target architecture with a certified toy substrate (Phase 3i). This document specifies how the concept of a *world model* maps onto the Growformer architecture and the AI Operating System. A full application-level world-model layer remains future work; JEPA-like adapters are the promotable *predictive content* under continual learning.

**Related:** [Growformer README](../README.md). [Growformer Whitepaper](GROWFORMER_WHITEPAPER.md) (DSPNS, Mirror/Main, authenticated routing, Continuum). [JEPA_ADAPTER_PROMOTION.md](JEPA_ADAPTER_PROMOTION.md) (encoder pin + predictor promote contract). [COMPETENCE_ROUTING_SPEC.md](COMPETENCE_ROUTING_SPEC.md) (cone / Task E certifiers). [AI Operating System](AI_OPERATING_SYSTEM.md). [GROWFORMER_CAUSAL_AI.md](../GROWFORMER_CAUSAL_AI.md) (Layer 0 grounding).

---

## 1. Scope and definitions

### 1.1 Purpose

This specification defines what a **world model** is in the context of the Growformer stack and the AI OS, and how the existing architecture provides the substrate for world-model capabilities. It does not specify a separate monolithic “world model module”; it specifies the *mapping* from world-model requirements to Growformer and AI OS components — including **frozen JEPA-like encoders** and **promoted predictor adapters**.

### 1.2 Definitions

| Term | Definition |
|------|-------------|
| **World model** | An internal simulation of reality that an AI uses to understand its environment, predict outcomes, and plan actions. It is the layer that turns agents from reactive tools into autonomous decision-makers. |
| **DSPNS** | Dynamically Structured Physical Neural System (Whitepaper §3.1): physics-determined topology, metabolically-constrained plasticity, consolidation-based continual learning, intrinsic-dimensionality-aware self-organization. |
| **Mirror Dimension** | Isolated neural environment for training one task; no consolidated knowledge; promotes to Main upon criterion (Whitepaper §3.3). |
| **Main Dimension** | Frozen consolidated knowledge store; receives promoted groups only; no gradient path from Mirror (Whitepaper §3.3). |
| **VirtualGroup** | Global scalar blend of frozen specialist *outputs*. Useful as a **baseline / floor**, not as the planning mechanism for region- or regime-switched tasks (Whitepaper §4.3.1 negative). |
| **JEPA-like encoder** | Frozen sensory front-end mapping observations → latents; hash-pinned; never trains after init ([JEPA_ADAPTER_PROMOTION.md](JEPA_ADAPTER_PROMOTION.md)). |
| **Predictor adapter** | Promotable next-latent (and affinity) head trained in Mirror on one dynamics regime; consolidated into Main under the encoder pin. |

### 1.3 Normative vs non-normative

- **Normative:** Sections 2–4, and §3.2 (JEPA adapter mapping + promotion contract pointer).
- **Non-normative:** Sections 5–6 (motivation, roadmap, audiences).

---

## 2. World-model requirements (capability set)

A world model, as used in robotics, reinforcement learning, autonomous systems, and AGI-oriented research, is expected to support the following. This list is the *requirement set* against which the architecture is mapped.

| ID | Capability | Description |
|----|------------|-------------|
| **WM-1** | **Spatial understanding** | Objects, geometry, motion, physical constraints. |
| **WM-2** | **Temporal understanding** | What happens next, what is likely, what is possible. |
| **WM-3** | **Causal structure** | If I do X, Y will happen (cause–effect). |
| **WM-4** | **Agent modeling** | Goals, intentions, and behaviors of other agents. |
| **WM-5** | **Long-term memory** | Persistent state beyond short context windows. |
| **WM-6** | **Planning ability** | Simulating multiple futures and choosing the best course of action. |

**Note:** LLMs excel at text but do not, by default, provide WM-1–WM-6 in a structured, actionable form. A world model is the missing layer that enables perceive → remember → reason → predict → plan → act in a coherent system.

---

## 3. Mapping: world-model capabilities → Growformer

The Growformer architecture (see [GROWFORMER_WHITEPAPER.md](GROWFORMER_WHITEPAPER.md)) provides the following mappings. These are architectural commitments: the specified components are the *locus* for the corresponding world-model capability.

| World-model capability | Growformer component | Whitepaper / implementation reference |
|------------------------|----------------------|----------------------------------------|
| **WM-5 Long-term memory** | **Main Dimension** (frozen consolidated groups); episodic storage (M6, Continuum). | §3.3 Mirror/Main; §5.4 deployment; README M6. |
| **WM-2 Temporal / WM-3 Causal** | **Frozen encoder + promoted predictors** for latent next-step / action-conditioned prediction; **authenticated router** (cone-class) selects which predictor applies. Language specialists remain for text-side “what if” within their engrams. | [JEPA_ADAPTER_PROMOTION.md](JEPA_ADAPTER_PROMOTION.md); Phase 3i (`--phase3i-jepa-wm`); Whitepaper §4.3.1 routing certifiers. |
| **WM-6 Planning** | **AI OS policy engine** runs short **latent rollouts** over the *routed* predictor(s). **Not** global VirtualGroup blending for switched regimes. Continuum feedback for outcome-based adaptation. | AI_OPERATING_SYSTEM §3.3; Whitepaper VG negative §4.3.1; §3.2 below. |
| **WM-4 Agent modeling** | **Multi-brain / multi-agent:** each specialized brain is a module; the meta-controller reasons about which agent should act. | README multiple brains; AI_OPERATING_SYSTEM §3.1–3.2. |
| **WM-1 Spatial** | **Primary:** JEPA / latent dynamics over observations (Phase 3i toy). **Explicit scene graphs:** Phase 3v (`--phase3v-scene-wm`) — objects + typed edges, frozen scene encoder, structure ablation. **Secondary:** network geometry (neurons in space) is substrate. | `jepa_adapters.rs`, `wm_scene.rs`; Whitepaper §3.1. |

**Summary:** Long-term memory is Main + checkpoints. Temporal / spatial prediction is **encoder (frozen) + predictor adapters (promoted)** with **authenticated dispatch**. Planning is **policy-layer rollouts**, not VG. Agent modeling is multi-brain OS. Layer 0 grounding (§3.1) complements language inference; it does not replace predictive specialists.

### 3.1 Declarative grounding graph (Layer 0, planned)

The tables above describe **where neural consolidation and orchestration** carry world-model-like behavior. Separately, a **small typed concept graph** (see [GROWFORMER_CAUSAL_AI.md § World grounding](../GROWFORMER_CAUSAL_AI.md#world-grounding-relational-knowledge-beneath-labeled-sentences)) is planned as **inference-time structure**: traverse or expand a few nodes to **enrich the query** before the program lattice and retrieval stack run.

**MVP in tree:** [`data/inference/grounding_expand.toml`](../data/inference/grounding_expand.toml) and `growformer::inference::grounding_expand` implement **rule-based keyword expansion** on the lattice shortcut path—an incremental step toward the graph, not the graph itself.

This is **not** a full world model: it does not satisfy WM-1–WM-6 by itself, and it is not a substitute for Main Dimension or JEPA predictors. It **reduces the gap** between raw text co-occurrence and **auditable relational structure** in language—most closely **adjacent to WM-3**. It belongs in the **language inference path**, versioned like other inference assets. Layer 0 and JEPA adapters are **complementary**, not substitutes.

### 3.2 JEPA / predictor adapters under continual learning (Phase 3i)

**Contract.** See [JEPA_ADAPTER_PROMOTION.md](JEPA_ADAPTER_PROMOTION.md): freeze and hash-pin the encoder; train predictor adapters only in Mirror; promote adapters into Main; no gradient into the encoder or other promoted predictors.

**Toy (implemented):** two dynamics regimes (inner rotation vs outer radial expand), balanced composite, adjustable-cone router on affinity scalars, certifiers = regime agreement / degeneracy / MSE vs VG and confidence-argmax floors.

```
Observation → Frozen_JEPA_encoder → z
z → Authenticated_router → Predictor_A | Predictor_B  (promoted)
selected predictor → AI_OS_planner (latent rollouts) → act
```

**Reproduce:** `cargo run --release --bin growformer-demos -- --phase3i-jepa-wm`  
**Artifact:** [`phase3i_jepa_wm_results.txt`](../phase3i_jepa_wm_results.txt).  
**Code:** [`src/dimension/jepa_adapters.rs`](../src/dimension/jepa_adapters.rs).

**What this does *not* claim:** full AMI/JEPA scale training; Luna chat as a JEPA surface; replacing Main with one mega world model; beating Task E language/geometry specialists by importing video models.

### 3.2.1 Energy-based JEPA adapters (Phase 3j)

Predictive specialists become **promoted energy landscapes** \(E_\theta(z, z')\) (EB-JEPA-style),
plus proposal + affinity heads. Planning = prefer low-energy successors; wide cone ≈ flat /
competing basins; regime separation certified by energy margin as well as regime agreement.

```
Observation → Frozen_encoder → z
z → Cone_router(affinity) → EnergyAdapter_A | EnergyAdapter_B
selected E → propose / score z' → AI_OS rollouts (descent on E)
```

**Reproduce:** `cargo run --release --bin growformer-demos -- --phase3j-energy-wm`  
**Artifact:** [`phase3j_energy_wm_results.txt`](../phase3j_energy_wm_results.txt).  
**Code:** [`src/dimension/energy_jepa.rs`](../src/dimension/energy_jepa.rs).  
**Contract:** [JEPA_ADAPTER_PROMOTION.md](JEPA_ADAPTER_PROMOTION.md) §8.

Not metabolic `energy_budget`. Geometric / probabilistic / neuro-symbolic successors should
**wrap this energy object**, not replace the promote–freeze contract.

### 3.3 Post–Task E composition rule

Whitepaper §4.3.1 showed global **VirtualGroup** fails on balanced switched tasks and that unauthenticated lattice routing can look accurate while collapsing. For world-model composition:

| Approach | Role |
| --- | --- |
| VirtualGroup / confidence argmax | **Baselines / floors** only |
| Adjustable-cone (or successor authenticated router) | **Dispatch** among frozen predictors / specialists |
| AI OS policy latent rollouts | **Planning** (WM-6) over the selected dynamics |

Do not map WM-6 to “blend all world-model heads with one global weight vector.”

---

## 4. Mapping: world model → AI Operating System

The AI OS (see [AI_OPERATING_SYSTEM.md](AI_OPERATING_SYSTEM.md)) is the execution and orchestration layer. The following table aligns world-model roles with AI OS components.

| World-model role | AI OS component | Spec reference |
|------------------|-----------------|----------------|
| **Executive / controller** | **Meta-controller (supervisor)** — decides which specialist / predictor acts, enforces policy, consumes events. | AI_OPERATING_SYSTEM §3.1. |
| **Modules in the world model** | **Specialized sub-agents (brains)** + **promoted predictor adapters** under a pinned encoder. | AI_OPERATING_SYSTEM §3.2; JEPA_ADAPTER_PROMOTION.md. |
| **Training / simulation ground** | **Simulation environment** — SpaceKit simulator, Mirror Dimensions (training), deployment (inference-only). | Whitepaper §3.3; AI OS Ecosystem. |
| **Planner** | **Policy engine** — short latent rollouts / heuristics / risk profiles; who runs when. | AI_OPERATING_SYSTEM §3.3; §3.3 above. |
| **Actuator interface** | **Syscall / tooling layer**. | AI_OPERATING_SYSTEM §3.4. |
| **Persistent world state** | **Memory layer** — episodic/working memory, M6, Main Dimension, checkpoints. | AI_OPERATING_SYSTEM §4; README M6. |

So: the **supervisor** is the executive; **specialized agents + predictors** are modules; the **simulation environment** is the training ground; the **policy engine** is the planner (rollouts, not VG); the **syscall layer** is the actuator interface; the **memory layer** is persistent world state.

---

## 5. Context and motivation (non-normative)

### 5.1 Why world models matter

Large language models are highly capable at text but do not inherently model physics, objects, cause–effect, or long-horizon planning. A **world model** is an internal simulation of the world that an AI uses to predict, plan, and act (e.g. LeCun/AMI Labs: perceive, remember, reason, predict, plan, act).

### 5.2 Why this matters for this architecture

The stack already provides:

- A **meta-brain** managing **specialized sub-brains**.
- **Promote–freeze continual learning** with authenticated routing (Phases 3g–3i).
- **State, memory, and policies** per agent; **WASM** / **SpaceKit** deployment.

Growformer supplies consolidation and honest dispatch; the AI OS supplies orchestration, policy rollouts, and tools. Together they support WM-1–WM-6 as mapped in §2–§4 without reintroducing a shared, continually fine-tuned backbone.

---

## 6. Future work and audiences (non-normative)

| Area | Description |
|------|-------------|
| **3k–3m** | Geometric / probabilistic / neuro-symbolic energy — **implemented** (see §10.8). |
| **3n Action energy** | **Implemented** (`--phase3n-action-wm`): \(E(z,a,z')\) + rollout planner. |
| **3o Composed stack** | **Implemented** (`--phase3o-compose-wm`): geo + ensemble + rules. |
| **3p Harder transfer** | **Implemented** (`--phase3p-hard-wm`): 8D obs, 3 regimes. |
| **3q Deploy contract** | **Implemented** (`--phase3q-deploy-wm`): JSON pin + `deploy_step` (not Luna). |
| **3r Beyond-toy** | **Implemented** (`--phase3r-beyond-toy`): close E-rank, bounce+central foreign domains, frozen encoder file, sim JSONL. |
| **3s Open ladder** | **Implemented** (`--phase3s-open-ladder`): offline-pretrained frozen vision encoder (D slot), visuomotor push–object log (C), SpaceKit `deploy_step` host (E). Real V-JEPA weight swap still optional. |
| **3t Act surface** | **Implemented** (`--phase3t-act-wm`): WM agents that **act** (disk + visuomotor); task return vs random/VG; acting host; not Luna/chat. |
| **3u V-JEPA bridge** | **Implemented** (`--phase3u-vjepa-wm`): pinned export bank + frozen student (`scripts/export_vjepa_features.py --mode hf|mock`); adapters only. |
| **3v Scene-graph WM** | **Implemented** (`--phase3v-scene-wm`): table+blocks scene graphs, typed edges, frozen encoder, energy/act adapters, structure ablation kill-gate; not chat. |
| **3w Scene host** | **Implemented** (`--phase3w-scene-host`): SpaceKit-callable `SceneHostSession` (`load_scene` / `step` / `act` / reload pin) over `SceneWmBundle`. |
| **Beyond-toy proof** | See §8 — A–F green; D strengthened by 3u export path; WM-1 scene rung via 3v; deploy via 3w. |
| **Spatial / SpaceTime** | Scene-graph + host green (3v/3w); SpaceKit stdio glue (`--wm-host-stdio`); still not chat-first. |
| **Layer 0 concept graph** | **Certifier scaffold** (`--layer0-concept-graph`): typed BFS expand + depth structure; complements energy ([GROWFORMER_CAUSAL_AI.md](../GROWFORMER_CAUSAL_AI.md)). |
| **Context-free MNIST / CIFAR** | **4a/4b** + **4d** full 5-task multi-seed CF (`--phase4d-cf-mnist-full`). **4c** synthetic Split-CIFAR smoke; **4e** gray CIFAR-10 lite; **4f PASS 10/10** frozen patch bank + input-only cosine k-NN router (`--phase4f-split-cifar-frozen`; 41% router-free, 68.2% task accuracy, zero forgetting). |
| **DM WM citizens** | **5a** Preview+ (`--phase5a-wm-dm`): citizens + sidecar + DM checkpoint roundtrip. |
| **brain.bin WM** | **5e** (`--phase5e-wm-brain`): language specialists + WM citizens in one BrainPackage. |
| **Product act-loop** | **5b** (`--phase5b-product-act`): disk return ship metric; visuomotor diagnostic. |
| **External product loop** | **5c** (`--phase5c-external-product`): DM + return + SpaceKit pin; chat non-certifier. |
| **Live SpaceKit episode** | **5f** (`--phase5f-live-spacekit`): multi-step ActingHostSession; return + pin. |
| **D′ lite vision JEPA** | **5d** (`--phase5d-vjepa-vision`): frozen vision teacher export (not mock); HF optional. |
| **D′ real-log JEPA** | **5g** (`--phase5g-vjepa-real-log`): logged frames → frozen export → adapters; HF via `--mode hf --log`. |
| **Full AMI / large JEPA** | **Deferred** until HF V-JEPA-at-scale and DM citizen path stay green. |
| **V-JEPA smoke** | `scripts/smoke_vjepa_export.sh` (mock CI; `VJEPA_MODE=hf` optional). |

### 8. Growing beyond the toy — proving the energy substrate

Toys (3i–3q) show the *shape* of the claim. A real proof needs **transfer under the same honesty protocol**, not more acronyms.

**Keep fixed (non-negotiable)**

1. Encoder frozen + hash-pinned; only adapters promote ([JEPA_ADAPTER_PROMOTION.md](JEPA_ADAPTER_PROMOTION.md)).
2. Certifiers: regime/regime-class agreement, degeneracy, energy margin, MSE vs VG — **never** composite score alone.
3. Planning = descent / low-energy rollouts on the *routed* landscape — not VirtualGroup blend.
4. Metabolic synapse `energy_budget` ≠ latent \(E(z,a,z')\).

**Ladder (do in order)**

| Rung | What to add | Kill if | Status |
| --- | --- | --- | --- |
| A. Action + compose | 3n + 3o stay green on new seeds | Plan ≤ random; regime ≤55% | **Green** (3n stretch closed via linear planning-rank head + CE; see 3r) |
| B. Harder synthetic | 3p-class: ≥3 regimes, higher-D, partial observability | Regime → chance; margin ≤0 | **Green** (`--phase3p-hard-wm`) |
| C. External dynamics | Physics toy / game / robot log **not** authored for Growformer | Needs regime labels at train → fail contract | **Green** — bounce+central (3r) + **visuomotor push–object camera log** (3s); regime labels eval-only. |
| D. Frozen real encoder | Replace synthetic encoder with pretrained JEPA/V-JEPA **weights frozen**; train only energy adapters | Fine-tuning backbone “a little” | **Green** — `data/wm/vjepa_export_v1.json` via `export_vjepa_features.py` (`--mode hf` = Meta V-JEPA 2; `--mode mock` = CI). Student/projector/backbone never trained in adapter loop (`--phase3u-vjepa-wm`). |
| E. Deploy loop | `deploy_step` in AI OS / SpaceKit sim; log energy, route, abstain | Pin drift; silent encoder update | **Green** — local sim (3r) + SpaceKit host protocol / `wm_deploy_host` (3s); see [WM_SPACEKIT_HOST.md](WM_SPACEKIT_HOST.md). |
| F. Product surface | Only after D–E: tools/agents that *act*; still not Luna chat as WM proof | Using chat accuracy as WM metric | **Green** (`--phase3t-act-wm`) — acting agents on disk + visuomotor; return vs random/VG; `ActingHostSession` JSON act; chat explicitly non-certifier. |

**What would count as proof**

- Same promote–freeze + certifiers on **at least two** external domains (e.g. simple physics + one visuomotor log).
- Action-conditioned energy improves **task return / goal score** vs random and vs VG, with regime agreement holding.
- Deployed bundle: fingerprint stable across process restart; abstain enriched near boundaries.
- Negative controls published: VG floor, constant-specialist degeneracy, leaking \(r\)/labels into the loss.

**What would not count**

- Beating the toy with more capacity on the same spiral/circles construction.
- Chat / Luna fingerprint wins.
- Claiming AMI/LeCun parity without D–E.
- Energy margin only on in-distribution held-out from the same generator.

---

## 7. References

- **JEPA adapter promotion:** [JEPA_ADAPTER_PROMOTION.md](JEPA_ADAPTER_PROMOTION.md).
- **Growformer Whitepaper:** [GROWFORMER_WHITEPAPER.md](GROWFORMER_WHITEPAPER.md).
- **Competence / cone routing:** [COMPETENCE_ROUTING_SPEC.md](COMPETENCE_ROUTING_SPEC.md).
- **Growformer README:** [README](../README.md).
- **Growformer Causal AI / Layer 0:** [GROWFORMER_CAUSAL_AI.md](../GROWFORMER_CAUSAL_AI.md).
- **AI Operating System:** [AI_OPERATING_SYSTEM.md](AI_OPERATING_SYSTEM.md).
