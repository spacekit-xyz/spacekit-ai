# Growformer Language Understanding and Generation [COMPLETE]

We can add language to Growformer without breaking the current architecture by treating it as **a front-end perception/action layer** over the existing continual-learning core.

## Implementation Status Snapshot [COMPLETE]
M = Milestone; 
- M1 (Language Embedding Foundation): complete
- M2 (Embedding-First Routing Validation): complete
- M3 (Intent-to-Action Layer): complete
  - Verified by `--validate-action-schema` gate on Stage A+B datasets (baseline and extended)
- M4 (Controlled Language Generation, template-only): complete
  - Verified by `--validate-generation` gate on Stage A+B extended dataset
- M5 (Continual Language Learning Integration): complete
  - Sequential retention training with real M5Learner (not mocked)
  - Domains: Python, Rust, JavaScript coding + design patterns + architectural patterns + Stage C multi-turn + Stage D adversarial
  - Replay buffer with anti-forgetting (configurable `--m5-replay-per-epoch`, `--m5-replay-prior-ratio`)
  - Full retention plan manifest: `data/language/m5/retention_eval_splits_full.json` (7 phases)
  - Automatic Mirror spawn trigger: K=10 consecutive low-confidence routing batches (`DimensionManager::track_confidence_for_auto_spawn`)
  - Domain taxonomy routing: task function, policy regime, language channel axes
  - Verified by `--m5-retention-eval` with mean_retention_ratio=1.000 across all 7 domains
- M6 (Production Agent Modes): complete
  - Two agent modes: `ContextFile` (retrieval-augmented) and `MicroBrain` (trained model)
  - Shared-state contract: context-file reads episodic summaries (read-only), micro-brain consumes retrieval snippets
  - Cross-mode handoff logging with mode origin, confidence, reason, and timestamp
  - SLO enforcement: latency P95, checkpoint domain count tracking
  - Acceptance report: `--acceptance-report` CLI and `GET /v1/acceptance` API endpoint
  - Mode switching: `POST /v1/mode` API endpoint and `set_mode()` service API
  - Verified by acceptance report: PASS (latency P95 < 50ms, checkpoint domains within limits)

## M0 Prerequisite: Encoder Decision (Must Be Fixed Before M1)

The encoder choice is the highest-impact dependency. Lock this before any implementation.

- **Primary default:** `sentence-transformers/all-MiniLM-L6-v2`
  - Dimensionality: `384`
  - Size: ~22 MB
  - Why: strong semantic quality per size, practical for edge/offline paths, fast inference.
- **Multilingual option:** `paraphrase-multilingual-MiniLM-L12-v2`
  - Use when multi-language support is required before M5.
- **High-accuracy fallback:** BERT-class encoder (`768` dims, much larger footprint)
  - Use only when M2 routing metrics fail with MiniLM and hardware budget allows.

Selection criteria (ranked): semantic retrieval quality, latency, model size, licensing, offline deployability.

## M0b Prerequisite: Global Bridge Calibration (Must Be Fixed Before M1)

Use one shared bridge mapping for all Mirrors and all promoted groups.

- Train the bridge once on a representative cross-domain calibration set.
- Freeze it after M1 validation.
- Reuse the same frozen bridge for every domain, Mirror, and promotion.
- Do not maintain per-Mirror bridge variants.

Rationale: cross-group routing requires all group embeddings to live in a comparable 64-dimensional space. Per-domain bridge training can create incompatible subspaces and unstable cosine routing.

### Calibration Dataset Spec (Required)

Build and approve a fixed cross-domain calibration set before M1 implementation.

- **Minimum coverage:**
  - At least `8` domains.
  - At least `500` samples per domain.
  - At least `20%` multilingual samples if multilingual support is in scope.
- **Required domain families:**
  - customer support,
  - coding/tool-use,
  - knowledge QA,
  - safety/policy refusal,
  - procedural instruction following,
  - short conversational turns,
  - multi-turn follow-ups,
  - adversarial/noisy prompts.
- **Label schema per sample:**
  - semantic intent,
  - tool/action target (or no-action),
  - policy regime,
  - language channel.
- **Ownership:**
  - Dataset curation owner: Applied ML lead.
  - Label quality owner: Domain annotation lead.
  - Final go/no-go signoff: Architecture owner before M1 gate closes.

## Target End State

- Same cloud backend supports two modes:
  - **Context-file agent** (prompt/retrieval driven).
  - **Trained micro-brain agent** (Growformer-driven with language I/O layers).
- Language becomes an interface, while durable behavior lives in the trained compact brain.

## Design Principles

- **Goal split**
  - Add `text -> embedding` (understanding input language).
  - Add `state -> text` (generating responses).
  - Keep Growformer as the decision/memory substrate, not the tokenizer/decoder.
- **Model strategy**
  - Don’t train a full LLM inside Growformer first.
  - Use a pre-trained language embedding model and optionally a pre-trained decoder model.
  - Let Growformer learn **task routing, retention, adaptation, and policy selection** over those embeddings.
- **Separation of concerns**
  - Keep tool/action decisions separate from free-form generation.
  - Keep encoder/decoder mostly stable early; let Growformer own adaptation.
  - Avoid full end-to-end co-training too early.

## Recommended Architecture

- **Language Encoder (new)**
  - Input: user text.
  - Output: fixed-size semantic vector (`d_model`), plus optional token-level vectors.
  - This is your language embedding layer.

- **Growformer Bridge (new)**
  - Projects encoder output into the dimensions Growformer expects.
  - Handles normalization, confidence, and temporal smoothing.
  - Writes into the existing `GroupEmbedding` / routing interface.
  - **Concrete spec (v1):**
    - Input dimension: `384` (MiniLM default).
    - Bridge output dimension: `64` for language routing.
    - Projection: `Linear(384 -> 64) + LayerNorm + confidence head`.
    - Temporal smoothing: EMA over turn embeddings (`alpha = 0.2`) for multi-turn routing (initial value, not final).
    - Rationale for `alpha = 0.2`: keeps routing responsive to the current turn while retaining enough short history to stabilize intent over brief dialogue bursts.
  - **Training lifecycle:**
    - Encoder is frozen in M1-M3.
    - Bridge is calibrated once globally (cross-domain) in M1.
    - Bridge is frozen after M1 and shared across all Mirrors/groups.
    - Mirror training adapts Growformer groups and router only, not the bridge mapping.

- **Growformer Core (existing)**
  - Uses the current mechanisms (main/mirror, promotion, frozen groups, routing, episodic memory).
  - Learns intent-to-skill mapping, dialogue policy, and long-horizon retention.

- **Language Generator — Single-Pass Binary Token Prediction**
  - No external decoder or templates. Pure Growformer substrate.
  - **TokenDictionary**: built from training corpus (up to 1024 entries). Provides a finite vocabulary for each domain. Acts as error-correction codebook — predicted IDs snap to nearest valid token.
  - **Binary token encoding**: each token ID encoded as 10 bits. Output layer = `MAX_TOKENS × 10` neurons (e.g. 32 slots × 10 bits = 320 neurons).
  - **Single forward pass**: the entire response is predicted at once (not autoregressive). Input embedding → substrate forward pass → 320-bit output → decode to token IDs → dictionary lookup → text.
  - **Training**: ONE substrate train_tick per sample (not per character). A secondary gradient pass reinforces EOS (all-zeros) for trailing token slots.
  - **Confidence truncation**: during decode, if all bits in a slot are near 0.5 (no confident prediction), generation stops. Prevents trailing repetition.
  - **Performance**: ~200x fewer substrate ticks per sample vs character-level autoregressive. Training that previously took days completes in minutes.
  - **Dictionary as knowledge routing**: per-group dictionaries constrain each group's output to domain-relevant tokens. The LearnedRouter selects the group → selects the vocabulary → generation is domain-specific.
  - Validated: "explain observer pattern" → `class Subject:def__init__(self):self._observers` from a 640-neuron substrate in one forward pass.

## Quantitative Gates (Pass/Fail)

These thresholds make milestones objective. Tune only after a documented ablation pass.

- **M2 routing**
  - Stage A intent accuracy: `>= 95%`
  - Median routing margin: `>= 0.25`
  - P10 routing margin: `>= 0.10` (fragile region floor)
  - OOD rejection (AUROC): `>= 0.90`
  - OOD false accept rate at operating threshold: `<= 5%`
- **M3 action mapping**
  - Action target accuracy: `>= 95%` on Stage A+B
  - Invalid/ambiguous fallback precision: `>= 98%`
- **M4 generation**
  - Task success non-regression: within `1%` absolute of M3
  - Hallucination rate threshold: no more than `+2%` absolute above template-only baseline
- **M5 continual retention**
  - Per-domain retention after sequential training: `>= 97%` of pre-sequence performance
- **M6 systems**
  - P95 end-to-end latency and memory limits must meet service SLOs defined by deployment tier.

## Domain Taxonomy and Boundary Policy (For M5)

Language domains are defined on three axes:

- **Task function** (support triage, medical QA, coding assist, etc.)
- **Policy regime** (safety/legal/format constraints)
- **Language channel** (English, Spanish, multilingual, etc.)

Domain handling rules:

- Same function + same policy + different language channel:
  - Start as one domain with language-conditioned routing.
  - Spawn a separate Mirror only if routing margin or retention drops below thresholds.
- Same function + different policy regime:
  - Always separate domain (new Mirror candidate).
- Overlap with existing groups:
  - If high confidence composition solves it, prefer VirtualGroup composition.
  - If residual stays high, spawn a new Mirror.

Spawn trigger for new Mirror:
- Route confidence below threshold for `K = 10` consecutive batches (initial value), or
- High residual after composition, or
- Safety-policy conflict with current group.

`K = 10` is chosen to suppress transient dips while still reacting within one short retraining window. Retune only with logged false-spawn and missed-spawn evidence.

## M1-M6 Milestone Plan

### M1 - Language Embedding Foundation

- Scope: Implement `text -> embedding` only (`Language Encoder` + basic `Growformer Bridge` projection/normalization).
- Out of scope: No natural-language generation yet.
- Out of scope (hard constraint): no encoder fine-tuning, no decoder training, no end-to-end co-training.
- Success criteria:
  - Encoder outputs stable fixed-size vectors for all supported inputs.
  - Global bridge calibration is completed on cross-domain samples and then frozen.
  - Growformer receives embeddings through the existing routing interface without architecture changes.
  - Baseline understanding metrics exist: intent accuracy, routing margin, OOD rejection.

### M2 - Embedding-First Routing Validation

- Scope: Use embeddings for classification/routing into existing groups (no tool execution yet).
- Goal: Prove semantic prompts map to correct groups with robust margins.
- Out of scope (hard constraint): no decoder integration, no joint encoder+bridge+core optimization.
- Success criteria:
  - Routing meets M2 quantitative gates on Stage A data.
  - Margin stability is confirmed across paraphrases and prompt variants.
  - OOD prompts are rejected or routed to fallback above the safety threshold.
  - EMA smoothing ablation is documented (`alpha in {0.0, 0.1, 0.2, 0.4}`), with selected alpha justified by routing stability vs turn-reactivity tradeoff.

### M3 - Intent-to-Action Layer

- Scope: Add structured action output from routed group (`intent -> action schema`), with machine-structured responses first.
- Design requirement: Keep tool decisions separate from language generation.
- Out of scope (hard constraint): no free-form natural language output path.
- Success criteria:
  - End-to-end flow works: text -> embedding -> routed group -> valid action JSON.
  - Action target accuracy meets M3 quantitative gates on Stage A + Stage B.
  - Invalid/ambiguous inputs trigger deterministic fallback behavior.

### M4 - Controlled Language Generation

- Scope: Add `state -> text` via constrained NLG (templates first; optional external decoder second).
- Rule: Do not merge policy/tool logic with free-form generation.
- Out of scope (hard constraint): no joint co-training of encoder/bridge/core/decoder.
- Success criteria:
  - Establish template-only hallucination baseline before decoder introduction.
  - Task success remains at or above M3 baseline after adding generation.
  - Generation quality remains within M4 quantitative gates.
  - Tool/action choices remain traceable to structured intermediate state.

### M5 - Continual Language Learning Integration

- Scope: Bring language domains into mirror/main promotion workflow (new domains train in mirror, promote via existing gates).
- Dataset focus: Stage C multi-turn + Stage D noisy/adversarial.
- Out of scope (hard constraint): no direct writes to promoted/frozen groups; no bridge retraining.
- **Region binding:** Generation and codegen heads are conditioned on `[raw_embedding; action_type_one_hot; group_id_one_hot]` so the routed group selects the correct region-specific attractor; training and inference use the same conditioning (see `action_classifier::group_id_one_hot`, service generation/codegen path).
- Success criteria:
  - New-domain learning does not regress prior-domain performance beyond retention budget.
  - Promotion gate criteria (accuracy/stability) are enforced before consolidation.
  - Sequential domain training shows measurable retention advantage versus non-isolated baseline.
  - Domain assignment follows taxonomy and Mirror spawn policy.
  - Cross-domain routing comparability is preserved (single shared frozen bridge across all groups).

### M6 - Production Agent Modes on Same Backend

- Scope: Operationalize both modes:
  - Context-file agent
  - Trained micro-brain agent
- Include system constraints: latency, memory footprint, checkpoint growth per added domain.
- Shared-state contract:
  - Context-file mode may read from micro-brain episodic summaries (read-only).
  - Micro-brain mode may consume retrieval snippets as additional input context.
  - Raw episodic memory is never directly mutated by context-file mode.
  - Every cross-mode handoff is logged with mode origin and confidence.
  - Logs are written to the central observability pipeline (same sink as model routing telemetry) with weekly review by on-call ML/platform owners.
- Success criteria:
  - Both modes run on the same cloud backend with shared APIs/infrastructure.
  - SLO targets are met for latency and memory; checkpoint growth stays within plan.
  - Final acceptance report includes understanding, generation, continual-learning, and system metrics.

### Cross-Phase Guardrails and Risks

- Biggest risk: mixing language fluency and agent policy learning too early.
- If everything is co-trained end-to-end at once, debugging becomes difficult and regressions become hard to localize.
- Keep encoder/decoder mostly stable early; let Growformer own adaptation.
- Avoid full end-to-end co-training until M4+ behavior is stable.
- Preserve architectural separation:
  - Language interface layer (I/O)
  - Growformer core (routing, retention, policy)
  - Generation surface (controlled output)
- Promotion failure policy:
  - If a Mirror fails promotion after `N = 3` cycles, trigger rollback workflow:
    1) tighten domain scope,
    2) retry with composition-first strategy,
    3) escalate to human labeling review.
  - If failures exceed `N_hard = 8`, escalate as an architectural incident (not only a data issue).
  - Rationale: `N = 3` catches persistent data/scope issues early; `N_hard = 8` marks repeated failure unlikely to be solved by relabeling alone and likely requiring architectural intervention.
  - Failed Mirrors do not write to Main Dimension.
- Adversarial correction policy (Stage D):
  - Frozen groups are not edited directly.
  - Corrections occur through router-level rejection, composition patching, or spawning a dedicated adversarial Mirror.
  - Promotion to Main only after adversarial robustness gates pass.

## Generation Architecture Evolution

The generation subsystem went through several iterations to find the approach that matches the Growformer substrate's strengths.

### Iteration 1: Character-level autoregressive (abandoned)
- 8 output neurons predict one bit per character, fed back autoregressively.
- ~200 sequential `train_tick` calls per sample (one per character).
- Substrate dynamics (physics, pruning, KWTA) run on every tick — designed for consolidation, not per-character optimization.
- Result: training took days for 1000 epochs. Loss plateaued. The approach fights the substrate's design.

### Iteration 2: Spectral encoding via DCT (explored, not adopted)
- Attempted DCT (Discrete Cosine Transform) compression of output sequences.
- Hypothesis: frequency decomposition would compress text into fewer coefficients.
- Finding: text bytes and token ID sequences are not smooth signals. K_lossless ≈ 85-95% of signal length — virtually no compression. DCT is wrong basis for discrete text.
- The `spectral.rs` module remains available for future signal-processing applications (e.g. routing dispatch).

### Iteration 3: Dictionary + binary token prediction (current)
- Insight from speech-to-text: decompose through stages of increasing discreteness.
- Dictionary reduces 130-character outputs to ~25 tokens. Binary encoding: 10 bits per token ID.
- Single forward pass predicts all 320 output bits simultaneously.
- ONE `train_tick` per sample — substrate dynamics run once at the right granularity (per-experience, not per-character).
- Dictionary provides error correction: predicted IDs snap to nearest valid token.
- 200x speed improvement over Iteration 1. Observable learning signal: loss 0.70 → 0.13 in 100 steps.

### Iteration 4: Algebraic codebook generation (current)
- Insight: even with binary token prediction, the substrate must predict 768–1920 raw bits. At 78% per-bit accuracy (loss ~0.22), the probability of a full token being correct is only ~7.4%. The bits-per-token problem is fundamentally a dimensionality problem.
- Solution: **algebraic codebook** decomposes the response space using group-theoretic factorization.
  1. Cluster training responses into **archetypes** (structural patterns with fixed tokens and variable slots).
  2. The substrate predicts only **(archetype index + slot values)** — typically 7–120 bits total.
  3. Decoding uses nearest-neighbor soft decode over archetype selection and slot filling.
- Result: **10–100× reduction** in prediction space. Loss reaches 0.003 in 200 steps vs. 0.22 in 3000 steps for raw binary.
- The codebook is built automatically from training data via positional token overlap clustering. No manual template authoring.
- Backward compatible: `GroupGenEnv` detects the codebook and switches encode/decode paths. Old brains without codebook continue using raw binary.
- Error correction pipeline (Gray code, Hamming ECC, soft decode) remains available in the raw binary fallback path.

### Key architectural principle
The Growformer substrate excels at structural tasks: classification, routing, pattern recognition, binary prediction. Sequential coherence across many steps is not its strength. The dictionary + binary token approach converts generation from a sequential prediction problem into a parallel binary classification problem — exactly what the substrate is built for. The algebraic codebook takes this further: it converts parallel binary classification into **factored classification** — which archetype, then which slot values — reducing the prediction space to match the substrate's native strengths.

## Training & Data Strategy

- **Stage A:** intent/small-task corpora (high precision labels, low ambiguity).
- **Stage B:** paraphrase robustness (same intent, many phrasings).
- **Stage C:** multi-turn stateful tasks (requires episodic recall).
- **Stage D:** adversarial/noisy prompts for safety and fallback behavior.

Keep labels at three levels:
- semantic intent,
- tool/action target,
- response style/tone constraints.

## Evaluation We Should Track

- **Understanding**
  - Intent accuracy, routing margin, OOD rejection rate.
- **Generation**
  - Task success rate, factual consistency, hallucination rate.
- **Continual learning**
  - Per-domain retention after sequential training (the current strongest metric).
- **Systems**
  - Latency, memory footprint, checkpoint growth per added language task.

## Group-Theoretic Foundations for Generation

The algebraic codebook is the first concrete application of group theory to Growformer generation. The following connections identify further mathematical structures that are either already present in the system or available for near-term implementation.

### Immediately actionable

**Hopf algebra structure (composable archetypes).** The codebook decomposition response → (archetype, slots) is a comultiplication in the sense of Hopf algebras. If archetypes are closed under composition — archetype A followed by archetype B maps to archetype C — then multi-step responses (reasoning chains, multi-turn dialogue) become sequences of archetype compositions. The `HopfCompositionTable` implements this: it decomposes archetypes into positional fragments and uses beam search to compose fragments from multiple archetypes. Transition scores are derived from **E8 root inner products** between quantized archetype prototypes (see below), providing algebraically exact compatibility.

**E8 lattice quantization (optimal sphere packing).** The bridge embedding is 64d (GEN_COND_DIM = 64 = 8 × 8). Each 8d subspace is quantized to the **E8 lattice** — the unique densest sphere packing in dimension 8 (Viazovska, 2017). The E8 lattice provides:

- **Archetype selection**: `E8Lattice::select_archetype()` replaces O(n×d) cosine scan with O(64) lattice nearest-point decoding. Provably optimal codeword separation (kissing number 240, density π⁴/384).
- **Hopf transition scoring**: `E8Lattice::compatibility_score()` computes root inner products between quantized archetype prototypes. The 240 roots of E8 have inner products in {-2, -1, 0, 1, 2}, providing discrete, algebraically exact compatibility — replacing heuristic cosine similarity.
- **Error correction**: The **extended Hamming code [8,4,4]** is the binary code underlying the E8 lattice construction. It detects 2-bit errors and corrects 1-bit errors, upgrading from standard Hamming [7,4,3].

Implementation: `E8Lattice` struct in `spectral.rs` with `nearest_point()`, `quantize_64d()`, `root_inner_product()`, `compatibility_score()`, `select_archetype()`. See Whitepaper §5.5.

**Leech lattice + ProjectModel (codebase spatial index).** The **Leech lattice** Λ₂₄ is the unique densest sphere packing in 24 dimensions (Cohn et al., 2017), constructed from 3 copies of E8 glued by the **extended Golay code [24,12,8]** (corrects 3-bit errors, detects 4). The Leech lattice provides:

- **ProjectModel**: Maps files, functions, types, modules to 24d Leech-quantized embeddings via a **hybrid embedding pipeline** with 6 signal channels (4d each): (1) **Structural skeleton** — AST-lite parsing via `CodeAnalyzer` (declarations, call graph, imports, cyclomatic complexity, nesting; 8 languages: Rust, Python, TypeScript, JavaScript, Go, C, C++, Java); (2) **Semantic content** — FNV-1a multi-hash projection of significant tokens; (3) **Relational graph** — import fan-out, API surface ratio, call graph density, module depth; (4) **Edit correlation** — `GitHistory` from git log (co-change frequency, churn, author diversity, recency); (5) **Test/quality signal** — test file detection, assertion density, documentation density; (6) **Pattern identity** — API surface ratio, trait/interface density, structural fingerprint.
- **Context conditioning**: `generation_with_context()` retrieves k nearest Leech neighbors for the active file and augments the generation prompt with related codebase context. This enables context-aware code generation — the system knows what imports are nearby, what tests cover this function, and what patterns are used in related files.
- **Nearest-neighbor queries**: `LeechLattice::nearest_neighbors()` finds related entities in Leech space for spatial reasoning across the full project.
- **Error correction**: Extended Golay [24,12,8] provides native error correction for the Leech construction.
- **Two-level hierarchy**: E8 (8d, local) for archetype selection and Hopf transitions; Leech (24d, global) for project-level spatial reasoning.

Implementation: `LeechLattice`, `ProjectModel`, `CodeAnalyzer`, `HybridEmbedder`, `GitHistory` in `spectral.rs`; `golay_encode()`, `golay_decode()` for ECC. REPL: `/index <path>` auto-loads git history when `.git` is present; `/project [file]`. See Whitepaper §5.5.

**Quantitative geometry of nilpotent groups (convergence bounds).** The physics-based neural environment is a metric space with layered competitive dynamics. Results on conjugator lengths in filiform nilpotent groups provide bounds on transformation distance within such layered structures. Each substrate layer corresponds to a step in the lower central series. If the substrate's dynamics can be verified as nilpotent, conjugator length bounds translate to **provable convergence guarantees**: a theoretical upper bound on training steps needed to move from one knowledge state to another. This would formalize the current empirical observation that algebraic generation converges in ~200 steps.

### Near-term relevant (Continuum phase)

**Profinite invariants (domain boundary detection).** Engram consolidation creates fixed points in weight space. The Mirror → group → Main hierarchy is structurally a profinite (inverse limit) tower. Stable commutator length — quantifying how "far" a group element is from being a product of commutators — provides a metric for **knowledge compatibility**: low stable commutator length between a new input's representation and existing group embeddings indicates clean composition; high length indicates a genuinely novel domain requiring a new Mirror. This could replace the current heuristic spawn trigger (K=10 consecutive low-confidence batches) with a theoretically grounded distance measure.

**Torsors and cryptographic verification.** In distributed inference, multiple brains produce outputs without a canonical ground truth — the output space is naturally a torsor. New work connecting torsors to Σ-protocols indicates a path toward **zero-knowledge proofs of correct inference**: proving a brain produced a specific output from a specific input without revealing weights. Combined with quantum-resistant authenticated data structures (SIS-based vector commitments, e.g. `spacekit-quantum-verkle`), this provides the cryptographic layer for a **brain marketplace** with provable inference. Directly supports the AI Operating System's federated trust model.

### Research directions (post-Continuum)

**Finite reductive groups (representation-theoretic capacity).** KWTA competitive dynamics create sparse basis representations. Character theory of finite groups of Lie type could yield principled initialization schemes and **a priori capacity estimates** — predicting how many neurons a group needs before training begins, based on the representation-theoretic complexity of the input distribution.

**Regularized determinants of the Rumin complex (spectral capacity).** Analytic tools from nilpotent Lie group theory (regularized determinants, spectral analysis on stratified groups) could provide a single scalar invariant characterizing the **information capacity** of a layer configuration. Useful for automatic architecture search and adaptive neurogenesis triggers.

### AI Safety Implications

The algebraic generation architecture produces safety properties as a byproduct of its design:

- **Bounded output space.** Every specialist's codebook is finite and enumerable. The set of possible outputs can be audited before deployment. Content outside the codebook cannot be generated regardless of input — this is structural, not a learned constraint.
- **Prompt injection resistance.** Adversarial inputs may cause misrouting or poor archetype selection, but cannot produce content outside the codebook's pattern space. The attack surface is qualitatively different from unbounded generation.
- **Deterministic frozen inference.** Consolidated specialists produce identical output for identical input, enabling certification: the brain at audit is the brain in production.

These properties are consequences of codebook factorization, structural isolation, and engram consolidation — the same mechanisms that provide generation quality and continual learning. See Whitepaper §5.6.

### Reference

These connections are documented in Whitepaper §5.5 (Group-Theoretic Foundations). The Hopf composition table is implemented (`HopfCompositionTable` in `group_gen.rs`). Nilpotent convergence verification is tracked in the development roadmap. AI safety properties are formally treated in Whitepaper §5.6.

## OCEAN Personality Profile

Conversational agents need behavioral identity beyond knowledge recall. The OCEAN (Big Five) personality model provides a compact, well-studied parameterization of behavioral style that integrates directly into the Growformer generation pipeline.

### Profile definition

Each agent (or per-group specialist) carries a 5-float OCEAN vector:

```
ocean: [O, C, E, A, N]   // each 0.0–1.0
```

| Trait | Low (0.0) | High (1.0) |
|-------|-----------|------------|
| **Openness** | Conservative, stays on-topic, prefers known patterns | Exploratory, creative, willing to combine unfamiliar concepts |
| **Conscientiousness** | Casual, flexible phrasing | Precise, structured, detail-oriented |
| **Extraversion** | Terse, minimal elaboration | Verbose, enthusiastic, expansive |
| **Agreeableness** | Neutral, matter-of-fact tone | Warm, empathetic, affirming |
| **Neuroticism** | Confident, assertive statements | Cautious, hedging, acknowledges uncertainty |

### Integration points

**1. Conditioning vector (generation).** The OCEAN vector modulates the last 5 dimensions of the 64d bridge embedding via `OceanProfile::condition_vector()`. Each dimension is shifted by `(value - 0.5) * 0.15`, providing a subtle directional bias without overriding content semantics. No dimension increase — personality is a secondary signal within the existing conditioning space.

**2. Hopf beam search (composition).** OCEAN's `hopf_diversity_bonus()` = `(openness - conscientiousness)` clamped to [-0.3, 0.3]. This bonus/penalty is applied to cross-archetype vs. same-archetype transitions during beam search via `compose_with_personality()`:
- **High openness** → positive diversity bonus → favors cross-archetype fragment mixing (creative, novel compositions).
- **High conscientiousness** → negative diversity bonus → favors same-archetype coherence (precise, structured responses).
- The underlying compatibility-scored transitions (E8 root inner products between quantized archetype prototypes) provide the algebraically exact base scoring; OCEAN modulates on top.

**3. EMA temporal smoothing (conversation).** Openness modulates the bridge's EMA alpha:
- High O → alpha = 0.35 (responsive to topic shifts, welcomes new directions)
- Low O → alpha = 0.10 (anchored to conversation history, resists topic drift)
- Default → alpha = 0.20

**4. Personality presets.** `OceanProfile` provides named presets via static constructors:
- `assistant()`: `[0.5, 0.8, 0.5, 0.7, 0.2]` — balanced professional
- `creative()`: `[0.9, 0.4, 0.8, 0.6, 0.3]` — exploratory, enthusiastic
- `engineer()`: `[0.4, 0.9, 0.3, 0.5, 0.2]` — precise, structured, concise
- `analyst()`: `[0.5, 0.9, 0.3, 0.5, 0.7]` — cautious, thorough
- Custom via REPL (`/ocean O C E A N`) or `set_personality()` API.

**5. Persistence.** Personality is a session-level setting. No additional training required to change personality — it modulates scoring, not learned parameters.

### Training strategy

Phase 1 (current): OCEAN modulates Hopf scoring, conditioning vector, and EMA alpha without requiring OCEAN-labeled training data. The presets shift behavior subtly.

Phase 2 (personality-aware): add OCEAN-labeled training samples where the same prompt has multiple responses with different style profiles. The network learns the mapping from OCEAN conditioning → slot vocabulary selection.

Phase 3 (emergent): during Continuum (train-while-on), user feedback implicitly shapes the OCEAN profile. Positive feedback on a creative response increases the O component for that session; positive feedback on a precise response increases C. The profile evolves through interaction.

## Conversation REPL (Implemented)

Multi-turn conversational inference over a trained brain. Implementation: `LanguageService::converse()` + `run_conversation_repl()` in `main.rs`.

### Architecture

```
┌─ User Input ──────────────────────────────────────────┐
│ "help me reset my password"                           │
└──────────────┬────────────────────────────────────────┘
               ▼
┌─ Context Augmentation ────────────────────────────────┐
│ Prepend recent history (3-turn sliding window)        │
│ "user: prev_q | agent: prev_a | user: current"       │
└──────────────┬────────────────────────────────────────┘
               ▼
┌─ Language Encoder → Bridge + EMA Smoother ────────────┐
│ 384d → 64d projection                                 │
│ state = alpha * current + (1-alpha) * prev_state      │
│ alpha modulated by OCEAN (E↑ = faster, C↑ = slower)   │
└──────────────┬────────────────────────────────────────┘
               ▼
┌─ Identity Check ─────────────────────────────────────┐
│ "who are you" → immediate identity response           │
│ (skips routing and generation)                        │
└──────────────┬────────────────────────────────────────┘
               ▼
┌─ OCEAN Personality Conditioning ─────────────────────┐
│ Modulate last 5 dims of 64d vector: (val-0.5)*0.15   │
│ Set diversity_bonus on all gen envs                    │
└──────────────┬────────────────────────────────────────┘
               ▼
┌─ VirtualGroup Generation (Levels 1–3) ───────────────┐
│ L1: Competitive multi-head → best confidence          │
│ L2: Cross-group sentence composition                  │
│ L3: Episodic memory cache/retrieve                    │
│ Hopf composition if conf < 0.9 (personality-aware)    │
└──────────────┬────────────────────────────────────────┘
               ▼
┌─ Response ───────────────────────────────────────────┐
│ [route: GeneralQuery g=1 conf=0.94]                   │
│ Password reset links expire... (conf=0.94)            │
└───────────────────────────────────────────────────────┘
```

### CLI interface

```
cargo run --release -- --infer [--brain brain.bin]

=== Growformer Conversation REPL ===
  Agent: Growformer (by swtch.ai)
  Personality [O=0.5 C=0.8 E=0.5 A=0.7 N=0.2]

Commands:
  /personality <preset>   Switch: assistant, creative, engineer, analyst
  /ocean O C E A N        Set custom OCEAN values (0.0-1.0)
  /reset                  Clear conversation history
  /history                Show conversation history
  /single <prompt>        Single-shot (no conversation context)
  /status                 Show brain + personality info
  quit | exit             Exit

[turn 1] > help me reset my password
  [route: SupportRequest g=0 conf=0.94]
  Password reset links expire... (conf=0.94)

[turn 2] > /personality creative
  Personality: creative (open, enthusiastic)
  [O=0.9 C=0.4 E=0.8 A=0.6 N=0.3]

[turn 3] > who are you
  I am Growformer, a Growformer Agent by swtch.ai...
```

### State management

- **EMA embedding**: accumulated across turns via `bridge_text()` (not `bridge_text_stateless()`). EMA alpha modulated by OCEAN personality. Resets on `/reset`.
- **OCEAN profile**: set at session start as `assistant()` preset, modifiable via `/personality <preset>` or `/ocean O C E A N`.
- **Turn history**: up to 20 turns stored in `ConversationContext`. Recent 3-turn window prepended to prompts for semantic grounding.
- **Group tracking**: active group and confidence displayed with each response.
- **Identity**: "who are you" / "what is your name" / "who made you" intercepted before routing.

Estimated: ~100 lines in `main.rs` under the `--chat` flag.
