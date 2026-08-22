# Growformer System Architecture

## Full System Overview

```mermaid
flowchart TD
    subgraph INPUT ["Input"]
        TEXT["User Prompt"]
        ENC["all-MiniLM-L6-v2\n768d raw embedding"]
        BRIDGE["Frozen Bridge\n768d → 128d"]
        TEXT --> ENC --> BRIDGE
    end

    subgraph ROUTING ["Routing & Classification"]
        ROUTER["LearnedRouter\n(InfraciliaryLattice)\nK-NN voting + field gradient bias\nselects group 0–13"]
        ACTION["ActionClassifier\n(InfraciliaryLattice)\n5 action types"]
        BRIDGE --> ROUTER
        BRIDGE --> ACTION
    end

    subgraph META_ROUTE ["GrowformerLang Meta-Routing"]
        META_CB["MetaCodebook\nconcept centroids\nLanguageProjectors\nconcept → group mapping"]
        INFER_CONCEPT["infer_concept()\ntext → MetaConcept\n(BinaryArithmetic, DataStructure, ...)"]
        INFER_OP["infer_operation_topic()\ntext → specific operation\n(addition_operation, ...)"]
        DETECT_LANG["detect_language()\ntext → TargetLanguage\n(Rust, Python, TS, Go)"]
        BRIDGE --> META_CB
        TEXT --> INFER_CONCEPT --> META_CB
        TEXT --> INFER_OP
        TEXT --> DETECT_LANG
    end

    subgraph UNDERSTAND ["Understanding Layer"]
        TOPIC["Topic Classifier\n(InfraciliaryLattice)\nsemantic intents"]
        VERB["Verb Classifier\n(InfraciliaryLattice)\n8 abstract action-verbs"]
        BRIDGE --> TOPIC
        BRIDGE --> VERB
    end

    subgraph META ["MetaBrain Coordinator"]
        direction TB
        MB_COORD["CentroidCoordinator\nnearest-neighbor fusion\nweights micro-brain outputs"]
        ARCH_BRAIN["ArchetypeBrain\n(InfraciliaryLattice)\nprograms across 14 groups"]
        TOPIC --> MB_COORD
        VERB --> MB_COORD
        ACTION --> MB_COORD
        ARCH_BRAIN --> MB_COORD
    end

    subgraph CONDITIONING ["Conditioning Pipeline"]
        ROTOR["Clifford Rotor Cl(1,7)\nSpaceTime Algebra\n28 bivector params\n7 boost + 21 rotation"]
        LANG_PROJ["LanguageProjector\nClifford rotor per language\nconcept → language-specific embedding"]
        BRIDGE --> ROTOR
        META_CB --> LANG_PROJ
    end

    subgraph GEN ["Per-Group Generation (×14 groups)"]
        direction TB
        DICT["TokenDictionary\n≤2048 entries\nsemantic-ordered vocab"]
        CODEBOOK["AlgebraicCodebook\narchetypes + variable slots"]
        LATTICE["InfraciliaryLattice\n(Paramecium)\none-pass learning\nequivariant STA scoring"]
        TOPIC_SUB["Topic Sub-Lattices\nper-operation sub-indices\nforced topic routing"]
        HOPF["Hopf Composition Table\n3-segment fragment mixing"]
        CLOZE["Cloze Learning\nfill-in-the-blank games\ncontrastive centroid drift"]
        DECODE["Algebraic Decode\nsoft archetype + slot decode\n→ token IDs → text"]
        DICT -.->|build| CODEBOOK
        LATTICE --> TOPIC_SUB
        CLOZE -.->|reward/punish| LATTICE
        CODEBOOK -.->|archetype/slot lookup| DECODE
        HOPF -.->|multi-archetype compose| DECODE
    end

    subgraph REASONING ["Reasoning Engine"]
        COGMAP["CognitiveMap\ngraph of all programs\ncross-group structural links"]
        REASON["ReasoningEngine\nSystem 1.5: wave settling\ncompositional assembly\ntransfer rotors"]
        SYS2["System 2\nWorkingMemory buffer\nStepOperator\nvariable-length chaining\nmax 6 steps"]
        COGMAP --> REASON
        REASON --> SYS2
    end

    subgraph COHERENCE ["Neural Coherence Analysis"]
        BAND["Band Decomposition\nδ=scalar θ=vector\nα/β=bivector γ=trivector"]
        ENS_COH["Ensemble Coherence\npower-weighted pairwise\nband synchrony"]
        COH_SEL["Coherence Select\nrelevance × synchrony\nmin_coherence=0.15"]
        BAND --> ENS_COH --> COH_SEL
    end

    subgraph METACOG ["MetaCognition (Reflective Quality Gate)"]
        REFLECT["Reflection Brain\n930+ (prompt,response) pairs\n124+ topic centroids"]
        SCORES["Coherence (0.45w)\nRelevance (0.35w)\nCompleteness (0.20w)\n→ quality score"]
        DECIDE["Accept / Retry / Degrade\nthreshold: 0.45\nmax retries: 2\ngraceful degradation"]
        REFLECT --> SCORES --> DECIDE
    end

    subgraph CODE ["Code Generation (6 groups)"]
        CODE_ENV["GroupCodeEnv\n(InfraciliaryLattice)\nlanguage-aware forced topic"]
    end

    ROUTER -->|group_id| LATTICE
    META_CB -->|concept override| LATTICE
    MB_COORD -->|conditioning 192d| LATTICE
    ROTOR -->|rotated embedding| LATTICE
    LANG_PROJ -->|projected embedding| CODE_ENV
    INFER_OP -->|forced topic hint| TOPIC_SUB
    DETECT_LANG -->|language filter| CODE_ENV
    ARCH_BRAIN -->|archetype_idx| DECODE
    MB_COORD -->|conditioning| CODE_ENV
    ROUTER -->|group_id| CODE_ENV
    SYS2 -->|"cross-domain\nambiguity"| DECODE
    TOPIC_SUB -->|"broad query\nmulti-topic"| COH_SEL
    COH_SEL -->|"coherent ensemble"| DECODE
    DECODE -->|candidate| REFLECT
    CODE_ENV -->|candidate| REFLECT

    subgraph OUTPUT ["Output"]
        GEN_OUT["Generated Response\nsingle forward pass"]
        CODE_OUT["Generated Code\nlanguage-specific"]
    end

    DECIDE -->|accepted| GEN_OUT
    DECIDE -->|accepted| CODE_OUT
    DECIDE -->|degraded| GEN_OUT

    classDef input fill:#e3f2fd,stroke:#1565c0,stroke-width:2px;
    classDef route fill:#fff3e0,stroke:#ef6c00,stroke-width:2px;
    classDef understand fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px;
    classDef meta fill:#fce4ec,stroke:#c62828,stroke-width:2px;
    classDef gen fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    classDef cond fill:#e0f7fa,stroke:#00695c,stroke-width:2px;
    classDef reason fill:#fff8e1,stroke:#f57f17,stroke-width:2px;
    classDef metalang fill:#e8eaf6,stroke:#283593,stroke-width:2px;

    class ENC,BRIDGE input
    class ROUTER,ACTION route
    class TOPIC,VERB understand
    class MB_COORD,ARCH_BRAIN meta
    class LATTICE,DICT,CODEBOOK,DECODE,HOPF,CODE_ENV,TOPIC_SUB,CLOZE gen
    class ROTOR,LANG_PROJ cond
    class COGMAP,REASON reason
    class META_CB,INFER_CONCEPT,INFER_OP,DETECT_LANG metalang
```

## Training Pipeline

```mermaid
flowchart LR
    subgraph S1 ["Stage 1: Router"]
        R["LearnedRouter\n(InfraciliaryLattice)\none-pass develop()"]
    end

    subgraph S2 ["Stage 2: Classifier"]
        AC["ActionClassifier\n(InfraciliaryLattice)\n5 action types"]
    end

    subgraph S2b ["Stage 2b: GrowformerLang"]
        MC["MetaCodebook\nconcept centroids\nLanguageProjectors"]
    end

    subgraph S25 ["Stage 2.5: Understanding"]
        UL["Understanding Layer\ntopic + verb classifiers\n(InfraciliaryLattice)"]
        MB["MetaBrain\nCentroidCoordinator\n+ ArchetypeBrain"]
        UL --> MB
    end

    subgraph S3 ["Stage 3: Generation"]
        IDX["IndexedGenEnv\none-pass lattice develop()\ntopic sub-lattice build\ncodebook + Hopf table"]
    end

    subgraph S35 ["Stage 3.5: Cloze"]
        CL["Cloze Learning\n6 rounds × K voters\ncontrastive reward/punish"]
    end

    subgraph S4 ["Stage 4: Export"]
        EX["Paramecium Post-Training\nCognitive Map + Reasoning\nBrain Export (.bin)"]
    end

    S1 --> S2 --> S2b --> S25 --> S3 --> S35 --> S4

    classDef stage fill:#e3f2fd,stroke:#1565c0,stroke-width:2px;
    class R,AC,MC,UL,MB,IDX,CL,EX stage
```

**Key change from previous architecture:** All MLP/NeuralEnvironment training has been replaced by one-pass `InfraciliaryLattice` (Paramecium) `develop()`. No backpropagation, no epochs, no iterative optimization. Training cost is O(n) where n = number of samples.

## Component Details

### Embedding & Conditioning Pipeline

| Component | Dimensions | Purpose |
|-----------|-----------|---------|
| Raw embedding (MiniLM) | 768d | Semantic representation of input text |
| Frozen bridge | 768d → 128d | Dimensionality reduction for routing |
| Clifford rotor Cl(1,7) | 28 bivector params (7 boost + 21 rotation) | SpaceTime Algebra geometric rotation |
| Understanding vector | 48d (32 topic + 16 verb) | Semantic intent conditioning |
| Language projector | 128d → 128d | Concept-to-language Clifford rotation |
| Generation conditioning | 192d total | Rotor output + understanding + zero-padding |

### SpaceTime Algebra Cl(1,7)

Upgrade from Euclidean Cl(8,0) to Minkowski-like Cl(1,7) with one timelike dimension:

| Property | Value |
|----------|-------|
| Algebra | Cl(1,7) — 1 timelike + 7 spacelike basis vectors |
| Boost bivectors | 7 (e₀∧eᵢ) — encode causal/temporal direction |
| Rotation bivectors | 21 (eᵢ∧eⱼ) — encode spatial/structural relations |
| Timelike dimension | e₀ carries `goal_magnitude` from UnderstandingLayer |
| Causal fingerprint | 7d boost bivector extraction |
| Spatial fingerprint | 21d rotation bivector extraction |

The timelike dimension encodes the "goal direction" of a prompt — how strongly the input drives toward action (high goal_magnitude for "write a function" vs. low for "what is X"). This gives embeddings a causal arrow that purely spatial algebra lacks.

### GrowformerLang (Meta-Programming Language)

A language-agnostic concept layer that separates *what* is being asked from *which programming language* it targets:

| Component | Purpose |
|-----------|---------|
| `MetaConcept` | 20+ abstract categories (BinaryArithmetic, DataStructure, TraitInterface, ...) |
| `MetaOp` | Abstract operations (Bind, BinaryOp, FnDef, Loop, PatternMatch, ...) |
| `TargetLanguage` | Rust, Python, TypeScript, Go, Generic |
| `LanguageProjector` | Per-language Clifford rotor mapping concept → language-specific embedding |
| `MetaCodebook` | Maps MetaConcepts to centroids, group indices, and per-language projectors |
| `infer_concept()` | Text → MetaConcept (uses action_target as ground truth during training) |
| `infer_operation_topic()` | Text → specific operation name (e.g., "subtraction_operation") |
| `detect_language()` | Text → TargetLanguage from keyword signals |

**Two-stage routing:**
1. `infer_concept()` classifies the abstract concept from text
2. `MetaCodebook.route_and_project()` finds the best group for that concept and applies the language-specific projector

This solved the inter-group routing problem for fine-grained coding categories.

### Topic Sub-Lattices & Forced Topic Routing

Each `IndexedGenEnv` contains multiple `TopicSubIndex` entries — small InfraciliaryLattice instances built from training samples grouped by `semantic_intent`. This provides within-group discrimination:

| Group | Topics (before) | Topics (after retagging) |
|-------|----------------|-------------------------|
| g2 (coding_arithmetic) | 1 ("coding_implementation") | 13 (addition_operation, subtraction_operation, ...) |

**Forced topic routing:** When `infer_operation_topic()` identifies a specific operation (e.g., "subtraction_operation"), the system directly queries the matching sub-lattice, bypassing cross-topic competition. This prevents high cosine similarity between sibling operations (add vs sub) from drowning out the correct result. The forced topic result is returned directly, skipping the field inhibition gate which would otherwise override it via global archetype slot inference.

**Language-aware code generation:** `forced_topic_response_lang()` accepts a language hint and filters sub-lattice programs by language-specific code markers (`fn ` for Rust, `def ` for Python), ensuring the generated code matches the requested language.

### Within-Group Equivariant Scoring

`nearest_response_in_lattice()` uses full multivector geometry instead of scalar cosine:

| Component | Weight | What it captures |
|-----------|--------|-----------------|
| Scalar cosine | 30% | Overall similarity (invariant) |
| Spatial bivector alignment | 35% | Rotation structure match (equivariant) |
| Causal bivector alignment | 20% | Temporal/goal direction match (equivariant) |
| Proximity (inverse distance) | 15% | Embedding space closeness |

Equivariant features preserve orientation and phase, enabling discrimination between operations that share the same symmetry class (e.g., add vs sub within arithmetic).

### Cloze (Fill-in-the-Blank) Learning

Post-lattice-development reinforcement that teaches programs to infer slot content from input semantics:

| Parameter | Value |
|-----------|-------|
| Rounds | 6 per group |
| K voters | 7 nearest programs |
| Reward rate | 0.10 (centroid drift toward correct) |
| Punish rate | 0.06 (contrastive: repel from wrong + attract toward own centroid) |

**Contrastive punishment:** Incorrect programs are simultaneously repelled *away* from the wrong input and attracted *toward* their original centroid. This creates sharper boundaries between related operations within the same group.

### Reasoning Engine

Cross-domain compositional inference using the CognitiveMap:

| Component | Purpose |
|-----------|---------|
| CognitiveMap | Graph of all programs across all groups, connected by structural similarity |
| ReasoningEngine (System 1.5) | Multi-group activation, wave settling (4 rounds), compositional assembly |
| System 2 (`reason_deliberate()`) | Variable-length deliberate chaining with WorkingMemory + StepOperator |
| Transfer rotors | Clifford rotors mapping structure from one domain to another (analogical reasoning) |
| `should_reason()` | System 1.5 heuristic: triggers when multi-group activation is ambiguous |
| `should_reason_deliberate()` | System 2 heuristic: triggers when confidence is 0.10–0.65 AND cross-domain ambiguity > 0.70 |

### MetaCognition (Reflective Quality Gate)

Post-generation quality evaluation via a generate-reflect-decide loop. Addresses content bleed, wrong-archetype selection, and low-confidence garbage.

| Component | Purpose |
|-----------|---------|
| `MetaCognition` (`src/metacognition.rs`) | Reflection brain trained from (prompt, response, topic) triples |
| `ReflectionScores` | Coherence (0.45w), relevance (0.35w), completeness (0.20w) → quality score |
| `ReflectionOutcome` | Accept (quality ≥ 0.45) / Retry (adjust conditioning, max 2) / Degrade (structured "I don't know") |
| Graceful degradation | When all retries fail, emits topic-specific "outside my training scope" message |
| `rebuild_metacognition()` | Automatically reconstructs from brain data on `load_brain()` |

**Training:** During brain training, every lattice program across all 14 gen + 7 code groups contributes a (prompt_emb, response_emb, topic) triple. The reflection brain learns per-topic centroids of what good (prompt, response) pairs look like in joint embedding space. At inference, the candidate response's joint embedding is measured against the appropriate topic centroid.

**Inference flow:** System 1 generates → MetaCognition evaluates → Accept/Retry/Degrade.

### System 2 Reasoning (Deliberate Multi-Step Chaining)

Extends the ReasoningEngine from fixed-round wave settling to variable-length deliberate chaining. Each step is a single lattice query or rotor application — no backprop, no autoregressive token loop.

| Component | Purpose |
|-----------|---------|
| `WorkingMemory` | Bounded scratchpad (capacity=8) of activated programs + partial conclusions |
| `StepOperator` | Decides next action: Retrieve / Transfer / Compose / Terminate |
| `Retrieve` | Query a structurally related group to diversify working memory |
| `Transfer` | Apply Cl(1,7) transfer rotor for analogical domain mapping |
| `Compose` | Sentence-level interleaving of working memory entries |
| `Terminate` | Coherence threshold (0.65) reached OR max steps (6) OR stall detected |
| `System2Config` | Configurable: max_steps, wm_capacity, coherence_threshold, min_activation |

**Biological analog:** Prefrontal working memory + inner speech. The StepOperator is a deliberate "select what to think about next" process, unlike the automatic wave settling of System 1.5.

Neuroscience-informed mapping:

| Brain Region | Growformer Component |
|-------------|---------------------|
| Medial Prefrontal Cortex | CognitiveMap (social/moral reasoning) |
| Temporoparietal Junction | UnderstandingLayer (theory of mind, semantic understanding) |
| Anterior Cingulate Cortex | MetaCognition (error monitoring, conflict detection) |
| Dorsolateral Prefrontal Cortex | System 2 WorkingMemory + StepOperator (deliberate reasoning, planning) |
| Inhibition | Causal Fingerprint gating (suppress irrelevant information) |
| Cross-cortical EEG Coherence | Neural Coherence Analysis (band-decomposed STA synchrony for ensemble composition) |

### Neural Coherence Analysis

Inspired by EEG coherence in neuroscience, where synchrony of neural oscillations across brain areas indicates functional connectivity. Growformer maps the Cl(1,7) SpaceTime Algebra grade structure to frequency bands:

| Band | STA Grade | Components | Analogy |
|------|-----------|------------|---------|
| δ (delta) | Grade 0 (scalar) | 1 | Global activation level alignment |
| θ (theta) | Grade 1 (vectors) | 8 | Intentional direction synchrony |
| α/β boost | Grade 2 (boost bivectors) | 7 | Temporal/causal coherence |
| α/β spatial | Grade 2 (rotation bivectors) | 21 | Structural/relational pattern synchrony |
| γ (gamma) | Grade 3 (trivectors) | 56 | Fine-grained semantic binding |

**Key neuroscience insight:** Coherence is primarily determined by the "sending" region (the source program's embedding quality), not just pairwise similarity. Programs with strong, well-defined grade-specific activations produce reliable coherence signals.

| Component | Purpose |
|-----------|---------|
| `BandCoherence` | Per-band coherence scores between two programs |
| `BandPower` | Signal strength per band for a single program (spectral power density) |
| `ensemble_coherence()` | Power-weighted mean pairwise coherence across a program set |
| `coherence_select()` | Greedy selection maximizing both relevance AND ensemble coherence |
| `coherence_matrix()` | Pairwise coherence matrix between topic sub-lattice centroids |

**Used for:** Broad query summarization — when a categorical question (e.g., "What is software architecture?") requires composing from multiple topic sub-lattices, coherence-guided selection ensures the chosen programs synchronize into a coherent ensemble rather than a disconnected list.

**Diagnostic gating:** If ensemble coherence falls below 0.15, the system recognizes it cannot compose a coherent multi-topic response and falls back to single-program retrieval. This mirrors how low cross-regional coherence in EEG indicates disconnected processing.

### MetaBrain Architecture

```mermaid
flowchart LR
    subgraph MICRO ["Micro-Brains (InfraciliaryLattice classifiers)"]
        TB["Topic Brain\nsemantic intents"]
        VB["Verb Brain\n8 classes"]
        AB["Action Brain\n5 classes"]
    end

    subgraph ARCH ["ArchetypeBrain"]
        PARA["InfraciliaryLattice\n(Paramecium)\nprograms across 14 groups\nwave propagation\ntrichocyst volley"]
    end

    subgraph COORD ["Coordinator"]
        CC["CentroidCoordinator\nnearest-neighbor fusion\n(replaced NeuralEnvironment)"]
    end

    TB -->|topic logits| CC
    VB -->|verb logits| CC
    AB -->|action logits| CC
    PARA -->|archetype + confidence| CC
    CC -->|"MetaResult:\nconditioning 192d\ngroup_idx, archetype_idx\ntopic, verb, action\nconfidence, volley"| OUT["Inference"]

    classDef micro fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    classDef arch fill:#fff3e0,stroke:#ef6c00,stroke-width:2px;
    classDef coord fill:#fce4ec,stroke:#c62828,stroke-width:2px;

    class TB,VB,AB micro
    class PARA arch
    class CC coord
```

### Data Groups (14 dynamic specialist domains)

Groups are dynamically discovered from unique `action_target` values in training data. Current layout:

| Group | Role | Gen Programs | Code Programs | Topic Sub-Lattices |
|-------|------|-------------|--------------|-------------------|
| g0 | Support/conversation | 65 | — | 11 |
| g1 | Search algorithms | 17 | 17 | 4 |
| g2 | Coding arithmetic | 21 | 18 | 13 |
| g3 | Data structures | 17 | 14 | 5 |
| g4 | Coding patterns | 90 | 56 | 17 |
| g5 | Sort algorithms | 28 | 24 | 5 |
| g6 | General conversation | 59 | — | 42 |
| g7 | Safety | 28 | — | 10 |
| g8 | Technical advice | 67 | — | 8 |
| g9 | Reasoning/composition | 18 | — | 19 |
| g10 | General knowledge | 149 | — | 31 |
| g11 | Architecture patterns | 140 | — | 14 |
| g12 | Design patterns | 44 | 20 | 11 |
| g13 | Multi-turn | 23 | — | 4 |

### Paramecium (InfraciliaryLattice)

The universal inference primitive — replaces all MLP/NeuralEnvironment components:

- **One-pass learning:** `develop()` creates behavioral programs from (embedding, text) pairs in O(n)
- **Wave propagation:** Input activates programs; activation spreads through the lattice
- **EMA centroid drift:** Program centroids adapt to input distribution (frozen after training)
- **Habituation:** Frequently-activated programs attenuate (novelty seeking)
- **Trichocyst volley:** Multi-program composition via ranked activation burst
- **Equivariant scoring:** Full STA multivector geometry for within-group discrimination

Used for: Router, ActionClassifier, Understanding Layer (topic/verb), ArchetypeBrain, all generation envs, all code envs, topic sub-lattices.

### Generation Architecture (Per-Group)

- **No autoregression:** Entire response predicted in a single forward pass
- **No backpropagation:** One-pass lattice development, O(n) training cost
- **Per-group isolation:** Each domain has its own `IndexedGenEnv`, `TokenDictionary`, and `AlgebraicCodebook`
- **Topic sub-lattices:** Within-group discrimination via operation-specific sub-indices
- **Forced topic routing:** Direct sub-lattice query when operation is known, bypassing cross-topic competition
- **Language-aware code gen:** Sub-lattice program filtering by target language markers
- **Algebraic codebook:** Responses clustered into archetypes (fixed tokens + variable slots)
- **Semantic token dictionary:** Vocabulary ordered by co-occurrence similarity + Gray coding
- **Hopf composition:** Multi-archetype responses via fragment mixing
- **Cloze learning:** Post-training contrastive reinforcement for slot inference

### E8 Lattice Quantization + Quantum Composition

The 128d bridge embedding decomposes into 16 × 8d subspaces, each quantized to the **E8 lattice** (densest sphere packing in 8d).

| Property | Value | Usage |
|----------|-------|-------|
| Dimension | 8 per subspace | Bridge 128d = 16 × 8d |
| Kissing number | 240 | Max equidistant archetype neighbors |
| Decoding | O(8) per subspace | Replaces O(n×d) cosine scan |
| Error correction | Extended Hamming [8,4,4] | 1-bit correction, 2-bit detection |

**Quantum group composition U_q(E8):** When multiple groups contribute, their E8 lattice points compose via quantum-deformed algebra with non-commutative R-matrix braiding.

### OCEAN Personality Conditioning

5-dimensional behavioral vector (each 0.0–1.0):

| Trait | Effect |
|-------|--------|
| **O**penness | Modulates Hopf cross-archetype composition (high → creative) |
| **C**onscientiousness | Biases slot prediction toward high-frequency vocab (high → precise) |
| **E**xtraversion | Influences archetype length selection (high → verbose) |
| **A**greeableness | Conditions toward affirming language patterns |
| **N**euroticism | Modulates confidence thresholds (high → cautious) |

### AI Safety Properties

1. **Bounded output space:** Codebook defines finite, enumerable response patterns. Content outside the codebook cannot be generated.
2. **Prompt injection resistance:** Adversarial inputs cause misrouting, not arbitrary text generation. Attack surface is "misselection among known patterns."
3. **Frozen deterministic inference:** Consolidated groups produce identical output for identical input across invocations, time, and hardware.

### Current Benchmark (~1600 samples, 14 groups)

| Component | Metric | Value |
|-----------|--------|-------|
| LearnedRouter | type | InfraciliaryLattice (K-NN + field gradient) |
| ActionClassifier | type | InfraciliaryLattice (one-pass) |
| MetaBrain coordinator | type | CentroidCoordinator (nearest-neighbor) |
| MetaCodebook | concepts | 20+ MetaConcepts with LanguageProjectors |
| Generation envs | groups | 14 gen + 6 code |
| Total programs | count | ~950 (gen + code) |
| Topic sub-lattices | count | ~200 across all groups |
| Cloze learning | accuracy | 94.6% slot fill (1470 games) |
| Brain checkpoint | size | ~81 MB |
| Training time | wall clock | ~27 min (one-pass, no epochs) |

Representative inference (single forward pass, no autoregression):

| Prompt | Response | Code |
|--------|----------|------|
| "write an addition function in Rust" | "Sum is the result of addition. Define a function with two parameters..." | `fn sum(a: f64, b: f64) -> f64 { a + b }` |
| "write a subtraction function in Rust" | "The difference of two numbers is computed by subtracting..." | `fn difference(a: f64, b: f64) -> f64 { a - b }` |
| "write a multiplication function in Rust" | "The product of two numbers is computed by multiplication..." | `fn product(a: f64, b: f64) -> f64 { a * b }` |
| "implement binary search in Python" | "Use two pointers converging toward the middle..." | `def binary_search(arr, target): lo, hi = 0, len(arr)-1 ...` |
| "help me reset my password" | "Password reset links expire after 30 minutes..." | — |
| "implement a stack using an enum in Rust" | "Use an enum with Box for heap allocation..." | `enum List<T> { Nil, Cons(T, Box<List<T>>) } ...` |

### Continual Learning (Split-MNIST, zero forgetting)

| Task | Digits | Accuracy | After All 5 Tasks | Forgetting |
|------|--------|----------|-------------------|------------|
| 0 | 0 vs 1 | 97.6% | 97.6% | 0% |
| 1 | 2 vs 3 | 96.7% | 96.7% | 0% |
| 2 | 4 vs 5 | 98.6% | 98.6% | 0% |
| 3 | 6 vs 7 | 97.1% | 97.1% | 0% |
| 4 | 8 vs 9 | 96.3% | 96.3% | 0% |
| **Avg** | | **97.3%** | **97.3%** | **0%** |

### File Map

| File | Component |
|------|-----------|
| `src/service.rs` | LanguageService: generation, codegen, meta-routing integration |
| `src/main.rs` | Training pipeline: all stages, cloze learning, brain export |
| `src/dimension/group_gen.rs` | IndexedGenEnv, topic sub-lattices, forced topic routing, equivariant scoring |
| `src/dimension/router.rs` | LearnedRouter (InfraciliaryLattice, K-NN + gradient bias) |
| `src/dimension/manager.rs` | DimensionManager, Clifford conditioning, structural fingerprints |
| `src/dimension/paramecium.rs` | InfraciliaryLattice (Paramecium): wave propagation, develop, respond |
| `src/growformer_lang.rs` | GrowformerLang: MetaConcept, MetaCodebook, LanguageProjector |
| `src/reasoning.rs` | CognitiveMap, ReasoningEngine (System 1.5), System 2 (WorkingMemory, StepOperator), transfer rotors |
| `src/metacognition.rs` | MetaCognition: Reflection Brain, ReflectionScores, graceful degradation |
| `src/coherence.rs` | Neural Coherence Analysis: band-decomposed STA coherence, ensemble selection, diagnostic gating |
| `src/cloze.rs` | Cloze learning: fill-in-the-blank games, contrastive drift |
| `src/understanding.rs` | UnderstandingLayer: topic/verb classifiers, goal_magnitude |
| `src/clifford.rs` | Cl(1,7) STA: multivectors, rotors, fingerprints, embed_bridge_vector |
| `src/meta_brain.rs` | MetaBrain: CentroidCoordinator, ArchetypeBrain fusion |
