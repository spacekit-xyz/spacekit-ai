# The Growformer Path to AGI

## A Different Scaling Law

The central claim of this document: **the path to AGI through Growformer is fundamentally different from the transformer path.** Transformers scale by making one brain bigger — more parameters, more compute, more data through the same monolithic architecture. Growformer scales by growing more specialist brains — structurally isolated, independently trained, algebraically composed.

The scaling laws are different because the architectures encode knowledge differently:

| | Transformer | Growformer |
|---|---|---|
| Knowledge encoding | Distributed across all weights | Localized in frozen engram topology |
| Add new domain | Retrain entire model | Grow one new group (~5–10 MB) |
| Training time per domain | Weeks on GPU clusters | Minutes on CPU |
| Forgetting risk | Catastrophic (requires RLHF, replay) | Zero (structural guarantee) |
| Deployment size | 4–175+ GB | 18 MB base + ~5 MB per domain |
| Scaling unit | Parameters (billions) | Groups (each with focused data) |
| Inference | Autoregressive (token by token) | Single forward pass (entire response) |

GPT-3 to GPT-4 wasn't a new architecture — it was the same architecture with more parameters and more compute. The knowledge remained distributed across billions of weights with no structural isolation. Every capability gained risked degrading another.

Growformer's scaling unit is the **specialist group**: a structurally isolated neural substrate with its own engrams, its own codebook, its own frozen topology. Adding "medical knowledge" doesn't mean retraining the whole system — it means growing a new group on medical data, freezing it, and letting the router dispatch to it. Existing groups are untouched.

The biological parallel is exact. The human brain doesn't have one giant homogeneous network. It has Broca's area for speech production, Wernicke's area for comprehension, the fusiform face area for face recognition, the hippocampus for memory formation — structurally distinct, functionally specialized regions. Each region is small. Intelligence emerges from their composition, not from any one region being enormous.

**AGI is a thousand small specialist brains with a good router, not one enormous brain with everything entangled.**

---

## Where We Are Today (March 2026)

The substrate works. The algebra works. The semantic dictionary works. Tool-use agents work. What's missing is breadth.

| Capability | Status | Evidence |
|---|---|---|
| Physics-based training | Complete | Neurons in 3D space, metabolic pruning, STDP, engram consolidation |
| Algebraic inference | Complete | E8 lattice quantization, Hopf composition, Hamming ECC, codebook factorization |
| Semantic token dictionary | Complete | Co-occurrence ordering so 1-bit errors land on synonyms; Observer pattern conf 0.57 → 0.92 |
| Zero forgetting | Complete | Structural guarantee — frozen groups receive zero gradient (Split MNIST: 97.3%, 0% forgetting) |
| Tool-use agent | Complete | Calculator, file reader, code runner, web search stub; g0-quality conversational wrappers |
| Deployable brain | Complete | 18 MB micro-brain — browser (WASM), mobile, IoT, edge |
| Base + augmentation model | Complete | Ship base brain, users train domain groups, structural isolation protects base |
| Production text quality | g0 only | Support responses at conf 0.93–0.97; identity at conf 1.00 |
| Multi-domain generation | Partial | g1 improving (Observer at 0.92) but complex compositions still low-confidence |

The bottleneck is **data density**, not architecture. ~350 samples across 2 groups. For AGI-class coverage: 10–20 groups, 500–2000 samples per group, across math, science, law, medicine, finance, creative writing, reasoning, planning, conversation, and instruction-following.

---

## Roadmap to AGI

### Phase 1 — Immediate (unblocks everything else)

#### 1. Data density across domains

The single biggest bottleneck. The group infrastructure is ready (`action_target` routing, `--auto` configuration, group splitting). What's needed:

- **10–20 specialist groups** (currently 2)
- **500–2000 samples per group** (currently ~175 per group)
- Domain coverage: mathematics, physics, biology, law, medicine, finance, creative writing, logical reasoning, planning/strategy, general conversation, instruction-following, code (Python, Rust, JS, Go, etc.)
- Each new domain group adds ~5–10 MB and trains in minutes on CPU

#### 2. Context window expansion

Currently 3-turn context with a fixed 64d conditioning vector. This limits reasoning depth and multi-step problem solving.

- **Short-term**: Increase turn window (3 → 10+), expand conditioning dimension (64 → 128+)
- **Long-term**: Episodic memory retrieval — the Leech lattice (24d) already supports spatial nearest-neighbor queries; use it to retrieve relevant past interactions as conditioning context

### Phase 2 — Near-term (1–3 months)

#### 3. Continuum — train-while-on

The ephaptic field provides immediate learning (1-sample adaptation). Full Continuum extends this to persistent online learning:

- **Feedback API**: User corrections become training signals — "that answer was wrong, the correct answer is X" triggers targeted retraining of the responsible group
- **Online training trigger**: Low-confidence responses automatically queue the group for retraining with the new data point
- **Checkpoint-on-schedule**: Periodic brain snapshots during production use
- This is where the system starts **growing in production**, not just at train time

#### 4. Schema abstraction — shared knowledge templates

When multiple groups converge on similar engram patterns, promote the common subgraph into a reusable schema. This is how general knowledge emerges:

- "Argument structure" is shared across law, debate, and science
- "Causal reasoning" is shared across medicine, engineering, and physics
- The schema captures the common form; groups specialize the content
- New groups receive schema-initialized weights, accelerating training
- Biological analog: hippocampal-to-neocortical consolidation

#### 5. Tool ecosystem expansion

The 4 built-in tools are a starting point. AGI requires:

- **Web browser** (fetch + parse + navigate)
- **Database queries** (structured data retrieval)
- **API orchestration** (chain multiple tool calls into workflows)
- **Memory read/write** (persistent long-term storage beyond engrams)
- **Reasoning tools** (symbolic math solver, logic engine, constraint satisfaction)
- **Sensory input** (image/audio processing via specialized groups)

### Phase 3 — Medium-term (3–6 months)

#### 6. World Models — predict, plan, act

The jump from "responsive agent" to "planning agent." The existing architecture maps directly:

- **Main Dimension** → world state representation
- **Mirror Dimension** → forward model (predict next state given action)
- **VirtualGroup** → planning (compose multiple specialist predictions into action sequences)
- **Policy engine** → action selection (which group to activate, which tool to use)

This is where the Growformer stops being reactive and starts being proactive — predicting outcomes, evaluating alternatives, and executing multi-step plans.

#### 7. Organogenesis — live specialist spawning

When Continuum feedback or persistent low-confidence indicates a genuinely new domain:

- Auto-spawn a new Mirror Dimension
- Train on the incoming data stream
- Promote to consolidated specialist when quality thresholds are met
- The auto-spawn trigger already exists (K=10 consecutive low-confidence batches)
- Together with Continuum: the system grows new organs on demand, autonomously

#### 8. Nilpotent convergence bounds — provable training guarantees

If the substrate's layered dynamics are nilpotent (each layer = one step in the lower central series), conjugator length bounds from quantitative geometric group theory provide **provable upper bounds** on training steps needed per knowledge state.

This transforms training from "run until loss plateaus" into "train for exactly N steps, guaranteed." Critical for production deployment where training time must be predictable.

### Phase 4 — Long-term (6–12 months)

#### 9. AI Operating System

A meta-controller that manages the ecosystem of specialist brains:

- Route between brains based on task type, confidence, and context
- Policy layer for safety constraints and access control
- Memory management across brains (shared episodic store, schema library)
- The multi-brain infrastructure already exists (`load_brain_as`, `set_active_brain`, `/v1/brains` API)
- The OS layer adds orchestration, scheduling, and resource management

#### 10. Cryptographic verification — provable inference

Torsors + Σ-protocols for zero-knowledge proofs of correct inference. Combined with SIS-based vector commitments for quantum-resistant authenticated knowledge stores.

- **Brain marketplace**: Users can verify that a brain produces certified outputs without seeing the brain's internals
- **Auditable Continuum trails**: Every learning step is cryptographically committed
- **Federated trust**: Organizations share specialist brains with provable properties

---

## Why This Path Is Viable

Three properties make the Growformer AGI path credible rather than aspirational:

**1. The substrate already works at production quality.** g0 produces text that's indistinguishable from hand-written responses. The algebraic generation pipeline (E8 quantization → codebook → Hopf composition → ECC) is mathematically grounded and empirically validated. This isn't a research prototype — it's a working system that needs more data, not more architecture.

**2. Scaling cost is linear, not quadratic.** Each new specialist group is independent. Training group N+1 doesn't touch groups 1 through N. There's no attention mechanism with O(n²) cost. There's no full-model retraining. The cost of going from 2 groups to 200 groups is 200× the cost of training one group — minutes on CPU, not months on GPU clusters.

**3. Zero forgetting is structural, not aspirational.** This isn't "we've reduced forgetting to 3%" — it's "frozen groups receive zero gradient, architecturally guaranteed." Every domain added is permanent. Every capability gained is preserved. The system only grows; it never regresses. This is the foundation that makes scaling to AGI feasible: you can build incrementally, one domain at a time, knowing that each step is permanent.

The question isn't whether the architecture can support AGI. The question is whether we can produce enough high-quality training data across enough domains. That's a data engineering problem, not a research problem. And it's a data engineering problem that gets easier as the tool ecosystem expands — because the agent itself can help generate, curate, and validate training data for its own new groups.

---

## One-Line Summary

**Growformer's path to AGI: grow a thousand small specialist brains on focused data, compose them algebraically, and let the system spawn new specialists autonomously as it encounters new domains.**
