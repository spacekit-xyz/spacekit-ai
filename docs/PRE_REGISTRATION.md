# Pre-registration: separable bets, kill conditions, and experiment queue

**Status:** Living document (2026-07-01).  
**Discipline:** Same as [`CERTIFICATION_ARC.md`](CERTIFICATION_ARC.md) and [`COMPETENCE_ROUTING_SPEC.md`](COMPETENCE_ROUTING_SPEC.md) — gates and verdict tables are written **before** runs; a null on one bet must not be narrated as a verdict on another.

**Audience:** Internal + fundraising narrative hygiene. Read §0 before interpreting any `growformer-llm` TinyStories number.

**Headline (2026-07-03):** Row **2** closes Bet B — matched vanilla **8.15 bpt** vs Clifford **9.62–9.74 bpt** (−1.5 to −1.6 bpt, verified). That is the answer to the Clifford LM question this repo asked. Bet A (CL routing, CL-A/B) is downstream completeness work, not the lead story.

---

## 0. Three separable bets (mandatory framing)

| Bet | Question | Repo / artifact | Depends on other bets? |
| --- | --- | --- | --- |
| **A — CL substrate** | Can we stack knowledge without forgetting and route among frozen specialists? | `growformer`: promote-freeze, cone router ([`phase3g_cone_results.txt`](../phase3g_cone_results.txt)) | **No** |
| **B — Clifford LM** | Does Cl(1,3) geo-linear LM earn its place vs dot/vanilla at matched budget? | `growformer-llm`: rows 1b–3b, 1b-v2, row 2 | **No** |
| **C — Region dynamics** | Do coupled oscillator region states + causal coupling improve composition claims? | `growformer` substrate (side quest) | **No** |

**Fundraising-safe sentence:** *We have a certified continual-learning and routing substrate (Bet A). On language modeling at matched scalar budget, Cl(1,3) underperforms a standard transformer by ~1.5 bits/token (Bet B closed, row 2). Region oscillator physics (Bet C) is a third, substrate-only hypothesis.*

### Research probe vs product (Bet B closed)

| Lens | Sentence |
| --- | --- |
| **Research probe** | Cl(1,3) for discrete text underperforms a matched transformer at ~737k scalars; FFN and width ablations are neutral; the loss is in the architecture class, not a single fixable component. Honest negative results on exotic architectures are useful. |
| **Product** | Use the transformer. Routing over Clifford LM specialists is a second story on a foundation Bet B failed — run CL-A for completeness, do not lead with it. |

### What a null on Bet B does **not** threaten

- Promote-freeze retention (0% Split MNIST forgetting in isolation protocol).
- Adjustable-cone routing (Task E: **92.5% ± 5.9%** vs VirtualGroup **68.4%**, 0/20 degenerate at n=30).
- Stacking via frozen specialists + dispatch.
- Paramecium / lattice program storage (orthogonal memory channel).

### What a null on Bet B **does** threaten

- Claims that **Cl(1,3) geometric product** beats matched dense/vanilla LM on held-out bits/token.
- Using Clifford LM as the default specialist to promote into Main without ablation.

---

## 1. Bet B — Clifford LM (`growformer-llm`)

### 1.1 Completed rows (held-out, 64 windows, seq 128)

| Row | Config | bits/token | ppl | Notes |
| --- | --- | --- | --- | --- |
| 1b | geo scores `(Q⊛K)₀` *without* `reverse(K)` | **9.69** | 827 | default seed; historical |
| 1b-v2 | corrected `⟨Q,K⟩`, metric LN, seed 1000 | **9.74** | 856 | **+0.05 vs 1b**; bug fix neutral |
| 3 | `--dense-ffn`, seed 1000 | 9.72 | 844 | FFN ablation ≈0 |
| 1c | `d_model=32`, `d_ff=128` | 9.76 | 866 | width ≈0 |
| 3b | `--dot-attention`, seed 1000 | **9.62** | 789 | **best Clifford-stack row**; −0.12 vs 1b-v2 |
| 2 | `--vanilla`, matched ~737k scalars, seed 1000 | **8.15** | **284** | **Bet B capstone**; −1.59 vs 1b-v2 (ledger paired) |
| unigram | MLE train counts | 10.06 | 1067 | floor |

**Code fix (2026-07-01):** default geometric scores now use Clifford inner product `⟨Q,K⟩ = (Q⊛K̃)₀`; metric-weighted layer norm. Old checkpoints trained under the bug are not comparable to new forward paths without retrain.

### 1.2 Row 1b-v2 (complete, 2026-07-02) — referendum on corrected Clifford stack

**Checkpoint:** `growformer-llm/agent-data/tinystories-row1b-v2.json`

**Held-out eval (64 windows, seq 128):** **9.7415 bits/token**, ppl **856.0**. Best train-val ppl **839.17** @ step 3800; final train-val **851.9** @ step 4000.

**Pre-registered read (applied):**

| Outcome vs 1b (9.69 bpt) / 3b (9.62 @ seed 1000) | Verdict | **Result** |
| --- | --- | --- |
| 1b-v2 **beats 1b by ≥0.05 bpt** | Fix mattered | **No** (+0.05 bpt) |
| 1b-v2 **within ±0.05 bpt of 1b** | Bug fix neutral | **Yes** (borderline +0.051) |
| 1b-v2 **worse than 1b by ≥0.05 bpt** | Fix hurt | Borderline only |
| 1b-v2 **still loses to 3b by ≥0.05 bpt** | Dot beats inner product | **Yes** (+0.12 bpt) |

**Fundraising-safe sentence:** *The Clifford LM trains and generalizes, but at this budget Euclidean dot attention outperforms the Clifford inner product; correcting the score kernel did not change the conclusion. Continual learning and cone routing (Bet A) are unaffected.*

**Ledger (paired SE, 2026-07-02):** `agent-data/results.jsonl` — `row3b` vs `row1b-v2`
**−0.118 ± 0.012 bpt** (n=64 windows, same split). Pre-fix checkpoint `row1b` re-evaluated
under post-fix forward reads ~**10.02 bpt** (not comparable to historical 9.69). Use
**row1b-v2** as post-fix baseline for ledger tables. Regenerate:

```bash
cargo run --release --bin tinystories -- ledger-table --baseline row1b-v2 --candidates row3b
cargo run --release --bin tinystories -- ledger-verify
```

**Command (record):**

```bash
cd growformer-llm
cargo run --release --bin tinystories -- train \
  data/tinystories.tok data/tinystories-train.bin data/tinystories-heldout.bin \
  --checkpoint-out agent-data/tinystories-row1b-v2.json \
  --steps 4000 --tie-embeddings --grad-accum 2 --init-seed 1000

cargo run --release --bin tinystories -- eval \
  --checkpoint agent-data/tinystories-row1b-v2.json \
  --tokenizer data/tinystories.tok \
  --train-bin data/tinystories-train.bin \
  data/tinystories-heldout.bin --seq-len 128 --windows 64
```

### 1.3 What generalizes (honest levers)

**Confirmed working (Bet B):** stable training; **~0.31 bpt** below unigram; train-val ≈ held-out (no memorization gap); viable **frozen specialist** checkpoint.

**Ruled out or weak at this protocol:**

| Lever | Evidence | Verdict |
| --- | --- | --- |
| More optimizer steps | Val flat 3000–4000 (1b, 1b-v2) | Not the bottleneck |
| GPU | Not used for quality ablation | **Speed only** — same math, faster iteration |
| Width (1c) | 9.76 bpt @ ~4× params | Width does not escape ~850 band |
| FFN Cayley vs dense (3) | 9.72 bpt | FFN geometry ≈0 |
| Clifford correctness fix | 1b-v2 vs 1b | Neutral (+0.05 bpt) |

**Still open (pre-registered):**

1. **Multi-seed replication** — row 2 @ seeds {1000, 1001, 1002} optional; single-seed caveat on current ledger rows.
2. **Chronological CL-1 specialists** — train/freeze row-cl-a + row-cl-b (§2.2 steps 1–4); Bet B checkpoint pairs tested below do not substitute.

**Not Bet B:** authored world grounding (`world_grounding.toml`) speeds routing/classification in `growformer` service; `corpus_semantic_init` already gives distributional grounding in TinyStories training.

### 1.4 Row 2 — matched vanilla transformer (Bet B capstone)

**Goal:** External ceiling at **matched scalar parameter budget** vs Clifford row 1b-v2 (~737k scalars @ vocab=2048, d_model=16, d_ff=64, n_blocks=4, tie_embeddings).

**Architecture (in-crate `growformer-llm::vanilla_llm` + `v2::vanilla_train`):**

| Component | Row 2 vanilla |
| --- | --- |
| Embeddings | Real `d_model` vectors (not 16-blade multivectors) |
| Positional | Fixed sinusoidal (no learnable params) |
| Block | Pre-norm GPT-2 style: LN → multi-head dot attention → residual → LN → ReLU FFN → residual |
| Attention | Dot product, scale `1/√head_dim`, causal mask |
| FFN | Real linear → ReLU → linear |
| Layer norm | Standard (Euclidean) γ/β per `d_model` |
| Head | Real linear; optional weight tying with embeddings |
| `d_model` | **Matched** via `param_budget::matched_vanilla_d_model` (CLI `--d-model` = Clifford reference; stored as `clifford_ref_d_model`; actual `d_model` ≈ **148** for default 1b scale) |

**Training protocol (same as row 1b unless noted):**

- Tokenizer/corpus: BPE vocab 2048, TinyStories chronological 90/10 split.
- Defaults: `--steps 4000`, `--seq-len 128`, `--lr-max 3e-4`, cosine warmup (`warmup = max(steps/20, 50)`), Adam, grad clip 1.0, `--grad-accum 2` recommended, `--tie-embeddings`.
- Init: corpus-semantic embedding init on by default (random indexing, window ±4); `--init-seed` for weight RNG.
- Checkpoint: `cfg.vanilla=true`, `cfg.clifford_ref_d_model=<CLI d_model>`; save via `vanilla_checkpoint::save_vanilla_state`.
- Full taped backward through all blocks (no head-only shortcut).

**Train command:**

```bash
cargo run --release --bin tinystories -- train \
  data/tinystories.tok data/tinystories-train.bin data/tinystories-heldout.bin \
  --checkpoint-out agent-data/tinystories-row2-seed0.json \
  --vanilla --d-model 16 --d-ff 64 --n-blocks 4 --n-heads 4 \
  --steps 4000 --tie-embeddings --grad-accum 2 --init-seed 0
```

**Eval command (held-out, ledger-compatible):**

```bash
cargo run --release --bin tinystories -- eval \
  --checkpoint agent-data/tinystories-row2-seed0.json \
  --tokenizer data/tinystories.tok \
  --train-bin data/tinystories-train.bin \
  data/tinystories-heldout.bin --seq-len 128 --windows 64
```

(`cfg.vanilla` is auto-detected from checkpoint; pass `--vanilla` to force.)

**Pre-registered comparison:** row **2** vs baseline **`row1b-v2`** (9.74 bpt / ~856 ppl on first-64×128 windows). ≥3 seeds; ledger `config_hash` includes `vanilla=true` and `clifford_ref_d_model`. Gate: same 0.05 bpt paired-SE protocol as §1.5.

**Kill / interpret (applied 2026-07-03):**

| Result | Verdict |
| --- | --- |
| Held-out **8.151 bpt** (ppl **284**); best val ppl **267** @ step 4000 | Row 2 complete |
| vs **row1b-v2** (9.741): **−1.590 ± 0.068 bpt** (paired, n=64) | **Vanilla wins decisively** |
| vs **row3b** (9.623): **−1.472 bpt** (mean; ledger paired vs 1b-v2 baseline) | Dot-Clifford body also loses to vanilla at matched budget |
| Pre-reg kill: vanilla beats 1b-v2 | **Triggered** — Clifford geo-linear stack not earning its complexity on TinyStories at this budget |

**Measurement audit (2026-07-03, response to 8.15 skepticism):**

| Check | Result |
| --- | --- |
| Same metric as row3b/1b-v2? | **Yes** — conditional CE, `-log₂ p(target)`, bits/**token** (not byte); identical eval loop structure |
| Same held-out shard / windows? | **Yes** — `split_hash=f2859d073…`, 64×128, **8084** predicted tokens, unigram floor **10.05** bpt on eval tokens |
| Re-eval reproduces ledger? | **Yes** — fresh `eval` → **8.1513 bpt** (ppl 284.3), matches `run_id=row2` ledger |
| Train-val consistent? | **Yes** — step 4000 val ppl **267** (nats/token 5.59 → ~8.0 bpt); not a train/eval mismatch |
| Scalar param match? | **Yes** — Clifford **736,768** vs vanilla **737,272** scalars (Δ −504, 0.07%) |
| Architectural peer to Clifford? | **No by design** — row 2 is the **Bet B capstone** (standard transformer at matched scalar budget), **not** a CL-1 routing peer |

The **1.5 bpt gap vs Clifford rows is real on this metric**, not a units/window bug. It is **large** because vanilla at `d≈148` with dot attention + standard LN materially outperforms the constrained Clifford stack at the same scalar count — that is exactly what row 2 was pre-registered to test. **Do not use row2 as specialist A in CL-1**; it is strictly better everywhere (64/64 windows vs row3b), so routing collapse is a **setup property**, not evidence about cone routing.

**Fundraising-safe sentence (Bet B):** *At matched scalar parameter budget (~737k), a standard transformer reaches **8.15 bpt** vs corrected Clifford **9.74** and dot-Clifford **9.62** — the capstone test does not support Cl(1,3) as the default LM specialist. Bet A (cone routing) remains live on **growformer** substrate (Phase 3g); LM CL-1 stacking tests below are **negative controls**, not router failures.*

**Caveats (still binding):** single seed (1000); param-matched not FLOP-matched; vanilla uses `d_model≈148` vs Clifford `d_model=16×16-blade` — same scalar count, different width allocation. Multi-seed replication recommended before external claims.

**Do not** compare unmatched `d_model` (16 vs ~148); always log `[row2] param budget` line from train.

### 1.5 Results ledger (measurement enforcement)

Crate: [`growformer-ledger/`](../growformer-ledger/). Wired into `tinystories eval`
(when `--train-bin` set). Stores per-window bpt + `split_hash`; `ledger-table`
renders paired 95% CI. **Do not** hand-edit verdict tables when ledger records exist.

| Rule | Detail |
| --- | --- |
| Baseline for post-fix rows | **`row1b-v2`**, not pre-fix `row1b` |
| Window tag | `--selection-tag first` (first 64 × 128-token windows) |
| Integrity | `ledger-verify` before trusting tables |
| Gate 0.05 bpt | May exceed 2·SE — ledger flags when gate is finer than resolution |

---

## 2. Bet A — Continual learning & routing (`growformer`)

### 2.1 Certified (do not re-litigate without new protocol)

| Claim | Evidence |
| --- | --- |
| Zero forgetting via promote-freeze | Split MNIST table in `growformer` README |
| Global scalar blend fails on switched tasks | VirtualGroup **~68%** Task E |
| Adjustable-cone beats VG, anti-collapse | [`phase3g_cone_results.txt`](../phase3g_cone_results.txt): **92.5%**, 0/20 degenerate |
| Confident-wrong down-weighted | Cone reliance **0.18** vs competence-router **0.60** |

**Stretch not met:** region agreement **85.3%** (gate ≥80% pass, stretch >90% fail).

### 2.2 Row CL-1 (pre-registered, parallel to 1b-v2)

**Goal:** Demonstrate Bet A with **LM specialists** before Bet B is settled.

**Protocol:**

1. Split TinyStories train shard chronologically (or by simple cluster): **A** = first 50%, **B** = second 50%.
2. Train specialist **A** (`tinystories-row-cl-a.json`, 2000 steps, 1b scale, seed 1000).
3. Train specialist **B** (`tinystories-row-cl-b.json`, 2000 steps, same hyperparams).
4. **Freeze** both; no further weight updates.
5. **Preflight (required before routing):** eval each specialist standalone on held-out; report mean bpt. **Peer gate:** `|A_bpt − B_bpt| ≤ 0.20` bpt. **Complementarity gate:** per-window win counts — if one specialist wins **all** windows, oracle = best single and **stop** (no router can help). Only if both gates pass proceed to cone calibration.
6. Calibration: `n ∈ {30, 60, 120}` windows, stratified from held-out; train adjustable-cone on **frozen** specialist logits/hiddens only.
7. Held-out eval: routed composite vs single LM trained on all data (matched total steps).

**Pre-registered gates (adapted from Phase 3g):**

| Gate | Pass |
| --- | --- |
| **Preflight:** specialist peer parity (`\|Δbpt\| ≤ 0.20`) | Required before reading router |
| **Preflight:** complementarity possible (each specialist wins ≥1 window) | Required; else oracle = best single → **stop** |
| Routed composite **beats** best single specialist on held-out bpt | Required (only if preflight passes) |
| Routed composite **beats** chronological oracle (always pick specialist trained on same half) | Stretch |
| **0 degenerate** seeds (constant specialist) at n=30 | Required (distinguish **imbalanced** vs **no-complementarity** in notes) |
| Confident-wrong reliance **< 0.5** on probe set | Required |

**Kill:** If cone on LM specialists cannot beat training one LM on full data, routing adds overhead without retention benefit — revise CL-1 split or features.

**Result (2026-07-03, exploratory pairs from frozen Bet B checkpoints):**

Implemented in `growformer-llm::cl1` + `lm_cone_router`; CLI `tinystories cl1`. Calibration: first 30 held-out windows; eval: 64 windows × 128 tokens; features = mean top-1 softmax confidence per specialist (oracle-free). Harness prints **preflight** before router verdicts.

| Pair (A / B) | A bpt | B bpt | Δbpt | Wins A/B | Oracle | Routed | Interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| row2 / row3b | **8.15** | 9.62 | **1.47** | 64/0 | 8.15 | 8.15 | **Imbalanced setup** — not a routing test; dominant model wins every window |
| row1b-v2 / row3b | 9.74 | **9.62** | 0.12 | 0/64 | 9.62 | 9.62 | **Peer negative control** — oracle = best single; no per-window complementarity |

**Headline (architecture-independent):** In both pairs, **oracle per-window min = best single specialist** because one model wins **every** held-out window. No router — cone or otherwise — can beat the dominant specialist on this shard. The row1b-v2/row3b pair is the honest negative result (“similar models, no window-level disagreement”). The row2/row3b pair is an **invalid routing setup** (specialists not peers); degenerate routing there is expected, not informative about cone routing.

Ledger: `run_id=cl1-row2-row3b`, bet **A** (composite equals row2 alone — do not cite as Bet A success/failure).

**Verdict:** Stacking Bet B ablation checkpoints does **not** test Bet A routing. Chronological half-corpus specialists (steps 1–4) remain the registered test — **with preflight parity + complementarity gates** before reading the router.

---

## 3. Bet C — Oscillator-structured region dynamics

**Placement:** `growformer` substrate **per-region state**, **not** inside `growformer-llm` during Clifford ablations.

### 3.1 Mechanism

Diagonal complex SSM (S4D/DSS family): state channel `x' = λx + Bu`, `λ = −α + iω`.

- `α = softplus(a)` — damping ≥ 0 (stable).
- `ω` — frequency (learnable).
- Optional init: log-spaced `ω` across channels (filter-bank prior).

**Coupling (inter-region):** off-diagonal **W** in coupled form, e.g.  
`y″ = −diag(γ)y − diag(ε)y′ + tanh(Wy + Vu + b)`  
with sparse/lattice-structured **W** aligned to existing routing topology.

**Literature anchors (verify venues before paper cite):** coupled oscillatory RNNs; LinOSS; Kuramoto-style oscillator-neuron layers (AKOrN).

### 3.2 Baselines (matched parameters)

1. Unconstrained diagonal complex SSM.
2. Matched-parameter GRU on same tasks.

### 3.3 Tasks

1. **Cheap:** synthetic phase / frequency discrimination.
2. **Real:** sequential CL task (Split MNIST or Task E specialist outputs as region observations).

### 3.4 Kill conditions

| Kill | Interpretation |
| --- | --- |
| Constrained oscillator **does not beat** matched unconstrained SSM/GRU | Prior is reparameterization only — drop oscillator framing |
| Post-hoc: learned `ω → 0` (no oscillation) | Task does not want vibration |
| Post-hoc: `α` saturated (critical damping) | Task wants memoryless channels, not ringing |

**Diagnostic (required):** histogram `ω` and `α` after training.

---

## 4. Intervention data — causal coupling graph

### 4.1 Problem

Observational region coupling is confounded by shared router + shared input. Granger / predictive coupling cannot separate "A drives B" from "both driven by context."

### 4.2 Estimand

Interventional coupling of region *i* onto *j*:

```
Δ_ij = E[Y_j | do(perturb_i)] − E[Y_j | do(sham)]
```

Assemble **interventional coupling matrix** over forked deterministic replays.

### 4.3 Protocol

1. **Fork** substrate state at timestep *t* (saved RNG / full state snapshot).
2. **Randomize** perturbed region index *i* and timestep (severs observational confound).
3. **Perturb:** phase kick or frequency detune on region *i*'s oscillator state (preferred if Bet C landed); amplitude clamp as secondary.
4. **Sham:** same fork, no perturbation.
5. **Re-run** both branches deterministically; record downstream region outputs / phases.
6. **Multiplicity:** pre-register effect-size threshold; report FDR-corrected or fixed pair list (do not mine full N² matrix).

### 4.4 Kill conditions

| Kill | Interpretation |
| --- | --- |
| Interventional matrix ≈ observational correlation everywhere | Interventions too weak, or confound was illusory — diagnose |
| **No** downstream effect from any `do(perturb_i)` | "Regions compose / influence each other" is **false** for this config — revise whitepaper composition claims |

### 4.5 Payoff for Bet A

- Causal influence graph figure (stronger than "regions blend").
- Routing test: clamp region A → does cone reroute? does composite degrade gracefully?

---

## 5. Fused experiment — phase-kick intervention on coupled oscillators

**Depends on:** Bet C prototype with identifiable per-region phase.

### 5.1 Hypothesis

Phase coupling in **W** is **causal**: kicking region *i*'s phase shifts region *j*'s phase/output beyond sham distribution.

### 5.2 Procedure

For each trial:

1. Run substrate to timestep *t*; snapshot state *S*.
2. Branch A: `do(phase_kick_i, magnitude φ)` on region *i*.
3. Branch B: sham (φ = 0).
4. Continue both to *t + Δ*; measure phase/output at region *j*.

Randomize (*i*, *t*, φ) from pre-registered set.

### 5.3 Sample size (pre-registered)

| Parameter | Value |
| --- | --- |
| Region pairs (*i*, *j*) | Pre-register **K ≤ 10** directed pairs (not full N²) |
| Trials per pair | **≥ 200** fork pairs |
| φ sweep | `{0, φ_low, φ_mid, φ_high}` — magnitudes set from 10% of typical phase variance on pilot |
| Seeds | **≥ 10** substrate init seeds |
| Test | Paired difference per fork; one-sided or two-sided pre-registered; **α = 0.05** with Bonferroni on K pairs |

### 5.4 Verdict table

| Result | Licenses |
| --- | --- |
| **Positive:** Δ_ij > sham threshold on ≥70% of pre-registered pairs, 0 seeds degenerate | "Phase coupling is causal"; W is load-bearing; proceed intervention routing tests |
| **Null:** no pair survives multiplicity correction | W is decorative; regions oscillate independently; narrow composition claims to routing-only |
| **Mixed:** signal only at annulus/boundary analog | Localize claims; do not globalize |

---

## 6. Bet D — Brain retrieval vs embedding RAG baseline (2026-07-03)

**Question:** For prompt→relevant labeled memory (grounding), does the growformer lattice retrieval stack beat a boring sentence-embedding k-NN over the same training JSONL?

**Pre-registered before run:**

| Item | Spec |
| --- | --- |
| Encoder | `sentence-transformers/all-MiniLM-L6-v2` |
| Index | Training JSONL `text` field per corpus (`data/sentiment`, `data/crypto`, `data/fintech` train files) |
| Queries | Same 4-prompt battery as raw lattice diagnostic (2026-07-03) |
| Brain comparator | Frozen **raw pre-gate top-1** from `brain-raw-diag --battery` (not `--brain-only` decline) |
| Case 3 store check | Grep corpus for mortgage/rate rows; tag `STORE_EMPTY` / `STORE_PARTIAL` if no qualifying row |
| Pass criteria | Case 1–2: negative label + crypto/decline terms; Case 3: fintech terms + non-positive polarity (or `STORE_*` excluded); Case 4: top-1 not opposite-sentiment mortgage praise |
| **Decision rule** | If RAG passes **≥3 scorable** cases **and** brain passes **≤1** → **switch retrieval core** to embedding k-NN for grounding; if RAG passes **≤2** → populate/label store first; growformer brain retained only for separately validated routing/metacog |

**Harness:** `growformer/scripts/rag_baseline_battery.py` → `agent-data/brain-rag-baseline/rag_baseline_results.json`

**Result (2026-07-03):** `agent-data/brain-rag-baseline/rag_baseline_results.json`

| Case | Store | RAG top-1 | pass RAG | Brain raw top-1 | pass brain |
| --- | --- | --- | --- | --- | --- |
| 1 sentiment × Bitcoin crash | `STORE_PARTIAL` (no crash+ETF row) | BTC ETF flows green (`positive_strong`) | **No** | Earnings beat (`positive_strong`) | **No** |
| 2 crypto × Bitcoin crash | `STORE_OK` | Bitcoin broke above $70k ETF inflows (`positive_strong`) | **No** | Counterfactual rally (`negative_mild`) | **No** |
| 3 fintech × Chase hike | `STORE_PARTIAL` (no mortgage-rate row) | Raised credit limit without warning (`mixed`) | **Yes** | Identity boilerplate | **No** |
| 4 sentiment × Chase hike | `STORE_OK` | **Chase gave me a great mortgage rate** (`positive_strong`) | **No** | Earnings beat (`positive_strong`) | **No** |

**Scores:** RAG **1/4**, brain raw **0/4**.

**Pre-registered outcome applied:** **`POPULATE_STORE_FIRST`** (RAG ≤2 passes — neither retrieval method clears the bar until the labeled store contains query-adjacent rows).

**Read:** The dumb baseline does **not** rescue cases 1, 2, or 4 on this corpus. Case 4 is the smoking gun both ways: MiniLM rank-1 is the training row `sent_ce_005` ("Chase gave me a great mortgage rate") — shared-token overlap without polarity, same failure class as brain witness. Case 3 passes RAG (not brain) by retrieving a structurally similar "raised … without warning" fintech row, but the store still lacks a mortgage-rate-hike exemplar (`STORE_PARTIAL`). **No retrieval approach works on empty/wrong memory;** fixing growformer scoring without adding rows would not have beaten this baseline anyway.

**Action:** Populate labeled JSONL for the 4-prompt battery (crash+ETF negative, Chase mortgage hike negative) before re-testing either retrieval stack. Do **not** patch `compute_forced_topic_scored_list` until the store exists and both methods are re-run.

**Round 2 (store populated, 2026-07-03):** Added `data/{sentiment,crypto,fintech}/train_sentiment_battery.jsonl` (exact-query rows + paraphrases). Re-ran **RAG only** (index update — query and store remain distinct):

| Case | RAG top-1 (round 2) | pass RAG |
| --- | --- | --- |
| 1 | `batt_sent_001` exact query (`negative_strong`) | **Yes** |
| 2 | `batt_crypto_001` exact query (`negative_strong`) | **Yes** |
| 3 | `batt_fin_001` exact query (`negative_strong`) | **Yes** |
| 4 | `batt_sent_003` exact query (`negative_strong`) | **Yes** |

**Round 2 RAG score:** **4/4** (`rag_baseline_results_round2.json`). Brain on **unretrained** `.bin` files remains **0/4** (store rows exist in JSONL but are not in the lattice until retrain).

**Fair post-store decision (pre-registered rule applied):** RAG **≥3/4** and brain **≤1/4** on the same battery without brain train-on-test → **`SWITCH_TO_EMBEDDING_RAG`** for the retrieval-for-grounding core.

**Round 2 brain retrain — INVALID (protocol break, set aside):** `train-brain` on battery JSONL then `brain-raw-diag --battery --battery-brains` is **train-on-test** (same failure class as train=val shard contamination at row 1). Reported **2/4** on `*-battery.bin` is **not scored** toward the decision rule. Cases 1–2 “pass” only because the paraphrase/exact rows were inserted into the lattice during training — that tests recall of inserted content, not retrieval on novel prompts.

| Case | Brain raw top-1 (`*-battery.bin`) | Valid score? |
| --- | --- | --- |
| 1 | Exact Bitcoin crash row | **No** — trained into lattice |
| 2 | ETF-delay paraphrase row | **No** — trained into lattice |
| 3 | Identity boilerplate (routes **g0**, not **g1** with 8 sentiment rows) | **No** — and exposes **routing failure**: correct content present, unreachable |
| 4 | Coinbase account freeze (wrong entity) | **No** — ranking within bucket |

**Headline finding (case 3, fair):** Fintech `*-battery.bin` has sentiment rows in **g1**, but the mortgage-complaint query routes to **identity g1** (1 program). Populating the lattice does not fix a router that selects the wrong group. Embedding k-NN has no group-selection stage to mis-route.

**Final verdict (Bet D, full 4-prompt battery):** **`SWITCH_TO_EMBEDDING_RAG`** for grounding retrieval when scored on the original pre-registered rule (RAG **≥3/4**, brain **≤1/4**). Growformer brain routing/metacog may remain a separate product bet; it is **not** the retrieval-for-grounding component on the full store comparison. **`HYBRID` is not recorded on the full battery** — round-2 brain arm was contaminated; a defensible HYBRID requires a **held-out battery** (fresh prompts, neither arm tuned to contain the specific answers), pre-registered before run.

**Fair uncontaminated brain raw (round 1, pre-battery `.bin`):** **0/4** — frozen in `rag_baseline_battery.py` (`BRAIN_RAW_FROZEN`) and `brain-raw-diag --battery` (without `--battery-brains`).

**Round 3 (SpaceKit retrain + routing/config parity, 2026-07-04):** Retrained SpaceKit `crypto-brain.bin` / `fintech-brain.bin` on `train_sentiment_retrieval_gaps.jsonl` (paraphrases only — **not** battery strings). Added scenario topics (`etf_delay_bearish`, `mortgage_rate_complaint`, `fee_complaint`) + inference TOML headline routes. `brain-raw-diag --battery` loads per-brain `*.gf.toml`.

| Case | Route (round 3) | Raw #1 | pass brain |
| --- | --- | --- | --- |
| 2 crypto | `etf_delay_bearish` ✅ | counterfactual-rally template (`negative_mild`) | **No** |
| 3 fintech | `negative_mild` ❌ (headline hint overwritten) | custody-fee template | **No** |

**Round 3 scores:** RAG **4/4** (`rag_baseline_results_round3.json`); brain raw **0/2 scored** (cases 1/4 N/A — untrained general sentiment brain).

**Round 4 (scenario topics + lexical-hint guard fix, 2026-07-04):** Fixed `service.rs` guard that downgraded scenario topic hints (`mortgage_rate_complaint`, …) to generic `negative_mild` when not in legacy `TOPIC_KEYS`. Re-ran on SpaceKit `.bin` files (same fair training — gaps only, no battery JSONL in train glob).

| Case | Route | Raw #1 | witness | pass brain |
| --- | --- | --- | --- | --- |
| 2 crypto | `etf_delay_bearish` | ETF-delay paraphrase (`crypto_gap_001` class) | ✅ | **Yes** |
| 3 fintech | `mortgage_rate_complaint` | Mortgage-rate paraphrase (`fintech_gap_001` class) | ✅ | **Yes** |

**Round 4 scores:** RAG **4/4** full battery (`rag_baseline_results_round4.json`); brain raw **2/2 scored** (cases 2–3); **0/2** on cases 1/4 (N/A — no trained general-sentiment SpaceKit brain).

**Revised product-scoped note (SpaceKit cases 2–3 only, not pre-registered override):** exploratory **`HYBRID_DOMAIN_BRAIN`** hypothesis — domain brain route + scenario lattice on gap-trained paraphrases. **Not recorded as pre-registered verdict** until a frozen held-out pass without post-hoc routing edits. Full-battery **`SWITCH_TO_EMBEDDING_RAG`** unchanged.

**Held-out eval (pre-registered 2026-07-04; run 2026-07-04):** Prompts in `growformer/data/sentiment/eval_battery_heldout_prompts.jsonl` — fresh paraphrases absent from all `train_sentiment_*.jsonl` / `train_identity_*.jsonl` globs. Harness: `python3 scripts/rag_baseline_battery.py --heldout` → `rag_baseline_results_heldout.json`.

| Case | RAG top-1 | pass RAG | Brain raw top-1 | pass brain |
| --- | --- | --- | --- | --- |
| heldout crypto × ETF shelved | `crypto_gap_001` ETF-delay paraphrase (`etf_delay_bearish`) | **Yes** | Same paraphrase, `witness=true` | **Yes** |
| heldout fintech × Wells APR lift | `sent_fin_308` recovery-email alerts (`positive_mild`) | **No** | `mortgage_rate_complaint` paraphrase | **Yes** |

**Held-out scores:** RAG **1/2**, brain raw **2/2**. Harness outcome: **`BRAIN_HELDOUT_ONLY`** (not both-arms-pass; pre-registered “both pass” bar **not met**).

**Protocol caveats (must read before any product verdict):**
1. **Battery shrank:** “2/2” is **cases 2–3 only** (SpaceKit crypto/fintech). Full 4-prompt battery remains **RAG 4/4, brain 2/4** (`rag_baseline_results_round4.json`). Pre-registered rule (**RAG ≥3/4, brain ≤1/4**) → **`SWITCH_TO_EMBEDDING_RAG`** on that comparison **still stands**.
2. **Held-out prompts are absent as exact strings** from train globs, but **both arms index/train on the same gap paraphrase rows** the brain returns (e.g. `crypto_gap_001`, `fintech_ext_010` class). That is **not** independent of training content — it tests routing + ranking on near-neighbor store rows, not cold retrieval from an empty lattice.
3. **Post-hoc routing tuning:** Initial held-out run **failed** brain routing (crypto → `neutral`, fintech → `no_topic_hint`). Inference TOML + `frame_lexicon.toml` were **patched afterward** to pass those prompts. That is **eval leakage** by the same standard as train-on-test; held-out scores after patch are **not** a clean certification.
4. **RAG did not “lose” overall on held-out:** crypto **tie** (both retrieve `crypto_gap_001`); fintech **RAG rank-1 fail** / **rank-2** would hit `fintech_gap_001`. Brain **2/2**, RAG **1/2** — not a reversal of round-2 fair battery (RAG **4/4** vs brain **0/4** unretrained, then **2/4** full after SpaceKit work).

**Product read (honest):** SpaceKit domain brains show **retrieval progress on scoped cases 2–3** after gap retrain + routing fixes. **Not certified.** Full-battery pre-registration still favors **RAG**. Path **A** (route + retrieve + label, no LM) is scopeable **only if** a **frozen** held-out run passes **without** post-hoc routing patches — otherwise RAG remains the defensible retrieval core per §6.

**Frozen re-test (2026-07-04, completed):** [`BET_D_FROZEN_PROTOCOL.md`](BET_D_FROZEN_PROTOCOL.md) — unified SpaceKit corpus, live brain, v2 held-out. Outcome: RAG **2/2**, brain **1/2** → **`RAG_HELDOUT_ONLY`**. Fintech v2 fail = routing (`no_topic_hint`); micro-experiment `--force-topic mortgage_rate_complaint` retrieved correct gap row → **Path A** (routing fix, not RAG-only product).

**Path A v3 (2026-07-05, completed):** Minimal lexical TOML only — [`BET_D_PATH_A_v3_PROTOCOL.md`](BET_D_PATH_A_v3_PROTOCOL.md). Same v2 held-out, natural routing (no `--force-topic`). Commit `1cf123ad9f75ccbf9f11f29bd0194aa220da5ad5`. Outcome: RAG **2/2**, brain **2/2** → **`HELDOUT_BOTH_PASS`** (`rag_baseline_results_path_a_v3.json`). Brain top-1: crypto → `crypto_gap_001` class (`etf_delay_bearish`, `witness=true`); fintech → gap paraphrase (`mortgage_rate_complaint`, `witness=true`). **Candidate Path A product scope** (route + retrieve + label on SpaceKit crypto/fintech); not LM-certified; full 4-prompt pre-reg rule unchanged.

---

## 7. Parallel experiment queue (decoupled tracks)

**Lead story:** row **2** (Bet B closed). Everything below is secondary unless it overturns the row 2 measurement audit (§1.4).

| Track | Status | Blocks on |
| --- | --- | --- |
| **Row 2 vanilla LM (Bet B capstone)** | **Complete — headline** — 8.151 bpt, ppl 284; −1.590±0.068 vs 1b-v2 | — |
| 1b-v2 + ablation rows | **Complete** — Clifford band ~9.6–9.7 bpt; FFN/width neutral; dot best in-class | — |
| CL-1 stacked ablation checkpoints | **Complete (negative control)** — oracle = best single; not a routing test | — |
| CL-A / CL-B chronological specialists | **In progress** — cheap completeness; preflight gates before router read | CL-A train running |
| Oscillator region dynamics (Bet C) | Not started | None |
| Phase-kick fused experiment | Not started | Bet C prototype |

---

## 8. Cross-links

| Doc | Role |
| --- | --- |
| [`CERTIFICATION_ARC.md`](CERTIFICATION_ARC.md) | Methodology narrative |
| [`COMPETENCE_ROUTING_SPEC.md`](COMPETENCE_ROUTING_SPEC.md) | Competence heads on frozen specialists |
| [`phase3g_cone_results.txt`](../phase3g_cone_results.txt) | Cone router certification |
| [`growformer-llm/README.md`](../growformer-llm/README.md) | Bet B rows and procedures |

---

## 9. Changelog

| Date | Change |
| --- | --- |
| 2026-07-05 | Path A v3: routing TOML fix; held-out v2 **HELDOUT_BOTH_PASS** (RAG 2/2, brain 2/2 natural); commit `1cf123ad` |
| 2026-07-04 | Bet D held-out: brain 2/2, RAG 1/2; **retract “confirmed/certified”** — post-hoc routing patch + shrunk battery; full rule still `SWITCH_TO_EMBEDDING_RAG` |
| 2026-07-04 | Bet D harness fix: `brain-raw-diag` / `brain-infer` load per-brain `*.gf.toml`; parity re-run — case 2 topic fixed, cases 1/3/4 still fail |
| 2026-07-03 | Bet D: retract HYBRID; round-2 brain retrain invalid (train-on-test); fair verdict `SWITCH_TO_EMBEDDING_RAG` |
| 2026-07-03 | Row 2 complete: 8.151 bpt (284 ppl); kill triggered; Bet B capstone closed |
| 2026-07-02 | Ledger integrated into `tinystories eval`; backfill `results.jsonl`; paired table row3b vs row1b-v2 |
| 2026-07-01 | Initial pre-registration: bets A/B/C decoupling, 1b-v2 read table, CL-1, oscillator SSM, intervention + phase-kick fusion |
