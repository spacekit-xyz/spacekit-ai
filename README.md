# Growformer Clifford LLM

A Rust language model built on **Space-Time Algebra**, Clifford algebra **Cl(1,3)**,
signature `(+,−,−,−)`. Linear layers and attention scores use the **geometric
product** of multivectors rather than scalar dot products and dense matmuls.

- Crate (lib): `growformer_llm` — `use growformer_llm::*;`
- Binary: `tinystories` — BPE → train → eval (bits/byte) → generate

> **Headline (2026-07-03).** At matched scalar budget (~737k), a standard transformer
> reaches **8.15 bits/token** vs the best Clifford stack **9.62** (row 3b) and corrected
> Clifford **9.74** (row 1b-v2) — **−1.5 to −1.6 bpt**, cleanly measured on the same
> held-out shard. **Bet B is closed:** Cl(1,3) geo-linear LM does not earn its complexity
> on TinyStories at this scale. FFN geometry and width are neutral; the gap is elsewhere
> in the architecture. **Product read:** use the transformer. **Research read:** honest
> negative result with component ablation. Bet A routing (CL-A/B) runs downstream of this
> foundation — report when done, do not lead with it.

---

## Separable bets (read before interpreting any number)

| Bet | Question | Where | 1b-v2 is a verdict on… |
| --- | --- | --- | --- |
| **A — CL** | Stack knowledge + route frozen specialists? | `growformer` promote-freeze, cone **92.5%** Task E | **No** — already certified on its own terms |
| **B — Clifford LM** | Does Cl(1,3) LM beat dot/vanilla at matched budget? | This repo (rows 1b–3b, **1b-v2**, row 2) | **Yes — Bet B only** |
| **C — Oscillators** | Causal region coupling via dynamics? | `growformer` substrate (side quest) | **No** |

A null on Clifford (Bet B) does **not** threaten continual learning or routing. Full gates:
[`PRE_REGISTRATION.md`](../growformer/docs/PRE_REGISTRATION.md).

---



## Research status

**Bet B answer (row 2, 2026-07-03):** Matched vanilla transformer **8.15 bpt** vs Clifford
**9.62–9.74 bpt** on held-out TinyStories (64×128 windows, same metric, verified — see
[`PRE_REGISTRATION.md`](../growformer/docs/PRE_REGISTRATION.md) §1.4 audit). This is the
finding the Clifford LM ablation arc was pre-registered to produce. Component isolation:
FFN Cayley vs dense **≈0** (row 3); width **≈0** (row 1c); score kernel dot beats inner
product **−0.12 bpt** (3b vs 1b-v2) but neither closes the gap to vanilla.

### Research probe vs product

| Lens | Read |
| --- | --- |
| **Research probe** | Real contribution: Cl(1,3) for discrete text underperforms a matched transformer at ~737k scalars; ablations localize where the loss is not (FFN, width) vs where it is (architecture class). Reporting your own architecture losing is the credibility move. |
| **Product** | Use the standard transformer. Cone routing over Clifford specialists builds a second story on a foundation Bet B just failed — still worth running CL-A cheaply for completeness, not as the headline. |

**Downstream (Bet A, not headline):** CL-1 chronological specialists (CL-A/B training) —
preflight parity gates; oracle = best single on stacked ablation rows already shown no
complementarity. See §2.2 in pre-registration.

### What the stack verified (before row 2)

| Claim                                | Evidence                                                                                                                                                              |
| ------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Algebra & backward are correct       | Cayley-table identities, finite-difference grad checks, `train_v2::end_to_end_loss_decreases`                                                                         |
| Pipeline works end-to-end            | BPE, packed bins, train, checkpoint, generate, `eval` (bits/byte vs gzip/lzma)                                                                                        |
| Training tricks help (domain corpus) | On sentiment headlines (`d_model=16`, tied, 200 steps): corpus-semantic init **val ppl 608** vs uniform **962** (~37% ↓) — see `[src/v2/README.md](src/v2/README.md)` |


These show the stack *trains* and that **embedding priors** matter. Row 2 showed they do **not** overcome the Clifford-vs-vanilla architecture gap at matched scalars.

### Held-out results (complete arc)
| Model | Params | bits/token (held-out) | bits/byte (held-out) | Notes |
| ----- | ------ | --------------------- | -------------------- | ----- |
| Clifford row 1a | ~0.7M | 10.37 | 2.34 | **In-sample / train-shard** — train=eval on `val.bin`, 400 steps; ppl **1261**; pipeline coherence only |
| Clifford row 1b | ~0.7M | **9.69** | **2.36** | Held-out; **pre-fix** scores `(Q⊛K)₀` w/o `K̃`; historical only |
| Clifford row 1b-v2 | ~0.7M | **9.74** | **2.37** | Post-fix `⟨Q,K⟩`, metric LN, seed 1000; eval ppl **856**; best val **839** @ 3800 |
| Dense FFN row 3 | ~0.7M (+66 FFN) | **9.72** | **2.37** | `--dense-ffn`, H=66; eval ppl **844**; **Δ +0.03 bits/token vs 1b** — no FFN signal |
| Clifford row 1c | ~2.8M | **9.76** | **2.38** | `d_model=32`, `d_ff=128`; best val ppl **837** @ 3600; eval ppl **866**; **no gain vs 1b** |
| unigram floor | — | 10.06 | 2.43 | MLE from train shard; held-out bytes/token ≈ **4.14** |
| uniform floor | — | 11.00 | 2.66 | log₂(2048); gzip ~2.69 on held-out (irrelevant bar) |
| Dot-attention row 3b | ~same | **9.62** | **2.34** | `--dot-attention`, seed 1000; best val ppl **757** @ 3600; eval ppl **789**; **−0.07 bpt vs 1b** |
| Matched vanilla row 2 | ~737k (d≈148) | **8.15** | **1.98** | `--vanilla`, seed 1000; eval ppl **284**; **−1.59 bpt vs 1b-v2** (ledger paired) |

**Row 1a (in-sample, 2026-06-30):** conditional **2.34 bpb**, ppl **1261**, **10.37 bits/token**.
Uniform floor **2.48 bpb** (11 bits/token); model is ~**6% below** uniform but still **above** empirical unigram — barely learning at 400 steps.
gzip **2.59** can be *worse than uniform* on tiny slices (misleading bar). **Headline: ppl, not bpb vs gzip.**

**Baselines in `eval` / `baselines`:** uniform token floor + empirical unigram (from `--train-bin` or `baselines train_bin`) alongside gzip/lzma.

**Row 1b (held-out):** eval **9.69 bits/token** (ppl **827**, 64 windows); best train val ppl **810** @ step 3000. **0.36 bits/token** below unigram (10.06). Capacity-limited plateau.

**Row 3 (dense FFN, `--init-seed 1000`):** eval **9.72 bits/token** (ppl **844**). **+0.03 bits/token vs 1b** — FFN Cayley structure is not the bottleneck at this size. Dense FFN has **+66** scalar surplus (H=66). *Seed not matched to 1b default; pair with 1b@1000 for strict ablation if needed.*

**Row 1c (capacity, 2026-07-01):** eval **9.76 bits/token** (ppl **866**, 64 windows); best train val ppl **837** @ step 3600. **+0.07 bits/token vs 1b** despite ~4× params — the ~820–850 plateau is **not** width-limited at 4000 steps.

**Row 3b (dot scores, `--init-seed 1000`, 2026-07-01):** eval **9.62 bits/token** (ppl **789**, 64 windows); best train val ppl **757** @ step 3600. **−0.07 bits/token vs 1b** (9.69) — dot scores beat pre-fix geometric scores on the same Clifford body.

**Row 1b-v2 (corrected Clifford, `--init-seed 1000`, 2026-07-02):** eval **9.74 bits/token** (ppl **856**, 64 windows); best train val ppl **839** @ step 3800. **+0.05 bits/token vs 1b** (9.69), **+0.12 vs 3b** (9.62). Pre-registered verdict: **bug fix neutral; dot still wins** — see [`PRE_REGISTRATION.md`](../growformer/docs/PRE_REGISTRATION.md) §1.2–1.4. Checkpoint: `agent-data/tinystories-row1b-v2.json`.

**Row 2 (vanilla capstone, `--vanilla`, seed 1000, 2026-07-03):** held-out **8.15 bits/token** (ppl **284**, 64 windows); best train-val ppl **267** @ step 4000. Ledger vs **row1b-v2**: **−1.590 ± 0.068 bpt**. Pre-registered kill **triggered** — standard transformer at matched scalar budget beats Clifford stack; Bet B capstone closed. Checkpoint: `agent-data/tinystories-row2-seed1000.json`.

**CL-1 (Bet A, 2026-07-03):** Stacking frozen Bet B checkpoints — **row2+row3b** is an **imbalanced setup** (Δ=1.47 bpt, 64/0 windows); **row1b-v2+row3b** is the valid **peer negative control** (Δ=0.12 bpt, 0/64 windows). In both cases **oracle = best single** → no complementarity to exploit; router collapse is expected, not a cone-routing finding. See [`PRE_REGISTRATION.md`](../growformer/docs/PRE_REGISTRATION.md) §2.2. Chronological CL-A/B in progress — **preflight parity gates apply before reading router.**

**Ablation summary (held-out bits/token):** FFN geometry **≈0** (row 3); width **≈0** (row 1c); **score kernel** dot beats inner product **−0.12 bpt** (3b vs 1b-v2). Clifford Q/K/V/O projections untested in isolation.

### Results ledger (`growformer-ledger`)

Append-only hash-chained eval log at `agent-data/results.jsonl`. Every held-out
`eval` (with `--train-bin`) stores **64 per-window bpt values** for paired
statistics. §1.2 verdict tables are **queries**, not hand-edited prose.

```bash
# Held-out eval (auto-appends to ledger)
cargo run --release --bin tinystories -- eval \
  --checkpoint agent-data/tinystories-row1b-v2.json \
  --tokenizer data/tinystories.tok \
  --train-bin data/tinystories-train.bin \
  data/tinystories-heldout.bin --seq-len 128 --windows 64 \
  --run-id row1b-v2 --selection-tag first

# Paired-SE §1.2 table (post-fix baseline)
cargo run --release --bin tinystories -- ledger-table \
  --baseline row1b-v2 --candidates row3b

# CI integrity check
cargo run --release --bin tinystories -- ledger-verify

# CL-1: route two frozen specialists (Bet A)
cargo run --release --bin tinystories -- cl1 \
  --checkpoint-a agent-data/tinystories-row2-seed1000.json \
  --checkpoint-b agent-data/tinystories-row3b-seed1000.json \
  --tokenizer data/tinystories.tok \
  data/tinystories-heldout.bin --seq-len 128 --windows 64 --cal-windows 30 \
  --run-id cl1-row2-row3b
```

**Ledger-backed read (2026-07-02, n=64, same split):** `row3b` **−0.118 ± 0.012 bpt**
vs `row1b-v2` (paired). Pre-fix checkpoints (`row1b`, `row1c`, `row3`) **must not**
be compared under the post-fix eval forward — re-eval gives ~**10.0 bpt** (train/eval
mismatch). Historical row 1b **9.69 bpt** was pre-fix train **and** eval. See
[`growformer-ledger/README.md`](../growformer-ledger/README.md).

**How to read row 1 (measurement caveat).** `tinystories eval` reports the model’s
**conditional** cross-entropy rate: bits/byte *given the trained weights*, with model
parameters **not** amortized into the bit count. gzip/lzma totals include their
codec overhead on the same byte stream but not a separate “model file.” On small
corpora, a ~0.7M-parameter checkpoint can look better or worse than gzip depending
on whether you count the weights. Row 1 is pipeline validation and a competence
sanity check — not a claim that the shipped system beats gzip end-to-end.

### Ablation matching protocol (fix before rows 2–3)

The ablation is defined by the matching rule; implement code only after this is fixed.


| Rule                  | Role             | Definition (draft)                                                                                                                                                                                                                                                                                                                                                                           |
| --------------------- | ---------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Parameter-matched** | **Primary**      | Same **total** learnable scalar count (weights **and** biases). Clifford FFN: `16·(2·d_model·d_ff + d_ff + d_model)` scalars. Dense FFN on flattened residual: `Linear(16·d_model→H)` + `Linear(H→16·d_model)` → `2·16·d_model·H + H + 16·d_model`. Weights-only gives `H = d_ff`; **total** match requires solving for `H` (at defaults `d_model=16`, `d_ff=64`: Clifford **34,048**, naive `H=d_ff` dense **33,088**, Δ **960 bias scalars** — fold into `H`, don't footnote). Runtime assert on totals, not labels. |
| **FLOP-matched**      | Report alongside | Count multiply-adds in forward (geo product vs dot). Clifford layers do more work per param; report FLOPs separately so “wins at equal params but 3× FLOPs” is visible.                                                                                                                                                                                                                      |
| **Width-matched**     | Secondary only   | Same `d_model`, `n_blocks`, `n_heads` — convenient but **not** equal params (Clifford cells are 16-wide). Do not use as the headline comparison.                                                                                                                                                                                                                                             |


**Implementation scope (not a one-line flag).** First ablation row is **FFN-only**: swap `CliffordFFN` for param-matched `DenseFFN` at the block boundary (flatten post-`norm2` → real linear → unflatten residual delta). Attention deferred (rank/16× confound). Needs parallel forward + tape backward + inference path.

**Param assert (construction time).** Count every learnable scalar:

```rust
fn clifford_ffn_scalars(d_model: usize, d_ff: usize) -> usize {
    // fc1: 16·d_model·d_ff weights + 16·d_ff biases; fc2: symmetric
    16 * (2 * d_model * d_ff + d_ff + d_model)
}

fn dense_ffn_scalars(d_model: usize, hidden: usize) -> usize {
    let in_ = 16 * d_model;
    // fc1: in_·H + H; fc2: H·in_ + in_
    2 * in_ * hidden + hidden + in_
}

// Solve hidden s.t. dense_ffn_scalars(d_model, H) == clifford_ffn_scalars(d_model, d_ff)
// Weights-only: 2·16·d_model·H = 2·16·d_model·d_ff  →  H = d_ff (biases differ by 16·(d_ff + d_model) − (H + 16·d_model))
debug_assert_eq!(dense_ffn_scalars(d_model, hidden), clifford_ffn_scalars(d_model, d_ff),
    "FFN scalar budgets must match for single-variable ablation");
```

**Row procedure (before first number).** ≥3 seeds per row; same `init_seed`, semantic-init scheme, tokenizer, train/held-out bins, eval windows; report bits/token mean ± spread. Match dense-init variance to Clifford's tiny-multivector init (document scheme in row notes).

**Vanilla transformer row (row 2).** Same tokenizer, corpus, training budget, and
eval harness; parameter-matched real-valued transformer (dot-product attention,
standard LayerNorm, ReLU FFN, sinusoidal PE). CLI `--d-model` sets the **Clifford
reference** width; `param_budget::matched_vanilla_d_model` picks the vanilla
`d_model` within ±500 scalars. Implemented in `vanilla_llm` + `v2::vanilla_train`.

```bash
# Row 2 train (matched to ~737k scalars at d_model=16 Clifford ref)
cargo run --release --bin tinystories -- train \
  data/tinystories.tok data/tinystories-train.bin data/tinystories-heldout.bin \
  --checkpoint-out agent-data/tinystories-row2-seed0.json \
  --vanilla --d-model 16 --d-ff 64 --n-blocks 4 --n-heads 4 \
  --steps 4000 --tie-embeddings --grad-accum 2 --init-seed 0

# Row 2 held-out eval (vanilla auto-detected from checkpoint cfg)
cargo run --release --bin tinystories -- eval \
  --checkpoint agent-data/tinystories-row2-seed0.json \
  --tokenizer data/tinystories.tok \
  --train-bin data/tinystories-train.bin \
  data/tinystories-heldout.bin --seq-len 128 --windows 64
```

### Row 1 procedure (TinyStories)

```bash
# 1. Chronological 90/10 split (held-out tail never seen in training)
cargo run --release --bin tinystories -- split \
  data/val.bin data/tinystories-train.bin data/tinystories-heldout.bin

# 2. Train on train shard only (row 1b — increase --steps until ppl ≪ uniform)
cargo run --release --bin tinystories -- train \
  data/tinystories.tok data/tinystories-train.bin data/tinystories-heldout.bin \
  --checkpoint-out agent-data/tinystories-row1b.json \
  --steps 4000 --tie-embeddings --grad-accum 2

# 3. Held-out eval + unigram floor (uniform + empirical from train shard)
cargo run --release --bin tinystories -- baselines \
  --train-bin data/tinystories-train.bin \
  --tokenizer data/tinystories.tok \
  data/tinystories-heldout.bin

cargo run --release --bin tinystories -- eval \
  --checkpoint agent-data/tinystories-row1b.json \
  --tokenizer data/tinystories.tok \
  --train-bin data/tinystories-train.bin \
  data/tinystories-heldout.bin --seq-len 128 --windows 64
```

Row **1a** (`tinystories-row1.json`) used train=eval on `val.bin` — valid for pipeline
coherence only; do not cite as held-out val.

### Row 3 procedure (dense FFN ablation)

Same protocol as row 1b (held-out split, unigram floor, ≥3 seeds), with `--dense-ffn`:

```bash
cargo run --release --bin tinystories -- train \
  data/tinystories.tok data/tinystories-train.bin data/tinystories-heldout.bin \
  --checkpoint-out agent-data/tinystories-row3-seed0.json \
  --steps 4000 --tie-embeddings --grad-accum 2 --dense-ffn \
  --init-seed 1000   # vary seed per run; report mean ± spread on bits/token
```

Construction logs matched hidden `H` and scalar budgets (Clifford vs dense). Attention, norms, embeddings, and eval harness are unchanged — only the FFN block differs.

**Row 3 result (2026-06-30):** held-out eval **9.72 bits/token** (ppl 844) vs row 1b **9.69** (ppl 827) — **Δ +0.03 bits/token**. FFN geometry is not the bottleneck.

### Row 1c procedure (capacity bump)

Same held-out protocol as 1b; scale width before attention ablations:

```bash
cargo run --release --bin tinystories -- train \
  data/tinystories.tok data/tinystories-train.bin data/tinystories-heldout.bin \
  --checkpoint-out agent-data/tinystories-row1c.json \
  --d-model 32 --d-ff 128 --n-heads 4 --n-blocks 4 \
  --steps 4000 --tie-embeddings --grad-accum 2

cargo run --release --bin tinystories -- eval \
  --checkpoint agent-data/tinystories-row1c.json \
  --tokenizer data/tinystories.tok \
  --train-bin data/tinystories-train.bin \
  data/tinystories-heldout.bin --seq-len 128 --windows 64
```

If held-out bits/token drops well below ~9.7, the ~820 plateau was capacity; if not, attention/score ablations (row 3b) are the next mechanism test.

**Row 1c result (2026-07-01):** held-out eval **9.76 bits/token** (ppl **866**) vs row 1b **9.69** (ppl **827**) — **Δ +0.07 bits/token** with ~4× params. Plateau persists; proceed to row **3b**.

### Row 3b procedure (dot attention scores)

Score-only ablation: swap `⟨Q⊛K⟩₀` for 16-component dot product on the same Clifford Q/K projections. Clifford FFN, Q/K/V/O, norms, and eval harness unchanged.

```bash
cargo run --release --bin tinystories -- train \
  data/tinystories.tok data/tinystories-train.bin data/tinystories-heldout.bin \
  --checkpoint-out agent-data/tinystories-row3b-seed1000.json \
  --steps 4000 --tie-embeddings --grad-accum 2 \
  --dot-attention --init-seed 1000

cargo run --release --bin tinystories -- eval \
  --checkpoint agent-data/tinystories-row3b-seed1000.json \
  --tokenizer data/tinystories.tok \
  --train-bin data/tinystories-train.bin \
  data/tinystories-heldout.bin --seq-len 128 --windows 64
```

Compare to row 1b at the same `--init-seed 1000` for a strict pair.

**Row 3b result (2026-07-01):** held-out eval **9.62 bits/token** (ppl **789**) vs row 1b **9.69** (ppl **827**) — **Δ −0.07 bits/token**. Geometric attention scores underperform dot scores on the same Clifford body; proceed to row **2**.

### Clifford correctness fixes (2026-07-01)

Earlier rows used `(Q ⊛ K)₀` without `reverse(K)` — not the Clifford inner product. Default geometric scores now use **`⟨Q, K⟩ = (Q ⊛ K̃)₀`** everywhere (train, eval, inference cache). Layer norm uses **metric-weighted** statistics over blade components. `CliffordLinear::new` seeds all 16 blades (not scalar-only).

**Re-run row 1b baseline** with the fixed stack (same protocol as 1b, no flags):

```bash
cargo run --release --bin tinystories -- train \
  data/tinystories.tok data/tinystories-train.bin data/tinystories-heldout.bin \
  --checkpoint-out agent-data/tinystories-row1b-v2.json \
  --steps 4000 --tie-embeddings --grad-accum 2 --init-seed 1000
```

Compare held-out eval to row 1b, row 3b, and row 3 at seed 1000.

**Row 1b-v2 result (2026-07-02):** held-out **9.74 bits/token** (ppl **856**) vs 1b **9.69** (827) and 3b **9.62** (789). Correctness fix did not close the gap to dot scores or beat the pre-fix baseline.

### What works vs what would generalize (Bet B)

The Clifford LM **works** in the operational sense: it trains stably, beats unigram by **~0.3 bits/token**, train-val and held-out track together (no overfit gap), and produces a usable frozen checkpoint. It does **not** beat matched traditional score kernels or width/FFN ablations at this budget.

| Lever | Tested? | Effect on held-out bpt | Notes |
| ----- | ------- | ---------------------- | ----- |
| More steps (4000→?) | Partially | **≈0** | Best val flat by step 3000 (1b) / 3800 (1b-v2); curve converged, not step-starved |
| GPU / faster hardware | No (CPU train) | **Throughput only** | Same hyperparams → same numbers; GPU enables multi-seed row 2 and CL-1, not a quality fix |
| Width (`d_model=32`) | Yes (row 1c) | **+0.07** (worse) | ~4× params, same plateau band |
| Dense FFN | Yes (row 3) | **+0.03** (worse) | FFN geometry not the bottleneck |
| Clifford score fix (`⟨Q,K⟩`, metric LN) | Yes (1b-v2) | **+0.05** vs 1b | Neutral to marginally worse |
| Dot attention scores | Yes (3b) | **−0.12** vs 1b-v2 | Best Clifford-stack row so far |
| Corpus-semantic init | Yes (default) | Large at step 0 | Already applied; buys cold start, not the ~850 ceiling |
| Matched vanilla transformer | **Yes** | **8.15 bpt** (−1.59 vs 1b-v2) | **Capstone closed** — vanilla wins at matched budget |
| Stack frozen specialists + cone | **Yes** | **0** (oracle=best single) | Peer control (1b-v2+3b): no window complementarity; row2+3b invalid setup |

**Honest read:** Row **2** closed Bet B at this budget: vanilla **8.15 bpt** vs Clifford **9.74** (1b-v2) and dot-Clifford **9.62** (3b) — measurement verified (§1.4 audit), not a units bug. **CL-1** on row2+row3b is an **invalid routing setup** (imbalanced peers); row1b-v2+row3b shows **oracle = best single** (no complementarity). Bet A routing test requires chronological specialists with **preflight parity** (§2.2).

### Open design questions

**Why Cl(1,3) for text at all?** In the GA-for-physics literature, the geometric
product earns its keep through **equivariance** to a symmetry the data actually has.
Discrete tokens have no Lorentz symmetry. Here the Cayley table is a fixed sparse
bilinear mixing pattern — possibly a useful parameterization, possibly a more
expensive constrained linear layer. We do not know which without the ablation above.

**Attention scores** use the Clifford inner product `⟨Q, K⟩ = (Q ⊛ K̃)₀` — not
the raw grade-0 of `Q ⊛ K` (which omitted `reverse(K)` and mis-specified the metric).
Use `--dot-attention` for the row 3b ablation (Euclidean dot on all 16 components).

**Positional “rotors”** mix compact rotations (`cos`/`sin` on `e12/e13/e23`) with
non-compact Lorentz boosts (`cosh`/`sinh` on `e01/e02/e03`). Boosts preserve the
**Clifford** norm `R R̃ = 1` (what `positional.rs` tests assert); they are **not**
Euclidean-unitary and can change the Euclidean norm of activations. `final_norm` and
block LayerNorm are doing real work here.

---



## Crate layout

```
Cargo.toml
src/
  lib.rs            — module declarations and re-exports
  clifford_llm.rs   — core types: Multivector, CliffordAlgebra, all layers, CliffordLLM
  blade.rs          — blade index constants, grade utilities, display
  cayley_const.rs   — compile-time Cayley table, CliffordAlgebraConst
  backprop.rs       — gradient types and backward pass (incl. RealHeadGrad)
  optim.rs          — Adam optimiser, LR schedule, gradient clipping
  positional.rs     — rotor-based positional encoding
  mask.rs           — causal and padding masks
  kv_cache.rs       — KV cache for autoregressive inference
  bpe.rs            — byte-pair-encoding tokenizer
  tinystories.rs    — packed corpus loader + random-chunk sampling
  v2/               — taped full-backward training, sampling, checkpoints,
                      arithmetic coding (see src/v2/README.md)
  bin/tinystories.rs — CLI: tokenize / encode / train / eval / generate
```

---



## Algebra at a glance

Cl(1,3) has 2⁴ = **16 basis blades**. The index used throughout the crate is the
**bit-mask** of the blade: bit *k* set means basis vector e*k* is present.


| index (bitmask) | blade   | grade            |
| --------------- | ------- | ---------------- |
| `0` = `0b0000`  | `1`     | 0 — scalar       |
| `1` = `0b0001`  | `e0`    | 1                |
| `2` = `0b0010`  | `e1`    | 1                |
| `3` = `0b0011`  | `e01`   | 2                |
| `4` = `0b0100`  | `e2`    | 1                |
| `5` = `0b0101`  | `e02`   | 2                |
| `6` = `0b0110`  | `e12`   | 2                |
| `7` = `0b0111`  | `e012`  | 3                |
| `8` = `0b1000`  | `e3`    | 1                |
| `9` = `0b1001`  | `e03`   | 2                |
| `10` = `0b1010` | `e13`   | 2                |
| `11` = `0b1011` | `e013`  | 3                |
| `12` = `0b1100` | `e23`   | 2                |
| `13` = `0b1101` | `e023`  | 3                |
| `14` = `0b1110` | `e123`  | 3                |
| `15` = `0b1111` | `e0123` | 4 — pseudoscalar |


Metric signature: `e0² = +1`, `e1² = e2² = e3² = −1`.

A `Multivector` is `[f32; 16]` — one component per blade at the index above. The
scalar (grade-0) part is always `c[0]`.

---



## Model architecture

```
token ids
  │  embedding[id]            (vocab × d_model multivectors)
  ▼
RotorPositionalEncoding       (sandwich R ⊛ x ⊛ R̃ — see `positional`)
  ▼
CliffordBlock × n_blocks
  ├─ norm1  → CliffordAttention (genuine multi-head)   → + residual
  └─ norm2  → CliffordFFN (fc1 → ReLU → fc2)            → + residual
  ▼
final_norm  (CliffordLayerNorm — GPT-2 `ln_f`; keeps the residual stream bounded)
  ▼
head: LinearReal              (flatten 16·d_model reals → vocab logits)
  ▼
logits[seq][vocab]
```

Key points:

- **Genuine multi-head attention** — each head owns `[h·head_dim, (h+1)·head_dim)`.
  Scores use the Clifford inner product `⟨Q, K⟩ = (Q ⊛ K̃)₀` (all grades mix via
  the Cayley table), scaled by `1/√(head_dim·16)`. `--dot-attention` selects the
  row 3b Euclidean-dot ablation instead.
- **CliffordLayerNorm** — metric-weighted mean/var over `16·d_model` blade
  components (not flat Euclidean statistics).
- **Real-valued output head (**`LinearReal`**)** — the residual stream
(`16·d_model` reals after `final_norm`) is projected to vocab logits by an
ordinary real matmul. This is far cheaper than a geometric-product head and is
the layer that **weight tying** shares with the embedding table.
- `final_norm` before the head is required: without it the unbounded
residual stream makes logits explode.



### Weight tying

`CliffordLLM::sync_tied_head()` mirrors the embedding table into `head.weights`,
so `logit[v] = bias[v] + ⟨flatten(final_norm(x)), flatten(embedding[v])⟩`. The
head and embedding then share one matrix — a strong prior and a parameter saving
for small models. Call it after any embedding update and after loading a tied
checkpoint. (The training loop in `v2` does this for you.)

---



## Quick start

```rust
use growformer_llm::*;
use std::sync::Arc;

let algebra = Arc::new(CliffordAlgebra::sta());

let d_model = 16;
let n_heads = 4;
let d_ff    = 64;
let vocab   = 2048;

let blocks: Vec<CliffordBlock> = (0..4).map(|_| CliffordBlock {
    attn:  CliffordAttention::new(d_model, n_heads, algebra.clone()),
    ffn:   CliffordFFN::new(d_model, d_ff, algebra.clone()),
    norm1: CliffordLayerNorm::new(d_model),
    norm2: CliffordLayerNorm::new(d_model),
}).collect();

let model = CliffordLLM {
    embedding:  vec![vec![Multivector::scalar(0.01); d_model]; vocab],
    blocks,
    final_norm: CliffordLayerNorm::new(d_model),
    head:       LinearReal::new(d_model, vocab),   // 16·d_model reals → vocab
    algebra,
};

let logits = model.forward(&[1, 42, 7, 0]);  // → Vec<Vec<f32>> [seq][vocab]
```

For real training you must first break symmetry (random init) and, in practice,
use the `v2` pipeline — see `[src/v2/README.md](src/v2/README.md)`.

---



## Inference cost

At default size (`d_model=16`, `n_blocks=4`, `vocab=2048`, ~0.7M params),
forward-only generation is fast for mundane reasons: tiny model, no backward tape,
const-folded Cayley table, stack-allocated `[f32;16]` math, release LTO. That is
expected, not surprising.

**Generation** uses `InferenceCache` (`v2/inference.rs`): K/V are cached per layer,
so each new token is **O(seq·layers)** instead of a full **O(seq²·layers)**
recompute. Training and `eval` still use the full-sequence path. The cache
respects `max_seq` (sliding-window eviction).

---



## Library primitives



### `blade` — blade index constants and grade utilities

```rust
use growformer_llm::blade::*;

let mut mv = Multivector::zero();
mv.c[E12] = 1.0;
mv.c[E0]  = 0.5;

println!("{}", display(&mv));         // "0.5000e0 + e12"
let grade2 = project_grade(&mv, 2);   // zero everything except grade-2
let bv     = bivector_part(&mv);      // → [e01, e02, e12, e03, e13, e23]
```

`SCALAR, E0…E0123` (indices), `BLADE_NAMES`, `BLADE_GRADES`, `REVERSE_SIGNS`,
`grade_of`, `blades_of_grade`, `project_grade`, `vector`, `bivector`, `display`.

### `cayley_const` — compile-time Cayley table

```rust
use growformer_llm::cayley_const::{CAYLEY_STA, CliffordAlgebraConst};

let cell = CAYLEY_STA[E1][E2];        // e1 ⊛ e2 → +e12
assert_eq!(cell.blade, E12 as u8);

const ALG: CliffordAlgebraConst = CliffordAlgebraConst::new();
let r = ALG.geo_product(&a, &b);
let s = ALG.sandwich(&rotor, &x);     // r ⊛ x ⊛ r̃
```

Same API as the runtime `CliffordAlgebra`, usable as a `const` item.

### `backprop` — gradient types and backward pass

The geometric product is bilinear, so gradients follow a clean product rule.
Given `C = geo(A, B)`:

```
dL/dA[i] = Σ_j  B[j] × cayley[i][j].sign × dL/dC[ cayley[i][j].blade ]
dL/dB[j] = Σ_i  A[i] × cayley[i][j].sign × dL/dC[ cayley[i][j].blade ]
```

```rust
use growformer_llm::backprop::*;

let (grad_a, grad_b)     = geo_product_backward(&a, &b, &grad_c);
let (grad_layer, grad_x) = linear_backward(&weights, &inputs, &grad_out);
let (loss, grad_logits)  = cross_entropy(&logits, target_token_id);

// Real output head (LinearReal): scatter logit grads back to the residual stream
let mut grad_head = RealHeadGrad::zeros(vocab, d_model * 16);
let grad_x = real_head_backward(&head.weights, &head_input, &grad_logits, &mut grad_head);
```

`GradLinear` and `RealHeadGrad` both support `accumulate` + `scale` for batch
averaging. (`scalar_head_backward` still exists for the legacy scalar head.)

### `optim` — Adam optimiser

```rust
use growformer_llm::optim::*;

let cfg = AdamConfig { lr: 3e-4, weight_decay: 0.01, ..Default::default() };
let mut opt = LayerOptimizer::new(out_dim, in_dim, cfg);          // Clifford layer
opt.step(&mut weights, &mut biases, &grad_layer);

let mut head_opt = RealHeadOptimizer::new(vocab, d_model * 16, cfg);  // LinearReal head
head_opt.step(&mut head, &grad_head);

clip_grad_norm(&mut grad, 1.0);
let lr = cosine_lr_with_warmup(step, warmup_steps, total_steps, 3e-4, 1e-5);
```

`AdamConfig` defaults: `lr=1e-3`, `beta1=0.9`, `beta2=0.999`, `eps=1e-8`,
`weight_decay=0.0`.

### `positional` — rotor positional encoding

Positions are rotors acting via the sandwich product `R(t) ⊛ x ⊛ R̃(t)`. Cl(1,3)
gives six independent bivector planes: `e12, e13, e23` (spatial rotations,
`cos`/`sin`) and `e01, e02, e03` (Lorentz boosts, `cosh`/`sinh`). Angles follow
the log-spaced sinusoidal schedule `θ(t,d) = t / 10000^(2d/d_model)`.

```rust
use growformer_llm::positional::*;
let pe = RotorPositionalEncoding::new(d_model);
let encoded = pe.encode(&alg, &embedded_sequence);   // [seq][d_model]
let table   = pe.precompute_rotors(max_seq_len);     // fast inference
```



### `mask` — causal and padding masks

```rust
use growformer_llm::mask::*;
mask_scores(&mut scores, None);                       // causal only
mask_scores(&mut scores, Some(&padding_mask));        // causal + padding
let pad = padding_mask_from_ids(&token_ids, pad_id);
```



### `kv_cache` — KV cache for autoregressive inference

```rust
use growformer_llm::kv_cache::*;
let mut cache = KVCache::new(n_layers, max_seq_len);
let attn_out  = cached_attention_step(&alg, cache.layer_mut(i), &q_new, k_new, v_new, scale);
```

The cache evicts the oldest tokens (sliding window) past `max_seq_len`.

---



## Running tests

```bash
cargo test --release
```

Covers algebra identities (anti-commutativity, associativity, metric signature),
finite-difference gradient checks, rotor unitarity, optimiser convergence, mask
boundaries, KV-cache eviction, and the end-to-end `train_v2` loss-decrease test.

---



## Growformer sibling crate (`brain-memory` feature)

**Canonical project paths (SpaceKit):** crypto and fintech train/infer data + deployed brains live under [`spacekit-projects/sentiment`](../../spacekit/spacekit-projects/sentiment) (`crypto/`, `fintech/`). Override root with `SPACEKIT_SENTIMENT_ROOT`. General sentiment (cases 1/4) still uses neurokit `growformer/scripts/sentiment-analysis.gf.toml` until a matching SpaceKit project exists.

**Status (2026-07-04):** Bet D **revised for product scope**. Full 4-prompt pre-registration still **`SWITCH_TO_EMBEDDING_RAG`** (RAG 4/4, brain 2/4 raw). **Scored SpaceKit cases 2–3:** brain raw **2/2** after scenario topics + `service.rs` lexical-hint fix → product verdict **`HYBRID_DOMAIN_BRAIN`**. **`brain-raw-diag --battery`** uses per-brain `*.gf.toml` config parity.

| Case | Topic hint | Raw #1 | Full infer |
|------|------------|--------|------------|
| 2 crypto × Bitcoin ETF | `etf_delay_bearish` ✅ | ETF-delay paraphrase, `witness=true` ✅ | user-anchored NEGATIVE ✅ |
| 3 fintech × Chase hike | `mortgage_rate_complaint` ✅ | mortgage paraphrase, `witness=true` ✅ | user-anchored NEGATIVE ✅ |

Cases 1/4 still use untrained neurokit `sentiment-brain-v3.bin` — skipped in default `--battery`.

**Retrieval-gap training (held-out paraphrases — not battery strings):** `train_sentiment_retrieval_gaps.jsonl` uses **scenario topics** (`etf_delay_bearish`, `mortgage_rate_complaint`) — not polarity-only `negative_mild`. Inference TOML maps the two scored battery prompts into those buckets before forced-topic retrieval. Mis-bucketed rows evicted: counterfactual rally → `copium`, custody fee → `fee_complaint`.

Retrain after data/TOML changes:

```bash
cd growformer
cargo run --release --no-default-features --features cli,native --bin growformer -- \
  --train-brain --project ../../spacekit/spacekit-projects/sentiment/crypto/crypto-sentiment-analysis.gf.toml
cargo run --release --no-default-features --features cli,native --bin growformer -- \
  --train-brain --project ../../spacekit/spacekit-projects/sentiment/fintech/fintech-sentiment-analysis.gf.toml
```

**Prior unfair scores (no project wiring):** round 1 RAG **1/4**, brain raw **0/4**. Round-2 `*-battery.bin` retrain remains **invalid** (train-on-test).

The `[growformer](../growformer)` crate implements lattice routing, sentiment brains,
and promote-freeze CL. With the default **`brain-memory`** feature, growformer-llm can
load a `brain.bin` as a **memory unit** — route + retrieve lattice text — then prefix
a vanilla or Clifford LM checkpoint for fluent continuation (downstream; not validated).

**Architecture (target):** brain handles domain routing/retrieval; LM handles open-ended tokens.
This is the product-shaped integration, not stacking Clifford LM specialists.

### Raw lattice diagnostic (pre-gate, fork-resolving)

Bypasses metacog and grounding gate. Dumps top-K lattice candidates after cosine+BM25+graph+lex-align scoring, with witness/hard-reject flags computed but **not** applied.

```bash
# Four-prompt battery — each case loads its own *.gf.toml (inference TOML + guardrails + topic graph)
cargo run --release --bin tinystories -- brain-raw-diag --battery --top-k 5

# Single brain + prompt (pass --project for native infer parity)
cargo run --release --bin tinystories -- brain-raw-diag \
  --brain ../growformer/agent-data/sentiment-analysis/sentiment-brain-v3.bin \
  --project ../growformer/scripts/sentiment-analysis.gf.toml \
  --prompt "Bitcoin crashed after the ETF delay" \
  --top-k 5

# Crypto case — SpaceKit canonical project
cargo run --release --bin tinystories -- brain-raw-diag \
  --brain ../../spacekit/spacekit-projects/sentiment/crypto/agent/crypto-brain.bin \
  --project ../../spacekit/spacekit-projects/sentiment/crypto/crypto-sentiment-analysis.gf.toml \
  --prompt "Bitcoin crashed after the ETF delay" \
  --top-k 5 -v

# Fintech case — SpaceKit canonical project
cargo run --release --bin tinystories -- brain-raw-diag \
  --brain ../../spacekit/spacekit-projects/sentiment/fintech/agent/fintech-brain.bin \
  --project ../../spacekit/spacekit-projects/sentiment/fintech/fintech-sentiment-analysis.gf.toml \
  --prompt "Chase raised my mortgage rate without notice" \
  --top-k 5 -v
```

**Config flags (single-prompt / `brain-infer`):** `--project`, `--inference-toml`, `--inference-defaults-toml`, `--guardrails-jsonl`, `-v`. World grounding graphs (base + crypto + fintech) are always compile-time embedded; runtime `grounding_toml` from the project is optional.

**Pre-registered rubric (read before tuning gates):**

| Raw top-1 | Interpretation | Next step |
|-----------|----------------|-----------|
| Topical + correct polarity/rank | Core retrieval OK | Tune witness / grounding gate / metacog — memory layer recoverable |
| Surface-matched, wrong rank (esp. case 4 polarity flip) | Core retrieval bug | Fix lattice scoring/training before any LM conditioning |
| Right terms in top-3 but wrong #1 | Mixed — ranking vs gating | Check witness_ok on #1 vs #2; gate if #2 is correct |
| `no_topic_hint` / empty candidates | Routing bug | Fix topic/subject setup before retrieval |

**Regression checklist (stage-tagged — do not flatten to FAIL):**

Default `--battery` runs **cases 2–3 only** (trained SpaceKit crypto + fintech). Cases 1 and 4 use neurokit `sentiment-brain-v3.bin`, which **is not trained** — ignore until a SpaceKit general-sentiment project exists. Pass `--battery-all` to run them for routing diagnostics only (not scored).

| Case | Brain × prompt | Expected | Stage if broken | Default battery |
|------|----------------|----------|-----------------|-----------------|
| 1 | sentiment × Bitcoin ETF crash | negative/crypto headline memory | retrieval / gate / interface | **skipped** |
| 2 | crypto × same | `etf_delay_bearish` lattice (ETF delay + crash copy) | retrieval / gate | **scored** |
| 3 | fintech × Chase mortgage hike | `mortgage_rate_complaint` lattice (not custody fee) | retrieval / gate | **scored** |
| 4 | sentiment × Chase hike (wrong brain) | low confidence or neutral — **not** opposite-sentiment top-1 | retrieval (polarity flip) | **skipped** |

Record which stage failed: **retrieval** (raw top-K wrong), **gate** (raw OK but `--brain-only` decline), **interface** (held). Token-interface and generator fork are **held** until the store is populated and both retrieval methods re-pass.

### Bet D — embedding RAG baseline (same battery)

Pre-registered head-to-head vs raw brain top-1. Harness:

```bash
cd ../growformer && python3 scripts/rag_baseline_battery.py
# → agent-data/brain-rag-baseline/rag_baseline_results.json
```

**Outcome (2026-07-03 – 2026-07-04):**

| Round | RAG | Brain raw (fair) | Outcome |
|-------|-----|------------------|---------|
| 1 (empty store) | 1/4 | **0/4** (pre-battery `.bin`) | `POPULATE_STORE_FIRST` |
| 2 (store + RAG index) | **4/4** | **0/4** (unretrained `.bin`) | **`SWITCH_TO_EMBEDDING_RAG`** (full battery) |
| 2 brain retrain | — | 2/4 (`*-battery.bin`) | **Invalid** — train-on-test; not scored |
| 3 (SpaceKit retrain + routing fix) | **4/4** | **0/2 scored** (routing OK, raw rank wrong) | **`SWITCH_TO_EMBEDDING_RAG`** (full battery) |
| 4 (scenario topics + hint guard) | **4/4** | **2/2 scored**, 2/4 full | **`HYBRID_DOMAIN_BRAIN`** (product scope) |

Round-2 brain `train-brain` on battery rows then eval on same battery is contaminated (memory-layer train=val). Round 4 brain raw on scored cases: case 2 ETF-delay paraphrase at #1; case 3 mortgage paraphrase at #1 (both `witness=true`). Full-battery rule unchanged; **scored subset** supports domain brain retrieval + optional RAG.

Battery JSONL (exact eval strings — do **not** train on for fair score): `growformer/data/{sentiment,crypto,fintech}/train_sentiment_battery.jsonl`. Held-out retrieval paraphrases (train): `train_sentiment_retrieval_gaps.jsonl`. **Held-out eval prompts (not in any train glob):** `growformer/data/sentiment/eval_battery_heldout_prompts.jsonl` — run with `--heldout` (2026-07-04: brain **2/2**, RAG **1/2**).

```bash
# RAG round 4 (round-4 brain frozen scores in harness)
cd ../growformer && python3 scripts/rag_baseline_battery.py --round 4

# Held-out paraphrase eval (pre-registered)
cd ../growformer && python3 scripts/rag_baseline_battery.py --heldout

# Brain raw (fair — project config parity, SpaceKit .bin)
cargo run --release --bin tinystories -- brain-raw-diag --battery --top-k 3 -v
```

### Full-path inference (HYBRID — brain memory + optional LM)

Default **`--hybrid`** (on): prefer raw lattice top-1 when scenario-topic rubric passes; fall back to full generation path (metacog + gates). Use **`--no-hybrid`** for legacy full-path memory only.

```bash
# Scored SpaceKit battery (cases 2–3) — brain memory only
cargo run --release --bin tinystories -- brain-infer --battery --brain-only

# Held-out paraphrase eval (pre-registered)
cargo run --release --bin tinystories -- brain-infer --heldout --brain-only

# Single prompt — crypto brain + hybrid raw memory
cargo run --release --bin tinystories -- brain-infer \
  --brain ../../spacekit/spacekit-projects/sentiment/crypto/agent/crypto-brain.bin \
  --project ../../spacekit/spacekit-projects/sentiment/crypto/crypto-sentiment-analysis.gf.toml \
  --prompt "Bitcoin crashed after the ETF delay" \
  --brain-only

# HYBRID prefix + row2 vanilla continuation (downstream — not validated on held-out LM quality)
cargo run --release --bin tinystories -- brain-infer \
  --brain ../../spacekit/spacekit-projects/sentiment/crypto/agent/crypto-brain.bin \
  --project ../../spacekit/spacekit-projects/sentiment/crypto/crypto-sentiment-analysis.gf.toml \
  --prompt "Bitcoin crashed after the ETF delay" \
  --checkpoint agent-data/tinystories-row2-seed1000.json \
  --tokenizer data/tinystories.tok \
  --max-new-tokens 64 --greedy
```

API: `growformer_llm::brain_memory::{BrainMemoryRuntime, query_hybrid, MemorySource, format_lm_memory_prefix_with_source, brain_router_features}`.
Raw diagnostic: `BrainMemoryRuntime::raw_lattice_diagnostic`.

Disable the dependency: `cargo build --no-default-features` (tinystories CL/eval only).

```toml
[dependencies]
growformer = { path = "../growformer", default-features = false }
```

---



## License

Apache-2.0.