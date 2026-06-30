# Growformer Clifford LLM

A Rust language model built on **Space-Time Algebra**, Clifford algebra **Cl(1,3)**,
signature `(+,−,−,−)`. Linear layers and attention scores use the **geometric
product** of multivectors rather than scalar dot products and dense matmuls.

- Crate (lib): `growformer_llm` — `use growformer_llm::*;`
- Binary: `tinystories` — BPE → train → eval (bits/byte) → generate

> **What this repo is.** A from-scratch Clifford-algebra transformer with a full
> training stack (tape backward, Adam, checkpoints, arithmetic coding). The *how*
> is documented in detail below and in `[src/v2/README.md](src/v2/README.md)`.
>
> **What it is not yet.** A demonstrated win for geometric algebra on language.
> Tokens do not live in Minkowski spacetime; Cl(1,3) here is a *hypothesis* about
> structured mixing, not a principled equivariance inductive bias the way GA layers
> are for E(3) point clouds or molecular geometry (Brandstetter et al., GATr). The
> experiment that would settle this — **matched-parameter bits/byte vs. a vanilla
> transformer and a dense-linear ablation** — has not been run yet. See
> [Research status](#research-status).

---



## Research status



### What is verified


| Claim                                | Evidence                                                                                                                                                              |
| ------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Algebra & backward are correct       | Cayley-table identities, finite-difference grad checks, `train_v2::end_to_end_loss_decreases`                                                                         |
| Pipeline works end-to-end            | BPE, packed bins, train, checkpoint, generate, `eval` (bits/byte vs gzip/lzma)                                                                                        |
| Training tricks help (domain corpus) | On sentiment headlines (`d_model=16`, tied, 200 steps): corpus-semantic init **val ppl 608** vs uniform **962** (~37% ↓) — see `[src/v2/README.md](src/v2/README.md)` |


These show the stack *trains* and that **embedding priors** matter. They do **not**
show that the geometric product beats a matched dense layer on language modeling quality.

### What is missing (the number that matters)

Hold-out **bits/byte on TinyStories** (or any fixed corpus), same tokenizer, same
training budget, three rows:


| Model                       | Params           | bits/byte (val)        | Notes                                                                                                                             |
| --------------------------- | ---------------- | ---------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| Clifford (row 1)            | ~0.7M @ defaults | **2.34** (conditional) | `tinystories-row1.json`: 400 steps, semantic init, tie, grad-accum 2, eval on `val.bin` 64×128 windows; **weights not amortized** |
| Matched vanilla transformer | same             | —                      | **needed**                                                                                                                        |
| Dense-linear ablation       | same             | —                      | swap `CliffordLinear` for real matmul of equal param count; isolates *structure*                                                  |


Row 1 eval (2026-06-30): conditional **2.34 bpb** vs gzip **2.59** vs lzma **4.34** on 35 583 bytes
(8024 text tokens). Pipeline validated; model is under-trained (400 steps, train=val shard).
**Beats gzip on conditional CE only** — not a shipped-size claim. Ablation still warranted once
training budget is increased.

Until that table exists, treat Cl(1,3) for text as **unproven engineering exploration**,
not a research result.

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
| **Parameter-matched** | **Primary**      | Same total learnable scalar count. A `CliffordLinear(out, in)` stores `out×in` multivector weights + `out` multivector biases → `16·(out×in + out)` floats. The dense ablation flattens each position to `16·d_model` reals and uses a standard `LinearReal`-style layer with the same scalar count (not the same `d_model` label). Answers: *does Cayley structure help at equal capacity?* |
| **FLOP-matched**      | Report alongside | Count multiply-adds in forward (geo product vs dot). Clifford layers do more work per param; report FLOPs separately so “wins at equal params but 3× FLOPs” is visible.                                                                                                                                                                                                                      |
| **Width-matched**     | Secondary only   | Same `d_model`, `n_blocks`, `n_heads` — convenient but **not** equal params (Clifford cells are 16-wide). Do not use as the headline comparison.                                                                                                                                                                                                                                             |


**Implementation scope (not a one-line flag).** A faithful dense ablation needs a
parallel layer type wired through **forward, tape backward, and inference cache** —
a second backward path, not a runtime toggle on `CliffordLinear` alone.

**Vanilla transformer row (row 2).** Same tokenizer, corpus, training budget, and
eval harness; parameter-matched real-valued transformer (dot-product attention,
standard LayerNorm, same head tying options). External baseline or in-crate — TBD.

### Row 1 procedure (TinyStories)

```bash
# Train on packed validation shard (small-run convention; replace with train.bin when available)
cargo run --release --bin tinystories -- train \
  data/tinystories.tok data/val.bin data/val.bin \
  --checkpoint-out agent-data/tinystories-row1.json \
  --steps 400 --tie-embeddings --grad-accum 2

# Conditional bits/byte (+ gzip/lzma on the same bytes)
cargo run --release --bin tinystories -- eval \
  --checkpoint agent-data/tinystories-row1.json \
  --tokenizer data/tinystories.tok \
  data/val.bin --seq-len 128 --windows 64
```

Semantic init is on by default for fresh training. Sentiment-corpus runs are a
separate track and do not fill row 1.

### Open design questions

**Why Cl(1,3) for text at all?** In the GA-for-physics literature, the geometric
product earns its keep through **equivariance** to a symmetry the data actually has.
Discrete tokens have no Lorentz symmetry. Here the Cayley table is a fixed sparse
bilinear mixing pattern — possibly a useful parameterization, possibly a more
expensive constrained linear layer. We do not know which without the ablation above.

**Attention scores** use the grade-0 part of `Qᵢ ⊛ K̃ⱼ` — a **Minkowski** (indefinite)
bilinear form, not a positive-definite dot product. Scores can be negative in ways a
standard dot-product attention cannot; LayerNorm and softmax may tame this, but it is
a plausible source of training quirks worth monitoring.

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

- **Genuine multi-head attention** — each head owns the channel slice
`[h·head_dim, (h+1)·head_dim)`, with its own Q/K/V projection and softmax.
Scores are the grade-0 (scalar) part of `Qᵢ ⊛ K̃ⱼ`, scaled by `1/√(head_dim·16)`.
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



## Growformer sibling crate (optional)

The separate `[growformer](../growformer)` crate implements lattice routing,
sentiment brains, and an Active Inference episode loop. It is **not** part of this
LM’s training path; use it when wiring LM outputs into domain-specific inference.
No claim is made here that spacetime algebra + free-energy framing improves
language modeling — that integration is exploratory.

```toml
[dependencies]
growformer = { path = "../growformer", default-features = false }
```

---



## License

Apache-2.0.