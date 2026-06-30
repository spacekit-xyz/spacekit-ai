# Growformer Clifford LLM

A Rust language model built on **Space-Time Algebra**, the Clifford algebra **Cl(1,3)**, signature `(+,−,−,−)`. Every linear layer and every attention score
is computed with the **geometric product** of multivectors rather than scalar
dot products and dense matmuls.

- Crate (lib): `growformer_llm` — `use growformer_llm::*;`
- Binary: `tinystories` — the end-to-end BPE → train → eval → generate pipeline

> This README is the **library / algebra reference**. The full training and
> inference pipeline (tape backward, init strategies, gradient accumulation, the
> `tinystories` CLI, prediction⇄compression eval) lives in
> `[src/v2/README.md](src/v2/README.md)`.

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



## Why inference is so fast

People are often surprised that generation/eval are near-instant. It is **not**
an asymptotic trick — it is a small model run forward-only with a const-folded
algebra:

1. **Tiny model.** Defaults are `d_model=16, n_heads=4, d_ff=64, n_blocks=4,
  vocab=2048` — on the order of ~0.7 M parameters. Everything fits in cache.
2. **Forward-only, no tape.** Training allocates an activation tape and runs the
  full bilinear backward through every geometric product (≈2× the forward work
   plus large gradient buffers). Inference does none of that — it is the cheap
   half of the graph.
3. **Compile-time Cayley table.** The 16×16 geometric-product table
  (`CAYLEY_STA`) is evaluated in `const fn`; there is no runtime algebra
   initialisation and the table stays resident in cache.
4. **Stack-allocated math.** A multivector is a fixed `[f32; 16]`; geometric
  products run on the stack with no heap allocation in the hot loop, which the
   optimiser vectorises well.
5. **Real output head.** `LinearReal` is a plain real matmul over `16·d_model`
  features — no blade mixing — even though it is the single largest layer
   (vocab-sized).
6. **Aggressive release build.** `opt-level=3`, `lto=true`, `codegen-units=1`.

**Caveat:** `tinystories generate` recomputes the full `O(seq²·layers)` forward
for every new token — it does *not* yet wire in the `kv_cache` module. It only
*feels* instant because the model and sequences are small; at larger
`d_model`/`seq` that recompute would dominate, and you should route generation
through `kv_cache` (`cached_attention_step`).

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



## Growformer integration (Active Inference)

The sibling `growformer` crate implements an **Active Inference** episode loop
(`growformer::active_inference`): internal `BeliefState`, inward `Observation`,
outward `Action` (Markov blanket). This crate does not contain that spine — use
`growformer` when wiring Clifford-LLM outputs into routing, lattice generation,
or MetaCognition.

```toml
[dependencies]
growformer = { path = "../growformer", default-features = false }
```

```rust
use growformer::active_inference::{ActiveInferenceSpine, BeliefState, EchoPolicy,
    Observation, QueuedEnvironment, SpineConfig};

let mut env    = QueuedEnvironment::from_observations([Observation::UserText("hi".into())]);
let mut belief = BeliefState::new();
let mut policy = EchoPolicy::new("> ");
let spine = ActiveInferenceSpine::new(SpineConfig::default());
let _trace = spine.run_episode(&mut env, &mut belief, &mut policy).unwrap();
```

For full turns, swap `EchoPolicy` for `RoutingGenerationMetacogEpisodePolicy`
(routing + lattice generation + MetaCognition). Enable
`LanguageService::enable_active_inference_replay_log()` to capture reflection
observations for offline replay.

---



## License

MIT or Apache-2.0 (your choice).