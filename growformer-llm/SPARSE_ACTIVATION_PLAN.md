# Sparse Activation (MoE) for growformer-llm : Implementation Plan

**Goal:** add *sparse activation*, a top-k mixture-of-experts (MoE) FFN, to both the
vanilla product stack and the Clifford research stack, so that only a subset of expert
FFNs fire per token. Wired through inference, training/backward, checkpoints, and the CLI.

**Scope decisions (from your answers):** target **both** stacks behind one shared
router/expert abstraction; core mechanism is **top-k MoE** with the **latency-vs-capacity
trade made explicit** and a recommended default.

---

## 1. What "sparse activation" buys you here, and the honest trade

"Sparse activation" in an LLM means: replace the single dense FFN with `N` expert FFNs and
a router that, per token, selects the top-`k` experts (`k ≪ N`). Only those `k` experts
compute. Two very different outcomes fall out of *how you size the experts*, and they must
not be conflated:

| Mode | Expert size | Active FFN FLOPs vs today | Total params | What improves |
|---|---|---|---|---|
| **Capacity** (Switch/MoE scaling) | each expert = today's `d_ff` | `k ×` today (so ~same at `k=1`) | `N ×` FFN params | quality per inference-FLOP |
| **Latency** (conditional compute) | each expert shrunk so `k · d_ff_expert ≤ d_ff` | **< today** | ≈ today or less | raw tokens/sec, at ~equal quality |

Key point for "improve inference": a top-1 router with experts the **same size** as today's
FFN keeps active compute essentially flat while multiplying capacity — you get a better model
at the *same* per-token cost, not a faster one. To actually *cut latency* on a fixed model you
must either shrink per-expert `d_ff` (so the selected experts sum to less than the current FFN)
or add a **null/skip expert** that lets easy tokens bypass the FFN entirely.

**Recommendation:** build the general top-k MoE FFN once, parameterised by `n_experts`,
`top_k`, and per-expert `d_ff`, such that **`n_experts = 1` reproduces today's dense FFN
bit-for-bit** (backward compatible, no checkpoint break). Default the product to a
**capacity** config (`n_experts=4, top_k=1`, experts at current `d_ff`) and expose a
**latency** config (`top_k=1`, shrunk experts, optional skip expert) behind the same flags.
Validate both against the existing bits/byte harness before changing any default.

At this repo's scale (`d_model=16, d_ff=64`, CPU) MoE frequently *underperforms* dense unless
the router is trained with a load-balancing loss and enough experts get gradient signal — so
this plan treats the bits/byte A/B as a gate, not a formality (§9).

---

## 2. Where it plugs into the current code

The FFN is the only thing that changes. Attention, KV cache, layer norm, embeddings, and the
output head are untouched — the FFN is position-local, so the `InferenceCache` needs **no**
cache-format change.

**Vanilla stack (`src/vanilla_llm.rs`)**
- `VanillaFFN` (lines 99–117): `fc1: d_model→d_ff`, ReLU, `fc2: d_ff→d_model`. This becomes the *expert*.
- `vanilla_forward_logits` (lines 175–214): the FFN call at line ~196 (`block.ffn.forward(&y)`) routes through the MoE.
- Training lives in `src/v2/vanilla_train.rs` (forward + backward for this stack).

**Clifford stack (`src/clifford_llm.rs`, `src/ffn.rs`, `src/v2/*`)**
- `FfnVariant` enum (`src/ffn.rs`, lines 97–123): today `Clifford(CliffordFFN)` | `Dense(DenseFFN)`. Add a `Moe` variant.
- Non-cached forward: `CliffordBlock::forward` (`clifford_llm.rs` line ~391).
- **Cached inference forward:** `block_forward_cached` (`src/v2/inference.rs` line ~117, `block.ffn.forward(&n2)`).
- **Taped forward (training):** `ffn_forward_taped` (`src/v2/tape.rs` line ~252) and the `FfnTape`/`FfnHidden` types (lines ~70–106).
- **Backward:** `ffn_backward` and `FfnGrad` (`src/v2/block_backward.rs` lines ~22–210).

**Router reuse note:** there is already an `LmAdjustableConeRouter` (`src/lm_cone_router.rs`)
— a learned 2-way cone router over confidence features. It is *not* a per-token gate over the
residual stream, so it is the wrong tool for N-way MoE gating. Build a small dedicated
`GatingNetwork` (below). Keep the cone router in mind only as an alternative for an explicit
**2-expert** configuration, where its narrow/wide margin logic could pick expert A vs B.

---

## 3. Core design

### 3.1 Gating network
A lightweight learned linear gate on the (flattened) post-`norm2` residual:

```
logits_g = W_g · x            # W_g: [n_experts × in_dim], in_dim = d_model (vanilla) or 16·d_model (clifford)
p        = softmax(logits_g)  # dense gate distribution
top_k    = argtopk(p, k)      # selected expert indices (deterministic tie-break by index)
g_e      = p_e / Σ_{j∈top_k} p_j   for e ∈ top_k   # renormalised over the chosen k
y        = Σ_{e∈top_k} g_e · expert_e(x)
```

- Use `LinearReal` (already in `real_linear.rs`) for `W_g` in both stacks — the Clifford
  gate reads the same flattened `16·d_model` features the output head uses (`flatten_mvs`).
- **Dropless** routing (no capacity factor / token dropping): every token computes exactly its
  `k` experts. Simpler and correct on CPU at this scale; capacity limits can come later if
  batched throughput ever matters.
- **Deterministic routing** is mandatory so the `cached_matches_full_forward` /
  `incremental_extend_matches_batch` parity tests (`inference.rs` lines ~200–238) still hold:
  the taped path and the cached path must select identical experts. Tie-break argmax by lowest
  index; add router noise **only** under a training flag, never at eval/generate.

### 3.2 Expert
An expert is exactly the existing FFN type for that stack:
- Vanilla: `VanillaFFN` (real `fc1/ReLU/fc2`).
- Clifford: `CliffordFFN` (geo-linear) or `DenseFFN` — reuse `FfnVariant` per expert so the
  MoE composes with the existing Clifford/dense ablation switch.

### 3.3 Optional skip / null expert (latency mode)
Reserve expert index 0 as an identity (returns zero delta → token bypasses the FFN). With
`top_k=1` this gives per-token conditional compute: the router learns which tokens need FFN
work. Gated behind a `moe_skip_expert` flag.

### 3.4 Load-balancing auxiliary loss (training only)
Without it the router collapses onto one expert. Use the Switch-Transformer importance/load
loss:

```
aux = n_experts · Σ_e (fraction_tokens_routed_to_e · mean_gate_prob_e)
loss = task_loss + moe_aux_weight · aux
```

Accumulate per optimiser step (respecting the existing `grad_accum` averaging in
`train_step_v2_accum`). Default `moe_aux_weight = 1e-2`.

---

## 4. New / changed types

```rust
// src/moe.rs  (new, shared by both stacks)
pub struct GatingNetwork { pub w_g: LinearReal, pub n_experts: usize, pub top_k: usize }
impl GatingNetwork {
    pub fn route(&self, x_flat: &[f32]) -> RouteDecision; // top-k indices + renormalised weights + full probs (for aux loss)
}
pub struct RouteDecision { pub experts: Vec<usize>, pub weights: Vec<f32>, pub probs: Vec<f32> }

// Vanilla: src/vanilla_llm.rs
pub struct VanillaMoeFFN { pub gate: GatingNetwork, pub experts: Vec<VanillaFFN> }
// enum so n_experts==1 stays the old path:
pub enum VanillaFfn { Dense(VanillaFFN), Moe(VanillaMoeFFN) }

// Clifford: src/ffn.rs  (extend existing enum)
pub enum FfnVariant {
    Clifford(CliffordFFN),
    Dense(DenseFFN),
    Moe { gate: GatingNetwork, experts: Vec<FfnVariant> },   // experts reuse the enum
}
```

**Tape (`src/v2/tape.rs`)** — extend `FfnHidden` with a `Moe` case recording, per token: the
gate logits/probs, the selected expert indices, and the selected experts' own hidden tapes
(so backward only touches experts that fired).

**Grad (`src/v2/block_backward.rs`)** — extend `FfnGrad` with `Moe(Vec<FfnGrad>, GatingGrad)`;
`GatingGrad` is a `RealHeadGrad` for `W_g`. Scale/clip/accumulate mirror the existing arms.

---

## 5. Backward pass (training)

For each token, gradient flows to (a) the selected experts and (b) the gate:
- **Experts:** `dL/d expert_e = g_e · grad_out`; recurse into the existing per-expert backward
  (`linear_backward` for Clifford, `real_linear_backward` for dense) — already implemented, just
  invoked per selected expert.
- **Gate weights:** `dL/d g_e = <grad_out, expert_e(x)>`; push through the top-k **renormalise**
  then **softmax** (standard Jacobian; straight-through on the discrete top-k selection — only
  selected logits receive gradient). Accumulate into `GatingGrad`.
- **Input:** sum the `dL/dx` contributions from each selected expert plus the gate's
  `dL/dx = (dL/d logits_g)·W_g`; this replaces the single `grad_ffn_in` returned today.
- Add `moe_aux_weight · d(aux)/d logits_g` to the gate gradient.

Wire `FfnGrad::Moe` into `StepGrads` and the Adam step in `apply_grads_v2` (one optimiser clock
tick per step, unchanged).

---

## 6. Config & checkpoints (`src/lm_config.rs`)

Add, all `#[serde(default)]` so **every existing checkpoint keeps loading**:

```rust
#[serde(default = "one")]        pub n_experts: usize,      // 1 = today's dense FFN (no-op)
#[serde(default = "one")]        pub moe_top_k: usize,      // 1 = switch-style
#[serde(default)]                pub moe_aux_weight: f32,   // 0.0 off; 1e-2 recommended
#[serde(default)]                pub moe_router_noise: f32, // train-only jitter; 0.0 at eval
#[serde(default)]                pub moe_skip_expert: bool, // reserve expert 0 as identity
#[serde(default)]                pub moe_expert_d_ff: usize,// 0 => use d_ff (capacity mode)
```

- `n_experts == 1` ⇒ construct the plain `Dense`/`Clifford` FFN, not the MoE wrapper — so the
  serialized model and forward numerics are identical to today (no schema bump needed for the
  default; bump the checkpoint schema version only when `n_experts > 1` writes expert arrays).
- `sync_tied_head` is unaffected (head ↔ embedding tie is independent of the FFN).
- Serialize experts as a `Vec` and the gate's `W_g`; keep the `.tok`-beside-`.json` convention.

---

## 7. Inference integration

- **Clifford (KV-cached):** `block_forward_cached` (`inference.rs`) calls `block.ffn.forward`;
  since the MoE forward is also just `&[Multivector] → Vec<Multivector>`, this is a drop-in.
  The KV cache holds only K/V (attention) — **no cache change**. Re-run the parity tests with
  `n_experts=4` to prove cached == full.
- **Vanilla:** swap the `block.ffn.forward(&y)` call in `vanilla_forward_logits`. (If you want
  KV-cached vanilla generation too, that's a separate item — today vanilla recomputes the full
  sequence; MoE doesn't change that either way.)
- **generate / eval CLIs** need no logic change beyond loading the new config fields.

---

## 8. Phased rollout

| Phase | Deliverable | Exit check |
|---|---|---|
| **0. Scaffolding** | `src/moe.rs`, config fields, enum variants; `n_experts=1` builds the old FFN | full test suite green, checkpoints still load, byte-identical logits vs pre-change at `n_experts=1` |
| **1. Forward MoE (inference)** | MoE forward in both stacks; frozen/random gate | `cached_matches_full_forward` + `incremental_extend_matches_batch` pass with `n_experts=4`; shapes correct |
| **2. Training + backward** | `GatingNetwork` grads, `FfnGrad::Moe`, aux loss, tape/backward wiring | new finite-diff gradient test on gate + experts; `end_to_end_loss_decreases` passes with MoE |
| **3. Eval & tuning** | bits/byte A/B vs dense at matched **active** FLOPs; expert-utilisation + balance logging | MoE ≥ dense on val bits/byte at equal active-FLOPs, no dead experts (>~5% load each) |
| **4. Latency mode** | shrunk experts + optional skip expert; CLI flags; latency microbench | tokens/sec improves vs dense at ≥ parity bits/byte |

Ship Phase 0–1 behind the default `n_experts=1` (pure no-op) so `main` stays safe throughout.

---

## 9. Testing & validation

- **Parity:** extend `inference.rs` tests to `n_experts=4, top_k=1/2` (cached == full).
- **Gradient:** finite-difference check on `W_g` and on a selected expert, in the style of the
  existing `tape::tests` / `block_backward::tests`.
- **Learning:** `train_v2::tests::end_to_end_loss_decreases` with MoE config (loss < 75% of start).
- **Balance:** after a short train, assert min expert load fraction above a floor (router not collapsed).
- **Quality gate:** run the `tinystories eval` bits/byte harness — MoE vs dense baseline at
  **matched active FLOPs** (this is the real "did sparse activation help" test, since bits/byte
  = compression rate is the repo's north-star metric).
- **Latency:** microbench tokens/sec via `InferenceCache` at fixed `n_experts`, dense vs
  top-1-shrunk-experts.

---

## 10. CLI (`src/bin/tinystories.rs`)

```bash
# capacity mode (recommended default to trial): 4 experts, switch routing
cargo run --release --bin tinystories -- train \
  data/tinystories.tok data/train.bin data/val.bin \
  --checkpoint-out agent-data/lm_moe.json --steps 4000 --seq-len 128 \
  --d-model 16 --n-heads 4 --d-ff 64 --n-blocks 4 \
  --n-experts 4 --moe-top-k 1 --moe-aux 0.01 --tie-embeddings --semantic-init

# latency mode: top-1, quarter-size experts + skip expert
  ... --n-experts 4 --moe-top-k 1 --moe-expert-d-ff 16 --moe-skip-expert

# eval / generate: unchanged commands; just point at the MoE checkpoint
```

`--n-experts 1` (default) = today's behaviour exactly.

---

## 11. Risks & mitigations

- **MoE < dense at tiny scale / router collapse** → aux loss on by default when `n_experts>1`;
  start `n_experts=4, top_k=1`; gate the default flip on the bits/byte A/B.
- **"Sparse ≠ faster" surprise** → the table in §1 and the latency mode make the trade explicit;
  don't flip the product default to capacity mode expecting a speedup.
- **Parity test breakage from nondeterministic routing** → deterministic argmax tie-break;
  router noise train-only.
- **Checkpoint compat** → `n_experts=1` writes the old format; schema bump only for `>1`.
- **CPU reality** → real latency wins need shrunk per-expert `d_ff` or the skip expert; top-1
  same-size experts are a capacity play, not a speed play.

---

## 12. File-change summary

| File | Change |
|---|---|
| `src/moe.rs` *(new)* | `GatingNetwork`, `RouteDecision`, routing + aux-loss helpers |
| `src/lm_config.rs` | new `#[serde(default)]` MoE fields |
| `src/vanilla_llm.rs` | `VanillaMoeFFN`, `VanillaFfn` enum, forward swap |
| `src/ffn.rs` | `FfnVariant::Moe` variant + forward |
| `src/clifford_llm.rs` | `CliffordBlock::forward` FFN call via enum (mostly transparent) |
| `src/v2/inference.rs` | `block_forward_cached` FFN call (drop-in) + extended parity tests |
| `src/v2/tape.rs` | `FfnHidden::Moe`, `ffn_forward_taped` MoE arm |
| `src/v2/block_backward.rs` | `FfnGrad::Moe`, `ffn_backward` MoE arm, gate grad |
| `src/v2/train_v2.rs` | aux loss into task loss; MoE grads into `StepGrads`/`apply_grads_v2` |
| `src/v2/vanilla_train.rs` | same for the vanilla training path |
| `src/v2/checkpoint.rs` + `vanilla_checkpoint.rs` | serialize experts + gate; schema bump for `n_experts>1` |
| `src/bin/tinystories.rs` | `--n-experts / --moe-top-k / --moe-aux / --moe-expert-d-ff / --moe-skip-expert` flags |
```
