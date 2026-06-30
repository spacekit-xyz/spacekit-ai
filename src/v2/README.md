# v2: training & inference pipeline

The `v2` module is the **full, tape-based training pipeline** for the Clifford
LLM, plus sampling, checkpointing, and the prediction⇄compression eval.

> For the algebra, the layer types (`Multivector`, `CliffordAttention`,
> `LinearReal`, …) and the library primitives (`blade`, `cayley_const`,
> `backprop`, `optim`, `positional`, `mask`, `kv_cache`), see the top-level
> `[../../README.md](../../README.md)`. This document does **not** repeat them.

What v2 adds over a forward pass:

- `model_forward_taped` records every activation into a `Tape`.
- A full backward pass — attention Q/K/V/O, both layer-norm γ/β pairs, the FFN,
the `final_norm`, the `LinearReal` head, **and** the embedding table all get
proper gradients (the legacy `train` module only updated the head).
- Adam everywhere, cosine LR with warmup, gradient clipping.
- **Weight tying**, three **embedding-init strategies**, and **gradient
accumulation** (effective batch size).

---



## Backward architecture

```
ids ─▶ model_forward_taped ─▶ Tape
                               │
                               ▼  cross_entropy at every loss-masked position
                           grad_logits
                               │
                               ▼  real_head_backward (LinearReal)  → grad on final_norm output
                          final_norm backward                      → grad_x (last block output)
                               │
                               ▼  for each block in reverse:
                          block_backward(tape, grad_x) → BlockGrads
                            • attention_backward (Q/K/V/O + per-head softmax)
                            • norm1 + norm2 (γ, β)
                            • FFN (fc1 → ReLU → fc2)
                               │
                               ▼  sparse embedding update (+ tied head-weight path)
```

The single step is split into two pure-ish halves so gradients can be
**averaged across microbatches** before one optimiser step:


| function                                   | role                                                                                               |
| ------------------------------------------ | -------------------------------------------------------------------------------------------------- |
| `compute_grads_v2(state, ex) -> StepGrads` | forward + full backward; **no** optimiser/model mutation. Grads are `1/n_loss`-scaled and clipped. |
| `apply_grads_v2(state, &StepGrads)`        | one Adam step over head, blocks, `final_norm`, embeddings; advances every optimiser clock once.    |
| `train_step_v2(state, ex)`                 | `compute` + `apply` for a single microbatch.                                                       |
| `train_step_v2_accum(state, &[ex])`        | accumulate over N microbatches, average, apply once (gradient accumulation).                       |
| `train_step_v2_head_only(state, ex)`       | forward + head-only backward (cheap sanity path; blocks/embeddings frozen).                        |


---



## Config (`TrainConfigV2`)

`TrainConfigV2::small(vocab)` defaults: `d_model=8, n_heads=2, d_ff=32, n_blocks=2, max_seq=128, lr_max=3e-4, lr_min=1e-5, warmup=100, grad_clip=1.0`.
(The `tinystories` CLI overrides the architecture knobs — its CPU defaults are
`d_model=16, n_heads=4, d_ff=64, n_blocks=4`.)

Training knobs worth knowing:


| field               | default | meaning                                                                        |
| ------------------- | ------- | ------------------------------------------------------------------------------ |
| `train_embeddings`  | `true`  | update the embedding table (false = freeze a fixed encoder)                    |
| `init_seed`         | fixed   | seed for random init (symmetry breaking)                                       |
| `tie_embeddings`    | `false` | share the embedding table with the output head (`sync_tied_head`)              |
| `structured_init`   | `false` | deterministic unit-norm Gaussian per token instead of tiny uniform noise       |
| `grad_accum`        | `1`     | microbatches averaged per optimiser step (effective batch size)                |
| `freeze_embeddings` | `false` | inheritance: freeze the shared base embedding                                  |
| `freeze_blocks`     | `0`     | inheritance: freeze blocks `[0, freeze_blocks)`; grads still flow through them |


All of the `#[serde(default)]` fields keep older checkpoints loadable.

---



## Embedding initialisation

The embedding table is the LLM's strongest lever at small scale. Three options,
in increasing order of prior strength:

1. **Random uniform** (default): tiny noise; every token starts near-identical.
2. `structured_embedding_init` (`--structured-init`), a deterministic
  unit-norm Gaussian per token (ported from growformer's
   `ChunkCodec::build_token_embeddings`). Distinct identities from step 0, and
   with tying a usable output classifier from step 0. Helps at larger scale; at
   `d_model=16` it is roughly a wash.
3. `corpus_semantic_init`, **random indexing**: each token gets a random index
   vector, then ±`window` neighbour vectors are distance-weighted-accumulated
   from the corpus, so tokens sharing contexts end up with similar embeddings
   (the distributional hypothesis, one pass). A genuine *semantic* prior, not
   just symmetry breaking — and the **default for fresh `tinystories` training**
   (opt out with `--no-semantic-init`; `--structured-init` selects #2).

**A/B (sentiment corpus, tied head, 200 steps, identical config):**


| init                | val NLL   | val ppl |
| ------------------- | --------- | ------- |
| random uniform      | 6.869     | 962     |
| structured random   | 7.094     | 1204    |
| **corpus-semantic** | **6.411** | **608** |


Corpus-semantic init cut validation perplexity ~37 % at equal step count — the
takeaway from the growformer audit: the *random* part of structured init does
not help, but *distributional structure* does.

---

## Gradient accumulation

`--grad-accum N` averages the gradients of `N` independent microbatches before a
single Adam step (one optimiser-clock tick per effective batch). It reduces
step-to-step gradient noise — valuable at `d_model=16` where the per-step loss
otherwise bounces — without the memory cost of a true batched forward. Total
chunks consumed = `steps × N`; the LR schedule still counts optimiser steps.

---

## `tinystories` CLI (end-to-end)

```bash
# 1. Train a BPE tokenizer and pack the corpus to token bins
cargo run --release --bin tinystories -- tokenize data/TinyStories-valid.txt 2048 data/tinystories.tok
cargo run --release --bin tinystories -- encode   data/TinyStories-valid.txt data/tinystories.tok data/val.bin

# 2. Train (recommended small-scale config)
cargo run --release --bin tinystories -- train \
  data/tinystories.tok data/train.bin data/val.bin \
  --checkpoint-out agent-data/lm.json \
  --steps 4000 --seq-len 128 \
  --d-model 16 --n-heads 4 --d-ff 64 --n-blocks 4 \
  --tie-embeddings --semantic-init --grad-accum 4

# 3. Bits/byte eval (prediction ⇄ compression vs gzip/lzma)
cargo run --release --bin tinystories -- eval \
  --checkpoint agent-data/lm.json --tokenizer data/tinystories.tok \
  data/val.bin --seq-len 128 --windows 32

# 4. Generate (pass the same .tok — the checkpoint omits the word list)
cargo run --release --bin tinystories -- generate \
  --checkpoint agent-data/lm.json --tokenizer data/tinystories.tok \
  --prompt "Once upon a time" --max-new-tokens 64
```

Useful `train` flags: `--head-only` (fast head-only sanity run),
`--init-from <ckpt>` (inherit a base model's architecture + weights, then
fine-tune), `--freeze-blocks N`, `--freeze-embeddings`, `--no-semantic-init`
(disable the default semantic init), `--structured-init`, `--semantic-window K`,
`--grad-accum N`, `--sample-every`, `--val-chunks`.

Notes:

- Corpus-semantic init is **on by default** for fresh training: it seeds
  embeddings from the **training corpus** post-construction, overrides
  `--structured-init`, and is ignored with `--init-from` (inherited embeddings
  are kept).
- If decoded text repeats (`the the the`), raise `--repetition-penalty` on
`generate`.
- Each training step is one taped forward over the chunk (teacher forcing) plus
a full backward through every Clifford block — that backward dominates CPU
time (~2× the forward).

---

## Prediction ⇄ compression

A model's cross-entropy in **bits/byte** is exactly its lossless compression
rate. `eval` reports the model's bits/byte against `gzip -9` and `lzma -9` on the
identical byte stream (via `spacekit-compressor`). The `arithmetic` module
(`ArithmeticEncoder`/`ArithmeticDecoder`) provides a round-trip range coder that
turns the model's next-token distribution into an actual bitstream, demonstrating
the equivalence directly.

---

## Sampling

```rust
use growformer_llm::v2::sample::*;

let cfg = SampleConfig {
    temperature: 0.8, top_p: Some(0.9), top_k: Some(50),
    repetition_penalty: 1.1, max_new_tokens: 256,
    stop_tokens: vec![growformer_llm::v2::data::special::EOS], seed: Some(42),
};
let next = sample_next(&logits_last, &context_ids, &cfg, &mut SimpleRng::new(42));
```

Presets: `SampleConfig::greedy()`, `::focused()` (T=0.7, top-p=0.9, rep=1.15),
`::creative()` (T=1.0, top-p=0.95, rep=1.1).

---

## Checkpoints

`save_lm_state` / `load_lm_state` serialise weights + `TrainConfigV2` to JSON
(schema-versioned; the `LinearReal` head and `final_norm` are stored explicitly).
The word list is **not** embedded — keep the `.tok` next to the `.json` and pass
both to `eval`/`generate`. Tied checkpoints call `sync_tied_head` on load so the
head mirror stays consistent.

---

## Gradient verification

```bash
cargo test --release tape::tests
cargo test --release attention_backward::tests
cargo test --release block_backward::tests
cargo test --release train_v2::tests::end_to_end_loss_decreases
```

`end_to_end_loss_decreases` trains the full model on one example for ~120 steps
and asserts the loss drops below 75 % of its starting value — the integration
check that the compute/apply split stays correct.

---

## License

Apache-2.0