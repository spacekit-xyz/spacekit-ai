# SpaceKit NCA

A Rust implementation of a transformer-style language model built on
**Space-Time Algebra** — Clifford algebra Cl(1,3), signature (+,−,−,−).

Every linear layer and every attention score is computed via the **geometric
product** of multivectors rather than scalar dot products and matrix
multiplication.

---

## Crate layout

```
Cargo.toml
src/
  lib.rs            — module declarations and re-exports
  clifford_llm.rs   — core types: Multivector, CliffordAlgebra, all layers
  blade.rs          — blade index constants, grade utilities, display
  cayley_const.rs   — compile-time Cayley table, CliffordAlgebraConst
  backprop.rs       — gradient types and backward pass
  optim.rs          — Adam optimiser, LR schedule, gradient clipping
  positional.rs     — rotor-based positional encoding
  mask.rs           — causal and padding masks
  kv_cache.rs       — KV cache for autoregressive inference
```

---

## Algebra at a glance

Cl(1,3) has 2⁴ = **16 basis blades**.  The index used throughout the crate is
the **bit-mask** of the blade: bit k set means basis vector eₖ is present.

| index (bitmask) | blade | grade |
|---|---|---|
| `0` = `0b0000` | `1` | 0 — scalar |
| `1` = `0b0001` | `e0` | 1 |
| `2` = `0b0010` | `e1` | 1 |
| `3` = `0b0011` | `e01` | 2 |
| `4` = `0b0100` | `e2` | 1 |
| `5` = `0b0101` | `e02` | 2 |
| `6` = `0b0110` | `e12` | 2 |
| `7` = `0b0111` | `e012` | 3 |
| `8` = `0b1000` | `e3` | 1 |
| `9` = `0b1001` | `e03` | 2 |
| `10` = `0b1010` | `e13` | 2 |
| `11` = `0b1011` | `e013` | 3 |
| `12` = `0b1100` | `e23` | 2 |
| `13` = `0b1101` | `e023` | 3 |
| `14` = `0b1110` | `e123` | 3 |
| `15` = `0b1111` | `e0123` | 4 — pseudoscalar |

Metric signature: `e0² = +1`, `e1² = e2² = e3² = −1`.

A `Multivector` is `[f32; 16]` — one component per blade at the index above.
The scalar (grade-0) part is always `c[0]`.

---

## Quick start

```rust
use spacekit_nca::*;
use std::sync::Arc;

let algebra_arc = Arc::new(CliffordAlgebra::sta());

let d_model = 8;
let n_heads = 2;
let vocab   = 256;

let blocks: Vec<CliffordBlock> = (0..4).map(|_| CliffordBlock {
    attn:  CliffordAttention::new(d_model, n_heads, algebra_arc.clone()),
    ffn:   CliffordFFN::new(d_model, 32, algebra_arc.clone()),
    norm1: CliffordLayerNorm::new(d_model),
    norm2: CliffordLayerNorm::new(d_model),
}).collect();

let model = CliffordLLM {
    embedding: vec![vec![Multivector::scalar(0.01); d_model]; vocab],
    blocks,
    head: CliffordLinear::new(d_model, vocab, algebra_arc.clone()),
    algebra: algebra_arc,
};

let logits = model.forward(&[1, 42, 7, 0]);  // → Vec<Vec<f32>> [seq][vocab]
```

---

## Modules

---

### `blade` — blade index constants and grade utilities

Named constants for every blade so you never use raw integers:

```rust
use spacekit_nca::blade::*;

let mut mv = Multivector::zero();
mv.c[E12] = 1.0;
mv.c[E0]  = 0.5;

println!("{}", display(&mv));         // "0.5000e0 + e12"
let grade2 = project_grade(&mv, 2);  // zero everything except grade-2
let bv     = bivector_part(&mv);     // → [e01, e02, e12, e03, e13, e23]
```

| symbol | type | description |
|---|---|---|
| `SCALAR`, `E0`…`E0123` | `usize` | Named blade indices (bit-mask values) |
| `BLADE_NAMES` | `[&str; 16]` | Human-readable name per index |
| `BLADE_GRADES` | `[u8; 16]` | Grade (0–4) per index |
| `REVERSE_SIGNS` | `[f32; 16]` | Sign flip for the reverse operation per grade |
| `grade_of(idx)` | `fn` | Grade of a blade index |
| `blades_of_grade(k)` | `fn` | All blade indices of grade k |
| `project_grade(mv, k)` | `fn` | Zero all components except grade k |
| `vector(a0,a1,a2,a3)` | `fn` | Build a grade-1 multivector |
| `bivector(...)` | `fn` | Build a grade-2 multivector from 6 components |
| `display(mv)` | `fn` | Pretty-print, skipping near-zero terms |

---

### `cayley_const` — compile-time Cayley table

The full 16×16 geometric product table is evaluated at compile time via
`const fn` and stored in the static `CAYLEY_STA`.  No heap allocation, no
initialisation cost.

```rust
use spacekit_nca::cayley_const::{CAYLEY_STA, CliffordAlgebraConst};

// Inspect a single cell
let cell = CAYLEY_STA[E1][E2];      // e1 ⊛ e2
assert_eq!(cell.sign,  1);           // → +e12
assert_eq!(cell.blade, E12 as u8);

// Use as a zero-cost const alternative to the runtime CliffordAlgebra
const ALG: CliffordAlgebraConst = CliffordAlgebraConst::new();
let result     = ALG.geo_product(&a, &b);
let r_rev      = ALG.reverse(&r);
let sandwiched = ALG.sandwich(&r, &x);  // r ⊛ x ⊛ r̃
let comm       = ALG.commutator(&a, &b);
```

`CliffordAlgebraConst` has the same API as the runtime `CliffordAlgebra` and
can be declared as a `const` item anywhere in your code.

---

### `backprop` — gradient types and backward pass

The geometric product is bilinear, so gradients follow a clean product rule.
Given `C = geo(A, B)`:

```
dL/dA[i] = Σ_j  B[j] × cayley[i][j].sign × dL/dC[ cayley[i][j].blade ]
dL/dB[j] = Σ_i  A[i] × cayley[i][j].sign × dL/dC[ cayley[i][j].blade ]
```

```rust
use spacekit_nca::backprop::*;

// Primitive backward through one geometric product
let (grad_a, grad_b) = geo_product_backward(&a, &b, &grad_c);

// Backward through a full CliffordLinear layer
let (grad_layer, grad_x) = linear_backward(&weights, &inputs, &grad_out);
// grad_layer: GradLinear { d_weights, d_biases }
// grad_x:     Vec<Multivector> to propagate further down

// Cross-entropy loss over scalar logits (output head)
let (loss, grad_logits) = cross_entropy(&logits, target_token_id);

// Scatter scalar logit grads into multivector grad_out for the head
let mv_grad = scalar_head_backward(&grad_logits);
```

`GradLinear` can be accumulated across a minibatch:

```rust
let mut total = GradLinear::zeros(out_dim, in_dim);
total.accumulate(&sample_grad);
total.scale(1.0 / batch_size as f32);
```

---

### `optim` — Adam optimiser

```rust
use spacekit_nca::optim::*;

let cfg = AdamConfig { lr: 3e-4, weight_decay: 0.01, ..Default::default() };

// Single multivector parameter update
let mut state = MvAdamState::zero();
let new_param = adam_step(&param, &grad, &mut state, &cfg);

// Layer-level wrapper (manages all weight and bias states internally)
let mut opt = LayerOptimizer::new(out_dim, in_dim, cfg);
opt.step(&mut weights, &mut biases, &grad_layer);

// Gradient utilities
clip_grad_norm(&mut grad, 1.0);                          // clip to max norm
let norm = grad_norm(&grad);

// Cosine decay with linear warmup
let lr = cosine_lr_with_warmup(step, warmup_steps, total_steps, 3e-4, 1e-5);
```

`AdamConfig` defaults: `lr=1e-3`, `beta1=0.9`, `beta2=0.999`, `eps=1e-8`,
`weight_decay=0.0`.

---

### `positional` — rotor positional encoding

Positions are encoded as **rotors** acting via the sandwich product
`R(t) ⊛ x ⊛ R̃(t)`.  Cl(1,3) provides six independent bivector planes:

| plane | kind | description |
|---|---|---|
| `e12`, `e13`, `e23` | rotation | ordinary spatial rotation (`cos`/`sin`) |
| `e01`, `e02`, `e03` | boost | Lorentz boost (`cosh`/`sinh`) |

```rust
use spacekit_nca::positional::*;

let alg = CliffordAlgebraConst::new();
let pe  = RotorPositionalEncoding::new(d_model); // cycles through all 6 planes

// Encode a full embedded sequence
let encoded = pe.encode(&alg, &embedded_sequence);  // [seq_len][d_model]

// Single position (for cached generation)
let enc_t = pe.encode_position(&alg, &token_mvs, position_t);

// Manual rotor construction
let r_rot   = make_rotor(theta, BivectorPlane::E12);  // spatial rotation
let r_boost = make_rotor(theta, BivectorPlane::E01);  // Lorentz boost
let rotated = apply_rotor(&alg, &r_rot, &x);

// Pre-compute rotors for fast inference
let rotor_table = pe.precompute_rotors(max_seq_len);
```

Angles follow the same log-spaced schedule as sinusoidal PE:
`θ(t, d) = t / 10000^(2d / d_model)`.

---

### `mask` — causal and padding masks

```rust
use spacekit_nca::mask::*;

// Inside CliffordAttention::forward, after computing raw scores and before softmax:
mask_scores(&mut scores, None);                      // causal mask only
mask_scores(&mut scores, Some(&padding_mask));       // causal + padding

// Build a padding mask from token ids (pad_id is typically 0)
let pad_mask = padding_mask_from_ids(&token_ids, pad_id);  // Vec<bool>

// Precomputed mask for repeated use without rebuilding
let mask = CausalMask::new(max_seq_len);
assert!(!mask.is_masked(3, 2));  // past position — visible
assert!( mask.is_masked(2, 3));  // future position — blocked
```

---

### `kv_cache` — KV cache for autoregressive inference

```rust
use spacekit_nca::kv_cache::*;

// Allocate once per generation session
let mut cache = KVCache::new(n_layers, max_seq_len);

// After computing k_i and v_i for the new token at each layer:
cache.push_all(layer_kvs);   // Vec<(Vec<Multivector>, Vec<Multivector>)>

// Retrieve all past K/V for attention in layer i
let past_k = cache.layer(i).all_k();  // &[Vec<Multivector>]
let past_v = cache.layer(i).all_v();

// Convenience: compute one cached attention step
let attn_out = cached_attention_step(
    &alg,
    cache.layer_mut(i),
    &q_new,     // query for new token  [d_model]
    k_new,      // key for new token    [d_model]
    v_new,      // value for new token  [d_model]
    scale,      // 1 / sqrt(head_dim × 16)
);

// Reset between independent sequences
cache.clear();
```

The cache enforces a hard `max_seq_len` limit by evicting the oldest tokens
(sliding window) when the sequence grows beyond it.

---

## Running tests

```bash
cargo test
```

Each module has its own unit tests.  The full suite covers: algebra identities
(anti-commutativity, associativity, metric signature), numerical gradient checks
via finite differences, rotor unitarity, optimiser convergence on a trivial
quadratic, mask boundary conditions, and KV cache eviction logic.

---

## Integration with your STA encoder

Your encoder maps a token id to `d_model` multivectors in Cl(1,3) — that is
exactly `Vec<Multivector>`.  Populate the embedding table at construction time:

```rust
let embedding: Vec<Vec<Multivector>> = (0..vocab)
    .map(|id| your_sta_encoder.encode(id))
    .collect();
```

Apply `RotorPositionalEncoding::encode` to the embedded sequence before the
first block (or inside the block after the pre-norm — both conventions work).

---

## License

MIT. See [`LICENSE`](LICENSE).

---

## Training

### 1. Prepare your data

Your JSONL file should sit at `data/train.jsonl`.  The loader expects the exact
schema shown above — `text`, `expected_response`, `split`, and `domain` are the
most important fields.

```
data/
  train.jsonl   ← set "split": "train" on training records
                   set "split": "val"   on validation records
```

### 2. Run training

```bash
cargo run --release -- train data/train.jsonl
```

Logs are written to stderr in this format:

```
[dataset] train=800 val=100 test=100 vocab=4312
[config]  vocab=4312 d_model=8 n_blocks=2 max_seq=128
[train]   epoch=1 step=10  loss=4.2341 lr=3.00e-4
[train]   epoch=1 step=20  loss=3.8812 lr=3.00e-4
[val]     step=100 val_loss=3.6201
```

### 3. Tune the config

Edit `TrainConfig::small` in `train.rs` or construct one directly in `main.rs`:

```rust
let mut cfg = TrainConfig::small(tokenizer.vocab_size());
cfg.d_model   = 16;    // larger model
cfg.n_blocks  = 4;
cfg.d_ff      = 64;
cfg.epochs    = 20;
cfg.lr_max    = 1e-4;
cfg.grad_clip = 0.5;
```

### 4. How the training sequence format works

Each example is encoded as:

```
[BOS] INPUT {text} [SEP] {expected_response} [EOS]
 ↑                  ↑
 no loss here        loss starts here
```

The `response_start` index marks where the loss switches on.
At every position `t >= response_start` the model is trained to predict
`token[t+1]` using cross-entropy.  The input and SEP tokens are teacher-forced
through the model but receive no gradient signal.

### 5. Plugging in your STA encoder

Replace the random embedding initialisation in `main.rs` with your encoder:

```rust
state.model.embedding = (0..tokenizer.vocab_size())
    .map(|id| your_sta_encoder.encode(id))   // → Vec<Multivector>
    .collect();
```

The embedding weights are not currently updated during training (they are
treated as a fixed STA representation).  To make them trainable, add a
`GradLinear`-style embedding gradient accumulator and an `embed_opt:
LayerOptimizer` in `ModelState`.

### 6. Generation after training

```bash
cargo run --release -- generate data/train.jsonl "Today was a rough day."
```

Output:
```
Prompt:   Today was a rough day.
Response: NEGATIVE (mild) — ...
```

---

## Full end-to-end training (v2)

The original `train` module only updated the output head and approximated the
attention gradients.  `train_v2` does the full backward pass: every linear
projection in attention (Q, K, V, O), both layer-norm γ/β pairs, the FFN, the
head, **and** the embedding table all receive proper gradients from a tape-based
backward.

### Architecture

```
data ──▶ model_forward_taped ──▶ Tape
                                  │
                                  ▼
                          cross_entropy at every loss position
                                  │
                                  ▼
                              grad_logits
                                  │
                                  ▼
                       head backward ─▶ grad_x
                                  │
                                  ▼
                  for each block (reverse order):
                      block_backward(tape, grad_x) ──▶ BlockGrads
                          • attention_backward (Q/K/V/O + softmax)
                          • norm1 + norm2 (γ, β)
                          • FFN (fc1 + fc2 + ReLU mask)
                                  │
                                  ▼
                          embedding sparse update
```

### Usage

```rust
use spacekit_nca::*;

let mut tokenizer = data::Tokenizer::new();
let dataset = data::Dataset::load("data/train.jsonl", &mut tokenizer, 256)?;

let mut cfg = TrainConfigV2::small(tokenizer.vocab_size());
cfg.train_embeddings = true;   // set false to freeze your STA encoder output
cfg.epochs = 20;

let mut state = ModelStateV2::new(cfg);
state.model.embedding = your_sta_encoder.embed_vocab(&tokenizer);

train_v2(&dataset, &mut state);
```

### Tape memory cost

For each forward pass `train_v2` retains:

| component        | size (per block)                  | scale with     |
|------------------|-----------------------------------|----------------|
| input + output   | `2 × seq × d_model × 16` floats   | seq            |
| Q, K, V cache    | `3 × seq × d_model × 16` floats   | seq            |
| attention scores | `seq × seq` floats                | seq²           |
| softmax weights  | `seq × seq` floats                | seq²           |
| FFN hidden       | `2 × seq × d_ff × 16` floats      | seq × d_ff     |
| norm stats       | `seq × 16 × d_model` floats       | seq            |

For the demo config (seq=128, d_model=8, d_ff=32, 2 blocks) this is roughly
**100 KB per block** plus the embedding/head — entirely tractable on CPU.

### Gradient verification

The tape and backward modules ship with finite-difference gradient checks:

```bash
cargo test --release tape::tests
cargo test --release attention_backward::tests
cargo test --release block_backward::tests
cargo test --release train_v2::tests::end_to_end_loss_decreases
```

The `end_to_end_loss_decreases` test trains the full model on a single example
for 30 steps and asserts the loss drops to less than 60% of its starting value.

---

## Sampling and streaming inference

```rust
use spacekit_nca::sample::*;

// Configure how the model picks tokens
let cfg = SampleConfig {
    temperature: 0.8,
    top_p: Some(0.9),
    top_k: Some(50),
    repetition_penalty: 1.1,
    max_new_tokens: 256,
    stop_tokens: vec![data::special::EOS],
    seed: Some(42),
};

// One-shot generation
let prompt_ids = tokenizer.encode_str("Today was a rough day.");
let mut rng = SimpleRng::new(42);
let next = sample_next(&logits, &prompt_ids, &cfg, &mut rng);

// Streaming with a callback
let generated = generate_stream(
    &prompt_ids,
    &cfg,
    &tokenizer,
    |ids| state.model.forward(ids),       // forward function
    |tok, piece| {                         // per-token callback
        print!("{piece} ");
        std::io::Write::flush(&mut std::io::stdout()).unwrap();
        true                                // return false to stop early
    },
);
```

Presets: `SampleConfig::greedy()`, `SampleConfig::focused()` (T=0.7, top-p=0.9,
rep=1.15), `SampleConfig::creative()` (T=1.0, top-p=0.95, rep=1.1).

---

## TinyStories experiment

TinyStories (Eldan & Li, 2023) is a corpus of ~2M short children's stories
with a constrained vocabulary, designed exactly for the "can a tiny model
learn coherent generation" question.  It's the right baseline for confirming
the full backward pass works end-to-end before scaling further.

### Files involved

| file                                | purpose                                        |
|-------------------------------------|------------------------------------------------|
| `src/bpe.rs`                        | byte-pair encoding tokenizer (~256–8k vocab)   |
| `src/tinystories.rs`                | corpus loader + packed binary format           |
| `src/bin/tinystories.rs`            | end-to-end CLI: tokenize / encode / train / generate |
| `scripts/get_tinystories.sh`        | download the dataset from HuggingFace          |

### Step-by-step pipeline

```bash
# 1. Download corpus (start with validation file — ~20 MB)
bash scripts/get_tinystories.sh

# 2. Train BPE tokenizer (one-time, ~5 min on full corpus)
cargo run --release --bin tinystories -- tokenize \
    data/TinyStories-valid.txt 2048 data/tinystories.tok

# 3. Encode corpus to packed binary (one-time, ~2 min on full corpus)
cargo run --release --bin tinystories -- encode \
    data/TinyStories-valid.txt data/tinystories.tok data/val.bin

# 4. Train (for the quick test, reuse val.bin for both train and val)
cargo run --release --bin tinystories -- train \
    data/tinystories.tok data/val.bin data/val.bin

# 5. Generate after training
cargo run --release --bin tinystories -- generate \
    data/tinystories.tok "Once upon a time"
```

For the full training corpus add `FULL=1`:

```bash
FULL=1 bash scripts/get_tinystories.sh
```

### Sizing the model honestly

The Cl(1,3) Clifford transformer uses 16 floats per "scalar" in a standard
transformer.  A linear layer with `d_model=16` has `16 × 16 × 16 = 4 096`
floats per (out, in) cell.  This is the parameter-count distinction to watch:

| comparison        | standard d_model | clifford d_model | equal params |
|-------------------|------------------|------------------|--------------|
| this run          | -                | 16               | -            |
| param-equal baseline | 64            | 16               | yes          |
| dim-equal baseline   | 16            | 16               | no (16× fewer in standard) |

For a fair "Clifford works as well as a transformer" comparison, train a
standard transformer at `d_model=64, n_blocks=4` against your Clifford
`d_model=16, n_blocks=4` and measure validation loss after the same wall-clock
training time.

### What good looks like

After ~2k steps on the validation file (no full training data), perplexity
should drop from ~vocab_size (random ≈ 2048) to ~100–200, and samples should
start to look like simple sentences with grammatical structure even if the
content is incoherent.  A sample like:

```
[sample] Once upon a time → there was a little girl named lily . she
         loved to play with her friends . one day , she went to the park
```

means the LM head is doing real work.  A sample like:

```
[sample] Once upon a time → the the the the the the the the
```

means you have a repetition collapse — increase `repetition_penalty` in
`SampleConfig` or use `SampleConfig::focused()`.

### Hardware expectations

CPU training is slow.  Expect ~50–500 ms per step at `d_model=16, seq=128,
n_blocks=4` on a modern laptop.  10k steps → 10–80 minutes.  This is enough
to demonstrate the pipeline works; scaling to interesting behaviour requires
either patience or a GPU port.


---

## Neural Cellular Automaton scratchpad

`CliffordGridNCA` is a 2D grid of multivector cells the agent can read from
and write to as persistent spatial memory.  Each cell holds `d_state`
multivectors; one CA step applies a local 3×3 perception (identity + Sobel-x
+ Sobel-y filters) followed by a tiny update network.

The grid persists across turns — the agent calls `step(n)` between thoughts
and the state evolves under its learned rule.  Damage to the grid (zeroing
a region) is repaired automatically over subsequent steps, exactly the
self-healing property that makes NCAs interesting as memory.

### Quick start

```rust
use spacekit_nca::{CliffordGridNCA, NcaCommand, NcaResponse};

// 32×32 grid, 4 multivectors per cell (64 floats per cell)
let mut nca = CliffordGridNCA::new(32, 32, 4, /*seed*/ 42);

// Seed an initial signal
nca.seed(16, 16, 1.0);

// Let it evolve for a while
nca.steps(20);

// Inspect what it grew
println!("alive cells: {}",   nca.alive_count());
println!("mean activity: {:.3}", nca.mean_activity());

// Read a patch
let patch = nca.read_region(14, 14, 5, 5);
```

### Agent tool-call interface

```rust
use spacekit_nca::{NcaCommand, NcaResponse};

// From inside a tool-call dispatcher:
let response = nca.execute(NcaCommand::Seed { x: 10, y: 10, magnitude: 1.0 });
let response = nca.execute(NcaCommand::Step { n: 5 });
let response = nca.execute(NcaCommand::ReadRegion { x: 8, y: 8, w: 5, h: 5 });

if let NcaResponse::Status { alive, activity, step } = nca.execute(NcaCommand::Status) {
    println!("step={step} alive={alive} activity={activity:.3}");
}
```

The eight commands cover everything an agent needs to use the grid as memory:

| command         | purpose                                            |
|-----------------|----------------------------------------------------|
| `Read`          | inspect a single cell                              |
| `Write`         | overwrite a cell                                   |
| `Seed`          | mark a cell alive with magnitude (convenience for write) |
| `ReadRegion`    | bulk read a rectangular patch                      |
| `Step`          | advance the CA by n steps                          |
| `Clear`         | reset the grid                                     |
| `Status`        | alive count + mean activity + step counter         |
| `Damage`        | zero out a circular patch (for repair experiments) |

### Knobs

| field             | default | what it does                                  |
|-------------------|---------|-----------------------------------------------|
| `fire_rate`       | 0.5     | per-cell update probability — stochastic mask |
| `alive_threshold` | 0.1     | cells with `state[0].c[0]` below this die     |
| `max_delta`       | 1.0     | clip on per-step component change             |

### What's not in this module (and what to add next)

The grid update uses the same `CliffordLinear` and `CliffordFFN` types as
everything else, but **the weights are random and the CA does not yet learn**.
For a useful agent scratchpad you have two paths:

1. **Use it as random-recurrent memory** — treat the CA as a fixed
   high-dimensional dynamical system the agent learns to read and write
   coherently.  No NCA training needed, just train the agent's tool-call policy
   to use it.  Reservoir-computing flavor.

2. **Train the CA to grow target patterns** — give it a pool of target images
   (or target multivector fields), apply the standard NCA training loop
   (seed → run N steps → MSE against target → backprop through the unrolled
   updates).  The tape/backward machinery from `train_v2` works for this with
   a few adapters — happy to add that as `nca_train.rs` if you want the
   trainable version.

Option 1 is what most "agent with scratchpad" papers actually do.  Option 2 is
the "research-direction" version where the CA dynamics themselves are learned.

