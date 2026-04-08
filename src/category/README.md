# Category Theory for Neural Networks

This is a deep question — it touches on **compositional generalization**, which is exactly where categorical structure earns its keep. Let's break it down honestly.

**Code in this folder** is behind the Cargo feature **`categorical`** (`#[cfg(feature = "categorical")] pub mod category` in `src/lib.rs`). Enable with `cargo build --features categorical` (not in default features, so lean `cargo build` omits this tree). Use `growformer::category::{…}` from other modules; files *inside* `src/category/` refer to siblings via `crate::category::training`, etc.

| File | Role |
|------|------|
| [`category.rs`](./category.rs) | `CategoricalDAG`, `MorphismKind`, `Layer`, `NaturalTransform`, `Network` |
| [`pythagoras.rs`](./pythagoras.rs) | `PythagorasNode`, Pythagorean dimensional splits |
| [`node.rs`](./node.rs) | `CategoricalNode` — DAG node + internal tree; `Layer` uses [`forward.rs`](./forward.rs) |
| [`forward.rs`](./forward.rs) | `record_embedding`, `bifunctor_branch_vectors`, `compose_aligned` (Hadamard + shape align) |
| [`embedding.rs`](./embedding.rs) | `SentenceEmbedder`, `CharHashEmbedder`, `TokenHashEmbedder` |
| [`linear_head.rs`](./linear_head.rs) | `LinearHead` — softmax CE + SGD for sentiment / aux logits |
| [`training.rs`](./training.rs) | `TrainingRecord`, JSONL I/O, `SentimentLabel`, optional `AuxCategory` |
| [`disentanglement.rs`](./disentanglement.rs) | Three-term loss, `cross_branch_dropout`, `SimpleRng` |
| [`curriculum.rs`](./curriculum.rs) | `CurriculumScheduler`, stage configs; uses `GrowthPolicy` from `growformer.rs` |
| [`growformer.rs`](./growformer.rs) | `GrowformerTrainer`, `TrainerConfig`, `GrowthPolicy`; real CE + bifunctor forward |
| [`inference.rs`](./inference.rs) | `InferenceResult`, `InferenceDetail`, `infer_from_embedding` |
| [`sentiment.rs`](./sentiment.rs) | `SentimentFunctor`, `ParsedInput`, optional toy baseline for `infer()` |

Re-exports for ergonomic imports: see [`mod.rs`](./mod.rs).

[`sentiment_generalization.rs`](./sentiment_generalization.rs) is **not** included in [`mod.rs`](./mod.rs); it reads like a draft `examples/` binary and its `use growformer::{ training, … }` paths predate the `category` namespace. Move or rewrite it under `examples/` with `growformer::category::{…}` if you want a runnable demo.

---

## Can Your Architecture Do This?

The short answer: **yes, but only if the categorical structure is actually enforced during training** — not just at the storage level. Here's why.

---

### What You're Actually Asking The Model To Learn

```
"I hate mondays"  →  negative/mild sentiment
"I hate tuesdays" →  ??? (must generalize)
```

This requires the model to learn **two separable things**:

1. The **sentiment morphism** — `"I hate X"` → `negative/mild`
2. The **object substitution** — `mondays`, `tuesdays`, ... are interchangeable objects in a category of *days*

A standard neural net learns these **entangled** — it memorizes `"I hate mondays"` as a whole token pattern. Your categorical approach can learn them **disentangled** — which is the key.

---

### How The Categorical DAG + Pythagoras Tree Enables This

The sentence decomposes into a **product** in your DAG:

```
"I hate mondays"
      │
   [Parse Node]
   /           \
[Sentiment]   [Entity]
morphism       morphism
"I hate X"    "mondays"
   │               │
negative/mild   Day-of-week
                category
```

The internal Pythagoras tree of the Parse Node stores this **bifunctor split**:

```
Parse Node (dim: c)
    /          \
Sentiment    Entity
 (dim: a)    (dim: b)
              /      \
           Day      Plurality
          (dim: b1)  (dim: b2)
```

Because `mondays` and `tuesdays` are **objects in the same category** (days of week), a **natural transformation** connects them — meaning the sentiment morphism applies identically to both, by categorical law.

---

### JSONL Training Structure

For this to work, your JSONL needs to encode the categorical structure explicitly:

```jsonl
{"input": "I hate mondays", "sentiment": "negative/mild", "entity_category": "day_of_week", "entity": "monday", "plural": true}
{"input": "I love fridays", "sentiment": "positive/mild", "entity_category": "day_of_week", "entity": "friday", "plural": true}
{"input": "I hate rain", "sentiment": "negative/mild", "entity_category": "weather", "entity": "rain", "plural": false}
{"input": "I hate meetings", "sentiment": "negative/mild", "entity_category": "event", "entity": "meeting", "plural": true}
```

The key insight is training **across categories** — `rain`, `meetings`, `mondays` — so the model learns that `"I hate X"` is a **functor** that maps any object `X` to `negative/mild`, independent of what `X` is.

---

### The Generalization Path in Rust

```rust
// Days of week as objects in a category
enum DayOfWeek { Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday }

// Natural transformation: any day maps to the same sentiment target
struct DaySubstitution;

impl NaturalTransform<DayOfWeek, DayOfWeek, SentimentLabel> for DaySubstitution {
    // Commutativity: swapping the day before or after sentiment
    // extraction gives the same result — this is what you're training
    fn transform(day: DayOfWeek) -> DayOfWeek {
        day // identity on the object, composition handles the rest
    }
}

// The sentiment functor — learned during training
struct SentimentFunctor;

impl Layer<ParsedInput, SentimentLabel> for SentimentFunctor {
    fn forward(&self, input: ParsedInput) -> SentimentLabel {
        // Internally: Pythagoras tree splits into
        // sentiment morphism + entity morphism
        // entity morphism is day-agnostic by training
        todo!()
    }
}
```

---

### What Can Go Wrong

| Risk | Cause | Mitigation |
|---|---|---|
| Model still memorizes `mondays` | JSONL lacks cross-day examples | Train on all 7 days + other entity categories |
| Entity/sentiment not disentangled | Pythagoras split not enforced during backprop | Add a **disentanglement loss** between the two child nodes |
| Plurality not generalized | `monday` vs `mondays` treated as different tokens | Normalize plurality in preprocessing or add a dedicated plurality node |
| Weak generalization to unseen days | Too few entity categories in training | Include non-day entities so the functor truly generalizes |

---

### The Disentanglement Loss

This is the critical piece — you need to explicitly encourage the Pythagoras children to learn **independent** representations:

```rust
// During training: penalize mutual information between
// the left (sentiment) and right (entity) child outputs
fn disentanglement_loss(
    sentiment_embedding: &[f32],
    entity_embedding: &[f32],
) -> f32 {
    // Cosine similarity penalty — push the two apart
    let dot: f32 = sentiment_embedding.iter()
        .zip(entity_embedding.iter())
        .map(|(a, b)| a * b)
        .sum();

    let norm_s: f32 = sentiment_embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_e: f32 = entity_embedding.iter().map(|x| x * x).sum::<f32>().sqrt();

    // Minimize this alongside your main loss
    (dot / (norm_s * norm_e)).abs()
}
```

---

### The Full Generalization Chain

```
Training:        "I hate mondays"  →  negative/mild
                 "I hate tuesdays" →  negative/mild  (if in JSONL)
                 "I hate rain"     →  negative/mild

Learned:         F("I hate X") = negative/mild  ∀ X  ← functor law

Inference:       "I hate wednesdays" → 
                 1. Parse node splits: ["I hate X"] + ["wednesdays"]
                 2. Entity node: wednesday ∈ Day_of_week (known category)
                 3. Natural transform: day substitution is identity on sentiment
                 4. Output: negative/mild  ✅
```

---

### Verdict

Growformer architecture **does give rise to this capability**, but it requires three things working together:

1. **JSONL structured** to separate entity categories from sentiment patterns
2. **Disentanglement loss** to ensure the Pythagoras split actually separates the two morphisms during backprop
3. **Cross-category training** so the sentiment functor generalizes beyond just days

The categorical structure isn't magic on its own — it's a **scaffold that makes the right generalization learnable**. The Pythagoras tree per node is what makes the bifunctor split *storable and inspectable*, which gives you far more control over this than a standard transformer would.

---

This is a sharp critique and most of it is right. Let me engage with it honestly, point by point, and tell you where I agree, where I'd push back, and what it means concretely for Growformer.

---

## Where the Critique Lands Clean ✅

**Point A — over-specified JSONL** is the strongest criticism. The current schema encodes `entity_category` as a hard label, which is exactly backwards for your architecture. You're right that Growformer's region formation should *discover* that mondays and tuesdays are the same kind of thing — not be told. The label becomes a crutch that short-circuits the emergence you're actually building toward.

**Point B — disentanglement loss** being necessary but not sufficient is also correct. Cosine separation alone is weak. The three additions proposed (orthogonality in expectation, cross-branch dropout, contrastive alignment) are all real improvements. Cross-branch dropout in particular is elegant — it forces each Pythagoras child to be *independently meaningful*, which is exactly the bifunctor law you want to enforce behaviourally.

**Point C — region specialization** doing the categorical work automatically is the key insight the critique is building toward, and it's correct in principle. If your region formation is working, days of week should cluster without you labeling them.

---

## Where I'd Push Back ⚠️

**The critique undersells the role of weak supervision.** Saying "let the model discover categories" is correct in theory, but practically you need enough signal to avoid degenerate solutions. Without *any* structural hint, the sentiment morphism can stabilize in a way that partially entangles the entity — and you won't know until inference breaks on an unseen day. The labels aren't wrong to include; they're wrong to treat as ground truth. The fix is making them soft, not removing them.

**"Developmental substrate" vs "classifier with metadata"** is a real distinction but it's overstated here. Even developmental systems benefit from scaffolding during early training. The question is whether you *remove the scaffolding* as regions stabilize — and that's a training curriculum question, not an architecture question.

---

## What This Means Concretely for Growformer

### 1. Revised minimal JSONL schema

Drop the explicit `entity_category`. Keep only what the model can't infer:

```jsonl
{"input": "I hate mondays", "sentiment": "negative/mild", "plural": true}
{"input": "I hate rain",    "sentiment": "negative/mild", "plural": false}
{"input": "I love fridays", "sentiment": "positive/mild", "plural": true}
```

The model discovers that `mondays`, `rain`, and `meetings` play structurally equivalent roles — because the sentiment morphism stabilizes the same way across all of them.

### 2. Three-stage training curriculum

```
Stage 1 — Scaffold (steps 0–N):
  Include soft entity_category labels as auxiliary loss (λ=0.3)
  Goal: seed initial region formation

Stage 2 — Loosen (steps N–2N):
  Drop auxiliary label loss entirely
  Enable cross-branch dropout (p=0.2)
  Add contrastive alignment loss on sentiment branch
  Goal: force the sentiment morphism to become entity-agnostic

Stage 3 — Harden (steps 2N+):
  Grow Pythagoras nodes only when disentanglement loss < threshold
  Prune nodes where branches have collapsed to similar representations
  Goal: stabilize compositionality, not memorization
```

### 3. Proper disentanglement loss stack

```rust
pub struct DisentanglementLoss {
    pub cosine_weight: f32,       // existing: push branches apart
    pub ortho_weight: f32,        // new: E[s·e] → 0 across batch
    pub contrastive_weight: f32,  // new: same-sentiment embeddings cluster
}

impl DisentanglementLoss {
    pub fn compute(
        &self,
        sentiment_batch: &[Vec<f32>],  // one per sample
        entity_batch:    &[Vec<f32>],  // one per sample
        sentiment_labels: &[SentimentLabel],
    ) -> f32 {
        let cosine   = self.cosine_term(sentiment_batch, entity_batch);
        let ortho    = self.orthogonality_term(sentiment_batch, entity_batch);
        let contrast = self.contrastive_term(sentiment_batch, sentiment_labels);

        self.cosine_weight    * cosine
        + self.ortho_weight   * ortho
        + self.contrastive_weight * contrast
    }

    // Orthogonality in expectation: mean(s·e) across batch → 0
    fn orthogonality_term(&self, s: &[Vec<f32>], e: &[Vec<f32>]) -> f32 {
        let n = s.len() as f32;
        s.iter().zip(e.iter())
            .map(|(si, ei)| dot(si, ei))
            .sum::<f32>() / n
    }

    // Contrastive: same-sentiment embeddings should be close,
    // different-sentiment embeddings should be far
    fn contrastive_term(
        &self,
        s: &[Vec<f32>],
        labels: &[SentimentLabel],
    ) -> f32 {
        let mut loss = 0.0f32;
        let mut count = 0;
        for i in 0..s.len() {
            for j in (i+1)..s.len() {
                let sim = cosine_sim(&s[i], &s[j]);
                let same = labels[i] == labels[j];
                // Pull same-class together, push different-class apart
                loss += if same { 1.0 - sim } else { sim.max(0.0) };
                count += 1;
            }
        }
        if count == 0 { 0.0 } else { loss / count as f32 }
    }

    fn cosine_term(&self, s: &[Vec<f32>], e: &[Vec<f32>]) -> f32 {
        s.iter().zip(e.iter())
            .map(|(si, ei)| cosine_sim(si, ei).abs())
            .sum::<f32>() / s.len() as f32
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let d = dot(a, b);
    let na = dot(a, a).sqrt();
    let nb = dot(b, b).sqrt();
    if na == 0.0 || nb == 0.0 { 0.0 } else { d / (na * nb) }
}
```

### 4. Cross-branch dropout — wired into the node

```rust
pub fn forward_with_dropout<F>(
    &self,
    input: A,
    kernel: &F,
    dropout_p: f32,
    rng: &mut impl rand::Rng,
) -> (Option<A>, Option<A>)
where
    F: Fn(&W, A) -> A,
    A: Clone,
{
    let (left, right) = self.composition.compose_bifunctor(input, kernel);
    let drop_left  = rng.gen::<f32>() < dropout_p;
    let drop_right = rng.gen::<f32>() < dropout_p;
    // Never drop both simultaneously
    match (drop_left, drop_right) {
        (true, false) => (None, Some(right)),
        (false, true) => (Some(left), None),
        _             => (Some(left), Some(right)),
    }
}
```

---

## The Distilled Position

The critique is right that the architecture will generalize via region-based compositionality, not symbolic substitution — and that's a genuine strength of Growformer over transformers. The adjustment is: treat the JSONL labels as **scaffolding to remove**, not structural truth to encode, and replace the single disentanglement term with the three-part loss stack (implemented in [`disentanglement.rs`](./disentanglement.rs)). The Pythagoras tree is still the right internal structure — it just needs the training signal to force each child to *earn* its independence.

---

## Integration with the rest of Growformer

### Feature flag

- **`categorical`** in `Cargo.toml`: `cargo build --features categorical`, `cargo test -p growformer --features categorical`.
- Default `cargo build` **omits** `category` for smaller dependency surface.

### Status

- **`GrowformerTrainer::step`** runs a **real** forward: `record_embedding` → `bifunctor_branch_vectors` on the parse node’s `PythagorasNode` → cross-branch dropout → **`LinearHead`** sentiment CE (SGD) + aux CE in scaffold → `combined_loss_full` for disentanglement. Parse-tree **Hadamard** weights are not yet updated from the task loss (heads learn first; backprop through the tree is the next step).
- **`TrainerConfig`** (`embed_dim`, `branch_dim`, `parse_node_id`, `lr`, `head_seed`) must match the parse `GrowformerNode`: `weights.len() == dim == embed_dim`. `branch_dim` is the width after aligning left/right branch vectors for heads and disentanglement.
- **Inference (heads + tree)**  
  - [`infer_head`](./growformer.rs) / [`infer_head_with_embedding`](./growformer.rs): compact [`InferenceResult`](./inference.rs) — sentiment from sentiment head, `inferred_category` from **aux head** (aligned with training).  
  - [`infer_head_detail`](./growformer.rs) / `_with_embedding`: full logits, softmax, confidences, plus `aux_heuristic` vs `aux_predicted` for calibration.  
  - [`infer_head_batch`](./growformer.rs) / [`infer_head_detail_batch`](./growformer.rs): same over many strings (hash embed per row).  
  - [`infer_from_embedding`](./inference.rs): free function if you already have parse tree + heads outside `GrowformerTrainer`.  
- **`infer` + `SentimentFunctor`**: separate toy classifier; `inferred_category` stays the **entity-string heuristic**, not the aux head.
- **`mock_embed`** remains a backward-compat alias for deterministic `char_hash_embed` when no `TrainingRecord.embedding` is set.
- Name overlap: the trainer is `growformer::category::GrowformerTrainer` (file `category/growformer.rs`), not `growformer_lang`.

### Extending with a real tokenizer / encoder

| Hook | Where to plug in |
|------|-------------------|
| Sentence vectors | Implement or call your encoder, then set `TrainingRecord.embedding` or replace [`record_embedding`](./forward.rs). |
| Stronger parse morphism | Replace or augment `hadamard` + `compose_aligned` with conv / attention kernels while keeping `node.dimension` and weight lengths consistent. |
| Tree weight training | Use `step()`’s branch gradients (from heads) to propagate into parse leaves, or add a second optimizer over `PythagorasNode` weights. |

`cross_branch_dropout` is applied per sample in `step()` via `SimpleRng`.

### JSONL and weak labels

**Native trainer schema** (aliases: `text` → `input`, `semantic_intent` → `sentiment` when the string is a valid label):

- `input` (or `text`), `sentiment` / `semantic_intent` (seven classes; see [`SentimentLabel`](./training.rs)), optional `plural`, optional `aux_category`, optional `embedding` (`Vec<f32>` in JSON).

**`growformer/data/sentiment/*.jsonl`**: `text` + `semantic_intent` plus ignored metadata. Optional `plural` and `embedding` are read if present. Load with [`TrainingBatch::from_sentiment_jsonl_dir`](./training.rs) + [`SentimentJsonlSelection::TrainFilesOnly`](./training.rs). [`semantic_intent_to_label`](./training.rs) maps `sarcastic` / `mixed` to distinct [`SentimentLabel::Sarcastic`](./training.rs) / [`Mixed`](./training.rs) (7-way linear head).

**Embeddings**: [`TrainingBatch::fill_missing_embeddings`](./training.rs) with [`TokenHashEmbedder`](./embedding.rs) or [`CharHashEmbedder`](./embedding.rs) (`SentenceEmbedder` trait). At train time, [`record_embedding`](./forward.rs) still aligns any stored vector to `TrainerConfig.embed_dim`.

**Plural**: [`TrainingBatch::reinforce_plural_with_heuristic`](./training.rs) sets `plural |= infer_plural_from_text` (tail-token `s` heuristic; see [`infer_plural_from_text`](./training.rs)).

**One-command training** (from the `growformer` crate root):

`cargo run --example categorical_sentiment_train --features categorical -- data/sentiment 400 128`

See `TrainingRecord::resolved_aux_category()` for Stage-1 aux when `aux_category` is omitted.

### Tests

- Lib: `cargo test -p growformer --features categorical --lib category::`
- Integration: `cargo test -p growformer --features categorical --test categorical_trainer`

### Curriculum behaviour

Stage transitions: **SCAFFOLD** (aux + no dropout) → **LOOSEN** (full disentanglement + dropout) → **HARDEN** (growth/pruning gated by last step’s disentanglement total). As heads fit the sentiment objective, branch vectors change and the ortho/cosine terms become more meaningful than with a fixed mock signal.

Canonical loss APIs: `DisentanglementLoss::compute`, `combined_loss_full`; cross-branch dropout: `disentanglement::cross_branch_dropout` (single RNG draw per sample; never drops both branches).