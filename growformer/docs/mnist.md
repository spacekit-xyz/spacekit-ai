# Split MNIST

**What Split MNIST is:**

Five sequential binary classification tasks, each presenting two digit classes:
- Task 1: 0 vs 1
- Task 2: 2 vs 3
- Task 3: 4 vs 5
- Task 4: 6 vs 7
- Task 5: 8 vs 9

Each task trains to completion, then the next task begins. The system must retain all prior tasks. Final evaluation measures accuracy on all five tasks simultaneously. The standard reported metric is average accuracy across all five after sequential training.

**The benchmark numbers to beat:**

EWC (Kirkpatrick et al. 2017) reports roughly 97% average accuracy on Split MNIST. Progressive Neural Networks report near 99% but with linear parameter growth. PackNet reports ~99% with fixed capacity. These are the numbers your results will be compared against directly.

**The input pipeline gap:**

MNIST images are 28×28 pixels — 784 floats. The Growformer currently expects low-dimensional continuous input. You need a preprocessing step that converts each image to a compact float vector before the Growformer sees it. Three options in increasing order of complexity.

Flatten and PCA down to 32-64 dimensions — simplest, loses spatial structure but preserves enough variance for digit classification. Implementable immediately, no new architecture.

Fixed random projection — project 784 floats down to 64 dimensions via a frozen random matrix. Theoretically justified by Johnson-Lindenstrauss. Same simplicity as PCA, no training required for the projection.

Small convolutional front-end — two conv layers producing a 64-128 dimensional feature vector, trained jointly with Task 1 and then frozen. Preserves spatial structure. More work but produces better features for the Growformer to operate on.

Start with PCA or random projection. If results are competitive that's the clean story — no auxiliary networks, pure Growformer. If accuracy is limited by the projection quality, add the conv front-end as a separate frozen embedding stage.

**Five Mirror promotions in sequence:**

Each task spawns a Mirror, trains to criterion, and promotes. After five promotions the Main Dimension holds five frozen groups. The routing test becomes meaningful at scale — five groups, cosine similarity routing, no task label. That's where the architecture either holds or reveals weaknesses.

Watch the routing margins as groups accumulate. Two groups showed clear separation for spiral vs circles because those tasks are geometrically very different. Digit pairs are all handwritten strokes — the embedding space may be much more crowded. Routing margin across five digit-pair groups is the critical diagnostic.

**The forgetting metric:**

After Task 5 trains, evaluate all five tasks. Report per-task accuracy and average. The Growformer's architectural prediction is near-zero forgetting on all five because Main Dimension is truly inert — Task 5 training cannot touch Tasks 1-4 at all. If the architecture is sound the forgetting number should be essentially 0% across all tasks, not just the most recent pair.

That would be a stronger result than EWC which achieves 97% through regularization — meaning 3% is still lost. The Growformer loses nothing by construction.

**The composition opportunity:**

After five groups are trained, test whether a VirtualGroup can solve a held-out composition task — distinguish even digits from odd digits (0/2/4/6/8 vs 1/3/5/7/9) without training a new group. This task cuts across all five trained groups. If the VirtualGroup finds blend weights that solve it from 30-50 examples, that is a result with no precedent in the continual learning literature.

**What success looks like for the paper:**

Average accuracy ≥97% across five tasks after sequential training, matching EWC. Zero forgetting on all prior tasks, beating EWC's 3% residual forgetting. Routing without task label working across five groups. Optional composition result on even/odd. Those four results together constitute a complete, publishable, directly comparable contribution to the continual learning literature.

The embedding choice affects the absolute accuracy numbers but not the forgetting numbers. The forgetting result is architecture-dependent and will be clean regardless of how good the embedding is. Lead with forgetting, support with accuracy.

Before running, three things to confirm are in place.

**The input pipeline decision:**

MNIST is 28×28 = 784 floats. The Growformer expects a compact float vector. You need a preprocessing decision before the first run. The fastest path is random projection — multiply each image by a frozen 784×64 random matrix, normalize, feed 64 floats to the Growformer. No training required for the projection, implementable in an hour, gives the Growformer a reasonable input space to work with.

PCA is slightly better quality but requires fitting on the training set first. Either works. Pick one and fix it before any training starts — the projection must be identical across all five tasks or the embedding space shifts between tasks and the routing breaks.

**The task sequence:**

```
Task 1: 0 vs 1
Task 2: 2 vs 3
Task 3: 4 vs 5
Task 4: 6 vs 7
Task 5: 8 vs 9
```

Each is a binary classifier. Each spawns a Mirror, trains to criterion, promotes. After all five, evaluate all tasks simultaneously. The forgetting number is the headline.

**The three numbers to report:**

Average accuracy across all five tasks after sequential training — this is what EWC reports and what you compare against. Per-task accuracy breakdown — which digit pairs are harder, does accuracy degrade on early tasks. Routing accuracy without task label — after five groups, does the cosine router correctly identify which digit pair an input belongs to without being told.

What preprocessing path are you going with? PCA or random projection?

---

## Advice and path to implementation

**Recommendations**

- **Preprocessing:** Start with **fixed random projection** (784→64). No training, no fit step, same projection for all tasks so the embedding space is stable. Use a single frozen matrix, normalize projected vectors (e.g. L2 or scale to [0,1] per dimension). If accuracy plateaus well below 97%, consider PCA or a small conv front-end later; lead with the simplest story.
- **Input dimension in the codebase:** The stack is currently hardcoded to 2D input (`[f32; 2]`, `mirror_layer_sizes: [2, 16, 16, 1]`, router `input_dim = 2`). For MNIST you need a single **variable-width path**: either (a) generalize the training/inference API to `&[f32]` (slice) and set `mirror_layer_sizes[0]` and router `input_dim` from a constant (e.g. 64), or (b) add an MNIST-only path with a fixed type alias (e.g. `MnistInput = [f32; 64]`) and dedicated entry points so existing 2D demos stay unchanged. Option (a) is cleaner long-term; option (b) is less invasive for a first benchmark.
- **Routing with five groups:** Digit tasks are more similar than spiral vs circles. Expect tighter routing margins; watch for confusion between adjacent digit pairs (e.g. 2/3 vs 4/5). If no-context routing degrades, report it and lean on the zero-forgetting result; you can still compare to EWC on accuracy and forgetting.
- **Order of work:** (1) Data + projection, (2) wire 64-dim input through one Mirror and verify training/eval, (3) five-task sequence + promotion, (4) final five-task evaluation and metrics, (5) optional even/odd composition.

**Implementation path (minimal)**

1. **MNIST data**
   - Add a dependency for MNIST (e.g. `mnist` crate or load from CSV/bytes). Alternatively use a small script to download and flatten 28×28 images into 784-dim vectors with labels 0–9.
   - Split into train/test (or use standard 60k/10k). For each task, filter to two classes and binary labels (e.g. 0/1 for task 1). Store as `Vec<(Vec<f32>, [f32; 1])>` (784 floats per image before projection).

2. **Projection module**
   - Implement a **random projection**: 784×64 matrix (e.g. Gaussian entries, fixed seed), computed once. For each image, `y = normalize(projection * x)` (or `x * projection.T` depending on layout). Output 64-dim `Vec<f32>` (or `[f32; 64]` if you keep fixed size).
   - Use the **same** projection for all five tasks (no refit). Normalize so scale is consistent across tasks.

3. **Variable input dimension in the pipeline**
   - Choose strategy (a) or (b) above.
   - **If (a):** Change training data type to `(Vec<f32>, [f32; 1])` (or `(&[f32], [f32; 1])` where the slice length equals a configurable input_dim). Update `train_epoch`, `train_mirror_epoch`, `evaluate_main_group`, `force_promote`, `train_and_set_router`, and any calibration/embedding calls to accept this type. Set `mirror_layer_sizes` to `[64, 32, 32, 1]` (or similar) and router `input_dim = 64`.
   - **If (b):** Add `mnist.rs` (or a `mnist` module) with `type MnistInput = [f32; 64]`, MNIST-specific `train_epoch_mnist`, `evaluate_mnist`, and a single entry point (e.g. `demo_split_mnist`) that builds a DimensionManager with `mirror_layer_sizes: vec![64, 32, 32, 1]`, loads and projects data, and runs the five-task loop. Router is trained with `input_dim = 64`. Leave existing 2D APIs as-is.

4. **Five-task sequence**
   - For task index t = 0..5: digit pair (0,1), (2,3), (4,5), (6,7), (8,9). Filter dataset to those labels, map labels to 0/1, project images to 64-dim.
   - Spawn mirror (e.g. `"task_0"` … `"task_4"`), train to criterion (e.g. accuracy ≥ 0.98 or max epochs), force-promote (or use existing promotion gate). After each promotion, Main has one more frozen group.
   - After all five: Main has five groups; router (if used) has `num_groups = 5`, `input_dim = 64`.

5. **Evaluation and reporting**
   - **Per-task accuracy:** For each task t, run the corresponding Main group on that task’s test set (projected); report accuracy. **Average accuracy** = mean of the five per-task accuracies (this is the number to compare to EWC’s ~97%).
   - **Forgetting:** Compare each task’s accuracy right after its promotion vs at the end of all five tasks. Report drop (should be ~0); “zero forgetting” is the headline.
   - **Routing without task label:** For each test sample, run the router (or cosine heuristic) to choose a group, then run that group’s predictor. Report routing accuracy (fraction where the chosen group is the correct task). Optionally report per-pair confusion.

6. **Optional: even/odd composition**
   - After five groups are trained, form a composition task: even (0,2,4,6,8) vs odd (1,3,5,7,9). Sample 30–50 projected examples with binary labels. Train a VirtualGroup over the five groups; report accuracy. No new Mirror; this tests whether blend weights can solve a cross-task concept from few examples.

**Checklist before first full run**

- [x] MNIST loaded and split by digit pair; labels 0/1 per task (`growformer::mnist::load_mnist_normalized`, `filter_digit_pair`).
- [x] One fixed random projection 784→64; same for all tasks; normalization applied (`RandomProjection`, `project_dataset`).
- [x] Pipeline accepts variable-length input (`Sample = (Vec<f32>, [f32; 1])` throughout; `mirror_layer_sizes[0]` from config).
- [x] `mirror_layer_sizes` and router `input_dim` set to 64 for MNIST (`vec![64, 32, 32, 1]` in `demo_split_mnist`).
- [x] Five mirrors promoted in order; final evaluation runs all five tasks and computes average accuracy and per-task accuracy (`cargo run -- --mnist`).

**Run:** `cargo run -- --mnist`. First time: from the repo root run `bash scripts/download_mnist.sh` to fetch and decompress MNIST into `./data`. Or set `MNIST_ROOT` to a directory containing the four decompressed `.ubyte` files. If the files are missing, the demo prints these instructions and exits without panicking.