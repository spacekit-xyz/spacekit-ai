# **When Composite Accuracy Lies: Few-Shot Routing Over Frozen Experts**

## Frozen Specialists, Compositional Reuse, and Dynamically Structured Physical Neural Systems

**Author:** Astor Rivera-Carcamo  
**Affiliation:** CTO, Founder, SWTCH Labs, SWTCH.AI (swtch.ai), SpaceKit.xyz (spacekit.xyz)
**Status:** Preprint — Not Peer Reviewed  
**Scope:** Parameter-isolated continual learning with compositional reuse of frozen specialists. Shared-substrate CL is explicitly out of scope. Deployment-stack and speculative algebraic material is in Appendices A–B.

---

## Abstract

Production systems that accumulate domain knowledge, clinical modules, robot skills, per-customer brains, often require **certifiable retention** of consolidated specialists while adapting to novel combinations of existing knowledge. Shared-substrate continual learning treats forgetting as a training dynamic to be constrained; **parameter isolation** (separate frozen subnets per task, as in Progressive Neural Networks) guarantees zero forgetting by construction but leaves open **routing**, **compact growth**, and **cheap composition** over frozen parts.

We introduce the **Growformer**, a parameter-isolated continual-learning substrate: per-task Mirrors promote to frozen groups in Main. On balanced Task E (20 seeds), **global scalar VirtualGroup blending** fails (**69.9%**); confidence argmax stays at floor (**69.5%**). Boundary authentication of expert-output LearnedRouter reveals a **representational–routing dissociation**: `f_circles ↔ r` = **0.87** yet region agreement = **55%** — the switching signal is in specialist outputs but the lattice router does not recover it reliably; 14/20 seeds collapse to constant-specialist routing (0% interior / 100% outer misroute). Global composition is a **finished negative result** with a **mechanistically specified** open problem: router training stability under sparse boundary coverage, not feature discovery. Split MNIST audits retention only.

---

## 1. Introduction

The ability to learn new tasks without corrupting old ones is a prerequisite for systems that operate in a dynamic world. A clinical decision support module must retain cardiology knowledge when oncology is added. A robot must retain navigation skills when manipulation is added. A deployed agent brain must retain one customer's consolidated specialists when another domain is trained.

Current deep learning systems fail this requirement when **one shared weight tensor** serves all tasks: fine-tuning overwrites prior functions. The continual learning literature addresses that setting with regularization (EWC, SI, MAS), architectural expansion (Progressive Networks, PackNet), and replay. Those methods compete on **how much forgetting remains** when parameters are shared.

### 1.1 Problem setting (what this paper does and does not claim)

This paper addresses a **different setting**, which we state explicitly to avoid conflating it with shared-substrate continual learning:


| Setting            | Shared-substrate CL (EWC, etc.)         | Growformer (this work)                                        |
| ------------------ | --------------------------------------- | ------------------------------------------------------------- |
| Weights            | One network, shared parameters          | Separate Mirror per task → frozen promoted group              |
| Forgetting         | Non-zero unless constrained             | **0% on frozen groups by construction** (parameter isolation) |
| Hard problem       | Interference during training            | **Gating-variable discovery**, routing, compact growth        |
| Benchmark emphasis | Retain accuracy on all tasks in one net | Retention invariant + cheap adaptation over frozen parts      |


We **do not** claim to solve catastrophic forgetting inside a single shared representation. That problem is real and remains open. We claim a **consolidation + routing + composition** substrate for deployable specialist brains where:

1. Each domain trains in an isolated Mirror, promotes to Main, and **freezes** (no further gradient on that subgraph).
2. Novel composite behaviour over frozen specialists is tested via **VirtualGroup** (global scalar blend) and input-dependent combiners (§4.3–4.3.1). Global blend **fails**; the localized open problem is **gating-variable discovery**, not combiner form.
3. Representations are **metabolically pruned** during Mirror training, yielding compact checkpoints (Section 5.4).

Our closest prior art is **Progressive Neural Networks** (Rusu et al. 2016): frozen columns per task, zero forgetting, linear memory growth. The Growformer differentiates on **physics-grown sparse topology per specialist**, **metabolically pruned per-group footprint** (linear in task count, smaller slope than unpruned columns), and **VirtualGroup composition** from few samples — not on the zero-forgetting property itself, which is inherited from isolation.

### 1.2 Contributions

1. **Mirror / Main / Promotion Gate** — architectural separation of fast isolated learning from frozen consolidated storage (CLS analog without replay).
2. **Finished negative result on global VirtualGroup** — scalar blending fails; unsupervised confidence stays at floor. Expert-output LearnedRouter **boundary-authenticated**: does not track `r < 0.4` (55% region agreement); middle rung **not established** (§4.3.1).
3. **Retention invariant** — Split MNIST demonstration that promoted groups receive no gradient from subsequent tasks (§4.1).
4. **Routing evaluation** — honest reporting of context-guided vs context-free specialist dispatch (§4.4).
5. **DSPNS substrate** — promote-freeze isolation, metabolic pruning, activity-gated topology growth (Sections 3.1–3.2).

Language deployment infrastructure (Appendix A) and speculative algebraic extensions (Appendix B) are documented but **not** part of the primary empirical claims of this preprint.

---

## 2. Related Work

### 2.1 Regularization-Based Continual Learning

Elastic Weight Consolidation (EWC, Kirkpatrick et al. 2017) estimates the importance of each parameter for prior tasks using the Fisher information matrix and adds a quadratic penalty to constrain movement of important weights during new task training. EWC achieves approximately 97% average accuracy on Split MNIST with approximately 3% residual forgetting. Synaptic Intelligence (SI) and Memory Aware Synapses (MAS) are related approaches using different importance estimators. All regularization approaches share a fundamental limitation, the penalty is a soft constraint. Forgetting is reduced, not eliminated.

### 2.2 Architectural Expansion and Parameter Isolation

Progressive Neural Networks (Rusu et al. 2016) add a complete new column of neurons per task, with lateral connections from new to old columns. Prior columns are frozen. Zero forgetting is achieved but memory grows linearly with task count. A 20-task system is 20× larger than a single-task system, with no pruning. PackNet (Mallya & Lazebnik 2018) iteratively prunes parameters after each task and dedicates the freed capacity to new tasks. Zero forgetting is achieved within fixed capacity but tasks compete for a finite sparse resource, limiting scalability.

**Relation to Growformer.** The Growformer's promote-freeze protocol is in the **parameter-isolation** family: each task's consolidated group is a separate frozen subgraph. Zero measured forgetting on held-out task splits follows from that design choice, not from a new learning rule that prevents interference inside shared weights. Our empirical focus relative to Progressive Networks is **(i)** metabolic sparsification reducing per-specialist footprint, **(ii)** VirtualGroup composition over frozen specialists, and **(iii)** geometry- and lattice-based routing — not the isolation mechanism itself.

### 2.3 Complementary Learning Systems

The Complementary Learning Systems (CLS) theory (McClelland et al. 1995, updated Kumaran et al. 2016) proposes that biological memory consolidation relies on two complementary systems, a fast-learning hippocampal system for rapid acquisition of new information, and a slow-learning neocortical system for gradual consolidation into stable long-term representations. Interference between systems is managed by their temporal separation, hippocampal memories are replayed during sleep and gradually integrated into neocortex. The Growformer implements an analogous separation computationally, a Mirror Dimension (hippocampal analog) for fast task learning, and a Main Dimension (neocortical analog) for frozen consolidated knowledge. The key distinction is that our separation is architectural rather than temporal, no replay is required. 

### 2.4 Mixture of Experts

Sparse Mixture of Experts systems (Shazeer et al. 2017, Fedus et al. 2022) route inputs to specialist subnetworks via a learned gating mechanism. Experts are trained jointly with a shared router. The Growformer's routing mechanism is functionally similar but fundamentally different in origin. Groups are trained sequentially and independently, routing emerges from embedding similarity rather than joint training, and new specialists can be added post-deployment without modifying existing specialists or retraining the router.

### 2.5 Compositional Reuse of Frozen Modules

**AdapterFusion** (Pfeiffer et al. 2021) learns scalar combination weights over frozen task-specific adapters in a shared transformer, enabling multi-task inference without retraining base weights. **Task arithmetic** (Ilharco et al. 2023) and **model merging** (Wortsman et al. 2022) combine fine-tuned model deltas or checkpoints via learned or hand-specified coefficients. These methods assume a **shared backbone** and operate in weight space.

**VirtualGroup** is closest in spirit to AdapterFusion — scalar blending over frozen specialists — but differs in substrate: each specialist is a **physically consolidated, promoted subgraph** (parameter isolation). Task E (§4.3.1) reports a **negative result** on global scalar blend and localizes the open problem to **gating-variable discovery**, not adapter-style weight tuning.

---

## 3. The Growformer Architecture

### 3.1 Defining Properties

The Growformer is a **Dynamically Structured Physical Neural System**, a category we introduce to distinguish it from existing neural network paradigms. Three **defining** properties anchor the category; a fourth is an **observed design pattern**, not a demonstrated law (Section 4.2).

**Activity-gated topology.** Network structure — which neurons exist, which are connected — emerges from competitive training dynamics (growth, pruning, optional neurogenesis) rather than fixed designer width alone. Neurons carry mass and 3D geometry; synapses below metabolic threshold are pruned on a fixed interval; co-firing strengthens surviving connections. *Full update equations for mass/velocity/geometry coupling are implementation-level; promotion freezes a group when mirror accuracy ≥ **85%**.*

**Metabolically-constrained plasticity.** Synapses have energy budgets. Connections that do not justify their metabolic cost are pruned. This produces representations that are as compact as the task requires, not as large as the architecture permits. Operationally: each training tick applies forward pass + backprop; synapses below a **pruning threshold** are removed on a fixed interval; surviving synapses are strengthened when pre- and post-neurons co-fire.

**Consolidation-based specialist storage.** Once a task is learned to criterion, its specialist structure is consolidated and frozen. Subsequent tasks train only in new isolated Mirrors. This is **parameter isolation**, the same retention guarantee as Progressive Networks; the open problems are routing, composition, and compact growth — not gradient interference inside frozen groups.

**Observed sparsification pattern (not a defining law).** Under full physics Mirror training, active hidden neuron counts often correlate with task structure (e.g. spiral 9–11 of 16 units). We report this as an empirical pattern in Section 4.2, not as a category axiom.

### 3.2 The Fractal Topology

The Growformer organizes knowledge hierarchically through a structure we call the Fractal Topology. The same pattern, observer, training space, consolidation, promotion gate, repeats at every scale.

At the **neuron scale**, individual neurons integrate local activation history and adapt their connectivity based on activity-dependent dynamics. Synapses that consistently carry useful signal are strengthened, dormant synapses are pruned.

At the **group scale**, a set of neurons differentiates into a functional specialist for a specific task domain. The group develops internal geometry, mass distribution, and connectivity that reflects the structure of its training domain. Upon meeting the consolidation criterion, the group is frozen as a complete, self-contained unit.

At the **global scale**, the GlobalObserver maintains a shared embedding space across all consolidated groups, routes new inputs to relevant specialists, and coordinates compositional reasoning across groups.

The fractal property, identical structure at every level, is not a design aesthetic. It is a functional requirement, each level needs the same capabilities (observe, learn, consolidate, gate) operating on the structures produced by the level below.

### 3.3 Mirror Dimension and Main Dimension

The central architectural innovation is the separation of learning from consolidation into two distinct environments.

The **Mirror Dimension** is a complete, isolated neural environment dedicated to training one task. It has its own competitive dynamics, its own geometry, its own mass budget. No consolidated knowledge is present. The new task trains with the full environment budget, without interference from prior task representations.

The **Main Dimension** is the consolidated knowledge store. It contains only frozen promoted groups. It never trains. It only receives promoted groups from Mirror Dimensions via the Promotion Gate.

When Task 2 trains, Task 1's consolidated group exists in Main with all neurons frozen. Task 2's Mirror has no access to Task 1's neurons. There is no gradient path between them. **Measured forgetting on Task 1 is therefore 0% by construction** — the same guarantee as freezing a Progressive Network column. The engineering question is not whether frozen weights change (they cannot), but whether the **router** selects the correct frozen specialist and whether **novel tasks** can be served by composition without full retraining.

### 3.4 Routing Over Frozen Specialists

After promotion, each group registers a **group embedding**, a fixed vector encoding the group's mean activation pattern over calibration data. The embedding is computed once and never updated.

**Continual-learning settings for routing.** We distinguish three evaluation regimes (following standard CL terminology):

- **Task-incremental:** task identity available at train time; at test time the system must select among frozen specialists **without** task labels.
- **Class-incremental:** classes arrive sequentially; the model must discriminate among all classes seen so far.
- **Task-free:** no task metadata at train or test time; routing must be inferred from input alone.

Growformer experiments in Section 4.4 evaluate **task-incremental retention with task-free dispatch** — the hard part is selecting the correct frozen group from input embeddings alone.

At inference time, routing uses one or more mechanisms:

- **GlobalObserver** — cosine similarity between the incoming activation pattern and registered group embeddings.
- **LearnedRouter** — optional InfraciliaryLattice trained on calibration data (K-NN + STA field-gradient bias).
- **MetaBrain** (language deployments) — centroid coordinator with topic classification and trichocyst volley over within-group programs.

**With task context** (oracle or high-confidence side information), routing margins reach **1.000** on MNIST splits — embeddings are orthogonal under context. **Without task context**, margins vary by input location in embedding space; robust task-free routing is an active evaluation target, not a settled result. LearnedRouter and language-intent routing (GLE calibration) are the primary paths to improve boundary behaviour.

### 3.5 Compositional Adaptation: Global Blend, Gating, and What to Condition On

**VirtualGroup (global).** Scalar weights *w* blend frozen specialist outputs: *ŷ(x) = w₁f₁(x) + w₂f₂(x)* with the **same weights for all** *x*, fit by one forward pass per training point plus least-squares solve. Frozen groups are never modified.

**Why global blend fails on region-switched tasks.** Composite labels in §4.3 follow a **spatial switch**: inner disk uses spiral specialist rule, outer annulus uses circles rule. The Bayes-optimal combiner is **piecewise** — use *f₁(x)* where region A applies and *f₂(x)* where region B applies — a function of *x*, not a constant vector. A single weight vector cannot represent a piecewise-defined target; the least-squares solve averages two half-correct experts and can land **below** either alone (Task E: **69.9%** vs singles **~74–77%**).

**Gating only helps on the correct latent — and on the correct features.** Input-dependent combiners are not uniformly better. **Confidence argmax** scores **69.5%**, at the floor with global blend (**69.9%**). Expert-output LearnedRouter reports **81.3% ± 7.8%** but **boundary authentication** shows **55% agreement** with the `r < 0.4` switch — not certified discovery (§4.3.1). Coordinate routing (**80.6%**) is ceiling-adjacent. Methods given the generative axis reach **91.5–100%**.

### 3.6 Neurogenesis

The Growformer implements mid-training neurogenesis, the addition of new neurons to a Mirror Dimension when existing capacity proves insufficient. A trigger fires when training loss exceeds a threshold after a minimum number of epochs (or when loss remains high for a consecutive streak of epochs). A new neuron is allocated or promoted from an optional **reserve pool** of pre-allocated neurons when configured, connected to adjacent layers with activity-proportional initialization, and immediately participates in competitive dynamics.

In testing, a spiral Mirror triggered neurogenesis at epoch 500 with loss 0.22, grew from 16 to 17 hidden neurons, and reached loss 0.13 after 100 additional epochs, demonstrating clean integration of new neurons into ongoing training dynamics. The reserve pool, when enabled, provides warm-start neurons for smoother integration.

### 3.7 Deployment Stack (Summary)

Language and agent deployments add an optional **inference control stack** — MetaCognition quality gating, ReflectiveField / DriveField present-state composition, basal-ganglia candidate selection, FragmentComposer, System 2 deliberation, Active Inference episode logging — that operates on **frozen** consolidated groups via conditioning and policy, not weight updates. This stack is **implemented** in the reference codebase but **not ablated or benchmarked** in the experiments of Section 4. A module-level summary appears in **Appendix A**. Primary preprint claims do not depend on it.

---

## 4. Experiments

**Reading guide.** Section 4.3 reports pilot composition (Tasks C–D); Section 4.3.1 reports Task E and **localizes the open problem** (gating-variable discovery). Section 4.1 is Split MNIST retention. Section 4.4 reports routing honestly.

### 4.1 Retention Invariant: Split MNIST

Split MNIST is widely used in continual learning but is **not a hard benchmark** — five binary tasks often yield high accuracy even for weak methods. We include it here to verify the **promote-freeze retention invariant**, not to claim state-of-the-art shared-substrate performance.

The MNIST digit dataset is split into five sequential binary classification tasks: 0 vs 1, 2 vs 3, 4 vs 5, 6 vs 7, and 8 vs 9. Each task trains to completion before the next begins.

**Setup.** MNIST images (28×28) are projected to a 64-dimensional input space via fixed random projection prior to Growformer processing. Each task spawns a Mirror Dimension, trains to accuracy criterion, and is promoted to the Main Dimension upon meeting the promotion threshold. Training set sizes range from 11,263 to 12,665 samples per task reflecting the natural class distribution in MNIST.

**Results.**


| Task        | Digits | Accuracy at Promotion | Accuracy After All 5 Tasks | Forgetting |
| ----------- | ------ | --------------------- | -------------------------- | ---------- |
| 0           | 0 vs 1 | 97.6%                 | 97.6%                      | 0.0%       |
| 1           | 2 vs 3 | 96.7%                 | 96.7%                      | 0.0%       |
| 2           | 4 vs 5 | 98.6%                 | 98.6%                      | 0.0%       |
| 3           | 6 vs 7 | 97.1%                 | 97.1%                      | 0.0%       |
| 4           | 8 vs 9 | 96.3%                 | 96.3%                      | 0.0%       |
| **Average** |        | **97.3%**             | **97.3%**                  | **0.0%**   |


The retention evaluation confirms that every task's accuracy is identical at promotion time and after all five tasks complete. The Main Dimension is structurally inert during subsequent task training. **0.0% forgetting here means frozen promoted weights were not updated** — the expected outcome of parameter isolation, not evidence of solving interference within a shared network.

**Comparison to prior work (retention only — not shared-substrate CL leaderboard).**


| System                      | Split MNIST Avg | Forgetting | Memory vs task count         |
| --------------------------- | --------------- | ---------- | ---------------------------- |
| EWC                         | ~97%            | ~3%        | Fixed shared net             |
| Progressive Neural Networks | ~99%            | ~0%        | Linear (full columns)        |
| **Growformer**              | **97.3%**       | **0.0%**   | Linear groups, pruned sparse |


The Growformer matches EWC accuracy on this toy split while exhibiting **0.0% measured forgetting** under isolation. Memory grows **linearly with promoted group count** (as in Progressive Networks), but each group's stored synapses are **metabolically pruned** during Mirror training — a smaller per-task slope than an unpruned column, not sub-linear growth in task count. **Harder benchmarks** (e.g. Split-CIFAR-100, task-free routing at scale) are planned; they are not reported in this preprint.

### 4.2 Foundational Tasks: Spiral and Circles

Prior to MNIST experiments, the Growformer was validated on two 2D classification benchmarks to establish foundational behavior.

**Double spiral classification**, a nonlinearly separable 2-class problem. The Growformer achieves 90.4–92.6% accuracy (seeds 42 and 7) matching an MLP baseline of 90.4% while self-organizing to 9-11 active neurons of 16 available, 40% fewer active neurons and 40% fewer synapses than the MLP.

**Concentric circles classification** with noise levels 0.05 and 0.25. The Growformer achieves 100% at low noise and 97.9% at high noise.

Under **full physics Mirror training** (prune/grow/geometry enabled), active hidden neuron counts correlate with task structure: the nonlinear spiral boundary typically recruits **9–11** of 16 hidden neurons, while concentric circles often retains more active units at comparable noise. We treat this as an **empirical metabolic sparsification pattern**, not a proven intrinsic-dimensionality law — the circles-vs-spiral neuron-count ordering is **not monotonic** across all training regimes (e.g. gradient-only promotion demos use fewer mirror epochs). Structural efficiency is a consequence of competitive dynamics, not a hand-tuned width parameter.

### 4.3 Pilot: Compositional Generalization via VirtualGroup (Tasks C–D)

**Scope.** VirtualGroup learns scalar blend weights over frozen groups without unfreezing them. These pilots (seed 42, `demo_phase3c_composition`) motivated the balanced evaluation in §4.3.1. They are **not** accuracy wins over single-expert baselines on held-out data.

**Task C: Area-imbalanced spiral-gated circles** (inner radius < 0.4 → spiral rule; outer → circles; uniform sampling in `[-1,1]²` so **~87%** of mass is outer).


| Method                 | Full Task C (n=100) | Train (n=30) | Held-out (n=70)             |
| ---------------------- | ------------------- | ------------ | --------------------------- |
| Spiral specialist      | **62.0%**           | —            | —                           |
| Circles specialist     | **88.0%**           | —            | —                           |
| **Oracle-best-single** | **88.0%**           | —            | **~88%** (circles wins)     |
| **VirtualGroup**       | —                   | **86.7%**    | **84.3%** (episodic recall) |


Blend weights **[0.345, 0.655]**. VirtualGroup **underperforms** oracle-best-single on held-out by ~4 points. Task C is structurally biased toward the circles specialist.

**Task D: Three-way composite** (n=100; moons added as third specialist).


| Method              | Full Task D | Train (n=40) | Held-out (n=60)             |
| ------------------- | ----------- | ------------ | --------------------------- |
| Best single (moons) | **78.0%**   | —            | —                           |
| **VirtualGroup**    | —           | **85.0%**    | **66.7%** (episodic recall) |


Train–held-out gap **18 points**; best single on full eval beats blend held-out by **11 points**.

### 4.3.1 Decisive Evaluation: Balanced Composite and the Locus of the Open Problem

Tasks C and D place most of their evaluation mass in one specialist's home region, so a single frozen expert can score well without composition. To remove that confound we construct **Task E**, a spiral-gated-circles composite with a **balanced 50/50** inner/outer split (inner: `r < 0.4` → spiral rule; outer → circles rule), a **stratified** 30-sample training set with the remaining 370 points held out, and report **mean ± std over 20 seeds (42–61)**. On a balanced task no single specialist can cover the evaluation set, so any method that beats the singles must be exploiting structure across them.

We evaluate four families on identical splits: (i) each frozen **single specialist**; (ii) **oracle-best-single**; (iii) **VirtualGroup** (global scalar blend); and (iv) **input-dependent combiners** — confidence argmax, **LearnedRouter** variants (feature type and training domain explicit in each row), learned threshold gate on radius, and logistic gate on radius. An **oracle region switch** (`r < 0.4`) is the diagnostic ceiling.


| Conditioning                               | Method                                       | Held-out (20 seeds)             |
| ------------------------------------------ | -------------------------------------------- | ------------------------------- |
| None (single)                              | Spiral specialist only                       | 76.6% ± 4.1%                    |
| None (single)                              | Circles specialist only                      | 73.5% ± 2.2%                    |
| None (global pick)                         | Oracle-best-single                           | 77.1% ± 3.4%                    |
| **None (global blend)**                    | **VirtualGroup (scalar)**                    | **69.9% ± 5.7%**                |
| None (retrain)                             | Direct composite Mirror                      | 71.0% ± 9.0%                    |
| **Input — unsupervised proxy**             | **Confidence argmax**                        | **69.5% ± 6.2%**                |
| Input — `(x,y)`, Task E labels             | LearnedRouter (coordinate; ceiling-adjacent) | 80.6% ± 6.9%                    |
| Input — expert outputs, Task E labels      | LearnedRouter `(f₁, f₂)` — boundary auth.    | 81.3% ± 7.8% (55% region agree) |
| Input — `(x,y)`, calibration task identity | LearnedRouter (deployment router)            | 76.6% ± 4.1%                    |
| Input — true latent                        | Logistic gate on `r`                         | 91.5% ± 5.1%                    |
| Input — true latent                        | Learned radius gate                          | 100.0% ± 0.0%                   |
| Diagnostic ceiling                         | Oracle region switch (`r < 0.4`)             | 100.0% ± 0.0%                   |


**Global scalar blending is a finished negative result.** VirtualGroup reaches **69.9%**, below both individual specialists and below oracle-best-single. The outcome is structural: a global blend cannot represent a piecewise-defined target.

**Methodological contribution (general).** On any balanced, region-switched composite over frozen specialists, **composite accuracy cannot distinguish routing from constant-specialist degeneracy**. A router that always picks one specialist scores ~75% on a 50/50 task while agreeing with the generative region switch on only half of points. **Region agreement** (router choice vs oracle region) and the **annulus misroute profile** (misroute rate in interior vs annulus vs outer) authenticate whether routing occurred. We report both alongside accuracy on Task E; the protocol transfers to any few-shot router benchmark.

**Boundary authentication (`--phase3e-boundary`, 20 seeds, 7400 held-out points).** Expert-output LearnedRouter reports **81.3% ± 7.8%** composite accuracy but **55.4% region agreement** and **margin↔(0.4−r) correlation 0.18** pooled — the mean is structurally misleading.

**Representational–routing dissociation.** `f_circles ↔ r` = **0.87**; `f_spiral ↔ r` ≈ **0**. The switching signal is in specialist outputs; the lattice router recovers it only rarely. The bottleneck is routing mechanism under sparse training, not feature richness.

**Degenerate routing (14/20 seeds at exactly 50% region agreement).** Annulus analysis (ε = 0.08): **0.0% interior / 100.0% outer** misroute — exactly constant always-spiral routing. Composite ~75% and region agreement 50% follow by construction on a balanced task.

**Ceiling seeds (5/20, composite ≥ 85%) — cross-tab closed.** Every ceiling seed has **margin↔r ≥ 0.44** (range **0.44–0.74**); **0/5** reach 85%+ with margin↔r < 0.30. Pearson(acc, margin↔r) = **0.47** across the cluster. No non-radius route to high accuracy was observed. Pooled annulus/interior misroute ratio **1.34×** (modest). Partial radius exploitation under favorable boundary coverage is confirmed; reliability fails on the other 14 seeds.

**Certifier for future routing numbers.** Region agreement, margin↔r correlation, and annulus misroute profile authenticate routing; composite accuracy alone is insufficient. Re-analysis: `--phase3e-boundary-analyze` on `phase3e_boundary_diagnostic.csv`.

**Unsupervised proxies stay at the floor.** Confidence argmax (**69.5%**) does not beat global blend (**69.9%**).

**Open problem (reframed).** The gating signal is present (`f_circles ↔ r` = 0.87). The failure is a router that collapses to constant-specialist degeneracy under sparse boundary coverage (14/20 seeds), recovering the signal only on favorable 30-point draws (5/20). Attack sample coverage and program stability; authenticate with region agreement, not composite accuracy.

### 4.4 Routing Evaluation (Context-Guided vs Context-Free)

After five MNIST task groups are promoted, we evaluate specialist dispatch under two regimes (Section 3.4):


| Regime             | Protocol                                                        | Result                                                                        |
| ------------------ | --------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| **Context-guided** | Task or regime identity available at routing time               | Margin **1.000** on all five tasks; group embeddings orthogonal under context |
| **Context-free**   | Cosine similarity over group embeddings only; **no task label** | Margins **vary** by task and by input location in embedding space             |


Context-free routing is the operationally relevant setting for deployed agents (user prompts do not carry task IDs). **Robust task-free routing is not settled in this preprint** — it is listed as primary future evaluation (Section 5.8).

**Failure mode.** When routing errs, the wrong **frozen** specialist is invoked — outputs remain bounded to that specialist's engrams, but task accuracy degrades. Routing error is the dominant risk in the parameter-isolated setting, not weight corruption.

---

## 5. Discussion

### 5.1 Retention Under Parameter Isolation

The 0.0% forgetting reported on Split MNIST is **guaranteed by the promote-freeze protocol**, not discovered by a novel anti-interference learning rule. Frozen Main Dimension groups have no gradient path from any Mirror Dimension training. That is the same structural guarantee as Progressive Neural Networks: **if specialists are isolated and frozen, they cannot forget by gradient overwrite**.

This is **not** a solution to shared-substrate continual learning. After Task E, the open problem is **router reliability**: specialist outputs carry the switching signal (`f_circles ↔ r` = 0.87) but lattice routing on 30 one-pass samples collapses to constant-specialist behavior on 14/20 seeds. Global VirtualGroup is a **finished negative result** (§4.3.1).

We report Split MNIST to make the invariant auditable, not to claim a breakthrough on the standard CL leaderboard.

### 5.2 Biological Motivation

The Growformer's architecture independently converges on two well-established principles from neuroscience.

The Complementary Learning Systems theory proposes the hippocampus as a fast-learning system for rapid acquisition and the neocortex as a slow-learning system for gradual consolidation. The Mirror/Main Dimension separation implements this computationally with fast learning in isolation and structural consolidation in a stable store.

Cortical column organization in mammalian neocortex suggests that stable representations are maintained in columnar structures with distinct boundaries. The Growformer's promoted groups, spatially separated, geometrically distinct, frozen against modification, are a functional analog.

Neither analogy is claimed as a mechanistic explanation. They are offered as independent motivation for isolate-then-consolidate design. Deployment-layer cortical motifs (MetaCognition, basal ganglia, neuromodulator gating) are summarized in Appendix A without empirical claims in this preprint.

### 5.3 Limitations and Honest Scope

**Shared-substrate CL is out of scope.** We do not train one shared representation across tasks with interference management. Claims about "solving catastrophic forgetting" apply only to **frozen isolated specialists**.

**Split MNIST is a retention demo, not a hard benchmark.** Near-perfect numbers on five binary tasks are expected for many methods. Stronger evidence requires harder splits (e.g. Split-CIFAR-100), multiple seeds, and confidence intervals — planned, not reported here.

**Task-free routing is incomplete.** Context-guided routing achieves margin 1.000; context-free cosine routing varies (Section 4.4). Until task-free routing is benchmarked under explicit protocol, deployed systems may require reliable intent classification or accept misrouting risk.

**Global VirtualGroup is a finished negative result.** On balanced Task E, scalar blending (**69.9%**) lands at the floor with **confidence argmax** (**69.5%**). Expert-output LearnedRouter (**81.3%**) does **not** align to the generative boundary (55% region agreement); middle rung **not established**. Radius-conditioned gates (**91.5–100%**) are **diagnostic ceilings**.

**Gating-variable discovery remains open.** Unsupervised proxies at floor (~70%); expert routing uncertified; privileged-axis gates at 92–100%. Specialist outputs encode position (`f_circles ↔ r` = 0.87) without the router recovering the switch.

**Input dimensionality.** MNIST uses fixed random projection (784→64); the ~70KB checkpoint figure counts **sparse promoted groups**, not the projection matrix.

**Neurogenesis.** Single-neuron mid-training neurogenesis is demonstrated; reserve pool is optional.

**Language and algebraic machinery.** A full language pipeline, factored generation, E8/Leech quantization, and Cl(1,7) rotors are **implemented** (Appendix A; engineering notes in Section 5.5). **Per-layer ablations and held-out benchmarks are open work.** Language continual-learning retention (7-domain mean ratio 1.0) is an internal milestone, not reproduced in Section 4.

**Code release.** Reference implementation exists; formal reproducibility package (seeds, configs, scripts) is in preparation.

### 5.4 Implications for Robotics, Edge and IoT Deployment

The complete five-task Split MNIST system serializes to approximately **70KB** for the **sparse promoted groups** (synapses that survived metabolic pressure during Mirror training). This figure **excludes** the fixed 784→64 MNIST projection matrix (~200KB if stored densely). Dead synapses consume no memory, no compute, and no representation.

Sparse inference, only the relevant group activates per input, means inference cost is constant regardless of total group count. A twenty-task system routes inputs through the same single forward pass as a two-task system. This scaling property is essential for resource-constrained deployment.

The deployment model is: train on desktop, promote groups, serialize the checkpoint, deploy inference-only to the target device. The device never trains. Its behavior is fixed and auditable after deployment. **Global scalar VirtualGroup** is not recommended for region-switched composites (§4.3.1); input-conditioned routing requires discovering the correct gating variable.

### 5.5 Engineering: Lattices, Factored Generation, and Clifford Conditioning

The language and generation path uses **concrete engineering structures** — E8 nearest-point program selection, Leech-quantized project indexing, factored response decomposition, Cl(1,7) bivector rotors for per-group conditioning — that reduce search space and enable traceable composition. Internal tests report factored training reaching loss 0.003 in ~200 steps vs 0.22 in 3000 steps for unfactored prediction on reference suites.

**What is demonstrated vs asserted.** Factored generation and lattice selection are **implemented and used in production brains**. The claim that grade-1 Cl(1,7) vectors "align" with E8 quantization is a **dimensional design choice** (8D grade-1, 8D E8 blocks) — it is **not** shown here via ablation that rotors outperform adapters or that lattice optimality buys measurable accuracy at fixed compute. **Formal guarantees** (optimal packing proofs, profinite convergence, cryptographic verification) are **conjectural** and deferred to Appendix B.

### 5.6 Structural Interpretability

A central concern with neural systems is that a trained model becomes an opaque collection of weights, a function that maps inputs to outputs with no human-readable explanation of the mapping. The Growformer's architecture provides a different property, **layered structural interpretability**. The system is not fully transparent at the synapse level, but every decision in the inference path is decomposable into auditable components.

**Routing is geometric.** Input classification is performed by measuring distance in a shared embedding space. For any input, the system can report which specialist group was selected, the quantitative confidence of that selection (cosine similarity), and the relative distances to all alternative groups. The routing decision is a geometric fact, not a softmax over opaque logits.

**Generation is factored.** The output of each specialist is not drawn from a continuous, unbounded distribution. It is assembled from a finite, enumerable set of structural components: a fixed response skeleton and a bounded set of variable positions, each with a finite vocabulary. For any generated output, the system can report which structural pattern was selected, which variable values were filled, and the confidence of each selection. The space of possible outputs for a given specialist is finite and auditable before any input is presented.

**Composition is traceable.** When compositional generation activates, fragment selection, compatibility scores, and search paths are recorded. **FragmentComposer** (Appendix A) uses a finite library of authored clauses with logged fragment IDs and scores.

**MetaCognition and selection are auditable.** Reflection scores (coherence, relevance, completeness), accept/retry/degrade outcomes, and optional basal-ganglia candidate values are available for post-hoc review when those modules are enabled.

**Knowledge is frozen and deterministic.** Consolidated specialist weights are fixed; the same input and policy configuration yield the same output. Inference **policy** (TOML guardrails) may differ by deployment; engrams do not.

**The boundary of interpretability.** Individual synaptic weights are not semantically readable. The claim is that the **decision path** — which specialist, which pattern or fragment, which scores — is decomposable, analogous to auditing which protocol a human team followed, not which molecule fired in a neuron.

### 5.7 Auditability and Bounded-Domain Deployment

This section addresses **auditability in bounded-domain agents** (support, compliance FAQ, certified workflows) — not open-ended alignment of general language models.

**Enumerable outputs aid audit, not omnibus safety.** Each specialist's outputs are drawn from finite lattice patterns and/or a finite fragment library. That makes pre-deployment enumeration and post-hoc tracing feasible. It does **not** imply safe behaviour on arbitrary open text: the system is **not** a competitive open-ended generator. Harmful content absent from the library cannot be emitted; harmful content **present** in an approved library remains a **content-governance** problem, not solved by architecture alone.

**Primary failure mode is misrouting, not unbounded generation.** Adversarial or ambiguous prompts may invoke the wrong frozen specialist or a poor pattern match. The attack surface is **misselection among known patterns**, not synthesis of arbitrary novel text — a different risk profile, appropriate for bounded deployments only.

**Frozen determinism aids certification.** Consolidated groups do not drift with continued training. Identical inputs and policy yield identical outputs across time and hardware, supporting audit replay in regulated settings.

These properties follow from parameter isolation and bounded output design. They must not be confused with **alignment** of general-purpose models.

### 5.8 Future Work

**Empirical priorities (to strengthen this preprint).**

- **Task-free routing benchmarks** — explicit protocol, multiple seeds, margin and accuracy without task IDs (MNIST splits; language intent holdouts).
- **Harder CL splits** — Split-CIFAR-100 or CORe50 under the same promote-freeze protocol, with Progressive Networks as isolated baseline.
- **Gating-variable discovery** — expert routing boundary-authenticated as non-discovery (55% region agree); understand why `f_circles` encodes `r` (0.87) without router recovering the switch.
- **VirtualGroup at scale** — language/vision composites; AdapterFusion-style baselines on matched protocols.
- **Per-layer ablations** — MetaCognition, basal ganglia, rotors vs adapters, factored vs unfactored generation (Appendix A stack).
- **Reproducibility package** — pinned seeds, configs, evaluation scripts.

**Implemented engineering (evaluation deferred).**

- Online feedback on routers/rotors with frozen groups; organogenesis on low-confidence streams; inference TOML policy; causal/world-grounding retrieval hints (Appendix A).

**Open research (Appendix B).**

- World-model substrates; profinite/nilpotent convergence theory; zero-knowledge inference proofs; categorical DAG trainer integration (Appendix B).

---

## 6. Conclusion

We presented the Growformer as a **parameter-isolated continual learning substrate**: each task trains in a Mirror, promotes to Main, and freezes. Measured forgetting on held-out task splits is **0% by construction** — not a solution to interference inside a shared weight tensor.

The **empirical contribution** is a **finished negative result** with a **mechanistically authenticated** open problem. Global VirtualGroup fails (**69.9%**). Specialist outputs encode the switching variable (`f_circles ↔ r` = **0.87**) but lattice routing recovers it on only **5/20** seeds; **14/20** collapse to constant-specialist degenerate routing (annulus analysis: **0% interior / 100% outer** misroute). The 81.3% composite mean was a mirage of degenerate floor plus lucky tail — region agreement (**55%**) is the certifier. Split MNIST audits retention.

---

## Appendix A: Deployment Inference Stack (Implemented, Not Evaluated Here)

Language agents add an optional control pipeline on frozen groups:

1. **GLE → LanguageBridge → routing** — text to embedding to specialist dispatch.
2. **Paramecium InfraciliaryLattice** — E8-quantized behavioural programs; three timescales (persistent / session / per-turn); trichocyst volley for top-K programs.
3. **ReflectiveField + DriveField** — Identity (OCEAN) ⊕ Activity ⊕ Drive, neuromodulator-gated retrieval gains.
4. **MetaCognition** — generate→reflect→decide; graceful degradation on low quality.
5. **BasalGanglia** — value-weighted selection over retrieval candidates.
6. **FragmentComposer** — finite authored clause assembly (`[fragment_compose]` policy).
7. **System 2 + Neural Coherence** — deliberate multi-step reasoning; band-decomposed ensemble checks.
8. **Active Inference spine + InferenceHarness** — episode replay; TOML/JSONL guardrails; brain package v2 plugins blob.

No section of this appendix has published per-layer ablation or held-out dialogue benchmarks in Section 4.

---

## Appendix B: Speculative Algebraic and Cryptographic Extensions (Conjectural)

The following are **research directions**, not results of this preprint:

- **Non-commutative multi-specialist composition** via continuous deformation (leader/follower ordering).
- **Profinite / nilpotent group connections** to convergence and spawn-trigger formalization.
- **Stable commutator length** as a compatibility metric between specialists.
- **Zero-knowledge proofs** of inference and authenticated brain marketplaces.

Lattice optimality and Cl(1,7)/E8 structure are used as **engineering choices** in the implementation; formal proofs and ablation superiority are **not** established here.

---

## References

Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., Rusu, A. A., ... & Hadsell, R. (2017). Overcoming catastrophic forgetting in neural networks. *Proceedings of the National Academy of Sciences*, 114(13), 3521-3526.

Rusu, A. A., Rabinowitz, N. C., Desjardins, G., Soyer, H., Kirkpatrick, J., Kavukcuoglu, K., ... & Hadsell, R. (2016). Progressive neural networks. *arXiv preprint arXiv:1606.04671*.

Mallya, A., & Lazebnik, S. (2018). PackNet: Adding multiple tasks to a single network by iterative pruning. *Proceedings of the IEEE conference on computer vision and pattern recognition*, 7765-7773.

McClelland, J. L., McNaughton, B. L., & O'Reilly, R. C. (1995). Why there are complementary learning systems in the hippocampus and neocortex: insights from the successes and failures of connectionist models of learning and memory. *Psychological review*, 102(3), 419.

Kumaran, D., Hassabis, D., & McClelland, J. L. (2016). What learning systems do intelligent agents need? Complementary learning systems theory updated. *Trends in cognitive sciences*, 20(7), 512-534.

Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G., & Dean, J. (2017). Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. *arXiv preprint arXiv:1701.06538*.

Pfeiffer, J., Kamath, A., Rücklé, A., Hwang, D., Ster, D., Vulić, I., ... & Gurevych, I. (2021). AdapterFusion: Non-destructive task composition for transfer learning. *Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics*, 487–503.

Ilharco, G., Ribeiro, M. T., Wortsman, M., Gururangan, S., Schmidt, L., Farhadi, A., & Hajishirzi, H. (2023). Editing models with task arithmetic. *International Conference on Learning Representations*.

Wortsman, M., Ilharco, G., Gadre, S. Y., Reese, E., Kembhavi, A., Taori, A., ... & Schmidt, L. (2022). Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time. *International Conference on Machine Learning*, 23965–23998.

---

*Preprint. Correspondence: Astor Rivera-Carcamo, SWTCH.AI.*