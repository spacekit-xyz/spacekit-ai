# When Composite Accuracy Lies: Few-Shot Routing Over Frozen Experts

### Parameter-Isolated Continual Learning, a Finished Negative on Global Composition, and Certifiers for Router Evaluation

> **Document role:** This is the technical and reproducibility companion. The canonical,
> self-contained paper for public readers is
> [Growformer: Adding Specialists Without Forgetting](GROWFORMER_PUBLIC_WHITEPAPER.md).

**Author:** Astor Rivera-Carcamo
**Affiliation:** SWTCH Labs, SWTCH.AI (swtch.ai), SpaceKit.xyz ([spacekit.xyz](http://spacekit.xyz))   
**Status:** Preprint, Not Peer Reviewed   
**Scope:** Parameter-isolated continual learning and routing/composition over frozen specialists. Shared-substrate continual learning is explicitly out of scope. The deployment inference stack and speculative algebraic material are **implemented but not evaluated here**; they are walled into Appendices A–D and carry no empirical claims in this preprint. **Frozen-encoder + promoted predictive (JEPA/energy) adapters** under the same promote–freeze contract are documented in [WORLD_MODELS.md](WORLD_MODELS.md) and Appendix E; they do **not** alter the Task E results in §4.3.1.

---

## Abstract

We study composition and routing over **frozen, parameter-isolated specialists**, the deployment-relevant setting where each task is trained in isolation, consolidated, and never updated, so retention is zero-forgetting by construction (as in Progressive Neural Networks) and the open problems are *which* specialist to invoke and *how* to combine them.

Our central result is a **finished negative with a mechanistically authenticated cause**. On a balanced, region-switched composite (Task E, 20 seeds), global scalar blending of frozen specialists (VirtualGroup) fails (**69.9%**), landing at the floor alongside an unsupervised confidence proxy (**69.5%**) and below both single specialists (**73.5–76.6%**). A learned lattice router on specialist outputs *appears* to succeed (**81.3%**), but **boundary authentication refutes this**: it agrees with the generative switch on only **55%** of points, and **14 of 20 seeds collapse to constant-specialist routing** (0% interior / 100% outer misroute), with the high mean assembled from that degenerate floor plus a lucky tail.

The deeper finding is a **representational–routing dissociation** for the lattice class: the switching variable is almost fully recoverable from a single specialist output (`f_circles ↔ r` = **0.87**), yet lattice routing does not reliably exploit it. An **adjustable-cone** router recovers a certified middle rung under region-supervised train (92.5% acc, 85.3% region agree, 0/20 degenerate) and under **label-free train** with specialist-output pseudo-labels (93.8% / 85.6%, 0/20 degenerate; no `r` in the loss) — both up to the same ~85% feature-information plateau, without overturning the lattice negative.

The reusable contribution is a **certification protocol**: on balanced switched tasks, composite accuracy *cannot* distinguish genuine routing from constant-specialist degeneracy; **region agreement**, the **annulus misroute profile**, and **margin↔latent correlation** can, and we show which certifier is decisive in which regime. Split MNIST (97.7%, 0.0% forgetting) is reported only to audit the promote-freeze retention invariant.

---

## 1. Introduction

Systems that accumulate domain knowledge over time, clinical decision modules, robot skills, per-customer agent brains.  must add new competence without corrupting existing competence. When one shared weight tensor serves all tasks, this fails: fine-tuning on a new task overwrites prior functions (catastrophic forgetting). The continual-learning literature addresses that shared-substrate setting with regularization (EWC, SI, MAS), architectural expansion (Progressive Networks, PackNet), and replay; these methods compete on *how much forgetting remains* when parameters are shared.

This paper studies a **different setting** and is careful not to conflate the two.

### 1.1 Problem Setting: What This Paper Does and Does Not Claim


|                | Shared-substrate CL (EWC, etc.) | This work                                                        |
| -------------- | ------------------------------- | ---------------------------------------------------------------- |
| Weights        | One network, shared parameters  | Separate frozen subnetwork per task                              |
| Forgetting     | Non-zero unless constrained     | **0% by construction** (parameter isolation)                     |
| Hard problem   | Interference during training    | **Routing and composition over frozen parts**                    |
| What we report | Retained accuracy in one net    | Retention invariant + an authenticated routing/composition study |


We **do not** claim to solve catastrophic forgetting inside a shared representation. That problem is real and remains open. We work in the parameter-isolation family, the same retention guarantee as Progressive Neural Networks (Rusu et al. 2016), where each task trains in isolation, promotes to a frozen store, and receives no further gradient. In that family, zero forgetting is *inherited from isolation*, not discovered; the genuinely open questions are **specialist dispatch (routing)** and **combination of frozen specialists (composition)**. This paper reports a negative result on the most natural composition method, authenticates its cause, and provides a protocol for evaluating routers honestly.

Our closest prior art is Progressive Neural Networks. We differ in using physics-grown sparse per-specialist topology and metabolically pruned footprints (Section 3), and in studying composition/routing over the frozen parts, not in the zero-forgetting property itself.

### 1.2 Contributions

Ordered by strength of evidence.

1. **A certification protocol for routing over frozen specialists (methodological, transferable).** On balanced, region-switched tasks, composite accuracy cannot separate genuine routing from constant-specialist degeneracy. We introduce **region agreement** (router choice vs. the generative region), the **annulus misroute profile** (misroute rate by distance to the switch boundary), and **margin↔latent correlation** as certifiers, and we show *which is decisive in which regime*: the annulus profile exposes degeneracy; margin↔latent certifies partial signal exploitation on the seeds that route. The protocol transfers to any frozen-expert router (AdapterFusion, model merging, MoE-style dispatch).
2. **A finished negative result on global composition (empirical).** Global scalar blending (VirtualGroup) over frozen specialists fails on balanced switched tasks (69.9%), and the natural unsupervised proxy (confidence argmax, 69.5%) stays at the same floor. A lattice router that *appears* to beat this (81.3%) is shown by authentication to be uncertified,  55% region agreement, 14/20 seeds degenerate.
3. **A qualified positive on anti-collapse under supervision (empirical).** An adjustable-cone router over specialist outputs (region-supervised train, oracle-free inference) reaches 92.5% held-out accuracy with 85.3% region agreement and **0/20** degenerate seeds (0/100 across an n-sweep), authenticated by certifiers the loss never optimized (§4.3.1). Accuracy is attributed to region supervision; the architecture earns anti-collapse.
4. **A middle rung under label-free train (empirical).** The same cone, trained with **no region / `r` in the loss** (pseudo-labels from specialist scalars + fixed polarity prior from `f_circles ↔ r` ≈ 0.87), reaches **93.8%** held-out accuracy and **85.6%** region agreement with **0/20** degenerate seeds (§4.3.1 Phase 3h; 6/6 pre-registered gates). This closes the label-free *train* axis on Task E under the stated contract; it does not overturn the lattice negative.
5. **A retention invariant (audited, scoped).** Split MNIST confirms that promoted groups receive no gradient from subsequent tasks (0.0% forgetting). This is the expected consequence of parameter isolation, reported for auditability, not as a leaderboard claim.
6. **The Growformer substrate (descriptive).** A promote-freeze architecture (Mirror → Main via a Promotion Gate) with physics-grown, metabolically pruned specialists. Described as the substrate the above results were measured on, not as a headline capability.

A language deployment stack (Appendix A), engineering use of lattice/Clifford structures (Appendix B), structural-interpretability properties (Appendix C), and speculative algebraic extensions (Appendix D) are **implemented but not evaluated** in this preprint and make no claims here.

---

## 2. Related Work

### 2.1 Regularization-Based Continual Learning

Elastic Weight Consolidation (EWC; Kirkpatrick et al. 2017) uses the Fisher information matrix to estimate per-parameter importance and adds a quadratic penalty constraining important weights during new-task training. It reaches ~97% average accuracy on Split MNIST with ~3% residual forgetting. Synaptic Intelligence and Memory Aware Synapses use different importance estimators. All share one limitation: the penalty is a soft constraint, so forgetting is reduced, not eliminated.

### 2.2 Architectural Expansion and Parameter Isolation

Progressive Neural Networks (Rusu et al. 2016) add a frozen column per task with lateral connections; zero forgetting, but memory grows linearly and unpruned. PackNet (Mallya & Lazebnik 2018) prunes after each task and reuses freed capacity; zero forgetting within fixed capacity, but tasks compete for a finite sparse resource.

**Relation to this work.** Our promote-freeze protocol is in the parameter-isolation family: each consolidated group is a separate frozen subgraph, and zero measured forgetting follows from that design choice, not from a new anti-interference rule. Our contribution is not the isolation mechanism but (i) metabolic sparsification of per-specialist footprint and (ii) the authenticated study of routing/composition over the frozen parts.

### 2.3 Complementary Learning Systems

CLS theory (McClelland et al. 1995; Kumaran et al. 2016) posits a fast hippocampal learner and a slow neocortical consolidator, with interference managed by temporal separation and replay. The Mirror/Main split implements an analogous fast-learning / stable-store separation, but **architecturally rather than temporally** — no replay is required. We treat this as motivation, not as a mechanistic claim.

### 2.4 Mixture of Experts

Sparse MoE (Shazeer et al. 2017; Fedus et al. 2022) routes inputs to subnetworks via a jointly trained gate. Our setting differs in origin: specialists are trained sequentially and independently, dispatch is over frozen groups, and specialists can be added post-deployment without retraining a shared router. The negative result in §4.3.1 is directly relevant to few-shot, post-hoc routing over such frozen experts.

### 2.5 Compositional Reuse of Frozen Modules

AdapterFusion (Pfeiffer et al. 2021) learns scalar combination weights over frozen adapters in a shared transformer. Task arithmetic (Ilharco et al. 2023) and model merging / model soups (Wortsman et al. 2022) combine fine-tuned deltas or checkpoints via learned or fixed coefficients. These assume a shared backbone and operate in weight space.

VirtualGroup is closest in spirit to AdapterFusion, scalar combination over frozen modules, but combines in **output space** over physically isolated specialists. We report a **negative** result for this combiner on switched tasks (§4.3.1) and do not claim superiority over adapter/merge baselines on shared-backbone benchmarks; those comparisons are future work.

---

## 3. The Growformer Substrate

We describe the architecture as the substrate on which the §4 results were measured. The properties below are **design patterns**, not a claimed new taxonomy; the one property we previously over-stated as a "law" is downgraded to an observed pattern (§3.1, §4.2).

### 3.1 Design Patterns

**Activity-gated topology.** Which neurons exist and connect emerges from competitive training dynamics (growth, pruning, optional neurogenesis) rather than from fixed designer width alone. Neurons carry mass and 3D geometry; synapses below a metabolic threshold are pruned on a fixed interval; co-firing strengthens survivors. Full update equations for the mass/geometry coupling are implementation-level; promotion freezes a group when mirror accuracy ≥ 85%.

**Metabolically-constrained plasticity.** Synapses have energy budgets; connections that do not justify their cost are pruned, yielding representations as compact as the task requires. Operationally: each training tick runs forward + backprop; sub-threshold synapses are pruned on a fixed interval; surviving synapses strengthen on co-firing.

**Consolidation-based specialist storage.** Once learned to criterion, a specialist is frozen; subsequent tasks train only in new isolated Mirrors. This is **parameter isolation**, the same retention guarantee as Progressive Networks. The open problems are routing and composition, not gradient interference inside frozen groups.

**Observed sparsification pattern (not a defining law).** Under full physics training, active neuron counts often correlate with task structure (e.g. spiral recruits 9–11 of 16 units). We report this as an empirical pattern in §4.2, explicitly *not* as an intrinsic-dimensionality law, the ordering is not monotonic across training regimes.

### 3.2 Mirror and Main

Learning and consolidation are separated into two environments. The **Mirror** is an isolated environment that trains one task with its own dynamics and budget, free of prior-task representations. The **Main** is the consolidated store: it holds only frozen promoted groups, never trains, and receives groups from Mirrors via the **Promotion Gate**.

When Task 2 trains, Task 1's frozen group sits in Main with no gradient path to Task 2's Mirror. **Measured forgetting on Task 1 is therefore 0% by construction** — the same guarantee as freezing a Progressive Network column. The live engineering question is whether the router selects the correct frozen specialist and whether novel tasks can be served by composition without full retraining. The rest of the paper shows that, for the natural composition method, the answer is currently *no*, and authenticates why.

### 3.3 Routing Over Frozen Specialists

After promotion, each group registers a fixed **group embedding** (mean activation over calibration data), computed once and never updated. We distinguish standard CL regimes, *task-incremental* (identity at train time, none at test), *class-incremental*, and *task-free* (no metadata at all), and evaluate **task-incremental retention with task-free dispatch**: the hard part is selecting the correct frozen group from input alone.

Dispatch mechanisms: **GlobalObserver** (cosine similarity to group embeddings), **LearnedRouter** (a lattice trained on calibration data; the object authenticated in §4.3.1), **KnnRouter** (an input-only cosine k-NN head used on the frozen CIFAR feature bank), and, in language deployments, **MetaBrain** (Appendix A). With task context, MNIST routing margins reach 1.000 (orthogonal embeddings). Input-only routing is now closed on the bounded in-repo rungs summarized in §4.4: full five-task Split MNIST (Phase 4d, multi-seed) and frozen-feature Split-CIFAR-10 lite (Phase 4f). These results do not overturn the sparse-boundary lattice negative in §4.3.1: they use different data regimes and, for 4f, avoid lossy lattice prototype compression.

### 3.4 Composition: Global Blend vs. Input-Conditioned Gating

**VirtualGroup (global).** Scalar weights blend frozen specialist outputs, `ŷ(x) = w₁f₁(x) + w₂f₂(x)`, with the **same weights for all x**, fit by one forward pass per training point plus a least-squares solve. Frozen groups are never modified.

**Why a global blend cannot represent a switched task.** Composite labels follow a spatial switch: inner disk → spiral rule, outer annulus → circles rule. The optimal combiner is *piecewise*, `f₁` where region A applies, `f₂` where region B applies, a function of x, not a constant vector. A single weight vector cannot represent a piecewise target; least-squares settles on a compromise that averages two half-correct experts and can land below either alone. §4.3.1 confirms this (69.9% vs. singles 73.5–76.6%) and, crucially, shows that simply switching to an input-dependent gate does **not** fix it unless the gate conditions on the right variable.

### 3.5 Neurogenesis

The substrate supports mid-training neurogenesis: when loss stays high past a minimum number of epochs, a neuron is added (or drawn from an optional reserve pool), wired with activity-proportional initialization, and enters competition immediately. In one run a spiral Mirror added a neuron at epoch 500 (loss 0.22 → 0.13 over 100 further epochs), integrating cleanly. This is a single demonstration, not an evaluated capability.

---

## 4. Experiments

**Reading guide.** §4.1 audits the retention invariant (Split MNIST). §4.2 establishes the 2D specialists. §4.3 reports the imbalanced pilots that *motivated* the decisive test. **§4.3.1 is the core of the paper**: the balanced evaluation, the negative result, and the boundary authentication that localizes the open problem. §4.4 reports MNIST routing honestly.

### 4.1 Retention Invariant: Split MNIST

Split MNIST (five binary tasks: 0v1 … 8v9) is a *weak* benchmark, high accuracy is expected even from weak methods, so we use it only to audit the promote-freeze invariant. MNIST images are projected to 64-d via a fixed random projection; each task trains a Mirror to criterion and promotes to Main.


| Task    | Digits | Acc. at promotion | Acc. after all 5 | Forgetting |
| ------- | ------ | ----------------- | ---------------- | ---------- |
| 0       | 0 vs 1 | 99.5%             | 99.5%            | 0.0%       |
| 1       | 2 vs 3 | 95.5%             | 95.5%            | 0.0%       |
| 2       | 4 vs 5 | 97.5%             | 97.5%            | 0.0%       |
| 3       | 6 vs 7 | 98.0%             | 98.0%            | 0.0%       |
| 4       | 8 vs 9 | 98.0%             | 98.0%            | 0.0%       |
| **Avg** |        | **97.7%**         | **97.7%**        | **0.0%**   |


Every task's accuracy is identical before and after subsequent training: the Main store is structurally inert during later tasks. **0.0% forgetting means frozen weights were not updated**, the expected outcome of parameter isolation, not evidence of solving shared-substrate interference.


| System                      | Split MNIST avg | Forgetting | Memory vs. task count        |
| --------------------------- | --------------- | ---------- | ---------------------------- |
| EWC                         | ~97%            | ~3%        | Fixed shared net             |
| Progressive Neural Networks | ~99%            | ~0%        | Linear (full columns)        |
| **Growformer**              | **97.7%**       | **0.0%**   | Linear groups, pruned sparse |


Memory grows **linearly** with promoted-group count (as in Progressive Networks); each group's stored synapses are metabolically pruned, giving a smaller per-task slope than an unpruned column, *not* sub-linear growth. A harder frozen-feature rung now exists on five binary CIFAR-10 class-pair tasks (Phase 4f; §4.4); full Split-CIFAR-100, CORe50, and matched isolated baselines remain future work.

### 4.2 Specialists: Spiral and Circles

Two 2D specialists are trained as the frozen parts used in composition. **Double spiral** (nonlinearly separable): 90.4–92.6% (seeds 42, 7), matching a 90.4% MLP baseline while self-organizing to 9–11 of 16 active neurons. **Concentric circles**: 100% at low noise, 97.9% at high noise. Active-neuron counts correlate with task structure under full physics training, but the spiral-vs-circles ordering is **not monotonic** across regimes, we report this as an empirical sparsification pattern, not a law.

### 4.3 Imbalanced Pilots (Tasks C–D) Motivation Only

These pilots (seed 42) motivated the balanced test in §4.3.1 and are **not** wins over single-expert baselines.

**Task C** (inner `r < 0.4` → spiral rule, outer → circles; uniform sampling, so ~87% of mass is outer). Circles alone scores 88.0%; VirtualGroup reaches 84.3% held-out — *below* the best single specialist by ~4 points. Task C is structurally biased toward circles, so it cannot demonstrate composition.

**Task D** (three-way; moons added). Best single (moons) 78.0%; VirtualGroup 85.0% on the train split but **66.7% held-out**,  an 18-point generalization gap from a 3-parameter fit on 40 points. The training-split "win" does not survive.

Both pilots place most evaluation mass in one specialist's home region. Removing that confound is the purpose of Task E.

### 4.3.1 Decisive Evaluation: Balanced Composite and the Locus of the Open Problem

**Task E** is a spiral-gated-circles composite with a **balanced 50/50** inner/outer split (inner `r < 0.4` → spiral rule; outer → circles rule), a **stratified** 30-sample training set, the remaining 370 points held out, over **20 seeds (42–61)**. On a balanced task no single specialist can cover the evaluation set, so any method that beats the singles must exploit structure across them.

We evaluate four families: single specialists; oracle-best-single; VirtualGroup (global scalar blend); and input-dependent combiners (confidence argmax; LearnedRouter, with feature type and training domain stated per row; logistic and learned radius gates). An oracle region switch (`r < 0.4`) is the diagnostic ceiling.


| Conditioning                          | Method                                             | Held-out (20 seeds)                 |
| ------------------------------------- | -------------------------------------------------- | ----------------------------------- |
| None (single)                         | Spiral only                                        | 76.6% ± 4.1%                        |
| None (single)                         | Circles only                                       | 73.5% ± 2.2%                        |
| None (global pick)                    | Oracle-best-single                                 | 77.1% ± 3.4%                        |
| **None (global blend)**               | **VirtualGroup (scalar)**                          | **69.9% ± 5.7%**                    |
| None (retrain)                        | Direct composite Mirror                            | 71.0% ± 9.0%                        |
| **Input — unsupervised proxy**        | **Confidence argmax**                              | **69.5% ± 6.2%**                    |
| Input — `(x,y)`, Task E labels        | LearnedRouter (coordinate)                         | 80.6% ± 6.9%                        |
| Input — expert outputs, Task E labels | LearnedRouter `(f₁,f₂)` — *boundary-authenticated* | 81.3% ± 7.8% (**55% region agree**) |
| Input — expert outputs, Task E labels | **Adjustable-cone** (oracle-free *inference*; region-supervised *train*) | **92.5% ± 5.9%** (**85.3% region agree**; **0/20 degenerate**) |
| Input — `(x,y)`, calibration identity | LearnedRouter (deployment router)                  | 76.6% ± 4.1%                        |
| Input — true latent                   | Logistic gate on `r`                               | 91.5% ± 5.1%                        |
| Input — true latent                   | Learned radius gate                                | 100.0% ± 0.0%                       |
| Diagnostic ceiling                    | Oracle region switch (`r < 0.4`)                   | 100.0% ± 0.0%                       |


**Phase-local baseline note.** The table above reports the canonical Phase 3e pooled evaluation (VirtualGroup 69.9%; confidence argmax 69.5%). The later Phase 3g and 3h certifiers recompute their baselines inside their own seeded evaluation paths and obtain VirtualGroup 68.4% and, in Phase 3h, confidence argmax 69.9%. Those phase-local values are retained in the pre-registered gate tables below; the 1.5-point / 0.4-point differences are rerun/split drift, not different methods or revised headline estimates.

**Global scalar blending is a finished negative result.** VirtualGroup (69.9%) sits below both singles and oracle-best-single. The cause is structural, not tuning: a constant blend cannot represent a piecewise target (§3.4). The blend is also unstable (±5.7%) where the circles specialist is nearly flat.

**Conditioning on the wrong variable is worse than not conditioning.** Confidence argmax is *input-dependent* yet scores 69.5%, at the floor with the global blend it ostensibly improves on. The contrast that matters is not global-vs-gated but **which latent the method conditions on**: nothing (blend) → 69.9%; the available unsupervised proxy (confidence) → 69.5%; the true switching latent (radius) → 91.5–100%. Methods that win are *handed the generative axis*.

#### The certification protocol (methodological contribution)

On any balanced, region-switched composite, **composite accuracy cannot distinguish routing from constant-specialist degeneracy**: a router that always picks one specialist scores ~75% on a 50/50 task while agreeing with the true region on only half of points. Three certifiers resolve this, and **each is decisive in a different regime**:

- **Region agreement** (router choice vs. oracle region): catches whether routing tracks the switch at all.
- **Annulus misroute profile** (misroute rate in interior vs. annulus vs. outer): **decisive for the degenerate cluster**: constant routing produces a 0%/100% interior/outer signature that is unmistakable.
- **Margin↔latent correlation**: **the tighter certifier for the ceiling cluster**, where the annulus ratio is noisy (one ceiling seed shows 0.89× annulus concentration despite margin↔r = 0.634).

We report all three alongside accuracy. The protocol transfers to any few-shot frozen-expert router.

#### Boundary authentication (`--phase3e-boundary`, 20 seeds, 7,400 held-out points)

The expert-output LearnedRouter's headline 81.3% does **not** survive authentication:

- **Region agreement 55.4%; pooled margin↔(0.4−r) correlation 0.18.** The router does not track the generative switch; the mean is structurally misleading.
- **Representational–routing dissociation.** `f_circles ↔ r` = **0.87**, `f_spiral ↔ r` ≈ 0. The switching signal is *present and singly recoverable* in specialist outputs, yet the router rarely exploits it. **The bottleneck is the routing mechanism under sparse training, not feature richness.**
- **Degenerate routing (14/20 seeds at exactly 50% region agreement).** Annulus analysis (ε = 0.08): **0.0% interior / 100.0% outer** misroute, exactly constant always-spiral routing, which yields ~75% accuracy and 50% region agreement by construction on a balanced task. The 81.3% mean is a mirage: a degenerate floor plus a lucky tail.
- **Ceiling seeds (5/20, composite ≥ 85%), cross-tab closed.** Every ceiling seed has margin↔r ≥ 0.44 (range 0.44–0.74); 0/5 reach ≥85% with margin↔r < 0.30; Pearson(acc, margin↔r) = 0.47 across the cluster. **No non-radius route to high accuracy exists.** These seeds do partial radius exploitation *when the 30-sample draw happens to cover the boundary*,  an existence proof of fragility, not of reliable routing.

#### Open problem (reframed by the evidence)

The gating *variable* is not missing, it is present at `r`-correlation 0.87. The failure is a router that **collapses to constant-specialist degeneracy under sparse boundary coverage** (14/20 seeds) and recovers the signal only on favorable draws (5/20). The open problem is therefore **router training stability and boundary-sample coverage**, not feature discovery. Any attack on it must be certified by region agreement and the annulus profile, *not* by composite accuracy, which we have shown can be degenerate at 75%.

#### Qualified positive: adjustable-cone anti-collapse (`--phase3g-cone`)

The lattice/VG negatives do **not** imply collapse is inevitable given region supervision. An **adjustable-cone** router (boundary-widening gate over oracle-free specialist features at inference; region labels used only at train time) is certified on the same Task E protocol:

| Pre-registered criterion | Result (n=30, 20 seeds) |
| --- | --- |
| Composite accuracy > VirtualGroup | **92.5%** vs 68.4% |
| Degenerate seeds (anti-collapse) | **0/20** (also **0/100** across n∈{20…200}) |
| Misroutes localized to annulus | annulus 26.2% > interior 8.7% |
| Confident-wrong reliance | **0.18** (competence router was ~0.60) |
| Region agreement ≥ 80% (spec gate) | **85.3%** (stretch >90% **not** met) |

Reproduce: `cargo run --release --bin growformer-demos -- --phase3g-cone` → [`phase3g_cone_results.txt`](../phase3g_cone_results.txt). Full protocol: [`COMPETENCE_ROUTING_SPEC.md`](COMPETENCE_ROUTING_SPEC.md) §10.

**Honest attribution.** Accuracy is attributable to **region supervision over learnable specialist features**. The cone architecture earns the **anti-collapse** claim (0 degenerate; n-sweep rises with coverage — the inverse of the competence-router decay). Margin↔r shaping was removed from the loss so that certifier is observational only. The 85%→~89% region-agreement plateau is read as the **~0.1-bit feature-information ceiling** in the annulus, not as under-training: the router tracks the feature budget and does not exceed it.

**What this does *not* claim.** It does not overturn the unsupervised lattice negative; does not claim `r` is recovered without region labels at train time; does not move the encoder or production-chat questions. Label-free train-time recovery is addressed next (Phase 3h).

#### Label-free train middle rung (`--phase3h-label-free`)

Phase 3h asks whether the cone's anti-collapse survives when **region / `r` are removed from the training loss** (eval certifiers only). Pseudo-labels come from specialist scalars alone: median / 2-means on `f_circles`, with polarity fixed by the known prior `f_circles ↔ r ≈ +0.87` (low circles → route spiral); near-boundary masks from specialist disagreement. Best strategy `circles_threshold` (n=30, 20 seeds):

| Pre-registered gate (§10.5) | Result |
| --- | --- |
| 0 degenerate seeds | **0/20** |
| Held-out > VirtualGroup | **93.8%** > 68.4% |
| Held-out > confidence argmax | **93.8%** > 69.9% |
| Region agreement ≥ 60% | **85.6% ± 7.2%** |
| Confident-wrong reliance < 0.50 | **0.12** |
| No region-agree decay n=20→120 | **85.2% → 85.9%** |

Ablations: `circles_cluster` 92.1% / 82.5% region; `bootstrap` 93.6% / 84.9% region — all 0 degenerate. Reproduce: `cargo run --release --bin growformer-demos -- --phase3h-label-free` → [`phase3h_label_free_results.txt`](../phase3h_label_free_results.txt). Protocol: [`COMPETENCE_ROUTING_SPEC.md`](COMPETENCE_ROUTING_SPEC.md) §10.5.

**Honest attribution.** No `r` enters the loss. The polarity prior is **fixed from prior analysis**, not re-estimated from region labels per seed. Region agreement lands on the same ~85% plateau as supervised 3g — consistent with the feature-information ceiling, not a claim of beating supervised routing. The lattice negative stands (different mechanism). Open axes shift to harder composites, multi-seed harder CL, open-world dispatch, and deployment transfer — not "is label-free train impossible on Task E."

### 4.4 Input-Only Routing on MNIST and Frozen-Feature CIFAR


| Regime | Protocol | Result |
| --- | --- | --- |
| Context-guided MNIST | Task identity available at routing | 100% agreement on all five tasks |
| Input-only MNIST (Phase 4d) | Five binary tasks; train-time task labels for the router; test input only; three seeds | **86.0%** router agreement vs. 17.1% embedding cosine; 84.9% mean task accuracy; 3/3 non-degenerate; **7/7 gates PASS** |
| Input-only CIFAR-10 lite (Phase 4f) | Five binary class-pair tasks; frozen hash-pinned 128-D patch encoder; cosine k-NN (`k=7`); test input only | **41.0%** router agreement vs. 20.5% embedding cosine; 68.2% mean task accuracy; zero forgetting; **10/10 gates PASS** |


Input-only routing is the operationally relevant case (inputs carry no task ID). Phases 4d and 4f close two bounded evaluation cells, not deployment-scale routing in general: MNIST remains easy, CIFAR-10 uses a fixed frozen patch bank and binary class-pair tasks, and 4f is a single-seed result only one point above its 40% gate. When routing errs, a wrong but frozen specialist is invoked: outputs stay bounded to that specialist's engrams, but task accuracy drops. **Routing error, not weight corruption, remains the dominant risk in the parameter-isolated setting**, which is exactly why the certification protocol of §4.3.1 matters.

---

## 5. Discussion

### 5.1 Retention Is Inherited; Routing Is the Real Problem

The 0.0% Split MNIST forgetting is guaranteed by promote-freeze, not discovered by an anti-interference rule, the same structural guarantee as a frozen Progressive Network column. Task E sharpened three routing results: (i) lattice / VirtualGroup collapse under sparse coverage (finished negative); (ii) cone anti-collapse under region-supervised train (qualified positive, Phase 3g); (iii) the same cone under **label-free train** via specialist-output pseudo-labels (Phase 3h, 6/6 gates). Specialist outputs carry the switching signal (`f_circles ↔ r` = 0.87); 3h shows that signal is usable as a train proxy without putting `r` in the loss.

### 5.2 Biological Motivation (Motivation Only)

The Mirror/Main split echoes Complementary Learning Systems (fast learner + stable consolidator) and cortical-column organization (stable, bounded representations) — offered as motivation for an isolate-then-consolidate design, not as mechanistic explanation. Deployment-layer cortical motifs are summarized in Appendix A with no empirical claims here.

### 5.3 Limitations and Honest Scope

- **Shared-substrate CL is out of scope.** "Solving catastrophic forgetting" applies only to frozen isolated specialists.
- **Split MNIST is a retention demo, not a hard benchmark.** Phase 4f adds a frozen-feature CIFAR-10 lite rung (68.2% task accuracy, 41.0% input-only routing, zero forgetting, 10/10 gates), but it is single-seed and uses five binary class-pair tasks. Stronger evidence still needs Split-CIFAR-100 / CORe50 with multiple seeds and intervals.
- **Global VirtualGroup fails** on balanced switched tasks (69.9%, at the floor with confidence argmax). The expert-output lattice router (81.3%) is **uncertified** (55% region agreement; 14/20 degenerate). A **qualified middle rung** exists under region-supervised train + cone (92.5% acc, 85.3% region agree, 0/20 degenerate; Phase 3g). A **label-free train** middle rung on the same cone passes all six §10.5 gates (93.8% acc, 85.6% region agree, 0/20 degenerate; Phase 3h) using specialist-output pseudo-labels and a fixed polarity prior — not radius features in the loss. Radius-conditioned gates (91.5–100%) remain **diagnostic ceilings**.
- **The open problem shifts:** Task E label-free *train* under the §10.5 contract is closed; full five-task input-only MNIST passes Phase 4d, and frozen-feature Split-CIFAR-10 lite passes Phase 4f. Remaining axes are harder composites, multi-seed harder CL, full Split-CIFAR-100 / CORe50, and deployment transfer of authenticated routing.
- **Task E is a single 2D construction.** Claims generalize only as far as "balanced region-switched composites over frozen specialists."
- **Context-free MNIST routing is closed only on the bounded Phase 4d protocol** (§4.4): five tasks, three seeds, 86.0% router agreement, 7/7 gates. It is not a claim of deployment-scale routing, open-world task discovery, or general class-incremental learning.
- **Predictive world-model adapters (Appendix E)** are a *related substrate*, not a claim of this preprint’s §4.3.1 results. Chat / Luna accuracy is never a world-model certifier.
- **Compute/footprint.** Historical engineering runs measured sparse promoted-group payloads at roughly 70KB, excluding the 784→64 projection matrix. That figure is not backed by a current committed five-task checkpoint artifact and is therefore descriptive, not a reproduced result of this preprint.
- **Deployment stack and algebraic machinery (Appendices A–D) are implemented but unevaluated.** No per-layer ablations or held-out benchmarks are reported; the 7-domain language retention figure is an internal milestone, not reproduced here.
- **Reproducibility.** A reference implementation exists; a pinned seeds/configs/scripts package is in preparation.

### 5.4 Deployment Footprint

Historical dense serialization (including the projection path) was approximately 200KB for the five-task Split MNIST configuration; no current committed artifact pins that measurement, so it is not treated as a benchmark result here. Inference activates only the routed group, so per-input cost is roughly constant in group count. The intended model is train-and-promote on a host, then deploy inference-only. Given §4.3.1, **global scalar VirtualGroup is not recommended** for region-switched composites. Input-conditioned cone routing is certified on Task E under the Phase 3g/3h contracts, while deployment-scale reliability, open-world dispatch, and harder-domain transfer remain open.

### 5.5 Future Work

- **Harder / transfer tests for label-free cone:** Phase 3h closed Task E under §10.5; next is whether specialist-output pseudo-labels + fixed polarity prior transfer to new switches, higher-D composites, and deployment traffic without reintroducing `r` into the loss. Do **not** chase >90% region agreement by leaking geometry into `cone_features`.
- **Extend the routing grid beyond the closed bounded cells**: expert-feature × calibration-identity (deployment-discovery cell), boundary-authenticated; open-world task discovery; class-incremental routing; and multi-seed replication of the Phase 4f k-NN result.
- **Harder CL splits**: Phase 4f closes the frozen-feature Split-CIFAR-10 lite rung (10/10 gates); next are full Split-CIFAR-100, CORe50, and Progressive Networks as the matched isolated baseline. DeepAugment remains optional and out-of-band; Phase 4c remains synthetic smoke only.
- **Frozen-encoder predictive adapters (world-model ladder):** promote–freeze + certifiers on energy / action / scene hosts — see [WORLD_MODELS.md](WORLD_MODELS.md) §8 (A–F green; 3i–3w) and Appendix E. Integration rungs landed: DM citizens (`--phase5a-wm-dm`), product/external loops (`--phase5b-product-act`, `--phase5c-external-product`), D′ lite frozen vision (`--phase5d-vjepa-vision`), language+WM `brain.bin` (`--phase5e-wm-brain`), live SpaceKit return+pin (`--phase5f-live-spacekit`), and D′ real-log JEPA (`--phase5g-vjepa-real-log`). **Full LeCun AMI / large-scale JEPA training remains deferred** until HF V-JEPA-at-scale stays green. Not a substitute for Task E claims.
- **Layer 0 concept graph (language path):** typed grounding graph certifiers (`--layer0-concept-graph`); complements JEPA predictors, does not replace them ([GROWFORMER_CAUSAL_AI.md](../GROWFORMER_CAUSAL_AI.md)).
- **SpaceKit host glue:** JSONL stdio over deploy / acting / scene hosts ([WM_SPACEKIT_HOST.md](WM_SPACEKIT_HOST.md); `--wm-host-stdio`).
- **Composition at scale**: language/vision composites; AdapterFusion / task-arithmetic baselines under matched protocols.
- **Evaluate the deployment stack**: per-layer ablations for the Appendix A modules; rotors-vs-adapters and factored-vs-unfactored generation under held-out benchmarks.
- **Reproducibility package**: pinned seeds, configs, evaluation scripts.

---

## 6. Conclusion

We studied composition and routing over frozen, parameter-isolated specialists. Retention is zero-forgetting by construction (Split MNIST audits the invariant), so the real questions are dispatch and combination. Our primary result is a **finished negative with an authenticated cause**: global scalar composition fails on balanced switched tasks (69.9%, at the floor with an unsupervised confidence proxy), and a lattice router that appears to succeed (81.3%) is shown by boundary authentication to be uncertified, 55% region agreement, 14/20 seeds collapsing to constant-specialist routing. A **qualified positive** (adjustable-cone, Phase 3g) shows anti-collapse under region-supervised train and oracle-free inference (92.5% accuracy, 85.3% region agreement, 0/20 degenerate). Phase 3h extends that rung to **label-free train** (no region/`r` in the loss): 93.8% accuracy, 85.6% region agreement, 0/20 degenerate, 6/6 pre-registered gates, via specialist-output pseudo-labels and a fixed polarity prior — without overturning the lattice negative. The switching variable is present in specialist outputs (`f_circles ↔ r` = 0.87); lattice routing does not recover it reliably, while cone routing reaches the feature-information ceiling under both supervised and label-free train contracts. The transferable contribution is the certification protocol — region agreement, the annulus misroute profile, and margin↔latent correlation — which any work routing over frozen experts can use to avoid mistaking a 75% degenerate mean for routing.

---

## Appendix A: Deployment Inference Stack (Implemented, Not Evaluated Here)

*Descriptive only. No claim in this appendix is benchmarked in §4; the primary results do not depend on any of it.*

Language/agent deployments add an optional control pipeline over frozen groups: (1) GLE → LanguageBridge → routing; (2) Paramecium InfraciliaryLattice — E8-quantized behavioural programs at three timescales, trichocyst volley for top-K programs; (3) ReflectiveField + DriveField — Identity (OCEAN) ⊕ Activity ⊕ Drive, neuromodulator-gated retrieval gains; (4) MetaCognition — generate→reflect→decide with graceful degradation; (5) BasalGanglia — value-weighted candidate selection; (6) FragmentComposer — finite authored-clause assembly; (7) System 2 + Neural Coherence — deliberate reasoning, band-decomposed ensemble checks; (8) Active Inference spine + InferenceHarness — episode replay, TOML/JSONL guardrails. None of these has published per-layer ablation or held-out dialogue benchmarks in this preprint.

## Appendix B: Engineering: Lattices, Factored Generation, Clifford Conditioning (Implemented, Not Evaluated Here)

*Descriptive only.* The generation path uses E8 nearest-point program selection, Leech-quantized indexing, factored response decomposition, and Cl(1,7) bivector rotors for per-group conditioning to reduce search space and enable traceable composition. Internal tests report factored training reaching loss 0.003 in ~200 steps vs. 0.22 in 3000 for unfactored prediction on reference suites. **What is asserted vs. shown:** factored generation and lattice selection are implemented; the grade-1 Cl(1,7)/E8 "alignment" is a dimensional design choice (8D/8D), **not** shown via ablation to outperform adapters or to buy accuracy at fixed compute. Formal guarantees are conjectural (Appendix D).

## Appendix C: Structural Interpretability and Bounded-Domain Auditability (Implemented, Not Evaluated Here)

*Descriptive only; these are architectural properties of the deployment stack, not benchmarked claims.* Routing is geometric (reportable group choice, cosine confidence, and distances to alternatives). Generation is factored (finite, enumerable structural components per specialist). Composition is traceable (logged fragment IDs and selection scores). MetaCognition and basal-ganglia selection expose reflection scores and candidate values when enabled. Consolidated weights are frozen and deterministic: identical input and policy yield identical output. Individual synaptic weights remain non-semantic; the claim is that the *decision path* is decomposable, akin to auditing which protocol a team followed.

For **bounded-domain agents** (support, compliance FAQ, certified workflows), enumerable outputs aid audit but do **not** imply open-text safety: the system is not a competitive open-ended generator, harmful content present in an approved library remains a content-governance problem, and the primary failure mode is **misrouting**, not unbounded generation. These properties must not be confused with alignment of general-purpose models, and none are evaluated in this preprint.

## Appendix D: Speculative Algebraic and Cryptographic Extensions (Conjectural)

Research directions, not results: non-commutative multi-specialist composition (leader/follower ordering); profinite/nilpotent group connections to convergence and spawn-trigger formalization; stable commutator length as a specialist-compatibility metric; zero-knowledge proofs of inference and authenticated brain marketplaces. Lattice/Clifford structures are engineering choices; formal proofs and ablation superiority are not established here.

## Appendix E: Predictive World-Model Adapters (Related Substrate; Not §4.3.1 Claims)

*Descriptive pointer only. No number in this appendix overrides or extends the Task E authentication in §4.3.1.*

Under the same **promote–freeze** discipline as this preprint, Growformer implements **frozen sensory encoders** (hash-pinned) and **promoted predictor / energy / action adapters** for latent next-step prediction, planning, and SpaceKit-callable hosts. Normative contract: [JEPA_ADAPTER_PROMOTION.md](JEPA_ADAPTER_PROMOTION.md). Capability map and beyond-toy ladder: [WORLD_MODELS.md](WORLD_MODELS.md) (§8 A–F green). Scene / acting / deploy JSON protocols: [WM_SPACEKIT_HOST.md](WM_SPACEKIT_HOST.md). Subsequent integration rungs promote acting/composed bundles as DimensionManager citizens (Phase 5a), certify a D′ lite frozen-vision bridge (Phase 5d), persist language specialists plus WM citizens in one `brain.bin` (Phase 5e), execute a multi-step SpaceKit acting-host episode gated by task return plus encoder pin (Phase 5f), and train adapters on a logged-visuomotor frozen-vision export (Phase 5g).

**What is asserted vs. shown.** Synthetic and visuomotor transfer, structure ablation, and host pin reload are certified in-repo demos (`--phase3i-jepa-wm` … `--phase3w-scene-host`). Phase 5d passes 6/6 on the D′ lite frozen-vision bridge; Phase 5e passes 10/10 gates for a language+WM `brain.bin`; Phase 5f passes 6/6 with return 7.8624 vs. random 3.7686 across 480 host steps and stable pin reload; Phase 5g passes 6/6 on a logged-visuomotor frozen-vision export. The **Meta V-JEPA 2 HF path remains optional and not certified here**: `scripts/export_vjepa_features.py --mode hf --log …` requires a Transformers build with `vjepa2` support; mock remains the CI path. The Phase 5b/5c visuomotor return remains diagnostic (`vm_ok=false` / `vm_diag=false`); disk return is the certified ship metric. **Chat / Luna accuracy is not a world-model certifier.** This appendix does not claim AMI/LeCun parity, shared-backbone continual learning, or replacement of Main Dimension by a single mega world model.

---

## References

Kirkpatrick, J., et al. (2017). Overcoming catastrophic forgetting in neural networks. *PNAS*, 114(13), 3521–3526.

Rusu, A. A., et al. (2016). Progressive neural networks. *arXiv:1606.04671*.

Mallya, A., & Lazebnik, S. (2018). PackNet: Adding multiple tasks to a single network by iterative pruning. *CVPR*, 7765–7773.

McClelland, J. L., McNaughton, B. L., & O'Reilly, R. C. (1995). Why there are complementary learning systems in the hippocampus and neocortex. *Psychological Review*, 102(3), 419.

Kumaran, D., Hassabis, D., & McClelland, J. L. (2016). What learning systems do intelligent agents need? *Trends in Cognitive Sciences*, 20(7), 512–534.

Shazeer, N., et al. (2017). Outrageously large neural networks: the sparsely-gated mixture-of-experts layer. *arXiv:1701.06538*.

Pfeiffer, J., et al. (2021). AdapterFusion: Non-destructive task composition for transfer learning. *EACL*, 487–503.

Ilharco, G., et al. (2023). Editing models with task arithmetic. *ICLR*.

Wortsman, M., et al. (2022). Model soups. *ICML*, 23965–23998.

---

*Preprint. Correspondence: Astor Rivera-Carcamo, SWTCH.AI.*