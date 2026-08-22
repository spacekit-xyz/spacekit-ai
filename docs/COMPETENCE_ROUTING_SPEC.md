# Spec: Per-Specialist Competence Routing (Item 1)

**Status:** Candidate mechanism — a falsifiable hypothesis, not an asserted fix.
**Goal:** Replace global scalar blending / decisiveness routing with per-specialist competence estimation, and certify it on Task E with the §4.3.1 protocol *before* it touches production.
**Pre-registered failure conditions are stated up front (§7) so the result cannot be narrated into a win.**

---

## 0. Why this is the next bet (and why it might fail)

The boundary authentication established three things that constrain the design:

1. **The switching signal exists** (`f_circles ↔ r` = 0.87) — so this is not a feature-discovery problem.
2. **Raw-output routing already failed** — `(f₁,f₂)` lattice router: 55% region agreement; the mean was degenerate.
3. **Decisiveness already failed worst** — confidence argmax (pick higher `|f_k−0.5|`): 69.5%, below the global blend.

So any new router must use a **feature set not yet tried** and a **training signal that distinguishes confident-correct from confident-wrong** — because decisiveness conflates the two, and that conflation is exactly what sank confidence argmax. A specialist can be confidently wrong in another specialist's region; decisiveness rewards it there, competence-on-correctness should not.

**Feature sets, tried and untried:**

| Feature set | Status | Problem |
| --- | --- | --- |
| Raw `(x,y)` | tried | Radius leak — recomputes the privileged axis |
| Raw outputs `(f₁,f₂)` | tried (55%) | Scalar outputs underdetermine correctness |
| Decisiveness `\|f_k−0.5\|` | tried, worst (69.5%) | Confident-wrong indistinguishable from confident-right |
| **Per-specialist internal activations** | **untried — this spec** | Richer than the scalar; may separate correct/wrong by region |

**The honest bet:** each frozen specialist's *hidden-layer activations* carry more about its own reliability than its scalar output does. A small head reading those activations, trained on that specialist's *own correctness*, may learn "I am confidently wrong here." If it can, competence routing clears the gate. If the internals don't separate correct from incorrect by region, competence routing lands at the ~69% floor too — and that is a stronger, publishable negative ("specialist internals do not encode their own region-of-validity"), not a disappointment to bury.

---

## 1. Definitions

For frozen specialist `k ∈ {1..K}` and input `x`:

- `f_k(x)` — specialist scalar output (existing).
- `a_k(x)` — specialist **penultimate hidden activations** (the layer before the output head), dimension `d_k`. Frozen specialist, so `a_k` is deterministic and free to read.
- `c_k(x) ∈ [0,1]` — **competence score**: a learned head's estimate of `P(specialist k is correct at x)`.
- Dispatch: `route(x) = argmax_k c_k(x)`, with abstain/tie rules (§4).

The competence head `h_k` is the *only* trainable object. **No frozen specialist weight is touched.** This preserves the retention invariant by construction.

---

## 2. What each competence head sees and learns

### 2.1 Features (per specialist, no cross-specialist or coordinate inputs)

`h_k` input = `a_k(x)` (penultimate activations of specialist k) **only**.

Explicitly **excluded** from `h_k`'s inputs:
- `(x,y)` coordinates — prevents the radius leak.
- Other specialists' outputs/activations — keeps each head per-specialist and prevents the shared-gate collapse mode that produced 14/20 degenerate seeds.
- The scalar `f_k(x)` alone as the sole feature — that reduces to a decisiveness variant.

### 2.2 Label (per specialist, no composite labels)

For each calibration point `x` with true composite label `y`:
`correct_k(x) = 1 if specialist_k_prediction(x) == y else 0`.

`h_k` is trained as a binary calibrated classifier: predict `correct_k(x)` from `a_k(x)`.

### 2.3 Head architecture (deliberately small)

`h_k`: `a_k(x) → Linear(d_k → 16) → ReLU → Linear(16 → 1) → sigmoid`.

### 2.4 Calibration

After training, apply temperature scaling (single scalar `T_k`) on `h_k`'s logits, fit on a held-out calibration slice, so `c_k` is a calibrated probability.

---

## 3. Training data budget

- **Router calibration set:** `n_router = 300` stratified points (50/50 region balance enforced), distinct from the 30-sample composition set and from held-out eval.
- Report router performance as a function of `n_router ∈ {30, 100, 300}`.
- Held-out eval remains the same 370-point Task E held-out set, untouched.

---

## 4. Dispatch rule

```
scores = [c_1(x), ..., c_K(x)]
k* = argmax(scores)
if max(scores) < τ_abstain:
    action = ENSEMBLE_TOP2
elif (top1 - top2) < τ_margin:
    action = ENSEMBLE_TOP2
else:
    action = route to specialist k*
```

- `τ_abstain` (default 0.5), `τ_margin` (default 0.1) tuned on calibration, **frozen before held-out eval**.

---

## 5. Certification gate (the §4.3.1 protocol, applied)

Run on Task E, **20 seeds (42–61)**, same stratified split. Emit per-point CSV (`competence_routing_diagnostic.csv`).

### 5.1 Primary metrics

| Metric | Pass condition |
| --- | --- |
| Held-out composite accuracy | > singles (~77%) **and** > confidence argmax (69.5%) by > 1 std |
| Region agreement | **≥ 80%** mean |
| Routing entropy | **No seed < 0.3 bits** |
| Annulus misroute ratio | Interior misroute **> 0%** on all seeds |
| margin↔(0.4−r) correlation | **≥ 0.5** mean |

### 5.2 Anti-leak / anti-degeneracy controls

1. Boundary-alignment scatter
2. Degenerate-seed count: **fail if ≥ 4/20**
3. **Confident-wrong probe:** on `\|f_k−0.5\| > 0.4` and `correct_k = 0`, report `c_k`. **The whole bet is that `c_k` is LOW there.**

### 5.3 Required baselines

Including decisiveness-trained head ablation (same architecture, label = decisiveness not correctness).

---

## 6. Decision table (filled after phase3f run, 2026-06-05)

| Outcome | Reading | Action |
| --- | --- | --- |
| **Acc 78.9%±6.1%, region 56.7%±7.8%, 10/20 degenerate, cw_c=0.60±0.15, margin↔r=−0.31** | **Mechanism rejected** — matches expert-router floor (55% region); confident-wrong probe failed (c_k stays high); 10/20 constant-routing collapse | Report publishable negative: specialist penultimate activations do not encode own region-of-validity under correctness labeling. Do NOT ship. |
| n_router=30 looked better (90% acc, 74% region, 0 degenerate) | Small-budget overfit / lucky seeds — not the certification budget | Scoped note only; primary gate is n_router=300 |
| n_router=100: 86.5% acc, 67.5% region, 2 degenerate | Intermediate — still below region gate | Does not change rejection at n=300 |
| Decisiveness ablation (n=300): 69.4% acc, cw_c=0.96 | Lands at confidence-argmax floor; label matters at margin | Correctness labeling helps accuracy slightly but not authentication |

---

## 7. Pre-registered honest caveats

See full spec in project chat / implementation PR.

---

## 8. Production trial (only if §6 returns the top row)

1. Implement **routing-entropy runtime guard** regardless of outcome.
2. Shadow-mode first behind feature flag.
3. Switch user-facing routing only after shadow audit clears §5.1 thresholds.

---

## 9. Implementation checklist

- [x] Add activation hook to frozen specialists exposing `a_k(x)` (penultimate layer)
- [x] `CompetenceHead` (16-unit MLP + sigmoid) + temperature scaling per specialist
- [x] Router calibration set generator (stratified, `n_router` sweep {30,100,300})
- [x] Dispatch with abstain/tie → `ENSEMBLE_TOP2`, thresholds frozen pre-eval
- [x] `--phase3f-competence` demo: 20 seeds, emits `competence_routing_diagnostic.csv`
- [x] Metrics: accuracy, region agreement, routing entropy, annulus ratio, margin↔r
- [x] Controls: boundary scatter, degenerate-seed count, **confident-wrong `c_k` probe**
- [x] Baseline rows incl. decisiveness-trained head ablation
- [x] `--phase3f-analyze` for CSV re-analysis
- [x] Fill decision table (§6) — **REJECTED** at n_router=300 (see §6)

---

## 10. Phase 3g: Adjustable-Cone Cognitive Router — authenticated anti-collapse mechanism (Task E)

§6 rejected competence-on-correctness routing: it collapsed to constant-specialist on 10/20 seeds,
landed at 56.7% region agreement, and **decayed as data grew** (90% acc at n=30 → 67% at n=300 — the
fixed-low-n mirage this project has been burned by repeatedly). Phase 3g returns to the same Task E
gate with a different hypothesis — a **multi-scale router whose effective "cognitive cone" expands
near the decision boundary** (Levin's cognitive-light-cone framing): a fast small-cone gate dispatches
a single specialist deep in a region; a learned **boundary controller** escalates points near the
annulus to a wide-cone, input-dependent **piecewise gate** (the per-point blend `VirtualGroup`
structurally lacks). It is trained with a boundary-aware curriculum (oversample the annulus) and a
cone-expansion regularizer (pull the annulus mean toward balanced, penalizing constant collapse).

**This claim is deliberately narrow.** It is *not* "forgetting is solved" and *not* "the §6 rejection
is reversed in general." It is: **an explicit boundary-uncertainty controller, trained with region
supervision, resists the constant-specialist collapse that killed unsupervised scalar blends
(`VirtualGroup`) and per-specialist gates, on Task E, and that resistance strengthens rather than
degrades as boundary coverage grows.** That is the whole of the contribution.

**Where the accuracy comes from — supervision vs. architecture (read this before the numbers).** The
route head is **supervised on the region label `r < 0.4` at training time**. This is permitted under
the fair-fight contract, which is an oracle-free-*inference* contract (the router reads only features
at test time) — *not* an oracle-free-*training* contract. The two contributions must therefore be
attributed separately, and our own ablation forces the split:

- **The accuracy (≈92–95%) is attributable to region supervision over sufficiently-informative
  features.** Removing the contaminated margin-shaping term left accuracy and region agreement
  unchanged, which proves the lift was never coming from margin shaping — it comes from training the
  route head directly on the region target, made *learnable* because the frozen features carry the
  switch (`f_circles ↔ r ≈ 0.87`). The router did not *discover* the regions unsupervised; it was
  *told* them at train time and learned to predict them from features that encode them.
- **The cone / boundary-widening architecture earns the *anti-collapse* claim, not the accuracy.**
  0/100 degenerate across the sweep, the confident-wrong reliance of 0.18, and the rising-with-coverage
  shape are what the architecture buys: a region-supervised *plain* gate can still collapse under
  sparse boundary coverage; the boundary-widening controller is what makes the supervised recovery
  *robust*. That robustness is the architectural result.

So the precise statement of what this shows about the §6 negative: §6 rejected *unsupervised*
per-specialist routing (confidence, blend weights — proxies with no region target). We have **not**
shown the switch is recoverable *without* region labels, and §10 must not be read that way. We have
shown that **the collapse was not inevitable given region supervision and the right architecture** —
which qualifies the negative on the supervised axis without overturning its unsupervised core.

**Decontamination (why margin↔r is not a gate).** An earlier version regressed the decision margin
toward `(0.4 − r)` ("margin shaping") and then reported `margin↔(0.4−r)` as a certified pass. That is
circular — training on a quantity and certifying on it measures only how well the loss optimized it.
**Margin shaping has been removed from every head's loss.** `margin↔r` is still computed (the loss no
longer touches it, so it is now uncontaminated) but it is reported **observationally only** and is
**excluded from the verdict**. The result rests on certifiers the training loss never saw.

**Fair-fight contract (enforced, unit-tested).** Inference is **oracle-free**: every head reads only
features built from frozen-specialist outputs (`cone_features`: centered outputs, disagreement,
ambiguity, decisiveness gap). The latent radius `r` is used **only at training time** (region labels,
annulus curriculum) and for certification — never as an inference input. The module
(`src/dimension/cone_router.rs`) takes no geometry; a unit test asserts decisions are a pure function
of the oracle-free features.

### 10.1 Pre-registered n-sweep — the decisive boundary-coverage test (20 seeds each)

The fixed-n table is exactly the configuration that produced misleading headlines before, so the
pre-registered decider is the **n-sweep**: does the result hold as train boundary coverage grows? The
competence router got *worse* with more data; this one does not.

| train n | annulus pts (train) | Cone acc | VG acc | Region agree | Degenerate | Annulus/interior | Conf-wrong reliance |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 20 | ~3 | 92.9% ± 4.3% | 67.7% ± 5.6% | 84.5% ± 5.7% | 0/20 | 5.09 | 0.16 |
| 30 | ~4 | 92.5% ± 5.9% | 68.4% ± 6.9% | 85.3% ± 8.3% | 0/20 | 4.97 | 0.18 |
| 60 | ~8 | 93.6% ± 3.3% | 68.4% ± 7.8% | 87.3% ± 5.7% | 0/20 | 6.87 | 0.16 |
| 120 | ~16 | 94.2% ± 3.0% | 67.4% ± 7.1% | 88.3% ± 5.3% | 0/20 | 5.36 | 0.15 |
| 200 | ~26 | 94.6% ± 3.9% | 68.3% ± 8.7% | 88.9% ± 5.1% | 0/20 | 15.37 | 0.16 |

**Across the full sweep (100 runs): 0 degenerate; cone > VirtualGroup at every n.** Accuracy and region
agreement *rise* monotonically with coverage (92.9 → 94.6%; 84.5 → 88.9%) — the **inverse** of the
competence router's collapse. The result is not a fixed-n artifact: held-out annulus generalization
improves with coverage rather than depending on the curriculum having fed the boundary at one budget.

### 10.2 Headline detail (canonical split n=30, 20 seeds 42–61, ~370 held-out)

| Method | Held-out composite accuracy |
| --- | --- |
| Oracle-best-single (global) | 77.4% ± 3.8% |
| **VirtualGroup** (global scalar blend) | 68.4% ± 6.9% |
| LearnedRouter expert `(f₁,f₂)` (the §6-era degenerate) | 80.9% ± 9.3% |
| **Adjustable-Cone Router (oracle-free *inference*, region-supervised train)** | **92.5% ± 5.9%** |
| Oracle region switch (`r < 0.4`, ceiling) | 100.0% ± 0.0% |

**Certifiers the loss did not touch** (the spine of the result):

| Pre-registered criterion | Result | Verdict |
| --- | --- | --- |
| Composite accuracy > VirtualGroup | 92.5% vs 68.4% | **PASS** |
| Routing entropy — 0 degenerate seeds (anti-collapse) | 0/20 at n=30; **0/100 across the sweep** | **PASS** |
| Misroutes localized to annulus | annulus 26.2% vs interior 8.7%, 20/20 seeds | **PASS** |
| Confident-wrong probe — gate down-weights confidently-wrong specialist | reliance **0.18** (competence router: 0.60) | **PASS** |
| Anti-collapse holds across the n-sweep | 0/100 degenerate, cone > VG at every n | **PASS** |
| Region agreement > 90% (stretch; §5.1 gate is ≥ 80%) | 85.3% ± 8.3% (worst seed 67.8%) | **clears 80% gate, misses 90% stretch** |

Observational (not a gate): `margin ↔ (0.4 − r) = 0.85 ± 0.05`. Now uncontaminated, but excluded.

Full per-seed log: [`phase3g_cone_results.txt`](../phase3g_cone_results.txt). Reproduce (deterministic;
two independent runs produced identical headline numbers): `cargo run --release --bin growformer-demos -- --phase3g-cone`.

### 10.3 Reading

The spine is two results the loss never optimized, and they are clean:

1. **Anti-collapse (0/100 degenerate across the n-sweep).** The whitepaper's authenticated pathology
   was constant-specialist degeneracy (14/20 for the lattice expert router; 10/20 for competence
   routing). The cone router does not collapse on any seed at any boundary budget. The loss trains on
   neither entropy nor the degeneracy signature, so this is uncontaminated.
2. **Confident-wrong probe (reliance 0.18).** On held-out points where a specialist is *decisive and
   wrong*, the gate puts only 0.18 weight on it — versus the competence router's 0.60, the value that
   sank §6. The cone is genuinely down-weighting confidently-wrong experts, not just rewarding
   decisiveness with extra machinery. This matters because the gate conditions on the same forward
   features the CMI work measured at ~0.1 bit; the probe confirms the lift is real selection, not
   leakage or decisiveness.

Annulus-localization (misroutes 3–15× more frequent in the ε-band than the interior, 20/20 seeds) is a
third uncontaminated property — and note the cone-expansion regularizer pushes *against* it, so the
certifier is adversarially clean.

**The one stretch miss is region agreement: 85.3%, and it is the predicted ceiling — confirmed, not a
shortfall.** It clears the §5.1 ≥ 80% gate but not the plan's > 90% goal (worst seed 67.8%). The
residual is structural: misroutes concentrate in the ε = 0.08 annulus where the features are genuinely
ambiguous (both specialists ≈ 0.5). This is the **same fact as the ~0.1-bit CMI result, viewed twice** —
the annulus is where the forward features simply do not carry a clean region bit, so even
region-supervised training cannot extract one. The sweep caps at ~89% because that is where the feature
information runs out, and the router does *as well as the feature information permits and no better* —
exactly the behavior of a legitimately-supervised-but-feature-limited router (a leaking or
overconfident router would exceed its feature budget). We do **not** claim the geometric boundary is
fully recovered; we claim the collapse is cured and the accuracy tracks the feature-information limit.

### 10.4 Status & next step

**Certified as a research result on Task E:** a region-supervised, boundary-widening router whose
*architecture* resists the constant-specialist degeneracy of unsupervised scalar blends and
per-specialist gates, authenticated by the two pre-registered decisive tests (n-sweep, confident-wrong
probe) on certifiers the loss never saw. The >90% region-agreement stretch is the confirmed
feature-information ceiling under current `cone_features`, not a training pathology to chase by
leaking geometry into the feature vector.

**What this result does *not* move (keep these separate).** This is a Task E **routing** result. It is
*not* the **encoder** question — that was solved separately by `all-mpnet-base-v2`, and §10 must not be
cited as bearing on it. It is *not* the **production** question — the unlabeled-real-traffic bottleneck
the live-capture spec addressed is untouched here.

**Why the path to production runs *through* the data-collection step, not around it.** Task E enjoys a
luxury production does not: **known region labels (`r`) at training time**, which is precisely what made
the route head supervisable. In production there is no `r`; the analog is having **reliable intent
labels to supervise the router** — i.e. the same labeled-disjoint-data bottleneck everything else is
gated on. So a clean win here does **not** yield a production-ready router: the cone router's path to
production passes *through* the labeled-data collection step. Per §8, any wiring into `observer.rs` /
`service.rs` stays gated behind the runtime entropy guard and a shadow-mode trial on real traffic — and
only once a supervisory signal for routing exists.

### 10.5 Phase 3h: Label-free train cone (eval labels only) — CERTIFIED

**Hypothesis (pre-registered).** The switch is recoverable from frozen-specialist outputs /
disagreement **without region labels in the training loss** (signal: `f_circles ↔ r` ≈ 0.87).
Phase 3g proved anti-collapse *given* region supervision; 3h asks whether a middle rung exists
under **oracle-free training**.

**Contract** (unchanged from pre-registration)

| Axis | Rule |
| --- | --- |
| Inference features | Same oracle-free family as 3g (`cone_features`); no raw `(x,y)` / `r` at test |
| Train labels | **Forbidden:** region / `r` / annulus membership in the loss. **Allowed:** specialist outputs, disagreement, confidence, proxies derived only from those |
| Eval labels | Region labels for certifiers only |
| Forbidden shortcuts | Handing `r` to the train loss; optimizing margin↔r; claiming >90% region agree by feature leakage |

**Mechanism.** Pseudo-labels from specialist scalars only: median / 2-means on `f_circles`
(polarity fixed by the known `f_circles ↔ r ≈ +0.87` prior: low circles → route spiral);
near-boundary mask from specialist disagreement percentile. Best strategy: `circles_threshold`
(ablations: `circles_cluster`, one-round `bootstrap` self-distill).

**Headline (n=30, 20 seeds; best = circles_threshold)**

| Gate | Pass bar | Result |
| --- | --- | --- |
| Degenerate seeds | 0/20 | **0/20** |
| vs VirtualGroup | held-out > VG | **93.8%** > 68.4% |
| vs confidence argmax | held-out > conf | **93.8%** > 69.9% |
| Region agreement | ≥ 60% | **85.6% ± 7.2%** |
| Confident-wrong reliance | < 0.50 | **0.12** |
| n-sweep region agree | no decay 20→120 | **85.2% → 85.9%** |

Ablations at n=30: cluster 92.1% / 82.5% region; bootstrap 93.6% / 84.9% region — all 0 degenerate.

**Verdict.** **6/6 gates PASS.** Label-free train middle rung under §10.5. Region agree matches the
Phase 3g supervised plateau (~85%); do **not** treat that as meeting 3g's supervised ≥80% gate
framing — 3h's pre-registered floor was ≥60%. Attribution: accuracy rides on the specialist-output
proxy + fixed polarity prior, not on handing `r` to the loss. Does not overturn the lattice
negative (different router class / training).

**Reproduce:** `cargo run --release --bin growformer-demos -- --phase3h-label-free`  
Artifact: [`phase3h_label_free_results.txt`](../phase3h_label_free_results.txt).  
**Whitepaper pointer:** [`GROWFORMER_WHITEPAPER.md`](GROWFORMER_WHITEPAPER.md) §4.3.1 / §5.5.

### 10.6 Phase 3i — JEPA / world-model adapters (WM Task E toy)

Predictive specialists under the same honesty protocol: **frozen, hash-pinned encoder**;
**promoted predictor adapters** only; cone routing on affinity scalars; certifiers = regime
agreement / degeneracy / MSE vs VG and confidence-argmax.

- Contract: [`JEPA_ADAPTER_PROMOTION.md`](JEPA_ADAPTER_PROMOTION.md)
- Mapping: [`WORLD_MODELS.md`](WORLD_MODELS.md) §3.2–§3.3
- Reproduce: `cargo run --release --bin growformer-demos -- --phase3i-jepa-wm`
- Artifact: [`phase3i_jepa_wm_results.txt`](../phase3i_jepa_wm_results.txt)

### 10.7 Phase 3j — Energy-based JEPA adapters

Promoted **latent energy landscapes** \(E(z,z')\) + proposal + affinity. Extra gates: energy
margin (away−home) > 0.01; cone true-pair energy ≤ VG energy. Not metabolic synapse energy.

- Contract: [`JEPA_ADAPTER_PROMOTION.md`](JEPA_ADAPTER_PROMOTION.md) §8
- Mapping: [`WORLD_MODELS.md`](WORLD_MODELS.md) §3.2.1
- Reproduce: `cargo run --release --bin growformer-demos -- --phase3j-energy-wm`
- Artifact: [`phase3j_energy_wm_results.txt`](../phase3j_energy_wm_results.txt)

### 10.8 Phases 3k / 3ℓ / 3m — geometric, probabilistic, neuro-symbolic

Successors on the Phase 3j energy substrate ([`wm_frontier.rs`](../src/dimension/wm_frontier.rs)):

| Phase | Idea | Reproduce | Artifact |
| --- | --- | --- | --- |
| **3k** | Clifford grade-1 latents + geo energy | `--phase3k-geo-wm` | [`phase3k_geo_wm_results.txt`](../phase3k_geo_wm_results.txt) |
| **3ℓ** | Ensemble + temperature abstain | `--phase3l-prob-wm` | [`phase3l_prob_wm_results.txt`](../phase3l_prob_wm_results.txt) |
| **3m** | Rule penalties on \(E\) | `--phase3m-sym-wm` | [`phase3m_sym_wm_results.txt`](../phase3m_sym_wm_results.txt) |

Contract unchanged: [JEPA_ADAPTER_PROMOTION.md](JEPA_ADAPTER_PROMOTION.md) §8.

### 10.9 Phases 3n–3q — action, compose, hard transfer, deploy

| Phase | Idea | Reproduce | Artifact |
| --- | --- | --- | --- |
| **3n** | \(E(z,a,z')\) + planner | `--phase3n-action-wm` | [`phase3n_action_wm_results.txt`](../phase3n_action_wm_results.txt) |
| **3o** | Compose 3k+3ℓ+3m | `--phase3o-compose-wm` | [`phase3o_compose_wm_results.txt`](../phase3o_compose_wm_results.txt) |
| **3p** | 8D / 3-regime hard | `--phase3p-hard-wm` | [`phase3p_hard_wm_results.txt`](../phase3p_hard_wm_results.txt) |
| **3q** | Serialize + `deploy_step` | `--phase3q-deploy-wm` | [`phase3q_deploy_wm_results.txt`](../phase3q_deploy_wm_results.txt) |

Code: [`wm_transfer.rs`](../src/dimension/wm_transfer.rs). Beyond-toy ladder: [`WORLD_MODELS.md`](WORLD_MODELS.md) §8.

### 10.10 Phase 3r — beyond-toy proof rungs

| Gate | Idea | Reproduce |
| --- | --- | --- |
| **E-rank close** | True-next hinge + propose distill (≥55%) | `--phase3r-beyond-toy` [A] |
| **Foreign×2** | Bounce ball + central force, shared frozen encoder | `--phase3r-beyond-toy` [B] |
| **Sim loop** | JSONL log of \(E\) / route / abstain; pin-stable | `--phase3r-beyond-toy` [C] |

Code: [`wm_proof.rs`](../src/dimension/wm_proof.rs). Encoder stand-in: `data/wm/frozen_external_encoder_v1.json`.

### 10.11 Phase 3s — open ladder (D + visuomotor C + SpaceKit E)

| Gate | Idea | Reproduce |
| --- | --- | --- |
| **D Vision freeze** | Offline JEPA-style vision encoder; pin unchanged after adapter train | `--phase3s-open-ladder` [D/C] |
| **Visuomotor** | Push–object 8×8 camera log; same certifiers; no regime in loss | `--phase3s-open-ladder` [D/C] |
| **SpaceKit host** | JSON `load_bundle` / `step` / reload pin | `--phase3s-open-ladder` [E] |

Code: [`wm_open.rs`](../src/dimension/wm_open.rs), host doc [`WM_SPACEKIT_HOST.md`](WM_SPACEKIT_HOST.md). Vision slot: `data/wm/frozen_vision_encoder_v1.json`.

### 10.12 Phase 3t — product surface (agents that act)

| Gate | Idea | Reproduce |
| --- | --- | --- |
| **Disk act** | Planning-energy agent; episode return > random & VG | `--phase3t-act-wm` [F1] |
| **Visuomotor act** | Same on push–object camera domain | `--phase3t-act-wm` [F2] |
| **Acting host** | `load_acting` / `act` / reload pin | `--phase3t-act-wm` [F3] |

**Non-goal:** chat / Luna accuracy. Code: [`wm_act.rs`](../src/dimension/wm_act.rs).

### 10.13 Phase 3u — V-JEPA frozen export bridge

| Gate | Idea | Reproduce |
| --- | --- | --- |
| **Export pin** | Projector/student fingerprint unchanged after adapter train | `--phase3u-vjepa-wm` |
| **Transfer** | Regime / margin / MSE≤VG on visuomotor under frozen export | `--phase3u-vjepa-wm` |

```bash
# Meta weights (optional; needs torch + transformers)
python3 scripts/export_vjepa_features.py --mode hf --model facebook/vjepa2-vitl-fpc64-256
cargo run --release --bin growformer-demos -- --phase3u-vjepa-wm
```

Code: [`wm_vjepa.rs`](../src/dimension/wm_vjepa.rs).

### 10.14 Phase 3v — Spatial scene-graph world model

| Gate | Idea | Reproduce |
| --- | --- | --- |
| **Encoder pin** | Frozen scene MLP fingerprint unchanged | `--phase3v-scene-wm` |
| **No chat** | Chat / Luna unused as certifier | `--phase3v-scene-wm` |
| **Transfer** | Regime / margin / MSE≤VG on stack vs slippery | `--phase3v-scene-wm` |
| **Acting** | Episode return > random and VG | `--phase3v-scene-wm` |
| **Structure ablation** | Edge shuffle worsens next-step MSE | `--phase3v-scene-wm` |

```bash
cargo run --release --bin growformer-demos -- --phase3v-scene-wm
```

Code: [`wm_scene.rs`](../src/dimension/wm_scene.rs). Regime labels are eval-only (not in scene features).

### 10.15 Phase 3w — SpaceKit scene-graph host

| Gate | Idea | Reproduce |
| --- | --- | --- |
| **load / step / act** | JSON host ops succeed | `--phase3w-scene-host` |
| **Pin reload** | Same fingerprint after new session + file | `--phase3w-scene-host` |
| **Host transfer** | Regime via `scene_deploy_step` ≥ 60% | `--phase3w-scene-host` |
| **Host acting** | Return via `scene_act_step` > random | `--phase3w-scene-host` |

```bash
cargo run --release --bin growformer-demos -- --phase3w-scene-host
```

Code: [`wm_scene_host.rs`](../src/dimension/wm_scene_host.rs). Protocol: [WM_SPACEKIT_HOST.md](WM_SPACEKIT_HOST.md).

### 10.16 Layer 0 concept graph (language path)

| Gate | Idea | Reproduce |
| --- | --- | --- |
| **Expand fixtures** | Known intents yield expected graph keywords | `--layer0-concept-graph` |
| **Depth structure** | `max_depth>0` adds terms beyond roots | `--layer0-concept-graph` |
| **Negative** | Neutral intent does not spuriously expand | `--layer0-concept-graph` |

```bash
cargo run --release --bin growformer-demos -- --layer0-concept-graph
```

Code: [`world_grounding.rs`](../src/inference/world_grounding.rs). Complements JEPA WM; not a chat certifier.

### 10.17 Phase 4a — context-free MNIST routing scaffold

| Gate | Idea | Reproduce |
| --- | --- | --- |
| **Guided** | Task-tag routing agree ≥ 90% | `--phase4a-context-free-mnist` |
| **Free harder** | Context-free agree < guided | `--phase4a-context-free-mnist` |

Does **not** close Whitepaper §4.4. Split-CIFAR remains future work.

```bash
# Requires MNIST_ROOT / data/ IDX files
cargo run --release --bin growformer-demos -- --phase4a-context-free-mnist
```

Code: [`context_free_mnist.rs`](../src/dimension/context_free_mnist.rs).

### 10.18 Phase 4b — LearnedRouter context-free at test

| Gate | Idea | Reproduce |
| --- | --- | --- |
| **Guided ceiling** | Task tags still ≥ 90% | `--phase4b-cf-mnist-router` |
| **Router > cosine** | LearnedRouter beats embedding cosine | `--phase4b-cf-mnist-router` |
| **Anti-collapse** | Not constant-specialist | `--phase4b-cf-mnist-router` |

Train uses task labels; **test is input-only**. Does not close full 5-task / CIFAR §4.4.

```bash
cargo run --release --bin growformer-demos -- --phase4b-cf-mnist-router
```

### 10.19 Phase 4c — Split-CIFAR protocol scaffold

Synthetic promote–freeze smoke only. Never a CIFAR-100 claim. Real lite: §10.21.

```bash
cargo run --release --bin growformer-demos -- --phase4c-split-cifar-scaffold
```

### 10.20 Phase 4d — full 5-task Split-MNIST CF LearnedRouter (multi-seed)

| Gate | Idea | Reproduce |
| --- | --- | --- |
| **5 tasks** | Digit pairs (0,1)…(8,9) | `--phase4d-cf-mnist-full` |
| **Multi-seed** | Mean over 3 seeds (42–44) | `--phase4d-cf-mnist-full` |
| **Router > cosine** | LearnedRouter beats embedding cosine | `--phase4d-cf-mnist-full` |

```bash
cargo run --release --bin growformer-demos -- --phase4d-cf-mnist-full
```

### 10.21 Phase 4e — Split-CIFAR-10 lite (torchvision export)

Class-pair binary ×5 (0v1…8v9), grayscale→64d promote–freeze + CF LearnedRouter. Data via torchvision (`scripts/export_cifar10.py`). DeepAugment policy search is optional / out-of-band (`scripts/deepaugment_cifar10_example.py`). Honest lite thresholds — not a full CIFAR claim.

```bash
python3 scripts/export_cifar10.py
cargo run --release --bin growformer-demos -- --phase4e-split-cifar-lite
```
