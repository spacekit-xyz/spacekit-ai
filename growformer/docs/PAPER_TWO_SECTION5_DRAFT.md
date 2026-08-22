# Paper-two §5–6 draft — Authentication gates for generalization claims

**Status:** Draft for insertion into whitepaper / paper-two. §5 (routing wall) numbers from `CMI_MEASUREMENT_SPEC.md` (2026-06-05); §6 (authentication gates / encoder & grounding) numbers from `GROUNDING_LOOP_SPEC.md` §§14–15.3, reproducible from deterministic verdict artifacts in `certify_artifacts/`. Whitepaper body not yet patched — paste when ready. §6 is the methods spine; §5.4 is its routing-side instance.

---

## Framing: one auditing discipline, two instantiations

A recurring pattern in modular ML systems — routing, grounding, distillation — is that standard held-out metrics (accuracy, AUROC, in-distribution coverage) can rise monotonically while the system learns nothing beyond surface statistics. The cause is always the same: the metric conflates surface recall with semantic generalization, and the eval set is constructed in a way that makes them indistinguishable. The result is a class of headline numbers — we call them **mirages** — that are correct measurements of the wrong quantity.

This paper contributes an auditing discipline that catches mirages *before* they become claims, demonstrated on the authors' own systems. The discipline has a single principle — **every generalization claim must survive a gate that removes the route by which a surface-only model could have produced the same number** — and two concrete instantiations, each adapted to its problem domain:

- **§5 (routing)** applies the principle to mutual-information estimation in per-specialist competence routing: a permutation null, repeated cross-validation, and an n-sweep that reveals lucky-coverage artifacts. It buries two headline numbers (81.3% router accuracy, 0.251-bit spiral MI) and establishes that per-specialist correctness information is bounded at ~0.1 bit.
- **§6 (encoders and grounding)** applies the same principle to text encoders and semantic grounding: a family of six authentication gates — feature-disjoint held-out, shuffle floor, provenance firewall, lexical positive control, strict-disjoint promotion, and a pre-flight eval-constructibility scan — that together decide whether a claimed accuracy reflects semantics or lexical overlap. It buries three additional headline numbers (20.7% supervised projection, a flagship 100% encoder, and the eval that produced it) and reaches a terminal result that standard metrics have no vocabulary for: that a generalization claim is *unresolvable in principle* because the data class a certifiable eval would require is structurally empty.

The five mirages are not five separate failures; they are one finding stated five ways. Across routing, grounding, and the language encoder, every probe of "does this system generalize semantically" returned negative or unmeasurable, and the encoder was repeatedly the lexical component. The contribution is not the negative result per se but the authentication procedure that produced it — and that most pipelines shipping equivalent numbers do not apply.

---

## 5. Information-Theoretic Bounds on Per-Specialist Forward-Signal Routing

The routing experiments in §4.3.1 established that three feature families fail on Task E with similar region-agreement floors (~55–57%). This section asks *why*: is the failure optimization (a better head would work) or representation (no head of this class can work)? We answer using conditional mutual information (CMI), with pre-registered falsification and estimator authentication applied before any mechanistic claim is written.

### 5.1 Estimation strategy

For frozen specialist \(k\), let \(A_k\) denote penultimate activations and \(Y_k = f_k(x)\) the scalar output. Inference is deterministic (frozen weights, no dropout), so \(Y_k = g(A_k)\) for a fixed readout. This yields the exact identity
\[
I(R;\, A_k \mid Y_k) = I(R;\, A_k) - I(R;\, Y_k),
\]
so the conditional quantity reduces to comparing two held-out classifier MI lower bounds rather than estimating a 16-dimensional conditional density. We report MI in bits as \(\hat H(R) - \widehat{CE}(R \mid S)\), with repeated stratified cross-validation for the parametric backend and a permutation null (B=500, shuffle \(R\) against \(A\)) to separate signal from estimator manufacture.

Routing requires each specialist's *own correctness* \(C_k = \mathbb{1}[\text{specialist } k \text{ matches composite label}]\), not region per se. We therefore report \(\Delta\mathrm{CMI}^C_k = I(C_k; A_k) - I(C_k; Y_k)\) as the load-bearing quantity — the information-theoretic form of the confident-wrong probe from phase 3f.

### 5.2 The correctness barrier (load-bearing)

Across 20 Task E seeds (42–61), 370 held-out points per seed:

| Quantity | Mean ± std |
| --- | --- |
| \(\Delta\mathrm{CMI}^C_{\mathrm{spiral}}\) | **0.097 ± 0.088 bit** |
| \(\Delta\mathrm{CMI}^C_{\mathrm{circles}}\) | **0.128 ± 0.107 bit** |

Activations add only ~0.1 bit about each specialist's own correctness beyond its scalar output. This is the quantity competence routing needed and did not find: the confident-wrong probe (mean \(c_k \approx 0.60\) on confidently-wrong points at \(n_{\mathrm{router}}=300\)) is the operational face of the same bound — a head trained on correctness cannot distinguish confident-right from confident-wrong when the forward signal carries only ~0.1 bit of incremental correctness information.

**Claim (armored):** Per-specialist competence routing failed because correctness is not present in the forward signal at useful strength — not because the competence heads were under-trained. The n-sweep inversion from phase 3f (accuracy and region agreement both *decline* as router calibration grows from 30 to 300 points) is consistent with this: more correctly-labeled data reveals asymptotic behavior near the scalar floor rather than unlocking hidden structure.

### 5.3 Region factorization (diagnostic — explains failure shape, not the thesis)

Region \(R = \mathbb{1}[r < 0.4]\) is fully present in the input (\(H(R) \approx 1.0\) bit, balanced) and recoverable from the cross-specialist joint (\(I(R;\, Y_1, Y_2) \approx 0.60\) bit). Per-specialist region information factorizes asymmetrically:

| Specialist | \(I(R;\, Y_k)\) | \(\Delta\mathrm{CMI}_k = I(R;A_k)-I(R;Y_k)\) |
| --- | --- | --- |
| Spiral | ≈ 0.004 bit (region-blind scalar) | First pass: 0.251 ± 0.098 — **artifact** (see §5.4) |
| Circles | 0.564 ± 0.153 bit | 0.068 ± 0.166 (≈ 0 — region in scalar) |

Circles *exports* region to its output; a scalar router on \(f_{\mathrm{circles}}\) already carries the switching bit. Spiral's scalar is region-blind — the binding constraint that forces always-spiral collapse (spiral cannot recuse itself in the outer region) even when circles could in principle be selected.

This asymmetry explains *which* degenerate routing pattern appears (constant spiral, not constant circles) without carrying the main impossibility claim. The main claim is §5.2.

### 5.4 Estimator authentication: two mirages buried

Two plausible positive numbers in this study were estimator artifacts exposed only by authentication:

1. **Expert-output router, 81.3% composite accuracy** — region agreement 55%, 14/20 seeds degenerate (§4.3.1).
2. **Spiral region MI, \(\Delta\mathrm{CMI}_{\mathrm{spiral}} = 0.251\)** — backend disagreement equal to effect size; single 60/40 split.

Spiral-resolve applied permutation null (B=500), repeated 10×5-fold CV, and a linear probe orthogonal to the output head (mean angle 90.1°):

| Instrument | \(I(R;\, A_{\mathrm{spiral}})\) pooled | Null-audited? |
| --- | --- | --- |
| Parametric MLP | 0.000 ± 0.000 (perm **p=0%**) | Yes |
| Linear probe | 0.000 ± 0.000 | Yes |
| PCA-kNN (m=2,3,5) | ~0.49 / 0.50 / 0.49 | **No** — single split, no permutation |

**Verdict on spiral activations:** Two null-audited instruments place \(I(R;\, A_{\mathrm{spiral}})\) at zero. The first-pass 0.251 was the MLP's manufactured floor — observed MI sat *inside* the permutation null at p=0%, meaning shuffled labels produced comparable apparent MI. The linear probe at 90.1° to the output head found nothing, ruling out a discarded region direction in the readout. The output-bottleneck hypothesis is specifically dead, not merely unsupported.

A third estimator (PCA-reduced k-NN) reports ~0.5 bit and **disagrees**. We do not treat this as outvoting the null-audited instruments; we treat it as **unresolved** — the same unaudited-estimator pattern that produced the 0.251 headline. Closing it would require a PCA-kNN permutation null (same B=500 protocol); we expect it to collapse as the MLP did, but have not run it. The conservative claim is: *below resolution*, resting on the two instruments that carry nulls, with the k-NN disagreement flagged rather than papered over.

This is the methodological contribution in miniature: a repeatable procedure that catches under-audited mirages in this problem class, applied twice to our own results before they became prose.

### 5.5 Synthesis

Three independent routing feature families failed empirically (raw outputs, decisiveness, penultimate activations with correctness labels). Information theory sharpens the failure:

- **Present:** region is in the cross-specialist joint (\(I(R;\, Y_1,Y_2) \approx 0.6\) bit).
- **Absent at useful strength for routing:** own-correctness beyond the scalar (\(\Delta\mathrm{CMI}^C \approx 0.1\) bit both specialists).
- **Asymmetric at the region level:** circles exports region; spiral's scalar is blind; whether spiral's activations recover region is below the resolution of null-audited estimators.

Per-specialist forward-signal routing — conditioning on one specialist's computation — is therefore information-bounded on the quantity routing actually needs (correctness), independent of whether a region signal exists somewhere in the joint. The switching variable is present yet inaccessible to this whole family of routers, not because optimization failed but because ~0.1 bit of correctness information cannot support reliable dispatch on a switched task.

Cross-specialist structure (the only family not yet ruled out) remains motivated by the joint MI but is explicitly out of scope here; any such mechanism must pass the same gate — n-sweep first, confident-wrong probe mandatory, region agreement at certification budget — with the prior warning that boundary-correlated disagreement will look successful at low \(n\) for the same lucky-coverage reason competence routing did.

---

## 6. Authentication Gates for Generalization Claims (Encoder & Grounding)

§5.4 buried two estimator mirages in the routing study by applying authentication before prose. This section states that move as the paper's primary contribution and shows it is not specific to mutual-information estimation: the same discipline, instantiated as a different gate family, catches *lexical recall masquerading as semantic generalization* in encoders and grounding routers. We demonstrate it on a series of our own systems — including a flagship language encoder whose model card reports 100% intent accuracy — and end on a result the gates can express but standard evaluation cannot: that a generalization claim is *unresolvable in principle on the available data*. Numbers are from `GROUNDING_LOOP_SPEC.md` (§§14–15.3) and are reproducible from deterministic verdict artifacts (`certify_artifacts/`).

### 6.1 The problem class and the thesis

A model that routes/grounds text by surface features can score arbitrarily high on a held-out set that shares those features with training, while carrying no concept-level generalization. Standard metrics are *silent* on the distinction: accuracy, AUROC, and in-distribution coverage all rise with lexical overlap, so a lexical separator and a semantic generalizer are observationally identical under them. The claim of this section:

**Thesis.** *Whether an apparent generalization number reflects semantics or surface recall is decidable by a small family of authentication gates, each of which removes one route by which a lexical model can manufacture the number. A claim that passes none-removed is not evidence of generalization; a claim that survives all gates is.*

### 6.2 The gate family

Each gate closes a specific manufacturing route. The unit of analysis is a labeled phrase set split into *propose* (training) and *certify* (held-out), with the encoder under test reduced to a routing function over concept centroids.

1. **Feature-disjoint held-out, at the encoder's own feature granularity.** Overlap is measured over the exact features the encoder consumes — words ∪ bigrams ∪ char-trigrams (`wbc`) for a hashing-style encoder — not over whole words. The load-bearing quantity is `disjoint_gen_a`: routing accuracy on certify phrases whose features are *absent* from their own concept's training yet *present* on some other concept (the "seen-elsewhere" sub-bin), so a hit reflects routing, not novel-token guessing. *Closes: surface-token leakage between train and test.*
2. **Shuffle floor.** The subject encoder is retrained on label-permuted data (B≥200); `semantic_floor_95` is the 95th percentile of its disjoint accuracy. The reported quantity is `disjoint_semantic_lift = disjoint_gen_a − semantic_floor_95`, certified only if its Wilson CI excludes 0. *Closes: a high score on a task so easy that permuted labels also score high.*
3. **Provenance / augmentation firewall.** Certify phrases must be `real_traffic`; no certify phrase's id may appear in the lineage of any training `augmented` phrase. *Closes: self-certification, where the eval is distilled/augmented from the training distribution.*
4. **Positive control (lexical lower bound must collapse).** A purely lexical baseline (character-n-gram nearest-centroid, "CATA") is run on the *same* certify set; it must collapse to the shuffle floor at overlap-0. If the lexical baseline scores high, the eval is lexically separable — an easy task — and the run is `INVALID`. *Closes: an eval that looks disjoint but is still solvable by surface alone.*
5. **Promotable-only-at-strict-disjoint.** When the strict (`wbc`) disjoint bin is empty on dense data, the gate may resolve lift at a coarser granularity (`wb`/`w`), but a pass so earned is `PASS_PROVISIONAL` and **not promotable**; only a strict `wbc` pass is. *Closes: laundering surface overlap by relaxing the disjointness definition until the bin populates.*
6. **Pre-flight eval acceptance + constructibility scan.** Before a certification run is spent, the candidate eval is checked for ≥ `n` feature-disjoint seen-elsewhere examples and lexical-baseline collapse; and the entire corpus is scanned leave-one-out to ask whether such examples *exist at all*. *Closes: spending effort certifying on an eval that structurally cannot carry the signal — and detects when no such eval is constructible.*

**Figure 1. The gate family: what each gate closes and what passes it.**

| Gate | Manufacturing route closed | What a surface-only model exploits | Passes iff |
| --- | --- | --- | --- |
| 1. Feature-disjoint held-out | surface-token leakage train→test | shared words/bigrams/trigrams between train and certify phrases for the same concept | `disjoint_gen_a` (accuracy on overlap-0, seen-elsewhere phrases) is measurable and positive |
| 2. Shuffle floor | trivially easy task | a task where even permuted labels yield high disjoint accuracy | `disjoint_semantic_lift` (gen\_a − floor\_95) > 0, Wilson CI excludes 0 |
| 3. Provenance firewall | self-certification via augmentation | eval drawn from or augmented by the same distribution as training | certify ⊆ `real_traffic`; no lineage crossing with training `augmented` phrases |
| 4. Lexical positive control | lexically separable eval | an eval whose disjoint bin is still solvable by surface n-grams alone | CATA (char-n-gram nearest-centroid) collapses to shuffle floor at overlap-0 |
| 5. Strict-disjoint promotion | relaxed disjointness laundering overlap | populating the disjoint bin by falling back to a coarser granularity (`w`/`wb`) where leakage persists | `PASS` only at `wbc` (union); coarser → `PASS_PROVISIONAL` (not promotable) |
| 6. Pre-flight acceptance + constructibility | certifying on an unconstructible eval | spending effort on an eval that structurally cannot carry a generalization signal | ≥ 8 feature-disjoint seen-elsewhere phrases exist; CATA collapses on them |

*Each gate is independently necessary: a number that survives gates 1–5 but fails gate 6 is unmeasurable, not wrong; a number that survives gates 2–6 but fails gate 1 is indistinguishable from memorization. The gates compose into a deterministic state machine that emits a single verdict per `(encoder, data_hash, seed)` tuple.*

Verdicts are produced by a deterministic state machine over these quantities — `{INVALID, BELOW_RESOLUTION, FAIL_MEMORIZATION, FAIL_COLLISION, PASS_PROVISIONAL, PASS}` — keyed by `(encoder, data_hash, seed)` and written to append-only artifacts, so a claimed improvement is distinguishable from an estimator mirage across encoder versions, not just at one snapshot. Crucially, `INVALID` and `BELOW_RESOLUTION` mean *not measured*, never *measured and bad*: the instrument refuses to read a number it cannot authenticate.

### 6.3 Demonstrations on our own systems (mirages buried)

The gates were applied to our own headline numbers. Five did not survive contact:

| # | System / headline | What the metric claimed | What a gate found | Gate |
| --- | --- | --- | --- | --- |
| 1 | Expert-output router, **81.3%** composite acc | competence routing | 55% region agreement, 14/20 seeds degenerate | n-sweep / confident-wrong (§5.4) |
| 2 | Spiral region MI, **0.251 bit** | recoverable region in activations | observed MI inside permutation null (p=0%); two null-audited probes at 0 | permutation null (§5.4) |
| 3 | Supervised projection, **20.7%** pooled | semantic grounding | disjoint lift at/under shuffle floor ⇒ `BELOW_RESOLUTION` | feature-disjoint + shuffle floor (§14) |
| 4 | Language encoder (GLE), **100%** intent acc, **AUROC 1.000** | semantic routing | a 2-way cosine-to-prototype lexical separator (see 6.4) | full gate family (§15.1) |
| 5 | GLE 100%, *certified in its own domain* | in-domain generalization | eval contains **zero** feature-disjoint examples ⇒ unresolvable (see 6.5) | acceptance + constructibility scan (§15.2–15.3) |

### 6.4 The composed mirage (case study: the 100%)

Mirage #4 is worth decomposing because it is the strongest headline and the most instructive failure. Tracing the number to its source, it decomposes into four lexical-overlap signatures stacked:

1. the distillation **teacher is a hashing encoder** — the student was distilled to mimic a *lexical* geometry;
2. distilled into a small student (so the student's "semantics" is a learned compression of lexical features);
3. scored on a **2-way** cosine-to-prototype separation between two maximally-distinct buckets (support vs. coding);
4. on a held-out split with **zero feature-disjoint examples**.

Each layer is independently the easy-task signature; the model card presented their composition as semantic routing with AUROC 1.000. Run zero-shot on a genuinely disjoint domain (a pet-companion grounding graph), the same encoder routes at **1.1%** — a *correct* measurement of an irrelevant question (a support encoder on out-of-domain traffic), and explicitly not the test of the headline. Run in its own domain through the gate, it returns `INVALID` — not because it scored badly, but because the eval cannot separate memorization from generalization.

### 6.5 The terminal result: an unconstructible eval (what the gates can express that metrics cannot)

The distinctive output of this instrument is a verdict standard evaluation has no vocabulary for: *this eval set is structurally incapable of distinguishing generalization from memorization.* For the GLE's home domain, every held-out phrase shares a feature with its own class's training at every granularity, so the disjoint bin is empty and **any** score on it — 100% or 15% — is silent on generalization.

A leave-one-out scan of the entire home corpus (≈1,485 phrases) settles whether a certifiable eval could be *constructed* at all. After removing two artifacts the gates themselves flag — `train==eval` self-overlap and singleton-class degeneracy — the count of feature-disjoint, seen-elsewhere phrases in eval-eligible classes is **0** in the support/coding subset and **1–2** across the full corpus, against a resolution threshold of 8.

**Claim (armored).** *The in-domain generalization claim is unresolvable in principle, not for lack of collection.* The data class it would require — same-concept, surface-disjoint phrasings — is structurally near-empty in this domain, consistent with a hashing-distilled encoder whose own notion of "same concept" is lexical: surface-disjoint-but-concept-preserving examples are scarce *by construction of the encoder*. No amount of data collection certifies the 100%.

### 6.6 Synthesis

Across routing (§5), grounding (§14), and the language encoder itself (§15), every probe of "does this system generalize semantically" returned negative or unmeasurable, and the encoder was repeatedly the lexical component. This is one finding stated several ways: **the language stack, as built, operates on lexical/surface structure throughout; the semantic generalization the metrics imply is not present anywhere it could be certified, including in the encoder's own domain.** The contribution is not the negative result per se but the *authentication procedure that produced it* — a gate family that caught five mirages on the authors' own systems, including a flagship 100%, and that most routing/distillation pipelines do not apply. We release it as a deterministic, reproducible contract (`--certify-encoder`, `--verify-disjoint-eval`, `--scan-disjoint-corpus`) so that others shipping the equivalent of our 100% can find out before, not after.

---

## Suggested figure caption (n-sweep inversion)

*Figure X.* Router calibration budget vs. held-out composite accuracy and region agreement (Task E, competence routing, 20 seeds). Both metrics decline as \(n_{\mathrm{router}}\) increases from 30 to 300 — the opposite of the usual sample-size curve — indicating that low-\(n\) accuracy is anti-correlated with routing truth on this task. The confident-wrong probe (not shown) was already failing at \(n=30\) (\(c_k \approx 0.51\)), exposing the headline before authentication.

---

## Abstract (consolidated)

> We present a family of authentication gates that decide whether an apparent generalization number reflects semantics or surface recall, and demonstrate them on our own systems. In routing, a permutation null and n-sweep inversion bury two headline numbers (81.3% composite accuracy, 0.251-bit mutual information) and establish that per-specialist correctness information is bounded at ~0.1 bit — an information-theoretic impossibility. In encoder and grounding evaluation, six gates — feature-disjoint held-out at the encoder's own feature granularity, a label-shuffle floor, a provenance/augmentation firewall, a lexical positive control, promotion only at strict disjointness, and a pre-flight eval-constructibility scan — bury three additional headline numbers, including a language encoder reporting 100% intent accuracy and AUROC 1.000 that the gates identify as a 2-way lexical separator on an eval with zero feature-disjoint examples. The procedure further returns a verdict standard evaluation cannot express: that the in-domain generalization claim is unresolvable *in principle*, because the domain's real traffic contains essentially no feature-disjoint concept-preserving examples. Across routing, grounding, and the encoder itself, every probe of "does this system generalize semantically" returned negative or unmeasurable; the five mirages are one finding stated five ways. We release the protocol as a deterministic, reproducible contract.

---

## Appendix A. How to adopt the gate protocol

The authentication gates are released as three deterministic CLI entrypoints. All verdicts are reproducible given `(encoder, data_hash, seed)` and are written to append-only artifact files (`certify_artifacts/verdict_<encoder>_<datahash>_<seed>.json`).

### A.1 Certify an encoder (full gate pipeline)

```bash
# Run the certifier on any encoder against a companion's grounding graph.
# encoder_id ∈ {supervised, cata, gle, gle_base, gle_m5_routing_tuned, ...}
cargo run --release --bin growformer-demos -- --certify-encoder <encoder_id> [companion_dir] [seed]

# Certify the GLE on its own support/coding domain (both 2-way and many-way).
cargo run --release --bin growformer-demos -- --certify-gle-indomain

# Re-read / compare any verdict artifact.
cargo run --release --bin growformer-demos -- --certify-verdict <path/to/verdict.json>
```

**What it does.** Runs all six gates in sequence: trains the encoder on propose-split phrases, evaluates feature-disjoint routing accuracy on certify-split phrases at `wbc`/`wb`/`w` granularity, runs B≥200 shuffle controls to establish the floor, runs lexical CATA as a positive control, checks provenance/augmentation firewall, and emits a single verdict artifact. The verdict state machine is deterministic and pure-functional (`decide_encoder_verdict` in `grounding_loop.rs`), testable independently of the routing machinery.

**Verdict meanings:**
- `PASS` — disjoint lift clear of floor at strict `wbc`; promotable.
- `PASS_PROVISIONAL` — lift cleared at a coarser level; real but leakier, *not* promotable.
- `FAIL_MEMORIZATION` — lift absent or CI includes 0; surface recall.
- `FAIL_COLLISION` — positive lift but graph edits introduced misroutes.
- `BELOW_RESOLUTION` — disjoint bin too small/noisy to read; need more traffic.
- `INVALID` — a validity gate failed (positive control didn't collapse, firewall tripped, or data insufficient). Check `invalid_reason` in the artifact.

### A.2 Pre-flight a candidate eval set

```bash
# Gate a candidate disjoint eval BEFORE spending a certification run on it.
cargo run --release --bin growformer-demos -- --verify-disjoint-eval <train.jsonl> <eval.jsonl> [action_target|semantic_intent]
```

**What it does.** Two encoder-free gates: (1) counts feature-disjoint seen-elsewhere phrases in the eval at `wbc` (needs ≥8); (2) runs CATA on the eval and checks it collapses at overlap-0. An eval that fails either gate cannot carry a generalization signal and should not be spent on a certification run.

### A.3 Check whether a certifiable eval is constructible at all

```bash
# Leave-one-out full-corpus scan: does this domain's real traffic contain feature-disjoint examples?
cargo run --release --bin growformer-demos -- --scan-disjoint-corpus [action_target|semantic_intent]
```

**What it does.** Scans all home-domain training data leave-one-out (avoiding self-overlap) and counts how many phrases in eval-eligible classes (≥4 phrases) are feature-disjoint from the rest of their class at `wbc`. If the count is < 8 across all traffic, the in-domain eval is structurally non-constructible — the generalization claim is unresolvable in principle. Reports `CONSTRUCTIBLE` vs. `STRUCTURALLY EMPTY`.

### Adapting to your own system

The gates are not specific to Growformer. To apply them to any encoder or router:

1. **Define the encoder's feature granularity.** If your encoder uses subword tokens, measure overlap at the subword level; if it uses character n-grams, use those. The principle is: overlap must be measured at the granularity the encoder *actually consumes*, not at a coarser level that hides leakage.
2. **Split propose/certify by provenance.** Certify phrases must be real traffic, not synthetic or augmented. If you can't provenance-tag your data, you can't close the self-certification route and the firewall gate is unenforceable.
3. **Run a lexical baseline on the same eval.** If a character-n-gram nearest-centroid (or your encoder's input tokenizer used as a bag-of-features) scores well on your disjoint bin, the eval is measuring an easy task, not semantics.
4. **Report the verdict, not the pooled accuracy.** A pooled number on a non-disjoint eval has the same evidential standing as no number at all. The gates exist to say which condition holds.
