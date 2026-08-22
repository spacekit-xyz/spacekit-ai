# The Certification Arc — methodology, and the results that test it

**Status:** internal consolidation, 2026-06-25. Living document; links to the specs and result
artifacts it summarizes rather than restating them in full.

## 0. Thesis

The contribution of this project is not any single model. It is a **discipline for deciding when a
machine-learning claim has been earned**, and a set of results produced by holding that discipline even
when it cost us the headline we wanted.

The discipline, in one sentence: **define the gate and the verdict before you run; certify only on data
and quantities the system did not produce or train on; and treat "not resolvable yet" as a first-class
outcome rather than rounding it to a pass.**

Everything below is organized to make that thesis falsifiable. §1 states the discipline as a set of
named rules, each paired with the specific failure it prevents. §2 walks the results in sequence —
a negative, a wall, a breach, a qualified-negative, an honest zero, and a pre-registered bar that refuses
a flattering comparison — and shows each one as *the discipline doing its job*, including the times it
did its job by refusing to certify something we believed. §3 consolidates what is and is not claimed. §4
records the current state and the honest next move.

The pattern to notice across §2: **every time a number looked good, the discipline asked one more
question, and the cases that survived that question are the only ones reported as passes.** The cases
that didn't survive are reported too — as negatives, as `INVALID`, as `BELOW_RESOLUTION`, as `0/47`.
That symmetry is the evidence the gate is real.

---

## 1. The discipline (the method is the result)

Each rule is stated with **the mechanism** and **the failure it prevents**. Cross-references point to
where the rule is implemented and exercised.

### 1.1 Certifier-first / pre-registration

**Mechanism.** Before a run, write down the metric, the threshold, and the decision table — what each
outcome will mean and what action it triggers. For the decisive experiments, pre-register the *tests*,
not just the metric (e.g. the n-sweep and the confident-wrong probe were named as decisive before the
router was tuned).

**Failure prevented.** Post-hoc goalpost-moving: choosing the metric or threshold after seeing the
numbers so that whatever happened reads as success. A pre-registered decision table converts a result
into a verdict instead of a narrative.

> Implemented: `COMPETENCE_ROUTING_SPEC.md` §5–6 (decision table filled *after* the run, thresholds
> frozen pre-eval); the cone-router pre-registration in §10.

### 1.2 Provenance firewall — you cannot certify on data you produced

**Mechanism.** Data carries provenance (`RealTraffic` vs authored). The certify split natively accepts
`RealTraffic`; authored data requires an explicit flag and is barred from being the certification
surface. The principle generalizes beyond data to *quantities*: see 1.4.

**Failure prevented.** Self-certification — scoring the system on a distribution the same system (or its
authors) generated, which is in-lexicon by construction and resolves nothing. This is not a hypothetical:
§2.5 shows 20 self-generated phrases bucketing **0-disjoint by construction**, the trap as a physical
property of capture.

> Implemented: `PhraseProvenance`, augmentation firewall; `GROUNDING_LOOP_SPEC.md` §18.2–18.3.

### 1.3 Feature-disjoint generalization, and constructibility

**Mechanism.** A generalization claim is read only on a **feature-disjoint held-out bin** — phrases
that share no surface feature (word / bigram / char-trigram, the `wbc` ladder) with their own class's
training. If no such bin can be constructed, the eval is `INVALID` and **no pooled number is read**.

**Failure prevented.** Lexical memorization masquerading as semantic generalization — "100% accuracy" on
a test whose held-out phrases share tokens with training. Sometimes the honest finding is *about the
eval, not the model*: the home-domain set cannot separate memorization from generalization at all.

> Implemented: disjointness scan / `audit_disjoint_eval`; `GROUNDING_LOOP_SPEC.md` §14–16.

### 1.4 Decontamination — never certify on a quantity the loss optimized

**Mechanism.** A certifier must be independent of the training objective. If the loss shapes a quantity,
that quantity is removed from the success criteria (or the shaping is removed from the loss). The result
must stand on certifiers the loss never touched.

**Failure prevented.** Circular certification — measuring "how well we optimized the thing we optimized"
and reporting it as evidence the mechanism works. See §2.4: `margin↔r` was trained-on via margin shaping;
it was stripped from the loss and demoted to observational.

> Implemented: `COMPETENCE_ROUTING_SPEC.md` §10 decontamination note; `cone_router.rs` (margin shaping
> removed from every head's loss).

### 1.5 Resolution-before-verdict — "not yet" is a real answer

**Mechanism.** A bin must be **resolvable** before any verdict is read: `n ≥ DISJOINT_MIN_N (47)` and
Wilson CI width `≤ DISJOINT_MAX_CI_WIDTH (0.30)`. An underpowered bin returns `BELOW_RESOLUTION`, whose
prescribed action is *grow the bin* — never *loosen the gate*.

**Failure prevented.** Reading a verdict off an underpowered sample because the point estimate is
favorable. §2.3 shows a +0.176 lift at n=34 correctly held as `BELOW_RESOLUTION` (CI width 0.314 > 0.30)
until the bin reached n=47 — a pass we wanted, refused until it was earned.

> Implemented: `GROUNDING_LOOP_SPEC.md` §17 (n=10→34→47), §18.4–18.5.

### 1.6 Certify the shipped artifact, not a proxy

**Mechanism.** Property-based certification of generative output runs against the **exact
`growformer_bg.wasm` the browser bundles**, in AgentHub load order — because the failure mode (garble /
`MASK`-leak) was wasm-specific and a native-only test would pass a broken browser build.

**Failure prevented.** Certifying a healthy proxy while shipping a broken artifact. Generation is
sampled, so properties (garble, forbidden, voice, required-signal) are gated, not exact strings.

> Implemented: `scripts/certify_chat.mjs`; `GROUNDING_LOOP_SPEC.md` §19.

---

## 2. The arc as evidence

Five results, in order, plus a sixth that is a *pre-registered bar* rather than a finished verdict. Two
are passes, one is a negative, one is an `INVALID`/wall, one is an honest zero; the sixth (§2.6) is the
discipline applied *before* a run, refusing a flattering comparison. The discipline is what makes them
trustworthy, and the through-line is that **each favorable number had to survive one more pre-registered
question.**

### 2.1 A clean negative — unsupervised forward-feature routing collapses

The competence router routed by per-specialist forward signals (confidence, blend weights) — unsupervised
proxies. At the certification budget (n_router=300) it landed at **78.9%±6.1% accuracy, 56.7% region
agreement, 10/20 degenerate (constant-specialist collapse), confident-wrong reliance 0.60, margin↔r =
−0.31** → **rejected**. Notably it *looked* good at n=30 (90% acc, 0 degenerate) and decayed as data
grew (90% → 86.5% @n=100 → 78.9% @n=300; region 56.7%) — the fixed-low-n mirage that 1.5 exists to catch.

**Discipline shown:** pre-registered decision table (1.1) + a confident-wrong probe that measured the
*mechanism*, not just accuracy. Reported as a publishable negative, not buried.

> `COMPETENCE_ROUTING_SPEC.md` §6.

### 2.2 A wall, correctly identified — the encoder eval cannot be made disjoint

The prior encoder (GLE) reported **100% intent accuracy**. Under 1.3 that number dissolved: it is a
**2-way, lexically-entangled** measurement, and the home-domain held-out sets have **no feature-disjoint
examples at any granularity** — every held-out phrase shares a feature with its own class's training. The
gate returned **`INVALID`** (empty disjoint bin) rather than reading the pooled number. As an encoder for
a *new* companion's graph, the GLE routes at pooled ~1.1%.

**Discipline shown:** the rarest move — a verdict *about the eval*, refusing to certify a 100% that was
real but meaningless. This is what told us the open problem was the **encoder**, not the router.

> `GROUNDING_LOOP_SPEC.md` §14–16.

### 2.3 A breach of the wall — mpnet capability PASS, earned at resolution

`all-mpnet-base-v2` was run against an authored, CATA-validated, feature-disjoint bench. The disjoint bin
was **grown n=10 → 34 → 47**. At n=34 the lift was already positive (**+0.176**) but the verdict was held
at **`BELOW_RESOLUTION`** because the CI width (0.314) exceeded the 0.30 gate. The bin was grown — *not*
the threshold loosened — until **n=47 cleared both the resolution gate and the lift-CI-excludes-zero
gate**. Verdict: **capability PASS**.

**Discipline shown:** 1.5 in its purest form — a favorable point estimate refused until it was powered.
**Scope guard (load-bearing):** this is a *capability* PASS ("mpnet can generalize on a fair disjoint
test"), **not** a deployment certification ("mpnet routes live traffic correctly"). That gap is §2.5.

> `GROUNDING_LOOP_SPEC.md` §17.

### 2.4 A qualified negative — the cone router resists collapse under supervision

Returning to the §2.1 task with a different hypothesis: a region-supervised, boundary-widening router
(oracle-free *inference*) whose effective "cognitive cone" expands near the decision boundary. The
pre-registered decisive tests (1.1):

- **n-sweep:** **0/100 degenerate** across 5 budgets × 20 seeds; cone > VirtualGroup at every n; accuracy
  and region agreement *rise* with coverage (92.9→94.6%, 84.5→88.9%) — the **inverse** of §2.1's decay.
- **Confident-wrong probe:** reliance **0.18** on confidently-wrong specialists, vs §2.1's **0.60** — the
  clean kill of the "decisiveness with extra steps" hypothesis, on held-out.
- **Decontamination (1.4):** `margin↔r` was trained-on via margin shaping; **removed from the loss** and
  demoted to observational (it held at 0.85 uncontaminated, but is excluded from the verdict). Accuracy
  held after removal — proving the lift never came from the contaminated term.

**Honest attribution (the sharp edge).** The route head is **supervised on the region label at train
time**; this is permitted under the oracle-free-*inference* contract, not an oracle-free-*training* one.
So the **accuracy is attributable to region supervision over learnable features**; the **cone architecture
earns the anti-collapse robustness**, not the accuracy. The result therefore *qualifies* the §2.1 negative
on the supervised axis ("collapse was not inevitable given supervision + the right architecture") without
overturning its unsupervised core. The one stretch miss — region agreement 85.3% (clears the ≥80% gate,
misses the >90% goal) — is the **confirmed ~0.1-bit CMI feature ceiling viewed twice**: the router does as
well as the feature information permits and no better, which is the signature of legitimate supervision.

> `COMPETENCE_ROUTING_SPEC.md` §10; `phase3g_cone_results.txt`.

### 2.5 An honest zero — deployment is gated on real disjoint traffic

The capability PASS (§2.3) becomes a deployment certification only on a **`resolvable` `RealTraffic`
disjoint bin**. First live drain of the production capture pipeline (storage node → drain → triage):
**pipeline green, but the disjoint bin is 0 / ~47.** The 20 captured records are self-generated test
chatter and bucket **0-disjoint by construction** — the provenance firewall (1.2) reappearing as a
physical property of capture: surface diversity is exactly what self-generated traffic lacks. **No amount
of internal traffic fills the bin; only diverse real users do.** Thresholds recorded **held**.

**Discipline shown:** writing down the zero. The blocker is named precisely as *data accumulation* — not
code, not the encoder, not the gate — which forecloses the conflation that has tried to creep in at every
turn, and forecloses the two shortcuts (synthetic fill; threshold-nudge).

> `GROUNDING_LOOP_SPEC.md` §18.9.

### 2.6 A refused flattering comparison — the semantic graph router's bar

The most recent capability question: does a knowledge graph (nodes embedded by mpnet, edges encoding
relations) route better than the embedding alone? The discipline's move here was to **reject the
flattering comparison** ("graph beats lexical" — cleared by using mpnet at all, it tests the encoder not
the graph) and make the **gate** the edge-ablation: E1/E2 (edge-using) must beat **E0 (flat mpnet
nearest-neighbor)** with a CI excluding zero, *and* the win must localize to a router-blind relational
slice.

**E0 is measured and it PASSes** (reproduced fresh, deterministic seed 42, 200-shuffle null): flat
mpnet-NN on the n=47 authored disjoint bench scores **lift +0.213, CI-lo +0.080, wbc**, with the CATA
positive control collapsing (−0.041) to confirm eval validity. Two consequences, both pre-committed:

- **The edges are an optimization, not a necessity.** Because E0 already clears the bench, you already
  have a working semantic router; E1/E2 must beat a *passing* baseline. Generalization lives in the
  **embedding**, and the open question is only whether **structure** adds anything on top.
- **E1 ≈ E0 is the pre-registered expected outcome.** A good embedding already encodes most relational
  structure; edges can only earn their place where geometry and true relations disagree (`contrasts_with`
  — semantically-near-but-opposite, the one thing NN is structurally bad at). A sub-CI edge is read as
  decoration, not "the graph helps." The relational slice is defined **router-blind** (annotator
  agreement, frozen before scoring) so a win cannot be authored into existence.

**Discipline shown:** 1.1 + 1.4 applied *preemptively* — the load-bearing comparison identified and the
flattering one refused before E1/E2 are built, so a decorative graph cannot read as a win. Either verdict
is real: edges earn their keep (compositional-routing result) or semantic NN suffices (a result a prettier
graph would have buried). This is a **capability** question on the authored bench; it does not move §2.5.

> `SEMANTIC_GRAPH_ROUTER_SPEC.md`; E0 artifact `certify_artifacts/verdict_all-mpnet-base-v2_3c5eefa076a8f5ba_42.json`.

### 2.x (parallel) — the same discipline on the generative path

§1.6 applied to chat output: 30 prompts × 6 = **180 generations, 100% pass / 0 garble** on the healthy
wasm engine; under fragment-composition ablation the raw decoder collapses and the gate serves **6/180 as
in-voice fallback with 0 garble reaching the user**. (**DEFECT-G1**, mitigated: a pre-gate garbled
*response* was found in capture; verified the served wasm = the §19-certified build, so no live leak,
and added a phrase-malformation quarantine so garble can't reach the labeled bin regardless. See §18.9.)

> `GROUNDING_LOOP_SPEC.md` §19.

---

## 3. What is and is not claimed

| Claimed | Not claimed |
| --- | --- |
| mpnet **can** generalize on a fair, resolution-cleared, feature-disjoint bench (capability PASS) | mpnet **routes live production traffic correctly** (deployment cert — pending §2.5) |
| A region-supervised, boundary-widening router **resists constant-specialist collapse**, and the resistance strengthens with coverage (Task E) | The switch is recoverable **without** region labels; "forgetting is solved"; the §6 negative is reversed in general |
| The cone's **accuracy** comes from region supervision over informative features | The cone **architecture** produced the accuracy (it produced the anti-collapse) |
| Flat mpnet-NN (E0) is a **deployable semantic router** on capability evidence (passes the disjoint bench) | Graph **edges** add value: E1/E2 > E0 is **unrun**, and **expected ≈ E0** (§2.6) — structure is unproven, generalization lives in the embedding |
| The discipline produces trustworthy verdicts, including negatives, `INVALID`, and `BELOW_RESOLUTION` | Any result here clears the **production** bar, or the **encoder** and **routing** questions are the same question |

Two standing separations: the Task E routing result does **not** move the encoder question (solved
separately by mpnet) or the production question (gated on real disjoint traffic); and a capability PASS is
**not** a deployment certification.

---

## 4. Current state & honest next move

- **Pre-registration (2026-07-01):** [`PRE_REGISTRATION.md`](PRE_REGISTRATION.md) — three decoupled bets (CL substrate / Clifford LM / oscillator dynamics). Row **1b-v2** (9.74 bpt, 2026-07-02) closed Bet B's score-kernel question: dot wins; cone routing and promote-freeze (Bet A) are unchanged.
- **Components:** proven (encoder capability PASS §2.3; cone anti-collapse §2.4; E0 deployable semantic
  router §2.6).
- **Pipeline:** proven (live drain §2.5).
- **Critical-path number:** disjoint bin **0 / 47**, `resolvable = no`, thresholds **held**.
- **Fill rate:** sub-linear in total traffic, set by user diversity we don't control — weeks-to-months,
  not days. A slow count is *the finding*, not a fault.
- **Shadow-mode deployment:** now **fully specced and armed** (`GROUNDING_LOOP_SPEC.md` §21):
  offline-shadow-first, centroid/bench **parity green** (E0 reproduces its verdict bit-for-bit), harvest
  harness wired. Readiness check (§21.6): **code-ready, data-blocked** — the current 20 captures are
  self-generated/in-lexicon (0-disjoint), so no run was manufactured. Posture is *"deployed on capability
  evidence, deployment certification pending disjoint-bin resolution"* — **not** synthetic fill, **not** a
  threshold-nudge.
- **DEFECT-G1:** mitigated — live leak ruled out (served wasm = §19-certified build; the garbled
  record predates the gate by ~16 h), and the labeling front now quarantines malformed phrases.

**The next move that changes position.** The build track is now exhausted of forward motion: every
capability mechanism (encoder, cone, graph) is benched or armed, and each comes back "the embedding
already does the work or the gate kills it." The only remaining mover of the critical-path number is
**real diverse third-party traffic**, which is a background process (product/consent, not code) that
shadow mode is armed to harvest and target the moment it arrives. The parallel track that does not compete
with the data clock is **this document** — the consolidated result the proven components already justify.
Running another capability experiment, however clean, is increasingly a sophisticated way to wait.

### 4.1 One wall, not two (framing correction)

It is tempting to name "routing-at-scale" as a second open wall. It is not. Every authenticated routing
failure was at *trivial* count (K=2 on Task E; 119 grounding nodes) and **none** was about count — each
was about the routing signal being lexical (§2.2) or ~0.1-bit (§2.1). Once the representation is semantic
(§2.3), routing among frozen specialists is **geometric nearest-neighbor** — what vector indexes solve at
billion-scale, by construction, with no jointly-trained gate to destabilize. Routing-at-scale is the
*shadow* the representation wall casts, not an independent wall.

So there is **one** wall — labeled disjoint data — and it splits into two gates that must be
parallelized, not conflated:

- **Capability** (encoder, cone router, distillation, any candidate mechanism — e.g. the
  **semantic graph router**, pre-registered in `SEMANTIC_GRAPH_ROUTER_SPEC.md`, whose E0 baseline is now
  measured and passing, §2.6) needs only an authored/held-out disjoint bench → **un-gatable now** (expand
  the n=47 bench; firewalled LLM paraphrases per `GROUNDING_LOOP_SPEC.md` §20.2).
- **Deployment** is the *only* thing that needs real traffic → run it as a **background** process behind
  shadow-mode mpnet (concrete protocol `GROUNDING_LOOP_SPEC.md` §21; strategic framing §20.3).

The wall does not fall to a clever mechanism; it falls to refusing to let one slow resource (real
disjoint traffic) block the three things that don't depend on it: **research** (authored bench),
**shipping** (shadow mode), and **deployment-evidence collection** (harvest while serving). Full
protocol: `GROUNDING_LOOP_SPEC.md` §20.

Sequencing: let capture run passively in the background; make capability decisions now on the authored
bench; produce the documents the proven components already justify. This file is the top-level index of
that arc.

### 4.2 The downstream-architecture pattern (why new mechanisms aren't the wall)

Four architectural ideas have arrived during the data wait — **diffusion routing**, **knowledge graph**,
**staged/curriculum learning**, and **Rete / working memory**. They form a clear shape, and naming it is
more useful than evaluating each fresh:

> **Every one of them is a way to *structure or use* meaning once it is in the system. Not one addresses
> how meaning *gets in*. Getting meaning in — same-concept-different-surface, resolved on real disjoint
> traffic — is the wall.** The mind productively generates the downstream architecture because the
> upstream bottleneck is an off-keyboard waiting game and the architecture is the part that's thinkable.

Each has a *correct form*, and it is always the same shape: **the sub-symbolic layer (mpnet) does the
meaning; the symbolic structure does the composition, downstream.** "Knowledge graph" → *semantic* graph
(nodes embedded, edges over resolved concepts; §2.6). "Rete / working memory" → **Rete over semantic
facts**: mpnet grounds a paraphrase to a concept-node, that resolved symbol enters working memory, and a
Rete-like network does incremental cross-turn rule composition over the *clean, already-resolved* symbols.
Rete's matching primitive is exact-symbol — structurally the lexical primitive §2.2 proved collapses on
paraphrase — so Rete *in front of* concepts re-introduces the 119-node lexical failure; Rete *behind*
mpnet (over resolved concepts) is a legitimate stateful-reasoning layer. Same verdict for all four:

**The rule for the next idea (there will be a fifth).** (1) It is almost certainly downstream
architecture, not a wall-breaker — confirm by asking *does it change how meaning enters the system, or
only how resolved meaning is used?* If the latter, it is parallel-track. (2) Its correct form is
sub-symbolic-grounding + symbolic-composition, never symbolic matching over raw surface. (3) It goes
through the gate as a **capability** experiment and **does not move 0/47** — even the correct form needs
mpnet to resolve paraphrase first, which is the deployment question gated on real traffic. (4) Working
memory / graphs / staged learning earn their place *after* a deployment-certified router needs them, not
before. Build them when the certified router demands the feature; until then they are the second floor,
and the first floor is certified by real traffic flowing to the armed shadow harness — which **no
downstream mechanism accelerates.**

Rete-over-semantic-facts is therefore **recorded, not specced**: a real future capability mechanism, the
correct form of a real instinct, deferred to when a certified router needs cross-turn state. Recording it
here closes the loop so the instinct need not be re-derived — and so the fifth idea is met with this rule,
not a fresh build.

## 5. Artifact & reproduction index

| Result | Spec section | Artifact / command |
| --- | --- | --- |
| Competence-routing negative | `COMPETENCE_ROUTING_SPEC.md` §6 | `--phase3f-competence` |
| Encoder wall / `INVALID` | `GROUNDING_LOOP_SPEC.md` §14–16 | `--scan-disjoint-corpus`, `--real-encoder-experiment` |
| mpnet capability PASS | `GROUNDING_LOOP_SPEC.md` §17 | `--verify-disjoint-eval`, `certify_encoder_pipeline` |
| Cone-router anti-collapse | `COMPETENCE_ROUTING_SPEC.md` §10 | `phase3g_cone_results.txt`; `--phase3g-cone` |
| Graph router bar — E0 PASS (lift +0.213) | `SEMANTIC_GRAPH_ROUTER_SPEC.md` §3.2.1 | `--real-encoder-experiment data/authored_disjoint_eval embeddings_all-mpnet-base-v2.json` |
| Chat-output certification | `GROUNDING_LOOP_SPEC.md` §19 | `scripts/certify_chat.mjs` |
| Deployment state (0/47) | `GROUNDING_LOOP_SPEC.md` §18.9 | `drain_capture.py` → `--audit-capture` |
| Shadow-mode protocol (armed, data-blocked) | `GROUNDING_LOOP_SPEC.md` §21 / §21.6 | offline-shadow: `--real-encoder-experiment` + `--audit-capture` |
