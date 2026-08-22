# Semantic Graph Router — pre-registered spec

**Status:** pre-registration (certifier-first). Gate defined *before* build, per the project discipline.
Capability architecture, gated by the authored disjoint bench (`GROUNDING_LOOP_SPEC.md` §20.1) —
buildable and testable **now**, downstream of the mpnet capability PASS (§17), not an alternative to it.

---

## 1. The distinction that decides everything

We already route over a grounded graph: `pet_world_grounding.toml` has concept nodes,
`is_a`/`adjacent_to`/`contrasts_with` edges, OCEAN anchors. **It routed at chance on held-out paraphrase
at 119 nodes** (§14–16). So "treat knowledge as a grounded graph" cannot be the answer to the wall — the
wall *appeared inside a grounded graph we already built*. Graph structure did not save it.

The reason is the **entry function**, not the topology:

| | Entry function | "the Roomba is freaking her out" → "vacuum/fear" node? | Disjoint test |
| --- | --- | --- | --- |
| **Lexical-entry graph** (current grounding TOML, Graphify-style) | phrase activates a node by **alias/surface overlap** | **No** — no lexical overlap with "vacuum" | **fails** (chance on held-out, §16) |
| **Semantic-entry graph** (this spec) | phrase routes by **cosine-NN to mpnet node embeddings** | **Yes** — lands near vacuum/fear by *meaning* | **to be tested** |

A richer graph with a lexical entry function is just a more elaborate lexical system; it fails the
disjoint test identically. What changed our position was **mpnet** (a semantic embedding), not graph
structure. So the legitimate architecture is: **a graph whose nodes are embeddings and whose entry is
semantic nearest-neighbor, with edges encoding the relational structure the embedding is weak at.**

- **Embedding is load-bearing for the wall** ("same meaning, different words").
- **Graph edges are load-bearing for structure** ("these concepts are relationally related"
  — is-a, contrast, adjacency — compositional structure cosine-NN alone misses).

This spec is downstream of mpnet. It is worth building **only if the edges earn their place** (§3.1).

---

## 2. Architecture

### 2.1 Nodes — embedded, not aliased

- Each concept node carries a **semantic embedding** = mpnet (or the eventual distilled student) of the
  node's canonical description and/or the centroid of its member phrases.
- **Provenance rule (load-bearing):** node embeddings come from the **semantic encoder**, never from
  co-occurrence / surface extraction. A Graphify-style extraction pipeline encodes *lexical*
  relationships dressed as semantic ones (it would place "vacuum" and "Roomba" far apart because they're
  lexically distinct and co-occur rarely) — that is the 119-node failure with a graph drawn around it.

### 2.2 Entry — semantic nearest-neighbor

- Route a phrase by cosine-NN between its mpnet embedding and node embeddings. This is the entry function
  that survives paraphrase; it is also the "routing-at-scale" answer (vector-index NN, scales by
  construction — `GROUNDING_LOOP_SPEC.md` §20.5). Entry is **not** the contribution; mpnet already does it.

### 2.3 Edges — the actual contribution

- Edges (`is_a`, `adjacent_to`, `contrasts_with`) encode relations the embedding cannot recover from
  surface meaning alone. The router *uses* them — candidate forms to pre-register and compare in §3:
  - **E0 — flat NN (no edges):** the ablation baseline. Route to top-1 node by embedding only.
  - **E1 — neighborhood-aggregated NN:** blend a node's score with its graph neighbors' (edge-weighted),
    so a phrase near a leaf also activates its `is_a` parent / `adjacent_to` siblings.
  - **E2 — contrast-resolved NN:** when top-2 nodes are `contrasts_with`-linked and similarity is close,
    use the edge to disambiguate (the relational structure breaks the tie the embedding can't).
- The pre-registered claim is **not** "the graph router passes" — it is "**E1/E2 beat E0** on the disjoint
  bench." If they don't, the edges add nothing and the graph is decoration over mpnet.

---

## 3. Pre-registered certification (write the gate first)

### 3.1 The decisive test — does the GRAPH add value over the EMBEDDING?

The headline certifier is the **ablation**, not the comparison to the lexical baseline:

> **PASS condition (pre-registered):** on the authored disjoint bench, an edge-using router (E1 or E2)
> beats **flat mpnet-NN (E0)** on disjoint-bin accuracy by a margin whose Wilson CI excludes zero —
> *and* the win localizes to phrases where relational structure is required (E0's errors that are
> graph-adjacent, not random). If E1/E2 ≈ E0, **the graph is rejected as decoration**; ship flat NN.

Why this and not "beats lexical": flat mpnet-NN already beats the lexical grounding (that's the §17 PASS).
"Beats lexical" is cleared by *using mpnet at all* and is **not** evidence the graph does work. Only
beating flat NN is.

#### 3.1.1 The edge-sensitive slice must be selected router-independently (or the win is circular)

The localization clause makes the **edge-sensitive eval slice** the new load-bearing artifact — and we
author it, which is the provenance trap one level up. If we select "cases where relational structure is
genuinely required" by our own intuition about where edges *should* help, we build an eval that confirms
the hypothesis by construction — the graph-clothing version of training on the certified metric (the
margin↔r circularity, §COMPETENCE_ROUTING_SPEC §10, reappearing).

**Rule:** membership in the edge-sensitive slice is defined by a **router-independent criterion**, fixed
before either router is scored:

- A phrase is "relational-required" iff **≥2 independent annotators agree** that resolving it correctly
  requires *composing or contrasting two concepts* (e.g. "semantically near concept A but the correct
  route is its `contrasts_with` neighbor B"), judged **blind to whether E1/E0 actually route it right**.
- The annotators never see router outputs; the slice is frozen before E0/E1/E2 are run.
- Report inter-annotator κ on slice membership; drop low-agreement items.

Without this, "E1 wins on the relational slice" means only "E1 wins on a slice chosen to make E1 win."

#### 3.1.2 Pre-committed expectation: E1 ≈ E0 is the *likely* outcome

We pre-commit to **expecting decoration**, because there is a structural reason it's likely: a good
semantic embedding **already encodes most of the relational structure** the edges would add. mpnet places
"vacuum" near "fear" because it learned that relation in meaning-space — so an `adjacent_to` edge between
them re-encodes what the embedding already has and contributes nothing on top. Edges can only earn their
place where **the embedding's geometry and the domain's true relations disagree**: concepts semantically
*near* but that should route *apart*, or semantically *far* but compositionally linked. Those cases exist
but may be **rare**, giving E1 ≈ E0 across most of the bench with at most a tiny edge-localized win.

Consequence (anti-mirage): a ~1-point E1 edge over E0 that does not clear the CI is **inside the noise**
and is read as decoration, not "the graph helps." The burden is on the edges to clear the bar, not on E0
to disprove them.

**Where to look hardest:** `contrasts_with`. Semantically-similar-but-opposite is the one relation an
embedding is *structurally* bad at — nearest-neighbor fails exactly there, and an explicit edge could
genuinely win. If edges earn their place anywhere, it is most likely here; E1's neighborhood-aggregation
(which an embedding mostly subsumes) is the least likely to.

### 3.2 Baselines (all on the same authored disjoint bench, §20.1)

| Baseline | Purpose |
| --- | --- |
| **Lexical grounding** (current TOML alias match) | the documented negative (§16) — context, not the bar |
| **Flat mpnet-NN (E0)** | **the bar** — does the graph beat the embedding alone? |
| **Shuffle floor** | label-permuted; `disjoint_semantic_lift` must clear its 95th pct |
| **Edge-using router (E1/E2)** | the candidate |

#### 3.2.1 E0 measured — the bar is set, and it clears the bench

Flat mpnet nearest-concept routing **is** the §17 capability experiment (`--real-encoder-experiment`):
route each phrase to the nearest concept node in mpnet space, certify on the n=47 authored disjoint bench.
Reproduced fresh (deterministic seed 42, 200-shuffle null, `certify_artifacts/verdict_all-mpnet-base-v2_3c5eefa076a8f5ba_42.json`):

| Encoder | Verdict | Lift | CI-lo | N | Level |
| --- | --- | --- | --- | --- | --- |
| cata (positive control) | `FAIL_MEMORIZATION` | −0.041 | −0.059 | 47 | wbc |
| supervised (homegrown) | `FAIL_MEMORIZATION` | −0.062 | −0.062 | 47 | wbc |
| **all-mpnet-base-v2 (E0)** | **`PASS`** | **+0.213** | **+0.080** | 47 | wbc |

CATA collapses (lift −0.041, below floor) → the eval is valid (tests semantics, not surface). E0's lift CI
lower bound (+0.080) excludes zero → **E0 PASSes the disjoint bench**.

**What this decides (build-order §7 step 1):** because E0 already clears the bench, **the edges are an
optimization question, not a necessity question.** We are not asking "do edges make routing work" (it
already works); we are asking "can edges improve a router that already passes." That raises the bar for
E1/E2 — they must beat a *passing* baseline, not a failing one — and is exactly why §3.1.2 pre-commits to
E1 ≈ E0 as the expected outcome. This is a capability bar on the authored bench (§6): it does not move the
0/47 deployment wall.

### 3.3 Metrics (reused, not invented)

- `disjoint_semantic_lift` = disjoint-bin accuracy − shuffle-floor 95th pct (must be > 0, CI excludes 0).
- Disjoint-bin **resolvable** precondition (`audit_disjoint_eval`: n ≥ 47, CI width ≤ 0.30) before any
  verdict is read — same gate as everything else.
- **CATA positive control:** pure lexical matching must **collapse** on the eval; if it doesn't, the eval
  is lexically separable (easy) and `INVALID`, not a graph win.
- **Edge-ablation delta:** E1/E2 accuracy − E0 accuracy, with CI; plus the localization check in §3.1.

### 3.4 Decision table (pre-registered)

| Outcome | Reading | Action |
| --- | --- | --- |
| E1/E2 > E0 (CI excludes 0) **and** win localizes to graph-adjacent errors | the graph edges do real relational work | keep the graph; certify as a capability result |
| E1/E2 ≈ E0 | embedding does all the work; edges are decoration | **ship flat mpnet-NN; reject the graph** |
| E1/E2 > E0 but only on in-lexicon phrases (CATA didn't collapse) | "win" is lexical leakage | `INVALID`; rebuild the eval |
| Disjoint bin not resolvable (n<47 / CI too wide) | underpowered | grow the authored bench (§20.1); no verdict |

### 3.5 Kill conditions

- If after a bounded build the edge-ablation delta is not positive with CI excluding zero on a resolvable,
  CATA-collapsing bench → **stop; ship flat NN.** Do not add more edge types hunting for a win (that is
  the threshold-nudge in graph clothing).

---

## 4. Graph-specific traps (named so they can't hide)

- **Graphs look like understanding.** Nodes, edges, hierarchy, semantic-sounding relations — a human sees
  meaning everywhere. That perceived meaningfulness is *exactly* the mirage condition (clusters always
  look meaningful). Legibility of the graph is **not** evidence it routes by meaning; only the disjoint
  gate is. Graphs are unusually good at looking meaningful, so the gate matters more here, not less.
- **Graphify/extraction edges are lexical in disguise.** Co-occurrence/surface-built graphs encode
  lexical relationships dressed as semantic ones. Hence §2.1: node embeddings and edge weights must trace
  to the semantic encoder or to human-authored relations, never to surface co-occurrence — and the
  ablation (§3.1) is what catches it if they don't.
- **"Passes where lexical failed" is the seductive non-test.** It is cleared by mpnet alone. The only
  test that isolates the graph is **E1/E2 vs E0**.

---

## 5. Fair-fight / decontamination

- Node embeddings, edge weights, and routing thresholds are built **only** from training data + authored
  relations — never from the eval. The bench stays held-out.
- Same firewall as §20: capability certification on authored / held-out disjoint phrases; **never** a
  deployment claim (that needs the `RealTraffic` bin, §18). LLM-paraphrase expansion of the bench is
  allowed under the §20.2 firewall (test XOR train, lineage-tracked, disjointness-verified).
- The edge-ablation is the decontamination: it prevents crediting the embedding's generalization to the
  graph.

---

## 6. Scope

- **Capability architecture, gated now.** Testable today on the authored bench; does **not** need the
  real-traffic bin. It does **not** overcome the data wall (§20) — it is downstream of the encoder that
  already broke the representation wall.
- **Routing-at-scale is not what this solves.** That is geometric NN (a vector index), already solved by
  construction once the space is semantic. This spec is about the **relational-composition** layer on top
  of NN — the part edges contribute, *if* §3 shows they do.
- **Relation to the cone router** (`COMPETENCE_ROUTING_SPEC.md` §10): the cone router is supervised,
  boundary-aware dispatch among frozen specialists on Task E; the semantic graph router is unsupervised
  semantic NN + relational edges over concept nodes. Both go through the same disjoint gate; neither is
  assumed to win without it.

---

## 7. Build order

1. **E0 first — build the thing that might make the graph unnecessary.** Embed nodes via mpnet, route by
   cosine-NN, certify on the authored bench. If E0 already clears the disjoint bench, you have a working
   semantic router *now*, and E1/E2 become **optimization, not necessity** — the question shifts from "do
   edges make it work" to "can edges improve a router that already works." That framing is the protection
   against crediting the graph for generalization the flat baseline would have provided alone (same logic
   as "run the validated teacher before building the distilled student"). Record E0 as the bar.
2. **Author the edge-sensitive eval slice** — disjoint phrases whose correct route requires relational
   structure (paraphrases that sit near the wrong node by raw similarity but resolve via is-a/contrast).
   Without this slice the ablation can't show localization (§3.1).
3. **Add E1, then E2** — measure the edge-ablation delta with CI on the resolvable, CATA-collapsing bench.
4. **Read the §3.4 table.** Keep the graph only if the edges earn it; otherwise ship flat NN and record
   the negative honestly (a real result: "on this domain, semantic NN suffices; relational edges add no
   measurable routing value").
