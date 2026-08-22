# Spec: Conditional Mutual Information Measurement — "Present but Inaccessible"

**Purpose:** Quantify whether per-specialist forward signals carry region/correctness information beyond the scalar output.
**Status:** First pass complete (`--cmi`). Spiral region MI under resolution review (`--cmi-spiral-resolve`).

---

## First-pass results (2026-06-05, `--cmi`, 20 seeds × 370 pts)

| Quantity | Mean ± std (backend-range) | Interpretation |
| -------- | -------------------------- | -------------- |
| H(R) | 1.000 ± 0.000 | Balanced ✓ |
| I(R; Y_spiral) | 0.004 ± 0.008 | Scalar region-blind ✓ |
| I(R; Y_circles) | 0.564 ± 0.153 | Region in circles scalar ✓ |
| I(R; A_spiral) MLP | 0.255 ± 0.101 (**±0.252**) | **Unmeasured** — backend disagreement = effect size |
| I(R; A_circles) MLP | 0.632 ± 0.167 (±0.103) | Mostly in scalar |
| **ΔCMI_spiral** | **0.251 ± 0.098** | Tripped falsification (>0.15) — **do not cite until spiral-resolve** |
| **ΔCMI_circles** | 0.068 ± 0.166 | ≈ 0 — entanglement holds for circles |
| I(R; (Y₁,Y₂)) | 0.596 ± 0.133 | "Present" half ✓ — joint carries region |
| **ΔCMI^C_spiral** | **0.097 ± 0.088** | **Load-bearing** — correctness barely above scalar |
| **ΔCMI^C_circles** | **0.128 ± 0.107** | **Load-bearing** — same |

### Interpretation discipline (post first-pass)

1. **Do not claim uniform information wall** — spiral first-pass ΔCMI fired falsification; spiral-resolve collapses it to below resolution.
2. **Do not claim output bottleneck** — not confirmed after permutation null.
3. **Do claim correctness barrier** — `ΔCMI^C ≈ 0.1` both specialists; explains cw_c ≈ 0.60.
4. **Paper §5 leads with ΔCMI^C**, not region ΔCMI_spiral.

---

## Spiral resolve (§spiral-resolve — `--cmi-spiral-resolve`)

Binary question: output bottleneck vs below resolution?

### Three instruments
1. Permutation null (B=500, linear probe, identical pipeline)
2. Linear probe + angle(probe w, output head w)
3. PCA-reduced k-NN MI at m ∈ {2,3,5}

### Pre-registered §4 decision

| Condition | Verdict |
| --- | --- |
| debiased > 0.10 AND linear > I(R;Y)+0.10 AND PCA-kNN > 0 | Output bottleneck CONFIRMED |
| observed inside null OR linear ≤ base+0.05 | Below resolution |
| else | Nonlinear, fragile |

### Spiral resolve results (2026-06-05, filled)

| Instrument | Pooled | Per-seed | Notes |
| ---------- | ------ | -------- | ----- |
| I(R; Y_spiral) | 0.021 | — | Region-blind scalar ✓ |
| I(R; A_spiral) MLP (10×5-fold CV) | **0.000 ± 0.000** | 0.067 ± 0.073 | perm **p=0%**, null95=0.000 |
| I(R; A_spiral) linear probe | **0.000 ± 0.000** | 0.040 ± 0.062 | probe⊥output **90.1°** avg |
| PCA-kNN MI (m=2,3,5) | 0.489 / 0.504 / 0.489 | — | Disagrees with linear/null — single-split, no permutation |
| De-biased (obs − null95) | **0.000** | — | |

**§4 verdict: Below resolution / near wall.** First-pass `ΔCMI_spiral=0.251` was single-split optimism; repeated CV + permutation null collapses it. **Output bottleneck not confirmed.**

**Instrument note:** PCA-kNN ~0.5 bit vs linear/null at 0. **Verdict rests on the two null-audited instruments agreeing at zero**, not on three-way consensus. PCA-kNN is flagged as disagreeing and not yet null-tested (optional close: PCA-kNN permutation null, same B=500).

**§5 prose draft:** `docs/PAPER_TWO_SECTION5_DRAFT.md` (ready to paste into whitepaper when pointed).

---

## Implementation

```bash
cargo run --release --bin growformer-demos -- --cmi
cargo run --release --bin growformer-demos -- --cmi-analyze
cargo run --release --bin growformer-demos -- --cmi-spiral-resolve
cargo run --release --bin growformer-demos -- --cmi-spiral-analyze
```

Artifacts: `cmi_diagnostic.csv`, `cmi_spiral_resolve.json`

---

## §8 template (fill after spiral-resolve + correctness anchor)

> Region is fully determined by the input (H(R) ≈ 1.0 bit) and recoverable from the cross-specialist joint (I(R;(Y₁,Y₂)) ≈ 0.60 bit). Whether spiral's activations carry region beyond its scalar is **below estimator resolution** under held-out linear/parametric classifiers with permutation null (de-biased ΔCMI_spiral = 0.000; first-pass 0.251 was single-split optimism). Circles exports region to its scalar (I(R;Y) ≈ 0.56); activations add little beyond it (ΔCMI_circles ≈ 0.07). **Per-specialist own-correctness information is bounded near the scalar for both specialists (ΔCMI^C ≈ 0.10 bit)** — the information form of the confident-wrong probe (cw_c ≈ 0.60). Per-specialist competence routing is therefore information-bounded on correctness; the always-spiral collapse follows from spiral's region-blind scalar (I(R;Y_spiral) ≈ 0), not from a confirmed activation-level bottleneck.
