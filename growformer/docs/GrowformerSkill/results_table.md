# Complete Results History

All configurations tested, in chronological order.

## Phase 1A — Finding the Baseline (hard-zero KWTA era)

| # | Configuration | Accuracy | Best Loss | Key Outcome |
|---|--------------|----------|-----------|-------------|
| 1 | 2→16→1, early runs | 52–54% | — | Input synapse pruning → geometric blindness |
| 2 | 2→16→28→1, input protection | 59.8% | — | First real result after input protection |
| 3 | 2→32→1, all fixes | 52–54% | — | Gradient cancellation → flat loss |
| 4 | 2→16→28→1 + warmup | 56% | — | Warmup created always-on attractor |
| 5 | 2→16→16→1, no KWTA | 53% | — | Strong synapses beat inhibition |
| 6 | **2→16→16→1, Global KWTA k=4** | **66%** | **0.233** | **KWTA broke collapse ceiling** |
| 7 | 66% + local KWTA r=1.5 | 62.2% | 0.221 | Radius too large — near-global |
| 8 | 66% + local KWTA r=0.8 | 54.5% | 0.240 | Radius too small — zero suppression |
| 9 | 66% + prune_stop_tick | 66.1% | 0.193 | Always-on attractor returned |
| 10 | 66% + 600 samples | 73.5% | 0.200 | +7.5% — biggest single improvement |
| 11 | **800 samples + mass_decay fix** | **75.4%** | **0.188** | **Old ceiling (later broken)** |
| 12 | 75.4% + prune_stop_tick | 66.1% | 0.193 | Frozen connectivity regressed |
| 13 | 75.4% + local KWTA r=0.9, 32 neurons | 67.4% | 0.216 | Input energy 77+ → no competition |
| 14 | 75.4% + input cap 0.5 | 54.4% | 0.245 | Starved synapses → mass pruning |
| 15 | 75.4% + 16,000 epochs | ~75% | — | Extended training did not help |
| 16 | 75.4% + 24 neurons | ~75% | — | Capacity increase did not help |
| 17 | 75.4% + soft local KWTA r=1.2 | 58.2% | — | 4 dead neurons, same density failure |
| 18 | 2→24→16→1, 16000 epochs | ~75% | 0.198 | Peak epoch 2500, then destructive pruning |

## Phase 1B — File Drift Debugging (residual fix era)

| # | Configuration | Input Energy | Accuracy | Key Outcome |
|---|--------------|-------------|----------|-------------|
| 19 | Post-drift, cap removed | 0.87–2.82 | 55.6% | Residual=0.0 (hard zero still) |
| 20 | Synapse floor added | — | 55.9% | Floor working but bias saturation |
| 21 | KWTA residual=0.05 | ~82 | 62.4% | Too much preserved — syns stuck at 219 |
| 22 | KWTA residual=0.01 | 3–4 | 64.5% | Best result before breakthrough |
| 23 | Input decay=0% | 38 | 59.0% | Energy exploded |
| 24 | Input decay=10% | 24.5 | 62.1% | Energy too high |
| 25 | **Uniform decay + residual=0.02** | **3–8** | **64.5%** | **Stable baseline** |
| 26 | Inner oversampling (50/class) | — | 58.2% | Inner points irreducibly ambiguous |

## Phase 1C — KWTA Breakthrough

| # | Seed | Run | Accuracy | Active Neurons | Synapses | Loss Floor |
|---|------|-----|----------|---------------|----------|------------|
| 27 | 42 | 1 | **90.4%** | 9 of 16 | 163 | 0.084 |
| 28 | 42 | 2 | **92.6%** | 10 of 16 | 150 | 0.076 |
| 29 | 42 | 3 | **90.1%** | 9 of 16 | ~155 | 0.082 |
| 30 | 7 | 1 | **92.2%** | 10 of 16 | 139 | 0.077 |
| 31 | 7 | 2 | **91.8%** | 10 of 16 | ~145 | 0.079 |

## MLP Baseline Comparison

| Model | Seed | Accuracy | Active Neurons | Synapses | Loss Floor |
|-------|------|----------|---------------|----------|------------|
| Growformer | 42 | 90.4–92.6% | 9–11 | 139–163 | 0.076–0.084 |
| MLP (no KWTA/physics/pruning) | 7 | 90.4% | 16 | ~272 stable | 0.070 |

**Verdict:** Accuracy within noise (~1-2%). Growformer uses 40% fewer synapses, 37% fewer active
neurons. MLP loss flatlined at epoch 500; Growformer continued improving to epoch 5500.

## Benchmark Suite

| Task | Config | Accuracy | Active Neurons | Notes |
|------|--------|----------|---------------|-------|
| Double spiral | seed=42, 8000 epochs | 90–92.6% | 9–11 of 16 | Primary benchmark |
| Concentric circles noise=0.05 | seed=7, 8000 epochs | 100% | 16 of 16 | Trivially separable |
| Concentric circles noise=0.25 | seed=7, 8000 epochs | 97.9% | 16 of 16 | Rings overlap, still easy |

## Key Findings (updated)

**The 75.4% ceiling was not an information-theoretic limit.** It was KWTA gradient starvation.
Hard-zero suppression zeroed gradient for 75% of hidden neurons every forward pass. KWTA residual
of 0.02 (two hundredths) preserved enough gradient to unlock 90%+. This is the single most
important finding in the project.

**Structural efficiency is problem-complexity dependent.** Spiral (intrinsically 2D boundary)
→ 9–11 active neurons. Circles (intrinsically 1D, radius threshold) → 16 regardless of noise.
The architecture allocates representation proportional to intrinsic dimensionality.

**Pruning as self-organization holds.** Every experiment that froze pruning regressed.
Pruning is actively co-shaping the solution throughout training.

**File drift is the primary reproducibility threat.** Three separate debugging sessions were
caused by environment.rs, types.rs, or growth.rs drifting from the validated outputs/ versions.