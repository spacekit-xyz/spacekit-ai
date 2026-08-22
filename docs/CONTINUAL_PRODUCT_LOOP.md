# Continual product loop (growformer companions + domain brains)

**Product chatbot stack:** growformer (Luna-style chat / Path A retrieve+compose).  
**Not** growformer-llm open generation.

This doc turns the “where to improve” list into an operable loop: **router adapter → train loop → fragments / multi-turn → ship gates**.

---

## 1. Fingerprint adapter (augment, don't override) — preferred for companions

Growformer already stores **neural fingerprints**:
- group structural fingerprints (grade-2 Cl(8)) for OOD / structure routing
- per-topic centroids + causal fingerprints for topic retrieve

`--train-fingerprint-adapter` EMA-blends new data into those fingerprints **without** touching lattices or the LearnedRouter:

```bash
cd growformer

cargo run --release --features cli --bin growformer -- \
  --train-fingerprint-adapter \
  --fingerprint-alpha 0.25 \
  --project ../../spacekit/spacekit-projects/companions/luna/luna.gf.toml \
  --brain ../../spacekit/spacekit-projects/companions/luna/agent/luna-v3-3d.bin \
  --brain-output agent-data/luna-fp-adapted.bin
```

**Contract**

| Piece | Behavior |
| --- | --- |
| Gen lattices / programs | Unchanged / stay frozen |
| LearnedRouter | Unchanged |
| Group + topic fingerprints | EMA-augmented (`fp ← (1−α)·old + α·new`) |
| Next step | Frozen prompt matrix before ship |

This is the right continual path when the brain is effectively one chat group (Luna `pet_chat` → legacy `support`/`coding` names): multi-class router retrain has nothing to learn and would wipe a useful router.

---

## 2. Router adapter (multi-class domain brains)

Routing is the usual failure mode on **multi-group** domain brains (wrong group → wrong lattice).  
You can retrain **only** the `LearnedRouter` (optional ActionClassifier) and keep frozen lattices. Router adapter **always** fingerprint-augments first; it **skips** LearnedRouter retrain when labels collapse to one class (companion safeguard).

```bash
cd growformer

cargo run --release --features cli --bin growformer -- \
  --train-router-adapter \
  --project ../../spacekit/spacekit-projects/companions/luna/luna.gf.toml \
  --brain ../../spacekit/spacekit-projects/companions/luna/agent/luna-v3-3d.bin \
  --brain-output agent-data/luna-router-adapted.bin \
  --brain-epochs 30

# Optional: also refresh ActionClassifier
cargo run --release --features cli --bin growformer -- \
  --train-router-adapter --also-classifier \
  --project … --brain … --brain-output …
```

**Contract**

| Piece | Behavior |
| --- | --- |
| Gen lattices | Unchanged / stay frozen |
| Fingerprints | Always EMA-augmented |
| Router | Rebuilt only if ≥2 label classes map to brain groups |
| Multi-turn rows | Adapters use **bare** user text (same as `converse()` routing) |
| Next step | Always run a frozen prompt matrix before ship |

---

## 3. Train in a loop

```bash
# Fingerprint augment loop (safest companion default)
PROJECT=../../spacekit/spacekit-projects/companions/luna/luna.gf.toml \
BRAIN=../../spacekit/spacekit-projects/companions/luna/agent/luna-v3-3d.bin \
MODE=fingerprint LOOPS=2 FP_ALPHA=0.25 \
GATE_CMD='echo "TODO: run companion matrix / Bet D held-out here; BRAIN=$BRAIN"' \
bash scripts/train_loop.sh

# Router refresh (domain brains with real multi-group labels)
MODE=router LOOPS=2 bash scripts/train_loop.sh

# Overlay CL: new shard → train small brain → merge → gate
MODE=overlay \
BRAIN=path/to/base.bin \
SHARD_DIR=path/to/new_jsonl_dir \
LOOPS=1 \
bash scripts/train_loop.sh
```

Modes: `fingerprint` (default), `router`, `overlay` (`--train-brain` on shard + `--merge-brain`), `full` (full retrain).

---

## 4. Better multi-turn + more fragmentation

Two complementary knobs:

### A. Multi-turn JSONL (lattice / history conditioning)

Keep / grow rows like Luna `luna_multiturn_v1.jsonl`:

- `pet.history` / `history` with prior user+pet turns  
- `conversation_turn` > 1  
- Full `expected_response` for that turn  

Full `--train-brain` already re-encodes multi-turn samples with the same context prefix + blend as `converse()` so follow-ups retrieve.

### B. More fragments (compose diversity)

Fragment composer needs competing opener/body/closer lines per intent. Mine candidates:

```bash
python3 scripts/expand_multiturn_to_fragments.py \
  --in ../../spacekit/spacekit-projects/companions/luna/data/luna_multiturn_v1.jsonl \
  --out ../../spacekit/spacekit-projects/companions/luna/data/luna_fragments_from_multiturn.jsonl \
  --archetype cheerful_companion
```

Then **manually review**, dedupe, append into `fragments_jsonl` referenced by `[inference]` / `inference_pets.toml`, redeploy. Do not auto-ship mined fragments without review (hallucination / tone risk).

Target: several fragments per high-traffic intent so drive/reflective fields have real competition (reduces fallthrough / bland defaults).

---

## 5. Interaction cycle (sample-efficient CL — no RL)

Live traffic → sparse fingerprint updates, with a human label gate:

```text
browser capture → drain_capture.py → --audit-capture → label_queue.jsonl
       → (human fills semantic_intent)
       → promote_label_queue.py → approved shard
       → MODE=fingerprint DATA_DIR=shard GATE_CMD=certify_chat
       → optional PROMOTE=1
```

```bash
# 1) Drain + triage (stops for human labeling)
STORAGE_URL=https://your-storage-node \
STAGE=drain_audit \
bash scripts/luna_interaction_cycle.sh

# 2) After labeling semantic_intent on capture_artifacts/label_queue.jsonl:
STAGE=promote_adapt \
REVIEWED_BY=you \
N=6 \
bash scripts/luna_interaction_cycle.sh

# Install only after gate PASS:
PROMOTE=1 STAGE=promote_adapt REVIEWED_BY=you bash scripts/luna_interaction_cycle.sh
```

Pieces:

| Script | Role |
| --- | --- |
| `scripts/drain_capture.py` | Storage → `traffic_*.jsonl` |
| `growformer-demos --audit-capture` | Triage → `label_queue.jsonl` |
| `scripts/promote_label_queue.py` | Reviewed queue → train/eval shard + manifest |
| `scripts/train_loop.sh` + `DATA_DIR` | Fingerprint EMA on approved shard only |
| `scripts/certify_chat.mjs` + `EXTRA_EVAL` | Authored matrix + capture holdout prompts |
| `scripts/luna_interaction_cycle.sh` | Orchestrates the above |

**Contract:** lattices stay frozen; only fingerprints move on capture shards; no auto-promote without certify PASS.

---

## 6. Where to improve — execution order

| # | Step | How |
| --- | --- | --- |
| 1 | Gate every ship | `GATE_CMD` on `train_loop.sh` + `certify_chat.mjs` |
| 2 | Interaction cycle | `luna_interaction_cycle.sh` (capture → label → fingerprint) |
| 3 | Scenario / pet topics as versioned TOML | SpaceKit `inference_*.toml` |
| 4 | Fingerprint CL (companions) | `MODE=fingerprint` + optional `DATA_DIR` |
| 5 | Overlay CL | `MODE=overlay` + merge + gate |
| 6 | Hybrid when router misses | Embedding k-NN + forced-topic (domain) |
| 7 | Enforce chat contract | Luna `inference_pets.toml` bounds |
| 8 | Fragment diversity | `expand_multiturn_to_fragments.py` + curated merge |
| 9 | Cone over specialists (later) | Task E cone when many overlays stack |

---

## Recommended companion workflow

1. Keep capture enabled in Agent Hub  
2. Periodically `STAGE=drain_audit` → human-label `label_queue.jsonl`  
3. `STAGE=promote_adapt` (fingerprint + certify); `PROMOTE=1` only on PASS  
4. Curate fragments / multi-turn for compose diversity  
5. Full `--train-brain` only when lattice content must change  

**Bottom line:** continuous learning for product = **interaction capture → human label → fingerprint augment → certify**, not open LM RL. Do not ship a router-adapted companion brain that reports mono-class / ~3% router accuracy.
