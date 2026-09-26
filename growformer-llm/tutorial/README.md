# Tutorial: train and run a Growformer domain assistant

This walks you from a clean checkout to a working assistant that answers crypto sentiment questions. Every command here was run end to end on the bundled project.

A Growformer assistant has two parts:

| Part | What it is | Role | Trained by |
|---|---|---|---|
| **Brain** (`agent/*.bin`) | Growformer lattice memory with routing, grounding and guardrails | **Explains the answer** from stored examples. Also handles identity, greetings and guardrails. | `train_brain.sh` (the `growformer` CLI) |
| **Label model** (`agent/label-model.json`) | TF-IDF + logistic-regression classifier over `semantic_intent` | **Picks the label.** The brain then explains from that label's examples. | `train_brain.sh` (`gf-llm label-train`) |
| **LM** (`*-lm.json` + `.tok`) | Small vanilla transformer (`growformer-llm`) | Optional. Rephrases the brain's answer (`polish`), or generates freely (`lm`, experimental) | `train_lm.sh` (the `gf-llm` CLI) |

```text
growformer-llm/tutorial/
├── README.md                 ← you are here
├── train_brain.sh            ← project → agent/<brain>.bin
├── train_lm.sh               ← project JSONL → chat LM checkpoint (+ eval, + card)
├── infer.sh                  ← ask the brain / brain+LM (one prompt, batch, or REPL)
├── benchmark.sh              ← held-out accuracy: brain vs label+brain
└── projects/
    └── crypto-sentiment/     ← complete example project (copy of spacekit-projects/sentiment/crypto)
        ├── crypto-sentiment-analysis.gf.toml   ← manifest: data, inference TOMLs, brain path
        ├── agent/                              ← brain goes here (*.bin is git-ignored; step 1 builds it)
        ├── data/
        │   ├── train_sentiment_*.jsonl         ← training rows (brain + LM)
        │   ├── inference_crypto.toml           ← inference rules
        │   ├── world_grounding_crypto.toml     ← grounding lexicon / concept graph
        │   ├── knowledge_graph*.toml           ← topic graph + sentiment overlay
        │   ├── inference_guardrails.jsonl      ← loaded at inference only
        │   └── plugins/, locale/, scams_info.md
        └── prompts/crypto-prompts-demo.txt     ← 25 verified demo prompts
```

Run every command from the **repo root** (`spacekit-ai/`).

---

## 0. Prerequisites

- Rust (stable) and `cargo`
- About 2 GB of disk for release builds

The scripts build the binaries on first use. To build them up front:

```bash
cargo build --release -p growformer-llm --bin gf-llm
cargo build --release -p growformer --no-default-features --features cli,native --bin growformer
```

## 1. Train the brain and label model (about 1 minute)

```bash
bash growformer-llm/tutorial/train_brain.sh growformer-llm/tutorial/projects/crypto-sentiment
# → agent/crypto-brain.bin     (Growformer brain)
# → agent/label-model.json     (label classifier)
```

Both read every `train_*.jsonl` in `data/`. The brain also uses the `[train]` settings in the manifest. The tutorial manifest sets `sentiment_spawn_threshold = 2.0`, which keeps every training row as its own stored example (see §6).

## 2. Ask the brain

```bash
T=growformer-llm/tutorial
P=$T/projects/crypto-sentiment

bash $T/infer.sh $P "Bitcoin crashed after the ETF delay"
# Assistant> NEGATIVE (mild) — ETF timeline slip plus overnight BTC selloff; regulatory delay driving price weakness.

bash $T/infer.sh $P "Layer 2 fees dropped to sub-cent but bridging back to mainnet still costs more than my lunch"
# Assistant> MIXED — L2 + bridge + cost contrast; dual valence, not one-sided NEG.

PROMPTS=$P/prompts/crypto-prompts-demo.txt bash $T/infer.sh $P   # all 25 demo prompts
bash $T/infer.sh $P                                              # interactive chat (quit to exit)
COMPOSE=diag bash $T/infer.sh $P "Bitcoin crashed after the ETF delay"   # routing / memory diagnostics
```

Replies start with a label, then an explanation taken from the most similar training example that has that label. Off-topic prompts such as "who are you" or "hey" get the brain's identity or greeting reply.

`infer.sh` uses `agent/label-model.json` automatically when it exists. To compare with the brain on its own, run `LABEL=0 bash $T/infer.sh $P "…"`.

## 3. Train the LM (optional)

```bash
bash growformer-llm/tutorial/train_lm.sh growformer-llm/tutorial/projects/crypto-sentiment
```

The script runs five steps:

1. **Corpus.** Converts JSONL rows to `### User:` / `### Assistant:` turns (`gf-llm jsonl-to-txt --chat`).
2. **Tokenize.** Builds a 1024-token BPE vocabulary, then encodes and splits 90/10 into train and held-out.
3. **Train.** Uses a 128-wide, 4-block vanilla transformer with AdamW, warmup plus cosine schedule, and a global gradient clip. It checks the held-out set every 200 steps and keeps the best checkpoint. Training stops early after 5 checks with no improvement.
4. **Eval.** Reports held-out bits/byte against uniform, unigram, gzip and lzma baselines.
5. **Card.** Writes `<name>-lm.gfcard.json` (arch, tokenizer, bits/byte) for fleet/runtime discovery.

Outputs go to `growformer-llm/agent-data/tutorial/<project>/`:

| File | Contents |
|---|---|
| `<name>-lm.json` | Best-validation weights (use this for inference) |
| `<name>-lm.last.json` + `.last.optim.json` | Final weights and optimizer state (for resuming) |
| `<name>.tok` | Tokenizer (always pair it with the checkpoint) |
| `<name>-lm.gfcard.json` | Specialist card |

Useful knobs (environment variables): `STEPS` (4000), `D_MODEL` (128), `D_FF` (512), `N_BLOCKS` (4), `VOCAB` (1024), `GRAD_ACCUM` (8), `LR_MAX` (1e-3), `PATIENCE` (5), `GF_THREADS` (all cores). To continue an interrupted run: `RESUME=1 bash …/train_lm.sh <project>`.

A full run takes roughly 30–60 minutes on a laptop CPU. Early stopping usually ends it sooner.

## 4. Use the LM

```bash
COMPOSE=polish bash $T/infer.sh $P "Bitcoin crashed after the ETF delay"   # brain answer, LM rephrase
COMPOSE=lm     bash $T/infer.sh $P "Bitcoin crashed after the ETF delay"   # experimental free generation
```

- **`polish`** only keeps the LM's rewrite if it preserves the brain's label and mostly reuses its words. Otherwise it returns the brain's answer unchanged. With a small LM, expect the brain's answer most of the time.
- **`lm`** is a research mode. Each project corpus is only about 30k tokens (a few hundred examples), so the LM learns the reply *format* but not reliable content. Treat its output as a smoke test, not an answer. To improve it, add a lot more data or pretrain a base model first (`scripts/pretrain_base_vanilla.sh`, see `DEVELOPER.md`).

## 5. Your own projects

Any folder with a `*.gf.toml` manifest and a `data/` folder works with the same three scripts:

```bash
SP=~/Projects/2026/spacekit/spacekit-projects

bash growformer-llm/tutorial/train_lm.sh $SP/sentiment/fintech
bash growformer-llm/tutorial/infer.sh    $SP/sentiment/fintech "Shares fell 12% after the company cut guidance"

bash growformer-llm/tutorial/train_lm.sh $SP/companions/luna
bash growformer-llm/tutorial/infer.sh    $SP/companions/luna "Hey Luna"
```

**Manifest (`*.gf.toml`).** All paths are relative to the manifest:

```toml
[train]
data_dir = "data"
brain_output = "agent/my-brain.bin"
[inference]
toml = "data/inference_x.toml"
guardrails_jsonl = "data/inference_guardrails.jsonl"
topic_graph = "data/knowledge_graph.toml"
grounding_toml = "data/world_grounding_x.toml"
[infer]
brain = "agent/my-brain.bin"
```

**Training row (`data/*.jsonl`, one per line).** `text` and `semantic_intent` are required. `expected_response` is what the assistant should say:

```json
{"text": "Whales moved 8,000 BTC onto Binance", "semantic_intent": "negative_mild", "expected_response": "NEGATIVE (mild) — large exchange inflow reads as sell pressure."}
```

**Which files the LM trains on.** `train_lm.sh` detects this automatically:

- **`data/` has `train_*.jsonl`** (sentiment-style). Only those files are used. Replies are cleaned: the label is kept, meta text is stripped, and replies are capped at 160 characters.
- **Otherwise** (companion-style, like Luna). Every `*.jsonl` whose rows have `semantic_intent` is used, and replies are kept verbatim. Fragment files (`*_fragments_*.jsonl`) and guardrails are skipped. The fragments are inference data (`[inference].fragments_jsonl`), and `gf-llm jsonl-to-txt` errors on them.

Override the cleaning with `CHAT_CLEAN=true|false`.

## 6. Measure and improve accuracy

`benchmark.sh` holds out 20% of the labelled rows (stratified by label). It trains the brain and label model on the rest, then scores both on the held-out prompts. Nothing in the project folder changes.

```bash
bash growformer-llm/tutorial/benchmark.sh growformer-llm/tutorial/projects/crypto-sentiment
SEED=11 SPAWN=0.92 bash growformer-llm/tutorial/benchmark.sh <project>   # another split + old merge setting
```

Crypto project, 73 held-out prompts, 3 seeds (7 / 11 / 13):

| Setup | Coarse accuracy | Hedge rate | Label accuracy (16 labels) | Distinct explanations (seed 7) |
|---|---|---|---|---|
| Brain alone, old merge (0.92) | 53.4% | 89.9% | – | 13 / 73 |
| Brain alone, no merge (2.0) | 52.5% | 84.9% | – | – |
| **Label model + brain, no merge** | **59.4%** | **0%** | 45.7% | **56 / 73** |

Per seed, label model + brain scored 61.6 / 56.2 / 60.3%, and the brain alone (old merge) scored 53.4 / 54.8 / 52.1%. Most of the brain's correct labels come from hedged replies whose leading label is a lexicon guess.

- **Coarse accuracy** means the reply's label lands in the right bucket: positive, negative, neutral, mixed or sarcastic.
- **Hedge rate** is the share of replies that are "I don't have enough information…".
- **Label accuracy** is an exact `semantic_intent` match across 16 labels.

**Why the label model helps.** The brain's own topic router scores about 11% on its *training* rows. That means topic-scoped retrieval usually searches the wrong label's examples, and the grounding gate then declines. The label model picks the label directly (`gf-llm label-train` / `label-predict`). The brain then returns the stored example with that label whose prompt is closest to yours.

**Why `sentiment_spawn_threshold = 2.0`.** By default, brain training merges rows whose embeddings are at least 0.92 cosine-similar. On this corpus that kept 4–7 of 302 answers, so every prompt with a given label got the same explanation. With 2.0 every row is kept: about 56 distinct explanations for 73 prompts instead of about 13. The trade-off is time: queries take about 1.8 s instead of 0.2 s on this project.

**What to do next.**

1. **More labelled rows.** On this corpus, a classifier's held-out accuracy rises from 45% to 54% to 59% to 64% as training rows go from 75 to 151 to 226 to 302. Add rows and re-run `benchmark.sh`.
2. **Balance the labels.** `euphoric` has 4 rows, `capitulation` 6 and `sarcastic` 14, versus 76 for `mixed`. Get every label to at least 30 rows, or merge rare labels into their coarse bucket.
3. **Use one reply format.** Start every `expected_response` with the label (`NEGATIVE (mild) — …`). About half the crypto rows don't.
4. **Label what the brain hedges on.** Collect real prompts that got a hedge or a wrong label, label them, add them to `data/`, and retrain. `brain-eval --json-out` lists every held-out row with its reply.

## The raw commands

The scripts are thin wrappers. With the binaries built:

```bash
GF=target/release/gf-llm
P=growformer-llm/tutorial/projects/crypto-sentiment
OUT=growformer-llm/agent-data/tutorial/crypto-sentiment; mkdir -p $OUT

# brain
(cd $P && ../../../../target/release/growformer --train-brain --project crypto-sentiment-analysis.gf.toml)

# LM
$GF jsonl-to-txt $P/data --chat --out $OUT/chat.txt           # add --chat-clean false for companions
$GF tokenize $OUT/chat.txt 1024 $OUT/crypto.tok
$GF encode   $OUT/chat.txt $OUT/crypto.tok $OUT/crypto.bin
$GF split    $OUT/crypto.bin $OUT/train.bin $OUT/heldout.bin --train-frac 0.9
$GF train $OUT/crypto.tok $OUT/train.bin $OUT/heldout.bin --checkpoint-out $OUT/crypto-lm.json \
  --turn-aligned --seq-len 256 --steps 4000 \
  --no-param-match --d-model 128 --d-ff 512 --n-heads 4 --n-blocks 4 --tie-embeddings \
  --grad-accum 8 --lr-max 1e-3 --val-every 200 --patience 5 --subject crypto
$GF eval --checkpoint $OUT/crypto-lm.json --tokenizer $OUT/crypto.tok \
  --train-bin $OUT/train.bin $OUT/heldout.bin --seq-len 256 --windows 32 --no-ledger

# inference (run from the project folder so data/ paths resolve)
cd $P
$GF chat --compose brain  --brain agent/crypto-brain.bin --project crypto-sentiment-analysis.gf.toml --message "Bitcoin crashed after the ETF delay"
$GF chat --compose polish --brain agent/crypto-brain.bin --project crypto-sentiment-analysis.gf.toml \
  --checkpoint ../../../agent-data/tutorial/crypto-sentiment/crypto-lm.json \
  --tokenizer  ../../../agent-data/tutorial/crypto-sentiment/crypto.tok --message "…"
$GF brain-infer --brain agent/crypto-brain.bin --project crypto-sentiment-analysis.gf.toml --prompt "…" --brain-only -v
$GF label-train data --out agent/label-model.json [--holdout-frac 0.2]   # label model (+ quick held-out score)
$GF label-predict --model agent/label-model.json --prompt "…"
$GF chat --compose brain --brain agent/crypto-brain.bin --project crypto-sentiment-analysis.gf.toml \
  --label-model agent/label-model.json --message "…"
$GF jsonl-split data --train-out /tmp/train.jsonl --test-out /tmp/test.jsonl --seed 7
$GF brain-eval --brain agent/crypto-brain.bin --project crypto-sentiment-analysis.gf.toml \
  --test /tmp/test.jsonl [--label-model agent/label-model.json] [--json-out results.json] [-v]
$GF generate --checkpoint <lm.json> --tokenizer <.tok> --prompt $'### User:\nBitcoin crashed after the ETF delay\n### Assistant:\n' --greedy
```

`--hybrid true|false` on `chat` and `brain-infer` turns the raw-lattice shortcut on or off (default on).

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `raw lattice diagnostic did not capture (no encoded routing path)` | Older builds aborted when a prompt didn't route into the lattice (for example, "Hey Luna" sent to the crypto brain). Current builds fall back to full generation. Rebuild `gf-llm`. |
| Reply repeats a headline similar to your prompt, with no label | Older builds returned the stored *prompt* of the nearest lattice hit instead of its answer. Rebuild `gf-llm`. |
| `NEUTRAL — I don't have enough information to explain this with confidence…` | The brain's router picked the wrong label or has no close example. Use the label model (`train_brain.sh` builds it; `infer.sh` uses it). Add rows for the phrasings it misses. |
| Every prompt with a given label gets the same explanation | The brain merged that label's rows. Set `[train].sentiment_spawn_threshold = 2.0` (or `chat_spawn_threshold` for companions) and retrain. |
| `polish`/`lm` output is garbled | The LM corpus is too small (see §4). `polish` falls back to the brain's answer, while `lm` doesn't. |
| `no training .jsonl files …` / `missing semantic_intent` | Pass a folder of labelled rows. Fragment files aren't training rows (see §5). |
| `brain not found` | Run `train_brain.sh`, or set `BRAIN=/path/to/brain.bin`. |
| A prompt meant for one project gives an odd answer from another | Each brain and LM is domain-specific. Luna prompts go to the Luna project, crypto prompts to crypto. |

`*.bin` and `agent-data/` are git-ignored, so brains and LM checkpoints stay local. Regenerate them with the scripts above.
