# Growformer System 1

Fast, non-generative decisions. Give it a **state** (text, JSON object or array) and **typed questions**. It returns schema-safe answers, each with a calibrated probability distribution and a confidence score. It never generates text, so there is nothing to parse.

| Primitive | Asks | Returns |
|---|---|---|
| **Choice** | Which of these options? | the option, plus a probability for each option |
| **Score** | Where on this ordered scale (e.g. 0–4)? | the probability-weighted mean, its standard deviation, and a probability for each point |
| **Noul** | Is this true? | `true`/`false` and `probability` = P(true) |

Every answer also carries `confidence` (the probability of the most likely outcome after calibration) and `abstain`. `abstain` is true when confidence falls below the question's `min_confidence` or the request's. Use it to send the case to a fallback or a human.

## How it works

```text
            ┌───────────── one state ─────────────┐
request ──► │ featurize once ─► head per question │──► confident? ──yes──► answer (≈1 ms)
            │                                      │        │no / no head
            │ open model: state prefix once,       │◄───────┘
            │ each question = short suffix,        │──► answer-token probabilities (no decoding)
            └──────────────────────────────────────┘
```

- **Fast heads.** These are trained per question: TF-IDF over the state feeding a logistic-regression head. Each head gets its own temperature, fitted on held-out rows so its confidences are calibrated. One feature pass is shared by every question.
- **Open-model backend** (feature `candle`). This handles questions that have no head, and heads that aren't confident:
  - It runs a small open-weight model (GGUF; Llama, Qwen2 or Qwen3) through [candle](https://github.com/huggingface/candle).
  - The prompt puts the state first. Llama and Qwen3 process the state once and reuse it for every question; Qwen2 recomputes it per question because its candle model can't be copied.
  - Each answer is read from the probability of its answer token (`A`/`B`/…, a digit, or `Yes`/`No`). No text is generated.
- **Probe heads** (feature `candle`, Qwen3 models). The open model reads the state once, with no question and no decoding. A small calibrated classifier per question then reads its hidden state (`train-probe`). Probes see what the text means, so they learn your labels from a few hundred examples, and every question is answered from that one pass.
- **Router.** A question goes to its fast head when the head's confidence is at or above a threshold; otherwise it goes to its probe if one is trained, else to the open model. `train-probe` sends a question straight to its probe when the probe beat the head on held-out rows; `eval --tune-router` picks a threshold per question from labelled data.
- **Distillation.** `distill` labels raw states with the open model, keeping the full probability distribution as soft labels. `train` then turns those labels into fast heads.

The crate is pure Rust, with no C or C++ dependencies. It builds for Linux, macOS, Windows and `wasm32-unknown-unknown`.

## Build

```bash
cargo build --release -p growformer-system1 --features server,candle --bin gf-system1   # CLI + HTTP server + open model
cargo build --release -p growformer-system1 --features cli --bin gf-system1             # heads only
cargo build --release -p growformer-system1 --target wasm32-unknown-unknown --lib --features wasm         # browser, heads only
cargo build --release -p growformer-system1 --target wasm32-unknown-unknown --lib --features wasm,candle  # browser + open model
```

`metal` and `cuda` features pass GPU support through to candle on native builds.

## Questions and data

`questions.toml`:

```toml
[[questions]]
id = "route"
text = "Which team should handle this ticket?"
type = "choice"
options = ["Billing", "Tech Support", "Cancellation"]

# Optional: one-line definitions the open model sees next to each option.
# Add them whenever option names are your own jargon.
definitions = { Billing = "payments, invoices, refunds", Cancellation = "wants to close the account" }

[[questions]]
id = "urgency"
text = "How urgent is this?"
type = "score"
min = 0
max = 4
labels = ["not urgent", "low", "normal", "high", "emergency"]   # optional; shown to the open model

[[questions]]
id = "refund"
text = "Is the customer asking for a refund?"
type = "noul"
min_confidence = 0.8          # below this, answers come back with abstain = true
```

Labelled JSONL, one row per line:

```json
{"state": "Charged twice, please refund", "labels": {"route": "Billing", "urgency": 3, "refund": true}}
{"state": {"subject": "App crash", "body": "error 500 on save"}, "labels": {"route": "Tech Support"}}
{"state": "…", "soft": {"route": {"Billing": 0.7, "Tech Support": 0.3}}}
```

To reuse an existing field instead of writing `labels`, add a `source` to the question:

```toml
source = { field = "semantic_intent", map = { negative_mild = "negative", positive_mild = "positive" } }
```

## CLI

```bash
S1=target/release/gf-system1

$S1 train --questions questions.toml --data data/ --out agent/support.s1.json.gz      # held-out report per question
$S1 decide --brain agent/support.s1.json.gz --state "Charged twice, please refund"
$S1 decide --brain agent/support.s1.json.gz --request req.json                      # inline questions, JSON state

# Open model for untrained questions / low-confidence heads
$S1 attach-lm --brain agent/support.s1.json.gz --arch qwen3 \
  --model models/Qwen3-0.6B-Q8_0.gguf --tokenizer models/tokenizer.json
$S1 eval --brain agent/support.s1.json.gz --data labelled/ --lm --fit-lm-temperature --tune-router --save
$S1 distill --brain agent/support.s1.json.gz --states unlabelled.jsonl --out distilled.jsonl
$S1 train --questions questions.toml --data distilled.jsonl --data data/ --out agent/support.s1.json.gz

# Probe heads: the open model's hidden state + your labels (features cached in <brain>.emb)
$S1 train-probe --brain agent/support.s1.json.gz --data data/
$S1 eval --brain agent/support.s1.json.gz --data test.jsonl            # head / probe / lm / routed side by side

# Label unlabelled text with the best available backend, then retrain the fast heads on it
$S1 distill --brain agent/support.s1.json.gz --states raw.txt --source best --out distilled.jsonl
$S1 stats --questions questions.toml --data data/                    # rows per outcome; flags thin ones

$S1 serve --brain agent/support.s1.json.gz --addr 127.0.0.1:8787
curl -s localhost:8787/v1/decide -H 'content-type: application/json' \
  -d '{"state":"my card was billed twice","questions":["route","refund"]}'
```

`eval` leaves out rows the heads were trained on (the brain stores a fingerprint of each training state), so its numbers are held-out by default; `--include-train-rows` keeps them. `--fit-lm-temperature / --tune-router` fits on one half of the remaining rows and reports on the other. A question's router threshold is only tuned with at least 50 fitting rows (`--min-router-rows`); fewer than that and the threshold is mostly noise. `train` keeps an existing brain's open-model settings but clears tuned router thresholds, since they were tuned against the old heads.

To get a held-out set, split before training:

```bash
gf-llm jsonl-split data --train-out split/train.jsonl --test-out test.jsonl --seed 7
$S1 train --questions questions.toml --data split/train.jsonl --out agent/support.s1.json.gz
$S1 eval  --brain agent/support.s1.json.gz --data test.jsonl --lm --fit-lm-temperature --tune-router --save
```

A Growformer project manifest can supply the paths:

```toml
[system1]
questions = "system1/questions.toml"
data = ["data"]
brain_output = "agent/support.s1.json.gz"

[system1.lm]                       # optional
arch = "qwen3"
model = "models/Qwen3-0.6B-Q8_0.gguf"
tokenizer = "models/tokenizer.json"
template = "chatml-nothink"
```

```bash
$S1 train --project my.gf.toml && $S1 serve --project my.gf.toml
```

## Request / response

```json
POST /v1/decide
{"state": {"subject": "Charged twice", "body": "please refund"},
 "questions": ["route", {"id": "vip", "text": "Is this a VIP customer?", "type": "noul"}],
 "min_confidence": 0.6}
```

```json
{"brain": "support", "latency_ms": 0.9, "answers": [
  {"id": "route", "type": "choice", "value": "Billing", "confidence": 0.91, "abstain": false,
   "distribution": [{"outcome": "Billing", "p": 0.91}, {"outcome": "Tech Support", "p": 0.06}, {"outcome": "Cancellation", "p": 0.03}],
   "source": "head"},
  {"id": "vip", "type": "noul", "value": false, "probability": 0.22, "confidence": 0.78, "abstain": false,
   "distribution": [{"outcome": "true", "p": 0.22}, {"outcome": "false", "p": 0.78}], "source": "lm"}]}
```

`source` is `head`, `probe`, `lm`, or `none`. `none` means no backend could answer; the answer then has `abstain: true` and an `error` field.

## WASM

```js
import init, * as s1 from "./growformer_system1.js";
await init();
s1.system1_load_brain(new Uint8Array(await (await fetch("support.s1.json.gz")).arrayBuffer()));
// optional, with the `candle` feature:
// s1.system1_load_lm(ggufBytes, tokenizerJsonBytes);
const res = JSON.parse(s1.system1_decide(JSON.stringify({ state: "refund please" })));
```

## Choosing an open model

Use any GGUF of a supported architecture together with its `tokenizer.json`. Small instruct models (about 0.5–2B parameters, quantised Q4/Q8) keep latency in the hundreds of milliseconds on a CPU. Calibrate and tune the router on your own labelled rows before trusting it.

To smoke-test the pipeline without downloading anything, run `cargo run --example tiny_model --features candle -- out/`. It writes a tiny random model; its answers are meaningless, but every code path runs.

## Improving quality

1. **Check the labels.** `stats` lists rows per outcome. Outcomes with fewer than about 30 rows are guesswork for any head: add examples or merge them into a broader outcome.
2. **Probe heads.** `train-probe` usually beats the TF-IDF heads by a wide margin on meaning-heavy labels (sarcasm, regret, "worked for once"). It costs one open-model pass per training row, cached in `<brain>.emb`, so retraining is fast.
3. **Distil.** Collect unlabelled text (JSONL or a `.txt` file, one state per line) and label it with `distill --source best`. Training the fast heads on it gives them far more wording to learn from. The labels are only as good as the backend that produced them, so keep hand labels for your project-specific outcomes.
4. **Measure on enough rows.** Use a held-out split of 150 rows or more; on 40 rows, differences under about 15 points are noise.

## Limits

- Heads are only as good as their labels. On the 375-row crypto sentiment set they reach about 59% (5-way sentiment) and 75% (bearish yes/no) on held-out rows.
- Open-model accuracy depends on the model you pick. Measure it with `eval --lm`.
- Probe heads need a Qwen3 model for now, and only work with the model they were trained on (re-run `train-probe` after switching models).
- Keep arithmetic, counting and date logic in application code. This crate is for semantic judgement.
