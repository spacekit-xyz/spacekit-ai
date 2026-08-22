# Developer guide — train & chat with growformer-llm

Build **small domain chatbots**: train a vanilla LM on your dialogue corpus, then
run an interactive chat loop. For **grounded** domain answers (crypto/fintech
labels, lattice memory), use Path A brain memory first; attach the LM for fluent
continuation.

This crate is **not** a general-purpose ChatGPT replacement. Expect ~0.7M-param
CPU models, short context (`max_seq` often 128), and best results when train and
chat share the same `### User:` / `### Assistant:` format.

---

## Chatbots (product path)

**Default compose = brain** — Path A lattice memory is the assistant reply:

```bash
cargo run --release --no-default-features --features vanilla-lm,brain-memory --bin gf-llm -- \
  chat --compose brain \
  --brain ../../spacekit/spacekit-projects/sentiment/crypto/agent/crypto-brain.bin \
  --project ../../spacekit/spacekit-projects/sentiment/crypto/crypto-sentiment-analysis.gf.toml \
  --message "Bitcoin crashed after the ETF delay"
```

| `--compose` | Behavior |
| --- | --- |
| `brain` (default) | Assistant = retrieved memory (no LM) |
| `polish` | Brain facts → LM one-sentence rewrite (needs checkpoint) |
| `lm` | Experimental LM chat (needs clean turn-aligned train) |

### Retrain LM for polish / lm (clean targets + turn-aligned)

```bash
CHAT=1 STEPS=8000 bash scripts/train_domain_vanilla.sh
# → --chat --chat-clean, --turn-aligned, seq_len=256
```

Assistant lines strip meta rationales (`third-party`, `first-person`, …).

---

## Architecture for chatbots

| Layer | Role | When to use |
| --- | --- | --- |
| **Path A brain** (`compose=brain`) | Route + retrieve + label from lattice | **Default product answers** |
| **Vanilla LM polish** | Rewrite memory into chat tone | After clean chat train |
| **Vanilla LM (`compose=lm`)** | Token continuation | Experimental only |
| **Clifford LM** (`--clifford`) | Historical research | Do not use for new chatbots |

**Recommended bot pattern**

1. User message → `chat --compose brain` (or `brain-infer --brain-only`)
2. Optional → `compose=polish` once a clean chat checkpoint exists
3. Keep system prompt short; train with `--turn-aligned` and `seq_len≥256` for hybrid/polish

---

## Clone and build

From the neurokit workspace (path deps: `growformer`, `growformer-ledger`, `spacekit-compressor`):

```bash
cd growformer-llm

# Product / chatbot binary (no Clifford)
cargo build --release --no-default-features --features vanilla-lm,brain-memory --bin gf-llm

# Same CLI as historical `tinystories` binary
./target/release/gf-llm --help
```

Full default features (includes Clifford for old checkpoints):

```bash
cargo build --release --bin gf-llm
```

---

## Chat prompt contract

Train and infer on the same markers (`growformer_llm::chat`):

```text
### System:
You are a concise domain assistant. …

### User:
Bitcoin crashed after the ETF delay

### Assistant:
Negative — ETF delay triggered a selloff.
```

JSONL rows should include `text` plus ideally `expected_response` (also accepts
`response` / `answer` / `assistant`). Fallback assistant text is `[semantic_intent]`.

---

## End-to-end: domain chatbot

### 1. Chat-formatted corpus

```bash
cargo run --release --no-default-features --features vanilla-lm,brain-memory --bin gf-llm -- \
  jsonl-to-txt ../growformer/data/crypto ../growformer/data/fintech \
  --chat --out data/domain/chat-both.txt
```

Or use the script (set `CHAT=1`):

```bash
CHAT=1 bash scripts/train_domain_vanilla.sh
```

### 2. Tokenize → encode → split → train

```bash
BIN=gf-llm
FEAT="--no-default-features --features vanilla-lm,brain-memory"

cargo run --release $FEAT --bin $BIN -- tokenize data/domain/chat-both.txt 2048 data/domain/chat-both.tok
cargo run --release $FEAT --bin $BIN -- encode data/domain/chat-both.txt data/domain/chat-both.tok data/domain/chat-both.bin
cargo run --release $FEAT --bin $BIN -- split data/domain/chat-both.bin \
  data/domain/chat-both-train.bin data/domain/chat-both-heldout.bin --train-frac 0.9

cargo run --release $FEAT --bin $BIN -- train \
  data/domain/chat-both.tok data/domain/chat-both-train.bin data/domain/chat-both-heldout.bin \
  --checkpoint-out agent-data/chat-domain-vanilla.json \
  --seq-len 128 --steps 4000 \
  --d-model 16 --d-ff 64 --n-blocks 4 --n-heads 4 \
  --tie-embeddings --init-seed 1000
```

Vanilla is the **default** (no `--vanilla` flag).

### 3. Chat REPL

```bash
cargo run --release $FEAT --bin $BIN -- chat \
  --checkpoint agent-data/chat-domain-vanilla.json \
  --tokenizer data/domain/chat-both.tok \
  --system "You are a concise crypto/fintech assistant."
```

One-shot (scripts / tests):

```bash
cargo run --release $FEAT --bin $BIN -- chat \
  --checkpoint agent-data/chat-domain-vanilla.json \
  --tokenizer data/domain/chat-both.tok \
  --message "Bitcoin crashed after the ETF delay"
```

### 4. Grounded hybrid (brain + LM chat)

```bash
cargo run --release $FEAT --bin $BIN -- chat \
  --checkpoint agent-data/chat-domain-vanilla.json \
  --tokenizer data/domain/chat-both.tok \
  --brain ../../spacekit/spacekit-projects/sentiment/crypto/agent/crypto-brain.bin \
  --project ../../spacekit/spacekit-projects/sentiment/crypto/crypto-sentiment-analysis.gf.toml \
  --message "Bitcoin crashed after the ETF delay"
```

Brain-only (no LM) remains available via `brain-infer --brain-only`.

---

## Library usage (embed in your bot)

```rust
use growformer_llm::{
    chat::ChatTranscript, default_chatbot_system, vanilla_forward_logits,
};
use growformer_llm::bpe::BpeTokenizer;
use growformer_llm::v2::vanilla_checkpoint::load_vanilla_state;
use growformer_llm::v2::sample::{sample_next, SampleConfig, SimpleRng};
use growformer_llm::v2::data::special;

let state = load_vanilla_state("agent-data/chat-domain-vanilla.json".as_ref())?;
let bpe = BpeTokenizer::load("data/domain/chat-both.tok")?;
let mut chat = ChatTranscript::with_system(default_chatbot_system());
chat.push_user("What happened to BTC after the ETF delay?");
let prompt = chat.render_for_completion();
// encode → vanilla_forward_logits → sample_next → push_assistant
```

---

## Quality expectations

- Held-out **bits/token** via `eval` measures language modeling, not chatbot helpfulness.
- Small models **hallucinate**; for factual domain bots prefer brain retrieve+label, then LM polish.
- Context is short — keep system text tiny and prune history (`ChatTranscript::truncate_to_token_budget`).
- Do not train on battery / held-out eval prompts if you care about fair scores.

---

### Chat quality bar

Held-out **ppl ~500+** (your first domain-both-chat run ~555) means the LM is
**not** ready for fluent chat — expect garble and repeated `### Assistant:` headers
(the CLI now stops on role markers). Treat Path A `brain-infer --brain-only` as the
answer path until chat ppl is much lower (ballpark &lt;100–150 on this corpus).

**Retrain tip:** `--chat` must **not** embed a long system prompt in every JSONL
example (that wastes `max_seq=128`). Rebuild corpus without `--system`, then train
longer:

```bash
CHAT=1 STEPS=12000 bash scripts/train_domain_vanilla.sh
```

---

## What is experimental

- Domain LM chat quality (fluency after few-k steps on JSONL)
- Hybrid brain+LM chat as a product UX
- Publishing to crates.io (path deps — ship via this workspace clone for now)
