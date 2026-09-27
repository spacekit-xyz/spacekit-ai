# Crypto & DeFi Sentiment Analysis

A Growformer micro-brain that classifies the sentiment of crypto, blockchain and DeFi text, and explains its reasoning. It's built for the way people actually write about crypto, including slang, sarcasm, hype and scam talk. It deploys to the SpaceKit Agent Hub as a pay-per-use agent.

Its sister project, [`sentiment/fintech`](../fintech), covers traditional finance.

## Labels

Each input gets one label plus a short rationale: **positive (strong / mild)**, **negative (strong / mild)**, **neutral**, **mixed** or **sarcastic**.

Example inputs from the demo set:

- "Bitcoin just printed a golden cross on the weekly chart for the first time since 2023"
- "Solana fees are practically zero but nobody's building anything people actually use"
- "My portfolio is up 400% this cycle but I still haven't taken profit because I'm a degenerate"

The demo prompt set covers five categories: **Price / Market**, **Crime / Fraud**, **DeFi / Protocol**, **Sarcasm / Mixed** and **Slang / Register**.

## Layout

```
crypto/
├── crypto-sentiment-analysis.gf.toml   # Growformer train + inference manifest
├── deploy.toml                         # Agent Hub deploy manifest (hub settings, marketplace, sample prompts)
├── agent/                              # Trained brain goes here (agent/crypto-brain.bin, not committed)
├── data/
│   ├── train_sentiment_crypto.jsonl            # Main training corpus
│   ├── train_sentiment_crypto_additions.jsonl  # Additional coverage
│   ├── train_sentiment_causal.jsonl            # Cause-and-effect reasoning examples
│   ├── train_sentiment_cata_routing.jsonl      # Category routing
│   ├── train_sentiment_retrieval_gaps.jsonl    # Fixes for retrieval misses
│   ├── inference_crypto.toml                   # Inference gates and rule lists (contrast, sarcasm, polarity)
│   ├── inference_guardrails.jsonl              # Guardrails (loaded at inference only)
│   ├── knowledge_graph.toml                    # Topic graph
│   ├── knowledge_graph_sentiment_overlay.toml
│   ├── world_grounding_crypto.toml             # Crypto lexicon and concept grounding
│   ├── scams_info.md                           # Research notes on scam categories used for dataset design
│   ├── plugins/                                # Brain plugin config
│   └── locale/                                 # Placeholder for non-English shards
└── prompts/                            # Demo and out-of-domain prompt sets
```

## Prerequisites

- The SpaceKit CLI (`spacekit`) on your `PATH`, or a build of the `growformer` binary
- To deploy, the crypto analysis agent WASM built from `spacekit-standard-library`
  (`spacekit_growformer_crypto_analysis.wasm`)

## Train

Trained brains (`*.bin`) are not committed. Train your own:

```bash
spacekit agent --train-brain --project crypto-sentiment-analysis.gf.toml
# writes agent/crypto-brain.bin
```

Or, from the `growformer` crate root:

```bash
cargo run --release --no-default-features --features cli,native --bin growformer -- \
  --train-brain --project /path/to/spacekit-projects/sentiment/crypto/crypto-sentiment-analysis.gf.toml
```

Training merges every `*.jsonl` in `data/`, except the guardrails file.

## Try it

```bash
spacekit agent infer --project crypto-sentiment-analysis.gf.toml \
  --prompt "Layer 2 fees dropped to sub-cent but bridging back to mainnet still costs more than my lunch"
```

`prompts/crypto-prompts-demo.txt` has 25 checked prompts across the five categories. The `fintech-prompts-crypto-unknown*.txt` files hold harder and out-of-domain cases.

## Deploy to the Agent Hub

1. Open `deploy.toml` and set:
   - `[artifacts].wasm` to the path of your `spacekit_growformer_crypto_analysis.wasm`
   - `[storage].url` to your storage node (it's commented out by default)
2. Deploy:

```bash
spacekit storage deploy --package deploy.toml
```

The agent is listed as **Crypto Sentiment Analysis** with tag **CRYPTO**, and the demo prompts from `deploy.toml` appear in the Hub UI. The CLI writes a `deploy-receipt.json`, which is git-ignored.

## Intended use

Good for tone and sentiment in crypto social posts, news headlines and community chat, and for testing how a classifier handles sarcasm and mixed sentiment.

**Not for** trading signals, investment decisions, fraud detection or compliance decisions. The training data is largely synthetic, so validate against your own data before relying on it.

## License

Apache 2.0 — see [LICENSE](LICENSE).
