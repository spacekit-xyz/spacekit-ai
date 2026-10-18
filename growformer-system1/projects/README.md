# System 1 use-case projects

Ready-to-run question sets for three use cases. Each folder has a manifest (`*.gf.toml`), questions with option definitions and routing rules (`system1/questions.toml`), and a small hand-written seed set (`data/seed.jsonl`).

| Project | Questions | Actions (`[[rules]]`) |
|---|---|---|
| `gaming/` | toxicity (0–4), abuse type, suspicious (cheating / scams), frustration (0–4), engagement (0–4), churn risk, support route | `crisis_support`, `block_and_escalate`, `mute`, `remove_message`, `investigate`, `priority_support`, `retention_offer`, `human_review`, default `allow` |
| `model-routing/` | domain, task, difficulty (0–4), risk (0–4), needs tools | `escalate:human`, `route:frontier`, `route:agent`, `route:code`, `route:small`, default `route:mid` |
| `guardrails/` | jailbreak, prompt injection, policy violation, sensitive data, tool-call error, response quality (0–4) | `support_resources`, `block`, `redact`, `retry_tool`, `regenerate`, `review`, default `pass` |

The gaming and model-routing seed sets are small (46–55 rows) and written by hand. Use them to **measure** the open model and fit its calibration, not to train production heads. `guardrails/` ships without examples: label checks from your own logged traffic (`guardrails/data/README.md`).

## Workflow

**All at once:** `bash growformer-system1/projects/run_all.sh` rebuilds `gf-system1`, creates the three brains with the open model attached, evaluates and calibrates the two that have seed data, and serves all three with a decision log (`SERVE=0` stops after eval; `MODEL` / `TOKENIZER` point at your model). Rebuild whenever you pull: an older binary has no `init`, `--log`, rules or groups.

Step by step, from the repo root (`S1=target/release/gf-system1`):

**1. Start with no training: the open model answers every question.**

```bash
P=growformer-system1/projects/gaming
$S1 init --project $P/gaming.gf.toml
$S1 attach-lm --brain $P/agent/gaming.s1.json.gz --arch qwen3 --template chatml-nothink \
  --model models/Qwen3-8B-Q4_K_M.gguf --tokenizer models/tokenizer.json
```

**2. Measure it on the seed set and fit its calibration.**

```bash
$S1 eval --brain $P/agent/gaming.s1.json.gz --data $P/data --lm --fit-lm-temperature --save
```

**3. Serve, with a decision log.**

```bash
$S1 serve --brain $P/agent/gaming.s1.json.gz --brain growformer-system1/projects/model-routing/agent/model-routing.s1.json.gz \
  --log decisions.jsonl
curl -s localhost:8787/v1/decide -H 'content-type: application/json' \
  -d '{"brain":"gaming","group":"chat","state":{"channel":"chat","text":"uninstall, you are useless"}}'
```

Each response carries `action` (`{"action": "mute", "rule": 3, "reason": "…"}`) next to the typed answers. Each log line holds every answer's value, confidence, source and full probability distribution, plus the action and latency. States are logged as a fingerprint unless you pass `--log-state`.

**4. Improve with your own data.** Label real traffic (or `distill --source best` your logged states), then `train` the fast heads and `train-probe` the probes, and `eval` on rows held out from training. See `../README.md`.

## Results: Qwen3-8B, no training

Measured with `run_all.sh` on a Mac (Metal), Qwen3-8B Q4_K_M, no heads or probes. `eval` fits calibration on one half of the seed rows and reports on the other half, so each question is scored on 6–17 rows: treat every figure as roughly ±20 points. Runs are deterministic (repeating one gives identical numbers).

"Before" uses one-word scale labels; "after" uses the per-point descriptions with examples now in `system1/questions.toml`. Questions whose wording didn't change scored the same in both runs.

| Brain | Question | n | Before | After | Calibration error (after) |
|---|---|---|---|---|---|
| gaming | support_route | 7 | 100% | 100% | 0.13 |
| gaming | churn_risk | 13 | 92% | 92% | 0.08 |
| gaming | abuse_type | 15 | 80% | 80% | 0.17 |
| gaming | suspicious | 15 | 80% | 80% | 0.15 |
| gaming | toxicity (0–4) | 15 | 53% | **80%** | 0.24 |
| gaming | frustration (0–4) | 13 | 77% | 69% | 0.21 |
| gaming | engagement (0–4) | 6 | 33% | **100%** | 0.00 |
| model-routing | needs_tools | 17 | 100% | 100% | 0.20 |
| model-routing | task | 17 | 88% | 88% | 0.10 |
| model-routing | difficulty (0–4) | 17 | 53% | **94%** | 0.15 |
| model-routing | risk (0–4) | 17 | 59% | **94%** | 0.24 |
| model-routing | domain | 17 | 65% | 65% | 0.33 |

Median latency per state (one model pass for all asked questions): gaming 1.3 → 2.0 s, model-routing 2.2 → 3.9 s. The longer descriptions are part of every question's prompt, so they cost time. Probe heads don't read question text, so training probes removes that cost.

**Reading it.**
- **The gains are overstated.** The new descriptions were written after reading the seed rows, and several examples are taken from them ("this matchmaking is garbage", "haven't logged in for weeks", "a multi-region payments database"). The seed set is no longer an independent test of the wording. Expect smaller gains on new text; measure on rows written or collected after this change.
- **What still holds:** scales described point by point, with "judge X, not Y" guidance between related scales, are much easier for the model than one-word labels. The direction of the change is real even if its size isn't.
- **Frustration** dropped one row (77% → 69% is 10 → 9 of 13): within noise.
- **Domain** didn't move. Its categories overlap (an itinerary is both `analysis` and `generate`); it needs labelled examples or fewer, sharper options.
- Yes/no and clear-cut choices were already strong: support routing, churn, abuse type, suspicious behaviour, task type, tool use.

**Next.**
1. Build an independent test set: 100+ real or freshly written rows per project that nobody looked at while writing the descriptions. Without it, further wording changes can't be judged.
2. Label 100–300 real requests per project (start from `decisions.jsonl` with `--log-state`) and `train-probe`. On the crypto project, probes beat the 8B alone by 16–25 points on house-specific labels and answered in about 300 ms.
3. Guardrails: label logged checks first; it has no measurement yet.

## Groups

`[groups]` names the questions to ask per input type, so a chat line isn't asked about billing:

```json
{"brain": "gaming", "group": "support", "state": {"channel": "support", "text": "charged twice"}}
```

Gaming groups: `chat`, `report`, `review`, `support`. Guardrail groups: `input`, `output`, `tool_call`, so each LLM input, reply and tool call gets only the checks that apply:

```json
{"brain": "guardrails", "group": "tool_call", "state": {"stage": "tool_call", "tool": "get_weather", "args": {}, "result": "error: missing city"}}
```

## Rules

Rules turn answers into one action; the first match wins.

```toml
[[rules]]
action = "mute"
any = true                                                   # any condition (default: all)
when = [{ id = "abuse_type", is = ["harassment", "sexual"] }, # choice value
        { id = "toxicity", min = 2.5 }]                      # score: expected value
```

- `is`: the answer is one of these values (`"true"` / `"false"` for yes/no questions).
- `min` / `max`: score → expected value; yes/no → P(true); choice → confidence.
- `min_confidence`: the answer's confidence is at least this.
- `abstain = true`: the brain wasn't sure (send to a person or a bigger model).

A condition on a question the request didn't ask never matches. Change rules without retraining:

```bash
$S1 update-rules --brain $P/agent/gaming.s1.json.gz --questions $P/system1/questions.toml
```
